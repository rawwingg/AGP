"""
The spatial policy: choose a mode, choose a region of the map, then choose a
continuous point inside that region.

The MDP's action is Add(x, y) | Del(x, y) | Stop with (x, y) anywhere in the
gallery (docs/agp_mdp_formulation.pdf, s5). Two earlier designs failed in ways
that shaped this one:

  * A flattened CNN summary feeding a single Gaussian over (x, y). The map is
    squeezed to 4x4 and then to 64 numbers before any coordinate is produced,
    and one Gaussian can only express one target, so with several uncovered
    pockets the mean lands between them. The deterministic policy stacked its
    guards in the middle of the gallery.
  * A mode encoded as a third continuous number cut into bands, which put Del
    between Add and Stop and stopped the policy from ever learning to stop.

Here the network keeps the map's layout to the end and the action factors as

    mode   ~ Categorical(3)                    Add / Del / Stop
    cell   ~ Categorical(G*G)                  which region of the bounding box
    offset ~ Beta(ax, bx) x Beta(ay, by)       where inside that region, in [0, 1)^2
    (x, y) = ((col + offset_x) / G, (row + offset_y) / G)

The point is continuous: every position in the gallery is reachable, the grid
only organises the choice. Cells partition the unit square, so the density of a
point is exactly P(cell) * Beta(offset) (up to the constant G^2), and PPO's
ratio is exact. Cells with no gallery interior are masked, and in stage 1 Del
and Stop are masked (`add_only`).

Wire format, unchanged so the environment and SB3 need nothing new:

    action = [mode, x, y]    mode in {0, 1, 2} as a float, x, y in [0, 1]
"""

import math

import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn
from torch.distributions import Beta, Categorical
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

N_MODES = 3          # Add, Del, Stop -- matches env.ADD / DEL / STOP
STOP = 2
MASKED_LOGIT = -1e8  # zeroes a choice's probability while keeping gradients finite
OFFSET_EPS = 1e-5    # keeps offsets strictly inside their cell, and Beta log-probs finite


# ---------------------------------------------------------------------------
# Network trunk
# ---------------------------------------------------------------------------
class SpatialTrunk(BaseFeaturesExtractor):
    """
    Image -> (G x G feature map, global vector, interior mask per cell).

    The local branch downsamples the 64x64 image only as far as the cell grid,
    so every cell keeps its own features. A second branch keeps going down to
    4x4 and summarises the whole gallery, together with the scalar stats, into
    a global vector: visibility is unlimited in range, so whether a spot is
    worth a guard depends on the entire gallery, not just its neighbourhood.
    The policy broadcasts that vector back onto every cell before scoring.

    Two coordinate channels are appended to the image so the convolutions,
    which are otherwise translation-invariant, know where in the gallery they are.
    """

    def __init__(self, observation_space: spaces.Dict, grid: int = 16,
                 channels: int = 32, global_dim: int = 256):
        super().__init__(observation_space, features_dim=global_dim)
        c_in, size, _ = observation_space["image"].shape       # channels-first after VecTransposeImage
        n_stats = observation_space["stats"].shape[0]
        def power_of_two(n):
            return n >= 1 and n & (n - 1) == 0
        # The local branch halves the image down to the grid; the global branch
        # halves the grid down to 4x4. Both need whole powers of two.
        if not (grid >= 4 and power_of_two(grid) and size % grid == 0
                and size // grid >= 2 and power_of_two(size // grid)):
            raise ValueError(f"grid {grid} must be a power of two in [4, {size // 2}]")
        self.grid = grid
        self.channels = channels

        ys, xs = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None], persistent=False)

        # --- local branch: image -> G x G ---
        local = [nn.Conv2d(c_in + 2, channels, 5, stride=2, padding=2), nn.ReLU()]
        for _ in range(int(math.log2(size // 2 // grid))):
            local += [nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU()]
        local += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU()]
        self.local = nn.Sequential(*local)

        # --- global branch: G x G -> 4 x 4 -> vector ---
        down = []
        for _ in range(int(math.log2(grid // 4))):
            down += [nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU()]
        self.down = nn.Sequential(*down, nn.Flatten(),
                                  nn.Linear(channels * 16, global_dim), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(global_dim + n_stats, global_dim), nn.ReLU())

    def forward(self, obs):
        image = obs["image"]
        x = torch.cat([image, self.coords.expand(image.shape[0], -1, -1, -1)], dim=1)
        cells = self.local(x)
        summary = self.merge(torch.cat([self.down(cells), obs["stats"]], dim=1))
        # Channel 0 is the gallery interior; a cell is live if any pixel of it is.
        interior = F.adaptive_max_pool2d(image[:, :1], self.grid)[:, 0] > 0.5
        return cells, summary, interior


# ---------------------------------------------------------------------------
# Action distribution
# ---------------------------------------------------------------------------
def _sample_beta(dist: Beta):
    """Sample a Beta on the CPU: Apple's MPS backend has no Beta/Dirichlet sampler.
    Only one offset per environment is drawn per step, so the round trip is cheap."""
    a, b = dist.concentration1, dist.concentration0
    if a.device.type != "mps":
        return dist.sample()
    return Beta(a.detach().cpu(), b.detach().cpu()).sample().to(a.device)


class SpatialDistribution:
    """Categorical mode x categorical cell x Beta offset, over [mode, x, y]."""

    def __init__(self, mode_logits, cell_logits, offset_params, grid: int):
        self.mode_dist = Categorical(logits=mode_logits)
        self.cell_dist = Categorical(logits=cell_logits)
        self.offset_params = offset_params          # (B, 4, G*G): ax, bx, ay, by per cell
        self.grid = grid

    def _offsets(self, cell):
        """The two Beta distributions belonging to the given cell of each sample."""
        index = cell[:, None, None].expand(-1, 4, 1)
        p = self.offset_params.gather(2, index).squeeze(2)
        return Beta(p[:, 0], p[:, 1]), Beta(p[:, 2], p[:, 3])

    def _to_action(self, mode, cell, ox, oy):
        g = self.grid
        ox = ox.clamp(OFFSET_EPS, 1 - OFFSET_EPS)
        oy = oy.clamp(OFFSET_EPS, 1 - OFFSET_EPS)
        x = ((cell % g).float() + ox) / g
        y = (torch.div(cell, g, rounding_mode="floor").float() + oy) / g
        return torch.stack([mode.float(), x, y], dim=1)

    def get_actions(self, deterministic: bool = False):
        if deterministic:
            mode = self.mode_dist.probs.argmax(dim=1)
            cell = self.cell_dist.probs.argmax(dim=1)
            bx, by = self._offsets(cell)
            return self._to_action(mode, cell, bx.mean, by.mean)
        mode = self.mode_dist.sample()
        cell = self.cell_dist.sample()
        bx, by = self._offsets(cell)
        return self._to_action(mode, cell, _sample_beta(bx), _sample_beta(by))

    def log_prob(self, actions):
        g = self.grid
        mode = actions[:, 0].round().long().clamp(0, N_MODES - 1)
        gx = actions[:, 1].clamp(0, 1) * g
        gy = actions[:, 2].clamp(0, 1) * g
        col = gx.floor().clamp(0, g - 1)
        row = gy.floor().clamp(0, g - 1)
        cell = (row * g + col).long()
        bx, by = self._offsets(cell)
        point = (self.cell_dist.log_prob(cell)
                 + bx.log_prob((gx - col).clamp(OFFSET_EPS, 1 - OFFSET_EPS))
                 + by.log_prob((gy - row).clamp(OFFSET_EPS, 1 - OFFSET_EPS)))
        # Stop ignores the point, so the point is not part of a Stop action.
        return self.mode_dist.log_prob(mode) + torch.where(mode == STOP, torch.zeros_like(point), point)

    def entropy(self):
        p = self.offset_params
        offset_entropy = (Beta(p[:, 0], p[:, 1]).entropy() + Beta(p[:, 2], p[:, 3]).entropy())
        expected_offset = (self.cell_dist.probs * offset_entropy).sum(dim=1)
        return self.mode_dist.entropy() + self.cell_dist.entropy() + expected_offset


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class SpatialPolicy(MultiInputActorCriticPolicy):
    """
    Actor-critic over `SpatialTrunk`. The mode head and the value head read the
    global vector; the cell scores and offsets are read off the map, one per cell.
    """

    def __init__(self, *args, **kwargs):
        # Not a constructor argument on purpose: SB3 saves constructor kwargs
        # into the checkpoint, and a stage-1 save would then reload as add-only.
        self.add_only = False
        kwargs.setdefault("features_extractor_class", SpatialTrunk)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        trunk = self.features_extractor
        channels, width = trunk.channels, trunk.features_dim
        self.grid = trunk.grid

        self.broadcast = nn.Linear(width, channels)
        # One output map per quantity: cell score, then the four Beta parameters.
        self.cell_head = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 5, 1),
        )
        self.mode_net = nn.Linear(width, N_MODES)
        self.value_net = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 1))

        # Start every head near "uniform": all cells equally likely, offsets
        # gently centred, modes even -- the same small-gain init SB3 uses.
        for layer in (self.cell_head[-1], self.mode_net):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.value_net[-1].weight, gain=1.0)

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _distribution(self, obs):
        cells, summary, interior = self.extract_features(obs)
        g = self.grid
        context = self.broadcast(summary)[:, :, None, None].expand(-1, -1, g, g)
        out = self.cell_head(torch.cat([cells, context], dim=1))

        cell_logits = out[:, 0].flatten(1).masked_fill(~interior.flatten(1), MASKED_LOGIT)
        # 1 + softplus keeps both Beta parameters >= 1: unimodal, never U-shaped.
        offset_params = 1.0 + F.softplus(out[:, 1:].flatten(2))

        mode_logits = self.mode_net(summary)
        if self.add_only:
            mode_logits = torch.cat([mode_logits[:, :1],
                                     torch.full_like(mode_logits[:, 1:], MASKED_LOGIT)], dim=1)
        return SpatialDistribution(mode_logits, cell_logits, offset_params, g), summary

    def forward(self, obs, deterministic: bool = False):
        dist, summary = self._distribution(obs)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self.value_net(summary), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        dist, summary = self._distribution(obs)
        return self.value_net(summary), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        return self._distribution(obs)[0]

    def predict_values(self, obs):
        _, summary, _ = self.extract_features(obs)
        return self.value_net(summary)


def prime_mode_head(policy: SpatialPolicy, add_prob: float):
    """
    Start the mode head at "mostly Add" for stage 2, independent of the state.

    Stage 1 masks Del and Stop, so their logits never receive a gradient.
    Zeroing the head and setting the Add bias gives P(Add) = add_prob in every
    state, with the rest split evenly between Del and Stop -- enough to explore
    both without throwing away the placement behaviour stage 1 learned.
    """
    others = (1.0 - add_prob) / (N_MODES - 1)
    with torch.no_grad():
        policy.mode_net.weight.zero_()
        policy.mode_net.bias.zero_()
        policy.mode_net.bias[0] = math.log(add_prob / others)
    policy.add_only = False
