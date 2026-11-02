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
    cell   ~ Categorical(G*G | mode)           which region of the bounding box
    offset ~ Beta(ax, bx) x Beta(ay, by)       where inside that region, in [0, 1)^2
    (x, y) = ((col + offset_x) / G, (row + offset_y) / G)

Add and Del each read their cell and offsets off their own map (`delete_map`).
With one shared map the point could not depend on the mode, so a Del pointed
where Add, two thirds of all actions, had learned to point -- at uncovered
space -- while free guards sit where coverage overlaps; through v10 most Dels
on hard galleries landed far from any free guard. Stop ignores the point.

The point is continuous: every position in the gallery is reachable, the grid
only organises the choice. Cells partition the unit square, so the density of a
point is exactly P(cell) * Beta(offset) (up to the constant G^2), and PPO's
ratio is exact. Cells with no gallery interior are masked, Del's map is also
masked to cells holding a guard (Del removes the nearest guard, so no target is
lost), in stage 1 Del and
Stop are masked (`add_only`), and Stop is masked while coverage is below the
threshold (`stop_below`) or while some guard is free to delete (`stop_while_free`).
Del is masked below the threshold as well (`delete_below`). In stage 2 Add is
masked in cells holding a guard that sees nothing no other guard sees
(`skip_redundant_cells`).

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

from .env import EnvConfig

N_MODES = 3          # Add, Del, Stop -- matches env.ADD / DEL / STOP
DEL = 1
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
    """Categorical mode x categorical cell given the mode x Beta offset, over [mode, x, y].

    There are two maps, indexed by `_map`: 0 serves Add (and Stop, whose point is
    ignored), 1 serves Del.
    """

    def __init__(self, mode_logits, cell_logits, offset_params, grid: int):
        self.mode_dist = Categorical(logits=mode_logits)
        self.cell_dist = Categorical(logits=cell_logits)   # batch (B, 2): one per map
        self.offset_params = offset_params          # (B, 2, 4, G*G): ax, bx, ay, by per map and cell
        self.grid = grid

    @staticmethod
    def _map(mode):
        return (mode == DEL).long()

    def _offsets(self, which, cell):
        """The two Beta distributions belonging to the given map and cell of each sample."""
        params = self.offset_params[torch.arange(cell.shape[0], device=cell.device), which]
        index = cell[:, None, None].expand(-1, 4, 1)
        p = params.gather(2, index).squeeze(2)
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
            which = self._map(mode)
            cell = self.cell_dist.probs.argmax(dim=2).gather(1, which[:, None]).squeeze(1)
            bx, by = self._offsets(which, cell)
            return self._to_action(mode, cell, bx.mean, by.mean)
        mode = self.mode_dist.sample()
        which = self._map(mode)
        cell = self.cell_dist.sample().gather(1, which[:, None]).squeeze(1)
        bx, by = self._offsets(which, cell)
        return self._to_action(mode, cell, _sample_beta(bx), _sample_beta(by))

    def log_prob(self, actions):
        g = self.grid
        mode = actions[:, 0].round().long().clamp(0, N_MODES - 1)
        gx = actions[:, 1].clamp(0, 1) * g
        gy = actions[:, 2].clamp(0, 1) * g
        col = gx.floor().clamp(0, g - 1)
        row = gy.floor().clamp(0, g - 1)
        cell = (row * g + col).long()
        which = self._map(mode)
        bx, by = self._offsets(which, cell)
        cell_log_prob = self.cell_dist.log_prob(cell[:, None].expand(-1, 2))
        point = (cell_log_prob.gather(1, which[:, None]).squeeze(1)
                 + bx.log_prob((gx - col).clamp(OFFSET_EPS, 1 - OFFSET_EPS))
                 + by.log_prob((gy - row).clamp(OFFSET_EPS, 1 - OFFSET_EPS)))
        # Stop ignores the point, so the point is not part of a Stop action.
        return self.mode_dist.log_prob(mode) + torch.where(mode == STOP, torch.zeros_like(point), point)

    def entropy(self, mode_weight: float = 1.0):
        """
        Joint entropy, with the mode term optionally weighted.

        The cell term (up to ln G^2) dwarfs the mode term (at most ln 3), so a
        single bonus strong enough to keep Del and Stop explored would also
        flatten placement. `mode_weight` scales the mode term alone.

        The point term is each map's entropy weighted by the chance of its mode;
        Stop has no point. The weights are detached so the bonus cannot steer the
        mode towards whichever map is currently more spread out (Add's, over the
        whole gallery, against Del's, over the guards). In stage 1 only Add is
        possible and this is the true entropy.
        """
        p = self.offset_params
        offset_entropy = (Beta(p[:, :, 0], p[:, :, 1]).entropy() + Beta(p[:, :, 2], p[:, :, 3]).entropy())
        point_entropy = self.cell_dist.entropy() + (self.cell_dist.probs * offset_entropy).sum(dim=2)
        probs = self.mode_dist.probs.detach()
        weights = torch.stack([probs[:, 0], probs[:, DEL]], dim=1)
        return mode_weight * self.mode_dist.entropy() + (weights * point_entropy).sum(dim=1)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class SpatialPolicy(MultiInputActorCriticPolicy):
    """
    Actor-critic over `SpatialTrunk`. The mode head and the value head read the
    global vector; the cell scores and offsets are read off the map, one per cell.
    """

    def __init__(self, *args, delete_map: bool = False, **kwargs):
        # Whether Del has its own cell map. A constructor argument on purpose, so
        # a stage-2 save reloads with the map; stage 1 never has one, and
        # `split_delete_map` adds it when stage 2 begins.
        self.delete_map = delete_map
        # Not constructor arguments on purpose: SB3 saves constructor kwargs
        # into the checkpoint, and a stage-1 save would then reload as add-only.
        self.add_only = False
        # Stop is masked while observed coverage is below this. Whether 97.9%
        # is short of 98% is a fact, not something worth spending samples on:
        # left to learn it, stage 2 kept stopping at 96-98% on hard galleries.
        self.stop_below = EnvConfig.coverage_threshold
        # Stop is also masked while some guard can be deleted without uncovering
        # the gallery. Given that guards cost reward, a free guard left at Stop is
        # never the better answer; left to learn it, v7 stopped on every one.
        self.stop_while_free = True
        # Del is masked below this too. Under the threshold a deletion can only
        # lose more coverage, and moving a guard nets nothing, so Add is the only
        # useful move; left to learn it, v8 kept deleting at 97% until two
        # refusals ended the episode uncovered. A Del from at or above the
        # threshold may still dip below it, and the mask then forces the repair.
        self.delete_below = EnvConfig.coverage_threshold
        # Add is masked in stage 2 in any cell holding a guard that sees nothing
        # no other guard sees: a guard there is known to add nothing. With the
        # state unchanged by such an Add, a deterministic policy repeats it;
        # v11 stacked 11 guards on one spot of a hard gallery at 97.6% coverage
        # until the budget ran out.
        self.skip_redundant_cells = True
        # Multiplies the mode term of the entropy bonus (see SpatialDistribution.entropy).
        self.mode_entropy_weight = 1.0
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
        if self.delete_map:
            self.delete_head = _new_delete_head(channels)
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
        hidden = self.cell_head[:-1](torch.cat([cells, context], dim=1))
        add_out = self.cell_head[-1](hidden)
        del_out = self.delete_head(hidden) if self.delete_map else add_out
        out = torch.stack([add_out, del_out], dim=1)                # (B, 2, 5, G, G)

        # Channel 3 of the image holds the guards; Del's own map keeps only their
        # cells. Without one, Del shares Add's map and mask, as before v11.
        del_live = interior
        if self.delete_map:
            del_live = interior & (F.adaptive_max_pool2d(obs["image"][:, 3:4].float(), g)[:, 0] > 0.5)
        add_live = interior
        if self.skip_redundant_cells and not self.add_only:
            add_live = interior & ~_redundant_cells(obs["image"][:, 4:5].float(), g)
        live = torch.stack([add_live, del_live], dim=1).flatten(2)
        cell_logits = out[:, :, 0].flatten(2).masked_fill(~live, MASKED_LOGIT)
        # 1 + softplus keeps both Beta parameters >= 1: unimodal, never U-shaped.
        offset_params = 1.0 + F.softplus(out[:, :, 1:].flatten(3))

        mode_logits = self.mode_net(summary)
        if self.add_only:
            mode_logits = torch.cat([mode_logits[:, :1],
                                     torch.full_like(mode_logits[:, 1:], MASKED_LOGIT)], dim=1)
        else:
            stats = obs["stats"]
            blocked = torch.zeros_like(stats[:, 0], dtype=torch.bool)
            if self.stop_below is not None:
                blocked |= stats[:, 0] < self.stop_below     # stats[0] is coverage
            if self.stop_while_free:
                blocked |= stats[:, 3] < 0.5                 # stats[3] is 0 while a guard is free
            mode_logits = mode_logits.clone()
            mode_logits[:, STOP] = mode_logits[:, STOP].masked_fill(blocked, MASKED_LOGIT)
            if self.delete_below is not None:
                short = stats[:, 0] < self.delete_below
                mode_logits[:, DEL] = mode_logits[:, DEL].masked_fill(short, MASKED_LOGIT)
        return SpatialDistribution(mode_logits, cell_logits, offset_params, g), summary

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "delete_map": self.delete_map}

    def forward(self, obs, deterministic: bool = False):
        dist, summary = self._distribution(obs)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self.value_net(summary), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        dist, summary = self._distribution(obs)
        # PPO only uses this entropy for its bonus, so the weighted form goes here.
        return (self.value_net(summary), dist.log_prob(actions),
                dist.entropy(mode_weight=self.mode_entropy_weight))

    def get_distribution(self, obs):
        return self._distribution(obs)[0]

    def predict_values(self, obs):
        _, summary, _ = self.extract_features(obs)
        return self.value_net(summary)


def _redundant_cells(channel4, grid: int):
    """
    Cells holding a guard drawn at 255 in channel 4: one whose deletion costs
    nothing (see Gallery.redundancy_mask). Guards are 3x3 blobs, so a guard's
    own pixel is one whose whole 3x3 neighbourhood is 255; testing only those
    keeps a blob that spills over a cell edge from masking the neighbour.
    """
    centre = -F.max_pool2d(-channel4, 3, stride=1, padding=1) > 254.5
    return F.adaptive_max_pool2d(centre.float(), grid)[:, 0] > 0.5


def _new_delete_head(channels: int) -> nn.Conv2d:
    """Del's cell score and offsets, read off the same hidden map as Add's.
    Small-gain init: uniform over the guards' cells, offsets gently centred."""
    head = nn.Conv2d(channels, 5, 1)
    nn.init.orthogonal_(head.weight, gain=0.01)
    nn.init.zeros_(head.bias)
    return head


def split_delete_map(model):
    """
    Give Del its own cell map when stage 2 begins.

    The new head starts uniform over the cells holding a guard rather than as a
    copy of Add's map, which points at uncovered space. The optimizer is rebuilt
    over all parameters so a save reloads into one parameter group, and the
    existing parameters keep their Adam state. `policy_kwargs` records the map,
    so stage-2 checkpoints rebuild it on load.
    """
    policy = model.policy
    if policy.delete_map:
        return
    policy.delete_head = _new_delete_head(policy.features_extractor.channels).to(policy.device)
    policy.delete_map = True
    old = policy.optimizer
    policy.optimizer = policy.optimizer_class(policy.parameters(), lr=old.param_groups[0]["lr"],
                                              **policy.optimizer_kwargs)
    for param, state in old.state.items():
        policy.optimizer.state[param] = state
    model.policy_kwargs = {**model.policy_kwargs, "delete_map": True}


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
