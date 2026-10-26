"""
The Art Gallery MDP as a Gymnasium environment.

This file is the executable form of docs/agp_mdp_formulation.pdf. Section
numbers in the comments refer to that document.

  State (s4)      the gallery instance plus the multiset of placed guards.
  Action (s5)     Add(x, y) | Del(x, y) | Stop -- one fixed-size vector.
  Observation     a 5-channel image of the gallery plus 5 scalars.
  Reward (s6)     the weighted basis in rewards.py.
  Episode ends    on Stop, on covering the gallery, or on exhausting the budget.

The same class serves both curriculum stages. A stage is nothing more than a
`RewardWeights` column plus the `allow_delete` flag, both of which the training
script sets at runtime with `VecEnv.set_attr`.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .gallery import Gallery
from .polygons import augment_record
from .rewards import RewardShape, RewardWeights, SATURATE


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------
# The document specifies a factored action: a categorical head over
# {Add, Del, Stop} and a point in the unit square. Stable-Baselines3 has no
# mixed action space, so the action travels as one 3-vector [mode, x, y] with
# the mode as an integer 0/1/2 in a float slot. policy.py supplies the matching
# distribution: a real categorical for the mode, a Gaussian for the point.
ADD, DEL, STOP = 0, 1, 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """Everything about the environment that is not a reward weight."""

    # --- episode limits (s5.4) ---
    # Greedy needs 1-2 guards on easy galleries and 5-8 on hard ones, so a
    # budget of 16 leaves room to over-saturate and then prune without ever
    # being so loose that running out of guards stops meaning anything.
    max_guards: int = 16          # hard budget; hitting it ends the episode as a failure
    max_steps: int = 40           # wall-clock safety net, not part of the MDP
    coverage_threshold: float = 0.98   # c-dagger: "covered enough"

    # --- observation ---
    img_size: int = 64

    # --- which actions are live (s5.2) ---
    allow_delete: bool = False    # stage 1 masks Del and Stop off entirely

    # --- coverage ratchet (s6.5) ---
    # A deletion is refused outright if it would drop coverage more than this
    # far below the best coverage seen in the episode. Makes catastrophic
    # unlearning impossible rather than merely expensive.
    ratchet_slack: float = 0.01

    # --- episode initialisation (s7.1 of the curriculum document) ---
    augment: bool = True              # apply a random D4 symmetry to the gallery
    presaturate_prob: float = 0.0     # fraction of episodes that start from an
                                      # over-saturated state instead of empty
    presaturate_extra: int = 4        # surplus guards added to that start state


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class ArtGalleryEnv(gym.Env):
    """Place and remove guards in a randomly drawn gallery until it is covered."""

    metadata = {"render_modes": []}

    def __init__(self,
                 records: List[dict],
                 config: Optional[EnvConfig] = None,
                 weights: Optional[RewardWeights] = None,
                 shape: Optional[RewardShape] = None,
                 seed: Optional[int] = None):
        super().__init__()
        if not records:
            raise ValueError("ArtGalleryEnv needs at least one gallery record")

        self.records = records
        self.config = config or EnvConfig()
        self.weights = weights or SATURATE
        self.shape = shape or RewardShape()
        self._rng = np.random.default_rng(seed)

        # --- spaces -------------------------------------------------------
        # Action: [mode, x, y]. mode is ADD/DEL/STOP as 0/1/2; x and y are in
        # [0, 1], normalised coordinates within the gallery's bounding box.
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                       high=np.array([2.0, 1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)

        # Observation: an image the CNN reads, and scalars the MLP reads.
        size = self.config.img_size
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(size, size, 5), dtype=np.uint8),
            "stats": spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
        })

        # --- episode state (filled in by reset) ---------------------------
        self.gallery: Optional[Gallery] = None
        self.k_ref: int = 1
        self.step_count: int = 0
        self.best_coverage: float = 0.0
        self.episode_return: float = 0.0

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # --- 1. draw a gallery, optionally under a random symmetry ---
        record = self.records[int(self._rng.integers(len(self.records)))]
        if self.config.augment:
            record = augment_record(record, int(self._rng.integers(8)))

        self.gallery = Gallery(record, img_size=self.config.img_size)
        # The greedy solver's guard count is the baseline the terminal parsimony
        # bonus is scored against. Fall back to something sane if absent.
        self.k_ref = max(int(record.get("k_ref", 8)), 1)

        # --- 2. optionally start from an over-saturated state (stage 2) ---
        # Deliberate over-saturation: lay down the greedy cover, then a handful
        # of extra guards at random. The extras are redundant by construction,
        # so the start state is guaranteed to contain free deletions. This is
        # the cheap stand-in for rolling out a frozen stage-1 policy, and it
        # gives the Del action a dense signal from its very first update.
        if self._rng.random() < self.config.presaturate_prob:
            self._presaturate(record)

        # --- 3. episode bookkeeping ---
        self.step_count = 0
        self.best_coverage = self.gallery.coverage
        self.episode_return = 0.0

        return self._observation(), {}

    def _presaturate(self, record: dict):
        """Fill the gallery with the greedy cover plus `presaturate_extra` spares."""
        budget = self.config.max_guards
        for x, y in record.get("greedy_guards", [])[:budget]:
            self.gallery.add_guard(float(x), float(y))

        minx, miny, maxx, maxy = self.gallery.bounds
        spares = min(self.config.presaturate_extra, budget - self.gallery.n_guards)
        placed, attempts = 0, 0
        while placed < spares and attempts < 100:
            attempts += 1
            x = self._rng.uniform(minx, maxx)
            y = self._rng.uniform(miny, maxy)
            if self.gallery.contains(x, y):
                self.gallery.add_guard(x, y)
                placed += 1

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------
    def step(self, action):
        weights = self.weights
        mode, x, y = self._decode(action)

        reward = 0.0
        terminated = False
        info = {"mode": mode}

        # =================== ADD ===========================================
        if mode == ADD:
            # Project into the gallery; aiming outside is legal but penalised
            # so the policy learns the shape rather than relying on the clamp.
            x, y, was_outside = self.gallery.project(x, y)
            if was_outside:
                reward -= weights.proj

            # Measure the scaffolding quantities BEFORE the state changes.
            in_frontier = not self.gallery.is_covered(x, y)
            gap = self.gallery.distance_to_nearest_guard(x, y)
            wall_distance = self.gallery.distance_to_boundary(x, y)

            # Coverage potential (+) and guard-count potential (-).
            delta_plus = self.gallery.add_guard(x, y)
            reward += weights.cov * delta_plus - weights.par

            # Scaffolds: reward aiming at uncovered space, following the
            # boundary-inward sweep, and not stacking guards on top of guards.
            if in_frontier:
                reward += weights.front
            reward += weights.sweep * self._sweep_kernel(wall_distance)
            if gap < self.shape.min_separation * self.gallery.diagonal:
                reward -= weights.sep

            info["delta"] = delta_plus

            # Budget exhausted: the decision problem is genuinely over, so this
            # is a terminal state (see the truncation note in _limits below).
            if self.gallery.n_guards >= self.config.max_guards:
                terminated = True

            # In stage 1 the episode's goal is coverage, so reaching it ends it.
            # In stage 2 we keep going, because there is still pruning to do.
            if (not self.config.allow_delete) and self.gallery.coverage >= self.config.coverage_threshold:
                terminated = True

        # =================== DEL ===========================================
        elif mode == DEL:
            # "Delete" points at a location; the guard nearest to it is the one
            # removed. This keeps the action fixed-size and order-independent,
            # instead of indexing into a list whose ordering is arbitrary.
            index = self.gallery.nearest_guard(x, y)

            if index is None:
                # Nothing to delete -- a wasted action, not a state change.
                reward -= weights.void
            else:
                cost = self.gallery.exact_exclusive_area(index)
                floor = self.best_coverage - self.config.ratchet_slack
                if self.gallery.coverage - cost < floor:
                    # The ratchet refuses deletions that would undo real work.
                    reward -= weights.void
                    info["blocked"] = True
                else:
                    delta_minus = self.gallery.remove_guard(index)
                    # Mirror of the Add case: coverage potential down, guard-count
                    # potential up, plus a bounded bonus for a near-free removal.
                    reward += (-weights.cov * delta_minus
                               + weights.par
                               + weights.prune * math.exp(-delta_minus / self.shape.sigma_prune))
                    info["delta"] = -delta_minus

        # =================== STOP ==========================================
        else:
            terminated = True
            if self.gallery.coverage < self.config.coverage_threshold:
                reward -= weights.stop      # stopping short of the goal

        # --- episode limits and terminal reward ----------------------------
        self.step_count += 1
        self.best_coverage = max(self.best_coverage, self.gallery.coverage)

        truncated = (not terminated) and self.step_count >= self.config.max_steps
        if terminated:
            reward += self._terminal_reward()

        self.episode_return += reward
        info.update(self._episode_info(terminated or truncated))

        return self._observation(), float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # Reward helpers
    # -----------------------------------------------------------------------
    def _terminal_reward(self) -> float:
        """
        Paid once, when the episode really ends (s6.4).

        Note this is NOT paid on truncation. Truncation is the step-limit safety
        net, an artefact of the implementation rather than part of the MDP;
        Stable-Baselines3 bootstraps the value function there instead, which is
        the correct handling and would be double-counted if we also paid out.
        """
        covered = self.gallery.coverage >= self.config.coverage_threshold
        if not covered:
            return -self.weights.fail

        # Guards saved relative to the greedy baseline, never negative -- the
        # agent is rewarded for beating greedy, not punished for matching it.
        saved = max(self.k_ref - self.gallery.n_guards, 0)
        return self.weights.goal + self.weights.pars * saved

    def _sweep_kernel(self, wall_distance: float) -> float:
        """
        Boundary-inward placement prior (s6.3).

        Early in an episode the preferred distance from the wall is small, and
        it grows as the guard budget is spent, so the policy is nudged to trace
        the perimeter first and then fill the interior. Returns a value in
        (0, 1], peaking when the guard sits at the currently preferred depth.
        """
        shape = self.shape
        progress = self.gallery.n_guards / max(self.config.max_guards, 1)
        mu = shape.sweep_mu_start + (shape.sweep_mu_end - shape.sweep_mu_start) * progress
        normalised = wall_distance / max(self.gallery.diagonal, 1e-9)
        return math.exp(-((normalised - mu) ** 2) / (2 * shape.sweep_sigma ** 2))

    # -----------------------------------------------------------------------
    # Action decoding
    # -----------------------------------------------------------------------
    def _decode(self, action):
        """Turn the raw 3-vector into (mode, x, y) in gallery coordinates."""
        action = np.asarray(action, dtype=np.float64)
        px, py = np.clip(action[1:], 0.0, 1.0)

        # Stage 1 masks Del and Stop. The policy masks them too; this keeps the
        # environment correct for any policy, including one not trained here.
        if not self.config.allow_delete:
            mode = ADD
        else:
            mode = int(np.clip(np.rint(action[0]), ADD, STOP))

        minx, miny, maxx, maxy = self.gallery.bounds
        return mode, minx + px * (maxx - minx), miny + py * (maxy - miny)

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------
    def _observation(self):
        """
        Five image channels plus five scalars.

        Channels 0-2 describe the gallery and the job left to do; channels 3-4
        describe the guards. Channel 4 is what makes pruning learnable at all:
        without it the policy cannot tell a load-bearing guard from a redundant
        one, and Del is a blind guess.
        """
        gallery = self.gallery
        size = self.config.img_size
        image = np.zeros((size, size, 5), dtype=np.uint8)

        image[:, :, 0] = gallery.interior_mask.astype(np.uint8) * 255   # where the gallery is
        image[:, :, 1] = gallery.covered_mask().astype(np.uint8) * 255  # already seen
        image[:, :, 2] = gallery.uncovered_mask().astype(np.uint8) * 255  # still to do
        image[:, :, 3] = gallery.guard_mask().astype(np.uint8) * 255    # guard positions

        deficits = gallery.exclusive_areas()
        image[:, :, 4] = gallery.redundancy_mask(self.shape.sigma_prune, deficits)  # cost to delete

        # Cheapest guard to remove, in units of the pruning tolerance. 0 means
        # "there is a free deletion available right now".
        cheapest = float(deficits.min()) if deficits.size else 1.0
        cheapest = min(cheapest / self.shape.sigma_prune, 1.0)

        stats = np.array([
            gallery.coverage,
            gallery.n_guards / self.config.max_guards,
            self.step_count / self.config.max_steps,
            cheapest,
            # Room left under the ratchet before a deletion gets refused.
            min(max(gallery.coverage - (self.best_coverage - self.config.ratchet_slack), 0.0), 1.0),
        ], dtype=np.float32)

        return {"image": image, "stats": np.clip(stats, 0.0, 1.0)}

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    def _episode_info(self, finished: bool) -> dict:
        """Per-episode metrics, emitted only on the final step of an episode."""
        if not finished:
            return {}
        coverage = self.gallery.coverage
        solved = coverage >= self.config.coverage_threshold
        return {
            "coverage": coverage,
            "n_guards": self.gallery.n_guards,
            "k_ref": self.k_ref,
            "solved": float(solved),
            # Guards used as a multiple of greedy: below 1.0 means we beat it.
            "guard_ratio": self.gallery.n_guards / self.k_ref,
            "tier": self.gallery.record.get("tier", "unknown"),
        }
