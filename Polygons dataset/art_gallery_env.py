"""
Art Gallery Problem — Gymnasium Environment for PPO (MultiInputPolicy)
======================================================================

Design rationale
----------------
Coordinates are normalized to [0, 1]:
    PPO operates across many episodes on polygons of varying size. Keeping all
    coordinates in a consistent [0, 1] range makes the neural network's job
    scale-invariant and speeds up learning.

Observations are padded to fixed shapes:
    Gymnasium and SB3 require constant-shape tensors. Different polygons have
    different numbers of vertices / holes / guards, so we pad everything to
    MAX_* sizes and provide boolean masks so the network can ignore padding zeros.

Masks are included:
    Without masks, the network cannot distinguish a real vertex at (0, 0) from a
    padded zero-slot. Masks provide that signal for free at negligible cost.

Invalid actions are penalized, not masked:
    PPO uses a continuous Box action space. There is no discrete "action mask"
    for continuous distributions — every real (x, y) pair is a valid action from
    the policy's perspective. We penalize invalid placements (outside polygon,
    inside hole, too close to another guard) so PPO learns to avoid them.

One PPO step = one guard placement:
    The Art Gallery Problem is solved sequentially (one guard at a time). This
    mapping keeps the environment simple and the reward signal dense.

Reward uses coverage gain (delta), not total coverage:
    Total coverage only tells the agent "how good is the whole solution so far".
    Delta coverage tells the agent "how useful was THIS specific guard". This
    dense, per-step signal gives PPO a much clearer gradient to follow.
"""

import sys
import math
import random
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon

# Ensure sibling files in "Polygons dataset/" are importable regardless of CWD
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from security import Guard, Security  # noqa: E402

# ---------------------------------------------------------------------------
# Padding constants (fixed observation shapes required by SB3)
# ---------------------------------------------------------------------------
MAX_OUTER_VERTICES = 32
MAX_HOLES          = 6
MAX_HOLE_VERTICES  = 16
MAX_GUARDS         = 10
MAX_EPISODE_STEPS  = 50   # hard cap on action attempts, including invalid ones

TARGET_COVERAGE    = 0.95   # Episode is "solved" once coverage >= this

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------
INVALID_ACTION_PENALTY  = -2.0
OUTSIDE_POLYGON_PENALTY = -2.0
INSIDE_HOLE_PENALTY     = -2.0
TOO_CLOSE_GUARD_PENALTY = -1.5
USELESS_GUARD_PENALTY   = -0.5   # guard placed correctly but added almost no area
GUARD_STEP_PENALTY      = -0.1   # small per-guard cost to encourage efficiency
COVERAGE_REWARD_SCALE   = 10.0   # scale coverage gain into a useful signal range
SUCCESS_BONUS           = 10.0   # one-time bonus for reaching TARGET_COVERAGE

MIN_COVERAGE_GAIN  = 0.001  # below this delta, guard is treated as "useless"

# Guards in normalized [0,1] space closer than this are considered duplicates.
# 0.03 ≈ 3 % of the bounding-box side length.
MIN_GUARD_DISTANCE = 0.03


class ArtGalleryEnv(gym.Env):
    """
    Continuous Art Gallery Problem environment compatible with SB3 PPO.

    Each episode uses one polygon sampled from the dataset.
    The agent places guards one at a time (one action = one guard).
    The episode terminates when coverage >= TARGET_COVERAGE or
    MAX_GUARDS guards have been placed.

    Parameters
    ----------
    dataset : list[dict]
        List of polygon records, each with keys:
        ``outer_points``, ``holes``, ``area``, and optionally
        ``id``, ``difficulty``, ``is_valid``.
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset: list):
        super().__init__()

        if not dataset:
            raise ValueError("dataset must be a non-empty list of polygon records.")
        self.dataset = dataset

        # Episode state — populated during reset()
        self.current_polygon: dict = None
        self.security: Security = None
        self.outer_only: ShapelyPolygon = None   # outer ring without holes
        self.guards: list = []                   # placed guard real-coordinates
        self.num_guards: int = 0
        self.episode_steps: int = 0              # counts valid and invalid attempts

        # Bounding-box for coordinate normalisation — recomputed each episode
        self.min_x = self.min_y = 0.0
        self.max_x = self.max_y = 1.0
        self.width  = 1.0
        self.height = 1.0

        # ------------------------------------------------------------------
        # Observation space
        # ------------------------------------------------------------------
        # Dict → MultiInputPolicy gives each component its own MLP encoder.
        # All coordinates are in [0, 1] (normalised).
        # Padded entries are marked by masks (1 = real, 0 = padding).
        self.observation_space = spaces.Dict({
            "outer":       spaces.Box(0.0, 1.0,
                                      shape=(MAX_OUTER_VERTICES, 2),
                                      dtype=np.float32),
            "holes":       spaces.Box(0.0, 1.0,
                                      shape=(MAX_HOLES, MAX_HOLE_VERTICES, 2),
                                      dtype=np.float32),
            "guards":      spaces.Box(0.0, 1.0,
                                      shape=(MAX_GUARDS, 2),
                                      dtype=np.float32),
            "outer_mask":  spaces.Box(0.0, 1.0,
                                      shape=(MAX_OUTER_VERTICES,),
                                      dtype=np.float32),
            "holes_mask":  spaces.Box(0.0, 1.0,
                                      shape=(MAX_HOLES, MAX_HOLE_VERTICES),
                                      dtype=np.float32),
            "guards_mask": spaces.Box(0.0, 1.0,
                                      shape=(MAX_GUARDS,),
                                      dtype=np.float32),
            "coverage":    spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "step":        spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        # ------------------------------------------------------------------
        # Action space
        # ------------------------------------------------------------------
        # Normalised (x, y) in [0, 1].  step() converts back to real coords.
        # One action = one guard placement attempt.
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

    # -----------------------------------------------------------------------
    # Coordinate helpers
    # -----------------------------------------------------------------------

    def _compute_bounds(self):
        """Compute bounding-box of the outer polygon (for normalisation)."""
        pts = self.current_polygon["outer_points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        # Guard against degenerate (zero-size) polygons
        self.width  = max(self.max_x - self.min_x, 1e-9)
        self.height = max(self.max_y - self.min_y, 1e-9)

    def _normalize_point(self, point):
        """
        Map a real (x, y) coordinate into [0, 1] using the episode bounding box.
        Returns a plain Python tuple.
        """
        x, y = float(point[0]), float(point[1])
        nx = float(np.clip((x - self.min_x) / self.width,  0.0, 1.0))
        ny = float(np.clip((y - self.min_y) / self.height, 0.0, 1.0))
        return nx, ny

    def _denormalize_point(self, action):
        """
        Convert a normalised [0, 1] action back to real polygon coordinates.
        Returns a plain Python tuple.
        """
        nx, ny = float(action[0]), float(action[1])
        return nx * self.width + self.min_x, ny * self.height + self.min_y

    # -----------------------------------------------------------------------
    # Padding helpers
    # -----------------------------------------------------------------------

    def _pad_points(self, points, max_points: int):
        """
        Pad a list of (x, y) points to ``max_points`` length.

        Returns
        -------
        padded : np.ndarray, shape (max_points, 2), float32
        mask   : np.ndarray, shape (max_points,),   float32
            mask[i] = 1.0 for real vertices, 0.0 for padding zeros.
        """
        n = min(len(points), max_points)
        padded = np.zeros((max_points, 2), dtype=np.float32)
        mask   = np.zeros(max_points,       dtype=np.float32)
        for i in range(n):
            nx, ny = self._normalize_point(points[i])
            padded[i, 0] = nx
            padded[i, 1] = ny
            mask[i] = 1.0
        return padded, mask

    def _pad_holes(self, holes: list):
        """
        Pad hole list to shape (MAX_HOLES, MAX_HOLE_VERTICES, 2).

        Returns
        -------
        padded : np.ndarray, shape (MAX_HOLES, MAX_HOLE_VERTICES, 2), float32
        mask   : np.ndarray, shape (MAX_HOLES, MAX_HOLE_VERTICES),    float32
        """
        padded = np.zeros((MAX_HOLES, MAX_HOLE_VERTICES, 2), dtype=np.float32)
        mask   = np.zeros((MAX_HOLES, MAX_HOLE_VERTICES),    dtype=np.float32)
        for hi, hole in enumerate(holes[:MAX_HOLES]):
            for vi, pt in enumerate(hole[:MAX_HOLE_VERTICES]):
                nx, ny = self._normalize_point(pt)
                padded[hi, vi, 0] = nx
                padded[hi, vi, 1] = ny
                mask[hi, vi] = 1.0
        return padded, mask

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _get_obs(self) -> dict:
        """Build the observation dict matching ``observation_space``."""
        outer_pad,  outer_mask  = self._pad_points(
            self.current_polygon["outer_points"], MAX_OUTER_VERTICES
        )
        holes_pad,  holes_mask  = self._pad_holes(
            self.current_polygon["holes"]
        )

        guards_pad  = np.zeros((MAX_GUARDS, 2), dtype=np.float32)
        guards_mask = np.zeros(MAX_GUARDS,       dtype=np.float32)
        for i, g in enumerate(self.guards[:MAX_GUARDS]):
            nx, ny = self._normalize_point(g)
            guards_pad[i, 0]  = nx
            guards_pad[i, 1]  = ny
            guards_mask[i]    = 1.0

        # Clip to [0, 1] to handle floating-point edge cases
        cov = float(np.clip(self.security.percent_coverage, 0.0, 1.0))

        return {
            "outer":       outer_pad,
            "holes":       holes_pad,
            "guards":      guards_pad,
            "outer_mask":  outer_mask,
            "holes_mask":  holes_mask,
            "guards_mask": guards_mask,
            "coverage":    np.array([cov],                            dtype=np.float32),
            # step normalised by MAX_GUARDS → always in [0, 1]
            "step":        np.array([self.num_guards / MAX_GUARDS],   dtype=np.float32),
        }

    # -----------------------------------------------------------------------
    # Guard-distance helper
    # -----------------------------------------------------------------------

    def _is_too_close_to_existing_guard(self, real_point) -> bool:
        """
        Return True if ``real_point`` is within MIN_GUARD_DISTANCE of any
        already-placed guard (distance measured in normalised [0, 1] space
        so the threshold is scale-invariant across polygon sizes).
        """
        if not self.guards:
            return False
        nx, ny = self._normalize_point(real_point)
        for g in self.guards:
            gx, gy = self._normalize_point(g)
            if math.hypot(nx - gx, ny - gy) < MIN_GUARD_DISTANCE:
                return True
        return False

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sample one polygon from the dataset (seeded when seed is provided)
        idx = int(self.np_random.integers(len(self.dataset)))
        self.current_polygon = self.dataset[idx]

        # (Re)build the Security simulation for this polygon
        self.security   = Security(self.current_polygon)

        # Outer polygon without holes — used to distinguish
        # "outside polygon" from "inside a hole"
        self.outer_only = ShapelyPolygon(self.current_polygon["outer_points"])

        # Reset episode counters
        self.guards     = []
        self.num_guards = 0
        self.episode_steps = 0

        # Compute normalisation bounds from the chosen polygon's outer ring
        self._compute_bounds()

        obs  = self._get_obs()
        info = {
            "coverage":   0.0,
            "num_guards": 0,
            "episode_steps": 0,
            "difficulty": self.current_polygon.get("difficulty", "unknown"),
            "polygon_id": self.current_polygon.get("id", -1),
        }
        return obs, info

    def step(self, action):
        self.episode_steps += 1

        # ------------------------------------------------------------------
        # 1. Clip and convert action to real coordinates
        # ------------------------------------------------------------------
        action      = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        real_x, real_y = self._denormalize_point(action)
        real_point  = Point(real_x, real_y)
        raw_action  = (float(action[0]), float(action[1]))
        real_action = (real_x, real_y)

        # ------------------------------------------------------------------
        # 2. Validate the guard placement
        # ------------------------------------------------------------------
        invalid_action = False
        invalid_reason = None
        reward         = 0.0
        delta_coverage = 0.0

        # Check in order: outside polygon → inside hole → too close to guard.
        # Using Shapely `contains` which handles holes automatically:
        #   outer_only.contains()  – True iff strictly inside the outer ring
        #   security.perimeter.contains() – True iff inside outer AND outside holes
        if not self.outer_only.contains(real_point):
            invalid_action = True
            invalid_reason = "outside_polygon"
            reward         = OUTSIDE_POLYGON_PENALTY

        elif not self.security.perimeter.contains(real_point):
            # Inside outer ring but not inside perimeter → inside a hole
            invalid_action = True
            invalid_reason = "inside_hole"
            reward         = INSIDE_HOLE_PENALTY

        elif self._is_too_close_to_existing_guard(real_action):
            invalid_action = True
            invalid_reason = "too_close_to_guard"
            reward         = TOO_CLOSE_GUARD_PENALTY

        # ------------------------------------------------------------------
        # 3. Valid placement — add guard and compute coverage gain
        # ------------------------------------------------------------------
        if not invalid_action:
            coverage_before = self.security.percent_coverage

            guard = Guard(real_x, real_y)
            self.security.add_guard(guard)   # raises ValueError only if logic above is wrong

            self.guards.append(real_action)
            self.num_guards += 1

            coverage_after = self.security.percent_coverage
            delta_coverage = coverage_after - coverage_before

            # Dense reward: coverage gain scaled up + small per-guard cost.
            # Using delta (not total) so the agent learns each guard's marginal value.
            reward = COVERAGE_REWARD_SCALE * delta_coverage + GUARD_STEP_PENALTY

            if delta_coverage < MIN_COVERAGE_GAIN:
                # Guard was placed legally but contributed almost no new visible area
                reward += USELESS_GUARD_PENALTY

        # ------------------------------------------------------------------
        # 4. Termination conditions
        # ------------------------------------------------------------------
        coverage_ratio = float(np.clip(self.security.percent_coverage, 0.0, 1.0))
        terminated     = coverage_ratio >= TARGET_COVERAGE
        truncated      = (
            self.num_guards >= MAX_GUARDS
            or self.episode_steps >= MAX_EPISODE_STEPS
        )

        if terminated:
            reward += SUCCESS_BONUS

        # ------------------------------------------------------------------
        # 5. Build observation and info
        # ------------------------------------------------------------------
        obs  = self._get_obs()
        info = {
            "coverage":       coverage_ratio,
            "num_guards":     self.num_guards,
            "episode_steps":   self.episode_steps,
            "delta_coverage": delta_coverage,
            "invalid_action": invalid_action,
            "invalid_reason": invalid_reason,
            "guards":         list(self.guards),
            "raw_action":     raw_action,
            "real_action":    real_action,
            "difficulty":     self.current_polygon.get("difficulty", "unknown"),
        }
        return obs, float(reward), terminated, truncated, info
