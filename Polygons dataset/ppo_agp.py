"""
PPO for the Art Gallery Problem (AGP)
=====================================

A single, self-contained script that:

  1. Generates a dataset of random polygon "galleries" (train + test).
  2. Wraps the gallery + Security coverage logic in a Gymnasium environment.
  3. Trains a PPO agent (Stable-Baselines3) to place guards that cover area.
  4. Tests the trained agent on held-out galleries and reports coverage.

The agent places guards ONE AT A TIME (we never remove guards here; see
docs/agp_ppo_design.md for the future "guard removal" idea).

How the "random placement" works
---------------------------------
PPO uses a stochastic policy: a neural network that outputs a 2D Gaussian over the
guard position (x, y). Early in training the network is untrained, so samples look
random (this is exploration). As it learns from the reward, the placements become
deliberate. We do NOT code random placement explicitly -- it is emergent and fades.

Reward (the "+area, -guard" rule)
---------------------------------
    reward = (new_area_covered / gallery_area) - GUARD_COST

Run it
------
    cd "Polygons dataset"
    source ../.venv/bin/activate

    python ppo_agp.py            # generate (if needed) -> train -> test
    python ppo_agp.py --generate # only (re)generate datasets
    python ppo_agp.py --train    # only train
    python ppo_agp.py --test     # only test (uses saved model)
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import shapely

from security import Security, Guard
from generate_polygon_dataset import generate_dataset


# ----------------------------------------------------------------------------
# Configuration (tweak these freely)
# ----------------------------------------------------------------------------
N_TRAIN = 500            # number of training galleries
N_TEST = 50              # number of test galleries
TIMESTEPS = 500_000      # total PPO training steps
SNAPSHOT_FREQ = 5_000    # every N steps, log a guard-placement image to TensorBoard

# PPO stability knobs. Defaults caused huge policy updates (approx_kl in the tens/hundreds)
# and late-training collapse; these keep each update small and well-behaved.
LEARNING_RATE = 1e-4     # smaller steps than the 3e-4 default
N_EPOCHS = 5             # fewer passes per rollout (default 10) -> less overfitting per update
TARGET_KL = 0.05         # early-stop an update if the policy moves too far (loosened so the
                         # action std doesn't collapse and pin every guard to one spot)
ENT_COEF = 0.01          # entropy bonus: keeps exploration alive so guards spread out
                         # instead of stacking on the first "good" location the policy finds
EVAL_FREQ = 10_000       # every N steps, evaluate on the test set and keep the best model

IMG_SIZE = 64            # observation is an IMG_SIZE x IMG_SIZE x 3 image fed to a CNN
GUARD_RANGE = 100        # max visibility distance per guard (world units); None = unlimited
GUARD_COST = 0.05        # penalty per guard placed (the "-guard" reward term)
COVERAGE_THRESHOLD = 0.90  # episode ends successfully at >= this coverage (reachable on hard maps)
MAX_GUARDS = 30          # hard cap on guards per episode (episode length limit)

# Placement shaping: reward putting the guard in the "right area".
# These guide the agent to aim INSIDE the gallery, at parts that still need covering.
# Set any of them to 0.0 to turn that piece of shaping off.
INVALID_PENALTY = 0.25   # placed OUTSIDE the gallery (or in a hole) -> wrong area
PLACEMENT_BONUS = 0.05   # placed inside the gallery in a still-UNCOVERED spot -> right area
REDUNDANT_PENALTY = 0.05  # placed inside but in an ALREADY-COVERED spot -> wasted

# Spacing: discourage placing guards too close together (clustering wastes guards).
# With LIMITED vision the relevant scale is the guard's range (two guards closer than a
# fraction of their range have heavily overlapping coverage), so "too close" is a fraction
# of GUARD_RANGE. If vision is unlimited we fall back to a fraction of the gallery diagonal.
MIN_GUARD_DIST_FRAC = 0.60       # guards closer than 60% of the vision range are "too close"
MIN_GUARD_DIST_FRAC_DIAG = 0.15  # fallback (unlimited vision): fraction of the diagonal
CLOSE_PENALTY = 0.10             # penalty when a new guard is too close to an existing one

# Completion: a one-off bonus when the episode ends by reaching COVERAGE_THRESHOLD. This
# gives a strong, unambiguous signal for finishing the job (vs. just nibbling area).
COMPLETION_BONUS = 1.0

# Early stop: only meant to catch a guard that adds ~NOTHING (re-placed on the same spot
# or fully inside already-covered area). The threshold is deliberately TINY so a guard
# making small-but-real progress still counts as productive. Earlier this was 1% of total
# area, which on hard/extreme maps flagged genuinely useful guards (each adding ~2-10%
# would pass, but the policy's overlapping aims added <1%) and truncated episodes at 2-3
# guards -- starving training of the long, high-coverage rollouts it needs to learn from.
USEFUL_AREA_FRAC = 0.002  # guard adding < 0.2% of total area = "unproductive" (≈ redundant)
PATIENCE = 5              # end the episode after this many unproductive guards in a row

# File locations.
# Datasets stay in the project, but training outputs are saved on Desktop/AGP so git/GitHub
# Desktop refreshes cannot delete the model checkpoints or TensorBoard logs.
TRAIN_PATH = Path("Polygons") / "train_polygons.json"
TEST_PATH = Path("Polygons") / "test_polygons.json"
RUNS_DIR = Path.home() / "Desktop" / "AGP"
MODEL_PATH = RUNS_DIR / "models" / "ppo_agp"     # SB3 appends .zip
LOG_DIR = RUNS_DIR / "logs" / "ppo_agp"
TEST_PREVIEW_DIR = RUNS_DIR / "test_previews"  # PNGs of the agent's test solutions


# ----------------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------------
def generate_datasets(n_train=N_TRAIN, n_test=N_TEST):
    """Generate train/test gallery datasets (mixed easy/mid/hard/extreme) and return them."""
    print(f"Generating {n_train} training galleries (4 difficulty tiers) -> {TRAIN_PATH}")
    train = generate_dataset(n_train, TRAIN_PATH)
    print(f"Generating {n_test} test galleries (4 difficulty tiers) -> {TEST_PATH}")
    test = generate_dataset(n_test, TEST_PATH)
    return train, test


def load_dataset(path):
    """Load a previously generated dataset, or raise a helpful error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python ppo_agp.py --generate` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# The Gymnasium environment
# ----------------------------------------------------------------------------
class ArtGalleryEnv(gym.Env):
    """One episode = cover a single gallery by placing guards one at a time."""

    metadata = {"render_modes": []}

    def __init__(self, polygons, img_size=IMG_SIZE, guard_range=GUARD_RANGE, guard_cost=GUARD_COST,
                 coverage_threshold=COVERAGE_THRESHOLD, max_guards=MAX_GUARDS,
                 invalid_penalty=INVALID_PENALTY, placement_bonus=PLACEMENT_BONUS,
                 redundant_penalty=REDUNDANT_PENALTY, min_guard_dist_frac=MIN_GUARD_DIST_FRAC,
                 min_guard_dist_frac_diag=MIN_GUARD_DIST_FRAC_DIAG,
                 close_penalty=CLOSE_PENALTY, completion_bonus=COMPLETION_BONUS,
                 useful_area_frac=USEFUL_AREA_FRAC,
                 patience=PATIENCE, random_reset=True):
        super().__init__()
        self.polygons = polygons              # list of gallery dicts
        self.img_size = img_size
        self.guard_range = guard_range        # max ray length for each guard's visibility
        self.guard_cost = guard_cost
        self.coverage_threshold = coverage_threshold
        self.max_guards = max_guards
        self.invalid_penalty = invalid_penalty      # wrong-area (outside) penalty
        self.placement_bonus = placement_bonus      # right-area (uncovered interior) bonus
        self.redundant_penalty = redundant_penalty  # already-covered placement penalty
        self.min_guard_dist_frac = min_guard_dist_frac  # "too close" distance (frac of vision range)
        self.min_guard_dist_frac_diag = min_guard_dist_frac_diag  # fallback (frac of diagonal)
        self.close_penalty = close_penalty          # penalty for placing guards too close
        self.completion_bonus = completion_bonus    # one-off bonus for reaching the threshold
        self.useful_area_frac = useful_area_frac    # below this (frac of TOTAL area) = unproductive
        self.patience = patience                    # consecutive unproductive guards -> stop
        self.random_reset = random_reset      # True: random gallery; False: cycle in order
        self._order_idx = 0                   # used when random_reset is False (evaluation)

        # Action: a point (x, y), each in [0, 1], later mapped onto the gallery bounds.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation is a DICT (MultiInputPolicy): a spatial image + non-spatial scalars.
        #   image (H, W, 3) uint8 -- handled by a CNN:
        #     ch0 = inside the gallery   (the room the agent must cover)
        #     ch1 = already covered      (what guards can currently see)
        #     ch2 = remaining to cover   (inside AND not yet covered = the "right area")
        #   stats (4,) float -- handled by an MLP; info the image cannot (easily) convey:
        #     [coverage_fraction, budget_used_fraction, remaining_centroid_x, _y]
        # - coverage / budget: the image can't show how much guard budget is spent (the
        #   covered channel is a union, so 1 vs 3 guards over the same area look identical).
        # - remaining_centroid (x, y in [0,1] of the bounding box): the center of mass of the
        #   still-uncovered pixels. This directly scaffolds the "aim at the remaining blob"
        #   behaviour -- the policy only has to learn a small offset from this hint instead
        #   of localizing the uncovered region from raw pixels. (It's a hint, not a command:
        #   for a split/U-shaped remainder the centroid may sit in covered space, and the
        #   policy is free to adjust; invalid aims are snapped inward anyway.)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8),
            "stats": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
        })

        # Episode state (filled in by reset()).
        self.security = None
        self.perimeter_area = None
        self.bounds = None        # (minx, miny, maxx, maxy)
        self.grid_x = None        # flattened pixel-center x coords
        self.grid_y = None        # flattened pixel-center y coords
        self.inside_mask = None   # (H, W) bool: pixel inside the gallery
        self.covered_mask = None  # (H, W) bool: pixel already covered by a guard
        self.steps = 0
        self.unproductive_streak = 0  # consecutive guards that added ~no new area

    # -- helpers -------------------------------------------------------------
    def _select_polygon(self):
        """Pick which gallery this episode uses."""
        if self.random_reset:
            return self.polygons[self.np_random.integers(len(self.polygons))]
        record = self.polygons[self._order_idx % len(self.polygons)]
        self._order_idx += 1
        return record

    def _build_grid(self):
        """Rasterize the gallery: figure out which pixels are inside it."""
        minx, miny, maxx, maxy = self.bounds
        xs = np.linspace(minx, maxx, self.img_size)
        ys = np.linspace(miny, maxy, self.img_size)
        gx, gy = np.meshgrid(xs, ys)        # each shape (H, W)
        self.grid_x = gx.ravel()
        self.grid_y = gy.ravel()
        # Vectorized point-in-polygon test (shapely 2.x) -> which pixels are inside.
        inside = shapely.contains_xy(self.security.perimeter, self.grid_x, self.grid_y)
        self.inside_mask = np.asarray(inside, dtype=bool).reshape(self.img_size, self.img_size)
        self.covered_mask = np.zeros_like(self.inside_mask, dtype=bool)

    def _get_obs(self):
        """Build the Dict observation: the (H, W, 3) image plus the (4,) stats vector."""
        remaining = self.inside_mask & (~self.covered_mask)
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img[:, :, 0] = self.inside_mask.astype(np.uint8) * 255    # ch0: gallery interior
        img[:, :, 1] = self.covered_mask.astype(np.uint8) * 255   # ch1: covered so far
        img[:, :, 2] = remaining.astype(np.uint8) * 255           # ch2: still to cover

        coverage = float(np.clip(self.security.percent_coverage, 0.0, 1.0))
        budget_used = float(np.clip(self.steps / self.max_guards, 0.0, 1.0))

        # Center of mass of the still-uncovered pixels, normalized to [0, 1] of the grid.
        # np.nonzero returns (row=y, col=x) indices; columns map to x, rows to y.
        rem_rows, rem_cols = np.nonzero(remaining)
        if rem_cols.size:
            denom = max(self.img_size - 1, 1)
            rem_cx = float(np.clip(rem_cols.mean() / denom, 0.0, 1.0))
            rem_cy = float(np.clip(rem_rows.mean() / denom, 0.0, 1.0))
        else:
            rem_cx = rem_cy = 0.5   # nothing left to cover -> neutral hint

        stats = np.array([coverage, budget_used, rem_cx, rem_cy], dtype=np.float32)
        return {"image": img, "stats": stats}

    # -- gym API -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        record = self._select_polygon()

        # Fresh Security instance => zero guards, empty coverage.
        self.security = Security(record, self.guard_range)
        self.perimeter_area = self.security.perimeter.area
        self.bounds = self.security.perimeter.bounds
        self.steps = 0
        self.unproductive_streak = 0
        self._build_grid()

        # "Too close" distance. With limited vision it scales with the guard's range
        # (overlapping fields of view); with unlimited vision we use the gallery diagonal.
        if self.guard_range:
            self.min_guard_dist = self.min_guard_dist_frac * self.guard_range
        else:
            minx, miny, maxx, maxy = self.bounds
            diagonal = math.hypot(maxx - minx, maxy - miny)
            self.min_guard_dist = self.min_guard_dist_frac_diag * diagonal

        return self._get_obs(), {}

    def _snap_into_gallery(self, x, y):
        """Return (x, y, was_invalid). If (x, y) is outside the gallery or in a hole,
        project it to the nearest valid interior point so a guard can still be placed."""
        from shapely.geometry import Point
        from shapely.ops import nearest_points

        pt = Point(x, y)
        if self.security.perimeter.contains(pt):
            return x, y, False

        # Nearest point on the gallery's filled area (a boundary point for outside/in-hole).
        nearest = nearest_points(self.security.perimeter, pt)[0]
        nx, ny = nearest.x, nearest.y
        # Nudge slightly inward toward a guaranteed-interior reference point until inside.
        ref = self.security.perimeter.representative_point()
        for t in (0.01, 0.05, 0.1, 0.25, 0.5):
            cx = nx + (ref.x - nx) * t
            cy = ny + (ref.y - ny) * t
            if self.security.perimeter.contains(Point(cx, cy)):
                return cx, cy, True
        # Fallback: the representative point is always strictly inside.
        return ref.x, ref.y, True

    def step(self, action):
        self.steps += 1

        # Map the normalized action into actual gallery coordinates.
        a = np.clip(action, 0.0, 1.0)
        minx, miny, maxx, maxy = self.bounds
        x = minx + float(a[0]) * (maxx - minx)
        y = miny + float(a[1]) * (maxy - miny)

        # If the aim is invalid (outside the gallery or inside a hole), snap it to the
        # nearest valid interior point. This GUARANTEES a guard is placed, so the
        # observation always changes -- which stops deterministic eval from getting stuck
        # repeating the same invalid action forever. We still record `was_invalid` so the
        # agent is penalized for aiming poorly (it learns to aim inside on its own).
        x, y, was_invalid = self._snap_into_gallery(x, y)
        guard = Guard(x, y)

        # add_guard returns the NEW area this guard contributes (overlap removed).
        new_area = self.security.add_guard(guard)

        # Update the covered image channel using this guard's coverage polygon.
        coverage_poly = self.security.all_guards[guard.get_position()]
        if not coverage_poly.is_empty:
            newly = shapely.contains_xy(coverage_poly, self.grid_x, self.grid_y)
            newly = np.asarray(newly, dtype=bool).reshape(self.img_size, self.img_size)
            self.covered_mask |= newly

        # --- Reward: deliberately SIMPLE (see docs/agp_ppo_design.md) ---
        # The only way to earn reward is to cover NEW area, minus a small per-guard cost.
        # A guard stacked on an existing one adds ~0 new area, so it just costs -guard_cost;
        # the only positive reward comes from moving to still-uncovered space. This pushes
        # guards apart on its own, without extra spacing/placement shaping terms.
        area_term = new_area / self.perimeter_area   # + for new area covered
        cost_term = self.guard_cost                  # - for each guard placed
        reward = area_term - cost_term

        # Track "unproductive" guards with an ABSOLUTE threshold: the guard must add
        # >= USEFUL_AREA_FRAC of the TOTAL gallery area to count as productive. A guard
        # stacked on an existing one adds ~0 area, so it lands in the unproductive branch.
        # (Absolute, not relative to remaining area, so limited-vision guards on big/complex
        # galleries aren't unfairly flagged just because the remaining area is huge.)
        productive_threshold = self.useful_area_frac * self.perimeter_area
        if new_area < productive_threshold:
            self.unproductive_streak += 1
        else:
            self.unproductive_streak = 0   # productive guard resets the streak

        coverage = self.security.percent_coverage
        terminated = coverage >= self.coverage_threshold   # covered enough -> success
        if terminated:
            reward += self.completion_bonus   # one-off bonus for finishing the job
        # End early if out of budget OR stuck placing unproductive guards repeatedly.
        out_of_budget = self.steps >= self.max_guards
        stuck = self.unproductive_streak >= self.patience
        truncated = out_of_budget or stuck

        info = {
            "coverage": coverage,
            "guards": len(self.security.all_guards),
            "stopped_early": bool(stuck and not out_of_budget),
            "aimed_invalid": bool(was_invalid),
        }
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info


# ----------------------------------------------------------------------------
# Live TensorBoard snapshots: render the agent's current guard placements
# ----------------------------------------------------------------------------
def _render_episode_figure(model, snapshot_env):
    """Run one deterministic episode and draw the gallery + guards + coverage."""
    import matplotlib
    matplotlib.use("Agg")  # no GUI window; we just want the figure for TensorBoard
    import matplotlib.pyplot as plt
    from polygon_viewer_mpl import draw_polygon

    obs, _ = snapshot_env.reset()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = snapshot_env.step(action)
        done = terminated or truncated

    # Build a record the viewer can draw: base gallery + the guards we just placed.
    base = snapshot_env.polygons[0]
    record = dict(base)
    record["guards"] = list(snapshot_env.security.all_guards.keys())
    record["area_coverage"] = info.get("coverage", 0.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_polygon(ax, record, guard_range=snapshot_env.guard_range)
    return fig, info


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(timesteps=TIMESTEPS, continue_training=False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.logger import Figure
    from stable_baselines3.common.monitor import Monitor

    polygons = load_dataset(TRAIN_PATH)
    env = ArtGalleryEnv(polygons, random_reset=True)

    # A fixed gallery used only for the snapshots, so we watch progress on the same room.
    snapshot_env = ArtGalleryEnv([polygons[0]], random_reset=False)

    # Held-out test galleries used to periodically score the policy and keep the BEST one,
    # so a late-training collapse can't overwrite a good earlier model.
    eval_env = Monitor(ArtGalleryEnv(load_dataset(TEST_PATH), random_reset=True))

    class TrainingMonitor(BaseCallback):
        """Logs galleries processed (a scalar) and guard-placement images to TensorBoard."""

        def __init__(self, snapshot_env, snapshot_freq):
            super().__init__()
            self.snapshot_env = snapshot_env
            self.snapshot_freq = snapshot_freq
            self.galleries_processed = 0
            self._last_snapshot = 0

        def _on_step(self):
            # Each finished episode == one gallery fully processed.
            for done in self.locals.get("dones", []):
                if done:
                    self.galleries_processed += 1
            self.logger.record("custom/galleries_processed", self.galleries_processed)

            # Periodically render the agent's current solution as an image.
            if self.num_timesteps - self._last_snapshot >= self.snapshot_freq:
                self._last_snapshot = self.num_timesteps
                fig, info = _render_episode_figure(self.model, self.snapshot_env)
                # Logging a matplotlib Figure puts it under the TensorBoard IMAGES tab.
                self.logger.record("rollout/guard_placement",
                                   Figure(fig, close=True),
                                   exclude=("stdout", "log", "json", "csv"))
                self.logger.record("custom/snapshot_coverage", info.get("coverage", 0.0))
                self.logger.record("custom/snapshot_guards", info.get("guards", 0))
            return True

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Use Apple GPU (MPS) when available; otherwise CPU.
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    monitor_cb = TrainingMonitor(snapshot_env, SNAPSHOT_FREQ)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_PATH.parent),
        log_path=str(LOG_DIR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=len(load_dataset(TEST_PATH)),
        deterministic=True,
        verbose=1,
    )

    if continue_training:
        # Resume from the best checkpoint if available, else the last saved model.
        best_path = MODEL_PATH.parent / "best_model"
        if best_path.with_suffix(".zip").exists():
            load_path = best_path
            print(f"Continuing from {load_path}.zip")
        elif MODEL_PATH.with_suffix(".zip").exists():
            load_path = MODEL_PATH
            print(f"Continuing from {load_path}.zip (no best_model found)")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {MODEL_PATH.parent}. Train from scratch first."
            )
        model = PPO.load(str(load_path), env=env, device=device)
        start = model.num_timesteps
        print(f"Checkpoint at {start:,} timesteps; training {timesteps:,} more "
              f"(target ~{start + timesteps:,} total).")
        model.learn(
            total_timesteps=timesteps,
            callback=[monitor_cb, eval_cb],
            reset_num_timesteps=False,  # keep TensorBoard step counter continuous
        )
    else:
        # MultiInputPolicy: a CNN reads the image and an MLP reads the stats vector;
        # SB3 combines them before the policy/value heads.
        model = PPO(
            "MultiInputPolicy", env, device=device, verbose=1, tensorboard_log=str(LOG_DIR),
            learning_rate=LEARNING_RATE,
            n_epochs=N_EPOCHS,
            target_kl=TARGET_KL,
            ent_coef=ENT_COEF,
        )
        print(f"Training PPO for {timesteps} timesteps on {len(polygons)} galleries...")
        model.learn(total_timesteps=timesteps, callback=[monitor_cb, eval_cb])
    model.save(str(MODEL_PATH))
    print(f"Saved final model to {MODEL_PATH}.zip")
    print(f"Best model (by test coverage) saved to {MODEL_PATH.parent / 'best_model'}.zip")
    return model


# ----------------------------------------------------------------------------
# Testing / evaluation
# ----------------------------------------------------------------------------
def test(save_previews=True):
    from stable_baselines3 import PPO

    polygons = load_dataset(TEST_PATH)
    # random_reset=False so we walk through every test gallery in order, once each.
    env = ArtGalleryEnv(polygons, random_reset=False)

    # Prefer the best model (highest test coverage during training); fall back to final.
    best_path = MODEL_PATH.parent / "best_model"
    if best_path.with_suffix(".zip").exists():
        model = PPO.load(str(best_path))
        print(f"Loaded best model: {best_path}.zip")
    else:
        model = PPO.load(str(MODEL_PATH))
        print(f"Loaded final model: {MODEL_PATH}.zip")

    # Set up the preview renderer (matplotlib, no GUI window).
    if save_previews:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from polygon_viewer_mpl import draw_polygon
        TEST_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))

    print(f"\nEvaluating on {len(polygons)} test galleries:\n")
    print(f"{'gallery':>8} | {'diff':>8} | {'guards':>6} | {'coverage':>9}")
    print("-" * 42)

    coverages, guard_counts = [], []
    by_difficulty = {}
    for _ in range(len(polygons)):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            # deterministic=True -> use the policy's mean action (no exploration noise).
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        record = env.polygons[(env._order_idx - 1) % len(env.polygons)]
        gallery_id = record["id"]
        difficulty = record.get("difficulty", "?")
        coverages.append(info["coverage"])
        guard_counts.append(info["guards"])
        by_difficulty.setdefault(difficulty, {"coverage": [], "guards": []})
        by_difficulty[difficulty]["coverage"].append(info["coverage"])
        by_difficulty[difficulty]["guards"].append(info["guards"])
        print(f"{gallery_id:>8} | {difficulty:>8} | {info['guards']:>6} | {info['coverage'] * 100:>8.2f}%")

        # Render this gallery with the guards the agent actually placed.
        if save_previews:
            preview = dict(record)
            preview["guards"] = list(env.security.all_guards.keys())
            preview["area_coverage"] = info["coverage"]
            draw_polygon(ax, preview, guard_range=env.guard_range)
            out = TEST_PREVIEW_DIR / f"test_polygon_{gallery_id}.png"
            fig.savefig(out, dpi=110, bbox_inches="tight")

    print("-" * 42)
    print(f"{'avg':>8} | {'all':>8} | {np.mean(guard_counts):>6.1f} | {np.mean(coverages) * 100:>8.2f}%")
    print("\nBy difficulty:")
    for tier in ("easy", "mid", "hard", "extreme"):
        if tier not in by_difficulty:
            continue
        d = by_difficulty[tier]
        print(f"  {tier:>8}: {len(d['coverage']):>3} galleries | "
              f"{np.mean(d['guards']):>4.1f} guards | {np.mean(d['coverage']) * 100:>6.2f}% coverage")

    if save_previews:
        plt.close(fig)
        print(f"\nSaved test previews to {TEST_PREVIEW_DIR}/")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PPO for the Art Gallery Problem.")
    parser.add_argument("--generate", action="store_true", help="(re)generate datasets only")
    parser.add_argument("--train", action="store_true", help="train only")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="resume from models/best_model.zip (or ppo_agp.zip)")
    parser.add_argument("--test", action="store_true", help="test only")
    parser.add_argument("--steps", type=int, default=TIMESTEPS,
                        help="training timesteps (additional steps when using --continue)")
    args = parser.parse_args()

    # If no specific flag is given, do the full pipeline.
    do_all = not (args.generate or args.train or args.test)

    if args.generate or do_all:
        if do_all and TRAIN_PATH.exists() and TEST_PATH.exists():
            print("Datasets already exist, skipping generation (use --generate to force).")
        else:
            generate_datasets()

    if args.train or do_all:
        train(timesteps=args.steps, continue_training=args.continue_training)

    if args.test or do_all:
        test()


if __name__ == "__main__":
    main()
