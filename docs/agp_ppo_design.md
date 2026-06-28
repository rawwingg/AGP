# PPO for the Art Gallery Problem — Design Notes

This document explains how the PPO agent in `Polygons dataset/ppo_agp.py` works,
answers the "does it place guards randomly?" question, and records future ideas.

## How guard placement / "randomness" works

We do **not** hand-code random placement. PPO uses a **stochastic policy**: a neural
network that, given the current state, outputs a probability distribution over the
guard position. Because our action is continuous `(x, y)`, that distribution is a
2D Gaussian (a mean position + a spread/standard deviation).

- **Early in training** the network is untrained, so the mean is arbitrary and the
  spread is large. Sampled placements therefore look essentially **random**. This is
  the exploration phase, and it happens automatically.
- **As training progresses**, the reward signal pushes the mean toward good locations
  and shrinks the spread. Placements become increasingly **deliberate**.

So it is a single policy that *starts out random and sharpens over time* — not a
manual "random mode then learned mode" switch.

## Environment (one episode = one gallery)

- **Reset**: pick a polygon, place **no** guards yet, compute the "uncovered" grid.
- **Action**: continuous `(x, y)` in `[0, 1]`, mapped onto the polygon's bounding box.
- **Step**: place one guard at the chosen point.
  - If the point is inside the gallery → add the guard, get the **new area covered**.
  - If it is outside (or in a hole) → invalid, small penalty, no guard added.
- We **only add** guards (never remove them), exactly as requested.

### Observation — an image fed to a CNN

The agent "sees" the gallery as an `IMG_SIZE x IMG_SIZE x 3` image (uint8, 0/255), and
the policy uses a **CNN** (`CnnPolicy`) so convolutions can reason about spatial structure
(where the holes are, where the uncovered regions sit). The three channels are:

- **ch0 — inside**: pixels inside the gallery (the room to cover).
- **ch1 — covered**: pixels currently visible to at least one guard.
- **ch2 — remaining**: inside AND not yet covered — i.e. the "right area" still to cover.

This is a richer perception than a flattened vector and matches how the problem is
naturally visual. Note this image is only the **perception**; the action (guard position)
is still continuous and all area/coverage math is exact (continuous), so this remains the
continuous Art Gallery Problem.

(The earlier version flattened a smaller grid into an MLP; the CNN replaces that.)

### Reward (the "+area, −guard, place in the right area" rule)

The reward is built from clear, separate pieces:

```
if guard is OUTSIDE the gallery (wrong area):
    reward = -INVALID_PENALTY            # no guard is placed
else:
    area_term      = new_area_covered / gallery_area     # + for new area
    cost_term      = GUARD_COST                           # - per guard
    placement_term = +PLACEMENT_BONUS  if the spot was uncovered (right area)
                     -REDUNDANT_PENALTY if the spot was already covered (wasted)
    spacing_term   = -CLOSE_PENALTY    if too close to an existing guard, else 0
    reward = area_term - cost_term + placement_term + spacing_term
```

- **`area_term` (+)**: fraction of **newly** covered area (overlap with already-covered
  regions contributes nothing — handled by `Security.add_guard`).
- **`cost_term` (−)**: a fixed `GUARD_COST` per guard, so the agent is discouraged from
  placing guards that add little new area. Drives "cover with **few** guards".
- **`placement_term` (the "right area" shaping)**: gives a dense, immediate signal about
  *where* the guard landed, which helps a lot early in training:
  - placed inside the gallery in a still-**uncovered** spot → `+PLACEMENT_BONUS` (right area),
  - placed inside but in an **already-covered** spot → `-REDUNDANT_PENALTY` (wasted),
  - placed **outside** the gallery → `-INVALID_PENALTY` and no guard added (wrong area).
- **`spacing_term` (the "not too close" rule)**: if the new guard is within
  `MIN_GUARD_DIST_FRAC` of the gallery's bounding-box diagonal from any existing guard,
  it gets `-CLOSE_PENALTY`. This discourages clustering guards on top of each other
  (which wastes guards on overlapping views). The threshold scales per gallery, so it
  works regardless of how big each gallery is.

The "uncovered vs covered" check is a **continuous** point-in-region test against the
current coverage (`Security.coverage_areas`), not a grid lookup. Each shaping constant can
be set to `0.0` to disable that piece.

Note: the discrete grid is **only the observation** (how the agent perceives the gallery).
The action (guard position) and all area/reward math are fully **continuous**.

### Placing on an existing guard is a no-op the agent learns to avoid

`Security.all_guards` is keyed by exact position, so re-placing on the same spot adds
no new area. Such a placement now collects the worst of the shaping terms at once:
`-REDUNDANT_PENALTY` (already-covered spot) **and** `-CLOSE_PENALTY` (too close), while
its `area_term` is ~0. The net reward is clearly negative, so the agent is pushed away
from "stacking" guards. It also counts as **unproductive** (see early stop below).

### Termination

- **terminated**: coverage ≥ `COVERAGE_THRESHOLD` (e.g. 95%) — the gallery is covered.
- **truncated** (episode ends without success) when either:
  - **out of budget**: reached `MAX_GUARDS` steps, or
  - **stuck / early stop**: `PATIENCE` guards in a row were *unproductive*. A guard is
    "unproductive" if it covers less than `USEFUL_AREA_FRAC` (default **10%**) of the
    area that was **still uncovered** when it was placed (this includes invalid
    out-of-bounds placements and guards stacked on existing ones). The threshold is
    **relative to the remaining gap**, so it adapts as the gallery fills: early on a
    guard must add a lot, and as less remains the bar drops proportionally. A productive
    guard resets the streak. This stops the agent from burning the whole guard budget
    re-placing in the same spot once it can no longer make progress, and
    `info["stopped_early"]` flags episodes that ended this way.

## Train / test split

- Generate `N_TRAIN` (e.g. 100) galleries for training and `N_TEST` (e.g. 10) for testing.
- Train PPO across the training galleries (a new random gallery each episode).
- Evaluate the trained policy on the held-out test galleries and report, per gallery,
  the final coverage % and number of guards used.

## Future ideas

### Guard removal / ablation (requested)

Right now the agent only **adds** guards. A natural extension:

- After the agent finishes placing guards, **remove each guard one at a time** and
  measure how much coverage is lost (`Security.remove_guard` already supports this).
- Guards whose removal barely changes coverage are **redundant** → the agent (or a
  post-processing step) could learn to avoid/prune them.
- This turns the problem into placement **and** pruning, and gives a cleaner measure of
  how "efficient" a solution is (coverage per guard).
- Could also be framed as letting the policy choose a "remove" action, making the action
  space mixed (add vs remove) — more complex, worth exploring later.

### Other improvements

- **CNN policy** on the occupancy grid (treat it as an image) instead of a flattened MLP,
  for better spatial reasoning.
- **Terminal bonus** for reaching full coverage, to more strongly reward "finishing".
- **Curriculum**: start with simple convex galleries, progress to many-holed ones.
- **Normalize coordinates** relative to the gallery centroid for scale invariance.
