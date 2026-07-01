# INFPPO Branch — Summary of Changes

This document summarizes what the **INFPPO** branch adds on top of `main`: a continuous Art Gallery Problem (AGP) solver trained with PPO (Proximal Policy Optimization) via Stable-Baselines3.

## Overview

The branch implements end-to-end reinforcement learning for guard placement:

1. Generate random polygon galleries (with holes).
2. Wrap gallery geometry and visibility in a Gymnasium environment.
3. Train a PPO agent to place guards that maximize coverage while minimizing guard count.
4. Evaluate on held-out galleries and save visual previews.

The main entry point is `Polygons dataset/ppo_agp.py` — a single script for dataset generation, training, and testing.

## New / Modified Files

| File | Purpose |
|------|---------|
| `Polygons dataset/ppo_agp.py` | Main PPO pipeline: env, train, test |
| `Polygons dataset/security.py` | Guard visibility (ray casting), coverage union, incremental area |
| `Polygons dataset/generate_polygon_dataset.py` | Batch polygon generation to JSON |
| `Polygons dataset/generate_polygons.py` | Low-level polygon + hole geometry |
| `Polygons dataset/polygon_viewer_mpl.py` | Matplotlib viewer for galleries and guard coverage |
| `docs/agp_ppo_design.md` | Detailed design notes and future ideas |
| `docs/ppo_sb3_setup.md` | SB3 install and smoke-test guide |
| `scripts/ppo_smoke_test.py` | CartPole sanity check for SB3 |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignores `logs/`, `models/`, `*.zip`, generated polygons |

## Core Design

### Problem formulation

- **Continuous action**: agent outputs `(x, y)` in `[0, 1]`, mapped to the gallery bounding box.
- **One guard per step**: guards are only added, never removed.
- **Episode**: one gallery per episode; ends on success, budget exhaustion, or early stop.

### Observation (`MultiInputPolicy`)

The agent receives a **dict** with two parts:

- **`image`** — `64×64×3` uint8 channels:
  - ch0: gallery interior
  - ch1: area already covered
  - ch2: remaining uncovered area
- **`stats`** — `[coverage_fraction, budget_used_fraction]`

A CNN processes the image; an MLP processes the scalars. This lets the policy know how much guard budget is spent — information the covered channel alone cannot convey.

### Reward function

```
reward = area_term - cost_term + placement_term + spacing_term + invalid_term
```

| Term | Meaning |
|------|---------|
| `area_term` | `+new_area / gallery_area` (incremental coverage only) |
| `cost_term` | `-GUARD_COST` per guard (default 0.05) |
| `placement_term` | `+0.05` if uncovered spot, `-0.05` if redundant |
| `spacing_term` | `-0.10` if too close to an existing guard |
| `invalid_term` | `-0.25` if aim was outside and had to be snapped inward |

### Robust placement (snap-to-valid)

If the agent aims outside the gallery or into a hole, `_snap_into_gallery` projects the point to the nearest valid interior location. A guard is still placed (with the invalid penalty). This prevents evaluation episodes from getting stuck repeating the same invalid action with 0 guards placed.

### Early stopping (patience)

Episodes truncate early if the agent places `PATIENCE` (3) guards in a row that are "unproductive" — each adds less than `USEFUL_AREA_FRAC` (10%) of the **remaining uncovered** area. This avoids wasting the full guard budget when stuck.

Other termination:

- **Success**: coverage ≥ `COVERAGE_THRESHOLD` (0.98)
- **Budget**: `MAX_GUARDS` (20) reached

### Training stability

Default PPO settings caused large policy updates and late-training collapse. The branch tunes:

| Hyperparameter | Value |
|----------------|-------|
| `LEARNING_RATE` | `1e-4` |
| `N_EPOCHS` | `5` |
| `TARGET_KL` | `0.03` |
| `ENT_COEF` | `0.0` |

An `EvalCallback` saves `best_model.zip` based on held-out test coverage so a late collapse does not overwrite the best policy.

### Hardware and outputs

- Auto-detects **Apple MPS** GPU when available, otherwise CPU.
- Models, TensorBoard logs, and test previews save to **`~/Desktop/AGP/`** (outside the repo) so `git pull` cannot delete checkpoints.

## `security.py` Fixes

- `coverage_areas` starts as an empty `Polygon()` (not a placeholder point).
- `add_guard()` returns **incremental** new area (overlap removed via `difference`).
- `remove_all_guards()` resets coverage correctly.

## Dataset

- **500** training galleries, **50** test galleries (generated to `Polygons/train_polygons.json` and `Polygons/test_polygons.json`).
- Polygons have random outer shapes and 1–4 holes.

## How to Run

```bash
source .venv/bin/activate
cd "Polygons dataset"

# Full pipeline (generate if missing → train → test)
python ppo_agp.py

# Individual steps
python ppo_agp.py --generate
python ppo_agp.py --train --steps 500000
python ppo_agp.py --continue --steps 500000   # resume from best_model.zip
python ppo_agp.py --test

# Monitor training
tensorboard --logdir ~/Desktop/AGP/logs/ppo_agp
```

## Test Output

`python ppo_agp.py --test` prints per-gallery coverage and guard counts, plus saves PNG previews to `~/Desktop/AGP/test_previews/`.

## Known Limitations & Next Steps

These were identified during development and are **not yet** on this branch:

1. **Limited guard vision** — guards currently see the full gallery (unlimited range). Adding a `GUARD_RANGE` parameter in `security.py` and the env is the next major feature.
2. **Difficulty tiers** — easy / mid / hard / extreme polygon generation for curriculum-style training.
3. **Early-stop tuning for limited vision** — when vision is capped, the 10%-of-remaining-area unproductive threshold can stop episodes too early (3–5 guards on hard maps). Recommended fix: use 1% of **total** gallery area instead.
4. **Completion bonus** — one-off reward when reaching the coverage threshold (stronger signal for finishing).
5. **Spacing scale** — with limited vision, "too close" should scale with guard range, not gallery diagonal.
6. **Coverage threshold** — 0.98 may be unreachable on complex galleries; 0.90 is a more practical target with limited range.

## Future Ideas (see `docs/agp_ppo_design.md`)

- Guard removal / pruning (measure redundancy via `Security.remove_guard`).
- Curriculum from simple to complex galleries.
- Terminal completion bonus and coordinate normalization.

## Commits on INFPPO (vs `main`)

- Polygon generation and `security.py` guard coverage
- SB3 baseline setup and smoke test
- Full `ppo_agp.py` PPO implementation
- GPU (MPS) support, early stopping, training stability fixes
- Multi-input observation, snap-to-valid, best-model checkpointing, Desktop output paths
