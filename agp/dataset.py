"""
Dataset construction.

A dataset is a JSON list of gallery records. Each record carries the geometry
(outer ring + holes) and, precomputed once here rather than at every episode
reset, the greedy baseline:

    k_ref           guard count after greedy cover then greedy prune
    k_greedy        guard count before pruning
    greedy_guards   the pre-prune guard positions
    greedy_coverage / pruned_coverage   what each phase actually achieved

k_ref is what the terminal parsimony bonus scores against, and greedy_guards is
what stage 2 uses as a pre-saturated start state. Computing them here costs a
few minutes once instead of a few milliseconds every reset.

Train, validation and test are generated from disjoint seeds and written to
separate files, so the held-out galleries are genuinely unseen geometry rather
than a reshuffle. Validation picks the best stage-2 checkpoint; test is only
ever used for the final score.
"""

import json
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .baselines import greedy_solution
from .polygons import TIER_NAMES, generate_gallery


DEFAULT_DIR = Path("data")

# How many galleries of each difficulty go into each split.
TRAIN_COUNTS: Dict[str, int] = {"easy": 200, "mid": 200, "hard": 200}
TEST_COUNTS: Dict[str, int] = {"easy": 30, "mid": 30, "hard": 30}
VAL_COUNTS: Dict[str, int] = {"easy": 20, "mid": 20, "hard": 20}

SPLITS = ("train", "test", "val")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _annotate(args):
    """Attach the greedy baseline to one record. Top level so Pool can pickle it."""
    record, threshold, max_guards, img_size = args
    record.update(greedy_solution(record, threshold=threshold, max_guards=max_guards,
                                  img_size=img_size, seed=record["id"]))
    return record


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_split(counts: Dict[str, int],
                seed: int,
                threshold: float = 0.98,
                max_guards: int = 16,
                img_size: int = 64,
                workers: Optional[int] = None) -> List[dict]:
    """
    Generate `counts[tier]` galleries per tier and annotate them with greedy.

    Geometry generation uses the `random` module (the sampler is written against
    it), so the seed is set once here; the greedy pass is deterministic given a
    record, which keeps the result reproducible despite running in parallel.
    """
    for tier in counts:
        if tier not in TIER_NAMES:
            raise ValueError(f"unknown tier {tier!r}; expected one of {TIER_NAMES}")

    random.seed(seed)
    np.random.seed(seed)

    # --- 1. geometry (cheap, serial) ---
    records: List[dict] = []
    for tier, count in counts.items():
        for _ in range(count):
            records.append(generate_gallery(tier, gallery_id=len(records)))

    # --- 2. greedy baseline (expensive, parallel) ---
    payload = [(record, threshold, max_guards, img_size) for record in records]
    with Pool(processes=workers) as pool:
        annotated = pool.map(_annotate, payload, chunksize=4)

    return annotated


def build(out_dir: Path = DEFAULT_DIR,
          train_counts: Optional[Dict[str, int]] = None,
          test_counts: Optional[Dict[str, int]] = None,
          val_counts: Optional[Dict[str, int]] = None,
          seed: int = 0,
          threshold: float = 0.98,
          max_guards: int = 16,
          img_size: int = 64,
          workers: Optional[int] = None,
          only: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Build the splits and write them to `out_dir`. Returns the paths written.

    `only` restricts the build to some splits; each split's seed is fixed, so
    building one leaves the others' contents unchanged.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    splits = [("train", train_counts or TRAIN_COUNTS, seed),
              ("test", test_counts or TEST_COUNTS, seed + 10_000),
              ("val", val_counts or VAL_COUNTS, seed + 20_000)]

    for name, counts, split_seed in splits:
        if only and name not in only:
            continue
        print(f"[dataset] building {name}: {counts}")
        records = build_split(counts, split_seed, threshold, max_guards, img_size, workers)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        written[name] = path
        _report(name, records)

    return written


def _report(name: str, records: List[dict]):
    """Print a short per-tier summary so a bad build is obvious immediately."""
    print(f"[dataset] {name}: {len(records)} galleries")
    for tier in TIER_NAMES:
        subset = [r for r in records if r["tier"] == tier]
        if not subset:
            continue
        k_ref = np.mean([r["k_ref"] for r in subset])
        k_greedy = np.mean([r["k_greedy"] for r in subset])
        coverage = np.mean([r["pruned_coverage"] for r in subset])
        print(f"           {tier:>5}  n={len(subset):3d}  "
              f"k_greedy={k_greedy:5.2f}  k_ref={k_ref:5.2f}  coverage={coverage:.3f}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load(path: Path) -> List[dict]:
    """Read a dataset JSON, with a useful error if it has not been built yet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the dataset first:  python -m agp.train dataset"
        )
    return json.loads(path.read_text(encoding="utf-8"))
