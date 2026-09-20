"""
Classical baselines: greedy cover and greedy prune.

These serve three purposes.

  1. `greedy_cover` produces k_ref, the reference guard count that the terminal
     parsimony bonus is scored against. Without a per-instance reference, "used
     few guards" is not comparable between an easy gallery and a hard one.

  2. Its output doubles as the pre-saturated start state for stage 2. Greedy set
     covers are provably within a log factor of optimal but are essentially
     never minimal, so a greedy solution is a state rich in free deletions --
     precisely the distribution the prune stage needs to see.

  3. `greedy_prune` is the thing the learned policy has to beat. Greedy-then-
     prune is the obvious non-learned pipeline; if PPO cannot improve on it,
     the approach has not earned its complexity.

Greedy selection runs on the raster rather than on exact polygons. Selecting a
guard needs only a ranking of marginal gains, and on the raster that ranking is
one numpy reduction instead of one polygon difference per candidate. The final
coverage figure is always re-measured exactly.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point

from .gallery import Gallery


Points = List[Tuple[float, float]]


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def candidate_points(gallery: Gallery, n_random: int = 40,
                     rng: Optional[np.random.Generator] = None) -> Points:
    """
    Where greedy is allowed to put a guard.

    Two families. Near-vertex points, because an optimal guard set can always be
    pushed towards corners and corners are where occlusion is decided; and
    uniform interior samples, which cover the open floor that no corner sees
    well. Vertices are nudged a little way towards the gallery's interior so
    they are strictly inside rather than sitting on an edge.
    """
    rng = rng or np.random.default_rng(0)
    points: Points = []

    # --- near-vertex candidates ---
    inward = max(gallery.diagonal * 0.01, 0.5)
    rings = [list(gallery.polygon.exterior.coords)]
    rings += [list(ring.coords) for ring in gallery.polygon.interiors]
    reference = gallery._interior_point

    for ring in rings:
        for vx, vy in ring[:-1]:                     # last coord repeats the first
            dx, dy = reference.x - vx, reference.y - vy
            norm = math.hypot(dx, dy) or 1.0
            cx, cy = vx + dx / norm * inward, vy + dy / norm * inward
            if gallery.contains(cx, cy):
                points.append((cx, cy))

    # --- uniform interior candidates ---
    minx, miny, maxx, maxy = gallery.bounds
    attempts = 0
    while len(points) < len(rings) + n_random and attempts < n_random * 40:
        attempts += 1
        px = rng.uniform(minx, maxx)
        py = rng.uniform(miny, maxy)
        if gallery.polygon.contains(Point(px, py)):
            points.append((px, py))

    return points


# ---------------------------------------------------------------------------
# Greedy cover
# ---------------------------------------------------------------------------
def greedy_cover(gallery: Gallery,
                 threshold: float = 0.98,
                 max_guards: int = 16,
                 n_random: int = 40,
                 rng: Optional[np.random.Generator] = None) -> Tuple[Points, float]:
    """
    Classic greedy maximum-coverage: repeatedly take the guard that adds the
    most currently-unseen area, until the gallery is covered or the budget runs
    out. Returns (guard positions, exact coverage achieved).

    The gallery is left with the chosen guards placed, so the caller can read
    off any further exact quantities.
    """
    gallery.reset_guards()
    candidates = candidate_points(gallery, n_random=n_random, rng=rng)
    if not candidates:
        return [], 0.0

    # Precompute each candidate's raster footprint once; the selection loop is
    # then pure numpy and costs nothing next to the visibility computations.
    masks = [gallery.rasterise(gallery.visibility_polygon(x, y)) for x, y in candidates]

    covered = np.zeros_like(gallery.interior_mask, dtype=bool)
    total = gallery.interior_pixels
    chosen: Points = []
    available = set(range(len(candidates)))

    while len(chosen) < max_guards and covered.sum() / total < threshold:
        best_index, best_gain = None, 0
        for index in available:
            gain = int(np.count_nonzero(masks[index] & ~covered))
            if gain > best_gain:
                best_index, best_gain = index, gain

        if best_index is None:      # nothing left adds anything new
            break

        covered |= masks[best_index]
        available.discard(best_index)
        chosen.append(candidates[best_index])

    # Replay the selection exactly, so the reported coverage is not a raster
    # approximation and the gallery ends up in the corresponding state.
    gallery.reset_guards()
    for x, y in chosen:
        gallery.add_guard(x, y)
    return chosen, gallery.coverage


# ---------------------------------------------------------------------------
# Greedy prune
# ---------------------------------------------------------------------------
def greedy_prune(gallery: Gallery, threshold: float = 0.98,
                 tolerance: float = 0.0) -> int:
    """
    Repeatedly delete the cheapest guard while coverage stays above threshold.

    `tolerance` optionally allows a deletion to cost that fraction of the
    gallery even if it is not entirely free, which is how the oracle trades a
    sliver of coverage for a whole guard. Returns the number of guards removed;
    the gallery is mutated in place.
    """
    removed = 0
    while gallery.n_guards > 0:
        costs = [gallery.exact_exclusive_area(i) for i in range(gallery.n_guards)]
        cheapest = int(np.argmin(costs))

        # Refuse the deletion if it would push us under the goal, unless it is
        # within the explicit tolerance.
        if gallery.coverage - costs[cheapest] < threshold and costs[cheapest] > tolerance:
            break

        gallery.remove_guard(cheapest)
        removed += 1
    return removed


def greedy_solution(record: dict, threshold: float = 0.98, max_guards: int = 16,
                    img_size: int = 64, seed: int = 0) -> dict:
    """
    Full greedy-then-prune pipeline for one gallery record.

    Returns the reference guard count k_ref (post-prune), the guard positions
    BEFORE pruning (the saturated start state for stage 2), and the coverage
    each phase reached.
    """
    rng = np.random.default_rng(seed)
    gallery = Gallery(record, img_size=img_size)

    guards, cover_coverage = greedy_cover(gallery, threshold, max_guards, rng=rng)
    greedy_prune(gallery, threshold)

    return {
        "k_ref": max(gallery.n_guards, 1),
        "k_greedy": len(guards),
        "greedy_guards": [[float(x), float(y)] for x, y in guards],
        "greedy_coverage": float(cover_coverage),
        "pruned_coverage": float(gallery.coverage),
    }
