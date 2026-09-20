"""
Random gallery generation.

A gallery is a simple outer polygon with a set of disjoint polygonal holes.
Three difficulty tiers are produced: ``easy``, ``mid`` and ``hard``. The old
``extreme`` tier (24-40 vertices, spikiness up to 0.45, up to 14 holes) is
deliberately NOT generated here -- those galleries are so skewed that the
visibility polygons degenerate into slivers and the learning signal is dominated
by geometric noise rather than by the placement decision.

The core polygon sampler is adapted from a StackOverflow answer:
    Source  - https://stackoverflow.com/a/25276331
    Posted  - Mike Ounsworth, modified by community (see post 'Timeline')
    License - CC BY-SA 4.0
"""

import math
import random
from typing import List, Sequence, Tuple

from shapely.geometry import Point, Polygon


# ---------------------------------------------------------------------------
# Difficulty tiers
# ---------------------------------------------------------------------------
# Each tier is a plain dict of sampling ranges, so adding or retuning a tier is
# a data change rather than a code change. `radius` etc. are sampled uniformly
# from the given (low, high) range; `vertices` and `holes` are integer ranges.
TIERS = {
    "easy": {
        "radius": (120, 180),
        "irregularity": (0.10, 0.25),
        "spikiness": (0.05, 0.12),
        "vertices": (8, 12),
        "holes": (1, 3),
        "hole_radius": (12, 28),
        "hole_vertices": (4, 6),
        "hole_irregularity": (0.03, 0.12),
        "hole_spikiness": (0.01, 0.06),
        "clearance": 10,
    },
    "mid": {
        "radius": (120, 190),
        "irregularity": (0.15, 0.35),
        "spikiness": (0.08, 0.18),
        "vertices": (10, 16),
        "holes": (2, 5),
        "hole_radius": (16, 38),
        "hole_vertices": (4, 8),
        "hole_irregularity": (0.05, 0.20),
        "hole_spikiness": (0.02, 0.10),
        "clearance": 8,
    },
    "hard": {
        "radius": (130, 200),
        "irregularity": (0.30, 0.55),
        "spikiness": (0.16, 0.32),
        "vertices": (16, 26),
        "holes": (5, 9),
        "hole_radius": (22, 55),
        "hole_vertices": (5, 10),
        "hole_irregularity": (0.10, 0.35),
        "hole_spikiness": (0.05, 0.18),
        "clearance": 6,
    },
}

TIER_NAMES = tuple(TIERS)
CENTER = (250.0, 250.0)


# ---------------------------------------------------------------------------
# Outer boundary sampler
# ---------------------------------------------------------------------------
def _random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Split a full turn into `steps` random angles that still sum to 2*pi."""
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    total = 0.0
    for _ in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        total += angle

    # Renormalise so the walk around the circle closes exactly.
    total /= 2 * math.pi
    return [a / total for a in angles]


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Sample a star-shaped polygon by walking around a circle.

    `irregularity` (0..1) varies the angular spacing between consecutive
    vertices; `spikiness` (0..1) varies each vertex's distance from the centre.
    Returns vertices in counter-clockwise order.
    """
    if not 0 <= irregularity <= 1:
        raise ValueError("irregularity must be in [0, 1]")
    if not 0 <= spikiness <= 1:
        raise ValueError("spikiness must be in [0, 1]")

    # Convert the normalised knobs into absolute angular / radial variances.
    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = _random_angle_steps(num_vertices, irregularity)

    points = []
    angle = random.uniform(0, 2 * math.pi)
    for step in angle_steps:
        radius = min(2 * avg_radius, max(0.0, random.gauss(avg_radius, spikiness)))
        points.append((center[0] + radius * math.cos(angle),
                       center[1] + radius * math.sin(angle)))
        angle += step
    return points


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------
def _random_point_in(polygon: Polygon, max_attempts: int = 1000) -> Point:
    """Rejection-sample a point inside `polygon`, falling back to a safe point."""
    min_x, min_y, max_x, max_y = polygon.bounds
    for _ in range(max_attempts):
        candidate = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(candidate):
            return candidate
    return polygon.representative_point()


def add_holes(outer_points: Sequence[Tuple[float, float]], spec: dict,
              max_attempts: int = 400):
    """
    Punch validated holes into an outer boundary.

    A candidate hole is accepted only if it is valid, non-trivial in area, fully
    inside the outer polygon, at least `clearance` away from the outer boundary
    and from every hole placed so far, and leaves the combined polygon valid.
    Rejected candidates are simply resampled, so the requested hole count is a
    target rather than a guarantee.

    Returns (outer_points, holes, shapely_polygon).
    """
    outer_polygon = Polygon(outer_points)
    clearance = spec["clearance"]

    # Holes must sit strictly inside this eroded copy of the gallery.
    placement_area = outer_polygon.buffer(-clearance)
    if placement_area.is_empty:
        return list(outer_points), [], outer_polygon

    target = random.randint(*spec["holes"])
    holes: List[List[Tuple[float, float]]] = []
    hole_polygons: List[Polygon] = []

    for _ in range(max_attempts):
        if len(holes) >= target:
            break

        centre = _random_point_in(placement_area)
        candidate_points = generate_polygon(
            center=(centre.x, centre.y),
            avg_radius=random.uniform(*spec["hole_radius"]),
            irregularity=random.uniform(*spec["hole_irregularity"]),
            spikiness=random.uniform(*spec["hole_spikiness"]),
            num_vertices=random.randint(*spec["hole_vertices"]),
        )
        candidate = Polygon(candidate_points)

        # --- validity gauntlet ---
        if not candidate.is_valid or candidate.area <= 1.0:
            continue
        if not placement_area.contains(candidate):
            continue
        if outer_polygon.exterior.distance(candidate) <= clearance:
            continue
        if any(candidate.distance(other) <= clearance for other in hole_polygons):
            continue
        if not Polygon(outer_points, holes=holes + [candidate_points]).is_valid:
            continue

        holes.append(candidate_points)
        hole_polygons.append(candidate)

    return list(outer_points), holes, Polygon(outer_points, holes=holes)


# ---------------------------------------------------------------------------
# One gallery record
# ---------------------------------------------------------------------------
def generate_gallery(tier: str, gallery_id: int = 0) -> dict:
    """
    Generate a single gallery record for the given difficulty tier.

    The record is the plain-dict form used everywhere downstream (and stored in
    the dataset JSON):  id, tier, outer_points, holes, area.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIER_NAMES}")
    spec = TIERS[tier]

    # Retry until the sampler produces a valid, non-degenerate polygon. In
    # practice this almost always succeeds first time.
    for _ in range(50):
        outer = generate_polygon(
            center=CENTER,
            avg_radius=random.uniform(*spec["radius"]),
            irregularity=random.uniform(*spec["irregularity"]),
            spikiness=random.uniform(*spec["spikiness"]),
            num_vertices=random.randint(*spec["vertices"]),
        )
        outer, holes, polygon = add_holes(outer, spec)
        if polygon.is_valid and polygon.area > 0:
            return {
                "id": gallery_id,
                "tier": tier,
                "outer_points": [[float(x), float(y)] for x, y in outer],
                "holes": [[[float(x), float(y)] for x, y in hole] for hole in holes],
                "area": float(polygon.area),
            }

    raise RuntimeError(f"could not generate a valid {tier} gallery after 50 attempts")


# ---------------------------------------------------------------------------
# D4 augmentation
# ---------------------------------------------------------------------------
# The Art Gallery Problem is equivariant under the eight symmetries of the
# square (four rotations x optional reflection), so transforming a gallery at
# episode reset multiplies the effective dataset by 8 for free. This is our
# stand-in for the domain randomisation the locomotion papers use.
_D4 = [
    (1, 0, 0, 1),    # identity
    (0, -1, 1, 0),   # rotate 90
    (-1, 0, 0, -1),  # rotate 180
    (0, 1, -1, 0),   # rotate 270
    (-1, 0, 0, 1),   # mirror in x
    (1, 0, 0, -1),   # mirror in y
    (0, 1, 1, 0),    # mirror in y = x
    (0, -1, -1, 0),  # mirror in y = -x
]


def augment_record(record: dict, index: int) -> dict:
    """
    Return `record` under the `index`-th dihedral symmetry (index taken mod 8).

    Coordinates are transformed about the gallery centre, so the polygon keeps
    its position and scale and only its orientation changes. Area is invariant,
    so it is carried over unchanged.

    Every point-valued field moves together -- the boundary, the holes AND the
    precomputed greedy guard positions. Transforming the geometry but leaving
    the guards behind would silently turn the stored greedy solution into a
    random one.
    """
    a, b, c, d = _D4[index % len(_D4)]
    cx, cy = CENTER

    def transform(points):
        out = []
        for x, y in points:
            dx, dy = x - cx, y - cy
            out.append([cx + a * dx + b * dy, cy + c * dx + d * dy])
        return out

    augmented = dict(record)
    augmented["outer_points"] = transform(record["outer_points"])
    augmented["holes"] = [transform(hole) for hole in record["holes"]]
    if record.get("greedy_guards"):
        augmented["greedy_guards"] = transform(record["greedy_guards"])
    return augmented
