"""
Gallery geometry: exact visibility polygons and coverage bookkeeping.

This is the "physics" of the MDP. Everything the environment needs to know about
a gallery and its guards lives here, and nothing in this file knows about
reinforcement learning.

Two representations are maintained side by side, on purpose:

  * EXACT (Shapely polygons). Used for every quantity that enters the reward --
    coverage fraction, the area a new guard adds, the area lost when a guard is
    deleted. Exact means the reward is not quantised by the raster.

  * RASTER (numpy boolean masks on an IMG_SIZE x IMG_SIZE grid). Used for the
    agent's observation, and for the cheap per-guard redundancy estimates shown
    to the agent each step. Recomputing every guard's exact exclusive area every
    step would cost O(k) polygon unions; on the raster it is a single numpy
    operation per guard.

Notation follows docs/agp_mdp_formulation.pdf:
    delta_plus   the new area a guard adds        (marginal gain)
    delta_minus  the area lost if a guard is cut  (redundancy deficit)
Both are reported as fractions of the gallery area, so they are scale-free.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid


# Angular epsilon used to shoot a ray just either side of each vertex, so the
# sweep captures both edges meeting at an occluding (reflex) corner.
_RAY_EPS = 1e-6


class Gallery:
    """
    One gallery instance plus the guards currently placed in it.

    A `Gallery` is created fresh at every episode reset, so it owns all mutable
    episode state: the guard list, their cached visibility polygons and masks,
    the exact covered region, and the per-pixel coverage count.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------
    def __init__(self, record: dict, img_size: int = 64):
        self.record = record
        self.img_size = img_size

        # --- static geometry -------------------------------------------------
        self.polygon = Polygon(record["outer_points"], holes=record["holes"])
        if not self.polygon.is_valid:
            self.polygon = make_valid(self.polygon)
        self.area = float(self.polygon.area)
        self.bounds = self.polygon.bounds                      # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = self.bounds
        self.diagonal = math.hypot(maxx - minx, maxy - miny)
        self.boundary = self.polygon.boundary                  # exterior + hole rings
        self._interior_point = self.polygon.representative_point()

        # Rays must be long enough to always escape the gallery.
        self._ray_length = max(self.diagonal * 3.0, 1.0)

        # Every vertex of the gallery, outer ring and holes alike. These are the
        # directions the visibility sweep needs to sample, held as flat arrays
        # so the sweep can be done in numpy rather than a Python loop.
        vertices = list(self.polygon.exterior.coords)
        for ring in self.polygon.interiors:
            vertices.extend(ring.coords)
        self._vertex_x = np.array([v[0] for v in vertices], dtype=np.float64)
        self._vertex_y = np.array([v[1] for v in vertices], dtype=np.float64)

        # --- raster grid -----------------------------------------------------
        self._build_grid()

        # --- mutable guard state --------------------------------------------
        self.reset_guards()

    def _build_grid(self):
        """Rasterise the gallery once: which pixels of the bbox are inside it."""
        minx, miny, maxx, maxy = self.bounds
        xs = np.linspace(minx, maxx, self.img_size)
        ys = np.linspace(miny, maxy, self.img_size)
        gx, gy = np.meshgrid(xs, ys)
        self._grid_x = gx.ravel()
        self._grid_y = gy.ravel()

        inside = shapely.contains_xy(self.polygon, self._grid_x, self._grid_y)
        self.interior_mask = np.asarray(inside, dtype=bool).reshape(self.img_size, self.img_size)
        # Guard against a pathological gallery that rasterises to nothing.
        self.interior_pixels = max(int(self.interior_mask.sum()), 1)

        # Rasterising a visibility polygon only ever needs the pixels that are
        # inside the gallery -- typically about 60% of the bounding box -- so
        # keep just those and skip the rest on every subsequent query.
        self._inside_index = np.flatnonzero(self.interior_mask.ravel())
        self._inside_x = self._grid_x[self._inside_index]
        self._inside_y = self._grid_y[self._inside_index]

    def reset_guards(self):
        """Drop all guards and return to the empty-gallery state."""
        self.guards: List[Tuple[float, float]] = []      # guard positions
        self._vis: List[Polygon] = []                    # cached visibility polygons
        self._masks: List[np.ndarray] = []               # cached raster masks
        self.covered = Polygon()                         # exact union of visibility
        self._count = np.zeros((self.img_size, self.img_size), dtype=np.int16)

    # -----------------------------------------------------------------------
    # Point queries
    # -----------------------------------------------------------------------
    def contains(self, x: float, y: float) -> bool:
        """True if (x, y) is strictly inside the gallery (not in a hole)."""
        return self.polygon.contains(Point(x, y))

    def project(self, x: float, y: float) -> Tuple[float, float, bool]:
        """
        Snap (x, y) into the gallery.

        The action space is the gallery's bounding box, which includes points
        outside the polygon and inside holes. Rather than rejecting those (which
        would leave the state unchanged and let a deterministic policy deadlock
        by repeating the same bad action forever), we project to the nearest
        interior point. Returns (x, y, was_outside).
        """
        point = Point(x, y)
        if self.polygon.contains(point):
            return x, y, False

        # Nearest point on the gallery -- lies exactly on the boundary.
        nearest = nearest_points(self.polygon, point)[0]
        nx, ny = nearest.x, nearest.y

        # Nudge inward along the line to a known-interior point until strictly
        # inside, so the guard does not sit exactly on an edge.
        ref = self._interior_point
        for t in (0.01, 0.05, 0.10, 0.25, 0.50):
            cx = nx + (ref.x - nx) * t
            cy = ny + (ref.y - ny) * t
            if self.polygon.contains(Point(cx, cy)):
                return cx, cy, True
        return ref.x, ref.y, True

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest wall or hole edge."""
        return float(self.boundary.distance(Point(x, y)))

    def is_covered(self, x: float, y: float) -> bool:
        """True if (x, y) is already visible to at least one placed guard."""
        return (not self.covered.is_empty) and self.covered.contains(Point(x, y))

    def nearest_guard(self, x: float, y: float) -> Optional[int]:
        """Index of the guard closest to (x, y), or None if there are none."""
        if not self.guards:
            return None
        return min(range(len(self.guards)),
                   key=lambda i: (self.guards[i][0] - x) ** 2 + (self.guards[i][1] - y) ** 2)

    def distance_to_nearest_guard(self, x: float, y: float) -> float:
        """Distance to the closest existing guard (inf when the gallery is empty)."""
        if not self.guards:
            return float("inf")
        return min(math.hypot(x - gx, y - gy) for gx, gy in self.guards)

    # -----------------------------------------------------------------------
    # Visibility
    # -----------------------------------------------------------------------
    def visibility_polygon(self, x: float, y: float) -> Polygon:
        """
        Exact visibility polygon of a guard at (x, y), by angular ray sweep.

        A ray is cast towards every gallery vertex, plus one just either side of
        it, and truncated at the first boundary hit. Taking the hits in angular
        order traces the visible region; clipping against the gallery keeps the
        result well behaved when a ray grazes an edge.

        The whole sweep is vectorised. This is the single most expensive
        operation in the environment -- it runs on every Add -- and doing the
        few hundred ray/boundary intersections as one batched Shapely call
        instead of a Python loop is worth roughly a threefold speedup.
        """
        if not self.polygon.covers(Point(x, y)):
            return Polygon()

        # --- 1. directions to sweep ---
        dx = self._vertex_x - x
        dy = self._vertex_y - y
        # Drop any vertex the guard is standing on; its direction is undefined.
        keep = (dx * dx + dy * dy) > 1e-18
        if keep.sum() < 3:
            return Polygon()

        base = np.arctan2(dy[keep], dx[keep])
        # One ray at the vertex and one just either side, so a reflex corner
        # yields both the near wall and the surface behind it.
        angles = np.unique(np.concatenate([base - _RAY_EPS, base, base + _RAY_EPS]))

        # --- 2. first boundary hit along each ray ---
        far_x = x + np.cos(angles) * self._ray_length
        far_y = y + np.sin(angles) * self._ray_length

        # Build all rays in one go: interleave origin and endpoint, then tell
        # Shapely that every consecutive pair is its own two-point line.
        n = angles.size
        points = np.empty((2 * n, 2), dtype=np.float64)
        points[0::2, 0] = x
        points[0::2, 1] = y
        points[1::2, 0] = far_x
        points[1::2, 1] = far_y
        rays = shapely.linestrings(points, indices=np.repeat(np.arange(n), 2))

        hits = shapely.intersection(rays, self.boundary)
        coords, source = shapely.get_coordinates(hits, return_index=True)
        if coords.shape[0] < 3:
            return Polygon()

        # --- 3. keep the nearest hit per ray ---
        # Sort by (ray, distance) and take the first row of each ray's block.
        squared = (coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2
        order = np.lexsort((squared, source))
        ordered_source = source[order]
        is_first = np.empty(ordered_source.size, dtype=bool)
        is_first[0] = True
        np.not_equal(ordered_source[1:], ordered_source[:-1], out=is_first[1:])
        # `angles` was sorted, so ray index order is already angular order.
        nearest = coords[order[is_first]]

        # Drop consecutive duplicates, which the three-rays-per-vertex scheme
        # produces wherever a corner is not actually occluding.
        differs = np.empty(nearest.shape[0], dtype=bool)
        differs[0] = True
        np.any(np.abs(np.diff(nearest, axis=0)) > 1e-9, axis=1, out=differs[1:])
        nearest = nearest[differs]
        if nearest.shape[0] < 3:
            return Polygon()

        visible = Polygon(nearest)
        if not visible.is_valid:
            visible = make_valid(visible)

        # --- 4. clip back to the gallery ---
        try:
            clipped = visible.intersection(self.polygon)
        except Exception:
            return Polygon()
        return _largest_polygonal_part(clipped)

    # -----------------------------------------------------------------------
    # Adding and removing guards
    # -----------------------------------------------------------------------
    def add_guard(self, x: float, y: float) -> float:
        """
        Place a guard at (x, y) and return delta_plus, the fraction of the
        gallery it newly covers (overlap with existing coverage excluded).
        """
        visible = self.visibility_polygon(x, y)
        new_area = visible.difference(self.covered).area if not visible.is_empty else 0.0

        self.guards.append((x, y))
        self._vis.append(visible)
        self.covered = self.covered.union(visible)

        # Keep the raster in step: mask for this guard, and the per-pixel count.
        mask = self.rasterise(visible).astype(np.int16)
        self._masks.append(mask)
        self._count += mask

        return new_area / self.area

    def remove_guard(self, index: int) -> float:
        """
        Delete guard `index` and return delta_minus, the fraction of the gallery
        that stopped being covered. Coverage is recomputed exactly.
        """
        before = self.covered.area

        self.guards.pop(index)
        self._vis.pop(index)
        self._count -= self._masks.pop(index)

        self.covered = unary_union(self._vis) if self._vis else Polygon()
        return (before - self.covered.area) / self.area

    def rasterise(self, polygon: Polygon) -> np.ndarray:
        """Boolean mask of the grid pixels inside `polygon`, clipped to the gallery."""
        mask = np.zeros(self.img_size * self.img_size, dtype=bool)
        if polygon.is_empty:
            return mask.reshape(self.img_size, self.img_size)
        hit = shapely.contains_xy(polygon, self._inside_x, self._inside_y)
        mask[self._inside_index] = np.asarray(hit, dtype=bool)
        return mask.reshape(self.img_size, self.img_size)

    # -----------------------------------------------------------------------
    # Derived quantities
    # -----------------------------------------------------------------------
    @property
    def n_guards(self) -> int:
        return len(self.guards)

    @property
    def coverage(self) -> float:
        """Exact covered fraction of the gallery, in [0, 1]."""
        if self.covered.is_empty:
            return 0.0
        return min(self.covered.area / self.area, 1.0)

    def exclusive_areas(self) -> np.ndarray:
        """
        Raster estimate of delta_minus for every guard, as area fractions.

        A pixel contributes to guard j only if guard j is the ONLY guard that
        sees it, so `_count == 1` isolates each guard's exclusive region in a
        single vectorised pass. Cheap enough to call every step.
        """
        if not self.guards:
            return np.zeros(0, dtype=np.float32)
        sole = self._count == 1
        return np.array(
            [float((mask.astype(bool) & sole).sum()) / self.interior_pixels
             for mask in self._masks],
            dtype=np.float32,
        )

    def exact_exclusive_area(self, index: int) -> float:
        """
        Exact delta_minus for one guard: the area lost if it were deleted.

        Used for the reward when a deletion actually happens; the raster
        estimate above is only for the observation.
        """
        if not self.guards:
            return 0.0
        remaining = [v for i, v in enumerate(self._vis) if i != index]
        without = unary_union(remaining).area if remaining else 0.0
        return max(self.covered.area - without, 0.0) / self.area

    # -----------------------------------------------------------------------
    # Observation channels
    # -----------------------------------------------------------------------
    def covered_mask(self) -> np.ndarray:
        """Pixels currently seen by at least one guard."""
        return self._count > 0

    def uncovered_mask(self) -> np.ndarray:
        """Pixels inside the gallery that no guard sees yet -- the work left."""
        return self.interior_mask & (self._count == 0)

    def guard_mask(self) -> np.ndarray:
        """
        Guard positions as a raster, so the policy can 'point at' a guard.

        Each guard is drawn as a small 3x3 blob rather than a single pixel,
        which survives the convolutional downsampling far better.
        """
        mask = np.zeros((self.img_size, self.img_size), dtype=bool)
        for x, y in self.guards:
            row, col = self._pixel_of(x, y)
            mask[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2] = True
        return mask

    def redundancy_mask(self, sigma: float,
                        deficits: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Per-guard redundancy, painted at each guard's position and scaled to
        0..255. Dark means 'nearly free to delete', bright means 'expensive'.

        `sigma` is the pruning tolerance scale from the reward weights, so the
        channel is calibrated to the same units the reward uses. Pass `deficits`
        if the caller has already computed them.
        """
        image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        if deficits is None:
            deficits = self.exclusive_areas()
        for (x, y), deficit in zip(self.guards, deficits):
            row, col = self._pixel_of(x, y)
            value = int(255 * min(1.0, deficit / max(sigma, 1e-9)))
            image[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2] = value
        return image

    def _pixel_of(self, x: float, y: float) -> Tuple[int, int]:
        """Map a gallery coordinate to its (row, col) pixel in the raster."""
        minx, miny, maxx, maxy = self.bounds
        col = int(round((x - minx) / max(maxx - minx, 1e-9) * (self.img_size - 1)))
        row = int(round((y - miny) / max(maxy - miny, 1e-9) * (self.img_size - 1)))
        return (int(np.clip(row, 0, self.img_size - 1)),
                int(np.clip(col, 0, self.img_size - 1)))


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def _largest_polygonal_part(geometry) -> Polygon:
    """Reduce an intersection result to a single polygon (empty if there is none)."""
    if geometry.is_empty:
        return Polygon()
    if geometry.geom_type == "Polygon":
        return geometry
    if hasattr(geometry, "geoms"):
        parts = [g for g in geometry.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if parts:
            return unary_union(parts)
    return Polygon()
