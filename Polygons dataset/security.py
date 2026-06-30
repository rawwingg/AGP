import json
import math
from pathlib import Path
import random
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
from shapely.validation import make_valid


LOCAL_DATASET_PATH = Path("Polygons") / "local_dataset.json"

class Guard:
    def __init__(self, x, y):
        self.position = Point(x, y)

    def set_position(self, x, y):
        self.position = Point(x, y)
    
    def get_position(self):
        return self.position.x, self.position.y

    def __eq__(self, other):
        if isinstance(other, Guard):
            return self.position == other.position
        return False

    def __repr__(self):
        return f"Guard at ({self.position.x}, {self.position.y})"

class Security:

    def __init__(self, poly, radius): #Initialize the Security class with the polygon data

        self.perimeter = Polygon(poly['outer_points'], holes=poly['holes'])
        self.area = poly['area']
        self.all_guards = dict()
        self.coverage_areas = Polygon()  # total covered region, starts empty
        self.percent_coverage = 0.0
        self.range = radius

    def add_guard(self, guard): #step function to add a guard to the list of guards
        if self.perimeter.contains(guard.position):
            area_coverage = self.get_area_coverage_of_guard(guard, radius = self.range)
            # area this guard adds that was not already covered (the RL "+new area" term)
            new_area = area_coverage.difference(self.coverage_areas).area
            self.all_guards[guard.get_position()] = area_coverage
            self.coverage_areas = self.coverage_areas.union(area_coverage)
            self.percent_coverage = (self.coverage_areas.area / self.perimeter.area)
            return new_area
        else:
            raise ValueError("Guard position is outside the polygon perimeter.")

    def get_area_coverage_of_guard(self, guard, radius = None, deg = 3):
        if not self.perimeter.covers(guard.position):
            return Polygon()  # empty

        # collect all polygon vertices (exterior + holes)
        all_vertices = list(self.perimeter.exterior.coords)
        for hole in self.perimeter.interiors:
            all_vertices.extend(list(hole.coords))

        x, y = guard.get_position()

        max_dist = None

        # ray distance: large enough to reach polygon boundary
        if radius is None:
            if self.range is None:
                bminx, bminy, bmaxx, bmaxy = self.perimeter.bounds
                max_dist = math.hypot(bmaxx - bminx, bmaxy - bminy) * 3.0
                if max_dist <= 0:
                    max_dist = 1000.0
            else:
                max_dist = self.range
        else:
            max_dist = radius

        # angles to cast rays (vertex angles ± small epsilon)

        eps = math.radians(deg)

        it = math.ceil(360/deg)

        hits = []
        for i in range(0, it):
            angle = eps * i
            ex = x + math.cos(angle) * max_dist
            ey = y + math.sin(angle) * max_dist
            ray = LineString([guard.position, Point(ex, ey)])

            # intersect with polygon boundary (prefer boundary for clean intersections)
            inter = ray.intersection(self.perimeter.boundary)

            if inter.is_empty:
                if Point(ex,ey).within(self.perimeter):
                    hits.append((angle,(ex,ey)))
                else:
                    continue

            # gather candidate points from intersection geometry
            candidates = []
            if inter.geom_type == "Point":
                candidates.append((inter.x, inter.y))
            elif inter.geom_type == "MultiPoint":
                for g in inter.geoms:
                    candidates.append((g.x, g.y))
            elif inter.geom_type == "LineString":
                candidates.extend(list(inter.coords))
            elif inter.geom_type == "MultiLineString":
                for g in inter.geoms:
                    candidates.extend(list(g.coords))
            else:
                # fallback: try to iterate geoms if available
                if hasattr(inter, "geoms"):
                    for g in inter.geoms:
                        if getattr(g, "geom_type", "") == "Point":
                            candidates.append((g.x, g.y))
                        elif getattr(g, "geom_type", "") == "LineString":
                            candidates.extend(list(g.coords))

            # pick the closest candidate to the guard along this ray
            best = None
            bestd = float("inf")
            for cx, cy in candidates:
                d = (cx - x) ** 2 + (cy - y) ** 2
                if d < bestd:
                    bestd = d
                    best = (cx, cy)

            if best is not None:
                hits.append((angle, best))

        if not hits:
            return Polygon()

        # sort by angle and deduplicate nearly-equal points
        hits.sort(key=lambda t: t[0])
        coords = []
        seen = set()
        for _angle, (px, py) in hits:
            key = (round(px, 8), round(py, 8))
            if key in seen:
                continue
            seen.add(key)
            coords.append((px, py))

        # must have at least 3 points for a polygon
        if len(coords) < 3:
            return Polygon()

        new_poly = Polygon(coords)
        if not new_poly.is_valid:
            new_poly = make_valid(new_poly)

        # clip to the perimeter to be safe
        try:
            clipped = new_poly.intersection(self.perimeter)
        except Exception:
            clipped = new_poly

        if clipped.is_empty:
            return Polygon()
        # if the result is a polygon or multipolygon, prefer polygon (union if needed)
        if clipped.geom_type == "Polygon":
            return clipped
        elif clipped.geom_type in ("MultiPolygon", "GeometryCollection"):
            # return the unary_union of polygon parts
            poly_parts = [g for g in getattr(clipped, "geoms", []) if g.geom_type == "Polygon"]
            if not poly_parts:
                return Polygon()
            return unary_union(poly_parts)
        else:
            return Polygon()
    
    def remove_guard(self, guard): #step function to remove a guards from the list of guards
        if guard.get_position() in self.all_guards:
            del self.all_guards[guard.get_position()]
            self.coverage_areas = unary_union(list(self.all_guards.values()))
            self.percent_coverage = (self.coverage_areas.area / self.perimeter.area)
            
    def remove_all_guards(self): #step function to remove all guards from the list of guards
        self.all_guards.clear()
        self.coverage_areas = Polygon()  # reset to empty coverage
        self.percent_coverage = 0.0

    def get_all_guards(self): #step function to return the list of guards
        return dict(zip(self.all_guards.keys(), [area.exterior.coords for area in self.all_guards.values()]))
    
    def set_range(self, range):
        self.range = range

    def __repr__(self):
        return f"Area = {self.area}, No of Guards = {len(self.all_guards)}, Area Coverage percentage = {self.percent_coverage * 100:.2f}%"
    
def new_dataset_with_guards(original_dataset_path = LOCAL_DATASET_PATH, new_dataset_path = Path("Polygons") / "local_dataset_with_guards.json", min_guards_per_polygon = 1, max_guards_per_polygon = 2, radius = None):

    new_polygons = []
    
    with open(original_dataset_path, 'r') as f:
        polygons = json.load(f)

    for polygon in polygons:
        security = Security(polygon, radius)
        polygon['guards'] = []
        guards_per_polygon = random.randint(min_guards_per_polygon, max_guards_per_polygon)
        for _ in range(guards_per_polygon):
            while True:
                x = random.uniform(security.perimeter.bounds[0], security.perimeter.bounds[2])
                y = random.uniform(security.perimeter.bounds[1], security.perimeter.bounds[3])
                guard = Guard(x, y)
                try:
                    security.add_guard(guard)
                    break
                except ValueError:
                    continue
            polygon['guards'].append((x,y))
        polygon["area_coverage"] = security.percent_coverage
        polygon["guard range"] = radius
        new_polygons.append(polygon)

    with open(new_dataset_path, 'w') as f:
        json.dump(new_polygons, f, indent=4)

def sort_points_clockwise(points, centre):
    if not points:
        return []
    
    sorted_points = sorted(points, key=lambda point: math.atan2(point[1] - centre[1], point[0] - centre[0]))
    
    return sorted_points

if __name__ == "__main__":
    new_dataset_with_guards(min_guards_per_polygon=1, max_guards_per_polygon=3, radius = 200)