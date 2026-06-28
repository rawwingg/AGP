# Source - https://stackoverflow.com/a/25276331
# Posted by Mike Ounsworth, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-20, License - CC BY-SA 4.0

import math, random
from typing import List, Tuple

from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points


# Source - https://stackoverflow.com/a/25276331
# Posted by Mike Ounsworth, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-20, License - CC BY-SA 4.0

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles


# Source - https://stackoverflow.com/a/25276331
# Posted by Mike Ounsworth, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-20, License - CC BY-SA 4.0

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))


def random_point_in_polygon(polygon: Polygon, max_attempts: int = 1000) -> Point:
    """Pick a random point inside a Shapely polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    for _ in range(max_attempts):
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            return point

    return polygon.representative_point()


def add_holes_to_polygon(
    outer_points,
    num_holes=1,
    hole_radius_range=(0.5, 2.0),
    hole_vertices_range=(4, 8),
    hole_irregularity_range=(0.05, 0.20),
    hole_spikeyness_range=(0.02, 0.10),
    clearance=0.2,
    max_attempts=200
):
    """
    Add random validated holes to an existing outer polygon point list.
    """
    outer_polygon = Polygon(outer_points)
    

    placement_area = outer_polygon.buffer(-clearance)
    if placement_area.is_empty:
        return outer_points, [], outer_polygon

    holes = []
    hole_polygons = []
    attempts = 0
    min_hole_area = 1 #we can adjust this threshold based on the expected scale of the polygons, or to create larger holes 

    while len(holes) < num_holes and attempts < max_attempts:
        attempts += 1

        center = random_point_in_polygon(placement_area)
        avg_radius = random.uniform(*hole_radius_range)
        num_vertices = random.randint(*hole_vertices_range)
        irregularity = random.uniform(*hole_irregularity_range)
        spikiness = random.uniform(*hole_spikeyness_range)

        candidate_points = generate_polygon(
            center=(center.x, center.y),
            avg_radius=avg_radius,
            irregularity=irregularity,
            spikiness=spikiness,
            num_vertices=num_vertices
        )
        candidate_polygon = Polygon(candidate_points)

        if not candidate_polygon.is_valid:
            continue
        if candidate_polygon.area <= min_hole_area:
            continue
        if not outer_polygon.contains(candidate_polygon):
            continue
        if outer_polygon.exterior.distance(candidate_polygon) <= clearance:
            continue
        if not placement_area.contains(candidate_polygon):
            continue
        if any(candidate_polygon.distance(hole) <= clearance for hole in hole_polygons):
            continue

        test_holes = holes + [candidate_points]
        test_polygon = Polygon(outer_points, holes=test_holes)
        if not test_polygon.is_valid:
            continue

        holes.append(candidate_points)
        hole_polygons.append(candidate_polygon)

    return outer_points, holes, Polygon(outer_points, holes=holes)


def save_polygon_preview(vertices: List[Tuple[float, float]],
                         holes=None,
                         output_path: str = "polygon_test.png",
                         image_size: Tuple[int, int] = (500, 500)) -> None:
    """Draw a polygon preview image so the generator can be tested quickly."""
    black = (0, 0, 0)
    white = (255, 255, 255)
    img = Image.new("RGB", image_size, white)
    draw = ImageDraw.Draw(img)

    draw.polygon(vertices, outline=black, fill=white)
    for hole in holes or []:
        draw.polygon(hole, outline=black, fill=white)
    draw.line(vertices + [vertices[0]], width=2, fill=black)
    for hole in holes or []:
        draw.line(hole + [hole[0]], width=2, fill=black)
    img.save(output_path)


if __name__ == "__main__":
    vertices = generate_polygon(center=(250, 250),
                                avg_radius=200,
                                irregularity=0.25,
                                spikiness=0.12,
                                num_vertices=14)

    outer_points, holes, polygon_with_holes = add_holes_to_polygon(
        vertices,
        num_holes=3,
        hole_radius_range=(10, 30),
        clearance=8
    )

    save_polygon_preview(outer_points, holes)
    print("Generated  vertices:")
    for vertex in vertices:
        print(f"  ({vertex[0]:.2f}, {vertex[1]:.2f})")
    print(f"Generated {len(holes)} holes.")
    print(f"Final Shapely polygon valid: {polygon_with_holes.is_valid}")
    print("Saved preview to polygon_test.png")
