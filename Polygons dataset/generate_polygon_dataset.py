import json
import random
from pathlib import Path

from generate_polygons import add_holes_to_polygon, generate_polygon


DEFAULT_OUTPUT_PATH = Path("Polygons") / "polygons.json"
# Brozers, use local_dataset.json path for testing (this wont be pushed to GitHub) otherwise use default
LOCAL_DATASET_PATH = Path("Polygons") / "local_dataset.json"

def points_to_json(points):
    """Convert tuples from the geometry functions into JSON-friendly lists."""
    return [[x, y] for x, y in points]


# Placeholder for later project-specific validation.
# def validate_polygon_record(outer_points, holes, shapely_polygon):
#     pass


def generate_polygons_with_holes(count, output_path=DEFAULT_OUTPUT_PATH):
    polygons = []

    for polygon_id in range(1, count + 1):
        outer_points = generate_polygon(
            center=(250, 250),
            avg_radius=random.uniform(120, 190),
            irregularity=random.uniform(0.15, 0.35),
            spikiness=random.uniform(0.08, 0.18),
            num_vertices=random.randint(10, 16)
        )

        outer_points, holes, shapely_polygon = add_holes_to_polygon(
            outer_points,
            num_holes=random.randint(1, 4),
            hole_radius_range=(10, 30),
            hole_vertices_range=(4, 8),
            hole_irregularity_range=(0.05, 0.20),
            hole_spikeyness_range=(0.02, 0.10),
            clearance=8,
            max_attempts=300
        )

        # TODO: Call validate_polygon_record(outer_points, holes, shapely_polygon)
        # once the validation rules for the project are finalized.

        polygons.append({
            "id": polygon_id,
            "outer_points": points_to_json(outer_points),
            "holes": [points_to_json(hole) for hole in holes],
            "is_valid": shapely_polygon.is_valid,
            "area": shapely_polygon.area
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(polygons, indent=2), encoding="utf-8")
    return polygons


if __name__ == "__main__":
    generated = generate_polygons_with_holes(10,LOCAL_DATASET_PATH)
    print(f"Saved {len(generated)} polygons to {LOCAL_DATASET_PATH}")
