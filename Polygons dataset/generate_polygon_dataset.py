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


def generate_polygons_with_holes(count, difficulty):
    polygons = []

    for polygon_id in range(1, count + 1):
        if difficulty == "easy":
            outer_points = generate_polygon(
                center=(250, 250),
                avg_radius=random.uniform(120, 180),
                irregularity=random.uniform(0.10, 0.25),
                spikiness=random.uniform(0.05, 0.12),
                num_vertices=random.randint(8, 12)
            )
            hole_settings = {
                "num_holes": random.randint(1, 3),
                "hole_radius_range": (12, 28),
                "hole_vertices_range": (4, 6),
                "hole_irregularity_range": (0.03, 0.12),
                "hole_spikeyness_range": (0.01, 0.06),
                "clearance": 10,
                "max_attempts": 200,
            }
        elif difficulty == "mid":
            outer_points = generate_polygon(
                center=(250, 250),
                avg_radius=random.uniform(120, 190),
                irregularity=random.uniform(0.15, 0.35),
                spikiness=random.uniform(0.08, 0.18),
                num_vertices=random.randint(10, 16)
            )
            hole_settings = {
                "num_holes": random.randint(2, 5),
                "hole_radius_range": (16, 38),
                "hole_vertices_range": (4, 8),
                "hole_irregularity_range": (0.05, 0.20),
                "hole_spikeyness_range": (0.02, 0.10),
                "clearance": 8,
                "max_attempts": 300,
            }
        elif difficulty == "hard":
            outer_points = generate_polygon(
                center=(250, 250),
                avg_radius=random.uniform(130, 200),
                irregularity=random.uniform(0.30, 0.55),
                spikiness=random.uniform(0.16, 0.32),
                num_vertices=random.randint(16, 26)
            )
            hole_settings = {
                "num_holes": random.randint(5, 9),
                "hole_radius_range": (22, 55),
                "hole_vertices_range": (5, 10),
                "hole_irregularity_range": (0.10, 0.35),
                "hole_spikeyness_range": (0.05, 0.18),
                "clearance": 6,
                "max_attempts": 600,
            }
        elif difficulty == "extreme":
            outer_points = generate_polygon(
                center=(250, 250),
                avg_radius=random.uniform(140, 220),
                irregularity=random.uniform(0.45, 0.75),
                spikiness=random.uniform(0.25, 0.45),
                num_vertices=random.randint(24, 40)
            )
            hole_settings = {
                "num_holes": random.randint(8, 14),
                "hole_radius_range": (28, 70),
                "hole_vertices_range": (6, 12),
                "hole_irregularity_range": (0.18, 0.45),
                "hole_spikeyness_range": (0.08, 0.25),
                "clearance": 4,
                "max_attempts": 900,
            }
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        outer_points, holes, shapely_polygon = add_holes_to_polygon(
            outer_points,
            **hole_settings
        )

        # TODO: Call validate_polygon_record(outer_points, holes, shapely_polygon)
        # once the validation rules for the project are finalized.

        polygons.append({
            "id": polygon_id,
            "difficulty": difficulty,
            "outer_points": points_to_json(outer_points),
            "holes": [points_to_json(hole) for hole in holes],
            "is_valid": shapely_polygon.is_valid,
            "area": shapely_polygon.area
        })

    return polygons


def generate_easy_polygons(count=10):
    return generate_polygons_with_holes(count, "easy")


def generate_mid_polygons(count=10):
    return generate_polygons_with_holes(count, "mid")


def generate_hard_polygons(count=10):
    return generate_polygons_with_holes(count, "hard")


def generate_extreme_polygons(count=10):
    return generate_polygons_with_holes(count, "extreme")


def generate_dataset(total_count, output_path=None):
    """
    Generate ``total_count`` polygons split equally across easy, mid, hard, and extreme.

    If ``total_count`` is not divisible by 4, the remainder goes to easy.

    Parameters
    ----------
    total_count : int
        Total number of polygons to generate (e.g. 100, 500).
    output_path : str or Path, optional
        If given, save the resulting list as JSON to this path.

    Returns
    -------
    list[dict]
        All generated polygon records with sequential IDs starting from 1.
    """
    tiers = ("easy", "mid", "hard", "extreme")
    per_tier = total_count // len(tiers)
    remainder = total_count - per_tier * len(tiers)  # extra easy galleries

    generated = []
    for difficulty in tiers:
        count = per_tier + (remainder if difficulty == "easy" else 0)
        if count == 0:
            continue
        print(f"  Generating {count} {difficulty} polygons...")
        generated.extend(generate_polygons_with_holes(count, difficulty))

    # Sequential IDs across the combined list
    for i, poly in enumerate(generated, start=1):
        poly["id"] = i

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(generated, indent=2), encoding="utf-8"
        )
        print(f"  Saved {len(generated)} polygons → {output_path}")

    return generated


if __name__ == "__main__":
    generated = []
    generated.extend(generate_easy_polygons(10))
    generated.extend(generate_mid_polygons(10))
    generated.extend(generate_hard_polygons(10))
    generated.extend(generate_extreme_polygons(10))

    for polygon_id, polygon in enumerate(generated, start=1):
        polygon["id"] = polygon_id

    LOCAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATASET_PATH.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(f"Saved {len(generated)} polygons to {LOCAL_DATASET_PATH}")
