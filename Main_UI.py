"""Matplotlib-based viewer for Art Gallery polygons.

Two modes:
    explore   — browse the dataset with randomly placed guards
                (data/local_dataset_with_guards.json)
    solution  — browse the PPO model's solutions
                (data/ppo_solutions.json)

Usage:
    python Main_UI.py                          # explore mode (default)
    python Main_UI.py --mode solution          # view PPO solutions
    python Main_UI.py --mode explore --save    # save PNGs for explore dataset
    python Main_UI.py --mode solution --save   # save PNGs for solutions
    python Main_UI.py --data path/to/file.json # override dataset path
"""

import sys
import argparse
import json
from pathlib import Path

import matplotlib
from shapely.geometry import Polygon

# Security lives in "Polygons dataset/" — add it to the path so the import works
# regardless of what directory Main_UI.py is run from.
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "Polygons dataset"))

from security import Guard, Security  # noqa: E402


# ---------------------------------------------------------------------------
# Default dataset paths (both relative to project root)
# ---------------------------------------------------------------------------
EXPLORE_DATASET_PATH  = _ROOT / "data" / "local_dataset_with_guards.json"
SOLUTIONS_DATASET_PATH = _ROOT / "data" / "ppo_solutions.json"
PREVIEW_DIR           = _ROOT / "data" / "previews"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
OUTER_FILL   = "#dbeafe"
OUTER_EDGE   = "#1d4ed8"
HOLE_FILL    = "#ffffff"
HOLE_EDGE    = "#dc2626"
GUARD_COLOR  = "#111827"
COVERAGE_FILL = "#fde047"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_polygons(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}.\n"
            f"  • For explore mode: run the dataset generator first.\n"
            f"  • For solution mode: run  python PPO/eval_ppo_multiinput.py --save-solutions"
        )
    return json.loads(data_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Drawing (unchanged — works for both modes because both datasets share the
# same record format: outer_points, holes, guards, area_coverage)
# ---------------------------------------------------------------------------

def _coverage_rings(coverage_polygon):
    """Return a list of exterior coordinate rings for a (possibly multi) polygon."""
    if coverage_polygon is None or coverage_polygon.is_empty:
        return []
    if coverage_polygon.geom_type == "Polygon":
        return [list(coverage_polygon.exterior.coords)]
    if coverage_polygon.geom_type.startswith("Multi") or coverage_polygon.geom_type == "GeometryCollection":
        rings = []
        for geom in coverage_polygon.geoms:
            if geom.geom_type == "Polygon" and not geom.is_empty:
                rings.append(list(geom.exterior.coords))
        return rings
    return []


def draw_polygon(ax, record):
    ax.clear()

    outer_points = record["outer_points"]
    holes        = record["holes"]
    guards       = record["guards"]
    security     = Security(record)

    # outer gallery
    outer = Polygon(outer_points)
    xs, ys = outer.exterior.xy
    ax.fill(xs, ys, facecolor=OUTER_FILL, edgecolor=OUTER_EDGE, linewidth=2, zorder=1)

    # holes punched out
    for hole in holes:
        hx = [p[0] for p in hole]
        hy = [p[1] for p in hole]
        ax.fill(hx, hy, facecolor=HOLE_FILL, edgecolor=HOLE_EDGE, linewidth=1.5, zorder=2)

    # each guard's visibility coverage (semi-transparent so overlaps are visible)
    for gx, gy in guards:
        coverage = security.get_area_coverage_of_guard(Guard(gx, gy))
        for ring in _coverage_rings(coverage):
            rx = [p[0] for p in ring]
            ry = [p[1] for p in ring]
            ax.fill(rx, ry, facecolor=COVERAGE_FILL, edgecolor="none", alpha=0.45, zorder=3)

    # guard positions
    if guards:
        ax.scatter([g[0] for g in guards], [g[1] for g in guards],
                   c=GUARD_COLOR, s=40, zorder=4, label="guards")

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_title(
        f"Polygon {record['id']}  |  {len(outer_points)} pts  |  {len(holes)} holes  |  "
        f"{len(guards)} guards  |  coverage {record.get('area_coverage', 0) * 100:.2f}%"
    )
    ax.figure.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

def run_interactive(polygons, mode: str):
    import matplotlib.pyplot as plt

    state = {"idx": 0}
    fig, ax = plt.subplots(figsize=(8, 8))

    mode_label = "PPO Solution Viewer" if mode == "solution" else "Polygon Explorer"
    fig.canvas.manager.set_window_title(f"Art Gallery — {mode_label}")

    def show(i):
        state["idx"] = i % len(polygons)
        draw_polygon(ax, polygons[state["idx"]])

    def on_key(event):
        if event.key in ("right", "n", " "):
            show(state["idx"] + 1)
        elif event.key in ("left", "p"):
            show(state["idx"] - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show(0)
    print(f"[{mode_label}] {len(polygons)} polygons loaded.")
    print("Use Left/Right arrow keys (or n/p) to navigate. Close the window to exit.")
    plt.show()


# ---------------------------------------------------------------------------
# Batch PNG save
# ---------------------------------------------------------------------------

def run_save(polygons, mode: str):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = PREVIEW_DIR / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    for record in polygons:
        draw_polygon(ax, record)
        out = save_dir / f"polygon_{record['id']}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {len(polygons)} preview images to {save_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Art Gallery polygon viewer.")
    parser.add_argument(
        "--mode",
        choices=["explore", "solution"],
        default="explore",
        help=(
            "explore  — browse dataset with random guards (default)\n"
            "solution — browse PPO model solutions"
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Render a PNG per polygon instead of opening an interactive window.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override the dataset path (uses mode default if not set).",
    )
    args = parser.parse_args()

    # Choose dataset path based on mode (unless overridden)
    if args.data:
        data_path = Path(args.data)
    elif args.mode == "solution":
        data_path = SOLUTIONS_DATASET_PATH
    else:
        data_path = EXPLORE_DATASET_PATH

    polygons = load_polygons(data_path)
    if not polygons:
        print("No polygons found in dataset.")
        return

    print(f"[Mode: {args.mode}] Loaded {len(polygons)} polygons from {data_path}")

    if args.save:
        run_save(polygons, args.mode)
    else:
        run_interactive(polygons, args.mode)


if __name__ == "__main__":
    main()
