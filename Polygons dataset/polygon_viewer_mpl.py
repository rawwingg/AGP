"""Matplotlib-based viewer for guarded polygons.

Replaces the Tk viewer, which renders blank on macOS's deprecated system Tk.
Matplotlib uses the native macOS backend (or Agg when saving), so no Tk needed.

Usage:
    python polygon_viewer_mpl.py            # interactive window, arrow keys / buttons to navigate
    python polygon_viewer_mpl.py --save     # write a PNG per polygon into Polygons/previews/
"""

import argparse
import json
from pathlib import Path

import matplotlib
from shapely.geometry import Polygon

from security import Guard, Security


LOCAL_DATASET_PATH = Path("Polygons") / "local_dataset_with_guards.json"
PREVIEW_DIR = Path("Polygons") / "previews"

OUTER_FILL = "#dbeafe"
OUTER_EDGE = "#1d4ed8"
HOLE_FILL = "#ffffff"
HOLE_EDGE = "#dc2626"
GUARD_COLOR = "#111827"
COVERAGE_FILL = "#fde047"


def load_polygons(data_path=LOCAL_DATASET_PATH):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Run generate_polygon_dataset.py and "
            f"security.py first to create the guarded dataset."
        )
    return json.loads(data_path.read_text(encoding="utf-8"))


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
    holes = record["holes"]
    guards = record["guards"]
    security = Security(record)

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
    ax.invert_yaxis()  # match image-style coordinates (y grows downward)
    ax.set_title(
        f"Polygon {record['id']}  |  {len(outer_points)} pts  |  {len(holes)} holes  |  "
        f"{len(guards)} guards  |  coverage {record.get('area_coverage', 0) * 100:.2f}%"
    )
    ax.figure.canvas.draw_idle()


def run_interactive(polygons):
    import matplotlib.pyplot as plt

    state = {"idx": 0}
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Guarded Polygon Viewer (matplotlib)")

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
    print("Use Left/Right arrow keys (or n/p) to navigate polygons. Close the window to exit.")
    plt.show()


def run_save(polygons):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    for record in polygons:
        draw_polygon(ax, record)
        out = PREVIEW_DIR / f"polygon_{record['id']}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {len(polygons)} preview images to {PREVIEW_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="View guarded polygons with matplotlib.")
    parser.add_argument("--save", action="store_true",
                        help="Render a PNG per polygon instead of opening a window.")
    parser.add_argument("--data", default=str(LOCAL_DATASET_PATH),
                        help="Path to the guarded dataset JSON.")
    args = parser.parse_args()

    polygons = load_polygons(args.data)
    if not polygons:
        print("No polygons found in dataset.")
        return

    if args.save:
        run_save(polygons)
    else:
        run_interactive(polygons)


if __name__ == "__main__":
    main()
