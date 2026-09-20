"""
Rendering: what the agent actually did, drawn.

One drawing routine, `draw_gallery`, is shared by everything:

  * the training viewer, which logs a grid of rollouts to TensorBoard every few
    thousand steps so progress is visible while training runs;
  * `python -m agp.viewer`, an interactive browser for a dataset or a trained
    checkpoint, for looking at single cases closely.

Matplotlib is used rather than Tk because the Tk build shipped with recent
macOS renders blank windows; matplotlib falls back to a native backend
interactively and to Agg when writing files.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
import numpy as np

from .dataset import DEFAULT_DIR, load
from .env import ArtGalleryEnv, EnvConfig
from .gallery import Gallery
from .rewards import PRUNE, RewardWeights


# Palette. Coverage is the loud colour because coverage is the thing to read
# off a panel at a glance.
OUTER_FILL = "#e5e7eb"
OUTER_EDGE = "#1f2937"
HOLE_FILL = "#ffffff"
HOLE_EDGE = "#b91c1c"
COVER_FILL = "#facc15"
GUARD_COLOR = "#111827"


# ---------------------------------------------------------------------------
# Core drawing
# ---------------------------------------------------------------------------
def _rings(geometry) -> List[list]:
    """Exterior coordinate rings of a polygon or multipolygon, for filling."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [list(geometry.exterior.coords)]
    if hasattr(geometry, "geoms"):
        return [list(g.exterior.coords) for g in geometry.geoms
                if g.geom_type == "Polygon" and not g.is_empty]
    return []


def draw_gallery(ax, gallery: Gallery, title: Optional[str] = None):
    """Draw one gallery, its holes, its covered region and its guards onto `ax`."""
    ax.clear()

    # --- gallery outline ---
    xs, ys = gallery.polygon.exterior.xy
    ax.fill(xs, ys, facecolor=OUTER_FILL, edgecolor=OUTER_EDGE, linewidth=1.6, zorder=1)

    # --- covered region, drawn over the floor but under the holes ---
    for ring in _rings(gallery.covered):
        ax.fill([p[0] for p in ring], [p[1] for p in ring],
                facecolor=COVER_FILL, edgecolor="none", alpha=0.55, zorder=2)

    # --- holes punched back out, so coverage never appears inside one ---
    for hole in gallery.polygon.interiors:
        hx, hy = hole.xy
        ax.fill(hx, hy, facecolor=HOLE_FILL, edgecolor=HOLE_EDGE, linewidth=1.2, zorder=3)

    # --- guards ---
    if gallery.guards:
        ax.scatter([g[0] for g in gallery.guards], [g[1] for g in gallery.guards],
                   c=GUARD_COLOR, s=26, zorder=4, edgecolors="white", linewidths=0.6)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
def rollout(model, record: dict, config: EnvConfig,
            weights: RewardWeights, deterministic: bool = True):
    """
    Run one full episode of `model` on one gallery and return its final state.

    Augmentation is switched off so the panel shows the gallery as stored,
    which makes successive snapshots of the same instance comparable.
    """
    config = EnvConfig(**{**config.__dict__, "augment": False})
    env = ArtGalleryEnv([record], config=config, weights=weights, seed=0)

    obs, _ = env.reset()
    done = False
    total = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated

    return env.gallery, total, info


def sample_records(records: List[dict], count: int) -> List[dict]:
    """
    Pick `count` galleries spread evenly through the list.

    Datasets are written tier by tier, so even spacing gives a panel from each
    difficulty. Taking the first few would show nothing but easy galleries,
    which are solved long before the interesting ones.
    """
    count = max(1, min(count, len(records)))
    indices = np.linspace(0, len(records) - 1, count).round().astype(int)
    return [records[i] for i in indices]


def rollout_figure(model, records: List[dict], config: EnvConfig,
                   weights: RewardWeights, columns: int = 4):
    """
    A one-row grid of finished episodes -- the training viewer's frame.

    Each panel's title carries the three numbers worth watching: coverage
    reached, guards used, and the greedy reference for that instance.
    """
    import matplotlib.pyplot as plt

    panels = sample_records(records, columns)
    columns = len(panels)
    figure, axes = plt.subplots(1, columns, figsize=(3.1 * columns, 3.4))
    axes = np.atleast_1d(axes)

    for ax, record in zip(axes, panels):
        gallery, total, info = rollout(model, record, config, weights)
        mark = "OK" if info.get("solved", 0.0) >= 1.0 else "--"
        draw_gallery(ax, gallery,
                     title=(f"{record['tier']} {mark}  "
                            f"cov {gallery.coverage * 100:.1f}%  "
                            f"k {gallery.n_guards}/{record.get('k_ref', '?')}  "
                            f"R {total:+.1f}"))

    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# Interactive browser
# ---------------------------------------------------------------------------
def browse(records: List[dict], model=None, config: Optional[EnvConfig] = None,
           weights: Optional[RewardWeights] = None):
    """
    Step through galleries with the arrow keys.

    With no model, shows the stored greedy solution. With a model, shows what
    the policy does on that gallery instead.
    """
    import matplotlib.pyplot as plt

    config = config or EnvConfig(allow_delete=True)
    weights = weights or PRUNE
    state = {"index": 0}
    figure, ax = plt.subplots(figsize=(7.5, 7.5))

    def show(index: int):
        state["index"] = index % len(records)
        record = records[state["index"]]

        if model is None:
            # Greedy reference: replay the stored guard set.
            gallery = Gallery(record, img_size=config.img_size)
            for x, y in record.get("greedy_guards", []):
                gallery.add_guard(x, y)
            label = "greedy"
            total = 0.0
        else:
            gallery, total, _ = rollout(model, record, config, weights)
            label = "policy"

        draw_gallery(ax, gallery,
                     title=(f"[{state['index'] + 1}/{len(records)}] {record['tier']} "
                            f"({label})   coverage {gallery.coverage * 100:.2f}%   "
                            f"guards {gallery.n_guards}   k_ref {record.get('k_ref', '?')}"
                            + (f"   return {total:+.2f}" if model else "")))
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "n", " "):
            show(state["index"] + 1)
        elif event.key in ("left", "p"):
            show(state["index"] - 1)

    figure.canvas.mpl_connect("key_press_event", on_key)
    show(0)
    print("Left/Right (or p/n) to move between galleries. Close the window to exit.")
    plt.show()


def save_previews(records: List[dict], out_dir: Path, model=None,
                  config: Optional[EnvConfig] = None,
                  weights: Optional[RewardWeights] = None):
    """Write one PNG per gallery -- the headless equivalent of `browse`."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = config or EnvConfig(allow_delete=True)
    weights = weights or PRUNE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(7.5, 7.5))
    for record in records:
        if model is None:
            gallery = Gallery(record, img_size=config.img_size)
            for x, y in record.get("greedy_guards", []):
                gallery.add_guard(x, y)
        else:
            gallery, _, _ = rollout(model, record, config, weights)
        draw_gallery(ax, gallery,
                     title=f"{record['tier']} #{record['id']}  "
                           f"coverage {gallery.coverage * 100:.2f}%  "
                           f"guards {gallery.n_guards}")
        figure.savefig(out_dir / f"{record['tier']}_{record['id']:04d}.png",
                       dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(f"[viewer] wrote {len(records)} previews to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Browse galleries and policies.")
    parser.add_argument("--data", default=str(DEFAULT_DIR / "test.json"),
                        help="dataset JSON to view")
    parser.add_argument("--model", default=None,
                        help="optional PPO checkpoint; without it, the greedy solution is shown")
    parser.add_argument("--save", default=None,
                        help="write PNGs to this directory instead of opening a window")
    parser.add_argument("--limit", type=int, default=None,
                        help="only look at the first N galleries")
    args = parser.parse_args()

    records = load(Path(args.data))
    if args.limit:
        records = records[: args.limit]

    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")

    if args.save:
        save_previews(records, Path(args.save), model=model)
    else:
        browse(records, model=model)


if __name__ == "__main__":
    main()
