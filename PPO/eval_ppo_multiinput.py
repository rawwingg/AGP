"""
PPO Evaluation Script — Art Gallery Problem (MultiInputPolicy)
==============================================================

Run from the project root after training:
    python PPO/eval_ppo_multiinput.py
    python PPO/eval_ppo_multiinput.py --episodes 20
    python PPO/eval_ppo_multiinput.py --save-solutions      # saves ppo_solutions.json
    python PPO/eval_ppo_multiinput.py --save-solutions --visualize

Options
-------
--model           Path to trained model (without .zip).
                  Default: ppo_art_gallery_multiinput  (project root)
--data            Path to polygon dataset JSON.
                  Default: data/polygon_dataset_90.json
--episodes        Number of random evaluation episodes. Default: 10
--save-solutions  Run one episode per polygon in the dataset and save
                  results to data/ppo_solutions.json  (used by Main_UI.py)
--visualize       Save a matplotlib PNG for the best episode of each difficulty.
--save-dir        Directory for visualisation PNGs. Default: eval_plots/
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# __file__ is PPO/eval_ppo_multiinput.py → parent = PPO/ → parent.parent = project root
# ---------------------------------------------------------------------------
_ROOT     = Path(__file__).parent.parent          # project root
_POLY_DIR = _ROOT / "Polygons dataset"
if str(_POLY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLY_DIR))

from art_gallery_env import ArtGalleryEnv          # noqa: E402
from stable_baselines3 import PPO                  # noqa: E402

DATASET_PATH   = _ROOT / "data" / "polygon_dataset_90.json"
SOLUTIONS_PATH = _ROOT / "data" / "ppo_solutions.json"
MODEL_PATH     = _ROOT / "ppo_art_gallery_multiinput"


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def _visualize_episode(record: dict, episode_info: dict, save_path: Path):
    """Draw polygon + placed guards and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from security import Guard, Security

    fig, ax = plt.subplots(figsize=(8, 8))

    outer       = record["outer_points"]
    holes       = record["holes"]
    guards_list = episode_info["guards"]

    xs = [p[0] for p in outer]
    ys = [p[1] for p in outer]
    ax.fill(xs, ys, facecolor="#dbeafe", edgecolor="#1d4ed8", linewidth=2, zorder=1)

    for hole in holes:
        hx = [p[0] for p in hole]
        hy = [p[1] for p in hole]
        ax.fill(hx, hy, facecolor="white", edgecolor="#dc2626", linewidth=1.5, zorder=2)

    if guards_list:
        sec = Security(record)
        for gx, gy in guards_list:
            cov = sec.get_area_coverage_of_guard(Guard(gx, gy))
            if cov.is_empty:
                continue
            rings = (
                [list(cov.exterior.coords)]
                if cov.geom_type == "Polygon"
                else [
                    list(g.exterior.coords)
                    for g in getattr(cov, "geoms", [])
                    if g.geom_type == "Polygon" and not g.is_empty
                ]
            )
            for ring in rings:
                ax.fill([p[0] for p in ring], [p[1] for p in ring],
                        facecolor="#fde047", edgecolor="none", alpha=0.45, zorder=3)

        ax.scatter([g[0] for g in guards_list], [g[1] for g in guards_list],
                   c="#111827", s=60, zorder=5, marker="*", label="guards")

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_title(
        f"Polygon {record.get('id','?')}  [{record.get('difficulty','?')}]  |  "
        f"guards={episode_info.get('num_guards',0)}  |  "
        f"coverage={episode_info.get('coverage',0)*100:.1f}%  |  "
        f"reward={episode_info.get('total_reward',0):.2f}  |  "
        f"invalid={episode_info.get('invalid_count',0)}"
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz] Saved → {save_path}")


# ---------------------------------------------------------------------------
# Run one episode on a specific polygon (used by --save-solutions)
# ---------------------------------------------------------------------------

def _run_episode_on_polygon(model, env, polygon_record: dict) -> dict:
    """
    Reset the env to ``polygon_record`` and run one deterministic episode.
    Returns an episode-info dict.
    """
    # Force the env to use this specific polygon
    env.dataset = [polygon_record]
    obs, info = env.reset()

    total_reward  = 0.0
    invalid_count = 0
    invalid_reasons: dict = defaultdict(int)
    done = False
    last_info = info

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        last_info = step_info
        if step_info.get("invalid_action"):
            invalid_count += 1
            invalid_reasons[step_info.get("invalid_reason", "unknown")] += 1

    return {
        "coverage":        last_info.get("coverage",  0.0),
        "num_guards":      last_info.get("num_guards", 0),
        "total_reward":    total_reward,
        "invalid_count":   invalid_count,
        "invalid_reasons": dict(invalid_reasons),
        "guards":          last_info.get("guards", []),
        "solved":          last_info.get("coverage", 0.0) >= 0.95,
    }


# ---------------------------------------------------------------------------
# Save solutions dataset (one solution per polygon)
# ---------------------------------------------------------------------------

def save_solutions(model, dataset: list, env, out_path: Path):
    """
    Run one deterministic episode per polygon and write a solutions JSON file
    in the same format as local_dataset_with_guards.json so Main_UI.py can
    display it directly.
    """
    print(f"\n[Solutions] Running one episode per polygon ({len(dataset)} total)...")
    solutions = []

    for i, poly in enumerate(dataset):
        ep = _run_episode_on_polygon(model, env, poly)
        solved_str = "✓" if ep["solved"] else "✗"
        print(
            f"  [{i+1:>3}/{len(dataset)}] id={poly.get('id','?'):>3}  "
            f"diff={poly.get('difficulty','?'):>4}  "
            f"cov={ep['coverage']*100:>5.1f}%  "
            f"guards={ep['num_guards']}  {solved_str}"
        )

        record = dict(poly)                             # copy original polygon fields
        record["guards"]                = [[g[0], g[1]] for g in ep["guards"]]
        record["area_coverage"]         = ep["coverage"]
        record["model_total_reward"]    = ep["total_reward"]
        record["model_invalid_count"]   = ep["invalid_count"]
        record["model_invalid_reasons"] = ep["invalid_reasons"]
        record["model_solved"]          = ep["solved"]
        solutions.append(record)

    # Restore full dataset so env works normally after this call
    env.dataset = dataset

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(solutions, fh, indent=2)

    solved_total = sum(1 for s in solutions if s["model_solved"])
    avg_cov = np.mean([s["area_coverage"] for s in solutions]) * 100
    print(f"\n[Solutions] Saved {len(solutions)} solutions → {out_path}")
    print(f"[Solutions] Solved: {solved_total}/{len(solutions)}  |  "
          f"Avg coverage: {avg_cov:.1f}%")
    return solutions


# ---------------------------------------------------------------------------
# Standard random-episode evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, env, num_episodes: int, visualize: bool, save_dir: Path):
    """Run ``num_episodes`` random deterministic episodes and print stats."""

    stats_by_diff = defaultdict(lambda: {
        "coverages": [], "num_guards": [], "rewards": [],
        "invalid_counts": [], "solved": 0, "count": 0,
    })
    best_episode: dict = {}

    print(f"\n{'='*70}")
    print(f"{'Episode':>8}  {'Diff':>6}  {'Coverage':>10}  "
          f"{'Guards':>6}  {'Reward':>10}  {'Invalid':>7}  {'Solved':>6}")
    print(f"{'='*70}")

    for ep in range(num_episodes):
        obs, info = env.reset()
        difficulty  = info.get("difficulty", "unknown")
        polygon_rec = env.current_polygon

        total_reward  = 0.0
        invalid_count = 0
        invalid_reasons: dict = defaultdict(int)
        done     = False
        last_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            last_info = step_info
            if step_info.get("invalid_action"):
                invalid_count += 1
                invalid_reasons[step_info.get("invalid_reason", "unknown")] += 1

        coverage   = last_info.get("coverage",  0.0)
        num_guards = last_info.get("num_guards", 0)
        guards     = last_info.get("guards",    [])
        solved     = coverage >= 0.95

        episode_info = {
            "coverage":        coverage,
            "num_guards":      num_guards,
            "total_reward":    total_reward,
            "invalid_count":   invalid_count,
            "invalid_reasons": dict(invalid_reasons),
            "guards":          guards,
            "difficulty":      difficulty,
            "polygon_id":      polygon_rec.get("id", -1),
        }

        s = stats_by_diff[difficulty]
        s["coverages"].append(coverage)
        s["num_guards"].append(num_guards)
        s["rewards"].append(total_reward)
        s["invalid_counts"].append(invalid_count)
        s["count"] += 1
        if solved:
            s["solved"] += 1

        if difficulty not in best_episode or coverage > best_episode[difficulty]["coverage"]:
            best_episode[difficulty] = {**episode_info, "_polygon_rec": polygon_rec}

        print(
            f"{ep+1:>8}  {difficulty:>6}  {coverage*100:>9.1f}%  "
            f"{num_guards:>6}  {total_reward:>10.3f}  {invalid_count:>7}  "
            f"{'YES' if solved else 'no':>6}"
        )
        print(f"         invalid reasons: {dict(invalid_reasons)}"
              f"   guards: {[f'({x:.1f},{y:.1f})' for x,y in guards]}")

    print(f"\n{'='*70}")
    print("SUMMARY BY DIFFICULTY")
    print(f"{'='*70}")
    for diff, s in sorted(stats_by_diff.items()):
        n = s["count"]
        print(
            f"  {diff:>6}  episodes={n:>3}  "
            f"avg_coverage={np.mean(s['coverages'])*100:>6.1f}%  "
            f"avg_guards={np.mean(s['num_guards']):>5.1f}  "
            f"avg_reward={np.mean(s['rewards']):>8.3f}  "
            f"avg_invalid={np.mean(s['invalid_counts']):>5.1f}  "
            f"solved={s['solved']}/{n}"
        )

    if visualize:
        print(f"\n[Viz] Saving best-episode plots to {save_dir}/")
        for diff, ep_info in best_episode.items():
            poly_rec  = ep_info.pop("_polygon_rec")
            save_path = save_dir / f"best_{diff}_cov{ep_info['coverage']*100:.0f}.png"
            _visualize_episode(poly_rec, ep_info, save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model on the Art Gallery Problem."
    )
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                        help="Path to trained model (without .zip).")
    parser.add_argument("--data", type=str, default=str(DATASET_PATH),
                        help="Path to polygon dataset JSON.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of random evaluation episodes (default: 10).")
    parser.add_argument("--save-solutions", action="store_true",
                        help=(
                            "Run one episode per polygon and save results to "
                            "data/ppo_solutions.json for viewing in Main_UI.py."
                        ))
    parser.add_argument("--solutions-path", type=str, default=str(SOLUTIONS_PATH),
                        help="Output path for the solutions file.")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a matplotlib PNG for the best episode of each difficulty.")
    parser.add_argument("--save-dir", type=str, default=str(_ROOT / "eval_plots"),
                        help="Directory for visualisation PNGs (default: eval_plots/).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[Error] Dataset not found: {data_path}")
        print("        Run PPO/train_ppo_multiinput.py first to generate it.")
        sys.exit(1)
    with open(data_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
    print(f"[Dataset] Loaded {len(dataset)} polygons from {data_path}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_path = args.model
    if not Path(model_path + ".zip").exists() and not Path(model_path).exists():
        print(f"[Error] Model not found: {model_path}.zip")
        print("        Run PPO/train_ppo_multiinput.py first to train the model.")
        sys.exit(1)

    print(f"[Model] Loading {model_path}...")
    env   = ArtGalleryEnv(dataset)
    model = PPO.load(model_path, env=env)
    print("[Model] Loaded successfully.\n")

    # ------------------------------------------------------------------
    # Save solutions (one per polygon) — used by Main_UI.py solution mode
    # ------------------------------------------------------------------
    if args.save_solutions:
        save_solutions(
            model=model,
            dataset=dataset,
            env=env,
            out_path=Path(args.solutions_path),
        )
        # Restore full dataset after save_solutions temporarily altered it
        env.dataset = dataset

    # ------------------------------------------------------------------
    # Standard random-episode evaluation
    # ------------------------------------------------------------------
    evaluate(
        model=model,
        env=env,
        num_episodes=args.episodes,
        visualize=args.visualize,
        save_dir=Path(args.save_dir),
    )


if __name__ == "__main__":
    main()
