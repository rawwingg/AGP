"""
PPO Training Script — Art Gallery Problem (MultiInputPolicy)
=============================================================

Run from the project root:
    python train_ppo_multiinput.py
    python train_ppo_multiinput.py --force-regenerate   # rebuild dataset

The script:
  1. Generates (or loads) a 120-polygon dataset (40 easy / 40 mid / 40 hard).
  2. Creates the ArtGalleryEnv wrapped around that dataset.
  3. Runs SB3 check_env to verify Gymnasium compatibility.
  4. Trains PPO for 200 000 timesteps.
  5. Saves the trained model to ppo_art_gallery_multiinput.zip.

TensorBoard logs are written to ./ppo_art_gallery_tensorboard/.
Launch with:
    tensorboard --logdir ppo_art_gallery_tensorboard
"""

import sys
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow imports from "Polygons dataset/" regardless of CWD
# ---------------------------------------------------------------------------
# __file__ is PPO/train_ppo_multiinput.py, so .parent is PPO/ and
# .parent.parent is the project root.
_ROOT     = Path(__file__).parent.parent          # project root
_POLY_DIR = _ROOT / "Polygons dataset"
if str(_POLY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLY_DIR))

from art_gallery_env import ArtGalleryEnv                            # noqa: E402
from generate_polygon_dataset import generate_polygons_with_holes    # noqa: E402

from stable_baselines3 import PPO                                    # noqa: E402
from stable_baselines3.common.env_checker import check_env           # noqa: E402

# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------
DATASET_PATH   = _ROOT / "data" / "polygon_dataset_90.json"
EXPECTED_COUNT = 120         # 40 easy + 40 mid + 40 hard


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(force: bool = False) -> list:
    """
    Generate 40 easy + 40 mid + 40 hard polygons and save to DATASET_PATH.
    Skips generation if the file already exists with the expected count,
    unless ``force`` is True.
    """
    if not force and DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if len(data) == EXPECTED_COUNT:
            print(f"[Dataset] Using cached dataset ({len(data)} polygons): {DATASET_PATH}")
            return data
        print(
            f"[Dataset] Cached file has {len(data)} polygons "
            f"(expected {EXPECTED_COUNT}). Regenerating..."
        )

    print("[Dataset] Generating 40 easy + 40 mid + 40 hard polygons...")
    generated = []
    for difficulty, count in [("easy", 40), ("mid", 40), ("hard", 40)]:
        print(f"  Generating {count} {difficulty} polygons...")
        generated.extend(generate_polygons_with_holes(count, difficulty))

    # Re-number IDs sequentially across the combined list
    for i, poly in enumerate(generated, start=1):
        poly["id"] = i

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as fh:
        json.dump(generated, fh, indent=2)
    print(f"[Dataset] Saved {len(generated)} polygons → {DATASET_PATH}")
    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on the Art Gallery Problem."
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate the polygon dataset even if it already exists.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total PPO training timesteps (default: 200000).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_art_gallery_multiinput",
        help="Output path for the saved model (without .zip extension).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    dataset = generate_dataset(force=args.force_regenerate)
    difficulties = {}
    for p in dataset:
        d = p.get("difficulty", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1
    print(f"[Dataset] Difficulty breakdown: {difficulties}")

    # ------------------------------------------------------------------
    # 2. Environment
    # ------------------------------------------------------------------
    env = ArtGalleryEnv(dataset)
    print(f"[Env] Created ArtGalleryEnv  obs_space={env.observation_space}  "
          f"act_space={env.action_space}")

    # ------------------------------------------------------------------
    # 3. Stable-Baselines3 environment check
    # ------------------------------------------------------------------
    print("\n[CheckEnv] Running SB3 environment checker...")
    check_env(env, warn=True)
    print("[CheckEnv] ✓ Passed!\n")

    # ------------------------------------------------------------------
    # 4. PPO model
    # ------------------------------------------------------------------
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(_ROOT / "ppo_art_gallery_tensorboard"),
    )
    print(f"[Model] PPO(MultiInputPolicy) created. "
          f"Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(f"[Training] Starting PPO for {args.timesteps:,} timesteps...\n")
    model.learn(total_timesteps=args.timesteps)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    model.save(args.model_path)
    print(f"\n[Training] Model saved → {args.model_path}.zip")
    print("           Run eval_ppo_multiinput.py to evaluate the trained model.")


if __name__ == "__main__":
    main()
