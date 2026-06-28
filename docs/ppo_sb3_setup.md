# Stable-Baselines3 PPO Setup

This guide gets PPO running locally on the `PPO` branch. It does **not** wire in the art gallery / `Security` code yet — that comes after you confirm training works.

Official references:

- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- [PPO algorithm page](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Getting started tutorial](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html)
- [Gymnasium docs](https://gymnasium.farama.org/) (SB3 uses Gymnasium envs)

---

## 1. Prerequisites

- Python **3.10–3.12** (SB3 + PyTorch support these well; avoid 3.13+ unless you verify wheels exist)
- macOS / Linux / WSL recommended for training scripts

Check your version:

```bash
python3 --version
```

---

## 2. Virtual environment

From the repo root (`AGP/`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## 3. Install dependencies

Create `requirements.txt` in the repo root (or install directly):

```txt
stable-baselines3[extra]>=2.3.0
gymnasium>=0.29.0
shapely>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.14.0
```

Install:

```bash
pip install -r requirements.txt
```

`stable-baselines3[extra]` pulls in PyTorch and optional logging/plotting helpers. On Apple Silicon, PyTorch usually installs with MPS (GPU) support automatically.

---

## 4. Smoke test (verify PPO works)

Before building the art gallery env, run PPO on a built-in Gymnasium env:

```bash
python scripts/ppo_smoke_test.py
```

Or run inline:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_smoke/")
model.learn(total_timesteps=10_000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
env.close()
```

You should see the reward climb over ~10k steps. If this works, SB3 is installed correctly.

---

## 5. Project layout (suggested)

```
AGP/
├── docs/
│   └── ppo_sb3_setup.md          # this file
├── scripts/
│   └── ppo_smoke_test.py         # CartPole sanity check
├── envs/                         # later: ArtGalleryEnv (Gymnasium)
├── logs/                         # tensorboard + checkpoints (gitignored)
├── models/                       # saved .zip policies (gitignored)
└── requirements.txt
```

Add to `.gitignore` if not already there:

```
logs/
models/
*.zip
```

---

## 6. Training with TensorBoard

While training:

```bash
tensorboard --logdir ./logs
```

Open http://localhost:6006 to watch reward, policy loss, value loss, etc.

---

## 7. Saving and loading a model

```python
model.save("models/ppo_cartpole")

from stable_baselines3 import PPO
model = PPO.load("models/ppo_cartpole", env=env)
```

---

## 8. PPO defaults worth knowing (for art gallery later)

When you build a custom continuous env (`Box` action space for guard `(x, y)`):

| Parameter | Typical starting value | Notes |
|-----------|------------------------|-------|
| `policy` | `"MlpPolicy"` | Use `"CnnPolicy"` if you use image observations |
| `learning_rate` | `3e-4` | SB3 default |
| `n_steps` | `2048` | Rollout length per env per update |
| `batch_size` | `64` | Must divide `n_steps * n_envs` |
| `n_epochs` | `10` | PPO epochs per rollout |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | Advantage estimation |
| `clip_range` | `0.2` | PPO clip |

Example skeleton for a future custom env:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# env = make_vec_env("ArtGallery-v0", n_envs=4)  # your env later
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/ppo_art_gallery/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
)
model.learn(total_timesteps=500_000)
model.save("models/ppo_art_gallery")
```

---

## 9. Custom Gymnasium environment (next step)

SB3 expects a [Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) env with:

- `reset()` → `obs, info`
- `step(action)` → `obs, reward, terminated, truncated, info`
- `observation_space` and `action_space` (use `gymnasium.spaces.Box` for continuous guard placement)

Tutorial: [Gymnasium — Env creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)

Once `ArtGalleryEnv` exists, register it and pass it to `PPO("MlpPolicy", env, ...)`.

---

## 10. Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'torch'` | Reinstall: `pip install "stable-baselines3[extra]"` |
| Slow on Mac | PyTorch MPS may help; SB3 uses CPU by default unless configured |
| `AssertionError: batch_size` | Ensure `batch_size` divides `n_steps * n_envs` |
| Gym import errors | Use `gymnasium`, not legacy `gym` |
| Shapely errors later | `pip install shapely>=2.0` |

---

## Quick links

| Resource | URL |
|----------|-----|
| SB3 GitHub | https://github.com/DLR-RM/stable-baselines3 |
| PPO paper (original) | https://arxiv.org/abs/1707.06347 |
| SB3 RL tips & tricks | https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html |
| Hyperparameter tuning | https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html |
