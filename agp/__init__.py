"""
AGP -- reinforcement learning for the continuous Art Gallery Problem.

The design this package implements is written up in:

    docs/agp_mdp_formulation.pdf        -- the MDP (state, action, dynamics, reward)
    docs/agp_staged_ppo_curriculum.pdf  -- the two-stage PPO curriculum

Module map
----------
    polygons.py   random gallery generation (easy / mid / hard tiers)
    gallery.py    exact visibility polygons + coverage bookkeeping
    dataset.py    build train/test datasets, including the greedy reference count
    baselines.py  greedy cover and greedy prune -- the baselines to beat
    rewards.py    the reward basis and the per-stage weight schedule
    env.py        the Gymnasium environment (the MDP itself)
    viewer.py     matplotlib rendering of a gallery, its guards and their coverage
    callbacks.py  training-progress viewer + stage scheduler for Stable-Baselines3
    train.py      command-line entry point: generate / train / test
"""

__all__ = [
    "polygons",
    "gallery",
    "dataset",
    "baselines",
    "rewards",
    "env",
    "viewer",
    "callbacks",
]
