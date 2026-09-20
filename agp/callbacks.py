"""
Stable-Baselines3 callbacks: the weight schedule, episode metrics, and the
training viewer.

Three concerns, three callbacks, deliberately kept separate:

  WeightSchedule  moves the reward weights through the stage's schedule.
  EpisodeMetrics  turns the environment's end-of-episode info into TensorBoard
                  scalars, so the numbers that matter (did it solve it, with
                  how many guards) are plotted rather than inferred from return.
  TrainingViewer  renders a row of finished episodes into TensorBoard every few
                  thousand steps. Return curves say whether learning is
                  happening; the pictures say what it has actually learned.
"""

from collections import deque
from typing import Callable, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

from .env import EnvConfig
from .rewards import RewardWeights


# ---------------------------------------------------------------------------
# Reward weight schedule
# ---------------------------------------------------------------------------
class WeightSchedule(BaseCallback):
    """
    Push freshly computed reward weights into every worker environment.

    `schedule` maps progress through the stage (0 to 1) to a RewardWeights. The
    environments are otherwise untouched, which is what lets a single env class
    serve both stages: the curriculum lives here, not in the MDP.
    """

    def __init__(self, schedule: Callable[[float], RewardWeights],
                 total_timesteps: int, update_every: int = 4096, verbose: int = 0):
        super().__init__(verbose)
        self.schedule = schedule
        self.total_timesteps = max(total_timesteps, 1)
        self.update_every = update_every
        self._start = 0
        self._last_update = 0

    def _on_training_start(self):
        # num_timesteps keeps counting across stages, so anchor progress to the
        # step count at which this stage began.
        self._start = self.num_timesteps
        self._apply(0.0)

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - self._start
        if elapsed - self._last_update >= self.update_every:
            self._last_update = elapsed
            self._apply(min(elapsed / self.total_timesteps, 1.0))
        return True

    def _apply(self, progress: float):
        weights = self.schedule(progress)
        self.training_env.set_attr("weights", weights)
        # Log the weights that move, so the schedule is visible next to its effect.
        self.logger.record("weights/progress", progress)
        for name in ("par", "prune", "front", "sweep", "sep"):
            self.logger.record(f"weights/{name}", getattr(weights, name))


# ---------------------------------------------------------------------------
# Episode metrics
# ---------------------------------------------------------------------------
class EpisodeMetrics(BaseCallback):
    """
    Rolling averages of the per-episode quantities the environment reports.

    Tracked overall and per difficulty tier, because an average across tiers
    hides the usual failure mode: easy galleries solved, hard ones abandoned.
    """

    KEYS = ("coverage", "n_guards", "solved", "guard_ratio")

    def __init__(self, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._overall = {key: deque(maxlen=window) for key in self.KEYS}
        self._by_tier = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "solved" not in info:          # only set on an episode's last step
                continue
            tier = info.get("tier", "unknown")
            per_tier = self._by_tier.setdefault(
                tier, {key: deque(maxlen=self.window) for key in self.KEYS})
            for key in self.KEYS:
                self._overall[key].append(info[key])
                per_tier[key].append(info[key])
        return True

    def _on_rollout_end(self):
        for key, values in self._overall.items():
            if values:
                self.logger.record(f"episode/{key}", float(np.mean(values)))
        for tier, buckets in self._by_tier.items():
            for key in ("solved", "guard_ratio"):
                if buckets[key]:
                    self.logger.record(f"tier_{tier}/{key}", float(np.mean(buckets[key])))


# ---------------------------------------------------------------------------
# Training viewer
# ---------------------------------------------------------------------------
class TrainingViewer(BaseCallback):
    """
    Render finished episodes into TensorBoard's IMAGES tab while training runs.

    Watch it with:   tensorboard --logdir runs
    """

    def __init__(self, records: List[dict], config: EnvConfig,
                 every: int = 20_000, columns: int = 4, verbose: int = 0):
        super().__init__(verbose)
        self.records = records          # rollout_figure samples evenly across tiers
        self.config = config
        self.every = every
        self.columns = columns
        self._next = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.every

        # Deferred import: rendering pulls in pyplot, which we do not want
        # loaded in worker processes.
        from .viewer import rollout_figure

        # Use the weights the environments are running right now, so the return
        # printed on each panel matches what the agent is being trained on.
        weights = self.training_env.get_attr("weights")[0]
        figure = rollout_figure(self.model, self.records, self.config, weights,
                                columns=self.columns)
        self.logger.record("viewer/rollouts", Figure(figure, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        return True
