"""
callbacks.py — training callbacks

Best practices:
- EvalCallback saves the best model automatically
- ProgressBarCallback shows timestep/episode progress
- Custom callback demonstrates how to log extra metrics
"""

from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    ProgressBarCallback,
)
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class EpisodeStatsCallback(BaseCallback):
    """
    Logs episode length mean/std to tensorboard.
    Hook for any custom per-episode metrics you want to track.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_lengths.append(info["episode"]["l"])

        # flush every 100 episodes
        if len(self._episode_lengths) >= 100:
            self.logger.record(
                "custom/ep_len_mean", np.mean(self._episode_lengths)
            )
            self.logger.record(
                "custom/ep_len_std", np.std(self._episode_lengths)
            )
            self._episode_lengths = []

        return True  # returning False stops training


class TrainingCallbacks:
    """
    Bundles all callbacks. Add/remove from build() as needed.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        save_path: str = "checkpoints/best",
        log_path: str = "logs",
        reward_threshold: float | None = None,
    ):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.log_path = log_path
        self.reward_threshold = reward_threshold

    def build(self) -> CallbackList:
        stop_cb = None
        if self.reward_threshold is not None:
            stop_cb = StopTrainingOnRewardThreshold(
                reward_threshold=self.reward_threshold, verbose=1
            )

        eval_cb = EvalCallback(
            eval_env=self.eval_env,
            best_model_save_path=self.save_path,
            log_path=self.log_path,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,      # use deterministic policy for eval
            render=False,
            callback_on_new_best=stop_cb,
        )

        return CallbackList([
            eval_cb,
            EpisodeStatsCallback(),
            ProgressBarCallback(),
        ])
