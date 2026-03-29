"""
envs/wrappers.py — environment construction + custom wrappers

Best practices:
- Always wrap with Monitor before VecEnv (captures episode stats)
- Use TimeLimit to cap episode length
- Keep custom wrappers small and composable
"""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor


def make_env(env_id: str, wrappers: list[dict] | None = None) -> gym.Env:
    """
    Factory function — always use this instead of gym.make() directly.
    Applies wrappers from config and ensures Monitor is always outermost.
    """
    env = gym.make(env_id)

    for w in (wrappers or []):
        env = _apply_wrapper(env, w)

    env = Monitor(env)  # must be outermost before VecEnv
    return env


def _apply_wrapper(env: gym.Env, wrapper_cfg: dict) -> gym.Env:
    name = wrapper_cfg["name"]
    kwargs = wrapper_cfg.get("kwargs", {})

    built_in = {
        "TimeLimit": TimeLimit,
        "ClipAction": gym.wrappers.ClipAction,
        "RescaleAction": gym.wrappers.RescaleAction,
    }

    if name in built_in:
        return built_in[name](env, **kwargs)

    # custom wrappers registered here
    custom = {
        "RewardScale": RewardScaleWrapper,
        "ObsNoise": ObsNoiseWrapper,
    }

    if name in custom:
        return custom[name](env, **kwargs)

    raise ValueError(f"Unknown wrapper: {name}")


# ---------------------------------------------------------------------------
# Custom wrappers
# ---------------------------------------------------------------------------

class RewardScaleWrapper(gym.RewardWrapper):
    """Scale rewards by a constant. Useful for stabilizing PPO/SAC."""

    def __init__(self, env, scale: float = 0.01):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale


class ObsNoiseWrapper(gym.ObservationWrapper):
    """
    Adds Gaussian noise to observations.
    Useful for robustness training / domain randomization.
    """

    def __init__(self, env, std: float = 0.01):
        super().__init__(env)
        self.std = std

    def observation(self, obs):
        import numpy as np
        return obs + np.random.normal(0, self.std, size=obs.shape).astype(obs.dtype)
