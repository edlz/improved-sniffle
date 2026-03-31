"""
envs/wrappers.py — environment construction + custom wrappers

Best practices:
- Always wrap with Monitor before VecEnv (captures episode stats)
- Use TimeLimit to cap episode length
- Keep custom wrappers small and composable
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor


def make_env(
    env_id: str,
    wrappers: list[dict] | None = None,
    retro_cfg: dict | None = None,
) -> gym.Env:
    """
    Factory function — always use this instead of gym.make() directly.
    Applies wrappers from config and ensures Monitor is always outermost.
    """
    if retro_cfg and retro_cfg.get("enabled", False):
        return make_retro_env(env_id, retro_cfg, wrappers)

    env = gym.make(env_id)

    for w in (wrappers or []):
        env = _apply_wrapper(env, w)

    env = Monitor(env)  # must be outermost before VecEnv
    return env


def make_retro_env(
    game: str,
    retro_cfg: dict,
    wrappers: list[dict] | None = None,
) -> gym.Env:
    """Factory for stable-retro environments."""
    import stable_retro

    # Register custom integration path if provided
    integration_path = retro_cfg.get("integration_path")
    if integration_path:
        abs_path = str(Path(integration_path).resolve())
        stable_retro.data.add_custom_integration(abs_path)

    # Resolve state
    state_str = retro_cfg.get("state", "default")
    if state_str == "none":
        state = stable_retro.State.NONE
    elif state_str == "default":
        state = stable_retro.State.DEFAULT
    else:
        state = state_str  # named save state

    # Resolve action type
    actions_str = retro_cfg.get("actions", "all").upper()
    use_restricted_actions = stable_retro.Actions[actions_str]

    env = stable_retro.make(
        game=game,
        state=state,
        use_restricted_actions=use_restricted_actions,
        render_mode=retro_cfg.get("render_mode", "rgb_array"),
        inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
    )

    for w in (wrappers or []):
        env = _apply_wrapper(env, w)

    env = Monitor(env)
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
        "StochasticFrameSkip": StochasticFrameSkip,
        "WarpFrame": WarpFrame,
        "FEThracia776Discretizer": FEThracia776Discretizer,
    }

    if name in custom:
        return custom[name](env, **kwargs)

    raise ValueError(f"Unknown wrapper: {name}")


# ---------------------------------------------------------------------------
# Retro wrappers
# ---------------------------------------------------------------------------

class StochasticFrameSkip(gym.Wrapper):
    """
    Repeat actions for n frames with a sticky probability.
    Standard for retro game RL — reduces effective framerate.
    """

    def __init__(self, env, n: int = 4, stickprob: float = 0.25):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self._last_action = None

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self.n):
            if self._last_action is not None and np.random.random() < self.stickprob:
                act = self._last_action
            else:
                act = action
            obs, reward, terminated, truncated, info = self.env.step(act)
            total_reward += reward
            if terminated or truncated:
                break
        self._last_action = action
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._last_action = None
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    """Grayscale + resize observation to (width x height). Standard for CNN policies."""

    def __init__(self, env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        import cv2
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class Discretizer(gym.ActionWrapper):
    """
    Base class that converts MultiBinary action space to Discrete.
    Subclass and define combos for specific games.
    """

    def __init__(self, env, combos: list[list[str]]):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self._buttons = env.unwrapped.buttons
        self._combos = combos
        self.action_space = gym.spaces.Discrete(len(combos))

    def action(self, action: int):
        arr = np.zeros(self.env.action_space.n, dtype=np.int8)
        for button in self._combos[action]:
            arr[self._buttons.index(button)] = 1
        return arr


class FEThracia776Discretizer(Discretizer):
    """Discrete action set for Fire Emblem: Thracia 776."""

    def __init__(self, env):
        super().__init__(env, combos=[
            [],                  # NOOP
            ["UP"],
            ["DOWN"],
            ["LEFT"],
            ["RIGHT"],
            ["A"],               # Confirm / select
            ["B"],               # Cancel / back
            ["X"],               # Info / status
            ["Y"],               # Quick menu
            ["L"],               # Scroll left
            ["R"],               # Scroll right
            ["START"],           # Menu
            ["A", "UP"],
            ["A", "DOWN"],
            ["A", "LEFT"],
            ["A", "RIGHT"],
        ])
