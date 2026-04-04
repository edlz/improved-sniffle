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

    from stable_baselines3.common.monitor import Monitor
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

    from stable_baselines3.common.monitor import Monitor
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
        "FEThracia776DiscretizerSmall": FEThracia776DiscretizerSmall,
        "RAMObsWrapper": RAMObsWrapper,
        "RewardWrapper": RewardWrapper,
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
    """Full discrete action set for Fire Emblem: Thracia 776."""

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
        ])


class FEThracia776DiscretizerSmall(Discretizer):
    """Trimmed action set for early training."""

    def __init__(self, env):
        super().__init__(env, combos=[
            [],                  # 0: NOOP
            ["UP"],              # 1: UP
            ["DOWN"],            # 2: DOWN
            ["LEFT"],            # 3: LEFT
            ["RIGHT"],           # 4: RIGHT
            ["A"],               # 5: Confirm / select
            ["B"],               # 6: Cancel / back
            ["R"],               # 7: Scroll right
            ["START"],           # 8: Menu
        ])


class RAMObsWrapper(gym.Wrapper):
    """Replace image obs with a flat vector of game state values from info."""

    GAME_KEYS = [
        "turn", "phase", "cursor_x", "cursor_y",
        "gold", "last_defeated", "capture",
    ]
    UNIT_KEYS = ["char", "class", "x", "y", "level", "exp",
                 "hp", "maxhp", "str", "mag", "skl", "spd", "def", "lck", "con"]
    N_PLAYERS = 16
    N_ENEMIES = 20

    def __init__(self, env):
        super().__init__(env)
        n_features = (len(self.GAME_KEYS)
                      + len(self.UNIT_KEYS) * self.N_PLAYERS
                      + len(self.UNIT_KEYS) * self.N_ENEMIES)
        self.observation_space = gym.spaces.Box(
            low=0, high=65535, shape=(n_features,), dtype=np.float32,
        )

    def _extract(self, info):
        feats = [float(info.get(k, 0)) for k in self.GAME_KEYS]
        for prefix, n in [("p", self.N_PLAYERS), ("e", self.N_ENEMIES)]:
            for i in range(n):
                for k in self.UNIT_KEYS:
                    feats.append(float(info.get(f"{prefix}{i}_{k}", 0)))
        return np.array(feats, dtype=np.float32)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # info is empty on reset — do a noop step to populate it
        _, reward, terminated, truncated, info = self.env.step(0)
        return self._extract(info), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._extract(info), reward, terminated, truncated, info


class RewardWrapper(gym.Wrapper):
    """Shaped reward based on RAM info for Fire Emblem: Thracia 776."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_info = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_info = None
        return obs, info

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute(self._prev_info, info) if self._prev_info else 0.0
        self._prev_info = info
        return obs, reward, terminated, truncated, info

    def _compute(self, prev, cur):
        if prev is None:
            return 0.0
        r = 0.0

        # Penalize each new turn
        # if cur.get("turn", 0) > prev.get("turn", 0):
        #     r -= 1.0

        # Player unit death — HP dropped to 0
        for i in range(48):
            if prev.get(f"p{i}_char", 0) == 0:
                continue
            if prev.get(f"p{i}_hp", 0) > 0 and cur.get(f"p{i}_hp", 0) == 0:
                r -= 10.0

        # Enemy unit death — HP dropped to 0
        for i in range(51):
            if prev.get(f"e{i}_char", 0) == 0:
                continue
            if prev.get(f"e{i}_hp", 0) > 0 and cur.get(f"e{i}_hp", 0) == 0:
                r += 2.0

        # Player HP damage taken
        for i in range(48):
            if prev.get(f"p{i}_char", 0) == 0:
                continue
            hp_prev = prev.get(f"p{i}_hp", 0)
            hp_now = cur.get(f"p{i}_hp", 0)
            if hp_now < hp_prev:
                r -= (hp_prev - hp_now) * 0.05

        # Enemy HP damage dealt
        for i in range(51):
            if prev.get(f"e{i}_char", 0) == 0:
                continue
            hp_prev = prev.get(f"e{i}_hp", 0)
            hp_now = cur.get(f"e{i}_hp", 0)
            if hp_now < hp_prev:
                r += (hp_prev - hp_now) * 0.1

        # Phase change (L button removed from action set, so no exploit)
        if cur.get("phase", 0) != prev.get("phase", 0):
            r += 1.0

        # Chapter clear
        if cur.get("chapter", 0) > prev.get("chapter", 0):
            r += 1000.0

        # Player unit movement
        for i in range(48):
            if cur.get(f"p{i}_char", 0) == 0:
                continue
            px, py = cur.get(f"p{i}_x", 0), cur.get(f"p{i}_y", 0)
            ox, oy = prev.get(f"p{i}_x", 0), prev.get(f"p{i}_y", 0)
            if px != ox or py != oy:
                r += 0.6

        # Player EXP gain
        for i in range(48):
            if cur.get(f"p{i}_char", 0) == 0:
                continue
            exp_now = cur.get(f"p{i}_exp", 0)
            exp_prev = prev.get(f"p{i}_exp", 0)
            if exp_now > exp_prev:
                r += (exp_now - exp_prev) * 0.1

        # Capture counter increased
        if cur.get("capture", 0) > prev.get("capture", 0):
            r += 5.0

        return r
