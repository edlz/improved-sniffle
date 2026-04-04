"""
evaluate.py — load a trained model and run evaluation episodes

Usage:
    # Retro env (uses config to reconstruct wrappers + frame stacking)
    python evaluate.py --model checkpoints/ppo_thracia776_v1/best_model \
                       --config configs/ppo_thracia776.yaml \
                       --episodes 10 --render --audio

    # Plain gymnasium env
    python evaluate.py --model checkpoints/ppo_cartpole_v1/best_model \
                       --env CartPole-v1 \
                       --episodes 20 --render
"""

import argparse
import sys
import numpy as np
import gymnasium as gym
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.wrappers import make_env, Discretizer

ALGORITHMS = [PPO, SAC, TD3]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(model_path: str, env):
    """Auto-detect algorithm from saved metadata."""
    path = Path(model_path)
    for algo in ALGORITHMS:
        try:
            return algo.load(path, env=env)
        except Exception:
            continue
    raise ValueError(f"Could not load model from {model_path}")


def _unwrap_vec(venv):
    """Unwrap VecEnv layers to get the first underlying gymnasium env."""
    env = venv
    while hasattr(env, "venv"):
        env = env.venv
    if hasattr(env, "envs"):
        env = env.envs[0]
    return env


def _find_wrapper(venv, cls):
    """Walk the gymnasium wrapper stack to find a specific wrapper type."""
    env = _unwrap_vec(venv)
    while env is not None:
        if isinstance(env, cls):
            return env
        env = getattr(env, "env", None)
    return None


def _print_results(rewards, lengths, episodes):
    print(f"\n--- Results over {episodes} episodes ---")
    print(f"Mean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean length : {np.mean(lengths):.1f}")
    print(f"Max reward  : {np.max(rewards):.2f}")


class PygameRenderer:
    """Renders retro env frames with audio and input overlay via pygame."""

    def __init__(self, retro_env, button_combos=None, scale=3, enable_audio=False):
        import pygame
        import pygame.locals as K
        self.pygame = pygame
        self.retro_env = retro_env
        self._combos = button_combos

        pygame.display.init()
        pygame.font.init()

        img = retro_env.get_screen()
        h, w = img.shape[:2]
        self._game_w = w * scale
        self._game_h = h * scale
        self._input_h = 48 if button_combos else 0
        self._log_h = 220
        self.screen = pygame.display.set_mode(
            (self._game_w, self._game_h + self._input_h + self._log_h)
        )
        pygame.display.set_caption("Thracia 776 — RL Agent")
        self.clock = pygame.time.Clock()
        self._fps = retro_env.em.get_screen_rate()
        self._font = pygame.font.SysFont("monospace", 22)
        self._log_font = pygame.font.SysFont("monospace", 14)
        self._change_log = []
        self._prev_info = {}
        self._frame_num = 0

        self._key_map = {
            K.K_UP: "UP", K.K_DOWN: "DOWN", K.K_LEFT: "LEFT", K.K_RIGHT: "RIGHT",
            K.K_z: "A", K.K_x: "B", K.K_a: "X", K.K_s: "Y",
            K.K_q: "L", K.K_w: "R", K.K_RETURN: "START",
        }
        self._human_override = False

        self._audio_stream = None
        if not enable_audio:
            return
        try:
            import sounddevice as sd
            rate = int(retro_env.em.get_audio_rate())
            self._audio_stream = sd.OutputStream(
                samplerate=rate, channels=2, dtype="int16", latency=0.5,
            )
            self._audio_stream.start()
            print(f"Audio enabled (rate={rate}Hz)")
        except Exception as e:
            print(f"Audio init failed ({e}), rendering video only")

    def render_step(self, action=None, info=None, reward=0.0):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False

        self._frame_num += 1

        frame = self.retro_env.get_screen()
        surf = self.pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = self.pygame.transform.scale(surf, (self._game_w, self._game_h))
        self.screen.blit(surf, (0, 0))

        if self._combos is not None and action is not None:
            self._draw_inputs(action)

        # Draw RAM log
        self._draw_log(info, reward)

        self.pygame.display.flip()

        if self._audio_stream is not None:
            raw = self.retro_env.em.get_audio()
            if len(raw):
                arr = np.array(raw, dtype=np.int16)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 2)
                self._audio_stream.write(np.ascontiguousarray(arr))
        else:
            self.clock.tick(self._fps)

        return True

    def get_human_action(self):
        """Return a discrete action index from keyboard, or None if no keys pressed."""
        if self._combos is None:
            return None
        keys = self.pygame.key.get_pressed()
        pressed = [btn for key, btn in self._key_map.items() if keys[key]]
        if not pressed:
            self._human_override = False
            return None
        self._human_override = True
        for i, combo in enumerate(self._combos):
            if sorted(combo) == sorted(pressed):
                return i
        for i, combo in enumerate(self._combos):
            if combo == pressed[:1]:
                return i
        return 0

    def _draw_badge(self, text, color, x, y_center):
        """Draw a rounded badge and return x advance."""
        pad = 6
        text_surf = self._font.render(text, True, (255, 255, 255))
        tw, th = text_surf.get_size()
        self.pygame.draw.rect(
            self.screen, color,
            (x, y_center - th // 2 - pad // 2, tw + pad * 2, th + pad),
            border_radius=4,
        )
        self.screen.blit(text_surf, (x + pad, y_center - th // 2))
        return tw + pad * 2

    def _draw_log(self, info, reward):
        if info is None:
            return
        y_start = self._game_h + self._input_h

        # Background
        self.pygame.draw.rect(self.screen, (20, 20, 20),
            (0, y_start, self._game_w, self._log_h))

        # Current state line
        state_line = (
            f"cursor=({info.get('cursor_x','?')},{info.get('cursor_y','?')})  "
            f"turn={info.get('turn','?')}  phase={info.get('phase','?')}  "
            f"sel={info.get('selected_char','?')}  "
            f"p0=({info.get('p0_x','?')},{info.get('p0_y','?')}) hp={info.get('p0_hp','?')}"
        )
        self.screen.blit(self._log_font.render(state_line, True, (200, 200, 200)),
            (8, y_start + 4))

        # Track changes
        changes = []
        watch = ["selected_char", "phase", "cursor_x", "cursor_y", "turn", "capture"]
        for i in range(5):
            watch += [f"p{i}_x", f"p{i}_y", f"p{i}_hp"]
        for i in range(15):
            watch += [f"e{i}_x", f"e{i}_y", f"e{i}_hp"]
        for k in watch:
            old = self._prev_info.get(k)
            new = info.get(k)
            if old is not None and old != new:
                changes.append(f"{k}:{old}->{new}")

        if changes or reward != 0:
            entry = f"[{self._frame_num:>6}] "
            if reward != 0:
                entry += f"r={reward:+.3f} "
            entry += "  ".join(changes)
            self._change_log.append((entry, reward != 0))
            if len(self._change_log) > 50:
                self._change_log.pop(0)

        self._prev_info = dict(info)

        # Draw scrolling log
        self.screen.blit(self._log_font.render("--- change log ---", True, (150, 150, 150)),
            (8, y_start + 22))
        visible = self._change_log[-11:]
        for i, (entry, has_reward) in enumerate(visible):
            color = (100, 255, 100) if has_reward else (180, 180, 180)
            self.screen.blit(self._log_font.render(entry[:100], True, color),
                (8, y_start + 38 + i * 16))

    def _draw_inputs(self, action):
        y = self._game_h
        y_center = y + self._input_h // 2
        self.pygame.draw.rect(self.screen, (20, 20, 20), (0, y, self._game_w, self._input_h))

        buttons = self._combos[int(action)]

        if self._human_override:
            tag, tag_color = "HUMAN", (200, 170, 30)
        else:
            tag, tag_color = "MODEL", (50, 120, 200)

        x = 8
        x += self._draw_badge(tag, tag_color, x, y_center) + 12

        btn_color = (60, 180, 90) if buttons else (120, 120, 120)
        for btn in (buttons or ["NOOP"]):
            x += self._draw_badge(btn, btn_color, x, y_center) + 6

    def close(self):
        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
        self.pygame.quit()


def evaluate_with_config(model_path: str, config_path: str, episodes: int, render: bool, audio: bool, vec_norm_path: str | None):
    """Evaluate using a training config (required for retro envs)."""
    cfg = load_config(config_path)
    retro_cfg = cfg["env"].get("retro")

    env = make_env(cfg["env"]["id"], cfg["env"].get("wrappers", []), retro_cfg=retro_cfg)
    venv = DummyVecEnv([lambda: env])

    if cfg["env"].get("frame_stack"):
        venv = VecFrameStack(venv, n_stack=cfg["env"]["frame_stack"])

    if vec_norm_path and Path(vec_norm_path).exists():
        venv = VecNormalize.load(vec_norm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = load_model(model_path, venv)

    renderer = None
    if render and retro_cfg and retro_cfg.get("enabled"):
        discretizer = _find_wrapper(venv, Discretizer)
        combos = discretizer._combos if discretizer else None
        renderer = PygameRenderer(
            _unwrap_vec(venv).unwrapped, button_combos=combos, enable_audio=audio,
        )

    rewards, lengths = [], []
    obs = venv.reset()

    for ep in range(episodes):
        done, ep_reward, ep_len = False, 0.0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            if renderer:
                human_action = renderer.get_human_action()
                if human_action is not None:
                    action = np.array([human_action])

            obs, reward_arr, done_arr, info_arr = venv.step(action)
            done = done_arr[0]
            ep_reward += reward_arr[0]
            ep_len += 1

            step_info = info_arr[0] if info_arr else {}
            if renderer and not renderer.render_step(action[0], info=step_info, reward=reward_arr[0]):
                done = True

        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"Episode {ep + 1:>3}: reward={ep_reward:.2f}, length={ep_len}")

    _print_results(rewards, lengths, episodes)
    if renderer:
        renderer.close()
    venv.close()


def evaluate(model_path: str, env_id: str, episodes: int, render: bool, vec_norm_path: str | None):
    """Evaluate with a plain gymnasium env (no config)."""
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    if vec_norm_path and Path(vec_norm_path).exists():
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(vec_norm_path, venv)
        venv.training = False
        venv.norm_reward = False
        eval_env = venv
    else:
        eval_env = None

    model = load_model(model_path, eval_env)

    rewards, lengths = [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        done, ep_reward, ep_len = False, 0.0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"Episode {ep + 1:>3}: reward={ep_reward:.2f}, length={ep_len}")

    _print_results(rewards, lengths, episodes)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--config", default=None, help="Training config YAML (required for retro envs)")
    parser.add_argument("--env", default=None, help="Gymnasium env ID (for non-retro envs)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--audio", action="store_true", help="Enable audio (requires sounddevice + libportaudio2)")
    parser.add_argument("--vec-norm", default=None, help="Path to vec_normalize.pkl")
    args = parser.parse_args()

    if args.config:
        evaluate_with_config(
            model_path=args.model,
            config_path=args.config,
            episodes=args.episodes,
            render=args.render,
            audio=args.audio,
            vec_norm_path=args.vec_norm,
        )
    elif args.env:
        evaluate(
            model_path=args.model,
            env_id=args.env,
            episodes=args.episodes,
            render=args.render,
            vec_norm_path=args.vec_norm,
        )
    else:
        parser.error("Either --config or --env is required")
