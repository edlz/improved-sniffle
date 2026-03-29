"""
evaluate.py — load a trained model and run evaluation episodes

Usage:
    python evaluate.py --model checkpoints/ppo_cartpole_v1/best_model \
                       --env CartPole-v1 \
                       --episodes 20 \
                       --render
"""

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pathlib import Path


def load_model(model_path: str, env):
    """Auto-detect algorithm from saved metadata."""
    path = Path(model_path)
    for algo in [PPO, SAC, TD3]:
        try:
            return algo.load(path, env=env)
        except Exception:
            continue
    raise ValueError(f"Could not load model from {model_path}")


def evaluate(model_path: str, env_id: str, episodes: int, render: bool, vec_norm_path: str | None):
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    # if training used VecNormalize, apply same stats at inference
    if vec_norm_path and Path(vec_norm_path).exists():
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(vec_norm_path, venv)
        venv.training = False     # don't update running stats at eval time
        venv.norm_reward = False  # don't normalize rewards at eval time
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

    print(f"\n--- Results over {episodes} episodes ---")
    print(f"Mean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean length : {np.mean(lengths):.1f}")
    print(f"Max reward  : {np.max(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--env", required=True, help="Gymnasium env ID")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--vec-norm", default=None, help="Path to vec_normalize.pkl")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        env_id=args.env,
        episodes=args.episodes,
        render=args.render,
        vec_norm_path=args.vec_norm,
    )
