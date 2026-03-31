"""
train.py — entry point for training

Usage:
    python train.py --config configs/ppo_cartpole.yaml
    python train.py --config configs/ppo_cartpole.yaml --checkpoint checkpoints/best_model
"""

import argparse
from pathlib import Path

import yaml
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv, VecFrameStack

from callbacks import TrainingCallbacks
from envs.wrappers import make_env

ALGORITHMS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config: dict, checkpoint: str | None = None):
    cfg = config

    # --- environment ---
    retro_cfg = cfg["env"].get("retro")

    # retro emulator allows only one instance per process — use SubprocVecEnv
    vec_cls = SubprocVecEnv if retro_cfg and retro_cfg.get("enabled") else DummyVecEnv

    env = make_vec_env(
        lambda: make_env(cfg["env"]["id"], cfg["env"].get("wrappers", []), retro_cfg=retro_cfg),
        n_envs=cfg["training"]["n_envs"],
        seed=cfg["training"]["seed"],
        vec_env_cls=vec_cls,
    )

    # normalize obs + rewards for continuous control tasks
    if cfg["env"].get("normalize", False):
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

    # frame stacking at VecEnv level (memory efficient)
    if cfg["env"].get("frame_stack"):
        env = VecFrameStack(env, n_stack=cfg["env"]["frame_stack"])

    eval_env = DummyVecEnv([
        lambda: make_env(cfg["env"]["id"], cfg["env"].get("wrappers", []), retro_cfg=retro_cfg)
    ])
    if cfg["env"].get("frame_stack"):
        eval_env = VecFrameStack(eval_env, n_stack=cfg["env"]["frame_stack"])

    # --- model ---
    algo_cls = ALGORITHMS[cfg["algorithm"]["name"]]

    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        model = algo_cls.load(checkpoint, env=env)
    else:
        model = algo_cls(
            policy=cfg["algorithm"]["policy"],
            env=env,
            device="cuda",  # falls back to cpu automatically if no GPU
            verbose=1,
            tensorboard_log=f"logs/{cfg['run_name']}",
            **cfg["algorithm"].get("hyperparams", {}),
        )

    # --- callbacks ---
    callbacks = TrainingCallbacks(
        eval_env=eval_env,
        eval_freq=cfg["training"]["eval_freq"],
        n_eval_episodes=cfg["training"]["n_eval_episodes"],
        save_path=f"checkpoints/{cfg['run_name']}",
        log_path=f"logs/{cfg['run_name']}",
    ).build()

    # --- train ---
    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=checkpoint is None,
    )

    # save final model + vecnormalize stats
    final_path = Path(f"checkpoints/{cfg['run_name']}/final_model")
    model.save(final_path)
    if cfg["env"].get("normalize", False):
        env.save(f"checkpoints/{cfg['run_name']}/vec_normalize.pkl")

    print(f"Training complete. Model saved to {final_path}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, checkpoint=args.checkpoint)
