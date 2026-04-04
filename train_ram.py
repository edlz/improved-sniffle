"""
train_ram.py — train on RAM observations instead of pixels

Usage:
    python train_ram.py
    python train_ram.py --checkpoint checkpoints/ppo_thracia776_ram_v1/best_model
"""

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from callbacks import TrainingCallbacks
from envs.wrappers import make_env

CONFIG_PATH = "configs/ppo_thracia776_ram.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config: dict, checkpoint: str | None = None):
    cfg = config
    retro_cfg = cfg["env"].get("retro")

    vec_cls = SubprocVecEnv if retro_cfg and retro_cfg.get("enabled") else DummyVecEnv

    env = make_vec_env(
        lambda: make_env(cfg["env"]["id"], cfg["env"].get("wrappers", []), retro_cfg=retro_cfg),
        n_envs=cfg["training"]["n_envs"],
        seed=cfg["training"]["seed"],
        vec_env_cls=vec_cls,
    )

    eval_env = DummyVecEnv([
        lambda: make_env(cfg["env"]["id"], cfg["env"].get("wrappers", []), retro_cfg=retro_cfg)
    ])

    save_path = f"checkpoints/{cfg['run_name']}"

    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        model = PPO.load(checkpoint, env=env)
    else:
        model = PPO(
            policy=cfg["algorithm"]["policy"],
            env=env,
            device="auto",
            verbose=1,
            tensorboard_log=f"logs/{cfg['run_name']}",
            **cfg["algorithm"].get("hyperparams", {}),
        )

    callbacks = TrainingCallbacks(
        eval_env=eval_env,
        eval_freq=cfg["training"]["eval_freq"],
        n_eval_episodes=cfg["training"]["n_eval_episodes"],
        save_path=save_path,
        log_path=f"logs/{cfg['run_name']}",
    ).build()

    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=checkpoint is None,
        progress_bar=True,
    )

    final_path = Path(f"{save_path}/final_model")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    train(cfg, checkpoint=args.checkpoint)
