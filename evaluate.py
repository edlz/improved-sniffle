"""
evaluate.py — evaluate a trained checkpoint with pygame rendering + RAM log

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/cleanrl_ppo/best.pt
    python evaluate.py --deterministic false
"""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch
import tyro
import stable_retro

from train import Agent, make_env, Args as TrainArgs
from envs.wrappers import RewardWrapper

stable_retro.data.add_custom_integration(str(Path("retro_data").resolve()))


@dataclass
class EvalArgs:
    checkpoint: str = "checkpoints/cleanrl_ppo/latest.pt"
    """path to checkpoint (or S3 URI with --s3)"""
    s3: bool = False
    """download checkpoint from S3 first"""
    deterministic: bool = True
    """use deterministic (greedy) actions"""
    scale: int = 3
    """display scale"""
    episodes: int = 1
    """number of episodes to run"""


def main():
    args = tyro.cli(EvalArgs)

    # Download from S3 if requested
    if args.s3:
        import subprocess
        local_path = Path(args.checkpoint).name
        local_path = f"checkpoints/cleanrl_ppo/{local_path}"
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.checkpoint} -> {local_path}")
        subprocess.run(["aws", "s3", "cp", args.checkpoint, local_path], check=True)
        args.checkpoint = local_path

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args_dict = ckpt.get("args", {})

    # Make env (single, raw for rendering)
    raw_env = stable_retro.make(
        game=train_args_dict.get("game", "FE776-Snes"),
        state=train_args_dict.get("state", "debug_save"),
        use_restricted_actions=stable_retro.Actions.ALL,
        render_mode="rgb_array",
        inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
    )

    # Wrap same as training but keep raw_env ref for rendering
    from envs.wrappers import (
        StochasticFrameSkip, FEThracia776DiscretizerSmall,
        RAMObsWrapper, RewardWrapper,
    )
    env = StochasticFrameSkip(raw_env,
        n=train_args_dict.get("frame_skip", 4),
        stickprob=train_args_dict.get("stickprob", 0.25))
    env = gym.wrappers.TimeLimit(env,
        max_episode_steps=train_args_dict.get("max_episode_steps", 4500))
    env = FEThracia776DiscretizerSmall(env)
    env = RewardWrapper(env)
    env = RAMObsWrapper(env)

    # Load agent
    obs_dim = np.array(env.observation_space.shape).prod()
    act_dim = env.action_space.n
    hidden = train_args_dict.get("hidden_size", 128)
    agent = Agent(obs_dim, act_dim, hidden=hidden).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    actions_names = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "R", "START"]

    # Pygame setup
    pygame.init()
    img = raw_env.get_screen()
    h, w = img.shape[:2]
    log_h = 260
    screen = pygame.display.set_mode((w * args.scale, h * args.scale + log_h))
    pygame.display.set_caption("CleanRL Eval — Thracia 776")
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    fps = raw_env.em.get_screen_rate()

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        change_log = []
        prev_info = dict(info)
        action_counts = {a: 0 for a in actions_names}

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    env.close()
                    pygame.quit()
                    return

            # Get action
            with torch.no_grad():
                obs_t = torch.Tensor(obs).unsqueeze(0).to(device)
                if args.deterministic:
                    logits = agent.actor(obs_t)
                    action = logits.argmax(dim=-1).item()
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_t)
                    action = action.item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1
            action_counts[actions_names[action]] += 1

            # Track changes
            changes = []
            watch = ["selected_char", "phase", "cursor_x", "cursor_y", "turn", "capture"]
            for i in range(5):
                watch += [f"p{i}_x", f"p{i}_y", f"p{i}_hp"]
            for i in range(15):
                watch += [f"e{i}_x", f"e{i}_y", f"e{i}_hp"]
            for k in watch:
                old = prev_info.get(k)
                new = info.get(k)
                if old is not None and old != new:
                    changes.append(f"{k}:{old}->{new}")

            if changes or reward != 0:
                entry = f"[{ep_len:>5}] {actions_names[action]:>5} "
                if reward != 0:
                    entry += f"r={reward:+.3f} "
                entry += "  ".join(changes)
                change_log.append((entry, reward != 0))
                if len(change_log) > 50:
                    change_log.pop(0)

            prev_info = dict(info)

            # Draw
            frame = raw_env.get_screen()
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, (w * args.scale, h * args.scale))
            screen.fill((20, 20, 20))
            screen.blit(surf, (0, 0))

            y = h * args.scale + 4
            state_line = (
                f"cursor=({info.get('cursor_x','?')},{info.get('cursor_y','?')})  "
                f"turn={info.get('turn','?')}  phase={info.get('phase','?')}  "
                f"sel={info.get('selected_char','?')}  "
                f"r={reward:+.3f}  total={ep_reward:.1f}  step={ep_len}"
            )
            screen.blit(font.render(state_line, True, (200, 200, 200)), (8, y))

            # Action distribution
            top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:5]
            act_line = "actions: " + "  ".join(f"{a}={c}" for a, c in top_actions)
            screen.blit(font.render(act_line, True, (200, 200, 200)), (8, y + 16))

            # Change log
            screen.blit(font.render("--- change log ---", True, (150, 150, 150)), (8, y + 36))
            visible = change_log[-12:]
            for i, (entry, has_reward) in enumerate(visible):
                color = (100, 255, 100) if has_reward else (180, 180, 180)
                screen.blit(font.render(entry[:120], True, color), (8, y + 52 + i * 16))

            pygame.display.flip()
            clock.tick(fps)

        print(f"Episode {ep + 1}: reward={ep_reward:.2f}, length={ep_len}")
        print(f"  Action dist: {dict(sorted(action_counts.items(), key=lambda x: -x[1]))}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
