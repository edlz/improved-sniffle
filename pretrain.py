"""
pretrain.py — behavioral cloning from recorded demos

Usage:
    python pretrain.py
    python pretrain.py --epochs 200 --lr 1e-3
    python pretrain.py --demo-dir demos --save-path checkpoints/cleanrl_ppo/pretrained.pt
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from train import Agent


@dataclass
class Args:
    demo_dir: str = "demos"
    """directory containing .npz demo files"""
    save_path: str = "checkpoints/cleanrl_ppo/pretrained.pt"
    """where to save pretrained weights"""
    epochs: int = 100
    """training epochs"""
    lr: float = 3e-4
    """learning rate"""
    batch_size: int = 256
    """batch size"""
    hidden_size: int = 128
    """must match train_cleanrl.py"""
    act_dim: int = 9
    """must match FEThracia776DiscretizerSmall"""


def load_demos(demo_dir):
    obs_all, act_all = [], []
    demo_files = sorted(Path(demo_dir).glob("demo_*.npz"))
    if not demo_files:
        raise FileNotFoundError(f"No demo files in {demo_dir}")

    for f in demo_files:
        data = np.load(f)
        obs_all.append(data["obs"])
        act_all.append(data["actions"])
        print(f"  Loaded {f.name}: {len(data['obs'])} frames")

    obs = np.concatenate(obs_all)
    actions = np.concatenate(act_all)
    print(f"Total: {len(obs)} frames from {len(demo_files)} demos")

    # Print action distribution
    unique, counts = np.unique(actions, return_counts=True)
    names = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "R", "START"]
    print("Action distribution:")
    for a, c in zip(unique, counts):
        print(f"  {names[a]:>6}: {c:6d} ({c/len(actions)*100:.1f}%)")

    # Downsample NOOP — keep all action frames, cap NOOPs at 2x non-NOOP count
    non_noop = actions != 0
    n_action = non_noop.sum()
    n_noop = (~non_noop).sum()
    max_noop = int(n_action * 0.25)  # 4:1 action:NOOP ratio

    if n_noop > max_noop:
        noop_idx = np.where(~non_noop)[0]
        keep_noop = np.random.choice(noop_idx, size=max_noop, replace=False)
        keep = np.sort(np.concatenate([np.where(non_noop)[0], keep_noop]))
        obs, actions = obs[keep], actions[keep]
        print(f"Balanced: kept {max_noop}/{n_noop} NOOPs, {n_action} actions → {len(obs)} total")

    return obs, actions


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    obs, actions = load_demos(args.demo_dir)
    obs_dim = obs.shape[1]

    obs_t = torch.FloatTensor(obs).to(device)
    act_t = torch.LongTensor(actions).to(device)

    # Create agent
    agent = Agent(obs_dim, args.act_dim, hidden=args.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(obs_t, act_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train
    best_loss = float("inf")
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_obs, batch_act in loader:
            logits = agent.actor(batch_obs)
            loss = criterion(logits, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_obs)
            correct += (logits.argmax(dim=-1) == batch_act).sum().item()
            total += len(batch_obs)

        avg_loss = total_loss / total
        accuracy = correct / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>4}/{args.epochs}  loss={avg_loss:.4f}  accuracy={accuracy:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": 0,
        "global_step": 0,
        "args": {
            "hidden_size": args.hidden_size,
            "game": "FE776-Snes",
            "state": "debug_save",
            "frame_skip": 4,
            "stickprob": 0.25,
            "max_episode_steps": 4500,
        },
    }, args.save_path)
    print(f"\nSaved pretrained model to {args.save_path}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"\nTo fine-tune with PPO:")
    print(f"  python train_cleanrl.py --checkpoint {args.save_path}")


if __name__ == "__main__":
    main()
