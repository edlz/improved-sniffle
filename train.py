"""
train.py — CleanRL-style single-file PPO for Fire Emblem: Thracia 776

Based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

Usage:
    python train.py
    python train.py --total-timesteps 20000000
    python train.py --checkpoint checkpoints/cleanrl_ppo/latest.pt
"""

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import stable_retro
from envs.wrappers import (
    FEThracia776DiscretizerSmall,
    RAMObsWrapper,
    RewardWrapper,
    StochasticFrameSkip,
)

stable_retro.data.add_custom_integration(str(Path("retro_data").resolve()))


@dataclass
class Args:
    exp_name: str = "cleanrl_ppo"
    """experiment name"""
    seed: int = 42
    """random seed"""
    cuda: bool = True
    """use GPU if available"""
    track: bool = False
    """track with wandb"""
    wandb_project_name: str = "fe-thracia"

    # Environment
    game: str = "FE776-Snes"
    state: str = "debug_save"
    max_episode_steps: int = 4500
    frame_skip: int = 4
    stickprob: float = 0.25

    # Training
    total_timesteps: int = 10_000_000
    num_envs: int = 2
    num_steps: int = 256
    """steps per env per rollout"""
    num_minibatches: int = 4
    update_epochs: int = 10
    anneal_lr: bool = True

    # PPO hyperparams
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    norm_adv: bool = True

    # Network
    hidden_size: int = 128
    """hidden layer size for actor/critic"""

    # Checkpointing
    checkpoint: str | None = None
    """path to checkpoint to resume from"""
    save_freq: int = 50
    """save checkpoint every N iterations"""
    save_path: str = "checkpoints/cleanrl_ppo"

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(args, idx):
    def thunk():
        env = stable_retro.make(
            game=args.game,
            state=args.state,
            use_restricted_actions=stable_retro.Actions.ALL,
            render_mode="rgb_array",
            inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
        )
        env = StochasticFrameSkip(env, n=args.frame_skip, stickprob=args.stickprob)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
        env = FEThracia776DiscretizerSmall(env)
        env = RewardWrapper(env)
        env = RAMObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, sync_tensorboard=True,
                   config=vars(args), name=run_name, save_code=True)

    writer = SummaryWriter(f"logs/{run_name}")
    writer.add_text("hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    if args.num_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        envs = SubprocVecEnv([make_env(args, i) for i in range(args.num_envs)])
        # Compatibility shim for CleanRL
        envs.single_observation_space = envs.observation_space
        envs.single_action_space = envs.action_space
    else:
        envs = gym.vector.SyncVectorEnv([make_env(args, 0)])
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = envs.single_action_space.n
    print(f"Obs dim: {obs_dim}, Action dim: {act_dim}")

    # Agent
    agent = Agent(obs_dim, act_dim, hidden=args.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    start_iteration = 1
    best_mean_return = -float("inf")

    # Resume from checkpoint
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iteration = ckpt.get("iteration", 0) + 1
        best_mean_return = ckpt.get("best_mean_return", best_mean_return)
        print(f"Resumed from {args.checkpoint} (iteration {start_iteration})")

    # Checkpointing directory
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start
    global_step = (start_iteration - 1) * args.batch_size
    start_time = time.time()
    if args.num_envs > 1:
        next_obs = envs.reset()
    else:
        next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(np.array(next_obs)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(start_iteration, args.num_iterations + 1):
        iter_returns = []

        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            step_result = envs.step(action.cpu().numpy())
            if args.num_envs > 1:
                # SubprocVecEnv returns (obs, reward, done, info)
                next_obs, reward, done_arr, infos = step_result
                next_done = done_arr.astype(np.float32)
                for info in infos:
                    if "episode" in info:
                        ep_r = info["episode"]["r"]
                        ep_l = info["episode"]["l"]
                        print(f"  step={global_step}  ep_return={ep_r:.1f}  ep_len={ep_l}")
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)
                        iter_returns.append(ep_r)
            else:
                # SyncVectorEnv returns (obs, reward, terminated, truncated, info)
                next_obs, reward, terminations, truncations, infos = step_result
                next_done = np.logical_or(terminations, truncations)
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            ep_r = info["episode"]["r"]
                            ep_l = info["episode"]["l"]
                            print(f"  step={global_step}  ep_return={ep_r:.1f}  ep_len={ep_l}")
                            writer.add_scalar("charts/episodic_return", ep_r, global_step)
                            writer.add_scalar("charts/episodic_length", ep_l, global_step)
                            iter_returns.append(ep_r)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(np.array(next_obs)).to(device)
            next_done = torch.Tensor(next_done).to(device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_max = torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if iteration % 10 == 0:
            print(f"iter={iteration}/{args.num_iterations}  steps={global_step}  "
                  f"SPS={sps}  pg_loss={pg_loss.item():.4f}  v_loss={v_loss.item():.4f}  "
                  f"entropy={entropy_loss.item():.3f}  explained_var={explained_var:.3f}")

        # Save checkpoint
        if iteration % args.save_freq == 0:
            ckpt_path = Path(args.save_path) / "latest.pt"
            torch.save({
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "global_step": global_step,
                "best_mean_return": best_mean_return,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

        # Save best model
        if iter_returns:
            mean_return = np.mean(iter_returns)
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                best_path = Path(args.save_path) / "best.pt"
                torch.save({
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "global_step": global_step,
                    "best_mean_return": best_mean_return,
                    "args": vars(args),
                }, best_path)
                print(f"  New best model! mean_return={mean_return:.1f} → {best_path}")

    # Save final
    final_path = Path(args.save_path) / "final.pt"
    torch.save({
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": args.num_iterations,
        "global_step": global_step,
        "best_mean_return": best_mean_return,
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Saved to {final_path}")

    envs.close()
    writer.close()
