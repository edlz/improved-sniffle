# rl_project

Stable-Baselines3 training package with best practices.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
# CartPole with PPO
python train.py --config configs/ppo_cartpole.yaml

# Hopper with SAC (requires MuJoCo)
python train.py --config configs/sac_hopper.yaml

# Resume from checkpoint
python train.py --config configs/ppo_cartpole.yaml --checkpoint checkpoints/ppo_cartpole_v1/best_model
```

## Evaluate

```bash
python evaluate.py --model checkpoints/ppo_cartpole_v1/best_model \
                   --env CartPole-v1 \
                   --episodes 20 \
                   --render
```

## Monitor training

```bash
tensorboard --logdir logs/
```

## Project structure

```
rl_project/
├── train.py                  # entry point
├── evaluate.py               # eval / inference
├── callbacks.py              # EvalCallback + custom hooks
├── configs/
│   ├── ppo_cartpole.yaml     # discrete action example
│   └── sac_hopper.yaml       # continuous control example
├── envs/
│   └── wrappers.py           # env factory + composable wrappers
├── checkpoints/              # best_model.zip saved here
└── logs/                     # tensorboard logs
```

## Key decisions

| Decision | Reason |
|---|---|
| Config-driven via YAML | Change hyperparams without touching code |
| `make_vec_env` + `Monitor` | Required for episode stats + parallelism |
| `VecNormalize` for continuous | Normalizing obs/rewards is critical for SAC/TD3 |
| `EvalCallback` | Saves best model automatically, no manual checkpointing |
| `deterministic=True` in eval | Stochastic policy at eval inflates variance |
| `device="cuda"` | Falls back to CPU automatically — safe default |
