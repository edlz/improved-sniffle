# Fire Emblem: Thracia 776 RL

Training an RL agent to play Fire Emblem: Thracia 776 (SNES) using Stable-Baselines3 and stable-retro.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place your Thracia 776 ROM as `retro_data/FireEmblemThracia776-Snes-v0/rom.sfc`.

## Train

```bash
python train.py --config configs/ppo_thracia776.yaml

# Resume from checkpoint
python train.py --config configs/ppo_thracia776.yaml --checkpoint checkpoints/ppo_thracia776_v1/best_model
```

## Evaluate

```bash
python evaluate.py --model checkpoints/ppo_thracia776_v1/best_model \
                   --env FireEmblemThracia776-Snes-v0 \
                   --episodes 10
```

## Monitor training

```bash
tensorboard --logdir logs/
```

## Project structure

```
├── train.py                  # entry point
├── evaluate.py               # eval / inference
├── callbacks.py              # EvalCallback + custom hooks
├── test_retro.py             # smoke test for retro env
├── configs/
│   └── ppo_thracia776.yaml   # PPO + CnnPolicy for SNES
├── envs/
│   └── wrappers.py           # env factory + retro wrappers
├── retro_data/
│   └── FireEmblemThracia776-Snes-v0/
│       ├── data.json         # RAM variable definitions
│       ├── scenario.json     # reward + done conditions
│       ├── metadata.json     # default state
│       ├── rom.sha           # ROM hash
│       └── rom.sfc           # ROM (gitignored)
├── checkpoints/              # best_model.zip saved here
└── logs/                     # tensorboard logs
```
