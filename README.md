# Fire Emblem: Thracia 776 RL

Training an RL agent to play Fire Emblem: Thracia 776 (SNES) using CleanRL PPO on RAM observations with stable-retro and gymnasium.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place your Thracia 776 ROM as `retro_data/FE776-Snes/rom.sfc`.

## Train

```bash
python train.py

# Resume from checkpoint
python train.py --checkpoint checkpoints/cleanrl_ppo/latest.pt

# Override defaults
python train.py --total-timesteps 20000000 --num-envs 4 --ent-coef 0.1
```

Saves `latest.pt` (every 50 iterations), `best.pt` (best mean episodic return), and `final.pt` to `checkpoints/cleanrl_ppo/`.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/cleanrl_ppo/best.pt
```

## Pretrain from demos

Record a demo with `debug_ram.py` (F6 to start/stop recording), then:

```bash
python pretrain.py
python train.py --checkpoint checkpoints/cleanrl_ppo/pretrained.pt
```

## Debug / record demos

```bash
python debug_ram.py
```

Arrow keys, Z=A, X=B, F=R, Enter=START, Tab=SELECT. F5/F9 save/load state, F6 record demo.

## Monitor training

```bash
tensorboard --logdir logs/
```

## AWS Batch training

```bash
# Edit .env.batch with your AWS config, then:
./scripts/setup_batch.sh

# Upload ROM
aws s3 cp retro_data/FE776-Snes/rom.sfc s3://YOUR_BUCKET/fe-thracia/cleanrl_ppo/rom.sfc

# Submit job
aws batch submit-job --job-name train-run-1 \
  --job-queue fe-thracia-queue \
  --job-definition fe-thracia-cleanrl
```

## Project structure

```
train.py                  # PPO training (CleanRL single-file)
evaluate.py               # eval with pygame rendering + RAM log
pretrain.py               # behavioral cloning from recorded demos
debug_ram.py              # interactive play + demo recording
envs/
  wrappers.py             # env factory + retro wrappers
configs/
  ppo_thracia776_ram.yaml # RAM observation config
  ppo_thracia776.yaml     # CNN/pixel config (legacy)
retro_data/
  FE776-Snes/
    data.json             # RAM variable definitions
    scenario.json         # reward + done conditions
    metadata.json         # default state
    rom.sfc               # ROM (gitignored)
demos/                    # recorded demos for pretraining
checkpoints/              # saved models (.pt)
logs/                     # TensorBoard logs
scripts/
  entrypoint.sh           # AWS Batch entrypoint
  setup_batch.sh          # AWS Batch infrastructure setup
legacy/                   # SB3-based training (deprecated)
```
