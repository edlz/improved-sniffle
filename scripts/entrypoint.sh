#!/bin/bash
set -euo pipefail

# --- Config from environment variables ---
S3_BUCKET="${S3_BUCKET:?Set S3_BUCKET env var}"
S3_PREFIX="${S3_PREFIX:-fe-thracia/cleanrl_ppo}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-4}"
ENT_COEF="${ENT_COEF:-0.15}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== FE Thracia 776 — CleanRL Training ==="
echo "S3: s3://${S3_BUCKET}/${S3_PREFIX}"
echo "Timesteps: ${TOTAL_TIMESTEPS}, Envs: ${NUM_ENVS}"

# --- Download ROM from S3 ---
echo "Downloading ROM..."
aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/rom.sfc" retro_data/FE776-Snes/rom.sfc

# --- Download latest checkpoint if exists ---
mkdir -p checkpoints/cleanrl_ppo
if aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/latest.pt" checkpoints/cleanrl_ppo/latest.pt 2>/dev/null; then
    echo "Resuming from existing checkpoint"
    CHECKPOINT_ARG="--checkpoint checkpoints/cleanrl_ppo/latest.pt"
else
    echo "No checkpoint found — starting fresh"
    CHECKPOINT_ARG=""
fi

# --- Download pretrained model if exists and no checkpoint ---
if [ -z "$CHECKPOINT_ARG" ]; then
    if aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/pretrained.pt" checkpoints/cleanrl_ppo/pretrained.pt 2>/dev/null; then
        echo "Starting from pretrained model"
        CHECKPOINT_ARG="--checkpoint checkpoints/cleanrl_ppo/pretrained.pt"
    fi
fi

# --- Sync checkpoints to S3 in background ---
sync_to_s3() {
    while true; do
        sleep 300  # every 5 minutes
        echo "[sync] Uploading checkpoints to S3..."
        aws s3 sync checkpoints/cleanrl_ppo/ "s3://${S3_BUCKET}/${S3_PREFIX}/" \
            --exclude "*" --include "*.pt" --quiet || true
        aws s3 sync logs/ "s3://${S3_BUCKET}/${S3_PREFIX}/logs/" --quiet || true
    done
}
sync_to_s3 &
SYNC_PID=$!

# --- Run training ---
python3 train.py \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --ent-coef "$ENT_COEF" \
    $CHECKPOINT_ARG \
    $EXTRA_ARGS

# --- Final sync ---
echo "Training complete. Final S3 sync..."
aws s3 sync checkpoints/cleanrl_ppo/ "s3://${S3_BUCKET}/${S3_PREFIX}/" \
    --exclude "*" --include "*.pt"
aws s3 sync logs/ "s3://${S3_BUCKET}/${S3_PREFIX}/logs/"

kill $SYNC_PID 2>/dev/null || true
echo "Done."
