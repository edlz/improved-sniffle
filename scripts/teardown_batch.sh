#!/bin/bash
set -euo pipefail

# --- Load config from .env.batch ---
ENV_FILE="${1:-.env.batch}"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found."
    exit 1
fi
set -a; source "$ENV_FILE"; set +a

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "=== Tearing down AWS Batch resources ==="
echo "Region: $AWS_REGION"
echo ""

# --- 1. Deregister job definitions ---
echo "Deregistering job definitions..."
REVISIONS=$(aws batch describe-job-definitions \
    --job-definition-name "$JOB_DEF_NAME" \
    --status ACTIVE \
    --query "jobDefinitions[].jobDefinitionArn" \
    --output text --region "$AWS_REGION" 2>/dev/null || true)
for arn in $REVISIONS; do
    aws batch deregister-job-definition --job-definition "$arn" --region "$AWS_REGION"
    echo "  Deregistered $arn"
done

# --- 2. Disable and delete job queue ---
echo "Disabling job queue..."
aws batch update-job-queue \
    --job-queue "$JOB_QUEUE_NAME" \
    --state DISABLED \
    --region "$AWS_REGION" 2>/dev/null || echo "  Queue not found or already disabled"

echo "Waiting for queue to disable..."
sleep 10

echo "Deleting job queue..."
aws batch delete-job-queue \
    --job-queue "$JOB_QUEUE_NAME" \
    --region "$AWS_REGION" 2>/dev/null || echo "  Queue not found"

# --- 3. Disable and delete compute environment ---
echo "Disabling compute environment..."
aws batch update-compute-environment \
    --compute-environment "$COMPUTE_ENV_NAME" \
    --state DISABLED \
    --region "$AWS_REGION" 2>/dev/null || echo "  Compute env not found or already disabled"

echo "Waiting for compute environment to disable..."
sleep 15

echo "Deleting compute environment..."
aws batch delete-compute-environment \
    --compute-environment "$COMPUTE_ENV_NAME" \
    --region "$AWS_REGION" 2>/dev/null || echo "  Compute env not found"

# --- 4. Delete ECR repository ---
echo "Deleting ECR repository (including all images)..."
aws ecr delete-repository \
    --repository-name "$ECR_REPO" \
    --force \
    --region "$AWS_REGION" 2>/dev/null || echo "  ECR repo not found"

# --- 5. S3 bucket (prompt before deleting) ---
echo ""
echo "S3 bucket s3://${S3_BUCKET} was NOT deleted (may contain checkpoints)."
echo "To delete it manually:"
echo "  aws s3 rb s3://${S3_BUCKET} --force"

echo ""
echo "=== Teardown complete ==="
