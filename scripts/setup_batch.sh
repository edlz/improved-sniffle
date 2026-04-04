#!/bin/bash
set -euo pipefail

# --- Load config from .env.batch ---
ENV_FILE="${1:-.env.batch}"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found. Copy .env.batch.example and fill in your values."
    exit 1
fi
set -a; source "$ENV_FILE"; set +a

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SUBNETS=$(aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" \
    --query "Subnets[0].SubnetId" --output text --region "$AWS_REGION")
SECURITY_GROUP=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=default" \
    --query "SecurityGroups[0].GroupId" --output text --region "$AWS_REGION")
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "Account:  $ACCOUNT_ID"
echo "Region:   $AWS_REGION"
echo "Subnet:   $SUBNETS"
echo "SG:       $SECURITY_GROUP"
echo "ECR:      $ECR_URI"
echo "S3:       s3://${S3_BUCKET}"
echo ""

# --- 1. Create S3 bucket ---
echo "Creating S3 bucket..."
aws s3 mb "s3://${S3_BUCKET}" --region "$AWS_REGION" 2>/dev/null || echo "Bucket exists"

echo "Upload your ROM:"
echo "  aws s3 cp retro_data/FE776-Snes/rom.sfc s3://${S3_BUCKET}/fe-thracia/cleanrl_ppo/rom.sfc"

# --- 2. Create ECR repository ---
echo "Creating ECR repository..."
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" 2>/dev/null || echo "Repo exists"

# --- 3. Build and push Docker image ---
echo "Building Docker image..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URI"
docker build -t "$ECR_REPO" .
docker tag "$ECR_REPO:latest" "$ECR_URI:latest"
docker push "$ECR_URI:latest"

# --- 4. Create Batch compute environment (GPU) ---
echo "Creating compute environment..."
aws batch create-compute-environment \
    --compute-environment-name "$COMPUTE_ENV_NAME" \
    --type MANAGED \
    --compute-resources '{
        "type": "EC2",
        "allocationStrategy": "BEST_FIT_PROGRESSIVE",
        "minvCpus": 0,
        "maxvCpus": '"$MAX_VCPUS"',
        "desiredvCpus": 0,
        "instanceTypes": ["'"$INSTANCE_TYPE"'"],
        "subnets": ["'"$SUBNETS"'"],
        "securityGroupIds": ["'"$SECURITY_GROUP"'"],
        "instanceRole": "'"$BATCH_INSTANCE_ROLE"'"
    }' \
    --region "$AWS_REGION" 2>/dev/null || echo "Compute env exists"

# --- 5. Create job queue ---
echo "Creating job queue..."
aws batch create-job-queue \
    --job-queue-name "$JOB_QUEUE_NAME" \
    --priority 1 \
    --compute-environment-order "order=1,computeEnvironment=$COMPUTE_ENV_NAME" \
    --region "$AWS_REGION" 2>/dev/null || echo "Job queue exists"

# --- 6. Register job definition ---
echo "Registering job definition..."
aws batch register-job-definition \
    --job-definition-name "$JOB_DEF_NAME" \
    --type container \
    --container-properties '{
        "image": "'"$ECR_URI:latest"'",
        "resourceRequirements": [
            {"type": "VCPU", "value": "4"},
            {"type": "MEMORY", "value": "15000"},
            {"type": "GPU", "value": "1"}
        ],
        "environment": [
            {"name": "S3_BUCKET", "value": "'"$S3_BUCKET"'"},
            {"name": "TOTAL_TIMESTEPS", "value": "10000000"},
            {"name": "NUM_ENVS", "value": "4"}
        ],
        "jobRoleArn": "arn:aws:iam::'"$ACCOUNT_ID"':role/'"$BATCH_JOB_ROLE"'",
        "executionRoleArn": "arn:aws:iam::'"$ACCOUNT_ID"':role/'"$BATCH_EXECUTION_ROLE"'"
    }' \
    --region "$AWS_REGION"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Before submitting, make sure you have:"
echo "  1. IAM role '${BATCH_INSTANCE_ROLE}' (for EC2 instances)"
echo "  2. IAM role '${BATCH_JOB_ROLE}' with S3 read/write to s3://${S3_BUCKET}"
echo "  3. IAM role '${BATCH_EXECUTION_ROLE}' (for ECS/Batch)"
echo "  4. Uploaded ROM: aws s3 cp retro_data/FE776-Snes/rom.sfc s3://${S3_BUCKET}/fe-thracia/cleanrl_ppo/rom.sfc"
echo ""
echo "Submit a job:"
echo "  aws batch submit-job --job-name train-run-1 --job-queue $JOB_QUEUE_NAME --job-definition $JOB_DEF_NAME"
