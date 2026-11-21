# Shared settings for the AWS training scripts (sourced, not run).
# Research account, us-east-1: on-demand H100 quota 64 vCPU = four p5.4xlarge (1 × H100 each) at once.
# us-east-2: 32 vCPU requested 2026-11-05 (us-east-1 had no free H100 from 2026-11-04 21:50 UTC). Launch there with
# AWS_REGION=us-east-2 (after `AWS_REGION=us-east-2 bash infra/aws/setup.sh` once); the bucket stays in us-east-1.
export AWS_PROFILE="${AWS_PROFILE:-m21-research}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export MC_REGIONS="${MC_REGIONS:-us-east-1 us-east-2 us-west-2}"    # status.sh / terminate.sh look in all of these
export MC_BUCKET_REGION="us-east-1"
export AWS_PAGER=""

export MC_PROJECT_TAG="mrcogito"
export MC_INSTANCE_TYPE="${MC_INSTANCE_TYPE:-p5.4xlarge}"
export MC_PRICE_PER_HOUR="${MC_PRICE_PER_HOUR:-6.88}"     # on-demand Linux us-east-1, checked 2026-11-04
export MC_BUDGET_USD="${MC_BUDGET_USD:-4000}"             # author's cap: $2,000 for the E33n goal (2026-11-04), raised to $4,000 on 2026-11-14 ("Increase our budget to $4000")
export MC_ROLE="mrcogito-train-node"
export MC_SG="mrcogito-train-ssh"
export MC_KEY="mrcogito-train"
export MC_KEY_FILE="${HOME}/.ssh/mrcogito-train.pem"
export MC_REPO_URL="https://github.com/ksopyla/MrCogito.git"
export MC_ROOT_GB="${MC_ROOT_GB:-200}"

_acct() { aws sts get-caller-identity --query Account --output text; }
export MC_ACCOUNT="${MC_ACCOUNT:-$(_acct)}"
export MC_BUCKET="${MC_BUCKET:-mrcogito-train-${MC_ACCOUNT}}"
