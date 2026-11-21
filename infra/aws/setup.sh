#!/bin/bash
# One-time (idempotent) setup in the research account:
#   - S3 bucket for data, checkpoints and results (private, encrypted, versioning off);
#   - IAM role + instance profile for the training nodes: that bucket only, plus SSM;
#   - security group: SSH from this machine's current public IP only (re-run to update the IP);
#   - EC2 key pair, private key saved to ~/.ssh/mrcogito-train.pem.
# Usage: bash infra/aws/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "== bucket ${MC_BUCKET}"
if ! aws s3api head-bucket --bucket "$MC_BUCKET" >/dev/null 2>&1; then
  aws s3api create-bucket --bucket "$MC_BUCKET" >/dev/null
  aws s3api put-public-access-block --bucket "$MC_BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "$MC_BUCKET" --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-tagging --bucket "$MC_BUCKET" --tagging "TagSet=[{Key=Project,Value=${MC_PROJECT_TAG}}]"
  echo "created"
else
  echo "exists"
fi

echo "== role ${MC_ROLE}"
if ! aws iam get-role --role-name "$MC_ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$MC_ROLE" --tags "Key=Project,Value=${MC_PROJECT_TAG}" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$MC_ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  echo "created"
fi
aws iam put-role-policy --role-name "$MC_ROLE" --policy-name bucket-rw --policy-document "{
  \"Version\":\"2012-10-17\",
  \"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::${MC_BUCKET}\"},
    {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:AbortMultipartUpload\"],
     \"Resource\":\"arn:aws:s3:::${MC_BUCKET}/*\"}]}"
if ! aws iam get-instance-profile --instance-profile-name "$MC_ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$MC_ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$MC_ROLE" --role-name "$MC_ROLE"
  echo "instance profile created (IAM needs ~10 s before first use)"
  sleep 10
fi

echo "== security group ${MC_SG}"
VPC=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=${MC_SG}" "Name=vpc-id,Values=${VPC}" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name "$MC_SG" --description "MrCogito training - SSH from the author IP only" \
    --vpc-id "$VPC" --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=${MC_PROJECT_TAG}}]" \
    --query GroupId --output text)
  echo "created $SG"
fi
MYIP=$(curl -s https://checkip.amazonaws.com)
# drop old SSH rules, then allow only the current IP
OLD=$(aws ec2 describe-security-groups --group-ids "$SG" \
  --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text)
for cidr in $OLD; do
  [ "$cidr" = "${MYIP}/32" ] && continue
  aws ec2 revoke-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$cidr" >/dev/null
done
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "${MYIP}/32" >/dev/null 2>&1 || true
echo "SSH allowed from ${MYIP}/32"

echo "== key pair ${MC_KEY}"
if ! aws ec2 describe-key-pairs --key-names "$MC_KEY" >/dev/null 2>&1 && [ -f "$MC_KEY_FILE" ]; then
  # a second region: reuse the private key we already hold (key pairs are per region; never overwrite the file)
  PUB=$(mktemp); ssh-keygen -y -f "$MC_KEY_FILE" > "$PUB"
  aws ec2 import-key-pair --key-name "$MC_KEY" --public-key-material "fileb://$PUB" \
    --tag-specifications "ResourceType=key-pair,Tags=[{Key=Project,Value=${MC_PROJECT_TAG}}]" >/dev/null
  rm -f "$PUB"
  echo "imported the public key of $MC_KEY_FILE into ${AWS_REGION}"
elif ! aws ec2 describe-key-pairs --key-names "$MC_KEY" >/dev/null 2>&1; then
  umask 077
  aws ec2 create-key-pair --key-name "$MC_KEY" --key-type ed25519 \
    --tag-specifications "ResourceType=key-pair,Tags=[{Key=Project,Value=${MC_PROJECT_TAG}}]" \
    --query KeyMaterial --output text > "$MC_KEY_FILE"
  chmod 600 "$MC_KEY_FILE"
  echo "created, private key at $MC_KEY_FILE"
else
  [ -f "$MC_KEY_FILE" ] && echo "exists ($MC_KEY_FILE)" || echo "WARNING: key pair exists but $MC_KEY_FILE is missing"
fi
