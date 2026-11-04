#!/bin/bash
# Launch one training node. It clones BRANCH over HTTPS, runs `uv sync`, then runs JOB (a script path inside the
# repo, run with bash from the repo root) in a tmux session "job", logging to /opt/dlami/nvme/job.log and S3.
# Safety: the node powers itself off after MAX_HOURS (shutdown behaviour = terminate), so a forgotten node costs at
# most MAX_HOURS × price. JOB should end with `sudo shutdown -h now` (infra/aws/node_run.sh does) to stop billing.
#
# Usage: bash infra/aws/launch.sh NAME MAX_HOURS JOB [BRANCH] [INSTANCE_TYPE]
#   e.g. bash infra/aws/launch.sh e33n-smoke 2 infra/aws/jobs/smoke.sh e33n-lsa-testbed
set -euo pipefail
source "$(dirname "$0")/env.sh"

NAME="$1"; MAX_HOURS="$2"; JOB="$3"; BRANCH="${4:-e33n-lsa-testbed}"; TYPE="${5:-$MC_INSTANCE_TYPE}"

case "$TYPE" in
  p5.*|g6*|g5*) AMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id ;;
  *)            AMI_PARAM=/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id ;;
esac
AMI=$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)
VPC=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=${MC_SG}" "Name=vpc-id,Values=${VPC}" \
  --query 'SecurityGroups[0].GroupId' --output text)

USERDATA=$(mktemp)
cat > "$USERDATA" <<EOF
#!/bin/bash
# hard stop: power off (= terminate) after ${MAX_HOURS} h whatever the job does
shutdown -h +$((MAX_HOURS * 60)) "max lifetime reached" || true
export HOME=/home/ubuntu
DISK=/opt/dlami/nvme
if [ ! -d \$DISK ]; then
  DEV=\$(lsblk -dpno NAME,MODEL | grep -i "Instance Storage" | head -1 | cut -d' ' -f1)
  mkdir -p \$DISK
  if [ -n "\$DEV" ]; then mkfs.ext4 -q -F \$DEV && mount \$DEV \$DISK; fi
fi
chown ubuntu:ubuntu \$DISK
if ! command -v aws >/dev/null; then   # stock Ubuntu (CPU nodes) has no AWS CLI; the Deep Learning AMI does
  apt-get update -qq && apt-get install -y -qq unzip >/dev/null
  curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
  unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install
fi
echo "MC_BUCKET=${MC_BUCKET}" > /etc/mrcogito.env
echo "MC_NAME=${NAME}" >> /etc/mrcogito.env
echo "MC_DISK=\$DISK" >> /etc/mrcogito.env
sudo -u ubuntu bash -lc '
  set -e
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=\$HOME/.local/bin:\$PATH
  cd /opt/dlami/nvme
  git clone --branch ${BRANCH} --single-branch ${MC_REPO_URL} MrCogito
  cd MrCogito
  tmux new-session -d -s job "bash infra/aws/node_run.sh ${JOB} 2>&1 | tee -a /opt/dlami/nvme/job.log"
'
EOF

ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$MC_KEY" \
  --security-group-ids "$SG" --iam-instance-profile "Name=${MC_ROLE}" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${MC_ROOT_GB},VolumeType=gp3,DeleteOnTermination=true}" \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=${MC_PROJECT_TAG}},{Key=MaxHours,Value=${MAX_HOURS}}]" \
  --user-data "file://${USERDATA}" --query 'Instances[0].InstanceId' --output text)
rm -f "$USERDATA"
echo "launched $NAME = $ID ($TYPE, max ${MAX_HOURS} h ≈ \$$(python3 -c "print(f'{$MAX_HOURS*$MC_PRICE_PER_HOUR:.0f}')") cap)"
aws ec2 wait instance-running --instance-ids "$ID"
IP=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "running at $IP — ssh -i $MC_KEY_FILE ubuntu@$IP ; tmux attach -t job"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launch $NAME $ID $TYPE max ${MAX_HOURS}h job ${JOB} branch ${BRANCH}" >> "$(dirname "$0")/runs.log"
