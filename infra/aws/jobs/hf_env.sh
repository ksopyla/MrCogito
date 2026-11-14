#!/bin/bash
# Sourced by AWS node jobs that read gated Hugging Face datasets (the Nemotron-CC family): exports HF_TOKEN from AWS
# Secrets Manager secret MC_HF_SECRET (default mrcogito/hf-token, us-east-1), which the node role mrcogito-train-node
# may read (policy hf-secret-read; the author stored the token and granted the read on 2026-11-14). Nothing here prints
# the token. Without it HF_TOKEN stays unset and gated sources fail loudly in their own build step.
_mc_hf=$(aws secretsmanager get-secret-value --region "${MC_SECRET_REGION:-us-east-1}" \
  --secret-id "${MC_HF_SECRET:-mrcogito/hf-token}" --query SecretString --output text 2>/dev/null) || _mc_hf=""
[ "$_mc_hf" = None ] && _mc_hf=""
if [ -n "$_mc_hf" ]; then
  export HF_TOKEN="$_mc_hf"
  echo "[hf_env] Hugging Face token read from Secrets Manager (${#_mc_hf} characters)"
else
  echo "[hf_env] no Hugging Face token readable: gated sources will fail"
fi
unset _mc_hf
