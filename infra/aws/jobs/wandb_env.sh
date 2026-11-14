#!/bin/bash
# Sourced by AWS node jobs (e33n_run.sh): W&B logs live when the node can read the API key from AWS Secrets Manager,
# offline (files in the run folder, synced to S3, uploadable later with `wandb sync`) otherwise.
# The key lives in secret MC_WANDB_SECRET (default mrcogito/wandb-api-key, us-east-1); the node role
# mrcogito-train-node needs secretsmanager:GetSecretValue on that one secret. The author stores the key and grants the
# read (author's choice, 2026-11-11: "add them to secret manager for future usage"); nothing here ever prints the key.
# Before going online the key is checked against the W&B API (also proves the node can reach it), because an
# online wandb.init that cannot connect stops the training run; any failure falls back to offline.
_mc_wb_key="${WANDB_API_KEY:-}"
if [ -z "$_mc_wb_key" ]; then
  _mc_wb_key=$(aws secretsmanager get-secret-value --region "${MC_SECRET_REGION:-us-east-1}" \
    --secret-id "${MC_WANDB_SECRET:-mrcogito/wandb-api-key}" --query SecretString --output text 2>/dev/null) || _mc_wb_key=""
  [ "$_mc_wb_key" = None ] && _mc_wb_key=""
fi
_mc_wb_entity=""
if [ -n "$_mc_wb_key" ]; then
  # the key goes to curl on stdin (-K -), never on its command line
  _mc_wb_entity=$(printf 'user = "api:%s"\n' "$_mc_wb_key" | curl -sf -m 20 -K - -H 'Content-Type: application/json' \
    -d '{"query":"{ viewer { entity } }"}' https://api.wandb.ai/graphql 2>/dev/null \
    | sed -n 's/.*"entity" *: *"\([^"]*\)".*/\1/p')
fi
if [ -n "$_mc_wb_entity" ]; then
  export WANDB_API_KEY="$_mc_wb_key" WANDB_MODE=online WANDB_ENTITY="${WANDB_ENTITY:-ksopyla}"
  echo "[wandb_env] W&B online (key accepted for ${_mc_wb_entity}; logging to ${WANDB_ENTITY})"
else
  unset WANDB_API_KEY
  export WANDB_MODE=offline
  echo "[wandb_env] W&B offline (no key in Secrets Manager, no read permission, or W&B unreachable)"
fi
unset _mc_wb_key _mc_wb_entity
