# PowerShell script to evaluate the Concept Encoder model on the GLUE benchmark.

# Activate poetry environment first if not already activated
# poetry shell

# Set HuggingFace cache directories to use local Cache folder
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $projectRoot "Cache"
$env:HF_DATASETS_CACHE = Join-Path $projectRoot "Cache\Datasets"

# Unset the old, deprecated TRANSFORMERS_CACHE variable to prevent warnings
Remove-Item Env:\TRANSFORMERS_CACHE -ErrorAction SilentlyContinue

# --- Evaluation Configuration ---
# This script is configured to evaluate the latest trained Concept Encoder model.
# Model path, tokenizer, and task can be adjusted as needed.

# NOTE: The model path points to the final saved model inside the training output directory.
# Please verify this is the correct model you wish to evaluate.
$modelPath = "./Cache/Training/weighted_mlm_H256L2C128_20251106_171559/weighted_mlm_H256L2C128_20251106_171559"

# Run the evaluation script
python training/evaluate_model_on_glue.py `
    --model_type "weighted_mlm" `
    --model_name_or_path "$modelPath" `
    --tokenizer_name "bert-base-uncased" `
    --task "mrpc" `
    --batch_size 32 `
    --epochs 5 `
    --learning_rate 2e-5 `
    --visualize `
    --save_model

Write-Host "GLUE evaluation completed!"
