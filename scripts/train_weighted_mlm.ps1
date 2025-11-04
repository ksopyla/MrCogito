# PowerShell script to train the weighted MLM model
# This uses the simplified weighted combination approach for initial experiments

# Activate poetry environment first if not already activated
# poetry shell

# Set HuggingFace cache directories to use local Cache folder
# Get the project root directory (parent of scripts folder)
$projectRoot = Split-Path -Parent $PSScriptRoot
# Use HF_HOME (new recommended way) - this sets the base cache directory
$env:HF_HOME = Join-Path $projectRoot "Cache"
# Set specific cache directories within HF_HOME structure
$env:HF_DATASETS_CACHE = Join-Path $projectRoot "Cache\Datasets"
# Note: TRANSFORMERS_CACHE is deprecated, but kept for backwards compatibility
# Models will be cached under HF_HOME/models/ by default

# Run training with weighted MLM model (Micro-2 config: 21M params)
python training/mlm_training.py `
    --model_type weighted_mlm `
    --hidden_size 256 `
    --num_hidden_layers 2 `
    --concept_size 128 `
    --mlm_probability 0.15 `
    --max_seq_length 256 `
    --dataset_name "Salesforce/wikitext" `
    --dataset_name_subset "wikitext-103-v1" `
    --dataset_cache_dir "./Cache/Datasets" `
    --per_device_train_batch_size 32 `
    --per_device_eval_batch_size 32 `
    --gradient_accumulation_steps 1 `
    --learning_rate 5e-4 `
    --num_train_epochs 1 `
    --warmup_steps 1000 `
    --logging_steps 500 `
    --eval_strategy "steps" `
    --eval_steps 500 `
    --save_steps 10000 `
    --output_dir "./Cache/Training/" `
    --seed 42 `
    --report_to "wandb" `
    --bf16

Write-Host "Training completed!"
