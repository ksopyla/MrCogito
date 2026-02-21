# PowerShell script to test the Masked Diffusion model locally
# This is a scaled-down version for testing architecture and stability on RTX 3080

$projectRoot = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $projectRoot "Cache"
$env:HF_DATASETS_CACHE = Join-Path $projectRoot "Cache\Datasets"

Remove-Item Env:\TRANSFORMERS_CACHE -ErrorAction SilentlyContinue

# Disable wandb locally for simple test
$env:WANDB_MODE = "disabled"

# Run testing for Diffusion model
# 2 layers, 256 hidden size, small batch size, max_steps 50
poetry run python training/train_diffusion.py `
    --hidden_size 256 `
    --num_hidden_layers 2 `
    --concept_num 128 `
    --decoder_layers 2 `
    --intermediate_size 512 `
    --dataset_name "Salesforce/wikitext" `
    --dataset_name_subset "wikitext-2-raw-v1" `
    --tokenizer_name "answerdotai/ModernBERT-base" `
    --dataset_cache_dir "./Cache/Datasets" `
    --per_device_train_batch_size 4 `
    --per_device_eval_batch_size 4 `
    --gradient_accumulation_steps 1 `
    --learning_rate 3e-4 `
    --max_steps 50 `
    --warmup_steps 10 `
    --logging_steps 5 `
    --eval_strategy "steps" `
    --eval_steps 25 `
    --save_strategy "steps" `
    --save_steps 25 `
    --load_best_model_at_end True `
    --metric_for_best_model "eval_loss" `
    --output_dir "./Cache/Training/" `
    --seed 42 `
    --report_to "none" `
    --concept_losses "none" `
    --loss_weighting "fixed" `
    --torch_compile False `
    --torch_compile_dynamic False `
    --bf16 `
    --optim "adamw_torch_fused" `
    --lr_scheduler_type "linear"

Write-Host "Local testing completed!"
