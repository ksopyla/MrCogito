# Local smoke test for the maintained perceiver denoising stack.

$ErrorActionPreference = "Stop"

$env:WANDB_MODE = "disabled"
$env:TOKENIZERS_PARALLELISM = "false"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host "=== Perceiver Denoising Local Smoke Test ===" -ForegroundColor Cyan

poetry run python training/train_concept_pretraining.py `
    --hidden_size 64 `
    --token_embedding_dim 64 `
    --num_hidden_layers 2 `
    --concept_num 8 `
    --intermediate_size 128 `
    --decoder_num_layers 2 `
    --use_bixt `
    --deletion_rate 0.5 `
    --dataset_name JeanKaddour/minipile `
    --tokenizer_name answerdotai/ModernBERT-base `
    --max_seq_length 64 `
    --num_train_epochs 1 `
    --learning_rate 3e-4 `
    --per_device_train_batch_size 4 `
    --max_steps 20 `
    --logging_steps 5 `
    --eval_strategy steps `
    --eval_steps 10 `
    --output_dir Cache/Training `
    --report_to none `
    --no_cuda `
    --bf16 False

Write-Host "=== Perceiver Denoising Local Test PASSED ===" -ForegroundColor Green
