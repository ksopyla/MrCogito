$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$env:HF_HOME = Join-Path $projectRoot "Cache"
$env:HF_DATASETS_CACHE = Join-Path $projectRoot "Cache\Datasets"
Remove-Item Env:\TRANSFORMERS_CACHE -ErrorAction SilentlyContinue

poetry run python training/train_perceiver_denoise.py `
    --hidden_size 512 `
    --token_embedding_dim 512 `
    --num_hidden_layers 6 `
    --concept_num 128 `
    --intermediate_size 2048 `
    --decoder_num_layers 3 `
    --use_bixt `
    --deletion_rate 0.6 `
    --objective_variant reconstruction `
    --dataset_name "JeanKaddour/minipile" `
    --tokenizer_name "answerdotai/ModernBERT-base" `
    --dataset_cache_dir "./Cache/Datasets" `
    --max_seq_length 512 `
    --per_device_train_batch_size 16 `
    --per_device_eval_batch_size 16 `
    --gradient_accumulation_steps 1 `
    --learning_rate 3e-4 `
    --num_train_epochs 1 `
    --warmup_steps 200 `
    --logging_steps 50 `
    --eval_strategy "steps" `
    --eval_steps 200 `
    --save_steps 200 `
    --output_dir "./Cache/Training/" `
    --seed 42 `
    --report_to "wandb" `
    --bf16
