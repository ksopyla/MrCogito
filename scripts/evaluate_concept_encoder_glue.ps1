# PowerShell script to evaluate the Concept Encoder model on the GLUE benchmark.
# Usage: .\scripts\evaluate_concept_encoder_glue.ps1 [-ModelPath "path/to/model"] [-Task "mrpc"]

param (
    [string]$ModelPath = "Cache/Training/weighted_mlm_H512L2C128_20251123_213949/weighted_mlm_H512L2C128_20251123_213949",
    [string]$Task = "mrpc"
)

# Activate poetry environment first if not already activated
# poetry shell

# Set HuggingFace cache directories to use local Cache folder
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $projectRoot "Cache"
$env:HF_DATASETS_CACHE = Join-Path $projectRoot "Cache\Datasets"

# Unset the old, deprecated TRANSFORMERS_CACHE variable to prevent warnings
Remove-Item Env:\TRANSFORMERS_CACHE -ErrorAction SilentlyContinue

# --- Evaluation Configuration ---
$TokenizerName = "bert-base-cased"

Write-Host "=== GLUE Evaluation Script for Concept Encoder ==="
Write-Host "Configuration:"
Write-Host "  - Project Root: $projectRoot"
Write-Host "  - HF Cache: $env:HF_HOME"
Write-Host "  - Model Path: $ModelPath"
Write-Host "  - Task: $Task"
Write-Host "  - Tokenizer: $TokenizerName"
Write-Host ""

# Check if model path exists
if (-not (Test-Path $ModelPath)) {
    Write-Warning "Model path does not seem to exist: $ModelPath"
    Write-Warning "Please check the path."
}

Write-Host "Starting evaluation..."

# Run the evaluation script
# Using 'weighted_mlm' model type triggers the new Weighted Classification Head
python training/evaluate_model_on_glue.py `
    --model_type "weighted_mlm" `
    --model_name_or_path "$ModelPath" `
    --tokenizer_name "$TokenizerName" `
    --task "$Task" `
    --batch_size 32 `
    --epochs 5 `
    --learning_rate 2e-5 `
    --visualize `
    --save_model

Write-Host ""
Write-Host "GLUE evaluation completed!"
