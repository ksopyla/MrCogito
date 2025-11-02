# PowerShell script to train the weighted MLM model
# This uses the simplified weighted combination approach for initial experiments

# Activate poetry environment first if not already activated
# poetry shell

# Run training with weighted MLM model (Micro-2 config: 21M params)
python training/mlm_training.py `
    --model_type weighted_mlm `
    --hidden_size 384 `
    --num_hidden_layers 3 `
    --concept_size 64 `
    --mlm_probability 0.15 `
    --max_seq_length 256 `
    --dataset_name "Salesforce/wikitext" `
    --dataset_name_subset "wikitext-103-raw-v1" `
    --per_device_train_batch_size 24 `
    --per_device_eval_batch_size 24 `
    --gradient_accumulation_steps 2 `
    --learning_rate 5e-4 `
    --num_train_epochs 10 `
    --warmup_steps 1000 `
    --logging_steps 50 `
    --eval_strategy "steps" `
    --eval_steps 500 `
    --save_steps 1000 `
    --output_dir "./Cache/Training/"
    --seed 42
    --bf16

Write-Host "Training completed!"
