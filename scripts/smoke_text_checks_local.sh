#!/usr/bin/env bash
# Local smoke of the text capability checks (draft text-v0): tiny dense / E31c / E31c+loop models
# trained for a few steps on the text-checks mix, then scored by evaluation/text_checks_eval.py.
# It checks that the data, the trainer and the scorer fit together — not what the models learn.
#
#   bash scripts/smoke_text_checks_local.sh                 # all three arches
#   ARCHES="e31c" MAX_STEPS=20 bash scripts/smoke_text_checks_local.sh
#
# Needs the smoke data first:
#   uv run python scripts/build_text_checks_data.py --stories smoke --out_dir Cache/text_checks/smoke \
#       --seq_len 1024 --n_lang_rows 300 --n_world_rows 600 --eval_lengths 512 1024 2048 --eval_items 6
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-Cache/text_checks/smoke}"
OUT="${OUT:-Cache/text_checks/smoke_runs}"
ARCHES="${ARCHES:-dense e31c e31c_loop}"
MAX_STEPS="${MAX_STEPS:-60}"
SEQ="${SEQ:-1024}"
BATCH="${BATCH:-4}"
LR="${LR:-1e-3}"
EVAL_LENGTHS="${EVAL_LENGTHS:-512 1024 2048}"
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export WANDB_MODE=disabled

TINY=(
    --model_family perceiver_ar --objective_variant causal_lm --decoder_type causal_ar
    --hidden_size 128 --intermediate_size 256 --token_embedding_dim 128
    --num_attention_heads 4 --num_kv_heads 1 --head_dim 32
    --par_pre_layers 1 --par_pre_window 16 --par_global_layers 1 --num_hidden_layers 2 --par_block 16
    --par_ngram_orders 2,3 --par_ngram_buckets 1024 --par_value_embed_layers "" --par_value_embed_dim 16
    --par_nope_every 0 --rope_theta 10000 --attn_backend sdpa --attn_pad_multiple 1
    --logit_softcap 30 --z_loss 1e-4 --use_liger False --chunked_ce_block_size 256
)
E31C=(
    --message_write latent_memory --message_raw_window 256 --lm_read closed --lm_context page_bidir
    --lm_window 256 --lm_stride 192 --lm_latents 8 --lm_latent_dim 64 --lm_heads 2 --lm_writer_dim 64
    --lm_enc_layers 1 --lm_rounds 1 --lm_reader_tokens 1 --lm_addr none --lm_slot_pos reader
)

for arch in $ARCHES; do
    case "$arch" in
        dense)     ARGS=(--par_mode dense) ;;
        e31c)      ARGS=(--par_mode perceiver "${E31C[@]}") ;;
        e31c_loop) ARGS=(--par_mode perceiver "${E31C[@]}" --message_loop_rounds 4
                         --message_loop_exit_aux 0.3 --message_loop_exit_targets answer) ;;
        *) echo "unknown arch $arch"; exit 2 ;;
    esac
    run="$OUT/$arch"
    echo "=== train $arch → $run"
    uv run python training/train_concept_pretraining.py \
        "${TINY[@]}" "${ARGS[@]}" \
        --pretokenized_manifest "$DATA/manifest.json" --tokenizer_name "$DATA/tokenizer" \
        --max_seq_length "$SEQ" --batch_packing_mode none \
        --per_device_train_batch_size "$BATCH" --per_device_eval_batch_size "$BATCH" \
        --gradient_accumulation_steps 1 --learning_rate "$LR" --warmup_steps 5 \
        --max_steps "$MAX_STEPS" --logging_steps 10 --eval_strategy steps --eval_steps "$MAX_STEPS" \
        --max_eval_samples 16 --save_strategy no --output_dir "$run" --logging_dir "$run/logs" \
        --seed 0 --optim adamw_torch --weight_decay 0.1 --max_grad_norm 1.0 \
        --lr_scheduler_type cosine --report_to none --overwrite_output_dir True \
        --remove_unused_columns True --disable_tqdm True --dataloader_num_workers 0 \
        --prediction_loss_only True 2>&1 | grep -viE "warn|^\s*$" | tail -25
    final=$(ls -td "$run"/*/final 2>/dev/null | head -1 || true)
    [ -n "$final" ] || { echo "no final/ under $run"; exit 1; }
    echo "=== eval $arch ($final)"
    # shellcheck disable=SC2086
    uv run python evaluation/text_checks_eval.py --checkpoint "$final" --device cpu \
        --items "$DATA/eval/id.jsonl" --out "$run/text_checks_id.json" --lengths $EVAL_LENGTHS \
        --max_items_per_cell 3 2>&1 | grep -viE "warn" | tail -30
    if [ "$arch" != "dense" ]; then
        echo "=== eval $arch, notebook off"
        # shellcheck disable=SC2086
        uv run python evaluation/text_checks_eval.py --checkpoint "$final" --device cpu \
            --items "$DATA/eval/id.jsonl" --out "$run/text_checks_id_none.json" --lengths $EVAL_LENGTHS \
            --max_items_per_cell 3 --message_override none --no_removed 2>&1 | grep -viE "warn" | tail -3
    fi
done
