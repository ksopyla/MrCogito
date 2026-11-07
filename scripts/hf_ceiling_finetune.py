#!/usr/bin/env python
"""Fine-tune a pretrained Hub causal LM on a text-checks / E33s data folder: a ceiling probe, not a model of ours.

It answers one question before our own models spend their budget: how well can a well-pretrained model of the same
size class do on this data? (E33s, 2026-10-22: SmolLM2-135M on the written-chain stages, so the real-QA gates have a
measured ceiling, as the design rules require.) Our models train through train_concept_pretraining.py; this tool
only wraps the HF Trainer around the same manifest loader and collator. The `read_targets` column (per-round targets
of our loop) is dropped: a plain transformer has no rounds.

    uv run accelerate launch --num_processes 2 --main_process_port 29504 scripts/hf_ceiling_finetune.py \\
        --model HuggingFaceTB/SmolLM2-135M --data <data dir> --out <dir> --tokens 5e7 [--lr 1e-4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hub id or local checkpoint (a previous stage's final)")
    p.add_argument("--data", required=True, help="data folder with manifest.json, tokenizer/, text_checks_meta.json")
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=float, required=True, help="training tokens (sets the step count)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rows_per_step", type=int, default=32, help="global batch in rows")
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_ckpt", action="store_true")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from data.data_collators import DataCollatorForCausalLM
    from data.dataset_preprocess import load_pretokenized_mix

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "text_checks_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(str(data / "tokenizer"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds, eval_ds = load_pretokenized_mix(str(data / "manifest.json"))
    drop = [c for c in ("read_targets",) if c in train_ds.column_names]
    if drop:
        train_ds = train_ds.remove_columns(drop)
        eval_ds = eval_ds.remove_columns([c for c in drop if c in eval_ds.column_names])
    eval_ds = eval_ds.select(range(min(256, len(eval_ds))))

    world = int(os.environ.get("WORLD_SIZE", "1"))
    accum = max(1, args.rows_per_step // (args.per_device_batch * world))
    mean_row = float(meta["mix"]["mean_row_tokens"])
    steps = max(1, math.ceil(args.tokens / (args.per_device_batch * world * accum * mean_row)))
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    targs = TrainingArguments(
        output_dir=str(out / "train"), per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch, gradient_accumulation_steps=accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=args.warmup, weight_decay=0.1,
        max_steps=steps, bf16=True, logging_steps=20, eval_strategy="steps", eval_steps=max(1, steps // 4),
        save_strategy="no", report_to="none", seed=args.seed, dataloader_num_workers=2,
        remove_unused_columns=False, max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorForCausalLM(tokenizer=tok))
    if trainer.is_world_process_zero():
        print(json.dumps({"model": args.model, "data": str(data), "steps": steps, "accum": accum, "world": world,
                          "tokens": args.tokens, "lr": args.lr, "mean_row_tokens": mean_row}), flush=True)
    trainer.train()
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        final = out / "final"
        trainer.save_model(str(final))
        tok.save_pretrained(str(final))
        (out / "model_path").write_text(str(final.resolve()))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("final", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
