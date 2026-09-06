#!/usr/bin/env python
"""E18 Perceiver AR v2 — CPU smoke of the REAL training entrypoint on a synthetic manifest.

Builds a tiny pretokenized manifest (random ids, causal_lm objective) under a temp dir and
runs `training/train_concept_pretraining.py` for a few Muon steps with
`--model_family perceiver_ar`. Exercises: arg validation, factory, collator, chunked CE,
eval with prediction_loss_only, Muon parameter routing.

  uv run python verification/e18_cpu_smoke.py            # 4 steps, sdpa backend, CPU
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def build_manifest(root: Path) -> Path:
    from datasets import Dataset

    random.seed(0)

    def rows(n, lo=200, hi=900):
        out = []
        for _ in range(n):
            L = random.randint(lo, hi)
            out.append({"input_ids": [random.randint(5, 4000) for _ in range(L)] + [2]})
        return out

    Dataset.from_list(rows(64)).save_to_disk(str(root / "train"))
    Dataset.from_list(rows(8)).save_to_disk(str(root / "eval"))
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "mix_id": "e18_smoke",
                "objective": "causal_lm",
                "max_seq_length": 1024,
                "seed": 0,
                "sources": [
                    {
                        "name": "smoke",
                        "weight": 1.0,
                        "train_path": str(root / "train"),
                        "eval_path": str(root / "eval"),
                    }
                ],
            }
        )
    )
    return manifest


def main() -> int:
    steps = int(os.environ.get("SMOKE_STEPS", "4"))
    with tempfile.TemporaryDirectory(prefix="e18smoke_") as td:
        root = Path(td)
        manifest = build_manifest(root)
        cmd = [
            sys.executable, "training/train_concept_pretraining.py",
            "--model_family", "perceiver_ar", "--objective_variant", "causal_lm",
            "--hidden_size", "64", "--token_embedding_dim", "16", "--num_hidden_layers", "2",
            "--intermediate_size", "128", "--par_pre_layers", "1", "--par_pre_window", "64",
            "--par_block", "128", "--num_kv_heads", "1", "--head_dim", "32",
            "--par_ngram_buckets", "256", "--par_value_embed_layers", "0",
            "--attn_backend", os.environ.get("SMOKE_BACKEND", "sdpa"), "--attn_pad_multiple", "64",
            "--use_liger", "False",
            "--pretokenized_manifest", str(manifest),
            "--tokenizer_name", os.environ.get("SMOKE_TOKENIZER", "HuggingFaceTB/SmolLM2-135M"),
            "--max_seq_length", "512", "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "1", "--learning_rate", "0.01",
            "--optimizer", "muon", "--muon_adamw_lr", "2e-4", "--weight_decay", "0.1",
            "--max_steps", str(steps), "--eval_strategy", "steps", "--eval_steps", "2",
            "--save_strategy", "no", "--output_dir", str(root / "out"),
            "--logging_dir", str(root / "logs"), "--logging_steps", "1", "--report_to", "none",
            "--prediction_loss_only", "True", "--remove_unused_columns", "True",
            "--disable_tqdm", "True", "--dataloader_num_workers", "0", "--seed", "1",
            "--max_eval_samples", "4",
        ]
        # Public tokenizer only: drop any (possibly expired) local HF token so the smoke never
        # fails on credentials. Set SMOKE_KEEP_TOKEN=1 to keep the ambient token.
        env = {**os.environ, "WANDB_MODE": "disabled"}
        if not os.environ.get("SMOKE_KEEP_TOKEN"):
            env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            env.pop("HF_TOKEN", None)
            env.pop("HUGGING_FACE_HUB_TOKEN", None)
        proc = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True)
        tail = "\n".join(line for line in (proc.stdout + proc.stderr).splitlines() if line.strip())[-6000:]
        print(tail)
        ok = proc.returncode == 0 and "eval_loss" in (proc.stdout + proc.stderr)
        print("\nE18 CPU SMOKE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
