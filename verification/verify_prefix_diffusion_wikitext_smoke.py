"""
Run a tiny end-to-end smoke test for prefix diffusion on WikiText.

This verifies that the real training entrypoint can:
1. load a Wikipedia-derived dataset from the new prefix-diffusion family
2. tokenize and split prefix/suffix examples with the sentence-boundary collator
3. execute a few training steps without touching the full Polonez launcher

Run:
    poetry run python verification/verify_prefix_diffusion_wikitext_smoke.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "training" / "train_prefix_diffusion.py"
    agent_memory = repo_root / "agent_memory"
    agent_memory.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="prefix_wikitext_smoke_",
        dir=agent_memory,
    ) as temp_dir:
        output_dir = Path(temp_dir) / "outputs"
        logging_dir = Path(temp_dir) / "logs"

        command = [
            sys.executable,
            str(train_script),
            "--hidden_size", "64",
            "--token_embedding_dim", "32",
            "--num_hidden_layers", "2",
            "--concept_num", "16",
            "--intermediate_size", "128",
            "--decoder_layers", "1",
            "--use_bixt", "True",
            "--bixt_token_ffn", "True",
            "--t_min", "0.3",
            "--label_smoothing", "0.1",
            "--elbo_weight", "True",
            "--dataset_name", "Salesforce/wikitext",
            "--dataset_name_subset", "wikitext-2-v1",
            "--tokenizer_name", "answerdotai/ModernBERT-base",
            "--max_seq_length", "128",
            "--test_size_percent", "0.1",
            "--prefix_ratio_min", "0.7",
            "--prefix_ratio_max", "0.8",
            "--split_strategy", "sentence_boundary",
            "--min_prefix_content", "8",
            "--min_suffix_content", "8",
            "--min_total_content_tokens", "24",
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "2",
            "--gradient_accumulation_steps", "1",
            "--learning_rate", "3e-4",
            "--num_train_epochs", "1",
            "--max_steps", "2",
            "--warmup_steps", "0",
            "--weight_decay", "0.01",
            "--max_grad_norm", "1.0",
            "--logging_steps", "1",
            "--eval_strategy", "no",
            "--save_strategy", "no",
            "--output_dir", str(output_dir),
            "--logging_dir", str(logging_dir),
            "--dataloader_num_workers", "0",
            "--dataloader_pin_memory", "False",
            "--bf16", "False",
            "--report_to", "none",
            "--optim", "adamw_torch",
            "--remove_unused_columns", "False",
            "--save_safetensors", "True",
            "--overwrite_output_dir", "True",
            "--seed", "42",
        ]

        env = os.environ.copy()
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        print("Running prefix diffusion WikiText smoke test...")
        print(" ".join(command))
        subprocess.run(command, cwd=repo_root, env=env, check=True)
        print("Smoke test passed.")


if __name__ == "__main__":
    main()
