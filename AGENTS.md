# AGENTS.md

Guidance for cloud agents working in this repository.

## Cursor Cloud specific instructions

### What this repo is

MrCogito is a **PyTorch research codebase** (encode → reason → decode concept encoder). There is no web server or docker-compose stack. End-to-end validation means: install deps → run unit tests → smoke-train a tiny model → optionally analyze/evaluate a checkpoint.

### Dependencies

- **Python 3.12** (see `.python-version`), managed with **[uv](https://docs.astral.sh/uv/)** and `uv.lock`.
- Install / refresh: `uv sync` (creates `.venv/` at repo root).
- Copy `.env.example` → `.env` before training or HF downloads. `HF_HOME=./Cache/hf_home` keeps caches inside the repo.

### Quality gates

- **Tests:** `uv run pytest tests/ -v` (79 passed, 9 skipped as of setup; no ruff/black/mypy/pre-commit configured).
- **PyTorch smoke:** `uv run python verification/torch_test.py`
- **Model forward smoke:** `PYTHONPATH=. uv run python verification/verify_perceiver.py` (pytest sets `pythonpath = ["."]` automatically).

### Hardware in Cloud VMs

- Cloud agent VMs are typically **CPU-only** (CUDA wheels install, but `torch.cuda.is_available()` is false).
- Use tiny configs for smoke training; full GPU training belongs on Odra/Polonez (see `.cursor/skills/experiment-run/SKILL.md`).

### Smoke training (canonical entrypoint)

`training/train_perceiver_denoise.py` is the main training script. For a fast local smoke run:

```bash
export WANDB_MODE=disabled   # required: --report_to none does NOT skip init_wandb()
uv run python training/train_perceiver_denoise.py \
  --dataset_name Salesforce/wikitext \
  --dataset_name_subset wikitext-2-raw-v1 \
  --hidden_size 64 --num_hidden_layers 2 --concept_num 8 --decoder_num_layers 1 \
  --intermediate_size 128 --max_seq_length 64 --max_steps 5 \
  --per_device_train_batch_size 2 --train_num_proc 2 --test_num_proc 1 \
  --save_strategy no --eval_strategy no --report_to none \
  --dataloader_num_workers 0 --output_dir ./Cache/Training/smoke_test
```

**Avoid** `JeanKaddour/minipile` for smoke tests — it tokenizes 1M rows and takes a long time on CPU.

Default production dataset/tokenizer: `JeanKaddour/minipile` + `answerdotai/ModernBERT-base` (see `README.md`).

### Concept analysis & evaluation

- **Concept geometry:** `uv run python analysis/run_concept_analysis.py --model_path <checkpoint_dir> --model_type perceiver_denoise`
- **GLUE eval:** `uv run python evaluation/evaluate_model_on_glue.py --model_path <checkpoint> --task mrpc`
- First HF Hub access may need `HF_TOKEN` in `.env` for gated assets; public datasets/tokenizers work without it.

### W&B caveat

`init_wandb()` in `training/utils_training.py` runs unconditionally. Disable logging with `WANDB_MODE=disabled` or `WANDB_DISABLED=true`, not only `--report_to none`.

### Remote training

Real multi-GPU runs use `scripts/train_perceiver_denoise_multigpu.sh` on Odra/Polonez. Do not expect meaningful training throughput on CPU-only cloud VMs.
