# AGENTS.md

Guidance for cloud agents and automated development environments working on the MrCogito / Concept Encoder codebase.

## Cursor Cloud specific instructions

### What this repo is

This is a **CLI-driven ML research codebase**, not a web app. There are no long-running services (no API server, database, or Docker Compose stack). End-to-end validation means: install deps → run unit tests → run verification scripts → optionally run a short training smoke test.

### Package manager and Python

- **Python 3.12** (see `.python-version`)
- **`uv`** for dependency management (`uv sync`, `uv run …`)
- On fresh cloud VMs, `uv` may not be on `PATH`. Install once if needed:
  `curl -LsSf https://astral.sh/uv/install.sh | sh` then `export PATH="$HOME/.local/bin:$PATH"`
- After every pull, refresh deps: `uv sync` (also runs automatically via the VM update script)

### Environment variables

Copy `.env.example` to `.env` for local paths and optional tokens. For smoke tests without W&B:

```bash
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HF_HOME=./Cache/hf_home
export PYTHONPATH=.
```

`HF_TOKEN` and `WANDB_API_KEY` are optional for public datasets and smoke runs.

### Linting

**No project linter is configured** (no ruff, flake8, mypy, or pre-commit). The quality gate is **pytest**.

### Tests and verification

| Command | Purpose |
|---|---|
| `uv run pytest tests/ -v` | Unit tests (79+ pass; pytest sets `pythonpath = ["."]` automatically) |
| `uv run python verification/torch_test.py` | PyTorch install + device check |
| `PYTHONPATH=. uv run python verification/verify_perceiver.py` | Perceiver forward-pass smoke |
| `PYTHONPATH=. uv run python verification/verify_sparse_decoding.py` | Sparse decoding correctness |
| `PYTHONPATH=. uv run python verification/verify_dimension_inversion.py` | Dimension inversion (1 known pre-existing failure in Test 8) |

Direct script invocations under `verification/` need `PYTHONPATH=.` because the repo is application-style (no installable package).

### E2E training smoke (hello-world)

Cloud VMs here are **CPU-only** (no NVIDIA GPU). Use a tiny model and **WikiText-2** (fast download); avoid `JeanKaddour/minipile` for smoke tests — it is large and can hang on first download.

```bash
uv run python training/train_perceiver_denoise.py \
  --hidden_size 64 --token_embedding_dim 64 \
  --num_hidden_layers 2 --concept_num 8 \
  --intermediate_size 128 --decoder_num_layers 2 \
  --use_bixt --deletion_rate 0.5 \
  --dataset_name Salesforce/wikitext \
  --dataset_name_subset wikitext-2-v1 \
  --tokenizer_name answerdotai/ModernBERT-base \
  --max_seq_length 64 --max_steps 20 \
  --per_device_train_batch_size 4 \
  --output_dir Cache/Training/smoke_test --report_to none \
  --bf16 False --use_cpu
```

Checkpoints land under `Cache/Training/`. Full-scale training runs on remote GPU servers (Polonez/Odra); see `.cursor/skills/experiment-run/SKILL.md`.

### Key entry points

- **Training:** `training/train_perceiver_denoise.py` (primary), `training/train_mlm.py`
- **Evaluation:** `evaluation/evaluate_model_on_glue.py`, `evaluation/evaluate_on_benchmark.py`
- **Analysis:** `analysis/run_concept_analysis.py`

### Git workflow note

Local dev docs target the `dev` branch; cloud agents on `main` should branch as `cursor/<name>-2ff9` per cloud task instructions.
