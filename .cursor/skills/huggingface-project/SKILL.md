---
name: huggingface-project
description: Publish and inspect Concept Encoder artifacts on the Hugging Face Hub. Use when uploading checkpoints, generating model cards, checking the ksopyla namespace, searching the Hub via MCP for related models, datasets, or papers, or comparing Hub baselines before release. Not for general research planning or run bookkeeping.
---

# HuggingFace Project Integration

Use this skill for Hugging Face Hub actions and Hub-specific discovery, not for experiment design or training/evaluation tracking.

## HF Space & Auth
- Authenticated user: `ksopyla` — https://huggingface.co/ksopyla
- Models uploaded as `ksopyla/concept-encoder-{run_name}`
- HF token: `HF_TOKEN` in `.env` (or `HUGGINGFACE_TOKEN`)
- MCP server: `user-hf-mcp-server` for searching models, datasets, papers

## Uploading Models to HF
Only upload after evaluation results are promising (see experiment-tracking workflow).
Interactive script lists local models from `Cache/Training`, matches with eval reports, uploads with auto-generated model card:
```powershell
# Windows
poetry run python scripts/upload_model_to_hf.py
# Linux (Polonez/Odra)
bash scripts/upload_model_to_hf.sh
# Non-interactive
poetry run python scripts/upload_model_to_hf.py --run-name perceiver_mlm_H512L6C128_20260208
# List only
poetry run python scripts/upload_model_to_hf.py --list-only
```
