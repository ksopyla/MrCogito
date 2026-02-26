#!/usr/bin/env python
# coding: utf-8
r"""
Upload Concept Encoder trained models to Hugging Face Hub.

Lists local trained models (from Cache/Training), matches them with Evaluation reports,
displays metrics and parameters, and uploads the selected model with a model card.

Usage:
    # On Polonez/Odra:
    #   bash scripts/upload_model_to_hf.sh

    # On Windows (with local models and eval reports):
    #   poetry run python scripts/upload_model_to_hf.py

    # With explicit paths:
    #   poetry run python scripts/upload_model_to_hf.py --training-dir Cache/Training --reports-dir Cache/Evaluation_reports
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    from huggingface_hub import HfApi, list_models, login
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: poetry install")
    sys.exit(1)

# Load .env for HF token (HUGGINGFACE_TOKEN or HF_TOKEN)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Ensure HF_TOKEN is set from any common env var name so huggingface_hub
# auto-detects it without requiring huggingface-cli login.
import os as _os
for _key in ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
    _tok = _os.environ.get(_key)
    if _tok:
        _os.environ.setdefault("HF_TOKEN", _tok)
        break

# Default paths (relative to project root)
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "Cache" / "Training"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "Cache" / "Evaluation_reports"
HF_USER = "ksopyla"


def get_run_folder_from_model_path(model_path: str) -> str:
    """Extract run folder name from model path (handles both flat and nested structure)."""
    path = model_path.replace("\\", "/").rstrip("/")
    parts = path.split("/")
    # Model path can be: .../run_name/run_name or .../run_name
    for part in reversed(parts):
        if part and re.match(r"^\w+_H\d+L\d+C\d+", part):
            return part
    return ""


def find_model_folder(run_dir: Path) -> Path | None:
    """
    Find the folder containing model files (config.json).
    Training saves to output_dir/run_name/run_name (nested) or output_dir (flat).
    """
    config_in_root = run_dir / "config.json"
    if config_in_root.exists():
        return run_dir

    # Check nested: run_name/run_name/config.json
    nested = run_dir / run_dir.name / "config.json"
    if nested.exists():
        return run_dir / run_dir.name

    return None


def scan_trained_models(training_dir: Path) -> list[dict]:
    """Scan Cache/Training for valid model checkpoints."""
    models = []
    if not training_dir.exists():
        return models

    for run_dir in sorted(training_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # Match run folders: model_type_H512L6C128_... or model_type_20251105-...
        if not re.match(r"^\w+_(H\d+L\d+C\d+|\d{8})", run_dir.name):
            continue

        model_folder = find_model_folder(run_dir)
        if model_folder is None:
            continue

        config_path = model_folder / "config.json"
        if not config_path.exists():
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception:
            config = {}

        # Try trainer_state for training args
        trainer_state_path = run_dir / "trainer_state.json"
        training_info = {}
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path) as f:
                    ts = json.load(f)
                    training_info = ts
            except Exception:
                pass

        models.append({
            "run_name": run_dir.name,
            "model_path": str(model_folder),
            "run_dir": str(run_dir),
            "config": config,
            "trainer_state": training_info,
        })
    return models


def load_evaluation_reports(reports_dir: Path) -> dict[str, list[dict]]:
    """
    Load all evaluation reports and group by run folder.
    Returns: {run_folder: [summary_rows with merged metadata, ...]}
    """
    by_run = {}
    if not reports_dir.exists():
        return by_run

    # Load summary first
    for f in reports_dir.glob("*-summary.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                model_name = str(row.get("model_name", ""))
                run_folder = get_run_folder_from_model_path(model_name)
                if not run_folder:
                    continue
                if run_folder not in by_run:
                    by_run[run_folder] = []
                by_run[run_folder].append(row.to_dict())
        except Exception:
            continue

    # Merge metadata (epochs, wandb_url, etc.) from metadata CSVs
    metadata_by_run = {}
    for f in reports_dir.glob("*-metadata.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                model_name = str(row.get("model_name", ""))
                run_folder = get_run_folder_from_model_path(model_name)
                if not run_folder:
                    continue
                if run_folder not in metadata_by_run:
                    metadata_by_run[run_folder] = row.to_dict()
        except Exception:
            continue

    # Merge metadata into first summary row per run
    for run_folder, meta in metadata_by_run.items():
        if run_folder in by_run and by_run[run_folder]:
            first = by_run[run_folder][0]
            for k, v in meta.items():
                if k not in first or (pd.isna(first.get(k)) and not pd.isna(v)):
                    first[k] = v

    return by_run


def load_results_for_metrics(reports_dir: Path) -> dict[str, list[dict]]:
    """Load detailed results (all metrics per task) grouped by run."""
    by_run = {}
    if not reports_dir.exists():
        return by_run

    for f in reports_dir.glob("*-results.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                model_name = str(row.get("model_name", ""))
                run_folder = get_run_folder_from_model_path(model_name)
                if not run_folder:
                    continue
                if run_folder not in by_run:
                    by_run[run_folder] = []
                by_run[run_folder].append(row.to_dict())
        except Exception:
            continue
    return by_run


def get_uploaded_models(hf_user: str) -> set[str]:
    """Get set of model repo names already on HF (concept-encoder / MrCogito related)."""
    try:
        api = HfApi()
        models = list(api.list_models(author=hf_user))
        return {m.id for m in models}
    except Exception:
        return set()


def _safe(val, default="?"):
    """Return default when val is None, NaN, or empty."""
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return val if str(val).strip() else default


def build_model_card(
    run_name: str,
    model_info: dict,
    eval_data: list[dict],
    results_data: list[dict],
    repo_id: str,
) -> str:
    """Build README.md model card with HF YAML front-matter, architecture details, and metrics."""
    config = model_info.get("config", {})

    # ── Architecture ──────────────────────────────────────────────────────────
    hidden_size    = config.get("hidden_size", "?")
    num_layers     = config.get("num_hidden_layers", "?")
    concept_num    = config.get("concept_num", "?")
    intermediate_size = config.get("intermediate_size", "?")
    use_bixt       = config.get("use_bixt", False)
    decoder_posonly = config.get("decoder_posonly", False)
    max_seq        = config.get("max_sequence_length", 512)

    # Infer training objective from run name / flags
    if "tsdae" in run_name.lower():
        objective = "TSDAE (token-deletion denoising)"
        obj_short = "tsdae"
    elif "diffusion" in run_name.lower():
        objective = "Masked Diffusion Language Modeling"
        obj_short = "diffusion"
    else:
        objective = "Masked Language Modeling (MLM)"
        obj_short = "mlm"

    encoder_arch = "BiXT bidirectional cross-attention" if use_bixt else "standard cross-attention"

    # ── Total parameters ──────────────────────────────────────────────────────
    total_params = "?"
    for row in eval_data:
        tp = row.get("total_params")
        try:
            if tp and not pd.isna(tp):
                total_params = f"{int(tp) / 1e6:.1f}M"
                break
        except (TypeError, ValueError):
            pass
    # Fallback: parse from run name (e.g. "...-61M-...")
    if total_params == "?":
        m = re.search(r"-(\d+)M-", run_name)
        if m:
            total_params = f"{m.group(1)}M"

    # ── Pretraining dataset / epochs ──────────────────────────────────────────
    pretrain_dataset = "JeanKaddour/minipile"
    pretrain_epochs  = "?"
    pretrain_lr      = "?"
    pretrain_wandb   = ""
    run_dir = Path(model_info.get("run_dir", ""))
    if run_dir.exists():
        for fname in ("training_args.json",):
            ta_path = run_dir / fname
            if ta_path.exists():
                try:
                    with open(ta_path) as f:
                        ta = json.load(f)
                    pretrain_dataset = ta.get("dataset_name", pretrain_dataset)
                    pretrain_epochs  = _safe(ta.get("num_train_epochs"), "?")
                    pretrain_lr      = _safe(ta.get("learning_rate"), "?")
                except Exception:
                    pass
        # Try wandb run URL from trainer_state
        ts_path = run_dir / "trainer_state.json"
        if ts_path.exists():
            try:
                with open(ts_path) as f:
                    ts = json.load(f)
                pretrain_wandb = ts.get("wandb_url", "")
            except Exception:
                pass

    # ── GLUE metadata (from first eval row) ───────────────────────────────────
    tokenizer_name = config.get("tokenizer_name", "answerdotai/ModernBERT-base")
    glue_epochs    = "?"
    glue_lr        = "?"
    glue_batch     = "?"
    glue_date      = ""
    glue_wandb     = ""

    if eval_data:
        first = eval_data[0]
        glue_epochs    = _safe(first.get("epochs"), "?")
        tokenizer_name = _safe(first.get("tokenizer_name"), tokenizer_name) or tokenizer_name
        glue_lr        = _safe(first.get("learning_rate"), "?")
        glue_batch     = _safe(first.get("batch_size"), "?")
        glue_date      = _safe(first.get("date"), "")
        glue_wandb     = _safe(first.get("wandb_url"), "")

    # ── Metrics tables ────────────────────────────────────────────────────────
    # Priority concept-quality tasks first
    PRIORITY_TASKS = ["stsb", "mrpc", "qqp", "mnli-matched", "mnli-mismatched"]
    SKIP_METRICS   = {"loss", "runtime", "samples_per_second", "steps_per_second"}

    task_results: dict[str, list[tuple]] = {}
    for r in results_data:
        task   = str(r.get("task", "?"))
        metric = str(r.get("metric", ""))
        if metric.startswith("sklearn_") or metric in SKIP_METRICS:
            continue
        score = r.get("score", 0)
        task_results.setdefault(task, []).append((metric, score))

    def _fmt(score):
        return f"{score:.4f}" if isinstance(score, float) else str(score)

    metrics_lines = []
    if task_results:
        metrics_lines += [
            "## Evaluation Results",
            "",
            "Concept-relevant tasks (primary evaluation signal):",
            "",
            "| Task | Metric | Score |",
            "|------|--------|-------|",
        ]
        # Priority tasks first, then the rest alphabetically
        ordered = [t for t in PRIORITY_TASKS if t in task_results]
        ordered += sorted(t for t in task_results if t not in PRIORITY_TASKS)
        for task in ordered:
            for metric, score in task_results[task]:
                metrics_lines.append(f"| {task} | {metric} | {_fmt(score)} |")

    dataset_hf_link = f"[{pretrain_dataset}](https://huggingface.co/datasets/{pretrain_dataset})"

    # ── HF YAML front-matter ──────────────────────────────────────────────────
    yaml_header = f"""\
---
language:
  - en
license: apache-2.0
tags:
  - concept-encoder
  - sentence-embeddings
  - semantic-similarity
  - perceiver
  - mrcogito
  - {obj_short}
datasets:
  - {pretrain_dataset}
library_name: transformers
pipeline_tag: feature-extraction
---"""

    # ── Concept architecture note ─────────────────────────────────────────────
    bixt_note = (
        "Uses **BiXT bidirectional cross-attention** — tokens and concepts "
        "update each other at every encoder layer, producing richer contextual "
        "concept representations."
    ) if use_bixt else (
        "Tokens attend to concepts via standard cross-attention. Each encoder "
        "layer refines the 128 concept vectors."
    )

    posonly_note = (
        "Decoder uses **position-only queries** (no input-embedding shortcut), "
        "forcing all token information to flow through the concept bottleneck."
    ) if decoder_posonly else (
        "Decoder uses input+position queries."
    )

    readme = f"""{yaml_header}

# {run_name}

Part of the **[MrCogito](https://github.com/ksopyla/MrCogito)** research project —
*Concept Encoder and Decoder*: a transformer architecture that compresses long token
sequences into a small number of semantic "concept tokens" via cross-attention, then
reconstructs or classifies from that compressed bottleneck.

**Project page:** https://ai.ksopyla.com/projects/concept-encoder/

## Architecture

```
Input tokens [B, L, D_tok]
      │
      ▼  cross-attention (L × layers)
Concept representations [B, C={concept_num}, H={hidden_size}]   ← bottleneck
      │
      ▼  Perceiver IO decoder (position queries → concepts)
Output tokens [B, L, vocab]
```

| Property | Value |
|---|---|
| Hidden size | {hidden_size} |
| Encoder layers | {num_layers} |
| Concept tokens (C) | {concept_num} |
| Intermediate size | {intermediate_size} |
| Max sequence length | {max_seq} |
| Parameters | ~{total_params} |
| Encoder attention | {encoder_arch} |
| Tokenizer | [{tokenizer_name}](https://huggingface.co/{tokenizer_name}) |

{bixt_note}

{posonly_note}

## Pretraining

| Property | Value |
|---|---|
| Objective | {objective} |
| Dataset | {dataset_hf_link} |
| Epochs | {pretrain_epochs} |
| Learning rate | {pretrain_lr} |
| WandB training logs | {pretrain_wandb if pretrain_wandb else "N/A"} |

## GLUE Fine-tuning

| Property | Value |
|---|---|
| Epochs per task | {glue_epochs} |
| Learning rate | {glue_lr} |
| Batch size | {glue_batch} |
| Evaluation date | {glue_date} |
| WandB eval logs | {glue_wandb if glue_wandb else "N/A"} |

{chr(10).join(metrics_lines) if metrics_lines else "*(No evaluation data attached to this upload.)*"}

## Known Limitations

- **Concept collapse:** Without explicit regularization, the pure MLM objective can
  collapse concept representations into a low-rank space (effective rank ~5/128).
  See [experiment log](https://github.com/ksopyla/MrCogito/blob/main/docs/2_Experiments_Registry/master_experiment_log.md).
- **CoLA ceiling:** Grammatical acceptability requires sub-word patterns that do not
  survive 4:1 token→concept compression; MCC ≈ 0 is architectural, not a bug.
- **GLUE concatenated pairs:** Pair tasks (MRPC, QQP, MNLI) encode both sentences
  into one shared concept set, which compresses the cross-sentence signal.

## Usage

```python
import torch
from transformers import AutoTokenizer
import sys
sys.path.append("path/to/MrCogito")  # project root

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver

tokenizer = AutoTokenizer.from_pretrained("{tokenizer_name}")
model = ConceptEncoderForMaskedLMPerceiver.from_pretrained("{repo_id}")
model.eval()

text = "Concept encoders compress tokens into semantic concept vectors."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    # Encode: tokens → concept representations [B, C=128, H=512]
    concept_repr = model.encoder(**inputs).last_hidden_state
    # Pool concepts to sentence embedding [B, H=512]
    sentence_embedding = concept_repr.mean(dim=1)

print(sentence_embedding.shape)  # torch.Size([1, 512])
```

## Citation

```bibtex
@misc{{mrcogito-concept-encoder-{run_name},
  author       = {{Sopyla, Krzysztof}},
  title        = {{MrCogito Concept Encoder: {run_name}}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  url          = {{https://huggingface.co/{repo_id}}},
  note         = {{Concept bottleneck encoder trained with {objective}
                  on {pretrain_dataset}}}
}}
```

## License

Apache 2.0
"""
    return readme


def main():
    parser = argparse.ArgumentParser(description="Upload Concept Encoder models to Hugging Face")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR,
                        help="Path to Cache/Training")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR,
                        help="Path to Cache/Evaluation_reports")
    parser.add_argument("--hf-user", type=str, default=HF_USER,
                        help="Hugging Face username")
    parser.add_argument("--repo-prefix", type=str, default="concept-encoder",
                        help="Prefix for HF repo name (e.g. concept-encoder-{run_name})")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list models, do not prompt for upload")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (or substring) to upload non-interactively, e.g. "
                             "perceiver_mlm_H512L6C128_20260208_211633")
    args = parser.parse_args()

    console = Console()

    # Ensure paths are absolute
    training_dir = args.training_dir if args.training_dir.is_absolute() else PROJECT_ROOT / args.training_dir
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else PROJECT_ROOT / args.reports_dir

    # Scan models
    models = scan_trained_models(training_dir)
    if not models:
        console.print("[yellow]No trained models found in[/yellow]", str(training_dir))
        console.print("Ensure models exist in Cache/Training (e.g. run on Polonez/Odra).")
        return 1

    # Load evaluation data
    eval_by_run = load_evaluation_reports(reports_dir)
    results_by_run = load_results_for_metrics(reports_dir)

    # Get already uploaded
    uploaded = get_uploaded_models(args.hf_user)

    # Build display list
    table = Table(title="Trained Models (select number to upload)", box=box.ROUNDED)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Run Name", style="green")
    table.add_column("Params", justify="right")
    table.add_column("Eval Tasks", justify="right")
    table.add_column("Main Metrics", style="dim")
    table.add_column("HF Status", justify="center")

    choices = []
    for i, m in enumerate(models, 1):
        run_name = m["run_name"]
        config = m["config"]
        params = config.get("hidden_size", "?")
        total_params = sum(
            r.get("total_params", 0) for r in (eval_by_run.get(run_name) or [])
        )
        if total_params:
            total_params = f"{total_params / 1e6:.1f}M"
        else:
            total_params = "?"

        eval_rows = eval_by_run.get(run_name, [])
        tasks = sorted(set(r.get("task", "") for r in eval_rows if r.get("task")))
        tasks_str = ", ".join(tasks[:4]) + ("..." if len(tasks) > 4 else "")

        results = results_by_run.get(run_name, [])
        main_metrics = []
        for r in results:
            metric = r.get("metric", "")
            if metric in ("accuracy", "matthews_correlation", "f1", "pearsonr", "spearmanr"):
                main_metrics.append(f"{r.get('task','')}:{r.get('score',0):.2f}")
        metrics_str = " ".join(main_metrics[:6]) if main_metrics else "-"

        repo_id = f"{args.hf_user}/{args.repo_prefix}-{run_name}"
        is_uploaded = repo_id in uploaded
        status = "[green]✓ Uploaded[/green]" if is_uploaded else "[dim]—[/dim]"

        table.add_row(str(i), run_name, total_params, tasks_str or "-", metrics_str or "-", status)
        choices.append({
            "index": i,
            "model": m,
            "run_name": run_name,
            "eval_data": eval_rows,
            "results_data": results,
            "repo_id": repo_id,
            "uploaded": is_uploaded,
        })

    console.print(table)
    console.print()
    if args.list_only:
        console.print("[dim]Use --list-only to see this list. Omit it to upload.[/dim]")
        return 0

    # ── Non-interactive selection via --run-name ───────────────────────────────
    if args.run_name:
        matched = [c for c in choices if args.run_name in c["run_name"]]
        if not matched:
            console.print(f"[red]No model matching run name:[/red] {args.run_name!r}")
            console.print("Available:", ", ".join(c["run_name"] for c in choices))
            return 1
        if len(matched) > 1:
            console.print(f"[red]Ambiguous run name[/red] {args.run_name!r} matches {len(matched)} models:")
            for m in matched:
                console.print(f"  {m['run_name']}")
            console.print("Provide a more specific substring.")
            return 1
        choice = matched[0]
        console.print(f"[cyan]Non-interactive mode:[/cyan] selected [green]{choice['run_name']}[/green]")
        if choice["uploaded"]:
            console.print(f"[yellow]Already on Hub:[/yellow] https://huggingface.co/{choice['repo_id']} — re-uploading.")
    else:
        # ── Interactive selection ──────────────────────────────────────────────
        console.print(Panel(
            "[dim]Enter model number to upload (or 'q' to quit). Already uploaded models are marked with ✓.[/dim]",
            title="Selection",
            border_style="dim",
        ))

        try:
            user_input = input("\nModel number: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cancel]Cancelled.[/cancel]")
            return 0

        if user_input.lower() in ("q", "quit", ""):
            return 0

        try:
            num = int(user_input)
        except ValueError:
            console.print("[red]Invalid number.[/red]")
            return 1

        choice = next((c for c in choices if c["index"] == num), None)
        if not choice:
            console.print(f"[red]No model with number {num}.[/red]")
            return 1

        if choice["uploaded"]:
            console.print(f"[yellow]Model already uploaded:[/yellow] https://huggingface.co/{choice['repo_id']}")
            overwrite = input("Overwrite? (y/N): ").strip().lower()
            if overwrite != "y":
                return 0

    # Login check — prefer token from env/dotenv, fall back to cached credentials
    hf_token_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        if hf_token_env:
            login(token=hf_token_env, add_to_git_credential=False)
            console.print("[green]HF login OK[/green] (token from env/.env)")
        else:
            login()  # uses cached credentials from huggingface-cli login
            console.print("[green]HF login OK[/green] (cached credentials)")
    except Exception as e:
        console.print(f"[red]HF login failed:[/red] {e}")
        console.print("Set HF_TOKEN in .env or run: huggingface-cli login")
        return 1

    # Build model card and upload
    model_card = build_model_card(
        choice["run_name"],
        choice["model"],
        choice["eval_data"],
        choice["results_data"],
        choice["repo_id"],
    )

    model_path = Path(choice["model"]["model_path"])
    if not model_path.exists():
        console.print(f"[red]Model path not found:[/red] {model_path}")
        return 1

    api = HfApi()
    repo_id = choice["repo_id"]

    console.print(f"\n[cyan]Creating repo[/cyan] {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    except Exception as e:
        console.print(f"[yellow]Repo may already exist:[/yellow] {e}")

    console.print(f"[cyan]Uploading model files[/cyan] from {model_path}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        commit_message=f"Upload {choice['run_name']}",
    )

    console.print("[cyan]Uploading model card[/cyan] README.md...")
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card with evaluation metrics",
    )

    console.print(f"\n[green]Done![/green] Model: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
