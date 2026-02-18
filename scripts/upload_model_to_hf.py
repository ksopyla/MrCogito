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

# Try load .env for HF token
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

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


def build_model_card(
    run_name: str,
    model_info: dict,
    eval_data: list[dict],
    results_data: list[dict],
    repo_id: str,
) -> str:
    """Build README.md model card with metrics, training info, and links."""
    config = model_info.get("config", {})
    trainer_state = model_info.get("trainer_state", {})

    # Model architecture
    model_type = config.get("model_type", "concept_encoder")
    hidden_size = config.get("hidden_size", "?")
    num_layers = config.get("num_hidden_layers", "?")
    concept_num = config.get("concept_num", "?")
    intermediate_size = config.get("intermediate_size", "?")

    # Pretraining dataset (from training_args.json or default)
    pretrain_dataset = "JeanKaddour/minipile"
    pretrain_epochs = "?"
    run_dir = Path(model_info.get("run_dir", ""))
    if run_dir.exists():
        ta_path = run_dir / "training_args.json"
        if ta_path.exists():
            try:
                with open(ta_path) as f:
                    ta = json.load(f)
                    pretrain_dataset = ta.get("dataset_name", pretrain_dataset)
                    pretrain_epochs = ta.get("num_train_epochs", pretrain_epochs)
            except Exception:
                pass

    # GLUE evaluation info from metadata
    glue_epochs = "?"
    tokenizer_name = config.get("tokenizer_name", "answerdotai/ModernBERT-base")
    learning_rate = "?"
    batch_size = "?"
    wandb_url = ""
    date = ""

    if eval_data:
        first = eval_data[0]
        glue_epochs = first.get("epochs", glue_epochs)
        if pd.isna(glue_epochs):
            glue_epochs = "?"
        tokenizer_name = first.get("tokenizer_name", tokenizer_name) or tokenizer_name
        learning_rate = first.get("learning_rate", learning_rate)
        batch_size = first.get("batch_size", batch_size)
        wandb_url = first.get("wandb_url", wandb_url)
        date = first.get("date", date)
        if pd.isna(date):
            date = ""

    # Build metrics table
    metrics_lines = []
    if results_data:
        # Group by task
        tasks = {}
        for r in results_data:
            t = r.get("task", "?")
            if t not in tasks:
                tasks[t] = []
            metric = r.get("metric", "")
            if not metric.startswith("sklearn_") and metric not in ("loss", "runtime", "samples_per_second", "steps_per_second"):
                tasks[t].append((metric, r.get("score", 0)))

        if tasks:
            metrics_lines.append("## Evaluation Results (GLUE)")
            metrics_lines.append("")
            metrics_lines.append("| Task | Metric | Score |")
            metrics_lines.append("|------|--------|-------|")
            for task in sorted(tasks.keys()):
                for metric, score in tasks[task]:
                    if isinstance(score, float):
                        score = f"{score:.4f}"
                    metrics_lines.append(f"| {task} | {metric} | {score} |")

    dataset_hf_link = f"[{pretrain_dataset}](https://huggingface.co/datasets/{pretrain_dataset})"
    total_params = config.get("vocab_size", "?")
    if eval_data and not pd.isna(eval_data[0].get("total_params")):
        total_params = f"{int(eval_data[0]['total_params']) / 1e6:.1f}M"

    readme = f"""# {run_name}

Concept Encoder model trained with Masked Language Modeling (MLM) for the [MrCogito](https://github.com/ksopyla/MrCogito) project.

## Model Details

- **Model type:** {model_type}
- **Architecture:** Hidden size {hidden_size}, {num_layers} layers, {concept_num} concept tokens
- **Intermediate size:** {intermediate_size}
- **Parameters:** ~{total_params}

## Pretraining (MLM)

- **Dataset:** {dataset_hf_link}
- **Tokenizer:** {tokenizer_name}
- **Epochs:** {pretrain_epochs}

## GLUE Fine-tuning (Evaluation)

- **Epochs:** {glue_epochs}
- **Learning rate:** {learning_rate}
- **Batch size:** {batch_size}
- **Evaluation date:** {date}

## Weights & Biases

Training logs: {wandb_url if wandb_url else "N/A"}

## Evaluation

{chr(10).join(metrics_lines) if metrics_lines else "No GLUE evaluation data available."}

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

## Citation

```bibtex
@misc{{mrcogito-concept-encoder,
  author = {{{{Krzysztof Sopyla}}}},
  title = {{MrCogito Concept Encoder}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
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

    # Login check
    try:
        login()
    except Exception as e:
        console.print(f"[red]HF login failed:[/red] {e}")
        console.print("Run: huggingface-cli login")
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
