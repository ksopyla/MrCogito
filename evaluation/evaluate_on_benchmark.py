#!/usr/bin/env python
# coding: utf-8

"""
Beyond-GLUE Benchmark Evaluation for Concept Encoder
-----------------------------------------------------
Evaluates concept encoder models on datasets beyond GLUE to test
whether concept representations capture genuine semantic understanding.

Supported benchmarks:
  - SICK (relatedness + entailment): Tests semantic similarity + NLI
  - PAWS (adversarial paraphrase): Tests meaning vs word-overlap understanding

These benchmarks complement GLUE by testing properties that are specifically
relevant to concept bottleneck architectures:
  - SICK relatedness: Direct concept embedding quality (continuous similarity)
  - SICK entailment: Compositional meaning preservation through bottleneck
  - PAWS: Whether concepts encode semantics, not surface form

Usage:
    python evaluation/evaluate_on_benchmark.py --benchmark sick_relatedness --model_type perceiver_denoise --model_name_or_path ./checkpoint
    python evaluation/evaluate_on_benchmark.py --benchmark stsb_zero_shot --model_type perceiver_denoise --model_name_or_path ./checkpoint
    python evaluation/evaluate_on_benchmark.py --benchmark all --model_type perceiver_denoise --model_name_or_path ./checkpoint
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import math
import time
import random
import logging
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score

import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    default_data_collator,
)
from datasets import load_dataset
import evaluate
import wandb
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.concept_encoder import ConceptEncoderConfig
from evaluation.concept_checkpoint_loader import (
    load_checkpoint_state_dict,
    load_concept_checkpoint_weights,
    select_concept_eval_model_class,
)
from evaluation.concept_eval_routing import (
    is_separate_pair_route,
    resolve_concept_eval_route,
)
from evaluation.wandb_identity import (
    build_eval_compare_fields,
    build_namespaced_eval_tags,
    lineage_to_wandb_config,
    resolve_eval_lineage,
)
from training.utils_training import get_hostname

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
except Exception:
    hf_token = None

DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
MODEL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Models"))
TOKENIZER_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Evaluation_reports"))

# ============================================================================
# Benchmark definitions
# ============================================================================

BENCHMARKS = {
    "sick_relatedness": {
        "dataset_id": "sick",
        "dataset_config": None,
        "num_labels": 1,
        "problem_type": "regression",
        "metrics": ["pearsonr", "spearmanr"],
        "primary_metric": "pearsonr",
        "input_columns": ["sentence_A", "sentence_B"],
        "label_column": "relatedness_score",
        "label_scale": 5.0,  # SICK relatedness is 1-5, normalize to 0-1
        "train_split": "train",
        "eval_split": "validation",
        "test_split": "test",
        "description": "SICK Relatedness — continuous semantic similarity (1-5 scale)",
        "why": "Direct concept embedding quality: do concepts preserve semantic similarity?",
    },
    "sick_entailment": {
        "dataset_id": "sick",
        "dataset_config": None,
        "num_labels": 3,
        "problem_type": "single_label_classification",
        "metrics": ["accuracy"],
        "primary_metric": "accuracy",
        "input_columns": ["sentence_A", "sentence_B"],
        "label_column": "label",
        "label_scale": None,
        "train_split": "train",
        "eval_split": "validation",
        "test_split": "test",
        "description": "SICK Entailment — 3-class NLI (entailment/neutral/contradiction)",
        "why": "Tests compositional meaning preservation through concept bottleneck",
    },
    "paws": {
        "dataset_id": "paws",
        "dataset_config": "labeled_final",
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "metrics": ["accuracy", "f1"],
        "primary_metric": "accuracy",
        "input_columns": ["sentence1", "sentence2"],
        "label_column": "label",
        "label_scale": None,
        "train_split": "train",
        "eval_split": "validation",
        "test_split": "test",
        "description": "PAWS — Adversarial paraphrase detection (word-scrambled pairs)",
        "why": "Bag-of-words models fail here. Tests if concepts encode meaning vs surface form.",
    },
    "stsb_zero_shot": {
        "dataset_id": "glue",
        "dataset_config": "stsb",
        "num_labels": 1,
        "problem_type": "regression",
        "metrics": ["pearsonr", "spearmanr"],
        "primary_metric": "pearsonr",
        "input_columns": ["sentence1", "sentence2"],
        "label_column": "label",
        "label_scale": None,
        "train_split": "train",
        "eval_split": "validation",
        "test_split": "validation",
        "description": "STS-B zero-shot — cosine similarity on separately encoded sentences",
        "why": "Primary semantic gate before any task-specific fine-tuning.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate concept encoder on beyond-GLUE benchmarks")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARKS.keys()) + ["all", "sick_all"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["weighted_mlm", "perceiver_denoise", "concept_ar", "diffusion_mlm", "prefix_diffusion"],
                        help="perceiver_denoise / concept_ar route via checkpoint_family metadata (sentence-pair vs weighted-pool).")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./Cache/Training/")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze the pretrained encoder and train only the task head (linear "
             "probe). Robust way to measure concept quality; avoids destroying a "
             "lightly-pretrained encoder via full fine-tuning on small datasets.",
    )
    parser.add_argument(
        "--pool_mode",
        type=str,
        default="mean",
        choices=["mean", "attention"],
        help="Concept-pooling for sentence-pair routes. 'mean' (default, backward "
             "compatible) averages the C concepts; 'attention' uses a single learned "
             "query (cross-attention over the C concepts) so distributed-across-slots "
             "information becomes visible. Use with --freeze_encoder for the probe tier.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=["none", "token_embed_mean", "teacher_hidden_mean"],
        help="Trivial-floor STS-B baseline (no concept model). 'token_embed_mean' = "
             "mean of --baseline_model input embeddings; 'teacher_hidden_mean' = mean "
             "of its last hidden states. Anchors the zero-shot STS-B number.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="HF model id for --baseline floors (shares our SmolLM2 tokenizer).",
    )
    parser.add_argument("--wandb_entity", type=str, default="ksopyla")
    parser.add_argument("--wandb_project", type=str, default="MrCogito")
    parser.add_argument(
        "--source_training_run_id",
        type=str,
        default=None,
        help="Optional explicit parent training run id/name in W&B.",
    )
    parser.add_argument(
        "--source_training_group",
        type=str,
        default=None,
        help="Optional explicit parent training W&B group (overrides API lookup).",
    )
    parser.add_argument(
        "--source_training_experiment_id",
        type=str,
        default=None,
        help="Optional explicit experiment id (e.g. E04) for eval lineage.",
    )
    parser.add_argument(
        "--source_checkpoint_step",
        type=int,
        default=None,
        help="Optional explicit checkpoint step; inferred from checkpoint-<step> when absent.",
    )
    parser.add_argument(
        "--source_checkpoint_epoch",
        type=float,
        default=None,
        help="Optional checkpoint epoch for easier train/eval retrieval in W&B.",
    )
    parser.add_argument(
        "--allow_unlinked_eval",
        action="store_true",
        help="Permit eval runs without resolved parent lineage (strict mode default is fail-fast).",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_benchmark_dataset(benchmark_name, tokenizer, max_length, pair_input_mode="concatenated"):
    """Load and preprocess a benchmark dataset."""
    cfg = BENCHMARKS[benchmark_name]

    try:
        if cfg["dataset_config"]:
            raw = load_dataset(cfg["dataset_id"], cfg["dataset_config"], cache_dir=DATASET_CACHE_DIR, trust_remote_code=True)
        else:
            raw = load_dataset(cfg["dataset_id"], cache_dir=DATASET_CACHE_DIR, trust_remote_code=True)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            logger.warning(f"Script-based loader failed for {cfg['dataset_id']}, retrying with parquet revision...")
            if cfg["dataset_config"]:
                raw = load_dataset(cfg["dataset_id"], cfg["dataset_config"],
                                   cache_dir=DATASET_CACHE_DIR, revision="refs/convert/parquet")
            else:
                raw = load_dataset(cfg["dataset_id"],
                                   cache_dir=DATASET_CACHE_DIR, revision="refs/convert/parquet")
        else:
            raise

    def preprocess(examples):
        texts_a = examples[cfg["input_columns"][0]]
        texts_b = examples[cfg["input_columns"][1]]
        if pair_input_mode == "separate":
            tokenized_a = tokenizer(
                texts_a,
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
            tokenized_b = tokenizer(
                texts_b,
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
            result = {
                "input_ids_a": tokenized_a["input_ids"],
                "attention_mask_a": tokenized_a["attention_mask"],
                "input_ids_b": tokenized_b["input_ids"],
                "attention_mask_b": tokenized_b["attention_mask"],
            }
        else:
            result = tokenizer(
                texts_a,
                texts_b,
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
        labels = examples[cfg["label_column"]]
        if cfg["label_scale"] is not None:
            labels = [float(l) / cfg["label_scale"] for l in labels]
        result["labels"] = labels
        return result

    train_ds = raw[cfg["train_split"]].map(
        preprocess, batched=True,
        remove_columns=raw[cfg["train_split"]].column_names,
        desc=f"Preprocessing {benchmark_name} train",
    )
    eval_ds = raw[cfg["eval_split"]].map(
        preprocess, batched=True,
        remove_columns=raw[cfg["eval_split"]].column_names,
        desc=f"Preprocessing {benchmark_name} eval",
    )

    logger.info(f"[{benchmark_name}] Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds


def build_compute_metrics(benchmark_name):
    """Build a compute_metrics function for the given benchmark."""
    cfg = BENCHMARKS[benchmark_name]

    def compute_metrics_fn(eval_pred):
        predictions, labels = eval_pred
        results = {}

        if cfg["problem_type"] == "regression":
            predictions = predictions[:, 0]
            results["pearsonr"] = pearsonr(predictions, labels)[0]
            results["spearmanr"] = spearmanr(predictions, labels)[0]
        else:
            predictions = np.argmax(predictions, axis=1)
            results["accuracy"] = accuracy_score(labels, predictions)
            if cfg["num_labels"] == 2:
                results["f1"] = f1_score(labels, predictions, average="binary")

        return results

    return compute_metrics_fn


def load_concept_model(args, benchmark_name):
    """Load concept encoder model for classification."""
    cfg = BENCHMARKS[benchmark_name]

    config = ConceptEncoderConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = cfg["num_labels"]
    config.problem_type = cfg["problem_type"]
    config.pool_mode = getattr(args, "pool_mode", "mean")
    route = resolve_concept_eval_route(
        config=config,
        requested_model_type=args.model_type,
        has_pair_inputs=len(cfg["input_columns"]) == 2,
    )

    model_class = select_concept_eval_model_class(route)
    model = model_class(config)

    ckpt = load_checkpoint_state_dict(args.model_name_or_path)
    if ckpt is None:
        logger.warning(f"No checkpoint found at {args.model_name_or_path}")
    else:
        loaded, skipped = load_concept_checkpoint_weights(model, ckpt, route)
        logger.info(f"Loaded {loaded} weights from checkpoint (skipped {skipped})")

    return model, route


def _architecture_id_from_config(config: ConceptEncoderConfig) -> str | None:
    if not all(hasattr(config, attr) for attr in ("hidden_size", "num_hidden_layers", "concept_num")):
        return None
    family = getattr(config, "checkpoint_family", "concept_encoder")
    decoder_layers = getattr(config, "decoder_num_layers", None)
    if decoder_layers is not None:
        return (
            f"{family}_H{config.hidden_size}"
            f"L{config.num_hidden_layers}"
            f"C{config.concept_num}"
            f"D{decoder_layers}"
        )
    return f"{family}_H{config.hidden_size}L{config.num_hidden_layers}C{config.concept_num}"


def _resolve_tokenizer_name(args, model=None) -> str:
    """Pick the tokenizer identifier to load and to put in W&B tags.

    Thin wrapper over `evaluation.wandb_identity.resolve_tokenizer_name_for_tag`
    using args + the loaded model's config. See that helper for precedence.
    """
    from evaluation.wandb_identity import resolve_tokenizer_name_for_tag

    return resolve_tokenizer_name_for_tag(
        arg_tokenizer_name=getattr(args, "tokenizer_name", None),
        arg_model_name_or_path=getattr(args, "model_name_or_path", None),
        config_tokenizer_name=getattr(getattr(model, "config", None), "tokenizer_name", None),
    )


def run_zero_shot_stsb(args):
    """Evaluate sentence-pair cosine similarity without fine-tuning."""
    benchmark_name = "stsb_zero_shot"
    cfg = BENCHMARKS[benchmark_name]
    model, route = load_concept_model(args, benchmark_name)
    if route.model_mode != "sentence_pair":
        raise ValueError(
            "Zero-shot STS-B requires a sentence-pair evaluation route with separate sentence encoding."
        )

    tokenizer_name = _resolve_tokenizer_name(args, model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=TOKENIZER_CACHE_DIR, token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    _, eval_ds = load_benchmark_dataset(
        benchmark_name,
        tokenizer,
        args.max_length,
        pair_input_mode=route.pair_input_mode,
    )
    dataloader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    predictions = []
    labels = []
    for batch in dataloader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            outputs = model(
                input_ids_a=batch["input_ids_a"],
                attention_mask_a=batch["attention_mask_a"],
                input_ids_b=batch["input_ids_b"],
                attention_mask_b=batch["attention_mask_b"],
                cosine_only=True,
                return_dict=True,
            )
        predictions.append(outputs.logits.squeeze(-1).cpu())
        labels.append(batch["labels"].cpu())

    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()
    results = {
        "pearsonr": pearsonr(predictions, labels)[0],
        "spearmanr": spearmanr(predictions, labels)[0],
    }

    total_params = sum(p.numel() for p in model.parameters())
    params_m = round(total_params / 1_000_000)
    params_label = f"{params_m}M{'-enc' if route.load_mode == 'encoder_only' else ''}"
    source_run_id = os.path.basename(args.model_name_or_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_name = f"bench-{benchmark_name}-{source_run_id}-{params_label}-{timestamp}"

    lineage = resolve_eval_lineage(
        model_path=args.model_name_or_path,
        source_training_run_id=args.source_training_run_id,
        source_training_group=args.source_training_group,
        source_training_experiment_id=args.source_training_experiment_id,
        source_checkpoint_step=args.source_checkpoint_step,
        source_checkpoint_epoch=args.source_checkpoint_epoch,
        allow_unlinked_eval=args.allow_unlinked_eval,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
    objective_family = getattr(model.config, "pretraining_objective", None)
    model_family = getattr(model.config, "checkpoint_family", args.model_type)
    architecture_id = _architecture_id_from_config(model.config)
    hostname = get_hostname()
    tags = build_namespaced_eval_tags(
        benchmark=benchmark_name,
        model_family=model_family,
        objective_family=objective_family,
        params_m=params_m,
        tokenizer_name=tokenizer_name,
        lineage=lineage,
        extra_tags=["beyond-glue", benchmark_name, args.model_type, hostname, "zero-shot"],
    )
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=report_name,
        job_type="benchmark_stsb_zero_shot",
        group=lineage.source_training_group,
        tags=tags,
        config={
            "benchmark": benchmark_name,
            "model_type": args.model_type,
            "model_path": args.model_name_or_path,
            "total_params": total_params,
            **lineage_to_wandb_config(lineage),
            **build_eval_compare_fields(
                model_family=model_family,
                params_m=params_m,
                objective_family=objective_family,
                tokenizer_name=tokenizer_name,
                architecture_id=architecture_id,
            ),
        },
    )
    wandb.log(results)
    wandb.finish()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    pd.DataFrame([{
        "benchmark": benchmark_name,
        "model_type": args.model_type,
        "model_path": args.model_name_or_path,
        "params_m": params_m,
        **results,
    }]).to_csv(os.path.join(REPORTS_DIR, f"{report_name}-results.csv"), index=False)

    logger.info("Results for stsb_zero_shot:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    return results


def run_zero_shot_stsb_baseline(args):
    """Trivial-floor STS-B baseline: mean-pool an external model's token embeddings or
    last hidden states, then cosine. No concept model involved — anchors the model number.
    """
    from transformers import AutoModel

    benchmark_name = "stsb_zero_shot"
    variant = args.baseline
    tokenizer_name = args.tokenizer_name or args.baseline_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=TOKENIZER_CACHE_DIR, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = AutoModel.from_pretrained(
        args.baseline_model, cache_dir=MODEL_CACHE_DIR, token=hf_token
    ).to(device).eval()
    embedding = teacher.get_input_embeddings()

    _, eval_ds = load_benchmark_dataset(
        benchmark_name, tokenizer, args.max_length, pair_input_mode="separate"
    )
    dataloader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator
    )

    def encode(input_ids, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        if variant == "token_embed_mean":
            h = embedding(input_ids)
        else:  # teacher_hidden_mean
            h = teacher(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

    predictions, labels = [], []
    for batch in dataloader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            z_a = encode(batch["input_ids_a"], batch["attention_mask_a"])
            z_b = encode(batch["input_ids_b"], batch["attention_mask_b"])
            cos = torch.nn.functional.cosine_similarity(z_a, z_b, dim=-1)
        predictions.append(cos.cpu())
        labels.append(batch["labels"].cpu())

    predictions = torch.cat(predictions).float().numpy()
    labels = torch.cat(labels).float().numpy()
    results = {
        "pearsonr": pearsonr(predictions, labels)[0],
        "spearmanr": spearmanr(predictions, labels)[0],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_name = f"bench-{benchmark_name}-BASELINE_{variant}-{os.path.basename(args.baseline_model)}-{timestamp}"
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pd.DataFrame([{
        "benchmark": benchmark_name,
        "variant": variant,
        "baseline_model": args.baseline_model,
        **results,
    }]).to_csv(os.path.join(REPORTS_DIR, f"{report_name}-results.csv"), index=False)

    logger.info(f"STS-B baseline [{variant}] on {args.baseline_model}:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")
    logger.info("Reference ceilings (cited): SimCSE-unsup ~0.76, SBERT ~0.84 Spearman.")
    return results


def run_benchmark(args, benchmark_name):
    """Fine-tune and evaluate on a single benchmark."""
    set_seed(args.seed)
    cfg = BENCHMARKS[benchmark_name]

    logger.info(f"\n{'='*60}")
    logger.info(f"  Benchmark: {cfg['description']}")
    logger.info(f"  Why: {cfg['why']}")
    logger.info(f"{'='*60}")

    model, route = load_concept_model(args, benchmark_name)
    tokenizer_name = _resolve_tokenizer_name(args, model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=TOKENIZER_CACHE_DIR, token=hf_token)

    if args.freeze_encoder and hasattr(model, "encoder"):
        frozen = 0
        for p in model.encoder.parameters():
            p.requires_grad = False
            frozen += p.numel()
        logger.info(f"Froze encoder ({frozen/1e6:.1f}M params); training task head only (linear probe).")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = round(total_params / 1_000_000)
    # Label clarifies the eval model is the encoder-only sub-model (no decoder/lm_head),
    # so reports don't imply the source checkpoint is this small.
    params_label = f"{params_m}M{'-enc' if route.load_mode == 'encoder_only' else ''}"

    train_ds, eval_ds = load_benchmark_dataset(
        benchmark_name,
        tokenizer,
        args.max_length,
        pair_input_mode=route.pair_input_mode,
    )

    num_batches = math.ceil(len(train_ds) / args.batch_size)
    logging_steps = max(1, num_batches // 10)

    source_run_id = os.path.basename(args.model_name_or_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"bench-{benchmark_name}-{source_run_id}-{params_label}-{timestamp}"
    lineage = resolve_eval_lineage(
        model_path=args.model_name_or_path,
        source_training_run_id=args.source_training_run_id,
        source_training_group=args.source_training_group,
        source_training_experiment_id=args.source_training_experiment_id,
        source_checkpoint_step=args.source_checkpoint_step,
        source_checkpoint_epoch=args.source_checkpoint_epoch,
        allow_unlinked_eval=args.allow_unlinked_eval,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, benchmark_name),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        lr_scheduler_type="linear",
        bf16=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=logging_steps,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model=cfg["primary_metric"],
        report_to=["wandb"],
        run_name=run_name,
    )

    if is_separate_pair_route(route):
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            max_length=args.max_length,
            pad_to_multiple_of=8,
        )

    objective_family = getattr(model.config, "pretraining_objective", None)
    model_family = getattr(model.config, "checkpoint_family", args.model_type)
    architecture_id = _architecture_id_from_config(model.config)
    hostname = get_hostname()
    tags = build_namespaced_eval_tags(
        benchmark=benchmark_name,
        model_family=model_family,
        objective_family=objective_family,
        params_m=params_m,
        tokenizer_name=tokenizer_name,
        lineage=lineage,
        extra_tags=["beyond-glue", benchmark_name, args.model_type, hostname],
    )
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        job_type=f"benchmark_{benchmark_name}",
        group=lineage.source_training_group,
        tags=tags,
        config={
            "benchmark": benchmark_name,
            "model_type": args.model_type,
            "model_path": args.model_name_or_path,
            "total_params": total_params,
            **lineage_to_wandb_config(lineage),
            **build_eval_compare_fields(
                model_family=model_family,
                params_m=params_m,
                objective_family=objective_family,
                tokenizer_name=tokenizer_name,
                architecture_id=architecture_id,
            ),
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(benchmark_name),
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    results = trainer.evaluate()
    logger.info(f"Results for {benchmark_name}:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_name = f"bench-{benchmark_name}-{source_run_id}-{params_label}-{timestamp}"
    pd.DataFrame([{
        "benchmark": benchmark_name,
        "model_type": args.model_type,
        "model_path": args.model_name_or_path,
        "params_m": params_m,
        "train_time_s": round(train_time, 1),
        **{k.replace("eval_", ""): v for k, v in results.items()},
    }]).to_csv(os.path.join(REPORTS_DIR, f"{report_name}-results.csv"), index=False)

    if args.save_model:
        trainer.save_model(os.path.join(args.output_dir, benchmark_name, "final_model"))

    wandb.finish()
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.benchmark == "all":
        benchmarks = list(BENCHMARKS.keys())
    elif args.benchmark == "sick_all":
        benchmarks = ["sick_relatedness", "sick_entailment"]
    else:
        benchmarks = [args.benchmark]

    for bm in benchmarks:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Running benchmark: {bm}")
        logger.info(f"{'#'*60}")
        if bm == "stsb_zero_shot":
            if getattr(args, "baseline", "none") != "none":
                run_zero_shot_stsb_baseline(args)
            else:
                run_zero_shot_stsb(args)
        else:
            run_benchmark(args, bm)


if __name__ == "__main__":
    main()
