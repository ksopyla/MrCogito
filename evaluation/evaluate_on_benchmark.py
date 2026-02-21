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
    python evaluation/evaluate_on_benchmark.py --benchmark sick_relatedness --model_type perceiver_mlm --model_name_or_path ./checkpoint
    python evaluation/evaluate_on_benchmark.py --benchmark paws --model_type perceiver_decoder_cls --model_name_or_path ./checkpoint
    python evaluation/evaluate_on_benchmark.py --benchmark all --model_type perceiver_mlm --model_name_or_path ./checkpoint
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
)
from datasets import load_dataset
import evaluate
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_weighted import ConceptEncoderForSequenceClassificationWeighted
from nn.concept_encoder_perceiver import (
    ConceptEncoderForSequenceClassificationPerceiver,
    ConceptEncoderForSequenceClassificationViaDecoder,
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
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate concept encoder on beyond-GLUE benchmarks")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARKS.keys()) + ["all", "sick_all"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["weighted_mlm", "perceiver_mlm", "perceiver_posonly_mlm", "perceiver_decoder_cls"])
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
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_benchmark_dataset(benchmark_name, tokenizer, max_length):
    """Load and preprocess a benchmark dataset."""
    cfg = BENCHMARKS[benchmark_name]

    if cfg["dataset_config"]:
        raw = load_dataset(cfg["dataset_id"], cfg["dataset_config"], cache_dir=DATASET_CACHE_DIR, trust_remote_code=True)
    else:
        raw = load_dataset(cfg["dataset_id"], cache_dir=DATASET_CACHE_DIR, trust_remote_code=True)

    def preprocess(examples):
        texts_a = examples[cfg["input_columns"][0]]
        texts_b = examples[cfg["input_columns"][1]]
        result = tokenizer(texts_a, texts_b, padding="max_length",
                           max_length=max_length, truncation=True)
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

    if args.model_type == "weighted_mlm":
        model_class = ConceptEncoderForSequenceClassificationWeighted
    elif args.model_type in ("perceiver_mlm", "perceiver_posonly_mlm"):
        model_class = ConceptEncoderForSequenceClassificationPerceiver
    elif args.model_type == "perceiver_decoder_cls":
        model_class = ConceptEncoderForSequenceClassificationViaDecoder
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model_class(config)

    checkpoint_path = os.path.join(args.model_name_or_path, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.model_name_or_path, "pytorch_model.bin")

    if os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(checkpoint_path)
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu")

        model_sd = model.state_dict()
        loaded, skipped = 0, 0

        if args.model_type == "perceiver_decoder_cls":
            for k, v in ckpt.items():
                if k.startswith("lm_head.") or k.startswith("loss_manager."):
                    continue
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    loaded += 1
                else:
                    skipped += 1
        else:
            for k, v in ckpt.items():
                if k.startswith("encoder.") and k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    loaded += 1
                else:
                    skipped += 1

        model.load_state_dict(model_sd)
        logger.info(f"Loaded {loaded} weights from checkpoint (skipped {skipped})")
    else:
        logger.warning(f"No checkpoint found at {args.model_name_or_path}")

    return model


def run_benchmark(args, benchmark_name):
    """Fine-tune and evaluate on a single benchmark."""
    set_seed(args.seed)
    cfg = BENCHMARKS[benchmark_name]

    logger.info(f"\n{'='*60}")
    logger.info(f"  Benchmark: {cfg['description']}")
    logger.info(f"  Why: {cfg['why']}")
    logger.info(f"{'='*60}")

    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=TOKENIZER_CACHE_DIR, token=hf_token)

    model = load_concept_model(args, benchmark_name)
    total_params = sum(p.numel() for p in model.parameters())
    params_m = round(total_params / 1_000_000)

    train_ds, eval_ds = load_benchmark_dataset(benchmark_name, tokenizer, args.max_length)

    num_batches = math.ceil(len(train_ds) / args.batch_size)
    logging_steps = max(1, num_batches // 10)

    source_run_id = os.path.basename(args.model_name_or_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"bench-{benchmark_name}-{source_run_id}-{params_m}M-{timestamp}"

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

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest",
        max_length=args.max_length, pad_to_multiple_of=8,
    )

    hostname = get_hostname()
    wandb.init(
        project="MrCogito",
        name=run_name,
        job_type=f"benchmark_{benchmark_name}",
        tags=["beyond-glue", benchmark_name, args.model_type, hostname],
        config={
            "benchmark": benchmark_name,
            "model_type": args.model_type,
            "model_path": args.model_name_or_path,
            "total_params": total_params,
            "source_run_id": source_run_id,
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
    report_name = f"bench-{benchmark_name}-{source_run_id}-{params_m}M-{timestamp}"
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
        run_benchmark(args, bm)


if __name__ == "__main__":
    main()
