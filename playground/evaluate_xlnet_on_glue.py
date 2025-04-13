#!/usr/bin/env python
# coding: utf-8

"""
GLUE Benchmark Evaluation for XLNet
-----------------------------------
This script fine-tunes and evaluates XLNet on the GLUE benchmark.
Adapted from the ModernBERT example.

Usage:
    python evaluate_xlnet_on_glue.py --task cola --batch_size 16 --epochs 3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import (
    XLNetModel,
    XLNetForSequenceClassification, 
    XLNetTokenizer,
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator
)
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
import evaluate
import logging
import time
import random
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import box


DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
MODEL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Models"))
TOKENIZER_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers"))

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configure arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune XLNet on GLUE")
    parser.add_argument(
        "--task",
        type=str,
        default="cola",
        choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
        help="GLUE task to train on"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="xlnet-base-cased",
        help="Model name or path to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA even when available"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the fine-tuned model"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize results with rich tables"
    )
    return parser.parse_args()

# GLUE task configurations
GLUE_TASKS = {
    "cola": {
        "num_labels": 2,
        "metrics": ["matthews_correlation"],
        "keys": {"sentence": "sentence", "label": "label"}
    },
    "mnli": {
        "num_labels": 3,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "premise", "sentence2": "hypothesis", "label": "label"}
    },
    "mrpc": {
        "num_labels": 2,
        "metrics": ["accuracy", "f1"],
        "keys": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"}
    },
    "qnli": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "question", "sentence2": "sentence", "label": "label"}
    },
    "qqp": {
        "num_labels": 2,
        "metrics": ["accuracy", "f1"],
        "keys": {"sentence1": "question1", "sentence2": "question2", "label": "label"}
    },
    "rte": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"}
    },
    "sst2": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "keys": {"sentence": "sentence", "label": "label"}
    },
    "stsb": {
        "num_labels": 1,
        "metrics": ["pearson", "spearmanr"],
        "keys": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"}
    },
    "wnli": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"}
    },
}

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Preprocess function for GLUE tasks
def preprocess_function(examples, tokenizer, max_length, task):
    task_keys = GLUE_TASKS[task]["keys"]
    
    # Handle single and pair sentence tasks
    if "sentence2" in task_keys:
        sentences1 = examples[task_keys["sentence1"]]
        sentences2 = examples[task_keys["sentence2"]]
        
        # Tokenize sentence pairs
        result = tokenizer(
            sentences1, 
            sentences2, 
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
    else:
        sentences = examples[task_keys["sentence"]]
        
        # Tokenize single sentences
        result = tokenizer(
            sentences,
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
    
    # Add labels
    if task_keys["label"] in examples:
        result["labels"] = examples[task_keys["label"]]
    
    return result

# Compute metrics for evaluation
def compute_metrics(task, metric_names):
    metrics = {name: evaluate.load(name) for name in metric_names}
    
    def compute_metrics_fn(eval_pred):
        predictions, labels = eval_pred
        
        # Handle regression task (STS-B)
        if task == "stsb":
            predictions = predictions[:, 0]
        else:
            predictions = np.argmax(predictions, axis=1)
        
        results = {}
        for name, metric in metrics.items():
            if name == "matthews_correlation":
                results[name] = metric.compute(predictions=predictions, references=labels)["matthews_correlation"]
            elif name == "f1":
                results[name] = metric.compute(predictions=predictions, references=labels)["f1"]
            elif name == "pearson":
                results[name] = metric.compute(predictions=predictions, references=labels)["pearson"]
            elif name == "spearmanr":
                results[name] = metric.compute(predictions=predictions, references=labels)["spearmanr"]
            else:
                results[name] = metric.compute(predictions=predictions, references=labels)["accuracy"]
        
        return results
    
    return compute_metrics_fn

def load_glue_dataset(task, tokenizer, max_length):
    """Load and preprocess a GLUE dataset for the given task."""
    try:
        # Validate task
        if task not in GLUE_TASKS:
            raise ValueError(f"Invalid task: {task}. Must be one of {list(GLUE_TASKS.keys())}")
        
        # Load dataset with error handling
        try:
            datasets = load_dataset("glue", task, cache_dir=DATASET_CACHE_DIR)
        except Exception as e:
            logger.error(f"Failed to load GLUE dataset for task {task}: {str(e)}")
            raise
        
        # Handle MNLI special case
        if task == "mnli":
            eval_dataset = datasets["validation_matched"]
            # Also load mismatched validation set for MNLI
            eval_mismatched = datasets["validation_mismatched"]
        else:
            eval_dataset = datasets["validation"]
            eval_mismatched = None
        
        # Preprocess datasets with error handling
        try:
            train_dataset = datasets["train"].map(
                lambda examples: preprocess_function(examples, tokenizer, max_length, task),
                batched=True,
                remove_columns=datasets["train"].column_names,
                desc="Preprocessing training data"
            )
            
            eval_dataset = eval_dataset.map(
                lambda examples: preprocess_function(examples, tokenizer, max_length, task),
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Preprocessing validation data"
            )
            
            if eval_mismatched is not None:
                eval_mismatched = eval_mismatched.map(
                    lambda examples: preprocess_function(examples, tokenizer, max_length, task),
                    batched=True,
                    remove_columns=eval_mismatched.column_names,
                    desc="Preprocessing validation mismatched data"
                )
        except Exception as e:
            logger.error(f"Failed to preprocess dataset for task {task}: {str(e)}")
            raise
        
        # Log dataset statistics
        logger.info(f"Dataset statistics for {task}:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(eval_dataset)}")
        if eval_mismatched is not None:
            logger.info(f"  Validation mismatched samples: {len(eval_mismatched)}")
        
        return train_dataset, eval_dataset, eval_mismatched
        
    except Exception as e:
        logger.error(f"Error in load_glue_dataset for task {task}: {str(e)}")
        raise

def finetune_xlnet_on_glue(args):
    """Fine-tune XLNet on a GLUE task and evaluate."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased", cache_dir=TOKENIZER_CACHE_DIR)
    model = XLNetForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=GLUE_TASKS[args.task]["num_labels"],
        problem_type="regression" if args.task == "stsb" else "single_label_classification",
        cache_dir=MODEL_CACHE_DIR
    )
    
    # Load and preprocess dataset
    train_dataset, eval_dataset, eval_mismatched = load_glue_dataset(args.task, tokenizer, args.max_length)
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.task}"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=GLUE_TASKS[args.task]["metrics"][0],
        push_to_hub=False,
        report_to="none",  # Disable wandb, tensorboard, etc.
        logging_dir=os.path.join(args.output_dir, f"{args.task}/logs"),
        logging_steps=10,
        disable_tqdm=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics(args.task, GLUE_TASKS[args.task]["metrics"]),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    logger.info(f"Training XLNet on {args.task}...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    logger.info(f"Evaluating XLNet on {args.task}...")
    eval_results = trainer.evaluate()
    
    # Evaluate on mismatched set for MNLI if available
    if eval_mismatched is not None:
        logger.info("Evaluating on MNLI mismatched validation set...")
        mismatched_results = trainer.evaluate(eval_dataset=eval_mismatched)
        eval_results.update({f"eval_mismatched_{k}": v for k, v in mismatched_results.items()})
    
    # Save model if requested
    if args.save_model:
        trainer.save_model(os.path.join(args.output_dir, f"{args.task}/final_model"))
    
    # Print results
    logger.info(f"Evaluation results for {args.task}:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value}")
    
    return eval_results

def visualize_results(results_df, args):
    """Create visualizations for the evaluation results using rich tables."""
    console = Console()
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # 1. Create a table showing all metrics for all tasks
    all_metrics_table = Table(title="GLUE Benchmark Results for XLNet", box=box.ROUNDED)
    all_metrics_table.add_column("Task", style="cyan")
    
    # Get unique metrics
    metrics = sorted(results_df["Metric"].unique())
    for metric in metrics:
        all_metrics_table.add_column(metric, style="green")
    
    # Add rows for each task
    for task in sorted(results_df["Task"].unique()):
        task_results = results_df[results_df["Task"] == task]
        row = [task]
        
        for metric in metrics:
            metric_value = task_results[task_results["Metric"] == metric]["Score"].values
            if len(metric_value) > 0:
                row.append(f"{metric_value[0]:.4f}")
            else:
                row.append("N/A")
        
        all_metrics_table.add_row(*row)
    
    # Display and save the table content to a file
    console.print(all_metrics_table)
    
    # 2. Create individual tables for each task
    for task in sorted(results_df["Task"].unique()):
        task_df = results_df[results_df["Task"] == task]
        
        task_table = Table(title=f"Results for {task.upper()}", box=box.ROUNDED)
        task_table.add_column("Metric", style="cyan")
        task_table.add_column("Score", style="green")
        
        for _, row in task_df.iterrows():
            task_table.add_row(row["Metric"], f"{row['Score']:.4f}")
        
        # Display the task table
        console.print(task_table)
        console.print("\n")
    
    # 3. Create a summary table with average scores
    summary_table = Table(title="Summary of Results", box=box.ROUNDED)
    summary_table.add_column("Task", style="cyan")
    summary_table.add_column("Avg Score", style="green")
    summary_table.add_column("# Metrics", style="blue")
    
    for task in sorted(results_df["Task"].unique()):
        task_df = results_df[results_df["Task"] == task]
        avg_score = task_df["Score"].mean()
        num_metrics = len(task_df)
        
        summary_table.add_row(task, f"{avg_score:.4f}", str(num_metrics))
    
    # Display the summary table
    console.print(summary_table)

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Evaluate all tasks if task is "all"
    if args.task == "all":
        tasks = list(GLUE_TASKS.keys())
    else:
        tasks = [args.task]
    
    # Store results
    results = []
    
    # Evaluate each task
    for task in tasks:
        logger.info(f"Processing task: {task}")
        
        # Update args with task
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task
        
        # Fine-tune and evaluate
        eval_results = finetune_xlnet_on_glue(task_args)
        
        # Store results
        for metric, score in eval_results.items():
            if metric.startswith("eval_"):
                metric_name = metric[5:]  # Remove "eval_" prefix
                results.append({
                    "Task": task,
                    "Metric": metric_name,
                    "Score": score
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(args.output_dir, "glue_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(results_df, args)
    
    # Print summary
    logger.info("Summary of results:")
    for task in set(results_df["Task"]):
        task_results = results_df[results_df["Task"] == task]
        logger.info(f"  {task}:")
        for _, row in task_results.iterrows():
            logger.info(f"    {row['Metric']}: {row['Score']:.4f}")

if __name__ == "__main__":
    main() 