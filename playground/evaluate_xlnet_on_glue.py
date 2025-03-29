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
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator
)
from datasets import load_dataset, load_metric
from datasets.utils.logging import disable_progress_bar
import evaluate
import logging
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        help="Visualize results"
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
    # Load dataset
    if task == "mnli":
        datasets = load_dataset("glue", task)
        eval_dataset = datasets["validation_matched"]
    else:
        datasets = load_dataset("glue", task)
        eval_dataset = datasets["validation"]
    
    # Preprocess datasets
    train_dataset = datasets["train"].map(
        lambda examples: preprocess_function(examples, tokenizer, max_length, task),
        batched=True,
        remove_columns=datasets["train"].column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length, task),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return train_dataset, eval_dataset

def finetune_xlnet_on_glue(args):
    """Fine-tune XLNet on a GLUE task and evaluate."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer and model
    tokenizer = XLNetTokenizer.from_pretrained(args.model_name_or_path)
    model = XLNetForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=GLUE_TASKS[args.task]["num_labels"],
        problem_type="regression" if args.task == "stsb" else "single_label_classification"
    )
    
    # Load and preprocess dataset
    train_dataset, eval_dataset = load_glue_dataset(args.task, tokenizer, args.max_length)
    
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
    
    # Save model if requested
    if args.save_model:
        trainer.save_model(os.path.join(args.output_dir, f"{args.task}/final_model"))
    
    # Print results
    logger.info(f"Evaluation results for {args.task}:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value}")
    
    return eval_results

def visualize_results(results_df, args):
    """Create visualizations for the evaluation results."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    
    # Plot results as bar chart
    plt.figure(figsize=(12, 8))
    
    # Get metrics for each task
    tasks = results_df["Task"].unique()
    
    for i, task in enumerate(tasks):
        task_df = results_df[results_df["Task"] == task]
        metrics = task_df["Metric"].tolist()
        scores = task_df["Score"].tolist()
        
        x = np.arange(len(metrics))
        plt.bar(x + i*0.2, scores, width=0.2, label=task)
        
        # Add value labels on bars
        for j, score in enumerate(scores):
            plt.text(x[j] + i*0.2, score + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.title("GLUE Benchmark Results for XLNet")
    plt.xticks(np.arange(len(metrics)) + 0.2*(len(tasks)-1)/2, metrics)
    plt.ylim(0, 1.0)
    plt.legend(title="Task")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(args.output_dir, "visualizations/glue_results.png"))
    plt.close()
    
    # Create heatmap of results
    plt.figure(figsize=(12, 8))
    
    # Pivot data for heatmap
    pivot_df = results_df.pivot(index="Task", columns="Metric", values="Score")
    
    # Plot heatmap
    im = plt.imshow(pivot_df.values, cmap="YlGn")
    
    # Add colorbar
    plt.colorbar(im, label="Score")
    
    # Add labels
    plt.xticks(np.arange(len(pivot_df.columns)), pivot_df.columns, rotation=45)
    plt.yticks(np.arange(len(pivot_df.index)), pivot_df.index)
    
    # Add values to cells
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            if not np.isnan(pivot_df.values[i, j]):
                plt.text(j, i, f'{pivot_df.values[i, j]:.4f}', 
                         ha="center", va="center", color="black")
    
    plt.title("GLUE Benchmark Results for XLNet")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(args.output_dir, "visualizations/glue_heatmap.png"))
    plt.close()

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