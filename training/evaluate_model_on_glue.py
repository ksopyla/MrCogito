#!/usr/bin/env python
# coding: utf-8

"""
GLUE Benchmark Evaluation for Transformer Models
-----------------------------------------------
This script fine-tunes and evaluates transformer models on the GLUE benchmark.
Supports XLNet and ConceptEncoder models.

Features:
- Supports all GLUE benchmark tasks (cola, mnli-matched, mnli-mismatched, mrpc, qnli, qqp, rte, sst2, stsb, wnli)
- Treats MNLI matched and mismatched as separate tasks with consistent dataset handling
- Uses a standardized configuration approach with clear dataset and split naming
- Provides rich visualizations of results
- Includes both Hugging Face and scikit-learn metric implementations for validation
- Comprehensive error handling and logging

The GLUE (General Language Understanding Evaluation) benchmark is a collection of
resources for training, evaluating, and analyzing natural language understanding systems.
This script fine-tunes a model on each GLUE task and evaluates its performance.

Usage:
    python evaluate_model_on_glue.py --model_type xlnet --task cola --batch_size 16 --epochs 3
    python evaluate_model_on_glue.py --model_type concept --model_name_or_path ./checkpoint --task mnli-matched --batch_size 32 --epochs 5
    python evaluate_model_on_glue.py --model_type concept --model_name_or_path ./checkpoint --task all --batch_size 32 --epochs 5 --visualize
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
    default_data_collator,
    DataCollatorWithPadding
)

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ConceptEncoder
from nn.concept_encoder import ConceptEncoderForSequenceClassification, ConceptEncoderConfig

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
import evaluate
import logging
import time
import random
from datetime import datetime
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
import wandb

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
    parser = argparse.ArgumentParser(description="Fine-tune a model on GLUE")
    parser.add_argument(
        "--model_type",
        type=str,
        default="xlnet",
        choices=["xlnet", "concept"],
        help="Type of model to fine-tune (xlnet or concept)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="cola",
        choices=["cola", "mnli-matched", "mnli-mismatched", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "all"],
        help="GLUE task to train on"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="xlnet-base-cased",
        help="Model name or path to use (pretrained model name or local checkpoint path)"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name to use (if different from model_name_or_path)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Cache/Training/",
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

# GLUE task configurations - merged dictionary with the best of both versions
GLUE_TASKS = {
    "cola": {
        "num_labels": 2,
        "metrics": ["matthews_correlation"],
        "keys": {"sentence": "sentence", "label": "label"},
        "abbr": "CoLA",
        "name": "Corpus of Linguistic Acceptability",
        "description": "Predict whether a sequence is a grammatical English sentence",
        "task_type": "Single-Sentence Task",
        "domain": "Misc.",
        "size": "8.5k",
        "dataset_names": {"train": "train", "validation": "validation", "test": "test"},
        "inputs": ["sentence"],
        "target": "label",
        "metric_funcs": [matthews_corrcoef],
        "dataset": "cola"
    },
    "mnli-matched": {
        "num_labels": 3,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "premise", "sentence2": "hypothesis", "label": "label"},
        "abbr": "MNLI-m",
        "name": "Multi-Genre Natural Language Inference (Matched)",
        "description": "Predict whether the premise entails, contradicts or is neutral to the hypothesis (matched domains)",
        "task_type": "Inference Tasks",
        "domain": "Misc.",
        "size": "393k",
        "dataset_names": {"train": "train", "validation": "validation_matched", "test": "test_matched"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "dataset": "mnli"
    },
    "mnli-mismatched": {
        "num_labels": 3,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "premise", "sentence2": "hypothesis", "label": "label"},
        "abbr": "MNLI-mm",
        "name": "Multi-Genre Natural Language Inference (Mismatched)",
        "description": "Predict whether the premise entails, contradicts or is neutral to the hypothesis (mismatched domains)",
        "task_type": "Inference Tasks",
        "domain": "Misc.",
        "size": "393k",
        "dataset_names": {"train": "train", "validation": "validation_mismatched", "test": "test_mismatched"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "dataset": "mnli"
    },
    "mrpc": {
        "num_labels": 2,
        "metrics": ["accuracy", "f1"],
        "keys": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"},
        "abbr": "MRPC",
        "name": "Microsoft Research Paraphrase Corpus",
        "description": "Predict whether two sentences are semantically equivalent",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "News",
        "size": "3.7k",
        "dataset_names": {"train": "train", "validation": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score, f1_score],
        "dataset": "mrpc"
    },
    "qnli": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "question", "sentence2": "sentence", "label": "label"},
        "abbr": "QNLI",
        "name": "Stanford Question Answering Dataset",
        "description": "Predict whether the context sentence contains the answer to the question",
        "task_type": "Inference Tasks",
        "domain": "Wikipedia",
        "size": "105k",
        "dataset_names": {"train": "train", "validation": "validation", "test": "test"},
        "inputs": ["question", "sentence"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "dataset": "qnli"
    },
    "qqp": {
        "num_labels": 2,
        "metrics": ["accuracy", "f1"],
        "keys": {"sentence1": "question1", "sentence2": "question2", "label": "label"},
        "abbr": "QQP",
        "name": "Quora Question Pair",
        "description": "Predict if two questions are a paraphrase of one another",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "Social QA questions",
        "size": "364k",
        "dataset_names": {"train": "train", "validation": "validation", "test": "test"},
        "inputs": ["question1", "question2"],
        "target": "label",
        "metric_funcs": [f1_score, accuracy_score],
        "dataset": "qqp"
    },
    "rte": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"},
        "abbr": "RTE",
        "name": "Recognize Textual Entailment",
        "description": "Predict whether one sentence entails another",
        "task_type": "Inference Tasks",
        "domain": "News, Wikipedia",
        "size": "2.5k",
        "dataset_names": {"train": "train", "validation": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "dataset": "rte"
    },
    "sst2": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "keys": {"sentence": "sentence", "label": "label"},
        "abbr": "SST-2",
        "name": "Stanford Sentiment Treebank",
        "description": "Predict the sentiment of a given sentence",
        "task_type": "Single-Sentence Task",
        "domain": "Movie reviews",
        "size": "67k",
        "dataset_names": {"train": "train", "validation": "validation", "test": "test"},
        "inputs": ["sentence"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "dataset": "sst2"
    },
    "stsb": {
        "num_labels": 1,
        "metrics": ["pearson", "spearmanr"],
        "keys": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"},
        "abbr": "STS-B",
        "name": "Semantic Textual Similarity Benchmark",
        "description": "Predict the similarity score for two sentences on a scale from 1 to 5",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "Misc.",
        "size": "7k",
        "dataset_names": {"train": "train", "validation": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [pearsonr, spearmanr],
        "dataset": "stsb"
    },
    "wnli": {
        "num_labels": 2,
        "metrics": ["accuracy"],
        "keys": {"sentence1": "sentence1", "sentence2": "sentence2", "label": "label"},
        "abbr": "WNLI",
        "name": "Winograd Schema Challenge",
        "description": "Predict if the sentence with the pronoun substituted is entailed by the original sentence",
        "task_type": "Inference Tasks",
        "domain": "Fiction books",
        "size": "634",
        "dataset_names": {"train": "train", "validation": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "dataset": "wnli"
    },
}

# Remove the old glue_tasks dictionary and related code
# Create a cleaner DataFrame from the merged dictionary
glue_df = pd.DataFrame([
    {
        "Abbr": task["abbr"],
        "Name": task["name"],
        "Task Type": task["task_type"],
        "Description": task["description"],
        "Size": task["size"],
        "Metrics": ", ".join(task["metrics"])
    }
    for task_name, task in GLUE_TASKS.items()
])

# Display information about the GLUE tasks
print(glue_df.style.set_properties(**{"text-align": "left"}))

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Preprocess function for GLUE tasks
def preprocess_function(examples, tokenizer, max_length, task):
    task_config = GLUE_TASKS[task]
    task_keys = task_config["keys"]
    
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
    task_config = GLUE_TASKS[task]
    sklearn_metrics = task_config["metric_funcs"] if "metric_funcs" in task_config else []
    
    def compute_metrics_fn(eval_pred):
        predictions, labels = eval_pred
        
        # Handle regression task (STS-B)
        if task == "stsb":
            predictions_raw = predictions[:, 0]
        else:
            predictions_raw = predictions
            predictions = np.argmax(predictions, axis=1)
        
        results = {}
        
        # Use HuggingFace evaluate metrics
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
        
        # Use scikit-learn/scipy metrics for validation
        for i, metric_func in enumerate(sklearn_metrics):
            metric_name = metric_func.__name__
            if metric_name not in results:
                # Use the appropriate prediction format based on the metric
                if metric_name in ['pearsonr', 'spearmanr']:
                    # For correlation metrics, use raw predictions for regression tasks
                    result = metric_func(predictions_raw, labels)
                    # pearsonr and spearmanr return a tuple (correlation, p-value), we want just the correlation
                    results[f"sklearn_{metric_name}"] = result[0]
                elif metric_name == 'f1_score':
                    # For F1 score, use the predicted classes
                    results[f"sklearn_{metric_name}"] = metric_func(labels, predictions, average='binary')
                else:
                    # For other metrics, use the predicted classes
                    results[f"sklearn_{metric_name}"] = metric_func(labels, predictions)
        
        return results
    
    return compute_metrics_fn

def load_glue_dataset(task, tokenizer, max_length):
    """Load and preprocess a GLUE dataset for the given task."""
    try:
        # Validate task
        if task not in GLUE_TASKS:
            raise ValueError(f"Invalid task: {task}. Must be one of {list(GLUE_TASKS.keys())}")
        
        task_config = GLUE_TASKS[task]
        dataset_name = task_config["dataset"]
        dataset_splits = task_config["dataset_names"]
        
        # Load dataset with error handling
        try:
            datasets = load_dataset("glue", dataset_name, cache_dir=DATASET_CACHE_DIR)
        except Exception as e:
            logger.error(f"Failed to load GLUE dataset for task {task}: {str(e)}")
            raise
        
        # Get validation set based on the task's dataset_names dictionary
        validation_split = dataset_splits["validation"]
        eval_dataset = datasets[validation_split]
        
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
        except Exception as e:
            logger.error(f"Failed to preprocess dataset for task {task}: {str(e)}")
            raise
        
        # Log dataset statistics
        logger.info(f"Dataset statistics for {task}:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset, None
        
    except Exception as e:
        logger.error(f"Error in load_glue_dataset for task {task}: {str(e)}")
        raise

def finetune_model_on_glue(args):
    """Fine-tune a model on a GLUE task and evaluate."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine tokenizer name
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    
    # Load tokenizer based on model type
    if args.model_type == "xlnet":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=TOKENIZER_CACHE_DIR)
    else:  # concept encoder
        # For ConceptEncoder, we typically use a standard tokenizer like BERT's
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=TOKENIZER_CACHE_DIR)
    
    # Load and initialize model based on model type
    if args.model_type == "xlnet":
        # Load XLNet model
        model = XLNetForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=GLUE_TASKS[args.task]["num_labels"],
            problem_type="regression" if args.task == "stsb" else "single_label_classification",
            cache_dir=MODEL_CACHE_DIR
        )
    else:  # concept encoder
        # First, load configuration and update with task-specific settings
        try:
            # Try to load the config from the model path
            config = ConceptEncoderConfig.from_pretrained(args.model_name_or_path)
            # Update config with task-specific settings
            config.num_labels = GLUE_TASKS[args.task]["num_labels"]
            config.problem_type = "regression" if args.task == "stsb" else "single_label_classification"
        except Exception as e:
            logger.warning(f"Could not load config from {args.model_name_or_path}: {e}")
            logger.warning("Creating a new config instead.")
            # Create new config if loading failed
            config = ConceptEncoderConfig(
                num_labels=GLUE_TASKS[args.task]["num_labels"],
                problem_type="regression" if args.task == "stsb" else "single_label_classification"
            )
        
        # Load or initialize ConceptEncoder model
        try:
            # Attempt to load from checkpoint
            model = ConceptEncoderForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                config=config,
                cache_dir=MODEL_CACHE_DIR
            )
            logger.info(f"Successfully loaded ConceptEncoder model from {args.model_name_or_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {args.model_name_or_path}: {e}")
            logger.warning("Initializing a new ConceptEncoderForSequenceClassification model instead.")
            # Initialize a new model with the config
            model = ConceptEncoderForSequenceClassification(config)
    
    # Load and preprocess dataset
    train_dataset, eval_dataset, _ = load_glue_dataset(args.task, tokenizer, args.max_length)
    
    # Calculate dynamic logging steps based on dataset size
    # Target approximately 10-15 logs per epoch for readability
    train_size = len(train_dataset)
    steps_per_epoch = max(1, train_size // args.batch_size // 2)  # Account for gradient accumulation of 2
    logging_steps = max(1, steps_per_epoch // 10)  # Aim for ~10 logs per epoch
    
    logger.info(f"Dataset size: {train_size}, Steps per epoch: {steps_per_epoch}, Dynamic logging steps: {logging_steps}")
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.task}"),
        logging_dir=os.path.join(args.output_dir, f"{args.task}/logs"),
        
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=500,
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=2.0,
        optim="adamw_torch",
        gradient_accumulation_steps=1,
        
        evaluation_strategy="epoch", # change to eval_strategy
        save_strategy="epoch",
        logging_steps=logging_steps,
        seed=42,

        load_best_model_at_end=True,
        dataloader_num_workers=2,
        metric_for_best_model=GLUE_TASKS[args.task]["metrics"][0],
        push_to_hub=False,
        report_to=["tensorboard", "wandb"],
        run_name=f"GLUE-{args.model_type}-{args.task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        
        disable_tqdm=False,
    )
    
    # Initialize Trainer
    compute_metrics_fn = compute_metrics(args.task, GLUE_TASKS[args.task]["metrics"])
    
    # Create an efficient data collator with dynamic padding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=args.max_length,
        pad_to_multiple_of=8  # Optimize for tensor core operations
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Initialize the wandb project
    wandb.init(
        project="MrCogito",
        config=vars(training_args),
        name=training_args.run_name,
        tensorboard=True,
        sync_tensorboard=True,
        tags=["glue", args.task, "finetuning", args.model_type, args.model_name_or_path],
        group=f"hostname-{os.environ['COMPUTERNAME']}"
    )

    # Train model
    logger.info(f"Training {args.model_name_or_path} ({args.model_type}) on {args.task}...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    logger.info(f"Evaluating {args.model_name_or_path} ({args.model_type}) on {args.task}...")
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
    """Create visualizations for the evaluation results using rich tables."""
    console = Console()
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # 1. Create a table showing all metrics for all tasks
    all_metrics_table = Table(title="GLUE Benchmark Results for ", box=box.ROUNDED)
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
        eval_results = finetune_model_on_glue(task_args)
        
        # Store results
        for metric, score in eval_results.items():
            if metric.startswith("eval_"):
                metric_name = metric[5:]  # Remove "eval_" prefix
                results.append({
                    "Task": task,
                    "Metric": metric_name,
                    "Score": score
                })
            elif metric.startswith("sklearn_"):
                # Store sklearn metric results
                results.append({
                    "Task": task,
                    "Metric": metric,
                    "Score": score
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(args.output_dir, f"glue_results_{args.model_type}.csv")
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