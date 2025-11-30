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
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
    DataCollatorWithPadding
)

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ConceptEncoder
from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_weighted import ConceptEncoderForSequenceClassificationWeighted
from nn.concept_encoder_perceiver import ConceptEncoderForSequenceClassificationPerceiver
from training.utils_training import get_hostname

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


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Add Hugging Face authentication token from .env file
try:
    from dotenv import load_dotenv
    import os
    
    # Load environment variables from .env file in the project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
    
    # Get HF token from environment variable
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    if hf_token:
        logger.info("Successfully loaded Hugging Face token from .env file")
    else:
        logger.warning("No Hugging Face token found in .env file. Add HUGGINGFACE_TOKEN or HF_TOKEN to your .env file")
        hf_token = None
        
except ImportError:
    logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
    hf_token = None
except Exception as e:
    logger.warning(f"Failed to load token from .env file: {e}")
    hf_token = None

DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
MODEL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Models"))
TOKENIZER_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers"))



# Configure arguments
def parse_args():
    """
    Parse command line arguments for GLUE evaluation script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with all configuration options
        
    Available arguments:
        --model_type: Type of model (bert-type, xlnet-type, concept-type)
        --task: GLUE task to evaluate on (cola, mrpc, etc. or 'all')
        --model_name_or_path: HuggingFace model name or local path
        --batch_size: Training and evaluation batch size
        --epochs: Number of training epochs
        --learning_rate: Learning rate for optimizer
        --visualize: Enable rich table visualizations
        And many more hyperparameter options...
    """
    parser = argparse.ArgumentParser(description="Fine-tune a model on GLUE")
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        choices=["bert-type", "xlnet-type", "concept-type", "sim_matrix_mlm", "concept_mlm", "weighted_mlm", "perceiver_mlm"],
        help="Type of model to fine-tune (bert, roberta, xlnet, or concept)"
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
        default="bert-base-cased", # bert-base-cased, xlnet-base-cased, roberta-base, distilbert-base-cased, distilbert-base-uncased-finetuned-sst-2-english
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
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer"
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer"
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer"
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
    """
    Set random seeds for reproducible results across different runs.
    
    Args:
        seed (int): Random seed value to use for all random number generators
        
    Note:
        Sets seeds for: random, numpy, torch, and CUDA (if available)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Preprocess function for GLUE tasks
def preprocess_function(examples, tokenizer, max_length, task):
    """
    Preprocess examples for GLUE tasks by tokenizing text and adding labels.
    
    Args:
        examples (dict): Batch of examples from the dataset
        tokenizer: HuggingFace tokenizer for text processing
        max_length (int): Maximum sequence length for tokenization
        task (str): GLUE task name to determine input format
        
    Returns:
        dict: Tokenized examples with input_ids, attention_mask, and labels
        
    Note:
        Handles both single-sentence tasks (CoLA, SST-2) and sentence-pair tasks (MRPC, etc.)
    """
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
    """
    Create a metrics computation function for the specified GLUE task.
    
    Args:
        task (str): GLUE task name
        metric_names (list): List of metric names to compute
        
    Returns:
        function: Metrics computation function that takes eval_pred and returns metric dict
        
    Note:
        Computes both HuggingFace evaluate metrics and scikit-learn metrics for validation.
        Handles regression tasks (STS-B) and classification tasks differently.
    """
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
    """
    Load and preprocess a GLUE dataset for the specified task.
    
    Args:
        task (str): GLUE task name (e.g., 'mrpc', 'cola', etc.)
        tokenizer: HuggingFace tokenizer for text preprocessing
        max_length (int): Maximum sequence length for tokenization
        
    Returns:
        tuple: (train_dataset, eval_dataset, test_dataset)
            - train_dataset: Preprocessed training dataset
            - eval_dataset: Preprocessed validation dataset  
            - test_dataset: None (not used in current implementation)
            
    Raises:
        ValueError: If task is not a valid GLUE task
        Exception: If dataset loading or preprocessing fails
        
    Note:
        Handles different validation split names for different tasks (e.g., MNLI has matched/mismatched)
    """
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
        # logger.info(f"Dataset statistics for {task}:")
        # logger.info(f"  Training samples: {len(train_dataset)}")
        # logger.info(f"  Validation samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset, None
        
    except Exception as e:
        logger.error(f"Error in load_glue_dataset for task {task}: {str(e)}")
        raise

def create_experiment_name(model_name, task, total_params, timestamp=None):
    """
    Create a consistent experiment name for files and wandb runs.
    
    Args:
        model_name (str): Full model name (e.g., 'distilbert-base-cased')
        task (str): GLUE task name (e.g., 'mrpc')
        total_params (int): Total number of model parameters
        timestamp (str, optional): Timestamp string, generates new if None
        
    Returns:
        tuple: (experiment_name, timestamp)
            - experiment_name: Formatted name like 'glue-mrpc-distilbert-base-cased-66M'
            - timestamp: Timestamp used (for consistency across files)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Clean model name for file system compatibility
    clean_model_name = model_name.lower()
    
    # Remove organization prefix (everything before the last slash)
    if '/' in clean_model_name:
        clean_model_name = clean_model_name.split('/')[-1]
    
    # Replace problematic characters with hyphens
    clean_model_name = clean_model_name.replace('_', '-')
    clean_model_name = clean_model_name.replace(' ', '-')
    clean_model_name = clean_model_name.replace('.', '-')
    
    # Remove multiple consecutive hyphens
    while '--' in clean_model_name:
        clean_model_name = clean_model_name.replace('--', '-')
    
    # Remove leading/trailing hyphens
    clean_model_name = clean_model_name.strip('-')
    
    # Format parameters in millions
    params_m = round(total_params / 1_000_000)
    params_str = f"{params_m}M"
    
    # Create experiment name
    experiment_name = f"glue-{task}-{clean_model_name}-{params_str}"
    
    return experiment_name, timestamp

def create_file_names(experiment_name, timestamp):
    """
    Create consistent file names for all experiment outputs.
    
    Args:
        experiment_name (str): Base experiment name
        timestamp (str): Timestamp string
        
    Returns:
        dict: Dictionary with file names for different report types
    """
    base_name = f"{experiment_name}-{timestamp}"
    
    return {
        'results': f"{base_name}-results.csv",
        'metadata': f"{base_name}-metadata.csv", 
        'summary': f"{base_name}-summary.csv"
    }

def get_model_specific_config(model_name_or_path):
    """
    Get model-specific configuration parameters for fine-tuning.
    
    This function centralizes model-specific settings to handle known issues
    and optimize training for different model architectures.
    
    Args:
        model_name_or_path (str): The model name or path
        
    Returns:
        dict: Dictionary containing model-specific configuration parameters
    """
    config = {
        'use_bf16': True,
        'use_fp16': False,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'warmup_steps': 500,
        'gradient_accumulation_steps': 1,
        'special_notes': []
    }
    
    model_name_lower = model_name_or_path.lower()
    
    # DeBERTa-specific configurations
    if "deberta" in model_name_lower:
        config.update({
            'use_bf16': False,
            'use_fp16': False,
            'max_grad_norm': 2.0,
            'special_notes': ['Mixed precision disabled due to BFloat16 overflow issues']
        })
        logger.info("DeBERTa model detected - applying DeBERTa-specific configurations")
    
    # ModernBERT-specific configurations
    elif "modernbert" in model_name_lower:
        config.update({
            'adam_beta2': 0.98,  # Critical for ModernBERT stability
            'adam_epsilon': 1e-6,
            'max_grad_norm': 2.0,
            'special_notes': ['ModernBERT requires adam_beta2=0.98 for stable training']
        })
        logger.info("ModernBERT model detected - applying ModernBERT-specific configurations")
    
    # ELECTRA-specific configurations
    elif "electra" in model_name_lower:
        config.update({
            'warmup_steps': 1000,
            'gradient_accumulation_steps': 2,
            'special_notes': ['ELECTRA benefits from longer warmup and gradient accumulation']
        })
        logger.info("ELECTRA model detected - applying ELECTRA-specific configurations")
    
    # XLNet-specific configurations  
    elif "xlnet" in model_name_lower:
        config.update({
            'warmup_steps': 1000,
            'max_grad_norm': 2.0,
            'special_notes': ['XLNet optimized with extended warmup']
        })
        logger.info("XLNet model detected - applying XLNet-specific configurations")
    
    # RoBERTa and BERT (standard configurations)
    elif any(model_type in model_name_lower for model_type in ["roberta", "bert", "distilbert"]):
        config.update({
            'max_grad_norm': 1.0,
            'special_notes': ['Using standard BERT/RoBERTa configurations']
        })
        logger.info(f"BERT-family model detected ({model_name_or_path}) - using standard configurations")
    
    # ALBERT-specific configurations
    elif "albert" in model_name_lower:
        config.update({
            'warmup_steps': 1000,
            'gradient_accumulation_steps': 2,
            'special_notes': ['ALBERT optimized with extended warmup and gradient accumulation']
        })
        logger.info("ALBERT model detected - applying ALBERT-specific configurations")
    
    return config

def print_experiment_configuration(args, model_config, total_params, trainable_params, train_dataset_size, steps_per_epoch, tokenizer_name):
    """
    Print a structured summary of the experiment configuration before training starts.
    """
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*50)
    
    logger.info(f"Task:           {args.task.upper()}")
    logger.info(f"Model Type:     {args.model_type}")
    logger.info(f"Model Path:     {args.model_name_or_path}")
    logger.info(f"Tokenizer:      {tokenizer_name}")
    logger.info(f"Device:         {torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')}")
    
    logger.info("-" * 50)
    logger.info("MODEL STATISTICS")
    logger.info(f"Total Params:      {total_params:,}")
    logger.info(f"Trainable Params:  {trainable_params:,} ({trainable_params/total_params:.1%} of total)")
    
    logger.info("-" * 50)
    logger.info("TRAINING DETAILS")
    logger.info(f"Dataset Size:      {train_dataset_size:,} samples")
    logger.info(f"Batch Size:        {args.batch_size}")
    logger.info(f"Epochs:            {args.epochs}")
    logger.info(f"Total Steps:       {steps_per_epoch * args.epochs:,}")
    logger.info(f"Learning Rate:     {args.learning_rate}")
    logger.info(f"Weight Decay:      {args.weight_decay}")
    logger.info(f"Warmup Steps:      {model_config['warmup_steps']}")
    logger.info(f"Max Grad Norm:     {model_config['max_grad_norm']}")
    logger.info(f"FP16/BF16:         {model_config['use_fp16']}/{model_config['use_bf16']}")
    
    if model_config['special_notes']:
        logger.info("-" * 50)
        logger.info("MODEL SPECIFIC SETTINGS")
        for note in model_config['special_notes']:
            logger.info(f"* {note}")
            
    logger.info("="*50 + "\n")

def finetune_model_on_glue(args):
    """
    Fine-tune a model on a specific GLUE task and evaluate performance.
    
    Args:
        args (argparse.Namespace): Command line arguments with model and training configuration
        
    Returns:
        tuple: (eval_results, experiment_metadata)
            - eval_results (dict): Raw evaluation metrics from trainer
            - experiment_metadata (dict): Comprehensive experiment information including:
                * experiment_id: Unique wandb run ID
                * model info: name, type, parameters
                * training info: times, hyperparameters
                * wandb_url: Direct link to experiment
                
    Note:
        Supports different model types: bert-type, xlnet-type, concept-type
        Automatically tracks experiment in Weights & Biases
        Handles both classification and regression tasks
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine tokenizer name
    # Strategy:
    # 1. If tokenizer_name is explicitly provided, use it (highest priority)
    # 2. If not, try to load tokenizer from the model directory (best practice)
    # 3. If not available, check if model config has 'tokenizer_name' stored (traceability)
    # 4. Fallback to model_name_or_path (works for HF Hub models)
    
    tokenizer_name = args.tokenizer_name
    
    if not tokenizer_name:
        # Check if tokenizer files exist in the model directory
        if os.path.isdir(args.model_name_or_path) and any(f.startswith("vocab") or f.startswith("tokenizer") for f in os.listdir(args.model_name_or_path)):
            tokenizer_name = args.model_name_or_path
            logger.info(f"Found tokenizer files in model directory. Using: {tokenizer_name}")
        # Check if config has stored tokenizer name
        elif hasattr(model, "config") and hasattr(model.config, "tokenizer_name"):
            tokenizer_name = model.config.tokenizer_name
            logger.info(f"Using stored tokenizer name from model config: {tokenizer_name}")
        else:
            tokenizer_name = args.model_name_or_path
            logger.info(f"Fallback: Using model path as tokenizer name: {tokenizer_name}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=TOKENIZER_CACHE_DIR, token=hf_token)
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        # If fallback failed, try the hardcoded default from training script
        default_tokenizer = "bert-base-cased"
        logger.warning(f"Attempting fallback to default tokenizer: {default_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(default_tokenizer, cache_dir=TOKENIZER_CACHE_DIR, token=hf_token)
    
    # Load and initialize model based on model type
    concept_model_types = ["sim_matrix_mlm", "concept_mlm", "weighted_mlm", "perceiver_mlm"]
    if args.model_type in concept_model_types:
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
            # Check if we should use the weighted version (e.g. if args.model_type is weighted_mlm)
            if args.model_type == "weighted_mlm":
                logger.info(f"Using Weighted Sequence Classification for model type: {args.model_type}")
                model_class = ConceptEncoderForSequenceClassificationWeighted
            elif args.model_type == "perceiver_mlm":
                logger.info(f"Using Perceiver Sequence Classification for model type: {args.model_type}")
                model_class = ConceptEncoderForSequenceClassificationPerceiver
            else:
                raise ValueError(f"Unsupported model type for classification: {args.model_type}")
                
            model = model_class.from_pretrained(
                args.model_name_or_path,
                config=config,
                cache_dir=MODEL_CACHE_DIR,
                token=hf_token
            )
            logger.info(f"Successfully loaded ConceptEncoder model from {args.model_name_or_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {args.model_name_or_path}: {e}")
            logger.warning("Initializing a new ConceptEncoderForSequenceClassificationWeighted model instead.")
            # Initialize a new model with the config
            if args.model_type == "weighted_mlm":
                 model = ConceptEncoderForSequenceClassificationWeighted(config)
            elif args.model_type == "perceiver_mlm":
                 model = ConceptEncoderForSequenceClassificationPerceiver(config)
            else:
                 raise ValueError(f"Unsupported model type for classification initialization: {args.model_type}")
    else:  # Standard transformer models like bert, roberta, xlnet
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=GLUE_TASKS[args.task]["num_labels"],
            problem_type="regression" if args.task == "stsb" else "single_label_classification",
            cache_dir=MODEL_CACHE_DIR,
            token=hf_token
        )
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Load and preprocess dataset
    train_dataset, eval_dataset, _ = load_glue_dataset(args.task, tokenizer, args.max_length)
    
    # Calculate dynamic logging steps based on dataset size
    # Target approximately 10-15 logs per epoch for readability
    train_size = len(train_dataset)
    steps_per_epoch = max(1, train_size // args.batch_size // 2)  # Account for gradient accumulation of 2
    logging_steps = max(1, steps_per_epoch // 10)  # Aim for ~10 logs per epoch
    
    # Get model-specific configuration
    model_config = get_model_specific_config(args.model_name_or_path)
    
    # Print grouped experiment configuration
    print_experiment_configuration(
        args, 
        model_config, 
        total_params, 
        trainable_params, 
        train_size, 
        steps_per_epoch,
        tokenizer_name
    )
    
    # Create experiment timestamp and run name
    experiment_name, timestamp = create_experiment_name(args.model_name_or_path, args.task, total_params)
    run_name = f"{experiment_name}-{timestamp}"
    
    
    # Override with command line arguments if provided, otherwise use model-specific defaults
    final_adam_beta1 = args.adam_beta1 if hasattr(args, 'adam_beta1') and args.adam_beta1 != 0.9 else model_config['adam_beta1']
    final_adam_beta2 = args.adam_beta2 if hasattr(args, 'adam_beta2') and args.adam_beta2 != 0.999 else model_config['adam_beta2']
    final_adam_epsilon = args.adam_epsilon if hasattr(args, 'adam_epsilon') and args.adam_epsilon != 1e-8 else model_config['adam_epsilon']
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.task}"),
        logging_dir=os.path.join(args.output_dir, f"{args.task}/logs"),
        
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=model_config['warmup_steps'],
        lr_scheduler_type="linear",
        bf16=model_config['use_bf16'],
        fp16=model_config['use_fp16'],

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=model_config['max_grad_norm'],
        optim="adamw_torch",
        adam_beta1=final_adam_beta1,
        adam_beta2=final_adam_beta2,
        adam_epsilon=final_adam_epsilon,
        gradient_accumulation_steps=model_config['gradient_accumulation_steps'],
        
        eval_strategy="epoch", # change to eval_strategy
        save_strategy="epoch",
        logging_steps=logging_steps,
        seed=42,

        load_best_model_at_end=True,
        dataloader_num_workers=2,
        metric_for_best_model=GLUE_TASKS[args.task]["metrics"][0],
        push_to_hub=False,
        report_to=["tensorboard", "wandb"],
        run_name=run_name,
        
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Create comprehensive config dictionary for wandb
    wandb_config = {
        '_name_or_path': args.model_name_or_path,
        'model_type': args.model_type,
        'task': args.task,
        'total_params': total_params,
        'trainable_params': trainable_params,
        **{k: v for k, v in vars(training_args).items() if not k.startswith('_')}
    }

    # Create wandb tags
    # Sanitize model name for wandb tag to avoid errors with long paths
    model_tag = args.model_name_or_path
    if os.path.isdir(model_tag):
        model_tag = os.path.basename(model_tag)
    
    # Strip timestamp from model name if it exists (e.g., _YYYYMMDD_HHMMSS)
    parts = model_tag.split('_')
    if len(parts) > 2 and parts[-2].isdigit() and len(parts[-2]) == 8 and parts[-1].isdigit() and len(parts[-1]) == 6:
        model_tag = '_'.join(parts[:-2])

    # Ensure tag is not too long
    if len(model_tag) > 63:
        model_tag = model_tag[:63]

    # Get hostname
    hostname = get_hostname()

    wandb_tags = [
        "glue",
        args.task,
        "finetuning",
        args.model_type,
        model_tag,
        hostname
    ]

    # Create group identifier matching training script if possible
    # Default fallback
    group_identifier = f"GLUE_{args.task}"
    
    # Try to reconstruct training group identifier: {model_type}_H{hidden}L{layers}C{concepts}
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "hidden_size") and hasattr(config, "num_hidden_layers") and hasattr(config, "concept_num"):
             # Use args.model_type (e.g. weighted_mlm) and config params
             group_identifier = f"{args.model_type}_H{config.hidden_size}L{config.num_hidden_layers}C{config.concept_num}"
             logger.info(f"Using training-compatible group identifier: {group_identifier}")
    
    # Initialize the wandb project
    wandb_run = wandb.init(
        project="MrCogito",
        id=run_name,  # Use run_name as a unique ID for resuming
        name=run_name,
        job_type=f"glue_{args.task}_evaluation",
        config=wandb_config,
        tags=wandb_tags,
        group=group_identifier,
        sync_tensorboard=True,
        notes=f"Fine-tuning {args.model_name_or_path} on GLUE task {args.task}"
    )

    # Train model
    logger.info(f"Training {args.model_name_or_path} ({args.model_type}) on {args.task}...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    logger.info(f"Evaluating {args.model_name_or_path} ({args.model_type}) on {args.task}...")
    eval_start_time = time.time()
    eval_results = trainer.evaluate()
    eval_time = time.time() - eval_start_time
    
    # Save model if requested
    if args.save_model:
        trainer.save_model(os.path.join(args.output_dir, f"{args.task}/final_model"))
    
    # Print results
    logger.info(f"Evaluation results for {args.task}:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value}")
    
    # Prepare comprehensive experiment metadata
    experiment_metadata = {
        'experiment_id': wandb_run.id,
        'experiment_name': run_name,
        'wandb_url': wandb_run.url,
        'wandb_project': wandb_run.project,
        'model_name': args.model_name_or_path,
        'model_type': args.model_type,
        'task': args.task,
        'timestamp': timestamp,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time_seconds': training_time,
        'eval_time_seconds': eval_time,
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'max_length': args.max_length,
        'seed': args.seed,
        'adam_beta1': final_adam_beta1,
        'adam_beta2': final_adam_beta2,
        'adam_epsilon': final_adam_epsilon,
        'warmup_steps': model_config['warmup_steps'],
        'max_grad_norm': model_config['max_grad_norm'],
        'gradient_accumulation_steps': model_config['gradient_accumulation_steps'],
        'use_bf16': model_config['use_bf16'],
        'use_fp16': model_config['use_fp16'],
        'model_specific_notes': '; '.join(model_config['special_notes'])
    }
    
    # Close wandb run
    wandb.finish()
    
    return eval_results, experiment_metadata

def create_experiment_results(eval_results, experiment_metadata):
    """
    Create standardized result entries from evaluation results and experiment metadata.
    
    Args:
        eval_results (dict): Raw evaluation results from trainer.evaluate()
        experiment_metadata (dict): Comprehensive experiment metadata
        
    Returns:
        list: List of result dictionaries with consistent structure
    """
    results = []
    
    for metric, score in eval_results.items():
        if metric.startswith("eval_"):
            metric_name = metric[5:]  # Remove "eval_" prefix
            results.append({
                "experiment_id": experiment_metadata['experiment_id'],
                "experiment_name": experiment_metadata['experiment_name'],
                "model_name": experiment_metadata['model_name'],
                "model_type": experiment_metadata['model_type'],
                "task": experiment_metadata['task'],
                "metric": metric_name,
                "score": score,
                "date": experiment_metadata['date'],
                "timestamp": experiment_metadata['timestamp'],
                "wandb_url": experiment_metadata['wandb_url'],
                "total_params": experiment_metadata['total_params'],
                "training_time_seconds": experiment_metadata['training_time_seconds'],
                "eval_time_seconds": experiment_metadata['eval_time_seconds']
            })
        elif metric.startswith("sklearn_"):
            # Store sklearn metric results
            results.append({
                "experiment_id": experiment_metadata['experiment_id'],
                "experiment_name": experiment_metadata['experiment_name'],
                "model_name": experiment_metadata['model_name'],
                "model_type": experiment_metadata['model_type'],
                "task": experiment_metadata['task'],
                "metric": metric,
                "score": score,
                "date": experiment_metadata['date'],
                "timestamp": experiment_metadata['timestamp'],
                "wandb_url": experiment_metadata['wandb_url'],
                "total_params": experiment_metadata['total_params'],
                "training_time_seconds": experiment_metadata['training_time_seconds'],
                "eval_time_seconds": experiment_metadata['eval_time_seconds']
            })
    
    return results

def create_summary_data(experiments_metadata, results_df):
    """
    Create summary data focusing on primary metrics for each experiment.
    
    Args:
        experiments_metadata (list): List of experiment metadata dictionaries
        results_df (pd.DataFrame): DataFrame containing all results
        
    Returns:
        list: List of summary dictionaries with primary metrics only
    """
    summary_data = []
    
    for exp in experiments_metadata:
        # Get primary metrics for this experiment
        exp_results = results_df[results_df['experiment_id'] == exp['experiment_id']]
        primary_metrics = exp_results[~exp_results['metric'].str.startswith('sklearn_')]
        
        if not primary_metrics.empty:
            # Get the main metric for the task (first metric in GLUE_TASKS)
            task_main_metric = GLUE_TASKS[exp['task']]['metrics'][0]
            main_score_results = primary_metrics[primary_metrics['metric'] == task_main_metric]
            
            if not main_score_results.empty:
                main_score = main_score_results['score'].iloc[0]
            else:
                # Fallback to first available metric if main metric not found
                main_score = primary_metrics['score'].iloc[0]
                task_main_metric = primary_metrics['metric'].iloc[0]
            
            summary_data.append({
                'experiment_name': exp['experiment_name'],
                'model_name': exp['model_name'],
                'task': exp['task'],
                'main_metric': task_main_metric,
                'score': main_score,
                'total_params': exp['total_params'],
                'training_time_min': round(exp['training_time_seconds'] / 60, 2),
                'date': exp['date'],
                'wandb_url': exp['wandb_url']
            })
    
    return summary_data

def save_evaluation_reports(results_df, experiments_df, summary_df, model_name, total_params, task, reports_dir):
    """
    Save all evaluation reports to the specified directory with consistent naming.
    
    Args:
        results_df (pd.DataFrame): Detailed results DataFrame
        experiments_df (pd.DataFrame): Experiments metadata DataFrame  
        summary_df (pd.DataFrame): Summary results DataFrame
        model_name (str): Model name for file naming
        total_params (int): Total model parameters for naming
        task (str): GLUE task name
        reports_dir (str): Directory to save reports
        
    Returns:
        tuple: Paths to (results_file, experiments_file, summary_file)
    """
    # Create consistent experiment name and timestamp
    experiment_name, timestamp = create_experiment_name(model_name, task, total_params)
    file_names = create_file_names(experiment_name, timestamp)
    
    # Save detailed results
    results_path = os.path.join(reports_dir, file_names['results'])
    results_df.to_csv(results_path, index=False)
    logger.info(f"Detailed results saved to {results_path}")
    
    # Save experiments metadata
    experiments_path = os.path.join(reports_dir, file_names['metadata'])
    experiments_df.to_csv(experiments_path, index=False)
    logger.info(f"Experiments metadata saved to {experiments_path}")
    
    # Save summary report
    summary_path = os.path.join(reports_dir, file_names['summary'])
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary report saved to {summary_path}")
    
    return results_path, experiments_path, summary_path

def print_results_summary(results_df):
    """
    Print a formatted summary of all results to the console.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame to summarize
    """
    logger.info("Summary of results:")
    for task in sorted(set(results_df["task"])):
        task_results = results_df[results_df["task"] == task]
        logger.info(f"  {task}:")
        
        # Group by experiment to avoid duplicates
        for experiment_id in task_results['experiment_id'].unique():
            exp_results = task_results[task_results['experiment_id'] == experiment_id]
            model_name = exp_results['model_name'].iloc[0]
            logger.info(f"    {model_name}:")
            
            for _, row in exp_results.iterrows():
                logger.info(f"      {row['metric']}: {row['score']:.4f}")

def print_file_locations(results_path, experiments_path, summary_path):
    """
    Print the locations of all generated report files.
    
    Args:
        results_path (str): Path to detailed results file
        experiments_path (str): Path to experiments metadata file
        summary_path (str): Path to summary report file
    """
    logger.info("\nGenerated Reports:")
    logger.info(f"  Detailed Results: {results_path}")
    logger.info(f"  Experiments Metadata: {experiments_path}")
    logger.info(f"  Summary Report: {summary_path}")

def visualize_results(results_df, args):
    """
    Create comprehensive visualizations for evaluation results using rich tables.
    
    This function creates multiple table views:
    1. Primary metrics table - main metric for each experiment
    2. Detailed task tables - all metrics for each task
    3. Model comparison table - side-by-side comparison (multi-task only)
    4. Experiment summary table - overview of all experiments
    
    Args:
        results_df (pd.DataFrame): Results DataFrame with experiment data
        args (argparse.Namespace): Command line arguments containing output_dir
    """
    console = Console()
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # Get unique tasks and models
    unique_tasks = sorted(results_df["task"].unique())
    unique_models = sorted(results_df["model_name"].unique())
    
    # 1. Create a table showing primary metrics for all tasks
    primary_metrics_table = Table(title=f"GLUE Benchmark Results - Primary Metrics", box=box.ROUNDED)
    primary_metrics_table.add_column("Model", style="cyan")
    primary_metrics_table.add_column("Task", style="magenta")
    primary_metrics_table.add_column("Primary Metric", style="yellow")
    primary_metrics_table.add_column("Score", style="green")
    primary_metrics_table.add_column("Training Time (min)", style="blue")
    
    # Filter to primary metrics only (exclude sklearn_ metrics)
    primary_results = results_df[~results_df['metric'].str.startswith('sklearn_')]
    
    # Add rows for each experiment
    for _, row in primary_results.iterrows():
        # Get the primary metric for this task
        task_primary_metric = GLUE_TASKS[row['task']]['metrics'][0]
        
        # Only show the primary metric for each task
        if row['metric'] == task_primary_metric:
            training_time_min = round(row['training_time_seconds'] / 60, 2)
            primary_metrics_table.add_row(
                row['model_name'],
                row['task'].upper(),
                row['metric'],
                f"{row['score']:.4f}",
                f"{training_time_min}"
            )
    
    console.print(primary_metrics_table)
    console.print("\n")
    
    # 2. Create individual tables for each task showing all metrics
    for task in unique_tasks:
        task_results = results_df[results_df["task"] == task]
        
        task_table = Table(title=f"Detailed Results for {task.upper()}", box=box.ROUNDED)
        task_table.add_column("Model", style="cyan")
        task_table.add_column("Metric", style="yellow")
        task_table.add_column("Score", style="green")
        task_table.add_column("Experiment", style="blue")
        
        for _, row in task_results.iterrows():
            # Truncate experiment name for display
            exp_name_short = row['experiment_name']
            task_table.add_row(
                row['model_name'],
                row['metric'],
                f"{row['score']:.4f}",
                exp_name_short
            )
        
        console.print(task_table)
        console.print("\n")
    
    # 3. Create a model comparison table (primary metrics only)
    if len(unique_tasks) > 1:
        comparison_table = Table(title="Model Comparison - Primary Metrics Only", box=box.ROUNDED)
        comparison_table.add_column("Model", style="cyan")
        
        # Add columns for each task
        for task in unique_tasks:
            task_primary_metric = GLUE_TASKS[task]['metrics'][0]
            comparison_table.add_column(f"{task.upper()}\n({task_primary_metric})", style="green")
        
        # Add rows for each model
        for model in unique_models:
            row_data = [model]
            
            for task in unique_tasks:
                task_primary_metric = GLUE_TASKS[task]['metrics'][0]
                model_task_result = results_df[
                    (results_df['model_name'] == model) & 
                    (results_df['task'] == task) & 
                    (results_df['metric'] == task_primary_metric)
                ]
                
                if not model_task_result.empty:
                    score = model_task_result['score'].iloc[0]
                    row_data.append(f"{score:.4f}")
                else:
                    row_data.append("N/A")
            
            comparison_table.add_row(*row_data)
        
        console.print(comparison_table)
        console.print("\n")
    
    # 4. Create experiment summary table
    experiment_summary_table = Table(title="Experiment Summary", box=box.ROUNDED)
    experiment_summary_table.add_column("Experiment", style="cyan")
    experiment_summary_table.add_column("Model", style="magenta")
    experiment_summary_table.add_column("Task", style="yellow")
    experiment_summary_table.add_column("Date", style="blue")
    experiment_summary_table.add_column("Parameters", style="green")
    
    # Get unique experiments
    unique_experiments = results_df.drop_duplicates(['experiment_id'])
    
    for _, row in unique_experiments.iterrows():
        # Format parameters in millions
        params_m = row['total_params'] / 1_000_000
        experiment_summary_table.add_row(
            row['experiment_name'],  # do not truncate for display
            row['model_name'],
            row['task'].upper(),
            row['date'],
            f"{params_m:.1f}M"
        )
    
    console.print(experiment_summary_table)

def main():
    """
    Main function to orchestrate GLUE evaluation experiments.
    
    This function:
    1. Parses command line arguments
    2. Sets up output directories
    3. Runs evaluation on specified task(s)
    4. Collects and processes results
    5. Generates comprehensive reports
    6. Optionally creates visualizations
    
    Returns:
        tuple: Paths to generated report files (results, metadata, summary)
        
    Example:
        Run single task: python script.py --task mrpc --model_name_or_path bert-base-cased
        Run all tasks: python script.py --task all --model_name_or_path bert-base-cased
        With visualization: python script.py --task mrpc --visualize
    """
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create Evaluation_reports directory for CSV reports
    reports_dir = os.path.join("./Cache", "Evaluation_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Evaluate all tasks if task is "all"
    if args.task == "all":
        tasks = list(GLUE_TASKS.keys())
    else:
        tasks = [args.task]
    
    # Store results and experiment metadata
    results = []
    experiments_metadata = []
    
    # Evaluate each task
    for task in tasks:
        logger.info(f"Processing task: {task}")
        
        # Update args with task
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task
        
        # Fine-tune and evaluate
        eval_results, experiment_metadata = finetune_model_on_glue(task_args)
        
        # Store experiment metadata
        experiments_metadata.append(experiment_metadata)
        
        # Store results with experiment metadata
        results.extend(create_experiment_results(eval_results, experiment_metadata))
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Create experiments metadata DataFrame
    experiments_df = pd.DataFrame(experiments_metadata)
    
    # Create summary data
    summary_data = create_summary_data(experiments_metadata, results_df)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Handle naming for single vs multiple tasks
    if args.task == 'all':
        # For multiple tasks, use a general name
        total_params = experiments_metadata[0]['total_params'] if experiments_metadata else 0
        experiment_name, timestamp = create_experiment_name(args.model_name_or_path, 'all-tasks', total_params)
        
        # Create file names manually for multi-task scenario
        file_names = create_file_names(experiment_name, timestamp)
        
        # Save with multi-task naming
        results_path = os.path.join(reports_dir, file_names['results'])
        experiments_path = os.path.join(reports_dir, file_names['metadata'])
        summary_path = os.path.join(reports_dir, file_names['summary'])
        
        results_df.to_csv(results_path, index=False)
        experiments_df.to_csv(experiments_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Multi-task results saved to {results_path}")
        logger.info(f"Multi-task metadata saved to {experiments_path}")
        logger.info(f"Multi-task summary saved to {summary_path}")
    else:
        # Single task - use the standard naming
        total_params = experiments_metadata[0]['total_params'] if experiments_metadata else 0
        
        # Save evaluation reports
        results_path, experiments_path, summary_path = save_evaluation_reports(
            results_df, experiments_df, summary_df, args.model_name_or_path, total_params, args.task, reports_dir
        )
    
    # Print results summary
    print_results_summary(results_df)
    
    # Print file locations
    print_file_locations(results_path, experiments_path, summary_path)
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(results_df, args)
    
    return results_path, experiments_path, summary_path

if __name__ == "__main__":
    main() 