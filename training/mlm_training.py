import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import wandb
import argparse
from datetime import datetime
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    set_seed,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    logging
)

import numpy as np
import torch
from dataclasses import dataclass, field

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_methods import ConceptEncoderForMaskedLM
from nn.concept_encoder_sim_matrix import ConceptEncoderWithSimMatrixForMaskedLM
from nn.concept_encoder_weighted import ConceptEncoderForMaskedLMWeighted
from nn.concept_encoder_perceiver import ConceptEncoderForMaskedLMPerceiver

from training.dataset_preprocess import load_and_preprocess_text_dataset
from training.utils_training import (
    get_parameter_breakdown,
    count_parameters,
    setup_distributed,
    is_main_process,
    get_hostname,
    log_system_info,
    log_model_info
)

# Initialize logger
logger = logging.get_logger(__name__)

# Model registry for cleaner initialization
MODEL_REGISTRY = {
    "sim_matrix_mlm": {
        "class": ConceptEncoderWithSimMatrixForMaskedLM,
        "description": "ConceptEncoder with similarity matrix for MLM"
    },
    "concept_mlm": {
        "class": ConceptEncoderForMaskedLM,
        "description": "Standard ConceptEncoder for MLM"
    },
    "weighted_mlm": {
        "class": ConceptEncoderForMaskedLMWeighted,
        "description": "ConceptEncoder with simplified weighted approach for MLM"
    },
    "perceiver_mlm": {
        "class": ConceptEncoderForMaskedLMPerceiver,
        "description": "ConceptEncoder with Perceiver IO decoding for MLM"
    }
}


@dataclass
class ModelArguments:
    model_type: str = field(
        default="weighted_mlm",
        metadata={"help": "Type of model to train", "choices": ["sim_matrix_mlm", "concept_mlm", "weighted_mlm", "perceiver_mlm"]}
    )
    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden size of the model"}
    )

    intermediate_size: int = field(
        default=1024,
        metadata={"help": "Internal feedforward network size of the model"}
    )
    num_hidden_layers: int = field(
        default=2,
        metadata={"help": "Number of transformer layers"}
    )
    concept_num: int = field(
        default=128,
        metadata={"help": "Number of concepts to train"}
    )
    

@dataclass
class DataTrainingArguments:
    mlm_probability: float = field(
        default=0.25,
        metadata={"help": "Probability for MLM masking"}
    )
    masking_type: str = field(
        default="random",
        metadata={"help": "Masking strategy", "choices": ["random", "whole_word"]}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum input sequence length"}
    )
    test_size_percent: float = field(
        default=0.1,
        metadata={"help": "Percentage of dataset to use for testing"}
    )
    dataset_name: str = field(
        default="Salesforce/wikitext",
        metadata={"help": "Dataset name to use for training from HuggingFace hub"}
    )
    dataset_name_subset: str | None = field(
        default=None, 
        metadata={"help": "Dataset name subset to use for training from HuggingFace hub, provide if exists"}
    )
    tokenizer_name: str = field(
        default="bert-base-uncased",
        metadata={"help": "Tokenizer name to use for training from HuggingFace hub"}
    )
    dataset_cache_dir: str | None = field(
        default="./Cache/Datasets",
        metadata={"help": "Directory to cache downloaded datasets. If not provided, uses ./Cache/Datasets"}
    )

def parse_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()


def main():
    # Setup distributed training
    local_rank = setup_distributed()
    
    # Setup logging - set transformers verbosity first
    if is_main_process():
        # Set transformers logging to info level
        logging.set_verbosity_info()
        # Ensure our logger is also at info level
        import logging as std_logging
        std_logging.basicConfig(
            level=std_logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            force=True
        )
    else:
        logging.set_verbosity_error()
    
    # Parse arguments
    model_args, data_args, training_args = parse_args()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Log system information (must be called after logging setup)
    log_system_info()
    
    # Log data configuration
    logger.info("="*60)
    logger.info("Data Configuration")
    logger.info("="*60)
    logger.info(f"Dataset: {data_args.dataset_name}")
    if data_args.dataset_name_subset:
        logger.info(f"Dataset subset: {data_args.dataset_name_subset}")
    logger.info(f"Tokenizer: {data_args.tokenizer_name}")
    logger.info(f"Max sequence length: {data_args.max_seq_length}")
    logger.info(f"MLM probability: {data_args.mlm_probability}")
    logger.info(f"Masking type: {data_args.masking_type}")
    logger.info(f"Test size: {data_args.test_size_percent * 100}%")
    logger.info(f"Cache directory: {data_args.dataset_cache_dir or './Cache/Datasets'}")
    
    # Load the tokenizer
    logger.info(f"\nLoading tokenizer: {data_args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    
    # Load and preprocess the dataset
    logger.info(f"Loading and preprocessing dataset...")
    train_ds, test_ds = load_and_preprocess_text_dataset(
        tokenizer, 
        data_args.dataset_name, 
        data_args.dataset_name_subset, 
        "text", 
        test_size_percent=data_args.test_size_percent,
        max_seq_length=data_args.max_seq_length,
        dataset_cache_dir=data_args.dataset_cache_dir
    )
    
    logger.info(f"Train dataset size: {len(train_ds):,}")
    logger.info(f"Test dataset size: {len(test_ds):,}")
    logger.info("="*60)
    
    # Create model config using model_args
    # Calculate appropriate number of attention heads based on hidden size
    # Each head should have at least 64 dimensions
    num_attention_heads = max(1, min(8, model_args.hidden_size // 64))
    
    # Ensure all special tokens are correctly mapped from tokenizer to config
    # We validate critical tokens that are required for the model/training to function
    
    # 1. Critical Tokens (Must exist)
    if tokenizer.pad_token_id is None:
        raise ValueError(
            f"Tokenizer '{data_args.tokenizer_name}' does not have a defined pad_token_id. "
            f"ConceptEncoder requires a pad token for embedding initialization and attention masking. "
            f"Please ensure the tokenizer has a pad token defined."
        )
    
    if tokenizer.mask_token_id is None:
        raise ValueError(
            f"Tokenizer '{data_args.tokenizer_name}' does not have a defined mask_token_id. "
            f"MLM training requires a mask token. "
            f"Please ensure the tokenizer has a mask token defined."
        )

    # 2. Optional Tokens (Use if available, else None)
    # We strictly use tokenizer's values or None, avoiding arbitrary defaults like 3/4/etc.
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    unk_token_id = tokenizer.unk_token_id

    config = ConceptEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        concept_num=model_args.concept_num,
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_sequence_length=data_args.max_seq_length,
        
        # Special Tokens
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=cls_token_id,
        sep_token_id=sep_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        unk_token_id=unk_token_id,
        
        tie_word_embeddings=False,
        tokenizer_name=data_args.tokenizer_name  # Store source tokenizer name for traceability
    )
    

        
    # Initialize the model using registry
    if model_args.model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_args.model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_args.model_type]["class"]
    
    # Check if we should load from a checkpoint or initialize fresh
    if training_args.resume_from_checkpoint and training_args.resume_from_checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {training_args.resume_from_checkpoint}")
        # If resume_from_checkpoint is a path, load from there
        # However, Trainer.train(resume_from_checkpoint=...) handles the loading of weights + optimizer state.
        # Here we are initializing the model object. 
        # If we provide a path to .from_pretrained, we load weights.
        # If we use resume_from_checkpoint in trainer.train(), it loads everything.
        # The standard HF pattern is to init config, init model (random), then let trainer load checkpoint.
        # BUT if the user provided --model_name_or_path pointing to a checkpoint, we should load weights here.
        pass 
    
    # Check if model_name_or_path is provided and is a directory (implies checkpoint/saved model)
    # In TrainingArguments, model_name_or_path is not a standard argument, it usually comes from ModelArguments
    # But we don't have model_name_or_path in ModelArguments definition above (it was missing).
    # Let's check if we can infer it or if we should add it to ModelArguments.
    
    # For now, we'll stick to initializing from config unless we want to explicitly support loading weights here.
    # If we want to continue training with NEW epochs but OLD weights, we should ideally load weights here.
    
    logger.info(f"Initializing model: {model_info['description']}")
    
    if model_args.model_name_or_path:
        logger.info(f"Loading model weights from: {model_args.model_name_or_path}")
        # When loading from a path, we usually want to use .from_pretrained
        # Ensure the config matches what we prepared or let it load from the path
        try:
            model = model_class.from_pretrained(model_args.model_name_or_path, config=config)
        except Exception as e:
            logger.warning(f"Failed to load via from_pretrained (might be a fresh directory?): {e}")
            logger.info("Falling back to fresh initialization")
            model = model_class(config)
    else:
        logger.info("Initializing fresh model from config")
        model = model_class(config)
    
    # Log detailed model information
    log_model_info(
        model, 
        config=config, 
        model_type=model_args.model_type,
        model_description=model_info['description']
    )
    
    # Data collator for dynamic masking
    if data_args.masking_type == "whole_word":
        # whole word masking - mask the random words (not neighbors)
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64
        )
    else:
        # random masking - the classic one, default
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=64
        )
    
    # Configure training arguments with defaults and timestamped directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a consistent run identifier using underscores (wandb best practice)
    # Format: model_type_H{hidden_size}L{layers}C{concept_num}_{timestamp}
    # IMPORTANT: Use underscores only (no hyphens) for wandb compatibility
    # This identifier will be used for run_name, logging_dir, and wandb.init(name)
    run_identifier = f"{model_args.model_type}_H{model_args.hidden_size}L{model_args.num_hidden_layers}C{model_args.concept_num}_{timestamp}"
    
    # Create a unique run ID for wandb (allows resuming runs if needed)
    # Format: same as run_identifier but can be used for resume functionality
    run_id = run_identifier
    
    # Set timestamped output directories - use consistent naming with run_name
    training_args.output_dir = os.path.join(training_args.output_dir or "./outputs", run_identifier)
    
    # CRITICAL: logging_dir structure affects panel names when sync_tensorboard=True
    # When sync_tensorboard=True, wandb creates panels from the directory structure
    # To ensure consistent panel names matching run names:
    # 1. Use the same base name as run_name
    # 2. Avoid nested subdirectories that don't match the run name
    # 3. The directory name becomes part of the panel name
    training_args.logging_dir = os.path.join(training_args.logging_dir or "./logs", run_identifier)
    
    # Use the same identifier for run_name to ensure consistency across wandb, Trainer, and directories
    # This ensures: wandb.run.name == TrainingArguments.run_name == logging_dir base name
    training_args.run_name = run_identifier
    
    # Log training configuration
    logger.info("="*60)
    logger.info("Training Configuration")
    logger.info("="*60)
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"Logging directory: {training_args.logging_dir}")
    logger.info(f"Run name: {training_args.run_name}")
    logger.info(f"Per-device train batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Per-device eval batch size: {training_args.per_device_eval_batch_size}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * device_count * training_args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Number of epochs: {training_args.num_train_epochs}")
    logger.info(f"Warmup steps: {training_args.warmup_steps}")
    logger.info(f"Weight decay: {training_args.weight_decay}")
    logger.info(f"Evaluation strategy: {training_args.eval_strategy}")
    logger.info(f"Save strategy: {training_args.save_strategy}")
    logger.info(f"Mixed precision: {'fp16' if training_args.fp16 else 'bf16' if training_args.bf16 else 'fp32'}")
    logger.info(f"Seed: {training_args.seed}")
    logger.info("="*60)
    
    # Set default values if not provided via command line
    training_args.per_device_train_batch_size = training_args.per_device_train_batch_size or 24
    training_args.num_train_epochs = training_args.num_train_epochs or 2
    training_args.learning_rate = training_args.learning_rate or 5e-4
    training_args.weight_decay = training_args.weight_decay or 0.01
    training_args.warmup_steps = training_args.warmup_steps or 1000
    training_args.seed = training_args.seed or 42
    training_args.logging_steps = training_args.logging_steps or 100
    training_args.eval_strategy = training_args.eval_strategy or "epoch"
    training_args.save_strategy = training_args.save_strategy or "epoch"
    training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps or 1
    training_args.per_device_eval_batch_size = training_args.per_device_eval_batch_size or training_args.per_device_train_batch_size
    
    
    # Set fixed training configuration
    training_args.overwrite_output_dir = True
    training_args.save_safetensors = False
    training_args.dataloader_num_workers = 2
    training_args.report_to = ["tensorboard", "wandb"]
    training_args.push_to_hub = False
    training_args.remove_unused_columns = True
    training_args.optim = "adamw_torch"
    training_args.max_grad_norm = 1.0
    training_args.use_cpu = False
    
    # Set fp16 as default (True), unless bf16 is explicitly set (then fp16=False)
    training_args.fp16 = not training_args.bf16
        
    
    # Clear eval/save steps if not using step-based strategy
    if training_args.eval_strategy != "steps":
        training_args.eval_steps = None
    if training_args.save_strategy != "steps":
        training_args.save_steps = None
    
    
    
    # Initialize wandb only on main process
    if is_main_process():
        
        # Get model parameter counts for logging
        total_params, trainable_params = count_parameters(model)
        
        # Create model identifier for W&B _name_or_path field
        model_name_or_path = f"concept-encoder-{model_args.model_type}"
        
        # Get hostname in cross-platform way
        hostname = get_hostname()
        
        wandb_tags = [
            model_args.model_type,
            "mlm-pretraining",
            data_args.dataset_name,
            data_args.masking_type,
            hostname,
            model_name_or_path
        ]
        if data_args.dataset_name_subset:
            wandb_tags.append(data_args.dataset_name_subset)
        
        
        # Create comprehensive config dictionary
        wandb_config = {
            # Model identifier (special field for W&B)
            '_name_or_path': model_name_or_path,
            
            # Model architecture
            'model_type': model_args.model_type,
            'hidden_size': model_args.hidden_size,
            'num_hidden_layers': model_args.num_hidden_layers,
            'concept_num': model_args.concept_num,
            'intermediate_size': model_args.intermediate_size,
            'num_attention_heads': config.num_attention_heads,
            'vocab_size': config.vocab_size,
            'max_sequence_length': config.max_sequence_length,
            'total_params': total_params,
            'trainable_params': trainable_params,
            
            # Data configuration
            'dataset_name': data_args.dataset_name,
            'dataset_name_subset': data_args.dataset_name_subset,
            'tokenizer_name': data_args.tokenizer_name,
            'max_seq_length': data_args.max_seq_length,
            'mlm_probability': data_args.mlm_probability,
            'masking_type': data_args.masking_type,
            'test_size_percent': data_args.test_size_percent,
            
            # Training configuration (from training_args)
            **{k: v for k, v in vars(training_args).items() if not k.startswith('_')}
        }
        
        logger.info("Initializing Weights & Biases...")
        # Initialize wandb with best practices for Hugging Face Transformers integration:
        # 
        # KEY PARAMETERS:
        # - id: Unique identifier for the run (allows resuming if needed)
        # - name: Human-readable run name (MUST match TrainingArguments.run_name exactly)
        # - job_type: Categorizes runs (e.g., "pretraining", "finetuning", "evaluation")
        # - group: Groups related runs for easier comparison (e.g., same model config)
        # - tags: For filtering and searching runs
        # - sync_tensorboard: Syncs TensorBoard logs (creates panels from logging_dir structure)        
        # Create group identifier for clustering related runs (same model config)
        group_identifier = f"{model_args.model_type}_H{model_args.hidden_size}L{model_args.num_hidden_layers}C{model_args.concept_num}"
        
        wandb.init(
            project="MrCogito",
            id=run_id,  # Unique identifier (allows resuming runs)
            name=training_args.run_name,  # MUST match TrainingArguments.run_name exactly
            job_type="mlm-pretraining",  # Categorizes the type of training job
            config=wandb_config,
            tags=wandb_tags,
            group=group_identifier,  # Group by model config for easier comparison
            sync_tensorboard=True,  # Syncs TensorBoard logs - panel names come from logging_dir structure
            notes=f"Model: {model_args.model_type}, Dataset: {data_args.dataset_name}"  # Add context
        )
        logger.info(f"W&B run initialized:")
        logger.info(f"  - Run ID: {wandb.run.id}")
        logger.info(f"  - Run name: {wandb.run.name}")
        logger.info(f"  - Run group: {wandb.run.group}")
        logger.info(f"  - Job type: {wandb.run.job_type}")
        logger.info(f"  - Logging dir: {training_args.logging_dir}")
        logger.info(f"  - Note: Panel names will be based on logging_dir structure when sync_tensorboard=True")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    logger.info("="*60)
    logger.info(f"Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(training_args.output_dir, training_args.run_name)
    logger.info(f"Saving final model to: {final_model_path}")
    trainer.save_model(final_model_path)
    
    # Explicitly save tokenizer to ensure it's available for evaluation
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Saved tokenizer to {final_model_path}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
