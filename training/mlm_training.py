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
from typing import List, Optional

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_methods import ConceptEncoderForMaskedLM
from nn.concept_encoder_sim_matrix import ConceptEncoderWithSimMatrixForMaskedLM
from nn.concept_encoder_weighted import ConceptEncoderForMaskedLMWeighted
from nn.concept_encoder_perceiver import (
    ConceptEncoderForMaskedLMPerceiver,
    ConceptEncoderForMaskedLMPerceiverPosOnly
)
from nn.concept_encoder_recursive_mlm import RecursiveConceptEncoderForMaskedLM
from nn.concept_encoder_recursive import RecursiveConceptEncoderConfig
from nn.loss_manager import LossConfig, ConceptLossStepCallback, get_available_losses

from data.dataset_preprocess import load_and_preprocess_text_dataset
from training.utils_training import (
    get_parameter_breakdown,
    count_parameters,
    setup_distributed,
    is_main_process,
    get_hostname,
    log_system_info,
    log_model_info,
    get_git_info,
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
        "description": "ConceptEncoder with Perceiver IO decoding for MLM (Input+Position queries)"
    },
    "perceiver_posonly_mlm": {
        "class": ConceptEncoderForMaskedLMPerceiverPosOnly,
        "description": "ConceptEncoder with Perceiver IO decoding for MLM (Position-only queries, pure Perceiver IO)"
    },
    "recursive_mlm": {
        "class": RecursiveConceptEncoderForMaskedLM,
        "description": "Recursive ConceptEncoder (1 shared layer, K iterations) with Perceiver IO decoding for MLM",
        "config_class": RecursiveConceptEncoderConfig,
    }
}


@dataclass
class ModelArguments:
    model_type: str = field(
        default="weighted_mlm",
        metadata={"help": "Type of model to train", "choices": list(MODEL_REGISTRY.keys())}
    )
    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # torch.compile is applied MANUALLY here (not via TrainingArguments.torch_compile) so we
    # can pass dynamic=True.  TrainingArguments.torch_compile should be kept False to avoid
    # double-compilation.  The Feb-2026 training instability (loss stuck at 7.0 vs 2.54, grad
    # explosion at step 8000–9000) was caused by the default static-shape compile tracing
    # triggering recompilation / eager-mode fallbacks on every batch due to the variable number
    # of masked tokens produced by DataCollatorForLanguageModeling.  dynamic=True resolves this
    # by emitting shape-agnostic code via symbolic integers.
    torch_compile_dynamic: bool = field(
        default=False,
        metadata={"help": "Compile model with torch.compile(dynamic=True) for stable training "
                  "with variable-shape tensors (e.g. sparse MLM masked token counts). "
                  "Keep TrainingArguments.torch_compile=False when this is True. "
                  "Backend is read from TrainingArguments.torch_compile_backend (default: inductor)."}
    )
    hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden size of the model (concept dimension, attention dimension)"}
    )
    token_embedding_dim: int = field(
        default=0,
        metadata={"help": "Token embedding dimension. 0 = same as hidden_size (backward compat). "
                  "When smaller than hidden_size, enables Dimension Inversion: tokens are cheap "
                  "(small vocab memory) while concepts are rich (large hidden_size)."}
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
    concept_position_type: str = field(
        default="none",
        metadata={"help": "Concept position encoding type: 'none' (orderless), "
                  "'sinusoidal' (fixed, no extra params), 'learned' (trainable)"}
    )


@dataclass
class LossArguments:
    """
    Arguments for loss configuration.
    
    Examples:
        # MLM only (no concept loss)
        --concept_losses none
        
        # MLM + orthogonality with fixed weight 0.1
        --concept_losses orthogonality --loss_weighting fixed --loss_weight 0.1
        
        # MLM + orthogonality with learnable weights (Kendall & Gal)
        --concept_losses orthogonality --loss_weighting kendall_gal
        
        # MLM + two concept losses
        --concept_losses orthogonality uniformity --loss_weighting kendall_gal
    """
    concept_losses: Optional[str] = field(
        default="orthogonality",
        metadata={
            "help": f"Concept loss types to use, space-separated. 'none' for no concept loss. "
                    f"Available: {get_available_losses()}"
        }
    )
    loss_weighting: str = field(
        default="kendall_gal",
        metadata={
            "help": "Loss weighting strategy: 'fixed', 'learnable', or 'kendall_gal'",
            "choices": ["fixed", "learnable", "kendall_gal"]
        }
    )
    loss_weight: float = field(
        default=0.1,
        metadata={
            "help": "Fixed weight for concept loss (only used with --loss_weighting fixed)"
        }
    )
    # Loss-specific parameters
    soft_ortho_threshold: float = field(
        default=0.1,
        metadata={"help": "Threshold for soft_orthogonality loss"}
    )
    uniformity_temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for uniformity loss"}
    )
    concept_loss_warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup steps for concept losses (0 = no warmup). "
                          "Only effective with fixed weighting."}
    )
    
    def to_loss_config(self) -> LossConfig:
        """Convert arguments to LossConfig."""
        if self.concept_losses is None or self.concept_losses.lower() == "none":
            return LossConfig.disabled()
        
        losses = self.concept_losses.split()
        
        loss_weights = {"task": 1.0}
        if self.loss_weighting == "fixed":
            per_loss_weight = self.loss_weight / len(losses) if losses else 0
            for loss_name in losses:
                loss_weights[loss_name] = per_loss_weight
        
        loss_params = {}
        if "soft_orthogonality" in losses:
            loss_params["soft_orthogonality"] = {"threshold": self.soft_ortho_threshold}
        if "uniformity" in losses or "combined" in losses:
            loss_params["uniformity"] = {"temperature": self.uniformity_temperature}
            loss_params["combined"] = {"temperature": self.uniformity_temperature}
        
        return LossConfig(
            concept_losses=losses,
            weighting_strategy=self.loss_weighting,
            loss_weights=loss_weights,
            loss_params=loss_params,
            warmup_steps=self.concept_loss_warmup_steps,
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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LossArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()


def main():
    # Setup distributed training
    local_rank = setup_distributed()

    
    # Setup logging - both console and file output for debugging
    import logging as std_logging
    if is_main_process():
        logging.set_verbosity_info()
        
        # Create a timestamped log file alongside the training output
        log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_dir = os.environ.get("LOG_DIR", "./Cache/logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filepath = os.path.join(log_dir, log_filename)
        
        # Configure root logger with both console and file handlers
        root_logger = std_logging.getLogger()
        root_logger.setLevel(std_logging.INFO)
        # Clear existing handlers to avoid duplicates on re-entry
        root_logger.handlers.clear()
        
        formatter = std_logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = std_logging.StreamHandler()
        console_handler.setLevel(std_logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (appends so multiple runs accumulate in the same dir)
        file_handler = std_logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
        file_handler.setLevel(std_logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_filepath}")
    else:
        logging.set_verbosity_error()
    
    # Parse arguments
    model_args, data_args, loss_args, training_args = parse_args()
    
    # Create loss configuration from arguments
    loss_config = loss_args.to_loss_config()
    
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
    # Only load on the main process to avoid memory spikes and redundant processing
    # The processed dataset is cached, so other processes will load it quickly
    # However, 'load_dataset' in dataset_preprocess is not distributed-aware by default in this script structure.
    # To be safe and simple: let main process load and cache, then others load.
    
    with training_args.main_process_first(desc="loading and tokenizing dataset"):
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
    # Pass special tokens directly to config
    # Resolve token_embedding_dim: 0 means same as hidden_size (backward compat)
    token_embedding_dim = model_args.token_embedding_dim if model_args.token_embedding_dim > 0 else None
    
    # When Dimension Inversion is active (token_dim < hidden_size), weight tying
    # is not possible because lm_head shape [hidden_size, vocab] != token_emb shape [vocab, token_dim]
    should_tie = token_embedding_dim is None or token_embedding_dim == model_args.hidden_size
    
    # Use model-specific config class if the registry specifies one
    config_class = MODEL_REGISTRY.get(model_args.model_type, {}).get("config_class", ConceptEncoderConfig)
    
    config_kwargs = dict(
        vocab_size=len(tokenizer),
        concept_num=model_args.concept_num,
        hidden_size=model_args.hidden_size,
        token_embedding_dim=token_embedding_dim,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_sequence_length=data_args.max_seq_length,
        concept_position_type=model_args.concept_position_type,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        tie_word_embeddings=should_tie,
        tokenizer_name=data_args.tokenizer_name,
    )
    
    config = config_class(**config_kwargs)
    

        
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
    
    model_info = MODEL_REGISTRY[model_args.model_type]
    logger.info(f"Initializing model: {model_info['description']}")
    
    # Log loss configuration
    logger.info("="*60)
    logger.info("Loss Configuration")
    logger.info("="*60)
    logger.info(f"Concept losses: {loss_config.concept_losses or 'none (MLM only)'}")
    logger.info(f"Weighting strategy: {loss_config.weighting_strategy}")
    if loss_config.weighting_strategy == "fixed":
        logger.info(f"Loss weights: {loss_config.loss_weights}")
    logger.info("="*60)
    
    # Models that support loss_config parameter
    models_with_loss_config = {"weighted_mlm", "perceiver_mlm", "perceiver_posonly_mlm", "recursive_mlm"}
    supports_loss_config = model_args.model_type in models_with_loss_config
    
    if model_args.model_name_or_path:
        logger.info(f"Loading model weights from: {model_args.model_name_or_path}")
        try:
            model = model_class.from_pretrained(model_args.model_name_or_path, config=config)
            # Set loss config after loading (not saved with model)
            if supports_loss_config and hasattr(model, 'set_loss_config'):
                model.set_loss_config(loss_config)
        except Exception as e:
            logger.warning(f"Failed to load via from_pretrained (might be a fresh directory?): {e}")
            logger.info("Falling back to fresh initialization")
            if supports_loss_config:
                model = model_class(config, loss_config=loss_config)
            else:
                model = model_class(config)
    else:
        logger.info("Initializing fresh model from config")
        if supports_loss_config:
            model = model_class(config, loss_config=loss_config)
        else:
            model = model_class(config)
            if loss_config.is_enabled:
                logger.warning(
                    f"Model type '{model_args.model_type}' does not support configurable loss. "
                    f"Concept loss settings will be ignored."
                )
    
    # Verify Flash Attention is available and will be used by SDPA.
    # F.scaled_dot_product_attention requires 4D tensors: [batch, heads, seq_len, head_dim].
    # nn.MultiheadAttention(need_weights=False) reshapes internally before calling SDPA,
    # so the actual training already uses the correct format — this test mirrors that.
    if torch.cuda.is_available() and is_main_process():
        _num_heads = config.num_attention_heads
        _head_dim  = config.hidden_size // _num_heads   # 512 / 8 = 64 for L6 model
        try:
            # [batch=1, heads=8, seq_q=128(concepts), head_dim=64]  Q = concepts
            _q = torch.zeros(1, _num_heads, config.concept_num, _head_dim,
                             dtype=torch.bfloat16, device="cuda")
            # [batch=1, heads=8, seq_k=512(tokens),  head_dim=64]  K/V = tokens
            _k = torch.zeros(1, _num_heads, 512, _head_dim,
                             dtype=torch.bfloat16, device="cuda")
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                torch.nn.functional.scaled_dot_product_attention(_q, _k, _k)
            logger.info(
                f"Flash Attention v2: ACTIVE ✓  "
                f"(heads={_num_heads}, head_dim={_head_dim}, dtype=bf16)"
            )
        except Exception as _fa_exc:
            logger.warning(
                f"Flash Attention not available — training will use memory-efficient / math SDPA. "
                f"Reason: {_fa_exc}"
            )
        finally:
            del _q, _k

    # Log detailed model information
    log_model_info(
        model, 
        config=config, 
        model_type=model_args.model_type,
        model_description=model_info['description']
    )

    # Apply torch.compile with dynamic=True AFTER model init, BEFORE Trainer creation.
    # Using dynamic=True prevents constant recompilation caused by variable masked-token
    # counts from DataCollatorForLanguageModeling (each batch has a different number of
    # ~15% masked positions, producing variable-size sparse tensors inside the model).
    # Keep training_args.torch_compile=False so HF Trainer does NOT compile again.
    if model_args.torch_compile_dynamic:
        if not torch.cuda.is_available():
            logger.warning("torch_compile_dynamic=True but no CUDA detected — skipping compile.")
        else:
            # Backend comes from TrainingArguments.torch_compile_backend (already defined there).
            # Default "inductor" is fine for RTX 3090 (Ampere, sm86).
            backend = getattr(training_args, "torch_compile_backend", None) or "inductor"
            logger.info(f"Applying torch.compile(dynamic=True, backend='{backend}') ...")
            model = torch.compile(
                model,
                dynamic=True,    # Handle variable masked-token shapes without recompilation
                fullgraph=False, # Allow graph breaks (safer for complex HF models)
                backend=backend,
            )
            logger.info("torch.compile applied successfully.")
    
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
    # Format: model_type_H{hidden_size}L{layers}C{concept_num}[_T{token_dim}][_pos{type}]_{timestamp}
    # IMPORTANT: Use underscores only (no hyphens) for wandb compatibility
    # This identifier will be used for run_name, logging_dir, and wandb.init(name)
    base_id = f"{model_args.model_type}_H{model_args.hidden_size}L{model_args.num_hidden_layers}C{model_args.concept_num}"
    # Append token_embedding_dim suffix only when Dimension Inversion is active
    if config.token_embedding_dim != config.hidden_size:
        base_id += f"_T{config.token_embedding_dim}"
    # Append concept position type suffix only when non-default
    if config.concept_position_type != "none":
        base_id += f"_pos{config.concept_position_type}"
    run_identifier = f"{base_id}_{timestamp}"
    
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
    logger.info(f"Eval steps: {training_args.eval_steps}")
    logger.info(f"Save strategy: {training_args.save_strategy}")
    logger.info(f"Save steps: {training_args.save_steps}")
    logger.info(f"Mixed precision: {'fp16' if training_args.fp16 else 'bf16' if training_args.bf16 else 'fp32'}")
    logger.info(f"Seed: {training_args.seed}")
    logger.info(f"Optimizer: {training_args.optim}")
    logger.info(f"LR scheduler: {training_args.lr_scheduler_type}")
    logger.info(f"Max grad norm: {training_args.max_grad_norm}")
    logger.info(f"Dataloader num workers: {training_args.dataloader_num_workers}")
    logger.info(f"Dataloader pin memory: {training_args.dataloader_pin_memory}")
    logger.info(f"torch.compile: {training_args.torch_compile}")
    logger.info(f"Save safetensors: {training_args.save_safetensors}")
    logger.info(f"Gradient checkpointing: {training_args.gradient_checkpointing}")
    logger.info(f"Load best model at end: {training_args.load_best_model_at_end}")
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
    
    
    # BUG FIX (2026-02-18): These were HARDCODED overrides that silently replaced CLI args!
    # Previously: training_args.optim = "adamw_torch" would overwrite --optim "adamw_torch_fused"
    # Now: only set sensible defaults that don't conflict with CLI-provided values.
    # Values that should always be set regardless of CLI:
    training_args.report_to = ["tensorboard", "wandb"]
    training_args.push_to_hub = False
    training_args.remove_unused_columns = True
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
        # Concept loss tags — makes filtering WandB runs by loss config easy
        if loss_config.is_enabled:
            wandb_tags.append(f"losses:{'+'.join(loss_config.concept_losses)}")
            wandb_tags.append(f"weighting:{loss_config.weighting_strategy}")
            if loss_config.weighting_strategy == "fixed":
                # Include the weight value so fixed runs are distinguishable
                task_weight = loss_config.loss_weights.get("task", 1.0)
                concept_weight = loss_config.loss_weights.get(
                    loss_config.concept_losses[0], loss_args.loss_weight
                )
                wandb_tags.append(f"concept_w:{concept_weight}")
        else:
            wandb_tags.append("losses:none")
        
        
        # Create comprehensive config dictionary
        wandb_config = {
            # Model identifier (special field for W&B)
            '_name_or_path': model_name_or_path,
            
            # Model architecture
            'model_type': model_args.model_type,
            'hidden_size': model_args.hidden_size,
            'token_embedding_dim': config.token_embedding_dim,
            'num_hidden_layers': model_args.num_hidden_layers,
            'concept_num': model_args.concept_num,
            'intermediate_size': model_args.intermediate_size,
            'num_attention_heads': config.num_attention_heads,
            'concept_position_type': config.concept_position_type,
            'vocab_size': config.vocab_size,
            'max_sequence_length': config.max_sequence_length,
            'total_params': total_params,
            'trainable_params': trainable_params,
            
            # Loss configuration
            'concept_losses': loss_config.concept_losses,
            'loss_weighting': loss_config.weighting_strategy,
            'loss_weights': loss_config.loss_weights if loss_config.weighting_strategy == "fixed" else "learnable",
            
            # Data configuration
            'dataset_name': data_args.dataset_name,
            'dataset_name_subset': data_args.dataset_name_subset,
            'tokenizer_name': data_args.tokenizer_name,
            'max_seq_length': data_args.max_seq_length,
            'mlm_probability': data_args.mlm_probability,
            'masking_type': data_args.masking_type,
            'test_size_percent': data_args.test_size_percent,
            
            # Code version traceability (links WandB run to exact git commit)
            **{f"git_{k}": v for k, v in get_git_info().items()},

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
        group_identifier = base_id  # Reuse the base identifier (without timestamp)
        
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
    
    callbacks = []
    if loss_config.warmup_steps > 0:
        callbacks.append(ConceptLossStepCallback())
        logger.info(f"Concept loss warmup: {loss_config.warmup_steps} steps")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
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
