"""Reproducibility-only trainer for the historical weighted-MLM baseline.

New research uses ``training/train_concept_pretraining.py``. The weighted model
remains in ``nn/`` because historical checkpoints still have evaluation support.
"""

import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import wandb
from datetime import datetime
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    set_seed,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    logging,
)

import torch
from dataclasses import dataclass, field
from typing import Optional

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_weighted import ConceptEncoderForMaskedLMWeighted
from nn.loss_manager import LossConfig, ConceptLossStepCallback, get_available_losses

from data.dataset_preprocess import load_and_preprocess_text_dataset
from training.utils_training import (
    setup_distributed,
    is_main_process,
    log_system_info,
    log_model_info,
    setup_file_logging,
    log_data_config,
    log_loss_config,
    log_training_config,
    setup_run_dirs,
    init_wandb,
)

logger = logging.get_logger(__name__)

# Model registry for cleaner initialization
MODEL_REGISTRY = {
    "weighted_mlm": {
        "class": ConceptEncoderForMaskedLMWeighted,
        "description": "ConceptEncoder with simplified weighted approach for MLM"
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
    local_rank = setup_distributed()

    if is_main_process():
        logging.set_verbosity_info()
        setup_file_logging()
    else:
        logging.set_verbosity_error()

    model_args, data_args, loss_args, training_args = parse_args()
    
    # Create loss configuration from arguments
    loss_config = loss_args.to_loss_config()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Log system information (must be called after logging setup)
    log_system_info()
    
    log_data_config(data_args, extra_fields={
        "MLM probability": data_args.mlm_probability,
        "Masking type": data_args.masking_type,
    })

    logger.info(f"Loading tokenizer: {data_args.tokenizer_name}")
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
    
    log_loss_config(loss_config)

    # Models that support loss_config parameter
    models_with_loss_config = {"weighted_mlm"}
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
    
    # Run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_id = f"{model_args.model_type}_H{model_args.hidden_size}L{model_args.num_hidden_layers}C{model_args.concept_num}"
    if config.token_embedding_dim != config.hidden_size:
        base_id += f"_T{config.token_embedding_dim}"
    if config.concept_position_type != "none":
        base_id += f"_pos{config.concept_position_type}"
    run_identifier = f"{base_id}_{timestamp}"

    setup_run_dirs(training_args, run_identifier)
    # MLM uses HF DataCollator which expects unused columns removed
    training_args.remove_unused_columns = True
    training_args.use_cpu = False

    if training_args.eval_strategy != "steps":
        training_args.eval_steps = None
    if training_args.save_strategy != "steps":
        training_args.save_steps = None

    log_training_config(training_args)

    init_wandb(
        training_args, model, config, data_args, loss_config,
        base_id, run_identifier,
        job_type="mlm-pretraining",
        model_type=model_args.model_type,
        wandb_tags=[model_args.model_type, "mlm-pretraining", data_args.masking_type],
        extra_config={
            "mlm_probability": data_args.mlm_probability,
            "masking_type": data_args.masking_type,
        },
    )
    
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
    
    logger.info("=" * 60)
    logger.info(f"Starting MLM pretraining: {datetime.now()}")
    logger.info("=" * 60)
    trainer.train()

    final_path = os.path.join(training_args.output_dir, run_identifier)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved model to: {final_path}")

    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()
