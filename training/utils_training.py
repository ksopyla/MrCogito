"""
Training utilities and helper functions for ConceptEncoder models.
"""
import logging as std_logging
import os
import platform
import subprocess
import torch
import wandb
from dataclasses import dataclass
from datetime import datetime, timedelta
from torch.nn import Module
from typing import Dict, List, Optional, Tuple, Any
from transformers import logging

logger = logging.get_logger(__name__)


def get_git_info() -> Dict[str, Optional[str]]:
    """
    Return the current git commit hash and nearest tag.

    Adds traceability between WandB training runs and the exact code version.
    Call this once in each training script and include the result in wandb.init config.

    Example usage in a training script:
        from training.utils_training import get_git_info
        git_info = get_git_info()
        wandb.init(..., config={"git_commit": git_info["commit"], **...})

    Returns:
        dict with keys:
          "commit"  : short SHA of HEAD (e.g. "54ee870"), or None if not in a git repo
          "commit_long": full 40-char SHA, or None
          "tag"     : nearest annotated/lightweight tag (e.g. "arch/tsdae-20260221"), or None
          "dirty"   : True if working tree has uncommitted changes, False otherwise
    """
    info: Dict[str, Optional[str]] = {
        "commit": None,
        "commit_long": None,
        "tag": None,
        "dirty": False,
    }
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["commit_long"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        try:
            info["tag"] = subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match", "HEAD"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
        except subprocess.CalledProcessError:
            # No exact tag — fall back to nearest tag + offset
            try:
                info["tag"] = subprocess.check_output(
                    ["git", "describe", "--tags", "--abbrev=4"],
                    stderr=subprocess.DEVNULL, text=True
                ).strip()
            except subprocess.CalledProcessError:
                info["tag"] = None
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["dirty"] = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def count_parameters(model: Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_parameter_breakdown(model: Module) -> Dict[str, Dict[str, int]]:
    """
    Get detailed parameter breakdown by model component for ConceptEncoder models.
    
    Categorizes parameters into:
    - token_embeddings: Token and position embeddings
    - concept_embeddings: Concept token embeddings
    - cross_attention: Cross-attention between concepts and tokens
    - self_attention: Self-attention between concepts
    - feedforward: Feed-forward network layers (Wi, Wo)
    - layer_norm: Layer normalization parameters
    - lm_head: Language modeling head and decoding layers
    - other: Other components (pooler, classifier, gates, etc.)
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts by component
    """
    breakdown = {}
    
    # Count parameters by major components
    component_params = {
        'token_embeddings': 0,
        'concept_embeddings': 0,
        'cross_attention': 0,
        'self_attention': 0,
        'feedforward': 0,
        'layer_norm': 0,
        'lm_head': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        name_lower = name.lower()
        
        # Token embeddings (including position embeddings)
        if 'token_embeddings' in name or 'token_position_embeddings' in name:
            component_params['token_embeddings'] += param_count
        
        # Concept embeddings
        elif 'concept_embeddings' in name:
            component_params['concept_embeddings'] += param_count
        
        # Cross attention (concept-token attention)
        elif 'concept_token_attn' in name or 'bixt_cross_attn' in name:
            component_params['cross_attention'] += param_count
        
        # Self attention (concept-concept attention)
        elif 'concept_self_attn' in name:
            component_params['self_attention'] += param_count
        
        # Feed-forward layers (Wi, Wo matrices) - exclude from LM head and gates
        elif ('wi' in name_lower or 'wo' in name_lower) and 'lm_head' not in name_lower and 'gate' not in name_lower:
            component_params['feedforward'] += param_count
        
        # Layer normalization
        elif 'norm' in name_lower or 'layernorm' in name_lower:
            component_params['layer_norm'] += param_count
        
        # LM head and decoding components
        elif any(x in name_lower for x in [
            'lm_head', 'concept_vocab_projection', 'lm_token_head',
            'concept_to_sequence', 'pre_lm_projection', 'concept_weights'
        ]):
            component_params['lm_head'] += param_count
        
        # Other components (gates, pooler, classifier, temperature, etc.)
        else:
            component_params['other'] += param_count
    
    # Convert to millions for readability and only include non-zero components
    for component, count in component_params.items():
        if count > 0:
            breakdown[component] = {
                'params': count,
                'params_m': count / 1e6
            }
    
    return breakdown



def setup_distributed(timeout_minutes: Optional[int] = None):
    """
    Setup for distributed training on multi-GPU single node.
    Returns local rank for the current process.

    Args:
        timeout_minutes: NCCL collective timeout. Must exceed the longest
            single-rank operation (e.g. first-time dataset tokenisation on
            rank 0 while other ranks wait at the ``main_process_first`` barrier).
            When None, reads the ``DDP_TIMEOUT`` env var (seconds — same knob the
            launchers pass to ``--ddp_timeout``), falling back to 30 min. This PG
            is created BEFORE TrainingArguments parsing, so ``--ddp_timeout``
            alone cannot protect the first barrier (caused a SIGABRT on Odra
            when first-time FineWeb-Edu tokenization took ~61 min).
    """
    if timeout_minutes is None:
        try:
            timeout_minutes = max(1, int(os.environ.get("DDP_TIMEOUT", "1800")) // 60)
        except ValueError:
            timeout_minutes = 30
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", -1))

        if local_rank != -1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl",
                    device_id=torch.device(f"cuda:{local_rank}"),
                    timeout=timedelta(minutes=timeout_minutes),
                )

            torch.cuda.set_device(local_rank)

        return local_rank
    return -1


def is_main_process():
    """
    Check if this is the main process (local_rank 0).
    Used to avoid duplicate logging/printing in multi-GPU training.
    """
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def broadcast_object(obj):
    """Broadcast a picklable object from rank 0 to all ranks.

    Used to agree on values that must be identical across DDP ranks but are
    derived from non-deterministic sources (e.g. a wall-clock ``run_id``).
    Computing such values independently per process lets ranks diverge — e.g.
    a second-resolution timestamp can differ across ranks and fork the output
    directory into ``..._HHMMSS`` / ``..._HHMMSS+1`` with duplicated checkpoints.

    No-op (returns ``obj`` unchanged) when distributed is not initialized
    (single-GPU / CPU runs).
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        holder = [obj]
        torch.distributed.broadcast_object_list(holder, src=0)
        return holder[0]
    return obj


def get_hostname():
    """
    Get hostname in a cross-platform way (works on Windows and Linux).
    """
    return platform.node()


def log_system_info():
    """
    Log system and CUDA information on main process only.
    """
    if not is_main_process():
        return
        
    logger.info("="*60)
    logger.info("System Information")
    logger.info("="*60)
    # Start training at date and time
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Hostname: {get_hostname()}")
    logger.info(f"Python version: {platform.python_version()}")
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  - Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"  - Compute capability: {props.major}.{props.minor}")
    logger.info("="*60)


def log_model_info(model: Module, config: Any = None, model_type: str = None, 
                   model_description: str = None):
    """
    Log model architecture and parameter information on main process only.
    
    Args:
        model: PyTorch model
        config: Model configuration object (optional)
        model_type: String identifier for the model type (optional)
        model_description: Human-readable description of the model (optional)
    """
    if not is_main_process():
        return
    
    logger.info("="*60)
    logger.info("Model Information")
    logger.info("="*60)
    
    if model_type:
        logger.info(f"Model type: {model_type}")
    if model_description:
        logger.info(f"Model description: {model_description}")
    
    # Log model class name
    logger.info(f"Model class: {model.__class__.__name__}")
    
    # Log configuration if provided
    if config:
        logger.info("\nModel Configuration:")
        config_attrs = ['hidden_size', 'token_embedding_dim', 'num_hidden_layers', 
                       'intermediate_size', 'num_attention_heads', 'concept_num', 
                       'concept_position_type', 'vocab_size', 'max_sequence_length']
        
        for attr in config_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                logger.info(f"  {attr.replace('_', ' ').title()}: {value}")
    
    # Get parameter counts
    total_params, trainable_params = count_parameters(model)
    logger.info(f"\nParameter Summary:")
    logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    if total_params != trainable_params:
        frozen_params = total_params - trainable_params
        logger.info(f"  Frozen parameters: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        logger.info(f"  Trainable percentage: {trainable_params/total_params*100:.1f}%")
    
    # Detailed parameter breakdown
    breakdown = get_parameter_breakdown(model)
    if breakdown:
        logger.info("\nParameter breakdown by component:")
        for component, info in breakdown.items():
            if info['params'] > 0:
                logger.info(f"  {component}: {info['params']:,} ({info['params_m']:.2f}M)")
    logger.info("="*60)


# ============================================================================
# Shared logging helpers — called by all training scripts
# ============================================================================

def setup_file_logging(log_dir: Optional[str] = None):
    """
    Add a timestamped file handler to the root logger.
    Useful for remote runs where console output may be lost.
    """
    if not is_main_process():
        return
    log_dir = log_dir or os.environ.get("LOG_DIR", "./Cache/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(
        log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    root = std_logging.getLogger()
    root.setLevel(std_logging.INFO)
    root.handlers.clear()

    formatter = std_logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console = std_logging.StreamHandler()
    console.setLevel(std_logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    fh = std_logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
    fh.setLevel(std_logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logger.info(f"Logging to file: {log_filepath}")


def resolve_dataset_identifier(data_args) -> str:
    """Return the dataset identifier that actually trained the model.

    Mirrors the data-loading priority in train_concept_pretraining.main():
    pretokenized manifest → dataset_mix_recipe → dataset_mix → dataset_name.
    The default ``dataset_name`` (e.g. ``JeanKaddour/minipile``) is misleading
    when a long-context mix is in use, because it is left untouched while the
    real data comes from the mix/manifest.
    """
    manifest = getattr(data_args, "pretokenized_manifest", None)
    if manifest:
        return manifest
    mix_recipe = getattr(data_args, "dataset_mix_recipe", None)
    if mix_recipe:
        return mix_recipe
    mix = getattr(data_args, "dataset_mix", None)
    if mix:
        return mix
    return data_args.dataset_name


def log_data_config(data_args, extra_fields: Optional[Dict[str, Any]] = None):
    """Log standardized 'Data Configuration' section to console."""
    if not is_main_process():
        return
    effective_dataset = resolve_dataset_identifier(data_args)
    logger.info("=" * 60)
    logger.info("Data Configuration")
    logger.info("=" * 60)
    logger.info(f"Dataset: {effective_dataset}")
    if effective_dataset == data_args.dataset_name and getattr(
        data_args, "dataset_name_subset", None
    ):
        logger.info(f"Dataset subset: {data_args.dataset_name_subset}")
    logger.info(f"Tokenizer: {data_args.tokenizer_name}")
    logger.info(f"Max sequence length: {data_args.max_seq_length}")
    logger.info(f"Test size: {data_args.test_size_percent * 100}%")
    if extra_fields:
        for k, v in extra_fields.items():
            logger.info(f"{k}: {v}")


def log_loss_config(loss_config):
    """Log standardized 'Loss Configuration' section to console."""
    if not is_main_process():
        return
    logger.info("=" * 60)
    logger.info("Loss Configuration")
    logger.info("=" * 60)
    logger.info(f"Concept losses: {loss_config.concept_losses or 'none'}")
    logger.info(f"Weighting strategy: {loss_config.weighting_strategy}")
    if loss_config.weighting_strategy == "fixed" and loss_config.is_enabled:
        logger.info(f"Loss weights: {loss_config.loss_weights}")
    if loss_config.warmup_steps > 0:
        logger.info(f"Concept loss warmup: {loss_config.warmup_steps} steps")


def log_training_config(training_args, extra_fields: Optional[Dict[str, Any]] = None):
    """
    Log standardized 'Training Configuration' section to console.
    Only logs derived/computed values; HF Trainer prints the rest.
    """
    if not is_main_process():
        return
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"Run name: {training_args.run_name}")
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    eff_batch = (
        training_args.per_device_train_batch_size
        * device_count
        * training_args.gradient_accumulation_steps
    )
    logger.info(f"Effective batch size: {eff_batch}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    mp = "bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "fp32")
    logger.info(f"Mixed precision: {mp}")
    if extra_fields:
        for k, v in extra_fields.items():
            logger.info(f"{k}: {v}")


def setup_run_dirs(training_args, run_identifier: str):
    """
    Configure output_dir, logging_dir and run_name on *training_args* so that
    every training script produces the same directory layout.
    """
    output_dir = training_args.output_dir or "./outputs"
    if not output_dir.endswith(run_identifier):
        training_args.output_dir = os.path.join(output_dir, run_identifier)

    if not training_args.logging_dir:
        training_args.logging_dir = os.path.join(
            os.path.dirname(training_args.output_dir), "logs", run_identifier,
        )
    elif not training_args.logging_dir.endswith(run_identifier):
        training_args.logging_dir = os.path.join(
            training_args.logging_dir, run_identifier,
        )

    training_args.run_name = run_identifier
    training_args.report_to = ["tensorboard", "wandb"]
    training_args.push_to_hub = False
    training_args.remove_unused_columns = False
    training_args.fp16 = not training_args.bf16


@dataclass(frozen=True)
class WandbRunIdentity:
    """Stable W&B identity facets for training runs."""

    experiment_id: Optional[str]
    model_family: str
    objective_family: str
    architecture_id: str
    group: str
    job_type: str
    tags: List[str]

    def to_config(self) -> Dict[str, Optional[str]]:
        return {
            "experiment_id": self.experiment_id,
            "model_family": self.model_family,
            "objective_family": self.objective_family,
            "architecture_id": self.architecture_id,
            "wandb_group": self.group,
            "wandb_job_type": self.job_type,
        }


def build_perceiver_wandb_identity(
    *,
    decoder_type: str,
    objective_variant: str,
    hidden_size: int,
    num_hidden_layers: int,
    concept_num: int,
    decoder_num_layers: int,
    checkpoint_family: str,
    pretraining_objective: str,
    use_bixt: bool,
    anchor_loss: bool = False,
    experiment_id: Optional[str] = None,
) -> WandbRunIdentity:
    """Derive W&B grouping metadata for the shared perceiver/AR entrypoint.

    The training script hosts multiple research families. W&B group/job_type
    should identify the family + objective, while run names stay unique via a
    timestamp added by the caller.
    """
    # Two human-legible axes surfaced as tags so W&B is scannable without decoding the
    # cryptic family names: how the decoder runs (parallel one-shot vs autoregressive) and
    # what the objective is (reconstruct the input vs generate unseen content).
    if decoder_type == "causal_ar":
        decoder_mode = "autoregressive"
        if objective_variant == "prefix_suffix":
            inferred_experiment = "E02"
            model_family = "concept_ar_prefix"
            objective_family = "prefix_suffix"
            task = "generation"
            job_type = "train_ar_generation_prefix_suffix"
        elif anchor_loss:
            inferred_experiment = "E03"
            model_family = "concept_ar"
            objective_family = "ar_reconstruction_anchor"
            task = "reconstruction"
            job_type = "train_ar_reconstruction_anchor"
        else:
            inferred_experiment = "E01"
            model_family = "concept_ar"
            objective_family = "ar_reconstruction"
            task = "reconstruction"
            job_type = "train_ar_reconstruction"
    else:
        # Parallel position-query Perceiver-IO decoder (denoising autoencoder, not diffusion).
        decoder_mode = "parallel"
        model_family = "perceiver_denoise"
        task = "reconstruction"
        if objective_variant == "reconstruction+contrastive":
            inferred_experiment = None
            objective_family = "reconstruction_contrastive"
            job_type = "train_parallel_reconstruction_contrastive"
        else:
            inferred_experiment = "E04"
            objective_family = "reconstruction"
            job_type = "train_parallel_reconstruction"

    resolved_experiment = experiment_id or inferred_experiment
    architecture_id = (
        f"{model_family}_H{hidden_size}"
        f"L{num_hidden_layers}"
        f"C{concept_num}"
        f"D{decoder_num_layers}"
    )
    group = f"{resolved_experiment}_{architecture_id}" if resolved_experiment else architecture_id
    tags = [
        "train",
        "concept-encoder",
        f"decoder:{decoder_mode}",
        f"task:{task}",
        model_family,
        checkpoint_family,
        decoder_type,
        objective_family,
        objective_variant,
        pretraining_objective,
    ]
    if resolved_experiment:
        tags.append(resolved_experiment)
    if anchor_loss:
        tags.extend(["anchor", "anchor-on"])
    if use_bixt:
        tags.append("bixt")
    tags = list(dict.fromkeys(tags))

    return WandbRunIdentity(
        experiment_id=resolved_experiment,
        model_family=model_family,
        objective_family=objective_family,
        architecture_id=architecture_id,
        group=group,
        job_type=job_type,
        tags=tags,
    )


def init_wandb(
    training_args,
    model: Module,
    config,
    data_args,
    loss_config,
    base_id: str,
    run_identifier: str,
    job_type: str,
    model_type: str,
    wandb_tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """
    Standardized wandb.init with full reproducibility config.

    Builds the wandb config dict from all argument groups and always includes
    ``**vars(training_args)`` so every HF TrainingArgument is captured.
    Tags always include hostname and dataset for easy filtering.
    """
    if not is_main_process():
        return

    total_params, trainable_params = count_parameters(model)
    hostname = get_hostname()

    tags = list(wandb_tags or [])
    identity_config = extra_config or {}
    configured_group = identity_config.get("wandb_group")
    experiment_id = identity_config.get("experiment_id")
    if configured_group and configured_group != base_id:
        raise ValueError(
            f"W&B identity mismatch: base group '{base_id}' differs from config wandb_group "
            f"'{configured_group}'."
        )
    if experiment_id:
        expected_prefix = f"{experiment_id}_"
        if not base_id.startswith(expected_prefix):
            raise ValueError(
                f"W&B identity mismatch: group '{base_id}' does not start with '{expected_prefix}'."
            )
        if experiment_id not in tags:
            raise ValueError(
                f"W&B identity mismatch: experiment tag '{experiment_id}' missing in wandb tags {tags}."
            )

    tags.extend([data_args.dataset_name, hostname])
    if getattr(data_args, "dataset_name_subset", None):
        tags.append(data_args.dataset_name_subset)
    if loss_config.is_enabled:
        tags.append(f"losses:{'+'.join(loss_config.concept_losses)}")

    # Effective dataset: prefer mix recipe / mix / pretokenized manifest over the
    # bare dataset_name (which stays at its default when a mix is in use). This is
    # what actually trained the model; the bare dataset_name is still kept below as
    # a separate config field for completeness. For the TAG, use a short form: a
    # pretokenized manifest PATH is ~90 chars and wandb rejects tags > 64 chars
    # (crashed wandb.init on pretokenized runs, 2026-07-01) — use the manifest basename.
    effective_dataset = resolve_dataset_identifier(data_args)
    dataset_tag = effective_dataset.rsplit("/", 1)[-1] if (
        effective_dataset and "/" in effective_dataset
    ) else effective_dataset
    if dataset_tag and dataset_tag not in tags:
        tags.append(dataset_tag)

    # The real optimizer family (selected by our --optimizer flag). HF's --optim is
    # kept at "adamw_torch_fused" for both arms because HF coerces --optim to its enum
    # and rejects "muon"; without this override the misleading HF value would surface
    # in the config under the "optim" key.
    effective_optim = identity_config.get("optimizer") or getattr(training_args, "optim", None)

    wandb_config: Dict[str, Any] = {
        "model_type": model_type,
        "hidden_size": config.hidden_size,
        "token_embedding_dim": config.token_embedding_dim,
        "num_hidden_layers": config.num_hidden_layers,
        "concept_num": config.concept_num,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "max_sequence_length": config.max_sequence_length,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "concept_losses": loss_config.concept_losses,
        "loss_weighting": loss_config.weighting_strategy,
        "dataset_name": effective_dataset,
        "dataset_name_hf_default": data_args.dataset_name,
        "tokenizer_name": data_args.tokenizer_name,
        "max_seq_length": data_args.max_seq_length,
        "compare_family": identity_config.get("model_family", model_type),
        "compare_objective": identity_config.get("objective_family"),
        "compare_architecture": identity_config.get("architecture_id"),
        "compare_tokenizer": data_args.tokenizer_name,
        "compare_params_m": round(total_params / 1_000_000),
        **{f"git_{k}": v for k, v in get_git_info().items()},
        **{k: v for k, v in vars(training_args).items() if not k.startswith("_")},
    }
    if effective_optim:
        wandb_config["optim"] = effective_optim
    if extra_config:
        wandb_config.update(extra_config)

    # W&B tags are capped at 64 chars (a longer tag raises a pydantic ValidationError
    # in wandb.init). Clamp as a safety net so no future long tag crashes the run.
    tags = [str(t)[:64] for t in tags if t]

    wandb.init(
        project="MrCogito",
        id=run_identifier,
        name=training_args.run_name,
        job_type=job_type,
        config=wandb_config,
        tags=tags,
        group=base_id,
        sync_tensorboard=True,
        notes=notes or f"Model: {model_type}, Dataset: {effective_dataset}",
    )
    logger.info(f"W&B run: {wandb.run.id} / {wandb.run.name}")
