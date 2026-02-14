"""
Training utilities and helper functions for ConceptEncoder models.
"""
import os
import platform
import torch
from torch.nn import Module
from typing import Tuple, Dict, Any
from transformers import logging
from datetime import datetime

logger = logging.get_logger(__name__)


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
        elif 'concept_token_attn' in name:
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



def setup_distributed():
    """
    Setup for distributed training on multi-GPU single node.
    Returns local rank for the current process.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        
        # Initialize process group if using distributed training (rank != -1)
        # This fixes warnings about "No device id provided via init_process_group"
        if local_rank != -1:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
            
            torch.cuda.set_device(local_rank)
            
        return local_rank
    return -1


def is_main_process():
    """
    Check if this is the main process (local_rank 0).
    Used to avoid duplicate logging/printing in multi-GPU training.
    """
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


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
