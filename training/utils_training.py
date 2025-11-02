"""
Training utilities and helper functions for ConceptEncoder models.
"""
import torch
from torch.nn import Module
from typing import Tuple, Dict, Any, Optional
from datasets import load_dataset


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


def print_model_parameters(model: Module, model_name: str = "Model") -> None:
    """
    Print model parameter counts in a formatted way.
    
    Args:
        model: PyTorch model
        model_name: Name to display in the output
    """
    total_params, trainable_params = count_parameters(model)
    print(f"\n{model_name} Parameters:")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Additional breakdown if there's a difference
    if total_params != trainable_params:
        frozen_params = total_params - trainable_params
        print(f"Frozen parameters: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        print(f"Trainable percentage: {trainable_params/total_params*100:.1f}%")


def get_parameter_breakdown(model: Module) -> Dict[str, Dict[str, int]]:
    """
    Get detailed parameter breakdown by model component.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts by component
    """
    breakdown = {}
    
    # Count parameters by major components
    component_params = {
        'embeddings': 0,
        'encoder': 0,
        'mlm_head': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        if 'embedding' in name:
            component_params['embeddings'] += param_count
        elif 'encoder' in name and 'embedding' not in name:
            component_params['encoder'] += param_count
        elif 'lm_head' in name or 'mlm' in name.lower() or 'concept_weights' in name:
            component_params['mlm_head'] += param_count
        else:
            component_params['other'] += param_count
    
    # Convert to millions for readability
    for component in component_params:
        breakdown[component] = {
            'params': component_params[component],
            'params_m': component_params[component] / 1e6
        }
    
    return breakdown


def format_time(seconds: float) -> str:
    """
    Format seconds into a readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def calculate_training_memory(model: Module, batch_size: int, seq_length: int, 
                            gradient_accumulation_steps: int = 1) -> Dict[str, float]:
    """
    Estimate GPU memory requirements for training.
    
    Args:
        model: PyTorch model
        batch_size: Training batch size
        seq_length: Sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        
    Returns:
        Dictionary with memory estimates in GB
    """
    total_params, _ = count_parameters(model)
    
    # Rough estimates (4 bytes per parameter)
    model_memory = total_params * 4 / (1024**3)  # Model weights
    optimizer_memory = model_memory * 2  # Adam optimizer states (2x model size)
    gradient_memory = model_memory  # Gradients
    
    # Activation memory (very rough estimate)
    # This is highly model-dependent
    effective_batch = batch_size // gradient_accumulation_steps
    activation_memory = effective_batch * seq_length * 1024 * 4 / (1024**3)  # Rough estimate
    
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    return {
        'model_gb': model_memory,
        'optimizer_gb': optimizer_memory,
        'gradients_gb': gradient_memory,
        'activations_gb': activation_memory,
        'total_gb': total_memory
    }



