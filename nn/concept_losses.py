"""
Concept Representation Loss Functions for ConceptEncoder models.

⚠️  DEPRECATION NOTICE:
    This module is deprecated in favor of `nn.loss_manager`.
    The new module provides a cleaner, more extensible architecture.
    
    For new code, use:
        from nn.loss_manager import LossManager, LossConfig
    
    This module is kept for backward compatibility and will be removed
    in a future version.

This module provides various loss functions designed to prevent dimensional collapse
and encourage diverse, well-distributed concept representations.

Loss Types:
-----------
1. orthogonality: Strict orthogonality between concept vectors (cosine sim = 0)
2. soft_orthogonality: Allows small correlations below a threshold
3. uniformity: Pushes concepts apart on hypersphere (softer than orthogonality)
4. vicreg: VICReg-style variance + covariance regularization
5. combined: Combination of variance + uniformity terms
6. none: No concept regularization loss

References:
-----------
- VICReg: Bardes et al., 2021 (https://hf.co/papers/2105.04906)
- Uniformity: Wang & Isola, 2020 "Understanding Contrastive Representation Learning"
- T-REGS: Mordacq et al., 2025 (https://hf.co/papers/2510.23484)
- Rethinking Uniformity: Fang et al., 2024 (https://hf.co/papers/2403.00642)
"""

import warnings

# Deprecation warning for users who import from this module
warnings.warn(
    "nn.concept_losses is deprecated. Use nn.loss_manager instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

from typing import Dict, Optional, Literal, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Loss Type Registry
# ============================================================================

ConceptLossType = Literal[
    "none",
    "orthogonality", 
    "soft_orthogonality",
    "uniformity", 
    "vicreg",
    "combined"
]

AVAILABLE_LOSSES = [
    "none",
    "orthogonality",
    "soft_orthogonality", 
    "uniformity",
    "vicreg",
    "combined"
]


# ============================================================================
# Individual Loss Functions
# ============================================================================

def compute_orthogonality_loss(concept_repr: torch.Tensor) -> torch.Tensor:
    """
    Encourage concept vectors to be strictly orthogonal to each other.
    
    This is the strongest constraint - forces cosine similarity between 
    different concepts to be exactly 0.
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        
    Returns:
        orthogonality_loss: scalar tensor
        
    Note:
        - Maximum number of perfectly orthogonal vectors = hidden_size
        - If concept_num > hidden_size, perfect orthogonality is impossible
    """
    # Normalize concepts to unit vectors
    concept_norm = F.normalize(concept_repr, p=2, dim=-1)  # [B, C, H]
    
    # Compute concept similarity matrix [B, C, H] @ [B, H, C] = [B, C, C]
    concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))  # [B, C, C]
    
    # Create identity matrix (target: concepts should be orthogonal)
    batch_size, concept_num = concept_sim.shape[:2]
    eye = torch.eye(concept_num, device=concept_sim.device).unsqueeze(0)
    eye = eye.expand(batch_size, -1, -1)
    
    # Compute loss: penalize non-diagonal elements (correlations)
    off_diagonal_mask = 1.0 - eye
    orthogonality_loss = (concept_sim * off_diagonal_mask).pow(2).sum() / (
        batch_size * concept_num * (concept_num - 1)
    )
    
    return orthogonality_loss


def compute_soft_orthogonality_loss(
    concept_repr: torch.Tensor, 
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Soft orthogonality loss that allows small correlations below threshold.
    
    This is a more flexible version that doesn't penalize small correlations,
    which may be natural for semantically related concepts.
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        threshold: Correlation values below this are not penalized (default: 0.1)
        
    Returns:
        soft_orthogonality_loss: scalar tensor
    """
    # Normalize concepts to unit vectors
    concept_norm = F.normalize(concept_repr, p=2, dim=-1)  # [B, C, H]
    
    # Compute concept similarity matrix
    concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))  # [B, C, C]
    
    batch_size, concept_num = concept_sim.shape[:2]
    eye = torch.eye(concept_num, device=concept_sim.device).unsqueeze(0)
    
    # Get off-diagonal elements
    off_diagonal_mask = 1.0 - eye
    off_diagonal_sim = (concept_sim * off_diagonal_mask).abs()
    
    # Only penalize correlations above threshold
    penalized_sim = F.relu(off_diagonal_sim - threshold)
    
    soft_ortho_loss = penalized_sim.pow(2).sum() / (
        batch_size * concept_num * (concept_num - 1)
    )
    
    return soft_ortho_loss


def compute_uniformity_loss(
    concept_repr: torch.Tensor, 
    temperature: float = 2.0
) -> torch.Tensor:
    """
    Uniformity loss on hypersphere - softer than strict orthogonality.
    
    Pushes concept representations to be uniformly distributed on the 
    unit hypersphere using a Gaussian kernel. This is inspired by 
    contrastive learning uniformity objectives.
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        temperature: Controls the spread of the Gaussian kernel (default: 2.0)
            - Higher values = weaker push apart
            - Lower values = stronger push apart
            
    Returns:
        uniformity_loss: scalar tensor
        
    Reference:
        Wang & Isola, "Understanding Contrastive Representation Learning"
    """
    # Normalize concepts to unit vectors
    concept_norm = F.normalize(concept_repr, p=2, dim=-1)  # [B, C, H]
    
    # Compute pairwise squared distances on the hypersphere
    # For unit vectors: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b = 2 - 2*a·b
    concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))  # [B, C, C]
    sq_dist = 2.0 - 2.0 * concept_sim  # [B, C, C]
    
    # Gaussian kernel: exp(-t * ||a - b||^2)
    # We want to minimize this (push concepts apart)
    batch_size, concept_num = concept_repr.shape[:2]
    eye = torch.eye(concept_num, device=concept_repr.device).unsqueeze(0)
    off_diag_mask = 1.0 - eye
    
    # Average over off-diagonal pairs
    uniformity_loss = (torch.exp(-temperature * sq_dist) * off_diag_mask).sum() / (
        batch_size * concept_num * (concept_num - 1)
    )
    
    return uniformity_loss


def compute_variance_loss(
    concept_repr: torch.Tensor, 
    eps: float = 1e-4,
    target_std: float = 1.0
) -> torch.Tensor:
    """
    Variance loss to prevent embedding collapse along dimensions.
    
    Encourages each hidden dimension to have variance above a target value
    across the batch and concepts. This prevents all concepts from collapsing
    to similar values.
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        eps: Small constant for numerical stability
        target_std: Target standard deviation (default: 1.0)
        
    Returns:
        variance_loss: scalar tensor
        
    Reference:
        VICReg: Bardes et al., 2021
    """
    batch_size, concept_num, hidden_size = concept_repr.shape
    
    # Flatten batch and concept dimensions
    concept_flat = concept_repr.reshape(-1, hidden_size)  # [B*C, H]
    
    # Compute standard deviation per dimension
    std = torch.sqrt(concept_flat.var(dim=0) + eps)  # [H]
    
    # Hinge loss: penalize dimensions with std below target
    variance_loss = F.relu(target_std - std).mean()
    
    return variance_loss


def compute_covariance_loss(concept_repr: torch.Tensor) -> torch.Tensor:
    """
    Covariance loss to decorrelate hidden dimensions.
    
    Encourages different dimensions of the representation to be uncorrelated,
    preventing redundant encoding of information.
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        
    Returns:
        covariance_loss: scalar tensor
        
    Reference:
        VICReg: Bardes et al., 2021
    """
    batch_size, concept_num, hidden_size = concept_repr.shape
    
    # Flatten batch and concept dimensions
    concept_flat = concept_repr.reshape(-1, hidden_size)  # [B*C, H]
    
    # Center the representations
    concept_centered = concept_flat - concept_flat.mean(dim=0)
    
    # Compute covariance matrix [H, H]
    n_samples = concept_centered.shape[0]
    cov = (concept_centered.T @ concept_centered) / (n_samples - 1)
    
    # Zero out diagonal (we only penalize off-diagonal)
    eye = torch.eye(hidden_size, device=cov.device)
    off_diag_cov = cov * (1.0 - eye)
    
    # Frobenius norm of off-diagonal elements
    covariance_loss = off_diag_cov.pow(2).sum() / hidden_size
    
    return covariance_loss


def compute_vicreg_loss(
    concept_repr: torch.Tensor,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    VICReg-style loss combining variance and covariance terms.
    
    This prevents both:
    - Variance collapse: All representations shrinking to small values
    - Dimensional collapse: Representations using only a subspace
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        variance_weight: Weight for variance term
        covariance_weight: Weight for covariance term
        eps: Small constant for numerical stability
        
    Returns:
        vicreg_loss: scalar tensor
        
    Reference:
        VICReg: Bardes et al., 2021 (https://hf.co/papers/2105.04906)
    """
    var_loss = compute_variance_loss(concept_repr, eps=eps)
    cov_loss = compute_covariance_loss(concept_repr)
    
    return variance_weight * var_loss + covariance_weight * cov_loss


def compute_combined_loss(
    concept_repr: torch.Tensor,
    variance_weight: float = 1.0,
    uniformity_weight: float = 1.0,
    temperature: float = 2.0,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Combined loss: Variance + Uniformity (recommended for most cases).
    
    This provides a balanced approach that:
    - Prevents variance collapse (via variance term)
    - Encourages concept diversity (via uniformity on hypersphere)
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        variance_weight: Weight for variance term
        uniformity_weight: Weight for uniformity term
        temperature: Temperature for uniformity loss
        eps: Small constant for numerical stability
        
    Returns:
        combined_loss: scalar tensor
    """
    var_loss = compute_variance_loss(concept_repr, eps=eps)
    uni_loss = compute_uniformity_loss(concept_repr, temperature=temperature)
    
    return variance_weight * var_loss + uniformity_weight * uni_loss


# ============================================================================
# Loss Configuration
# ============================================================================

@dataclass
class ConceptLossConfig:
    """
    Configuration for concept representation loss during training.
    
    This is separate from model architecture config (ConceptEncoderConfig)
    following the Single Responsibility Principle:
    - ConceptEncoderConfig = what the model IS (architecture, saved with model)
    - ConceptLossConfig = how the model is TRAINED (behavior, not saved with model)
    
    Attributes:
        loss_type: Type of concept loss to use.
            Options: "none", "orthogonality", "soft_orthogonality", 
                     "uniformity", "vicreg", "combined"
        loss_weight: Fixed weight for concept loss when not using learned weights.
        use_learned_loss_weights: If True, use Kendall & Gal uncertainty weighting
            which learns the optimal balance between task loss and concept loss.
        soft_ortho_threshold: Threshold for soft_orthogonality loss.
            Correlations below this value are not penalized.
        uniformity_temperature: Temperature for uniformity/combined loss.
            Lower values = stronger push apart, higher = weaker.
        variance_weight: Weight for variance component in vicreg/combined.
        covariance_weight: Weight for covariance component in vicreg.
        uniformity_weight: Weight for uniformity component in combined.
        eps: Small constant for numerical stability.
        
    Example:
        >>> # For ablation: no concept loss
        >>> config = ConceptLossConfig(loss_type="none")
        
        >>> # For ablation: soft orthogonality with higher threshold
        >>> config = ConceptLossConfig(
        ...     loss_type="soft_orthogonality",
        ...     soft_ortho_threshold=0.2,
        ...     use_learned_loss_weights=True
        ... )
        
        >>> # Recommended: combined with learned weights
        >>> config = ConceptLossConfig(
        ...     loss_type="combined",
        ...     use_learned_loss_weights=True
        ... )
    """
    # Loss type selection
    loss_type: ConceptLossType = "orthogonality"
    
    # Loss weighting strategy
    loss_weight: float = 1.0  # Fixed weight (used if use_learned_loss_weights=False)
    use_learned_loss_weights: bool = True  # Kendall & Gal uncertainty weighting
    
    # Loss-specific parameters
    soft_ortho_threshold: float = 0.1
    uniformity_temperature: float = 2.0
    variance_weight: float = 1.0
    covariance_weight: float = 1.0
    uniformity_weight: float = 1.0
    eps: float = 1e-4
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.loss_type not in AVAILABLE_LOSSES:
            raise ValueError(
                f"Unknown loss_type: {self.loss_type}. "
                f"Available: {AVAILABLE_LOSSES}"
            )
    
    @property
    def is_enabled(self) -> bool:
        """Check if concept loss is enabled."""
        return self.loss_type != "none"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            "loss_type": self.loss_type,
            "loss_weight": self.loss_weight,
            "use_learned_loss_weights": self.use_learned_loss_weights,
            "soft_ortho_threshold": self.soft_ortho_threshold,
            "uniformity_temperature": self.uniformity_temperature,
            "variance_weight": self.variance_weight,
            "covariance_weight": self.covariance_weight,
            "uniformity_weight": self.uniformity_weight,
            "eps": self.eps
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ConceptLossConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def disabled(cls) -> "ConceptLossConfig":
        """Factory method for disabled concept loss (for inference/baseline)."""
        return cls(loss_type="none")
    
    @classmethod
    def default(cls) -> "ConceptLossConfig":
        """Factory method for default (orthogonality with learned weights)."""
        return cls(loss_type="orthogonality", use_learned_loss_weights=True)
    
    @classmethod
    def recommended(cls) -> "ConceptLossConfig":
        """Factory method for recommended config (combined with learned weights)."""
        return cls(loss_type="combined", use_learned_loss_weights=True)


def get_concept_loss_fn(
    loss_type: ConceptLossType,
    config: Optional[ConceptLossConfig] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get the concept loss function based on type.
    
    Args:
        loss_type: Type of loss function to return
        config: Optional configuration for loss parameters
        
    Returns:
        A callable that takes concept_repr and returns a loss scalar
        
    Example:
        >>> config = ConceptLossConfig(loss_type="uniformity", uniformity_temperature=3.0)
        >>> loss_fn = get_concept_loss_fn("uniformity", config)
        >>> loss = loss_fn(concept_repr)
    """
    if config is None:
        config = ConceptLossConfig(loss_type=loss_type)
    
    if loss_type == "none":
        return lambda x: torch.tensor(0.0, device=x.device, requires_grad=False)
    
    elif loss_type == "orthogonality":
        return compute_orthogonality_loss
    
    elif loss_type == "soft_orthogonality":
        return lambda x: compute_soft_orthogonality_loss(
            x, threshold=config.soft_ortho_threshold
        )
    
    elif loss_type == "uniformity":
        return lambda x: compute_uniformity_loss(
            x, temperature=config.uniformity_temperature
        )
    
    elif loss_type == "vicreg":
        return lambda x: compute_vicreg_loss(
            x,
            variance_weight=config.variance_weight,
            covariance_weight=config.covariance_weight,
            eps=config.eps
        )
    
    elif loss_type == "combined":
        return lambda x: compute_combined_loss(
            x,
            variance_weight=config.variance_weight,
            uniformity_weight=config.uniformity_weight,
            temperature=config.uniformity_temperature,
            eps=config.eps
        )
    
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {AVAILABLE_LOSSES}"
        )


# ============================================================================
# Monitoring Metrics
# ============================================================================

@torch.no_grad()
def compute_concept_metrics(concept_repr: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics to monitor concept space utilization during training.
    
    These metrics help diagnose:
    - Dimensional collapse (low effective rank)
    - Concept correlation (high max/mean correlation)
    - Representation spread (variance metrics)
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        
    Returns:
        Dictionary with metrics:
        - max_concept_correlation: Maximum absolute correlation between concepts
        - mean_concept_correlation: Mean absolute correlation between concepts
        - effective_rank: Effective rank of concept representations
        - mean_concept_norm: Average L2 norm of concept vectors
        - std_concept_norm: Std of L2 norms (uniformity indicator)
        - dimension_variance_mean: Mean variance across hidden dimensions
        - dimension_variance_min: Minimum variance (collapse indicator)
    """
    batch_size, concept_num, hidden_size = concept_repr.shape
    
    # Normalize for correlation computation
    concept_norm = F.normalize(concept_repr, p=2, dim=-1)
    concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))
    
    # Remove diagonal
    eye = torch.eye(concept_num, device=concept_sim.device).unsqueeze(0)
    off_diag = concept_sim * (1.0 - eye)
    
    # Correlation metrics
    max_correlation = off_diag.abs().max().item()
    mean_correlation = off_diag.abs().sum().item() / (batch_size * concept_num * (concept_num - 1))
    
    # Effective rank (nuclear norm / spectral norm) - averaged over batch
    concept_mean = concept_repr.mean(dim=0)  # [C, H]
    try:
        U, S, V = torch.svd(concept_mean)
        effective_rank = (S.sum() / S.max()).item()
    except RuntimeError:
        # SVD may fail for degenerate cases
        effective_rank = float('nan')
    
    # Norm statistics
    concept_norms = concept_repr.norm(dim=-1)  # [B, C]
    mean_norm = concept_norms.mean().item()
    std_norm = concept_norms.std().item()
    
    # Dimension variance (collapse indicator)
    concept_flat = concept_repr.reshape(-1, hidden_size)
    dim_variance = concept_flat.var(dim=0)  # [H]
    
    return {
        "max_concept_correlation": max_correlation,
        "mean_concept_correlation": mean_correlation,
        "effective_rank": effective_rank,
        "mean_concept_norm": mean_norm,
        "std_concept_norm": std_norm,
        "dimension_variance_mean": dim_variance.mean().item(),
        "dimension_variance_min": dim_variance.min().item(),
    }


def log_concept_metrics(
    concept_repr: torch.Tensor,
    step: int,
    prefix: str = "concept",
    logger = None
) -> Dict[str, float]:
    """
    Compute and optionally log concept metrics.
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        step: Training step number
        prefix: Prefix for metric names
        logger: Optional logger (e.g., wandb, tensorboard writer)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = compute_concept_metrics(concept_repr)
    
    # Add prefix to keys
    prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    if logger is not None:
        # Assume logger has a log method (like wandb)
        if hasattr(logger, 'log'):
            logger.log(prefixed_metrics, step=step)
    
    return prefixed_metrics


# ============================================================================
# Utility Functions
# ============================================================================

def check_concept_loss_feasibility(
    concept_num: int, 
    hidden_size: int,
    loss_type: ConceptLossType
) -> Optional[str]:
    """
    Check if the chosen loss type is feasible given concept_num and hidden_size.
    
    Args:
        concept_num: Number of concept vectors
        hidden_size: Dimension of hidden representations
        loss_type: Type of concept loss
        
    Returns:
        Warning message if there's an issue, None otherwise
    """
    if loss_type in ["orthogonality", "soft_orthogonality"]:
        if concept_num > hidden_size:
            return (
                f"WARNING: concept_num ({concept_num}) > hidden_size ({hidden_size}). "
                f"Perfect orthogonality is mathematically impossible with {loss_type} loss. "
                f"Consider using 'uniformity' or 'combined' loss instead, or reduce concept_num."
            )
    return None

