"""
Loss Management System for ConceptEncoder Models.

This module provides a clean, extensible architecture for managing losses
in ConceptEncoder models. It follows SOLID principles and design patterns
to enable easy experimentation with different loss combinations.

Design Patterns Used:
- Strategy Pattern: For loss functions and weighting strategies
- Composite Pattern: For combining multiple losses
- Factory Pattern: For creating loss components from configuration

Key Benefits:
- Add new losses without modifying model classes
- Easy experimentation with different loss combinations
- Clean separation between model architecture and training behavior
- Support for static, learnable, and uncertainty-based weighting

Usage:
    >>> from nn.loss_manager import LossManager, LossConfig
    >>> 
    >>> # Configuration for MLM + orthogonality loss with learnable weights
    >>> config = LossConfig(
    ...     concept_losses=["orthogonality"],
    ...     weighting_strategy="learnable",
    ...     loss_weights={"orthogonality": 0.1}
    ... )
    >>> 
    >>> loss_manager = LossManager(config)
    >>> total_loss = loss_manager.compute_total_loss(
    ...     task_loss=mlm_loss,
    ...     concept_repr=concept_repr
    ... )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Union, Literal, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Type Definitions
# ============================================================================

ConceptLossType = Literal[
    "none",
    "orthogonality",
    "soft_orthogonality", 
    "uniformity",
    "vicreg",
    "variance",
    "covariance",
    "combined"
]

WeightingStrategyType = Literal[
    "fixed",      # Static weights (e.g., 0.8 * mlm + 0.2 * ortho)
    "learnable",  # Simple learnable weights (nn.Parameter)
    "kendall_gal" # Uncertainty-based (Kendall & Gal, 2018)
]


# ============================================================================
# Abstract Base Classes (Strategy Pattern)
# ============================================================================

class ConceptLossComponent(ABC):
    """
    Abstract base class for concept loss functions.
    
    Implement this interface to add new concept losses without
    modifying any existing code (Open/Closed Principle).
    
    Example:
        >>> class MyNewLoss(ConceptLossComponent):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_new_loss"
        ...     
        ...     def compute(self, concept_repr: torch.Tensor) -> torch.Tensor:
        ...         return my_loss_computation(concept_repr)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this loss component."""
        pass
    
    @abstractmethod
    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the loss value.
        
        Args:
            concept_repr: [batch_size, concept_num, hidden_size]
            **kwargs: Additional parameters (e.g., threshold, temperature)
            
        Returns:
            Scalar loss tensor
        """
        pass
    
    def validate_input(self, concept_repr: torch.Tensor) -> None:
        """Optional validation of input tensor."""
        if concept_repr.dim() != 3:
            raise ValueError(
                f"{self.name} expects 3D tensor [batch, concepts, hidden], "
                f"got {concept_repr.dim()}D"
            )


class WeightingStrategy(ABC, nn.Module):
    """
    Abstract base class for loss weighting strategies.
    
    Implement this interface to add new weighting schemes without
    modifying any existing code.
    
    The strategy takes a dictionary of losses and returns a weighted total.
    """
    
    @abstractmethod
    def forward(
        self, 
        losses: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute weighted sum of losses.
        
        Args:
            losses: Dictionary mapping loss names to loss values
            step: Optional training step for scheduling
            
        Returns:
            Weighted total loss
        """
        pass
    
    def get_weights_for_logging(self) -> Dict[str, float]:
        """Return current weights for logging/monitoring."""
        return {}


# ============================================================================
# Concrete Loss Components (Strategy Implementations)
# ============================================================================

class OrthogonalityLoss(ConceptLossComponent):
    """
    Strict orthogonality between concept vectors (cosine sim = 0).
    
    This is the strongest constraint - forces different concepts
    to be completely uncorrelated.
    """
    
    @property
    def name(self) -> str:
        return "orthogonality"
    
    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize concepts to unit vectors
        concept_norm = F.normalize(concept_repr, p=2, dim=-1)  # [B, C, H]
        
        # Compute similarity matrix
        concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))  # [B, C, C]
        
        # Create identity mask
        batch_size, concept_num = concept_sim.shape[:2]
        eye = torch.eye(concept_num, device=concept_sim.device).unsqueeze(0)
        eye = eye.expand(batch_size, -1, -1)
        
        # Penalize off-diagonal elements
        off_diagonal_mask = 1.0 - eye
        orthogonality_loss = (concept_sim * off_diagonal_mask).pow(2).sum() / (
            batch_size * concept_num * (concept_num - 1)
        )
        
        return orthogonality_loss


class SoftOrthogonalityLoss(ConceptLossComponent):
    """
    Soft orthogonality that allows small correlations below threshold.
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "soft_orthogonality"
    
    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        threshold = kwargs.get("threshold", self.threshold)
        
        concept_norm = F.normalize(concept_repr, p=2, dim=-1)
        concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))
        
        batch_size, concept_num = concept_sim.shape[:2]
        eye = torch.eye(concept_num, device=concept_sim.device).unsqueeze(0)
        
        off_diagonal_mask = 1.0 - eye
        off_diagonal_sim = (concept_sim * off_diagonal_mask).abs()
        
        # Only penalize correlations above threshold
        penalized_sim = F.relu(off_diagonal_sim - threshold)
        
        return penalized_sim.pow(2).sum() / (
            batch_size * concept_num * (concept_num - 1)
        )


class UniformityLoss(ConceptLossComponent):
    """
    Pushes concept representations to be uniformly distributed on hypersphere.
    
    Softer than strict orthogonality, inspired by contrastive learning.
    """
    
    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature
    
    @property
    def name(self) -> str:
        return "uniformity"
    
    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        temperature = kwargs.get("temperature", self.temperature)
        
        concept_norm = F.normalize(concept_repr, p=2, dim=-1)
        concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))
        
        # Squared distance on hypersphere
        sq_dist = 2.0 - 2.0 * concept_sim
        
        batch_size, concept_num = concept_repr.shape[:2]
        eye = torch.eye(concept_num, device=concept_repr.device).unsqueeze(0)
        off_diag_mask = 1.0 - eye
        
        # Gaussian kernel - minimize to push concepts apart
        uniformity_loss = (torch.exp(-temperature * sq_dist) * off_diag_mask).sum() / (
            batch_size * concept_num * (concept_num - 1)
        )
        
        return uniformity_loss


class VarianceLoss(ConceptLossComponent):
    """
    Prevents embedding collapse along dimensions (VICReg-style).
    """
    
    def __init__(self, target_std: float = 1.0, eps: float = 1e-4):
        self.target_std = target_std
        self.eps = eps
    
    @property
    def name(self) -> str:
        return "variance"
    
    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        target_std = kwargs.get("target_std", self.target_std)
        eps = kwargs.get("eps", self.eps)
        
        batch_size, concept_num, hidden_size = concept_repr.shape
        concept_flat = concept_repr.reshape(-1, hidden_size)
        
        std = torch.sqrt(concept_flat.var(dim=0) + eps)
        variance_loss = F.relu(target_std - std).mean()
        
        return variance_loss


class CovarianceLoss(ConceptLossComponent):
    """
    Decorrelates hidden dimensions (VICReg-style).
    """
    
    @property
    def name(self) -> str:
        return "covariance"
    
    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, concept_num, hidden_size = concept_repr.shape
        concept_flat = concept_repr.reshape(-1, hidden_size)
        
        # Center the representations
        concept_centered = concept_flat - concept_flat.mean(dim=0)
        
        # Compute covariance matrix
        n_samples = concept_centered.shape[0]
        cov = (concept_centered.T @ concept_centered) / (n_samples - 1)
        
        # Penalize off-diagonal elements
        eye = torch.eye(hidden_size, device=cov.device)
        off_diag_cov = cov * (1.0 - eye)
        
        return off_diag_cov.pow(2).sum() / hidden_size


class VICRegLoss(ConceptLossComponent):
    """
    Combined variance + covariance regularization (VICReg, Bardes 2021).
    """
    
    def __init__(
        self, 
        variance_weight: float = 1.0, 
        covariance_weight: float = 1.0,
        eps: float = 1e-4
    ):
        self.variance_loss = VarianceLoss(eps=eps)
        self.covariance_loss = CovarianceLoss()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
    
    @property
    def name(self) -> str:
        return "vicreg"
    
    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        var_w = kwargs.get("variance_weight", self.variance_weight)
        cov_w = kwargs.get("covariance_weight", self.covariance_weight)
        
        var_loss = self.variance_loss.compute(concept_repr, **kwargs)
        cov_loss = self.covariance_loss.compute(concept_repr, **kwargs)
        
        return var_w * var_loss + cov_w * cov_loss


class TREGSMSTLoss(ConceptLossComponent):
    """
    MST-based uniformity regularization (T-REGS, Mordacq et al., 2025).

    Approximates the Minimum Spanning Tree length using nearest-neighbor
    distances between concept vectors. Maximizing MST length forces concepts
    to spread uniformly through space — it simultaneously prevents dimensional
    collapse AND promotes uniformity, two properties that VICReg variance
    metrics can miss.

    Reference:
        "T-REGS: Minimum Spanning Tree Regularization for Self-Supervised
        Learning", Mordacq et al., 2025 — https://hf.co/papers/2510.23484
    """

    @property
    def name(self) -> str:
        return "t_regs_mst"

    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        # concept_repr: [B, C, H]
        # Normalize to unit sphere so distances are geometry-preserving
        concept_norm = F.normalize(concept_repr, p=2, dim=-1)  # [B, C, H]

        # Pairwise L2 distances: [B, C, C]
        distances = torch.cdist(concept_norm, concept_norm, p=2)

        # Mask self-distances with large value so they don't become the minimum
        eye_mask = torch.eye(
            concept_repr.size(1), device=concept_repr.device
        ).unsqueeze(0) * 1e9
        distances = distances + eye_mask

        # Nearest-neighbor distance for each concept — O(C²) per batch item
        nn_distances = distances.min(dim=-1).values  # [B, C]

        # Maximize total NN distance (proxy for MST length) → minimise its negative
        return -nn_distances.mean()


class CombinedLoss(ConceptLossComponent):
    """
    Variance + Uniformity combination (recommended for most cases).
    """
    
    def __init__(
        self,
        variance_weight: float = 1.0,
        uniformity_weight: float = 1.0,
        temperature: float = 2.0,
        eps: float = 1e-4
    ):
        self.variance_loss = VarianceLoss(eps=eps)
        self.uniformity_loss = UniformityLoss(temperature=temperature)
        self.variance_weight = variance_weight
        self.uniformity_weight = uniformity_weight
    
    @property
    def name(self) -> str:
        return "combined"
    
    def compute(self, concept_repr: torch.Tensor, **kwargs) -> torch.Tensor:
        var_w = kwargs.get("variance_weight", self.variance_weight)
        uni_w = kwargs.get("uniformity_weight", self.uniformity_weight)
        
        var_loss = self.variance_loss.compute(concept_repr, **kwargs)
        uni_loss = self.uniformity_loss.compute(concept_repr, **kwargs)
        
        return var_w * var_loss + uni_w * uni_loss


# ============================================================================
# Loss Component Registry (Factory Pattern)
# ============================================================================

LOSS_REGISTRY: Dict[str, type] = {
    "orthogonality": OrthogonalityLoss,
    "soft_orthogonality": SoftOrthogonalityLoss,
    "uniformity": UniformityLoss,
    "variance": VarianceLoss,
    "covariance": CovarianceLoss,
    "vicreg": VICRegLoss,
    "combined": CombinedLoss,
    "t_regs_mst": TREGSMSTLoss,
}


def register_loss(name: str, loss_class: type) -> None:
    """
    Register a new loss component.
    
    This allows adding new losses without modifying this file.
    
    Example:
        >>> class MyCustomLoss(ConceptLossComponent):
        ...     @property
        ...     def name(self): return "my_custom"
        ...     def compute(self, x): return x.sum()
        >>> 
        >>> register_loss("my_custom", MyCustomLoss)
    """
    if not issubclass(loss_class, ConceptLossComponent):
        raise TypeError(f"{loss_class} must inherit from ConceptLossComponent")
    LOSS_REGISTRY[name] = loss_class


def get_available_losses() -> List[str]:
    """Return list of all registered loss names."""
    return list(LOSS_REGISTRY.keys())


def create_loss_component(
    name: str, 
    **kwargs
) -> ConceptLossComponent:
    """
    Factory function to create loss components.
    
    Args:
        name: Loss type name (must be registered)
        **kwargs: Parameters passed to loss constructor
        
    Returns:
        Configured loss component
        
    Raises:
        ValueError: If loss name is not registered
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss type: '{name}'. "
            f"Available: {get_available_losses()}"
        )
    
    loss_class = LOSS_REGISTRY[name]
    return loss_class(**kwargs)


# ============================================================================
# Weighting Strategies
# ============================================================================

class FixedWeighting(WeightingStrategy):
    """
    Static weighted sum of losses.
    
    Example:
        total = 0.8 * task_loss + 0.15 * ortho_loss + 0.05 * uniformity_loss
    """
    
    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights: Dictionary mapping loss names to their weights
                     Must include 'task' for the main task loss
        """
        super().__init__()
        self.weights = weights
        
        # Validate task weight is present
        if "task" not in weights:
            weights["task"] = 1.0
    
    def forward(
        self, 
        losses: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        for name, loss in losses.items():
            weight = self.weights.get(name, 0.0)
            total = total + weight * loss
        
        return total
    
    def get_weights_for_logging(self) -> Dict[str, float]:
        return {f"weight/{k}": v for k, v in self.weights.items()}


class LearnableWeighting(WeightingStrategy):
    """
    Simple learnable weights using softmax normalization.
    
    Weights are learned during training to find optimal balance.
    Uses softmax to ensure weights sum to 1.
    """
    
    def __init__(self, loss_names: List[str], init_value: float = 0.0):
        """
        Args:
            loss_names: Names of all losses (including 'task')
            init_value: Initial value for raw weights (before softmax)
        """
        super().__init__()
        self.loss_names = ["task"] + [n for n in loss_names if n != "task"]
        self.num_losses = len(self.loss_names)
        
        # Raw weights (before softmax)
        self.raw_weights = nn.Parameter(
            torch.full((self.num_losses,), init_value)
        )
    
    def forward(
        self, 
        losses: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> torch.Tensor:
        # Get normalized weights
        weights = F.softmax(self.raw_weights, dim=0)
        
        total = torch.tensor(0.0, device=self.raw_weights.device)
        
        for i, name in enumerate(self.loss_names):
            if name in losses:
                total = total + weights[i] * losses[name]
        
        return total
    
    def get_weights_for_logging(self) -> Dict[str, float]:
        weights = F.softmax(self.raw_weights, dim=0)
        return {
            f"weight/{name}": weights[i].item() 
            for i, name in enumerate(self.loss_names)
        }


class KendallGalWeighting(WeightingStrategy):
    """
    Uncertainty-based weighting (Kendall & Gal, CVPR 2018).
    
    Learns task-dependent uncertainty (homoscedastic uncertainty)
    to automatically balance multiple losses.
    
    The formula for each loss i:
        L_weighted = (1 / (2 * sigma_i^2)) * L_i + log(sigma_i)
                   = 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i
    
    Where log_var_i = log(sigma_i^2) is the learned parameter.
    """
    
    def __init__(self, loss_names: List[str], init_log_var: float = 0.0):
        """
        Args:
            loss_names: Names of all losses (including 'task')
            init_log_var: Initial value for log variance (0.0 = weight ~1.0)
        """
        super().__init__()
        self.loss_names = ["task"] + [n for n in loss_names if n != "task"]
        self.num_losses = len(self.loss_names)
        
        # Log variance parameters (log(sigma^2))
        self.log_vars = nn.Parameter(
            torch.full((self.num_losses,), init_log_var)
        )
    
    def forward(
        self, 
        losses: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.log_vars.device)
        
        for i, name in enumerate(self.loss_names):
            if name in losses:
                log_var = self.log_vars[i]
                # precision = exp(-log_var) = 1 / sigma^2
                # weighted_loss = 0.5 * precision * loss + 0.5 * log_var (regularization)
                precision = torch.exp(-log_var)
                weighted = 0.5 * precision * losses[name] + 0.5 * log_var
                total = total + weighted
        
        return total
    
    def get_weights_for_logging(self) -> Dict[str, float]:
        precisions = torch.exp(-self.log_vars)
        return {
            f"weight/{name}": precisions[i].item()
            for i, name in enumerate(self.loss_names)
        }


# ============================================================================
# Loss Configuration
# ============================================================================

@dataclass
class LossConfig:
    """
    Configuration for the loss management system.
    
    This dataclass provides a clean interface for configuring losses
    without modifying model code.
    
    Attributes:
        concept_losses: List of concept loss types to use (empty = no concept loss)
        weighting_strategy: How to combine losses ('fixed', 'learnable', 'kendall_gal')
        loss_weights: Initial/fixed weights for each loss (used with 'fixed' strategy)
        loss_params: Additional parameters for specific losses
        
    Examples:
        >>> # MLM only (baseline)
        >>> config = LossConfig(concept_losses=[])
        
        >>> # MLM + orthogonality with fixed weight
        >>> config = LossConfig(
        ...     concept_losses=["orthogonality"],
        ...     weighting_strategy="fixed",
        ...     loss_weights={"task": 1.0, "orthogonality": 0.1}
        ... )
        
        >>> # MLM + 2 losses with learnable weights
        >>> config = LossConfig(
        ...     concept_losses=["orthogonality", "uniformity"],
        ...     weighting_strategy="kendall_gal"
        ... )
    """
    # What concept losses to use
    concept_losses: List[str] = field(default_factory=list)
    
    # How to weight losses
    weighting_strategy: WeightingStrategyType = "fixed"
    
    # Weights for fixed strategy (ignored for learnable strategies)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"task": 1.0})
    
    # Parameters for specific losses (e.g., threshold, temperature)
    loss_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure task weight exists
        if "task" not in self.loss_weights:
            self.loss_weights["task"] = 1.0
        
        # Validate loss names
        available = get_available_losses()
        for loss_name in self.concept_losses:
            if loss_name not in available:
                raise ValueError(
                    f"Unknown loss '{loss_name}'. Available: {available}"
                )
    
    @property
    def is_enabled(self) -> bool:
        """Check if any concept loss is enabled."""
        return len(self.concept_losses) > 0
    
    @classmethod
    def disabled(cls) -> "LossConfig":
        """Factory for disabled concept loss (MLM only)."""
        return cls(concept_losses=[])
    
    @classmethod
    def default(cls) -> "LossConfig":
        """Factory for default config (orthogonality with Kendall-Gal)."""
        return cls(
            concept_losses=["orthogonality"],
            weighting_strategy="kendall_gal"
        )
    
    @classmethod
    def from_dict(cls, d: Dict) -> "LossConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "concept_losses": self.concept_losses,
            "weighting_strategy": self.weighting_strategy,
            "loss_weights": self.loss_weights,
            "loss_params": self.loss_params,
        }


# ============================================================================
# Loss Manager (Facade Pattern)
# ============================================================================

class LossManager(nn.Module):
    """
    Central manager for loss computation in ConceptEncoder models.
    
    This class encapsulates all loss-related logic, keeping model classes
    focused on architecture only. It follows the Facade pattern to provide
    a simple interface to the complex loss subsystem.
    
    Responsibilities:
    - Create and manage loss components
    - Apply weighting strategy
    - Compute total loss from task loss + concept losses
    - Provide logging/monitoring interface
    
    Example:
        >>> # In model __init__
        >>> self.loss_manager = LossManager(LossConfig(
        ...     concept_losses=["orthogonality", "uniformity"],
        ...     weighting_strategy="kendall_gal"
        ... ))
        >>> 
        >>> # In forward pass
        >>> if labels is not None:
        ...     mlm_loss = cross_entropy(logits, labels)
        ...     total_loss = self.loss_manager(
        ...         task_loss=mlm_loss,
        ...         concept_repr=concept_repr
        ...     )
    """
    
    def __init__(self, config: Optional[LossConfig] = None):
        """
        Initialize the loss manager.
        
        Args:
            config: Loss configuration. None = disabled (task loss only)
        """
        super().__init__()
        
        self.config = config or LossConfig.disabled()
        
        # Create loss components
        self.loss_components: nn.ModuleDict = nn.ModuleDict()
        self._loss_component_instances: Dict[str, ConceptLossComponent] = {}
        
        for loss_name in self.config.concept_losses:
            params = self.config.loss_params.get(loss_name, {})
            component = create_loss_component(loss_name, **params)
            # Store as a wrapper module for proper parameter tracking
            self._loss_component_instances[loss_name] = component
        
        # Create weighting strategy
        self.weighting_strategy: Optional[WeightingStrategy] = None
        
        if self.config.is_enabled:
            all_loss_names = ["task"] + self.config.concept_losses
            
            if self.config.weighting_strategy == "fixed":
                self.weighting_strategy = FixedWeighting(self.config.loss_weights)
            elif self.config.weighting_strategy == "learnable":
                self.weighting_strategy = LearnableWeighting(all_loss_names)
            elif self.config.weighting_strategy == "kendall_gal":
                self.weighting_strategy = KendallGalWeighting(all_loss_names)
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {self.config.weighting_strategy}"
                )
    
    @classmethod
    def create_for_model(
        cls,
        concept_num: int,
        hidden_size: int,
        loss_config: Optional[LossConfig] = None
    ) -> "LossManager":
        """
        Factory method that validates loss feasibility and creates LossManager.
        
        This centralizes all loss setup logic in one place, removing duplication
        from model classes. Models should use this instead of direct instantiation.
        
        Args:
            concept_num: Number of concept vectors in the model
            hidden_size: Hidden dimension size of the model
            loss_config: Loss configuration. None = disabled (task loss only)
            
        Returns:
            Configured LossManager instance
            
        Example:
            >>> # In model __init__
            >>> self.loss_manager = LossManager.create_for_model(
            ...     concept_num=config.concept_num,
            ...     hidden_size=config.hidden_size,
            ...     loss_config=loss_config
            ... )
        """
        # Validate loss feasibility before creating manager
        if loss_config is not None and loss_config.is_enabled:
            warnings = check_loss_feasibility(
                concept_num,
                hidden_size,
                loss_config.concept_losses
            )
            for warning in warnings:
                from transformers.utils import logging
                logger = logging.get_logger(__name__)
                logger.warning(warning)
        
        return cls(loss_config)
    
    @property
    def is_enabled(self) -> bool:
        """Check if concept loss is enabled."""
        return self.config.is_enabled
    
    @property
    def has_learnable_weights(self) -> bool:
        """Check if weights are learnable parameters."""
        return self.config.weighting_strategy in ["learnable", "kendall_gal"]
    
    def forward(
        self,
        task_loss: torch.Tensor,
        concept_repr: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        return_breakdown: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total weighted loss.
        
        Args:
            task_loss: Main task loss (e.g., MLM, classification)
            concept_repr: Concept representations [batch, concepts, hidden]
            step: Training step (for scheduling/logging)
            return_breakdown: If True, return dict with individual losses
            
        Returns:
            total_loss if return_breakdown=False
            dict with 'total' and individual losses if return_breakdown=True
        """
        losses = {"task": task_loss}
        
        # Compute concept losses
        if self.config.is_enabled and concept_repr is not None:
            for loss_name, component in self._loss_component_instances.items():
                params = self.config.loss_params.get(loss_name, {})
                losses[loss_name] = component.compute(concept_repr, **params)
        
        # Apply weighting
        if self.weighting_strategy is not None:
            total = self.weighting_strategy(losses, step)
        else:
            # No concept losses, just return task loss
            total = task_loss
        
        if return_breakdown:
            losses["total"] = total
            return losses
        
        return total
    
    def get_logging_dict(self, prefix: str = "loss") -> Dict[str, float]:
        """
        Get a dictionary of metrics for logging.
        
        Args:
            prefix: Prefix for metric names
            
        Returns:
            Dictionary with loss names and weights for logging
        """
        metrics = {}
        
        if self.weighting_strategy is not None:
            weights = self.weighting_strategy.get_weights_for_logging()
            metrics.update({f"{prefix}/{k}": v for k, v in weights.items()})
        
        return metrics


# ============================================================================
# Utility Functions
# ============================================================================

def check_loss_feasibility(
    concept_num: int,
    hidden_size: int,
    loss_names: List[str]
) -> List[str]:
    """
    Check if chosen losses are mathematically feasible.
    
    Args:
        concept_num: Number of concept vectors
        hidden_size: Dimension of hidden representations
        loss_names: List of loss names to check
        
    Returns:
        List of warning messages (empty if all ok)
    """
    warnings = []
    
    ortho_losses = {"orthogonality", "soft_orthogonality"}
    used_ortho = set(loss_names) & ortho_losses
    
    if used_ortho and concept_num > hidden_size:
        warnings.append(
            f"concept_num ({concept_num}) > hidden_size ({hidden_size}). "
            f"Perfect orthogonality is mathematically impossible. "
            f"Consider using 'uniformity' or 'combined' loss instead."
        )
    
    return warnings

