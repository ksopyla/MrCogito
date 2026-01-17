"""
Concept Encoder Analysis Toolkit
================================

A comprehensive suite of metrics and visualization tools for understanding
what concepts learn, their relationships, and their token interactions.

Based on research literature:
- VICReg (Bardes et al., 2021): Variance-Invariance-Covariance analysis
- AttentionViz (Yeh et al., 2023): Joint query-key embedding visualization
- Perceiver IO (Jaegle et al., 2021): Cross-attention bottleneck analysis
- Probing Tasks (Miaschi et al., 2020): Linguistic property probing
- Intrinsic Dimensionality (Aghajanyan et al., 2020): Effective dimensionality
- Neural Collapse (Galanti et al., 2021): Representation collapse analysis
- T-REGS (Mordacq et al., 2025): Uniformity and collapse metrics

Author: Krzysztof Sopyla
Project: MrCogito - Concept Encoder Research
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import json

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ============================================================================
# 1. CONCEPT SPACE GEOMETRY METRICS
# ============================================================================

@torch.no_grad()
def compute_concept_geometry_metrics(concept_repr: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive geometric metrics for concept representations.
    
    Based on VICReg (Bardes et al., 2021) and T-REGS (Mordacq et al., 2025).
    
    Args:
        concept_repr: [batch_size, concept_num, hidden_size]
        
    Returns:
        Dictionary with geometry metrics:
        - effective_rank: How many dimensions are actually used (SVD-based)
        - uniformity: How uniformly distributed on hypersphere
        - alignment: How aligned concepts are (collapse indicator)
        - isotropy: Whether all dimensions are equally used
        - variance_explained_95: Dimensions needed for 95% variance
    """
    batch_size, concept_num, hidden_size = concept_repr.shape
    
    metrics = {}
    
    # Flatten across batch and compute on average representation
    concept_mean = concept_repr.mean(dim=0)  # [C, H]
    
    # === 1. Effective Rank (nuclear norm / spectral norm) ===
    # Measures how many dimensions are actually being used
    try:
        U, S, V = torch.svd(concept_mean)
        # Effective rank = (sum of singular values) / max singular value
        effective_rank = (S.sum() / (S.max() + 1e-8)).item()
        metrics['effective_rank'] = effective_rank
        
        # Normalized effective rank (0-1 scale)
        max_possible_rank = min(concept_num, hidden_size)
        metrics['effective_rank_normalized'] = effective_rank / max_possible_rank
        
        # Variance explained by top-k components
        variance_explained = (S ** 2) / ((S ** 2).sum() + 1e-8)
        cumsum = variance_explained.cumsum(0)
        dims_for_95 = (cumsum < 0.95).sum().item() + 1
        metrics['dimensions_for_95_variance'] = dims_for_95
        metrics['top_1_variance_ratio'] = variance_explained[0].item()
        metrics['top_5_variance_ratio'] = variance_explained[:5].sum().item() if len(S) >= 5 else 1.0
        
    except RuntimeError:
        metrics['effective_rank'] = float('nan')
        metrics['effective_rank_normalized'] = float('nan')
        metrics['dimensions_for_95_variance'] = float('nan')
    
    # === 2. Concept Pairwise Similarity (Collapse Detection) ===
    concept_norm = F.normalize(concept_repr, p=2, dim=-1)  # [B, C, H]
    concept_sim = torch.bmm(concept_norm, concept_norm.transpose(1, 2))  # [B, C, C]
    
    # Remove diagonal
    eye = torch.eye(concept_num, device=concept_sim.device).unsqueeze(0)
    off_diag_mask = 1.0 - eye
    off_diag = concept_sim * off_diag_mask
    
    metrics['max_concept_similarity'] = off_diag.max().item()
    metrics['mean_concept_similarity'] = off_diag.abs().sum().item() / (batch_size * concept_num * (concept_num - 1))
    metrics['std_concept_similarity'] = off_diag[off_diag_mask.bool()].std().item()
    
    # === 3. Uniformity Loss (Wang & Isola, 2020) ===
    # Lower is better - concepts are well spread on hypersphere
    sq_dist = 2.0 - 2.0 * concept_sim
    uniformity = (torch.exp(-2.0 * sq_dist) * off_diag_mask).sum() / (
        batch_size * concept_num * (concept_num - 1)
    )
    metrics['uniformity_loss'] = uniformity.item()
    
    # === 4. Isotropy (Mu et al., 2018) ===
    # How evenly the embeddings use all dimensions
    concept_flat = concept_repr.reshape(-1, hidden_size)  # [B*C, H]
    concept_centered = concept_flat - concept_flat.mean(dim=0)
    
    # Covariance matrix eigenvalues
    cov = (concept_centered.T @ concept_centered) / (concept_centered.shape[0] - 1)
    try:
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.clamp(min=1e-8)
        
        # Isotropy = min(eigenvalue) / max(eigenvalue)
        isotropy = (eigenvalues.min() / eigenvalues.max()).item()
        metrics['isotropy'] = isotropy
        
        # Participation ratio (effective number of dimensions)
        # PR = (sum(λ))^2 / sum(λ^2)
        participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        metrics['participation_ratio'] = participation_ratio.item()
        metrics['participation_ratio_normalized'] = (participation_ratio / hidden_size).item()
        
    except RuntimeError:
        metrics['isotropy'] = float('nan')
        metrics['participation_ratio'] = float('nan')
    
    # === 5. Variance Statistics (VICReg-style) ===
    std_per_dim = concept_flat.std(dim=0)
    metrics['mean_dimension_std'] = std_per_dim.mean().item()
    metrics['min_dimension_std'] = std_per_dim.min().item()
    metrics['max_dimension_std'] = std_per_dim.max().item()
    metrics['collapsed_dimensions'] = (std_per_dim < 1e-4).sum().item()
    metrics['collapsed_dimensions_ratio'] = metrics['collapsed_dimensions'] / hidden_size
    
    # === 6. Concept Norm Statistics ===
    concept_norms = concept_repr.norm(dim=-1)  # [B, C]
    metrics['mean_concept_norm'] = concept_norms.mean().item()
    metrics['std_concept_norm'] = concept_norms.std().item()
    metrics['min_concept_norm'] = concept_norms.min().item()
    metrics['max_concept_norm'] = concept_norms.max().item()
    
    return metrics


# ============================================================================
# 2. ATTENTION PATTERN ANALYSIS
# ============================================================================

class ConceptAttentionExtractor:
    """
    Extract and analyze attention patterns between concepts and tokens.
    
    Based on:
    - AttentionViz (Yeh et al., 2023)
    - Quantifying Attention Flow (Abnar & Zuidema, 2020)
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: ConceptEncoder model (weighted or perceiver variant)
        """
        self.model = model
        self.attention_maps = {}
        self.hooks = []
        
    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights."""
        self.attention_maps = {}
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # MultiheadAttention returns (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.attention_maps[name] = attn_weights.detach()
            return hook
        
        # Find attention modules and register hooks
        if hasattr(self.model, 'encoder'):
            encoder = self.model.encoder
            for layer_idx, layer in enumerate(encoder.layers):
                # Cross-attention: concepts attending to tokens
                if hasattr(layer, 'concept_token_attn'):
                    hook = layer.concept_token_attn.register_forward_hook(
                        get_attention_hook(f'layer_{layer_idx}_cross')
                    )
                    self.hooks.append(hook)
                
                # Self-attention: concepts attending to concepts
                if hasattr(layer, 'concept_self_attn'):
                    hook = layer.concept_self_attn.register_forward_hook(
                        get_attention_hook(f'layer_{layer_idx}_self')
                    )
                    self.hooks.append(hook)
        
        # Decoder attention for perceiver models
        if hasattr(self.model, 'decoder_cross_attn'):
            hook = self.model.decoder_cross_attn.register_forward_hook(
                get_attention_hook('decoder_cross')
            )
            self.hooks.append(hook)
            
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    @torch.no_grad()
    def extract_attention(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for a given input.
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            
        Returns:
            Dictionary with attention maps from each layer
        """
        self._register_attention_hooks()
        self.model.eval()
        
        try:
            # Forward pass with attention output
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        finally:
            self._remove_hooks()
            
        return self.attention_maps
    
    @torch.no_grad()
    def compute_concept_token_affinity(
        self,
        attention_maps: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute which tokens each concept attends to most.
        
        Returns:
            Dictionary with:
            - concept_to_token: [concept_num, seq_length] aggregated attention
            - dominant_concepts: [seq_length] which concept dominates each position
        """
        results = {}
        
        # Aggregate cross-attention across layers
        cross_attns = [v for k, v in attention_maps.items() if 'cross' in k]
        
        if cross_attns:
            # Average across layers and heads
            # Attention shape: [batch, heads, query_len, key_len]
            stacked = torch.stack(cross_attns, dim=0)  # [layers, B, H, Q, K]
            avg_attn = stacked.mean(dim=[0, 1, 2])  # [Q, K] = [concepts, tokens]
            
            results['concept_to_token_affinity'] = avg_attn
            results['dominant_concept_per_token'] = avg_attn.argmax(dim=0)
            results['token_attention_entropy'] = self._compute_entropy(avg_attn, dim=0)
            results['concept_attention_entropy'] = self._compute_entropy(avg_attn, dim=1)
            
        return results
    
    def _compute_entropy(self, probs: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute entropy along a dimension (higher = more uniform attention)."""
        probs = probs.clamp(min=1e-8)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum(dim=dim)
        return entropy


# ============================================================================
# 3. CONCEPT SPECIALIZATION ANALYSIS
# ============================================================================

@dataclass
class ConceptSpecializationResult:
    """Results from concept specialization analysis."""
    concept_token_counts: Dict[int, Dict[int, int]]  # concept -> {token_id: count}
    concept_position_bias: torch.Tensor  # [concept_num, max_position]
    concept_clustering: Dict[str, float]  # Clustering quality metrics
    token_to_concept_mapping: Dict[int, List[int]]  # token_id -> top concepts
    

class ConceptSpecializationAnalyzer:
    """
    Analyze what each concept has specialized to learn.
    
    Based on probing task methodology (Miaschi et al., 2020).
    """
    
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.concept_token_stats = defaultdict(lambda: defaultdict(int))
        self.concept_position_counts = None
        
    @torch.no_grad()
    def collect_statistics(
        self,
        dataloader,
        max_batches: int = 100
    ) -> None:
        """
        Collect concept-token co-occurrence statistics from data.
        
        Args:
            dataloader: DataLoader yielding batches with 'input_ids' and 'attention_mask'
            max_batches: Maximum number of batches to process
        """
        self.model.eval()
        extractor = ConceptAttentionExtractor(self.model)
        
        config = self.model.config if hasattr(self.model, 'config') else self.model.encoder.config
        concept_num = config.concept_num
        max_seq_len = config.max_sequence_length
        
        self.concept_position_counts = torch.zeros(concept_num, max_seq_len)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)
            
            # Get attention maps
            attention_maps = extractor.extract_attention(input_ids, attention_mask)
            
            # Compute concept-token affinity
            affinity = extractor.compute_concept_token_affinity(attention_maps)
            
            if 'dominant_concept_per_token' in affinity:
                # For each position, record which concept dominates
                dominant = affinity['dominant_concept_per_token']  # [seq_len]
                
                # Aggregate token-concept co-occurrences
                for batch_i in range(input_ids.shape[0]):
                    for pos in range(input_ids.shape[1]):
                        if attention_mask is None or attention_mask[batch_i, pos] == 1:
                            token_id = input_ids[batch_i, pos].item()
                            concept_idx = dominant[pos].item() if len(dominant.shape) == 1 else dominant[batch_i, pos].item()
                            
                            self.concept_token_stats[concept_idx][token_id] += 1
                            if pos < max_seq_len:
                                self.concept_position_counts[concept_idx, pos] += 1
    
    def get_concept_vocabulary(self, concept_idx: int, top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Get the top tokens associated with a specific concept.
        
        Args:
            concept_idx: Which concept to analyze
            top_k: Number of top tokens to return
            
        Returns:
            List of (token_string, count) tuples
        """
        token_counts = self.concept_token_stats[concept_idx]
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(self.tokenizer.decode([tok_id]), count) for tok_id, count in sorted_tokens]
    
    def compute_concept_entropy(self) -> torch.Tensor:
        """
        Compute entropy for each concept's token distribution.
        
        Higher entropy = concept attends to diverse tokens
        Lower entropy = concept is specialized to few tokens
        
        Returns:
            [concept_num] tensor of entropy values
        """
        entropies = []
        
        for concept_idx in sorted(self.concept_token_stats.keys()):
            counts = torch.tensor(list(self.concept_token_stats[concept_idx].values()), dtype=torch.float)
            if counts.sum() > 0:
                probs = counts / counts.sum()
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                entropies.append(entropy.item())
            else:
                entropies.append(0.0)
                
        return torch.tensor(entropies)


# ============================================================================
# 4. PROBING TASK FRAMEWORK
# ============================================================================

class ConceptProbingTask(nn.Module):
    """
    Linear probing classifier to test what information concepts encode.
    
    Based on probing task methodology for understanding neural representations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        probe_type: str = 'concept_mean'  # 'concept_mean', 'concept_max', 'all_concepts'
    ):
        super().__init__()
        self.probe_type = probe_type
        
        if probe_type == 'all_concepts':
            raise NotImplementedError("Use concept_mean or concept_max for now")
        
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, concept_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concept_repr: [batch_size, concept_num, hidden_size]
            
        Returns:
            logits: [batch_size, num_labels]
        """
        if self.probe_type == 'concept_mean':
            pooled = concept_repr.mean(dim=1)  # [B, H]
        elif self.probe_type == 'concept_max':
            pooled = concept_repr.max(dim=1).values  # [B, H]
        else:
            pooled = concept_repr.mean(dim=1)
            
        return self.classifier(pooled)


PROBING_TASKS = {
    # Syntactic probing tasks
    'sentence_length': {
        'description': 'Predict binned sentence length (short/medium/long)',
        'num_labels': 3,
        'type': 'surface'
    },
    'word_content': {
        'description': 'Predict if specific words are present',
        'num_labels': 2,
        'type': 'surface'
    },
    'tree_depth': {
        'description': 'Predict parse tree depth',
        'num_labels': 4,
        'type': 'syntactic'
    },
    'top_constituent': {
        'description': 'Predict top constituent of parse tree',
        'num_labels': 20,
        'type': 'syntactic'
    },
    'past_present': {
        'description': 'Predict tense (past vs present)',
        'num_labels': 2,
        'type': 'syntactic'
    },
    'subject_number': {
        'description': 'Predict subject number (singular vs plural)',
        'num_labels': 2,
        'type': 'syntactic'
    },
    'object_number': {
        'description': 'Predict object number (singular vs plural)', 
        'num_labels': 2,
        'type': 'syntactic'
    },
    # Semantic probing tasks
    'semantic_odd_man_out': {
        'description': 'Identify semantically odd word',
        'num_labels': 2,
        'type': 'semantic'
    },
    'coordination_inversion': {
        'description': 'Detect if coordinated clauses are swapped',
        'num_labels': 2,
        'type': 'semantic'
    },
}


# ============================================================================
# 5. VISUALIZATION UTILITIES
# ============================================================================

class ConceptVisualizer:
    """
    Visualization tools for concept analysis.
    
    Generates publication-quality figures for:
    - Concept embedding space (UMAP/t-SNE)
    - Attention heatmaps
    - Concept similarity matrices
    - Training dynamics
    """
    
    def __init__(self, save_dir: str = './Cache/Outputs/concept_analysis'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            # Set publication-quality defaults
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.figsize': (10, 8),
                'figure.dpi': 150,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight'
            })
    
    def plot_concept_similarity_matrix(
        self,
        concept_repr: torch.Tensor,
        title: str = "Concept Similarity Matrix",
        save_name: str = "concept_similarity.png"
    ) -> Optional[str]:
        """
        Plot pairwise cosine similarity between concepts.
        
        Args:
            concept_repr: [batch_size, concept_num, hidden_size] or [concept_num, hidden_size]
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available for visualization")
            return None
            
        # Handle batch dimension
        if concept_repr.dim() == 3:
            concept_repr = concept_repr.mean(dim=0)  # [C, H]
            
        concept_repr = concept_repr.detach().cpu()
        
        # Compute similarity
        concept_norm = F.normalize(concept_repr, p=2, dim=-1)
        sim_matrix = (concept_norm @ concept_norm.T).numpy()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        ax.set_xlabel('Concept Index')
        ax.set_ylabel('Concept Index')
        ax.set_title(title)
        
        # Add diagonal reference
        n = sim_matrix.shape[0]
        ax.plot([0, n-1], [0, n-1], 'k--', alpha=0.3, linewidth=1)
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_concept_embedding_space(
        self,
        concept_repr: torch.Tensor,
        method: str = 'umap',  # 'umap', 'tsne', 'pca'
        color_by: Optional[torch.Tensor] = None,
        title: str = "Concept Embedding Space",
        save_name: str = "concept_embeddings.png"
    ) -> Optional[str]:
        """
        Visualize concept embeddings in 2D space.
        
        Args:
            concept_repr: [concept_num, hidden_size] or [batch, concept_num, hidden_size]
            method: Dimensionality reduction method
            color_by: Optional tensor for coloring points
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
            
        # Handle batch dimension
        if concept_repr.dim() == 3:
            concept_repr = concept_repr.mean(dim=0)
            
        X = concept_repr.detach().cpu().numpy()
        
        # Dimensionality reduction
        if method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
        elif method == 'tsne' and HAS_SKLEARN:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
            X_2d = reducer.fit_transform(X)
        elif method == 'pca' and HAS_SKLEARN:
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X)
        else:
            print(f"Warning: {method} not available, falling back to PCA")
            if HAS_SKLEARN:
                reducer = PCA(n_components=2)
                X_2d = reducer.fit_transform(X)
            else:
                return None
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if color_by is not None:
            colors = color_by.detach().cpu().numpy() if torch.is_tensor(color_by) else color_by
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='viridis', s=100, alpha=0.7)
            plt.colorbar(scatter, ax=ax)
        else:
            colors = np.arange(X.shape[0])
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='tab20', s=100, alpha=0.7)
        
        # Annotate points with concept indices
        for i, (x, y) in enumerate(X_2d):
            ax.annotate(str(i), (x, y), fontsize=8, alpha=0.7)
        
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        ax.set_title(title)
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        title: str = "Concept-Token Attention",
        save_name: str = "attention_heatmap.png"
    ) -> Optional[str]:
        """
        Plot attention weights between concepts and tokens.
        
        Args:
            attention_weights: [concept_num, seq_length]
            tokens: List of token strings
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
            
        attn = attention_weights.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        im = ax.imshow(attn, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        # Set labels
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Concept Index')
        ax.set_title(title)
        
        # Add token labels if not too many
        if len(tokens) <= 50:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_training_dynamics(
        self,
        metrics_history: Dict[str, List[float]],
        title: str = "Concept Metrics During Training",
        save_name: str = "training_dynamics.png"
    ) -> Optional[str]:
        """
        Plot how concept metrics evolve during training.
        
        Args:
            metrics_history: Dictionary mapping metric names to lists of values
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
            
        n_metrics = len(metrics_history)
        fig, axes = plt.subplots(
            (n_metrics + 2) // 3, 3, 
            figsize=(15, 4 * ((n_metrics + 2) // 3))
        )
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, (name, values) in enumerate(metrics_history.items()):
            if idx < len(axes):
                ax = axes[idx]
                ax.plot(values, linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel(name)
                ax.set_title(name.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for idx in range(len(metrics_history), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        
        return save_path


# ============================================================================
# 6. COMPREHENSIVE ANALYSIS RUNNER
# ============================================================================

@dataclass
class ConceptAnalysisReport:
    """Complete analysis report for a concept encoder model."""
    model_name: str
    geometry_metrics: Dict[str, float]
    attention_analysis: Dict[str, any]
    specialization_analysis: Dict[str, any]
    visualization_paths: Dict[str, str]
    probing_results: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'geometry_metrics': self.geometry_metrics,
            'attention_analysis': {k: str(v) for k, v in self.attention_analysis.items()},
            'specialization_analysis': self.specialization_analysis,
            'visualization_paths': self.visualization_paths,
            'probing_results': self.probing_results
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ConceptAnalyzer:
    """
    Main class for comprehensive concept encoder analysis.
    
    Usage:
        analyzer = ConceptAnalyzer(model, tokenizer)
        report = analyzer.run_full_analysis(dataloader)
        report.save('analysis_report.json')
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        output_dir: str = './Cache/Outputs/concept_analysis'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.visualizer = ConceptVisualizer(output_dir)
        self.attention_extractor = ConceptAttentionExtractor(model)
        self.specialization_analyzer = ConceptSpecializationAnalyzer(model, tokenizer)
        
    @torch.no_grad()
    def run_full_analysis(
        self,
        dataloader,
        model_name: str = "concept_encoder",
        max_batches: int = 50
    ) -> ConceptAnalysisReport:
        """
        Run comprehensive analysis on the concept encoder.
        
        Args:
            dataloader: DataLoader for analysis
            model_name: Name for the report
            max_batches: Maximum batches to analyze
            
        Returns:
            ConceptAnalysisReport with all metrics and visualizations
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Collect concept representations
        all_concept_reprs = []
        all_attention_maps = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
            
            # Get encoder outputs
            if hasattr(self.model, 'encoder'):
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True
                )
                concept_repr = encoder_outputs.last_hidden_state
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                concept_repr = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
            
            if concept_repr is not None:
                all_concept_reprs.append(concept_repr.cpu())
            
            # Get attention maps
            attn_maps = self.attention_extractor.extract_attention(input_ids, attention_mask)
            all_attention_maps.append(attn_maps)
        
        # Aggregate concept representations
        if all_concept_reprs:
            concept_repr_cat = torch.cat(all_concept_reprs, dim=0)
        else:
            concept_repr_cat = None
        
        # === 1. Geometry Metrics ===
        geometry_metrics = {}
        if concept_repr_cat is not None:
            geometry_metrics = compute_concept_geometry_metrics(concept_repr_cat)
        
        # === 2. Attention Analysis ===
        attention_analysis = {}
        if all_attention_maps:
            # Use first batch for attention visualization
            first_attn = all_attention_maps[0]
            affinity = self.attention_extractor.compute_concept_token_affinity(first_attn)
            attention_analysis = affinity
        
        # === 3. Specialization Analysis ===
        self.specialization_analyzer.collect_statistics(dataloader, max_batches=max_batches)
        concept_entropies = self.specialization_analyzer.compute_concept_entropy()
        
        specialization_analysis = {
            'concept_entropies_mean': concept_entropies.mean().item(),
            'concept_entropies_std': concept_entropies.std().item(),
            'concept_entropies_min': concept_entropies.min().item(),
            'concept_entropies_max': concept_entropies.max().item(),
        }
        
        # Add top tokens for first few concepts
        for i in range(min(5, len(self.specialization_analyzer.concept_token_stats))):
            top_tokens = self.specialization_analyzer.get_concept_vocabulary(i, top_k=10)
            specialization_analysis[f'concept_{i}_top_tokens'] = [t[0] for t in top_tokens]
        
        # === 4. Generate Visualizations ===
        visualization_paths = {}
        
        if concept_repr_cat is not None:
            # Similarity matrix
            path = self.visualizer.plot_concept_similarity_matrix(
                concept_repr_cat,
                title=f"{model_name} - Concept Similarity",
                save_name=f"{model_name}_similarity.png"
            )
            if path:
                visualization_paths['similarity_matrix'] = path
            
            # Embedding space
            for method in ['pca', 'tsne', 'umap']:
                path = self.visualizer.plot_concept_embedding_space(
                    concept_repr_cat,
                    method=method,
                    title=f"{model_name} - Concepts ({method.upper()})",
                    save_name=f"{model_name}_{method}.png"
                )
                if path:
                    visualization_paths[f'embedding_{method}'] = path
        
        # Attention heatmap
        if 'concept_to_token_affinity' in attention_analysis:
            # Get tokens for first sample
            first_batch = next(iter(dataloader))
            first_ids = first_batch['input_ids'][0].tolist()
            tokens = [self.tokenizer.decode([tid]) for tid in first_ids]
            
            path = self.visualizer.plot_attention_heatmap(
                attention_analysis['concept_to_token_affinity'],
                tokens,
                title=f"{model_name} - Concept-Token Attention",
                save_name=f"{model_name}_attention.png"
            )
            if path:
                visualization_paths['attention_heatmap'] = path
        
        return ConceptAnalysisReport(
            model_name=model_name,
            geometry_metrics=geometry_metrics,
            attention_analysis={k: str(v.shape) if torch.is_tensor(v) else v 
                              for k, v in attention_analysis.items()},
            specialization_analysis=specialization_analysis,
            visualization_paths=visualization_paths
        )


# ============================================================================
# 7. TRAINING CALLBACK FOR METRIC LOGGING
# ============================================================================

class ConceptMetricsCallback:
    """
    Callback for logging concept metrics during training.
    
    Integrates with HuggingFace Trainer or custom training loops.
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 100,
        metrics_to_log: List[str] = None
    ):
        self.log_every_n_steps = log_every_n_steps
        self.metrics_to_log = metrics_to_log or [
            'effective_rank',
            'mean_concept_similarity',
            'uniformity_loss',
            'isotropy',
            'collapsed_dimensions_ratio'
        ]
        self.metrics_history = defaultdict(list)
        
    def on_step_end(
        self,
        step: int,
        concept_repr: torch.Tensor,
        trainer=None
    ) -> Optional[Dict[str, float]]:
        """
        Called at the end of each training step.
        
        Args:
            step: Current training step
            concept_repr: Current concept representations
            trainer: Optional HuggingFace Trainer for logging
            
        Returns:
            Dictionary of metrics if this is a logging step
        """
        if step % self.log_every_n_steps != 0:
            return None
            
        metrics = compute_concept_geometry_metrics(concept_repr)
        
        # Filter to requested metrics
        logged_metrics = {
            f"concept/{k}": v 
            for k, v in metrics.items() 
            if k in self.metrics_to_log
        }
        
        # Store history
        for k, v in logged_metrics.items():
            self.metrics_history[k].append(v)
        
        # Log to trainer if available
        if trainer is not None and hasattr(trainer, 'log'):
            trainer.log(logged_metrics)
        
        return logged_metrics
    
    def get_history(self) -> Dict[str, List[float]]:
        return dict(self.metrics_history)


if __name__ == "__main__":
    # Example usage
    print("Concept Analysis Toolkit loaded successfully!")
    print(f"Available metrics: {list(compute_concept_geometry_metrics.__doc__.split('Returns:')[1].split('-')[1:5])}")
    print(f"Probing tasks available: {list(PROBING_TASKS.keys())}")

