# Concept Encoder Analysis Framework

A comprehensive framework for understanding, visualizing, and validating what concepts learn in the ConceptEncoder architecture.

## 📚 Research Background

This framework is grounded in several key research papers:

| Paper | Year | Key Contribution | Application to Concepts |
|-------|------|------------------|------------------------|
| VICReg (Bardes et al.) | 2021 | Variance-Invariance-Covariance regularization | Collapse detection, diversity metrics |
| Perceiver IO (Jaegle et al.) | 2021 | Cross-attention bottleneck | Concept-token interaction analysis |
| AttentionViz (Yeh et al.) | 2023 | Joint query-key visualization | Attention pattern analysis |
| Probing Tasks (Miaschi et al.) | 2020 | Linguistic property probing | What information concepts encode |
| Intrinsic Dimensionality (Aghajanyan et al.) | 2020 | Effective dimensionality | Understanding concept space utilization |
| T-REGS (Mordacq et al.) | 2025 | Uniformity and MST regularization | Distribution quality |
| Neural Collapse (Galanti et al.) | 2021 | Representation collapse | Collapse detection and prevention |

## 🔧 Framework Components

### 1. Concept Space Geometry (`compute_concept_geometry_metrics`)

Measures the geometric properties of learned concept representations:

#### Dimensionality Metrics
- **Effective Rank**: `sum(σ) / max(σ)` - how many SVD components are significantly used
- **Dimensions for 95% Variance**: Number of components needed to explain 95% of variance
- **Participation Ratio**: `(Σλ)² / Σλ²` - effective number of dimensions used

#### Similarity Metrics
- **Max Concept Similarity**: Highest cosine similarity between any two concepts (collapse indicator)
- **Mean Concept Similarity**: Average pairwise similarity
- **Uniformity Loss**: Gaussian kernel-based measure of distribution on hypersphere

#### Distribution Metrics
- **Isotropy**: `min(λ) / max(λ)` of covariance eigenvalues - dimension utilization balance
- **Collapsed Dimensions**: Count of dimensions with std < 1e-4

### 2. Attention Pattern Analysis (`ConceptAttentionExtractor`)

Extracts and analyzes how concepts attend to input tokens:

- **Concept-Token Affinity**: Which tokens each concept focuses on
- **Attention Entropy**: How focused vs. distributed the attention is
- **Dominant Concept per Position**: Which concept "claims" each token position

### 3. Concept Specialization Analysis (`ConceptSpecializationAnalyzer`)

Determines what each concept has learned to specialize in:

- **Concept Vocabulary**: Top tokens associated with each concept
- **Concept Entropy**: Low entropy = specialized, high entropy = general
- **Position Bias**: Whether concepts prefer certain positions

### 4. Probing Tasks (`ConceptProbingTask`)

Linear probing to test what linguistic information is encoded:

| Task | Type | Description |
|------|------|-------------|
| sentence_length | Surface | Predict binned sentence length |
| past_present | Syntactic | Predict tense |
| subject_number | Syntactic | Predict singular/plural |
| tree_depth | Syntactic | Predict parse tree depth |
| semantic_odd_man_out | Semantic | Identify odd word |

## 📊 Key Metrics for Paper

### Primary Metrics (Always Report)

| Metric | Good Value | Interpretation |
|--------|-----------|----------------|
| Effective Rank (normalized) | > 0.5 | Concepts use diverse dimensions |
| Max Concept Similarity | < 0.5 | Concepts are distinct |
| Uniformity Loss | < 0.1 | Concepts well-distributed |
| Isotropy | > 0.01 | Balanced dimension usage |

### Collapse Indicators (Red Flags)

- Effective rank < 0.3 * max_possible → dimensional collapse
- Max similarity > 0.8 → concept collapse
- Collapsed dimensions > 10% of hidden_size → sparse utilization

## 🎨 Visualization Types

### 1. Concept Similarity Matrix
**Purpose**: Visualize pairwise relationships between concepts
**Interpretation**: 
- Diagonal should be 1 (self-similarity)
- Off-diagonal should be near 0 for orthogonal concepts
- Clusters indicate groups of related concepts

### 2. Singular Value Spectrum
**Purpose**: Understand effective dimensionality
**Interpretation**:
- Sharp drop = low effective dimension (potential collapse)
- Gradual decay = well-utilized space

### 3. Embedding Space (PCA/t-SNE/UMAP)
**Purpose**: 2D visualization of concept relationships
**Interpretation**:
- Evenly spread = good diversity
- Tight clusters = potential collapse
- Outliers = specialized concepts

### 4. Concept-Token Attention Heatmap
**Purpose**: Understand what each concept attends to
**Interpretation**:
- Sparse rows = specialized concepts
- Dense rows = general concepts
- Patterns = learned linguistic structures

### 5. Layer-wise Evolution
**Purpose**: How concepts evolve through layers
**Interpretation**:
- Increasing effective rank = progressive refinement
- Stable similarity = consistent representations

## 📈 Training Dynamics Monitoring

Use `ConceptMetricsCallback` during training:

```python
from analysis.concept_analysis import ConceptMetricsCallback

callback = ConceptMetricsCallback(
    log_every_n_steps=100,
    metrics_to_log=['effective_rank', 'mean_concept_similarity', 'uniformity_loss']
)

# In training loop:
metrics = callback.on_step_end(step, concept_repr)
if metrics:
    wandb.log(metrics)
```

### Expected Training Dynamics

| Phase | Effective Rank | Similarity | Uniformity |
|-------|----------------|------------|------------|
| Early | Low/Increasing | Variable | High |
| Middle | Stabilizing | Decreasing | Decreasing |
| Late | Stable | Low | Low |

## 🔬 Research Questions to Answer

1. **What do concepts encode?**
   - Run probing tasks on concept representations
   - Analyze concept vocabulary (most associated tokens)
   
2. **Are concepts diverse?**
   - Check effective rank and similarity metrics
   - Visualize similarity matrix
   
3. **Do concepts specialize?**
   - Analyze concept entropy distribution
   - Look for low-entropy specialized concepts
   
4. **How do concepts interact with tokens?**
   - Visualize attention patterns
   - Compute concept-token affinity
   
5. **Does training prevent collapse?**
   - Monitor geometry metrics during training
   - Compare with/without regularization losses

## 📁 Output Structure

```
Cache/Outputs/concept_analysis/
├── concept_similarity_matrix.png
├── singular_value_spectrum.png
├── concept_embeddings_pca.png
├── concept_embeddings_tsne.png
├── concept_embeddings_umap.png
├── dimension_usage.png
├── concept_specialization.png
├── attention_heatmap.png
├── layer_evolution.png
├── training_dynamics.png
└── analysis_report.json
```

## 🚀 Quick Start

```python
from analysis.concept_analysis import ConceptAnalyzer

# Load your model
model = ConceptEncoderForMaskedLMPerceiver.from_pretrained("path/to/model")

# Create analyzer
analyzer = ConceptAnalyzer(model, tokenizer)

# Run full analysis
report = analyzer.run_full_analysis(dataloader, model_name="my_model")

# Save report
report.save("./Cache/Outputs/concept_analysis/report.json")
```

## 📖 References

1. Bardes, A., Ponce, J., & LeCun, Y. (2021). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. https://hf.co/papers/2105.04906

2. Jaegle, A., et al. (2021). Perceiver: General Perception with Iterative Attention. https://hf.co/papers/2103.03206

3. Yeh, C., et al. (2023). AttentionViz: A Global View of Transformer Attention. https://hf.co/papers/2305.03210

4. Miaschi, A., et al. (2020). Linguistic Profiling of a Neural Language Model. https://hf.co/papers/2010.01869

5. Aghajanyan, A., et al. (2020). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. https://hf.co/papers/2012.13255

6. Mordacq, J., et al. (2025). T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning. https://hf.co/papers/2510.23484

7. Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. https://hf.co/papers/2005.00928

