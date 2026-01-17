
# Embedding Spaces Encoding Capabilities 

I have tried to understand the information encoding capabilities of the embedding spaces via the Shannon and other information theory concepts.

What I wanted to understand better is the relationship between the number of possible values for the variable K and the dimension of the embedding space D.


## Information Encoding Capabilities

Lets define main two variables: 

* K - number of possible values for the variable, eg for a binary K=2, for a ternary K=3, for a decimal K=10
* D - dimension of the embedding space, eg for a 1D embedding space D=1, for a 2D embedding space D=2, for a 3D embedding space D=3

Then the number of possible values for the variable is K^D.

The maximum value encoded by the embedding space is K^D - 1.

What I wanted to understand better is the relationship between the number of possible values for the variable K and the dimension of the embedding space D.

To increase the information encoding capabilities of the embedding vector we can:
1. Increase the number of possible values for the variable K
2. Increase the dimension of the embedding space D

---

## Research Questions & Answers

### Question 1: Dimension Increase Relationship

**Q:** If I increase D by 1, how many D-dim with K values vectors can I fit into the same embedding space?

**A:** The relationship is: K^(D+1) / K^D = K

**Research Insight:** While this is mathematically correct, practical utilization is much lower. Neural networks don't uniformly utilize embedding space.

**Key Paper:** *"Measuring the Intrinsic Dimension of Objective Landscapes"* (Li et al., 2018)
- Found that many problems have **much smaller intrinsic dimensions** than parameter space suggests
- Once parameter space is large enough, extra dimensions serve to increase the **solution manifold**, not problem-solving capacity
- [Paper Link](https://hf.co/papers/1804.08838)

---

### Question 2: Floating Point Precision (FP16, BFloat16)

**Q:** How does the K^D relationship apply to computer floating point precision?

**Research Answer:**

| Format | Mantissa Bits | Exponent Bits | Unique Values | Effective K |
|--------|---------------|---------------|---------------|-------------|
| FP32 | 23 | 8 | ~4.3×10^9 | Very high |
| BF16 | 7 | 8 | ~65,536 | ~256 effective |
| FP16 | 10 | 5 | ~65,536 | ~1024 effective |
| FP8 (E4M3) | 3 | 4 | ~256 | ~8-16 effective |

**Key Finding:** The "effective K" is NOT the total representable values, but the number of **meaningfully distinct values** that training can differentiate.

**Key Papers:**

1. **"A Study of BFLOAT16 for Deep Learning Training"** (Kalamkar et al., 2019)
   - BFloat16 maintains same **dynamic range as FP32** (8-bit exponent)
   - Achieves SOTA results **without hyperparameter tuning** unlike FP16
   - **Key insight: Range matters more than precision for DNN training**
   - [Paper Link](https://hf.co/papers/1905.12322)
   - **Relevance:** Can train ConceptEncoder in BF16 without quality loss

2. **"FP8 Formats for Deep Learning"** (Micikevicius et al., 2022)
   - Proposes E4M3 (4-bit exp, 3-bit mantissa) and E5M2 formats
   - Successfully trained models up to **175B parameters** with FP8
   - **Key insight: Training tolerates significant precision reduction if dynamic range preserved**
   - [Paper Link](https://hf.co/papers/2209.05433)
   - **Relevance:** Future optimization path for ConceptEncoder training

3. **"To FP8 and Back Again"** (Lee et al., 2024)
   - Found reduced precision affects **training stability more than final accuracy**
   - Key issue: Gradient updates can be "cancelled" by rounding in low precision
   - [Paper Link](https://hf.co/papers/2405.18710)
   - **Relevance:** Explains why some training runs may be unstable

---

### Question 3: Embedding Space Utilization

**Q:** Are embeddings uniformly distributed? Do training algorithms influence usage?

**Research Answer:** **NO, embeddings are NOT uniformly distributed.** Training algorithms strongly influence distribution.

**Key Phenomena Discovered:**
1. **Dimensional Collapse:** Features occupy only a low-dimensional subspace
2. **Anisotropy:** Embeddings cluster in narrow cones rather than spreading uniformly
3. **Neural Collapse:** In classification, features from same class collapse to their mean

**Key Papers:**

1. **"Revealing the Utilized Rank of Subspaces of Learning"** (Garg et al., 2024)
   - **ViT-B/16 on ImageNet utilizes only 35% of available space**
   - **ViT-L/16 utilizes only 20%** (larger model = lower utilization!)
   - Self-supervised pre-training increases utilization to ~70%
   - [Paper Link](https://hf.co/papers/2407.04797)
   - **Relevance:** Concepts might use only 20-35% of their dimensions without regularization

2. **"Redundancy, Isotropy, and Intrinsic Dimensionality of Prompt-based Text Embeddings"** (Tsukagoshi et al., 2025)
   - Keeping only **25% of dimensions** results in very slight performance degradation
   - For classification/clustering: reducing to **<0.5% of dimensions** causes minimal degradation!
   - Embeddings for different tasks show different isotropy
   - [Paper Link](https://hf.co/papers/2506.01435)
   - **Relevance:** Concepts for MLM may need different dimensions than concepts for classification

3. **"VICReg: Variance-Invariance-Covariance Regularization"** (Bardes et al., 2021)
   - Without explicit regularization, encoders tend to produce **constant or collapsed embeddings**
   - Proposed explicit variance term on each dimension to prevent collapse
   - [Paper Link](https://hf.co/papers/2105.04906)
   - **Relevance:** Directly inspired orthogonality/uniformity losses in ConceptEncoder

4. **"Rethinking The Uniformity Metric in Self-Supervised Learning"** (Fang et al., 2024)
   - Classic uniformity metrics **fail to detect dimensional collapse**
   - Proposed new metric that identifies when embeddings occupy only low-dimensional subspace
   - [Paper Link](https://hf.co/papers/2403.00642)
   - **Relevance:** Need proper metrics to monitor concept space utilization

5. **"T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning"** (Mordacq et al., 2025)
   - MST-based regularization simultaneously mitigates collapse AND promotes uniformity
   - Works on arbitrary compact Riemannian manifolds
   - [Paper Link](https://hf.co/papers/2510.23484)
   - **Relevance:** Alternative regularization approach for concept embeddings

6. **"Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"** (Aghajanyan et al., 2020)
   - Pre-trained models have **very low intrinsic dimension**
   - RoBERTa can achieve 90% performance with only **200 trainable parameters**!
   - **Larger models tend to have lower intrinsic dimension** after pre-training
   - [Paper Link](https://hf.co/papers/2012.13255)
   - **Relevance:** Explains why concept space might work with fewer dimensions than expected

---

### Question 4: Post-Training Compression

**Q:** Can we further compress learned embeddings via quantization, clustering, etc.?

**Research Answer:** **YES, significant compression is possible.**

**Key Papers:**

1. **"Cramming 1568 Tokens into a Single Vector and Back Again"** (Kuratov et al., 2025) ⭐
   - **Most relevant paper for ConceptEncoder!**
   - Found per-sample optimization can achieve **up to 1500x compression** (1568 tokens → 1 vector)
   - Current encoder-based methods achieve only ~10x compression
   - **Gap of 2 orders of magnitude** between current and theoretically attainable
   - **Key insight: Compression limit is determined by cross-entropy loss (uncertainty to reduce), not sequence length!**
   - [Paper Link](https://hf.co/papers/2502.13063)
   - **Relevance:** Theoretical upper bound for how much concepts can compress tokens

2. **"Learning Low-Rank Representations for Model Compression"** (Zhu et al., 2022)
   - Combines dimensionality reduction with vector quantization (LR²VQ)
   - Achieved **43x compression** on ResNet-18 with 2.8% accuracy improvement
   - [Paper Link](https://hf.co/papers/2211.11397)
   - **Relevance:** Post-training compression techniques for concept embeddings

3. **"FastText.zip: Compressing text classification models"** (Joulin et al., 2016)
   - Product quantization for word embeddings
   - **100x compression** with only slight accuracy degradation
   - [Paper Link](https://hf.co/papers/1612.03651)
   - **Relevance:** Can apply to token embeddings for smaller vocab memory footprint

4. **"Vector Quantization for Recommender Systems: A Review"** (Liu et al., 2024)
   - Comprehensive survey of VQ methods
   - Covers efficiency-oriented and quality-oriented approaches
   - [Paper Link](https://hf.co/papers/2405.03110)
   - **Relevance:** Systematic overview of compression techniques

5. **"Unified Scaling Laws for Compressed Representations"** (Panferov et al., 2025)
   - Shows "capacity" metric based on fitting random Gaussian data predicts parameter efficiency
   - Works across sparse, quantized, and vector-quantized formats
   - [Paper Link](https://hf.co/papers/2506.01863)
   - **Relevance:** Framework for understanding compression vs. performance trade-offs

--

## Key Papers Summary Table

| Paper | Year | Key Finding | Relevance to ConceptEncoder |
|-------|------|-------------|----------------------------|
| Cramming 1568 Tokens | 2025 | 1500x compression possible, 10x achieved | Upper bound for concept compression |
| Revealing Utilized Rank | 2024 | ViT uses 20-35% of space | Need regularization for utilization |
| VICReg | 2021 | Variance+covariance prevents collapse | Inspired concept losses |
| Intrinsic Dimensionality | 2020 | 200 params can achieve 90% | Concepts may work with fewer dims |
| BFloat16 Study | 2019 | Range > precision | Can train in BF16 |
| Low-Rank Bottleneck | 2020 | Attention has expressivity limits | Inform attention head design |
| Large Concept Models | 2024 | Sentence-level concepts work | Validates concept approach |
| SONAR | 2023 | 1024-dim for 200 languages | Reference for concept dim |

---

## Attention & Transformer Architecture Papers

1. **"Low-Rank Bottleneck in Multi-head Attention Models"** (Bhojanapalli et al., 2020)
   - Scaling between heads and head size creates low-rank bottleneck
   - **Recommends: Head size should scale with sequence length**
   - [Paper Link](https://hf.co/papers/2002.07028)
   - **Relevance:** Informs cross-attention design between concepts and tokens

2. **"On the Benefits of Rank in Attention Layers"** (Amsel et al., 2024)
   - **Dramatic trade-offs** between rank and number of heads
   - Some functions require **full-rank attention** or exponentially many low-rank heads
   - For short contexts, **depth can compensate** for low-rank
   - [Paper Link](https://hf.co/papers/2407.16153)
   - **Relevance:** Justifies using full hidden_size for attention

3. **"Representational Strengths and Limitations of Transformers"** (Sanford et al., 2023)
   - Embedding dimension directly affects what functions can be approximated
   - Sparse averaging tasks show transformers scale logarithmically vs. polynomially for other architectures
   - [Paper Link](https://hf.co/papers/2306.02896)
   - **Relevance:** Theoretical grounding for embedding dimension choices

---

## Information Bottleneck Theory Papers

1. **"Opening the Black Box of DNNs via Information"** (Shwartz-Ziv & Tishby, 2017)
   - Training has two phases: **fitting** and **compression**
   - Most epochs spent on compression, not fitting labels
   - Networks converge to Information Bottleneck bound
   - [Paper Link](https://hf.co/papers/1703.00810)
   - **Relevance:** Theoretical foundation for concept compression

2. **"How Does Information Bottleneck Help Deep Learning?"** (Kawaguchi et al., 2023)
   - Controlling information bottleneck **bounds generalization error**
   - New generalization bounds scale with degree of information bottleneck
   - [Paper Link](https://hf.co/papers/2305.18887)
   - **Relevance:** Justifies concept bottleneck architecture

3. **"Learning to Compress: Local Rank and Information Compression"** (Patel & Shwartz-Ziv, 2024)
   - Networks exhibit bias toward **low-rank solutions** during training
   - Rank decreases during final training phase
   - Reduced rank = compressed mutual information
   - [Paper Link](https://hf.co/papers/2410.07687)
   - **Relevance:** Explains why concept regularization is needed

4. **"To Compress or Not to Compress - Self-Supervised Learning and Information Theory"** (Shwartz-Ziv & LeCun, 2023)
   - Comprehensive review of SSL from information-theoretic standpoint
   - Unified framework for self-supervised information-theoretic learning
   - [Paper Link](https://hf.co/papers/2304.09355)
   - **Relevance:** Framework for understanding concept learning objectives

---

## Geometric Properties Papers

1. **"Geometric Properties of Neural Multivariate Regression"** (Andriopoulos et al., 2025)
   - Neural collapse **degrades performance** in regression (unlike classification)
   - Key finding: **ID_H > ID_Y prevents over-compression**
   - Two regimes: over-compressed and under-compressed
   - [Paper Link](https://hf.co/papers/2510.01105)
   - **Relevance:** Concept intrinsic dim should exceed token prediction intrinsic dim

2. **"Poincaré Embeddings for Learning Hierarchical Representations"** (Nickel & Kiela, 2017)
   - Hyperbolic space allows **parsimonious representations** of hierarchies
   - Hyperbolic embeddings outperform Euclidean on hierarchical data
   - [Paper Link](https://hf.co/papers/1705.08039)
   - **Relevance:** Alternative embedding space for hierarchical concept relationships

3. **"Representation Tradeoffs for Hyperbolic Embeddings"** (De Sa et al., 2018)
   - 2D hyperbolic achieves MAP 0.989 vs. 200D Euclidean at 0.87 on WordNet
   - Provides precision-dimensionality tradeoff bounds
   - [Paper Link](https://hf.co/papers/1804.03329)
   - **Relevance:** Hyperbolic concepts could be dramatically more efficient

---



### Metrics to Monitor

1. **effective_rank** - Target: > 80% of concept_num
2. **mean_concept_correlation** - Target: < 0.3
3. **dimension_variance_min** - Target: > 0.1

---

## Future Research Directions

Based on the literature review:

1. **Hyperbolic Concept Space** - Could dramatically reduce required dimensions
2. **Adaptive Concept Count** - Dynamic number of concepts based on input complexity
3. **Multi-Scale Concepts** - Hierarchical concepts at different granularities
4. **Diffusion-Based Concept Decoding** - As explored in Large Concept Models
5. **Concept Quantization** - Post-training compression to int8/int4

---

## Deep Dive: Tiny Token Embeddings + Concept Aggregation Paradigm

### My Core Hypothesis

**Contrarian Insight:** Token embeddings could be dramatically smaller (16-32D instead of 768D+) if concepts aggregate information via cross-attention into larger representations (256-1024D).

This is supported by converging evidence from multiple research streams:

### Evidence 1: Intrinsic Dimensionality is Shockingly Low

| Paper | Finding | Implication |
|-------|---------|-------------|
| Tsukagoshi & Sasano (2025) | Embeddings with 1000+ dims have intrinsic dimensionality of **10-37** | 16-32D tokens could suffice |
| Aghajanyan et al. (2020) | RoBERTa achieves 90% performance with **200 trainable params** | Effective information is tiny |
| Li et al. (2018) | Extra dimensions increase **solution manifold**, not capacity | Diminishing returns on dim |

**Key Paper:** "Redundancy, Isotropy, and Intrinsic Dimensionality of Prompt-based Text Embeddings" (2025)
- Keeping only 25% of dimensions: minimal performance loss
- For classification/clustering: **<0.5% of dimensions** works!
- [Link](https://aclanthology.org/2025.findings-acl.1330/)

### Evidence 2: Cross-Attention as Knowledge Aggregator

Architectures that validate the "aggregate small → large" approach:

| Architecture | Token Processing | Aggregation Mechanism |
|--------------|-----------------|----------------------|
| **Perceiver IO** | Arbitrary inputs | Cross-attention to 256-512 latents |
| **Set Transformer** | Input set | Inducing points |
| **Funnel-Transformer** | Progressive pooling | Sequence compression |
| **Compressive Transformer** | Compress past | Memory vectors |

**Key Paper:** "Attention Bottlenecks for Multimodal Fusion" (Nagrani et al., 2021)
- Forces information through **small bottleneck latents**
- Model must **condense most relevant information**
- [Link](https://hf.co/papers/2107.00135)

### Evidence 3: Theoretical Compression Limits

**"Cramming 1568 Tokens into a Single Vector"** (Kuratov et al., 2025):
- Achieved **1500x compression** via per-sample optimization
- Current encoders achieve only ~10x
- **Compression limit = cross-entropy of sequence, NOT length**
- [Link](https://hf.co/papers/2502.13063)

**Implication:** With 128 concepts for 1536+ tokens (12:1 ratio), you have massive headroom.

### The "Dimension Inversion" Principle

| Component | Current Practice | Proposed | Rationale |
|-----------|-----------------|----------|-----------|
| Token Embed | 768-4096D | **16-32D** | Intrinsic dim is 10-37 |
| Concept Embed | - | **256-512D** | Aggregation needs capacity |
| Ratio | - | 1:8 to 1:16 | Matches ALBERT factorization |

### Information Bottleneck Justification

```
Token Embedding (X) → Concept (T) → Output (Y)
                 ↓           ↓
            compress     preserve task info
```

Small tokens + large concepts naturally enforces:
- min I(X;T): forced compression at input (small dims)
- max I(T;Y): sufficient capacity for task (large concept dims)

---

## Advanced Regularization Techniques

### T-REGS: MST-based Uniformity (Novel, 2025)

From **"T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning"**:

```python
def mst_uniformity_loss(concepts):
    """
    MST length as uniformity metric.
    Longer MST = more spread out = better space utilization.
    Simultaneously prevents dimensional collapse AND promotes uniformity.
    """
    distances = torch.cdist(concepts, concepts)
    nn_distances = distances.topk(k=2, dim=-1, largest=False).values[:, :, 1]
    mst_approx = nn_distances.sum(dim=-1).mean()
    return -mst_approx  # Maximize MST length
```

**Advantage over VICReg:** Detects dimensional collapse that variance metrics miss.

### Intrinsic Dimension Monitoring

From "Geometric Properties of Neural Multivariate Regression" (2025):

**Key Finding:** ID_H > ID_Y prevents over-compression in regression.

```python
def estimate_intrinsic_dimension_twonn(embeddings, k=5):
    """Two-NN estimator for monitoring concept space utilization."""
    distances = torch.cdist(embeddings, embeddings)
    knn_dists = distances.topk(k=k+2, dim=-1, largest=False).values[:, :, 1:]
    mu = knn_dists[:, :, 1] / (knn_dists[:, :, 0] + 1e-8)
    intrinsic_dim = 1 / torch.log(mu).mean()
    return intrinsic_dim
```

**Target:** Concept ID should be 2-3x the task prediction complexity.

### Attention Sparsity Encouragement

Encourage concepts to specialize on different token groups:

```python
def attention_entropy_loss(attention_weights, target_entropy=1.0):
    """Low entropy = concepts attend to specific token groups."""
    probs = attention_weights + 1e-8
    entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
    return (entropy - target_entropy).pow(2)
```

---

## Novel Loss Functions for Knowledge Compression

### Multi-Scale Reconstruction Loss

Inspired by Cramming paper - compression quality scales with predictability:

```python
def multi_scale_reconstruction_loss(concepts, original_tokens, decoder, 
                                     scales=[1, 2, 4, 8]):
    """Reconstruct at multiple granularities."""
    total_loss = 0
    for scale in scales:
        pooled_tokens = F.avg_pool1d(original_tokens.transpose(1,2), scale).transpose(1,2)
        decoded = decoder(concepts, target_length=pooled_tokens.shape[1])
        weight = 1.0 / scale  # Finer scales weighted higher
        total_loss += weight * F.mse_loss(decoded, pooled_tokens)
    return total_loss
```

### Contrastive Concept Alignment

From Sub-Sentence Encoder paper:

```python
def contrastive_concept_loss(concepts_1, concepts_2, temperature=0.07):
    """Contrastive loss between concept views."""
    z1, z2 = F.normalize(concepts_1, dim=-1), F.normalize(concepts_2, dim=-1)
    pos_sim = (z1 * z2).sum(dim=-1) / temperature
    neg_sim = torch.bmm(z1, z2.transpose(-1, -2)) / temperature
    labels = torch.arange(concepts_1.shape[1], device=concepts_1.device)
    return F.cross_entropy(neg_sim.view(-1, concepts_1.shape[1]), 
                           labels.repeat(concepts_1.shape[0]))
```

---

## Experimental Roadmap

### Priority 1: Token Dimension Ablation

```python
token_dims = [16, 32, 64, 128, 256]
concept_dim = 256  # Fixed
concept_num = 128  # Fixed

# Hypothesis: 16-32D achieves 85-95% of 256D performance
```

### Priority 2: Dimension Inversion Test

```python
configs = [
    {"token_dim": 32, "concept_dim": 128},   # 1:4
    {"token_dim": 32, "concept_dim": 256},   # 1:8
    {"token_dim": 32, "concept_dim": 512},   # 1:16
]
```

### Priority 3: Regularization Comparison

```python
regularizations = ["none", "orthogonality", "vicreg", "t_regs_mst", "combined"]
# Track: effective rank, intrinsic dimension, concept utilization
```

### Priority 4: Compression Ratio Study

```python
# How many tokens per concept?
ratios = [4, 8, 16, 32]  # tokens per concept
```

---

## Hyperparameter Recommendations

### Token Embeddings (Aggressive Settings)
| Parameter | Conservative | Aggressive |
|-----------|-------------|------------|
| Dimension | 64 | **16-32** |
| Vocab Size | 32K | 128K+ |
| Projection | Linear | Small MLP |

### Concept Embeddings
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Dimension | 256-512 | 8-16x token dim |
| Number | 64-256 | 4:1 to 32:1 compression |

### Cross-Attention
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Heads | 8-16 | Higher rank = more expressive |
| Head Dim | concept_dim/heads | Full utilization |

### Regularization Weights
| Loss | Weight | Increase If |
|------|--------|-------------|
| Orthogonality | 0.01 | correlation > 0.5 |
| Variance | 0.1 | variance_min < 0.5 |
| MST Uniformity | 0.05 | clustering observed |

---

## Future Research Directions

### Hyperbolic Concept Space

From "Poincaré Embeddings" (2017):
- 2D hyperbolic achieves MAP 0.989 vs 200D Euclidean at 0.87
- Concepts could be dramatically more efficient in hyperbolic space

### Learned Concept Count

Dynamic allocation based on input complexity:

```python
class AdaptiveConcepts(nn.Module):
    def __init__(self, max_concepts, hidden_size):
        self.concept_pool = nn.Parameter(torch.randn(max_concepts, hidden_size))
        self.selector = nn.Linear(hidden_size, 1)
    
    def forward(self, token_features):
        importance = torch.sigmoid(self.selector(self.concept_pool))
        k = estimate_k_from_complexity(token_features)
        return self.concept_pool[importance.topk(k).indices]
```

### Multi-Scale Concepts

```python
# Hierarchical concept granularities
token_level_concepts = nn.Parameter(torch.randn(256, 64))   # Fine
phrase_level_concepts = nn.Parameter(torch.randn(64, 128))  # Medium
sentence_level_concepts = nn.Parameter(torch.randn(16, 256)) # Coarse
```

---

## Additional Key Papers

### Efficient Transformers with Small Dimensions

1. **"Greenformers: Low-Rank Approximation"** (Cahyawijaya, 2021)
   - Low-rank transformer factorization
   - BERT compression by 30%+
   - [Link](https://hf.co/papers/2108.10808)

2. **"EfficientFormer: ViT at MobileNet Speed"** (Li et al., 2022)
   - Dimension-consistent design for efficiency
   - [Link](https://hf.co/papers/2206.01191)

### Token Pooling and Aggregation

3. **"Token Pooling in Vision Transformers"** (Marin et al., 2021)
   - Softmax-attention is low-pass filter → redundancy can be pruned
   - [Link](https://hf.co/papers/2110.03860)

4. **"Hierarchical Transformers Are More Efficient"** (Nawrot et al., 2021)
   - Hourglass architecture with downsampling/upsampling
   - [Link](https://hf.co/papers/2110.13711)

5. **"Funnel-Transformer"** (Dai et al., 2020)
   - Progressive compression of hidden states
   - Re-invests FLOPs in model capacity
   - [Link](https://hf.co/papers/2006.03236)

### Late Chunking and Contextual Aggregation

6. **"Late Chunking: Contextual Chunk Embeddings"** (Günther et al., 2024)
   - Chunking AFTER transformer → preserves context
   - [Link](https://hf.co/papers/2409.04701)

---

*Last updated: January 2026*
*Research conducted using HuggingFace paper search and web sources*
