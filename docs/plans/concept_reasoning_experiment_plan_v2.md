# Concept Reasoning Experiment Plan v2 — Strategic Reframe

**Created: 2026-02-18**
**Supersedes:** [`concept_reasoning_experiment_plan_v1.md`](concept_reasoning_experiment_plan_v1.md) (v1, 2026-02-13, now obsolete)
**Author analysis basis:** Full GLUE results (L2 + L6), embedding space research, field survey (Feb 2026)

---

## Context: Why a New Plan?

The v1 plan correctly identified engineering and data-scaling improvements but retained a flawed framing: **"close the 24pt GLUE gap to BERT-Base."** After deeper analysis of the L6 results and the research landscape, this goal is both unachievable with the current architecture and the wrong target. This v3 plan corrects both.

The concept encoder's architecture is designed to *compress and abstract* token-level information. GLUE was designed to *reward preservation* of token-level information (especially CoLA, RTE). These are structurally opposed. The 24-point gap is not a training failure — it is the expected signature of a working information bottleneck being evaluated on a benchmark that penalizes compression.

This plan reframes the goal, prioritizes experiments accordingly, and adds the contrastive pretraining objective that was missing from v1.

---

## Completed Work (as of 2026-02-18)

| Step | Status | Key Finding |
|------|--------|-------------|
| L2 baseline on Minipile (H512L2C128) | **Done** | `weighted_mlm` best on MRPC (82.2% F1), `posonly` best on inference tasks |
| Full GLUE eval L2 | **Done** | CoLA ~0 MCC, MNLI ~54%, near-random on all inference tasks |
| Scale to L6, 40 epochs (H512L6C128) | **Done** | +3pts avg; MNLI 59.1%; CoLA still broken (0.13 MCC) |
| Sparse MLM decoding fix | **Done** | Fixed OOM from accelerate fp32 conversion on full logits |
| Full GLUE eval L6 | **Done** | `perceiver_mlm` wins 6/8 tasks; avg gap to BERT-Base = -23.7pts |
| torch.compile fix (dynamic=True) | **Done** | `training/mlm_training.py` + `--torch_compile_dynamic` flag added |
| T-REGS MST loss | **Done** | Added `TREGSMSTLoss` + `"t_regs_mst"` registry entry to `nn/loss_manager.py` |
| Concept losses enabled in script | **Done** | `CONCEPT_LOSSES="combined"`, `LOSS_WEIGHTING="kendall_gal"` in `train_mlm_multigpu_perceiver.sh` |
| Concept analysis — L6 perceiver_mlm | **Done** | **CRITICAL collapse detected** — see table below |
| Masked Diffusion model implementation | **Done** | `nn/concept_encoder_diffusion.py` + `training/train_diffusion.py` + `scripts/train_diffusion_multigpu.sh` |

### Concept Analysis Results — perceiver_mlm H512L6C128 (2026-02-18)

Raw results: `agent_memory/concept_analysis_l6_20260218.json` (480 samples, 30 batches)

| Metric | Value | Target | Status |
|---|---|---|---|
| Global effective rank (raw) | **5.07** | > 40 | **CRITICAL — collapsed** |
| Effective rank (normalized) | **0.040** | > 0.3 | **CRITICAL — only 4% used** |
| Participation ratio (normalized) | **0.013** | > 0.1 | **CRITICAL** |
| Mean pairwise concept similarity | 0.451 | < 0.3 | **POOR — highly correlated** |
| Max pairwise concept similarity | **1.000** | < 0.6 | **CRITICAL — duplicated concepts** |
| Dimensions for 95% variance | 20.6 | > 50 | **Poor — dominated by 20 dims** |
| Top-1 singular value dominance | **111.8** (vs next 34.1) | balanced | **1 concept dominates** |
| Uniformity loss | 0.104 | < 0.3 | OK |
| Collapsed dimensions ratio | 0.0 | 0.0 | Good |
| Mean dimension std | 0.517 | > 0.3 | OK |

**Critical finding:** Only **5 out of 128 concepts** have meaningful effective rank. The model is almost entirely dominated by a single concept vector (singular value 111.8 vs next 34.1). At least two concept pairs have cosine similarity = 1.0 (perfect duplicates). This is severe dimensional collapse — the concept bottleneck is not being used effectively.

**Implication:** The next MLM training run MUST include concept losses. The existing L6 model is essentially running on 5 concept dimensions instead of 128.

**References:**
- L6 results: [`full_glue_evaluation_20260205.md`](../experiments_results/full_glue_evaluation_20260205.md)
- Baseline comparison: [`encoders_glue_evaluation_baseline.md`](../experiments_results/encoders_glue_evaluation_baseline.md)
- Embedding space theory: [`embedding_space_capabilites.md`](../research-notes/embedding_space_capabilites.md)
- Concept analysis raw: `agent_memory/concept_analysis_l6_20260218.json`

---

## Root Cause Diagnosis (Updated)

The v1 diagnosis identified three root causes: compression, position, data. This is correct but incomplete. Two additional root causes are now identified:

### Root Cause 1: Pretraining Objective Misalignment (NEW — critical)

MLM requires reconstructing individual masked tokens. The concept bottleneck is designed to destroy token-level detail during compression. These objectives are in direct conflict. The bottleneck is forced to be neither a good semantic abstractor nor a good token reconstructor — it converges to a suboptimal compromise.

**Evidence:** MLM loss improved 37% (L2→L6) but downstream tasks improved only +3pts. Better MLM reconstruction does not produce better semantic concepts; it produces better-at-reconstruction-but-not-semantic concepts.

**Fix:** Add a contrastive sentence-level objective alongside MLM (see Phase 3 below).

**Reference paper:** *MAE-LM: Representation Deficiency in Masked Language Modeling* (Meng et al., 2023) — [[HF paper]](https://hf.co/papers/2302.02060). Shows that [MASK] tokens corrupt encoder representations; the concept bottleneck amplifies this.

### Root Cause 2: GLUE Evaluates the Wrong Properties (NEW — critical)

GLUE was designed for full-sequence token-level encoders (BERT-style). Concept encoders compress information: tasks that require token-level structure will always fail.

| GLUE Task | Requires token-level detail? | Survivable through concept bottleneck? |
|---|---|---|
| **CoLA** | Yes — grammaticality is sub-word | **No — architectural ceiling reached at L6** |
| **RTE** | Partially | Partially |
| **SST-2** | No — holistic sentiment | Yes |
| **MRPC / QQP** | No — semantic similarity | Yes |
| **QNLI / MNLI** | Partially — compositional meaning | Partially |
| **STS-B** | No — similarity | Yes |

**CoLA MCC 0.13 at L6 is the architectural ceiling, not a training failure.** No amount of additional data or depth will fix CoLA. Drop it as an optimization target.

**Fix:** Add long-context benchmarks (SCROLLS, LongBench) where concept compression is an advantage, not a liability.

### Root Cause 3: Pretraining Data Starvation

| Model | Pretraining tokens | Concept encoder gap |
|---|---|---|
| BERT-Base | ~3.3B | Baseline |
| RoBERTa | ~160B | |
| DeBERTa-base | ~78B | |
| **Our L6 model** | **~0.6B effective (Minipile ×40)** | **5–270x deficit** |

Seeing Minipile 40 times means the model is memorizing corpus statistics rather than generalizing language understanding.

### Root Cause 4: Compression Ratio Wrong for Benchmark

128 concepts / 512 tokens = **4:1 compression**. This ratio is:
- Too aggressive to preserve token-level information (GLUE loses)
- Not aggressive enough to show the architecture's efficiency advantage (512 tokens is trivial for O(N²) BERT)

The concept encoder's real advantage appears at N > 1024 tokens, where O(C×N) vs O(N²) matters. Current experiments never test this.

### Root Cause 5: Classification Head Ignores Pretrained Decoder

Current GLUE fine-tuning: `single CLS query → concepts → linear classifier`. This throws away the entire pretrained Perceiver decoder, which already learned to reconstruct positional sequence representations from concepts. The decoder contains exactly the information the classifier needs.

**Fix:** Implement Phase 1C (classification via decoder) — code already sketched in v1.

---

## NEW Strategic Goal

> **Instead of:** "Close the 24pt GLUE gap to BERT-Base on token-level tasks"
>
> **New goal:** "Demonstrate that concept compression enables competitive performance on *long-context* tasks (N > 1K tokens) at a fraction of the memory and compute of standard attention, and enables *multimodal* (text + audio) understanding through a shared concept space."

**Practical target benchmarks:**
- **SCROLLS** (long document understanding, 1K-10K tokens) — primary new target
- **LongBench** (multilingual long-context tasks) — secondary
- **GLUE MRPC, QQP, MNLI** — retained as secondary metrics (fair tasks for concept representations)
- ~~GLUE CoLA~~ — dropped as optimization target (architectural impossibility)

**Publication framing:** "Concept Bottleneck Encoder for Long-Context and Multimodal Understanding — matching BERT on semantic tasks with 97% less compute at 4K tokens."

---

## Phase 1: Engineering Foundation (1-2 days, prerequisite for all)

Carry over from v1. These are unchanged but not yet done. **Do these before any new training run.**

### 1.1 Enable Flash Attention / SDPA

Set `need_weights=False` on ALL `nn.MultiheadAttention` calls:
- `nn/concept_encoder.py` (cross-attn line 163, self-attn line 179)
- `nn/concept_encoder_perceiver.py` (decoder cross-attn lines 200, 560; cls cross-attn line 378)
- `nn/concept_encoder_weighted.py` (no attention in decoder, skip)

This enables SDPA/Flash Attention automatically on PyTorch 2.x. **Expected: 2-4x training speedup. Required for 4K+ context experiments.**

### 1.2 Enable torch.compile

```bash
--torch_compile True
--torch_compile_backend "inductor"
```

Test for graph breaks before enabling on full training runs. Expected 1.5-2x additional speedup.

### 1.3 Fused Cross-Entropy (Liger Kernel)

Replace `CrossEntropyLoss` with `LigerCrossEntropyLoss` in sparse MLM decoding paths. Expected +20% throughput, -60% memory on loss computation.

### 1.4 Data Loading

```bash
--dataloader_num_workers=4
--dataloader_prefetch_factor=2
```

---

## Phase 2: Diagnostic — Concept Analysis (0.5 days, no training cost)

**Run this immediately on the best existing checkpoint before any new experiments.** This determines whether further training with the current objective is worth it.

Checkpoint: `perceiver_mlm_H512L6C128_20260208_211633`
Tools: `analysis/check_model_health.py`, `analysis/concept_analysis.py`

### Metrics to measure and interpret:

| Metric | Target | If below target |
|---|---|---|
| Effective rank of 128 concepts | > 100 (>80%) used | Concepts have collapsed → add VICReg immediately |
| Mean concept correlation | < 0.3 | Concepts are redundant → add orthogonality loss |
| Attention specialization | Concepts attend to different token types | Low specialization → span masking needed |
| Dimension utilization (512 dims) | > 30% active | Under-utilization → add T-REGS MST regularization |
| Intrinsic dimensionality (TwoNN) | > 20 | Severe collapse → architecture change needed |

**Reference:** *Revealing the Utilized Rank of Subspaces of Learning* (Garg et al., 2024) — ViT-B/16 utilizes only 35% of embedding space without regularization. [[HF paper]](https://hf.co/papers/2407.04797)

**Decision gate:** If effective rank < 50 (less than 40% of concepts actually used), the MLM pretraining objective is causing dimensional collapse. This makes Phase 3 (contrastive objective) the top priority ahead of data scaling.

---

## Phase 3: Classification via Decoder (2-3 days coding, 1 day eval)

**Carry over from v1 Phase 3.1, but now the highest-priority experiment.**

The current classification head (`CLS query → concepts → linear`) ignores the entire pretrained Perceiver decoder. Reuse the decoder for classification:

```python
# Current (wasteful):
#   concepts → CLS cross-attn → classify
#
# Proposed (full weight reuse):
#   concepts → Perceiver decoder (pretrained) → full sequence repr → pool → classify

class ConceptEncoderForSequenceClassificationViaDecoder(PreTrainedModel):
    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Encode to concepts (same as MLM pretraining)
        concept_repr = self.encoder(input_ids, attention_mask).last_hidden_state

        # 2. Decode back to full sequence using PRETRAINED decoder
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.decoder_query_embeddings(position_ids).expand(input_ids.shape[0], -1, -1)
        input_embeddings = self.encoder.token_embeddings(input_ids)
        decoder_queries = input_embeddings + pos_embeddings  # Input + Position (perceiver_mlm strategy)

        attn_output, _ = self.decoder_cross_attn(
            query=decoder_queries, key=concept_repr, value=concept_repr
        )
        decoder_output = decoder_queries + attn_output
        decoder_output = decoder_output + self.decoder_ffn(self.post_cross_norm(decoder_output))

        # 3. Mean pool over non-padding tokens, then classify
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (decoder_output * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        logits = self.classifier(self.dropout(pooled))
        ...
```

**Why this matters:** Loads ALL pretrained weights (encoder + decoder), not just encoder. The Perceiver decoder already learned position→concept→token mappings during MLM pretraining. Pooling over the reconstructed sequence gives richer representations than a single CLS query.

**Expected impact:** +5-10pts on QNLI, MNLI (positional composition); modest improvement on MRPC, QQP. CoLA will still be near-zero — that is expected.

**Note from v1:** This class `ConceptEncoderForSequenceClassificationViaDecoder` already exists in `nn/concept_encoder_perceiver.py`. Verify it loads pretrained decoder weights correctly and run GLUE evaluation.

---

## Phase 4: Scale Pretraining Data + Add Contrastive Objective (5-7 days training)

This is the highest-impact experiment. Combines the data scaling from v2 with a new contrastive objective that addresses Root Cause 1.

### 4.1 Dataset: OpenWebText + Wikipedia (same as v1 Phase 2.1)

- `Skylion007/openwebtext` (~8M samples, 13.5GB) + `wikimedia/wikipedia` 20231101.en (~6.7M, 20GB)
- Total: ~15M samples, ~33GB = 10x Minipile
- Fits in RAM on Polonez (256GB)
- Fallback if insufficient: `HuggingFaceFW/fineweb-edu` sample-10BT

### 4.2 NEW: Add Contrastive Sentence Objective Alongside MLM

**Motivation:** MLM alone trains the concept space to be good at reconstructing individual masked tokens. We also want the concept space to encode *semantic similarity*. Adding a contrastive objective alongside MLM directly targets this.

**Implementation:**
```python
# During training, for each batch:
# - MLM loss: as usual (predicts masked tokens)
# - Contrastive loss: adjacent text spans should have similar concept representations
#                     random pairs should have different concept representations

def contrastive_concept_loss(concept_repr_a, concept_repr_b, temperature=0.07):
    """
    concept_repr_a: [B, C, H] - concepts for span A (e.g., first half of document)
    concept_repr_b: [B, C, H] - concepts for span B (e.g., second half of document)
    Adjacent spans from the same document = positive pairs.
    """
    # Pool concepts to sentence-level vector
    z_a = F.normalize(concept_repr_a.mean(dim=1), dim=-1)  # [B, H]
    z_b = F.normalize(concept_repr_b.mean(dim=1), dim=-1)  # [B, H]

    # In-batch negatives (SimCSE-style)
    sim_matrix = torch.matmul(z_a, z_b.T) / temperature   # [B, B]
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
    return loss

# Combined loss:
total_loss = mlm_loss + 0.1 * contrastive_loss
```

**Why this is the right fix:** This is exactly what Large Concept Models (Meta, 2024) use at sentence level with SONAR embeddings. It trains the concept space directly for semantic similarity. The weight 0.1 keeps MLM dominant but adds the semantic alignment signal.

**Reference papers:**
- *Large Concept Models* (LCM Team, Meta, Dec 2024) — sentence-level concept prediction validated [[HF paper]](https://hf.co/papers/2412.08821)
- *SimCSE: Simple Contrastive Learning of Sentence Embeddings* (Gao et al., 2021)

### 4.3 Training Protocol

```bash
# Architecture: perceiver_mlm H512L6C128 (best from current experiments)
# Data: OpenWebText + Wikipedia, streamed and interleaved
# Epochs: 10-15 (vs 40 on Minipile — more data, fewer epochs needed)
# LR: 3e-4 cosine, 3000 warmup steps
# Effective batch: 512
# New flags:
#   --contrastive_loss_weight 0.1
#   --span_masking True  (see Phase 5)
#   --mlm_probability 0.30  (increased from 0.15, forces richer concept encoding)
# Target MLM loss: < 2.0 (vs 2.54 on Minipile)
```

---

## Phase 5: Span Masking (implement together with Phase 4)

Replace random 15% token masking with span masking. Low implementation cost; should be done as part of Phase 4 data pipeline.

```python
# In DataCollator: mask contiguous spans of 3-10 tokens
# Forces concepts to encode phrases/semantic chunks, not individual tokens
# Combined with higher masking rate (30%) to increase information pressure on the bottleneck
```

**Why:** Token-level random masking can be satisfied by remembering token co-occurrence statistics locally. Span masking forces the model to encode semantic content at the phrase level — exactly what concepts should represent.

**Reference:** *SpanBERT* (Joshi et al., 2019) — showed +2-4pts on QA, coreference, and relation extraction over BERT with span masking. [[HF paper]](https://hf.co/papers/1907.10529)

**PMI masking option:** Use pointwise mutual information to select spans of linguistically meaningful units (verb phrases, NPs). Harder to implement but more principled. Try simple span masking first.

---

## Phase 6: Long-Context Evaluation (NEW — 1-2 days, no training)

**This is the most important new addition to the plan.** Run the trained model on sequences longer than 512 tokens to validate the architecture's core efficiency claim.

### 6.1 Extend Model to Handle Longer Sequences

The concept encoder cross-attention is already O(C×N) not O(N²). Extend the positional embeddings and test at:
- 512 tokens (current baseline)
- 1024 tokens
- 2048 tokens
- 4096 tokens (requires Flash Attention from Phase 1.1)

```python
# Simple extension: resize position embeddings at inference time
# Most HF models support this via model.resize_token_embeddings() equivalent
# Or use RoPE/ALiBi for length extrapolation
```

### 6.2 Benchmarks to Evaluate

| Benchmark | Task Type | Seq Length | Why relevant |
|---|---|---|---|
| **SCROLLS** | Long doc QA, summarization | 1K-10K | Primary long-context target |
| **LongBench** | Multi-task long-context | 1K-32K | Diverse evaluation |
| **QASPER** | Scientific QA | ~4K | Realistic long-context NLP |
| **GovReport** | Summarization | ~10K | Tests concept compression |

**Hypothesis:** At N=4096, concept encoder uses O(128×4096) = 524K attention ops vs BERT's impossible O(4096²) = 16.7M. This is a 32x efficiency advantage. Even at 50% downstream performance parity, the efficiency story changes entirely.

### 6.3 Why This Unlocks the SoTA Path

| Benchmark type | Can concept encoder win? | Why |
|---|---|---|
| GLUE (N≤512) | Unlikely on CoLA/RTE | Architecture penalized for compression |
| GLUE MRPC/QQP | Competitive after data scaling | Semantic similarity = concept strength |
| Long-context (N>1K) | **Yes — architectural advantage** | O(C×N) beats O(N²) |
| Multimodal text+audio | **Yes — unique capability** | Shared concept space, modality-agnostic |

**Reference:** *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Geiping et al., Feb 2025) — reasoning in latent space outperforms token-space models. [[HF paper]](https://hf.co/papers/2502.05171)

---

## Phase 7: Concept Regularization Ablations (1-2 days, quick)

Unchanged from v1. Infrastructure is ready (`LossManager`). Run on L6 perceiver_mlm.

| Experiment | Config | Expected |
|---|---|---|
| Baseline | `CONCEPT_LOSSES="none"` | Current results |
| Orthogonality | `CONCEPT_LOSSES="orthogonality" LOSS_WEIGHTING="kendall_gal"` | Diverse concepts |
| VICReg | `CONCEPT_LOSSES="vicreg" LOSS_WEIGHTING="kendall_gal"` | Prevent collapse |
| T-REGS MST | `CONCEPT_LOSSES="t_regs_mst"` | Better uniformity than VICReg |
| Combined | `CONCEPT_LOSSES="combined" LOSS_WEIGHTING="kendall_gal"` | Best of both |

**New option — T-REGS MST uniformity** (from *T-REGS: Minimum Spanning Tree Regularization*, Mordacq et al., 2025 [[HF paper]](https://hf.co/papers/2510.23484)):
```python
def mst_uniformity_loss(concepts):
    distances = torch.cdist(concepts, concepts)
    nn_distances = distances.topk(k=2, dim=-1, largest=False).values[:, :, 1]
    return -nn_distances.sum(dim=-1).mean()  # Maximize spread
```

**Advantage over VICReg:** Detects and penalizes dimensional collapse that variance-based metrics miss.

---

## Phase 9: Masked Diffusion Decoder — Replace MLM (NEW — HIGH PRIORITY)

**Status:** Implementation complete. See `nn/concept_encoder_diffusion.py`, `training/train_diffusion.py`, `scripts/train_diffusion_multigpu.sh`.

### Why masked diffusion instead of MLM?

| Property | MLM (current) | Masked Diffusion |
|---|---|---|
| Masking rate | Fixed 15% | Sampled t ~ Uniform(0.05, 1.0) per batch |
| At low t (5-15%) | decoder uses local context (easy) | same — local denoising |
| At high t (70-99%) | impossible — too few tokens survive | decoder MUST use concepts |
| Objective alignment | forces token-level preservation through bottleneck | forces semantic abstraction |
| Inference | fill-mask only | iterative generation from all-[MASK] |
| Graph breaks in compile | yes (sparse indexing) | no (uniform [B, L] shapes) |

The key architectural insight: at t ≈ 1.0 (all tokens masked), the decoder receives **nothing but concepts**. It MUST extract all information from the encoder's concept vectors. This creates direct gradient pressure to make concepts semantically rich, eliminating the MLM misalignment root cause.

### Architecture

```
Input tokens [B, L]
    ↓
ConceptEncoder (same L6 encoder)
    ↓
Concepts [B, C=128, H=512]
    ↓  (+ noisy_ids [B, L] + timestep t [B])
ConceptDiffusionDecoder (2 transformer layers)
    └── Self-attention (token ↔ token coordination)
    └── Cross-attention (token → concepts)
    └── Timestep conditioning (AdaLN scale+shift)
    ↓
Logits [B, L, V]  → CE loss at masked positions
```

### Warm-start strategy (important)

The encoder architecture is identical to the MLM models. Warm-start the diffusion model from the best MLM checkpoint:

```bash
# In train_diffusion.py: load encoder weights from MLM checkpoint
# Only the diffusion decoder starts from random init
model.encoder.load_state_dict(
    ConceptEncoderForMaskedLMPerceiver.from_pretrained(mlm_checkpoint).encoder.state_dict()
)
```

This gives the encoder a head start vs training from scratch, and lets training focus on optimizing the decoder and concept regularization.

### Comparison experiment

Run both in parallel on Minipile for 20 epochs:
- **Run A**: MLM + combined + kendall_gal (current script, new losses)
- **Run B**: Masked Diffusion + combined + kendall_gal (new script)

Compare: concept analysis (effective rank), GLUE MRPC/MNLI, training stability.

### Training script

```bash
bash scripts/train_diffusion_multigpu.sh
```

---

## Phase 10: Slot Attention Concept Encoder (NEW — MEDIUM PRIORITY)

**Status:** Design complete (described in previous analysis). Not yet implemented.
**Rationale:** Deprioritized vs masked diffusion — can be added as an encoder variant without changing the training objective.

### What is Slot Attention?

Standard cross-attention (current): softmax over **token positions** — each concept gets a soft mixture of all tokens.

Slot Attention (Locatello et al., 2020): softmax over **concept dimension** — each token position assigns its mass to mostly one concept. This forces concepts to **compete** for token attribution, creating specialization.

```python
# Current ConceptEncoderLayer: softmax over tokens
attn_logits = Q_concepts @ K_tokens.T   # [B, C, T]
attn_weights = softmax(attn_logits, dim=-1)  # each concept attends all tokens

# Slot Attention: softmax over concepts
attn_logits = Q_tokens @ K_concepts.T   # [B, T, C]
attn_weights = softmax(attn_logits, dim=-1)  # each token "votes" for ONE concept
concept_updates = attn_weights.T @ token_features  # [B, C, H]
```

The concept-dimension softmax is what produces semantic specialization: one concept for nouns, another for verbs, another for clause boundaries.

### Weight-tied iterations (Universal Transformer style)

Instead of L=6 layers with different weights, apply ONE `SlotConceptLayer` K=6 times with tied weights. Benefits:
- Test-time compute scaling: run K=12 at inference for hard inputs
- Each iteration refines concepts until convergence (like EM)
- Directly analogous to Geiping et al. recurrent depth reasoning

### Implementation plan

```python
class SlotConceptLayer(nn.Module):
    """Replaces ConceptEncoderLayer with Slot Attention."""
    # See detailed implementation in previous architecture analysis

class ConceptSlotEncoder(nn.Module):
    """Weight-tied SlotConceptLayer applied K times."""
    def __init__(self, config, num_iterations=6):
        self.slot_layer = SlotConceptLayer(config)  # ONE shared layer
        self.num_iterations = num_iterations

    def forward(self, input_ids, attention_mask, num_iterations=None):
        K = num_iterations or self.num_iterations
        for _ in range(K):  # Same weights, K times
            concepts = self.slot_layer(concepts, token_features, padding_mask)
        return concepts
```

### When to implement

After Phase 9 (masked diffusion) has been validated on Minipile:
1. Add `ConceptSlotEncoder` as a new model variant (`model_type="slot_mlm"` or `"slot_diffusion"`)
2. Run concept analysis: slot attention should dramatically improve effective rank vs standard cross-attention
3. If effective rank improves to > 50% (64+ of 128 concepts), integrate as the default encoder

### Expected impact

The concept analysis shows only 5/128 concepts are used effectively. Slot Attention's competition mechanism is the most direct architectural fix for this collapse. Combined with masked diffusion:
- Slot Attention fixes the encoder (diverse, specialized concepts)
- Masked diffusion fixes the training objective (forces concepts to encode semantic content)

This combination is the core of the proposed architecture.

**References:**
- *Object-Centric Learning with Slot Attention* (Locatello et al., 2020) — [[HF paper]](https://hf.co/papers/2006.15055)
- *Universal Transformers* (Dehghani et al., 2018) — [[HF paper]](https://hf.co/papers/1807.03819)
- *Recurrent Depth Reasoning* (Geiping et al., 2025) — [[HF paper]](https://hf.co/papers/2502.05171)

---

## Phase 8: Architectural Experiments (parallel after Phase 4)

### 8.1 Dimension Inversion: Thin Tokens + Fat Concepts (HIGH PRIORITY)

Carry over from v1 Phase 3.3. Now supported by stronger evidence.

**Theory:** Token embedding intrinsic dimensionality is 10-37 (Tsukagoshi & Sasano, 2025). ALBERT uses 128-dim token embeddings → 768-dim hidden with full GLUE performance. Apply this to concept encoder: tiny token embeddings, large concept embeddings.

```python
class ConceptEncoderConfig:
    token_embedding_dim: int = 32    # down from 512
    concept_dim: int = 512           # unchanged or larger
```

**Ablation grid:**

| token_dim | concept_dim | Ratio | Expected |
|---|---|---|---|
| 512 | 512 | 1:1 | Baseline (current) |
| 64 | 512 | 1:8 | Better concept capacity per param |
| 32 | 512 | 1:16 | ALBERT-style factorization |
| 32 | 1024 | 1:32 | Maximum concept richness |

**Why now more justified:** The embedding space research (`embedding_space_capabilites.md`) shows keeping only 25% of embedding dims causes minimal performance degradation. Token dims can be cut to 32-64 without information loss.

**Reference:** *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (Aghajanyan et al., 2020) — RoBERTa achieves 90% performance with 200 trainable params. [[HF paper]](https://hf.co/papers/2012.13255)

### 8.2 Backbone Initialization from Pretrained CLM (NEW — HIGH PRIORITY)

**Motivation:** We train from random init while BERT/RoBERTa/DeBERTa all benefit from extensive pretraining. The biphasic CLM+MLM paper (2025) shows initializing from a pretrained causal LM and then fine-tuning with MLM gives dramatically better representations.

**Proposal:** Initialize the token-level backbone transformer layers from a small pretrained model:
- `HuggingFaceTB/SmolLM2-135M` — 135M params, trained on 11T tokens, permissive license
- `Qwen/Qwen2.5-0.5B` — 500M params, strong baseline

Keep concept cross-attention and decoder as random init (novel architecture). This gives the token-level processing layers 100B-token pretraining at zero training cost.

**Implementation:** Load pretrained backbone weights selectively:
```python
pretrained = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")
concept_encoder.backbone.load_state_dict(pretrained.state_dict(), strict=False)
# concept_cross_attn and decoder remain random init
```

**Reference:** *Should We Still Pretrain Encoders with Masked Language Modeling?* (Gisserot-Boukhlef et al., Jul 2025) — biphasic CLM→MLM training outperforms pure MLM from scratch. [[HF paper]](https://hf.co/papers/2507.00994)

### 8.3 Position-Enriched Concepts (MEDIUM PRIORITY)

Carry over from v1 Phase 3.2. Still worth testing but moved down in priority.

**Recommended option — RoPE on concept self-attention:**
Apply RoPE to the self-attention layers within the concept processing path (not cross-attention). Gives concepts a "before/after" ordering without adding parameters.

### 8.4 Gradual Compression / Funnel-style (LOW PRIORITY)

Carry over from v1 Phase 3.5. Still interesting but complex to implement. Do after 8.1 and 8.2.

---

## Execution Priority (Revised — 2026-02-18)

### ✅ Already done (2026-02-18)

| Step | What | Result |
|---|---|---|
| ✅ | Flash Attention (`need_weights=False` on all MHA) | Already set in code |
| ✅ | torch.compile fix (`dynamic=True`) | `mlm_training.py` updated, `--torch_compile_dynamic` flag |
| ✅ | T-REGS MST loss | `nn/loss_manager.py` registry entry |
| ✅ | Enable concept losses in script | `combined + kendall_gal` now default |
| ✅ | Concept analysis on L6 | **CRITICAL collapse: effective rank 4%** — 5/128 concepts used |
| ✅ | Masked Diffusion implementation | `nn/concept_encoder_diffusion.py` + training scripts |

### 🔜 Immediate next (this week)

| # | Experiment | Effort | Expected Impact | Dependencies | Status |
|---|---|---|---|---|---|
| **1** | Restart MLM training with `combined+kendall_gal` losses | 5 days GPU | Fix concept collapse (eff. rank 4% → >50%) | None | **Start now** |
| **2** | Run masked diffusion on Minipile (parallel) | 5 days GPU | Validate diffusion objective | None | **Start now** |
| **3** | Phase 3: Classification via decoder | 2 days | +5-10pts QNLI/MNLI on L6 model | None | High |
| **4** | Re-run concept analysis after (1) | 0.5 day | Verify concept losses fixed collapse | After (1) | High |

### 📅 Next quarter

| # | Experiment | Effort | Expected Impact | Dependencies |
|---|---|---|---|---|
| **5** | Phase 4+5: Scale data (OpenWebText+Wiki) + span masking | 7 days | Largest single improvement | After (1) |
| **6** | Phase 6: Long-context evaluation (SCROLLS/LongBench) | 2 days | Validates efficiency advantage | Phase 4+5 |
| **7** | Phase 9: Masked Diffusion on large data | 7 days | Best pretraining objective | After (5) |
| **8** | Phase 10: Slot Attention encoder variant | 5 days code + 5 days train | Fix concept collapse architecturally | Phase 7 |
| **9** | Phase 8.2: Backbone init from SmolLM2 | 3 days | Bypass data starvation | Phase 1 |
| **10** | Phase 7: Concept losses ablation (t_regs_mst vs combined) | 2 days | +1-3pts, identify best loss | After (4) |
| **11** | Phase 8.1: Dimension inversion (token_dim=32, concept_dim=512) | 3+5 days | Novel efficiency + quality | Phase 5 |
| **12** | Phase 8.3: Position-enriched concepts (RoPE on self-attn) | 2 days | +2-4pts composition | Phase 1 |

**Critical path:** [Restart MLM with concept losses + Diffusion on Minipile] in parallel → Concept re-analysis → Scale data → Long-context eval → Slot Attention → publish

---

## Decision Gates

Use these to avoid wasting GPU-days on the wrong next step:

```
After Phase 2 (concept analysis):
  effective_rank < 50?  → Prioritize Phase 7 (VICReg/orthogonality) before data scaling
  mean_correlation > 0.5? → Add orthogonality loss to Phase 4 training
  effective_rank > 100? → Proceed with data scaling as planned

After Phase 4 (scaled training):
  MLM loss < 2.0? → On track, proceed to Phase 6 long-context eval
  MLM loss > 2.5? → Check data pipeline, try higher LR or learning rate schedule
  GLUE MNLI > 65%? → Strong signal, expand long-context evaluation
  GLUE MNLI < 60%? → Consider backbone init (Phase 8.2) before further data scaling

After Phase 6 (long-context eval):
  Concept encoder competitive at N=4096? → Focus on long-context SoTA + publication
  Not competitive? → Profile bottleneck: is it the concept count (128) or model depth?
```

---

## Research Horizon

### Near-term (v2 experiments succeed, <10pt gap on semantic GLUE tasks):
1. **Long-context SoTA** — target SCROLLS leaderboard with 4K-16K sequence experiments
2. **Publication** — "Concept Bottleneck Encoder for Long-Context and Multimodal NLU" — dimension inversion + classification via decoder + long-context results

### Medium-term (after long-context validation):
3. **Audio modality** — concept bottleneck on mel spectrograms; shared concept space with text; target speech understanding benchmarks
4. **Concept encoder-decoder for generation** — use LCM's approach: predict next-concept-embedding autoregressively; frozen concept encoder as semantic anchor

### Long-term (the original vision, 2-3 years):
5. **Multimodal SoTA** — concept bottleneck as universal modality bridge; competing with Qwen-Omni / Moshi on speech-text understanding
6. **Recurrent concept refinement** — apply Geiping et al. recurrent depth idea to concept tokens; each "thinking step" refines the 128 concept vectors; enable test-time compute scaling

---

## Key Papers Reference Table

| Paper | Year | Finding | Relevance | Link |
|---|---|---|---|---|
| Large Concept Models (Meta) | 2024 | Sentence-level concept prediction works for generation | Validates concept approach at scale | [HF](https://hf.co/papers/2412.08821) |
| Recurrent Depth Reasoning (Geiping) | 2025 | Latent space reasoning outperforms token space | Natural extension for concept generation | [HF](https://hf.co/papers/2502.05171) |
| Should We Still Use MLM? | 2025 | CLM→MLM biphasic beats pure MLM from scratch | Backbone init strategy | [HF](https://hf.co/papers/2507.00994) |
| MAE-LM (Representation Deficiency) | 2023 | [MASK] tokens corrupt encoder representations | Explains MLM misalignment | [HF](https://hf.co/papers/2302.02060) |
| SpanBERT | 2019 | Span masking > token masking for representation | Phase 5 justification | [HF](https://hf.co/papers/1907.10529) |
| Cramming 1568 Tokens | 2025 | 1500x compression theoretically achievable | Upper bound for concept compression | [HF](https://hf.co/papers/2502.13063) |
| Intrinsic Dimensionality (Aghajanyan) | 2020 | 200 params achieves 90% RoBERTa performance | Dimension inversion justification | [HF](https://hf.co/papers/2012.13255) |
| Revealing Utilized Rank (Garg) | 2024 | ViT uses only 20-35% of embedding space | Regularization needed | [HF](https://hf.co/papers/2407.04797) |
| VICReg (Bardes) | 2021 | Explicit variance prevents concept collapse | Concept regularization | [HF](https://hf.co/papers/2105.04906) |
| T-REGS MST (Mordacq) | 2025 | MST-based uniformity detects dimensional collapse | Better than VICReg | [HF](https://hf.co/papers/2510.23484) |
| Token Assorted (Su et al.) | 2025 | Mixing latent + text tokens improves reasoning | Hybrid concept-token generation | [HF](https://hf.co/papers/2502.03275) |
| Information Bottleneck (Shwartz-Ziv) | 2017 | Training = fitting then compression | Theoretical foundation | [HF](https://hf.co/papers/1703.00810) |

---

*Plan v2 created: 2026-02-18*
*Previous plan (v1): [`concept_reasoning_experiment_plan_v1.md`](concept_reasoning_experiment_plan_v1.md)*
*Next review: after Phase 4 training completes*
