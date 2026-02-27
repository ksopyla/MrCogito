# MrCogito — Concept Encoder Research

> A personal research journey into building an efficient, reasoning-capable transformer that thinks in "concepts" rather than tokens.

**Author:** Krzysztof Sopyla — [ai.ksopyla.com](https://ai.ksopyla.com) · [GitHub](https://github.com/ksopyla) · [LinkedIn](https://www.linkedin.com/in/krzysztof-sopyla/)
**Project page:** [ai.ksopyla.com/projects/concept-encoder](https://ai.ksopyla.com/projects/concept-encoder)

---

## What is this?

This is an open research project, not a polished library. I'm exploring a simple but ambitious idea: **what if a transformer didn't attend over every token, but instead compressed long sequences into a small number of dense "concept tokens" and reasoned from there?**

I'm doing this as a solo researcher, learning as I go. Expect experiments that fail, ideas that get revised, and honest documentation of what worked and what didn't.

---

## The Core Idea

Standard transformers use self-attention with O(N²) complexity — at 128K tokens, the attention matrix alone is enormous. The concept encoder replaces that with **cross-attention** between a small set of learned concept tokens (C) and the full input sequence (N):

```
Standard self-attention:  O(N × N)  — quadratic, memory-breaking at long contexts
Concept cross-attention:  O(C × N)  — linear in N, C scales gracefully with N
```

| Sequence length N | Concepts C | Concept O(C×N) | Self-attn O(N²) | Speedup |
|---|---|---|---|---|
| 512 | 128 | 65K | 262K | 4x |
| 4,096 | 512 | 2.1M | 16.7M | **8x** |
| 32,768 | 2,048 | 67M | 1.07B | **16x** |
| 1,048,576 | 8,192 | 8.6B | 1.1T | **128x** |

At 1M tokens, full self-attention is computationally intractable. Concept attention stays tractable — while forcing the model to produce increasingly abstract, semantic representations.

**Inference pipeline (current text focus):**
```
Input text (N tokens)
  → Encoder: cross-attention compresses N tokens into C concept tokens
  → Reasoning: recursive concept refinement (K iterations, weight-tied)
  → Decoder: generates text output from refined concepts
```

**End goal (audio):**
```
User speech (mel-spectrogram)
  → Audio adapter: maps audio into the same concept space
  → Reasoning: recursive refinement (shared weights with text)
  → Audio decoder (Talker): generates speech tokens from concepts
```

---

## Research Vision

The long-term goal is an **audio conversational and reasoning model** grounded in a concept bottleneck. Before adding audio, I need to prove the text concept bottleneck actually works — that it produces rich, non-collapsed semantic representations that support generation and reasoning.

**Training objective evolution** (the planned journey from reconstruction to reasoning):

| Phase | Objective | What it trains | How I'll know it works |
|---|---|---|---|
| Phase 0 | Self-reconstruction (MLM/diffusion/TSDAE) | Concept compression quality | STS-B > 0.75, concept rank > 64/128 |
| Phase 1 | **Prefix generation** (encode prefix, decode suffix) | Semantic concepts + generative decoder | Suffix perplexity < 3.0 |
| Phase 2 | **Variable-depth recursive training** | Latent reasoning via iteration | Better accuracy with more K iterations |
| Phase 3 | Instruction fine-tuning | Task-following generation | Instruction-following benchmarks |
| Phase 4 | Progressive sequence length (512 → 4K → 1M) | Long-context concept abstraction | SCROLLS, LongBench |

---

## Current State (Feb 2026)

I'm in **Phase 0** — still fighting to produce concept representations that aren't collapsed.

### Best Results So Far

| Task | Score | Notes |
|---|---|---|
| MRPC F1 | 82.73% | ViaDecoder evaluation (Perceiver L6) |
| STS-B Pearson | 0.650 | ViaDecoder evaluation (Perceiver L6) |
| QQP F1 | 73.35% | ViaDecoder evaluation |
| MNLI-m Acc | 59.75% | ViaDecoder evaluation |
| **Concept effective rank** | **5/128 (4%)** | Severe collapse — the main problem |

The semantic task numbers look reasonable for the model size. But the concept effective rank of 5/128 means only 4% of concept space is being utilized — the model is essentially using 5 directions in 128-dimensional space. That's **concept collapse**, and it's the root problem I'm working to fix.

### What I've Tried and What Failed

Honest accounting of failed approaches (I think documenting failures is as important as documenting successes):

| Approach | Outcome | Why it failed |
|---|---|---|
| `combined` loss + Kendall-Gal weighting | Rank 95% but GLUE collapsed (STS-B -46%) | Kendall-Gal muted MLM gradient; concepts geometrically diverse but semantically empty |
| `combined` loss + fixed weight 0.1 | Rank 12%, GLUE regressed | Combined loss can't prevent intra-sample collapse without destroying task quality |
| CLS-query classification head | 128:1 information collapse | Single attention query flattens all concept structure |
| Diffusion L2 self-reconstruction | Rank 2× better (10/128), STS-B 0.138 (near-random) | L2 architecture too shallow + missing ELBO weighting + self-reconstruction allows surface hashing |

Root cause analyses live in `docs/4_Research_Notes/`.

### Architecture Overhaul (Feb 21, 2026)

Diagnosing 5 structural misalignments in the original MLM+Perceiver setup led to a complete overhaul:
- **TSDAE training** (denoising autoencoder — 83× stronger gradient signal per concept vs sparse MLM)
- **BiXT** (bidirectional cross-attention — fixes static token embeddings)
- **PosOnly decoder** (separate sentence encoding via positional queries)
- **Weighted concept pooling** (richer downstream readout)
- **VICReg + t_regs_mst regularization with warmup** (cross-batch dimensional health + within-sample concept diversity)

Implementation complete. Training not yet started — waiting on GPU availability.

---

## Research Tracks

The work is organized into 5 parallel tracks, with Track A being the critical path everything else depends on.

### Track A — Fix Concept Quality (Critical Path)

**Goal:** Find the training objective that produces concept rank > 64/128 AND STS-B Pearson > 0.70.

| Experiment | Status |
|---|---|
| TSDAE PosOnly on Minipile | Implemented, awaiting GPU |
| TSDAE + BiXT on Minipile | Implemented, awaiting GPU |
| Masked Diffusion (warm-start from L6 MLM) | In progress on Polonez cluster |
| L6 Diffusion ablation (ELBO + t_min=0.3) | Code done, not yet run |
| Prefix generation training (encode prefix → decode suffix) | Not started (after diffusion results) |
| t_regs_mst + VICReg regularization | Implemented, untested |

**Key insight from SODA (CVPR 2024):** Self-reconstruction (X→X) permits surface-level hashing. Cross-content generation (encode A → generate B) forces genuine semantic compression. Prefix generation (Phase 1) is how I plan to implement this for text.

**Decision gate:** If any objective achieves rank > 64 AND STS-B > 0.70 → proceed to Track B. If all fail rank > 30 → implement Slot Attention as architectural fallback.

### Track B — Data Scaling

**Goal:** Scale from Minipile (0.6B tokens) to OpenWebText + Wikipedia (5B+ tokens) with the winning objective from Track A.

Data sources: `Skylion007/openwebtext`, `wikimedia/wikipedia`, fallback to `HuggingFaceFW/fineweb-edu`.

**Decision gate:** STS-B > 0.75 → proceed to audio. MNLI-m > 65% → ready for reasoning benchmarks.

### Track C — Architectural Innovations

**Goal:** Test variants that improve concept quality or enable reasoning.

- **Recursive Concept Encoder** (C1): Weight-tied encoder applied K times. 47% fewer params than standard. Code done, not trained.
- **Test-time compute scaling** (C3): Run more iterations at inference. No retraining needed.
- **Dimension Inversion** (C4): Token embedding dim = 32, concept dim = 512. Concentrates model capacity in concept space.
- **Slot Attention** (C5): Architectural fallback if Track A objectives all fail.

Inspired by: TRM (7M-param recursive model beats LLMs 1000× its size on ARC-AGI), Recurrent Depth (Geiping 2025 — test-time recurrence equals 103B-param equivalent), Coconut (Meta 2024 — latent reasoning outperforms CoT).

### Track D — Long-Context and Reasoning

**Goal:** Validate the efficiency advantage at N > 1K tokens. Demonstrate test-time compute scaling.

Targets: SCROLLS (1K–10K token documents), LongBench, HellaSwag, CommonsenseQA, ProntoQA. Long-term: ARC-AGI via visual adapter.

### Track E — Audio Modality

**Goal:** Map mel-spectrograms into the frozen text concept space. Build a "Concept-Talker."

**Gate:** Do not start until STS-B > 0.75 AND concept rank > 64/128 AND zero-shot STS-B > 0.60. The concept space must be proven semantically useful before mapping audio into it.

Reference architectures: Qwen2.5-Omni (Thinker-Talker), Moshi (full-duplex, inner monologue), SLAM (single-GPU recipe).

---

## Success Criteria

Before moving forward, I need:

- **Concept effective rank > 64/128** (50% of concept space utilized)
- **STS-B Pearson > 0.75** (semantic quality via fine-tuned decoder)
- **Zero-shot STS-B cosine similarity > 0.60** (ground truth of concept quality without fine-tuning)
- **Prefix generation suffix loss < 3.0** (concepts support generation, not just classification)

---

## Project Structure

```
MrCogito/
├── nn/                              # Core model implementations
│   ├── concept_encoder.py           # Perceiver MLM (primary)
│   ├── concept_encoder_recursive.py # Recursive (weight-tied) encoder
│   ├── concept_encoder_diffusion.py # Masked diffusion decoder
│   ├── concept_encoder_tsdae.py     # TSDAE denoising autoencoder
│   └── loss_manager.py             # VICReg + t_regs_mst concept losses
├── training/                        # Training scripts
│   ├── mlm_training.py             # MLM pretraining
│   ├── train_diffusion.py          # Diffusion decoder training
│   ├── train_tsdae.py              # TSDAE training
│   ├── evaluate_model_on_glue.py   # GLUE benchmark evaluation
│   └── utils_training.py           # Git info, WandB helpers
├── scripts/                         # Launch scripts
│   ├── train_diffusion_multigpu.sh  # Linux multi-GPU (Odra/Polonez)
│   ├── test_diffusion_local.ps1    # Windows local test run
│   └── ...
├── analysis/                        # Concept analysis tools
│   ├── check_model_health.py
│   └── analyze_concept_space.py
├── tests/                           # Unit tests
│   ├── test_loss_manager.py        # 26 tests (VICReg, t_regs_mst, warmup)
│   └── ...
├── docs/
│   ├── 1_Strategy_and_Plans/       # Roadmap, active TODOs
│   ├── 2_Experiments_Registry/     # master_experiment_log.md + run reports
│   ├── 3_Evaluations_and_Baselines/ # Canonical baselines
│   ├── 4_Research_Notes/           # Root cause analyses, diagnoses
│   └── 5_Archive/                  # Superseded roadmaps/plans
├── agent_memory/                    # Scratch files from AI-assisted sessions
├── CHANGELOG.md                    # Engineering log (what changed + why)
└── pyproject.toml                  # Poetry dependencies
```

---

## Setup

### Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/) for dependency management
- CUDA 12.8 (for GPU training)
- Windows 11 (local dev) or Linux (cluster training)

### Install

```powershell
git clone https://github.com/ksopyla/MrCogito.git
cd MrCogito
poetry install
```

### Verify

```powershell
poetry run python verification/torch_test.py
```

### Run Tests

```powershell
poetry run pytest tests/ -v
```

---

## Training

### Local (Windows, single GPU)

```powershell
# TSDAE training
poetry run python training/train_tsdae.py --model_type perceiver_mlm --hidden_size 512 --num_layers 6 --num_concepts 128

# Diffusion training
.\scripts\test_diffusion_local.ps1
```

### Cluster (Linux, multi-GPU)

```bash
# Diffusion training with VICReg + t_regs_mst regularization
bash scripts/train_diffusion_multigpu.sh
```

### Compute Infrastructure

- **Local**: RTX 3080 laptop (10GB VRAM) — for tests and quick experiments
- **Polonez**: 4× RTX 3090 — primary training cluster
- **Odra**: 3× RTX 3090 — secondary cluster

---

## Evaluation

All evaluations use ViaDecoder (fine-tuned lightweight decoder on top of frozen concepts) and are logged to WandB project `MrCogito`.

```powershell
# GLUE evaluation on a checkpoint
poetry run python evaluation/evaluate_model_on_glue.py \
  --model_path "Cache/Training/your_checkpoint" \
  --task mrpc
```

Results are saved to `Cache/Evaluation_reports/` and logged to `docs/2_Experiments_Registry/master_experiment_log.md`.

---

## Experiment Tracking

All training runs use [Weights & Biases](https://wandb.ai/):

- **Project**: `MrCogito`
- **Run naming**: `[dataset]-[task]-[model]`
- **Git info**: every run records `git describe --tags --always` in config

Every training run is also logged manually in `docs/2_Experiments_Registry/master_experiment_log.md` with the git tag, hyperparameters, and key results.

---

## Key References

Papers that most influenced the current design:

| Paper | Key finding for this project |
|---|---|
| [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) (Wang 2021) | Denoising autoencoder for sentence embeddings — 83× stronger gradient signal per concept |
| [SODA](https://openaccess.thecvf.com/content/CVPR2024/html/Hudson_SODA_Bottleneck_Diffusion_Models_for_Representation_Learning_CVPR_2024_paper.html) (Hudson, CVPR 2024) | Bottleneck model learns semantics only when decoder generates *related but different* content to input |
| [TRM](https://hf.co/papers/2510.04871) (Jolicoeur-Martineau 2025) | 7M-param recursive model beats LLMs 1000× its size on ARC-AGI |
| [Recurrent Depth](https://hf.co/papers/2502.05171) (Geiping 2025) | 3.5B model with test-time recurrence matches 103B equivalent |
| [Coconut](https://github.com/facebookresearch/coconut) (Meta 2024) | Latent chain-of-thought outperforms token-space CoT; curriculum from explicit to fully latent |
| [LLaDA](https://arxiv.org/abs/2502.09992) (Nie 2025) | Masked diffusion LLM at 8B scale; ELBO = weighted MLM losses with 1/t weighting |
| [MDLM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb0b13cc515724ab8015bc978fdde0ad-Abstract-Conference.html) (Sahoo, NeurIPS 2024) | Simplified ELBO for masked diffusion |
| [BiXT](https://arxiv.org/abs/2402.12138) (Hiller 2024) | Bidirectional cross-attention for Perceiver — fixes static token embeddings |
| [SimCSE](https://hf.co/papers/2104.08821) (Gao 2021) | Contrastive learning for sentence embeddings (+4pt STS-B) |
| [T-REGS MST](https://hf.co/papers/2510.23484) (Mordacq 2025) | MST-based regularization that detects and prevents dimensional collapse |
| [Large Concept Models](https://hf.co/papers/2412.08821) (Meta 2024) | Sentence-level concept prediction works for generation at scale |
| [Cramming 1568 Tokens](https://hf.co/papers/2502.13063) (2025) | 1500× compression is theoretically achievable |

---

## Honest Assessment

This is hard. I've been fighting concept collapse for weeks. The architecture is theoretically sound — cross-attention between concept tokens and input tokens is a well-established pattern (Perceiver, Flamingo, LCM) — but making the training objective produce genuinely diverse, semantically meaningful representations is non-trivial.

Things I've learned the hard way:
- **Self-reconstruction is not enough.** The model learns to hash surface tokens into concepts, not to extract semantics.
- **Regularization alone doesn't fix collapse.** If the task loss dominates, the regularizer gets ignored. If you weight regularizer too high, the task quality collapses.
- **The evaluation metric matters.** High GLUE scores with rank-5 concepts means the downstream head is doing all the work, not the concepts.

If I can get the concept space to be geometrically rich AND semantically useful, the efficiency story at long contexts becomes very compelling. That's the bet.

---

## Citation

If you use this work or find it useful, please cite:

```bibtex
@misc{mrcogito2025,
  title={MrCogito: Concept Bottleneck Encoder for Long-Context Reasoning},
  author={Sopyla, Krzysztof},
  year={2025},
  url={https://github.com/ksopyla/MrCogito}
}
```

---

## License

[MIT License](LICENSE)
