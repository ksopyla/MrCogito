# MrCogito

> What if a transformer didn't attend over every token, but instead compressed long sequences into a small number of dense "concept tokens" and reasoned from there?

**Author:** Krzysztof Sopyla -- [ai.ksopyla.com](https://ai.ksopyla.com) | [GitHub](https://github.com/ksopyla) | [LinkedIn](https://www.linkedin.com/in/krzysztof-sopyla/)
**Project page:** [ai.ksopyla.com/projects/concept-encoder](https://ai.ksopyla.com/projects/concept-encoder/)
**Experiments:** [wandb.ai/ksopyla/MrCogito](https://wandb.ai/ksopyla/MrCogito)

---

## Why This Project Exists

The standard transformer is remarkable, but self-attention is O(N^2). At 128K tokens the attention matrix is enormous. At 1M tokens it is computationally intractable. The field's answer so far has been better hardware, clever approximations, and bigger clusters.

I want to explore a different direction.

Instead of making self-attention cheaper, what if the model did not need it at all for most of its reasoning? What if it compressed the input into a compact semantic state and then operated on that?

This is not a new idea in isolation -- Perceivers, Flamingo, and Meta's Large Concept Models all use forms of cross-attention bottlenecks. But most of those systems treat the bottleneck as a means to an end (efficiency or multimodal fusion), not as the primary locus of reasoning.

MrCogito asks: **can a small set of concept tokens become the model's working memory?**

If the answer is yes, that opens several doors at once:

- **Efficient long-context processing.** O(C*N) instead of O(N^2), where C is the number of concepts and C << N.
- **Test-time compute scaling.** A recursive, weight-tied encoder can run more iterations at inference to "think harder" without retraining.
- **Modality-agnostic reasoning.** If different inputs (text, audio) map into the same concept space, the reasoning module does not need to know the modality.
- **Interpretable internal states.** A small concept bank is easier to inspect than 32K hidden states.

The bet is straightforward: if I can make the concept bottleneck carry real semantic content, the rest of the architecture becomes simpler, cheaper, and more composable.

---

## The Core Architecture

```
Input text (N tokens)
  --> Encoder: cross-attention compresses N tokens into C concept tokens
  --> Reasoning: recursive concept refinement (K iterations, weight-tied)
  --> Decoder: generates text output from refined concepts
```

The encoder uses cross-attention between a small set of learned concept tokens (C) and the full input sequence (N). This produces a compact representation in concept space:

| Sequence length N | Concepts C | Compression | Self-attn O(N^2) | Concept O(C*N) | Speedup |
|---|---|---|---|---|---|
| 512 | 128 | 4:1 | 262K | 65K | 4x |
| 4,096 | 512 | 8:1 | 16.7M | 2.1M | **8x** |
| 32,768 | 2,048 | 16:1 | 1.07B | 67M | **16x** |
| 1,048,576 | 8,192 | 128:1 | 1.1T | 8.6B | **128x** |

At 1M tokens, full self-attention is impossible on any current hardware. Concept attention with C=8K remains tractable -- while forcing the model to produce increasingly abstract, semantic representations.

The current model is small by design: **~21M parameters** in the Micro-2 configuration. This is deliberate. I want to prove the architecture works before scaling, not prove that enough parameters can brute-force any objective.

---

## The End Vision

The long-term goal is an **audio conversational and reasoning model** grounded in a concept bottleneck:

```
User speech (mel-spectrogram)
  --> Audio adapter: maps audio features into concept space
  --> Reasoning: recursive concept refinement (shared weights with text)
  --> Audio decoder (Talker): generates speech tokens from concepts
```

Text and audio would share the same concept space and the same reasoning module. The only modality-specific parts are the adapter and the decoder.

This vision has six phases, each gated by concrete success criteria:

| Phase | What | Gate |
|---|---|---|
| 1 | Prove concept bottleneck captures semantics | STS-B > 0.70, concept rank > 64/128 |
| 2 | Stronger representations, data scaling, architecture variants | STS-B > 0.75, prefix loss < 3.0 |
| 3 | Full text generation from concepts | Coherent multi-sentence output |
| 4 | Instruction fine-tuning (SFT) | AlpacaEval, MT-Bench |
| 5 | Recursive reasoning, test-time compute scaling | K=12 beats K=6 on reasoning benchmarks |
| 6 | Audio modality (Concept-Talker) | Speech-to-concept-to-speech working |

**Current state (March 2026):** Phase 1 -- fighting concept collapse. All downstream phases depend on solving this.

---

## What I Have Learned So Far

### The Concept Collapse Problem

The best checkpoint so far reaches decent downstream numbers: MRPC = 82.7%, STS-B = 0.650, QQP = 73.4%. Those look reasonable for a 21M parameter model.

But the internal geometry tells a different story. I allocated 128 concept slots, and the model uses effectively **5 of them**. That is 4% utilization. The concepts have collapsed into a low-dimensional subspace.

This is the core problem. High downstream scores with collapsed concepts means the evaluation head is doing all the work, not the concept space. And a collapsed concept space cannot support generation, reasoning, or modality transfer.

### What Failed

I document failures because I think they are as instructive as successes.

| Approach | What happened |
|---|---|
| Combined loss + Kendall-Gal weighting | Concept rank jumped to 95%, but all semantic metrics collapsed -- concepts were diverse but empty |
| Combined loss + fixed weight | Rank stuck at 12%, everything regressed |
| CLS-query classification head | 128:1 information collapse -- a single query flattens all concept structure |
| Diffusion L2 self-reconstruction | Rank 2x better but STS-B near-random (0.138) |
| Deep diffusion + ELBO + VICReg | Rank barely moved (5.74/128), STS-B 0.174 |
| Prefix generation v1 (without BiXT) | Rank 6.19/128, STS-B 0.337 -- better direction but underpowered |

### What I Believe Now

**The root insight:** Self-reconstruction (feed in X, reconstruct X) teaches the model to build a positional hash function, not to extract semantics. The decoder needs to generate content the encoder never saw. This is the SODA principle (Hudson et al., 2024).

- Self-reconstruction through a bottleneck optimizes the wrong information path
- Prefix-conditioned generation (encode prefix, decode suffix) is the most promising current direction
- BiXT (bidirectional cross-attention) is no longer optional -- the token side needs to evolve alongside concepts
- Geometric diversity without semantic content is not a win
- Regularization alone cannot fix collapse if the training objective rewards the wrong shortcut

Root cause analyses live in `docs/4_Research_Notes/`. A fuller account of this diagnostic journey: [Quicker Failures Lead to Better Questions](https://ai.ksopyla.com/posts/quicker-failures-better-questions/).

---

## Research Tracks

The work is organized into parallel tracks, with Track A as the critical path.

**Track A -- Fix Concept Quality.** Find the training objective that produces concept rank > 64/128 AND STS-B > 0.70. Current candidates: TSDAE denoising, prefix generation (encode prefix, decode suffix), masked diffusion with ELBO fixes. If all fail rank > 30, Slot Attention is the architectural fallback.

**Track B -- Data Scaling.** Scale from Minipile (0.6B tokens) to OpenWebText + Wikipedia (5B+ tokens) with the winning objective from Track A.

**Track C -- Architectural Innovations.** Recursive concept encoder (weight-tied, 47% fewer params), dimension inversion (token_dim=32, concept_dim=512), test-time compute scaling (more iterations at inference without retraining).

**Track D -- Text Generation.** Transition from reconstruction to full text generation from concepts via diffusion or autoregressive decoders.

**Track E -- Long-Context.** Validate the efficiency advantage on sequences > 1K tokens (SCROLLS, LongBench).

**Tracks F, G, H -- SFT, Reasoning, Audio.** Future phases, gated on concept quality and generation working first.

Full roadmap: [`docs/1_Strategy_and_Plans/roadmap.md`](docs/1_Strategy_and_Plans/roadmap.md)

---

## Architecture Variants

Four decoder approaches are implemented, each testing a different hypothesis about how to get semantic content into concept space:

| Variant | Module | Training script | Idea |
|---|---|---|---|
| **Perceiver Denoise** | `nn/concept_encoder_perceiver.py` | `training/train_perceiver_denoise.py` | TSDAE denoising autoencoder with BiXT and position-only decoder |
| **Weighted MLM** | `nn/concept_encoder_weighted.py` | `training/train_mlm.py` | Weighted concept pooling + masked language modeling |
| **Recursive MLM** | `nn/concept_encoder_recursive.py` | `training/train_recursive_mlm.py` | Weight-tied encoder applied K times (TRM-inspired) |
| **Diffusion** | `nn/concept_encoder_diffusion.py` | `training/train_diffusion.py` | Masked diffusion decoder with AdaLN-Zero |
| **Prefix Diffusion** | `nn/concept_encoder_diffusion.py` | `training/train_prefix_diffusion.py` | Encode prefix, generate suffix via diffusion (SODA-inspired) |

The maintained primary path is `perceiver_denoise`. Other variants are active experiments or baselines.

---

## Project Structure

```
MrCogito/
|-- nn/                                 # Core model implementations
|   |-- concept_encoder.py              # Shared encoder config and core blocks
|   |-- concept_encoder_perceiver.py    # Perceiver denoising + ViaDecoder models
|   |-- concept_encoder_recursive.py    # Recursive (weight-tied) encoder
|   |-- concept_encoder_diffusion.py    # Masked diffusion decoder
|   |-- concept_encoder_weighted.py     # Weighted MLM decoder
|   |-- loss_manager.py                 # VICReg + t_regs_mst concept losses
|   +-- concept_losses.py              # Loss function implementations
|-- training/                           # Training scripts
|   |-- train_perceiver_denoise.py      # Canonical perceiver denoising
|   |-- train_mlm.py                    # Weighted MLM baseline
|   |-- train_recursive_mlm.py          # Isolated recursive experiment
|   |-- train_diffusion.py              # Diffusion decoder training
|   |-- train_prefix_diffusion.py       # Prefix-conditioned diffusion
|   +-- utils_training.py              # Shared logging, WandB, git helpers
|-- evaluation/                         # Benchmark evaluation
|   |-- evaluate_model_on_glue.py       # GLUE benchmark evaluation
|   |-- evaluate_on_benchmark.py        # STS-B zero-shot, SICK, PAWS
|   +-- concept_eval_routing.py        # Checkpoint-driven evaluator routing
|-- analysis/                           # Concept space analysis tools
|-- tests/                              # Unit tests
|-- scripts/                            # Launch scripts (Windows + Linux)
|-- docs/
|   |-- 1_Strategy_and_Plans/           # Roadmap, active TODOs
|   |-- 2_Experiments_Registry/         # Master experiment log + run reports
|   |-- 3_Evaluations_and_Baselines/    # Canonical baselines
|   |-- 4_Research_Notes/               # Root cause analyses, diagnoses
|   +-- 5_Archive/                     # Superseded roadmaps/plans
|-- CHANGELOG.md                        # Engineering log (what changed + why)
+-- pyproject.toml                     # uv / PEP 621 dependencies
```

---

## Setup

### Prerequisites

- Python 3.12 (managed by [uv](https://docs.astral.sh/uv/))
- [uv](https://docs.astral.sh/uv/) for dependency / environment management
- CUDA 12.8 for GPU training (Linux / Windows). On macOS the project installs CPU/MPS wheels automatically.

### Install

```bash
git clone https://github.com/ksopyla/MrCogito.git
cd MrCogito
uv sync
```

### Verify

```bash
uv run python verification/torch_test.py
```

### Run Tests

```bash
uv run pytest tests/ -v
```

---

## Training

### Local (single GPU)

```powershell
# Perceiver denoising (canonical path)
poetry run python training/train_perceiver_denoise.py --hidden_size 512 --num_hidden_layers 6 --concept_num 128

# Smoke test
.\scripts\test_perceiver_denoise_local.ps1
```

### Cluster (multi-GPU via DDP)

```bash
bash scripts/train_perceiver_denoise_multigpu.sh
bash scripts/train_diffusion_multigpu.sh
```

### Compute

| Name | Hardware | Role |
|---|---|---|
| Local | RTX 3080 laptop (10 GB VRAM) | Smoke tests, debugging |
| **Polonez** | 4x RTX 3090 (24 GB each) | Primary training cluster |
| **Odra** | 3x RTX 3090 (24 GB each) | Secondary cluster, parallel experiments |

This is modest hardware by industry standards. But it is enough for Phase 1-2 experiments on the Minipile dataset (0.6B tokens).

---

## Evaluation

All evaluations use ViaDecoder (fine-tuned lightweight decoder on top of frozen concepts) unless otherwise noted.

```bash
poetry run python evaluation/evaluate_model_on_glue.py \
  --model_path "Cache/Training/your_checkpoint" \
  --task mrpc
```

Every training run is logged to [Weights & Biases](https://wandb.ai/ksopyla/MrCogito) with full hyperparameters, git commit hash, training curves, concept analysis metrics, and GLUE results. A `master_experiment_log.md` in the repo carries human-written summaries: what I tried, what happened, and what I concluded.

---

## Key Papers and Influences

| Paper | Key contribution to MrCogito |
|---|---|
| [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/) (Wang 2021) | Denoising autoencoder -- 83x stronger gradient signal per concept vs sparse MLM |
| [SODA](https://openaccess.thecvf.com/content/CVPR2024/html/Hudson_SODA_Bottleneck_Diffusion_Models_for_Representation_Learning_CVPR_2024_paper.html) (Hudson, CVPR 2024) | Bottleneck model learns semantics only when decoder generates different content than encoder saw |
| [TRM](https://hf.co/papers/2510.04871) (Jolicoeur-Martineau 2025) | 7M-param recursive model beats LLMs 1000x its size on ARC-AGI |
| [Recurrent Depth](https://hf.co/papers/2502.05171) (Geiping 2025) | Test-time recurrence -- 3.5B model matches 103B equivalent |
| [Coconut](https://github.com/facebookresearch/coconut) (Meta 2024) | Latent chain-of-thought outperforms token-space CoT |
| [BiXT](https://arxiv.org/abs/2402.12138) (Hiller 2024) | Bidirectional cross-attention fixes static token embeddings in Perceivers |
| [Large Concept Models](https://hf.co/papers/2412.08821) (Meta 2024) | Sentence-level concept prediction works for generation at scale |
| [LLaDA](https://arxiv.org/abs/2502.09992) (Nie 2025) | Masked diffusion language model at 8B scale -- validates diffusion for text |
| [SimCSE](https://hf.co/papers/2104.08821) (Gao 2021) | Contrastive learning for sentence embeddings |
| [T-REGS MST](https://hf.co/papers/2510.23484) (Mordacq 2025) | MST-based regularization that detects and prevents dimensional collapse |

---

## FAQ

**"Why not just fine-tune an existing model?"**
Because the research question is about the architecture, not the task. I want to know whether concept bottlenecks can work as the primary representation, not whether I can get good STS-B scores (there are easier ways).

**"Why so small? 21M parameters is tiny."**
Deliberately. If the architecture cannot produce good concepts at 21M, scaling will not fix the fundamental problem. If it can, scaling will make it better. Small models also mean faster iteration and lower compute costs -- important for a solo researcher.

**"What happens if concept collapse cannot be solved?"**
I have a fallback path (Slot Attention) and a pivot option (decoder-only with concept conditioning). But I believe the current diagnostic work points to solvable problems -- wrong training objective, not wrong architecture.

---

## Updates

- **2026-03-08** -- Perceiver V2 denoising reset: canonical denoising stack, checkpoint-declared evaluation routing, retired legacy perceiver MLM interfaces.
- **2026-03-08** -- Published [Quicker Failures Lead to Better Questions](https://ai.ksopyla.com/posts/quicker-failures-better-questions/): diagnostic journey through concept collapse.
- **2026-03-07** -- Prefix diffusion v2 hardening: BiXT-only, sentence-boundary splits, evaluation contracts.
- **2026-02-21** -- Architecture overhaul: BiXT, TSDAE, PosOnly decoder, ViaDecoder evaluation, VICReg + t_regs_mst regularization.
- **2026-02-08** -- Best baseline checkpoint: Perceiver MLM L6, 40 epochs on Minipile. MRPC 82.7%, STS-B 0.650, concept rank 5/128.

---

## Citation

```bibtex
@misc{mrcogito2025,
  title   = {MrCogito: Concept Bottleneck Encoder for Long-Context Reasoning},
  author  = {Sopyla, Krzysztof},
  year    = {2025},
  url     = {https://github.com/ksopyla/MrCogito}
}
```

## License

[MIT License](LICENSE)

---

This is an open research project. The repo is public and MIT-licensed. If you find this work interesting, the most useful things you can do are: read the code, open an issue, or tell me what I am getting wrong.
