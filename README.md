# MrCogito

> **A research project building models that compress long context into latent _concept_ vectors, reason in that concept space, and decode back to text or other modalities.**

**Author:** Krzysztof Sopyla — [ai.ksopyla.com](https://ai.ksopyla.com) · [GitHub](https://github.com/ksopyla) · [LinkedIn](https://www.linkedin.com/in/krzysztof-sopyla/)
**Project page:** [ai.ksopyla.com/projects/concept-encoder](https://ai.ksopyla.com/projects/concept-encoder/)
**Experiments:** [wandb.ai/ksopyla/MrCogito](https://wandb.ai/ksopyla/MrCogito)

---

## Vision

Today's language models reason by expanding thought into text tokens. That works, but it is an extremely narrow channel: every intermediate idea has to be serialized into a vocabulary item, attended over again, and paid for again in the context window.

This project explores a different primitive:

> **Compress input into dense latent concepts, refine those concepts recursively, and generate output from the refined concept state.**

The end goal is a foundation-model architecture defined by four bold targets:

- **🧠 Reasoning in latent space.** The model should spend most of its compute refining continuous concept states, not just emitting visible chain-of-thought tokens.
- **📜 10M-token context as the north star.** Reached not by forcing full self-attention to scale forever, but by compressing long streams into a far smaller set of concept vectors.
- **🔀 One concept substrate across modalities.** Text, audio, and eventually other inputs map into the same reasoning space, with modality-specific adapters and decoders only at the edges.
- **🤝 A path to latent agent communication.** Once concept vectors are a reliable semantic interface, models can exchange concepts directly instead of talking through text. _That multi-agent layer lives outside this repository — but it is the foundation this work is meant to enable._

The core bet is simple: **if concept vectors can carry rich semantic and generative state, long-context reasoning becomes cheaper, more inspectable, and more composable than token-only reasoning.**

---

## Why This Project Exists

The frontier's answer to long context has been more of the same: better hardware, clever attention approximations, bigger clusters. I want a different bet — not making self-attention cheaper, but asking whether the model needs it at all for most of its reasoning.

A few older ideas I think were ahead of their time, and that I believe belong together:

- **Perceiver**[^perceiver] — a small set of latent queries can cross-attend to a huge input and decouple compute from sequence length. Better parameter and compute utilization through cross-attention — but used for perception, not as the seat of reasoning.
- **Embedding space is wildly underused.** A single dense vector can be made to hold ~1,500 tokens of recoverable text[^cramming], yet we still spend enormous capacity on giant vocabularies and wide token embeddings. I believe most of that capacity goes to surface form, not meaning — the latent space is the asset we're not exploiting.
- **Reasoning can move into latent space.** Looping the same weights[^ouro] and predicting continuous vectors instead of discrete tokens[^calm] let a model "think" beyond the token channel and punch above its parameter count[^huginn].

My bet is that **combining these into one end-to-end trained architecture** — a cross-attention concept bottleneck plus a recursive reasoning core operating in latent space — opens a different paradigm than the scaling race. Cross-attention bottlenecks themselves aren't new (Flamingo[^flamingo], Large Concept Models[^lcm]), but they're treated as a means to an end, not as the **primary locus of reasoning**. The question I keep asking is sharper:

> **Can a small set of concept vectors become the model's working memory — the thing it actually reasons over?**

If yes, the payoff compounds: `O(C·N)` long context with `C ≪ N`, test-time compute scaling by running more refinement steps without retraining, modality-agnostic reasoning over a shared concept space, and an internal state small enough to actually inspect. It also opens a channel the token interface can't: **letting agents talk to each other in concept space instead of text** — early work shows passing latent state between models is both faster and more accurate than exchanging tokens[^latentmas][^interlat]. It may not work — it's a genuine research bet — but I think it's the more interesting direction, and the one the scaling race is leaving unexplored.

---

## Architecture

The architecture follows an **encode → reason → decode** pattern, with the concept space at its center:

```text
Input tokens / frames  (N)
        │
        ▼
   Encoder            cross-attention compresses N inputs into C concepts   (C ≪ N)
        │
        ▼
 Reasoning core       optional recursive refinement over the concepts
        │
        ▼
   Decoder            generates text, speech, or another modality from concepts
```

A small set of learned **concept queries** cross-attends to the full input sequence. This shifts the dominant attention pattern from `O(N²)` token-to-token attention to `O(C·N)` concept-to-token attention — and the concept count `C` grows far more slowly than the input length `N`.

| Input length `N` | Concepts `C` | Attention savings vs. full self-attention |
|---|---|---|
| 512 | 128 | ~4× |
| 32K | 2K | ~16× |
| 1M | 8K | ~128× |
| **10M** | hierarchical | **north-star target** — needs hierarchical concepts, recurrence, and memory |

This is not just an efficiency trick. The bottleneck is meant to **force abstraction**: concepts become the model's working memory, the object reasoning operates on, and the interface decoders read from.

---

## How the Project Works

This is an **active research project**: the north star is fixed, but the route is discovered one milestone at a time. Each milestone is a hypothesis, run as a small experiment with explicit success and kill criteria — and almost everything downstream can change once the first real training run is evaluated.

- **The first milestone is concept quality:** find a training objective *and* architecture that form a quality concept bottleneck — semantically meaningful, geometrically diverse, and actually used by the decoder.
- **The next milestone is recursive reasoning over those concepts:** the broad idea is to refine concepts with a shared block applied multiple times, but the details are deliberately open and will be shaped by what the first experiments reveal.

The day-to-day focus moves quickly, so it is not pinned here. The living source of truth is the agenda and the active experiment spec:

- **Live agenda:** [`docs/1_Strategy_and_Plans/agenda.md`](docs/1_Strategy_and_Plans/agenda.md)
- **Active experiment:** [`docs/experiments_specs/`](docs/experiments_specs/)
- **Run ledger:** [`docs/2_Experiments_Registry/master_experiment_log.md`](docs/2_Experiments_Registry/master_experiment_log.md)

---

## What Has Been Learned

The project has already run many small-scale experiments. The useful lesson is not "the old model worked" — it is the opposite: easy objectives exposed exactly *where* concept bottlenecks fail.

- **Self-reconstruction is the wrong pressure.** When the encoder sees the same content the decoder reconstructs, the system learns positional or surface shortcuts instead of semantics.
- **Diversity alone is not meaning.** Some regularizers raise effective rank while damaging downstream semantics.
- **Parallel decoders are not enough.** Position-only reconstruction can train a useful probe, but it does not prove the model can *generate*.
- **Bidirectional token ↔ concept interaction matters.** The token side and concept side must evolve together; static token embeddings leave the bottleneck too weak.
- **Concept collapse is measurable.** Effective rank, pairwise concept cosine, STS-B, and concept-ablation loss are all tracked, because loss curves alone can be misleading.

The method is deliberately incremental: **one experiment, one changed variable, explicit success and kill criteria.** Past runs are treated as evidence that improved understanding — not as wins or losses.

---

## Long-Term Roadmap

The path is genuinely open, but the direction is stable:

| Stage | Goal |
|---|---|
| 1 · **Concept quality** | Concepts that are semantically rich, geometrically non-collapsed, and useful for generation. |
| 2 · **Concept-conditioned generation** | Move from representation probes to real AR or diffusion generation from concepts. |
| 3 · **Instruction following** | Encode instructions into concepts and generate useful responses through the bottleneck. |
| 4 · **Recursive latent reasoning** | Apply a shared reasoning block repeatedly over concepts, with more refinement steps available at inference time. |
| 5 · **Long context** | Scale length through concept compression, memory, and curricula — with **10M tokens** as the north-star target. |
| 6 · **Audio-native reasoning** | Map speech into concepts, reason without mandatory text round-tripping, and decode back to speech. |

The eventual audio path:

```text
User speech  →  audio adapter  →  concept space  →  recursive refinement  →  talker / decoder  →  spoken response
```

Text is the first proving ground because it gives fast iteration, mature datasets, and clear evaluation. The larger ambition is a **modality-agnostic concept space**.

---

## Repository Guide

```text
.
├── nn/                         # Core PyTorch model components
│   ├── concept_encoder.py        # Shared config, encoder, BiXT-style blocks
│   ├── concept_encoder_perceiver.py
│   ├── concept_encoder_weighted.py
│   ├── concept_encoder_recursive_mlm.py
│   ├── concept_losses.py
│   └── loss_manager.py
├── training/                   # Training entrypoints and shared utilities
│   ├── train_perceiver_denoise.py
│   ├── train_mlm.py
│   ├── train_prefix_diffusion.py
│   └── utils_training.py
├── evaluation/                 # GLUE, STS-B, PAWS/SICK, checkpoint routing
├── analysis/                   # Concept-rank and geometry analysis
├── scripts/                    # Local and multi-GPU launch scripts
├── docs/
│   ├── 1_Strategy_and_Plans/     # Current agenda and long-term vision
│   ├── experiments/             # Frozen experiment specs and plans
│   ├── 2_Experiments_Registry/   # Run ledger and reports
│   ├── 3_Evaluations_and_Baselines/
│   ├── 4_Research_Notes/
│   └── 5_Archive/               # Historical plans (not current truth)
├── parked/                     # Revivable but inactive experiment families
├── tests/
├── verification/
├── CHANGELOG.md
└── pyproject.toml              # uv / PEP 621 dependencies
```

The foundation is shared on purpose: new experiments are **config-selectable extensions** over the common code, not one-off training forks.

---

## Setup

**Prerequisites**

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- CUDA for serious training; macOS CPU/MPS is suitable for smoke tests only

**Install**

```bash
git clone https://github.com/ksopyla/MrCogito.git
cd MrCogito
uv sync
```

**Verify the environment**

```bash
uv run python verification/torch_test.py
uv run pytest tests/ -v
```

---

## Training and Evaluation

Main maintained training entrypoint:

```bash
uv run python training/train_perceiver_denoise.py \
  --hidden_size 512 \
  --num_hidden_layers 6 \
  --concept_num 128
```

Remote multi-GPU launchers live in `scripts/`, for example:

```bash
bash scripts/train_perceiver_denoise_multigpu.sh
```

Evaluate a checkpoint:

```bash
uv run python evaluation/evaluate_model_on_glue.py \
  --model_path "Cache/Training/your_checkpoint" \
  --task mrpc
```

Every serious run is logged to a **public [Weights & Biases project](https://wandb.ai/ksopyla/MrCogito)** — anyone can follow the work live, with full hyperparameters, git commit, losses, concept metrics, and evaluation results. Human-readable conclusions live in the experiment registry.

---

## Influences

This work sits at the intersection of several converging lines of research:

- **Perceiver / Perceiver IO** — cross-attention bottlenecks as a general encode-reason-decode skeleton.
- **Flamingo, BLIP-2** — learned query bottlenecks that condition strong decoders.
- **SODA** — representation learning through bottleneck *generation*, not trivial self-reconstruction.
- **BiXT** — bidirectional token ↔ concept interaction.
- **Large Concept Models, SONAR-LLM** — generation and reasoning above the token level.
- **Coconut & latent chain-of-thought** — reasoning in continuous hidden states instead of only text.
- **Recurrent / recursive transformers** — test-time compute scaling through repeated refinement.
- **Latent multi-agent communication** — the future direction where concept vectors become the channel between cooperating models.

These are influences, not dependencies. The research question is whether a compact concept state can become the **primary working memory** of a generative model.

---

## Status

This is an active research repository — public, MIT-licensed, and intentionally transparent about negative results. The aim is not to polish a benchmark number, but to prove whether a concept bottleneck can support generation, and then reasoning.

- **Long-term vision:** [`docs/1_Strategy_and_Plans/vision_and_goals.md`](docs/1_Strategy_and_Plans/vision_and_goals.md)
- **Live agenda & experiments:** [`docs/1_Strategy_and_Plans/agenda.md`](docs/1_Strategy_and_Plans/agenda.md) · [`docs/experiments_specs/`](docs/experiments_specs/)
- **Live experiment tracking:** the open [Weights & Biases project](https://wandb.ai/ksopyla/MrCogito) — every run, public.
- Historical diffusion and recursive branches are parked, not discarded.

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

[^perceiver]: Jaegle et al., *Perceiver: General Perception with Iterative Attention*, ICML 2021 — <https://arxiv.org/abs/2103.03206>. See also *Perceiver IO*, ICLR 2022 — <https://arxiv.org/abs/2107.14795>.
[^cramming]: Kuratov et al., *Cramming 1568 Tokens into a Single Vector and Back Again*, ACL 2025 — <https://arxiv.org/abs/2502.13063>.
[^ouro]: Ouro team, *Ouro: Looped Language Models*, 2025 — <https://arxiv.org/abs/2510.25741>.
[^calm]: Tencent & Tsinghua, *CALM: Continuous Autoregressive Language Models*, 2025 — <https://arxiv.org/abs/2510.27688>.
[^huginn]: Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning (Huginn)*, NeurIPS 2025 — <https://arxiv.org/abs/2502.05171>.
[^flamingo]: Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022 — <https://arxiv.org/abs/2204.14198>.
[^lcm]: LCM team (Meta), *Large Concept Models: Language Modeling in a Sentence Representation Space*, 2024 — <https://arxiv.org/abs/2412.08821>.
[^latentmas]: Princeton, UIUC & Stanford, *LatentMAS: Multi-Agent Collaboration in Latent Space*, 2025 — <https://arxiv.org/abs/2511.20639>.
[^interlat]: Zhejiang & Alibaba, *Interlat: Inter-Agent Communication in Latent Space*, ACL 2026 — <https://arxiv.org/abs/2511.09149>.
