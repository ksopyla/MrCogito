# MrCogito — Evaluation Protocol (canonical)

**Updated:** 2026-06-16 · The single reference for *what we measure, why, and when*. Sister docs:
`experiment-evaluate` skill (the *how to run*), `engineering_specs/concept_information_eval_upgrade.md`
(the change that introduced the research tiers), `canonical_baselines.md` (reference numbers).

## Purpose
Decide whether the **concept bottleneck** stores meaningful, distributed, useful information and
supports compression / reasoning / generation — without fooling ourselves. The recurring project
failure is *underspecified measurement* (train/eval mismatch, readouts that bypass the bottleneck,
metrics that reward shortcuts), so this protocol is built around three rules.

## Three rules (apply to every tier)
1. **Separate the two axes.** *Faithfulness* = does information survive the bottleneck?
   *Utility* = is the surviving information usable? Never quote one as the other.
2. **No-bypass readouts.** Probe with a **frozen encoder + a control baseline + a necessity ablation**
   (zero/shuffle concepts; later zero loops). A trainable head over an unfrozen encoder measures the
   head, not the concepts.
3. **Stage by maturity.** Long-context and reasoning tiers are noise on collapsed / incoherent
   checkpoints. Each tier has a *gate*; do not run a tier before its gate opens.

---

## Two separate evaluation programs
- **Research track (Tiers L0–L4):** direction-finding. Architecture-aware probes that tell us *what to
  fix next*. This is the daily driver.
- **External-comparability track (lighteval):** leaderboard-style comparison to public models. Tells us
  *where we stand*, not what to fix. **Deferred** until the backbone is proven and we scale (see below).

---

## Research track — tiers

| Tier | What it measures | Axis | Gate to run | When we use it | Status |
|---|---|---|---|---|---|
| **L0 Geometry** | within-sample concept RankMe, anti-collapse, active-slot fraction | faithfulness-precursor | always (seconds) | every checkpoint, automatically | **built** |
| **L1 Generation faithfulness** | concept-ablation ΔCE; round-trip token recovery (teacher-forced + free-running); latent specificity `z(x)` vs `z(x')` (acc-drop + symmetric-KL); active-slot fraction | faithfulness | concepts non-trivial (post-training) | every serious run; the primary "do concepts carry & get used" signal now | **built** |
| **L2 Representation utility** | mean-vs-attention pool + STS-B floors (built); SentEval probing + control-task selectivity, MTEB STS/clustering/retrieval (deferred remote slice) | utility | L0 healthy | when claiming concepts hold *useful* info; comparing checkpoints' representation quality | **partly built** |
| **L3 Compression / long context** | reconstruction-vs-compression-ratio curve (built, reuses L1 round-trip); held-out-segment PPL & small-scale recall (RMT / RULER-MQAR) deferred — need >512 context | both | de-collapse **and** coherent reconstruction (+ context-length extension for >512) | when testing the compression vision / long-context experiments (E05+) | **harness built; gated to run** |
| **L4 Reasoning depth** | depth-controlled synthetics (p-hop, CLUTRR k-hop, ListOps depth, k-hop QA, graph-by-diameter) as **accuracy-vs-depth curves** + necessity ablation + iso-FLOP/iso-param baselines | utility | L3 + a recursion/latent-reasoning mechanism (E08) | when claiming latent/depth reasoning; the falsifiable depth curve | **gated / design** |

### What each tier means and when to run it
- **L0 — Geometry (always).** Cheap health/collapse check on the concept set. Runs on every checkpoint;
  a failed L0 (collapsed, dead slots) means higher tiers are not yet worth running.
- **L1 — Generation faithfulness (now, every run).** The honest core for *does the decoder use the
  concepts and can they be decoded back*. Reuses the AR model + FineWeb-Edu; no external data. This is
  the current primary research signal while we attack collapse. *Implemented* in
  `analysis/concept_generation_eval.py` (round-trip recovery, latent specificity), wired into
  `analysis/run_concept_analysis.py` (`--generation_eval`). Note: KL/MI posterior-collapse metrics from
  the VAE literature do **not** apply — our concepts are a deterministic encoding, not a sampled latent;
  active-slot fraction + specificity play that role here.
- **L2 — Representation utility (now, for representation claims).** The direct **GLUE replacement**:
  frozen-encoder probes (linear + attention-pool, with selectivity controls) + zero-training MTEB
  STS/clustering/retrieval. Answers "is the surviving info usable, and is it distributed across slots?"
- **L3 — Compression / long context (after de-collapse).** The first tier that tests the actual vision
  (compress long sequences into concepts). The **reconstruction-vs-compression-ratio curve is
  implemented** (it reuses the L1 round-trip primitive, bucketed by `⌈seq_len/C⌉`); segment-PPL and
  downstream recall (RULER/NIAH/RMT) are deferred custom tasks that need a context-length extension
  beyond the current 512.
- **L4 — Reasoning depth (after a recursion mechanism, E08).** Tests whether reasoning flows through the
  bottleneck. Only meaningful as a **depth curve with a necessity ablation** — if accuracy is unchanged
  when concepts/loops are zeroed, the depth claim is dead. Always paired with iso-FLOP and iso-param
  baselines (the standard latent-reasoning failure mode is gains that vanish under ablation).

---

## External-comparability track — lighteval (deferred)
- **What it is:** Hugging Face `lighteval` — a harness with a `custom`-model backend, a 1000+ task
  catalog (MMLU, HellaSwag, ARC, GSM8K, MATH, BBH, RULER, …) and few-shot prompting. It runs a model as
  a standard LM (`loglikelihood` + `generate`).
- **What it answers:** "Is our concept-conditioned model a *competent general LM* vs public baselines?"
  — comparability, **not** concept-specific evidence (short prompts + MC loglikelihood can bypass the
  bottleneck; same caveat as full-finetune GLUE).
- **One-time enabler:** a lighteval **custom-model adapter** mapping `prompt → encoder → concepts →
  decoder` (concept-conditioned scoring/generation). The decoder-as-plain-LM shortcut is forbidden — it
  bypasses the bottleneck. Every lighteval run is paired with a **zero-concept control**.
- **When we use it (not now):** only **after the backbone is established** (concepts + reasoning + a
  chosen AR/diffusion decoder proven on L1–L4) and we **scale up**. Then run size-appropriate lists:
  - **SmolLM2-135M subset** (HellaSwag, ARC-easy, PIQA, OpenBookQA, WinoGrande, CommonsenseQA, BoolQ) for
    small (~135M) models;
  - **SmolLM3 list** (adds MMLU-Pro, MATH, HumanEval+/MBPP+, RULER 32k–128k) for **1B–3B** models.
- **Why deferred:** at 135M from-scratch on 512-token context, most SmolLM3-scale tasks are near-chance
  and RULER needs long context — the numbers would not inform direction.

---

## When do we run what (summary)
- **Every checkpoint:** L0.
- **Now (collapse era, E01–E0x):** L0 + L1 + L2 — find the direction.
- **After de-collapse + coherent generation:** add L3 (compression / long context).
- **After a recursion mechanism (E08):** add L4 (reasoning depth).
- **After the backbone is proven, on scale-up to 1B–3B:** turn on the lighteval comparability track
  (SmolLM2-135M subset for small models, SmolLM3 list for 1B–3B).

---

## References

**Geometry / collapse (L0)**
- RankMe — Garrido et al., ICML 2023, [arXiv:2210.02885](https://arxiv.org/abs/2210.02885)
- Unsupervised embedding-quality metrics — Tsitsulin et al., ICML 2023, [arXiv:2305.16562](https://arxiv.org/abs/2305.16562)

**Generation faithfulness / latent usage (L1)**
- Sentence VAE & posterior collapse — Bowman et al., CoNLL 2016, [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)
- Skip-VAE KL/MI/active-units — Dieng et al., AISTATS 2019, [arXiv:1807.04863](https://arxiv.org/abs/1807.04863)
- Optimus (latent + AR decoder, AU/MI) — Li et al., EMNLP 2020, [arXiv:2004.04092](https://arxiv.org/abs/2004.04092)
- Vec2Text round-trip recovery — Morris et al., EMNLP 2023, [arXiv:2306.05443](https://arxiv.org/abs/2306.05443)
- CALM (round-trip / BrierLM) — 2025, [arXiv:2510.27688](https://arxiv.org/abs/2510.27688)
- LD4LG (MAUVE/Gen-PPL/diversity/memorization) — NeurIPS 2023, [arXiv:2212.09462](https://arxiv.org/abs/2212.09462)
- MAUVE — Pillutla et al., NeurIPS 2021, [arXiv:2102.01454](https://arxiv.org/abs/2102.01454)
- Classifier-free guidance (conditioning dropout) — Ho & Salimans, 2022, [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)

**Representation utility / probing (L2)**
- SentEval — Conneau & Kiela, LREC 2018, [arXiv:1803.05449](https://arxiv.org/abs/1803.05449); probing tasks [arXiv:1805.01070](https://arxiv.org/abs/1805.01070)
- Control tasks & selectivity — Hewitt & Liang, EMNLP 2019, [arXiv:1909.03368](https://arxiv.org/abs/1909.03368)
- MDL probing — Voita & Titov, EMNLP 2020, [arXiv:2003.12298](https://arxiv.org/abs/2003.12298)
- MTEB — Muennighoff et al., EACL 2023, [arXiv:2210.07316](https://arxiv.org/abs/2210.07316); MMTEB [arXiv:2502.13595](https://arxiv.org/abs/2502.13595)
- BEIR (zero-shot retrieval) — Thakur et al., NeurIPS 2021, [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
- Set Transformer / PMA pooling — Lee et al., ICML 2019, [PMLR v97](http://proceedings.mlr.press/v97/lee19d/lee19d.pdf)

**Compression / long context (L3)**
- ICAE (in-context autoencoder) — Ge et al., ICLR 2024, [arXiv:2307.06945](https://arxiv.org/abs/2307.06945)
- AutoCompressor (segment-PPL) — Chevalier et al., EMNLP 2023, [arXiv:2305.14788](https://arxiv.org/abs/2305.14788)
- Gisting — Mu et al., NeurIPS 2023, [arXiv:2304.08467](https://arxiv.org/abs/2304.08467)
- Recurrent Memory Transformer — Bulatov et al., NeurIPS 2022, [arXiv:2207.06881](https://arxiv.org/abs/2207.06881)
- RULER — Hsieh et al., COLM 2024, [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)
- BABILong — Kuratov et al., NeurIPS 2024, [arXiv:2406.10149](https://arxiv.org/abs/2406.10149)
- NIAH — Kamradt, 2023, [repo](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

**Reasoning depth (L4)**
- Coconut (continuous thought) — Hao et al., 2024, [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- Recurrent-depth "Huginn" — Geiping et al., 2025, [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
- Ouro / LoopLM — 2025, [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)
- Looped transformers & depth — Saunshi et al., ICLR 2025, [arXiv:2502.17416](https://arxiv.org/abs/2502.17416)
- CLUTRR (k-hop) — Sinha et al., EMNLP 2019, [arXiv:1908.06177](https://arxiv.org/abs/1908.06177)
- ListOps — Nangia & Bowman, NAACL 2018, [arXiv:1804.06028](https://arxiv.org/abs/1804.06028)
- Synthetic k-hop QA — 2025, [arXiv:2505.17923](https://arxiv.org/abs/2505.17923)
- Graph connectivity vs depth — 2025, [arXiv:2510.19753](https://arxiv.org/abs/2510.19753)
- "Are latent reasoning models interpretable?" (necessity-ablation audit) — 2026, [arXiv:2604.04902](https://arxiv.org/abs/2604.04902)
- TRM/ARC audit — 2025, [arXiv:2512.11847](https://arxiv.org/abs/2512.11847)

**External comparability**
- lighteval — [docs](https://huggingface.co/docs/lighteval/index) · [repo](https://github.com/huggingface/lighteval)
- SmolLM3 (eval list, lighteval, RULER) — [blog](https://huggingface.co/blog/smollm3) · [model](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)
