# Text capability checks: calibration ladder — what the literature says (2026-10-10)

**Question.** After the 30M round on data v1 (Odra, 9–10 Oct 2026: 600M tokens = 20 tokens/param, 3,173
steps of ~189k tokens) every model, dense included, sits at the guessing rate on every task. Is the
missing ingredient data/steps or model capacity, and does the author's two-axis ladder make sense?

**Author's proposal (2026-10-10).** (1) 30M on 1–1.5B tokens; (2) 50M on 1–1.5B (max 2B), depending on (1);
(3) 100M with more tokens.

**Verdict.** The ladder is the standard shape (size × token multiple, dense only, until dense passes). Four
adjustments from the literature: count optimizer steps, not only tokens; score the probability of the right
answer, not only exact match; one failing seed is not a verdict; grow 50M by depth first. Step 1 started on
Odra 2026-10-10 05:50 UTC (dense, 30M, 1.5B tokens = 7,931 steps, same v1 mix, 2.3 passes).

## 1. Our own evidence, re-read (r1v1 30M round)

Dense is far less surprised by the right answer when the fact is in the document than when it is removed
(mean nats per answer token, lower = more confident; held-out-name exam at the stated length):

| task | 1k with fact | 1k fact removed | 4k with fact | 4k fact removed |
|---|---|---|---|---|
| quote | 0.38 | 6.56 | 0.64 | 6.17 |
| lookup | 0.53 | 3.68 | 0.59 | 3.47 |
| keyed | 0.87 | 3.63 | 0.96 | 3.67 |
| latest | 0.72 | 4.09 | 0.77 | 3.96 |
| compose | 0.71 | 3.74 | 0.82 | 3.75 |

Yet its first-token pick is at the guessing rate (lookup 20 % vs 25 %), and the copy probe on repeated real
text barely drops (1.90 → 1.65 nats/token at gap 0). Reading: dense has learned *"the answer is a word from
this document"* (an in-context word cache: once the first piece of an invented name is given, the rest is
predicted from the document) but not *"which person's"* (binding the asked name to its fact), and no general
copying. This is the known stage order in small transformers — in-context unigram statistics first, then
induction/bigram copying (Edelman et al. 2024, arXiv 2402.11004; Bietti et al. 2023, arXiv 2306.00802) — so the model is mid-way, not stuck at
zero. The memory models barely use the document at 4k (E31c lookup 1.91 vs 2.33; no-memory control 1.83 vs
2.35), i.e. the notebook carries no name identity yet.

## 2. Evidence by claim

**Copying from context needs little capacity, but arrives late and abruptly.**
- Induction heads form in a narrow window at roughly 2.5–5B tokens "for language models of every size
  (provided they have more than one layer)" (Olsson et al. 2022, arXiv 2209.11895).
- Pythia (2.1M tokens/step): induction heads emerge soon after ~2B tokens (~step 1,000), similar for
  70M–2.8B (Tigges et al. 2024, arXiv 2407.10827; Yin & Steinhardt 2025, arXiv 2502.14010).
- Timing is set by the number of updates, batch and context length, not model size (50M–7B): fitted
  U = e^13.26 · B^−0.37 · C^−0.62 (Aoyama, Wilcox, Schneider, ICML 2026, arXiv 2511.16893). Our C = 4096 is
  outside the fitted range (4–2048); the fit under-predicts Pythia ~2×.
- 2-layer attention solves multi-query associative recall at model width 64 for lengths 64–512
  (Zoology, Arora et al. 2023, arXiv 2312.04927). Transformers learn copying with ~100× fewer samples than
  state-space models at ~160M (Jelassi et al. 2024, arXiv 2402.01032).
- Implication: 30M (8 layers × 576) has ample capacity for lookup/keyed/quote; the first lever is steps.
  1.5B tokens = 7,931 steps; 2B = ~10,600 steps. E33 on its synthetic ladder saw composition jump after
  6–10k steps.

**Larger models learn faster per token.** "Larger models are significantly more sample-efficient"
(Kaplan et al. 2020, arXiv 2001.08361). So 50M at the same tokens is a fair capacity test: compare at
equal tokens and equal data.

**Two-hop composition is the hard one.** Natural pretraining at 1.3B params / 100B tokens "fails even the
simplest 2-hop reasoning" in context (30–36 %, near random), while synthetic data teaches it to small models
(Allen-Zhu, Physics of LMs Part 4.1, arXiv 2512.17351). Depth matters more than width for reasoning (Part 2.1,
arXiv 2407.20311; MobileLLM, arXiv 2402.14905: deep-thin wins at 125M); k hops need ~⌊log₂k⌋+2 layers
(Sanford et al. 2024, arXiv 2402.09268). A 3-layer toy model jumps from chance to 100 % on in-context 2-hop at
~800 steps (Guo et al. 2025, arXiv 2502.13913) — on a pure toy task, not mixed pretraining.

**Model ladders vary size and token multiple, and score near-chance tasks by probability.**
- OLMo ladder: 190M/370M/760M/1.3B × {1, 2, 5, 10} × Chinchilla; predict task loss (log-loss of the correct
  answer) first, then accuracy (Bhagia et al. 2024, arXiv 2412.04403).
- DataDecide: 4M–1B at 100 tokens/param, 3 seeds; the probability of the correct answer is as good as or
  better than accuracy at small scale and turns near-chance tasks predictable (Magnusson et al. 2025,
  arXiv 2504.11393).
- Gadre et al. 2024 (arXiv 2403.08540): keep a task only if a 154M model is ≥10 points above chance.
- Exact match on a multi-token answer looks sudden even when the underlying skill grows smoothly
  (Schaeffer et al. 2023, arXiv 2304.15004).

**Seeds.** Time to acquire in-context learning is long-tailed; near the threshold many runs never acquire it
(Nguyen & Reddy 2024, arXiv 2412.00104). E33 saw one run in three never jump. Init seed shifts timing in
transformers (Singh et al. 2024, arXiv 2404.07129).

**Repeating data is fine at these budgets.** Up to 4 epochs ≈ fresh data for loss (Muennighoff et al. 2023,
arXiv 2305.16264). 1.5B tokens on the 660M mix = 2.3 passes; 2B = 3.0. Exams use held-out names, so
memorising training worlds does not help.

**Batch.** Critical batch size scales mainly with data, ~300–380k tokens at 85M / ~1–1.7B tokens (Zhang et al.
2024, arXiv 2410.21676); our ~189k tokens/step is below it, so steps are not being wasted on oversized batches.

**What tiny models do on text.** TinyStories (Eldan & Li 2023, arXiv 2305.07759): grammar first at the
smallest sizes; consistency (tracking names/facts in a ~512-token story) appears from hidden 128; "for
context-tracking the number of layers is more important"; 1-layer models fail consistency. Small public
models are trained far beyond 20 tokens/param (SmolLM2-135M 2T, MobileLLM-125M 1T, Pythia-70M 300B) — for
general quality, not for our question.

## 3. Ladder protocol (proposed)

| step | model | tokens | steps | Odra time (3 GPUs) | decide |
|---|---|---|---|---|---|
| 1 | dense 30M (8 × 576) | 1.5B, extend to 2B if the answer probability is still improving | 7,931 | ~2.2 h | see rules |
| 2 | dense 50M (deeper first) | same as the best of step 1, same mix | — | ~3.5 h (est.) | capacity vs steps |
| 3 | dense 100M | 2–4B (fresh mix) | — | ~8–15 h (est.) | budget for the four-model round |

Scored at 10/25/50/75/100 % of steps (learning curve), plus the copy probe. Decision rules after step 1:
- Pick rises above guessing on quote/lookup/keyed by the end → token-limited; set the round budget there
  (or at 2B), and run the four-model round.
- Answer probability keeps improving but pick stays at guessing → still on the way; extend to 2B, then step 2
  at equal tokens.
- Nothing moves from 600M to 1.5B → not tokens: step 2 at the same tokens; if 50M is flat too, the data format
  is the limit (answers are ~0.1 % of training tokens: one question per ~2k-token document) — raise the
  question density (Zucchet et al. 2025, arXiv 2505.17863: within-sequence repetition speeds emergence ~4×;
  Chan et al. 2022, arXiv 2205.05055) before scaling further.
- A single failing seed is not a verdict: add a second seed before concluding "not tokens".
