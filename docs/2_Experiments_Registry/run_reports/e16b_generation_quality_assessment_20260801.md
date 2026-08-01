# E16b — Free-run generation quality vs base Gemma (Tier 1.5) — `checkpoint-7900`

**Date:** 2026-08-01
**Machine:** Odra (GPU 0, RTX 3090) + local playground sniff (MPS)
**Run ID (training):** `backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850)
**Checkpoint:** `Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850/checkpoint-7900`
**Control:** frozen `google/gemma-3-1b-pt` (same tokenizer family, no LoRA / no concepts)
**Related:** [E16b mechanism report](e16b_longctx_muon_1b_20260725.md) · [spec](../../experiments_specs/done_success/E16b_longctx_muon_1b.md)

**Artifacts:**
- Scale assessment JSON: `Cache/Evaluation_reports/e16b_ckpt7900_generation_assessment_scale.json`
- Earlier CPU vibe-check (partial): `Cache/Evaluation_reports/e16b_ckpt7900_generation_quality.json`
- Shell log: `Cache/logs/eval_e16b_generation_assessment_20260801_124035.log`
- Runner: `analysis/run_e16b_generation_assessment.py`

---

## Goal

Assess **free-running generation** of the E16b best checkpoint across generation
lengths and prompt-context lengths, explain the playground repetition loops, and
decide whether chat-template SFT is the right fix — with a matched **base Gemma**
control so “we broke the backbone” is falsifiable.

This is **not** a re-run of the E16b mechanism gate (Tier-1 RankMe / ΔCE already
passed on 2026-07-25). It is the Tier-1.5 generation vibe-check the mechanism
report explicitly deferred.

---

## Protocol

| Item | Value |
|---|---|
| Models | E16b `BackboneConceptLM` (`shared_depth_recurrent`, C=128, K=512) vs `gemma-3-1b-pt` |
| Concept modes | `real`, `zero` (E16b only) |
| Prompt styles | continuation; Gemma turn-format chat probe |
| Decode | greedy (primary); nucleus sample T=0.8 / top_p=0.95 (subset) |
| Short bank | 6 continuation prompts; 6 chat-wrapped counterparts; max **256** new tokens |
| Length cutoffs | 32 / 64 / 128 / 256 (`distinct_1`, `distinct_2`, REP-3) |
| Context sweep | 2 long docs; prompt prefixes **128 / 512 / 1024 / 2048**; **128** new tokens |
| Dtype / device | bf16 / CUDA |

Metrics: `distinct_n` (Li et al.), REP-3 (Welleck). Falling `distinct_1` + rising
REP-3 with length is the repetition-loop signature.

---

## Results

### 1. Short-prompt continuation (greedy) — length profile

Mean over the 6-prompt bank:

| Condition | @32 d1 / r3 | @64 | @128 | @256 |
|---|---|---|---|---|
| **E16b `real`** | 0.41 / 0.38 | 0.17 / 0.75 | 0.08 / 0.88 | **0.04 / 0.94** |
| **E16b `zero`** | 0.74 / 0.04 | 0.48 / 0.30 | 0.27 / 0.55 | 0.15 / 0.73 |
| **Base Gemma** | 0.70 / 0.05 | 0.49 / 0.24 | 0.33 / 0.42 | 0.16 / 0.71 |

**Reading:** E16b with live concepts collapses into near-total repetition by 256
tokens. Turning concepts **off** (`zero`) is substantially healthier than `real`
on free-run diversity — the opposite of the teacher-forced ablation story.
Base Gemma under greedy also repeats (phrase loops), but stays in prose; E16b
`real` often exits language into digit/punctuation attractors.

### 2. Sampling vs greedy

| Condition | @256 d1 / r3 |
|---|---|
| E16b `real` greedy | 0.04 / 0.94 |
| E16b `real` sample | 0.15 / 0.36 |
| Base sample | **0.49 / 0.03** |

Sampling helps E16b but does **not** close the gap to base. Base + nucleus is
healthy; E16b + nucleus is still degraded.

### 3. Chat-template probe

Both E16b and base were trained as **base PT**, not instruct. Chat format makes
free-run worse or no better for E16b (`concrete concrete…`, instruction echoes).
Base also fails the chat probe (often echoes the user turn). **Missing chat SFT
does not explain continuation loops** — plain `Once upon a time…` already fails.

### 4. Prompt-context length sweep (128 new tokens)

| Prompt length | E16b d1 / r3 | Base d1 / r3 |
|---|---|---|
| 128 | 0.16 / 0.74 | 0.21 / 0.63 |
| 512 | 0.48 / 0.38 | 0.34 / 0.42 |
| 1024 | **0.06 / 0.90** | 0.39 / 0.35 |
| 2048 | **0.07 / 0.91** | **0.50 / 0.20** |

**Reading:** longer prompt context **helps base** and **hurts E16b** free-run.
This is the opposite of what one would hope from a long-context concept-memory
win under free generation — even though teacher-forced beyond-local ΔCE cleared
at these lengths.

### 5. Qualitative signature (matched prompts)

| Prompt | E16b `real` (greedy) | Base (greedy) |
|---|---|---|
| renewable energy… | `…efficiently 1.1.1.1.1…` | prose, phrase-loopy but readable |
| Once upon a time… | `wise old man. . . . . .` | Alice story continues |
| Scientists have long… | `brain is a computer, and whether…` (tight loop) | universe / inflation prose |
| gradient is… | one good definition → `1.1.1.1…` | definition → mild phrase loop |

Local playground samples (MPS, sampling) match the same pattern: fluent opener,
then word/phrase collapse (`shy shy shy…`, `theory of relativity theory is a…`).

---

## Interpretation

### Mechanism success ≠ generation success

E16b remains a **done_success** on its registered question (beyond-local causal
concept use under teacher-forced CE). Free-run generation is a **separate axis**
and currently fails hard. The 2026-07-25 report already warned that ΔCE is not
literary quality; this eval quantifies that warning.

### Three cooperating failure modes

1. **Exposure bias.** Training is gold-prefix CE. Free-run conditions on the
   model’s own tokens; after the first local slip, high-probability cycles win
   (`1.1.1…` from FinePDF-like outline style in the mix; punctuation runs; phrase
   repeats).
2. **Concept write feedback under self-generation.** `zero` ≫ `real` on free-run
   diversity. Live concept writes from bad tokens appear to **poison** the
   trajectory even though corrupting concepts under gold prefixes raises CE
   (Tier-1). Teacher-forced usefulness ≠ free-run stability.
3. **Fluency regression vs base.** Frozen Gemma weights + LoRA r16 + 1B tokens of
   plain causal CE moved open-ended behavior away from the base. Base sampling is
   fine; E16b sampling is not — so this is not “1B base models always loop.”

### Did we break Gemma?

**Partially, for open-ended free-run — not for weights or the memory mechanism.**

- Backbone weights are frozen; LoRA + concept graft change the effective next-token map.
- Matched prompts: base stays in language; E16b `real` often leaves language within
  ~20–40 tokens under greedy.
- Chat templates are orthogonal: both base PT and E16b fail instruction-shaped
  prompts; that only restates “not an instruct model.”

### Is chat SFT the fix?

**No — not as the primary fix.**

Continuation already fails. Chat made E16b worse in the probe. Vision Phase 4
(SFT) belongs **after** non-degenerate continuation; SFT on a looper yields a
polite looper.

---

## How to fix it (recommended layers)

### Layer 0 — Decode band-aids (hours; demos only)

- Prefer sampling over greedy for interactive use.
- Add `repetition_penalty` / no-repeat-ngram to `BackboneConceptLM.generate`.
- Probe **freeze-`z` decode:** encode the prompt with writes, then generate with
  concepts **read-only** (no write from new tokens). If that alone kills most
  loops, it confirms failure mode (2) and defines the right inference contract.

Success for Layer 0: playground continuations stay in prose for ≥128 tokens
without retraining. Not a research fix.

### Layer 1 — Primary research experiment (do next)

**One coherent bet: free-run-stable concept memory.**

Hypothesis: concepts help under gold prefixes but poison free-run when written
from the model’s own tokens.

Preferred formulation (**1A — read-only suffix / frozen-concept decode**):

- Train (or fine-tune from `checkpoint-7900`): write concepts on document tokens;
  after a prefix, freeze `z` and train/eval the suffix with reads only.
- Optional mix: light **scheduled sampling** so some write prefixes see model
  tokens before freeze.
- Keep Tier-1 beyond-local Δ as a non-regression constraint.

Alternative (**1B — free-run CE stage**): short continued train sampling
continuations + CE (and/or unlikelihood on repeated n-grams), without changing
the write/read contract — weaker architectural claim, faster to try.

**Falsify / succeed when:** matched short bank, greedy **and** sample, @128/@256 —
E16b `real` within ~0.1 REP-3 of base **and** `real` ≥ `zero` on usable text;
Δshuffle/Δstatic≥1024 remain ≫ 0.01.

Do **not** lead with “more plain 4K CE” or chat SFT — more teacher-forced CE is
what produced this attractor (cf. E05: loss↓ with generation loops).

### Layer 2 — Protect Gemma fluency (second experiment, if Layer 1 is partial)

- Smaller / slower LoRA, or freeze LoRA after a memory phase and train only
  write/read gates.
- Short fluency-replay mix at low LR while holding a memory auxiliary.

Kill if memory Δ returns to E16a-near-null while fluency returns — that only
undoes E16b.

### Layer 3 — Stronger objective pressure (if needed)

Prefix→suffix / delayed-recall pressure (E02 lesson) **plus** free-run diversity
logged every N steps as a first-class gate — not CE alone.

### Layer 4 — Chat SFT (only after Layer 1)

Instruction following on a model that already continues without digit/punctuation
attractors.

---

## Decision

1. Record free-run generation as an **open failure mode** on the E16b platform —
   does not reopen the Tier-1 mechanism verdict.
2. Elevate free-run vibe-check (`distinct_1` / REP-3 + base control) to a
   **required companion gate** for future backbone_concept generation claims.
3. **Next fix experiment:** Layer 1A (frozen-concept / read-only suffix) or 1B
   (free-run CE stage) from `checkpoint-7900` — draft via `experiment-design`
   before code.
4. Defer chat SFT until continuation clears the Layer-1 gate.
5. Semantic probe (STS-B + floors) on E16b remains useful and orthogonal; do not
   substitute it for free-run metrics.

---

## Appendix — earlier CPU vibe-check (context)

`e16b_ckpt7900_generation_quality.json` (CPU, max_new=1024, fewer prompts) already
showed the same kill signature: `continuation|real` @1024 `distinct_1≈0.018`,
`rep_3≈0.976`; chat|real worse; `real≈shuffle` on short prompts. The 2026-08-01
Odra run supersedes it for base comparison and context-length claims.

*Related: [e16b_longctx_muon_1b_20260725.md](e16b_longctx_muon_1b_20260725.md),
`agenda.md`, `experiment-evaluate` Tier 1.5.*
