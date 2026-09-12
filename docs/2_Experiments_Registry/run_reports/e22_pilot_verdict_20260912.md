# E22 pilot verdict — a live, diverse concept array that the decoder reads for register, not for facts

**Date:** 2026-09-12 (train 13:17 → ≈ 20:00 UTC; eval + diagnosis same evening)
**Machine:** Odra (3× RTX 3090) for arms A and C · Polonez for arm A′ and the dense control
**Spec:** [E22](../../experiments_specs/done_failed/E22_perceiver_concept_lm.md) · plan
[E22_plan](../../experiments_specs/done_failed/E22_perceiver_concept_lm_plan.md)
**Root-cause note:** [e22_root_cause_20260912](../../4_Research_Notes/e22_root_cause_20260912.md) (the full evidence and reasoning; this report is the ledger summary)
**Git:** `8914029` (as run) → `0f4b4f9` (`near`/`far` ablation + `concept_xattn_scope`)
**Artifacts:** Odra `Cache/eval/e22_armA/` (`concept.json`, `concept_nearfar.json`, `longctx_suite.json`, `tasks.json`, `summary.md`), `Cache/eval/e22_armC/`; finals on Odra `Cache/Training/…131735_resumed/final`, `…152511/final`; NAS `e22/`

---

## Goal

Test the one bet in the E22 spec: a from-scratch ~128M-compute LM whose only route past a 1024-token
decoder segment is a length-proportional, positionally allocated concept array (one pooled slot per 16
tokens from a 6-layer SWA encoder, processed by a 4-layer latent transformer) becomes load-bearing and
useful under plain next-token CE at 32k.

## Runs

| run id | arm | what it tested | status |
|---|---|---|---|
| `perceiver_concept_H768e6r16c1l4d8s1024_20260912_131735` (+ `_resumed`) | **A** | the bet | 560 steps · ≈ 0.28B tokens (56% of the 0.5B pilot) · evaluated |
| `perceiver_concept_H768e6r16c1l4d8s1024_20260912_152511` | **C** | same decoder, no encoder / array / cross-attention | 560 steps · evaluated |
| `perceiver_concept_H768e6r16c1l4d8s1024_20260912_143726` | A′ | duplicate arm A on Polonez | final archived to NAS; not evaluated separately |
| `perceiver_ar_dense_H768L0g0s18N2048_20260912_161135` | dense | 18-layer full-causal control | crashed ≈ step 390 (`ENOSPC`); `checkpoint-320` lost in the cleanup sweep |

## Gate results

| gate | criterion | result | verdict |
|---|---|---|---|
| **S1** load-bearing | Δ_none ≥ 0.30 on ≥ 4k history; Δ_shuffled ≥ 0.20 | Δ_none **0.25–0.27** (> 10σ) but **0.22 already inside segment 0** where no far slot exists; Δ_shuffled 0.19–0.29, and shuffled < none at 16k–32k | ❌ (used, distance-blind) |
| **S2** useful | A ≤ 0.97 × C on positions ≥ 4k | A 4.210 / 4.151 / 4.162 vs C 4.215 / 4.155 / **4.133** → ratio ≈ 1.00 | ❌ |
| **S3** compression price | A within 3% of dense | at step 320: A 4.691 vs dense 4.616 (+1.6%) — dense then lost | unmeasured (on track) |
| **S4** recall | passkey ≥ 0.5 @32k; keyed-recall first-token ≥ 25% | passkey **0.0** / multikey 0.0 / vt 0.0; tasks **4.8%** (C floor 3.5%) | ❌ |
| **S5** geometry | RankMe(z) ≥ 64 | **265** of 768; adjacent-slot cosine 0.69; pooler `wo` ‖W‖ 11.2 (init 0) | ✅ |
| **K1** | at 40% budget A not below C on far positions | met (A ≈ C throughout) | triggered |
| K2 | Δ_none < 0.05 | not triggered (0.25) | — |
| K3 / K4 | eval rising / throughput < 40% dense | not triggered | — |

## The decisive measurement (added post-run)

`near` / `far` paired ablation on arm A (64 rows, `concept_nearfar.json`): removing only the **far** slots
costs **0.054 / 0.057 / 0.059 / 0.046** nats from 1k to 32k (flat); removing only the **near**
(own-segment) slots costs **0.026 / 0.022 / 0.018 / 0.030**; removing everything costs 0.25. Segment 0
behaves as the built-in control (0.000 and 0.220). So the array's value decomposes as ≈ 0.17 nats of
document-level content present redundantly in every slot, ≈ 0.05 of far-specific content — the memory
the bet was about, 6× under the gate — and ≈ 0.03 of local bypass through the `cpos ≤ pos` mask. The
array itself is diverse (RankMe 265), so the content is there; the decoder was never trained to read it.

## Verdict

**killed — K1 met, S1/S2/S4 missed; S5 passed.** Two root causes ([note §3](../../4_Research_Notes/e22_root_cause_20260912.md)):
(1) next-token CE on natural text is worth ≈ 0.05 nats beyond 1k tokens at this scale, and the array
captured exactly that — the objective never paid for far content (the E18 law, now on a structurally
closed model); (2) the cross-attention mask admitted same-segment slots, so the array was never the only
route for anything and Δ_none over-states channel value 5×. The suspected third cause — smooth,
collapsed slots — is refuted by S5.

**Banked:** the `perceiver_concept` family (first ledger design with positional slots, a transformer
over slots and a segment-closed decoder; trains stably; 96 B/token state); the `near`/`far` concept
ablation with a built-in zero; `concept_xattn_scope=exclusive`; the finding that a CE-trained concept
array is a good *document embedding* and a poor *memory*; two new design laws (objective must pay;
exclusivity is two-sided). **Lost:** dense control checkpoint (S3 needs a rerun if ever wanted).

**Next:** E23 — same platform, exclusive scope, an objective that pays for far content
(far-repeat-weighted CE + 30% dense-label long-range rows), gated on the far marginal and on recall.
