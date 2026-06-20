# Concept-Information Eval Upgrade — engineering spec + implementation plan

- **Type:** engineering (eval-foundation). **Not** an `E0NN` experiment — it changes *how we measure*, not *what we train*. No training-code or training-contract changes.
- **Status:** in implementation (opened 2026-06-15)
- **Owner:** Krzysztof Sopyla
- **Serves:** the agenda "Unified eval interface" engineering item, and the chronic question behind E01/E02/E03: *do the 128 concepts actually store meaningful, distributed information?*
- **Read-only on checkpoints.** Re-running the new probes on E01/E02/E03 best checkpoints is a **follow-up** (needs a GPU box), not part of this change.

## Problem (why this is needed)
The current protocol cannot answer its own headline question, for three concrete reasons found in the code:

1. **Every semantic probe pools the structure away.** Zero-shot STS-B, GLUE, SICK, PAWS all do
   `concepts [B,128,H] → mean(dim=1) → [B,H]` before scoring
   (`nn/concept_encoder_perceiver.py:803` `_pool_concepts`, `:570`, `:705`). A rank-1 collapsed
   concept set and a rank-128 healthy set can yield the **same** mean vector, so the semantic eval is
   mathematically blind to the de-collapse that E03+ exists to create. Empirically: E02 has collapsed
   slot rank (11.57) yet the best STS-B (0.702); E03 held rank steady yet STS-B stayed flat (0.556).
   The geometry goal and the semantic readout are decoupled by construction.
2. **STS-B numbers have no floor or ceiling**, so 0.556 / 0.702 are uninterpretable. Mean-of-word-embeddings
   is a famously strong STS baseline (~0.4–0.6 Spearman); without that floor we cannot tell whether the
   concepts beat bag-of-embeddings at all.
3. **Three different "rank" numbers are used interchangeably** as if they were one object:
   - `effective_rank` = SVD of **batch-averaged** concepts (`concept_analysis.py:83,90`) → ~10/128.
     Measures *slot redundancy after averaging*, conflates within-sample and cross-sample structure.
   - `manifold_rankme` = RankMe over **cross-sample** pooled embeddings (`:191`) → ~167.
     Measures *embedding diversity across inputs*, not concept-set rank (and >128 by construction).
   - **within-sample concept-set rank** (the true de-collapse object: how many independent directions do
     ONE input's 128 concepts span?) — **does not exist in code**.

GLUE full fine-tune is additionally misleading: it unfreezes the encoder and trains a head, so it
re-routes around the bottleneck and measures fine-tuning capacity, not what the concepts store.

## What we measure after this change (the contract)
Separate the question into two independent tiers and make each interpretable:

- **Tier 2 — zero-shot semantic gate (fixed pooling, no params).** Keep mean-pool STS-B exactly as-is
  for backward-compatibility with E01/E02/E03, but report it **next to two trivial floors** so the number
  is anchored. The ceiling is cited from literature (SimCSE-unsup ≈ 0.76, SBERT ≈ 0.84 Spearman), not run.
- **Tier 2.5 — frozen-encoder readout probe (NEW).** Freeze the encoder, train only a tiny head, and
  report **mean-pool vs attention-pool** side by side on the pair tasks. If attention-pool ≫ mean-pool,
  information *is* distributed across the 128 slots and mean-pool was hiding it; if equal, the set is
  genuinely collapsed in the way that matters. This is the readout that can finally *see* de-collapse.
- **Tier 1 — geometry, disambiguated.** Add a clean **within-sample concept-set rank** as the PRIMARY
  de-collapse metric; keep slot-mean `effective_rank` as a labeled SECONDARY diagnostic; relabel
  `manifold_rankme` as "cross-sample embedding diversity" (a downstream-retrieval property, not concept rank).

## What we remove / demote (authorized 2026-06-15)
- **GLUE full fine-tune is demoted** from the "do concepts store info" evidence line. The script stays,
  but the protocol only credits the **frozen-encoder probe** variant. Full-finetune GLUE is, at most, a
  downstream-utility footnote — never concept-content evidence.
- **The batch-averaged slot rank is demoted** from headline to secondary diagnostic.
- **Cross-sample RankMe is relabeled** so it is never again quoted as "concept rank".
- No new long-document probe here (the compression/long-context test belongs to **E05**; out of scope).

## Falsifiable value check (how we know this upgrade was worth it)
The upgrade pays off iff, on at least one existing checkpoint, the new probes **change a conclusion**:
- the attention-pool probe beats mean-pool by a margin clearly above seed noise (target ≥ +0.03 STS-B /
  +2 acc), **or** confirms equality (collapse is real) — either is decision-relevant; **and**
- the trivial floor lands close enough to the model's zero-shot STS-B (within ~0.05) to reveal that some
  "successes" were near bag-of-embeddings. If neither happens, the probe added cost without signal —
  record that and drop it.

---

# Implementation plan (repo-rooted, step by step)

Each step is a config/flag over existing scripts; no new training fork; backward-compatible defaults.

### Step 1 — geometry metric hygiene (`analysis/concept_analysis.py`, `analysis/run_concept_analysis.py`)
- Add `compute_within_sample_concept_rank(concept_repr: [B,C,H]) -> {within_sample_rankme_mean, within_sample_rankme_std}`:
  for each sample, RankMe (`exp(entropy(normalized singular values))`) of its `[C,H]` matrix, averaged over
  the batch. This is the **primary** de-collapse number.
- Do **not** delete `effective_rank`; in the runner's printed report + JSON, label it
  `slot_mean_effective_rank (secondary diagnostic)` and label `manifold_rankme`
  `cross_sample_embedding_rankme (downstream diversity, not concept rank)`. Add the new key to the JSON.
- Update the printed "Concept Health" summary ordering: primary = within-sample rank, then ablation ΔCE,
  then secondaries.

### Step 2 — zero-shot STS-B trivial floors (`evaluation/evaluate_on_benchmark.py`)
- Add `--baseline {none, token_embed_mean, teacher_hidden_mean}` (default `none`) handled inside / beside
  `run_zero_shot_stsb`:
  - `token_embed_mean`: mean-pool the **model's own input-embedding table** rows over each sentence
    (masked), cosine. Pure bag-of-embeddings floor, no encoder forward.
  - `teacher_hidden_mean`: load frozen `HuggingFaceTB/SmolLM2-135M` (eval, no_grad), mean-pool last hidden
    states over non-pad tokens, cosine. The "what the anchor teacher alone gives" floor.
- Tag CSV / W&B rows with a `variant` field (`model` | `token_embed_mean` | `teacher_hidden_mean`) so the
  three numbers sit in one report. Cite the SimCSE/SBERT ceiling in the run report text.

### Step 3 — attention-pool probe (`nn/concept_encoder_perceiver.py`, `evaluation/evaluate_on_benchmark.py`, `evaluation/evaluate_model_on_glue.py`)
- In `ConceptEncoderForSentencePairClassification`, add config field `pool_mode` (`"mean"` default |
  `"attention"`). Add a small `AttentionPool` module: one learned query `[1,H]`, single-head
  cross-attention over the `[B,C,H]` concepts → `[B,H]`, followed by the existing `pool_norm`.
  `_pool_concepts` switches on `pool_mode`. `"mean"` path is byte-identical to today (backward compat).
- Add `--pool_mode {mean,attention}` to both eval CLIs; thread into `config` before model construction.
- The probe tier = `--freeze_encoder --pool_mode {mean,attention}` (frozen encoder, tiny trainable head).
  No default behavior changes when the flags are absent.

### Step 4 — protocol docs (`.cursor/skills/experiment-evaluate/SKILL.md`)
- Rewrite Tier 2 (add floors), insert Tier 2.5 (frozen-encoder mean-vs-attention probe), demote GLUE
  full-finetune to a footnote, and replace the rank language with the disambiguated three-metric naming.

### Step 5 — tests (`tests/`)
- `test_within_sample_concept_rank`: synthetic collapsed concepts (all slots equal) → rank ≈ 1; diverse
  random concepts → rank ≫ 1; monotonic between.
- `test_sentence_pair_pool_modes`: `pool_mode="attention"` forward returns correct shapes for both
  `cosine_only` and classifier paths; `pool_mode="mean"` output unchanged vs the pre-change code path.

### Validation (local, no GPU training)
- `uv run pytest tests/ -k "pool or rank" -v`.
- MPS smoke of the metric + a randomly-initialized sentence-pair model forward in both pool modes.
- A real best-vs-checkpoint re-run on E01/E02/E03 is the **follow-up** (remote), handed to `experiment-run`
  / `experiment-evaluate`, then `experiment-track`.

---

# Staged eval ladder (from the 2026-06-15 literature scan)

Four parallel `research-scout` scans (representation probing, long-context compression, latent
reasoning, generation-from-bottleneck) converge on two principles that should shape the *whole*
eval suite, not just the patch above.

**Principle 1 — separate the two axes.** Every credible eval pins exactly one of:
- **Faithfulness** — does information *survive* the bottleneck? (reconstruction CE, round-trip
  recovery, concept-ablation ΔCE, KL/MI/active-units)
- **Utility** — is the surviving info *usable*? (STS, clustering, retrieval, linguistic probing,
  reasoning accuracy)
Each probe must use a **frozen encoder + a control baseline + a necessity ablation**, or it
measures the readout, not the concepts (this is the GLUE-finetune failure, and the project's
recurring comparability weakness).

**Principle 2 — stage by checkpoint maturity.** Long-context and reasoning benches are *premature*
on collapsed, incoherent checkpoints (E01–E03); they would only produce noise. Unlock each rung by
a gate.

| Rung | Gate to unlock | Measures (axis) | Goal served | Status |
|---|---|---|---|---|
| **L0 Geometry** | always | within-sample RankMe, anti-collapse, active slots (faithfulness-precursor) | prerequisite | **built** |
| **L1 Generation faithfulness** | concepts non-trivial | ΔCE (built); **round-trip token recovery**; teacher-forced vs free-running CE; latent specificity `z(x)` vs `z(x')`; KL/MI/AU | generation | **next (now)** |
| **L2 Representation utility** | L0 healthy | **SentEval probing + control-task selectivity**; **MTEB STS/clustering/retrieval (zero-training)**; mean-vs-attention pool (built) | representation / "useful info" | **next (now)** |
| **L3 Compression** | de-collapse + coherent recon | **ICAE-style reconstruction curves** (score vs compression-ratio & slot-count); RMT/RULER small-scale recall; held-out segment PPL w/ compressed prefix | long context | gated |
| **L4 Reasoning depth** | L3 + recursion (E08) | depth-dependent synthetics (p-hop, CLUTRR, ListOps, k-hop QA, graph-by-diameter) + necessity ablation + iso-FLOP baselines | reasoning | gated |

## Per-rung adopt / watch / reject (with sources)

**L1 — generation faithfulness (cheapest honest evidence; no new datasets):**
- *Adopt* **round-trip token recovery** (encode→decode, Token-F1 / exact-match) — Vec2Text
  (EMNLP 2023, arXiv:2306.05443), CALM round-trip (arXiv:2510.27688).
- *Adopt* **teacher-forced vs free-running reconstruction CE** reported together (they diverge for
  AR decoders; rarely both reported).
- *Adopt* **latent specificity** (decode same prefix with `z(x)` vs `z(x')`; output divergence
  should track input divergence) — Optimus (arXiv:2004.04092), latent-vacancy (arXiv:1905.11975).
- *Adopt* **collapse trio KL / MI / active-units** as early warning — Dieng 2019 (arXiv:1807.04863),
  Optimus.
- *Watch* (defer until generation is coherent) **MAUVE + Gen-PPL + distinct-n + 4-gram
  memorization** — LD4LG (arXiv:2212.09462), Cosmos (arXiv:2506.21170).

**L2 — representation utility (the direct GLUE replacement):**
- *Adopt* **SentEval probing** (WordContent, BShift, Tense, SubjNum, TreeDepth…) as frozen linear
  probes **with Hewitt control-task selectivity** (arXiv:1909.03368) — tells you *what kind* of
  info concepts hold (arXiv:1805.01070).
- *Adopt* **MTEB zero-training subset**: STS + Clustering (v-measure) + small Retrieval/BEIR slice
  (arXiv:2210.07316, MMTEB arXiv:2502.13595) — clustering/retrieval need *no head* and reward the
  distributed structure mean-pool hides.
- *Adopt* **attention-pool (Set Transformer PMA) vs mean-pool** — built; PMA (ICML 2019) is the
  canonical permutation-invariant set readout.
- *Watch* **MDL / V-information probing** (Voita & Titov EMNLP 2020, arXiv:2003.12298) for
  publication-grade rigor later.
- *Reject* **full-finetune GLUE** as concept evidence (re-routes around the bottleneck).

**L3 — compression / long context (gated on de-collapse + coherent recon):**
- *Adapt* **ICAE-style reconstruction-faithfulness curves** (arXiv:2307.06945) — the cleanest match
  to "N tokens → C concepts"; the bridge from the recon objective to the long-context vision.
- *Adapt* **small synthetic recall** RMT copy/associative-recall (arXiv:2207.06881) and RULER
  S-NIAH / MQAR at 4–16K (arXiv:2404.06654) — proven viable at ~137M (BABILong arXiv:2406.10149).
- *Adopt* **held-out segment PPL with compressed prefix** (AutoCompressor, arXiv:2305.14788).

**L4 — reasoning depth (design now, run after E08):**
- *Watch/Design* depth-dependent synthetics with a controllable difficulty knob: p-hop induction &
  i-GSM (arXiv:2502.17416), CLUTRR k-hop (arXiv:1908.06177), ListOps (arXiv:1804.06028), synthetic
  k-hop QA (arXiv:2505.17923), graph-connectivity-by-diameter (arXiv:2510.19753).
- *Adopt as a rule* the **necessity ablation** (zero concepts / zero loops) + **iso-FLOP & iso-param
  baselines** for any latent/recursion claim — apparent depth gains routinely vanish to zero-latent
  ablation or come from ensembling/ID-conditioning (audit arXiv:2604.04902; TRM/ARC arXiv:2512.11847).

## Sequencing
1. **Now (this doc):** L1 round-trip + free-running CE + latent specificity (reuse the AR model and
   FineWeb-Edu; no new deps) and L2 SentEval + MTEB-subset frozen probes.
2. **On de-collapse:** open L3 (its own spec) — reconstruction-vs-compression-ratio harness.
3. **On E08 recursion:** open L4 (its own spec) — one controllable depth-synthetic + necessity ablation.

> Full scouted source material lives in the agent transcripts for the four 2026-06-15 scans
> (representation probing, long-context compression, latent reasoning, generation-from-bottleneck).

---

# Implementation pass 2 (2026-06-16) — L1 + L3 in the analysis harness

Scope decision: implement the **dependency-free, CPU-testable** core that reuses the existing
`concept_ar` model and `analysis/run_concept_analysis.py` harness — namely **L1 generation
faithfulness** and the **L3 compression-faithfulness curve** (they share one round-trip primitive).
**L2 SentEval + MTEB is explicitly deferred** to a separate remote slice (it needs external datasets
and would add heavy deps — `mteb`/`sentence-transformers` — that can't be validated locally; the
zero-shot STS-B floors + mean/attention pool already cover the cheap L2 signal). L3's long-context
*recall* tasks (RULER/RMT) and segment-PPL stay gated (need context-length extension); the
reconstruction-vs-compression-ratio curve below is the L3 piece that is meaningful on current 512-token
reconstruction checkpoints.

### Step 6 — `analysis/concept_generation_eval.py` (new, reusable)
All functions take `batches = [(input_ids, attention_mask), …]` (CPU) like `compute_ar_concept_ablation`,
and reuse `model.encode_concepts`, `model._shift_right`, `model.decode_logits` (exact teacher-forcing
convention — no re-derived shift).
- `compute_roundtrip_recovery(model, batches, device, concept_num, free_running_examples, eos_id, max_new_tokens)`:
  - **Teacher-forced token accuracy** — argmax of `decode_logits` vs labels over non-pad positions (L1).
  - **Free-running recovery** — greedy decode from start token, compare to gold: position exact-match +
    **token-F1** (multiset overlap) on a small sample (L1).
  - **Compression curve** — teacher-forced recovery bucketed by `ceil(seq_len / C)` (= compression
    ratio); recovery vs ratio (L3).
- `compute_latent_specificity(model, batches, device)`: teacher-forced accuracy with **matched vs
  row-shuffled** concepts → `specificity_acc_drop`, plus mean symmetric-KL of next-token distributions
  (matched vs shuffled). Confirms outputs are specific to *this* input's concepts (L1).
- Helper `token_f1(pred_ids, gold_ids)`.

### Step 7 — wire into `analysis/run_concept_analysis.py`
- New flags: `--generation_eval` (default on for `concept_ar`), `--free_running_examples` (default 8).
- Reuse the already-collected `ablation_batches`; print an **"L1/L3 — Generation & compression
  faithfulness"** section; add results under `result["generation_faithfulness"]` in the JSON.

### Step 8 — tests (`tests/test_concept_generation_eval.py`)
- Tiny random `ConceptEncoderForConditionalLM` on CPU: shapes + ranges (accuracies in [0,1]); `token_f1`
  correctness on known cases; **specificity is ~0 for an untrained model and the API returns a finite
  drop**; compression-curve buckets cover the lengths present.

### L2 (deferred remote slice — documented, not built this pass)
- `evaluation/` probing runner for **SentEval tasks** (linear probe + Hewitt control-task selectivity)
  and **MTEB STS/Clustering/Retrieval** via the `mteb` package, frozen-encoder, mean-vs-attention pool.
  Build when a GPU box is available and a checkpoint is worth probing; add `mteb`/`sentence-transformers`
  as an optional dependency group then.

## Risks
- **Free-running recovery is O(examples × tokens)** — keep `free_running_examples` small (≈8); the
  teacher-forced accuracy + compression curve are the cheap batched signals.
- **Attention-pool head capacity could mask a weak bottleneck.** Mitigation: single learned query, single
  head, frozen encoder, and always reported *against* mean-pool — the delta is the signal, not the absolute.
- **Token-embed floor depends on the model's own (trained) embedding table**, so it is a *lower bound that
  already includes some learning*; that is the intended, conservative floor. The teacher-hidden floor is the
  external reference.
- **Backward compatibility:** all new behavior is behind new flags / new JSON keys; absent flags reproduce
  today's numbers exactly.
