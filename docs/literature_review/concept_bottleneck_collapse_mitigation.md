# Research note — making the concept bottleneck non-bypassable + anti-collapse losses

**Date:** 2026-07-09
**Responds to:** the E05 Muon long (2 ep) collapse diagnosis — [`e05_muon_long_2ep_collapsed_20260709.md`](../2_Experiments_Registry/run_reports/e05_muon_long_2ep_collapsed_20260709.md) §"Mechanism (deep-dive)".
**Question:** what *easy, proven, SoTA* objective change or added loss would (a) make the 128-concept bottleneck **non-bypassable** and (b) **prevent the rank-1 collapse** the run exhibited?
**Scope:** primary sources only, arXiv/proceedings verified. All IDs below were checked against the arXiv abstract page.

---

## 1. The two failure modes (so we fix the right one)

The diagnosis split the E05 failure into two coupled mechanisms, and the literature has two corresponding fix families:

1. **Decoder bypass (objective is bypass-able).** The K=128 windowed decoder predicts the suffix from its *own* within-window tokens, routing around the concepts. The decoder's input access is what makes the bypass possible.
2. **Dimension / posterior collapse (the redundant directions die).** Once the bypass makes the concept channel redundant (`∇L ≈ 0` there), weight decay selectively shrinks those singular directions → rank-1 (`enc.L5.bixt.rv_lat`, stable rank 1.79, 834/1536 dead rows).

**Key literature confirmation for mechanism 2:** Jing et al., *"Understanding Dimensional Collapse in Contrastive Self-supervised Learning"* (arXiv:2110.09348, ICLR 2022) explicitly names **implicit regularization (a weight-decay-like effect)** as a *cause* of dimensional collapse, diagnosed by **spectral analysis of the feature covariance matrix** (eigenvalues vanish in the collapsed directions) — exactly the `bixt.rv_lat` singular-value signature we measured. The remedy the SSL literature converged on is the decorrelation / variance regularizers in Family C below.

---

## 2. Four families of fixes (the recurring structure across SSL + VAE + CBM literature)

### Family A — Destroy the decoder's input access (make the bypass *impossible by construction*)
The cleanest "non-bypassable." If the decoder cannot read the suffix tokens, it must read them from the concepts.

- **TSDAE** (Wang/Reimers/Gurevych, EMNLP 2021 Findings; arXiv:2104.06979): the decoder's cross-attention is **confined to a single fixed-size sentence vector** — "Our decoder decodes only from a fixed-size sentence representation … it does not have access to all contextualized word embeddings … this bottleneck should force the encoder to produce a meaningful sentence representation." Best noise = **deletion, ratio 0.6** (App. A). Decisive ablation (Table 4): starting from **BART/T5** (powerful decoders) gives *lower training loss but worse embeddings* — "overfitting the reconstruction behavior." ⇒ over-capable decoders bypass; corrupting/constraining the decoder's input forces a useful latent.
- **BERT/MLM** span masking and **MAE** (He et al., CVPR 2022; arXiv:2111.06377, ~75% masking + lightweight decoder): same principle — mask the input the decoder would otherwise copy.
- **MrCogito status:** the encoder-side TSDAE deletion is already on (`deletion_rate=0.6`). The decoder-side lever — `DECODER_WORD_DROPOUT` (`nn/concept_encoder_perceiver.py:1214-1216`, replaces decoder-input embeddings with a learned mask with prob `p`, training-only) — is **fully implemented and currently 0**. This is the one-config-change non-bypassable knob.

### Family B — Predict in latent space, not token space (no decoder to bypass through)
Remove the token decoder entirely; predict a *latent target* the local context genuinely cannot supply.

- **I-JEPA** (Assran et al., CVPR 2023; arXiv:2301.08243): predict target-block *representations* from a context block in latent space — no pixel/token decoder. Anti-collapse via **asymmetric architecture** (EMA target encoder + narrow predictor, App. A.1). Decisive ablation (Table 7): replacing latent targets with **pixel targets drops linear-probe 66.9 → 40.7** — the latent target is what makes the representation carry semantics.
- **CPC / InfoNCE** (van den Oord et al., 2018; arXiv:1807.03748): predict the **future** latent `z_{t+k}` from context `c_t` via InfoNCE; `I(x_{t+k};c_t) ≥ log N − L_N`. The target is future, so a local decoder that copies from its input cannot solve it.
- **MrCogito status:** this is exactly **E06** (latent-space prediction, already specced) — the bigger, more principled bet, deferred. Listed here for completeness; *not* the "easy" lever.

### Family C — Anti-collapse regularizers on the latent (directly fight rank collapse)
Additive loss terms that forbid the trivial/degenerate solutions — orthogonal to the objective, drop-in.

- **VICReg** (Bardes/Ponce/LeCun, ICLR 2022; arXiv:2105.04906): three terms — **variance** (per-dimension std hinge `max(0, γ − std)`, *explicitly forbids the all-zero/constant solution*; using std not var is load-bearing), **covariance** (off-diagonal of the batch covariance → 0, *forbids rank-1/redundancy*), invariance (MSE between views). Defaults λ=μ=25, ν=1, γ=1. Ablation (Table 7): removing variance ⇒ immediate collapse. The two anti-collapse terms are applied **per-branch independently** — no negatives, no EMA, no stop-gradient required.
- **Barlow Twins** (Zbontar et al., ICML 2021; arXiv:2103.03230): cross-correlation matrix → identity (on-diag 1, off-diag 0); no negatives/batch-size/predictor.
- **W-MSE / whitening** (Ermolov et al., ICML 2021; arXiv:2007.06346): hard full-rank enforcement by whitening the batch covariance (stronger than VICReg's soft penalty; cost = matrix inversion).
- **Orthogonal regularization** ("Preventing Dimensional Collapse in SSL via Orthogonal Regularization," NeurIPS 2024; arXiv:2411.00392): enforce orthogonality on weight matrices/features to maintain rank directly. (Synergy note: Muon *already* orthogonalizes its updates via NS5 — OR on the concept matrices would be a natural complement.)
- **Free-bits / capacity annealing** (VAE lineage): free-bits (Kingma et al., arXiv:1606.04934) — per-dimension KL hinge guaranteeing a minimum information per latent; β/cyclical annealing (Burgess arXiv:1804.03599; Fu et al. arXiv:1903.10145) — schedule β so dimensions come online. The *per-dimension info-floor* idea is structurally identical to VICReg's per-dimension variance hinge.

### Family D — Cap decoder capacity (structural anti-bypass)
A weak/shallow decoder *cannot* do the reconstruction work, so the latent must. Evidence: TSDAE Table 4 (BART/T5 bypass), I-JEPA narrow predictor (Table 14: width 384 → 70.7 vs 1024 → 68.4), MAE's asymmetric lightweight decoder. Architectural; defer (a decoder redesign, not a knob).

---

## 3. Adopt / Adapt / Watch / Reject (for E05)

| Lever | Family | Verdict | Rationale |
|---|---|---|---|
| **Decoder word-dropout on the suffix** (`DECODER_WORD_DROPOUT=0.3-0.5`, *already wired*) | A | **ADOPT (primary)** | One config change; TSDAE-proven; kills the bypass by construction (decoder can't copy destroyed tokens). Once concepts carry gradient pressure, wd can no longer selectively shrink them → fixes *both* mechanisms at the root. |
| **VICReg variance+covariance loss on the concept matrix** | C | **ADOPT (additive)** | Drop-in loss term; directly counteracts the measured rank-1 collapse + Jing-et-al. wd-driven collapse; no negatives/EMA needed; cheap. Belt-and-suspenders on top of Family A. |
| CPC/InfoNCE on future-suffix concept latents | B | **ADAPT** | Non-bypassable and principled, but needs a target-encoder/view machinery; adapt to the concept setting as a follow-up if A+C underperform. |
| **JEPA concept prediction (latent target, no decoder)** | B | **WATCH (= E06)** | The most principled non-bypassable architecture, but a bigger change; keep as the separate E06 bet. |
| Free-bits / β-cyclical annealing (VAE) | C | **WATCH** | E05 has no KL posterior, so the VAE machinery doesn't transfer directly; only the *per-dim info-floor* idea maps (already covered by VICReg's variance hinge). |
| W-MSE whitening; orthogonal regularization | C | **WATCH** | Stronger/alternative anti-collapse; whitening is less stable (matrix inversion); OR could pair with Muon. Try only if VICReg is insufficient. |
| Weak/shallow decoder | D | **WATCH (deferred)** | Structural anti-bypass but a decoder redesign; not a knob. |

*Nothing is a hard **Reject** — all are plausible; the table is ordered by ease × directness × proof for the E05 mechanism.*

---

## 4. Concrete recommendation for E05 (→ spec [E05c](../experiments_specs/ahead/E05c_anticollapse_extension.md))

**Two minimal, additive, proven levers, each mapped to a diagnosed sub-mechanism:**

1. **Non-bypassable objective (root cause):** `DECODER_WORD_DROPOUT=0.3` (then sweep 0.2/0.5) — destroy the suffix tokens the bypass copies. Reuses the existing knob; one config change; TSDAE-proven. *Expected:* Δzero_beyond / Δshuffle_beyond rise above the Stage-1 floor (the decoder is forced to read concepts for the dropped positions); concept rank stops collapsing because the concept channel now carries gradient.
2. **Direct anti-collapse loss (wd-driven collapse):** a VICReg-style penalty on the `[B, C, H]` concept batch — variance hinge `Σ_j max(0, γ − std(z_:j))` (keep each of the C·H concept dims active) + covariance `Σ_{i≠j} C(Z)²_{ij}` (decorrelate → forbid rank-1). Applied to the encoder's concept output each step, small weight (start ν≈1e-3 relative to the AR CE). *Expected:* holds within-sample RankMe up even if any residual bypass survives; directly opposes the Jing-et-al. implicit-regularization collapse.

**Why not JEPA/CPC for E05c:** they are the better long-term answer (and = E06), but they are a new objective/machinery, not an *easy* E05 extension. E05c stays additive so it isolates the objective/loss question from the architecture question.

**Caveat / extrapolation limit (flagged honestly):** nearly all the evidence above is vision-SSL or vision/text-VAE. No paper directly measures **effective rank of a text concept-bottleneck** under these regularizers. The recommendation is *structurally* well-grounded (the decorrelation/variance principles are architecture-agnostic) but **empirically unverified in this exact setting** — which is exactly what E05c tests.

---

## Sources
- VICReg — arXiv:2105.04906 (ICLR 2022) · code github.com/facebookresearch/vicreg
- Barlow Twins — arXiv:2103.03230 (ICML 2021)
- TSDAE — arXiv:2104.06979 (EMNLP 2021 Findings)
- MAE — arXiv:2111.06377 (CVPR 2022)
- I-JEPA — arXiv:2301.08243 (CVPR 2023)
- CPC / InfoNCE — arXiv:1807.03748
- Free-bits (Kingma, IAF) — arXiv:1606.04934 (NeurIPS 2016)
- Variational Lossy Autoencoder — arXiv:1611.02731 (ICLR 2017)
- Cyclical annealing — arXiv:1903.10145 (NAACL 2019); β-VAE / Burgess — arXiv:1804.03599
- W-MSE / whitening — arXiv:2007.06346 (ICML 2021)
- Understanding Dimensional Collapse / DirectCLR — arXiv:2110.09348 (ICLR 2022)
- Orthogonal Regularization for SSL — arXiv:2411.00392 (NeurIPS 2024)
- Concept Bottleneck Models — arXiv:2007.04612 (ICML 2020); Label-Free CBMs — arXiv:2304.06129

*Related: E05 run report [`e05_muon_long_2ep_collapsed_20260709.md`](../2_Experiments_Registry/run_reports/e05_muon_long_2ep_collapsed_20260709.md); E05 spec [`E05_windowed_decoder_concept_memory.md`](../experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md); control spec [`E05b_wd_confound_control.md`](../experiments_specs/done_success/E05b_wd_confound_control.md); extension spec [`E05c_anticollapse_extension.md`](../experiments_specs/ahead/E05c_anticollapse_extension.md).*
