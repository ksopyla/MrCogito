# E06 — Latent-space prediction (anchor as the primary objective)

- **Status:** canceled — rejected before implementation/run because it was an auxiliary
  representation objective rather than the selected architecture-level direction
- **Serves:** the [agenda](../../1_Strategy_and_Plans/agenda.md) collapse focus and the "**reason in latent space**" Vision. Removes the token-level bypass *entirely* by moving the learning signal out of token space: the concepts must predict a frozen teacher's per-token **representations**, not reconstruct tokens. JEPA / data2vec / CPC family ([2301.08243], [2202.03555], [1807.03748]).
- **Implementation plan:** E06_latent_space_prediction_plan.md *(to author after approval)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-06-14 · closed —

> One changed variable vs E03: the anchor moves from **auxiliary** (`λ·MSE` added to token CE) to the **primary** objective (token CE removed or down-weighted to a tiny readout). Everything else (teacher, head, data) is E03's.

## Hypothesis
If the concept bottleneck is trained **primarily** to predict the frozen SmolLM2-135M per-token hidden states (standardized MSE / JEPA-style), with token cross-entropy removed (or kept at a tiny weight only as an interpretable readout), then the per-sample manifold RankMe and zero-shot STS-B will be **≥ the E03 anchor-ON run**, because the objective lives entirely in representation space where there is **no teacher-forced token side-channel to bypass** — at the cost of generation fluency, which we explicitly do not optimize here.

## Builds-on
- **Foundation (reuse, no fork):** E03's `AnchorDistillHead` + frozen-teacher path + `compute_anchor_loss` (`anchor_standardize=True` for data2vec-style target normalization to prevent representation collapse), `ConceptEncoder`/BiXT, shared entrypoint. New: a config switch to make the anchor the primary loss.
- **Init / checkpoint:** random init; optional warm-start from the E03 anchor checkpoint.
- **Baseline to beat:** the **E03 anchor-ON** run (manifold RankMe + STS-B), and E02 STS-B 0.702.

## The single change
**Loss weighting:** `total = λ_tok·CE + anchor_MSE` with **λ_tok = 0** (pure latent prediction) — vs E03's `total = CE + λ_anchor·MSE` (token CE primary). Optionally a tiny `λ_tok` (e.g. 0.05) to keep a token readout for inspection. Teacher, head, data, tokenizer, seq-len identical to E03.

## Success criteria (set BEFORE running)
- **Primary:** manifold **RankMe ≥ E03-anchor** AND zero-shot **STS-B ≥ E03-anchor STS-B** (≥ 0.65 target).
- **Anti-collapse sanity:** anchor MSE well below a mean-prediction baseline; concept anisotropy < 0.4 (targets standardized → guards the data2vec-style constant-collapse failure).

## Kill criteria (set BEFORE running)
- By 25% budget: if manifold RankMe **and** STS-B are both **≤ E03-anchor**, latent-only training adds nothing over the auxiliary form → stop.
- If representations collapse (anchor MSE → 0 with anisotropy → 1, i.e. constant outputs) despite standardization → stop, revisit target normalization (per-dim running stats).

## Plan
- **Data:** `HuggingFaceFW/fineweb-edu` `sample-10BT`, SmolLM2-135M tokenizer (== E03; shared tokenizer mandatory for 1:1 teacher token alignment).
- **Compute:** Odra/Polonez 3–4× RTX 3090, bf16; one extra frozen-teacher forward/step (as E03).
- **Steps / epochs:** matched to E03 (0.3-epoch gate → 1 epoch).
- **Launch (after the `λ_tok` / primary-anchor switch lands):**
  ```bash
  EXPERIMENT_ID=E06 DECODER_TYPE=causal_ar ANCHOR_LOSS=true ANCHOR_MODEL=HuggingFaceTB/SmolLM2-135M \
  ANCHOR_STANDARDIZE=true TOKEN_LOSS_WEIGHT=0.0 \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 CONCEPT_NUM=128 \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  SEED=42 NUM_EPOCHS=0.3 PER_DEVICE_BATCH_SIZE=24 GRADIENT_ACCUMULATION_STEPS=2 \
  uv run bash scripts/train_perceiver_denoise_multigpu.sh
  ```
- **New foundation code:** a reusable `token_loss_weight` knob (default 1.0 = unchanged) so the token CE can be down-weighted/disabled while the anchor MSE drives training; thread through `ModelArguments` + launcher. Consider a future masked-span / future-segment latent target (CPC-style) as a follow-up variable.

## Result
No run was launched. The design was deliberately canceled before implementation.
