# E03 — De-collapse via a frozen-encoder hidden-state anchor

- **Status:** anchor-ON warmup done (2026-06-15); matched control queued on Odra next
- **Serves:** the encoder→AR-decoder Current focus in [agenda.md](../1_Strategy_and_Plans/agenda.md), and the chronic open problem behind it — **concept collapse** (effective rank stuck at 5–10/128 across ~60 runs). Attacks the root cause that blocks every downstream bet (AR generation, recursion, diffusion). The frozen-hidden-state anchor is the shared "validate the bottleneck first" ingredient the diffusion literature (Cosmos/LDLM/CALM) and the recursion line both rely on — and the one ingredient the team brief's P1 matrix does **not** currently contain.
- **Implementation plan:** [E03_concept_anchor_decollapse_plan.md](E03_concept_anchor_decollapse_plan.md) *(the HOW — repo-rooted design)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-06-13 · closed —

> One experiment = one hypothesis = one changed variable. Implementation is a config
> (`--anchor_loss true`) over the shared `train_perceiver_denoise.py` entrypoint and its
> launcher — NOT a new fork. The control arm is the *same config with the anchor off*. The
> spec is **frozen once the run starts**; results live in the registry + run report, not here.

## Hypothesis
If we add an **auxiliary per-token hidden-state distillation loss** — the concept bottleneck must
reconstruct a **frozen pretrained LM's per-token last-hidden-states** (MSE, via a parallel
position-query decoder) **on top of** the E01 concept-conditioned AR objective — then the concepts
will **de-collapse**: concept **effective rank** will rise to **≥ (matched no-anchor control) + 16
and ≥ 32/128**, and **zero-shot STS-B** will rise by **≥ 0.03 over the control (and ≥ 0.62)**,
**without** degrading AR generation (eval CE no worse than control by > ~0.2 nats; concept-ablation
**ΔCE ≥ 0.5**) — because forcing the orderless 128-concept set to regenerate a strong LM's
position-specific semantic representations spreads information across many concepts and grounds them
in external semantics, removing the degrees of freedom that let a self-supervised bottleneck collapse
to a low-rank subspace.

## Builds-on
- **Foundation (reuse + extend, no fork):**
  - Reuse as-is: the E01 AR stack — `ConceptEncoder` + BiXT encoder, `ConceptCausalDecoderStack`,
    `ConceptEncoderForConditionalLM` (`decoder_type="causal_ar"`); `DataCollatorForTSDAE`;
    `load_and_preprocess_text_dataset`; the shared entrypoint `training/train_perceiver_denoise.py`
    and launcher `scripts/train_perceiver_denoise_multigpu.sh`.
  - **Reuse the existing `PerceiverDecoderStack`** (parallel, position-only decoder, already present
    in `nn/concept_encoder_perceiver.py` and kept intact since E01 as the STS-B / representation
    probe) — now repurposed as an **MSE regression head** that predicts the frozen teacher's per-token
    hidden states from the concepts. No new decoder architecture.
  - New **reusable, config-selectable** components (via `research-implement`): a frozen-teacher wrapper
    + the anchor auxiliary loss, gated by `anchor_loss` (default `False`) so old configs/checkpoints
    are unaffected. **The anchor is reusable** by later experiments (it is the Stage-A validation that
    a diffusion decode or a recursion loop would build on).
- **Init / checkpoint:** random init (same as E01 — train from scratch). The frozen teacher
  (`HuggingFaceTB/SmolLM2-135M`) is loaded in eval mode, **no gradients**, used only to produce MSE
  targets.
- **Baseline to beat:** a **matched no-anchor control** = the identical E01 config with `anchor_loss`
  off, run at the same seed / data / steps / GPUs in the same session. This control doubles as a fresh
  E01-architecture baseline and removes the dependency on the offline Polonez E01 final
  (`concept_ar_H768L6C128D4_20260607_172931`). Broader collapsed-rank reference: 5–10/128; E01's own
  de-collapse gate was rank > 32/128.

## The single change
**Add the auxiliary anchor loss.** Total objective changes from E01's
`L = CE_AR` to:

`L = CE_AR + λ · MSE( standardize(h_teacher) , proj(decoder_pos(concepts)) )`

where `h_teacher` = frozen `SmolLM2-135M` per-token last-hidden-states on the **same `input_ids`**
(same tokenizer → 1:1 token alignment, **zero alignment code**), `decoder_pos` = the existing
`PerceiverDecoderStack` reading the 128 concepts with position queries, `proj` = a `Linear(hidden_size
→ 576)` to the teacher's hidden size, MSE taken over non-pad positions only. Everything else is held
**identical to E01**: `decoder_type=causal_ar`, FineWeb-Edu `sample-10BT`, SmolLM2 tokenizer
(pad=eos), `hidden_size=768`, `token_embedding_dim=256`, encoder `num_hidden_layers=6`,
`concept_num=128`, decoder `decoder_num_layers=4`, `intermediate_size=2048`, `hidden_act=silu`,
`norm_type=rmsnorm`, `decoder_pos_type=rope`, `max_seq_length=512`, `deletion_rate=0.6`,
`decoder_word_dropout=0.2`.

> **Why this is one variable, not several.** The control arm is byte-for-byte E01 config with the
> anchor disabled; the experiment arm flips `anchor_loss=true`. The MSE head reuses an existing module.
> The frozen teacher adds a forward pass and a `Linear`, but contributes **no trainable confound** to
> the encoder/decoder beyond the anchor gradient itself.
>
> **Defaults to calibrate in smoke tests (not design forks):** standardize `h_teacher`
> (per-dim mean/var, Cosmos/LDLM practice) for stable MSE; start `λ` (`anchor_loss_weight`) so the
> standardized MSE term is the same order of magnitude as `CE_AR` (initial guess `λ≈0.5–1.0`); teacher
> target layer = last hidden state. Confirm `SmolLM2-135M` hidden size (expected **576**) at
> implementation.

### De-collapse — the mechanism, and how we detect a *misleading* success
The risk is that effective rank rises **without** the concepts becoming useful or staying usable:
- **Rank inflation without semantics:** the anchor could spread concepts to mimic the (high-rank)
  teacher states while STS-B stays flat. → STS-B is a **co-primary** gate, not secondary.
- **Anchor wins, generation breaks:** the MSE could dominate and degrade the AR decoder. → AR eval CE
  and concept-ablation ΔCE are **guardrails**; the anchor must de-collapse *without* sacrificing the
  bottleneck's usability.
- **Trivial copying:** concepts could become a lossy copy of teacher states rather than a useful
  compression. → we still measure ΔCE (concepts must remain causally used by the AR decoder) and
  anti-collapse cosine stats.

## Success criteria (set BEFORE running) — judged on the matched pair (anchor ON vs OFF)
> **Amendment 2026-06-14 (metric upgrade, pre-verdict):** after the Phase-0 finding that the
> headline "effective rank" SVDs the **batch-averaged** concepts (so it measures *slot redundancy*,
> not the per-sample representation geometry that downstream tasks use), the **primary de-collapse
> metric is the per-sample manifold RankMe** + **early-position ablation Δ** (both added to
> `analysis/run_concept_analysis.py`). Slot-mean effective rank is kept only as a secondary
> diagnostic. The anchor-ON warmup is already running mislabeled as E01 in W&B (launched without
> `EXPERIMENT_ID=E03`); the **matched control and any rerun MUST pass `EXPERIMENT_ID=E03 ANCHOR_LOSS=…`**.
1. **De-collapse (headline, upgraded):** **manifold RankMe(anchor) ≥ RankMe(control) + 8** (per-sample
   pooled embeddings) AND **early-position Δzero(anchor) ≥ Δzero(control)**. Secondary/diagnostic:
   slot-mean effective rank(anchor) ≥ control + 16; anti-collapse mean pairwise cosine < 0.4, max < 0.8.
2. **Useful, not just spread (co-primary):** zero-shot **STS-B(anchor) ≥ STS-B(control) + 0.03** AND
   **≥ 0.62** (≥ prior best 0.607), from mean-pooled concepts.
3. **Generation not broken (guardrails):** AR eval **CE(anchor)** no worse than control by **> ~0.2
   nats**; concept-ablation **early Δzero ≥ 0.5** (concepts still causally used).
4. **Anchor actually learns:** anchor **MSE decreases** monotonically and the per-token regression is
   well below a mean-prediction baseline (sanity that the head + concepts carry the signal).

## Kill criteria (set BEFORE running)
- By **25% of the step budget**: if **rank(anchor) − rank(control) < +5** → the anchor is not
  de-collapsing, **stop**.
- By **25% of the step budget**: if AR eval **CE diverges upward** vs the control, or anchor **MSE does
  not decrease** → the anchor destabilizes/short-circuits training, **stop**.
- Compute cap: **> ~60 GPU-hours** across the pair without clearing the 25% gates → **stop**.

## Plan
- **Data:** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, same subsample/preprocessing as E01
  (eos appended per document). **Tokenizer must be `HuggingFaceTB/SmolLM2-135M`** — the anchor target
  requires the teacher and our model to share the tokenizer for 1:1 token alignment.
- **Model:** identical to E01 (see "The single change"). Anchor arm adds frozen `SmolLM2-135M` (eval,
  no grad) + the `PerceiverDecoderStack` MSE head + `Linear(768→576)`.
- **Compute:** Odra / Polonez, 3–4× RTX 3090, bf16, AdamW-fused + cosine (as in the reference
  launcher). Polonez is currently **down (needs restart)**; Odra is running **E02** — E03 queues after
  Odra frees or after Polonez is back. Each arm is cheap (frozen teacher forward + small MSE head).
- **Steps / epochs:** **(1)** a short **warm-up gate** (≈0.3 epoch, like the E01/E02 warmups) on **both
  arms** to check the 25% kill gates — anchor MSE decreasing, rank moving above control, AR CE not
  diverging. **(2)** if healthy, a matched **fuller pair** (≈1 epoch / E01 budget) for the final
  rank / STS-B / ΔCE verdict.
- **Launch (env-var overrides on the shared launcher; the only diff between arms is the anchor block):**
  ```bash
  # Experiment arm (anchor ON)
  EXPERIMENT_ID=E03 DECODER_TYPE=causal_ar \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
  OBJECTIVE_VARIANT=reconstruction DELETION_RATE=0.6 DECODER_WORD_DROPOUT=0.2 \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  ANCHOR_LOSS=true ANCHOR_MODEL=HuggingFaceTB/SmolLM2-135M \
  ANCHOR_LOSS_WEIGHT=0.5 ANCHOR_STANDARDIZE=true \
  bash scripts/train_perceiver_denoise_multigpu.sh

  # Control arm (anchor OFF) — identical command minus the ANCHOR_* block (== E01 config);
  # keep EXPERIMENT_ID=E03 so W&B groups it with the matched E03 pair.
  ```
- **New foundation code (reusable, via `research-implement`):** config fields `anchor_loss`
  (default `False`), `anchor_model_name`, `anchor_loss_weight`, `anchor_standardize`,
  `anchor_target_layer` (default last); a frozen-teacher wrapper that runs the anchor model on
  `input_ids` and returns standardized per-token hidden states (eval, `no_grad`); the anchor MSE head
  reusing `PerceiverDecoderStack` + `Linear(hidden_size→teacher_hidden)`; wire the auxiliary loss into
  `ConceptEncoderForConditionalLM.forward` and log `anchor_mse`; ensure the new family/metrics are
  visible to `analysis/run_concept_analysis.py` and `evaluation/concept_eval_routing.py`. Preserve the
  checkpoint eval contract and backward-compatible defaults.

## Result
**Anchor-ON warmup completed 2026-06-15. Control arm (anchor-OFF) not yet run — full verdict pending.**

- Run id: `concept_ar_H768L6C128D4_20260614_164206` (anchor-ON, 0.3 epoch) / TBD (control)
- WandB (anchor-ON): [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_H768L6C128D4_20260614_164206)
- Run report: `docs/2_Experiments_Registry/run_reports/e03a_anchor_on_warmup_20260615.md`
- Verdict: **inconclusive** — anchor arm passes all kill gates (MSE ↓, AR CE stable, concept ablation strong), but the matched control must be run to evaluate de-collapse criteria.
