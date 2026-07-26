# E05c — Non-bypassable objective: decoder word-dropout on the suffix

- **Status:** **on hold / unrun.** [E05b](../done_success/E05b_wd_confound_control.md)
  resolved the weight-decay confound on 2026-07-11; E05c remains a valid objective-side
  follow-up but is deprioritized behind the pretrained-backbone/interface track.
- **Serves:** the E05 collapse diagnosis — the K=128 windowed **decoder bypass** is the root attractor (it predicts the suffix from its *own* within-window tokens, routing around the concepts). This experiment makes the bypass **impossible by construction** (TSDAE principle: destroy the decoder's input access → it must read concepts). The cheap, proven, config-only extension the E05 track was missing. Mechanistic follow-up; E10 remains the headline focus.
- **Implementation plan:** none needed — config only (the knob exists).
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-09
- **Literature:** [concept_bottleneck_collapse_mitigation.md](../../literature_review/concept_bottleneck_collapse_mitigation.md) (Family A).

> One experiment = one hypothesis = one changed variable. The single variable is **decoder word-dropout** on the suffix (0 → 0.3). Held fixed = the collapsed Muon 0.5-ep recipe.

## Hypothesis
If the within-window decoder bypass is the root cause of the concept collapse, then destroying the decoder's direct access to suffix tokens (`DECODER_WORD_DROPOUT=0.3`, TSDAE-style: replace 30% of decoder-input embeddings with a learned mask, training-only) will **force the decoder to read those tokens from the concepts** → `Δshuffle_beyond` rises above the Stage-1 floor (≥0.3) and within-sample RankMe recovers (≥20). Because the concept channel then carries gradient pressure, wd can no longer selectively shrink it (the diagnosed wd-driven collapse mechanism is defused at the root). TSDAE (arXiv:2104.06979) shows exactly this: confining/corrupting decoder input forces a meaningful bottleneck, and over-capable/uncorrupted decoders bypass.

## Builds-on
- **Foundation:** the shared E05 launcher; `DECODER_WORD_DROPOUT` is already wired end-to-end — `train_perceiver_denoise_multigpu.sh:51` → `:265 --decoder_word_dropout`, consumed at `nn/concept_encoder_perceiver.py:1214-1216` (replaces decoder-input embeddings with the learned `self.dropout_embedding` with prob `p`, **training-only**; eval/`ce_intact` stay clean per `:1810`). No new code.
- **Init / checkpoint:** random init, seed 42 (identical to the Muon arm).
- **Baseline to beat:** `concept_ar_prefix_H768L6C128D4_20260702_031956` (**E05 Muon 0.5-ep arm**, wd=0.1, word-dropout=0) — within-sample RankMe **10.57**, **Δshuffle_beyond 0.209** (fails Stage-1 floor ≥0.3), Δzero_beyond 0.41, STS-B 0.518, eval_loss 2.606.

## The single change
`DECODER_WORD_DROPOUT=0.3` (was 0.0). Held fixed = the stabilized Muon 0.5-ep recipe: `OPTIMIZER=muon`, LR 0.01, wd 0.1, `MUON_ADAMW_LR=2e-4`, clip 0.5, effective batch 72, `NUM_EPOCHS=0.5`. (Muon baseline chosen so the fix is tested on the *actually collapsed* model; wd/Muon are held identical to the baseline so word-dropout is the only variable.)

## Success criteria (set BEFORE running)
- **Δshuffle_beyond ≥ 0.3** (Stage-1 floor; baseline 0.209) — the primary "concepts are now necessary" gate.
- **Within-sample RankMe ≥ 20** (baseline 10.57) — the collapse reverses.
- Δzero_beyond ≥ 1.0 (baseline 0.41) — secondary read that the decoder genuinely reads concepts.
- Stable end-to-end (no divergence).

## Kill criteria (set BEFORE running)
- Divergence: grad_norm > 1e4, loss non-finite, or eval_loss rising over 3 evals.
- If at 0.25 ep Δshuffle_beyond is still < 0.2, the bypass is not killed by p=0.3 → stop and retry p=0.5 (or move to span-masking / the VICReg co-loss in E05d) before spending the full budget.
- If eval_loss explodes (word-dropout too aggressive for the AR objective), lower p.

## Plan
- **Data:** `smollm3_inspired_2k_e05` (same pretokenized manifest).
- **Compute:** Odra (3× 3090); ~**75 GPU-h** (≈ the Muon 0.5-ep arm).
- **Steps / epochs:** 0.5 ep / 69,142 steps.
- **Checkpointing:** same improved cadence as E05b — `SAVE_TOTAL_LIMIT=40 EVAL_STEPS=2000 SAVE_STEPS=2000`.
- **Launch:**
  ```bash
  OPTIMIZER=muon DECODER_WORD_DROPOUT=0.3 \
  SAVE_TOTAL_LIMIT=40 EVAL_STEPS=2000 SAVE_STEPS=2000 \
  SKIP_PRETOKENIZE=1 bash scripts/launch_e05.sh
  ```
  *(sweep `DECODER_WORD_DROPOUT` 0.2 / 0.3 / 0.5 if needed; the knob is continuous.)*
- **New foundation code:** none — config only.

## If it works / doesn't
- **Works (gates met):** the windowed decoder's bypass was the root cause and is cheaply killable. This rehabilitates the E05 from-scratch windowed decoder (and informs E10's concept read). Next: confirm semantics hold (run the Tier-1+STS-B eval), then optionally stack the VICReg co-loss ([E05d](E05d_concept_vicreg.md)) for belt-and-suspenders.
- **Bypass killed (Δshuffle_beyond up) but RankMe still low:** wd is still collapsing now-necessary-but-weak directions → add VICReg (E05d) or reduce wd on `bixt`.
- **Bypass not killed (Δshuffle_beyond flat):** p too low, or the windowed self-attention reconstructs from the *un-dropped* neighbors — move to contiguous span-masking (force multi-token gaps) in E05d-adjacent.

## Result
<Filled in AFTER, by experiment-track.>
- Run id: `<run_id>`
- WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
