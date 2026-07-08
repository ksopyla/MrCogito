# E10 — Pretrained-backbone concept memory (Gemma-3-1B graft: global→concept read + recurrent write)

- **Status:** draft
- **Serves:** the platform pivot decided 2026-07-08 — stop paying the from-scratch language-acquisition
  cost on every run; graft the concept machinery onto a **pretrained decoder** and make the C concepts a
  **running memory** (E09's write op) with **windowed generation** (E05's coherence window). This is the
  long-context bet's new Stage-A: O(C·N) processing on a backbone that already speaks English, testable
  for length extrapolation beyond the training horizon.
- **Implementation plan:** [E10_gemma_backbone_concept_memory_plan.md](E10_gemma_backbone_concept_memory_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-08 · closed —

> **This is a platform-introduction experiment (E01/E08-style), not a one-knob A/B.** The graft
> (backbone + concept read path + concept write op) is one coordinated mechanism; splitting it would
> dissolve the hypothesis. Attribution comes from the **matched no-concept control arm** (identical
> backbone, LoRA, masks, data, budget — concepts removed), not from a single-variable frame the
> paradigm cannot satisfy. Design variants (in-sequence mem-tokens = E11, per-layer KV prefix = E12)
> are separate follow-up specs.

## Hypothesis
If we take frozen **`google/gemma-3-1b-pt`** (whose attention is already factored into 22 local
sliding-window-512 layers + 4 global layers), **replace the global layers' full-attention reach with
reads of C=128 concept slots** (tokens attend [concepts + local window] instead of all N), and make
the concepts a **gated recurrent state written from each decoded 512-token block** (BiXT write op,
zero-init gate), then after a LoRA fine-tune the concept arm will **recover ≥ 40% of the long-range
CE gap** between the windowed-only control and full-attention Gemma at positions beyond local reach —
because the backbone was pretrained to route long-range information through exactly those 4 global
layers, and the concept state is the only long-range channel we leave open. (Falsifiable: if the
concept arm's beyond-local CE is not measurably below the matched no-concept control, concepts carry
nothing the window doesn't.)

## Builds-on
- **Foundation:** `training/train_perceiver_denoise.py` + `scripts/train_perceiver_denoise_multigpu.sh`
  (the single shared entrypoint; the graft is a new config-selectable model family on it, never a fork);
  `nn/concept_encoder.py:BiXTCrossAttention` (`update_tokens=False`) as the concept **write op** —
  the E09 `ConceptWriteHead` design, executed here on the pretrained backbone; the
  pretokenize→manifest→train spine (`scripts/pretokenize_mix.py`); HF `Gemma3ForCausalLM`
  (transformers 4.57.6, per-layer-type mask-dict API).
- **Init / checkpoint:** backbone = `google/gemma-3-1b-pt` frozen + LoRA (r=16, q/k/v/o all layers);
  new params (concept init z0, write head, gates) trained in full. **Zero-init gate α=0 ⇒ the
  concept state is inert at step 0** (writes are identity), and `concept_num=0` reproduces the pure
  windowed-blocks control path exactly.
- **Baseline to beat:**
  - **Control arm (trained):** same backbone/LoRA/masks/data/budget with `CONCEPT_NUM=0` — the
    windowed-only Gemma. The concept arm must beat it at beyond-local positions.
  - **Upper baseline (no training):** intact `gemma-3-1b-pt`, full global attention — the CE our
    O(C·N) graft tries to approach. Measured in Stage 0 together with the untrained windowed-only
    CE; their difference is the **gap G** the concepts must close.
  - Prior-line reference (not a target): E05 Adam beyond-window Δshuffle 0.39, STS-B 0.452 — the
    from-scratch numbers this platform is meant to leapfrog.

## The single change (platform-level)
Concept I/O grafted onto a frozen pretrained backbone, Design C ("global→concept"):
1. **Read:** C=128 concept slots enter the sequence as soft embeddings; **only the 4 global layers'
   tokens may attend them** (mask-dict surgery). All token↔token attention is sliding-window-512
   causal in **every** layer (the global layers lose full reach) → O(N·(K+C)) total.
2. **Write:** after each 512-token block, `z ← z + tanh(α)·RMSNorm(BiXT_lat←tok(z, block_hiddens))`
   with α zero-init (the E09 recurrence, fed by the backbone's own final-layer hidden states).
3. **Recurrent encode = recurrent decode:** the prefix/document is consumed block-by-block through
   the same write op — there is **no separate encoder**; input length is unbounded at fixed memory.

**Held fixed:** the backbone (frozen weights, tokenizer, sliding window 512, layer pattern), block
size = write cadence = 512, one-block carry for window continuity, plain next-token CE, LoRA config,
data mix, budget, seed. Control arm differs ONLY in `CONCEPT_NUM=0`.

## Success criteria (set BEFORE running)
- **Stage 0 (prerequisite, no training):** on held-out long docs, measure per-position-bucket CE for
  (a) intact Gemma full attention and (b) the E10 block-recurrent protocol with concepts off
  (windowed blocks + one-block carry — `analysis/run_e10_stage0.py`). Define
  **G = CE(b) − CE(a) averaged over positions ≥ 1024** (beyond one window+carry of local reach) at
  seq 2048, and the same at 8192. **G must be ≥ 0.05 nats at 2048** — otherwise windowing doesn't
  hurt Gemma at this length, the experiment cannot detect concept value at seq 2048, and the training
  seq/eval protocol must be re-scoped to longer sequences BEFORE any GPU spend.
- **PRIMARY (decisive):** at the final checkpoint, concept-arm CE at positions ≥ 1024 is below the
  matched control arm by **≥ 0.4·G** (concepts recover ≥ 40% of the long-range gap) on the held-out
  eval split at seq 2048.
- **Length extrapolation (the 10M-path signal):** at eval seq 8192 (16 blocks — 4× the training
  horizon), concept arm still beats control at positions ≥ 1024 by **≥ 0.2·G₈ₖ** (no state
  collapse with block count).
- **Ablation sanity:** Δshuffle (real vs batch-shuffled concept state) **≥ 0.1 nats** at
  beyond-local positions — the decoder reads the *content* of the state, not just its presence.
- **MUST-NOT-REGRESS:** concept-arm CE at positions < 512 (pure local regime) within **+0.02 nats**
  of control (concepts must not tax short-range fluency); within-sample RankMe of the final concept
  state ≥ 0.3·C (no collapse of the recurrent state).

## Kill criteria (set BEFORE running)
- **Stage 0 gate:** G < 0.05 nats at seq 2048 → do not train; re-scope to longer sequences first.
- **Training:** concept arm minus control at positions ≥ 1024 is **≤ 0.01 nats at 50% of budget**
  (gate never learned to open / state carries nothing) → stop, report the ablation read.
- **Collapse:** within-sample RankMe of the concept state < 0.15·C at any eval, or eval CE rising
  over 3 consecutive evals (divergence signature) → stop.

## Plan
- **Data:** the proven `smollm3_inspired_2k_e05` mix recipe, **re-pretokenized with the Gemma
  tokenizer** (`google/gemma-3-1b-pt`, 262K vocab) at seq 2048 → new manifest
  `smollm3_inspired_2k_e05` under the Gemma tokenizer cache key. Held-out 8K eval slice: a small
  fineweb-edu long-doc set pretokenized at 8192 (Stage-0/extrapolation eval only, not training).
- **Compute:** Odra 3× RTX 3090 (concept arm), Polonez (control arm, can run in parallel).
  Budget **~2B tokens per arm** (LoRA fine-tune; ICAE/RMT-scale), est. ~50–70 GPU-h per arm.
  bf16, gradient checkpointing on, per-device batch tuned to fit (start 4 × accum 6).
- **Steps / epochs:** ~0.1 epoch of the mix (≈2B tokens); cosine LR 1e-4 (LoRA-typical), warmup 500,
  `max_grad_norm` 0.5, seed 42.
- **Launch:** `bash scripts/launch_e10.sh` (thin wrapper over the shared launcher, pattern of
  `launch_e05.sh`) — concept arm default; control arm: `CONCEPT_NUM=0 bash scripts/launch_e10.sh`.
- **New foundation code (reusable, config-selectable — via `research-implement`):**
  - `nn/backbone_concept_lm.py` — `BackboneConceptLM` (+ its `PretrainedConfig`): wraps a HF
    backbone with per-layer-type mask surgery, concept slots, block-recurrent forward,
    `ConceptWriteHead` (reuses `BiXTCrossAttention`), `concept_ablation_ce` + `encode_concepts`
    (metric-contract compatibility with the existing trainer eval hooks).
  - `BACKBONE_MODEL` / `CONCEPT_NUM` / block + LoRA knobs wired through the shared entrypoint
    (existing model families byte-identical when `BACKBONE_MODEL` is unset).
  - `peft` added as a dependency (LoRA).
- **Known deviations from prior-line assumptions (accepted, recorded):** token embeddings are the
  backbone's native 1152-dim (frozen — the "tiny token embedding" asymmetry applies to our write-op
  economics, not the backbone's embedding table); the SmolLM2 tokenizer is replaced by Gemma's
  (pretrained-backbone constraint), so CE numbers are **not** comparable with E01–E09.

## Result
<Filled in AFTER, by experiment-track. Link out; do not paste full results here.>
- Run id: `<run_id>`
- WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>

## References
- Design discussion (2026-07-07/08 chat): Design C chosen over A (in-sequence mem-tokens → E11) and
  B (per-layer KV prefix → E12).
- `docs/literature_review/recurrent_memory_transformers.md` — RMT §B (pretrained-backbone retrofit,
  the on-ramp precedent), Block-Recurrent §B (gated write), Infini-attention §D (iterated-memory
  collapse warning → zero-init gate + RankMe kill-gate).
- LongLoRA (arXiv:2309.12307) — attention-pattern surgery + LoRA on a pretrained LM works.
- Gemma 3 (google/gemma-3-1b-pt): 26L, H1152, SWA 512, 5:1 local:global — the ready-made socket.
