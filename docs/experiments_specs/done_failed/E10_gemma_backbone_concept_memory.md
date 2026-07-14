# E10 — Pretrained-backbone concept memory (Gemma-3-1B graft: global→concept read + recurrent write)

- **Status:** **done / failed mechanism gate (2026-07-13).** The 100M concept-arm pilot
  completed stably, but all beyond-local recurrent-state ablations stayed below 0.001 nats.
  E10b–E10e subsequently closed the calibration branch without recovering persistent memory;
  the unmatched control leaves the original recovery-fraction comparison unresolved, not active.
- **Serves:** the platform pivot decided 2026-07-08 — stop paying the from-scratch language-acquisition
  cost on every run; graft the concept machinery onto a **pretrained decoder** and make the C concepts a
  **running memory** (E09's write op) with **windowed generation** (E05's coherence window). This is the
  long-context bet's new Stage-A: O(C·N) processing on a backbone that already speaks English, testable
  for length extrapolation beyond the training horizon.
- **Implementation plan:** [E10_gemma_backbone_concept_memory_plan.md](E10_gemma_backbone_concept_memory_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-08 · closed 2026-07-13

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
- **RECURRENCE ATTRIBUTION (co-primary):** on the same paired documents/positions, real recurrent
  state beats (a) the learned static `z0`/read branch with writes disabled by **≥ 0.05 nats** at
  2K and (b) a previous-block-only state by **≥ 0.02 nats** at 8K; paired-bootstrap 95% CIs must
  exclude zero. Concept-vs-no-concept measures utility; these within-arm ablations establish that
  any gain comes from accumulated document memory rather than static prompt/additional capacity.
- **Length extrapolation (the 10M-path signal):** at eval seq 8192 (16 blocks — 4× the training
  horizon), concept arm still beats control at positions ≥ 1024 by **≥ 0.2·G₈ₖ** (no state
  collapse with block count).
- **Ablation sanity:** Δshuffle (real vs batch-shuffled concept state) **≥ 0.1 nats** at
  positions ≥ 1024 (past the explicit one-block carry) — the decoder reads the *content* of the
  recurrent state, not merely its presence or redundant information still available locally.
- **MUST-NOT-REGRESS:** concept-arm CE at positions < 512 (pure local regime) within **+0.02 nats**
  of control (concepts must not tax short-range fluency); within-sample RankMe of the final concept
  state ≥ 0.3·C (no collapse of the recurrent state).

## Kill criteria (set BEFORE running)
- **Stage 0 gate:** G < 0.05 nats at seq 2048 → do not train; re-scope to longer sequences first.
- **Training:** define paired improvement as `control CE − concept CE`. At the matched 50%-token
  checkpoints, stop if improvement at positions ≥1024 is **≤ 0.01 nats**; if the control is not
  yet available, use the within-arm recurrence proxy `static CE − recurrent CE ≤ 0.01` together
  with near-zero read/write gates, then confirm against control before any extension.
- **Collapse:** within-sample RankMe of the concept state < 0.15·C at any eval, or eval CE rising
  over 3 consecutive evals (divergence signature) → stop.

## Plan
- **Data:** the proven `smollm3_inspired_2k_e05` mix recipe, **re-pretokenized with the Gemma
  tokenizer** (`google/gemma-3-1b-pt`, 262K vocab) at seq 2048 → new manifest
  `smollm3_inspired_2k_e05` under the Gemma tokenizer cache key. Before training, freeze a
  deterministic, train-disjoint FineWeb-Edu eval-only manifest at 8192; the same long documents
  truncated to 2K and 8K are used for paired Stage-0 and final comparisons.
- **Compute:** Odra 3× RTX 3090 (concept arm), Polonez (control arm, can run in parallel).
  Budget **~2B tokens per arm** (LoRA fine-tune; ICAE/RMT-scale), est. ~50–70 GPU-h per arm.
  bf16, gradient checkpointing on, effective batch 72. Odra calibration (2026-07-11):
  per-device 8 × 3 GPUs × accum 3 (19.97 GiB peak; batch 10 OOM); matched Polonez control:
  per-device 6 × 4 GPUs × accum 3.
- **Steps / epochs:** 2B non-padding Gemma-token target per arm (deterministic rounding ≤ one
  optimizer batch); derive the epoch fraction and optimizer steps from the completed manifest
  checksum/token count. Save fixed ~10%-budget
  checkpoints and compare the arms at matched 50% and 100% token exposure. Cosine LR 1e-4
  (LoRA-typical), warmup 500, `max_grad_norm` 0.5, seed 42.
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
- **Readiness audit (2026-07-11):** corrected the launcher to the frozen effective batch 72
  (calibrated as `8 × 3 GPUs × accum 3` on Odra; matched by `6 × 4 × 3` on Polonez); aligned live Δshuffle with
  the pre-registered ≥1024 region; added live within-sample RankMe + read/write-gate telemetry;
  enabled `backbone_concept` Tier-1 geometry/ablation checkpoint analysis; and added a production
  gradient-checkpointing regression test. The mix now explicitly declares `causal_lm`
  compatibility. A calibration-discovered tokenizer-only multimodal id outside the text model
  vocabulary is sanitized during tokenization and model-bounded in the collator. Local targeted
  suite: 33 passed.
- **Known deviations from prior-line assumptions (accepted, recorded):** token embeddings are the
  backbone's native 1152-dim (frozen — the "tiny token embedding" asymmetry applies to our write-op
  economics, not the backbone's embedding table); the SmolLM2 tokenizer is replaced by Gemma's
  (pretrained-backbone constraint), so CE numbers are **not** comparable with E01–E09.

## Result
- Run id: `backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847` (100M concept-arm pilot)
- WandB: [training run](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847)
- Run report: [e10_100m_concept_pilot_20260711.md](../../2_Experiments_Registry/run_reports/e10_100m_concept_pilot_20260711.md)
- Verdict: **negative mechanism result; original paired comparison unresolved** — concepts
  stayed non-collapsed (RankMe 77.1; centered 123.1), but recurrent-memory utility was
  below 0.001 nats. The matched control needed for the original recovery-fraction criterion
  was never run; E10b–E10e nevertheless closed this interface by repeatedly failing direct
  recurrence-attribution gates.

## Follow-ups (indicative path, not a committed schedule)

E10 is the **mechanism test** for fixed-memory block recurrence on a real LM. It trains at seq
2048 and evaluates extrapolation at 8K (4×). The project's stated Vision
(`docs/1_Strategy_and_Plans/vision_and_goals.md`) targets 1M-token context (C scaling to 8K–16K,
128× compression); 10M is the hardware ceiling on 24 GB cards (needs 80 GB / ~30 GPUs per the
F6 sequence-parallelism work, 2026-06-27). The gap between E10 and that Vision is large by design —
E10 exists to falsify (or validate) the one architectural property the whole long-context bet
depends on: **fixed-memory recurrence over unbounded input** (block size 512, write every block,
positions reset per block ⇒ per-block compute and memory are constant in sequence length; the same
architecture that trains at 2K can in principle run at 2K / 8K / 32K / 128K / 1M / 10M — the loop
just iterates more times).

The path from E10 to the 1M/10M Vision is staged. Each step is **conditional on E10's outcome** —
this is an indicative plan, not a committed schedule.

| Stage | Length | What it tests | Trigger condition |
|---|---|---|---|
| **E10** (this spec) | train 2K, eval 8K | does the recurrence work at all? does the concept arm beat the matched control? | 100M concept pilot complete: recurrence null; matched control pending |
| **E10 extrapolation probe** (no retraining) | eval 16K, 32K, 64K on the trained checkpoint | how far does a 2K-trained recurrence actually extrapolate before the state collapses or saturates? | always — cheap, single-GPU, runs after E10 |
| **C-scaling sweep** | train 2K with C ∈ {128, 256, 512, 1024}, eval 8K | is 128 enough at longer reach, or does the bottleneck need more slots? | E10 PRIMARY passes but extrapolation regresses → suspect information capacity |
| **Length-curriculum arm** | train 2K-majority + 4K/8K tail (LongLoRA-style) | does training on longer docs extend the recurrence's reach? | E10 extrapolation criterion fails (concept state collapses past 8K) |
| **Mechanism-variant A/B** | E11 (in-sequence mem-tokens) vs E12 (per-layer KV prefix) | is "global→concept" (Design C) the right read path, or is depth-starved? | E10 read gates g_ℓ stay near zero (Stage 0 of E10 flags this); E11/E12 specs already drafted |
| **1M demonstration** | eval 1M on 3× 3090 (F6 sequence parallelism) | does the unrolled loop hold useful state over ~2K blocks? | the extrapolation probe above stays healthy to at least 64K |

### What E10 deliberately does NOT prove about 1M/10M (recorded so we don't fool ourselves)

1. **It tests 4× extrapolation (2K→8K), not 500× (2K→1M).** The recurrent-state collapse failure
   mode (Infini-attention §D warning; the spec's RankMe kill-gate exists for this) could manifest
   anywhere between 8K and 1M even if 8K is fine. The extrapolation probe is the cheap first read.
2. **C is fixed at 128, the 2K-regime number from the Vision table.** Whether 128 slots suffice at
   1M is the C-scaling sweep's question. The within-sample RankMe guard is the leading indicator:
   if rank stays healthy as inference length grows, 128 may suffice; if it collapses past 32K, the
   bottleneck needs more slots.
3. **Linear ≠ free.** O(N·(K+C)) is linear in N, but at N=1M with K=512 that is ~2K blocks × a
   per-block forward of a 1B model — hours per sequence on a 3090 even with the concept machinery.
   The Vision's 1M claim is "tractable vs O(N²) full attention", not "fast".

### Falsification branches (decide before spending the next budget tier)

- **E10 PRIMARY fails** (concept arm ≈ control at positions ≥ 1024) → concepts carry nothing the
  window doesn't on a pretrained backbone. The global→concept design is wrong; pivot to E11/E12
  before any longer-context work.
- **E10 PRIMARY passes but extrapolation collapses past 8K** → the recurrence works in-distribution
  but doesn't generalize. Trigger the length-curriculum arm; if that also fails, the fixed-C
  recurrence paradigm is in trouble and the Vision's "C scales with N" leg needs a mechanism for
  *growing* C at inference (not just training a bigger fixed C).
- **E10 read gates stay near zero** → the global-layer read path is depth-starved or mis-initialized.
  Pivot to E12 (per-layer KV prefix) which injects at every layer, not just the 4 global ones.
- **E10 passes cleanly on all five criteria** → proceed to the extrapolation probe, then C-scaling,
  then the 1M demonstration, in that order.

## References
- Design discussion (2026-07-07/08 chat): Design C chosen over A (in-sequence mem-tokens → E11) and
  B (per-layer KV prefix → E12).
- `docs/literature_review/recurrent_memory_transformers.md` — RMT §B (pretrained-backbone retrofit,
  the on-ramp precedent), Block-Recurrent §B (gated write), Infini-attention §D (iterated-memory
  collapse warning → zero-init gate + RankMe kill-gate).
- LongLoRA (arXiv:2309.12307) — attention-pattern surgery + LoRA on a pretrained LM works.
- Gemma 3 (google/gemma-3-1b-pt): 26L, H1152, SWA 512, 5:1 local:global — the ready-made socket.
- Stage 0 results (2026-07-09): `docs/2_Experiments_Registry/run_reports/e10_stage0_gap_curve_20260709.md`
  — G(pos≥1024) ≥ 0.05 gate cleared at every measured length (2K/4K/8K/16K); G(8K)/G(2K) = 1.12×
  (gentle growth ⇒ no curriculum amendment, spec as-written is well-posed).
