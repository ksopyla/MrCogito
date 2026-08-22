# E17e — Starve the local window (K=256)

- **Status:** approved (named 2026-08-22; implement as config over E17d)
- **Serves:** Priority 1 / SG1–SG2: make previous-window concepts necessary for FinePDFs
  next-token CE by cutting the local computer in half, instead of retuning gates.
- **Implementation plan:** [E17e_starved_local_window_plan.md](E17e_starved_local_window_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-08-22 · closed —

> E17e is one coherent bet: **keep E17d's four depth-private attn-residual concept
> layers and no token carry, and shrink the local token window / write cadence from
> K=512 to K=256.** This is the starve diagnostic of the E17d five-whys, not a new
> cell. `E18` stays reserved on the unmerged addressable-RAM branch; do not reuse it.

## Hypothesis
If Gemma's local softmax and the concept write cadence both run at **K=256** instead of
K=512, with E17d's attn-residual four-bank cell and previous windows existing only as
banks, then by 300M tokens permuting those banks will raise CE in the **late half of
each 256-token window** (`delta_permutation_block_256_512` = offsets 128–256 of each
block) to **≥ 0.10**, because FinePDFs next-token CE can no longer be solved inside a
512-token local computer after ~64 tokens of the current page.

## Builds-on
- **Foundation:** E17d cell in `nn/backbone_concept_lm.py` (`concept_read_placement=
  "attn_residual"`, `inference_carry_policy="drop_after_first"`, untied additive
  writers). Shared `training/train_concept_pretraining.py` → `scripts/launch_e10.sh`.
  New reusable behavior: `concept_block` is the authority for the local window;
  loaded Gemma `sliding_window` is aligned to it (hub Gemma-3-1B ships 512).
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42,
  C=128, **K=256**, four banks, seq 4096. Do **not** warm-start E17d: write cadence
  and the pretrained SWA mask both change; a resume would confound the starve.
- **Baseline to beat:** E17d
  `backbone_concept_gemma_3_1b_pt_K512_concept_20260817_141227`
  (`checkpoint-2660`, eval 2026-08-18). Late-half of a K=512 window (true tokens
  256–512) Δperm **0.044** CI [0.039, 0.049]; first-64 Δperm **0.75**; RankMe
  **43.2 / 58.7 / 65.9 / 76.8**; eval_loss **2.365**; free-run `real` greedy @256
  **0.185/0.595** (`real=shuffle`).
- **Materially new vs E17d:** the local computer is halved and banks update **16
  times per 4096 tokens instead of 8**. Same cell, same mix, same no-carry rule.
  This is the remaining untested cause from the E17c/E17d five-whys (local softmax
  already covers CE), not a gate/LR/read-placement retune.

## The architectural bet

E17d already removed the previous-window **token carry** and put concept mix in the
attention residual. Late-page Δperm stayed **0.044** because each forward still
gives every token a **512-token local softmax**. After ~64 tokens of a new window,
FinePDFs CE does not need the banks.

E17e keeps that cell and cuts the local path:

```text
E17d (K=512, eight windows per 4096):
  local  = WindowedTokenAttn(x, W=512)
  global = DedicatedCrossAttn(Q=x, KV=z_g)
  h      = h + local + global
  write z_g after each 512-token window

E17e (K=256, sixteen windows per 4096):
  local  = WindowedTokenAttn(x, W=256)   # THIS is the starve
  global = DedicatedCrossAttn(Q=x, KV=z_g)
  h      = h + local + global
  write z_g after each 256-token window  # twice the write cadence
```

Hub Gemma-3-1B ships `sliding_window=512` on the text config **and** on each
`Gemma3Attention`. Construction today raises unless `concept_block` equals that
value. E17e makes `concept_block` the authority and patches every copy so one mask
serves local layers, global layers, and the write loop.

**Metric naming (do not misread the E17d key).** Intra-block bins are fractions of
`concept_block`. The logged key `delta_permutation_block_256_512` means **the
second half of the current window** (frac 0.5–1.0). At K=256 that is tokens
**128–256 of each 256-token block**, not document offsets 256–512. Compare
late-half to E17d's late-half (0.044), not to E17d's 128–256 bin (0.10).

**Out of scope:** dropping token–token attention at layers 5/11/17/23 (full Bet A;
follow-up if K=256 still leaves a sufficient local computer); K=128; MQAR / needle
train-time (Bet B); Titans surprise-write; VICReg; collapsing to one bank; 1B
until the 300M late-half gate passes; gate-init A/Bs.

## Why this is not a safe retread
E17d listed "shrinking the sliding window below K=512" as out of scope and then
failed with a healthy, unused memory. Literature on SWA hybrids (Infini, NHA w>0)
says extra memory is ignored until the window cannot cover next-token CE. This
run is that starve, at the size the user named (half of 512), on the cell that
already proved geometry can stay healthy. Analogy: a cortex patch whose receptive
field is too large never has to read the compressed notebook.

## Success criteria (set BEFORE running)
Evaluate at 100M (kill), 300M (mechanism verdict), 1B only if 300M passes.
Same immutable held-out split as E17d. Default forward is already carryless.

- **Primary (300M):** late-half of each K=256 window, `CE(permuted all banks) −
  CE(real) ≥ 0.10` nats on `delta_permutation_block_256_512`; 95% CI lower bound
  **> 0.05**. E17d late-half was **0.044**.
- **Depth abstraction (300M):** single-bank permutation has a positive 95% CI for
  at least **3/4** banks on that same late-half bin.
- **Geometry (300M):** within-sample RankMe **≥ 19.2/128 for every bank**.
- **Generation (300M, matched no-carry decode):** `real` greedy @256 distinct-1
  **≥ 0.20** and REP-3 **≤ 0.60**, and `real` beats `shuffle` on at least one of
  those.
- **No broad LM crash:** held-out eval loss **≤ 2.70** (E17d 2.365 with bar 2.50;
  some rise is expected because the local computer is weaker).

## Kill criteria (set BEFORE running)
- **Before GPU training:** unit tests for (a) `concept_block=256` aligns Gemma
  `sliding_window` on config and `Gemma3Attention`, (b) write cadence is 16
  windows per 4096, (c) E17d K=512 still constructs, (d) `launch_e17e.sh` pins
  `--concept_block 256`.
- **Any checkpoint:** non-finite loss/grads, three consecutive eval-loss
  increases, or any bank RankMe **< 10**.
- **100M:** stop if late-half all-bank Δperm **< 0.03** (E17d replay) **or** only
  bank 0 has a positive Δ. That means K=256 still leaves a sufficient local
  computer — next is concept-only global layers or a train-time associative
  objective, not another half-window retune in this spec.
- **300M:** do not launch 1B if the primary late-half gate misses 0.10 or fewer
  than 3 banks participate.

## Plan
- **Data:** immutable `e16b_long_4k_v1` Gemma-tokenized mix; causal LM; seq 4096.
  Pressure/upweight is **off**. Same 4k tree; no re-pretokenize.
- **Compute:** Polonez, 4× RTX 3090, **300M** non-padding tokens. Start from
  E17d's bs=8 accum=2 (effective batch 64); K=256 may change activation memory
  (16 windows, shorter Q). Recalibrate microbatch by real tok/s only if bs=8
  OOM or is underfilled. Not 1B.
- **Launch (after implementation):** wrapper `scripts/launch_e17e.sh` pinning
  E17d knobs plus `CONCEPT_BLOCK=256`, then `launch_e10.sh`.
  ```bash
  EXPERIMENT_ID=E17e CONCEPT_BLOCK=256 CONCEPT_IO_MODE=per_layer_banks \
  CONCEPT_READ_MODE=dedicated CONCEPT_READ_PLACEMENT=attn_residual \
  TIE_CONCEPT_WRITER=false CONCEPT_WRITE_MODE=additive WRITE_GATE_INIT=0.1 \
  MEMORY_CARRY_DROPOUT=1.0 INFERENCE_CARRY_POLICY=drop_after_first \
  MEMORY_PRESSURE_TOKENS=0 MEMORY_PRESSURE_WEIGHT=1.0 \
  READ_CONCEPT_NORM=true READ_GATE_INIT=0.1 \
  OPTIMIZER=muon LEARNING_RATE=0.01 MUON_ADAMW_LR=2e-4 \
  MAX_SEQ_LENGTH=4096 PRETOKENIZE_MIX=e16b_long_4k_v1 \
  TARGET_TOKENS=300000000 SKIP_PRETOKENIZE=1 \
  bash scripts/launch_e10.sh
  ```
- **New foundation code:** align backbone `sliding_window` to `concept_block`
  (config + per-layer `Gemma3Attention`). `launch_e10.sh` must honor a caller
  `CONCEPT_BLOCK` instead of overwriting 512. No new model class.

## Result
<Filled in AFTER, by experiment-track. Link out; do not paste full results here.>
- Run id: `<run_id>`
- WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
