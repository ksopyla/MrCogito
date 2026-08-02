# E17 — Four-bank per-global-layer concept memory (the write-path structural fix)

- **Status:** active — running on Polonez (100M done, continuing to 1B). **Topology NOT
  falsified:** the 100M init-0.01 run is confounded by the cold-start gate init (gates stay dead
  on *both* topologies at init 0.01); the fair test is per-layer + init 0.3 (next).
- **Serves:** the E16b platform's open failure mode — free-run generation degeneration caused
  by a dead concept write path. Restores effective concept bandwidth (BAPO channel `a`) by
  privatizing the memory so each write gets a clean "self" gradient.
- **Implementation plan:** [E17_four_bank_concept_memory_plan.md](E17_four_bank_concept_memory_plan.md) *(authored by `implementation-plan` after approval; the HOW)*
- **Owner / dates:** Krzysztof Sopyła · opened 2026-08-01 · closed —

> One experiment = one changed variable vs E16b: the **concept-state topology**. Everything
> else — backbone, LoRA, K, C, objective, data, optimizer, gate init, parameter count — is
> identical to E16b. This is **not** E13 (per-layer memory at all 26 layers via a new
> KV-prefix read interface, gated on E12); it is the cheaper 4-bank intermediary that E13's
> own "Explicitly not E13" section says *"needs its own spec."*

## Hypothesis
If each of Gemma's four global layers (5/11/17/23) reads and writes its **own** 128-slot
concept bank — with the writer still **tied** across the four (identical parameter count to
E16b) — then the per-layer write gates, which stayed dead (|tanh(α)|≈0.05, no trend) through
**1B tokens** of E16b, will open to |tanh(α)|≥0.1 by **50M tokens**, and free-run greedy
distinct-1@256 will recover to ≥0.5 (REP-3≤0.25) with `real`≥`zero` — because privatizing the
banks makes each write **"selfish"** (a layer reads the bank it wrote last block), giving the
write gate a clean gradient the shared topology denies it.

## Builds-on
- **Foundation:** `nn/backbone_concept_lm.py` `BackboneConceptLM` (E16b path); the single
  shared entrypoint `training/train_concept_pretraining.py` → `scripts/train_concept_pretraining_multigpu.sh`
  → `scripts/launch_e10.sh`. New capability lands as a reusable config value
  `concept_io_mode="per_layer_banks"` — **not a fork**.
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42, read/write
  gate init 0.01 — **the same init recipe as E16b** (not a resume). Keeping `WRITE_GATE_INIT=0.01`
  is deliberate: it isolates topology as the single variable. (The higher-init pre-check is a
  *separate* experiment on the shared topology, not this one.)
- **Baseline to beat:** E16b `backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850`
  (`checkpoint-7900`). Its write gates finished at ±0.05 (dead) at 1B tokens; free-run `real`
  greedy distinct-1@256 = **0.04** / REP-3 = **0.94** (vs base Gemma 0.78 / 0.00), and
  `real` was *worse* than `zero` on free-run diversity.
- **Materially new vs E16b:** state privatization — four private `[B,128,H]` banks instead of
  one shared `[B,128,H]`, with **identical machinery** (same tied `ConceptWriteHead`, same
  per-layer read/write gates); only the learned bank initializations scale 4× (`concept_init
  [4,128,H]` vs `[128,H]`, ≈+0.08% of trainable params — necessary and negligible). The read
  dispatches each global layer to its own bank; within a block each layer reads the bank it
  wrote last block (selfish), not a shared accumulator written by all shallower depths. This is
  a **structural topology change**, not a retune.
- **Materially new vs E13:** only 4 banks (at the 4 full-attention layers), reusing E16b's
  cross-attention read + tied BiXT write — no new KV-prefix interface (E13's E12 prerequisite).

## The architectural bet
Replace E16b's shared state and update schedule:

```text
E16b (shared_depth_recurrent):
  one z_b [B,128,H]; layer 5 reads z_b → tied write → z_b^(5);
  layer 11 reads z_b^(5) → tied write → z_b^(11); … ; z_{b+1} = z_b^(23)
  (a write at depth d only ever helps depths > d and future blocks → altruistic → starved)

E17 (per_layer_banks):
  four banks z5_b, z11_b, z17_b, z23_b, each [B,128,H], each init from its own learned concept_init_ℓ
  layer ℓ reads zℓ_b → tied write (same ConceptWriteHead weights for all four) → zℓ_{b+1}
  (each layer reads the bank it wrote last block → selfish → clean write gradient)
```

The tied writer + per-layer gates keep the **machinery identical to E16b**; only the learned
bank initializations grow 4× (≈+0.08% of trainable params). The state routing changes from a
shared accumulator to four private banks. Reads stay at the same four layers using each
layer's native Q/K/V/O (LoRA-adapted), unchanged. Raw causal-LM objective, data, K=512,
C=128, optimizer (Muon), and gate init are all held fixed vs E16b.

**Out of scope (follow-ups after a positive signal):** untying the writer (4 separate writers
— a capacity confound, E13 territory); the KV-prefix read interface (E12); varying the number
of banks; BAPO-hard reasoning probes (that is the E08 decomposition axis, not this fix).

## Why this is not a safe retread
This is not a knob tweak and not E13. It is a **gradient-flow / information-topology** change
motivated by a specific failure diagnosis ([write-path report](../../2_Experiments_Registry/run_reports/e16b_write_path_and_topology_diagnosis_20260801.md)):
E16b's writes are dead because the shared topology makes them altruistic (a write only helps
later layers) and forces four depth-specialized features through one accumulator. **Analogy: a
standard transformer already gives every layer its OWN private KV cache** — E16b's shared bank
violates that natural per-layer-private-memory structure; E17 restores it in compressed,
recurrent form at exactly the four layers that need cross-window memory. Framed in BAPO terms,
the bet is on **effective** bandwidth (open the write valve), explicitly NOT on nominal
capacity (more slots) — BAPO Thm 10 says the latter does not help, and E17's metrics are
designed to tell those apart.

## Success criteria (set BEFORE running)
All evaluated at the **100M-token report checkpoint** and re-confirmed at the full
**1B-token run** (matching E16b's budget — the E16→E16b trajectory shows short budgets
misjudge this architecture, so the verdict is the 1B number, with 100M as the intermediate
signal):

- **Mechanism — the load-bearing discriminator (did the writes open?):** all **4** per-layer
  write gates have `|tanh(α)| ≥ 0.1` (E16b was ≈0.05 at 1B with no trend). Without this, any
  CE/diversity gain is the BAPO-Thm-10 "wrong axis" (capacity) failure mode, not a topology win.
- **Primary — the actual symptom (free-run generation):** E17 `real` **greedy** distinct-1@256
  **≥ 0.5** AND REP-3@256 **≤ 0.25** (base Gemma is 0.78/0.00; E16b was 0.04/0.94), measured by
  the [generation-quality runner](../../analysis/run_e16b_generation_assessment.py) vs the
  matched base-Gemma control. AND **`real` ≥ `zero`** on free-run diversity — the E16b
  inversion (concepts-on worse than concepts-off) must flip.
- **Effective `a` per bank:** per-bank `Δshuffle_beyond ≥ 0.01` and `Δstatic_beyond ≥ 0.01`
  nats at positions ≥1024 (the E16 mechanism gate, computed independently for each of the 4 banks).
- **No regression:** teacher-forced beyond-window CE no more than +0.05 nats above E16b at
  matched exposure; at least 3 of 4 read gates open (`|tanh|≥0.1`).
- **Geometry guard:** within-sample RankMe ≥ 38.4/128 for every bank (report min + median).

## Kill criteria (set BEFORE running)
**Per user direction: do NOT early-stop on write-gate or free-run-diversity grounds.** The
E16 (50M, failed) → E16b (1B, cleared) trajectory shows this architecture needs the full
budget before the mechanism verdict is trustworthy; a flat write-gate curve at 25M is NOT a
kill signal (E16b's gates were flat for most of training). Run the full 1B; the success
metrics above are reported at 100M as an intermediate signal, not a stop condition.

Stop ONLY for genuine divergence / instability:
- Non-finite loss or gradients.
- Held-out eval CE rising at three consecutive evaluations.
- Any bank's within-sample RankMe < 19.2/128 (geometric collapse).

## Plan
- **Data:** `e16b_long_4k_v1` (Gemma-tokenized, 4K) — the same immutable manifest as E16b.
  Raw causal-LM objective, sequence length 4096.
- **Compute:** **Polonez, 4× RTX 3090.** ≈ **115 GPU-h** (≈ E16b's 1B), ≈ **~30 h wall** on 4
  GPUs (E16b was ~38 h on 3). Per-bank concept state is tiny vs token activations, so per-step
  memory/compute are ≈ E16b; calibrate microbatch first (4 private banks add a small footprint).
- **Steps / epochs:** **1B non-padding tokens** (matching E16b), warmup 500, effective batch
  **72** (per-device 3 × accum 6 × 4 GPUs — held equal to E16b's 72 for a clean A/B). Mandatory
  **report checkpoint at 100M** (save + run the generation-quality eval + gate telemetry there),
  but **do not stop** — continue to 1B.
- **Launch (Polonez, all 4 GPUs):**
  ```bash
  EXPERIMENT_ID=E17 CONCEPT_IO_MODE=per_layer_banks \
  READ_CONCEPT_NORM=true READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 \
  OPTIMIZER=muon LEARNING_RATE=0.01 MUON_ADAMW_LR=2e-4 MUON_MOMENTUM=0.95 \
  WEIGHT_DECAY=0.1 CONCEPT_MEMORY_LR= \
  MAX_SEQ_LENGTH=4096 PRETOKENIZE_MIX=e16b_long_4k_v1 \
  TARGET_TOKENS=1000000000 WARMUP_STEPS=500 AUTO_INTERVALS=1 \
  SAVE_TOTAL_LIMIT=12 SKIP_PRETOKENIZE=1 \
  PER_DEVICE_BATCH_SIZE=3 GRADIENT_ACCUMULATION_STEPS=6 \
  bash scripts/launch_e10.sh
  ```
  (`CONCEPT_MEMORY_LR` stays empty — Muon routes the gate scalars to AdamW @ 2e-4, same as E16b;
  the diagnosis showed the dead writes are not an LR problem.)
- **Parallel diagnostic (Odra, 3× RTX 3090):** the cheap write-init falsification — a fresh
  *shared* `shared_depth_recurrent` run with `WRITE_GATE_INIT=0.3` (read init unchanged) for
  ~100M tokens, to separate the cold-start-init cause from the topology cause. Uses existing
  code; runs concurrently with E17 implementation.
- **New foundation code (reusable, not a fork):** add `concept_io_mode="per_layer_banks"` to
  `nn/backbone_concept_lm.py` — a per-layer state holder (4 banks + 4 learned `concept_init_ℓ`),
  read dispatch routing each `GlobalLayerWithConceptRead` to its own bank, the existing tied
  `ConceptWriteHead` applied per-layer to its bank, per-layer read/write gates, checkpoint
  round-tripping, and per-bank `real/zero/shuffle/static/frozen` ablation + gate telemetry.
  Implement via `research-implement` after the plan; extend the shared args/config/launcher
  plumbing and tests; **no new training script**.

## Result (100M checkpoint — running; topology NOT falsified)
**E17 is the right topology to test, not a failure.** At the 100M checkpoint (step 790) the per-bank
write gates stayed dead (`write_0..3 = 0.0014 / 0.0109 / 0.0067 / -0.0033`, `|tanh(α)| ≤ 0.011`,
init 0.01) — but this is the **cold-start init confound**, not a topology verdict: the Odra
write-init falsification shows init 0.01 deadens the gates on *both* topologies (init 0.3 holds them
open at 0.20–0.32 on the *same* shared topology). With the gates inert, E17's per-layer write
structure never engaged, so the topology's effect on concept quality / generation is **unmeasured**.
E17 is a progression of E16b (per-layer = the transformer's natural per-layer-private-memory
structure; the E13 direction) and remains the bet to pursue — to be evaluated **with open gates
(init 0.3)**: per-layer + init 0.3 vs shared + init 0.3 is the decisive test. The init-0.01 run
continues to 1B on Polonez (a record under the cold-start confound). Separately, init 0.3 opens the
gates and free-run diversity recovers at 100M — encouraging, **not** success (needs proper
generation-quality assessment at scale). E17a (untied writers) remains a valid future variant. See
[report](../../2_Experiments_Registry/run_reports/e17_falsified_init_is_the_cause_20260802.md).
- Run id: `backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805` (Polonez, continuing to 1B)
- WandB: https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805
- Run report: `docs/2_Experiments_Registry/run_reports/e17_falsified_init_is_the_cause_20260802.md`
- Verdict: **topology not falsified — confounded by cold-start init at 100M; the fair (open-gate)
  topology test is next.**

## References
- Diagnosis backing: [E16b write-path & topology diagnosis](../../2_Experiments_Registry/run_reports/e16b_write_path_and_topology_diagnosis_20260801.md)
- Decoding failure + Layer-0 probe: [E16b generation quality report](../../2_Experiments_Registry/run_reports/e16b_generation_quality_assessment_20260801.md)
- E16b (baseline): [spec](../done_success/E16b_longctx_muon_1b.md) · [mechanism report](../../2_Experiments_Registry/run_reports/e16b_longctx_muon_1b_20260725.md)
- E13 (full 26-layer per-layer memory, distinct bet): [spec](E13_layerwise_recurrent_kv_memory.md)
- BAPO / information-flow lens: [literature review](../../literature_review/reasoning_bandwidth_information_flow.md)
