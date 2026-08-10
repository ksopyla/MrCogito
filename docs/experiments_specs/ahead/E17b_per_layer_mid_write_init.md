# E17b — Per-layer banks with mid write-gate init (0.1)

- **Status:** draft — awaiting approval to launch on Polonez
- **Serves:** free-run recovery on the E16b long-context platform, now that E17
  (init 0.01) showed a **partial** free-run lift with dead writes, and shared init 0.3
  showed that opening writes alone does **not** fix free-run on shared topology.
- **Implementation plan:** [E17b_per_layer_mid_write_init_plan.md](E17b_per_layer_mid_write_init_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-08-10 · closed —

> One experiment = one changed variable vs E17: **`WRITE_GATE_INIT` 0.01 → 0.1**.
> Topology stays `per_layer_banks`. Everything else matched to E17 / E16b.

## Hypothesis
If E17's four per-layer concept banks are trained with `WRITE_GATE_INIT=0.1`
(tanh≈0.1 — the registered open-gate floor, midway in log-space between the dead
cold-start 0.01 and the aggressive shared-control 0.3), then by **100M tokens** all
four write gates stay at `|tanh(α)| ≥ 0.1`, and by **1B tokens** free-run `real`
greedy distinct-1@256 reaches **≥ 0.5** with REP-3@256 **≤ 0.25** and `real ≥ zero`
— because the banks get a live write path (unlike E17 init 0.01) without the shared
topology's free-run poison that accompanied init 0.3 on `shared_depth_recurrent`.

## Builds-on
- **Foundation:** `nn/backbone_concept_lm.py` `concept_io_mode="per_layer_banks"` (E17);
  `scripts/launch_e17b.sh` → `launch_e10.sh` → shared trainer. **Config only — no new fork.**
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42;
  `READ_GATE_INIT=0.01` unchanged; **only** `WRITE_GATE_INIT=0.1`.
- **Baseline to beat:**
  - E17 init 0.01 (`…20260807_195730`): free-run `real`@256 **0.21/0.59**, writes dead,
    Δbeyond ~0.004 — partial gen lift, no mechanism.
  - Shared init 0.3 (`…20260807_090248`): Δshuf_beyond **1.69**, free-run `real`@256
    **0.06/0.90** — mechanism open, free-run still broken.
  - Absolute bar: same as E17 (d1≥0.5, REP-3≤0.25, `real≥zero`).
- **Materially new vs E17:** write-gate cold-start set to the open-gate threshold (0.1),
  not another topology change. **Materially new vs shared init-0.3:** same open-ish init
  on *per-layer* banks (the structure E17 already showed is freer-run-friendlier with
  dead writes).

## The architectural bet
Keep E17's selfish per-layer banks; change only the write-gate prior so training does
not start with the valve glued shut. 0.1 is chosen as the **middle**:

| init_w | tanh(init) | Role in ledger |
|---|---|---|
| 0.01 | ≈0.01 | E16b / E17 — cold-start; writes stay dead |
| **0.1** | **≈0.10** | **This run — open-gate floor / geometric mid** |
| 0.3 | ≈0.29 | Shared control — opens mechanism, free-run still fails |

Analogy: E17 privatized the memory channels; E17b opens the write valves just enough
to use them. Shared+0.3 already showed that slamming the valves open on a shared
accumulator restores teacher-forced ΔCE without curing free-run — so 0.3 is the wrong
default for the per-layer fair test.

## Why this is not a safe retread
The user explicitly requested a higher / mid write init after E17's partial free-run
win and the shared-0.3 free-run failure. This is the missing cell of the 2×2
(topology × init) that the 100M sniff and E17 1B jointly define — not an optimizer
knob hunt.

## Success criteria (set BEFORE running)
Evaluate at **100M** (report checkpoint) and re-confirm at **1B**:

- **Mechanism:** all 4 write gates `|tanh(α)| ≥ 0.1` at 100M and at 1B (no depth drifts
  to ~0 the way shared write_1 did under init 0.3).
- **Primary — free-run:** `real` greedy distinct-1@256 **≥ 0.5** AND REP-3@256 **≤ 0.25**,
  with **`real ≥ zero`**, via `analysis/run_e16b_generation_assessment.py` vs base Gemma.
- **Effective `a`:** per-bank (or aggregate) Δshuffle_beyond **≥ 0.01** and Δstatic_beyond
  **≥ 0.01** at positions ≥1024.
- **No free-run regression vs E17:** `real`@256 distinct-1 **≥ 0.21** even if the absolute
  bar is missed (must not collapse back to E16b/shared-0.3 ~0.05).
- **Geometry:** within-sample RankMe ≥ 38.4/128.

## Kill criteria (set BEFORE running)
Same stability kills as E17 (non-finite loss/grads; eval CE rising three evals; RankMe
< 19.2). **Do not early-stop on free-run at 100M alone** — shared-0.3 taught that 100M
gen sniffs lie — but **do** require writes still open at 100M; if all four `|tanh| < 0.05`
at the 100M report checkpoint, stop (cold-start confound repeating; raise init rather
than burn 1B).

## Plan
- **Data:** `e16b_long_4k_v1` (immutable, same as E16b/E17).
- **Compute:** Polonez 4×3090; ≈ E17 wall (~58 h / ~230 GPU-h at 1B).
- **Steps:** 1B tokens, warmup 500, eff batch 72, report @100M + full Tier-1.5 there
  and at 1B.
- **Launch:**
  ```bash
  bash scripts/launch_e17b.sh
  # equivalent:
  EXPERIMENT_ID=E17b WRITE_GATE_INIT=0.1 bash scripts/launch_e17.sh
  ```
- **New foundation code:** none — config only.

## Result
- Run id: —
- WandB: —
- Run report: —
- Verdict: —

## References
- E17 partial success: [spec](../done_success/E17_four_bank_concept_memory.md) ·
  [1B gen](../../2_Experiments_Registry/run_reports/e17_lowinit_1b_generation_20260810.md)
- Shared init-0.3 control: [report](../../2_Experiments_Registry/run_reports/e16b_shared_init030_1b_20260810.md)
- E16b free-run failure: [report](../../2_Experiments_Registry/run_reports/e16b_generation_quality_assessment_20260801.md)
