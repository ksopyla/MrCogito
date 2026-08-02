# E17 falsified + write-gate INIT confirmed as the dead-write cause

**Date:** 2026-08-02
**Runs:**
- **E17** — `backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805` (Polonez, `per_layer_banks`, init 0.01). [W&B](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805).
- **Odra write-init falsification** — `backbone_concept_gemma_3_1b_pt_K512_concept_20260801_203613` (Odra, `shared_depth_recurrent`, `WRITE_GATE_INIT=0.3`).
**Related:** [E17 spec](../../experiments_specs/ahead/E17_four_bank_concept_memory.md) · [write-path diagnosis](e16b_write_path_and_topology_diagnosis_20260801.md) · [E16b generation report](e16b_generation_quality_assessment_20260801.md)

---

## TL;DR
**The dead-write cause is the cold-start *initialization* (0.01), not the shared-depth topology.**
E17 (per-layer banks, the structural/topology fix) is **FALSIFIED at 100M** — its write gates stayed
dead. The cheap Odra falsification (higher write init, same shared topology) **held the gates open**.
The diagnosis process worked: the cheap falsification found the cause; the expensive topology bet did
not pay off.

## 1. E17 — per-layer banks did NOT open the write gates (FALSIFIED)
At the 100M-token checkpoint (step 790), `concept_gate_metrics()` on `checkpoint-790`:

| gate | start (init 0.01) | **@100M** |
|---|---|---|
| write_0 (L5)  | 0.010 | **0.0014** |
| write_1 (L11) | 0.010 | **0.0109** |
| write_2 (L17) | 0.010 | **0.0067** |
| write_3 (L23) | 0.010 | **-0.0033** |

All `|tanh(α)| ≤ 0.011` — essentially unchanged from init, and flatter than E16b at the same point.
Read gates opened only weakly (0.06–0.12, increasing with depth) — a *consequence* of empty banks
(dead writes → nothing useful to read). `eval_loss @790 = 2.28`; no free-run diversity eval is wired
into the training loop.

**Verdict:** the "selfish-write" topology hypothesis is falsified. Privatizing the banks did **not**
give the write gate a usable gradient. The run continues toward 1B (no-early-kill policy) but the
mechanism signal — the gate trajectory itself — is the decisive metric here, and it has flatlined;
it will not flip at 1B.

## 2. Odra write-init falsification — higher init HOLDS the gates open (cause confirmed)
Same `shared_depth_recurrent` topology as E16b, only `WRITE_GATE_INIT` 0.01 → 0.3, ~100M tokens
(checkpoint-791):

| gate | start (tanh 0.3 ≈ 0.29) | **@100M** |
|---|---|---|
| write_0 | 0.29 | **0.324** |
| write_1 | 0.29 | **0.195** |
| write_2 | 0.29 | **0.268** |
| write_3 | 0.29 | **0.320** |

The write gates **stayed open** (~0.20–0.32) on the *exact* topology whose gates were dead in E16b.
So the cold-start init (0.01) — not the topology — is what starved the writes (the step-(a) diagnosis,
cause 2a). Higher init gives the write a non-trivial contribution from step 1 → enough gradient to
*hold* the gate open.

## 3. Combined reading
- **Cause 2a (cold-start init) confirmed; cause 2b (shared-depth topology) rejected.** Per-layer
  privatization (E17) and writer tying (E17a) are both downstream of a problem that lives entirely in
  the gate *initialization*.
- **The fix direction is the write-gate init** (≥0.3), now empirically shown to open the gates on the
  E16b topology. The decisive next test is whether that actually recovers *generation*: with live
  writes the banks carry real prefix content instead of a learned constant, so the constant-bias
  free-run attractor should vanish.
- **E17a (untied writers) is now moot** — its activation gate ("E17 gates opened but quality capped")
  is not met (E17's gates never opened). Move to `canceled/`.

## 4. Decisive next experiment (recommended)
**`shared_depth_recurrent` + `WRITE_GATE_INIT=0.3` (or higher), 1B tokens, with free-run diversity
(distinct-1/REP-3 vs base Gemma) measured on the 100M / 1B checkpoints.** This is "E16b with the
correct write-gate init" — the cheapest possible change (one env var) on an architecture whose gates
are now shown to open. Success: free-run `real` distinct-1@256 recovers toward base (~0.78) with
`real ≥ zero`, at gates |tanh(α)| ≳ 0.2.

Open secondary question (only after the init fix recovers generation): does per-layer topology matter
*once gates are open* (E17 + init 0.3)? Topology can only be fairly judged with live writes.

## 5. Process note
The two-run design paid off: the cheap falsification (Odra, ~12 GPU-h) identified the cause and the
fix direction; the expensive topology experiment (E17, ~115 GPU-h) cleanly falsified the alternative.
The negative E17 result is a successful falsification, not a process failure — the hypothesis was bold
and pre-registered.

## 6. Known infra issue (filed)
E17's first launch died in pretokenize on invalid UTF-8 in `finepdfs_100BT` (`_map_resilient` isn't
resilient to parquet string decode errors). Workaround: staged E16b's already-tokenized data
Odra→NAS→Polonez. Proper fix (read text as binary, decode `errors='replace'`) captured as a TODO at
`scripts/pretokenize_mix.py:_map_resilient` (commit `53c1334`).

## 7. Update (2026-08-02): CONFIRMED — init 0.3 recovers generation (100M)
The decisive test ran: the generation-quality assessment on the Odra init-0.3 checkpoint
(`...20260801_203613/checkpoint-791`, ~100M, write gates OPEN 0.20–0.32) vs base Gemma — a clean
single-variable isolation vs E16b (same `shared_depth_recurrent` topology; only the write-gate init
differs).

| condition (greedy @256) | E16b (init 0.01, dead gates) | **init-0.3 (gates open)** | base |
|---|---|---|---|
| `real` distinct-1 | ~0.04 | **0.29** | 0.19 |
| `real` REP-3 | ~0.94 | **0.42** | 0.67 |
| `real` sample distinct-1 | — | **0.45** | 0.49 |

- **`real ≥ zero`** (0.29 vs 0.28) — the E16b inversion (concepts-on *worse* than off) is gone.
- Text stays in coherent prose ("…Alice… set off on her adventure…") instead of E16b's `1. 2. 3.` collapse.
- Context sweep (128–2048 prompt tokens): in base's ballpark, no catastrophic collapse.

**Verdict: the write-gate init is the fix, confirmed by single-variable isolation.** Same topology as
E16b, only init differs → gates open → generation recovers to ≈ base. The dead-write-gate cause of
E16b's repetition is corroborated end-to-end (cause → mechanism → symptom → fix). Next: scale the
init-0.3 config to 1B and adopt free-run diversity as a standing training gate. Artifacts:
`Cache/Evaluation_reports/e16bfg_ckpt791_generation.json`,
`Cache/logs/eval_falsif_ckpt791_20260802_074204.log`.
