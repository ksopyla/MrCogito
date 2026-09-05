# E17 (per-layer topology) + the write-gate-init finding

**Date:** 2026-08-02
*(Reframe 2026-08-02: an earlier draft of this report headlined E17 as "FALSIFIED." That framing
was wrong — it conflated "the selfish gradient opens cold gates" (not supported) with "per-layer
topology is the right structure" (still the right bet, untested fairly). E17 is **not** falsified;
see below. The data below is unchanged; only the interpretation is corrected. Filename retained for
link stability.)*

**Runs:**
- **E17** — `backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805` (Polonez, `per_layer_banks`, init 0.01). [W&B](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805).
- **Odra write-init falsification** — `backbone_concept_gemma_3_1b_pt_K512_concept_20260801_203613` (Odra, `shared_depth_recurrent`, `WRITE_GATE_INIT=0.3`).
**Related:** [E17 spec](../../experiments_specs/done_failed/E17_four_bank_concept_memory.md) · [1B gen report](e17_lowinit_1b_generation_20260810.md) · [write-path diagnosis](e16b_write_path_and_topology_diagnosis_20260801.md) · [E16b generation report](e16b_generation_quality_assessment_20260801.md)

---

## TL;DR (corrected)
- **E17 is the right topology to test — a progression of E16b, not a failure.** Per-layer concept
  banks restore the transformer's natural per-layer-private-memory structure (each layer its own
  memory, like KV caches; the E13 direction; respects depth specialization). It is the
  well-motivated next step and remains the bet to pursue.
- **E17's 100M run (init 0.01) is confounded, not a verdict.** The write gates stayed dead at
  init 0.01 — but the Odra falsification shows init 0.01 deadens the gates on **both** topologies
  (it is the cold-start *init*). So E17's per-layer topology was **never fairly tested**: its write
  structure was inert. E17 is **not falsified**; it must be evaluated with open gates.
- **init 0.3 opens the gates and free-run diversity recovers at 100M — encouraging, NOT success.**
  A single distinct-1 snapshot at 100M is not a generation-quality verdict; a proper assessment at
  scale is required.
- **The decisive topology test:** per-layer + init 0.3 (E17 with open gates) vs shared + init 0.3 —
  does per-layer topology improve generation quality / effective-`a` once the gates are live? This
  is the real E17 question, now well-defined and testable.

## 1. E17 @100M (init 0.01) — write gates dead (cold-start confound, not a topology verdict)
`concept_gate_metrics()` on `checkpoint-790`:

| gate | start (init 0.01) | @100M |
|---|---|---|
| write_0 (L5)  | 0.010 | 0.0014 |
| write_1 (L11) | 0.010 | 0.0109 |
| write_2 (L17) | 0.010 | 0.0067 |
| write_3 (L23) | 0.010 | -0.0033 |

All `|tanh(α)| ≤ 0.011` — unchanged from init. **But this is the cold-start init confound (see §2),
not evidence against the per-layer topology.** With the gates inert, the per-layer write structure
never engaged, so the topology's effect on concept quality / generation is **unmeasured** by this
run. Read gates opened only weakly (0.06–0.12) — a consequence of empty banks. `eval_loss @790 = 2.28`.

## 2. Odra write-init falsification @100M (shared topology, init 0.3) — gates HELD OPEN
Same `shared_depth_recurrent` topology as E16b, only `WRITE_GATE_INIT` 0.01 → 0.3, ~100M tokens
(checkpoint-791):

| gate | start (tanh 0.3 ≈ 0.29) | @100M |
|---|---|---|
| write_0 | 0.29 | 0.324 |
| write_1 | 0.29 | 0.195 |
| write_2 | 0.29 | 0.268 |
| write_3 | 0.29 | 0.320 |

The gates **stayed open (0.20–0.32)** on the *same* topology whose gates were dead in E16b. So the
cold-start **initialization** (0.01) — not the topology — is what deadens the gates. This means
E17's init-0.01 run (§1) was testing the topology with the gates switched off; it cannot adjudicate
the topology question.

## 3. init-0.3 free-run diversity @100M — encouraging, NOT success
Generation assessment on the §2 checkpoint vs base Gemma (single variable vs E16b: same shared
topology, only init differs):

| metric (greedy @256) | E16b (init 0.01, dead gates) | init-0.3 (gates open) | base |
|---|---|---|---|
| `real` distinct-1 | ~0.04 | 0.29 | 0.19 |
| `real` REP-3 | ~0.94 | 0.42 | 0.67 |
| `real` sample distinct-1 | — | 0.45 | 0.49 |

`real ≥ zero` (0.29 vs 0.28) and the text stays in prose instead of E16b's `1. 2. 3.` collapse.
**This is an encouraging single-variable signal that opening the gates matters — but it is a 100M
distinct-1 snapshot, not a generation-quality verdict.** A proper assessment (more prompts, longer
generation, qualitative coherence, at 1B scale, vs base at matched training) is needed before
claiming the generation problem is solved.

## 4. Why E17 (per-layer topology) is still the right bet
- **Natural transformer structure:** every layer keeps its own KV cache (private memory). E16b's
  single shared bank violates that; E17 restores it (the E13 framing).
- **Depth specialization:** low vs high layers carry different information; per-layer banks respect
  that (the original intuition that motivated E17).
- **BAPO:** per-layer banks could carry more *effective* bandwidth (each depth its own channel) — a
  meaningful question once the gates are open (Thm 10 cautions about *nominal* capacity, which is
  exactly why the open-gate test must measure effective-`a`, not assume it).
- The decisive comparison — per-layer + init 0.3 vs shared + init 0.3 — is now well-defined.

## 5. Next steps
1. **Proper generation-quality assessment** of the init-0.3 config at scale (resume to 1B, then
   assess — more prompts, longer, qualitative, vs base at matched tokens).
2. **The fair topology test:** E17 + `WRITE_GATE_INIT=0.3` (per-layer, open gates) vs shared + init
   0.3 — the real E17 comparison (per-layer topology's quality contribution with live writes).
3. E17 (init 0.01) continues to 1B on Polonez per the no-early-kill policy (a record of the
   per-layer topology under the cold-start confound); the topology verdict awaits the open-gate test.

## 6. Process note
The two-run design worked: the cheap Odra falsification isolated the gate-init cause. The earlier
"E17 falsified" framing over-reached — it treated "the selfish gradient does not overcome a 0.01
cold start" (true) as "the per-layer topology is wrong" (not established). E17's topology can only
be judged once the gates are open; that test is the next step.

## 7. Known infra issue (filed)
E17's first launch died in pretokenize on invalid UTF-8 in `finepdfs_100BT` (`_map_resilient` isn't
resilient to parquet string decode errors). Workaround: staged E16b's already-tokenized data
Odra→NAS→Polonez. Proper fix captured as a TODO at `scripts/pretokenize_mix.py:_map_resilient`
(commit `53c1334`).
