# E24 tiny BAPO ladder — E18 vs dense vs encoder-decoder

**Date:** 2026-09-13
**Machine:** Cursor Cloud VM (CPU) for tiny / S0 hunts; Odra 3× RTX 3090 for medium (launched after this write-up)
**Run ID:** `bapo_tiny_packed` + S0 hunts (`recall_single`, `select_1decoy`, `chain_shuffled`, encdec PE rerun)
**WandB:** n/a (CPU probe; no `compute/*` audit — there is no W&B run)
**Raw log:** `/opt/cursor/artifacts/bapo_tiny_packed.log`, `/opt/cursor/artifacts/bapo_s0.log`
**Best checkpoint:** none (on-the-fly rows; models discarded after each rung)
**Git commit:** `b78e780` (probe that produced these numbers)
**Git tag:** —
**Related TODO:** E24 observation base for later gating / selective-read / compression

---

## Goal

Measure what the E18 architecture (one global causal read + SWA stack) can and cannot do on a
DNA-alphabet BAPO ladder, against a matched dense decoder-only and a suffix-only encoder-decoder,
with a **75% dense solvability control** and closed-form prizes (bits, information flow, bytes/token).
Tiny (seq=128, ~0.59M params) is the solvability proof. Medium (4k) / large are the scale-up.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` (`dense` / `perceiver`) + probe-only `EncoderDecoderLM` |
| Width / depth | H=128, pre=1, global=1, stack=2 (~**0.595M**); encdec 2+2 (~**0.79M**) |
| Value embeddings | layers 0 and 1 (VE on E18's global read) |
| Dataset | on-the-fly `data/symbolic_tasks.py` via `data/bapo_ladder.py`, A=4 |
| Objective | teacher-forced CE on the packed answer span only |
| Steps | advertised 800, K1 = 4× (dense up to 3200); S0 hunts to 4000 |
| Batch / LR | 32, AdamW 3e-3, 50-step warmup then constant (not OneCycle) |
| Compute | CPU-hours, no GPU, no W&B `compute/*` |

Packed answers (the Arm-A lesson): `far_copy` 32 tok / **64 bits**; `recall`/`select` 16 tok / **32 bits**;
`chain*` key_len=13 / **26 bits**. `e18_local` (`global_layers=0`) is the leak control.

## Training Outcome

Four architectures on the packed tiny core, then solvability hunts that shrink the contrast set
until dense hits 75% or the rung is declared ill-posed.

| recipe | prize | dense | E18 | encdec | e18_local | calibrated? |
|---|---|---|---|---|---|---|
| `far_copy` | 64 bits | **99.4%** @2400 · flow **0.989** · 63.3 bits | **99.2%** @1400 · flow **0.985** · 63.0 bits | 24.8% (PE rerun) | 24.7% | **yes** |
| `recall_single` (`n_distractors=0`) | 32 bits | **99.2%** @2850 · flow **0.980** · 31.4 bits | **25.3%** · flow **0** | 25.3% | 24.1% | **yes** |
| `select_1decoy` | 32 bits | **99.2%** @1900 · flow **0.978** · 31.3 bits | **87.0%** · flow **0.817** · 26.1 bits | 25.3% | 25.2% | **yes** |
| `chain_ordered` | 26 bits | **93.1%** @3200 · flow **0.896** · 23.3 bits | **97.5%** · flow **0.945** · 24.6 bits | 27.6% | 26.0% | **yes** |
| `chain_shuffled` (`n_distractors=0`) | 26 bits | 33.5% @4000 · flow 0.059 | skipped | — | — | **no** |

Default 2–3-item MATCH2 (`recall` / `select` without the single-fact shrink) stayed at dense **30–39%**
even at H=256 / 4000 steps. Those rungs are K1 (ill-posed at this size), not an E18 failure.

## Concept Health

Not a language-model run. Geometry / RankMe / STS-B are not in scope.

Information-theoretic scores (`evaluation/bapo_metrics.py`):

- **Nominal cache `a`** (bf16 KV bytes / token): dense 512, E18 128 (4× smaller), encdec 256, e18_local 0.
- **Effective `a`** is recovered bits, not the cache size. On `recall_single` E18's cache is the same
  128 B/tok as on `far_copy`, but recovered bits go from 63 → **0**.
- **Bytes / input token** (recovered): `far_copy` 0.062 for both dense and E18 (ceiling 0.0625);
  `recall_single` 0.031 dense vs **~0** E18.

## Evaluation

Plots (learning curves, heatmap, information flow, recovered bits, bytes/token) live in the
PR artifacts under `bapo_tiny_capability/`. CSV: `capability_table.csv`.

### Gates vs this tiny evidence

| gate | result |
|---|---|
| **S0** instrument | `far_copy`, `recall_single`, `select_1decoy`, `chain_ordered` have dense ≥75%. Default MATCH2 and shuffled `chain` do **not**. |
| **S1** positional | **PASS.** E18 matches dense on packed `far_copy` (flow 0.985 vs 0.989) and learns *faster* (1400 vs 2400). |
| **S2** content | **PASS on keyed recall.** Dense 99% / 31 bits; E18 recovers **0 bits** for 4000 steps. Matches E18b (4.49% vs dense 99.33%) with a closed-form prize. `select_1decoy` is a `keymark` vs `decoy` type cue, not MATCH2 — E18 87% there does not contradict S2. |
| **S3** composition | **Half.** Ordered DFA is easy (E18 97.5% ≥ dense 93%). Shuffled chain is **not dense-solvable** at 0.59M / 4000 (33.5%) — do not score the ordered-vs-shuffled gap. |
| **S4** plots | **PASS** (this report). |
| **K1** | Default MATCH2 and shuffled chain are ill-posed at this size. Stopped; did not score E18. |
| **K2** | `e18_local` stayed at chance on every retrieval rung. Tasks do not leak. |
| **K3** | Not triggered: E18 does **not** match dense on `recall_single`. |

## Interpretation

E18 at ~0.6M is a **positional INDEX machine with a content-addressing wall**. The same global
read that copies 64 bits of a far span (and did 32k copy at 125M) cannot retrieve a 32-bit value
by its key, even when there is only one planted fact and the dense control is at 99%. That is
the E18b diagnosis with a known prize and an exact floor.

`select_1decoy` is easier than MATCH2 because the marker *type* is unique (`keymark` vs `decoy`).
E18 can use that cue (87%, 26 of 32 bits). Do not read it as "E18 can content-address."

The 2+2 H=128 encoder-decoder is **not** a competitive matched baseline in this budget: after
sinusoidal PE removed a bag-of-tokens leak, copy sat at chance. Load-bearing comparison is
**dense vs E18**. A fairer encoder-decoder would need RoPE, more depth, or the same SWA+read
recipe with a bidirectional prefix — out of scope for this probe.

Shuffled REACHABILITY (`A→B` edges in random order) is a dense wall at this size, not an E18
kill. Ordered DFA is easy for both, and slightly easier for E18 — consistent with "local
transition + one global read beats full attention on a sequential scan."

## Decision

- Keep E24 in `ahead/`. Tiny is the observation base; it does not close the ID.
- Score E18 only on `CALIBRATED_RECIPES` (`--recipe far_copy recall_single select_1decoy chain_ordered`).
- First GPU rung: the same four recipes at `--scale medium` (seq=4096), H=256, still <100M.
  Do not launch default MATCH2 or shuffled `chain` at 4k until dense hits 75% at that width.
- Next architecture work (gating, selective read, better compression) should target the
  **recall_single** wall: the read's keys are not content-discriminative. `far_copy` is the
  control that the channel exists.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E24_e18_bapo_capability_ladder.md`, `docs/engineering_specs/bapo_capability_ladder.md`, `agenda.md`*
