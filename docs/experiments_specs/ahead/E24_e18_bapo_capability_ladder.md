# E24 — E18 capability limits on a BAPO DNA ladder

- **Status:** in progress (tiny DNA measured; GPU bridge 512/1024 measured; 4k S0 not closed). **2026-09-13:** a second
  family (Glyph: typed vocab 16/32, structured noise) is specified and implemented as
  config-selectable generators; it does **not** change this experiment's DNA hypothesis.
  See [`glyph_capability_ladder.md`](../../4_Research_Notes/glyph_capability_ladder.md).
- **Serves:** the Vision's "does the long-range channel carry *addressable* content, and how
  many bits?" question, as the observation base for later gating / selective-read / compression
  work. Uses the DNA-alphabet framework that made E22/E23's channel measurable, scaled onto
  **E18** vs matched dense transformers.
- **Implementation plan:** [E24_e18_bapo_capability_ladder_plan.md](E24_e18_bapo_capability_ladder_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-13
- **Engineering foundation:** [bapo_capability_ladder.md](../../engineering_specs/bapo_capability_ladder.md)

> One experiment = one coherent claim about *what E18 can and cannot do* on a calibrated
> synthetic ladder. Not a new architecture. A/B of width/LR is out of scope.

## Hypothesis
If E18 is a bounded-attention prefix oracle whose only unbounded layer is **one** full-causal
read, then on the DNA BAPO ladder, at matched params <100M: (a) a dense decoder-only control
reaches ≥75% on every tiny user-core rung (the tasks are solvable); (b) E18 matches dense on
positional `far_copy` (INDEX / bandwidth) and lags on content-addressed `recall` / `select`
and on shuffled `chain` (REACHABILITY-hard), while `chain_ordered` (DFA-easy) is strictly
easier for E18 than shuffled `chain`; (c) recovered bits — the measured prefix bandwidth `a`
— stay flat as seq_len grows on hard tasks for E18, whereas dense keeps recovering until the
task's BAPO-hard wall; (d) a symmetric encoder-decoder (`b=0` on the suffix, full prefix `a`)
sits between them on content tasks. **Because** E18's ledger already showed positional copy
99.9998% @32k and keyed-recall 4.49% vs dense 99.33%, and BAPO Thm 10 says raising depth at a
fixed bottleneck does not raise effective `a`.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (`par_mode=perceiver|dense`), new reusable
  `nn/encdec_lm.py`, `data/symbolic_tasks.py` + `data/bapo_ladder.py`,
  `evaluation/bapo_metrics.py`, `verification/bapo_capability_probe.py`. Shared causal-LM
  entrypoint unchanged.
- **Init / checkpoint:** random init, <100M. No pretrained warm-start.
- **Baseline to beat:** E18b dense keyed-recall **99.33%** (task learnable) vs E18 **4.49%**;
  E18 positional copy **99.9998%** @32k; DNA far_copy Arm D **100%** / Arm C **25%** at seq=128
  (`symbolic_long_context_suite.md`).
- **Materially new:** not another natural-text CE run. A calibrated BAPO-hard/easy ladder
  with closed-form prizes, a 75% dense solvability gate, encoder-decoder as a third inductive
  bias, and information-flow plots. The sibling agent's perceiver_concept 100% map is a
  different architecture and is not this experiment.

## The architectural bet
E18's single global read is **sufficient for positional INDEX/copy and insufficient for
content-addressed and multi-hop tasks**, at a constant effective `a`, once each task is
proven solvable by a matched dense decoder-only. The encoder-decoder control tests whether
"no raw suffix→prefix route" (the BAPO `b=0` decoder) is itself enough to force a useful
`a`, or whether E18's failure is the *one-layer* key quality (E18b's diagnosis).

## Why this is not a safe retread
E18/E18b already ran keyed-recall on *text-shaped* rows mixed into an LM mix. This ladder
fixes the three things that made those numbers hard to read: unknown prize, no exact floor,
no BAPO-hard composition tasks, no encoder-decoder control, no bits/token. It is a
measurement instrument with a falsifiable capability claim, not a LR sweep.

## Success criteria (set BEFORE running)
- **S0 (instrument):** every tiny user-core rung (`far_copy`, `recall`, `select`,
  `chain_ordered`, `chain`) has dense accuracy ≥ 75% at the stated tiny budget. If not, fix
  the generator / pack the loss / add steps — do not score E18.
- **S1 (positional):** E18 `far_copy` information_flow ≥ 0.75 × dense on tiny (and, when
  GPU, on medium). Consistent with E18's 32k copy result.
- **S2 (content):** E18 `recall` and `select` information_flow ≤ 0.5 × dense on tiny. Consistent
  with E18b's 4.49% vs 99%.
- **S3 (composition):** E18 `chain_ordered` > E18 `chain` by ≥ 0.20 accuracy, and shuffled
  `chain` is where E18 vs dense gap is largest.
- **S4 (plots):** learning-curve, heatmap, information-flow, recovered-bits, and
  bytes/token plots exist for the tiny run and are checked into artifacts / the run report.

## Kill criteria (set BEFORE running)
- **K1:** dense < 75% on any tiny user-core rung after 4× the default step budget and packed
  `span_len` — the rung is ill-posed; stop and fix the generator.
- **K2:** `e18_local` > chance+0.15 on a retrieval rung — the task leaks; stop.
- **K3:** E18 matches dense (±5 acc points) on `recall`, `select`, *and* shuffled `chain` at
  tiny. Then the "one-layer read cannot content-address" claim is false at this scale and the
  E18b text result does not generalise; do not scale the ladder, re-diagnose.

## Plan
- **Data:** on-the-fly `data/symbolic_tasks.py` via `data/bapo_ladder.py` rungs. No shard
  required for tiny; `scripts/build_symbolic_dataset.py` for medium/large GPU mixes.
- **Compute:** tiny on CPU (Cloud VM or laptop). Medium/large on Odra/Polonez, still
  `max_params < 100e6`.
- **Steps / epochs:** tiny default 800 steps, K1 = 4× (dense may train to 3200), batch 32,
  hidden 128 (~1M params), early-stop at 99% acc. Answers packed to ≥16 supervised tokens
  (`value_len` / `key_len` / `span_len`); dense is trained first and other arches are skipped
  if it misses 75%.
- **Launch:** `uv run python verification/bapo_capability_probe.py --scale tiny --out …`
  then `analysis/plot_bapo_capability.py`. Medium: same with `--scale medium` on GPU.
- **New foundation code:** see the plan. Reusable: extra DNA tasks, ladder, metrics, encdec
  baseline, probe, plots. No new `train_*.py`.

## Result
- Run id: `bapo_tiny_packed` (CPU) + `bapo_bridge512` (Polonez/Odra GPU)
- WandB: n/a (probe; no `compute/*`)
- Run reports: [`e24_tiny_bapo_ladder_20260913.md`](../../2_Experiments_Registry/run_reports/e24_tiny_bapo_ladder_20260913.md) · [`e24_bridge512_bapo_ladder_20260913.md`](../../2_Experiments_Registry/run_reports/e24_bridge512_bapo_ladder_20260913.md)
- Verdict: **mixed** — S1 pass at seq=128 and **512** right-align `far_copy` (E18 100% / 64 bits); **S1 fail at 1024** (dense 99.9% / 64 bits, E18 **0 bits**). S2 holds on 512 `recall_single` even with fixed offset (E18 0 bits vs dense 32). 4k spread and 512 chain are K1. Do not score E18 at 4k until a dense replica is ≥75%.
