# Capability suite — the standard exams for new architectures

- **Status:** active · suite version `2026-09-24.v1`
- **Code:** `evaluation/capability_suite.py` (definition) ·
  `scripts/run_capability_suite.py` (plan / launch) ·
  `analysis/capability_scorecard.py` (score + verdict) · probe `verification/bapo_capability_probe.py`
- **Tests:** `tests/test_capability_suite.py`, `tests/test_bapo_probe_tiers.py`
- **How to run it end to end:** the `capability-suite` skill (`.cursor/skills/capability-suite/SKILL.md`)
- **Related:** [small-model protocol](small_model_capability_protocol.md) (training rules the suite
  inherits) · [E30 review](../experiments_specs/ahead/E30_sliding_window_perceiver.md#built-vs-intended-review-2026-09-24)

## Why

Every architecture so far was scored on its own flag combinations, so results were hard
to compare and easy to misread: a frozen preset once quietly became a 16-bit exam instead
of the recorded 64-bit one, and the 1024-token runs used 16–32 eval rows with one seed.
The suite fixes one ladder of exams, one set of model sizes, one training policy and one
set of scoring rules. A new architecture runs the same exams as every past one, is
compared with their recorded results, and gets a plain verdict on whether it has earned
more compute.

It answers three questions, in order:
1. **Can it learn to pick the right information and reason over it?** Levels 0–4, then 5.
2. **Is it better than what we had?** Every cell carries the recorded results of the full
   model, the full read (E18), the averaged notebook (E21) and the sliding-window
   notebook (E30).
3. **Does it scale?** The same cells at 5M → 10M → 30M → 50M; the scorecard checks the trend.

## The ladder (difficulty goes up)

All exams are on-the-fly synthetic rows with an exact information content ("prize", in
bits): the answer is random, so a model that recovers `b` bits has carried `b` bits of
evidence from far back. DNA rows use 4 letters and random filler (exact floor). Glyph rows
(L5) use a typed vocabulary and **language-like Markov filler**, where the noise looks
plausible. Pass = median answer-letter accuracy ≥ 75 % (chance is 25 % on DNA, 12.5 % on Glyph).

| level | question it answers | cells (length · prize) |
|---|---|---|
| **L0 Learns at all** | can it learn to copy and to look up at all? | `L0.copy-128` (128 · 64) · `L0.lookup-128` (128 · 32) |
| **L1 Carries a fact** | does a whole fact survive the channel? | `L1.copy-256` · `L1.lookup-256` (256 · 48) · `L1.copy-512` (512 · 64) · `L1.lookup-512` (512 · 48) — 512-token cells use a fixed fact offset, as recorded |
| **L2 Picks signal over a lookalike** | keep the fact, ignore a same-shaped decoy | `L2.lookalike-128` (128 · 32) · `L2.lookalike-1k` (1024 · 64) |
| **L3 Long reach** | retrieval over a long book | `L3.lookup-1k` (1024 · 64) · `L3.lookup-2k` (2048 · 64) · stretch `L3.copy-1k` |
| **L4 Multi-step reasoning** | follow a 4-hop chain of facts | `L4.chain-1k` (1024 · 64) · `L4.chain-2k` (2048 · 64) |
| **L5 Language-like noise** | same skills when the filler is plausible "text" | `L5.fact-512`, `L5.story-512`, `L5.chain-512` (512 · 72) · `L5.fact-1k` (1024 · 96) |
| **L6 BAPO-hard stretch** | tasks a bounded channel should not solve easily | `L6.shuffled-1k` (shuffled 2-hop) · `L6.unique-256` — reported, never gating |

Stretch cells never gate a level. L5 has no recorded ceilings yet: its first run on the
dense model calibrates it.

**What the levels do not cover yet:** natural text. The next rung (planted facts in real
prose, and a next-token smoke at 512 tokens) is specified in
[E31](../experiments_specs/ahead/E31_sliding_window_latent_memory.md#exam-ladder-dna--structured-noise--text)
and will be added as L7 when its generator exists. Passing L0–L6 means "the mechanism
works in a controlled setting", not "it works on language".

## Sizes (width-matched, the E30 ledger family)

Four layers — one local layer (16-token reach), one global read, two local layers —
head width 64, one key/value head. Only the width changes.

| size | hidden | dense params | cap (`--max_params`) |
|---|---|---|---|
| `5m` | 384 | 5.11 M | 7 M |
| `10m` | 512 | 8.97 M | 12 M |
| `30m` | 960 | 31.00 M | 40 M |
| `50m` | 1216 | 49.54 M | 60 M |

A candidate should stay within ±5 % of the dense count at each size, or report the gap.
All four sizes fit one RTX 3090 per job (Odra: 3 GPUs, Polonez: 4).

## Training policy (inherited from the small-model protocol)

| size | ≤ 256 tokens | 512 | ≥ 1024 |
|---|---|---|---|
| 5m / 10m | 1e-3 | 3e-4 + warm residuals* | 1e-4 + warm* |
| 30m | 3e-4 + warm | 3e-4 + warm | 1e-4 + warm |
| 50m | 2e-4 + warm* | 2e-4 + warm* | 5e-5 + warm |

\* not yet measured in the ledger: the runner flags these; use `--lr_pair` (also runs lr/2)
on the first run at that size. Step-size cliffs are real (1e-4 passes, 2e-4 zeros the
2048 full read).

Budget: 800 × 4 steps at ≤ 256 tokens (batch 32), 1200 × 4 at 512 (batch 32) and 1024
(batch 8), 1200 × 4 at 2048 (batch 4); the probe extends once while eval loss is still
falling. Eval set: **256 rows** (the ledger used 16–32). The dense model trains next to the
candidate in every job — it is the ceiling on the same data, seed and budget.

## Tiers

| tier | levels | seeds | use it for | rough cost for one candidate |
|---|---|---|---|---|
| `screen` | L0–L2 | 1 | "does it learn at all?" at 5M and 10M | ~1–2 GPU-hours per size |
| `standard` | L0–L4 | 2 | the comparison run at 30M (then 50M) | ~one night on Odra's 3 GPUs per size |
| `full` | L0–L6 | 3 | a claim run; includes language-like noise and stretch cells | ~1–2 nights on both servers |

Costs are estimates from the E30 runs (≈ 0.04–0.14 s/step at ≤ 512 tokens and 30M;
1024-token cells take 1–3 GPU-hours each).

## Workflow for a new architecture

1. **Register it** in `evaluation/bapo_models.py` (`ARCHES` + `build_model`), config-selectable
   on the shared foundation. Add any arch-only probe flags to `ARCH_FLAGS` in the suite.
2. **Screen small:** `--tier screen --sizes 5m 10m` (one GPU, or locally with `--mode local`).
3. **Standard at 30M**, then 50M if 30M looks promising.
4. **Score:** `analysis/capability_scorecard.py` → `scorecard.md` / `.html` / `.json` / `cells.csv`.
5. Record the verdict and the frontier in the experiment's spec and the master log
   (`experiment-track`), linking the scorecard.

On a server (git-only sync — never copy files):
```bash
ssh odra
cd ~/dev/MrCogito && git pull --ff-only
uv run python scripts/run_capability_suite.py --arch <new_arch> --sizes 30m --tier standard \
  --mode scripts --gpus 0 1 2 --host odra --out Cache/capability/<new_arch>_standard_30m
bash Cache/capability/<new_arch>_standard_30m/launch/odra_start_all.sh   # one byobu window per GPU
# later
uv run python analysis/capability_scorecard.py --in_dir Cache/capability/<new_arch>_standard_30m
```
Scripts are resumable: a finished cell leaves a `DONE` marker and is skipped on restart.
Probe exit code 2 means "results written, dense missed 75 % on this cell" and counts as done.

## Scoring rules

- **Cell pass:** median answer accuracy over seeds ≥ 75 %. With several step sizes, the best one counts.
- **Level:** `pass` when every gating cell passes; `uncalibrated` when every miss is on a cell
  where the dense model also missed on the same run (the exam was not learnable at this
  budget — no evidence against the candidate); otherwise `fail`.
- **Frontier:** the highest level L with every level 0..L passing.
- **Compared with the past:** each cell shows the best recorded architecture (excluding the
  candidate itself) at the same size, and the dense ceiling on this run.
- **Size trend:** per cell, bits across sizes. It **regresses** if any step up in size loses more
  than 2 bits, **improves** if the total gain is more than 2 bits.

### Scale-up verdict

`scale up` needs all of:
1. a run at **≥ 30M**;
2. frontier **≥ L2** at the largest size (every L0–L2 cell passes);
3. at least one **L3/L4** cell that passes or ties/beats the best past architecture;
4. **no cell regresses** with size;
5. training speed **≥ 0.5×** the dense model at the largest size.

`promising — fix before scaling` = some level passes and there is a hard-cell win or an
improving trend, but a rule above fails (the scorecard lists which). Otherwise `not ready`.
"Scale up" means: worth a larger model or a longer text run, not a 1T-token pretrain.

**Where the past architectures stand (30M, from the references):** the dense model passes
L0–L2 and the 1024 chain but scores 0 on the 1024/2048 lookup (a trainability wall). E30
passes L0–L1 and the 1024 chain but misses the 1024 lookalike and lookup (~55 %), so its
frontier is L1 → "promising — fix before scaling". E21 fails L2 and L4.

## Comparability rules

- A change to any cell, size, policy value or scoring rule **bumps `SUITE_VERSION`**. The
  scorecard prints the version; compare only like with like.
- The frozen exams carry their prize in bits; `EXPECTED_PRIZE_BITS` and the tests fail if a
  preset silently changes the exam.
- References are copied from the run reports with their source file; add new ones when a
  suite run finishes (`REFERENCES` in `evaluation/capability_suite.py`).
- Reported per cell: recovered bits, accuracy ± standard error over rows, accuracy per
  answer letter, examples to 75 % (confirmed over 2 evals), training speed, parameters.
