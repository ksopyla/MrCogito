---
name: capability-suite
description: Run the MrCogito capability suite — the standard, graded exams (L0 learns at all → L1 carries a fact → L2 picks signal over a lookalike → L3 long reach → L4 multi-step reasoning → L5 language-like noise → L6 BAPO-hard stretch) that every new architecture trains on from scratch at 5M / 10M / 30M / 50M on Odra / Polonez, scored against recorded past architectures with a scale-up verdict. Use when the user wants to test, benchmark, screen, compare or "exam" a new or changed architecture, asks whether an architecture can learn to pick information or reason, asks whether it is worth scaling / more compute, wants 5M–50M probes, or asks to run, monitor, score or record the capability suite. Owns the end-to-end process (register → plan → smoke → sync → launch → monitor → score → record → change the suite). Not for evaluating pretrained checkpoints on STS-B/GLUE (experiment-evaluate), generic training runs (experiment-run), or designing the architecture (experiment-design).
---

# Capability suite

One standard ladder of exams for every architecture, so results are comparable across
experiments and "should we scale this?" gets a rule-based answer instead of a feeling.

- **Spec (rules, tables, rationale):** `docs/engineering_specs/capability_suite.md`
- **Definition:** `evaluation/capability_suite.py` (levels, 19 cells, sizes, step-size and
  budget policy, tiers, recorded references, scale rule, `SUITE_VERSION`)
- **Runner:** `scripts/run_capability_suite.py` · **Scorecard:** `analysis/capability_scorecard.py`
- **Trainer (unchanged, shared):** `verification/bapo_capability_probe.py` + `evaluation/bapo_models.py`
- **Tests:** `tests/test_capability_suite.py`, `tests/test_bapo_probe_tiers.py`

**Boundary.** This skill runs the suite and interprets its scorecard. Server hardware and
network facts → `remote-servers`; generic launches, env vars and Byobu habits →
`experiment-run`; pretrained-checkpoint benchmarks → `experiment-evaluate`; writing results
into the ledger → `experiment-track` (step 8 below says what to hand it); how to tell the
author → `research-comms`.

## The suite in one screen

| level | question | cells (tokens) |
|---|---|---|
| L0 Learns at all | copy / look up at all? | copy-128, lookup-128 |
| L1 Carries a fact | a whole fact survives? | copy-256, lookup-256, copy-512, lookup-512 |
| L2 Picks signal over a lookalike | fact vs same-shaped decoy | lookalike-128, lookalike-1k |
| L3 Long reach | retrieval in a long book | lookup-1k, lookup-2k (+ stretch copy-1k) |
| L4 Multi-step reasoning | 4-hop chain | chain-1k, chain-2k |
| L5 Language-like noise | same, in plausible "text" filler (Glyph) | fact-512, story-512, chain-512, fact-1k |
| L6 BAPO-hard stretch | reported, never gating | shuffled-1k, unique-256 |

| size | hidden | dense params | | tier | levels | seeds | use |
|---|---|---|---|---|---|---|---|
| 5m | 384 | 5.11 M | | screen | L0–L2 | 1 | does it learn at all (5m, 10m) |
| 10m | 512 | 8.97 M | | standard | L0–L4 | 2 | the comparison run (30m, then 50m) |
| 30m | 960 | 31.0 M | | full | L0–L6 | 3 | a claim run |
| 50m | 1216 | 49.5 M | | | | | |

Every job trains **the dense model next to the candidate** (same data, seed, budget): that
is the ceiling on the same replica. Pass = median answer accuracy ≥ 75 %.

## The process

### 0 · Preconditions (before any GPU)
- The architecture is **registered** in `evaluation/bapo_models.py`: a name in `ARCHES`, a
  branch in `build_model(...)` that builds it from `ArchSpec`, config-selectable on the shared
  foundation (no fork; see `research-implement`). Old arch names keep building identically.
- Arch-only probe flags (if any) go into `ARCH_FLAGS` in `evaluation/capability_suite.py`
  — never hand-edit generated commands (a different flag set is a different exam).
- Parameter count at each size is within ±5 % of the dense count (the probe prints
  params; the `--max_params` cap per size aborts larger builds). If it cannot be, say so in
  the report.
- `uv run pytest -q tests/test_capability_suite.py tests/test_bapo_probe_tiers.py` passes.
- The experiment spec states its targets **on the suite** (target frontier level, cells that
  must beat a recorded reference) — `experiment-design` step 5.

### 1 · Plan (local, seconds)
```bash
uv run python scripts/run_capability_suite.py --arch <arch> --sizes 5m 10m --tier screen
```
Prints every job and command, writes `plan.json`. Check the job count and the line
`N (size, cell) pairs use a step size not yet measured` — for those, add `--lr_pair` on the
first run at that size (also runs lr/2; the scorecard keeps the better one).

Typical sequence for a new architecture:
1. `--tier screen --sizes 5m 10m` → must reach at least L0–L1 before spending a night.
2. `--tier standard --sizes 30m` → the comparison with E18 / E21 / E30 / dense.
3. `--tier standard --sizes 50m` (or `full` at 30m for a claim) → the scaling trend and the verdict.

Other knobs: `--levels 3 4` or `--cells L3.lookup-1k` (targeted reruns), `--seeds N`,
`--controls dense e30` (also re-train a past architecture next to the candidate, useful at
a size with no references), `--extra "--flag value"` (appended to every job; use sparingly,
record it).

### 2 · Local smoke (minutes, catches crashes before GPU time)
```bash
uv run python scripts/run_capability_suite.py --arch <arch> --sizes 5m --levels 0 \
  --mode local --budget_scale 0.02 --eval_rows 16 --out /tmp/cap_smoke_<arch>
uv run python analysis/capability_scorecard.py --in_dir /tmp/cap_smoke_<arch>
```
Expect `done: N ok, 0 failed` and levels `uncalibrated` (2 % budget is too short for the
dense model — that is correct, not a failure). A crash here is a bug in the architecture or
its registration; fix it before touching a server. For a full-coverage check use
`--tier full --seeds 1` (≈ 30–40 min on a laptop CPU at 5m).

### 3 · Sync the code (git only)
Commit and push the branch; on each server:
```bash
ssh odra 'cd ~/dev/MrCogito && git fetch && git checkout <branch> && git pull --ff-only'
```
Never rsync/scp code. If the pull aborts with "untracked working tree files would be
overwritten", remove only those colliding untracked files (list them with
`git diff --diff-filter=A --name-only HEAD..origin/<branch>`), never `reset --hard`/`stash -u`.
Repo on both servers: `/home/ksopyla/dev/MrCogito`.

### 4 · Check the GPUs are free
Odra has 3 × RTX 3090, Polonez 4 × RTX 3090 (facts: `remote-servers`). Ask the
`server-checker` agent for a snapshot, or `ssh odra 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'`.
Use only idle GPUs; one job per GPU at a time (the scripts run their jobs sequentially).

### 5 · Generate the launch scripts **on the server** and start them
`Cache/` is gitignored, so scripts are generated where they run:
```bash
ssh odra 'cd ~/dev/MrCogito && export PATH="$HOME/.local/bin:$PATH" && \
  uv run python scripts/run_capability_suite.py --arch <arch> --sizes 30m --tier standard \
    --mode scripts --gpus 0 1 2 --host odra --out Cache/capability/<arch>_standard_30m && \
  bash Cache/capability/<arch>_standard_30m/launch/odra_start_all.sh'
```
- The starter opens one window per GPU in the byobu session `capability` (created if
  missing). Attach with `byobu attach -t capability`.
- Jobs are balanced by estimated cost; each writes `job.json`, `probe.log`, the rung JSON,
  and a `DONE` marker when finished. Re-running a GPU script **skips DONE cells** — safe
  after a crash or reboot.
- Splitting across servers: give each host its own `--out` (e.g. sizes 5m/10m on Polonez,
  30m on Odra, or the two seeds on different hosts). The scorecard merges folders (step 7).
- Naming: `Cache/capability/<arch>_<tier>_<sizes>[_<tag>]`.

### 6 · Monitor
```bash
ssh odra 'cd ~/dev/MrCogito/Cache/capability/<run> && echo "done $(find . -name DONE | wc -l) / $(find . -name job.json | wc -l)"; grep -h "^EXIT" launch/*.log 2>/dev/null | tail'
ssh odra 'tail -n 5 ~/dev/MrCogito/Cache/capability/<run>/30m/L3.lookup-1k/seed0/probe.log'
```
The byobu window prints `JOB …` / `EXIT … <code>` per cell. Exit codes:
- `0` finished, dense passed; `2` finished, **dense missed 75 %** on this cell (uncalibrated
  here; counts as done);
- anything else = crash → read that cell's `probe.log` (OOM at 2048 tokens → lower `--batch`
  via a targeted rerun with `--cells … --extra "--batch 2"` and note it; NaN → rerun with
  `--lr_pair`). Fix, then re-run the GPU script: finished cells are skipped.
Long-reach cells (1024–2048 tokens) take 1–3 GPU-hours each at 30m; a probe that extends
once ("eval CE still falling … extending") is normal, not a hang.

### 7 · Score
On the server (or locally after bringing the result folders back):
```bash
uv run python analysis/capability_scorecard.py --in_dir Cache/capability/<run>
# merge halves run on two hosts, once both folders are on one machine:
uv run python analysis/capability_scorecard.py --in_dir <odra_run> <polonez_run> --out_dir Cache/capability/<arch>_merged
```
Bring results back read-only from the server (results, not code — the git-only rule is
about code): `ssh odra 'tar -C ~/dev/MrCogito -czf - Cache/capability/<run>' | tar -xzf -`.

Outputs: `scorecard.md` (paste-ready), `scorecard.html` (level grid + bars with the best
past architecture and the dense ceiling as ticks), `scorecard.json`, `cells.csv`.

**Reading it:**
- **Verdict:** `scale up` / `promising — fix before scaling` / `not ready`, with the rules
  that failed listed underneath.
- **Frontier level** per size = the highest level with every level up to it passing.
- **Level status:** `pass`, `fail`, `uncalibrated` (every miss was on a cell the dense model
  also missed — says nothing against the candidate), `stretch k/n` (L6), `not-run`.
- Per cell: bits / prize, accuracy ± standard error, the dense bits on this run, the best
  past architecture at this size, examples to 75 %, step size, seeds.
- `per_position_acc` (in the rung JSON / `scorecard.json`) shows which answer letters
  survive — use it to diagnose a partial score (e.g. a salience wall: first letters right,
  later ones at chance).
- Compare like with like: same `SUITE_VERSION` (printed at the top) and same size.

### 8 · Record and report
Hand to `experiment-track`:
- master log row: arch, suite version, tier, sizes, verdict, frontier per size, link to the
  scorecard (keep the scorecard files under `Cache/`; paste `scorecard.md` into a run report);
- the experiment spec's **Result**: which target cells passed/missed vs its criteria;
- **new references**: append the candidate's median bits per (cell, size) to `REFERENCES`
  in `evaluation/capability_suite.py` with the run report as `source`, so the next
  architecture is compared against it. Adding references does not change the exams, so it
  does not bump `SUITE_VERSION`.

Tell the author (per `research-comms`): the verdict in one sentence, the frontier level at
each size in plain words ("learns and carries facts up to 512 tokens, misses the 1024-token
lookalike"), the one or two cells that beat or miss the best past architecture, and the
next step the verdict implies:
- `scale up` → a bigger model or a text smoke (the L7 text rung, when it exists);
- `promising — fix before scaling` → fix the listed failing rule (usually one level or a
  regressing cell) with targeted `--cells` reruns, not a new full sweep;
- `not ready` → back to design; do not spend more compute on this version.

## Changing the suite
The suite is a standard: change it deliberately.
- Any change to a cell, a size, the step-size/budget policy, eval rows or a scoring rule →
  **bump `SUITE_VERSION`**, update the spec tables, add or adjust tests, and note in the
  CHANGELOG why. Results across versions are not directly comparable.
- New cell: pick the level by what it tests, freeze it as a recipe + overrides, set its
  prize (the test checks it builds with that prize), run the dense model on it once to
  calibrate before it gates anything. New levels (e.g. L7 natural text) follow the same rule.
- Never pin an answer-length field (`value_len`, `span_len`, `key_len`) in a recipe override
  without re-checking the prize: overrides are applied after answer packing.
- New size: keep the 4-layer width-matched shape unless the suite version changes; record
  the dense parameter count in `SIZES`.

## Pitfalls
- **Hand-edited commands** — produce a different exam; add flags via `ARCH_FLAGS` or `--extra`
  and record them.
- **One seed** — the screen tier is a screen; claims need `standard`/`full` seeds. Gaps
  smaller than ~2 × the reported accuracy SE are ties.
- **Step-size cliffs** — a zero at a size with an unmeasured step size is not a verdict; rerun
  with `--lr_pair`.
- **Uncalibrated ≠ failed** — if the dense model misses too, the exam was not learnable at
  that budget; it counts neither way.
- **Long cells look stuck** — 2048-token cells run at batch 4 and extend once; check `probe.log`
  before killing.
- **`uv: command not found` over ssh** — prefix `export PATH="$HOME/.local/bin:$PATH"`
  (the generated scripts already do).
- **Busy GPUs** — check first (step 4); two jobs on one 3090 slow both and can OOM at 2048.
