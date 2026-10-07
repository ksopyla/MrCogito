# Capability checks — one leveled series of learning-capability tests

- **Status:** v4 structure agreed 2026-10-04 (author's decisions below); task definitions in code
  (`evaluation/capability_tasks.py`, version `v4-draft-2026-10-04`). Runs still use the v3 runners
  (suite + length battery) until the v4 runner lands — see "Migration".
- **Skill:** `capability-checks` (`.cursor/skills/capability-checks/SKILL.md`) runs it end to end.
- **Results:** ledger `docs/2_Experiments_Registry/results/capability/` · dashboard
  `docs/3_Evaluations_and_Baselines/capability_board.html` · raw folders on the NAS.

## What this is for

We are building our own architecture, so the question is **can this architecture *learn*** to pick,
hold, compose and reason over information — trained from scratch on synthetic exams whose answers carry
an exact amount of information. (Trained text checkpoints are a different question, answered by
`experiment-evaluate`.)

## How it grew, and why it is reorganized

There is **one exam engine**: `data/bapo_ladder.py` generates the tasks (random-letter "books" with
planted facts, chains, decoys) and `verification/bapo_capability_probe.py` trains and scores a model on
one task. Everything else was a configuration of that engine, added experiment by experiment:

| layer | added | how it configured the engine |
|---|---|---|
| ad-hoc probes | E24–E29 | per-experiment tasks and settings, 16–32 eval rows, one seed |
| graded suite v3 | 2026-09-25 | 19 frozen cells L0–L6, from scratch at the test length, 5M–50M, 3 seeds, pass on the mean over answer letters |
| length battery v1 | E31b, 2026-09-27 | lookup / chains / hard exams started from the run's own lookup-2k weights, laddered to 128k, first letter |
| experiment exams | E33/E33a | parallel chains with a 2 → 3 → 4-hop curriculum and replay |

The layers disagreed: the same task appeared in two places with different training (lookup at 2k: 98 %
in the suite, 67 % in the battery), two scoring rules, different seeds, levels that mixed task type with
length (L3 "long reach" vs L4 "reasoning"), shortcut tasks still counted, and the hardest capabilities
(holding many facts, real multi-hop reasoning, length transfer) missing from the suite.

## The structure (v4)

**Levels say what the model must do**, each building on the previous. **Length and size are separate
dials**, measured the same way on every level.

```
 LEVEL                                         × LENGTH                                   × SIZE
 C0 Carry         copy a span                    train at its length (128 → 2k);            screen at 10M,
 C1 Address       look up one fact               tasks trained at ≥ 1k are then read at      compare at 30M,
 C2 Discriminate  the fact among 1 → 8 decoys    every length up to 128k with no more        trend at 50M
 C3 Hold many     recall 1 of 8 → 16 facts       training (the length ladder)
 C4 Compose       in-order chain, 4 → 8 hops
 C5 Reason        parallel chains, 2 → 4 hops    ← research frontier
 C6 Aggregate     the fact that appears once, count
 C7 Language-like the same skills in Markov "text" filler   (C8 real text, when it exists)
 Flawed (listed, never evidence): shuffled chains without decoy chains, triple-match, majority
```

## Rules (author's decisions, 2026-10-04)

1. **From scratch only.** Every capability result starts from random init on our own architecture. No
   checkpoint from another run, experiment or pretrained model. (E33a's arm fine-tuned from E31's weights is
   labeled `not-from-scratch` and excluded from capability claims.)
2. **An in-run curriculum is allowed only when written in the task definition** (`curriculum` in
   `capability_tasks.py`): it starts from random init and is identical for every architecture, including
   the dense ceiling. Chaining stages as separate jobs is an implementation detail of the same schedule.
3. **Reading at longer lengths is evaluation, not training**, and applies to every task trained at ≥ 1k.
4. **Score = first-letter accuracy** (the first answer letter is predicted with no answer letters in
   context). Pass = median over **seeds 0, 1, 2** ≥ 75 %. Chance is 25 % on DNA letters, 12.5 % on Glyph.
   **Exams with several same-shaped candidates** (the parallel chains, the keyed lookups) score the
   **picked candidate** instead: the answer is decoded greedily and the planted candidate nearest to it
   must be the asked one. Their first letter has a guessing floor of ~40 % (answer with any candidate: the
   commonest first letter among the candidates wins); the picked candidate has 1 / #candidates, and a lossy
   copy of the right fact still counts. Every task lists its guessing floor (`floor`) next to chance, and
   every report prints it beside the score (E33a diagnosis, 2026-10-05).
5. **The dense model trains next to the candidate** on the same data, seed and budget: the ceiling. A
   miss where dense also misses is "uncalibrated", not a failure.
   *Learnability reference for C5 (author, 2026-10-06):* if the standard 4-layer dense control cannot learn
   a C5 exam, an **8-layer dense model** (same width, learnability check only) decides whether the task is
   learnable; if it passes, C5 is frozen with it as the reference and E31/E33 results on C5 count as
   evidence. The 4-layer dense stays the comparison everywhere else.
6. **Flawed tasks** stay defined with their reason so they are recognised; they never gate, never enter a
   level or package, are labeled by the runner and the scorecard, and are crossed out on the dashboard.
7. **One protocol for every model** (author, 2026-10-06). A task's recipe is its exam (`recipe` + `args`),
   its training budget (`train`; empty = the suite v3 budget) and its written curriculum, all in
   `capability_tasks.py` and identical for every architecture and the dense ceiling. Any change to them is
   a protocol change: it applies to all models, bumps `VERSION`, and gets a dated line in the calibration
   log below. Runs are built from the task definition (`task_job` in the study plan), never from hand-typed
   flags.

## Levels and tasks

Source of truth: `evaluation/capability_tasks.py` (generated table; `calibrating` = the from-scratch recipe
is not fixed yet, see Migration step 3).

| task | what | train length | status | curriculum / length |
|---|---|---|---|---|
| **C0 Carry** | | | | |
| `C0.copy-128` | copy 32 letters, 128-token book | 128 | active | — |
| `C0.copy-256` | copy 24 letters, 256-token book | 256 | active | — |
| `C0.copy-512` | copy 32 letters, 512-token book (fixed offset) | 512 | active | — |
| `C0.copy-1k` | copy 32 letters, 1024-token book | 1024 | active | ladder → 128k |
| **C1 Address** | | | | |
| `C1.lookup-128` | one fact, 128-token book | 128 | active | — |
| `C1.lookup-256` | one fact, 256-token book | 256 | active | — |
| `C1.lookup-512` | one fact, 512-token book (fixed offset) | 512 | active | — |
| `C1.lookup-1k` | one fact, 1024-token book | 1024 | active | ladder → 128k |
| `C1.lookup-2k` | one fact, 2048-token book | 2048 | active | ladder → 128k |
| `C1.lookup-16k` | one fact, trained up to a 16k book | 16384 | active | curriculum: one run from random init: 2k (C1.lookup-2k recipe) → 8k (1500 steps, batch 16, step 5e-5) → 16k (1000 steps, batch 8, step 5e-5); ladder → 128k |
| `C1.keyed4-1k` | 1 of 4 facts by its key, 1024 tokens (picked candidate, floor 25 %) | 1024 | **calibrating** | ladder → 128k |
| `C1.edge-1k` | 1 of 4 edges by its start node, 1024 tokens (picked candidate, floor 12.5 %) | 1024 | **calibrating** | ladder → 128k |
| **C2 Discriminate** | | | | |
| `C2.lookalike-128` | fact vs 1 look-alike, 128 tokens | 128 | active | — |
| `C2.lookalike-1k` | fact vs 1 look-alike, 1024 tokens | 1024 | active | ladder → 128k |
| `C2.decoy8-1k` | fact among 8 look-alikes, 1024 tokens | 1024 | **calibrating** | ladder → 128k |
| **C3 Hold many** | | | | |
| `C3.recall8-1k` | recall 1 of 8 facts, 1024 tokens | 1024 | **calibrating** | ladder → 128k |
| `C3.recall16-1k` | recall 1 of 16 facts, 1024 tokens | 1024 | **calibrating** | ladder → 128k |
| **C4 Compose** | | | | |
| `C4.chain4-1k` | in-order 4-hop chain, 1024 tokens | 1024 | active | ladder → 128k |
| `C4.chain4-2k` | in-order 4-hop chain, 2048 tokens | 2048 | **calibrating** | ladder → 128k |
| `C4.chain8-1k` | in-order 8-hop chain, 1024 tokens | 1024 | **calibrating** | ladder → 128k |
| **C5 Reason** | | | | |
| `C5.pchain2-1k` | parallel 2-hop chain among 3 decoy chains, 16-letter nodes, one edge past the asked node (picked candidate, floor 8.3 %) | 1024 | **calibrating** | curriculum: candidate (to be fixed by calibration): one run from random init, 1 hop (C1.edge-1k) → 2 hops; a stage ends when its picked-candidate score reaches 90 % or its budget runs out; ladder → 128k |
| `C5.pchain3-1k` | parallel 3-hop chain among 3 decoy chains, 16-letter nodes, one edge past the asked node (picked candidate, floor 6.2 %) | 1024 | **calibrating** | curriculum: candidate (to be fixed by calibration): one run from random init, 1 hop (C1.edge-1k) → 2 → 3 hops; a stage ends when its picked-candidate score reaches 90 % or its budget runs out; ladder → 128k |
| `C5.pchain4-1k` | parallel 4-hop chain among 3 decoy chains, 16-letter nodes, one edge past the asked node (picked candidate, floor 5 %) | 1024 | **calibrating** | curriculum: candidate (to be fixed by calibration): one run from random init, 1 hop (C1.edge-1k) → 2 → 3 → 4 hops; a stage ends when its picked-candidate score reaches 90 % or its budget runs out; ladder → 128k |
| **C6 Aggregate** | | | | |
| `C6.unique-256` | the fact that appears once, 256 tokens | 256 | active | — |
| `C6.unique-1k` | the fact that appears once, 1024 tokens | 1024 | **calibrating** | ladder → 128k |
| `C6.count-1k` | count, 1024 tokens | 1024 | **calibrating** | ladder → 128k |
| **C7 Language-like** | | | | |
| `C7.fact-512` | one fact in Markov 'text', 512 tokens | 512 | active | — |
| `C7.story-512` | a fact keyed by a word, 512 tokens | 512 | active | — |
| `C7.chain-512` | in-order hops in structured filler, 512 tokens | 512 | active | — |
| `C7.fact-1k` | one fact in Markov 'text', 1024 tokens | 1024 | active | — |
| **Flawed (never evidence)** | | | | |
| ~~`X.pchain2-1k-v1`~~ | parallel 2-hop chain, 32-letter nodes, first letter (E31b / E33a) | 1024 | flawed | guessable: any link target scores 44 % on the first letter, a chain end 25 % of whole answers; every arm sat on that floor (use C5) |
| ~~`X.pchain3-1k-v1`~~ | parallel 3-hop chain, 32-letter nodes, first letter (E31b / E33a) | 1024 | flawed | guessable: any link target scores 41 % on the first letter, a chain end 25 % of whole answers; every arm sat on that floor (use C5) |
| ~~`X.pchain4-1k-v1`~~ | parallel 4-hop chain, 32-letter nodes, first letter (E31b / E33a) | 1024 | flawed | guessable: any link target scores 39 % on the first letter, a chain end 25 % of whole answers; every arm sat on that floor (use C5) |
| ~~`X.shuffled2-1k`~~ | shuffled 2-hop chain, no decoy chains | 1024 | flawed | shortcut: the answer is the only node that is never the start of a hop, so it is found without following any hop (use C5 parallel chains) |
| ~~`X.shuffled3-1k`~~ | shuffled 3-hop chain, no decoy chains | 1024 | flawed | same no-hop shortcut as the shuffled 2-hop chain (use C5 parallel chains) |
| ~~`X.match3-1k`~~ | the fact planted three times | 1024 | flawed | at 1k the book holds the triple plus only 2 single facts: averaging all facts gives the majority letter at each position, no matching needed |
| ~~`X.majority-1k`~~ | majority letter of the book | 1024 | flawed | about 2/3 of the book is the winner, so any sample answers it |

## Packages (what a check runs)

| package | levels | size · seeds | length ladder | use |
|---|---|---|---|---|
| **screen** | C0–C2 | 10M · seed 0 | no | does it learn at all — a few GPU-hours |
| **core** | C0–C5 | 30M · seeds 0, 1, 2 | yes, to 128k | the full check every variant gets; the no-harm comparison uses it |
| **frontier** | C5–C6 | 30M · seeds 0, 1, 2 | yes | deeper tasks for the current research question; they join core once stable |
| **language** | C7 | 30M · seeds 0, 1, 2 | no | the same skills in plausible filler; real text (C8) when its generator exists |

A size trend (5M → 50M) reuses the same package at more sizes.

## Comparing variants: champion and no-harm

**Champion:** `e31_li_m1` (E31 latent memory, one reader entry per latent), chosen 2026-10-01 in E31b.
**No-harm rule:** a capability is lost when the variant scores more than 5 points below the champion on
any task and length where the champion passes (≥ 75 %), comparing medians over the same seeds. "Not run"
is not a pass. When a variant takes over, change `--champion` in `analysis/capability_board.py` and here.

## Past results: re-scoring without losing anything

Nothing is deleted or re-run. The ledger keeps per-job details (accuracy per answer letter, bits,
settings, every ladder length), so every past job is re-labeled onto a v4 task and re-scored on first
letter, with an honest match label (`legacy_suite` / `legacy_battery` in `capability_tasks.py`):

| match | meaning | counts as v4 evidence? |
|---|---|---|
| `same` | same exam, from scratch, frozen settings (all suite v3 cells; the lookup 2k → 8k → 16k stages) | yes |
| `settings-differ` | same exam from scratch, other step size / budget / rows (battery lookup at 2k, dense hard exams, suite cells run at a step size other than the suite default — e.g. the E31 lookup-2k runs at 5e-5) | shown, marked |
| `curriculum-differs` | from scratch through a schedule that is not the v4 one (battery chains and hard exams started from the run's own lookup-2k weights; E33a curricula) | shown, marked; not used for no-harm until the v4 recipe matches |
| `not-from-scratch` | started from another run's checkpoint (E33a fine-tuned arm) | no |
| `flawed` | a flawed task | no, crossed out |

**Reports and summaries are append-only:** each past report that cites capability numbers gets a dated
note — "Re-scored under capability checks v4 (YYYY-MM-DD): …, see the dashboard" — above the original
numbers, which stay as recorded. Coverage: E30, E31, E31b, E33a (in the ledger); E24–E29 (raw on the NAS,
collect first). E01–E22 and the Gemma backbone runs are text / pretrained models, out of scope.

## Migration

1. **Definitions** — `capability_tasks.py`, flawed flags in the suite runner and scorecard, this spec,
   skills and rules. *(done 2026-10-04)*
2. **Dashboard + re-run list** *(done 2026-10-04)* — the board is organized by level (C0–C7): one task table
   with every task, training-length and 128k scores, match labels, row status (active, partial, missing,
   calibrating, flawed), task definitions on hover and as a guide, one model filter; nothing re-run or
   re-scored. The generated re-run list: [`capability_reruns.md`](../3_Evaluations_and_Baselines/capability_reruns.md).
   Dated re-scoring notes in past reports follow once the re-runs land.
3. **Calibration study** (Odra, ~1 night) — for every `calibrating` task, dense and `e31_li_m1` from scratch:
   a step-size pair, the v3 budget vs 2×, and for chains/parallel chains the candidate in-run curriculum vs
   none. The cheapest recipe on which dense passes becomes frozen (`active`); if none does, the task
   stays `calibrating` and is reported, never gated.
4. **v4 runner** — one runner for a package: from-scratch jobs, written curricula, length ladders, seeds
   0–2, dense ceiling, ledger output. It replaces `run_capability_suite.py` + `battery_jobs` for new runs;
   the old runners stay for reproducing v3 / battery v1 results.
5. **Version freeze** — `VERSION = "v4"` once steps 3–4 land; the suite's `SUITE_VERSION` and
   `BATTERY_VERSION` stay frozen as history.

### Calibration log

Every recipe decision, dated. Dense, from random init, seed 0 unless noted; "picked" = the picked-candidate
score, read against the task's guessing floor. Ledgers: `results/capability/study/calibrate_reasoning*.json`.

| date | round | finding | decision |
|---|---|---|---|
| 2026-10-05 | 0 (E33a diagnosis) | old parallel chains: every arm, dense included, sat on the link-target guessing floor (first letter ~40 %) | `X.pchain*-1k-v1` flawed; C5 rebuilt with a chain overhang, 16-letter nodes and the picked-candidate score; keyed C1 tasks added (v4-draft-2026-10-05) |
| 2026-10-05 | 1 (Odra) | at the v3 budget dense passes C1.keyed4 (picked 79.5 %) but not C1.edge (10 %, floor 12.5 %); C5 pchain2/3 at the floor at 1× and 4×, at both step sizes and through the 1 → 2 → 3 hop schedule; C6.count at chance at 4× | count and C5 stay `calibrating` |
| 2026-10-05 | 2 (Polonez) | at **4× budget** dense passes C1.edge in every variant (8- / 16-letter nodes, 256- / 512 / 1k books, with or without overhang, after lookup: 92–98 %) and C3.recall8 (98 %); recall8 with 16-letter values fails (28 %); lookup → edge → pchain2 falls to the floor on pchain2 (12.5 %) | `train`: C1.edge-1k and C3.recall8-1k at 4× (`TRAIN_1K_X4`), C1.keyed4-1k at 1× (`TRAIN_1K`), for every model (v4-draft-2026-10-06); confirm on dense seeds 0–2 (`confirm_calibration`) before `active` |
| 2026-10-05 | 2 (E33 reasoning) | E31 arms pass edge (~99 %) from lookup weights, then lose hop 1 on pchain2: the question does not say how many hops, so stage 2 punishes stage 1 | C5 needs the hop count in the question (author approved 2026-10-06); C5 stays `calibrating` until dense passes the new exam |
| 2026-10-06 | exam change | the generator can state the hop count in the question (`--hop_count_in_question`: k hop markers after the start node; off by default, recorded exams bit-identical; floors unchanged: picked 12.5 / 8.3 / 6.2 / 5 % for 1–4 hops) and mix hop counts in one stage (`--replay_hops`) | C5.pchain2/3/4-1k state the hop count, train at 4× (`TRAIN_1K_X4`) for every model (v4-draft-2026-10-06.2); round 3 calibrates dense from scratch: direct, 1 → 2 → 3 hop curriculum, and the curriculum with half the rows at the previous hop count |
| 2026-10-06 | confirm (Polonez) | dense seeds 0–2 on the written recipes: C1.keyed4-1k picked 95 / 94 / 86 %, C1.edge-1k picked 96 / 95 / 97 %, C3.recall8-1k first letter 97 / 98 / 94 % | C1.keyed4-1k, C1.edge-1k and C3.recall8-1k **active** (v4-draft-2026-10-06.3): every model runs them with these recipes |
| 2026-10-06 | 3 (Polonez) | C5 with the hop count in the question, 4× budget: 1 hop picked 96 %; 2 hops direct 6.9 % (floor 8.3), after 1 hop 12.1 %, after 1 hop with half the rows kept at 1 hop 13.9 %; 3 hops direct 7.4 % (floor 6.2), 1 → 2 → 3 6.6 %, mixed 11.3 %. Teacher-forced accuracy stays ~92 %: dense completes a node once its first letter is given but never picks it | C5 stays `calibrating`, never gated; the second hop is not learned by dense at 4× in any schedule — next: a larger budget or a smaller exam (fewer / shorter chains), decided with the E33 reasoning session |
| 2026-10-06 | protocol | a 4-layer dense control may lack the depth for 2 dependent multi-token lookups in one pass (E33 reasoning) | author: an 8-layer dense model is the C5 learnability reference when the 4-layer one fails (rule 5); round 4 adds it (`calibrate_c5_deep`) next to short nodes, 2 vs 4 chains and a 16× budget (`calibrate_c5_r4`) |
| 2026-10-06 | 4 (Odra) | dense, hop count in the question, 16× budget (19.2k steps; the probe extends while loss falls): 2 hops with 4 chains — 4-letter nodes picked 8.6 % (per-letter accuracy jumped to 70 % at ~16k steps, the choice stayed at the floor 8.3 %), 8-letter 11.7 %; 2 chains — 8-letter 27.3 %, 4-letter 27.1 % (floor 16.7 %); 3 hops, 4-letter 6.7 % (floor 6.2 %); **8-layer dense**: 16-letter nodes at 4× 7.0 %, 4-letter at 16× 3.9 %; the 256-token stage does not fit 4 chains of 8-letter nodes | no C5 recipe yet; neither depth (8 layers), budget (16×), node length nor fewer chains makes dense pick the second hop — next: one-token names (E33 reasoning diagnostics, `--n_symbols`) |
| 2026-10-07 | 5 (Odra + Polonez) | recipe A (final answer only), dense, one written run from random init, 1 hop (4×) → 2 hops (16×): 8-letter names — 1 hop picked 95.9 %, 2 hops 9.8 %, 2 hops with half the rows at 1 hop 12.1 % (floor 8.3 %); 4-letter names — 1 hop fails at 4× (15.8 %, floor 12.5 %), later stages stopped. E33 reasoning diagnostics (from E31 lookup weights, seed 1): the loop with progress exits + whole-name loss + half the rows at 1 hop passes 2 hops at ~98 % (one hop per round) | two recipes from now on, each calibrated on dense first — **A** (`C5.pchain*`): composition, final answer only, no intermediate targets (a loop uses answer exits, no name loss); **B** (`C5.path*`): chained lookup with a scratchpad, every model writes the path (`--chain_answer_path`). The E33 loop result (final answer only, node targets on internal states via progress exits / name loss) is a third setting with no dense counterpart yet: a mechanism result, not a capability cell |
| 2026-10-07 | 5b (Odra) | **recipe B passes for dense**: 8-letter names, answer = written path, one run from random init, 1 hop (4×, picked 98.0 %) → 2 hops (16× cap, half the rows at 1 hop): picked 90.0 % on the final node (floor 8.3 %), every letter 87 %, done in ~25 min | new tasks `C5.path2-1k` / `C5.path3-1k` (recipe B: `--chain_answer_path`, 8-letter names, `TRAIN_1K_X16`, written curriculum 1 hop → … with half the rows at the previous hop count), `calibrating` until dense seeds 0–2 pass (`confirm_c5_path`); 3 hops calibrated on dense (`calibrate_c5_path3`); v4-draft-2026-10-07. Recipe A (`C5.pchain*`) stays `calibrating` |
| 2026-10-07 08:44 | confirm (Odra) | dense seeds 0–2 on `C5.path2-1k` as written: 1 hop 97 / 97 / 94 %, 2 hops picked 93 / 88 / 90 % (floor 8.3 %); `C5.path3-1k` seed 0: 1 → 2 → 3 hops picked 96 / 91 / 89 % (floor 6.2 %) | **`C5.path2-1k` active** (v4-draft-2026-10-07.2): every model runs it with this exam, budget and curriculum; `C5.path3-1k` confirmed next on seeds 0–2 (`confirm_c5_path3`) |

Until step 4, run new variants with the v3 runners (`suite.md`, `battery.md` in the skill) and label the
results as legacy protocol.

## Where results live (the storage contract)
The same rule covers every result in the project
([`results/README.md`](../2_Experiments_Registry/results/README.md)): `Cache/` is scratch, cited
results are committed under `docs/2_Experiments_Registry/results/`, raw folders go to the NAS.

| layer | what | where | who writes it |
|---|---|---|---|
| raw | every log, rung JSON, ladder JSON, checkpoint | server `Cache/capability/<run>` or `Cache/study/<study>` | the runners |
| archive | the full raw folder | NAS `/nas/ml_data/mrcogito/results/{capability,study}/<name>.<host>/` | `scripts/pull_capability_results.sh` |
| **ledger** (source of truth for comparisons) | compact JSON per (folder, host): per job and arch accuracy ± SE, first letter, per-letter accuracy, bits, speed, every ladder length | `docs/2_Experiments_Registry/results/capability/{suite,study}/<name>.<host>.json`, committed | `scripts/pull_capability_results.sh` (collector `analysis/capability_ledger.py`) |
| scorecard | suite verdict and frontier for one run | `analysis/capability_scorecard.py --in_dir <ledger files or folders>` | on demand |
| **board** (the visual summary) | all variants: no-harm, length battery, suite grid | `docs/3_Evaluations_and_Baselines/capability_board.html`, published as an Artifact | `analysis/capability_board.py` |
| registry | what it means | master log row, spec Result, run report, agenda line | `experiment-track` |
| references | suite-cell bits of past architectures for the scorecard | `REFERENCES` in `evaluation/capability_suite.py` | `capability-checks` step "record" |

**When to pull.** After each phase finishes, at least once a day during a multi-day study (Polonez
can shut down for heat at any time), and always before a write-up or a visual summary. Pulling is
idempotent: the ledger file is rewritten from the folder, the NAS copy only grows.

**W&B is not used for these results.** The probes log only a final summary, run offline in their
hundreds, and their numbers are small enough to version in git, where every agent can read them
without network access. W&B stays the home of text-training runs and their checkpoint evaluations.

**Hand-written HTML reports** (architecture explainers, study narratives) may still be written,
but their numbers must come from the ledger, and the standard visual comparison is the board.


## Maintaining the checks
- **A task changes** (recipe, length, budget, curriculum, rows) → it is a new task id or a new `VERSION`;
  results across versions are compared only through the match labels.
- **A new task** runs first on dense (and the champion) to calibrate, then becomes `active` in a level.
- **A task turns out to have a shortcut** → set `status="flawed"` with the reason; never delete it.
- **After each result:** pull → commit the ledger → regenerate the dashboard → `experiment-track`.
- Suite v3 (`docs/engineering_specs/capability_suite.md`) and battery v1 (`battery_jobs`) are kept as the
  history of how past results were produced.
