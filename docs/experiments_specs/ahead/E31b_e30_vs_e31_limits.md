# E31b — E30 vs E31 limits study (same platform, same protocol, 1k → 128k)

- **Status:** running. Wave 1 launched 2026-09-27 on Odra (3 GPUs) and Polonez (4 GPUs).
- **Serves:** choosing the memory for [E32](E32_two_level_concept_memory.md) (two-level concept
  memory) on evidence, not on the E31 suite alone. That suite compared E31 against an
  E30 that had a 16-token reach and an absolute address.
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-27 · closed —
- **Code:** branch `e31-latent-memory`.
  - Plan: `scripts/study_plans/e30_vs_e31.py`.
  - Runner: `scripts/run_study_queue.py`.
  - Cost bench: `verification/memory_cost_bench.py`.
  - Results: `Cache/study/e30_vs_e31/<job>/` (probe JSON, `ladder.json`, checkpoint) and
    `Cache/capability/li_full_30m/` (suite).

## Questions (from the author, 27 Sep)
1. Why did 48 tokens per latent fail for E31? Sweep 6 → 64 tokens per latent and find
   hyperparameters that work.
2. What is the right token-to-latent ratio for E31, and how was E30's set?
3. Head-to-head: E30 vs E31 trained on the same checks.
4. Why is E30 more efficient? Is it only configuration?
5. Harder reasoning and multi-hop tasks: find each architecture's limits.
6. Use the capability-suite framework. The same training protocol for both; test up to 128k.

## Arms (all length-invariant, all on the E31 platform)
| arch | writer | memory entries | address | params |
|---|---|---|---|---|
| `e30_li` | E30 perceiver: K = 32 static queries per 256-window (stride 192) pool **main-path layer-0 states**, 64-token causal reach (`e30_ctx`) | N/6 | slot keys RoPE'd at QUERY (`--swp_slot_pos boundary`, new) | 31.3 M (writer 0.13 M) |
| `e31_li` | E31 page writer: 2 bidirectional layers over the window on 128-d embeddings; 32 latents × 512; competition | 5 per latent → **0.83 N** | no window-start sinusoid; keys at QUERY | 35.7 M (writer 4.48 M) |
| `e31_li_m1` | same writer, **one reader entry per latent** | **N/6** (same as E30) | same | 35.1 M (writer 3.95 M) |
| `dense` | none; 4 full-attention layers | N × 4 layers | RoPE | 31.2 M (ceiling at training length) |

`e31_li_m1` is the matched comparison: the same number of reader entries as E30, so any gap
left is the writer, not the memory size. **Platform:** H 960, 4 layers (1 SWA pre, 1 global
read, 2 SWA), head_dim 64, 1 KV head, 128-d token embedding, no hashed n-grams, warm
residuals, `--message_raw_window 256` (the memory is the only path longer than 256 tokens,
and cost is linear). Every run uses 256 eval rows, and ladders use 64 rows per length.

## Protocol (the same for every arch)
- **1k:** 1200 × 4 steps, batch 32, step 1e-4. **2k:** 1200 × 6 steps, batch 32 (2
  micro-batches), step **5e-5**. Both extend once when eval CE is still falling.
- **Curriculum stages** (from the E31 ladder): 8k (1500 steps, batch 16) → 16k (1000
  steps, batch 8), step 5e-5, each starting from the previous stage's weights.
- **Harder exams start from the arch's own lookup-2k weights**, then train at 1k with step
  5e-5. With the invariant memory, chain from scratch did not take off; from lookup
  weights it did.
- **Length ladder** after every stage: 2k (or 1k) → 128k, 64 rows per length, accuracy by fact
  depth, peak GPU memory, seconds per row.
- **Seeds:** takeoff is stochastic at 2k. Headline claims need ≥ 2 seeds. Single-seed cells are
  marked as such.

## Block A — ratio sweep (Q1, Q2)
Exams at 1k:
- **lookup** (`recall_single`): one 32-letter fact.
- **recall8** (`recall`, 8 facts, 4-letter keys): the memory must keep all 8 facts (512 bits),
  because it does not know the question when it writes.

| arm | window / stride | K | m | tokens / latent | tokens / reader entry | entries at 1M |
|---|---|---|---|---|---|---|
| e31 K32 m5 (E31 default) | 256 / 192 | 32 | 5 | 6 | 1.2 | 870k |
| e31 K32 m1 | 256 / 192 | 32 | 1 | 6 | 6 | 175k |
| e31 K16 m1 | 256 / 192 | 16 | 1 | 12 | 12 | 87k |
| e31 K8 m1 | 256 / 192 | 8 | 1 | 24 | 24 | 44k |
| e31 K4 m1 | 256 / 192 | 4 | 1 | 48 | 48 | 22k |
| e31 K4 m1 s256 | 256 / 256 | 4 | 1 | 64 | 64 | 16k |
| e31 W512 K8 m1 (the failed coarse point) | 512 / 384 | 8 | 1 | 48 | 48 | 22k |
| e31 K8 m4 | 256 / 192 | 8 | 4 | 24 | 6 | 175k |
| e31 K4 m8 | 256 / 192 | 4 | 8 | 48 | 6 | 175k |
| e31 W512 K8 m8 | 512 / 384 | 8 | 8 | 48 | 6 | 175k |
| e30 K32 / K16 / K8 / K4 | 256 / 192 | 32–4 | — | 6 / 12 / 24 / 48 | same | 175k–22k |
| e30 K4 s256, e30 W512 K8 | 256 / 256, 512 / 384 | 4, 8 | — | 64, 48 | same | 16k, 22k |

Plus dense on recall8 as the ceiling. Every arm runs a ladder over 1k–16k.

**Why 48 tokens per latent might have failed**, and the arm that separates each cause:
- **H1 read bandwidth.** One reader entry is one 64-d key and one 64-d value (1 KV head). A
  whole 32-letter fact must come out of that single vector, and the answer stack must unpack
  letter *i* from it. *Test:* K4 m8 and W512 K8 m8 (48 tokens per latent, 6 per entry). If
  those pass and K4 m1 fails, the bottleneck is the read, not the latent.
- **H2 write selectivity.** Few latents with competition over a mostly-noise window may not
  bind the one fact. *Test:* the K sweep at m = 1, with per-letter accuracy.
- **H3 window length.** In the 22 Sep coverage sweep, E30 died at 512-token windows even with 64
  queries ("the kill is the window, not the note count"). *Test:* W512 K8 vs W256 K4, both at
  48 tokens per latent.
- **H4 optimisation.** The coarse m = 1 run left chance late (step ≈ 2,900) and was still rising
  at 37 % when the budget ended. The m = 5 run never left chance. Both were one seed each, at
  2k, with the absolute address. *Test:* 1k exams (reliable takeoff), the invariant
  address, and a second seed at the boundary points in wave 2.

**How E30's ratio was set:** the 22 Sep coverage sweep
([report](../../2_Experiments_Registry/run_reports/e30_coverage_and_breadth_20260922.md)) tried
about 3, 6 and 11 tokens per slot (128 / 256 / 512 windows, K = 32) at 1k. 3 and 6 passed the
chain, 11 (a 512 window) scored zero, and the default stayed at 6 (W = 8K, stride 0.75 W). That
sweep had E30's 16-token reach and changed window length and ratio together, so this study
measures the ratio again.

**Decision rule for "the right ratio":** the largest tokens per entry whose median bits are
≥ 0.9× the 6-token arm on both exams, with a 16k ladder ≥ 0.9× as well. It is reported
separately for tokens per latent (write compression) and tokens per reader entry (read
bandwidth). E32's coarse level needs the write ratio; its fine level needs the read ratio.

## Block B — head-to-head capability suite (Q3, Q6)
Capability suite `2026-09-25.v3`, full tier (L0–L6), 30m, **2 seeds**:
- arches `e30_li e31_li e31_li_m1`, with dense as the control;
- `--extra "--message_raw_window 256"`;
- the two 2k cells use `--lr_scale 0.5` (step 5e-5; at 1e-4 every arch except E30 stayed at
  chance).

The scorecard gives the frontier level and verdict per arch. It adds a third seed where
two seeds disagree across the pass line.

## Block C — length (Q3, Q6)
For each arch and seeds 0 and 1:
- lookup: 2k → 8k → 16k, ladder to 128k after each stage;
- 4-hop chain from the lookup-2k weights: 2k → 8k, ladder to 128k.

`e31_li` lookup (both seeds) and chain (seed 1) are imported from the E31 ladder runs; the
same flags were used. The pass line is ≥ 75 % at 32k and a flat depth profile. The study also
records the longest length ≥ 75 % and the accuracy at 128k.

## Block D — harder exams (Q5)
Each exam runs at 1k, starting from the arch's lookup-2k weights, with a ladder over
1k → 128k. Dense runs from scratch at 1k as the training-length ceiling. Wave 1 uses seed 1.

| exam | recipe | BAPO class | what it probes |
|---|---|---|---|
| recall8 / recall16 | `recall`, 8 / 16 facts, 4-letter keys | MATCH2 | capacity: many facts kept without knowing the question |
| decoy8 | `select`, 8 decoys | MATCH2 + noise | picking the fact among 8 look-alikes |
| shuf2 / shuf3 | `chain`, 2 / 3 shuffled hops, no distractors | REACHABILITY *in name* | **has a shortcut** (below); kept as a control |
| pchain2 / pchain3 | `chain_parallel`: 2 / 3 shuffled hops among 3 decoy chains | REACHABILITY | dependent lookups in one read; the start node picks the chain (added 27 Sep, seed 0) |
| chain8 | `chain_ordered`, 8 hops | DFA | long in-order composition |
| unique | `unique` | Σ-hard | the fact that appears once (no key) |
| match3 | `match3` | MATCH3-hard | the fact planted three times |
| count / majority | `count`, `majority` | aggregation | whole-book statistics (1 answer token, 2-bit prize) |

**Shortcut found in the shuffled chain (27 Sep).** With no distractor edges, the answer is
the only node that is a target but never a source, so it can be found without following a
hop. E31-LI reached 96 % on shuf3 at 1k and 80 % at 16k; that measures sink detection, not
3-hop reasoning. Distractor edges do not remove it (a distractor's source is never a
target). `chain_parallel` plants 3 decoy chains of the same length: 4 sinks, and only the
start node in the question tells which one is the answer (test:
`test_chain_parallel_removes_pure_target_shortcut`). **The suite's L6.shuffled-1k cell has
the same shortcut**, so the 96–98 % scores there (E31, E30_ctx, dense) are not evidence of
multi-hop reasoning.

**Two more exams are easier than their BAPO class suggests at this size** (read from the generator):
- `match3` at bridge_1k has the triple plus only 2 single facts. An attention read that
  *averages* all facts gets the majority letter at each position, with no matching needed.
- `majority` fills about 2/3 of the book with the winner, so any sample of the book answers it.

Both stay in the study as controls. They are not reported as MATCH3 or MAJORITY capability.
Genuinely hard versions need ≥ 30 single facts (a 2k+ book) and a balanced body.

The expected limits, to be confirmed or refuted:
- one exclusive read after QUERY cannot do shuffled hops ≥ 3 at length, or aggregation;
- recall16 at length tests the precision of content addressing over many similar slots.

## Block E — cost (Q4)
`verification/memory_cost_bench.py` measures, at 8k → 1M with batch 1 on an RTX 3090:
- seconds per row;
- peak memory;
- the writer's share of forward time;
- memory entries and parameters.

Arches: `e18_local`, `e30_li`, `e31_li`, `e31_li_m1`, and `dense` up to 128k.

**Prior (to confirm):** the gap has three parts.
1. **Memory size.** 0.83 N vs N/6 entries comes from m = 5. That part is pure configuration,
   and m = 1 closes it.
2. **Writer compute.** E31 runs its own 2-layer, 256-wide page encoder over every token
   (4/3 overlap), plus 2 latent rounds. E30 scores the main path's existing K/V with one
   960→128 projection. That part is structural: config can shrink it (1 encoder layer, a
   smaller width) but never to E30's level, because E30 reuses the main path.
3. **Parameters.** E31 has +4.3 M; E30's writer is 0.13 M. Also structural.

Block A with m = 1 shows whether the configuration part costs accuracy.

## Wave plan and budget
- **Wave 1** (≈ 170 GPU-hours, ≈ 30 h wall):
  - Odra: Block A, Block C seed 1, Block D seed 1.
  - Polonez GPUs 2–3: Block C seed 0 and the dense hard ceilings.
  - Polonez GPUs 0–1: Block B.
- **Wave 2** (conditional):
  - second seeds at the ratio boundary;
  - Block D at 2k for the exams that pass at 1k;
  - hyperparameters at the chosen ratio: latent width 256 / 512 / 1024, encoder layers
    1 / 2, rounds 1 / 2 / 3, E30 pre-encoder reach 64 / 128, E30 query heads;
  - the cost bench once a GPU is free.
- **Wave 3:** whatever wave 2 leaves open, then the E32 prototype.

## Result
*(interim, updated as runs finish; seed in the job name; accuracy in %)*

**28 Sep, early morning. E30's length failure was also the address.** Lookup, trained at 2k
only, then run on the length ladder:

| arm (2k only) | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|---|
| e30_li s0 | 97 | 95 | 91 | 77 | 63 | 45 | 34 |
| e30_li s1 | 98 | 94 | 71 | 54 | 35 | 31 | 26 |
| e31_li s0 | 97 | 96 | 91 | 76 | 56 | 40 | 32 |
| e31_li s1 | 98 | 97 | 94 | 87 | 69 | 51 | 36 |
| e30_ctx (old absolute address), s2 | 88 | 83 | 57 | 29 | 25 | 25 | 25 |

With slot keys at QUERY, the E30 writer transfers with length about as well as E31: one seed
each way. The 8k / 16k curriculum stages are running.

**Ratio sweep, lookup-1k (seed 0, first arms):**

| arm | tok/latent | tok/entry | 1k | 2k | 4k | 8k | 16k |
|---|---|---|---|---|---|---|---|
| e31 K32 m1 | 6 | 6 | 99 | 99 | 98 | 95 | 88 |
| e31 K8 m1 | 24 | 24 | 77 | 79 | 79 | 76 | 67 |
| e31 W512 K8 m1 | 48 | 48 | 42 | 42 | 42 | 38 | 32 |
| e31 W512 K8 m8 | 48 | **6** | 45 | 44 | 42 | 38 | 33 |

- **m = 1 costs nothing.** K32 m1 matches the E31 default (m = 5) at 1k, with 5× fewer memory entries.
- **Read bandwidth does not rescue the 512-token window.** Eight reader entries per latent
  (m8: 6 tokens per entry) score the same as m1. So H1 is rejected for the long window: the
  loss is in the write, whether that is window length (H3) or latent count (H2). W256 K4 (48
  tokens per latent, 256 window) separates H2 from H3; it is queued.

**Harder exams, e31_li seed 1, trained at 1k from lookup-2k weights:**

| exam | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k | note |
|---|---|---|---|---|---|---|---|---|---|
| recall8 (8 facts) | 98 | 98 | 96 | 86 | 68 | 48 | 34 | 31 | |
| recall16 (16 facts) | 97 | 98 | 93 | 79 | 56 | 40 | 31 | 30 | |
| shuf2 | 98 | 97 | 97 | 95 | 81 | 52 | 41 | 31 | shortcut |
| shuf3 | 96 | 96 | 95 | 91 | 80 | 62 | 44 | 36 | shortcut |
| match3 | 99 | 98 | 96 | 92 | 87 | 72 | 55 | 37 | averaging-easy |

- Multi-fact recall learns quickly from lookup weights (13–26k examples). With 1k training
  alone it transfers about 8×, the same length profile as lookup trained only at 2k. The
  curriculum stages are what carried lookup to 128k.

**Dense at 1k, from scratch (the training-length ceiling):**
- recall8 and recall16: **chance**.
- decoy8: 99 %, but 24 % at 2k.
- shuf2 / shuf3: 99 % at 1k (the shortcut), 76 % / 29 % at 2k.

Dense from scratch is therefore not a fair ceiling for exams where the memory arms start from
lookup weights. Wave 2 starts dense from its own lookup checkpoint.

**Chain from lookup, e31_li seed 0:** chance after 7,200 steps (no extension; CE flat). Seed 1
took off at step ≈ 4,000. Takeoff is stochastic; wave 2 retries with a doubled budget.
