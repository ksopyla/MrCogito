# E33a — Read, think, reread: a tied loop over the layers above the global read

- **Status:** done — **inconclusive** (closed 2026-10-05): no capability lost; reasoning untested (the parallel-chain
  exam was a guessing floor for every model). Approved 2026-10-03 (the author asked to design, plan, implement and run
  in one request; alignment page reviewed in chat on 2 Oct; the author removed the checklist step).
- **Serves:** Vision priority 3, "reason in concept space", on top of the E31 latent memory (vision priorities 1–2).
  It is the first multi-hop bet after E31 was chosen and E30 parked (1 Oct).
- **Implementation plan:** [E33a_reread_loop_plan.md](E33a_reread_loop_plan.md)
- **Alignment page:** [e33a_architecture.html](../../3_Evaluations_and_Baselines/e33a_architecture.html)
  (information flow, the vectors that change per loop, gradients, queries/keys, literature).
- **Owner / dates:** Krzysztof Sopyla · opened 2026-10-03 · closed 2026-10-05
- **ID:** a flavour of [E33](../ahead/E33_iterative_concept_reads.md) (iterative concept reads). The loop moves *up*: it now
  contains the answer-forming layer. The design pages of 1–2 Oct called it an "E34 candidate"; under the ID rules it
  is E33a.

## Hypothesis
If the E31 question side loops the **global read plus the next local layer** 4 times with tied weights, decodes a
guess after every loop through an untied answer layer (deep supervision), and keeps the memory read-only, then the
parallel 3-hop chain at 1k goes from **46 % to ≥ 75 % first-letter accuracy**, with no new weight matrices and
lookup unchanged. **Because:**
- k hops need k dependent reads. Each read must ask with what the previous read returned (BAPO reachability;
  memory-network hops).
- The layer inside the loop is where "found B (as a target)" becomes "ask for B (as a source)". E33 had only one
  feed-forward between reads for that conversion.
- A loss at every loop gives each read a direct error signal. The ARC Prize analysis of HRM found this, not the
  hierarchy, drives recursive models.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (PerceiverARLM, exclusive global read, E33 round targets),
  `nn/latent_memory.py` (E31 writer), `verification/bapo_capability_probe.py`, `verification/length_ladder.py`,
  `scripts/run_study_queue.py`. No fork. New behaviour is config: `message_loop_*` on PerceiverARConfig and
  matching probe flags.
- **Init / checkpoint (author's call, 3 Oct: "it could be important that the model learns to use the loop"):**
  - **main arm: random init with the loop on from step 0.** E31 (page encoder, length-invariant address, one reader
    entry per latent) is trained on lookup at 2k with R = 4, then on the chain curriculum.
  - **comparison arm:** the same loop fine-tuned from `len_lookup_e31_li_m1_s1`, the past single-read lookup
    checkpoint (the start of E33/E33b/E33c). With R = 1 the new model computes exactly that checkpoint's function.
- **Baselines to beat** (first letter at 1k, 8-letter nodes, 4 parallel chains, chance 25 %, ±3 points):

  | run | pchain2 | pchain3 |
  |---|---|---|
  | single read (`e33b_*_R1_s1`) | 45 | 46 |
  | E33 tied read loop, R4 (`e33b_*_R4_s1`) | 44 | 45 |
  | E33 R4 + per-round node targets (`e33c_*_R4aux_s1`) | 44 | 45 |
  | dense, 30M (own lookup-1k weights, seed 0) | 48 | 44 |

- **Materially new vs E33:**
  1. The loop contains the answer-forming local layer (core = global read + local), not the read alone.
  2. An untied answer layer after the loop, also used to decode every loop's guess. E33 decoded its round guesses
     through the head while skipping the two upper layers.
  3. A no-harm protocol: lookup replay in training, per-loop evaluation, and a writer-frozen arm.

## The architectural bet
The four main-path layers are regrouped. No layer is added:

| role | layers | runs |
|---|---|---|
| prelude ("read the question") | local layer 1 (16 back) | once |
| **core ("read, then think")** | **global read (own side + memory) → local layer 2** | **R = 4, tied** |
| answer layer ("formulate the answer") | local layer 3 + final norm + head | after the last loop; during training also after every loop (exits) |

- **Memory:** written once by the E31 writer and read-only during the loop (the notebook). Its keys and values are
  fixed across loops; queries, question-side keys/values and the residual state change each loop.
- **Loop marker:** a zero-initialised vector per loop (R × 960, about 0.01 % of the model), the same as E33's round
  embedding. Each block's existing embedding re-injection (α·x + β·x0) runs every loop.
- **Exits:** loss = final CE + 0.3 × Σ_{r<R} CE(exit r). Exit targets:
  - **progress** (default): exit r predicts chain node r+1, the terminal from the last hop on;
  - **answer**: every exit predicts the final answer (Ouro style). Rows without chain nodes, such as lookup replay,
    use the answer.
- **Replay:** 25 % of each batch is the lookup exam (`recall_single`), so retrieval is trained alongside.
- **Skips:** the U-net skips keep their meaning per loop. Local layer 2 takes the global read's output from the same
  loop; the answer layer takes the prelude's output.

## Arms (1k chain stages, 8-letter nodes, curriculum pchain2 → pchain3 → pchain4, 25 % lookup replay)
| arm | start | change | isolates |
|---|---|---|---|
| **loop from step 0 ★ (seeds 1, 2)** | random init → lookup-2k **with the loop** | R = 4, core = read + local, progress exits | the bet: a model that learns to use the loop from the start |
| loop fine-tuned | past checkpoint `len_lookup_e31_li_m1_s1` | as ★ | does learning the loop from the start matter? |
| single-read control | past checkpoint | R = 1, same replay | the baseline under the same data |
| answer exits | ★'s loop-trained lookup weights (seed 1) | every exit predicts the answer | works without intermediate-node labels? |
| writer frozen | ★'s loop-trained lookup weights | writer weights frozen in the chain stages | retrieval unchanged by chain training; is a fixed notebook enough? |
| prelude re-injection | ★'s loop-trained lookup weights | + the prelude output added back each loop (one scalar gate, zero init) | does reminding the loop of the question help? |
| wide core | random init → its own loop lookup-2k | core = read + both local layers, head only after | more thinking per loop vs an untied answer layer |

27 jobs (study phase `e33a`), about 69 GPU-h on Odra's 3 GPUs (≈ 23 h wall).

E33's own replica (the read-only loop) is not rerun: its numbers above share this init and protocol.

## Why this is not a safe retread
E33 tested a read-only loop: −2 to +6 points over 10 cells. Here the loop contains the layer that must convert a
read result into the next query, and every loop is supervised through the real answer path. Analogy: a student
with a notebook rereads it with a sharper question after each attempt (cue-dependent recall). The residual stream is
working memory; the latents are the fixed episodic store. In the literature: End-to-end Memory Networks (hops over
one memory), and the prelude / tied core / untied coda layout of Huginn and Mixture-of-Recursions.

## Success criteria (set BEFORE running)
Ceiling. The same memory answers a single fact among look-alikes at 1k at 100 % (decoy8, `hard_decoy8_e31_li_m1_s1`)
and lookup at 98 %. Three perfectly composed reads therefore reach about 94–97 %. The gate is 0.8× that ceiling.
- **S1 (reasoning):** loop ★ pchain3 at 1k **≥ 75 %** first letter on **both** seeds, and ≥ 20 points above the
  single-read control.
- **S2 (four hops):** loop ★ pchain4 (R = 4) ≥ 60 % on both seeds.
- **S3 (no harm):** lookup (replay exam, 1k) within 2 points of the past single-read checkpoint (98 %), at the last
  exit and at exit 1. Lookup ladder of the final pchain4 checkpoint (1k–32k) within 5 points of that checkpoint's
  ladder (first letter: 99 / 96 / 94 at 2k / 8k / 32k). For the from-scratch arm, its loop-trained lookup stage
  must also reach that level (it is the same exam and budget).
- **S4 (trace):** on pchain3, exit r's first letter matches node r+1 on ≥ 70 % of rows for r = 1, 2.
- **S5 (length):** loop ★ pchain3 ladder at 4k ≥ 0.8× its 1k score. Report to 16k.

## Kill criteria (set BEFORE running)
- **K1:** loop ★ pchain3 < single-read control + 15 points on both seeds after the curriculum (extended once). The
  loop placement is not the bottleneck. Next: latent thoughts (sequence-recurrent reads) or memory-side
  consolidation.
- **K2:** S1 passes but S3 fails in every arm except the writer-frozen one → keep the writer frozen. If that arm also
  fails S1, report a trade-off, not a win.
- **K3 (time):** any job with eval CE flat (< 0.02 nats change over its last 1200 steps) and first letter < 50 % at
  4800 steps → stop that chain.

## Plan
- **Data:** DNA `chain_parallel` (4 chains, 8-letter nodes, `--key_len 8`) at 1024 tokens, hops 2 / 3 / 4, plus 25 %
  `recall_single` replay. Eval: 256 rows per stage (±3 points), first letter and per-exit first letter.
- **Compute:** Odra, 3 × RTX 3090 (Polonez offline). 27 jobs, about 69 GPU-h (≈ 23 h wall). The 2k lookup stages
  with the loop are the long ones, at about 5.5 h each.
- **Steps:** per stage 1200 × 4 at step size 5e-5, batch 32, extended once if eval CE is still falling (the study
  protocol `L1K_FT`).
- **Launch:** study phase `e33a` in `scripts/study_plans/e30_vs_e31.py`, run by `scripts/run_study_queue.py` on Odra
  in byobu session `study` (windows `e33a_g0..2`). Every job's command is generated from the plan, never hand-edited.
- **New foundation code:** `message_loop_rounds`, `message_loop_span`, `message_loop_inject`,
  `message_loop_exit_aux`, `message_loop_exit_targets` (PerceiverARConfig); probe flags for these plus
  `--replay_recipe`, `--replay_frac`, `--freeze_writer` and per-exit eval; `--loop_rounds` in the length ladder.
  All are reusable for any exclusive-read model.

## Generality: what carries over to text (added 3 Oct, at the author's request)
The loop is a general mechanism. Only two training aids here are specific to synthetic exams:
| part | general? | in text |
|---|---|---|
| prelude → [global read + local] × R → answer layer | **yes**: runs at every position, no QUERY needed | unchanged; compute ≈ 10 layer passes per token instead of 4 |
| exits with **answer** targets | **yes**: the answer is the next token | unchanged (Ouro-style loss at every loop) |
| exits with **progress** targets | **no**: needs the generator's intermediate nodes | not available; kept only as a scaffold arm (an upper bound with step labels) |
| lookup replay | data trick | the text mix plays this role |
| writer frozen | yes | optional |

- **The general claim is judged on the answer-exit arm** (2 seeds: `loopans_s1` and `loopans_s2`). The
  progress-exit arm (`loop_s1/s2`) shows how much step labels add.
- The capability checks and the suite use answer exits only, so E33a gets no supervision E31 never had.
- **Gap outside E33a:** E31's memory is read only by question-side tokens, after QUERY (the slot mask in
  `build_message_mask`, `slot_side < side`). The "text mode" in the E31 spec is not implemented: every token
  reading the windows that closed before it. The text trainer (`training/concept_pretraining_factories.py`) also
  does not expose the latent memory or the loop. Training on text needs that read mode first. It is E31's
  planned text rung and is independent of the loop.

## Capability checks: no past capability lost (added 3 Oct, at the author's request)
*Process: [capability checks](../../engineering_specs/capability_checks.md) — the battery below is
`BATTERY_VARIANTS["e33a"]` (phase `battery_e33a`, alias `e33a_e31b`); results go to the ledger and the board.*
E33a gets the same battery as E31, so the final comparison is like for like. All of it runs on Odra after the
`e33a` phase. The queues are chained, about 3 days in total.
1. **E31b protocol on the loop** (study phase `e33a_e31b`, seeds 1 and 2, starting from each seed's loop-trained
   lookup-2k weights). The same exams, steps and ladders as `e31_li_m1` in E31b:
   - lookup with the 8k and 16k stages, ladder 2k → 128k;
   - the in-order 4-hop chain at 2k, plus an 8k stage, ladder to 128k;
   - the hard exams at 1k (recall8, recall16, decoy8, unique, match3, chain8), ladder 1k → 128k;
   - the recall 8k stage.

   First-letter accuracy throughout, 128 rows per length.
2. **Capability suite, full tier, 30M, 3 seeds** (`Cache/capability/e33a_full_30m`, arch `e33a_loop`, the same
   `--message_raw_window 256` as the E31 suite `li_full_30m`). Every cell is trained from scratch with the loop on.
   - Dense controls are not rerun: the suite data is deterministic per seed, so the scorecard merges this folder
     with `li_full_30m`, which holds dense, `e31_li` and `e31_li_m1` on the same seeds and suite version.
   - `--accum_mult 2` halves the micro-batches only; the effective batch is unchanged.
3. **Comparison report** at the end: E33a vs E31 (`e31_li_m1`, `e31_li`) on every cell and ladder above, plus the
   reasoning gain on the parallel chains. Each E31 number comes from E31b (first letter) and `li_full_30m`.

**No-harm rule:** a capability counts as lost if E33a falls more than 5 points below `e31_li_m1` (first letter) on
any cell or ladder length where `e31_li_m1` passes (≥ 75 %), on the median of its seeds.

## Result
- **Runs:** study `e30_vs_e31` jobs `e33a_*` (Odra, 2026-10-03 → 05); suites `e33a_full_30m` (Polonez seed 0, Odra seeds
  1–2) and `e33a_lookup2k_lr5e-5` (Odra). No W&B (capability probes).
- **Report:** [e33a_reread_loop_20261005.md](../../2_Experiments_Registry/run_reports/e33a_reread_loop_20261005.md) ·
  diagnosis [e33a_loop_diagnosis_20261004.md](../../4_Research_Notes/e33a_loop_diagnosis_20261004.md).
- **Ledger:** `results/capability/suite/e33a_full_30m.{polonez,odra}.json`, `suite/e33a_lookup2k_lr5e-5.odra.json`,
  `study/e30_vs_e31.odra.json` (`e33a_*`).
- **Verdict: inconclusive.** S1/S2/S4/S5 not reached: every arm, the R = 1 control and dense sat at the exam's guessing
  floor (44 / 41 / 39 % for 2 / 3 / 4 hops), so K1 is formally met but uninformative; the v1 parallel-chain exams are
  now `flawed`. No harm (S3 in the v4 sense): all 17 non-flawed suite tasks within 3 points of `e31_li_m1` (median of
  seeds 0–2); lookup-2k at 5e-5 99 vs 98; the lookup 2k → 16k curriculum reaches 93–100 % to 128k on one seed of two,
  as E31 does (one good seed each).
