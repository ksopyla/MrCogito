# E33t — the catalogue read: notes index the text, they are not read; balanced filing, a trained index, a new place per pass, two levels

- **Status:** **approved 2026-11-07** ("Aligned, build it") on the alignment page
  [e33t_architecture.html](../../3_Evaluations_and_Baselines/e33t_architecture.html), every decision as shown there.
- **Serves:** vision priority 2 (long context through the notebook) and the author's 2026-11-07 aim: chase LSA's
  quality with a simpler design of our own that can reason over memory and reach > 10M tokens (tested smaller first),
  then scale to 100–200M if 30M matches. Direct follow-up of E33p stage 1.
- **Implementation plan:** [E33t_plan.md](E33t_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-07 · closed —
- **ID:** E33t. Drafted as "E33s" on 2026-11-07 (commits 56ce7d5 … 2179a9d on `e33n-lsa-testbed`), but `dev` has
  owned E33s since 2026-10-21 (multi-hop on real text taught by written reasoning, `E33s_written_chains_real_data.md`);
  renamed the same day before any run. E33q / E33r are text-check runs, E33w / E33x the iCoT runs; E33u, E33v unused.
- **Evidence behind it:** E33p stage 1 analysis page
  [e33p_vs_lsa.html](../../3_Evaluations_and_Baselines/e33p_vs_lsa.html) (2026-11-07): the measured limits L1–L6
  below. LSA (arXiv 2610.42147) as the reference read; SwAV's equipartition (balanced soft assignment by Sinkhorn
  iterations) for the filing; the E33a/E33c loop as the place where lookups chain.

## What E33p stage 1 measured (the limits this spec answers)
30M, 16k rows, positions ≥ 2k; share of the gap G between the window control (0 %) and the dense ceiling (100 %).
Notes-open and the full note read are seed 1 on AWS; dense, window and LSA-native are seed 42 at home (same-seed
controls on AWS follow during stage 1).

| measure | notes-open (E33p) | LSA-native | source |
|---|---|---|---|
| share of G at 4k / 16k / 32k / 64k | 47 / 39 / 31 / 21 % | 97 / 99 / 100 / 127 % | long-document eval, G(16k) = 0.199 nats |
| **L1** notes with no member token | 30.4 % (largest note per page ≈ 14 of 8 on average) | 0 % (≤ 16 keys by construction) | `writer/empty_notes`, `writer/members_max_mean` |
| **L2** queries whose 512-token budget is full | 8–19 % (≈ 44 valid of 64 notes) | 66 % | `select/*/cap_full`, `select/*/groups` |
| **L3** reading-layer mass at 8–16k, pass 1 | summaries 37 %, opened 35 %, window 28 % | opened 67 %, sinks 5 %, window 29 % | `read/r0/8192-16384/*` |
| **L4** picks repeated from the previous pass | 37–47 % | 18–25 % | `select/r*/overlap` |
| **L5** fade with length | −26 points from 4k to 64k | +30 points | as above; the summary softmax grows with the note count |
| **L6** training speed at 16k (one RTX 3090) | 10.6k tokens/s | 17.6k (dense 25.0k) | `perf/real_tokens_per_second` |
| **L7** dense top-32 far keys opened, pass 1 → 4 | 28 → 27 % (full read: 19 → 13 %) | 40 → 68 % | flow probe B1, 16 rows × 512 queries, dense s42 reference |
| **L7** copy source opened on far-copy queries, pass 1 → 4 | 12 → 36 % | 31 → 87 % (92 % in pass 2) | flow probe B2 |
| **L7** reading-layer mass on the copy source, pass 4 | 3.6 % | 9.7 % (dense 9.9 %) | flow probe B2 |

The full note read (E33) recovers 6 %, so opening exact tokens is what works; how they are found is what is weak
(L7: our search finds the copy source a third as often as LSA's, and does not improve over the passes, while LSA's
goes from 31 % in pass 1 to 92 % in pass 2: its passes already work as successive lookups).

## Hypothesis
If the notes stop being read and become only a **catalogue** for the reading layer, with every page token filed under
exactly one balanced note (none empty, none over 16), the catalogue trained to predict where the read's attention
lands, the 512-token budget filled by tokens instead of by note count, and each loop pass searching only where the
same query's earlier passes did not look, then the 30M model recovers **≥ 80 % of the window→dense gap at 16k**
(E33p notes-open: 39 %; LSA-native shows 99 % is reachable at this size) and **keeps it to 64k**, while a two-level
search (pages first, then the notes inside the best pages) holds that quality with a per-query search that grows
with the number of pages rather than tokens.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (perceiver_ar, the E31c page writer, the closed read, the E33a loop),
  `nn/exact_read.py` (groups, per-query choice, the flex `open_term`, chunked table lookup since 561fe26),
  `nn/lsa_buckets.py` (the reference read), the LSA testbed (`scripts/run_e33n.py`, `infra/aws/jobs/e33n_run.sh` with
  multi-GPU and spot support since 7d8222f), the instrumentation (`nn/flow_stats.py`, `training/flow_callback.py`,
  `evaluation/flow_probe.py`), evals `evaluation/long_doc_loss.py`, `evaluation/copy_split.py`,
  `evaluation/run_lsa_suite.py`. One trainer entrypoint; the new arms are config.
- **Init / checkpoint:** random init, E33p's protocol unchanged.
- **Baseline to beat (E33p stage 1, 16k, positions ≥ 2k):** notes-open 2.4696 (39.4 % of G) · LSA-native 2.3503
  (99.2 %) · dense 2.3488 · window 2.5480. Same-seed values replace these when the AWS seed-1 controls finish.
- **Materially new:** vs E33p, the notes leave the read (they only choose what to open) and gain a training signal
  of their own; filing is balanced; the budget counts tokens; passes are made to look in new places; pages stop
  overlapping; a two-level search. Vs LSA, the index is a **trained writer** (it sees the whole page both ways and
  is taught by the read where queries actually look), not a parameter-free clustering of keys, and the two levels
  come from pages and notes instead of a separate coarse mode.

## The architectural bet
The backbone, data and loop are E33p's (30M: width 576, 2 pre + reading + 5 stack layers, loop of 4 over [reading
layer + 1 local layer], one KV head of 64 in the reading layer, no RoPE there). The writer is E31c's latent-memory
page writer (2 layers, 32 latents, 2 rounds); only its role changes. Per closed page and per loop pass:

1. **Pages without overlap.** Pages of 256 tokens with stride 256 (E33p: stride 192). A token belongs to exactly one
   page and one note; at 16k: 64 pages, 2,048 notes (E33p: 85 pages, 2,720 notes).
2. **Balanced filing.** The writer's assignment A[t, n] (softmax over the page's 32 notes) is balanced by 3 Sinkhorn
   iterations so every note receives on average 256 / 32 = 8 tokens (SwAV's equipartition), giving Â. Each token
   is filed under its largest Â entry, with a hard cap of 16 per note (overflow goes to the next-best note with
   room). No note is empty unless its page has fewer live tokens than notes (document ends).
3. **Note keys.** κ_n = Σ_t Â[t, n] k_t / Σ_t Â[t, n] over the page, k_t the reading layer's own key of token t in
   this pass (E33p's tied key, now on the balanced assignment). Page key: π_p = mean of its tokens' keys.
4. **Choice per query and pass** (no gradient): score s_i(n) = max over query heads of cos(q_{i,h}, κ_n) (one head
   with a match is enough). Candidates are notes of closed pages of the query's document with a member older than
   its window, **minus every note the same query opened in an earlier pass** (a pass looks somewhere new). Notes are
   taken best first, their members oldest first, **until 512 tokens are held** (no fixed note count; at most 128
   notes for the kernel's bounds).
   - **Two-level arm:** the query first scores page keys π_p (same rule), keeps its best 16 pages, and scores only
     their notes (16 × 32 = 512 note scores instead of all of them).
5. **One softmax** over **4 learned empty slots** (a key and a value each, shared by all positions: a place for
   attention that finds nothing; LSA uses the first 4 tokens of the document instead) ‖ the last 512 raw tokens ‖
   the opened raw tokens. No note summaries in the read.
6. **Index loss (new).** The note scores learn to predict where the read's attention lands. Per query (a sample of
   256 query positions per row and pass): target t_g = attention mass of the read on group g's opened tokens, for
   the chosen groups, plus the mass the same query would give to 32 sampled unchosen eligible groups (their members'
   keys scored with the query, no values needed), renormalised over that set; prediction p_g = softmax_g(τ · s_i(g))
   with a learned temperature. L_index = KL(t ‖ p), target without gradient, weight λ = 0.1 of the LM loss. Its
   gradient reaches the writer's assignment through κ (via Â) and the reading layer's keys; the LM loss still trains
   everything else. Without it the writer would get no signal once the summaries leave the read.

- **Budget per query:** 4 + 512 + 512 keys, flat in length. Search cost per query: all notes (flat) or pages + 512
  notes (two-level). Memory for the reading layer: one 64-number key and value per token (256 bytes in bf16:
  2.6 GB at 10M tokens).
- **Arms (stage A, 30M):**
  - **catalogue (main):** steps 1–6, flat search.
  - **catalogue, two-level:** the same with the page-first search.
  - **controls:** E33p's seed-1 dense, window and LSA-native, trained on the same AWS node and protocol.
- **Information budget:** an opened token keeps its full key and value (exactly what the dense ceiling reads); the
  catalogue's job is only to point. What can be lost is recall: the share of the tokens the dense ceiling reads most
  that the 512 opened tokens contain (E33p probe B1), and the copy source on far-copy tokens (probe B2).

## Why this is not a safe retread
It is not a knob on E33p. It removes the summary read that every notebook arm since E30 kept (the notebook becomes
an index only), gives the index its own training signal from the read (LSA's index is untrained; E33p's learned only
through the summaries), and uses the loop passes as successive lookups over a two-level catalogue. Analogy: a
library catalogue: drawers (pages) and cards (notes); the catalogue is never read for content, it is only searched;
each card covers about the same number of books; a librarian who learns from which books readers actually opened;
and a reader who, after the first visit, searches the drawers not yet opened.

## Training protocol
E33p's: FineWeb-Edu GPT-2 1.5B tokens at 4k (constant LR 1e-3 after 1 % warmup), then LongDocs-lite 0.2B tokens at
16k (3 % warmup, 1 − sqrt decay to 0.1 ×, fresh optimizer); 64 rows of 4k per optimizer step (micro 4 × accumulation
4 × 4 GPUs on the AWS node). Seed 1, the same as the AWS controls.

## Evaluation
E33p's suite, all on the fixed code (561fe26): long-document loss by position at 4k–128k (8 rows per source; 64 rows
at 16k and 64k for the verdict, bootstrap over documents), the copy split (near / far / no copy), RULER at 4k and
16k, core benchmarks, and the flow probe against the seed-1 dense reference (recall of its top-32, dense mass in the
opened set, copy-source hit rate, attention by distance). Eval switches inside each new model: opened tokens off,
pass exclusion off (every pass may re-pick), index loss's temperature reported. Speed at 4k and 16k on the same node.

## Instrumentation
E33p's A and B unchanged, plus: filing balance (members per note: min, max, share empty), index-loss value and the
KL per pass, the rank correlation between note score and read mass on the chosen notes, distinct tokens opened over
the 4 passes, and for the two-level arm the share of the dense top-32 that falls in the 16 kept pages (page recall).

## Success criteria (set BEFORE running)
Shares of G use same-seed controls (AWS seed 1) on positions ≥ 2k; the dense ceiling (100 %) and LSA-native (99 %
at 16k in E33p) show the marks are reachable at this size and data.
- **S1 (quality):** the main arm recovers ≥ 80 % of G at 16k.
- **S2 (no fade):** its share at 64k is at least its share at 16k minus 5 points (LSA-native: +28 points).
- **S3 (it finds what matters):** recall of the dense top-32 in pass 4 ≥ 0.9 × LSA-native's (E33p: 0.68, so ≥ 0.61;
  notes-open 0.27), copy source opened in pass 4 ≥ 0.9 × LSA-native's (0.87, so ≥ 0.78; notes-open 0.36), and on
  far-copy tokens it closes ≥ 0.8 × the share LSA-native closes (pre-flight: the copy split on E33p's LSA-native and
  notes-open).
- **S4 (two levels hold):** the two-level arm is within 5 points of the main arm's share at 16k and at 64k.
- **S5 (cost):** training speed at 16k ≥ 0.9 × LSA-native's on the same node (E33p notes-open: 0.60 ×).
- **S6 (no harm):** loss on positions < 512 at 16k within 0.01 nats of the window control; 4k held-out loss after
  the trunk ≤ the window control's; core benchmark mean within 1 point of the window control's.

## Kill criteria (set BEFORE running)
- **K1:** after the decay the main arm recovers < 50 % of G at 16k → stop E33t; the catalogue changes do not explain
  the gap to LSA (then test LSA's key clustering inside our writer as the index, a new spec).
- **K2 (index not learning):** by 25 % of the trunk the rank correlation between note score and read mass on the
  chosen notes is < 0.2 → stop and debug the index loss before spending the run.
- **K3 (health):** loss spikes > 0.5 that do not recover within 1k steps, or a trunk loss more than 0.1 above the
  window control at 25 % of the run → stop that run and debug.
- **K4 (cost):** a stage-A run takes more than 1.3 × LSA-native's wall time on the same node → fix the kernel before
  stage B.

## Plan
- **Pre-flight (free, home servers, during stage 1):** the copy split on E33p's LSA-native and notes-open (S3's
  ceiling); the flow probe on every E33p model (recall of the dense top-32 per pass).
- **Stage 0 (Mac + one GPU):** tests — Sinkhorn filing gives ≤ 16 and ≥ 1 members per note on full pages; the cap's
  overflow rule; pass exclusion never re-opens a note within a query; with the budget ≥ all eligible tokens the read
  equals the dense read restricted to closed pages plus the window; no future leak (content and row length); the
  index loss's gradient reaches the writer and the keys and none flows through the choice; the two-level search with
  16 ≥ all pages equals the flat search. Then speed and memory at 4k and 16k on one GPU (K4 early check), and a tiny
  pipeline check on the Mac.
- **Stage 0 result (2026-11-07 07:55–08:35 UTC, Polonez GPU 1, RTX 3090, `infra/aws/jobs/e33t_stage0.sh` at
  `47339bf`):** 66 GPU tests pass (flex = SDPA reference, forward and backward; worst relative gradient difference
  2e-6 catalogue, 8e-6 two-level); the tiny pipeline check passes for both arms (trunk then decay, instrumentation
  on). Speed, tokens/s: catalogue trunk 4k (micro 4) 23,571, 21.2 GB · decay 16k (micro 1): catalogue 13,854,
  two-level 15,485, LSA-native 18,697 (same GPU), E33p notes-open 10.8k (stage 0, Odra). Projected full run
  (1.5B at 4k + 0.2B at 16k) ≈ 1.12 × LSA-native's time: **K4 passes**; at 16k alone the catalogue is 0.74 × LSA
  (S5 asks ≥ 0.9, judged after stage A).
- **Stage A (30M, AWS 4 × L40S spot):** the main arm and the two-level arm after the seed-1 controls of stage 1
  (queue order dense, window, LSA-native, then pages-open), about 5 hours and $30 per model. Gate: S1–S6, K1–K4.
- **Stage A follow-up, the pass exclusion off in training (arm `catalogue_reuse`, approved by the author
  2026-11-07 ~21:00 UTC):** the catalogue with `lm_open_new_per_pass` False and nothing else changed (same 72,687,765
  parameters), seed 1, the stage-A recipe, then the planted-password probe on it (`infra/aws/jobs/e33t_reuse.sh`,
  chained on the AWS node after the stage-1 seed-1 window and LSA-native controls, about $31). Why — the probe
  (`evaluation/flow_probe.py --passkey`, 2026-11-07 20:17–20:23 UTC, 40 rows at 4k, 20 at 16k; S3
  `runs/e33t/probe_passkey/`): the catalogue opens the key on pass 1 (94 % at 4k, 71 % at 16k, a third of its read on
  it) but returns on passes 2–4 only 7–14 % of the time, while the LSA port and dense put 37–41 % of their read on the
  key on pass 4; first answer token right 10 % (LSA port 25 %, dense 30 %, notes-open 0 %). The exclusion switched off
  only at test time reopens it (85 %) but gives it 6 % of the read and lifts answers to 20 %: the passes must learn it.
  Preliminary stage-A main arm (scored 2026-11-07 19:58 UTC against the home seed-42 controls): 75 % of G at 16k,
  99 % at 64k. Same gates S1–S6 as the main arm; plus the probe's first-token score against the LSA port's.
- **Capability checks (if S1 holds):** the core package on the home servers (C0–C5 from scratch, seeds 0–2, ladders
  to 128k), no-harm rule against the E31 champion; targets `C1.lookup-16k` ≥ 75 % first letter at 64k and
  `C4.chain4-1k` ≥ 75 % first letter (the passes-as-lookups claim).
- **Stage B (if S1 holds): page and note sizes.** Around the stage-A winner: 4, 8 and 16 tokens per note at
  256-token pages (64 / 32 / 16 notes), and 512-token pages with 64 notes. Four runs, about $120.
  **Approved by the author 2026-11-08 ~17:30 UTC:** the winner is `catalogue_reuse` (8 tokens per note at
  256-token pages, already trained), so three new runs (`infra/aws/jobs/e33t_b.sh`): `catalogue_reuse_n4`
  (64 notes, 72,769,685 parameters / 30,328,597 without word tables), `catalogue_reuse_n16` (16 notes, 72,646,805 /
  30,205,717) and `catalogue_reuse_p512` (512-token pages, 64 notes, 72,900,757 / 30,459,669); only the writer's
  note slots differ. Same gates as stage A against the seed-1 controls; the winner is the best 16k share whose 64k
  share does not fall below it (S2), ties broken by RULER at 4k.
  **Machines:** the planned g6e.12xlarge spot node was refused at 17:40 UTC (MaxSpotInstanceCountExceeded in every
  us-east-1 zone: the account's 48-vCPU spot G quota is held by the 10M-context runs; on-demand G quota is 0). The
  author chose one on-demand H100 (p5.4xlarge, $6.88/h) per model, launched from ~18:00 UTC with a retry every 5 min
  while H100s are short, 16 h hard stop each; micro-batch 8 (trunk) / 2 (decay) on the 80 GB card, same rows per
  optimizer step as the 4-GPU runs (64 / 16). Expected ~8–10 h and ~$60–70 per model. Caveat for the comparison:
  one GPU instead of four, so the same data in a different micro-batch split (numerics only, no change of recipe).
  A spot quota raise to 96 vCPUs (two 4-GPU nodes) was requested 2026-11-08 17:56 UTC for later stages.
  **Moved to the home servers (author, 2026-11-09 ~20:20 UTC: "train what has been scheduled"):** no H100 came free
  in 138 rounds (2026-11-08 17:58 → 2026-11-09 20:19 UTC, every us-east-1 zone, every 5 then 10 min) and the L40S
  spot quota (64 vCPU by then, case to 96 still open) stayed held by the 10M-context node, so the AWS loop was
  stopped at 20:22 UTC with no machine launched ($0). Launched 20:23 UTC at `0227838`, seed 1, the stage-A recipe
  (`E33N_EXPERIMENT=E33t E33N_FLOW=1 E33N_SEED=1 E33N_RULER_LENGTHS=4096 E33N_MICRO_TRUNK=4 E33N_MICRO_DECAY=1
  E33N_CORE_BATCH=8 bash infra/aws/jobs/e33n_run.sh ARM s30`, runs in `Cache/e33t_runs/`, W&B online):
  `catalogue_reuse_n4` on Odra GPUs 0–1 (`E33N_GPUS=2`), `catalogue_reuse_p512` on Polonez GPUs 0–1 and
  `catalogue_reuse_n16` on Polonez GPU 2 (both `E33N_BURST_H=5.5 E33N_COOL_MIN=20`, the heat rule; Odra GPU 2 and
  Polonez GPU 3 idle on purpose). Same 64 rows per optimizer step (4 × 8 × 2 GPUs, 4 × 16 × 1). p512's first start
  lost rank 1 to a SIGSEGV one second in, while both Polonez jobs were syncing the venv after the pull; relaunched
  20:24 UTC, fine. Expected (stage-0 speed, 23.6k tokens/s per 3090 at 4k): ~12 h on two GPUs, ~24 h on one, so
  n4 and p512 about 09:00–10:00 UTC on 2026-11-10 and n16 about 21:00 UTC.
- **Stage C: long documents.** The winner with the two-level search, its choice stored as a short list (a gather
  read instead of the [S × G] table), trained at 16k plus a short 64k stage; evaluated at 128k, 1M and beyond.
  Two runs, about $80. **Design proposed 2026-11-08 (plan §10, awaiting approval):** the list read for evaluation
  only (the table is 275 GB at 1M), the 64k stage for the stage-B winner, the LSA port and the window control alike,
  two levels first as an eval switch on the flat winner (a two-level model only if the switch loses), about $30–50.
  Only one validation document reaches 1M tokens, so 1M is judged mainly by the planted password.
  **Approved by the author 2026-11-11 (before 06:45 UTC, "Yes, start it") with the stage-B winner `catalogue_reuse_n4`;**
  run on the home servers ($0 on AWS), see Result.
- **Stage D (if A–C match LSA-native at 30M, with the author's approval):** 100M (`m100`: 3B + 0.4B tokens) — the
  winner, LSA-native and the two controls, about 20–25 hours each on the 4 × L40S node, $500–600 in all.
  **Moved ahead of stage C and joined to E37's 100M full run (author, 2026-11-10 ~05:37 UTC: "Yes, add it"):** the
  E37 session already trains window, dense, notes-open, the LSA port and E37's concept memory on the full m100
  protocol (3.0B tokens at 4k, then 0.4B at 16k; E37 stage B, `infra/aws/jobs/e37_full.sh`), so only the catalogue
  model is added, on E37's single 4 × L40S spot slot after e37a, the original LSA (if no H100 takes it) and the LSA
  port: start about 14:00–18:00 UTC Nov 11, or about 01:30 UTC Nov 11 if an H100 takes the original LSA, earlier if
  the G spot quota rises to 96. The E37 session launches it in its queue, seed 42 like its other 100M models:
  `MC_SPOT=1 bash infra/aws/launch.sh e37-b-l40s-cat 30 "infra/aws/jobs/e37_full.sh 42 <arm>" e37-catalogue
  g6e.12xlarge`. The arm is the stage-B winner, `catalogue_reuse_n4` (named 2026-11-10 22:20 UTC, see Result).
  Queued by the E37 session about 22:25 UTC Nov 10 on node `e37-b-l40s-catalogue-reuse-n4` (30 h cap), results to
  `s3://mrcogito-train-093337158062/runs/e37/e33n_m100_catalogue_reuse_n4_tok3.0e9_seed42/`. Size at m100: 100.6M parameters without
  the word tables, 166.4M in total, the same as window and notes-open (the LSA port 96.5M / 162.4M), so 3.4B tokens
  is 20.4 per parameter, about compute-optimal. Code: branch `e37-catalogue` (a986d08) = this branch + `arch-10m`
  (3079db4), conflicts only side-by-side additions, tests 124 pass. Memory check at m100 on Odra card 2 (24 GB),
  05:41–05:56 UTC, `infra/aws/jobs/speed_check.sh`, peak by nvidia-smi: catalogue_reuse_n4 trunk at 4k 10.1 GB at
  micro 1 and 16.8 GB at micro 2 (about 6.7 GB per row), so the node's micro 4 needs about 30 GB of the L40S's
  48 GB; decay at 16k, micro 1 (the node's setting), 22.7 GB; catalogue_reuse_p512 trunk at micro 2 16.7 GB (n16
  has fewer notes than n4). Speed on one 3090: 12.3k tokens/s at 4k (micro 2), 7.7k at 16k. Estimated 16–19 h on
  4 × L40S (within the job's 30 h cap): about $80–140 on spot, $190 on demand, on the $2,000 goal.
- **Budget:** about $800–900 for A–D on top of stage 1 (≈ $220 in all), inside the $2,000 AWS goal.
- **Launch (stage A, AWS):** `MC_SPOT=1 bash infra/aws/launch.sh e33t-a 20 "infra/aws/jobs/e33p_s1_multi.sh 1
  catalogue catalogue_2l" e33n-lsa-testbed g6e.12xlarge` with `E33N_EXPERIMENT=E33t` (new arms in
  `scripts/run_e33n.py`).
- **New foundation code:** in `nn/exact_read.py` and the closed-read path of `nn/perceiver_ar_lm.py`:
  `lm_note_read` (`gist` | `index`: summaries in the read or not), `lm_filing` (`argmax` | `balanced`, Sinkhorn
  iterations and cap), `lm_open_score` (`sum_softmax` | `max_cos`), `lm_open_fill` (`notes` | `tokens`),
  `lm_open_new_per_pass` (bool), `lm_open_levels` (1 | 2, pages kept), `lm_sinks` (count of learned empty slots),
  `lm_index_loss` (weight, sampled queries and negatives) through `nn/loss_manager.py`; non-overlapping pages are the
  existing `lm_stride = lm_window`. Defaults reproduce E33p bit for bit (tests). Stage C adds the gather read.

## Risks
- **Pass exclusion may hurt** where rereading the same tokens helps; the eval switch (exclusion off) measures it
  without a retrain, and the instrumentation shows distinct tokens per query over the passes. Seen 2026-11-07 on
  planted passwords (stage A follow-up above); the switch alone did not recover it, so `catalogue_reuse` retrains.
- **The index loss may teach the scores to follow the read without improving recall** (a self-fulfilling loop:
  the read only sees what was chosen). The 32 sampled unchosen groups per query are there to break it; K2 watches it.
- **Sinkhorn balancing could split a coherent group** (a long name spread over two notes); the cap of 16 and the
  probe's recall show whether this costs anything.
- **Learned empty slots** are a different sink than LSA's first tokens; if attention collapses onto them (> 50 % of
  the mass at far positions) the read is idle: watched in `read/*/mass_sinks`.
- **Two levels can drop a needed page** before its notes are scored; page recall is logged, and S4 decides.

## Result
- **Stage A + follow-up, scored 2026-11-08 ~16:50 UTC against the same-seed controls** (AWS seed 1, 4 × L40S;
  share of G on positions ≥ 2k, token-weighted over sources; 128k: loss, lower is better, dense is worse than the
  window there). Runs: catalogue 2026-11-07 11:24–15:50 UTC, catalogue_2l 15:51–20:49, window s1 2026-11-07 20:49 –
  11-08 01:03, LSA port s1 01:03–05:05 (evals to 06:12), catalogue_reuse 07:22–12:03 (probe to 12:04); results in
  `s3://…/runs/e33t/` and `runs/e33p/`.

  | model | 4k | 16k | 32k | 64k | 128k loss |
  |---|---|---|---|---|---|
  | window | 0 | 0 | 0 | 0 | 2.2765 |
  | dense | 100 | 100 | 100 | 100 | 2.3209 |
  | LSA port (`lsa`) | 98.2 | 95.3 | 99.3 | 153.3 | 2.1441 |
  | notes-open | 42.8 | 37.4 | 30.1 | 23.4 | — |
  | catalogue | 70.0 | 71.1 | 72.3 | 115.2 | 2.1738 |
  | catalogue_2l | 67.4 | 67.9 | 67.0 | 99.5 | 2.1933 |
  | **catalogue_reuse** | **96.9** | **91.4** | **94.0** | **151.6** | **2.1395** |

- **Main arm (catalogue):** S1 missed (71.1 < 80); S2 passes; K1 not hit. **S4:** the two-level arm is 3.2 points
  behind at 16k (holds) and 15.7 behind at 64k (fails).
- **catalogue_reuse (pass exclusion off in training, nothing else changed):** S1 passes (91.4); S2 passes; S5 0.91 ×
  the LSA port's 16k speed (decay 39.4 vs 35.9 min; passes narrowly); the whole run 1.11 × the LSA port's wall time
  (K4 passes); S6 passes (16k loss on positions < 512: 2.6082 vs the window's 2.6095; core mean 40.78 vs 40.83).
  S3 not measured in its spec form (the flow probe's dense top-32 recall was not run on it).
- **RULER at 4k (mean of tasks):** catalogue_reuse 14.1 (single needle 94 %), catalogue 6.0, catalogue_2l 1.2, LSA
  port 15.2, dense 16.0, notes-open 5.0, window 4.1. Core benchmark means all 40.1–41.0 (no harm anywhere).
- **Planted-password probe on catalogue_reuse:** pass 1 now rarely opens the key (23 % at 4k; catalogue 94 %), passes
  2–4 open it 95–97 % at 4k (63–67 % at 16k) and the last pass puts 0.38 of its read on it (LSA port s42 0.37, dense
  s1 0.41): the passes return to the key the way the LSA port's do. First answer token right 15 % at 4k (6/40) and
  10 % at 16k (2/20), below the LSA port's 25 % / 30 % (seed 42) and dense's 30 % / 15 %; with 40 and 20 rows the
  difference is within sampling noise (about ±7 points at 4k).
- **Verdict:** the per-pass exclusion was the main cause of the gap to LSA. Switching it off in training (same
  72,687,765 parameters) moves the catalogue read from 71 % to 91 % of G at 16k, level with the LSA port at 64k, and
  gives the best 128k loss of all models. It does not beat the LSA port at ≤ 32k (1–5 points behind) and trains
  about 10 % slower. catalogue_reuse is the stage-A winner for stages B–C.
- **Stage B (note and page sizes), home servers, seed 1, scored against the AWS seed-1 controls** (same measures as
  stage A; `catalogue_reuse` = 32 notes of 8 tokens per 256-token page is the stage-A winner it must beat):

  | model | 4k | 16k | 32k | 64k | 128k loss | RULER 4k | core mean |
  |---|---|---|---|---|---|---|---|
  | catalogue_reuse (32 notes, AWS) | 96.9 | 91.4 | 94.0 | 151.6 | 2.1395 | 14.1 | 40.78 |
  | catalogue_reuse_n4 (64 notes, Odra) | 91.2 | 87.9 | 90.8 | 146.8 | 2.1463 | 10.9 | 38.90 |
  | catalogue_reuse_p512 (512-token pages, 64 notes, Polonez) | 89.0 | 84.0 | 85.8 | 133.2 | 2.1608 | 15.5 | 39.31 |
  | catalogue_reuse_n16 (16 notes, Polonez) | 79.0 | 73.0 | 70.7 | 106.2 | 2.1855 | 8.9 | 39.91 |
  | catalogue_reuse (32 notes, Odra retrain, same-hardware reference) | 93.0 | 86.2 | 89.4 | 143.7 | 2.1586 | 9.9 | 39.84 |

  - **n4** (Odra GPUs 0–1): trunk 2026-11-09 20:23 → 11-10 06:07 UTC (9.7 h), decay 06:07 → 08:14, evals to 08:44.
    Below the 32-note winner at every length (3.5 points at 16k), on RULER and on core; S1 and S2 pass, S6 misses
    twice: core mean 1.9 below the window's 40.83 (lowest of any model so far; arc_easy 38.8 and sciq 62.1 against
    40.8–44.6 and 63.5–70.4 elsewhere; one seed on different hardware), and 16k loss on positions < 512 2.6204 vs
    the window's 2.6095 (0.011; dense 2.6250 and the LSA port 2.6258 miss it as well). Its 128k pass ran out of
    memory when core shared card 0 (fix in `e33n_run.sh`, two-GPU runs now keep the long-document loss on its own
    card); rerun alone on Odra GPU 0 08:55–09:04 UTC (`eval_longdoc_128k.json`, hansard has no 128k document).
  - **p512** (Polonez GPUs 0–1, bursts): trunk 2026-11-09 20:24 → 11-10 06:48 UTC (one 20 min cooldown at 01:54),
    decay 06:49 → 09:05, evals to 09:34 (128k fit this time: no out-of-memory). Below the 32-note winner at every
    length (7.4 points at 16k) and on the 128k loss; S1 passes (84.0), S2 passes; RULER at 4k 15.5, the best of
    the reading models (the LSA port 15.2) and just below dense's 16.0, single needle 88 %; S6 misses on core (1.5 below the window) and passes
    on the short positions (2.6169, 0.007 above the window).
  - **n16** (16 notes of 16 tokens per 256-token page; Polonez GPU 2 alone, bursts): trunk 2026-11-09 20:23 →
    11-10 15:34 UTC (three 20 min cooldowns), decay 15:34 → 19:29, evals one after another to 20:12 UTC (no
    out-of-memory). Far below the 32-note winner: 18.4 points at 16k, 2.5 × the noise floor, and the worst
    reading model at every length; **S1 misses** (73.0 < 80), S2 passes (106.2 at 64k); RULER at 4k 8.9, single
    needle 46 % (the 32-note winner 94 %); S6 passes (core 0.9 below the window, short positions 2.6144, 0.005 above
    it). Halving the notes to 16 costs 18 points at 16k, doubling them to 64 costs 3.5: 32 notes of 8 tokens is the
    best of the three note sizes.
  - **Noise floor for these gaps:** the controls trained twice (AWS seed 1 vs home seed 42, so seed and hardware both
    change) move by 0.0005 / 0.0114 / 0.0031 / 0.0160 nats at 16k (window / dense / LSA port / notes-open), up to 7.6
    points of G. n4's 3.5-point gap sits inside it and p512's 7.4 at its edge. **Same-hardware reference** (free,
    Odra idle): `catalogue_reuse` itself retrained at home, seed 1, the same command as n4 apart from the note count
    (`e33n_s30_catalogue_reuse_seed1`, Odra GPUs 0–1, tmux `e33t-b-n8home`), started 2026-11-10 09:46 UTC, expected
    about 22:00 UTC with n16. The winner is scored against both 32-note runs.
  - **Same-hardware reference result:** trunk 09:46 → 19:22 UTC (9.6 h, n4 9.7 h), decay 19:22 → 21:25, evals to
    21:59 UTC. The same model, seed and command as the AWS run scores 5.2 points lower at 16k on Odra (86.2 vs 91.4;
    16k loss on positions ≥ 2k 2.3666 vs 2.3555 nats), RULER at 4k 9.9 vs 14.1 (single needle 81 % vs 94 %), core
    39.84 vs 40.78. The home runs lose the same tasks (sciq 62–65 vs 68–70, arc_easy 39–41 vs 42–45 on AWS), and
    the controls' reading models (dense, LSA port) are also worse at home, so the hardware gap looks systematic,
    not seed noise. On the same machine, n4 against it: 91.2 / 87.9 / 90.8 / 146.8 vs 93.0 / 86.2 / 89.4 / 143.7
    (n4 behind at 4k, ahead at 16k, 32k and 64k), 128k loss 2.1463 vs 2.1586, RULER 10.9 vs 9.9 (per-task nearly
    identical: single needle 80 vs 81 %), core 38.90 vs 39.84, short positions 2.6204 vs 2.6218. Against the home
    seed-42 controls the order is the same (16k: n4 92.9, 32-note retrain 91.1, p512 88.8, n16 77.1, so n16 misses
    S1 on either set of controls). S6's short-position check fails for every model trained at home, the 32-note
    retrain included (0.012 above the AWS window, 0.017 above the home window), so it does not separate the arms.
- **Stage-B verdict (2026-11-10 ~22:15 UTC): `catalogue_reuse_n4` wins by the rule written before the runs** (best
  16k share with the 64k share not below it, ties broken by RULER at 4k), on the same-hardware comparison the
  reference was trained for: 87.9 vs 86.2 at 16k, RULER 10.9 vs 9.9. The margin (1.7 points, 0.0036 nats) is a
  third of the hardware gap between two runs of the same model, so 64 and 32 notes are a near-tie; 16 notes is
  clearly worse (S1 misses), 512-token pages sit 2–4 points behind with a better needle score (RULER 15.5, run on
  Polonez, a third machine). Named to the E37 session at 22:20 UTC for stage D's 100M full run
  (`infra/aws/jobs/e37_full.sh 42 catalogue_reuse_n4`, branch `e37-catalogue` a986d08, where the memory check at
  m100 was already done on n4). Stage B spent $0 on AWS (all four runs on Odra and Polonez).
- **Stage C (long documents), home servers, $0 on AWS.** Approved by the author 2026-11-11 with `catalogue_reuse_n4`.
  - **Code** (branch e33n-lsa-testbed, fb91fa8 → 7c4ca8e): the `long` phase (`E33N_LONG=1`: rows of 65,536, 0.1B
    tokens, 381 steps of 4 rows, 1e-4 → 1e-5, manifest seed 2042 and trainer seed 1 for every arm, so all three see the
    same rows; train rows at 64k: arxiv 693 is the smallest source); layer recomputation for the notebook models (a
    recomputed block replays the pass state its forward saw: the first try failed with CheckpointError, 85 vs 68 saved
    tensors, because the loss had cleared the index-loss list; tests: same loss and gradients with and without it); the
    list read (`ChoiceList` + `attend_list`, `open_impl("list")`, evaluation only); evaluation memory at 1M (feed-forward
    in pieces without gradient, one loop state kept); `evaluation/long_passkey.py`, `--two_level` and
    `infra/aws/jobs/e33t_c_evals.sh`.
  - **Memory at 64k, micro 1, Odra 3090** (2026-11-11 06:50–07:01 UTC): without recomputation the reuse catalogue and the
    LSA port run out of memory (24 GB); with it 15.6 GB and 6.6 GB.
  - **List read = table read** on trained 16k models, fp32, two held-out 16k documents (`list_read_check.py`, 07:02–07:05
    UTC): mean loss equal to 6–7 digits (catalogue_reuse_n4 1.42260 / 1.62415, LSA port 1.40922 / 1.58024); mean
    per-token difference 1e-6–5e-6, largest 0.002 (catalogue) and 0.022 (LSA port, one token): near-tied choices that
    round differently after four loop passes. CPU tests at test scale: logits equal to 1e-5, opened tokens equal query
    by query. Cost of the list read (bf16, one row): 128k 11 s / 3.8 GB, 256k 30 s / 7.5 GB, 512k 85 s / 15.0 GB, 1M
    276 s / 19.7 GB (catalogue_reuse_n4; the LSA port 11 s / 3.4 GB, 22 s / 6.6 GB, 55 s / 13.1 GB at 128k–512k). 1M
    first ran out of memory on 2.9 GB feed-forward activations, fixed at 07:15 UTC.
  - **64k phase launched 2026-11-11 07:02 UTC**, one GPU per model, layer recomputation, evals after it as
    `eval_*_long.json` (4k–128k loss, RULER 4k, core): `catalogue_reuse_n4` (Odra GPU 0, 3,220 tokens/s, due about
    15:40 UTC), the LSA port (home seed 42, `e33p_runs/e33n_s30_lsa`, Odra GPU 1, 6,850 tokens/s, due about 11:10 UTC)
    and the window control (home seed 42, `e33p_runs/e33n_s30_window`, Polonez GPU 0, 25,600 tokens/s, due about
    08:10 UTC). Each then runs `e33t_c_evals.sh` (256k and 1M loss, password at 128k and 1M; the catalogue also with
    two levels, 16 pages) on its own GPU. The same evals on the 16k-only models started 07:20 UTC (catalogue and then the
    LSA port on Odra GPU 2, the window control on Polonez GPU 1).
  - **Chain bug** (found 11:10 UTC): `chain_c.sh` put the optional `${TL:+E33T_TWO_LEVEL=..}` word before the other
    `VAR=..` prefixes, so bash took the next prefix as the command name (exit 127) and the window control's 256k–1M
    evals did not start at 08:26 UTC. Fixed (settings exported first); the waiting chains were stopped by PID and
    restarted 11:11 UTC. The 64-notes model's evals were later moved off its training GPU (`6285b87`,
    `E33T_SKIP_FLAT`: flat search on Odra GPU 1, two levels on GPU 2, standard evals on GPU 0), about 3 h earlier.
  - **Timeline (2026-11-11 UTC):** 64k phases 07:02 → window 08:10 (Polonez, 25,600 tokens/s), LSA port 11:07 (Odra,
    6,850), catalogue_reuse_n4 15:45 (Odra, 3,220: the slowest, about 31 % GPU use). Last eval 17:37 UTC. Odra's
    hottest card 73 °C. Results: `docs/2_Experiments_Registry/results/evaluations/E33t_C_*` and the NAS
    (`/nas/ml_data/mrcogito/results/reports/<host>/MrCogito-lsa/`).
  - **Gain over the window control trained the same way** (nats, loss on positions ≥ 2k, token-weighted over sources;
    4k–128k with the table read and 8 documents per source (37 at 64k, 21 at 128k), 256k four documents (1.82M scored
    tokens), 1M one usgpo document (1.05M scored tokens, an illustration); home controls: window and LSA port seed 42,
    catalogue_reuse_n4 the Odra stage-B run, seed 1). Window losses: 16k-trained 2.6506 / 2.5480 / 2.6742 / 2.6007 /
    2.2784 / 1.7891 / 0.9031, after the 64k stage 2.6365 / 2.5314 / 2.6584 / 2.5824 / 2.2590 / 1.7675 / 0.8793.

    | model | 4k | 16k | 32k | 64k | 128k | 256k | 1M |
    |---|---|---|---|---|---|---|---|
    | LSA port, 16k-trained | 0.143 | 0.198 | 0.220 | 0.172 | 0.108 | 0.102 | 0.054 |
    | LSA port, after the 64k stage | 0.137 | 0.193 | 0.223 | 0.195 | **0.168** | 0.157 | 0.037 |
    | catalogue_reuse_n4, 16k-trained | 0.143 | 0.185 | 0.204 | 0.171 | 0.132 | 0.139 | 0.016 |
    | catalogue_reuse_n4, after the 64k stage | 0.137 | 0.181 | 0.206 | 0.180 | 0.162 | **0.178** | **0.089** |
    | n4, two levels (eval switch), 16k-trained | | | | | | 0.107 | 0.049 |
    | n4, two levels (eval switch), after the 64k stage | | | | | | 0.109 | 0.039 |
    | n4, notebook off, 16k-trained | −0.016 | −0.020 | −0.022 | −0.017 | −0.015 | −0.019 | −0.017 |
    | n4, notebook off, after the 64k stage | −0.025 | −0.029 | −0.032 | −0.027 | −0.027 | −0.033 | −0.035 |
    | LSA port, opened tokens off, after the 64k stage | | | | −0.045 | −0.047 | −0.058 | −0.051 |

  - **Planted password** (exact / first digit right, of 20 depths; `evaluation/long_passkey.py`, teacher-forced):

    | model | 128k, 16k-trained | 1M, 16k-trained | 128k, after 64k | 1M, after 64k |
    |---|---|---|---|---|
    | window control | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
    | LSA port | 1 / 2 | 0 / 1 | 2 / 3 (depths 12.5, 97.5 %) | **1 / 2** (depth 2.5 %) |
    | catalogue_reuse_n4 | 0 / 1 | 0 / 0 | 2 / 2 (depths 72.5, 82.5 %) | 0 / 0 |
    | n4, two levels | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
    | n4, notebook off | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

  - **No harm from the 64k stage:** core mean (7 tasks) window 40.34 → 40.40, LSA port 40.59 → 40.49,
    catalogue_reuse_n4 38.90 → 38.87; RULER at 4k (mean × 100) 2.9 → 3.3, 13.9 → 14.0, 10.9 → 11.2.
- **Stage-C verdict (2026-11-11 ~17:45 UTC):**
  - **The list read works:** equal to the table read to 6–7 digits, and it takes the 30M catalogue to 1M tokens on one
    3090 (276 s, 19.7 GB per row). The flat search is affordable at 1M, so the search cost is not yet the wall.
  - **The 64k stage helps where it should:** 0.1B tokens leave 4k–32k unchanged (±0.006) and lift the gain at 128k
    (catalogue 0.132 → 0.162, LSA port 0.108 → 0.168) and 256k (0.139 → 0.178, 0.102 → 0.157). The window control
    itself gains 0.014–0.024 from it (more at longer lengths), which the comparison removes.
  - **Catalogue vs the LSA port after the 64k stage:** level up to 128k (LSA port ahead by 0.006–0.017 at 16k–128k,
    about the 0.016-nat rerun noise at 16k), the catalogue ahead at 256k (0.021, four documents) and on the one 1M
    document (0.052). The past-128k lead is suggestive, not proven: four documents and one seed.
  - **The gain is the notebook's:** with it off the catalogue sits 0.015–0.035 below the window control at every length.
  - **Passwords stay rare past 128k** for every model (at most 2 of 20 at 128k, 1 of 20 at 1M, the LSA port's): at 30M
    and this data neither read retrieves a five-digit key reliably that far; the loss gain comes from the text, not
    from exact recall of one planted fact.
  - **Two levels lose as an eval switch** (0.109 vs 0.178 at 256k, 0.039 vs 0.089 at 1M, 0 vs 2 passwords at 128k).
    By plan §10.7 the next step would be a two-level training run; it is deferred because the flat search fits at 1M
    and the cost only matters past it (the 10M-token aim). Open for the author.
  - **Stage D** (catalogue_reuse_n4 at 100M on AWS with the E37 full protocol) runs since 2026-11-11 16:19 UTC.
- **Stage D, 100M on the E37 full protocol (done 2026-11-12 06:19 UTC; scored 07:40 UTC from the raw JSONs):**
  - **Run:** `catalogue_reuse_n4` at m100 (100.6M parameters without the word tables, 166.4M in total), seed 42,
    3.0B tokens at 4k then 0.4B at 16k, no 64k stage. Node `e37-b-l40s-catalogue-reuse-n4` (i-09f3edf49946aa5f6,
    4 × L40S spot): start 2026-11-11 16:20, trunk 16:21–02:47 (10.4 h; the LSA port 9.4 h, e37a 10.5 h on the same
    node type), decay 02:47–04:50 (2.0 h; the LSA port 1.7 h, so 0.82 × its 16k speed), evals 04:50–06:19 UTC; exit 0,
    no NaN; about 14 h, about $100 on spot; node terminated. Trunk eval loss 3.125 (LSA port 3.121), decay eval 2.274
    (LSA port 2.255, e37a 2.253). Results: `s3://mrcogito-train-093337158062/runs/e37/e33n_m100_catalogue_reuse_n4_tok3.0e9_seed42/`
    and `docs/2_Experiments_Registry/results/evaluations/E33t_D_catalogue_reuse_n4_m100/`.
  - **Comparison:** E37's 100M models trained the same way (seed 42, same node type): e37a (E37's concept memory),
    the LSA port and the original LSA. E37's method: loss on positions ≥ 2k, token-pooled over 5 sources (4 at 128k);
    core uses acc_norm where present. The full-budget window and dense controls still wait for H100s, so the shares of
    the window→dense gap (S1, S2) and S6's window comparison cannot be scored yet.

    | model | 4k | 16k | 32k | 64k | 128k | 16k, positions < 512 |
    |---|---|---|---|---|---|---|
    | catalogue_reuse_n4 | 2.342 | 2.189 | 2.275 | 2.270 | 2.058 | 2.400 |
    | LSA port | 2.322 | 2.169 | 2.254 | 2.245 | 1.994 | 2.384 |
    | e37a | 2.320 | 2.166 | 2.254 | 2.250 | 1.990 | 2.388 |
    | original LSA | 2.284 | 2.135 | 2.215 | 2.201 | 1.925 | 2.369 |
    | n4, notebook off | 2.496 | 2.386 | 2.489 | 2.433 | 2.123 | |
    | what n4's notebook carries | 0.155 | 0.197 | 0.214 | 0.163 | 0.065 | |

  - **RULER (mean of 13 tasks, 100 prompts each):** catalogue 0.231 at 4k / 0.183 at 16k; LSA port 0.255 / 0.209;
    e37a 0.185 / 0.136; original LSA 0.139 / 0.082. Per task (4k / 16k), catalogue vs the LSA port: single needle
    0.97 / 0.98 vs 0.96 / 0.95; single-2 (essay, number) 0.69 / 0.38 vs 0.88 / 0.84; single-3 (UUID) 0.43 / 0.41 vs
    0.55 / 0.27 (best of the four at 16k); multi-query 0.18 / 0.16 vs 0.05 / 0.04 (best of the four); multi-value
    0.18 / 0.14 vs 0.13 / 0.14; multi-key-1 0.24 / 0.18 vs 0.26 / 0.27; word frequency 0.00 / 0.00 vs 0.27 / 0.09;
    HotpotQA 0.12 / 0.08 vs 0.10 / 0.09; SQuAD 0.08 / 0.04 vs 0.03 / 0.01; the rest near zero for all.
  - **Core mean (7 tasks, acc_norm where present):** catalogue 41.6, LSA port 41.6, e37a 41.6, original LSA 42.0 — a tie.
  - **Criteria at 100M:** S5 misses (0.82 × the LSA port's 16k speed; 0.90 × at 4k). S6's core part holds against
    every model available (within 1 point); near text at 16k is 0.016 nats worse than the LSA port. S1, S2 wait for the
    full-budget window and dense controls.
  - **Later pass (E37's protocol, launched by the E37 session 2026-11-12 06:32 UTC):** two one-L40S spot nodes, branch
    `e37-catalogue-later` (= `e37-catalogue` a986d08 + E37's `e37_ruler_later.sh`, `e37_ruler_later_node.sh`,
    `evaluation/val_loss.py`; model code unchanged): `e37-later-cat-64k` RULER at 64k (about 12 h);
    `e37-later-cat-rest` validation loss (trunk final on FineWeb-Edu 4k; decay final on FineWeb-Edu 4k and
    LongDocs-lite 16k), then RULER at 8k and 32k (about 8 h). Results to the run's S3 folder (`eval_val.json`,
    `later/ruler_<L>_<task>.json`, merged `eval_ruler_later.json`). Both ran slower than estimated (about 72 min per
    task at 32k and 64k); `cat-rest` was reclaimed at 17:26 UTC with 7 of 13 32k tasks saved and relaunched in us-west-2
    at 19:03 UTC; both due about 02:30–03:00 UTC Nov 13.
    **Result (finished 2026-11-13 02:27 UTC; files read and checked here about 08:15 UTC; same files and limits for all four
    models, 100 documents a task at 8k and 32k, the first 25 at 64k):** RULER mean of 13 tasks at 8k / 32k / 64k:
    n4 0.197 / 0.125 / 0.101; LSA port 0.241 / 0.174 / 0.151; E37a 0.187 / 0.106 / 0.088; original LSA 0.119 /
    0.040 / 0.028. n4 is second of four at every length from 4k to 64k, behind the LSA port by 0.024–0.050; its score as a
    share of the LSA port's falls from 0.82–0.91 at 4k–16k to 0.72 / 0.67 at 32k / 64k. Single needle (niah_single_1) 0.98 / 0.98 / 0.96
    (LSA port 0.96 / 0.96 / 0.96; E37a 0.95 / 0.96 / 0.92; original 0.09 / 0.03 / 0.04). The essay needles fade past 16k,
    as in E37a: number (single_2) 0.53 / 0.08 / 0.04 (LSA port 0.92 / 0.55 / 0.28), UUID (single_3) 0.32 / 0.07 / 0.00
    (LSA port 0.53 / 0.20 / 0.04). Multi-key-1 0.20 / 0.19 / 0.12 (LSA port 0.22 / 0.31 / 0.20). Validation loss
    (`evaluation/val_loss.py`, the trainer's eval rows): FineWeb-Edu 4k after the trunk 3.112 (LSA port 3.111, E37a
    3.105), after the decay 3.200 (3.193, 3.187), LongDocs-lite 16k after the decay 2.250 (2.232, 2.228); original LSA
    3.093 / 3.162 / 2.193 (its own trainer). **Reading:** the catalogue keeps the single needle to 64k but loses the
    harder needles at the same lengths as E37a. Whether a 64k training stage helps at 100M (at 30M it put n4 ahead
    past 128k) is still open and not run.
  - **The predecessor at the same budget (E33p notes_open, run by the E37 session, 2026-11-12 06:39–21:43 UTC, 4 × L40S
    spot, about $110; scored here from `runs/e37/e33n_m100_notes_open_tok3.0e9_seed42/` with the same pooling):** loss
    past 2k 2.379 / 2.282 / 2.393 / 2.367 / 2.104 at 4k / 16k / 32k / 64k / 128k, 0.04–0.12 nats behind n4
    (0.09–0.12 from 16k to 64k); notebook off 2.497 / 2.388 / 2.488 / 2.432 / 2.120, so its notebook carries 0.118 /
    0.106 / 0.096 / 0.066 / 0.016 (n4: 0.155 / 0.197 / 0.214 / 0.163 / 0.065); near text (< 512 at 16k) 2.370, the best
    of the five (n4 2.400); RULER 0.066 / 0.018 at 4k / 16k (its 1.0B screen 0.116 / 0.072; n4 0.231 / 0.183); core 41.1
    (n4 41.6). **The catalogue read is a clear step up from the read it replaced at 100M**: 1.3–2.5× the
    notebook's carry up to 64k and 4× at 128k, and 3.5–10× the needle-puzzle mean.
- **Stage-D verdict (2026-11-12 ~07:45 UTC):**
  - **At 100M with the standard recipe the catalogue does not match the LSA port:** 0.02 nats behind up to 64k and
    0.065 behind at 128k. At 4k–64k the gap is close to its 0.016-nat gap on near text, so much of it is the model's
    general quality, not the read; at 128k the near-text losses are equal (1.709 vs 1.709) and the gap is the read's.
  - **The read fades at 128k:** the notebook carries 0.15–0.21 nats up to 64k but 0.065 at 128k. At 30M the 16k-trained
    catalogue kept most of its gain at 128k (0.132 nats over the window vs 0.171 at 64k), and the 64k stage is what put
    it ahead past 128k; stage D had no 64k stage.
  - **Needle puzzles:** close to the LSA port with a different profile (better on several-key and UUID needles, worse
    on the essay number and word frequency), above e37a and far above the original LSA. General knowledge: a tie.
  - **Open for the author:** the fair test of the 30M finding is the same 0.1B-token 64k stage at 100M for the
    catalogue and the LSA port (and the window control once it exists). Without it, the LSA port stays the stronger
    read at 100M.
- **Stage D+, the 64k stage at 100M (approved by the author 2026-11-13: "Yes, run it", to "Run the 64k training stage
  at 100M for n4 and the LSA port (about $50–80)?"):** `catalogue_reuse_n4` and the LSA port only (the approval does
  not cover the window control). From each stage-D decay final on S3, `E33N_LONG=1 E33N_LONG_TOKENS=1e8
  E33N_GRAD_CKPT=1 E33N_GPUS=4 E33N_CKPT_S3=1` (knob added in a729ca7): LongDocs-lite rows of 65,536, manifest seed
  2042 and trainer seed 1 for both, micro 1 × accumulation 2 × 4 GPUs = 8 rows a step, 191 steps, learning rate
  6.4e-5 → 6.4e-6 (1−sqrt), the 30M stage's tokens. Then the stage-D tests with a `_long` suffix (long-document loss
  4k–128k with the notebook on and off, RULER 13 tasks at 4k–32k, core); no 64k RULER pass unless asked. Memory at
  100M is not yet checked (30M at 64k with recomputation: n4 15.6 GB, the LSA port 6.6 GB). Handed to the E37 session
  2026-11-13 10:47 UTC, on a 4 × L40S spot node per model, hard stop 8 h; estimate about 1–1.5 h of training and
  about 1.5 h of tests per model, about $50–80 in total.
  **Result (run by the E37 session; read here 2026-11-14 ~07:00 UTC from its E37 spec entries, b10f0012 and 037ee1fc
  on `arch-10m`; files `long/`, `long_evals/`, `eval_*_long.json` in each run's S3 folder):**
  - *Runs:* n4 on `e37-long-catalogue-reuse-n4` (i-07e15db9d75d8c8ac): training 19:16–20:54, tests 20:54–21:37 UTC
    Nov 13, about 2.4 h, about $18; ~22 GB of 46 GB a card, ~32 s a step. LSA port on `e37-long-lsa`
    (i-0806f36773e567902): training 21:57–22:58, tests 22:58–23:33 UTC, about 1.7 h, about $13; ~12 GB a card, ~19 s
    a step. So at 64k n4 trains at about 0.59 × the LSA port's speed (16k: 0.82 ×). Both nodes shut down by themselves.
  - *Loss on positions ≥ 2k* (5 sources, 4 at 128k), stage D → D+:

    | model | 4k | 16k | 32k | 64k | 128k | positions < 512 at 16k |
    |---|---|---|---|---|---|---|
    | n4 | 2.342 → 2.339 | 2.189 → 2.187 | 2.275 → 2.264 | 2.270 → **2.232** | 2.058 → **1.943** | 2.400 → 2.399 |
    | LSA port | 2.322 → 2.320 | 2.169 → 2.166 | 2.254 → 2.244 | 2.245 → **2.212** | 1.994 → **1.923** | 2.384 → 2.382 |
    | n4 − LSA port | +0.019 | +0.021 | +0.020 | +0.020 | +0.020 | +0.017 |

    n4's notebook off is unchanged by the stage (−0.002 to −0.005), so its gain is the catalogue read's. E37a 1.990
    and the original LSA 1.925 at 128k (stage B, no 64k stage).
  - *RULER* (mean of 13 tasks, 4k / 8k / 16k / 32k): n4 0.231 / 0.197 / 0.183 / 0.125 → 0.224 / 0.191 / 0.180 /
    **0.158** (32k essay number 0.08 → 0.19, UUID 0.07 → 0.21; single needle 0.99 at every length); LSA port 0.255 /
    0.241 / 0.209 / 0.174 → 0.247 / 0.233 / 0.203 / 0.172 (essay number 0.88 / 0.92 / 0.84 / 0.55 → 0.81 / 0.82 / 0.70 /
    0.38; UUID at 32k 0.20 → 0.28). Core: n4 41.6 → 41.6, LSA port 41.6 → 41.4. RULER here uses the cached decoding;
    on the same model at 4k it matches the old loop on 10 of 13 tasks for n4 and 7 of 13 for the LSA port (the rest
    differ by one or two documents, bf16 near-ties), so mean differences under ~0.005 are not results.
  - **Reading:** at 100M the 64k stage does what it did at 30M on long text: nothing up to 16k, −0.04 nats at 64k and
    −0.12 at 128k for n4 (−0.03 / −0.07 for the LSA port). n4's 128k gap to the LSA port shrinks from 0.064 to 0.020
    nats, the same 0.02 it trails by at every other length and on nearby text, so what remains is the model's general
    quality, not the read. Unlike 30M, n4 does not pull ahead: the LSA port stays 0.02 nats better at every length and
    better on the needle puzzles (0.172 vs 0.158 at 32k), though it gives up part of its essay-number needle.
- **The full-budget controls and the criteria at 100M** (window and dense trained by the E37 session on the same
  protocol, seed 42, 4 × L40S spot: dense 22:14–08:03 UTC Nov 12–13, window 08:34–18:56 UTC Nov 13; shares from its
  `full_gap.py` with the same pooling, E37 spec fb95eb14):
  - Loss on positions ≥ 2k at 4k / 16k / 32k / 64k / 128k: window 2.461 / 2.351 / 2.449 / 2.397 / 2.086, dense 2.319 /
    2.163 / 2.256 / 2.332 / 2.194 (dense reads with no position code and falls below the window past 32k, so it is no
    ceiling there). Near text (< 512 at 16k): window 2.367, dense 2.394. Core: window 42.1, dense 42.2. RULER at 4k /
    16k: window 0.047 / 0.011, dense 0.210 / 0.175.
  - Share of the window → dense gap at 4k / 16k / 32k: **n4 84 / 86 / 90 %**, LSA port 98 / 97 / 101, E37a 99 / 99 /
    101, notes-open 58 / 37 / 29, original LSA 125 / 115 / 121. Gain over the window at 64k / 128k (nats): n4 0.127 /
    0.027 (after stage D+ 0.165 / 0.143, against a window control without a 64k stage; at 30M the window gained
    0.014–0.024 from it), LSA port 0.153 / 0.092 (after D+ 0.185 / 0.163), E37a 0.147 / 0.096.
  - **S1 passes** (86 % ≥ 80 %), 11–13 points below the LSA port and E37a. **S2:** as a share undefined past 32k (no
    ceiling); in nats n4's gain over the window is 0.162 / 0.174 / 0.127 / 0.027 at 16k / 32k / 64k / 128k: it holds
    to 32k and fades past it on 16k training, and holds to 128k after the 64k stage (as at 30M). **S5 misses** (0.82 ×
    the LSA port's speed at 16k, about 0.59 × at 64k). **S6:** core holds (41.6 vs the window's 42.1, within 1 point);
    nearby text misses (+0.033 nats against the 0.01 bar; the LSA port +0.017 and E37a +0.021 miss it too); the 4k
    held-out part is not scored (no matching file for the window). S3 and S4 were not measured at 100M.
- **Needle puzzles with cached decoding (E37 session, branch `e37-needles`; stage-D weights; all 100 documents a
  task; `eval_ruler_needles.json` in each run folder, read here 2026-11-14 ~06:30 UTC):** RULER mean at 64k: LSA port
  0.139, **n4 0.099**, E37a 0.083, dense 0.073 (the earlier first-25-documents pass: 0.151 / 0.101 / 0.088 / —).
  At 128k: LSA port 0.128, dense 0.083; n4, E37a and the original LSA still running. n4 at 64k: single needle 0.97,
  essay number 0.03, UUID 0.01, multi-key-1 0.12; at 4k cached 0.228 (old loop 0.231).
- **Capability checks, core package (run by the capability session; finished 2026-11-14 02:29:43 UTC; full verdict,
  per-task table and ledger on branch `e33t-capability`, commit 24b163c3, not yet merged here):** the no-harm check
  against the champion E31 fails — 9 cells kept, 50 lost on 10 tasks, 14 not run — and both targets are missed
  (`C1.lookup-16k` not run after its 2k stage stayed at chance; `C4.chain4-1k` 23–25 % first letter, chance 25 %, the
  champion 68 %). The read learns tasks whose book fits its 256-token raw window (15 of 15 runs at 128–256 tokens),
  seldom learns to open a page from random weights (24 of 56 first stages from 512 tokens up), and what it learned at
  1k does not carry to 2k (2-hop path 98 → 7–64 %, 3-hop 97–99 → 3–19 %; the champion holds 79–95 % to 16k). The
  startup issue (`catalogue-startup`) stays open, so these cells read "lost", never "cannot".
- **Overall verdict (2026-11-14 ~07:00 UTC):** the catalogue read is a clear step up from the read it replaced (at
  100M 86 % of the gap against notes-open's 37 %, needle puzzles 3.5–10 × higher) and at 30M it matched the LSA
  port and led it past 128k. At 100M it passes S1 but stays 0.02 nats behind the LSA port (and E37a) at every length,
  also after the 64k stage, trails the LSA port on the needle puzzles at every length, trains slower (S5 misses), and
  fails the capability no-harm check. No E33t arm beats the LSA port at 100M, so the LSA port and E37a remain the
  leads; closing this spec (and moving it to its lifecycle folder) is open for the author.
