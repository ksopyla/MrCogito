# E38 — Two 100M models on a modern data mix: longer budget, SmolLM2 tokenizer, far-context data

- **Status:** leaderboard run 1 of 4 launched (2026-11-15, 05:24 UTC): the 150M concept-memory model
  (`e33n_m120t_e37a_muon0.01_tok8.0e9_seed42`, Muon) on g6e.12xlarge spot `i-0712f5801fcca98ee`, us-east-1b, max 48 h.
  **Muon test result (finished 04:09 UTC Nov 15, $15):** held-out loss at the same steps, AdamW → Muon:
  step 500 2.905 → 2.742, 1,000 2.541 → 2.450, 1,500 2.403 → 2.342, 2,000 2.323 → 2.283. At step 2,000 Muon is
  0.040 lower, above the 0.03 pass mark, so all four leaderboard runs use Muon (`--muon_peak 0.01`). The gap narrowed at
  every check (0.164, 0.091, 0.061, 0.040), so the gain may shrink further over 11.5B tokens; the AdamW v1
  checkpoint-4000 (2.168) stays as a reference point. The slot came free at 05:13 UTC when the E37 session parked its
  e37a seed-7 run at checkpoint-4004 (it resumes after the fourth leaderboard run). Order on the one slot: concept
  memory, LSA port, catalogue, fair full-attention control, each starting after the previous one's tests are in S3.
  Earlier (2026-11-14, 21:36 UTC, author's "Yes, stop it"): the v1 concept-memory long run
  (`e33n_m100_e37a_tok8.0e9_seed42`, 162M total, over the board's 150M cap) was stopped at about step 4,300 of 15,259;
  the four 150M runs (`m120t` on `m21mix_v2`, seed 42) are queued by a launcher on the laptop
  (`Cache/keepers/e38-board-chain.sh` and `e38_pick_opt.py`, not in git) that applied the pass mark and started four
  keepers. Per model about 37 h and $295 (spot about $8.00/h): stage 1 at 105.6k tokens/s (measured 06:25 UTC Nov 15
  on this run: 13% below the v1 AdamW 100M run's 121k; cards at 21.5 of 48 GB), stage 2 about 71k and stage 3 about
  59k (estimates scaled by the same 13%), tests about 1.5 h. Expected done: concept memory Mon 16 Nov ~18:30 UTC,
  LSA port Wed 18 ~07:30, catalogue Thu 19 ~20:30, fair full-attention control Sat 21 ~09:30. W&B online, run name
  "150M concept memory, Muon, stage 1 of 3". Two lanes since 06:54 UTC Nov 15 (author: "Could you try to spin another
  machine for next schedule model, do we have capacity and quota? Maybe next L40 or H100?"): a second g6e.12xlarge
  does not fit (G spot 64 vCPU in us-east-1, 48 in use; 8 in us-east-2 and us-west-2; G on-demand 0 / 8 / 0; the
  increases to 96 are still open), so the LSA port's keeper asks for one on-demand p5.4xlarge (P on-demand 64 vCPU in
  us-east-1; P spot 8, too small) every 10 min, falling back to the L40S slot after the concept-memory run's tests.
  First try 06:54 UTC: no capacity in any us-east-1 zone. The catalogue waits for the concept-memory tests and the
  LSA port's start (`events.log` in S3); the fair full-attention control waits for the catalogue's start and takes the
  L40S slot, or the H100 once the LSA port is done. One H100 is an estimate of 55-60 h and about $410 per model
  (same rows per step: 8 rows at 4k and 2 at 16k per pass on 80 GB). Data: `m21mix_v1` in S3 since 07:58 UTC
  Nov 14, `m21mix_v2` since 19:53 UTC (author, 2026-11-14: "Train two best models 100M with a longer data budget, with
  a better tokenizer and data mix"; "Increase our budget to $4000"; "SmolLM2, bigger body").
- **Serves:** the meridian21 goal of 2026-11-13 (first models and a demo for investors: show that the memory
  architecture uses far context on real long documents) and Vision priority 2 (long context). Not a new
  architecture: the two best 100M reads of E37 stage B, unchanged, on better data. A data and protocol experiment,
  explicitly asked for by the author.
- **Implementation plan:** this spec's "Plan" section.
- **Owner / dates:** Krzysztof Sopyla · opened 2026-11-14 · closed —
- **Evidence behind it:**
  - E37 stage B (100M, 3.0B FineWeb-Edu + 0.4B LongDocs-lite, GPT-2 tokens): concept memory keeps 99 % of the
    window→dense gap at 16k, the LSA port 97 %; every model scores 41–42 % on the core quizzes (guessing ≈ 29 %),
    which is a data-budget ceiling, not an architecture difference.
  - With 3× more tokens (1.0B → 3.0B) full attention's far-context gain stayed at 0.19 nats: web text rarely needs
    far context, so more web alone does not grow the thing we sell.
  - The 64k stage (0.1B tokens of 64k rows) lowered the 128k loss of the catalogue model 2.058 → 1.943 and of the
    LSA port 1.994 → 1.923 (E37 spec, 2026-11-14): length in training matters and is cheap.
  - Exact facts in real text are the weak spot: concept memory finds an essay-haystack number 6 % of the time at
    32k (LSA port 55 %).
  - Published small-model recipes (survey of 2026-11-14, sources in "Builds-on"): a web-heavy main phase, a tail
    richer in code, math and curated text during the decay, and a long-context stage with 34–40 % long documents.

## Hypothesis
The 100M reads of E37 are limited by their data, not by their design. On a mix built like the published small-model
recipes, with 3.4× the tokens, long documents in two length stages (16k and 64k) and a few percent of rows whose
answer provably sits thousands of tokens back, both models (a) predict held-out long documents better than their
E37 versions at every length to 128k, measured in bits per byte so the tokenizers compare, (b) leave the quiz floor,
and (c) find exact facts in real text past 16k — while concept memory keeps its far-context benefit.

## Builds-on
- **Models (unchanged):** `scripts/run_e33n.py` arms `e37a` (concept memory, 8 keys per concept, 512 opened tokens,
  one summary per other concept, length scale at 16,384) and `lsa` (LSA's released flat read ported to the same
  reading layer), size `m100` (width 896, 3 prelude layers, reading layer + 1 local layer looped 4 times, one KV
  head of 64, local windows 128). FFN width fitted as before; only the word tables change with the vocabulary
  (49,152 instead of 50,257).
- **Architecture page (2026-11-21):** `docs/3_Evaluations_and_Baselines/e38_150m_architectures.html` walks the four
  `m120t` leaderboard models (concept memory, topic index, catalogue, full attention) step by step, with shapes and
  trainer-exact weights; their NN Builder designs are `tools/nn-builder/designs/e38_*_150m.json`.
- **Baselines to beat:** E37 stage B, seed 42 — `e33n_m100_e37a_tok3.0e9_seed42` and `e33n_m100_lsa_tok3.0e9_seed42`
  (S3 `runs/e37/`), and their seed-7 repeats (queued 2026-11-13).
- **What is materially new:** the data and the protocol — tokenizer, mix, three length stages, 11.5B tokens — and
  the training speed settings of 2026-11-14 (`--fast`). The architecture is deliberately the same, so any change
  in the scores is the data's.
- **Recipes this follows** (read at the primary sources on 2026-11-14): SmolLM2 (2502.02737: web 90 → 58 %, code
  10 → 24 %, math 0 → 14 %, 40 % long documents in the context extension), SmolLM3 (web/code/math 85/12/3 →
  63/24/13, then 32k and 64k stages), Olmo 3 (2512.13961: long-context stage 34 % long PDFs), Nemotron 3 Nano
  (2512.20848: 20 % long-document QA and 1 % synthetic retrieval in its long-context phase), MobileLLM-R1
  (2509.24945), Qwen3 (2505.09388: a 32k stage with 75 % of text 16–32k long), DeepSeek-V4 and Kimi K3 (length
  curricula inside pretraining). None trains at 16k–64k on a 10–30B budget; that part is ours.

## Why this is not a safe retread
It is a protocol experiment, not an architecture one, and the author asked for it by name. What it risks: that the
architecture's far-context benefit shrinks when the base model gets stronger (a better local model needs the far
context less), and that constructed far-context rows teach a template instead of a skill. Both would be findings.

## Arms
| arm | what | order |
|---|---|---|
| `e37a` | concept memory, seed 42 | first |
| `lsa` | the LSA port, seed 42 | second |

Not in this spec (each ≈ one more run): a full-attention control with rotary positions on the same mix (needed
before any "matches full attention" claim on this data; decide after the E37 `dense_rope` result), and the
catalogue model.

## Data: `m21mix_v1` (SmolLM2 tokenizer, 49,152; `<|endoftext|>` = 0 starts every document)
Every source is public and ungated, downloaded anonymously on an AWS CPU node and written as uint16 token shards
to `s3://…/data/m21mix_v1/` (`scripts/build_m21_mix.py`, `scripts/build_reasoning_rows.py`,
`infra/aws/jobs/m21_data_build.sh`). No Hugging Face token and no laptop step. The gated Nemotron packages
(CC-v2.1, CC-Math, Code) are left out for that reason; the open Nemotron "Specialized" subsets are in.

| phase | rows | tokens | sources (share of the phase's tokens) |
|---|---|---|---|
| trunk | 4,096, documents packed | 8.0B | FineWeb-Edu (score ≥ 3) 42 %, DCLM-Edu (≥ 3) 24 %, code files (Common Pile Stack v2 Edu) 10 %, Nemotron Wiki-Rewrite 8 %, FineMath 4+ 6 %, Nemotron RQA 6 %, Nemotron Math-Textbooks 4 % |
| decay | 16,384 | 3.0B | long documents 60 %: long PDFs 22 %, Gutenberg books 12 %, arXiv 5 %, DOAB 3 %, US GPO 3 %, Hansard 3 %, code packed per repository 12 %; web replay 20 % (FineWeb-Edu 14 %, DCLM-Edu 6 %, tokens the trunk did not read); Nemotron InfiniByte-Reasoning 7 %; multi-hop libraries 6 %; planted-fact lookups 3 %; chat (smol-smoltalk, chat template as text) 4 % |
| long | 65,536 | 0.5B | long PDFs 30 %, books 25 %, arXiv 4 %, DOAB 3 %, US GPO 3 %, code repositories 15 %, multi-hop libraries 10 %, lookups 5 %, web replay 5 % |

- **Long documents** are whole documents of ≥ 16,384 tokens, one contiguous span per row; the 64k phase reads the
  ones long enough for it.
- **Multi-hop libraries:** the context paragraphs of 20–160 2WikiMultihopQA training items, shuffled, then all the
  questions with their evidence chain and answer at the end of the row, so an answer needs paragraphs thousands of
  tokens back. HotpotQA, MuSiQue, SQuAD and BABILong are not used: they are in our tests.
- **Lookups:** 4–24 invented facts planted in a real book span, asked at the end; 15 % of rows ask for a fact that
  was never stated ("not stated in the text") and 15 % of facts are revised later (the answer is the latest). The
  templates avoid the wording of the RULER needles, but RULER is no longer zero-shot for these models and is
  reported as "after lookup training".
- **Validation:** per source, a hash-held-out set; for the five Common Pile long-document sources the same
  documents as LongDocs-lite's validation, so E37 and E38 models can be scored on the same documents.

### Data build record (2026-11-14)
Built 07:23–07:58 UTC on one 32-core spot node (`m21-data`, $0.71) plus a 16-core node for the constructed rows
(`m21-reason`, 07:32–07:46 UTC, $0.19): **13.76B train tokens, 20 sources, 174 shards, 25.7 GiB** in
`s3://…/data/m21mix_v1/`, every shard named in `mix.json` present. Five sources (finemath, math_textbooks, books, pdfs,
hansard) were built and uploaded but lost their `done.json` to a folder-sync race; the markers were copied by hand on
the node and both job scripts now copy file by file.

| source | train tokens | documents (or rows) | shards | bytes per token |
|---|---|---|---|---|
| fineweb_edu | 4.000B | 3,849,215 | 41 | 4.56 |
| dclm_edu | 2.300B | 1,663,910 | 24 | 4.40 |
| pdfs | 1.200B | 23,055 | 13 | 3.58 |
| code_files | 0.850B | 733,178 | 9 | 3.27 |
| books | 0.750B | 7,255 | 8 | 3.90 |
| wiki_rewrite | 0.700B | 599,554 | 8 | 3.16 |
| code_repos | 0.600B | 13,436 | 7 | 3.26 |
| rqa | 0.520B | 64,170 | 6 | 3.02 |
| finemath | 0.520B | 318,485 | 6 | 3.13 |
| math_textbooks | 0.350B | 163,795 | 4 | 3.59 |
| doab | 0.300B | 3,154 | 4 | 3.98 |
| arxiv | 0.300B | 9,052 | 4 | 2.74 |
| usgpo | 0.300B | 3,807 | 4 | 4.15 |
| hansard | 0.300B | 5,002 | 4 | 4.78 |
| infinibyte | 0.250B | 13,813 | 3 | 3.54 |
| multihop_16k | 0.200B | 12,207 rows | 3 | 3.64 |
| chat | 0.130B | 145,575 | 2 | 4.44 |
| lookup_16k | 0.100B | 6,104 rows | 2 | 3.90 |
| multihop_64k | 0.060B | 916 rows | 1 | 3.64 |
| lookup_64k | 0.030B | 458 rows | 1 | 3.94 |

Supply (`scripts/run_e33n.py mixinfo`): one pass over the data is larger than the budget in every phase, so no row is
read twice — trunk 8.50B tokens a pass (limited by the code files) for 8.0B, decay 3.25B (limited by chat) for 3.0B,
long 0.60B (limited by the 64k multi-hop libraries) for 0.5B. No source is empty at any phase's length and every
evaluation source has rows at 4k, 16k and 64k.

## Training protocol
`scripts/run_e33n.py --mix DATA/m21mix_v1 --fast`, size m100, 128 rows of 4k per optimizer step in every phase
(128 × 4k, 32 × 16k, 8 × 64k).
- **trunk:** 8.0B tokens, warmup 1 %, constant LR 6.4e-4 (as E37).
- **decay:** 3.0B tokens from the trunk's weights, fresh optimizer, 3 % warmup, 1 − sqrt decay to 0.1 × (as E37, 7.5×
  the tokens).
- **long:** 0.5B tokens from the decay's weights, 6.4e-5 → 6.4e-6, each layer recomputed in the backward (as E37's
  64k stage, 5× the tokens).
- **Speed settings (`--fast`, measured 2026-11-14):** Liger's fused linear + cross-entropy (the same loss: equal in
  fp32, 7e-5 apart in bf16) and the compiled concept grouping (same labels), with 4 / 1 / 1 rows per pass and 48 GB
  card (as E37: larger passes add no speed on an L40S). AdamW, betas 0.9 / 0.95, weight decay 0.1, clip 1.0, bf16, flex attention — unchanged.
- **Compute:** one 4 × L40S spot node per model (a checkpoint to S3 at least every 800 steps and ten per phase; `infra/aws/keep_running.sh`
  relaunches after an interruption and stops before the budget cap). No on-demand H100 was available on 2026-11-14
  (07:13 UTC, all six zones). Measured on one L40S with `--fast`: 30.3k tokens/s at 4k (old settings 20.3k) and
  20.4k at 16k (old 15.2k), with the E37 micro-batches (4 and 1 rows per pass). On four cards that is about 18 h
  for the trunk, 10 h for the decay and an estimated 4–5 h for the 64k phase: about 37 h and $260 per model with
  its tests.

## Evaluation
- Long-document loss by position at 4k–128k on the seven long sources, in nats per token and bits per byte
  (`evaluation/long_doc_loss.py --tokenizer`); the E37 models rescored with bits per byte on the same five sources.
- The far read's value inside each model: the same rows with the summaries / opened tokens off.
- RULER, 13 tasks at 4k–32k (cached decoding), then 64k and 128k on one-card nodes; essay-haystack needles
  reported apart from the filler-text ones.
- Core quizzes; BLiMP, SWDE and BABILong (the "extra" suite).
- The 16k model's long-document loss, to see what the 64k phase changed.

## Success criteria (set BEFORE running)
Ceilings: the E37 stage-B scores of the same arm (measured). For the quizzes the ceiling of the size class is
SmolLM2-135M (2T tokens, 170× ours); it has not been scored with our core suite yet, so S3 is stated against our own
E37 score and is re-read once that reference is measured (plan step 6).
- **S1 (long documents):** bits per byte on the five shared long sources, positions ≥ 2k, at 16k and at 64k: at
  least 3 % lower than the same arm's E37 stage-B model, for both arms.
- **S2 (length):** at 128k the E38 `e37a` loss in bits per byte is no higher than at 64k by more than 1 % (no fade).
- **S3 (knowledge):** core mean at least 3 points above the same arm's E37 score (41–42 %; guessing ≈ 29 %), for
  both arms.
- **S4 (facts in real text):** `e37a` essay-haystack number needle at 32k ≥ 30 % (E37: 6 %; the LSA port's 55 % is
  the measured ceiling at this size).
- **S5 (far context still used):** switching the far read off costs `e37a` at least 0.10 nats on positions ≥ 2k at
  16k (E37: dense − window = 0.19 nats).

## Kill criteria (set BEFORE running)
- **K1:** the trunk's validation loss on FineWeb-Edu after 3.0B tokens is more than 2 % worse in bits per byte than
  the E37 trunk at 3.0B → the mix or the tokenizer hurts plain web modelling; stop and inspect before the decay.
- **K2:** training speed on the node is below 0.9 × the E37 runs' (79k tokens/s on 4 × L40S) → the fast settings
  misbehave under four-card training; fall back to the old settings.
- **K3:** S5 fails by half (< 0.05 nats) → the stronger local model no longer needs the notebook at 16k; report it
  and move the claim to 64k–128k only.

## Leaderboard recipe (added 2026-11-14, evening)
The author wants the 100M models on the Tiny-ML Leaderboard (huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard)
and, after comparing tokenizers, chose **"SmolLM2, bigger body"** (19:20 UTC). The board admits models up to 150M
parameters in total, a shared input/output table counted once; every model trained so far (about 162M with separate
tables) is over the cap, including the running long concept-memory run.

- **Shared table:** `--lm_head_tied` (`nn/perceiver_ar_lm.TiedFactorizedHead`): the output reuses the input table
  E [49,152 × 384] through one projection, logits = (h·Pᵀ)·Eᵀ. Loss, gradients and save/load match an explicit tied
  reference; the fused GPU loss equals the chunked one with it (Odra, 18 GPU tests).
- **Size `m120t`:** width 896, 3 local + 9 stacked layers (7 at m100), FFN 2,816; body 120.8M (m100: 96.5M).
  Totals: concept memory, LSA port and fair full attention 141.51M (the trainer counts 141,506,208), catalogue
  145.61M. It trains only from a built mix and refuses a vocabulary larger than SmolLM2's.
- **Data:** `m21mix_v2` (Nemotron-CC High-Quality, Diverse-QA and Math in the web half; SmolLM2 tokens). None of the
  board's tests (BLiMP, ARC, WikiText) is in any training source; the board flags disclosed benchmark training.
- **Optimizer:** AdamW, or Muon from one peak (`--muon_peak`: trunk and decay at the peak, the 64k phase at a tenth)
  if the paid Muon test passes its mark (≥ 0.03 lower held-out loss than AdamW at the same step).
- **Job:** `infra/aws/jobs/e38_board.sh SEED OPT ARM...` (OPT = adam | muon:PEAK); same phases, tests and resume.
- **Memory:** 4 rows of 4k and 1 row of 16k each peak at about 20 GB on a 24 GB card (Odra probe), so the 48 GB AWS
  cards keep the m100 pass sizes.
- **Scoring:** `evaluation/run_lm_eval_suite.py --tier leaderboard` (BLiMP acc over 67 sets, ARC-Easy acc, WikiText-2
  byte perplexity, 0-shot) and `evaluation/tiny_ml_board.py` (the board's formula; five entries reproduce the live
  page's scores to 2 decimals; snapshot of 100 entries, 2026-11-14).
- **Pass mark for the board (set before running):** at least the board's 12B-token reference, JugnuLM-53M
  (BLiMP 78.1, ARC-Easy 51.4, WikiText-2 2.04) on BLiMP and ARC-Easy, i.e. the middle of the board; ARC-Easy
  < 45 % after the full budget means the recipe, not the size, is the limit.

## List questions run (added 2026-11-23)
The author, 2026-11-23 15:23 UTC, on the 150M models reading several far values and writing one of them again:
"find a way to fix those issues, work on the concept model ... try to find a root cause and solution ... use Odra
and Polonez for checks, do not afraid to use AWS for training". A run of this family on a new data mix, not a new
experiment: the concept-memory model, the same recipe, stages 2 and 3 again on `m21mix_v3`.

**Root cause (measured 2026-11-23 on Odra and Polonez, probe `scripts/concept_use_probe.py`).** Every planted-fact
question of `m21mix_v2` has one value for an answer, and a fact stated twice is answered with its latest value. No
row asks for several far values in one answer, so no model of the family has that skill:
- On held-out questions in our own wording (48 rows of 16k), before any change: concept memory answers 0.91 of the
  single-value questions exactly, 0.00 of the all-values questions with every value and 0.00 of the several-facts
  questions; full attention 0.80 / 0.00 / 0.00; the topic index 0.94 / 0.00 / 0.00. Each writes one value and ends
  the line.
- On RULER's two several-values tasks all three models fail at the same three places. (1) Starting the list: after
  a several-keys question concept memory writes ":" first in 80 % / 62 % / 12 % of prompts at 4k / 8k / 16k; with
  the far read restricted to the planted sentences 98 % / 100 % / 96 %, with every token readable word for word and
  no summaries 84 % / 70 % / 8 %, with the summaries' weight held at the 4k level 6 % at 16k: the query there is not
  specific, the grouping, the choice of 512 tokens and the summaries are not the cause. (2) The first of several
  values under one key: right 0.46–0.68 (distinct keys: 0.96–1.00); the read blends the planted values. (3) Moving
  on: handed its own answer so far, concept memory writes a new value 0.58–0.86 of the time at 8k (full attention
  0.44–0.74, topic index 0.58–0.96), and 0.74–0.96 once the written values are hidden from the far read.
- Sharper read weights (× 1.5, × 2) and summaries out make the free answers worse (8k several keys 0.34 → 0.12 and
  0.22), so the read is not too blunt in general.
- **The skill is learnable in minutes.** A fine-tune of the released weights on Polonez (60M tokens of 16k rows,
  half list rows, half the 16k stage's own sources; 15:52–16:51 UTC): held-out all-values answers with every value
  0.00 → 0.73, several-facts exact 0.00 → 0.80, single-value 0.91 → 0.97. RULER, whose wording the rows avoid (the
  banned-words check), board protocol, 100 prompts: all values 0.31 / 0.34 / 0.24 → 0.82 / 0.74 / 0.47 at 4k / 8k /
  16k, several keys 0.40 / 0.32 / 0.11 → 0.62 / 0.48 / 0.27; the other 11 tasks' mean 0.513 / 0.460 / 0.408 →
  0.488 / 0.460 / 0.402.
- **That fine-tune is not the fix:** it saw 16k rows only and lost the 64k calibration the 64k stage had taught
  (64k documents 2.293 → 2.477 nats per token, summaries now a cost; 16k documents 2.126 → 2.133), and the UUID
  needle fell (0.29 → 0.02 at 4k). The list rows belong in stages 2 and 3 at a few per cent, not in a heavy
  afterthought.

- **Added 18:10 UTC, twelve minutes after the launch and before any result of the run (marks unchanged): the
  control changes how RULER reads.** The same fine-tune with single-value lookup rows in place of the list rows
  (Polonez, 16:52–17:44 UTC): our list exam stays at 0.00 / 0.00 (single-value 0.95), so the list skill comes from
  the list rows. But RULER rises almost as much as with the list rows (board protocol at 4k: all values 0.31 →
  0.74, several keys 0.40 → 0.52; the list fine-tune 0.82 and 0.62), and the UUID needle falls the same way
  (0.29 → 0.04). The released model had already read more lookup tokens (115M at 3–5 % of its rows) than either
  fine-tune added (30M at 50 %), so RULER's two tests mostly follow how strongly the END of training is tuned to
  copying from far context, not the list skill. Consequence for this run: L1 is the test of the list rows; L2 may
  fail with L1 passed, which would say that 3–5 % of constructed rows teach the skill but do not move RULER, and
  that a retrieval-rich final stage is the lever (a home-card test of one is under way).

- **Recipe changed before any result of the run (2026-11-23, 20:10 UTC; marks unchanged).** The run on
  `m21mix_v3` (stage 2 started 18:55:42 UTC on spot g6e.12xlarge `i-08bdd35208065be00`; speed 67.9k tokens/s over
  its first 300 steps, held-out loss 2.306 at step 200 against the E38 run's 2.295) was stopped and its node
  terminated at 20:10 UTC, about $9, because two home-card tests that finished at 19:29 UTC showed a better recipe:
  - **Recall rows** (`recall_16k` / `recall_64k`: the list sets with half of the asked items written as a statement
    to complete, "As stated above, the serials of X are" → " a, b, c."), the same fine-tune as above (Odra,
    18:32–19:29 UTC): RULER board protocol all values 0.31 / 0.34 / 0.24 → 0.99 / 0.97 / 0.91 at 4k / 8k / 16k,
    several keys 0.40 / 0.32 / 0.11 → 0.88 / 0.86 / 0.86; the list now begins with its first value in 86–98 % of
    several-keys prompts (12 % at 16k before); the other 11 tasks 0.499 / 0.459 / 0.409 (E38 run 0.513 / 0.460 /
    0.408); our list exam 0.63 all values, 0.74 several facts exact. Costs: similar-needles task (multi-key 2) at
    16k 0.76 → 0.42 (it now picks a neighbouring key), the UUID needle 0.29 → 0.06 at 4k. Not yet measured on 64k
    documents.
  - **A 64k stage after the question-only fine-tune** (Polonez, 18:19–19:29 UTC, 30M tokens of 64k rows, 60 %
    constructed): 64k documents 2.477 → 2.320 nats per token (E38 run 2.293) with the summaries useful again
    (switching them off costs 0.069, the E38 run 0.068); multi-key 2 back to 0.94 / 0.90 / 0.81; other 11 tasks
    0.518 / 0.481 / 0.443; several keys 0.78 / 0.74 / 0.38, all values 0.93 / 0.88 / 0.63; the UUID needle stays
    low (0.04 / 0.05 / 0.04: the model answers " a special magic uuid. It's a unique identifier…").
  So the run now reads **`m21mix_v4`** (`data/mix_recipes/m21_mix_v4.json`): v2 plus `recall_16k` at 3 % of
  stage 2 and `recall_64k` at 5 % of stage 3, from the `pdfs` share; everything else as v3. Job:
  `infra/aws/jobs/e38_lists.sh e37a m21mix_v4 recall`, folder `runs/e38/e33n_m120t_e37a_muon0.01-recall_tok8.0e9_seed42/`.
  RULER's two several-values tasks now share their **sentence form** (a statement to complete) with training rows,
  not their words (the banned-words check holds): report them as trained-form, as the card already does for the
  single lookups. The marks L1–L3 and KL1–KL3 apply to this run unchanged ($220 from 20:10 UTC; the v3 attempt's
  $9 and the start-up failures' $2.70 are outside it).

- **Home-card check of the v4 recipe, all release tests (2026-11-23, 20:12–22:33 UTC; diagnostic, not the run).**
  The recall fine-tune above plus a 64k stage (Polonez, 20:12–21:16 UTC, same 64k stage as the question-only
  model), then the release protocol (Polonez: list exam, start probe, RULER 4k–16k; Odra: documents, core, RULER
  32k), against the E38 run:
  - RULER 13-task mean 0.594 / 0.556 / 0.501 / 0.451 at 4k / 8k / 16k / 32k (E38 run 0.488 / 0.439 / 0.373 /
    0.319); the other 11 tasks 0.532 / 0.492 / 0.439 / 0.382 (0.513 / 0.460 / 0.408 / 0.347), better at every
    length; several keys 0.89 / 0.84 / 0.83 / 0.82, all values 0.97 / 0.97 / 0.86 / 0.84 — the gains hold at 32k,
    a length no training row has; multi-key 2 back to 0.93 / 0.79 / 0.75 / 0.52 (0.92 / 0.84 / 0.76 / 0.41).
  - Cost: the UUID needle 0.10 / 0.15 / 0.03 / 0.07 (0.29 / 0.27 / 0.07 / 0.07). The control fine-tune with
    single-value lookup rows loses it too (0.04 at 4k), so it comes from fine-tuning on a large constructed share,
    not from the list rows; 80 of 100 replies at 4k begin " a unique identifier that is …" instead of the ID (an
    answer-start failure). Across the saved replies of the E38 run and this model at 4k–32k, every reply that
    begins with an ID begins with the right one (105 of 105; full attention 145 of 146): the far read finds the
    36-character ID whenever the answer starts, and the score is the share of replies that start (E38 run 29 / 27
    / 7 / 7 %, full attention 71 / 69 % at 4k / 8k). Our rows never say "uuid" and their IDs are 6–10
    characters. This run mixes the rows at 3–5 % from the start of stage 2, as v2 did.
  - Documents (nats per token, lower is better): 64k 2.319 (2.293, +1.1 %), 16k 2.147 (2.126), 128k 2.450 (2.451,
    a tie; past 61k tokens of context the full context is now 0.030 worse than a 61k cut, the E38 run 0.060);
    summaries useful again (switched off: +0.069 at 64k, +0.071 at 128k). WikiText 0.972 bits per byte (0.964).
    Core 7 mean, the card's metrics: 46.9 (46.2).
  - Our list exam at 16k: single 0.93, all values 0.59, several facts exact 0.72 (L1 asks 0.90 / 0.65 / 0.70).
  As a fine-tune it misses only L3's document marks, by about 0.01 nats, and the UUID mark; the run is the test.

- **The recall run, training and its AWS tests (done 2026-11-25 03:07 UTC; read 06:20 UTC).** Spot g6e.12xlarge,
  one interruption (~11:30 UTC 11-24, resumed from its checkpoint). Nodes $149.27 of $220 (KL3 met); KL1 and KL2
  met during stage 2 (see above). Against the E38 run ("the released model"), same job, same scripts and settings:
  - Held-out loss at the end of stage 2 1.996 (1.996; L3 asks ≤ 2.001) and of stage 3 1.938 (1.936; ≤ 1.941):
    met. Core 7 mean, the card's metrics (acc): 46.27 (46.23; ≥ 45.7): met. WikiText 0.964 bits per byte (0.964).
  - Long-document loss (the job's test: longdocs-lite, 7 sources, nats per token): 2.066 / 1.890 / 1.881 / 1.775
    at 4k / 16k / 64k / 128k (2.067 / 1.888 / 1.879 / 1.775), a tie at every length.
  - RULER as the job runs it (100 prompts, cached decoding; not the card's per-task answer lengths), 4k / 8k /
    16k / 32k: 13-task mean 0.572 / 0.526 / 0.405 / 0.360 (0.433 / 0.397 / 0.339 / 0.304); the other 11 tasks
    0.569 / 0.530 / 0.419 / 0.377 (0.456 / 0.413 / 0.372 / 0.333); all values 0.59 / 0.41 / 0.35 / 0.38 (0.29 /
    0.31 / 0.20 / 0.21); several keys 0.58 / 0.59 / 0.29 / 0.15 (0.33 / 0.29 / 0.10 / 0.07); the UUID needle 0.97 /
    0.91 / 0.44 / 0.59 (0.02 / 0.02 / 0.00 / 0.00); multi-key 3 0.34 / 0.15 / 0.08 / 0.02 (0.00 / 0.01 / 0.01 /
    0.00). **The cost: multi-key 2 (similar needles) 0.73 / 0.62 / 0.25 / 0.13 (0.92 / 0.85 / 0.67 / 0.42).** The
    home-card check had it back after its 64k stage; the full run, with recall rows from the start of stage 2, has
    not.
  - L1, L2 and the document, other-11 and UUID items of L3 are set on the card's protocol (list exam; RULER with
    each task's own answer length, every reply saved; the card's 64k and 16k document tests): run on Polonez from
    06:11 UTC on the run's final weights (`scripts/release_evals.sh`, cards 1 and 3; list exam and first-token
    probe on card 0).
  - **L1 met (06:17 UTC):** our list exam at 16k (48 held-out list rows): single-value answers 0.94, all-values
    answers 0.67 (values found 0.91), several facts exact 0.85 (L1 asks 0.90 / 0.65 / 0.70; the released model 0.00
    on both list forms; the home-card fine-tunes 0.63 / 0.59 all values, 0.74 / 0.72 several facts).
  - **The card's protocol, all release tests (Polonez, 06:17–08:05 UTC; `scripts/compare_release_evals.py
    --before docs/release/hf_card_concept_memory_150m`; copies in `Cache/results/recall_release/`).** RULER at
    4k / 8k / 16k / 32k against the card: 13-task mean 0.567 / 0.527 / 0.392 / 0.362 (0.488 / 0.439 / 0.373 /
    0.319); the other 11 tasks 0.567 / 0.530 / 0.398 / 0.381 (0.513 / 0.460 / 0.408 / 0.347); all values 0.59 /
    0.43 / 0.35 / 0.36 (0.31 / 0.34 / 0.24 / 0.24); several keys 0.54 / 0.60 / 0.36 / 0.16 (0.40 / 0.32 / 0.11 /
    0.09); the UUID needle 0.97 / 0.88 / 0.46 / 0.53 (0.29 / 0.27 / 0.07 / 0.07); word counting (fwe) 0.20 / 0.27 /
    0.12 / 0.17 (0.15 / 0.16 / 0.02 / 0.06); common words (cwe) 0.56 / 0.52 / 0.35 / 0.17 (0.52 / 0.42 / 0.27 /
    0.13). **The cost: multi-key 2 (one number among similar keys) 0.72 / 0.63 / 0.26 / 0.20 (0.92 / 0.84 / 0.76 /
    0.41).** Of its wrong replies, 26 of 28 at 4k, 35 of 37 at 8k and 64 of 74 at 16k are a 7-digit number (the
    form of a value in the text, so most likely a neighbouring key's), the rest invented ("123456789",
    "1000…"); every wrong reply of the released model is a 7-digit number. The single-value needles 1.00 / 1.00 at
    4k–16k, 0.92 at 32k on the second one (0.98).
  - Documents (nats per token, lower is better): 64k 2.291 (2.293), 16k 2.125 (2.126), 128k 2.461 (2.451, +0.4 %;
    past 61k tokens of context the full context is now 0.072 worse than a 61k cut, the released model 0.060).
    Summaries off costs +0.067 at 64k (+0.068) and +0.022 at 16k (+0.023): unchanged. Core 7 mean (acc) 46.37
    (46.22); WikiText 0.964 bits per byte (0.964); BLiMP 79.9 (80.6, −0.7 points; not a mark). Tiny-ML board
    place 26th (the card's 24th).
  - **L2 missed on all values, met on several keys:** all values 0.59 / 0.43 / 0.35 (asks ≥ 0.60 / 0.60 / 0.40),
    several keys 0.54 / 0.60 (asks ≥ 0.50 / 0.40). This is the outcome the 18:10 UTC note above foresaw for
    3–5 % of constructed rows: L1 passed and RULER's all-values task moved less than with a retrieval-rich end of
    training (the home-card v4 check: 0.97 / 0.97 / 0.86).
  - **L3 met:** 64k documents 2.291 ≤ 2.305, 16k 2.125 ≤ 2.137; the other 11 tasks +0.054 / +0.070 / −0.010 at
    4k / 8k / 16k (within 0.03 or better); the UUID needle at 4k 0.97 ≥ 0.19; core 46.37 ≥ 45.7; held-out losses
    as above. Not covered by L3 and lost: multi-key 2 at 16k (0.76 → 0.26) and 32k (0.41 → 0.20).

**Data: `m21mix_v3`** (`data/mix_recipes/m21_mix_v3.json`) = v2 plus two constructed sets
(`scripts/build_reasoning_rows.py --only list_16k,list_64k`): lookup rows whose question block also asks for all the
values of one thing (2–5 values at 16k, 2–6 at 64k, each stated on its own; answer in the order of the text) and
for 2–4 single facts in one question (answer in the order asked; those facts are not asked on their own). 3 % of
stage 2 and 5 % of stage 3, taken from the `pdfs` share; every other source, weight, budget and every evaluation set
is v2's. Stage 1 reads no constructed rows, so it is the same under both mixes.

**Run:** `infra/aws/jobs/e38_lists.sh e37a`: its own folder
`runs/e38/e33n_m120t_e37a_muon0.01-lists_tok8.0e9_seed42/`, seeded with the stage-1 final weights of the E38
concept-memory run (same seed, same optimizer), then stage 2 (3.0B tokens of 16k rows), stage 3 (0.5B of 64k rows)
and the tests, as every E38 board run. One spot g6e.12xlarge (4 × L40S); the E38 run took 19.3 h for these stages
and the tests. W&B name "150M concept memory with list questions, Muon, stage 2 of 3".

**Pass marks (set before the run; "the E38 run" = the released concept-memory model, same stage 1):**
- **L1 (the skill, our wording):** on the held-out list rows at 16k, all-values answers with every value ≥ 0.65 and
  several-facts answers exact ≥ 0.70 (the fine-tune reached 0.73 and 0.80 with a third of the list tokens this
  run sees), single-value answers ≥ 0.90.
- **L2 (RULER's wording, never trained):** all values ≥ 0.60 at 4k and 8k and ≥ 0.40 at 16k; several keys ≥ 0.50
  at 4k and ≥ 0.40 at 8k (the fine-tune: 0.82 / 0.74 / 0.47 and 0.62 / 0.48; the E38 run 0.31 / 0.34 / 0.24 and
  0.40 / 0.32).
- **L3 (nothing else lost):** 64k documents ≤ 2.305 nats per token and 16k documents ≤ 2.137 (the E38 run's 2.293
  and 2.126 + 0.5 %); the mean of the other 11 RULER tasks within 0.03 of the E38 run's at 4k, 8k and 16k; the UUID
  needle at 4k ≥ 0.19 (0.29 − 0.10; run-to-run noise at 100 prompts is ± 0.09); core mean ≥ 45.7 (46.22 − 0.5);
  held-out loss at the end of stage 2 ≤ 2.001 and of stage 3 ≤ 1.941 (the E38 run's 1.996 and 1.936 + 0.005; the
  evaluation sets are the same).

**Stop marks:**
- **KL1:** held-out loss at step 1,000 of stage 2 more than 0.03 above the E38 run's at that step → stop: 3 % of
  list rows must not cost general text.
- **KL2:** stage-2 speed below 0.9 × the E38 run's on the same node type → report before going on.
- **KL3 (money):** $220 for this run's nodes (expected about $150); the family cap of $5,000 stands (spent
  $3,089.18 on 2026-11-23 15:25 UTC, E39's run has $450 of it reserved).

**Not claimed:** RULER's two tests are no longer untrained for this model in the sense the card already states for
the single lookups (similar shape, different wording). Order of the values in an all-values answer is not learnable
by a far read without a position code; the mark counts values, not their order. E39's stages 2 and 3 are not
changed by this section: whether they read v3 is the author's call (its stage 1 is the same either way).

## Plan
1. Engineering (done 2026-11-14, branch `eng-100m`): cache + tuning + architecture branches merged; `--fast`;
   read-once / answer-many decoding; `--mix`; job and keeper scripts.
2. Build `m21mix_v1` on a CPU spot node (`infra/aws/jobs/m21_data_build.sh`), check the supply table in `mix.json`
   (rows available vs needed per source and phase).
3. Pipeline check on Odra: a few steps of each phase on a small build, and the tests on the resulting model.
4. L40S speed check (`infra/aws/jobs/eng_check_node.sh`) to fix the micro-batches.
5. Launch `e37a`, then `lsa` (`infra/aws/jobs/e38_full.sh 42 e37a lsa`), kept alive by `keep_running.sh`.
6. Tests, bits-per-byte rescoring of the E37 models, needle nodes at 64k / 128k, SmolLM2-135M on the core suite
   (the size-class reference for S3), the board.

## Risks
- **Two things change at once** (tokenizer and data, plus budget): the comparison with E37 is in bits per byte on
  shared documents, and the trunk's K1 check isolates the tokenizer + web part at equal tokens.
- **Constructed rows leak a template:** they are ≤ 9 % of the decay and 15 % of the long phase; tests that share
  their format (RULER) are labelled; the long-document loss and the essay needles do not share it.
- **Spot interruptions** over multi-day runs: a checkpoint to S3 at least every 800 steps (under an hour of work), automatic relaunch.
- **Licences:** the Nemotron synthetic subsets are CC BY 4.0 but their cards note that the generating models'
  terms may apply; FineWeb-Edu, DCLM-Edu and FineMath are ODC-By / CC BY; Common Pile is openly licensed. Weights
  stay private or gated until the IP decision.
- **No full-attention control on this mix** in this spec: no "matches full attention" claim from E38 alone.

## Result
—
