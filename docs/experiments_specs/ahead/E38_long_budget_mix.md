# E38 — Two 100M models on a modern data mix: longer budget, SmolLM2 tokenizer, far-context data

- **Status:** approved and in build (author, 2026-11-14: "Train two best models 100M with a longer data budget, with
  a better tokenizer and data mix; try to find some multi-hop or reasoning data in Hugging Face"; budget raised the
  same day: "Increase our budget to $4000"). Data builders and launcher written 2026-11-14; runs pending the data
  build and a free 4-GPU slot.
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

## Training protocol
`scripts/run_e33n.py --mix DATA/m21mix_v1 --fast`, size m100, 128 rows of 4k per optimizer step in every phase
(128 × 4k, 32 × 16k, 8 × 64k).
- **trunk:** 8.0B tokens, warmup 1 %, constant LR 6.4e-4 (as E37).
- **decay:** 3.0B tokens from the trunk's weights, fresh optimizer, 3 % warmup, 1 − sqrt decay to 0.1 × (as E37, 7.5×
  the tokens).
- **long:** 0.5B tokens from the decay's weights, 6.4e-5 → 6.4e-6, each layer recomputed in the backward (as E37's
  64k stage, 5× the tokens).
- **Speed settings (`--fast`, measured 2026-11-14):** Liger's fused linear + cross-entropy (the same loss: equal in
  fp32, 7e-5 apart in bf16) and the compiled concept grouping (same labels), with 8 / 2 / 1 rows per pass and 48 GB
  card. AdamW, betas 0.9 / 0.95, weight decay 0.1, clip 1.0, bf16, flex attention — unchanged.
- **Compute:** one 4 × L40S spot node per model (checkpoints every 400 steps to S3; `infra/aws/keep_running.sh`
  relaunches after an interruption and stops before the budget cap), or one on-demand H100 per model if that
  starts sooner. Estimate before the L40S speed check: 35–45 h and $250–330 per model.

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
- **Spot interruptions** over multi-day runs: checkpoints every 400 steps (~40 min) to S3, automatic relaunch.
- **Licences:** the Nemotron synthetic subsets are CC BY 4.0 but their cards note that the generating models'
  terms may apply; FineWeb-Edu, DCLM-Edu and FineMath are ODC-By / CC BY; Common Pile is openly licensed. Weights
  stay private or gated until the IP decision.
- **No full-attention control on this mix** in this spec: no "matches full attention" claim from E38 alone.

## Result
—
