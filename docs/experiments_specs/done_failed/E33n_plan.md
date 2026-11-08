# E33n — implementation plan (as built, 2026-11-04)

Spec: [E33n_lsa_testbed_topk_notes.md](E33n_lsa_testbed_topk_notes.md). Branch `e33n-lsa-testbed`.

## What was built (all config-selectable, no fork)

| piece | where | notes |
|---|---|---|
| top-k note read | `nn/perceiver_ar_lm.py`: `lm_read_topk`, `MessageCtx.note_topk`, `note_topk_threshold`, cut in `attend_message` | per (row, query head, query) the k-th best score among visible notes, fp32, query chunks of 1,024, no gradient; flex: `score_mod` sets notes below it to −∞; SDPA: the same cut in the dense mask. Exact selection, **same cost as the full note read** (the kernel still visits every note block) — the speed-up (gather of k notes, or LSA buckets) is a later step and only matters at 1M+. Not combined with E33d/E33e note terms. Tests: `tests/test_note_topk.py` |
| token-shard rows | `data/bin_token_dataset.py`, trainer `--bin_manifest` | modded-nanogpt GPT-2 `.bin` shards, memory-mapped; `stream` rows (trunk) and `doc` rows (one document per row, decay); per-token doc ids → the existing causal-LM collator masks every document start. Tests: `tests/test_bin_token_dataset.py` |
| LongDocs-lite | `scripts/build_longdocs_lite.py` | Common Pile filtered gutenberg / arxiv / doab / usgpo / hansard (LSA's source cards), docs ≥ 16k GPT-2 tokens, 300M train tokens per source, every 100th kept doc held out. Built 2026-11-04 in 11 min on a c7i.4xlarge: 2,906 / 8,911 / 3,217 / 3,092 / 5,088 docs kept, 2.2–2.9M held-out tokens per source |
| run driver | `scripts/run_e33n.py` | arms × sizes × phases → one trainer command; FFN width fitted on the non-embedding count (window arm), shared by all arms of a size |
| evals | `evaluation/long_doc_loss.py`, `evaluation/run_lsa_suite.py`, generation in `evaluation/lm_eval_perceiver_ar.py` | loss by position bucket at 4k–128k with the notebook on/off; RULER's 13 lm-eval tasks (GPT-2-sized prompts, string match, answers capped at 32 tokens, greedy, no cache); LSA's 8 core tasks |
| AWS | `infra/aws/` | see its README: self-terminating nodes, spend heartbeats, zone fallback, launch retry |

## Arms as configured (`scripts/run_e33n.py`)
- shared: perceiver_ar, plain input, GPT-2 vocabulary, factorised input table (e = 256 / 384 / 512), untied output layer,
  one KV head, local windows 128 (pre + stack), reading layer without RoPE (`par_global_nope`), E33 loop 4 rounds,
  span 1, exits off; flex attention, bf16, fused AdamW (0.9, 0.95), weight decay 0.1, clip 1.0.
- window: `_E31C` writer + `message_override none` (writer built and run, notes masked) — equal compute to the notebook arms.
- dense: `par_mode perceiver`, no writer, `message_boundary_token_id −1` → the reading layer is full causal over raw keys.
- e33: `_E31C` writer (32 notes / 256-token page, stride 192), raw window 512, closed read.
- e33n: e33 + `lm_read_topk 64`.

| size | width | pre / stack | FFN | non-embedding | total |
|---|---|---|---|---|---|
| s30 | 576 | 2 / 5 | 1472 | 30.2M (dense 26.3M) | 72.7M |
| m100 | 896 | 3 / 7 | 2624 | 100.6M (96.5M) | 166.4M |
| l300 | 1280 | 4 / 9 | 4608 | 300.0M (295.9M) | 393.0M |

## Schedule and batch (driver defaults)
- tokens per optimizer step: 0.26M (s30), 0.52M (m100), 1.05M (l300) in both phases (rows of 4k or 16k).
- trunk: `constant_with_warmup`, warmup 1 %; decay: `warmup_stable_decay` with warmup 3 %, `num_decay_steps` = 97 %,
  `decay_type 1-sqrt`, `min_lr_ratio 0.1`, weights from the trunk's `final`, fresh optimizer.
- peak learning rate 1e-3 (s30), scaled by 1/width above (6.4e-4, 4.5e-4) unless stage 0 says otherwise.

## Stage 0, measured on the home servers (2026-11-04 23:20 – 2026-11-05 00:33 UTC, no AWS cost)
No on-demand H100 was free in us-east-1 from 2026-11-04 21:50 UTC (every zone, retried every 2–3 min), and the
account has H100 quota in no other region and none in SageMaker. Stage 0 therefore ran on Odra / Polonez RTX 3090s
(`infra/aws/jobs/home_gpu_check.sh`, `lr_check.sh`); the H100 stage 0 (`stage0.sh`) still runs when capacity appears.

Speed (tokens/s, one RTX 3090, flex attention, bf16) and peak memory:

| size | rows | window | dense | e33 (full read) | e33n (top-64) |
|---|---|---|---|---|---|
| s30 | 4k, micro 4 | 29.1k, 21.2 GB | 38.4k, 16.6 GB | 28.1k, 21.2 GB | 26.2k, 21.2 GB |
| s30 | 16k, micro 1 | — | 25.5k, 8.6 GB | — | 16.6k, 13.2 GB |
| m100 | 4k, micro 2 | — | 22.1k, 13.3 GB | — | 14.6k, 16.6 GB |

- The top-k read compiles and trains under flex attention (first GPU run). Its cost is ~7 % over the full read; the
  writer (run by the window arm too) is the larger cost: ~25 % at s30 against the dense arm.
- Peak learning rate (window control, s30, 0.1B tokens, warmup 5 %, 1−sqrt decay over the last 20 %), validation
  loss: 5e-4 → 4.731, **1e-3 → 4.408**, 2e-3 → 4.516, 4e-3 → 5.124. The default 1e-3 stays; m100 / l300 keep
  their width-scaled peaks (6.4e-4, 4.5e-4).
- Data on the home servers is rebuilt from the public sources by `data_build.sh` and is identical to the S3 copy
  (same documents kept and tokens per source, same shard hash).

## Stage 1 location (2026-11-05 00:33 UTC)
30M runs on the home servers, free: Odra GPU 0 / 1 / 2 = window / e33 / e33n, Polonez GPU 0 = dense in ≤ 5.5 h bursts
with 20 min cooldowns (`E33N_BURST_H`, heat rule). Expected ~13–19 h per arm on a 3090. AWS stays for stage 2
(100M) and stage 3 (300M). Runs: `~/dev/MrCogito-lsa/Cache/e33n_runs/e33n_s30_<arm>/` on each server.

## Risks seen while building
- The output layer is untied and full width: with a 50k vocabulary it is a large share of compute at s30.
- No whole-model compile; host syncs in the notebook path (mask building, writer diagnostics) cost throughput —
  measured in stage 0 before any long run.
- Uncached RULER decoding: cost grows with length × answer tokens; capped at 32 new tokens and 32k length.
- On-demand H100 capacity in us-east-1 was unavailable on 2026-11-04 22:00 UTC in every zone (launch retries).
