# E18 main run on AWS — launch plan (draft 2026-09-07, awaiting pilot gates + explicit go)

Companion to [E18](E18_perceiver_ar_v2_baseline.md) (gates M1–M4) and
[E18_plan](E18_perceiver_ar_v2_baseline_plan.md) (implementation). This page is the one-pager
for the go/no-go: what gets trained, on what, for how much, and what must be true first.
Nothing here is launched until the user says go.

## 0. Preconditions (all must hold)

| # | Precondition | Owner | Status 2026-09-07 |
|---|---|---|---|
| 1 | Pilot gates: P1 ✅ (0.1%), P2 ✅ (copy@32k 99.9998%), P4 ✅ (1.02×); **P3 re-scored on the geometry arms** (reach ablation, spec amendment 2026-09-09) — stage B was an optimizer regression, not a P3 result | me, arms A/C/B running on Polonez (`Cache/jobs/e18_geometry_arms.sh`) | ⏳ |
| 2 | P2 copy task solved (or explained) | me, rerun from `checkpoint-500` when GPUs free | two runs stuck at ln(256); deferred |
| 3 | Packed-document training path (`BATCH_PACKING_MODE=pack`) | done, tested (packed == unpacked per-token loss) | ✅ `e7be9f6` |
| 4 | Multi-node launch (`NUM_MACHINES`, `MACHINE_RANK`, `MAIN_PROCESS_IP`) | done in the generic launcher; untested on real multi-node | ✅ `a663137`, needs a 2-node dry run |
| 5 | Main-run recipe `data/mix_recipes/e18_main_stage1_v1.json` | **Nemotron-first** (user decision 2026-09-07): 13 of 15 sources are Nemotron (CC-v2.1 web, Code-v1 synthetic code, CC-Math-v1, Specialized-v1/v1.2, SFT-v1); only the long-document tier (FinePDFs, PG-19) is external. Row weights need re-check with `manifest_token_stats.py` after pretokenization | ✅ v1 |
| 6 | HF access: CC-v2.1 / CC-v2 / CC-Math-v1 / Code-v1 / SFT-v1 / Specialized ✅; **Nemotron-CC-Code-v1 and Nemotron-Pretraining-Code-v2 requested 2026-09-07** (Company Meridian21Lab, krzysztof.sopyla@meridian21lab.com, NVIDIA Data Agreement for Model Training accepted) | NVIDIA approval (manual) | pending — swap in per the recipe's `pending_swap` note |
| 7 | AWS: p5 Capacity Block quota, S3 bucket, IAM for the nodes, EFA-enabled AMI with CUDA 12.8+ | **user / AWS account** | pending |
| 8 | FA3 backend (`ATTN_BACKEND=flash`) validated on one H100 against flex (loss identical to 1e-3) | me, first hour of the block | pending |
| 9 | PG-19 test/validation decontamination of the training pool | me, before any reported PG-19 number | pending |

## 1. Model (frozen unless the pilot says otherwise)

| knob | value | note |
|---|---|---|
| `HIDDEN_SIZE / INTERMEDIATE_SIZE` | 1280 / 3456 | SwiGLU |
| `NUM_LAYERS` (stack) / `PAR_PRE_LAYERS` / `PAR_GLOBAL_LAYERS` | 20 / 2 / 1 | 23 attention layers total |
| `PAR_BLOCK / PAR_PRE_WINDOW` | 4096 / 1024 | window-N stack; one full-causal read |
| `NUM_KV_HEADS / HEAD_DIM` | 2 / 128 | GQA, 10 query heads |
| `TOKEN_EMBEDDING_DIM / PAR_NGRAM_BUCKETS` | 256 / 131072 | tiny hashed n-gram input |
| `PAR_VALUE_EMBED_LAYERS` | 0,7,14 | value embeddings |
| params | ≈ 540M dense-equivalent + 33M hashed tables | `analytic_param_count` |
| `ATTN_BACKEND` | flash (FA3) once validated, else flex | flex is the reference |
| optimizer | Muon lr 0.01 / AdamW 2e-4, wd 0.1, clip 0.5, WSD schedule (10% decay) | pilot-calibrated triple |

Dense control: same everything with `PAR_MODE=dense` for stage 1 only (M1/M2 need it).

## 2. Context schedule and budget

Cost basis: p5.48xlarge Capacity Block **$41.5/node-h** (8×H100 80 GB); H100 bf16 ≈ 989 TFLOPs;
FLOPs/token ≈ 6·N for the 8k–32k stages, ≈ 6.9 GFLOP at 256k and ≈ 12 GFLOP at 512k
(feasibility note §4). Two MFU columns because long-context flex kernels will not reach 35%.

| stage | seq | tokens | data | H100-h @35% / @20% | node-h @20% | $ @20% |
|---|---|---|---|---|---|---|
| 1a | 8k → 32k (packed) | 300B | `e18_main_stage1_v1` | 780 / 1,370 | 170 | 7.1K |
| 1b dense control | 8k → 32k (packed) | 300B | same | 780 / 1,370 | 170 | 7.1K |
| 2 | 256k (packed, doc masks) | 50B | 1a recipe re-weighted to 35% long tier + ProLong-64K/512K if terms allow | 280 / 490 | 60 | 2.5K |
| 3 | 512k (YaRN, batch 1) | 20B | ProLong-512K + repo-level code | 190 / 330 | 40 | 1.7K |
| evals + 1M demo | — | — | RULER, NIAH grid, PG-19, WikiText-103, lm-eval | ~40 node-h (p5e for 1M) | 40 | 1.9K |
| **total** | | | | | **~480 node-h** | **~$20K** |

Headroom: the remaining ~$70K covers one full rerun of 1a (if the pilot forces a config change), a
1T-token version of stage 1 (~$24K at 20% MFU) if the 300B result is close to the SmolLM2-360M
bar, and storage/egress. Jean Zay (12.5k H100-h normalized) is reserved for E19–E21.

Wall time on the 300B stage: one node at 20% MFU ≈ 490k tok/s → **7 days**; two nodes ≈ 3.5 days.
Proposal: one **2-node × 7-day** block runs 1a and 1b in parallel (one node each), then a
**2-node × 3-day** block (p5e/H200 preferred) for stages 2–3 and the 1M demo.

## 3. Launch sequence (per block)

1. **Node prep (both nodes):** clone `MrCogito` at a tagged commit (`train/e18_main_stage1`), `uv sync`,
   `HF_TOKEN` with Nemotron access, `WANDB_API_KEY`, mount a 4 TB gp3 volume at `/data`
   (`HF_HOME=/data/hf_home`, `DATASETS_TOK_DIR=/data/hf_home/datasets_tok_smollm3_32k`), EFA
   env (`FI_PROVIDER=efa`, `NCCL_PROTO=simple`).
2. **Pretokenize once** on node 0 (CPU-bound, ~6–10 h for ~350B tokens at 96 vCPU):
   `PRETOKENIZE_MIX=e18_main_stage1_v1 TOKENIZER_NAME=HuggingFaceTB/SmolLM3-3B MAX_SEQ_LENGTH=32768
   uv run python scripts/pretokenize_mix.py …` → `aws s3 sync` the tok tree; node 1 pulls it.
   Then `scripts/manifest_token_stats.py` to confirm token shares; adjust weights in the manifest.
3. **FA3 check (1 GPU, 10 min):** `ATTN_BACKEND=flash` vs `flex` on the same batch; require
   |Δloss| < 1e-3 and no NaN in 50 steps. Fall back to flex if it fails.
4. **2-node dry run (30 min):** `MAX_STEPS=50` with `NUM_MACHINES=2 MACHINE_RANK={0,1}
   MAIN_PROCESS_IP=<node0>`; confirm tokens/s scales and checkpoint save + resume works.
5. **Stage 1a / 1b:** each node single-machine (8 GPUs), `BATCH_PACKING_MODE=pack`, seq 32k from the
   start (packing makes 8k vs 32k a per-step cost question only; 32k from the start avoids a
   context-extension step), `PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=8` → 2M tokens/step,
   `TARGET_TOKENS=3e11`, `SAVE_STEPS` every ~2 h, `SAVE_TOTAL_LIMIT=3`, a cron `aws s3 sync` of
   the run dir every 30 min, `AUTO_INTERVALS=1`, eval on PG-19 validation buckets + passkey.
6. **Stage 2 → 3 — context-extension protocol (hard rule, from the stage-B regression 2026-09-08):**
   never restart a converged model at the peak lr with a fresh optimizer. Stage B did exactly that
   (weights-only `MODEL_NAME_OR_PATH`, Muon 0.01, 500-step re-warmup, no decay) and came out uniformly
   ~2% worse at every position after 0.5B tokens. For each extension stage: (a) prefer
   `--resume_from_checkpoint` so Muon momentum carries over, with `max_steps` extended for the new
   budget; if the dataset/seq change makes resume impractical, warm-start weights at **≤ 20% of peak lr**
   (Muon 2e-3 / AdamW 4e-5) with ≤ 100 warmup steps and a **decay to zero** over the stage (WSD tail or
   cosine); (b) `BATCH_PACKING_MODE=pack` so every sequence is full-length; (c) keep ~⅔ short data in the
   mix to protect short-context loss; (d) before committing the stage, run the reach probe on the
   warm-start checkpoint at the new length (a model that already extrapolates needs a gentler stage).
   Mechanics: warm start from 1a `final`, `MAX_SEQ_LENGTH=262144` then
   `524288` with `ROPE_THETA` raised (YaRN factor recorded in the run report), 2 nodes via
   `NUM_MACHINES=2`, `ATTN_PAD_MULTIPLE=4096`.
7. **Evals** per M1–M4, then `experiment-track`.

## 4. Decision points during the run

- **1a at 30B tokens (~day 1):** loss within 3% of the dense control's curve at equal tokens, else
  stop both and revisit `PAR_BLOCK` / `PAR_PRE_WINDOW` (kill criterion K1 of the spec).
- **1a at 100B:** lm-eval quick set (HellaSwag, ARC-E, PIQA) vs SmolLM2-135M at equal tokens;
  a miss by > 3 points triggers the 1T-token decision rather than a config change.
- **End of 1a:** M1 + long-position buckets vs dense (M2 first half). Only then buy the second block.

## 5. Risks specific to AWS

| risk | mitigation |
|---|---|
| Capacity Block not available in the region/dates | book 2–3 weeks ahead; p4de (A100) fallback is 2.3× cheaper per hour but ~2.5× slower and has no FA3 |
| Node loss mid-run (blocks are not interruptible, but hardware fails) | S3 sync every 30 min; `--resume_from_checkpoint` tested in step 4 |
| Data pipeline slower than GPUs at 32k packed | pretokenized + `PackedDataset` (arrow random access); measure `perf/real_tokens_per_second` in the dry run; raise `DATALOADER_NUM_WORKERS` (no fork issue on EC2) |
| Nemotron synthetic subsets' pass-through licence terms | capped at 15% of tokens; recorded in the model card |
| Recompiles / dynamo cache blowups at 256k with flex | `attn_pad_multiple` keeps shapes static; per-forward mask memo landed; dynamo cache limits raised |
