# E16b — Long-Context Muon 1B Implementation Plan

- **Spec:** [E16b_longctx_muon_1b.md](E16b_longctx_muon_1b.md) · **Status:** done / success (Tier-1 2026-07-25) · validated long-context path
- **Authored by:** `implementation-plan` · for → `research-implement`

> Compound scale-up authorized by the user. Architecture stays E16; change the
> operating regime (4K + long mix + Muon + 1B).

## 1. Source & fit

- **Origin:** E16/E16a kept healthy concepts but null-to-tiny beyond-local causal
  use under 2K plain CE. User rejects further tiny calibrations and authorizes a
  bold long-context / high-budget Muon run.
- **Architecture mapping:** data + sequence length + optimization budget.
  Forward graph unchanged (`shared_depth_recurrent`).

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `BackboneConceptLM` shared-depth path | reuse | `nn/backbone_concept_lm.py` |
| Muon optimizer | reuse E16a recipe | `nn/muon.py`, trainer |
| Recipe loader / pretokenize | reuse | `data/dataset_preprocess.py`, `scripts/pretokenize_mix.py` |
| 4K mix recipe | **new** | `data/mix_recipes/e16b_long_4k_v1.json` |
| E10 launcher | reuse with overrides | `scripts/launch_e10.sh` |
| E16b protocol wrapper | **new** | `scripts/launch_e16b.sh` |
| Launcher tests | extend | `tests/test_training_launcher_parameter_flow.py` |

## 3. Forward pass

Symbols: `B` microbatch, `N=4096`, `K=512`, `C=128`, `H=1152`.

```text
8 blocks of K=512 tokens
z starts as concept_init
for each block:
  Gemma layers with concept read+tied write at L6/12/18/24
  z carried to next block
loss = next-token CE over the sequence
```

Beyond-local ablation region for gates: positions ≥2048 (half the sequence),
matching the longer carry distance.

## 4. Inputs & data

- Recipe: `e16b_long_4k_v1` (FinePDFs 0.30, PG19 0.18, DCLM 0.15, Wikipedia-en
  0.12, FineWeb-Edu 0.10, Stack-Edu 0.10, FineMath 0.05).
- Pretokenize on Odra:
  ```bash
  uv run python scripts/pretokenize_mix.py \
    --mix e16b_long_4k_v1 \
    --tokenizer google/gemma-3-1b-pt \
    --max_seq_length 4096 \
    --cache_dir "$DATASETS_TOK_DIR" \
    --raw_dir "$RAW_ARCHIVE_DIR" \
    --raw_archive_dir "$RAW_ARCHIVE_DIR" \
    --manifest "$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json" \
    --objective causal_lm --seed 42 \
    --train_num_proc 8 --test_num_proc 4 --jobs 1
  ```
- Training uses `SKIP_PRETOKENIZE=1` + that manifest.

## 5. Loss & objective

Unchanged causal-LM CE. Live ablations / RankMe / gates unchanged.

## 6. Config & launch

`scripts/launch_e16b.sh` pins:

```bash
EXPERIMENT_ID=E16b
CONCEPT_IO_MODE=shared_depth_recurrent
READ_CONCEPT_NORM=true
READ_GATE_INIT=0.01
WRITE_GATE_INIT=0.01
OPTIMIZER=muon
LEARNING_RATE=0.01
MUON_ADAMW_LR=2e-4
MUON_MOMENTUM=0.95
WEIGHT_DECAY=0.1
CONCEPT_MEMORY_LR=
MAX_SEQ_LENGTH=4096          # must override launch_e10's hardcoded 2048
PRETOKENIZE_MIX=e16b_long_4k_v1
TARGET_TOKENS=1000000000
WARMUP_STEPS=500
AUTO_INTERVALS=1
SAVE_TOTAL_LIMIT=12
SKIP_PRETOKENIZE=1
```

`launch_e10.sh` currently forces `MAX_SEQ_LENGTH=2048`. Fix: change to
`${MAX_SEQ_LENGTH:-2048}` and allow `PRETOKENIZE_MIX` / `MANIFEST` overrides
(already partially supported). Same for any other hard pins that block 4K.

Microbatch: calibrate on Odra starting from `PER_DEVICE_BATCH_SIZE=4` at 4K;
preserve effective batch 72 via gradient accumulation.

## 7. Tests & smoke

- Recipe loads and weights sum to 1.
- `launch_e16b` pins Muon + 4K + mix + 1B.
- `launch_e10` accepts `MAX_SEQ_LENGTH=4096` override.
- Remote: short Muon 4K calibration (~10–20M tokens or ~50–100 steps) for VRAM
  before the 1B run.

## 8. Risks

- **VRAM:** 4K + four depth writes may force small microbatch; calibrate first.
- **Wall time:** ~several days on Odra for 1B; monitor and do not kill on early
  small deltas.
- **Multi-factor confound:** accepted; the research question is whether *this
  regime* unlocks concept use, not which single factor did.
- **8K deferred:** if 4K calibration has ≥6 GiB free/GPU and throughput is
  healthy, a later E16c/8K arm can reuse the same launcher.

## 9. Sequence

1. Patch `launch_e10.sh` seq-length override; add `launch_e16b.sh` + tests.
2. Sync to Odra; start 4K Gemma pretokenize in Byobu.
3. VRAM calibrate Muon@4K; freeze microbatch/accum.
4. Launch 1B training in Byobu `E16b`.
