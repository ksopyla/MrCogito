# E16a — Shared Depth-Recurrent Workspace Optimizer A/B Implementation Plan

- **Spec:** [E16a_muon_optimizer_ab.md](E16a_muon_optimizer_ab.md) · **Status:** done / failed short-ctx gate
- **Authored by:** `implementation-plan` · for → `research-implement`

> Scope is exactly the spec's optimizer-recipe comparison. Architecture, data,
> context, objective, initialization, and token exposure are unchanged between arms.

## 1. Source & fit

- **Origin:** the completed E16 run showed stable training and healthy concepts but
  null persistent-state ablations at 50M. The user hypothesizes that optimization
  and exposure, rather than architecture, are limiting.
- **Repository evidence:** stabilized Muon previously lowered E05 CE much faster,
  but its long run collapsed concept geometry. E16a therefore treats causal
  ablations and RankMe—not CE—as the deciding metrics.
- **Architecture mapping (ONE):** optimization only. The E16 encoding/recurrent
  bottleneck and decoder path are byte-unchanged.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `BackboneConceptLM` | reuse as-is | `nn/backbone_concept_lm.py` |
| `ConceptWriteHead` / four depth gates | reuse as-is | `nn/backbone_concept_lm.py` |
| `PerceiverDenoiseTrainer.create_optimizer` | reuse as-is | `training/concept_pretraining_trainer.py` |
| `Muon` | reuse as-is | `nn/muon.py` |
| canonical generic runner | reuse as-is | `scripts/train_concept_pretraining_multigpu.sh` |
| E10 backbone protocol wrapper | delegate to it | `scripts/launch_e10.sh` |
| E16a protocol wrapper | new thin config wrapper | `scripts/launch_e16a.sh` |
| E16a unattended A/B chain | new orchestration only | `scripts/launch_e16a_pipeline.sh` |
| launcher contract tests | extend | `tests/test_training_launcher_parameter_flow.py` |

No model, loss, collator, checkpoint, or evaluation-routing code changes.

## 3. Forward pass

Symbols: `B`=microbatch, `N=2048`, `K=512`, `C=128`, `H=1152`,
`L=26`, `V=262144`.

```text
(B,N) → split into four K-token blocks

For each block b:
  token window [B,≤1024,H]
  z_b [B,C,H]

  Gemma layers 1..6  → concept read at L6  → tied write → z_b^(6)
  Gemma layers 7..12 → concept read at L12 → tied write → z_b^(12)
  Gemma layers 13..18→ concept read at L18 → tied write → z_b^(18)
  Gemma layers 19..24→ concept read at L24 → tied write → z_b^(24)
  remaining layers; z_(b+1)=z_b^(24)

  hidden [B,K,H] → chunked frozen LM head → next-token CE
```

Both arms execute this identical graph. Only parameter-update construction differs:

- Adam: LoRA groups at `1e-4`; concept-memory groups at `3e-4`.
- Muon: eligible 2D matrices use orthogonalized updates at the calibrated matrix
  LR; 1D and high-aspect parameters use the optimizer's AdamW fallback at `2e-4`.
  `concept_memory_lr` must be unset because the current trainer deliberately rejects
  differential role-based LR with Muon.

## 4. Inputs & data

- **Dataset:** existing immutable
  `${DATASETS_TOK_DIR}/smollm3_inspired_2k_e05_gemma_manifest.json`.
- **Tokenizer:** `google/gemma-3-1b-pt`.
- **Collator/objective:** existing `backbone_concept` causal-LM factory route;
  no preprocessing, packing, masking, split, or label changes.
- **Sequence:** N=2048 and K=512, with E16's one-block token carry.
- Both arms use seed 42 and the same manifest/order contract.

## 5. Loss and training objective

- Existing token-count-normalized next-token CE in `BackboneConceptLM`.
- No auxiliary objective or changed loss weighting.
- Existing live eval computes real/static/shuffle/zero/one-block concept ablations,
  concept geometry, and depth-gate telemetry at matched token intervals.

## 6. Config and launch

### Thin protocol wrapper

Add `scripts/launch_e16a.sh`. It pins all E16 invariants, branches only on
`OPTIMIZER`, then delegates to `scripts/launch_e10.sh`.

Common:

```bash
EXPERIMENT_ID=E16a
CONCEPT_IO_MODE=shared_depth_recurrent
READ_CONCEPT_NORM=true
READ_GATE_INIT=0.01
WRITE_GATE_INIT=0.01
TARGET_TOKENS=100000000
WARMUP_STEPS=100
MAX_GRAD_NORM=0.5
AUTO_INTERVALS=1
SAVE_TOTAL_LIMIT=12
SKIP_PRETOKENIZE=1
```

Adam:

```bash
OPTIMIZER=adam
LEARNING_RATE=1e-4
CONCEPT_MEMORY_LR=3e-4
WEIGHT_DECAY=0
```

Muon:

```bash
OPTIMIZER=muon
LEARNING_RATE=0.01
CONCEPT_MEMORY_LR=
MUON_ADAMW_LR=2e-4
MUON_MOMENTUM=0.95
WEIGHT_DECAY=0.1
```

The wrapper must reject optimizer values other than `adam|muon`. Environment
overrides remain possible for the pre-registered Muon calibration LR fallback and
short target budget.

### Calibration

Before either full arm, run a fresh Muon stability calibration:

```bash
EXPERIMENT_ID=E16a_calibration OPTIMIZER=muon \
TARGET_TOKENS=10000000 WARMUP_STEPS=50 \
LR_SCHEDULER_TYPE=constant_with_warmup \
WANDB_MODE=disabled bash scripts/launch_e16a.sh
```

Inspect finite loss/gradients, sustained peak-LR behavior, RankMe, gate activity,
and VRAM. If matrix LR 0.01 fails the spec's stability gate, retry once with
`LEARNING_RATE=0.005`; otherwise freeze 0.01.

### Unattended full pair

Add `scripts/launch_e16a_pipeline.sh`:

```text
set -euo pipefail
verify the immutable manifest exists
run OPTIMIZER=adam TARGET_TOKENS=100M launch_e16a.sh
only after exit 0, run OPTIMIZER=muon TARGET_TOKENS=100M launch_e16a.sh
emit stable arm-start/arm-complete sentinels
```

This serializes the arms on Odra's three GPUs and prevents Muon from starting after
an Adam crash. The selected calibrated Muon LR is passed as a pipeline environment
value; the script does not tune.

## 7. Tests and smoke

- Extend `tests/test_training_launcher_parameter_flow.py` to assert:
  - E16a delegates to `launch_e10.sh`;
  - shared-depth mode, gates, 2K context, 100M budget, and seed are pinned;
  - Adam forwards differential concept LR and wd=0;
  - Muon forwards matrix/fallback LR, momentum, wd=0.1, and an empty
    `CONCEPT_MEMORY_LR`;
  - invalid optimizer values fail;
  - pipeline orders Adam before Muon and uses `&&`/`set -e` semantics.
- Reuse `tests/test_optimizer_muon.py` for Muon construction and backbone support.
- Run:
  ```bash
  bash -n scripts/launch_e16a.sh scripts/launch_e16a_pipeline.sh
  uv run pytest tests/test_optimizer_muon.py \
    tests/test_training_launcher_parameter_flow.py -v
  ```
- Remote calibration is the required CUDA smoke because local MPS does not validate
  Muon's DDP/VRAM/throughput behavior.

## 8. Risks and tradeoffs

- **Muon concept collapse:** E05's principal warning. Cheapest signal: RankMe plus
  static/shuffle deltas at every 10% interval. Lower CE cannot override a collapse.
- **Different wd/LRs:** these are required parts of each practical optimizer recipe;
  E16a compares deployable recipes, not a pure optimizer equation. Do not attribute
  any result specifically to weight decay or one LR.
- **Muon ignores `concept_memory_lr`:** intentional current contract. The fallback
  and matrix branches route by tensor shape, so gate/norm scalars and matrices do
  not reproduce Adam's role-based 3× LR.
- **Delayed instability:** use sustained-LR calibration rather than a short cosine
  smoke that decays before the failure region.
- **Sequential wall time:** approximately 12–15 hours total on Odra. The pipeline
  preserves clean GPU ownership and comparable system conditions.
- **Negative result:** if neither 100M arm reaches causal use, E16b should not be
  justified by CE alone; the result instead rejects optimizer choice as the missing
  ingredient under the 2K/plain-CE protocol.

## 9. Implementation sequence

1. Add and contract-test the thin E16a wrapper.
2. Add and syntax-test the sequential pipeline.
3. Run targeted local tests.
4. Sync to Odra and run Muon sustained-LR calibration.
5. Freeze the passing pre-registered Muon LR.
6. Launch the unattended Adam→Muon full pipeline in a named Byobu session.
