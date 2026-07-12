# E10e — Calibrated Concept Memory at the Original 100M Budget Implementation Plan

- **Spec:** [E10e_calibrated_memory_100m.md](E10e_calibrated_memory_100m.md) · **Status:** approved
- **Authored by:** `implementation-plan` · configuration-only extension of E10d

## 1. Source & fit

E10d had the strongest short-horizon recurrence signal of the calibration ladder but failed its
pre-registered 25M improvement criterion. E10e is a deliberate, user-authorized endurance
diagnostic: it changes only the target training exposure so the calibrated path can be observed
under the original E10 100M-token and 1,449-step schedule.

## 2. Reuse map

Reuse E10d unchanged:

| Component | Action |
|---|---|
| `BackboneConceptLM` | normalized read branch and 0.01 read/write gate initialization |
| `PerceiverDenoiseTrainer` | differential AdamW groups |
| `train_concept_pretraining.py` | existing validation and W&B configuration |
| `launch_e10.sh` | existing Gemma/data/protocol wrapper |

No code or test changes are required.

## 3. Forward pass, inputs, and loss

Identical to E10d: Gemma causal next-token CE, blockwise recurrent write every 512 tokens, and
normalized 1%-gated reads in the four global layers. Data is the existing Gemma-tokenized
`smollm3_inspired_2k_e05` manifest at seq 2048.

## 4. Config and launch

The sole configuration difference from E10d is:

```text
TARGET_TOKENS: 50,000,000 → 100,000,000
```

This sets the schedule to approximately 1,449 optimizer steps. Keep warmup at 50 steps; preserve
the original E10 budget's cosine phase at matched token exposure.

```bash
EXPERIMENT_ID=E10e READ_CONCEPT_NORM=true \
READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 \
CONCEPT_MEMORY_LR=3e-4 LEARNING_RATE=1e-4 OPTIMIZER=adam \
WEIGHT_DECAY=0 TARGET_TOKENS=100000000 WARMUP_STEPS=50 \
SKIP_PRETOKENIZE=1 MAX_SEQ_LENGTH=2048 CONCEPT_NUM=128 \
bash scripts/launch_e10.sh
```

## 5. Monitoring and decision

- Verify W&B config: `experiment_id=E10e`, read norm enabled, both gate inits 0.01,
  `concept_memory_lr=3e-4`, LoRA LR `1e-4`, seed 42, effective batch 72, and approximately
  1,449 estimated steps.
- Preserve checkpoints at the normal ~10% cadence.
- At checkpoint ~720 (~50M target tokens), apply the frozen kill criterion from the spec.
- At the final checkpoint, evaluate recurring state under real/static/shuffle/zero/one-block
  ablations and compare with E10 and E10d at matched exposure.

## 6. Risks

- The failed 25M E10d gate means 100M is an explicit endurance diagnostic, not an expected
  recovery; do not reinterpret lower CE or larger gates as concept utility.
- The 100M schedule is intentionally different from E10d's 50M schedule. This repairs the
  E10-versus-recovery cosine-phase comparability issue but means the run is not a pure
  token-matched E10d continuation.
