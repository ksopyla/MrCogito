# E10c — Small Nonzero Memory Gates Implementation Plan

- **Spec:** [E10c_nonzero_memory_gates.md](E10c_nonzero_memory_gates.md) · **Status:** implemented
- **Authored by:** `implementation-plan` · for → `research-implement`

## 1. Source & fit

- **Origin:** E10 initialized both serial residual gates at zero. At a zero read gate, the
  concept state and read normalization receive no loss gradient; at a zero write gate, BiXT
  parameters receive no gradient until `alpha` opens. E10b repairs read scale but deliberately
  leaves this staged bootstrap intact.
- **Synthesis verdict:** **Adapt.** The
  [Block-Recurrent Transformer](https://proceedings.neurips.cc/paper_files/paper/2022/file/d6e0bbb9fc3f4c10950052ec2359355c-Paper-Conference.pdf)
  warns that ignored recurrence is a stable local optimum and uses small/remember-biased
  initialization. E10c keeps Flamingo-style bounded `tanh` gates but starts the complete serial
  path at a small 1% scale.
- **Architecture mapping (ONE):** concept read/write initialization policy.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `BackboneConceptConfig.read_gate_init/write_gate_init` | reuse existing fields | `nn/backbone_concept_lm.py` |
| `GlobalLayerWithConceptRead.gate` | reuse existing initialization | `nn/backbone_concept_lm.py` |
| `ConceptWriteHead.alpha` | reuse existing initialization | `nn/backbone_concept_lm.py` |
| `ModelArguments` | expose the two fields with zero defaults | `training/concept_pretraining_args.py` |
| backbone factory | pass model args into config | `training/concept_pretraining_factories.py` |
| generic launcher | add `READ_GATE_INIT` / `WRITE_GATE_INIT` | `scripts/train_concept_pretraining_multigpu.sh` |
| W&B config | log both raw and effective initial values | `training/train_concept_pretraining.py` |

No new module, model family, objective, optimizer, dataset, or launcher is required.

## 3. Forward pass

Symbols: `B`=batch, `S`=2048, `K`=512, `C`=128, `H`=1152.

```text
read_l = tanh(0.01) * ConceptRead_l(x, RMSNorm(z))
hidden = GemmaLayer(hidden) + read_l

write = tanh(0.01) * RMSNorm(BiXT(RMSNorm(z), RMSNorm(h_block)))
z_next = z + write
```

Shapes and O(C·N) complexity are unchanged. The initial effective scale is
`tanh(0.01) ≈ 0.0099997`.

## 4. Inputs & data

Identical to E10b: current Gemma-tokenized `smollm3_inspired_2k_e05` manifest,
`DataCollatorForCausalLM`, seq 2048, no chat formatting or preprocessing changes.

## 5. Loss & training objective

Unchanged full next-token CE. Reuse real/shuffle/zero/static/one-block evaluations.
No auxiliary loss or position weighting.

## 6. Config & launch

Add backward-compatible fields:

```python
# sketch
class ModelArguments:
    read_gate_init: float = 0.0
    write_gate_init: float = 0.0
```

Wire both into `BackboneConceptConfig`. The generic launcher defines:

```bash
READ_GATE_INIT="${READ_GATE_INIT:-0.0}"
WRITE_GATE_INIT="${WRITE_GATE_INIT:-0.0}"
```

and passes them only when `BACKBONE_MODEL` is nonempty.

Launch after E10b evaluation:

```bash
EXPERIMENT_ID=E10c \
READ_CONCEPT_NORM=true \
READ_GATE_INIT=0.01 \
WRITE_GATE_INIT=0.01 \
TARGET_TOKENS=50000000 \
WARMUP_STEPS=50 \
SKIP_PRETOKENIZE=1 \
bash scripts/launch_e10.sh
```

Keep `OPTIMIZER=adam`, `LEARNING_RATE=1e-4`, and `WEIGHT_DECAY=0`.

## 7. Tests & smoke

- Extend `tests/test_backbone_concept_lm.py` helper to accept config overrides.
- Assert configured raw gate parameters equal 0.01 and effective metrics equal `tanh(0.01)`.
- Assert default zero values preserve all existing zero-init equivalence tests.
- On a first backward pass with both gates at 0.01, require finite nonzero gradients for
  `concept_init`, at least one BiXT projection, each read RMSNorm gain, gate scalars, and LoRA-B.
- Extend factory/parser tests to prove values reach `BackboneConceptConfig`.
- Extend launcher contract tests to prove env → CLI propagation.

Commands:

```bash
uv run pytest tests/test_backbone_concept_lm.py \
  tests/test_concept_pretraining_parameter_flow.py \
  tests/test_training_launcher_parameter_flow.py -v
```

## 8. Risks & tradeoffs

- **1% may still be too small.** Cheapest signal: first-step gradient ratios and 25M recurrence
  deltas. Fallback: a separate 0.03 gate-init experiment, not an in-run change.
- **Both gates change together.** This tests the coordinated serial-path policy requested by the
  user; it cannot attribute a gain to read versus write gate independently.
- **Random BiXT writes can perturb memory immediately.** The bounded 1% scale limits this;
  local CE is the kill guard.
- **Gate magnitude is not utility.** Success remains real-vs-static/shuffle CE, not merely gates
  becoming nonzero.
