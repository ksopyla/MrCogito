# E10d — Differential Concept-Memory LR Implementation Plan

- **Spec:** [E10d_differential_concept_lr.md](E10d_differential_concept_lr.md) · **Status:** implemented
- **Authored by:** `implementation-plan` · for → `research-implement`

## 1. Source & fit

- **Origin:** E10 measured concept/BiXT relative update scales 1e4–1e5 below LoRA-B. E10b
  repairs the activation scale and E10c removes the zero-gate dead zone; E10d asks the narrower
  remaining question: whether the new memory modules need a higher LR than LoRA.
- **Synthesis verdict:** **Adapt.** Adapter training commonly uses higher LRs for newly initialized
  modules, and [LoRA+](https://proceedings.mlr.press/v235/hayou24a.html) shows that uniform
  learning rates need not be optimal across low-rank factors. We use a conservative 3× concept
  multiplier rather than trying to compensate directly for the raw 1e4–1e5 gradient ratio.
- **Architecture mapping (ONE):** optimizer parameter grouping. The forward pass is unchanged.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `OptimizerArguments` | add optional `concept_memory_lr` | `training/concept_pretraining_args.py` |
| `PerceiverDenoiseTrainer.create_optimizer` | add backbone AdamW grouping path | `training/concept_pretraining_trainer.py` |
| HF decay/no-decay classification | reuse through `Trainer.get_decay_parameter_names` | `training/concept_pretraining_trainer.py` |
| trainer construction | pass the optional LR | `training/train_concept_pretraining.py` |
| generic launcher | pass `CONCEPT_MEMORY_LR` only when set | `scripts/train_concept_pretraining_multigpu.sh` |
| W&B config | log requested/effective group LRs | `training/train_concept_pretraining.py` |

No optimizer fork, model change, new launcher, or objective change is required.

## 3. Forward pass

Identical to E10c:

```text
tokens → blockwise Gemma + normalized 1%-gated concept reads
       → 1%-gated recurrent BiXT write
       → next-token CE
```

Only optimizer updates differ:

```text
LoRA A/B:                                  AdamW lr = 1e-4
concept_init, write_head, read gates/norm: AdamW lr = 3e-4
```

## 4. Inputs & data

Identical to E10c: existing Gemma manifest, seq 2048, current mix and collator.

## 5. Loss & training objective

Identical full next-token CE and concept-state ablations. Weight decay remains zero in E10d,
although the reusable optimizer grouping must preserve HF's decay/no-decay split for future use.

## 6. Config & launch

Add:

```python
# sketch
@dataclass
class OptimizerArguments:
    concept_memory_lr: Optional[float] = None
```

`None` calls `super().create_optimizer()` exactly as before. A set value is valid only when:

- checkpoint family is `backbone_concept`
- optimizer choice is `adam`
- all trainable parameters classify as either LoRA or explicit concept-memory parameters

Parameter classification:

```python
# sketch
def is_lora(name):
    return "lora_A" in name or "lora_B" in name

def is_concept_memory(name):
    return (
        name == "concept_init"
        or name.startswith("write_head.")
        or name.endswith(".gate")
        or ".read_branch.concept_norm." in name
    )
```

Fail fast on an unknown trainable parameter instead of silently assigning the wrong LR. Build
decay/no-decay subgroups at each LR using `Trainer.get_decay_parameter_names`.

For Muon, raise a clear `ValueError`: differential concept LR is an AdamW-only capability.

The launcher adds an optional argument:

```bash
CONCEPT_MEMORY_LR="${CONCEPT_MEMORY_LR:-}"
```

and passes `--concept_memory_lr` only when nonempty.

Launch:

```bash
EXPERIMENT_ID=E10d \
READ_CONCEPT_NORM=true \
READ_GATE_INIT=0.01 \
WRITE_GATE_INIT=0.01 \
CONCEPT_MEMORY_LR=3e-4 \
LEARNING_RATE=1e-4 \
OPTIMIZER=adam \
TARGET_TOKENS=50000000 \
WARMUP_STEPS=50 \
SKIP_PRETOKENIZE=1 \
bash scripts/launch_e10.sh
```

## 7. Tests & smoke

Create/extend optimizer tests:

1. `concept_memory_lr=None` retains the HF AdamW path.
2. E10 tiny model with `3e-4` creates concept decay/no-decay groups at 3e-4 and LoRA groups
   at 1e-4; all trainable parameters appear exactly once.
3. Group names include global wrapped-layer LoRA paths and all concept modules.
4. One CPU optimizer step changes both a LoRA parameter with nonzero grad and a concept parameter.
5. Unknown trainable parameters fail fast.
6. `optimizer_choice=muon` plus `concept_memory_lr` raises a targeted error.
7. Parser, launcher, trainer-construction, and W&B tests preserve/log the value.

Commands:

```bash
uv run pytest tests/test_optimizer_muon.py \
  tests/test_concept_pretraining_main.py \
  tests/test_concept_pretraining_parameter_flow.py \
  tests/test_training_launcher_parameter_flow.py -v
```

## 8. Risks & tradeoffs

- **The groups may optimize at different effective scales despite a 3× nominal LR.**
  Cheapest signal: recurrence deltas, gate trajectories, and first-step gradient probes.
  Fallback: tune the multiplier in a separate experiment.
- **Custom grouping can drift from future parameter names.** Fail-fast exhaustive partitioning
  makes drift visible instead of silently applying the base LR.
- **Higher memory LR may overfit or destabilize gates.** Bounded gates, clipping 0.5, local CE,
  RankMe, and the 25M checkpoint are the safeguards.
- **Muon is intentionally unsupported.** E05 showed that lower CE under Muon can coincide with
  worse concept use; optimizer-family testing remains a separate experiment.
