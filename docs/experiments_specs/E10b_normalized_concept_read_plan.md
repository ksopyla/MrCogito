# E10b — Normalized Concept Read Implementation Plan

- **Spec:** [E10b_normalized_concept_read.md](E10b_normalized_concept_read.md) · **Status:** implemented
- **Authored by:** `implementation-plan` · for → `research-implement`

> This plan implements one experimental change: normalize the recurrent concept state
> before Gemma's concept-read K/V projections. The user approved implementation and a
> 50M-token Odra pilot on 2026-07-12.

## 1. Source & fit

- **Origin:** E10's 100M pilot found a connected but severely scale-starved path:
  `concept_init` RMS ≈0.0297, gated reads at 1e-9–1e-4 of the residual stream, concept/BiXT
  relative update scale 1e4–1e5 below LoRA-B, and every recurrent-state ablation <0.001
  nats. See
  `docs/2_Experiments_Registry/run_reports/e10_100m_concept_pilot_20260711.md`.
- **Research fit:** **Adapt.** Frozen-backbone grafts such as
  [Flamingo](https://papers.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf)
  and [LLaMA-Adapter](https://openreview.net/pdf?id=d4UiXAHN2W) protect the pretrained
  residual stream with a gated attachment; the
  [Block-Recurrent Transformer](https://proceedings.neurips.cc/paper_files/paper/2022/file/d6e0bbb9fc3f4c10950052ec2359355c-Paper-Conference.pdf)
  additionally shows that ignored recurrent state is a hard local optimum. We retain E10's
  bounded residual gate but normalize the low-scale recurrent values before the frozen
  attention projections.
- **Architecture mapping (ONE):** decoder/read interface. No change to Gemma's layers,
  concept write recurrence, LoRA targets, data, loss, or optimizer.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `BackboneConceptLM` block loop and recurrent state | reuse as-is | `nn/backbone_concept_lm.py` |
| `ConceptReadBranch` | extend with an optional concept-side RMSNorm per wrapped global layer | `nn/backbone_concept_lm.py` |
| `GlobalLayerWithConceptRead` | extend constructor wiring only | `nn/backbone_concept_lm.py` |
| `BackboneConceptConfig` | extend with safe boolean default | `nn/backbone_concept_lm.py` |
| `ModelArguments` | extend with matching CLI field | `training/concept_pretraining_args.py` |
| backbone model factory | pass the new field into `BackboneConceptConfig` | `training/concept_pretraining_factories.py` |
| generic launcher | expose `READ_CONCEPT_NORM`; E10 wrapper pins it only at launch | `scripts/train_concept_pretraining_multigpu.sh`, `scripts/launch_e10.sh` |
| causal collator, trainer, E10 ablations | reuse as-is | `training/concept_pretraining_factories.py`, `training/concept_pretraining_trainer.py` |

No new model family, registry entry, training script, collator, optimizer, or data recipe is
needed.

## 3. Forward pass

Symbols: `B`=microbatch, `S`=2048 sequence tokens, `K`=512 block tokens,
`N≤2048`=Gemma window including local carry, `C`=128 concepts,
`H`=Gemma hidden size 1152, `V`=Gemma vocabulary.

```text
input_ids [B,S]
  → split into four K-token blocks
  → Gemma local hidden stream x [B,N,H]

recurrent state z [B,C,H]
  → read_concept_norm(z) when enabled                    [B,C,H]
  → Gemma global-layer k_proj/v_proj                     [B,C,Hkv]
  → q(x) attends to concept K/V                          [B,N,H]
  → x + tanh(read_gate_l) * concept_read                 [B,N,H]

block hidden states [B,K,H] + prior z [B,C,H]
  → existing `ConceptWriteHead` / BiXT                   [B,C,H]
  → z_next                                               [B,C,H]

final hidden stream [B,S,H]
  → tied Gemma lm_head                                   [B,S,V]
```

Complexity stays O(S·C) for concept reads plus E10's bounded local Gemma attention. The
change does not introduce full-sequence O(S²) attention.

The normalization is applied before **both** K and V projections. Gemma already normalizes
Q/K internally; applying it only to K would be largely erased by K normalization and would
leave the measured V-scale mismatch untouched.

## 4. Inputs & data

- **Dataset:** existing pretokenized `smollm3_inspired_2k_e05` Gemma manifest.
- **Preprocessing:** reuse `scripts/pretokenize_mix.py` output; no re-tokenization,
  packing, truncation, document splitting, QA synthesis, or chat template.
- **Collator:** reuse `DataCollatorForCausalLM` selected by
  `training/concept_pretraining_factories.py` for `objective=causal_lm`.
- **Masking:** unchanged next-token labels with padding ignored.

This deliberately postpones 4K and a new long-document mix. At fixed 100M tokens, 4K would
roughly halve optimizer updates (~1,449 → ~725), confounding whether normalization or fewer
updates caused the result.

## 5. Loss & training objective

- **Objective:** unchanged full next-token cross-entropy from `BackboneConceptLM.forward`.
- **Concept losses / anchor loss:** remain disabled for the backbone family by
  `training/concept_pretraining_args.py`.
- **Weighting:** unchanged; all non-padding causal targets contribute.
- **Ablations:** reuse trainer-time real/shuffled/static/zero/one-block concept-state
  comparisons, with the pre-registered primary region at positions ≥1024.

No auxiliary objective, answer-only mask, SFT formatting, or counterfactual QA is introduced.

## 6. Config & launch

### Backward-compatible config

Add:

```python
# sketch
@dataclass
class BackboneConceptConfig:
    ...
    read_concept_norm: bool = False
```

Mirror it as `ModelArguments.read_concept_norm: bool = False`, pass it through the existing
backbone factory, and expose:

```bash
READ_CONCEPT_NORM="${READ_CONCEPT_NORM:-false}"
```

The generic launcher passes `--read_concept_norm "$READ_CONCEPT_NORM"` only for the
`backbone_concept` argument group. `scripts/launch_e10.sh` keeps its historical default
unchanged; E10b activates the option through an environment override.

### Module construction and checkpoint behavior

```python
# sketch
class ConceptReadBranch(nn.Module):
    def __init__(self, layer, hidden_size, *, normalize_concepts=False, eps=None):
        self.layer = layer
        self.concept_norm = (
            nn.RMSNorm(hidden_size, eps=eps)
            if normalize_concepts
            else nn.Identity()
        )

    def forward(self, hidden_states, concepts, attention_mask=None):
        z_read = self.concept_norm(concepts.to(hidden_states.dtype))
        ...
        k = attn.k_proj(z_read)
        v = attn.v_proj(z_read)
```

Use Gemma's configured RMS epsilon when available. Instantiate `nn.Identity` when disabled so
the default model has no new parameters or checkpoint keys. Old E10 checkpoints therefore
retain their exact state-dict contract and forward behavior.

### Launch

```bash
EXPERIMENT_ID=E10b \
READ_CONCEPT_NORM=true \
TARGET_TOKENS=50000000 \
WARMUP_STEPS=50 \
SKIP_PRETOKENIZE=1 \
bash scripts/launch_e10.sh
```

Keep `OPTIMIZER=adam`, `LEARNING_RATE=1e-4`, `WEIGHT_DECAY=0`,
`MAX_SEQ_LENGTH=2048`, `CONCEPT_NUM=128`, and the same seed/effective batch.

### Logging identity

Ensure W&B/config logs include `read_concept_norm=true` and use an E10b run/group label.
Do not change the checkpoint evaluation contract (`backbone_concept`, contract version 1).

## 7. Tests & smoke

Add or update `tests/test_backbone_concept_lm.py`:

1. **Backward-compatible default:** `read_concept_norm=False` creates `nn.Identity`, adds no
   state-dict key, and preserves the existing zero-gate backbone-equivalence assertion.
2. **Enabled shape/finite test:** tiny model forward with the option enabled returns finite
   loss/logits with unchanged shapes.
3. **Normalization scale:** for deliberately low-RMS concepts, the tensor entering `v_proj`
   has RMS approximately one when enabled and preserves low RMS when disabled.
4. **Gradient reach:** with a small nonzero test-only read gate, the RMSNorm weight and
   concept state receive finite nonzero gradients.
5. **Checkpoint compatibility:** a default-config state dict loads into the default model
   without missing/unexpected keys; the enabled variant round-trips its norm weight.

Update `tests/test_training_launcher_parameter_flow.py`:

6. The generic backbone argument group forwards `READ_CONCEPT_NORM`.
7. The historical E10 default remains false; an environment override reaches the Python args.

Run:

```bash
uv run pytest tests/test_backbone_concept_lm.py \
  tests/test_training_launcher_parameter_flow.py -v
```

Local MPS smoke:

```bash
READ_CONCEPT_NORM=true MAX_STEPS=3 TARGET_TOKENS=0 \
PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 \
bash scripts/launch_e10.sh
```

If the launcher cannot safely use the remote pretrained manifest locally, run the existing
tiny synthetic backbone smoke fixture instead; assert finite loss, nonzero gate gradient, and
no new linter diagnostics.

## 8. Risks & tradeoffs

- **Normalization raises read amplitude but not necessarily useful memory.**
  Cheapest signal: `static−real` and `Δshuffle` beyond 1024, not gate magnitude.
  **Fallback:** if both remain ≤0.002 at 25M, stop and test small nonzero gate initialization
  as a separate experiment.
- **Random new norm gain can perturb Gemma once gates open.**
  Cheapest signal: local CE and read-gate trajectory.
  **Fallback:** freeze norm gain at one; do not lower the gate and change LR in the same run.
- **K normalization may hide part of the intervention.**
  This is expected; the intended scale correction is primarily the V path.
- **Zero write/read gates still produce a staged bootstrap.**
  This experiment tests whether corrected V scale is enough to bootstrap it. If not, E10c
  should isolate `read_gate_init=write_gate_init=0.01`; differential LR comes only after
  gradients are demonstrably present.
- **A positive CE result may still be static-memory use.**
  Requiring both `static−real` and shuffled-state degradation distinguishes recurrent writing
  from merely learning a useful fixed prompt.

## 9. Explicitly deferred follow-ups

These are not implementation tasks in E10b:

1. Small nonzero read/write gate initialization.
2. Differential AdamW parameter groups for LoRA, BiXT/z0, gates, and norms.
3. 4K sequence length at the same data mix.
4. A new 4K long-document data recipe.
5. Hybrid Muon restricted to eligible BiXT matrices.
6. Chat/SFT formatting or memory-only QA.

Each changes a separate causal variable and requires its own spec after E10b's result.
