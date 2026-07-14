# E16 — Shared Depth-Recurrent Concept Workspace Implementation Plan

- **Spec:** [E16_shared_depth_recurrent_concepts.md](E16_shared_depth_recurrent_concepts.md) · **Status:** implemented and run; failed the 2026-07-14 mechanism gate
- **Authored by:** `implementation-plan` · for → `research-implement`

> Implement exactly one change versus E10e: move the existing tied concept write
> from one post-block application to four interleaved applications after Gemma's
> existing concept-reading layers. Keep one shared state and preserve the E10e path
> as the backward-compatible default.

## 1. Source & fit

- **Origin:** E10b–E10e showed healthy concept geometry and a short carry-adjacent
  signal, but no persistent use from one post-block write. The design adapts the
  differentiable shared-state recurrence pattern reviewed in
  [`recurrent_memory_transformers.md`](../../literature_review/recurrent_memory_transformers.md),
  while retaining MrCogito's single compressed concept workspace.
- **Synthesis verdict:** **Adapt.** Take repeated gated refinement and tied update
  weights; reject E13's 26 independent memories and any growing raw-token/KV store.
- **Architecture mapping:** concept bottleneck only. E16 changes the schedule by
  which the shared state is refined through model depth; it does not add a reasoner,
  loss, dataset, decoder, or independent layer memories.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `BackboneConceptConfig` | extend accepted `concept_io_mode` values; default remains `global_kv` | `nn/backbone_concept_lm.py` |
| `ConceptReadBranch` | reuse as-is | `nn/backbone_concept_lm.py` |
| `GlobalLayerWithConceptRead` | reuse its read path as-is | `nn/backbone_concept_lm.py` |
| `ConceptWriteHead` | extend to one shared BiXT writer with either one legacy scalar gate or an ordered vector of depth gates | `nn/backbone_concept_lm.py` |
| `BackboneConceptLM` block loop | extend with a config-selected explicit Gemma layer loop and interleaved writes | `nn/backbone_concept_lm.py` |
| `ModelArguments.concept_io_mode` | update validation/help only | `training/concept_pretraining_args.py` |
| differential optimizer grouping | recognize depth-gate parameters as concept-memory parameters | `training/concept_pretraining_trainer.py` |
| E10 launcher/config plumbing | reuse as-is; it already forwards `CONCEPT_IO_MODE` | `scripts/launch_e10.sh`, `scripts/train_concept_pretraining_multigpu.sh` |
| checkpoint/eval routes | reuse as-is under `checkpoint_family=backbone_concept` | `evaluation/concept_eval_routing.py`, `analysis/run_concept_analysis.py`, `analysis/run_e10_comparison.py` |

No new model family, training entrypoint, launcher, collator, loss, or evaluation
script is introduced.

## 3. Forward pass

Symbols: `B`=batch, `N=2048`, `K=512`, `C=128`, `H=1152`,
`L=26`, and `Q<=2K` includes the one-block token carry.

```text
z_0: learned concept_init [C,H] → expand → [B,C,H]

for each K-token block b:
  dec_ids / mask                     [B,Q]
  token embeddings                   [B,Q,H]

  Gemma layers 1..26 execute in order.
  At each full-attention layer g in human layers 6/12/18/24:
    tokens read current z             [B,Q,H] ← attention over [B,C,H]
    current-block tail                h_g = hidden[:,-block_len:,:]
    shared tied writer                update_g = Write(z, h_g) [B,C,H]
    depth-specific scalar gate        z ← z + tanh(alpha_g) * update_g

  final Gemma norm                    [B,Q,H]
  next-token CE on current block only
  final shared z carries to block b+1 [B,C,H]
```

The write consumes the hidden state returned by the wrapped global layer, including
that layer's concept-read residual, and only the current block tail—not carry-token
states. The last interleaved update is the block output; E16 performs no additional
post-block write.

### Checkpoint-safe execution

Do not mutate `z` from inside `GlobalLayerWithConceptRead.forward()`. A write inside
a gradient-checkpointed layer would be replayed during backward against stale mutable
state. Add a private explicit one-block text-model helper on `BackboneConceptLM`
that:

1. reproduces `Gemma3TextModel.forward` embedding/position/mask/final-norm behavior;
2. invokes each decoder layer in order using the same per-layer attention-mask
   selection as Hugging Face;
3. leaves each layer's checkpointing behavior intact;
4. applies the tied write in the parent helper after a global layer returns.

The existing monolithic `self.backbone.model(...)` path remains untouched for
`global_kv` and `concept_num=0`.

### Ablation semantics

| Mode | E16 behavior |
|---|---|
| `real` | all four reads and chained writes use the evolving shared state |
| `static` | reads use initial `z`; all four writes are disabled |
| `zero` | no state, reads, or writes |
| `shuffle` / `permutation` | each read sees the deranged state; writes continue on the unpermuted recurrent path, matching E10 |
| `one_block` | at each depth, the writer's residual base is reset to `concept_init`; this prevents accumulation through depth and blocks while retaining a one-update signal |

Existing `concept_ablation_ce`, per-position metrics, and
`encode_concepts()` signatures remain unchanged.

## 4. Inputs and data

- **Dataset:** existing immutable Gemma-tokenized
  `smollm3_inspired_2k_e05` manifest.
- **Tokenizer:** `google/gemma-3-1b-pt`.
- **Collator:** existing causal-LM collator selected by the
  `backbone_concept`/`causal_lm` factory; no data code changes.
- **Masking:** sequence length 2048, K=512 window and one-block token carry exactly
  as E10e. The explicit layer loop receives the same additive mask dictionary for
  sliding and former-global layers.

## 5. Loss and training objective

- Reuse next-token `ChunkedLMHeadCE` and the existing token-count-normalized
  causal-LM loss in `BackboneConceptLM`.
- No auxiliary loss and no changed weighting. This isolates update schedule from
  E10e.
- Keep the zero-valued differentiable `z` tie used to make every recurrent
  parameter visible to DDP with `find_unused_parameters=False`.

## 6. Config, telemetry, and launch

- Accept `BackboneConceptConfig.concept_io_mode="shared_depth_recurrent"`; retain
  `"global_kv"` as the serialized/default behavior for old checkpoints.
- `ConceptWriteHead` owns shared BiXT/norm weights plus:
  - legacy `alpha` for `global_kv`; or
  - `depth_alphas[n_global]` for E16, ordered by discovered
    `attention_type=="full_attention"` layers.
- Discover layer indices from the backbone rather than hard-coding 5/11/17/23.
  Gemma-3-1B must discover exactly four; tests may use fewer.
- Extend `concept_gate_metrics()` with stable keys
  `concept_gates/write_0..3` and `concept_gates/write_layer_5|11|17|23`.
  Preserve legacy `concept_gates/write` only for `global_kv`.
- Existing W&B identity remains `backbone_concept` and adds the already-supported
  `io-shared_depth_recurrent` tag/config value.
- Exact launch:

```bash
EXPERIMENT_ID=E16 CONCEPT_IO_MODE=shared_depth_recurrent \
READ_CONCEPT_NORM=true READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 \
CONCEPT_MEMORY_LR=3e-4 LEARNING_RATE=1e-4 OPTIMIZER=adam \
WEIGHT_DECAY=0 TARGET_TOKENS=50000000 WARMUP_STEPS=50 \
SKIP_PRETOKENIZE=1 MAX_SEQ_LENGTH=2048 CONCEPT_NUM=128 \
SAVE_TOTAL_LIMIT=4 \
bash scripts/launch_e10.sh
```

Before the full command, run an Odra memory/throughput calibration with the same
model and sequence settings but a tiny target budget. Preserve effective batch 72
by adjusting microbatch and gradient accumulation.

## 7. Tests and smoke

Add targeted coverage to `tests/test_backbone_concept_lm.py`:

- mode construction discovers global layers and creates one depth gate per layer;
- global_kv still has one gate and unchanged post-block behavior;
- E16 performs exactly one write per global layer and no post-block write;
- later global reads observe the state produced by earlier depth writes;
- real/static/zero/shuffle/one-block APIs and output keys remain compatible;
- every depth gate receives a finite gradient;
- non-reentrant gradient checkpointing produces finite, deterministic gradients;
- `save_pretrained`/`from_pretrained` round-trip preserves logits, state, mode, and
  gates;
- `concept_num=0` remains the unchanged control path;
- gate telemetry exposes all discovered depth writes.

Extend argument/launcher parameter-flow tests only where existing tests enumerate
allowed/help-text values.

Verification commands:

```bash
uv run pytest tests/test_backbone_concept_lm.py -v
uv run pytest tests/test_concept_pretraining_parameter_flow.py \
  tests/test_training_launcher_parameter_flow.py -v
uv run pytest tests/ -v
```

Run the existing tiny-backbone finite-loss/backward smoke on CPU/MPS with
`concept_io_mode=shared_depth_recurrent`, two token blocks, open 0.01 gates, and
assert finite loss plus finite gradients for every depth gate and the shared writer.

## 8. Risks and tradeoffs

- **Hugging Face forward drift:** duplicating Gemma's layer loop can diverge from
  upstream mask/position/cache behavior. Cheapest signal: a test comparing helper
  versus native model hidden states with writes/reads disabled. Keep the helper
  private and minimal.
- **Gradient-checkpoint recomputation:** mutable in-layer writes would silently
  corrupt recurrence. Cheapest signal: checkpointed vs non-checkpointed loss and
  gradient equivalence on a tiny model.
- **Activation memory:** four differentiable BiXT writes per block retain more
  activations. Cheapest signal: Odra calibration with `nvidia-smi`, throughput, and
  1–2 GiB headroom before launch.
- **Repeated updates can destabilize/collapse:** the existing RMSNorm/sandwich norm,
  0.01 gates, RankMe guard, and 25M recurrence gate bound the risk.
- **Plain CE may still ignore memory:** this is the intended falsification. If both
  beyond-local deltas remain ≤0.002 at 25M, kill E16 rather than changing objective
  inside the run.

## 9. Implementation sequence

1. Refactor `ConceptWriteHead` for tied multi-depth gates while keeping E10 tests
   unchanged.
2. Add the private checkpoint-safe explicit layer-loop helper and E16 block-loop
   branch.
3. Preserve all ablation modes and expose per-depth telemetry/optimizer roles.
4. Add round-trip, gradient-checkpoint, equivalence, ablation, and parameter-flow
   tests.
5. Run the targeted and complete local suites.
6. Update engineering traceability, then sync through Git and calibrate/launch E16
   on Odra.
