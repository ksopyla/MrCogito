---
name: research-implement
description: Implement and extend the Concept Encoder PyTorch foundation (nn/, training/, data/, evaluation/) from an approved experiment spec and its implementation plan. Use when writing or modifying encoder/reasoning/decoder modules, losses, data collators, the training entrypoint, or training bash launchers in this codebase. Knows the existing modules and the encode→reason→decode patterns, adds capability as reusable config-selectable components (never per-experiment forks), and preserves old code/checkpoints for reproducibility. Not for experiment scoping (experiment-design), writing the design plan (implementation-plan), result logging (experiment-track), or doc cleanup (docs-hygiene).
---

# Research Implement (Concept Encoder codebase)

This is the **implementation half** of the workflow: `experiment-design` produces a
frozen spec in `docs/experiments_specs/<ID>.md`, `implementation-plan` produces the repo-rooted
design in `docs/experiments_specs/<ID>_plan.md`; this skill turns them into code in the
**shared foundation**. Implement *exactly* the spec's single change — nothing more.

Read the spec (`Builds-on` + `The single change`) **and its `<ID>_plan.md`** (reuse map,
forward pass, data, loss, config, tests) before touching code — follow the plan; if it is
missing or wrong, hand back to `implementation-plan` rather than improvising. Then locate
the right module below. Default to **extending** an existing module; add new code only as a
**reusable, config-selectable** component.

## Codebase map (what you build on)
- **Config + encoder** — `nn/concept_encoder.py`: `ConceptEncoderConfig`, `ConceptEncoder` (N tokens → C concepts), layers `ConceptEncoderLayer` (cross-attn) and `BiConceptEncoderLayer` / `BiXTCrossAttention` (BiXT). Key knobs: `hidden_size` H, `concept_num` C, `token_embedding_dim` (token↔concept asymmetry), `num_hidden_layers`, `use_bixt`.
- **Decoders / heads** — `nn/concept_encoder_perceiver.py`: `PerceiverDecoderLayer` / `PerceiverDecoderStack` (position/query cross-attention to concepts), `ConceptEncoderForDenoisingPerceiver` (maintained denoise model), `…ForSequenceClassificationViaDecoder`, `…ForSentencePairClassification`, `…ForSequenceClassificationPerceiver` (eval heads), `…PosOnly`.
- **Baseline** — `nn/concept_encoder_weighted.py`: `weighted_mlm` model + weighted-pool classifier.
- **Losses** — `nn/loss_manager.py` (`LossManager`, `LossConfig`, loss components, `ConceptLossStepCallback`, `register_loss`) + `nn/concept_losses.py` (functional losses + concept metrics).
- **Data** — `data/data_collators.py` (collators, e.g. TSDAE/denoise) and `data/dataset_preprocess.py`.
- **Eval / analysis** — `evaluation/concept_eval_routing.py` (checkpoint → eval route), `concept_checkpoint_loader.py`, `evaluate_model_on_glue.py`, `evaluate_on_benchmark.py`; `analysis/run_concept_analysis.py` + `analysis/concept_analysis.py` (effective rank, pairwise similarity, singular values).
- **Parked (revivable, do NOT import from the live tree)** — `parked/` (recursive, diffusion families). See `parked/README.md`.

## The three patterns — name which one you're touching
- **ENCODING:** cross-attention compresses N input tokens → C concept vectors, O(C·N). BiXT is the bidirectional variant; token↔concept asymmetry via `token_embedding_dim ≠ hidden_size`. This is the part prone to **collapse** — always check effective rank / pairwise similarity (`run_concept_analysis.py`) when you change it.
- **REASONING (latent):** concept → concept refinement over the C concepts (e.g. weight-tied iterations — currently in `parked/` recursive). Any new reasoning block operates concept→concept and must be a **config-selectable** module, not hard-wired.
- **DECODING:** produce tokens/logits from concepts via query cross-attention to concepts (`PerceiverDecoderStack`) → `lm_head`; `ViaDecoder` reuses the decoder for classification. New generation heads (e.g. autoregressive) are **concept-conditioned** and live here as reusable heads. Keep decoding O(C·N) — do **not** reintroduce O(N²) token self-attention (a past regression).

## How to add capability (configs over forks)
1. Put the code in the right home: encoder layer → `nn/concept_encoder.py`; decoder/head → `nn/concept_encoder_perceiver.py`; loss → `nn/loss_manager.py` (`register_loss`); collator → `data/data_collators.py`.
2. Expose it through config: add a field to `ConceptEncoderConfig` (or `LossConfig`) and select the variant by a config/CLI value. Register model types in the entrypoint's `MODEL_REGISTRY`.
3. **Never** create `training/train_<idea>.py` or `nn/concept_encoder_<idea>.py` per experiment. One shared entrypoint; experiments are args/configs (see `.cursor/rules/experiment-discipline.mdc`).
4. Document `forward()` input/output shapes (`[B, N, H]` → `[B, C, H]` → …) in the docstring. Add a small unit test in `tests/` for the new module (tiny random tensors, shape + loss-finite checks).

## Reproducibility — do NOT break old experiments
- Never delete or rewrite old model code, configs, or checkpoint-loading paths just because they are off the current direction. **Old checkpoints must keep loading.**
- To retire something, **park it** (`git mv` into `parked/`, tag a snapshot) — do not delete. Hand bulk pruning to `docs-hygiene`. See `parked/README.md`.
- Add new config fields with **backward-compatible defaults**; don't rename/remove existing ones that checkpoints depend on.
- Preserve the checkpoint **evaluation contract**: new trainable families must save the eval metadata `concept_eval_routing.py` needs (`checkpoint_family`, `evaluation_contract_version`, canonical single/pair modes).

## Unparking — bringing `parked/` code back
When a spec decides to revive a parked family, do NOT just import from `parked/`. Move it back into the live tree, then review and align it with the current foundation — the parked code froze at `pre-consolidation-20260605` and the foundation has moved since.

```
- [ ] 1. There is an approved spec naming the materially new ingredient (e.g. warm-start). Without it, stop — this is experiment-design's call.
- [ ] 2. git mv the family back: parked/nn/* → nn/, parked/training/* → training/, parked/tests/* → tests/, parked/scripts/* → scripts/ (preserve history).
- [ ] 3. Re-wire it into the live foundation: fix imports; re-register the model in the entrypoint MODEL_REGISTRY and in analysis/run_concept_analysis.py MODEL_CLASSES; restore any eval choices / routing it needs.
- [ ] 4. Align with the current foundation: reconcile against the current ConceptEncoderConfig fields, BiXT/encoder layers, PerceiverDecoderStack, loss_manager API, and data collators. Update drifted call sites; keep the unparked code's old checkpoints loadable.
- [ ] 5. Restore the checkpoint evaluation contract (metadata fields) if the family is trainable+evaluated.
- [ ] 6. Run its tests (now under tests/) + a tiny local MPS smoke run; fix what drifted.
- [ ] 7. Update parked/README.md (remove the revived rows), training_eval_matrix.md (status → maintained/active), agenda.md, and CHANGELOG.md.
```

Review honestly while unparking: only restore what the spec needs, flag anything that no longer fits the current direction, and don't silently resurrect dead assumptions. If large parts conflict with the current foundation, say so and prefer reimplementing that piece as a fresh reusable component over force-fitting old code.

## Training entrypoint init standard
Every `training/train_*.py` `main()` follows this exact sequence; canonical reference `training/train_concept_pretraining.py` / `training/train_mlm.py`, helpers in `training/utils_training.py`. `training/train_perceiver_denoise.py` is a temporary compatibility wrapper and is not the implementation reference.

```python
def main():
    setup_distributed()                       # 1. NCCL init, device assignment (before anything else)
    if is_main_process():                      # 2. logging verbosity (or info is swallowed)
        logging.set_verbosity_info(); setup_file_logging()
    else:
        logging.set_verbosity_error()
    # 3. parse_args_into_dataclasses()
    set_seed(training_args.seed)               # 4. seed, system + data config
    log_system_info(); log_data_config(...)
    # 5. tokenizer + dataset; 6. log dataset sizes after load/filter
    # 7. config + loss config + model init
    log_loss_config(loss_config); log_model_info(model, config=..., model_type=...)
    # 8. Flash-Attention probe (main process); 9. optional torch.compile(dynamic=True)
    setup_run_dirs(training_args, run_identifier)   # 10. run dirs + guards
    training_args.use_cpu = False
    if training_args.eval_strategy != "steps": training_args.eval_steps = None
    if training_args.save_strategy != "steps": training_args.save_steps = None
    log_training_config(...); init_wandb(...)       # 11. config log + W&B
    trainer.train()                                  # 12. train
    trainer.save_model(final_path); tokenizer.save_pretrained(final_path)   # 13. save
    if wandb.run and is_main_process(): wandb.finish()
```

Imports: `from training.utils_training import (init_wandb, is_main_process, log_data_config, log_loss_config, log_model_info, log_system_info, log_training_config, setup_distributed, setup_file_logging, setup_run_dirs)`.

Checklist for new/modified entrypoints: `setup_distributed()` first · verbosity set per-rank · dataset sizes logged after load · Flash-Attn probe after model init · `use_cpu=False` + eval/save step guards after `setup_run_dirs` · `is_main_process()` guard on `wandb.finish()`.

## Training bash launchers
Reference: `scripts/train_perceiver_denoise_multigpu.sh`. Pattern: set CUDA/NCCL/HF env → declare `"${VAR:-default}"` knobs → `accelerate launch --num_processes=$NUM_GPUS --multi_gpu --mixed_precision=bf16 training/<entrypoint>.py --args …` → pipe through `scripts/clean_tee.py` to `Cache/logs/`.
- To run a **new experiment variant**: add `"${VAR:-default}"` knobs and pass the new config arg to the **existing** launcher — do not copy the script. Override at launch time (`HIDDEN_SIZE=768 bash scripts/train_perceiver_denoise_multigpu.sh`).
- Always expose and set checkpoint retention for training launchers (`SAVE_TOTAL_LIMIT`, passed to
  `--save_total_limit`). Full-corpus runs with frequent saves can fill `/home` if retention is
  unbounded; default to a small finite value (3–5) unless the experiment explicitly needs every
  intermediate checkpoint. Keep final `trainer.save_model(...)` output separate from rotating periodic
  checkpoints.
- Write a **new** launcher only for a genuinely new entrypoint, and keep it structurally identical to the reference.

## Hardware / numerics (this project)
- Local = Apple **MPS**, smoke tests only (`PYTORCH_ENABLE_MPS_FALLBACK=1`); real runs on remote **RTX 3090 (24 GB)**. `liger-kernel` is Linux/CUDA-only — never import it from cross-platform code.
- Preserve the O(C·N) advantage; keep AMP/bf16 + cosine schedule as in the reference launcher.

## Handoffs
- Scope / write the spec → `experiment-design`. Record results → `experiment-track`. Retire or prune code/docs → `docs-hygiene`. Choose research direction from literature → `research-synthesis`.
