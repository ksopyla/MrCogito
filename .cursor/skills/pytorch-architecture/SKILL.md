---
name: pytorch-architecture
description: Design and implement Concept Encoder PyTorch modules safely. Use when sketching tensor shapes, writing or refactoring nn.Module code, checking memory or DDP behavior, choosing AMP or torch.compile patterns, or improving training-time performance and numerical stability. Not for experiment logging, changelog updates, or high-level research prioritization.
---

# PyTorch Architecture Design & Best Practices

Use this skill for model implementation and systems-level PyTorch decisions, not for run bookkeeping or deciding which research hypothesis to test next.

## Research Design Workflow

### Before Writing Code
1. **Sketch the forward pass** with concrete tensor shapes at each step (e.g., `[B, N, H]` → cross-attention → `[B, C, H]`). Document shapes in docstrings.
2. **PyTorch & Hardware First**: Keep PyTorch and engineering principles in mind while designing new architectures. Think about memory layout, vectorization, and GPU utilization (e.g., maximizing FLOPs, minimizing memory bandwidth bottlenecks, utilizing fused kernels like Flash Attention, LingerKernel, etc.).
3. **Always check the project goals and roadmap** to ensure the new architecture is aligned with the project goals and roadmap.
4. **Estimate memory footprint**: will it fit in 10GB VRAM (local RTX 3080) or 24GB (remote RTX 3090)?
5. **Define decision gates** before training: "If effective rank > X, proceed. If < Y, hypothesis is wrong." Write gates into `active_todos.md`.
6. **Plan ablation**: what varies (one thing at a time), what stays fixed, which metric decides.

### Architecture Implementation
- Use `nn.Module` subclasses. Separate encoder, decoder, and loss into distinct modules.
- Document expected input/output shapes in every `forward()` docstring.
- Write a unit test with small random tensors before integration testing.

## Training Script Initialization Standard

Every `training/train_*.py` script MUST follow this exact initialization sequence in `main()`. The canonical reference is `training/train_mlm.py`; all helpers live in `training/utils_training.py`.

### Required init sequence (in order)

```python
def main():
    # 1. Distributed setup (NCCL init, CUDA device assignment)
    setup_distributed()

    # 2. Logging verbosity — without this, all logger.info() calls are swallowed
    if is_main_process():
        logging.set_verbosity_info()
        setup_file_logging()          # timestamped log file in Cache/logs/
    else:
        logging.set_verbosity_error()

    # 3. Argument parsing
    ...parse_args_into_dataclasses()...

    # 4. Seed, system info, data config
    set_seed(training_args.seed)
    log_system_info()
    log_data_config(data_args, extra_fields={...})

    # 5. Tokenizer + dataset loading
    ...
    # 6. Dataset size logging (after load/filter)
    logger.info(f"Train dataset size: {len(train_ds):,}")
    logger.info(f"Test dataset size: {len(test_ds):,}")

    # 7. Config, loss config, model init
    ...
    log_loss_config(loss_config)
    log_model_info(model, config=config, model_type=..., model_description=...)

    # 8. Flash Attention probe (main process only)
    if torch.cuda.is_available() and is_main_process():
        ...sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)...

    # 9. Optional torch.compile(dynamic=True)
    ...

    # 10. Run dirs + training args guards
    setup_run_dirs(training_args, run_identifier)
    training_args.use_cpu = False
    if training_args.eval_strategy != "steps":
        training_args.eval_steps = None
    if training_args.save_strategy != "steps":
        training_args.save_steps = None

    # 11. Training config log + W&B init
    log_training_config(training_args, extra_fields={...})
    init_wandb(training_args, model, config, data_args, loss_config, ...)

    # 12. Trainer creation + train
    ...
    trainer.train()

    # 13. Save + W&B finish (main process guard)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    if wandb.run and is_main_process():
        wandb.finish()
```

### Required imports from utils_training

```python
from training.utils_training import (
    init_wandb,
    is_main_process,
    log_data_config,
    log_loss_config,
    log_model_info,
    log_system_info,
    log_training_config,
    setup_distributed,
    setup_file_logging,
    setup_run_dirs,
)
```

### Checklist for new or modified training scripts

- [ ] `setup_distributed()` called before anything else
- [ ] `logging.set_verbosity_info()` + `setup_file_logging()` on main process
- [ ] `logging.set_verbosity_error()` on non-main processes
- [ ] Dataset sizes logged after loading
- [ ] Flash Attention probe after model init
- [ ] `training_args.use_cpu = False` after `setup_run_dirs`
- [ ] Eval/save step guards after `setup_run_dirs`
- [ ] `is_main_process()` guard on `wandb.finish()`


