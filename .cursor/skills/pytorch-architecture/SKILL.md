---
name: pytorch-architecture
description: Research-oriented PyTorch architecture design workflow and non-obvious best practices. Use when designing new model architectures, prototyping nn.Module classes, planning ablations, or optimizing training performance. Covers tensor shape sketching, decision gates, mixed precision, DDP, and numerical stability.
---

# PyTorch Architecture Design & Best Practices

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

## Non-Obvious PyTorch Best Practices

### Mixed Precision Training
Use `torch.cuda.amp` with `GradScaler` for faster training and lower memory:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(data)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Data Parallel (DDP)
- Framework: `accelerate` with NCCL backend.
- **CRITICAL**: NEVER use `.expand()` in model forward pass — use `.repeat()` instead. `.expand()` creates views that cause NCCL gradient sync deadlocks. See `docs/debugging/nccl_timeout_fix_2025-11-11.md`.
- Effective batch size = `PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM_STEPS`.

### torch.compile
Apply `torch.compile` to the model for graph-mode acceleration:
```python
compiled_model = torch.compile(model)
```
Check if this do not cause graph breaks. If it does, use `dynamic=True`.

### DataLoader Optimization
```python
DataLoader(dataset, batch_size=B, shuffle=True,
           num_workers=min(os.cpu_count(), 8),
           pin_memory=True, persistent_workers=True)
```

