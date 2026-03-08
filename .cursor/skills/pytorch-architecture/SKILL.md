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



