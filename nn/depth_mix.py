"""E40 — depth-mixed attention inputs: DeepCrossAttention's GRN-v3 mixes for a layer's query, key and value inputs.

Spec: docs/experiments_specs/done_failed/E40_depth_mixed_read.md. Source: Heddes et al., "DeepCrossAttention: Supercharging
Transformer Residual Connections", ICML 2025, arXiv 2502.06785 (literature: docs/literature_review/
depth_wise_residual_connections.md).

**Columns.** A forward is a sequence of layer runs (the reading layer's loop counts each pass: 19 runs for the 150M
models, `run_schedule`). Before run t the stream is s_t; s_0 = the embedding. The depth columns are the stream's
increments c_0 = s_0, c_j = s_j − s_{j−1} (each run's output, plus anything added between runs such as the loop
markers, minus its input), so Σ_j c_j = s_t exactly, whatever α, β and the U-net skips did inside the blocks.

**One mix (GRN-v3).**  g = Σ_j (b_j + relu(w · rms(c_j) / d)) ⊙ c_j,  b_j ∈ R^d per column, w ∈ R^d per mix (one
input-dependent scalar per token and column). Written as  g = s_t + Σ_j (b'_j + relu(w · rms(c_j) / d)) ⊙ c_j  with
b = 1 + b': at b' = 0, w = 0 every term is an exact zero, so g = s_t and the model is exactly the one it started
from (a warm start is unchanged at step 0). relu here has gradient 1 at 0 (`relu0`): with PyTorch's relu (gradient 0
at 0) a w that starts at 0, as the paper's does, can never move; the unofficial code instead starts w at small noise,
which breaks the exact start. rms(c) (no gain) makes the gate's input scale-free, as the unofficial code's RMSNorm.
The gate reads the MEAN over the d dimensions (÷ d), not the sum: Adam moves each weight by about its rate per step,
so with the sum one step moved a gate by ~0.7 at rate 1e-3 (E40 speed check, 2026-11-24: the mixed arms' loss rose
0.01–0.025 in ten steps) while b moved by 0.001; with the mean both move alike.

**One block with mixes.** Three mixes give the query, key and value inputs g_q, g_k, g_v. The block's own input mix
(α · · + β · x0 + σ · skip, `Block.mix`) is applied to each; attention reads its query from g_q, its keys from g_k
and its values from g_v; the feed-forward reads g_q + attention (the paper: the query input is the residual of the
attention and the feed-forward input); the stream continues as  mix(s_t) + attention + feed-forward, so the next
column is this run's increment (the paper stacks f_t = attention + feed-forward). At the start g_q = g_k = g_v = s_t
and the block is the original one.

**Modes** (`par_depth_mix`):
  read  — the reading layer's runs only (each loop pass its own three mixes), over every column before the run:
          x0, the three local layers' increments and the earlier passes' reading-layer and looped-layer increments.
  dca2  — 2-DCA on every run (the paper's k-DCA, k = 2): columns c_0, the sum of the middle ones, and the last two.
Neither changes anything else (the residual bookkeeping, the U-net skips, the loop markers).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

MODES = ("none", "read", "dca2")
_EPS = 1e-6


def relu0(z: torch.Tensor) -> torch.Tensor:
    """relu with gradient 1 at 0 (where z ≥ 0), so a gate weight that starts at exactly 0 can learn."""
    return torch.where(z >= 0, z, torch.zeros_like(z))


def _rms(c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return c * torch.rsqrt(c.float().pow(2).mean(-1, keepdim=True) + eps).to(c.dtype)


class GRN3(nn.Module):
    """One GRN-v3 mix over `n_cols` columns of width `d`, in the delta form (exact identity at init)."""

    def __init__(self, n_cols: int, d: int):
        super().__init__()
        self.n_cols = int(n_cols)
        self.b_delta = nn.Parameter(torch.zeros(self.n_cols, d))   # b = 1 + b_delta (per column, per dimension)
        self.w = nn.Parameter(torch.zeros(d))                      # the gate's input weights (shared by columns)

    def forward(self, cols: list, plain: torch.Tensor) -> torch.Tensor:
        """cols: n_cols tensors [B, S, d] summing to `plain` (the stream). Returns g [B, S, d]."""
        if len(cols) != self.n_cols:
            raise ValueError(f"GRN3 built for {self.n_cols} columns, got {len(cols)}")
        out = plain
        w = self.w
        for j, c in enumerate(cols):
            gate = relu0((_rms(c) * w.to(c.dtype)).mean(-1, keepdim=True))            # [B, S, 1]
            out = out + (self.b_delta[j].to(c.dtype) + gate) * c
        return out


def run_schedule(cfg) -> list:
    """The layer index of every run of a forward, in order (the reading layer's loop unrolled)."""
    n = len(cfg.layer_patterns())
    R = int(getattr(cfg, "message_loop_rounds", 1) or 1)
    if R <= 1:
        return list(range(n))
    gi = int(cfg.global_layer_index)
    span = int(getattr(cfg, "message_loop_span", 1) or 1)
    core = list(range(gi, gi + 1 + span))
    return list(range(gi)) + core * R + list(range(gi + 1 + span, n))


def _mix3(x: torch.Tensor, W: torch.Tensor, Bd: torch.Tensor, *cols: torch.Tensor) -> tuple:
    """The three mixes of one run at once: g_m = x + Σ_j (b'_mj + relu0((w_m · c_j) / (rms(c_j) d))) ⊙ c_j for
    m = query, key, value. x [B, S, d]; W [3, d] (the gate weights); Bd [3, J, d] (b'); cols J × [B, S, d]. Each
    column's rms and its three gates (one matrix product) are shared by the three mixes; same values as `GRN3`."""
    d = x.shape[-1]
    outs = [x, x, x]
    for j, c in enumerate(cols):
        inv = torch.rsqrt(c.float().pow(2).mean(-1, keepdim=True) + _EPS).to(c.dtype)   # 1 / rms(c), [B, S, 1]
        gate = relu0((c @ W.t().to(c.dtype)) * (inv / d))                               # [B, S, 3]
        for m in range(3):
            outs[m] = outs[m] + (Bd[m, j].to(c.dtype) + gate[..., m:m + 1]) * c
    return tuple(outs)


_MIX3_COMPILED = None


def mix3(x, W, Bd, cols) -> tuple:
    """`_mix3`, through torch.compile when DEPTH_MIX_COMPILE=1, on CUDA, with gradients (training): the elementwise
    work fused (E40 speed check, 2026-11-24: eager, the reading-layer mixes cost 15 % of the training speed at 16k).
    One compiled graph per column count."""
    global _MIX3_COMPILED
    if x.is_cuda and torch.is_grad_enabled() and os.environ.get("DEPTH_MIX_COMPILE", "0") == "1":
        if _MIX3_COMPILED is None:
            import torch._dynamo.config as dyn_cfg

            dyn_cfg.cache_size_limit = max(int(getattr(dyn_cfg, "cache_size_limit", 8)), 64)
            _MIX3_COMPILED = torch.compile(_mix3, dynamic=False)
        return _MIX3_COMPILED(x, W, Bd, *cols)
    return _mix3(x, W, Bd, *cols)


def dca2_columns(t: int) -> int:
    """Columns a 2-DCA mix reads before run t (t + 1 increments exist): all of them up to 4, then 4."""
    return min(t + 1, 4)


class DepthMix(nn.Module):
    """The mixes of one model: for every mixed run t, three GRN3 (query, key, value inputs), `runs[str(t)]`."""

    def __init__(self, cfg):
        super().__init__()
        mode = str(getattr(cfg, "par_depth_mix", "none") or "none")
        if mode not in MODES or mode == "none":
            raise ValueError(f"par_depth_mix must be one of {MODES[1:]} to build mixes, got {mode!r}")
        self.mode = mode
        self.schedule = run_schedule(cfg)
        gi = int(cfg.global_layer_index)
        d = int(cfg.hidden_size)
        self.runs = nn.ModuleDict()
        for t, li in enumerate(self.schedule):
            if mode == "read" and li != gi:
                continue
            n_cols = t + 1 if mode == "read" else dca2_columns(t)
            self.runs[str(t)] = nn.ModuleList([GRN3(n_cols, d) for _ in range(3)])

    def mixed(self, t: int) -> bool:
        return str(t) in self.runs

    def columns(self, states: list, t: int) -> list:
        """The columns run t's mixes read, from the stream states s_0..s_t (each the input of a run)."""
        if len(states) != t + 1:
            raise RuntimeError(f"run {t}: {len(states)} stream states")
        inc = lambda j: states[0] if j == 0 else states[j] - states[j - 1]   # noqa: E731
        if self.mode == "read" or t + 1 <= 4:
            return [inc(j) for j in range(t + 1)]
        # 2-DCA: the model input, the sum of the middle increments (s_{t−2} − s_0), the last two increments
        return [states[0], states[t - 2] - states[0], inc(t - 1), inc(t)]

    def inputs(self, states: list, t: int) -> Optional[tuple]:
        """(g_q, g_k, g_v) for run t, or None when the run has no mixes."""
        key = str(t)
        if key not in self.runs:
            return None
        cols = self.columns(states, t)
        mixes = self.runs[key]
        W = torch.stack([m.w for m in mixes])                                           # [3, d]
        Bd = torch.stack([m.b_delta for m in mixes])                                    # [3, J, d]
        return mix3(states[t], W, Bd, cols)


class DepthState:
    """The stream states of one forward (the inputs of every run so far), fed to the mixes run by run."""

    def __init__(self, mix: DepthMix):
        self.mix = mix
        self.states: list = []

    def start(self, x: torch.Tensor) -> int:
        """Record the input of the next run; returns its run index."""
        self.states.append(x)
        return len(self.states) - 1

    def inputs(self, t: int) -> Optional[tuple]:
        return self.mix.inputs(self.states, t)
