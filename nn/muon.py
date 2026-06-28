"""Muon optimizer (Jordan et al. 2024) — Momentum + Orthogonalization via Newton-Schulz.

For 2D weight matrices, Muon maintains a momentum buffer and orthogonalizes it with a
5th-order Newton-Schulz iteration (a zero-power approximation, (G G^T)^-1/2 G) before the
step. The orthogonalized update gives every singular direction equal magnitude, which
empirically yields ~1.5-2x faster transformer convergence than AdamW per step, at ~1/2
the optimizer memory (one momentum buffer vs AdamW's m + v).

Very rectangular matrices (embeddings, lm_head — aspect ratio > ``max_aspect``) and 1D
params (norms, biases) fall back to an AdamW update, because Newton-Schulz on a
[49152 x 768] matrix would materialise a [49152 x 49152] intermediate. Reference:
github.com/KellerJordan/Muon.

LR conventions differ from AdamW: Muon uses a higher LR (~0.02) for the matrix params and
a separate AdamW LR (~2-3e-3) for the fallback params. Tune per architecture.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer

# Newton-Schulz 5th-order coefficients (optimal for 5 iterations, from the Muon repo).
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate G @ (G^T G)^-1/2 (the orthogonal factor) via a 5th-order Newton-Schulz
    iteration. Works on the last two dims; keeps G tall internally so the [m x m]
    intermediate stays small. Returns the same shape/dtype as G."""
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT                                   # keep tall: [m<=n]
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT                               # [m, m]
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon for 2D matrices (aspect <= max_aspect) + AdamW fallback for the rest.

    Args:
        lr: Muon learning rate for matrix params (try ~0.02).
        momentum: momentum coefficient for the Muon buffer (0.9-0.95).
        nesterov: Nesterov-style momentum for the orthogonalized update.
        ns_steps: Newton-Schulz iterations (5 is the tuned default).
        adamw_lr: LR for the AdamW fallback (embeddings / lm_head / 1D params).
        adamw_betas, adamw_eps: AdamW params for the fallback.
        weight_decay: decoupled weight decay (applied to both branches).
        max_aspect: 2D params with max(rows,cols)/min(rows,cols) > this use AdamW.
    """

    def __init__(
        self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
        adamw_lr=3e-3, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, weight_decay=0.0,
        max_aspect=8.0,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            weight_decay=weight_decay, max_aspect=max_aspect,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mu_lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns = group["ns_steps"]
            wd = group["weight_decay"]
            a_lr = group["adamw_lr"]
            b1, b2 = group["adamw_betas"]
            a_eps = group["adamw_eps"]
            max_aspect = group["max_aspect"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                use_muon = g.ndim >= 2 and min(g.shape) > 1 and \
                    (max(g.shape) / min(g.shape)) <= max_aspect
                state = self.state[p]
                if use_muon:
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p)
                    buf = state["momentum"]
                    buf.mul_(mom).add_(g)
                    update = g.add(buf, alpha=mom) if nesterov else buf
                    update = zeropower_via_newtonschulz5(update, steps=ns)
                    # scale so the update magnitude is comparable across shapes
                    scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
                    if wd:
                        p.mul_(1 - mu_lr * wd)
                    p.add_(update, alpha=-mu_lr * scale)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    state["step"] += 1
                    m, v = state["m"], state["v"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bias = 1 - b2 ** state["step"]
                    vhat = (v / bias).sqrt_().add_(a_eps)
                    if wd:
                        p.mul_(1 - a_lr * wd)
                    p.addcdiv_(m, vhat, value=-a_lr)
        return loss
