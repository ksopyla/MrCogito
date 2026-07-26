"""Numerical-correctness tests for ChunkedLMHeadCE (F2 fix).

The custom autograd Function must match the FULL lm_head+CE path on three counts:
  1. the scalar loss (mean over non-ignored positions),
  2. the gradient w.r.t. hidden ([B,N,H]) — what flows back into the decoder,
  3. the gradient w.r.t. the lm_head weight ([V,H]) — what the optimizer sees.
Run on CPU in float32 for tight tolerances; the Function is dtype-agnostic.
"""
import torch
import torch.nn.functional as F

from nn.concept_encoder_perceiver import ChunkedLMHeadCE


def _full_ce(hidden, weight, labels, ignore_index=-100):
    logits = F.linear(hidden, weight)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index
    )


def _run(block_size, ignore=True):
    torch.manual_seed(0)
    B, N, H, V = 2, 64, 16, 50
    hidden0 = torch.randn(B, N, H, dtype=torch.float32)
    weight0 = torch.randn(V, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, N))
    if ignore:
        labels[0, -8:] = -100  # a block-worth of ignored positions

    # --- reference: full CE with autograd ---
    h_ref = hidden0.clone().requires_grad_(True)
    w_ref = weight0.clone().requires_grad_(True)
    loss_ref = _full_ce(h_ref, w_ref, labels)
    loss_ref.backward()

    # --- chunked via the custom Function ---
    h_chk = hidden0.clone().requires_grad_(True)
    w_chk = weight0.clone().requires_grad_(True)
    loss_chk = ChunkedLMHeadCE.apply(h_chk, w_chk, labels, block_size)
    loss_chk.backward()

    return loss_ref, loss_chk, h_ref.grad, h_chk.grad, w_ref.grad, w_chk.grad


def test_loss_matches_full():
    for bs in (1, 16, 64, 128):
        lr, lc, *_ = _run(bs)
        assert torch.allclose(lr, lc, atol=1e-5), f"bs={bs}: {lr.item()} vs {lc.item()}"


def test_hidden_grad_matches_full():
    for bs in (1, 16, 64):
        _, _, hr, hc, _, _ = _run(bs)
        assert torch.allclose(hr, hc, atol=1e-4), f"bs={bs}: max abs diff {(hr - hc).abs().max()}"


def test_weight_grad_matches_full():
    for bs in (1, 16, 64):
        _, _, _, _, wr, wc = _run(bs)
        assert torch.allclose(wr, wc, atol=1e-4), f"bs={bs}: max abs diff {(wr - wc).abs().max()}"


def test_frozen_weight_skips_gradient_but_preserves_hidden_gradient():
    torch.manual_seed(1)
    hidden0 = torch.randn(2, 32, 16)
    weight = torch.randn(50, 16)  # frozen: requires_grad=False
    labels = torch.randint(0, 50, (2, 32))

    h_ref = hidden0.clone().requires_grad_(True)
    _full_ce(h_ref, weight, labels).backward()
    h_chk = hidden0.clone().requires_grad_(True)
    ChunkedLMHeadCE.apply(h_chk, weight, labels, 8).backward()

    assert weight.grad is None
    assert torch.allclose(h_ref.grad, h_chk.grad, atol=1e-4)


def test_grad_zero_at_ignored_positions():
    """Gradients at ignored (label=-100) positions must be exactly zero in both paths."""
    _, _, hr, hc, _, _ = _run(16, ignore=True)
    ignored = slice(-8, None)
    assert torch.all(hr[0, ignored] == 0)
    assert torch.all(hc[0, ignored] == 0)


def test_scales_with_upstream_grad():
    """When the Function's output is scaled before .backward() (as happens when
    loss_manager adds concept losses upstream of task_loss, giving grad_out != 1),
    both the hidden and weight gradients must scale by the same factor as full CE."""
    torch.manual_seed(2)
    h0 = torch.randn(2, 32, 16, dtype=torch.float32)
    w0 = torch.randn(50, 16, dtype=torch.float32)
    lab = torch.randint(0, 50, (2, 32))
    scale = 3.0

    hr = h0.clone().requires_grad_(True)
    wr = w0.clone().requires_grad_(True)
    (_full_ce(hr, wr, lab) * scale).backward()

    hc = h0.clone().requires_grad_(True)
    wc = w0.clone().requires_grad_(True)
    (ChunkedLMHeadCE.apply(hc, wc, lab, 8) * scale).backward()

    assert torch.allclose(hr.grad, hc.grad, atol=1e-4)
    assert torch.allclose(wr.grad, wc.grad, atol=1e-4)
