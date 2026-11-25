"""E40 — depth-mixed attention inputs (nn/depth_mix.py; spec docs/experiments_specs/done_failed/E40_depth_mixed_read.md).

What must hold:
  D1  the run schedule unrolls the reading layer's loop; 'read' mixes only its runs, over every earlier column;
      'dca2' mixes every run over at most four columns; the columns of a run sum to its input exactly;
  D2  at initialisation the model with mixes equals the model without them (same weights), forward and loss,
      in both modes (the warm start is unchanged: P3);
  D3  gradients reach b' and w at initialisation (w through relu's gradient 1 at 0), and the mixed model learns;
  D4  moved off the identity, the mixes still never read ahead (later tokens do not move earlier logits);
  D5  gradient checkpointing gives the same loss and gradients as the plain backward;
  D6  a warm start from a model without mixes leaves only the mixes missing.
Cached decoding with mixes: tests/test_incremental_decode.py (arms concept_dmread, concept_dca2).
"""
from __future__ import annotations

import pytest
import torch

from nn.depth_mix import DepthMix, GRN3, dca2_columns, relu0, run_schedule
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM
from tests.test_concept_memory import CONCEPT
from tests.test_latent_memory import cfg as base_cfg

DEEP = dict(pre_layers=2, stack_layers=3, message_loop_rounds=3)


def _pair(mode, deep=False, seed=3):
    """(model without mixes, the same weights with mixes of `mode`)."""
    kw = dict(CONCEPT, **(DEEP if deep else {}))
    torch.manual_seed(seed)
    a = PerceiverARLM(base_cfg(**kw)).eval()
    for layer in a.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
        layer.alpha.data.uniform_(0.5, 1.1)          # trained-looking input mixes and skips
        layer.beta.data.uniform_(-0.3, 0.3)
        if layer.sigma is not None:
            layer.sigma.data.uniform_(-0.3, 0.8)
    if a.loop_emb is not None:
        a.loop_emb.data.normal_(0, 0.5)
    b = PerceiverARLM(base_cfg(**kw, par_depth_mix=mode)).eval()
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("depth_mix.") for k in missing)
    return a, b


def _ids(S=96, seed=5):
    return torch.randint(3, 60, (2, S), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------ D1
def test_schedule_and_columns_of_the_150m_models():
    c = PerceiverARConfig(hidden_size=896, num_attention_heads=14, head_dim=64, num_kv_heads=1, par_mode="perceiver",
                          pre_layers=3, global_layers=1, stack_layers=9, message_loop_rounds=4, global_nope=True,
                          par_global_read="concept", message_boundary_token_id=-1)
    sch = run_schedule(c)
    assert sch == [0, 1, 2] + [3, 4] * 4 + list(range(5, 13)) and len(sch) == 19
    read = DepthMix(PerceiverARConfig(**{**c.to_dict(), "par_depth_mix": "read"}))
    assert sorted(int(k) for k in read.runs) == [3, 5, 7, 9]
    assert [read.runs[str(t)][0].n_cols for t in (3, 5, 7, 9)] == [4, 6, 8, 10]
    n_read = sum(p.numel() for p in read.parameters())
    assert n_read == 3 * 896 * (4 + 6 + 8 + 10) + 3 * 4 * 896          # under 0.1 % of 141.5M
    dca = DepthMix(PerceiverARConfig(**{**c.to_dict(), "par_depth_mix": "dca2"}))
    assert len(dca.runs) == 19 and [dca.runs[str(t)][0].n_cols for t in range(6)] == [1, 2, 3, 4, 4, 4]
    assert [dca2_columns(t) for t in range(19)] == [min(t + 1, 4) for t in range(19)]


@pytest.mark.parametrize("mode", ["read", "dca2"])
def test_columns_sum_to_the_run_input(mode):
    torch.manual_seed(0)
    _, b = _pair(mode, deep=True)
    states = [torch.randn(2, 7, 32) for _ in range(12)]
    for t in range(12):
        cols = b.depth_mix.columns(states[:t + 1], t)
        torch.testing.assert_close(sum(cols), states[t], atol=1e-5, rtol=1e-5)
        assert len(cols) == (t + 1 if mode == "read" else min(t + 1, 4))


# ------------------------------------------------------------------ D2
@pytest.mark.parametrize("deep", [False, True], ids=["small", "deep"])
@pytest.mark.parametrize("mode", ["read", "dca2"])
def test_identity_at_initialisation(mode, deep):
    a, b = _pair(mode, deep)
    ids = _ids()
    with torch.no_grad():
        la, lb = a(ids).logits, b(ids).logits
    assert torch.equal(la, lb), (la - lb).abs().max()
    a.train(), b.train()
    ra, rb = a(ids, labels=ids), b(ids, labels=ids)
    assert torch.equal(ra.loss, rb.loss)


# ------------------------------------------------------------------ D3
def test_relu0_has_gradient_one_at_zero():
    z = torch.zeros(3, requires_grad=True)
    relu0(z).sum().backward()
    assert torch.equal(z.grad, torch.ones(3))
    g = GRN3(2, 4)
    c = [torch.randn(1, 3, 4), torch.randn(1, 3, 4)]
    x = c[0] + c[1]
    assert torch.equal(g(c, x), x)


@pytest.mark.parametrize("mode", ["read", "dca2"])
def test_gradients_reach_the_mixes_and_they_learn(mode):
    _, b = _pair(mode)
    b.train()
    ids = _ids()
    out = b(ids, labels=ids)
    out.loss.backward()
    for name, p in b.depth_mix.named_parameters():
        assert p.grad is not None and float(p.grad.abs().max()) > 0, name
    opt = torch.optim.Adam(b.depth_mix.parameters(), lr=3e-2)
    first = float(out.loss)
    for _ in range(25):
        opt.zero_grad()
        loss = b(ids, labels=ids).loss
        loss.backward()
        opt.step()
    assert float(loss) < first - 1e-3


# ------------------------------------------------------------------ D4
@pytest.mark.parametrize("mode", ["read", "dca2"])
def test_no_future_leak_with_moved_mixes(mode):
    _, b = _pair(mode)
    for p in b.depth_mix.parameters():
        p.data.normal_(0, 0.3)
    ids = _ids(seed=9)[:1]
    with torch.no_grad():
        base = b(ids).logits
        for t in range(40, 96, 11):
            ids2 = ids.clone()
            ids2[0, t] = (ids2[0, t] + 7) % 57 + 3
            d = (base - b(ids2).logits)[0, :t].abs().amax(-1)
            assert (d < 1e-5).all(), (t, d.nonzero().flatten()[:5].tolist())


# ------------------------------------------------------------------ D5
@pytest.mark.parametrize("mode", ["read", "dca2"])
def test_gradient_checkpointing_matches(mode):
    _, b = _pair(mode)
    for p in b.depth_mix.parameters():
        p.data.normal_(0, 0.2)
    b.train()
    ids = _ids(seed=11)
    grads = []
    for ckpt in (False, True):
        b.gradient_checkpointing = ckpt
        b.zero_grad()
        loss = b(ids, labels=ids).loss
        loss.backward()
        grads.append((float(loss), {n: p.grad.clone() for n, p in b.named_parameters() if p.grad is not None}))
    (l0, g0), (l1, g1) = grads
    assert abs(l0 - l1) < 1e-6 and g0.keys() == g1.keys()
    for n in g0:
        torch.testing.assert_close(g0[n], g1[n], atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------ D6
def test_config_round_trip_and_refusals():
    c = base_cfg(**CONCEPT, par_depth_mix="read")
    assert PerceiverARConfig(**c.to_dict()).par_depth_mix == "read"
    with pytest.raises(ValueError, match="par_depth_mix"):
        base_cfg(**CONCEPT, par_depth_mix="all")
    _, b = _pair("read")
    with pytest.raises(NotImplementedError):
        b._run_layers(_ids(), None, None, None, capture_input_of=1)


# ------------------------------------------------------------------ the shared three-mix kernel
@pytest.mark.parametrize("J", [1, 4, 10])
def test_mix3_equals_three_grn3(J):
    from nn.depth_mix import mix3

    torch.manual_seed(J)
    d = 16
    g = [GRN3(J, d) for _ in range(3)]
    for m in g:
        m.b_delta.data.normal_(0, 0.3)
        m.w.data.normal_(0, 2.0)
    cols = [torch.randn(2, 5, d) * (j + 1) for j in range(J)]
    x = sum(cols)
    want = [m(cols, x) for m in g]
    got = mix3(x, torch.stack([m.w for m in g]), torch.stack([m.b_delta for m in g]), cols)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("J", [4, 10])
def test_mix3_compiled_matches_eager_with_gradients(J, monkeypatch):
    import nn.depth_mix as dm

    torch.manual_seed(0)
    d, S = 64, 256
    W = (torch.randn(3, d, device="cuda") * 2).requires_grad_()
    Bd = (torch.randn(3, J, d, device="cuda") * 0.3).requires_grad_()
    cols = [torch.randn(2, S, d, device="cuda", requires_grad=True) for _ in range(J)]
    x = sum(cols).detach().requires_grad_()
    res = []
    for flag in ("0", "1"):
        monkeypatch.setenv("DEPTH_MIX_COMPILE", flag)
        for t in (W, Bd, x, *cols):
            t.grad = None
        out = dm.mix3(x, W, Bd, cols)
        (sum(o.pow(2).sum() for o in out)).backward()
        res.append(([o.detach() for o in out], [t.grad.clone() for t in (W, Bd, x, *cols)]))
    for a, b in zip(res[0][0] + res[0][1], res[1][0] + res[1][1]):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
