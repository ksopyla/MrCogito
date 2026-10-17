"""E33h — count heads: a summing read beside the global read. Each head adds σ(a·q·k + b) over the notes of closed
pages (each token of the document once: a page owns its first `stride` tokens) and the raw tokens no closed page owns
yet, and adds the coded count to the read through a zero-initialised projection.

Roadmap: docs/4_Research_Notes/text_t4_t7_roadmap_20261017.md §4.5.
"""
import math

import torch

from nn.perceiver_ar_lm import note_count
from tests.test_e33d_note_age import DEV, LOOP, WIDE
from tests.test_text_memory import TEXT, make, text_row

COUNT = dict(lm_count_heads=2)


def _ctx_and_keys(S=200, seed=0, **kw):
    torch.manual_seed(seed)
    m = make(**TEXT, **WIDE, **kw).to(DEV)
    ids = text_row(S=S).to(DEV)
    with torch.no_grad():
        m(ids)
    return m, m._last_message_ctx


def _brute(qc, kb, kr, ctx, scale, window, stride, a, b):
    B, S, nc, dh = qc.shape
    C = kb.shape[1]
    out = torch.zeros(B, S, nc)
    for bi in range(B):
        for t in range(S):
            d = int(ctx.doc[bi, t])
            closes = [int(ctx.slot_close[bi, c]) for c in range(C)
                      if ctx.slot_doc[bi, c] == d and 0 <= ctx.slot_close[bi, c] <= t]
            last = max(closes) if closes else -(10 ** 9)
            for h in range(nc):
                tot = 0.0
                for c in range(C):
                    sc, sp = int(ctx.slot_close[bi, c]), int(ctx.slot_pos[bi, c])
                    if ctx.slot_doc[bi, c] != d or not (0 <= sc <= t) or sp - (sc - window + 1) >= stride:
                        continue
                    tot += 1 / (1 + math.exp(-(a[h] * float(qc[bi, t, h] @ kb[bi, c, 0]) * scale + b[h])))
                for r in range(max(0, t - window + 1), t + 1):
                    if int(ctx.doc[bi, r]) != d or d < 0 or r <= last - (window - stride):
                        continue
                    tot += 1 / (1 + math.exp(-(a[h] * float(qc[bi, t, h] @ kr[bi, r, 0]) * scale + b[h])))
                out[bi, t, h] = tot
    return out


def test_count_matches_the_brute_force_reference():
    m, ctx = _ctx_and_keys(S=120, lm_window=32, lm_stride=24, message_raw_window=32)
    ctx = type(ctx)(**{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in vars(ctx).items()})
    S, C = ctx.doc.shape[1], ctx.slot_doc.shape[1]
    torch.manual_seed(1)
    qc, kb, kr = torch.randn(1, S, 2, 8), torch.randn(1, C, 1, 8), torch.randn(1, S, 1, 8)
    a, b = torch.tensor([1.0, 0.5]), torch.tensor([-1.0, 0.5])
    got = note_count(qc, kb, kr, ctx, scale=8 ** -0.5, window=32, stride=24, a=a, b=b, chunk=17)
    ref = _brute(qc, kb, kr, ctx, 8 ** -0.5, 32, 24, a.tolist(), b.tolist())
    assert torch.allclose(got, ref, atol=1e-4), float((got - ref).abs().max())


def test_each_event_is_counted_once_before_and_after_its_page_closes():
    """Events are tokens whose raw key matches, and (once their page has closed) one note per event standing at the
    event's position. The count at t must be the number of events up to t at every t: raw while no closed page owns
    the event, from its note afterwards, never both (pages overlap by window − stride tokens)."""
    m, ctx = _ctx_and_keys(S=150, lm_window=32, lm_stride=24, message_raw_window=32)
    ctx = type(ctx)(**{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in vars(ctx).items()})
    S, C = ctx.doc.shape[1], ctx.slot_doc.shape[1]
    sc, sp = ctx.slot_close[0], ctx.slot_pos[0]
    owned = [c for c in range(C) if ctx.slot_doc[0, c] >= 0 and sc[c] >= 0 and int(sp[c]) - (int(sc[c]) - 32 + 1) < 24]
    events, note_of = [], {}
    for c in owned:
        e = int(sp[c])
        if e not in note_of:
            note_of[e] = c
            events.append(e)
    events = sorted(events)[::3][:8]
    assert len(events) >= 3
    kb, kr = torch.zeros(1, C, 1, 4), torch.zeros(1, S, 1, 4)
    for e in events:
        kb[0, note_of[e], 0] = 1.0
        kr[0, e, 0] = 1.0
    qc = torch.ones(1, S, 1, 4)
    n = note_count(qc, kb, kr, ctx, scale=1.0, window=32, stride=24, a=torch.tensor([10.0]), b=torch.tensor([-20.0]))
    want = torch.tensor([sum(e <= t for e in events) for t in range(S)], dtype=torch.float32)
    valid = ctx.doc[0] == ctx.doc[0, events[0]]
    assert torch.allclose(n[0, valid, 0], want[valid], atol=1e-3), (n[0, valid, 0] - want[valid]).abs().max()


def test_zero_projection_leaves_the_model_unchanged_and_learns():
    torch.manual_seed(3)
    m = make(**TEXT, **LOOP, **WIDE, **COUNT, message_raw_window=8).to(DEV)
    ids = text_row(S=48).to(DEV)
    att = m.layers[m.config.global_layer_index].attn
    assert att.note_counter is not None and not att.note_counter.out.weight.any()
    with torch.no_grad():
        base = m(ids).logits
        cnt = att.note_counter
        att.note_counter = None
        assert torch.allclose(m(ids).logits, base, atol=1e-5)
        att.note_counter = cnt
    out = m(ids, labels=ids)
    out.loss.backward()
    assert cnt.out.weight.grad is not None and cnt.out.weight.grad.abs().sum() > 0
    with torch.no_grad():
        cnt.out.weight.normal_(0, 0.02)
    m.zero_grad()
    m(ids, labels=ids).loss.backward()
    assert cnt.wq.weight.grad.abs().sum() > 0 and cnt.b.grad.abs().sum() > 0


def test_text_checks_arch_e33h_is_e33r_plus_count_heads():
    from evaluation.text_checks import ARCHES, count_params, model_args

    r = model_args("e33r_plain", "screen", 1152)
    h = model_args("e33h_plain", "screen", 1152)
    assert {k for k in h if h[k] != r.get(k)} == {"lm_count_heads"} and ARCHES["e33h_plain"].has_notebook
    assert count_params(h) >= count_params(r)
