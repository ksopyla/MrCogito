"""E33p instrumentation part B: the cross-arm probe's recorder on tiny models (no dataset)."""
from __future__ import annotations

import numpy as np
import torch

from evaluation.flow_probe import ProbeRecorder, pick_queries
from tests.test_flow_stats import make


def _run(m, ids, q, src, ref=None, save_ref=False):
    S = ids.shape[1]
    tok = torch.linspace(0, S - 1, 32).round().long()
    rec = ProbeRecorder(torch.from_numpy(q), tok, ref=ref, save_ref=save_ref, far_src=torch.from_numpy(src),
                        window=32)
    if getattr(m, "memory_writer", None) is not None:
        geo = m.memory_writer.geometry(S)
        st = torch.tensor(geo.starts)
        rec._page_tok = (st[:, None] + torch.arange(geo.window)[None]).clamp(max=S - 1)
    rec.begin(ids)
    m._set_flow(rec)
    with torch.no_grad():
        m.hidden_states(ids)
    m._set_flow(None)
    stats = rec.finish(m)
    return rec, stats


def test_pick_queries_mixes_far_copies_and_random_positions():
    rng = np.random.default_rng(0)
    base = rng.integers(3, 80, 300)
    ids = np.concatenate([base, rng.integers(3, 80, 900), base])  # the last 300 tokens copy the first 300
    q, s = pick_queries(ids, len(ids), 64, seed=1, min_pos=600)
    assert len(q) == 64 and len(set(q.tolist())) == 64 and q.min() >= 600
    assert (s >= 0).sum() >= 16 and np.all(ids[s[s >= 0]] == ids[q[s >= 0] + 1])


class _CharTok:
    """One token per character (ids = code points): enough to place a pass key."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_passkey_row_points_each_answer_query_at_the_key_inside_the_needle():
    import random

    from evaluation.flow_probe import passkey_row

    S = 1200
    filler = [5] * 2000
    ids, q, src, answer = passkey_row(_CharTok(), filler, S, 0.3, random.Random(0))
    assert len(ids) == S and len(answer) == 6 and len(q) == len(src) == 6        # " 12345": space + 5 digits
    assert ids[-len(answer):] == answer
    assert np.all(q + 1 == np.arange(S - 6, S))                                   # each query predicts one answer token
    assert [ids[s] for s in src] == answer and src[0] < 0.3 * S + 30               # the key inside the needle
    assert np.all(q - src > 512)                                                  # beyond the 512 window


def test_probe_dense_ref_then_opening_and_notebook_arms():
    S = 200
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(3, 80, (1, S), generator=g)
    ids[0, 150:180] = ids[0, 20:50]                                         # far copies (beyond the 32-token window)
    q = np.arange(100, 199, 3)
    src = np.full(len(q), -1)
    for n, i in enumerate(q):
        if 150 <= i + 1 < 180:
            src[n] = i + 1 - 130
    dense = make("dense").eval()
    rec, stats = _run(dense, ids, q, src, save_ref=True)
    assert set(rec.new_ref) == {0, 1, 2, 3}
    pos, prob = rec.new_ref[0]
    assert pos.shape == (len(q), 256) or pos.shape[1] <= S
    assert any(k.startswith("lens/") for k in stats)
    ref = rec.new_ref
    for arm in ("notes_open", "lsa", "e33", "window"):
        m = make(arm).eval()
        r2, _ = _run(m, ids, q, src, ref=ref)
        if arm == "window":
            assert not any("recall32" in k for k in r2.metrics)
        else:
            assert "r0/recall32" in r2.metrics and 0 <= r2.metrics["r0/recall32"][0] <= 1, arm
            assert "r0/src_hit" in r2.metrics, arm
        assert "r0/src_mass" in r2.metrics
        assert any(k.startswith("L") for k in r2.acts) and next(iter(r2.acts.values())).shape[0] == 32
        tot = float(r2.dist[0].sum())                       # raw-key attention only (notes take the rest)
        assert set(r2.dist) == {0, 1, 2, 3} and tot <= len(q) + 1e-3
        if arm in ("lsa", "window"):
            assert abs(tot - len(q)) < 1e-3, arm


def test_linear_cka():
    from scripts.compare_flow import linear_cka

    torch.manual_seed(0)
    x = torch.randn(500, 16)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert abs(linear_cka(x, x) - 1) < 1e-9
    assert abs(linear_cka(x, 3 * x @ q) - 1) < 1e-9          # rotation and scale invariant
    assert linear_cka(x, torch.randn(500, 16)) < 0.2
