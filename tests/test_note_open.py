"""E33p — note-indexed exact read: tied note keys, each query opens the raw tokens of its best notes (or pages).

Spec: docs/experiments_specs/ahead/E33p_note_indexed_exact_read.md · plan: E33p_plan.md (§7, T1–T11, T13, T14).
"""
from __future__ import annotations

import pytest
import torch

from nn.exact_read import (
    ORD_MOD,
    choose,
    counts,
    dense_open,
    doc_starts,
    note_groups,
    note_open_table,
    tied_note_keys,
)
from nn.perceiver_ar_lm import PerceiverARLM, dense_message_mask, make_message_mask_pred
from tests.e33p_golden import GOLDEN, build, compute, golden_doc_ids, golden_rows
from tests.test_latent_memory import lm_cfg
from tests.test_text_memory import TEXT

RW = 32
TIED = dict(**TEXT, message_loop_rounds=4, global_nope=True, message_raw_window=RW, lm_note_key="tied")
NOTES = dict(**TIED, lm_open="notes", lm_open_k=4, lm_open_cap=32)
PAGES = dict(**TIED, lm_open="pages", lm_open_k=1, lm_open_cap=32)


def make(seed=0, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(lm_cfg(**kw)).eval()
    for layer in m.layers:  # warm residuals so every path is visible at init
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def row(S=160, seed=1, B=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 80, (B, S), generator=g)


def packed_docs(S=160, cut=70):
    d = torch.zeros(1, S, dtype=torch.long)
    d[0, cut:] = 1
    return d


# ------------------------------------------------------------------ T1 config
def test_config_defaults_and_validation():
    c = lm_cfg(**TEXT)
    assert (c.lm_note_key, c.lm_open, c.lm_open_k, c.par_global_read) == ("writer", "none", 0, "dense")
    assert lm_cfg(**NOTES).lm_open == "notes"
    bad = [
        dict(NOTES, lm_open_k=0),                      # opening needs k ≥ 1
        dict(TIED, lm_open="notes", lm_open_k=4, lm_note_key="writer"),  # the notes must be the index
        dict(NOTES, message_raw_window=0),             # opened tokens lie beyond the window
        dict(NOTES, global_nope=False),                # q·κ is the tied identity only without RoPE
        dict(NOTES, lm_read_topk=4),
        dict(NOTES, lm_null_latent=True),
        dict(NOTES, lm_reader_tokens=2),
        dict(NOTES, lm_open="everything"),
        dict(NOTES, lm_open_cap=0),
        dict(TIED, lm_note_key="other"),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            lm_cfg(**kw)


# ------------------------------------------------------------------ T2 old configs bit-identical
def test_old_configs_are_bit_identical_to_the_golden_logits():
    import sys

    want = torch.load(GOLDEN, weights_only=True)
    got = compute()
    exact = sys.platform == "darwin"  # the golden file was made on the Mac CPU; other CPUs round differently
    for name in want:
        if exact:
            assert torch.equal(want[name]["logits"], got[name]["logits"]), name
        else:
            assert torch.allclose(want[name]["logits"], got[name]["logits"], atol=1e-4, rtol=1e-4), name
        assert want[name]["keys"] == got[name]["keys"], name
        assert want[name]["n_params"] == got[name]["n_params"], name


def test_tied_arms_have_the_e33_parameters_and_init():
    e33 = build("e33")
    for kw in (NOTES, PAGES, TIED):
        m = make(seed=0, **kw)
        assert sorted(m.state_dict()) == sorted(e33.state_dict())
        for (n, p), (n2, p2) in zip(m.state_dict().items(), e33.state_dict().items()):
            assert n == n2 and torch.equal(p, p2), n


# ------------------------------------------------------------------ helpers on a real forward
def _forward_state(kw=NOTES, S=160, doc=None, seed=0):
    m = make(seed=seed, **kw)
    ids = row(S)
    with torch.no_grad():
        m(ids, doc_ids=doc)
    return m, ids, m._last_message_ctx


# ------------------------------------------------------------------ T3 assignment
def test_assignment_is_the_competition_softmax_before_token_renormalisation():
    m, ids, ctx = _forward_state()
    no = ctx.index
    A = no.assign                                                     # [1, n_w, K, W]
    pick = ctx.window_valid
    s = A.sum(2)                                                      # over notes
    assert torch.allclose(s[pick], torch.ones_like(s[pick]), atol=1e-5)
    assert torch.all(A.permute(0, 1, 3, 2)[~pick] == 0)
    # recompute by hand from the writer's last-round logits
    w = m.memory_writer
    a = w._last_assign                                                # [Bn, hl, Kp, W]
    assert torch.allclose(a[:, :, : no.per_page].float().mean(1).reshape(A.shape), A)


# ------------------------------------------------------------------ T4 tied key
def test_tied_key_is_the_assignment_weighted_mean_of_raw_keys():
    torch.manual_seed(3)
    B, n_w, K, W, S, dh = 1, 3, 4, 8, 20, 8
    starts = [0, 6, 12]
    page_tok = (torch.tensor(starts)[:, None] + torch.arange(W)[None]).clamp(max=S - 1)
    A = torch.rand(B, n_w, K, W)
    k = torch.randn(B, S, 1, dh)
    kap = tied_note_keys(A, k, page_tok)
    q = torch.randn(dh)
    for w in range(n_w):
        for n in range(K):
            ref = sum(A[0, w, n, t] * k[0, page_tok[w, t], 0] for t in range(W)) / A[0, w, n].sum()
            assert torch.allclose(kap[0, w * K + n, 0], ref, atol=1e-5)
            dots = torch.stack([q @ k[0, page_tok[w, t], 0] for t in range(W)])
            assert torch.allclose(q @ kap[0, w * K + n, 0], (A[0, w, n] * dots).sum() / A[0, w, n].sum(), atol=1e-4)


# ------------------------------------------------------------------ T5 membership & codes
def test_membership_codes_and_counts():
    m, ids, ctx = _forward_state(S=160, doc=packed_docs())
    gi = ctx.index.groups
    no = ctx.index
    geo = m.memory_writer.geometry(160)
    best = no.assign.argmax(2)                                        # [1, n_w, W]
    pick = ctx.window_valid
    S = 160
    members: dict[int, list[int]] = {}
    for w, st in enumerate(geo.starts):
        for t in range(geo.window):
            if pick[0, w, t]:
                members.setdefault(w * no.per_page + int(best[0, w, t]), []).append(st + t)
    ds = doc_starts(ctx.doc)
    # every member has one code per covering page (in some layer), with its rank from its document's start
    for g, ps in members.items():
        for p in ps:
            codes = [int(c) for c in gi.codes[:, 0, p] if int(c) >= 0 and int(c) // ORD_MOD == g]
            assert len(codes) == 1, (g, p)
            assert codes[0] % ORD_MOD == sum(1 for x in ps if int(ds[0, p]) <= x < p)
    # no code for a token that is not a member
    for li in range(gi.codes.shape[0]):
        for p in range(S):
            c = int(gi.codes[li, 0, p])
            if c >= 0:
                assert p in members[c // ORD_MOD]
    # counts from the prefix tables equal a brute-force count
    torch.manual_seed(0)
    for _ in range(200):
        g = int(torch.randint(0, gi.n_groups, ()))
        lo, hi = sorted(int(x) for x in torch.randint(0, S, (2,)))
        want = sum(1 for p in members.get(g, []) if lo <= p <= hi)
        got = int(counts(gi, torch.tensor([[g]]), torch.tensor([[lo]]), torch.tensor([[hi]]))[0, 0])
        assert got == want


# ------------------------------------------------------------------ T7 choice = brute force
def _brute_open(q, kappa, ctx, no, raw_window):
    """Python loop per query: the opened positions, as a set, plus the chosen groups in order."""
    B, S, h, dh = q.shape
    K = no.per_page
    gi = no.groups
    C = kappa.shape[1]
    geo_members: dict[int, list[int]] = {}
    for li in range(gi.codes.shape[0]):
        for p in range(S):
            c = int(gi.codes[li, 0, p])
            if c >= 0:
                geo_members.setdefault(c // ORD_MOD, []).append(p)
    for g in geo_members:
        geo_members[g].sort()
    out = []
    for i in range(S):
        vis = [n for n in range(C) if int(ctx.slot_doc[0, n]) == int(ctx.doc[0, i]) >= 0
               and 0 <= int(ctx.slot_close[0, n]) <= i]
        s_note = torch.zeros(C)
        if vis:
            lg = torch.einsum("hd,cd->hc", q[0, i].float(), kappa[0, vis, 0].float()) / dh ** 0.5
            s_note[vis] = lg.softmax(-1).sum(0)
        if no.mode == "pages":
            score = s_note.view(-1, K).sum(-1)
            vis_g = sorted({n // K for n in vis})
        else:
            score, vis_g = s_note, vis
        ds = int(doc_starts(ctx.doc)[0, i])
        elig = [g for g in vis_g if any(ds <= p <= i - raw_window for p in geo_members.get(g, []))]
        elig.sort(key=lambda g: -float(score[g]))
        chosen = elig[: no.k]
        opened, used = set(), 0
        for g in chosen:
            for p in geo_members[g]:
                if ds <= p <= i - raw_window:
                    if used < no.cap:
                        opened.add(p)
                    used += 1
        out.append((opened, chosen))
    return out


@pytest.mark.parametrize("kw", [NOTES, PAGES, dict(NOTES, lm_open_cap=7)], ids=["notes", "pages", "cap7"])
def test_choice_equals_brute_force(kw):
    S = 160
    m, ids, ctx = _forward_state(kw=kw, S=S, doc=packed_docs())
    no = ctx.index
    torch.manual_seed(5)
    h, dh = m.config.num_attention_heads, m.config.head_dim
    q = torch.randn(1, S, h, dh)
    kappa = torch.randn(1, ctx.n_slots, 1, dh)
    tab = note_open_table(q, kappa, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc,
                          raw_window=RW, chunk=48)
    got = dense_open(tab.T, no.groups.codes, cap=no.cap, raw_window=RW)[0]           # [S, S]
    ref = _brute_open(q, kappa, ctx, no, RW)
    n_open = 0
    for i in range(S):
        want = ref[i][0]
        have = {int(j) for j in got[i].nonzero().flatten()}
        # the dense rule may also mark tokens of chosen groups inside the window: never, by construction
        assert have == want, (i, sorted(have ^ want))
        n_open += len(want)
    assert n_open > 0
    # chunking does not change the table
    tab2 = note_open_table(q, kappa, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc,
                           raw_window=RW, chunk=0)
    assert torch.equal(tab.T, tab2.T) and torch.equal(tab.need, tab2.need)


def test_window_only_notes_are_skipped():
    """A note whose members all sit inside the query's window never takes a place: the next best does."""
    S = 160
    m, ids, ctx = _forward_state(kw=NOTES, S=S)
    no = ctx.index
    gi = no.groups
    C = ctx.n_slots
    i = 120
    vis = (ctx.slot_doc[0] == 0) & (ctx.slot_close[0] >= 0) & (ctx.slot_close[0] <= i)
    elig = vis & (gi.g_minpos[0] <= i - RW)
    win_only = (vis & ~elig).nonzero().flatten()
    if win_only.numel() == 0:
        pytest.skip("no window-only visible note at this geometry")
    score = torch.zeros(1, 1, C)
    score[0, 0, win_only] = 10.0                                     # the window-only notes score best
    score[0, 0, elig] = torch.linspace(1.0, 0.5, int(elig.sum()))
    ids_, off = choose(score, elig[None, None], gi, torch.tensor([i]), m=no.k, cap=no.cap, raw_window=RW)
    assert not set(ids_[0, 0][off[0, 0] < no.cap].tolist()) & set(win_only.tolist())
    assert (off[0, 0] < no.cap).sum() == min(no.k, int(elig.sum()))


# ------------------------------------------------------------------ T6 open-all = dense
def test_opening_everything_equals_the_dense_read():
    """k ≥ every note, cap ≥ S, one document, notes masked: the read equals a twin whose raw window is unbounded."""
    S = 160
    kw = dict(NOTES, lm_open_k=10_000, lm_open_cap=4096)
    m = make(**kw)
    twin_kw = dict(TIED, message_raw_window=0)
    twin = make(**twin_kw)
    twin.load_state_dict(m.state_dict())
    ids = row(S)
    with torch.no_grad(), m.open_override("gist_off"), twin.open_override("gist_off"):
        a = m(ids).logits
        b = twin(ids).logits
    # tokens outside every page (unpicked) are not members: only pages' picked tokens open; with one document every
    # token is picked, so the read is the full causal read
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_gist_on_matches_a_hand_built_reference():
    S = 160
    m, ids, ctx = _forward_state(kw=NOTES, S=S)
    no = ctx.index
    torch.manual_seed(1)
    h, dh = m.config.num_attention_heads, m.config.head_dim
    q = torch.randn(1, S, h, dh)
    k, v = torch.randn(1, S, 1, dh), torch.randn(1, S, 1, dh)
    kap = tied_note_keys(no.assign, k, no.page_tok)
    vb = torch.randn(1, ctx.n_slots, 1, dh)
    from nn.perceiver_ar_lm import attend_message

    tab = note_open_table(q, kap, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc, raw_window=RW)
    out = attend_message(q, k, v, kap, vb, ctx=ctx, key_valid=None, backend="sdpa", opened=tab, gist=True)
    # reference: explicit softmax over raw (window ∪ opened) ‖ visible notes
    op = dense_open(tab.T, no.groups.codes, cap=no.cap, raw_window=RW)[0]
    ii = torch.arange(S)[:, None]
    jj = torch.arange(S)[None]
    raw = (jj <= ii) & ((ii - jj < RW) | op)
    notes = (ctx.slot_doc[0] >= 0) & (ctx.slot_close[0] >= 0) & (ctx.slot_close[0][None] <= ii)
    mask = torch.cat([raw, notes], -1)
    lg = torch.einsum("shd,td->hst", q[0], torch.cat([k, kap.to(k.dtype)], 1)[0, :, 0]) / dh ** 0.5
    p = lg.masked_fill(~mask[None], float("-inf")).softmax(-1)
    ref = torch.einsum("hst,td->shd", p, torch.cat([v, vb], 1)[0, :, 0])
    assert torch.allclose(out[0], ref, atol=1e-5)


# ------------------------------------------------------------------ T8 causality & window
def _future_leaks(m, ids, doc=None):
    bad = []
    with torch.no_grad():
        base = m(ids, doc_ids=doc).logits
        for t in range(0, ids.shape[1], 7):
            ids2 = ids.clone()
            ids2[0, t] = (ids2[0, t] + 7) % 77 + 3
            d = (base - m(ids2, doc_ids=doc).logits)[0, :t].abs().amax(-1)
            moved = (d > 1e-5).nonzero().flatten().tolist()
            if moved:
                bad.append((t, moved[:5]))
    return bad


@pytest.mark.parametrize("kw", [NOTES, PAGES], ids=["notes", "pages"])
@pytest.mark.parametrize("packed", [False, True])
def test_no_future_leak(kw, packed):
    m = make(**kw)
    ids = row(160)
    doc = packed_docs() if packed else None
    assert _future_leaks(m, ids, doc) == []


def test_opened_entries_are_old_same_document_and_closed():
    S = 160
    m, ids, ctx = _forward_state(kw=NOTES, S=S, doc=packed_docs())
    no = ctx.index
    torch.manual_seed(2)
    q = torch.randn(1, S, m.config.num_attention_heads, m.config.head_dim)
    kap = torch.randn(1, ctx.n_slots, 1, m.config.head_dim)
    tab = note_open_table(q, kap, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc, raw_window=RW)
    op = dense_open(tab.T, no.groups.codes, cap=no.cap, raw_window=RW)[0]
    i, j = op.nonzero(as_tuple=True)
    assert i.numel() > 0
    assert torch.all(j <= i - RW)
    assert torch.all(ctx.doc[0, i] == ctx.doc[0, j])
    assert (op.sum(-1) <= no.cap).all()


# ------------------------------------------------------------------ T9 flex predicate = dense mask
def test_flex_predicate_equals_the_dense_mask():
    from nn.exact_read import open_term

    S = 160
    m, ids, ctx = _forward_state(kw=NOTES, S=S, doc=packed_docs())
    no = ctx.index
    torch.manual_seed(4)
    q = torch.randn(1, S, m.config.num_attention_heads, m.config.head_dim)
    kap = torch.randn(1, ctx.n_slots, 1, m.config.head_dim)
    tab = note_open_table(q, kap, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc, raw_window=RW)
    for gist in (True, False):
        term = open_term(tab.T, no.groups.codes, cap=no.cap, raw_window=RW)
        pred = make_message_mask_pred(S, ctx, None, opened=term, gist=gist)
        qi = torch.arange(S)[:, None].expand(S, S + ctx.n_slots)
        kv = torch.arange(S + ctx.n_slots)[None].expand(S, S + ctx.n_slots)
        b = torch.zeros_like(qi)
        flex = pred(b, None, qi, kv)
        od = dense_open(tab.T, no.groups.codes, cap=no.cap, raw_window=RW)
        dense = dense_message_mask(S, ctx, None, q.device, opened=od, gist=gist)[0, 0]
        assert torch.equal(flex, dense)
    # the key-block lists cover every opened entry
    ex = tab.extra_blocks()[0]                                        # [nqb, nkb]
    op = dense_open(tab.T, no.groups.codes, cap=no.cap, raw_window=RW)[0]
    for i, j in zip(*op.nonzero(as_tuple=True)):
        assert int(j) // 128 in set(ex[int(i) // 128].tolist())


# ------------------------------------------------------------------ T10 gradients
def test_gradients_reach_the_index_and_not_through_the_choice():
    S = 160
    m = make(**dict(NOTES)).train()
    ids = row(S)
    with m.open_override("off"):
        out = m(ids, labels=ids)
    out.loss.backward()
    w = m.memory_writer
    g = m.layers[m.config.global_layer_index].attn
    assert w.wq.weight.grad is not None and w.wq.weight.grad.abs().sum() > 0      # through A (competition logits)
    assert g.wk.weight.grad.abs().sum() > 0
    for mod in (w.to_k, w.to_k_read):                                               # the writer's own key is unused
        assert mod.weight.grad is None or mod.weight.grad.abs().sum() == 0


def test_value_gradient_reaches_a_token_only_as_an_opened_token():
    """Loss on the last position only: a value far outside every window reaches it only as an opened token."""
    S = 160
    torch.manual_seed(0)
    kw = dict(NOTES, lm_open_k=10_000, lm_open_cap=4096, message_loop_rounds=1)
    m = make(**kw)
    ids = row(S)
    gi = m.config.global_layer_index
    attn = m.layers[gi].attn
    cache = {}

    def hook(mod, inp, outp):
        outp.retain_grad()
        cache["v"] = outp

    h = attn.wv.register_forward_hook(hook)
    for mode, expect_nonzero in ((None, True), ("off", False)):
        m.zero_grad()
        with m.open_override(mode):
            logits = m(ids).logits
        logits[0, -1].sum().backward()
        gv = cache["v"].grad[0, 10].abs().sum()
        assert (gv > 0) == expect_nonzero, (mode, float(gv))
    h.remove()


# ------------------------------------------------------------------ T11 switches
def test_open_off_equals_a_no_open_tied_twin_and_switches_restore():
    S = 160
    m = make(**NOTES)
    twin = make(**TIED)
    twin.load_state_dict(m.state_dict())
    ids = row(S)
    with torch.no_grad():
        with m.open_override("off"):
            a = m(ids).logits
        b = twin(ids).logits
        assert torch.allclose(a, b, atol=1e-6)
        on = m(ids).logits
        assert not torch.allclose(on, a, atol=1e-4)                   # opening changes the read
        with pytest.raises(ValueError):
            with m.open_override("bogus"):
                pass
        assert m._open_eval == "on"
        with m.open_override("gist_off"):
            c = m(ids).logits
        assert not torch.allclose(c, on, atol=1e-4)
        assert torch.equal(m(ids).logits, on)


# ------------------------------------------------------------------ T13 trains
@pytest.mark.parametrize("kw", [NOTES, PAGES], ids=["notes", "pages"])
def test_trains_two_steps(kw):
    m = make(**kw).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ids = row(160, B=2)
    losses = []
    for _ in range(2):
        out = m(ids, labels=ids)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(out.loss.detach()))
    assert all(map(lambda x: x == x, losses))


# ------------------------------------------------------------------ T13 the driver's arms through the trainer's factory
def test_driver_arms_count_like_their_controls_and_every_arm_loops():
    from types import SimpleNamespace

    from scripts.run_e33n import EOT, VOCAB, count, model_args
    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    assert count(model_args("notes_open", "s30")) == count(model_args("e33", "s30"))
    assert count(model_args("pages_open", "s30")) == count(model_args("e33", "s30"))
    assert count(model_args("lsa", "s30")) == count(model_args("dense", "s30"))
    fields = set(ModelArguments.__dataclass_fields__)
    tok = type("T", (), {"pad_token_id": EOT, "bos_token_id": EOT, "eos_token_id": EOT,
                         "__len__": lambda self: VOCAB})()
    for arm in ("dense", "lsa", "e33", "notes_open"):
        ma = ModelArguments(**{k: v for k, v in model_args(arm, "tiny").items() if k in fields})
        ma.attn_backend, ma.use_liger = "sdpa", False
        with torch.device("meta"):
            m, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=512, tokenizer_name="gpt2"))
        assert m.config.message_loop_rounds == 4, arm      # E33n's dense arm was built with 1 (factory bug)
        if arm == "notes_open":
            assert m.config.lm_open == "notes" and m.config.lm_note_key == "tied"
        if arm == "lsa":
            assert m.config.par_global_read == "lsa" and m.config.lsa_top_k == 64
