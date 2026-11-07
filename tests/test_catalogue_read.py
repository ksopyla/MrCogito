"""E33s — the catalogue read: the notes only choose what opens; balanced filing, max-head cosine choice, a new place per
loop pass, a two-level page → note search, learned empty slots, the index loss.

Spec: docs/experiments_specs/ahead/E33s_catalogue_read.md · plan: E33s_plan.md (§7).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nn.exact_read import (
    balance_assign,
    counts,
    file_capped,
    index_loss,
    note_open_table,
    page_keys,
    tied_note_keys,
)
from nn.perceiver_ar_lm import PerceiverARLM, _attend_catalogue
from tests.test_latent_memory import lm_cfg
from tests.test_note_open import RW, TIED, _future_leaks, packed_docs, row

CAT = dict(**TIED, lm_stride=16, lm_open="notes", lm_open_k=8, lm_open_cap=24, lm_note_read="index",
           lm_filing="balanced", lm_filing_cap=6, lm_open_score="max_cos", lm_open_new_per_pass=True, lm_sinks=2,
           lm_index_loss=0.1, lm_index_queries=16, lm_index_negatives=3)
CAT2 = dict(CAT, lm_open_levels=2, lm_open_pages=2)


def make(seed=0, train=False, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(lm_cfg(**kw))
    for layer in m.layers:  # warm residuals so every path is visible at init
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m.train() if train else m.eval()


def state(kw=CAT, S=160, doc=None, seed=0):
    m = make(seed=seed, **kw)
    ids = row(S)
    with torch.no_grad():
        m(ids, doc_ids=doc)
    return m, ids, m._last_message_ctx


# ------------------------------------------------------------------ config
def test_config_validation():
    with pytest.raises(ValueError):
        lm_cfg(**dict(TIED, lm_open="notes", lm_open_k=4, lm_sinks=2))                   # slots need index read
    with pytest.raises(ValueError):
        lm_cfg(**dict(TIED, lm_note_read="index"))                                        # index read needs notes
    with pytest.raises(ValueError):
        lm_cfg(**dict(CAT, lm_filing_cap=3))                                              # 3 · 4 notes < 16 tokens
    c = lm_cfg(**CAT)
    assert c.lm_note_read == "index" and c.lm_filing == "balanced" and c.lm_open_new_per_pass


def test_catalogue_adds_only_slots_and_temperature():
    base = make(**dict(TIED, lm_stride=16, lm_open="notes", lm_open_k=8, lm_open_cap=24))
    cat = make(**CAT)
    extra = {n for n, _ in cat.named_parameters()} - {n for n, _ in base.named_parameters()}
    assert extra and all(n.endswith(("sink_k", "sink_v", "index_log_tau")) for n in extra)
    gi = cat.config.global_layer_index
    assert torch.count_nonzero(cat.layers[gi].attn.sink_v) == 0


# ------------------------------------------------------------------ filing
def test_sinkhorn_balances_the_notes_and_keeps_tokens_normalised():
    torch.manual_seed(1)
    B, n_w, K, W = 2, 3, 4, 16
    A = torch.softmax(torch.randn(B, n_w, K, W) * 3, dim=2)
    valid = torch.ones(B, n_w, W, dtype=torch.bool)
    valid[1, 2, 10:] = False
    Ah = balance_assign(A, valid, iters=60)
    assert torch.allclose(Ah.sum(2)[valid], torch.ones(int(valid.sum())), atol=1e-5)
    assert torch.all(Ah.sum(2)[~valid] == 0)
    col = Ah.sum(-1)                                                                      # [B, n_w, K]
    target = valid.sum(-1, keepdim=True).float() / K
    assert torch.allclose(col, target.expand_as(col), rtol=1e-3, atol=1e-3)
    few = balance_assign(A, valid, iters=3).sum(-1)
    raw = (A * valid[:, :, None]).sum(-1)
    assert (few - target).abs().mean() < (raw - target).abs().mean()                     # 3 steps already balance more


@pytest.mark.parametrize("adversarial", [False, True])
def test_capped_filing_files_every_token_once_within_the_cap(adversarial):
    torch.manual_seed(2)
    B, n_w, K, W, cap = 2, 3, 4, 16, 5
    if adversarial:  # every token prefers note 0
        A = torch.full((B, n_w, K, W), 0.01)
        A[:, :, 0] = 0.97
    else:
        A = torch.softmax(torch.randn(B, n_w, K, W) * 3, dim=2)
    valid = torch.ones(B, n_w, W, dtype=torch.bool)
    valid[0, 1, :5] = False
    lab = file_capped(A, valid, cap=cap)
    assert torch.all((lab >= 0) == valid)
    cnt = F.one_hot(lab.clamp_min(0), K) * valid[..., None]
    assert int(cnt.sum(2).max()) <= cap
    if not adversarial:  # without pressure every token gets its best note when it has room
        best = A.argmax(2)
        assert (lab == best)[valid].float().mean() > 0.6


def test_model_filing_is_balanced():
    m, ids, ctx = state()
    no = ctx.index
    assert no.labels is not None and no.read == "index"
    lab, pick = no.labels[0], no.pick[0]
    full = pick.all(-1)
    cnt = F.one_hot(lab[full], no.per_page).sum(1)
    assert int(cnt.max()) <= m.config.lm_filing_cap and int(cnt.min()) >= 1


# ------------------------------------------------------------------ choice
def test_max_cos_choice_takes_the_best_notes_until_the_cap():
    m, ids, ctx = state()
    no = ctx.index
    torch.manual_seed(3)
    S = ids.shape[1]
    q = torch.randn(1, S, m.config.num_attention_heads, m.config.head_dim)
    k = torch.randn(1, S, 1, m.config.head_dim)
    kappa = tied_note_keys(no.assign, k, no.page_tok)
    tab = note_open_table(q, kappa, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc, raw_window=RW)
    gi = no.groups
    G = gi.n_groups
    s = torch.einsum("shd,gd->shg", F.normalize(q[0], dim=-1), F.normalize(kappa[0, :, 0], dim=-1)).amax(1)
    ds = gi.doc_start[0]
    for i in range(RW + 40, S, 17):
        vis = (ctx.slot_doc[0] == ctx.doc[0, i]) & (ctx.slot_close[0] >= 0) & (ctx.slot_close[0] <= i)
        gid = torch.arange(G)
        cnt = counts(gi, gid[None], torch.full((1, G), int(ds[i])), torch.full((1, G), i - RW))[0]
        elig = vis & (cnt > 0)
        order = [int(g) for g in s[i].argsort(descending=True) if bool(elig[g])][: no.k]
        want, used = set(), 0
        for g in order:
            if used >= no.cap:
                break
            want.add(g)
            used += int(cnt[g])
        got = set(int(g) for g in (tab.T[0, i] < no.cap).nonzero().flatten())
        assert got == want, i


def test_a_pass_never_reopens_a_note():
    m, ids, ctx = state()
    no = ctx.index
    S = ids.shape[1]
    seen = None
    torch.manual_seed(4)
    k = torch.randn(1, S, 1, m.config.head_dim)
    kappa = tied_note_keys(no.assign, k, no.page_tok)
    total = 0
    for _ in range(4):
        q = torch.randn(1, S, m.config.num_attention_heads, m.config.head_dim)
        tab = note_open_table(q, kappa, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc,
                              raw_window=RW, exclude=seen)
        chosen = tab.T < tab.cap
        if seen is not None:
            assert not bool((chosen & seen).any())
        total += int(chosen.sum())
        seen = chosen if seen is None else seen | chosen
    assert total > 0 and int(seen.sum()) == total


def test_two_levels_with_every_page_kept_equal_the_flat_search():
    m, ids, ctx = state(kw=dict(CAT2, lm_open_pages=64))
    no = ctx.index
    S = ids.shape[1]
    torch.manual_seed(5)
    q = torch.randn(1, S, m.config.num_attention_heads, m.config.head_dim)
    k = torch.randn(1, S, 1, m.config.head_dim)
    kappa = tied_note_keys(no.assign, k, no.page_tok)
    kw = dict(doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc, raw_window=RW)
    two = note_open_table(q, kappa, no, page_key=page_keys(k, no.page_tok, no.pick), **kw)
    no.levels = 1
    flat = note_open_table(q, kappa, no, **kw)
    assert torch.equal(two.T, flat.T)


def test_two_levels_keep_only_the_best_pages():
    m, ids, ctx = state(kw=CAT2)
    no = ctx.index
    S = ids.shape[1]
    torch.manual_seed(6)
    q = torch.randn(1, S, m.config.num_attention_heads, m.config.head_dim)
    k = torch.randn(1, S, 1, m.config.head_dim)
    kappa = tied_note_keys(no.assign, k, no.page_tok)
    tab = note_open_table(q, kappa, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc, raw_window=RW,
                          page_key=page_keys(k, no.page_tok, no.pick))
    pages = (tab.T[0] < tab.cap).view(S, -1, no.per_page).any(-1).sum(-1)
    assert int(pages.max()) <= no.keep_pages and int(pages.max()) >= 1


# ------------------------------------------------------------------ read
def test_catalogue_read_matches_a_hand_built_softmax():
    m, ids, ctx = state()
    no = ctx.index
    S = ids.shape[1]
    h, dh = m.config.num_attention_heads, m.config.head_dim
    torch.manual_seed(7)
    q, k, v = torch.randn(1, S, h, dh), torch.randn(1, S, 1, dh), torch.randn(1, S, 1, dh)
    sk, sv = torch.randn(2, dh), torch.randn(2, dh)
    kappa = tied_note_keys(no.assign, k, no.page_tok)
    tab = note_open_table(q, kappa, no, doc=ctx.doc, slot_close=ctx.slot_close, slot_doc=ctx.slot_doc, raw_window=RW)
    out = _attend_catalogue(q, k, v, sk, sv, ctx=ctx, key_valid=None, backend="sdpa", opened=tab)
    from nn.exact_read import dense_open
    op = dense_open(tab.T, no.groups.codes, cap=tab.cap, raw_window=RW, doc=ctx.doc)[0]
    for i in (5, RW + 7, 100, S - 1):
        j = torch.arange(S)
        ok = (j <= i) & (ctx.doc[0] == ctx.doc[0, i]) & (((i - j) < RW) | op[i])
        keys = torch.cat([k[0, ok, 0], sk])
        vals = torch.cat([v[0, ok, 0], sv])
        p = torch.softmax(q[0, i] @ keys.T / math.sqrt(dh), dim=-1)
        assert torch.allclose(out[0, i], p @ vals, atol=1e-5), i
    assert bool(op.any())


@pytest.mark.parametrize("kw", [CAT, CAT2], ids=["flat", "two_level"])
@pytest.mark.parametrize("packed", [False, True])
def test_no_future_leak(kw, packed):
    m = make(**kw)
    ids = row(160)
    doc = packed_docs() if packed else None
    assert _future_leaks(m, ids, doc) == []


def test_eval_switches_change_the_read():
    m = make(**CAT)
    ids = row(160)
    with torch.no_grad():
        on = m(ids).logits
        with m.open_override("off"):
            off = m(ids).logits
        with m.open_override("reuse"):
            reuse = m(ids).logits
    assert not torch.allclose(on, off) and not torch.allclose(on, reuse)


# ------------------------------------------------------------------ index loss
def test_index_loss_trains_the_catalogue_not_the_query():
    m, ids, ctx = state()
    no = ctx.index
    S = ids.shape[1]
    h, dh = m.config.num_attention_heads, m.config.head_dim
    torch.manual_seed(8)
    q = torch.randn(1, S, h, dh, requires_grad=True)
    k = torch.randn(1, S, 1, dh, requires_grad=True)
    A = no.assign.detach().clone().requires_grad_(True)
    kappa = tied_note_keys(A, k, no.page_tok)
    tab = note_open_table(q.detach(), kappa.detach(), no, doc=ctx.doc, slot_close=ctx.slot_close,
                          slot_doc=ctx.slot_doc, raw_window=RW)
    log_tau = torch.tensor(math.log(10.0), requires_grad=True)
    loss, st = index_loss(q, k, torch.randn(2, dh), kappa, tab, doc=ctx.doc, raw_window=RW, n_queries=32, n_neg=3,
                          log_tau=log_tau, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss) and float(loss) > 0 and float(st["set"]) >= 2
    loss.backward()
    assert q.grad is None or float(q.grad.abs().sum()) == 0                               # the query is not trained
    assert float(A.grad.abs().sum()) > 0 and float(log_tau.grad.abs()) > 0               # the catalogue and τ are
    assert float(k.grad.abs().sum()) > 0                                                  # keys, through κ


def test_training_forward_adds_the_index_loss_and_reaches_the_writer():
    m = make(train=True, **CAT)
    ids = row(160)
    out = m(ids, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    gi = m.config.global_layer_index
    att = m.layers[gi].attn
    assert att.index_log_tau.grad is not None and float(att.index_log_tau.grad.abs()) > 0
    assert att.sink_k.grad is not None and float(att.sink_v.grad.abs().sum()) > 0
    writer_grads = [p.grad for n, p in m.named_parameters() if n.startswith("memory_writer") and p.grad is not None]
    assert writer_grads and sum(float(g.abs().sum()) for g in writer_grads) > 0
    m0 = make(train=True, **dict(CAT, lm_index_loss=0.0))
    m0.load_state_dict({k_: v_ for k_, v_ in m.state_dict().items() if "index_log_tau" not in k_}, strict=False)
    torch.manual_seed(0)
    l1 = m(ids, labels=ids).loss
    l0 = m0(ids, labels=ids).loss
    assert not torch.allclose(l1, l0)
