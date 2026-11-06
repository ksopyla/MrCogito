"""E33p — the LSA-native arm: LSA's released flat read on our reading layer (nn/lsa_buckets.py).

Plan: docs/experiments_specs/ahead/E33p_plan.md §7 (T1, T6, T8, T11, T12, T13 for the LSA arm).

T12 holds the port to LSA's own code: `_lsa_build_flat_buckets`, `_lsa_members`, `_lsa_scores` and
`_lsa_select_indices` below are verbatim copies (renamed only) of github.com/Prefrontal-AI/latent-sparse-attention
@ 5ebea73, src/lsa/training/native/lsa.py and src/lsa/training/native/kernels/reference.py:

    Copyright 2026 Prefrontal AI
    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
    with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
    (full text: third_party/LSA_LICENSE). Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nn.exact_read import dense_open
from nn.lsa_buckets import closed_block_count, flat_buckets, lsa_groups, lsa_open_table, pool_keys, ward_rounds
from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg

# ---------------------------------------------------------------- LSA reference (verbatim, Apache-2.0, see above)
PAD = -1
HEAD_SCORE_ELEMENTS = 2**26


def _lsa_ward_merge(pool_keys_, pool_sizes, bucket_count, bucket_cap):
    bucket_size = pool_sizes.sum(1).max().item() / bucket_count
    if bucket_cap < 2 * bucket_size:
        raise ValueError(
            f"the bucket cap must be at least twice the bucket size, {2 * bucket_size:g} - got {bucket_cap}"
        )
    labels, reached = ward_rounds(pool_keys_, pool_sizes, bucket_count, bucket_cap)
    if not reached.all():
        raise ValueError(
            f"{int((~reached).sum())} blocks stopped short of {bucket_count} buckets under the cap"
        )
    return labels


@torch.no_grad()
def _lsa_build_flat_buckets(index_keys, config):
    sequence_count, _, index_dim = index_keys.shape
    block_size, per_block = config.block_size, config.buckets_per_block
    blocks = closed_block_count(index_keys.shape[1], block_size)
    device = index_keys.device
    rows = (
        index_keys[:, : blocks * block_size].float().reshape(sequence_count * blocks, block_size, index_dim)
    )
    pools = pool_keys(rows, config.pool_size)
    pool_sizes = torch.full(pools.shape[:2], float(config.pool_size), device=device)
    labels = _lsa_ward_merge(pools, pool_sizes, per_block, config.bucket_cap)
    # Bucket of each key within its block [N * blocks, block_size]
    key_labels = labels.repeat_interleave(config.pool_size, dim=1)
    key_sums = torch.zeros(rows.shape[0], per_block, index_dim, device=device)
    key_sums.scatter_add_(1, key_labels[..., None].expand(-1, -1, index_dim), rows)
    centroids = key_sums / key_sums.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    members = _lsa_members(key_labels, config)
    block_starts = block_size * (torch.arange(rows.shape[0], device=device) % blocks)
    members = torch.where(members >= 0, members + block_starts[:, None, None], -1)
    return (
        centroids.reshape(sequence_count, blocks * per_block, index_dim),
        members.reshape(sequence_count, blocks * per_block, config.bucket_cap),
    )


def _lsa_members(key_labels, config):
    groups, block_size = key_labels.shape
    per_block = config.buckets_per_block
    device = key_labels.device
    order = key_labels.argsort(dim=1, stable=True)
    sorted_labels = key_labels.gather(1, order)
    counts = torch.nn.functional.one_hot(key_labels, per_block).sum(1)
    starts = counts.cumsum(1) - counts
    rank = torch.arange(block_size, device=device)[None] - starts.gather(1, sorted_labels)
    members = torch.full((groups, per_block, config.bucket_cap), -1, dtype=torch.int32, device=device)
    members[torch.arange(groups, device=device)[:, None], sorted_labels, rank] = order.to(torch.int32)
    return members


def _lsa_scores(index_queries, centroids):
    q = index_queries.float()
    if q.dim() == 3:
        return torch.einsum("bnd,bmd->bnm", q, centroids)
    b, n, h, _ = q.shape
    rows = max(1, HEAD_SCORE_ELEMENTS // (b * h * max(1, centroids.shape[1])))
    out = [torch.einsum("bnhd,bmd->bnhm", q[:, s : s + rows], centroids).amax(2) for s in range(0, n, rows)]
    return torch.cat(out, dim=1)


def _lsa_select_indices(index_queries, q_pos, buckets, block, bucket_size, top_p, top_k, window_size, sink_size):
    b, n_q = index_queries.shape[:2]
    selected = torch.full((b, n_q, top_k), PAD, dtype=torch.long, device=index_queries.device)
    if buckets is None:
        return selected
    centroids, members = buckets
    m, cap = members.shape[1:]
    bucket_block = torch.arange(m, device=index_queries.device) // (block // bucket_size)
    pick_count = min(top_p, m)
    rows_at_once = max(1, 2**25 // (b * pick_count * cap))
    batch = torch.arange(b, device=index_queries.device)[:, None, None]
    for s in range(0, n_q, rows_at_once):
        pos = q_pos[s : s + rows_at_once]
        visible = bucket_block[None, :] < (pos // block)[:, None]
        ranked = _lsa_scores(index_queries[:, s : s + rows_at_once], centroids)
        top = ranked.masked_fill(~visible, float("-inf")).topk(pick_count, dim=-1)
        ids = members[batch, top.indices].flatten(2)
        live = (top.values > float("-inf")).repeat_interleave(cap, dim=-1)
        valid = live & (ids >= sink_size) & (ids <= (pos - window_size)[None, :, None])
        rank = valid.cumsum(-1) - 1
        taken = valid & (rank < top_k)
        # Every key not taken lands in a spare last slot, dropped after
        packed = torch.full((b, len(pos), top_k + 1), PAD, dtype=torch.long, device=index_queries.device)
        packed.scatter_(-1, torch.where(taken, rank, top_k), torch.where(taken, ids, PAD))
        selected[:, s : s + rows_at_once] = packed[..., :top_k]
    return selected
# ---------------------------------------------------------------- end of the LSA reference


GEO = dict(block=32, bucket=8, pool=4, cap=16, top_k=16, window=32, sinks=4)


def _lsa_config():
    return SimpleNamespace(block_size=GEO["block"], buckets_per_block=GEO["block"] // GEO["bucket"],
                           pool_size=GEO["pool"], bucket_cap=GEO["cap"])


def _qk(S=200, h=3, d=8, seed=0, B=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, S, h, d, generator=g), torch.randn(B, S, 1, d, generator=g)


def _ours(q, k, doc):
    built = lsa_groups(k, doc, block=GEO["block"], per_block=GEO["block"] // GEO["bucket"], pool=GEO["pool"],
                       cap=GEO["cap"], sinks=GEO["sinks"])
    centroids, gi = built
    width = 2 * GEO["top_k"] // GEO["bucket"]
    tab = lsa_open_table(q, centroids, gi, top_k=GEO["top_k"], width=width, raw_window=GEO["window"], chunk=64)
    return centroids, gi, tab, width


# ------------------------------------------------------------------ T12 port = LSA
def test_buckets_equal_lsa():
    q, k = _qk()
    keys = F.normalize(k[:, :, 0].float(), dim=-1)
    want_c, want_m = _lsa_build_flat_buckets(keys, _lsa_config())
    got_c, labels = flat_buckets(keys, block=GEO["block"], per_block=GEO["block"] // GEO["bucket"], pool=GEO["pool"],
                                 cap=GEO["cap"])
    assert torch.equal(got_c, want_c)
    B, nblk, blk = labels.shape
    per = GEO["block"] // GEO["bucket"]
    for b in range(B):
        for g in range(nblk * per):
            mine = sorted((g // per) * blk + t for t in range(blk) if int(labels[b, g // per, t]) == g % per)
            theirs = sorted(int(x) for x in want_m[b, g] if int(x) >= 0)
            assert mine == theirs


def test_selection_equals_lsa_on_one_document():
    S = 200
    q, k = _qk(S=S)
    B = q.shape[0]
    doc = torch.zeros(B, S, dtype=torch.long)
    centroids, gi, tab, width = _ours(q, k, doc)
    iq, ik = F.normalize(q.float(), dim=-1), F.normalize(k[:, :, 0].float(), dim=-1)
    buckets = _lsa_build_flat_buckets(ik, _lsa_config())
    sel = _lsa_select_indices(iq, torch.arange(S), (buckets[0], buckets[1].long()), GEO["block"], GEO["bucket"],
                              width, GEO["top_k"], GEO["window"], GEO["sinks"])
    got = dense_open(tab.T, gi.codes, cap=GEO["top_k"], raw_window=GEO["window"])
    n = 0
    for b in range(B):
        for i in range(S):
            want = {int(x) for x in sel[b, i] if int(x) >= 0}
            have = {int(x) for x in got[b, i].nonzero().flatten()}
            assert have == want, (b, i, sorted(have ^ want))
            n += len(want)
    assert n > 0


def test_packed_documents_take_only_own_document_keys():
    """Our change: a key of another document is never taken and does not use a place; sinks per document."""
    S, cut = 200, 90
    q, k = _qk(S=S, B=1)
    doc = torch.zeros(1, S, dtype=torch.long)
    doc[0, cut:] = 1
    centroids, gi, tab, width = _ours(q, k, doc)
    got = dense_open(tab.T, gi.codes, cap=GEO["top_k"], raw_window=GEO["window"], doc=doc)[0]
    iq = F.normalize(q.float(), dim=-1)
    labels = flat_buckets(F.normalize(k[:, :, 0].float(), dim=-1), block=GEO["block"],
                          per_block=GEO["block"] // GEO["bucket"], pool=GEO["pool"], cap=GEO["cap"])[1][0]
    per = GEO["block"] // GEO["bucket"]
    members = {}
    for blk in range(labels.shape[0]):
        for t in range(GEO["block"]):
            members.setdefault(blk * per + int(labels[blk, t]), []).append(blk * GEO["block"] + t)
    for i in range(S):
        ds = 0 if i < cut else cut
        vis = [g for g in range(centroids.shape[1]) if (g // per + 1) * GEO["block"] <= i]
        sc = (iq[0, i] @ centroids[0, vis].T).amax(0) if vis else torch.zeros(0)
        order = [vis[int(x)] for x in sc.topk(min(width, len(vis))).indices] if vis else []
        want, used = set(), 0
        for g in order:
            for p in members[g]:
                if ds + GEO["sinks"] <= p <= i - GEO["window"]:
                    if used < GEO["top_k"]:
                        want.add(p)
                    used += 1
        have = {int(x) for x in got[i].nonzero().flatten()}
        assert have == want, (i, sorted(have ^ want))


# ------------------------------------------------------------------ the LSA arm in the model
LSA = dict(message_boundary_token_id=-1, message_loop_rounds=4, global_nope=True, num_kv_heads=1,
           par_global_read="lsa", lsa_window=32, lsa_block=32, lsa_bucket=8, lsa_pool=4, lsa_bucket_cap=16,
           lsa_top_k=16, lsa_sinks=4)
DENSE = dict(message_boundary_token_id=-1, message_loop_rounds=4, global_nope=True, num_kv_heads=1)


def make(seed=0, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(cfg(**kw)).eval()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def row(S=200, seed=1, B=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 80, (B, S), generator=g)


def test_lsa_config_validation():
    c = cfg(**LSA)
    assert c.par_global_read == "lsa" and c.lsa_width == 4
    for bad in (dict(LSA, num_kv_heads=2), dict(LSA, global_nope=False), dict(LSA, lsa_bucket_cap=8),
                dict(LSA, lsa_bucket=6), dict(LSA, par_global_read="other"), dict(LSA, attn_backend="flash")):
        with pytest.raises(ValueError):
            cfg(**bad)


def test_lsa_arm_has_the_dense_parameters():
    a, d = make(**LSA), make(**DENSE)
    assert sorted(a.state_dict()) == sorted(d.state_dict())
    for (n, p), (n2, p2) in zip(a.state_dict().items(), d.state_dict().items()):
        assert n == n2 and torch.equal(p, p2)


def test_lsa_with_everything_taken_equals_the_dense_read():
    a = make(**dict(LSA, lsa_top_k=4096, lsa_width=4096))
    d = make(**DENSE)
    ids = row()
    with torch.no_grad():
        assert torch.allclose(a(ids).logits, d(ids).logits, atol=1e-5)


def test_lsa_off_reads_sinks_and_window_only():
    a = make(**LSA)
    ids = row()
    with torch.no_grad():
        on = a(ids).logits
        with a.open_override("off"):
            off = a(ids).logits
        assert not torch.allclose(on, off, atol=1e-4)
        assert torch.equal(a(ids).logits, on)


@pytest.mark.parametrize("packed", [False, True])
def test_lsa_no_future_leak(packed):
    m = make(**LSA)
    ids = row()
    doc = None
    if packed:
        doc = torch.zeros(1, 200, dtype=torch.long)
        doc[0, 90:] = 1
    with torch.no_grad():
        base = m(ids, doc_ids=doc).logits
        for t in range(0, 200, 9):
            ids2 = ids.clone()
            ids2[0, t] = (ids2[0, t] + 7) % 77 + 3
            d = (base - m(ids2, doc_ids=doc).logits)[0, :t].abs().amax(-1)
            assert float(d.max()) <= 1e-5 if t else True, (t, float(d.max()))


def test_lsa_trains_two_steps():
    m = make(**LSA).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ids = row(B=2)
    for _ in range(2):
        out = m(ids, labels=ids)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
