"""Document packing (data/packed_dataset.py) + collator doc_ids + model equivalence.

The load-bearing test is `test_packed_losses_equal_unpacked`: with cross-document masks,
per-document position reset and doc-bounded n-gram hashes, every token of a packed sequence
must get exactly the loss it gets when its document is run alone.
"""
import json

import numpy as np
import pytest
import torch
from datasets import Dataset

from data.data_collators import DataCollatorForCausalLM
from data.packed_dataset import (
    PackedDataset,
    build_or_load_packed_bins,
    pack_rows,
    packed_cache_path,
    packing_stats,
)
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM, attend

V = 97


def tiny_cfg(**kw):
    base = dict(
        vocab_size=V, hidden_size=32, intermediate_size=64, token_embedding_dim=8,
        par_mode="perceiver", pre_layers=1, pre_window=4, global_layers=1, stack_layers=3, block=6,
        num_attention_heads=4, num_kv_heads=2, head_dim=8, rope_theta=10000.0, nope_every=2,
        ngram_orders=(2, 3), ngram_buckets=64, value_embed_layers=(0, 3), value_embed_dim=4,
        logit_softcap=30.0, z_loss=1e-4, chunked_ce_block_size=5, use_liger=False,
        attn_backend="sdpa", attn_pad_multiple=1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    base.update(kw)
    return PerceiverARConfig(**base)


class _Tok:
    pad_token_id = 0
    unk_token_id = None

    def __len__(self):
        return V


def _realistic_lengths(n, seed=0):
    rng = np.random.default_rng(seed)
    # web-like: log-normal body with a heavy tail, plus a few > capacity rows
    lengths = np.exp(rng.normal(6.5, 1.0, size=n)).astype(np.int64) + 8
    lengths[: n // 50] = rng.integers(6000, 12000, size=n // 50)
    return lengths


# ------------------------------------------------------------------ pack_rows


def test_pack_rows_covers_every_row_once_and_respects_capacity():
    lengths = _realistic_lengths(5000)
    cap = 4096
    bins = pack_rows(lengths, cap, seed=3)
    flat = np.concatenate(bins)
    assert sorted(flat.tolist()) == list(range(len(lengths)))
    for b in bins:
        assert np.minimum(lengths[b], cap).sum() <= cap
    # rows longer than the capacity end up alone
    for b in bins:
        if (lengths[b] > cap).any():
            assert len(b) == 1


def test_pack_rows_is_dense_and_deterministic():
    lengths = _realistic_lengths(20000, seed=1)
    cap = 8192
    a = pack_rows(lengths, cap, seed=11)
    b = pack_rows(lengths, cap, seed=11)
    c = pack_rows(lengths, cap, seed=12)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    assert [x.tolist() for x in a] != [x.tolist() for x in c]
    stats = packing_stats(a, lengths, cap)
    assert stats["fill_ratio"] > 0.97, stats
    assert stats["num_bins"] < len(lengths)


def test_pack_rows_edge_cases():
    assert pack_rows([], 16, seed=0) == []
    bins = pack_rows([16, 16, 16], 16, seed=0)
    assert sorted(len(b) for b in bins) == [1, 1, 1]
    bins = pack_rows([1, 1, 1, 1], 4, seed=0, close_slack=0)
    assert len(bins) == 1 and len(bins[0]) == 4
    with pytest.raises(ValueError):
        pack_rows([1], 0, seed=0)


# ------------------------------------------------------------------ cache


def test_packed_bins_cache_round_trip_and_staleness(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"sources": []}))
    lengths = _realistic_lengths(3000, seed=4)
    bins = build_or_load_packed_bins(manifest, lengths, capacity=2048, seed=5)
    path = packed_cache_path(manifest, 2048, 5)
    assert path.exists()
    again = build_or_load_packed_bins(manifest, lengths, capacity=2048, seed=5)
    assert [b.tolist() for b in bins] == [b.tolist() for b in again]
    # different lengths under the same path -> rebuilt, not served stale
    other = build_or_load_packed_bins(manifest, lengths + 1, capacity=2048, seed=5)
    assert [b.tolist() for b in other] != [b.tolist() for b in bins] or len(other) != len(bins)
    # corrupt file -> rebuilt
    path.write_bytes(b"garbage")
    rebuilt = build_or_load_packed_bins(manifest, lengths, capacity=2048, seed=5)
    assert [b.tolist() for b in rebuilt] == [b.tolist() for b in bins]


# ------------------------------------------------------------------ dataset + collator


def _rows(n, seed=0, lo=3, hi=12):
    rng = np.random.default_rng(seed)
    return [rng.integers(3, V, size=int(rng.integers(lo, hi))).tolist() for _ in range(n)]


def test_packed_dataset_items_and_collator_contract():
    rows = _rows(12)
    ds = Dataset.from_dict({"input_ids": rows})
    lengths = np.asarray([len(r) for r in rows])
    cap = 24
    bins = pack_rows(lengths, cap, seed=0)
    packed = PackedDataset(ds, bins, capacity=cap)
    assert len(packed) == len(bins)
    item = packed[0]
    expect_ids = [t for r in bins[0] for t in rows[r]]
    assert item["input_ids"] == expect_ids
    assert item["doc_ids"] == [k for k, r in enumerate(bins[0]) for _ in rows[r]]

    collator = DataCollatorForCausalLM(_Tok(), max_length=cap, model_vocab_size=V)
    batch = collator([packed[i] for i in range(len(packed))])
    ids, am, labels, doc = batch["input_ids"], batch["attention_mask"], batch["labels"], batch["doc_ids"]
    assert ids.shape == am.shape == labels.shape == doc.shape
    assert ids.shape[1] <= cap
    assert (doc[am == 0] == -1).all()
    assert (doc[am == 1] >= 0).all()
    assert (labels[am == 0] == -100).all()
    # doc starts (t > 0) carry -100; everything else real mirrors input_ids
    starts = torch.zeros_like(am, dtype=torch.bool)
    starts[:, 1:] = (doc[:, 1:] != doc[:, :-1]) & (doc[:, 1:] >= 0)
    assert (labels[starts] == -100).all()
    inner = (am == 1) & ~starts
    assert torch.equal(labels[inner], ids[inner])
    # number of masked starts == number of docs beyond the first, per row
    n_docs = torch.tensor([len(b) for b in bins])
    assert torch.equal(starts.sum(1), n_docs - 1)


def test_collator_without_doc_ids_is_unchanged():
    rows = _rows(4)
    collator = DataCollatorForCausalLM(_Tok(), max_length=16, model_vocab_size=V)
    batch = collator([{"input_ids": r} for r in rows])
    assert "doc_ids" not in batch
    with pytest.raises(ValueError):
        collator([{"input_ids": rows[0], "doc_ids": [0] * len(rows[0])}, {"input_ids": rows[1]}])


def test_packed_dataset_preserves_precomputed_labels():
    rows = _rows(6, seed=2)
    labels = [[-100] * (len(r) // 2) + r[len(r) // 2 :] for r in rows]
    ds = Dataset.from_dict({"input_ids": rows, "labels": labels})
    bins = pack_rows([len(r) for r in rows], 32, seed=0)
    packed = PackedDataset(ds, bins, capacity=32, preserve_labels=True)
    item = packed[0]
    assert item["labels"] == [t for r in bins[0] for t in labels[r]]
    collator = DataCollatorForCausalLM(_Tok(), max_length=32, model_vocab_size=V, preserve_precomputed_labels=True)
    batch = collator([item])
    doc = batch["doc_ids"]
    starts = torch.zeros_like(doc, dtype=torch.bool)
    starts[:, 1:] = (doc[:, 1:] != doc[:, :-1]) & (doc[:, 1:] >= 0)
    assert (batch["labels"][starts] == -100).all()


# ------------------------------------------------------------------ model equivalence


@torch.no_grad()
def test_packed_losses_equal_unpacked():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = PerceiverARLM(cfg).eval()
    # non-zero-init projections so attention actually matters
    for p in model.parameters():
        if p.ndim == 2:
            p.normal_(0, 0.2)
    rows = _rows(5, seed=7, lo=4, hi=10)
    ds = Dataset.from_dict({"input_ids": rows})
    cap = sum(len(r) for r in rows) + 3  # everything in one bin, plus right padding
    bins = pack_rows([len(r) for r in rows], cap, seed=0)
    assert len(bins) == 1
    packed = PackedDataset(ds, bins, capacity=cap)
    collator = DataCollatorForCausalLM(_Tok(), max_length=cap, model_vocab_size=V)
    # pad the packed row by adding a dummy short row in the batch so S > packed length
    batch = collator([packed[0], {"input_ids": rows[0][:2], "doc_ids": [0, 0]}])
    out, per, valid = model(
        batch["input_ids"], batch["attention_mask"], labels=batch["labels"],
        doc_ids=batch["doc_ids"], return_per_token_loss=True,
    )
    per_packed = per[0][valid[0]]

    ref = []
    for r in bins[0]:
        ids = torch.tensor([rows[r]])
        _, p, v = model(ids, labels=ids.clone(), return_per_token_loss=True)
        ref.append(p[0][v[0]])
    ref = torch.cat(ref)
    assert per_packed.shape == ref.shape
    torch.testing.assert_close(per_packed, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out.loss, (ref.sum() + per[1][valid[1]].sum()) / (ref.numel() + valid[1].sum()))


def test_block_mask_memo_is_shared_across_layers():
    """Flex memo: one create_block_mask per (pattern, window) per forward, output == sdpa."""
    pytest.importorskip("torch.nn.attention.flex_attention")
    torch.manual_seed(0)
    B, S, h, g, dh = 2, 16, 4, 2, 8
    q, k, v = (torch.randn(B, S, n, dh) for n in (h, g, g))
    doc_ids = torch.tensor([[0] * 6 + [1] * 7 + [-1] * 3, [0] * 16])
    memo: dict = {}
    try:
        out1 = attend(q, k, v, pattern="swa", window=5, key_valid=None, doc_ids=doc_ids,
                      backend="flex", block_masks=memo)
    except Exception as exc:  # flex on this CPU/torch build unsupported -> nothing to memoise
        pytest.skip(f"flex_attention unavailable on CPU here: {exc}")
    out2 = attend(q, k, v, pattern="swa", window=5, key_valid=None, doc_ids=doc_ids,
                  backend="flex", block_masks=memo)
    assert list(memo.keys()) == [("swa", 5, True)]
    ref = attend(q, k, v, pattern="swa", window=5, key_valid=None, doc_ids=doc_ids, backend="sdpa")
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out1, ref, rtol=1e-4, atol=1e-4)
