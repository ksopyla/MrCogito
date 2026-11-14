"""E33n data path: fixed rows from GPT-2 token shards, with doc ids that the causal-LM collator masks."""
import json

import numpy as np
import torch

from data.bin_token_dataset import BinRowDataset, load_bin_manifest, open_shard, write_shard
from data.data_collators import DataCollatorForCausalLM

EOT = 50256


def _docs(lengths, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in lengths:
        out += [EOT] + rng.integers(0, 50000, n - 1).tolist()
    return out


def test_shard_roundtrip(tmp_path):
    toks = _docs([5, 9, 3])
    write_shard(tmp_path / "a.bin", toks)
    assert open_shard(str(tmp_path / "a.bin")).tolist() == toks


def test_stream_rows_and_doc_ids(tmp_path):
    toks = _docs([10, 6, 20])  # 36 tokens
    write_shard(tmp_path / "a.bin", toks)
    ds = BinRowDataset([{"name": "a", "files": [str(tmp_path / "a.bin")]}], seq_len=12, rule="stream", seed=0)
    assert len(ds) == 3
    rows = sorted((ds[i]["input_ids"].tolist(), ds[i]["doc_ids"].tolist()) for i in range(len(ds)))
    full = {tuple(r) for r, _ in rows}
    assert tuple(toks[0:12]) in full and tuple(toks[24:36]) in full
    for ids, doc in rows:
        assert doc[0] == 0
        for t in range(1, len(ids)):  # a new id exactly at each <|endoftext|>
            assert doc[t] - doc[t - 1] == int(ids[t] == EOT)


def test_doc_rows_never_cross_documents(tmp_path):
    toks = _docs([30, 10, 25])
    write_shard(tmp_path / "a.bin", toks)
    ds = BinRowDataset([{"name": "a", "files": [str(tmp_path / "a.bin")]}], seq_len=12, rule="doc", seed=0)
    assert len(ds) == 2 + 2  # 30 → 2 rows, 10 → none, 25 → 2 rows
    for i in range(len(ds)):
        assert ds[i]["doc_ids"].max().item() == 0


def test_weights_and_manifest(tmp_path):
    write_shard(tmp_path / "a.bin", _docs([400], 1))
    write_shard(tmp_path / "b.bin", _docs([400], 2))
    man = {"seq_len": 16, "rule": "stream", "seed": 3,
           "train": [{"name": "a", "files": [str(tmp_path / "a.bin")], "weight": 3.0},
                     {"name": "b", "files": [str(tmp_path / "b.bin")], "weight": 1.0}],
           "eval": [{"name": "a", "files": [str(tmp_path / "a.bin")], "rows": 4}]}
    (tmp_path / "m.json").write_text(json.dumps(man))
    tr, ev, _ = load_bin_manifest(str(tmp_path / "m.json"))
    assert tr.rows_per_source["a"] == 3 * tr.rows_per_source["b"]
    assert len(ev) == 4


def test_collator_masks_doc_starts(tmp_path):
    toks = _docs([5, 7])
    write_shard(tmp_path / "a.bin", toks)
    ds = BinRowDataset([{"name": "a", "files": [str(tmp_path / "a.bin")]}], seq_len=12, rule="stream")
    tok = type("T", (), {"pad_token_id": EOT, "unk_token_id": None, "__len__": lambda self: 50257})()
    batch = DataCollatorForCausalLM(tok, max_length=12, model_vocab_size=50257)([ds[0]])
    assert batch["doc_ids"].tolist()[0] == [0] * 5 + [1] * 7
    assert batch["labels"][0, 5].item() == -100  # the next document's first token is never a target
    assert torch.equal(batch["input_ids"][0], torch.tensor(toks))
