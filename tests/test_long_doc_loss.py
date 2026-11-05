"""E33n long-document eval: the per-row document key groups the spans cut from one document (bootstrap unit)."""
import numpy as np

from data.bin_token_dataset import BinRowDataset, write_shard
from evaluation.long_doc_loss import doc_key

EOT = 50256


def test_doc_key_groups_spans_of_one_document(tmp_path):
    rng = np.random.default_rng(0)
    lengths = [40, 9, 25]  # rows of 8: doc 0 gives 5 spans, doc 1 gives 1, doc 2 gives 3
    toks, starts = [], []
    for n in lengths:
        starts.append(len(toks))
        toks += [EOT] + rng.integers(0, 50000, n - 1).tolist()
    write_shard(tmp_path / "v.bin", toks)
    ds = BinRowDataset([{"name": "v", "files": [str(tmp_path / "v.bin")]}], seq_len=8, rule="doc", seed=7)
    assert len(ds) == 5 + 1 + 3
    keys = [doc_key(ds, i) for i in range(len(ds))]
    assert sorted(set(keys)) == sorted(f"0:{s}" for s in starts)
    assert sorted(keys.count(f"0:{s}") for s in starts) == [1, 3, 5]
    for i, k in enumerate(keys):  # each row's tokens lie inside its keyed document
        s = int(k.split(":")[1])
        e = next((t for t in starts if t > s), len(toks))
        row = ds[i]["input_ids"].tolist()
        assert any(toks[o:o + 8] == row for o in range(s, e - 7))
