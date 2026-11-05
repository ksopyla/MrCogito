"""E33n long-document eval: the per-row document key groups the spans cut from one document (bootstrap unit); scoring
a source leaves the model's own notebook setting alone (the window control stays notes-off)."""
import numpy as np

from data.bin_token_dataset import BinRowDataset, write_shard
from evaluation.long_doc_loss import doc_key, score_source

EOT = 50256


def test_scoring_keeps_the_window_controls_notes_off(tmp_path):
    from tests.test_note_topk import make
    rng = np.random.default_rng(1)
    eot = 96  # the tiny model's vocabulary is 97 tokens
    toks = []
    for _ in range(3):
        toks += [eot] + rng.integers(0, 96, 199).tolist()
    write_shard(tmp_path / "v.bin", toks)
    ds = BinRowDataset([{"name": "v", "files": [str(tmp_path / "v.bin")]}], seq_len=96, rule="doc", eot=eot, seed=7)
    window = make(message_override="none")  # writer built, notes masked: the window control
    reading = make()
    reading.load_state_dict(window.state_dict())
    for notes_off in (True, False):
        first = score_source(window, ds, 96, 2, "cpu", notes_off)
        second = score_source(window, ds, 96, 2, "cpu", notes_off)
        assert window._message_override == "none"
        assert first["notes_on"]["loss"] == second["notes_on"]["loss"]
    # the check has teeth: the same weights reading their notes score differently
    assert score_source(reading, ds, 96, 2, "cpu", False)["notes_on"]["loss"] != first["notes_on"]["loss"]


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
