import json

import numpy as np
from datasets import Dataset

from data.length_cache import (
    compute_or_load_interleaved_lengths,
    length_cache_paths,
)


def _manifest_and_dataset(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"mix_id": "tiny", "seed": 42, "max_seq_length": 4096, "sources": []})
    )
    dataset = Dataset.from_dict(
        {"input_ids": [[1], [2, 3, 4], [5, 6], list(range(4096))]}
    )
    return manifest, dataset


def test_length_cache_roundtrip_and_manifest_invalidation(tmp_path):
    manifest, dataset = _manifest_and_dataset(tmp_path)

    first = compute_or_load_interleaved_lengths(manifest, train_ds=dataset)
    np.testing.assert_array_equal(first, np.array([1, 3, 2, 4096], dtype=np.int32))

    npz_path, meta_path = length_cache_paths(manifest)
    assert npz_path.exists()
    assert meta_path.exists()
    metadata = json.loads(meta_path.read_text())
    assert metadata["n_rows"] == 4
    assert metadata["seed"] == 42
    assert metadata["max_seq_length"] == 4096

    # A valid cache is reused without reading dataset rows.
    class _NoReadDataset:
        def __len__(self):
            return 4

        def select_columns(self, _columns):
            raise AssertionError("valid cache should avoid a dataset scan")

    cached = compute_or_load_interleaved_lengths(
        manifest,
        train_ds=_NoReadDataset(),
    )
    np.testing.assert_array_equal(cached, first)

    # The manifest hash is part of the index-space identity.
    payload = json.loads(manifest.read_text())
    payload["seed"] = 43
    manifest.write_text(json.dumps(payload))
    recomputed = compute_or_load_interleaved_lengths(manifest, train_ds=dataset)
    np.testing.assert_array_equal(recomputed, first)
    assert json.loads(meta_path.read_text())["seed"] == 43
    assert not list(tmp_path.glob("*.tmp"))


def test_length_cache_recomputes_for_row_count_mismatch(tmp_path):
    manifest, dataset = _manifest_and_dataset(tmp_path)
    compute_or_load_interleaved_lengths(manifest, train_ds=dataset)

    shorter = dataset.select(range(2))
    lengths = compute_or_load_interleaved_lengths(manifest, train_ds=shorter)

    np.testing.assert_array_equal(lengths, np.array([1, 3], dtype=np.int32))
