"""Tests for the Hugging Face datasets length sidecar cache."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from datasets import Dataset, load_from_disk

from data.length_cache import (
    _length_batch,
    cache_path_for_manifest,
    compute_or_load_interleaved_lengths,
    length_cache_paths,
    load_length_cache,
    save_length_cache,
)


def _write_manifest(tmp_path: Path, *, seed: int = 42, max_seq_length: int = 128) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "mix_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "mix_id": "test_mix",
                "seed": seed,
                "max_seq_length": max_seq_length,
                "sources": [{"name": "a", "weight": 1.0, "path": "a"}],
            }
        )
    )
    return manifest


def _make_dataset(lengths: list[int]) -> Dataset:
    return Dataset.from_dict({"input_ids": [[1] * n for n in lengths]})


def test_length_batch_helper_returns_int_lengths():
    out = _length_batch({"input_ids": [[1, 2, 3], [4], []]})
    assert out == {"length": [3, 1, 0]}


def test_save_and_load_roundtrip_hf_dataset(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    lengths = np.array([3, 1, 8, 2], dtype=np.int32)
    save_length_cache(manifest, lengths)

    cache_dir = cache_path_for_manifest(manifest)
    assert cache_dir.is_dir()
    assert (cache_dir / "dataset_info.json").is_file()

    loaded_ds = load_from_disk(str(cache_dir))
    assert loaded_ds.column_names == ["length"]
    assert loaded_ds.features["length"].dtype == "int32"
    assert list(loaded_ds["length"]) == [3, 1, 8, 2]

    loaded = load_length_cache(manifest)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, lengths)
    assert loaded.dtype == np.int32


def test_cache_hit_skips_scan(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    save_length_cache(manifest, np.array([4, 5, 6], dtype=np.int32))

    def _boom(*_args, **_kwargs):
        raise AssertionError("scan should not run on a valid cache hit")

    with patch("data.length_cache._compute_lengths", side_effect=_boom):
        out = compute_or_load_interleaved_lengths(
            manifest,
            train_ds=_make_dataset([4, 5, 6]),
        )
    np.testing.assert_array_equal(out, np.array([4, 5, 6], dtype=np.int32))


def test_invalid_seed_forces_rescan(tmp_path: Path):
    manifest = _write_manifest(tmp_path, seed=1)
    save_length_cache(manifest, np.array([1, 2], dtype=np.int32))
    _write_manifest(tmp_path, seed=2)

    out = compute_or_load_interleaved_lengths(
        manifest,
        train_ds=_make_dataset([9, 8]),
        num_proc=1,
    )
    np.testing.assert_array_equal(out, np.array([9, 8], dtype=np.int32))


def test_row_count_mismatch_forces_rescan(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    save_length_cache(manifest, np.array([1, 2, 3], dtype=np.int32))

    out = compute_or_load_interleaved_lengths(
        manifest,
        train_ds=_make_dataset([4, 5]),
        num_proc=1,
    )
    np.testing.assert_array_equal(out, np.array([4, 5], dtype=np.int32))


def test_compute_matches_python_len(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    lengths = [1, 7, 3, 12, 2]
    ds = _make_dataset(lengths)
    out = compute_or_load_interleaved_lengths(manifest, train_ds=ds, num_proc=1)
    np.testing.assert_array_equal(out, np.array(lengths, dtype=np.int32))


def test_num_proc_does_not_change_values(tmp_path: Path):
    lengths = [4, 1, 9, 2, 8, 3, 7, 5]
    ds = _make_dataset(lengths)
    out_serial = compute_or_load_interleaved_lengths(
        _write_manifest(tmp_path / "serial"),
        train_ds=ds,
        num_proc=1,
    )
    out_parallel = compute_or_load_interleaved_lengths(
        _write_manifest(tmp_path / "parallel"),
        train_ds=ds,
        num_proc=2,
    )
    np.testing.assert_array_equal(out_serial, out_parallel)
    np.testing.assert_array_equal(out_parallel, np.array(lengths, dtype=np.int32))


def test_force_rebuild_overwrites_cache(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    save_length_cache(manifest, np.array([1, 1, 1], dtype=np.int32))
    out = compute_or_load_interleaved_lengths(
        manifest,
        train_ds=_make_dataset([9, 8, 7]),
        force=True,
        num_proc=1,
    )
    np.testing.assert_array_equal(out, np.array([9, 8, 7], dtype=np.int32))


def test_save_cleans_tmp_and_old_dirs(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    save_length_cache(manifest, np.array([1, 2], dtype=np.int32))
    cache_dir = cache_path_for_manifest(manifest)
    assert not Path(str(cache_dir) + ".tmp").exists()
    assert not Path(str(cache_dir) + ".old").exists()


def test_legacy_npz_is_not_accepted(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    np.savez_compressed(
        tmp_path / "mix_manifest.json.lengths.npz",
        lengths=np.array([1, 2, 3], dtype=np.int32),
        seed=np.int64(42),
        max_seq_length=np.int64(128),
        n_rows=np.int64(3),
        sha256=np.array("deadbeef", dtype=object),
    )
    assert load_length_cache(manifest) is None
    out = compute_or_load_interleaved_lengths(
        manifest,
        train_ds=_make_dataset([4, 5, 6]),
        num_proc=1,
    )
    np.testing.assert_array_equal(out, np.array([4, 5, 6], dtype=np.int32))
    assert cache_path_for_manifest(manifest).is_dir()
    assert not (tmp_path / "mix_manifest.json.lengths.npz").exists()


def test_length_cache_paths_are_arrow_dir(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    dataset_dir, meta_path = length_cache_paths(manifest)
    assert dataset_dir == Path(f"{manifest}.lengths")
    assert not str(dataset_dir).endswith(".npz")
    compute_or_load_interleaved_lengths(
        manifest,
        train_ds=_make_dataset([2, 3]),
        num_proc=1,
    )
    assert dataset_dir.is_dir()
    assert meta_path.is_file()
    assert json.loads(meta_path.read_text())["format"] == "hf_datasets_arrow"


def test_valid_cache_does_not_read_dataset_rows(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    dataset = _make_dataset([1, 3, 2, 8])
    first = compute_or_load_interleaved_lengths(manifest, train_ds=dataset, num_proc=1)

    class _NoReadDataset:
        def __len__(self):
            return 4

        def select_columns(self, _columns):
            raise AssertionError("valid cache should avoid a dataset scan")

    cached = compute_or_load_interleaved_lengths(manifest, train_ds=_NoReadDataset())
    np.testing.assert_array_equal(cached, first)
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.old"))
