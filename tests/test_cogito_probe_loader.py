"""CogitoProbe Hub filter + id resolution (no Hub download in CI)."""
from __future__ import annotations

import random
from pathlib import Path

import pytest
from datasets import Dataset

from data.dataset_preprocess import (
    COGITO_PROBE_HUB,
    filter_cogito_probe,
    resolve_cogito_probe_id,
)
from evaluation.evaluate_cogito_probe import shuffle_filler_row

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_family_and_hub_id():
    assert resolve_cogito_probe_id("bits") == "ksopyla/cogito-probe-bits"
    assert resolve_cogito_probe_id("ksopyla/cogito-probe-arith") == "ksopyla/cogito-probe-arith"
    assert set(COGITO_PROBE_HUB) == {"bits", "bind", "arith", "props"}
    with pytest.raises(ValueError):
        resolve_cogito_probe_id("fineweb")


def test_filter_seq_and_variant_drops_32k():
    rows = []
    for seq, variant, n in ((1024, "fixed", 3), (1024, "scaled", 2), (32768, "fixed", 4)):
        for i in range(n):
            rows.append(
                {
                    "seq_len": seq,
                    "variant": variant,
                    "task": "recall_packed",
                    "input_ids": [1] * seq,
                    "labels": [-100] * seq,
                }
            )
    ds = Dataset.from_list(rows)
    keep = filter_cogito_probe(ds, seq_len=1024, variant="fixed")
    assert len(keep) == 3
    assert set(keep["seq_len"]) == {1024}
    assert set(keep["variant"]) == {"fixed"}
    long = filter_cogito_probe(ds, seq_len=32768, variant="fixed")
    assert len(long) == 4
    with pytest.raises(ValueError, match="empty"):
        filter_cogito_probe(ds, seq_len=4096, variant="fixed")


def test_filter_bind_hops_vs_gist():
    rows = []
    for task, n in (("attr_color", 3), ("who_place", 2), ("hop_friend_place", 4)):
        for _ in range(n):
            rows.append(
                {
                    "seq_len": 1024,
                    "variant": "fixed",
                    "task": task,
                    "input_ids": [1] * 8,
                    "labels": [-100] * 8,
                }
            )
    ds = Dataset.from_list(rows)
    hops = filter_cogito_probe(ds, seq_len=1024, variant="fixed", task="hop_friend_place")
    gist = filter_cogito_probe(ds, seq_len=1024, variant="fixed", task="attr_color")
    assert len(hops) == 4
    assert set(hops["task"]) == {"hop_friend_place"}
    assert len(gist) == 3
    assert set(gist["task"]) == {"attr_color"}


def test_shuffle_filler_keeps_evidence_query_and_labels():
    row = {
        "input_ids": [1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 99, 7, 8, 9],
        "labels": [-100] * 11 + [7, 8, 9],
        "evidence_end": 4,
        "answer_start": 11,
        "task": "prop_color",
    }
    out = shuffle_filler_row(row, random.Random(0), query_token_id=99)
    assert out["input_ids"][:4] == [1, 2, 3, 4]
    assert out["input_ids"][10] == 99
    assert out["input_ids"][11:] == [7, 8, 9]
    assert out["labels"] == row["labels"]
    assert sorted(out["input_ids"][4:10]) == [10, 11, 12, 13, 14, 15]
    assert out["input_ids"][4:10] != [10, 11, 12, 13, 14, 15]


def test_launch_e29_wires_hops_gist_and_props_shuffle():
    text = (REPO_ROOT / "scripts" / "launch_e29.sh").read_text()
    assert "hop_friend_place" in text
    assert "attr_color" in text
    assert "who_place" in text
    assert "ksopyla/cogito-probe-props" in text
    assert "--shuffle_filler" in text
    assert "PAR_MODE" in text and "perceiver" in text
    assert "PAR_MESSAGE_IDENTITY_SLOTS" in text


def test_odra_queue_skips_bind_after_bits():
    e29 = (REPO_ROOT / "scripts" / "launch_e29.sh").read_text()
    e28 = (REPO_ROOT / "scripts" / "launch_e28.sh").read_text()
    cont = (REPO_ROOT / "scripts" / "e21_queue_continue.sh").read_text()
    skip_at = e29.find("e29_park_if_bind")
    train_at = e29.find('bash "$SCRIPT_DIR/launch_e28.sh"')
    assert 0 <= skip_at < train_at
    assert "e29_park_if_bind" in e28
    assert 'SKIP_E29="${SKIP_E29:-1}"' in cont
    assert "Do not start Hub bind" in cont
    assert (REPO_ROOT / "scripts" / "skip_e29_guard.sh").is_file()
