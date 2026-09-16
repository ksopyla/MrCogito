"""CogitoProbe Hub filter + id resolution (no Hub download in CI)."""
from __future__ import annotations

import pytest
from datasets import Dataset

from data.dataset_preprocess import (
    COGITO_PROBE_HUB,
    filter_cogito_probe,
    resolve_cogito_probe_id,
)


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
