import json

import pytest

from data.dataset_preprocess import (
    DATASET_MIXES,
    apply_mix_weight_override,
    load_mix_recipe,
    resolve_mix_sources,
)


def test_load_mix_recipe_from_packaged_id():
    recipe = load_mix_recipe("smollm3_inspired_2k")
    assert recipe["mix_id"] == "smollm3_inspired_2k"
    assert len(recipe["sources"]) >= 3
    for src in recipe["sources"]:
        assert src.get("hf_id") or src.get("data_files")
        assert isinstance(src["text_columns"], list)
        assert isinstance(src["split"], str) and src["split"]
        assert src["weight"] > 0


def test_e16b_4k_mix_recipe_weights_and_policy():
    recipe = load_mix_recipe("smollm3_inspired_4k_e16b")
    assert recipe["mix_id"] == "smollm3_inspired_4k_e16b"
    assert recipe["seq_len_target"] == 4096
    total = sum(src["weight"] for src in recipe["sources"])
    assert abs(total - 1.0) < 1e-9
    names = {src["name"] for src in recipe["sources"]}
    assert {"finepdfs_100BT", "dclm_baseline", "fineweb_edu"}.issubset(names)
    assert recipe["long_context_policy"]["target_min_pct_docs_over_4k"] >= 18.0


def test_resolve_mix_sources_supports_registry_mixes():
    sources = resolve_mix_sources("long_2k_base_v1")
    assert len(sources) == len(DATASET_MIXES["long_2k_base_v1"])
    total_weight = sum(s["weight"] for s in sources)
    assert abs(total_weight - 1.0) < 1e-6


def test_apply_mix_weight_override_by_name():
    sources = resolve_mix_sources("smollm3_inspired_2k")
    updated = apply_mix_weight_override(
        sources,
        {"fineweb_edu": 0.5, "dclm_baseline": 0.3, "stack_edu_python": 0.2},
    )
    idx = {s["name"]: s for s in updated}
    assert idx["fineweb_edu"]["weight"] == pytest.approx(0.5)
    assert idx["dclm_baseline"]["weight"] == pytest.approx(0.3)
    assert idx["stack_edu_python"]["weight"] == pytest.approx(0.2)


def test_apply_mix_weight_override_unknown_key_raises():
    sources = resolve_mix_sources("smollm3_inspired_2k")
    with pytest.raises(ValueError, match="unknown source key"):
        apply_mix_weight_override(sources, {"not_a_source": 0.9})


def test_load_mix_recipe_list_payload_compat(tmp_path):
    recipe_path = tmp_path / "tiny_mix.json"
    recipe_path.write_text(
        json.dumps(
            [
                {
                    "name": "tiny_source",
                    "dataset": "my-org/my-dataset",
                    "text_column": "content",
                    "split": "train",
                    "weight": 1.0,
                    "max_samples": 1000,
                }
            ]
        ),
        encoding="utf-8",
    )

    recipe = load_mix_recipe(str(recipe_path))
    assert recipe["mix_id"] == "tiny_mix"
    assert len(recipe["sources"]) == 1
    src = recipe["sources"][0]
    assert src["hf_id"] == "my-org/my-dataset"
    assert src["text_columns"] == ["content"]
    assert src["max_samples"] == 1000
