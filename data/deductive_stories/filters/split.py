"""Train/validation/test split with template-id firewall."""

from __future__ import annotations

from collections import defaultdict

from data.deductive_stories.graph import detective, se_debug
from data.deductive_stories.schema import DeductiveExample

_TRAIN_FAMILIES = set(detective.TRAIN_TEMPLATES) | set(se_debug.TRAIN_TEMPLATES)
_TEST_FAMILIES = set(detective.TEST_TEMPLATES) | set(se_debug.TEST_TEMPLATES)


def assert_template_firewall(examples: list[DeductiveExample]) -> None:
    train_ids = {e.template_id for e in examples if e.split == "train"}
    val_ids = {e.template_id for e in examples if e.split == "validation"}
    test_ids = {
        e.template_id for e in examples if e.split in {"test", "test_ood_noise"}
    }
    overlap = (train_ids | val_ids) & test_ids
    if overlap:
        raise ValueError(f"Train/val templates overlap test holdout: {sorted(overlap)}")
    illegal_train = (train_ids | val_ids) & _TEST_FAMILIES
    illegal_test = test_ids & _TRAIN_FAMILIES
    if illegal_train:
        raise ValueError(f"Train/val used test-only templates: {sorted(illegal_train)}")
    if illegal_test:
        raise ValueError(f"Test used train-only templates: {sorted(illegal_test)}")


def assign_splits_by_template(
    examples: list[DeductiveExample],
) -> list[DeductiveExample]:
    """Ensure split field matches template family; validation shares train families."""
    for ex in examples:
        if ex.template_id in _TEST_FAMILIES:
            if ex.split not in {"test", "test_ood_noise"}:
                ex.split = "test"
        elif ex.template_id in _TRAIN_FAMILIES:
            if ex.split in {"test", "test_ood_noise"}:
                raise ValueError(
                    f"Example {ex.example_id} has train template {ex.template_id} "
                    f"but split={ex.split}"
                )
        else:
            raise ValueError(f"Unknown template_id: {ex.template_id}")
    assert_template_firewall(examples)
    return examples


def summarize_splits(examples: list[DeductiveExample]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.split] += 1
    return dict(counts)
