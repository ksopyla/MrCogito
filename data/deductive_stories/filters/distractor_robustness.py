"""Distractor-robustness helper (lightweight v0)."""

from __future__ import annotations

from data.deductive_stories.noise.inject import inject_ood_haystack
from data.deductive_stories.schema import DeductiveExample


def story_with_extra_distractors(example: DeductiveExample) -> str:
    """Return a noisier copy of the story for optional teacher checks."""
    return inject_ood_haystack(
        example.story_text,
        seed=example.graph.seed + 99,
        n_paragraphs=2,
    )


def mark_distractor_view(example: DeductiveExample) -> DeductiveExample:
    """Tag metadata for eval subsets that already include dense distractors."""
    if example.split in {"test", "test_ood_noise"}:
        example.filter_notes.append("distractor_policy:eval_dense")
    else:
        example.filter_notes.append("distractor_policy:train_moderate")
    return example
