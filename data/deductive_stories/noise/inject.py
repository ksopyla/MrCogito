"""Noise injection for deductive stories."""

from __future__ import annotations

import random

from data.deductive_stories.schema import DeductiveExample

# Short OOD haystack paragraphs (public-domain style filler; not from PG19 dump).
_OOD_PARAS = (
    "The old mill by the river had stood empty since the flood of '09, its wheel "
    "locked with rust and its ledger books warped beyond reading.",
    "Farmers debated grain prices at the market square while gulls argued over "
    "scraps near the fish stalls, indifferent to any private investigation.",
    "A librarian reshelved atlases of distant coastlines, pausing only to note "
    "that volume C-14 was missing its endpaper map of tidal flats.",
    "In a quiet laboratory, a botanist measured leaf angles under glass and "
    "wrote numbers that would never appear in any crime or outage report.",
)


def inject_ood_haystack(
    story: str,
    *,
    seed: int,
    n_paragraphs: int = 3,
) -> str:
    rng = random.Random(seed)
    inserts = [rng.choice(_OOD_PARAS) for _ in range(max(0, n_paragraphs))]
    if not inserts:
        return story
    parts = story.split("\n\n")
    for para in inserts:
        idx = rng.randrange(0, max(1, len(parts)))
        parts.insert(idx, para)
    return "\n\n".join(parts)


def apply_split_noise_policy(example: DeductiveExample) -> DeductiveExample:
    """Raise distractor tagging / optional OOD inserts by split policy."""
    if example.split == "train":
        example.noise_kind = "in_domain_red_herring"
        example.graph.noise_kind = example.noise_kind
        return example
    if example.split == "validation":
        example.noise_kind = "in_domain_red_herring_dense"
        example.graph.noise_kind = example.noise_kind
        return example
    if example.split in {"test", "test_ood_noise"}:
        if example.split == "test_ood_noise" or example.noise_kind == "ood_haystack":
            example.story_text = inject_ood_haystack(
                example.story_text,
                seed=example.graph.seed + 17,
                n_paragraphs=4,
            )
            example.noise_kind = "ood_haystack"
            example.graph.noise_kind = "ood_haystack"
        else:
            example.noise_kind = "in_domain_red_herring_dense"
            example.graph.noise_kind = example.noise_kind
    return example
