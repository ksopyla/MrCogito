"""Length-view materialization via chapter subsetting (approx token budget)."""

from __future__ import annotations

from data.deductive_stories.schema import ChapterOutline, DeductiveExample


def _approx_tokens(text: str) -> int:
    # Rough English heuristic; good enough for bucketing without loading tokenizers.
    return max(1, int(len(text.split()) * 1.3))


def _chapter_blocks(story: str) -> list[str]:
    if "## " not in story:
        return [story]
    parts = story.split("\n\n## ")
    if len(parts) == 1:
        return [story]
    blocks = [parts[0]]
    for part in parts[1:]:
        blocks.append("## " + part if not part.startswith("## ") else part)
    # Fix first block if it started with ##
    if blocks and blocks[0].startswith("## "):
        pass
    elif story.startswith("## ") and not blocks[0].startswith("## "):
        blocks[0] = story.split("\n\n", 1)[0]
    return [b.strip() for b in blocks if b.strip()]


def _support_chapter_indices(
    outline: list[ChapterOutline],
    support_ids: set[str],
) -> list[int]:
    indices = []
    for i, chapter in enumerate(outline):
        if support_ids.intersection(chapter.cover_event_ids):
            indices.append(i)
        elif not chapter.is_distractor and chapter.cover_event_ids:
            # Keep some critical context chapters even if not direct support.
            indices.append(i)
    return sorted(set(indices))


def materialize_length_views(
    example: DeductiveExample,
    *,
    targets: dict[str, int] | None = None,
) -> DeductiveExample:
    """Fill story_2k/4k/8k/16k from chapter subsets of story_text.

    Condensation rule: keep chapters covering support_node_ids first, then add
    remaining chapters until the approximate token budget is hit.
    """
    targets = targets or {
        "story_2k": 2048,
        "story_4k": 4096,
        "story_8k": 8192,
        "story_16k": 16384,
    }
    full = example.story_text.strip()
    blocks = _chapter_blocks(full)
    support: set[str] = set()
    for ids in example.support_node_ids or []:
        support.update(ids)
    if not example.support_node_ids and example.graph.queries:
        for q in example.graph.queries:
            support.update(q.support_node_ids)

    preferred = _support_chapter_indices(example.outline, support) if example.outline else list(range(len(blocks)))
    order = list(preferred)
    for i in range(len(blocks)):
        if i not in order:
            order.append(i)

    views: dict[str, str] = {}
    for name, budget in targets.items():
        chosen: list[str] = []
        for idx in order:
            if idx >= len(blocks):
                continue
            candidate = "\n\n".join(chosen + [blocks[idx]])
            if chosen and _approx_tokens(candidate) > budget:
                break
            chosen.append(blocks[idx])
            if _approx_tokens("\n\n".join(chosen)) >= int(budget * 0.9):
                break
        views[name] = "\n\n".join(chosen) if chosen else full[: max(200, budget * 4)]

    example.story_2k = views["story_2k"]
    example.story_4k = views["story_4k"]
    example.story_8k = views["story_8k"] if _approx_tokens(full) >= 6000 else full
    # If full story is shorter than 8k target, keep full as canonical.
    if _approx_tokens(full) < targets["story_8k"]:
        example.story_8k = full
    example.story_16k = full if _approx_tokens(full) <= targets["story_16k"] else views["story_16k"]
    # Canonical long story for eval.
    example.story_text = example.story_8k or full
    return example
