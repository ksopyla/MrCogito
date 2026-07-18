"""Near-duplicate detection via graph fingerprint + optional story Jaccard."""

from __future__ import annotations

import json

from data.deductive_stories.schema import DeductiveExample


def _ngrams(text: str, n: int = 3) -> set[str]:
    norm = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    norm = " ".join(norm.split())
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard(a: str, b: str, *, n: int = 3) -> float:
    sa, sb = _ngrams(a, n=n), _ngrams(b, n=n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def graph_fingerprint(example: DeductiveExample) -> str:
    """Exact identity key for solver-verified content (not narrative surface)."""
    payload = {
        "domain": example.graph.domain,
        "template_id": example.graph.template_id,
        "seed": example.graph.seed,
        "entities": [
            {"id": e.id, "name": e.name, "type": e.type, "attrs": e.attrs}
            for e in example.graph.entities
        ],
        "gold": [{"qid": g.qid, "normalized": g.normalized} for g in example.graph.gold],
        "events": [
            {"id": e.id, "type": e.type, "text_seed": e.text_seed, "attrs": e.attrs}
            for e in example.graph.events
        ],
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def dedup_examples(
    examples: list[DeductiveExample],
    *,
    threshold: float = 0.85,
) -> tuple[list[DeductiveExample], list[str]]:
    """Greedy keep-first dedup on exact graph fingerprint.

    Narrative boilerplate must not collapse distinct graphs. Optional story
    Jaccard applies only for non-mock writers at a very high threshold.
    """
    del threshold  # reserved for future soft graph matching
    kept: list[DeductiveExample] = []
    kept_fps: set[str] = set()
    dropped_ids: list[str] = []
    for ex in examples:
        fp = graph_fingerprint(ex)
        duplicate = fp in kept_fps
        if not duplicate and ex.writer_model and ex.writer_model != "mock_llm" and ex.story_text:
            for prev in kept:
                if (
                    prev.writer_model
                    and prev.writer_model != "mock_llm"
                    and prev.story_text
                    and jaccard(ex.story_text, prev.story_text) >= 0.97
                ):
                    duplicate = True
                    break
        if duplicate:
            dropped_ids.append(ex.example_id)
            ex.accepted = False
            ex.filter_notes.append("dedup_drop")
        else:
            kept.append(ex)
            kept_fps.add(fp)
    return kept, dropped_ids
