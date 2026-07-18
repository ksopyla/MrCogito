"""Chapter outlines from event graphs (no LLM)."""

from __future__ import annotations

from data.deductive_stories.schema import ChapterOutline, EventGraph


def build_outline(graph: EventGraph, *, chapters_target: int = 8) -> list[ChapterOutline]:
    """Group events into chapters with fact-coverage checklists.

    Critical (non-distractor) events are covered first; distractors fill remaining
    chapter budget so long narratives can pad without changing gold.
    """
    critical = [e for e in graph.events if not e.attrs.get("distractor")]
    distractors = [e for e in graph.events if e.attrs.get("distractor")]

    # Aim for ~1-2 critical events per early chapter.
    chapters: list[ChapterOutline] = []
    chunk = max(1, (len(critical) + chapters_target - 1) // max(1, chapters_target // 2))
    for i in range(0, len(critical), chunk):
        group = critical[i : i + chunk]
        chapters.append(
            ChapterOutline(
                chapter_id=f"C{len(chapters)+1}",
                title=f"Chapter {len(chapters)+1}: {group[0].type.replace('_', ' ')}",
                cover_event_ids=[e.id for e in group],
                cover_entity_ids=sorted({a for e in group for a in e.actors}),
                bullet_facts=[e.text_seed for e in group if e.text_seed],
                is_distractor=False,
            )
        )

    for event in distractors:
        chapters.append(
            ChapterOutline(
                chapter_id=f"C{len(chapters)+1}",
                title=f"Chapter {len(chapters)+1}: side lead",
                cover_event_ids=[event.id],
                cover_entity_ids=list(event.actors),
                bullet_facts=[event.text_seed] if event.text_seed else [],
                is_distractor=True,
            )
        )

    # Pad with empty atmospheric chapters if still short of target (writer fills).
    while len(chapters) < chapters_target:
        chapters.append(
            ChapterOutline(
                chapter_id=f"C{len(chapters)+1}",
                title=f"Chapter {len(chapters)+1}: atmosphere",
                cover_event_ids=[],
                cover_entity_ids=[],
                bullet_facts=[
                    "Add scene-setting detail that does not introduce new decisive facts."
                ],
                is_distractor=True,
            )
        )
    return chapters
