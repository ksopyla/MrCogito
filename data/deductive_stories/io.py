"""Serialize / deserialize pipeline intermediates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.deductive_stories.schema import (
    ChapterOutline,
    DeductiveExample,
    EventGraph,
)


def example_to_dict(example: DeductiveExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "domain": example.domain,
        "template_id": example.template_id,
        "split": example.split,
        "graph": example.graph.to_dict(),
        "outline": [
            {
                "chapter_id": c.chapter_id,
                "title": c.title,
                "cover_event_ids": c.cover_event_ids,
                "cover_entity_ids": c.cover_entity_ids,
                "bullet_facts": c.bullet_facts,
                "is_distractor": c.is_distractor,
            }
            for c in example.outline
        ],
        "story_text": example.story_text,
        "story_2k": example.story_2k,
        "story_4k": example.story_4k,
        "story_8k": example.story_8k,
        "story_16k": example.story_16k,
        "questions": example.questions,
        "answers": example.answers,
        "answers_raw": example.answers_raw,
        "support_node_ids": example.support_node_ids,
        "hop_depths": example.hop_depths,
        "distractor_ratio": example.distractor_ratio,
        "noise_kind": example.noise_kind,
        "writer_model": example.writer_model,
        "judge_model": example.judge_model,
        "solver_id": example.solver_id,
        "generation_version": example.generation_version,
        "filter_notes": example.filter_notes,
        "accepted": example.accepted,
    }


def example_from_dict(data: dict[str, Any]) -> DeductiveExample:
    outline = [ChapterOutline(**c) for c in data.get("outline") or []]
    ex = DeductiveExample(
        example_id=data["example_id"],
        domain=data["domain"],
        template_id=data["template_id"],
        split=data["split"],
        graph=EventGraph.from_dict(data["graph"]),
        outline=outline,
        story_text=data.get("story_text", ""),
        story_2k=data.get("story_2k", ""),
        story_4k=data.get("story_4k", ""),
        story_8k=data.get("story_8k", ""),
        story_16k=data.get("story_16k", ""),
        writer_model=data.get("writer_model", ""),
        judge_model=data.get("judge_model", ""),
        solver_id=data.get("solver_id", ""),
        generation_version=data.get("generation_version", ""),
        filter_notes=list(data.get("filter_notes") or []),
        accepted=bool(data.get("accepted", True)),
    )
    ex.sync_qa_from_graph()
    if data.get("questions"):
        ex.questions = list(data["questions"])
    if data.get("answers"):
        ex.answers = list(data["answers"])
    if data.get("answers_raw"):
        ex.answers_raw = list(data["answers_raw"])
    if data.get("noise_kind"):
        ex.noise_kind = data["noise_kind"]
    return ex


def save_examples_jsonl(examples: list[DeductiveExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(example_to_dict(ex), ensure_ascii=True) + "\n")


def load_examples_jsonl(path: Path) -> list[DeductiveExample]:
    out: list[DeductiveExample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(example_from_dict(json.loads(line)))
    return out
