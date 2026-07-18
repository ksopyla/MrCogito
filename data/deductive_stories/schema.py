"""Shared schema for deductive-stories graphs and published rows."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GENERATION_VERSION = "deductive_stories_v0.1"
SCHEMA_VERSION = "deductive_graph_v0"

DomainName = Literal["detective", "se_debug", "logistics", "riddle"]
SplitName = Literal["train", "validation", "test", "test_ood_noise"]


def normalize_answer(value: Any, *, answer_type: str = "string_norm") -> str:
    """Canonical exact-match form for scoring."""
    if value is None:
        return ""
    if answer_type == "bool":
        if isinstance(value, bool):
            return "yes" if value else "no"
        text = str(value).strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return "yes"
        if text in {"false", "no", "n", "0"}:
            return "no"
        return text
    if answer_type == "number":
        num = float(value)
        if num.is_integer():
            return str(int(num))
        return f"{num:.6g}"
    if answer_type == "entity_id":
        return str(value).strip().lower()
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class Entity:
    id: str
    type: str
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    id: str
    type: str
    time: int
    actors: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    text_seed: str = ""


@dataclass
class Relation:
    src: str
    dst: str
    type: str


@dataclass
class Query:
    qid: str
    type: str
    prompt: str
    answer_type: str
    support_node_ids: list[str] = field(default_factory=list)
    hop_depth: int = 1


@dataclass
class GoldAnswer:
    qid: str
    value: str
    normalized: str
    solver: str


@dataclass
class EventGraph:
    domain: DomainName
    template_id: str
    seed: int
    entities: list[Entity]
    events: list[Event]
    relations: list[Relation]
    queries: list[Query]
    gold: list[GoldAnswer]
    hidden_state: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    distractor_ratio: float = 0.0
    noise_kind: str = "none"

    def entity_by_id(self) -> dict[str, Entity]:
        return {e.id: e for e in self.entities}

    def event_by_id(self) -> dict[str, Event]:
        return {e.id: e for e in self.events}

    def gold_by_qid(self) -> dict[str, GoldAnswer]:
        return {g.qid: g for g in self.gold}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventGraph:
        return cls(
            domain=data["domain"],
            template_id=data["template_id"],
            seed=int(data["seed"]),
            entities=[Entity(**e) for e in data["entities"]],
            events=[Event(**e) for e in data["events"]],
            relations=[Relation(**r) for r in data["relations"]],
            queries=[Query(**q) for q in data["queries"]],
            gold=[GoldAnswer(**g) for g in data["gold"]],
            hidden_state=dict(data.get("hidden_state") or {}),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            distractor_ratio=float(data.get("distractor_ratio") or 0.0),
            noise_kind=str(data.get("noise_kind") or "none"),
        )


@dataclass
class ChapterOutline:
    chapter_id: str
    title: str
    cover_event_ids: list[str]
    cover_entity_ids: list[str]
    bullet_facts: list[str]
    is_distractor: bool = False


@dataclass
class DeductiveExample:
    example_id: str
    domain: DomainName
    template_id: str
    split: SplitName
    graph: EventGraph
    outline: list[ChapterOutline] = field(default_factory=list)
    story_text: str = ""
    story_2k: str = ""
    story_4k: str = ""
    story_8k: str = ""
    story_16k: str = ""
    questions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    answers_raw: list[str] = field(default_factory=list)
    support_node_ids: list[list[str]] = field(default_factory=list)
    hop_depths: list[int] = field(default_factory=list)
    distractor_ratio: float = 0.0
    noise_kind: str = "none"
    writer_model: str = ""
    judge_model: str = ""
    solver_id: str = ""
    generation_version: str = GENERATION_VERSION
    filter_notes: list[str] = field(default_factory=list)
    accepted: bool = True

    def sync_qa_from_graph(self) -> None:
        self.questions = [q.prompt for q in self.graph.queries]
        self.answers_raw = [g.value for g in self.graph.gold]
        self.answers = [g.normalized for g in self.graph.gold]
        self.support_node_ids = [list(q.support_node_ids) for q in self.graph.queries]
        self.hop_depths = [int(q.hop_depth) for q in self.graph.queries]
        self.distractor_ratio = float(self.graph.distractor_ratio)
        self.noise_kind = self.graph.noise_kind
        self.template_id = self.graph.template_id
        self.domain = self.graph.domain
        if self.graph.gold:
            self.solver_id = self.graph.gold[0].solver

    def to_public_row(self) -> dict[str, Any]:
        self.sync_qa_from_graph()
        return {
            "example_id": self.example_id,
            "domain": self.domain,
            "template_id": self.template_id,
            "split": self.split,
            "story_text": self.story_8k or self.story_text,
            "story_2k": self.story_2k,
            "story_4k": self.story_4k,
            "story_8k": self.story_8k or self.story_text,
            "story_16k": self.story_16k,
            "questions": list(self.questions),
            "answers": list(self.answers),
            "answers_raw": list(self.answers_raw),
            "support_node_ids": [list(x) for x in self.support_node_ids],
            "event_graph": self.graph.to_json(),
            "hop_depths": list(self.hop_depths),
            "distractor_ratio": self.distractor_ratio,
            "noise_kind": self.noise_kind,
            "writer_model": self.writer_model,
            "judge_model": self.judge_model,
            "solver_id": self.solver_id,
            "generation_version": self.generation_version,
            "text": pack_training_text(
                self.story_2k or self.story_text,
                self.questions,
                self.answers_raw,
            ),
        }


def pack_training_text(story: str, questions: list[str], answers: list[str]) -> str:
    blocks = [story.strip()]
    for question, answer in zip(questions, answers, strict=True):
        blocks.append(f"Question: {question}\nAnswer: {answer}")
    return "\n\n".join(blocks)
