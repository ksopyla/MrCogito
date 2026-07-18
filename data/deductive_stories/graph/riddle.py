"""Riddle domain adapter (v1 stub)."""

from __future__ import annotations

from dataclasses import dataclass

from data.deductive_stories.graph.base import register_adapter
from data.deductive_stories.schema import EventGraph, Query


@dataclass
class RiddleAdapter:
    domain: str = "riddle"

    def generate(self, *, seed: int, split: str, distractor_ratio: float = 0.35) -> EventGraph:
        raise NotImplementedError(
            "riddle adapter is v1 — implement after detective/se_debug MVP"
        )

    def solve(self, graph: EventGraph, query: Query) -> str:
        raise NotImplementedError("riddle adapter is v1")


register_adapter(RiddleAdapter())
