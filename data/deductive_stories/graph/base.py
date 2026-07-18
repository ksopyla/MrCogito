"""Shared solver interface and registry for domain adapters."""

from __future__ import annotations

import uuid
from typing import Callable, Protocol

from data.deductive_stories.schema import (
    DeductiveExample,
    EventGraph,
    GoldAnswer,
    Query,
    normalize_answer,
)

SolverFn = Callable[[EventGraph, Query], str]


class DomainAdapter(Protocol):
    domain: str

    def generate(self, *, seed: int, split: str, distractor_ratio: float) -> EventGraph:
        ...


_ADAPTERS: dict[str, DomainAdapter] = {}


def register_adapter(adapter: DomainAdapter) -> DomainAdapter:
    _ADAPTERS[adapter.domain] = adapter
    return adapter


def get_adapter(domain: str) -> DomainAdapter:
    if domain not in _ADAPTERS:
        # Import adapters lazily so registration happens.
        from data.deductive_stories.graph import detective as _detective  # noqa: F401
        from data.deductive_stories.graph import se_debug as _se_debug  # noqa: F401
        from data.deductive_stories.graph import logistics as _logistics  # noqa: F401
        from data.deductive_stories.graph import riddle as _riddle  # noqa: F401
    if domain not in _ADAPTERS:
        raise KeyError(f"Unknown domain adapter: {domain}")
    return _ADAPTERS[domain]


def list_domains() -> list[str]:
    get_adapter("detective")  # force registration
    return sorted(_ADAPTERS)


def solve_query(graph: EventGraph, query: Query) -> str:
    """Recompute gold from the graph (no LLM)."""
    adapter = get_adapter(graph.domain)
    solver = getattr(adapter, "solve", None)
    if solver is None:
        raise RuntimeError(f"Adapter {graph.domain} has no solve()")
    raw = solver(graph, query)
    return normalize_answer(raw, answer_type=query.answer_type)


def verify_graph_gold(graph: EventGraph) -> list[str]:
    """Return mismatch descriptions; empty means consistent."""
    errors: list[str] = []
    gold_map = graph.gold_by_qid()
    for query in graph.queries:
        gold = gold_map.get(query.qid)
        if gold is None:
            errors.append(f"missing gold for {query.qid}")
            continue
        recomputed = solve_query(graph, query)
        if recomputed != gold.normalized:
            errors.append(
                f"{query.qid}: gold={gold.normalized!r} recomputed={recomputed!r}"
            )
    return errors


def make_example_from_graph(
    graph: EventGraph,
    *,
    split: str,
    example_id: str | None = None,
) -> DeductiveExample:
    errors = verify_graph_gold(graph)
    if errors:
        raise ValueError("Graph gold inconsistent: " + "; ".join(errors))
    example = DeductiveExample(
        example_id=example_id or str(uuid.uuid4()),
        domain=graph.domain,
        template_id=graph.template_id,
        split=split,  # type: ignore[arg-type]
        graph=graph,
    )
    example.sync_qa_from_graph()
    return example


def attach_gold(
    graph: EventGraph,
    *,
    solver_name: str,
) -> EventGraph:
    gold: list[GoldAnswer] = []
    for query in graph.queries:
        raw = get_adapter(graph.domain).solve(graph, query)  # type: ignore[attr-defined]
        gold.append(
            GoldAnswer(
                qid=query.qid,
                value=str(raw),
                normalized=normalize_answer(raw, answer_type=query.answer_type),
                solver=solver_name,
            )
        )
    graph.gold = gold
    return graph
