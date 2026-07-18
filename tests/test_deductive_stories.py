"""Unit tests for deductive-stories graph solvers, splits, and scoring."""

from __future__ import annotations

import pytest

from data.deductive_stories.expand.outline import build_outline
from data.deductive_stories.expand.writer import expand_story
from data.deductive_stories.filters.dedup import dedup_examples, jaccard
from data.deductive_stories.filters.split import (
    assert_template_firewall,
    assign_splits_by_template,
)
from data.deductive_stories.graph.base import (
    get_adapter,
    make_example_from_graph,
    verify_graph_gold,
)
from data.deductive_stories.materialize.length_views import materialize_length_views
from data.deductive_stories.schema import normalize_answer


@pytest.mark.parametrize(
    "value,answer_type,expected",
    [
        ("Yes", "bool", "yes"),
        (True, "bool", "yes"),
        (False, "bool", "no"),
        (3.0, "number", "3"),
        (3.14, "number", "3.14"),
        ("  Avery Cole ", "string_norm", "avery cole"),
        ("E4", "entity_id", "e4"),
    ],
)
def test_normalize_answer(value, answer_type, expected):
    assert normalize_answer(value, answer_type=answer_type) == expected


@pytest.mark.parametrize("domain", ["detective", "se_debug"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_solver_round_trip(domain, split):
    adapter = get_adapter(domain)
    graph = adapter.generate(seed=123, split=split, distractor_ratio=0.4)
    assert verify_graph_gold(graph) == []
    # Re-solve after serialize round-trip.
    from data.deductive_stories.schema import EventGraph

    restored = EventGraph.from_dict(graph.to_dict())
    assert verify_graph_gold(restored) == []
    assert len(graph.queries) == 3
    assert len(graph.gold) == 3


def test_template_firewall_disjoint():
    det = get_adapter("detective")
    se = get_adapter("se_debug")
    examples = [
        make_example_from_graph(det.generate(seed=1, split="train"), split="train"),
        make_example_from_graph(det.generate(seed=2, split="test"), split="test"),
        make_example_from_graph(se.generate(seed=3, split="train"), split="train"),
        make_example_from_graph(se.generate(seed=4, split="test"), split="test"),
    ]
    assign_splits_by_template(examples)
    assert_template_firewall(examples)
    train_t = {e.template_id for e in examples if e.split == "train"}
    test_t = {e.template_id for e in examples if e.split == "test"}
    assert not (train_t & test_t)


def test_firewall_raises_on_overlap():
    det = get_adapter("detective")
    train_ex = make_example_from_graph(det.generate(seed=7, split="train"), split="train")
    # Force illegal: put a train example into test split with train template.
    train_ex.split = "test"
    with pytest.raises(ValueError):
        assign_splits_by_template([train_ex])


def test_mock_expand_and_length_views():
    adapter = get_adapter("detective")
    graph = adapter.generate(seed=99, split="train", distractor_ratio=0.5)
    ex = make_example_from_graph(graph, split="train")
    ex.outline = build_outline(graph, chapters_target=6)
    story, writer = expand_story(
        graph, ex.outline, mock_llm=True, words_per_chapter=100
    )
    assert writer == "mock_llm"
    assert "Chapter" in story
    ex.story_text = story
    materialize_length_views(ex)
    assert ex.story_2k
    assert ex.story_8k
    row = ex.to_public_row()
    assert "Question:" in row["text"]
    assert "Answer:" in row["text"]


def test_dedup_jaccard():
    assert jaccard("hello world", "hello world") == 1.0
    assert jaccard("aaaa", "bbbb") < 0.5
    adapter = get_adapter("se_debug")
    a = make_example_from_graph(adapter.generate(seed=1, split="train"), split="train")
    b = make_example_from_graph(adapter.generate(seed=1, split="train"), split="train")
    # Identical graphs (same seed) should dedup even with different story filler.
    a.story_text = "same story text for dedup " * 20
    b.story_text = "totally different surface form " * 20
    kept, dropped = dedup_examples([a, b], threshold=0.9)
    assert len(kept) == 1
    assert len(dropped) == 1
    # Distinct seeds survive shared mock filler.
    c = make_example_from_graph(adapter.generate(seed=2, split="train"), split="train")
    a.writer_model = b.writer_model = c.writer_model = "mock_llm"
    filler = ("Meanwhile clerks filed routine paperwork. " * 30)
    a.story_text = b.story_text = c.story_text = filler
    kept2, dropped2 = dedup_examples([a, c], threshold=0.9)
    assert len(kept2) == 2
    assert dropped2 == []
