"""Text capability checks — the world-document generator (data/text_world.py, draft text-world-v0)."""
import random
import re

import pytest

from data.text_world import (
    TASKS,
    build_item,
    is_heldout_name,
    make_eval_record,
    make_train_document,
    renamed_filler,
    sample_names,
)

STORIES = [
    "Once upon a time, Lily had a red ball. Lily played in the park with Tom. Tom was happy.",
    "Max went to the market with Mom. Max lived in a big house. They bought bread and went home.",
    "Sue saw a bird. The bird sang a song. Sue smiled and waved at the bird.",
    "Ben had a friend called Sam. Ben and Sam built a fort. It was a great day.",
] * 5
ntok = lambda s: len(s.split())  # noqa: E731  (word count stands in for the tokenizer)


def rec(task, seed=0, length=300, depth=None, split="id"):
    return make_eval_record(task, seed, split, length, depth, STORIES, ntok)


@pytest.mark.parametrize("task", TASKS)
def test_records_are_deterministic(task):
    assert rec(task, 5) == rec(task, 5)
    assert rec(task, 5)["prompt"] != rec(task, 6)["prompt"]


@pytest.mark.parametrize("task", TASKS)
def test_answer_is_a_candidate_and_floor_is_set(task):
    r = rec(task, 1)
    assert r["answer"] in r["candidates"] and r["answer"].endswith(".")
    assert 0.0 <= r["floor"] <= 0.5
    assert r["prompt"].endswith("Answer:")


@pytest.mark.parametrize("task", ["quote", "lookup", "keyed", "latest", "compose"])
def test_evidence_removed_twin_drops_the_answer(task):
    """Answers are invented (or random word strings): once the evidence is gone, they are nowhere."""
    for s in range(5):
        r = rec(task, s)
        ans = r["answer"].strip(" .")
        assert ans in r["prompt"]
        assert ans not in r["prompt_removed"]


def test_twin_has_the_same_length_and_filler():
    r = rec("keyed", 3, length=400, depth="middle")
    a, b = ntok(r["prompt"]), ntok(r["prompt_removed"])
    assert abs(a - b) < 40
    assert r["prompt"].split()[:5] == r["prompt_removed"].split()[:5]


@pytest.mark.parametrize("length", [200, 600, 1500])
def test_documents_hit_their_length(length):
    for task in TASKS:
        n = ntok(rec(task, 2, length=length)["prompt"] + rec(task, 2, length=length)["answer"])
        assert length * 0.9 <= n <= length + 2, (task, n)


def test_depth_places_the_single_fact():
    early = rec("lookup", 4, length=600, depth="early")
    late = rec("lookup", 4, length=600, depth="late")
    ans = early["answer"].strip(" .")
    assert early["prompt"].index(ans) < len(early["prompt"]) * 0.25
    assert late["prompt"].index(ans) > len(late["prompt"]) * 0.75


def test_latest_answer_is_not_the_last_place_mentioned():
    for s in range(10):
        r = rec("latest", s, length=500)
        body = r["prompt"].split("\n\nQuestion:")[0]
        places = re.findall(r"moved (?:away )?to (\w+)|live in (\w+)", body)
        last = [a or b for a, b in places][-1]
        assert last != r["answer"].strip(" .")


def test_compose_follows_the_sister_chain():
    rng = random.Random(7)
    item = build_item("compose", rng, "id")
    text = " ".join(f.text for f in item.facts)
    q = re.search(r"of ([A-Z]\w+) live", item.question).group(1)
    hops = item.meta["hops"]

    def sister(p):
        for pat in (rf"(\w+) was the sister of {p}\b", rf"\b{p} had a sister called (\w+)",
                    rf"\b{p} and {p}'s sister (\w+)", rf"(\w+) was {p}'s sister", rf"\b{p} grew up with a sister named (\w+)"):
            m = re.search(pat, text)
            if m:
                return m.group(1)
        raise AssertionError(p)

    cur = q
    for _ in range(hops):
        cur = sister(cur)
    assert re.search(rf"\b{cur}\b.*?{item.answer}|{item.answer}.*?\b{cur}\b", text)


def test_count_answers_are_balanced():
    answers = [rec("count", s)["answer"] for s in range(140)]
    assert len(set(answers)) == 7


def test_eval_names_are_held_out_of_training():
    rng = random.Random(0)
    assert not any(is_heldout_name(n) for n in sample_names(rng, 50, "train"))
    assert all(is_heldout_name(n) for n in sample_names(rng, 50, "eval"))
    doc = make_train_document(3, STORIES, 400, ntok)
    assert "Question:" in doc and doc.rstrip().endswith(".")


def test_filler_never_states_a_relation_about_the_cast():
    sents = renamed_filler(STORIES[1], ["Lumo"], random.Random(0))
    assert all("Lumo" not in s or not re.search(r"lived|market|friend", s) for s in sents)
    assert any("Lumo" in s for s in renamed_filler(STORIES[2], ["Lumo"], random.Random(0)))
