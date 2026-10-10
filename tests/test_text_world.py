"""Text capability checks — the world-document generator (data/text_world.py, text-world-v1; v0 kept)."""
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


def rec(task, seed=0, length=300, depth=None, split="id", version="v1"):
    return make_eval_record(task, seed, split, length, depth, STORIES, ntok, version=version)


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


def test_compose_follows_the_sister_chain_v0():
    rng = random.Random(7)
    item = build_item("compose", rng, "id", version="v0")
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


# ---------------------------------------------------------------- v1: fixes from the v0 audit
def _audit(r):
    from analysis.text_checks_audit import rule_reader, shortcuts, split_doc

    sents, q = split_doc(r["prompt"])
    return rule_reader(r["task"], sents, q), shortcuts(r["task"], sents, q)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("split", ["id", "harder", "paraphrase"])
def test_v1_rule_reader_recovers_every_answer(task, split):
    for s in range(15):
        r = rec(task, s, length=500, split=split, depth="early" if task in ("quote", "lookup", "keyed") else None)
        got, _ = _audit(r)
        assert got == r["answer"].strip(" ."), (task, split, s, got)


def test_v1_lookup_states_every_home():
    r = rec("lookup", 2)
    assert len(r["candidates"]) == 4 and r["floor"] == 0.25
    assert all(c.strip(" .") in r["prompt"] for c in r["candidates"])


def test_v1_latest_needs_the_name():
    hits = [(_audit(rec("latest", s, length=500))[1]["busiest mover's last place (ignores the name)"]
             == rec("latest", s, length=500)["answer"].strip(" .")) for s in range(40)]
    assert sum(hits) < 20  # v0: all 40


def test_v1_compose_chain_never_returns_to_the_asked_person():
    for s in range(30):
        item = build_item("compose", random.Random(s), "harder")
        assert "teacher" in item.question
        q = re.search(r"of ([A-Z]\w+) live", item.question).group(1)
        own = next(f.text for f in item.facts if q in f.text and any(c in f.text for c in item.candidates))
        assert item.answer not in own


def test_v1_count_decoys_visit_the_same_place():
    from analysis.text_checks_audit import shortcuts, split_doc

    right = 0
    for s in range(60):
        r = rec("count", s, length=500)
        sents, q = split_doc(r["prompt"])
        right += shortcuts("count", sents, q)["all visits to the place (ignores the name)"] == r["answer"].strip(" .")
    assert right < 20  # v0: about half


def test_v1_filler_never_names_a_property_of_the_cast():
    story = "Lily was very brave. Lily had a red ball. The sun was warm."
    sents = renamed_filler(story, ["Lumo"], random.Random(0), version="v1")
    assert not any("brave" in x for x in sents) and any("ball" in x for x in sents)
    assert any("brave" in x for x in renamed_filler(story, ["Lumo"], random.Random(0), version="v0"))


def test_v1_single_fact_is_not_the_first_or_last_of_its_kind():
    from analysis.text_checks_audit import shortcuts, split_doc

    for depth, key in (("early", "first place"), ("late", "last place")):
        for s in range(10):
            r = rec("keyed", s, length=600, depth=depth)
            sents, q = split_doc(r["prompt"])
            assert shortcuts("keyed", sents, q)[key] != r["answer"].strip(" ."), (depth, s)


def test_versions_are_recorded_and_checked():
    assert rec("quote", 1)["version"] == "text-world-v1"
    assert rec("quote", 1, version="v0")["version"] == "text-world-v0"
    with pytest.raises(ValueError):
        rec("quote", 1, version="v9")


def test_v1_training_documents_can_ask_several_cast_members():
    one = make_train_document(11, STORIES, 300, ntok)
    many = make_train_document(11, STORIES, 300, ntok, questions=8)
    assert one.count("Question:") == 1 and many.count("Question:") > 1
    qs = re.findall(r"Question: (.*?)\nAnswer: (.*?)\.", many)
    assert len({q for q, _ in qs}) == len(qs)  # different people, no repeats
    first_q = re.search(r"Question: (.*?)\n", one).group(1)
    assert qs[0][0] == first_q  # the asked person comes first
    assert make_train_document(11, STORIES, 300, ntok, questions=8, version="v0").count("Question:") == 1


# ---------------------------------------------------------------- v2: every guessing rate below 10 %
def rec2(task, seed=0, length=1500, depth=None, split="id"):
    return make_eval_record(task, seed, split, length, depth, STORIES, ntok, version="v2")


@pytest.mark.parametrize("split", ["id", "harder", "paraphrase"])
def test_v2_every_guessing_rate_is_below_10_percent(split):
    for task in TASKS:
        r = rec2(task, 1, split=split)
        assert r["floor"] < 0.10 and abs(r["floor"] - 1 / len(r["candidates"])) < 1e-9, (task, split, r["floor"])
        assert r["answer"] in r["candidates"] and len(set(r["candidates"])) == len(r["candidates"])


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("split", ["id", "harder", "paraphrase"])
def test_v2_rule_reader_recovers_every_answer(task, split):
    for s in range(10):
        r = rec2(task, s, split=split, depth="middle" if task in ("quote", "lookup", "keyed") else None)
        got, _ = _audit(r)
        assert got == r["answer"].strip(" ."), (task, split, s, got)


@pytest.mark.parametrize("task", ["lookup", "keyed", "latest", "compose"])
def test_v2_candidate_places_start_with_different_letters(task):
    """The tokenizer's first token of an invented name is its capital letter: different letters make
    the picked-candidate score decidable at the first token."""
    for s in range(20):
        firsts = [c.strip()[0] for c in rec2(task, s)["candidates"]]
        assert len(set(firsts)) == len(firsts), (task, s, firsts)


def test_v2_deduce_asks_for_the_property_of_one_chain_among_twelve():
    from data.text_world import PROPERTIES_V2

    r = rec2("deduce", 3)
    assert r["prompt"].rstrip().endswith("like?\nAnswer:")
    props = [c.strip(" .") for c in r["candidates"]]
    assert len(props) == 12 and set(props) <= set(PROPERTIES_V2)
    assert all(re.search(rf"Every \w+ is {p}\.", r["prompt"]) for p in props)  # every chain ends in its own property
    assert not re.search(rf"Every \w+ is {props[0]}\.", r["prompt_removed"])


@pytest.mark.parametrize("task", TASKS)
def test_v2_shortcuts_stay_near_the_guessing_rate(task):
    hits = {}
    for s in range(60):
        depth = ("early", "middle", "late")[s % 3] if task in ("quote", "lookup", "keyed") else None
        r = rec2(task, s, length=1200, depth=depth)
        for k, v in _audit(r)[1].items():
            if v is not None:
                hits.setdefault(k, []).append(v == r["answer"].strip(" ."))
    floor = rec2(task, 0)["floor"]
    for k, v in hits.items():
        # latest's known partial floor: tracking moves but ignoring whose they are gives 1 in 5 people
        allowed = 1 / 5 if (task, k) == ("latest", "busiest mover's last place (ignores the name)") else floor
        assert sum(v) / len(v) <= allowed + 0.15, (task, k, sum(v) / len(v))


def test_v2_count_answers_cover_zero_to_ten():
    assert len({rec2("count", s, length=900)["answer"] for s in range(220)}) == 11
