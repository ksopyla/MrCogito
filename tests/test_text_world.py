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


# ---------------------------------------------------------------- v2.1: exams added on top of v2 (eval only)
@pytest.mark.parametrize("split", ["crowded", "paraphrase2", "steps"])
def test_v21_added_splits_need_v2_and_say_so(split):
    from data.text_world import ADDED_SPLITS

    task = ADDED_SPLITS[split][0]
    with pytest.raises(ValueError):
        make_eval_record(task, 0, split, 600, None, STORIES, ntok, version="v1")
    r = rec2(task, 0, split=split, depth="middle" if task in ("quote", "lookup", "keyed") else None)
    assert r["version"] == "text-world-v2.1" and rec2(task, 0)["version"] == "text-world-v2"


@pytest.mark.parametrize("split", ["crowded", "paraphrase2", "steps"])
def test_v21_rule_reader_recovers_every_answer(split):
    from data.text_world import ADDED_SPLITS

    for task in ADDED_SPLITS[split]:
        for s in range(8):
            depth = ("early", "middle", "late")[s % 3] if task in ("quote", "lookup", "keyed") else None
            r = rec2(task, s, length=3000, split=split, depth=depth)
            got, _ = _audit(r)
            assert got == r["answer"].strip(" ."), (split, task, s, got)
            assert r["answer"].strip(" .") not in r["prompt_removed"] or task in ("count", "deduce")


@pytest.mark.parametrize("task", ["quote", "lookup", "keyed"])
def test_v21_crowded_packs_every_fact_into_the_band(task):
    """Every fact sentence sits inside the 1,024-token band around the asked fact (word count stands in for tokens)."""
    from data.text_world import CROWD_TOKENS, ADDED_SPLITS

    assert task in ADDED_SPLITS["crowded"]
    for s, depth in enumerate(("early", "middle", "late")):
        r = rec2(task, s, length=6000, split="crowded", depth=depth)
        lo, hi = r["meta"]["fact_band"]
        assert abs((hi - lo) - CROWD_TOKENS / 6000) < 1e-3
        body = r["prompt"].rpartition("\n\nQuestion:")[0]
        item = build_item(task, random.Random(r["seed"]), "crowded", "v2")
        for f in item.facts:
            at = body.find(f.text)
            assert at >= 0
            frac = len(body[:at].split()) / max(1, len(body.split()))
            assert lo - 0.03 <= frac <= hi + 0.03, (task, depth, frac, lo, hi)
        spread = rec2(task, s, length=6000, depth=depth)  # the id split spreads them over the document
        assert "fact_band" not in spread["meta"]


def test_v21_paraphrase2_uses_only_unseen_sentence_shapes():
    from analysis.text_checks_audit import split_doc
    from data.text_world import HELDOUT_TEMPLATES_2, TEMPLATES
    from analysis.text_checks_audit import _compile

    seen = [_compile(rel, t) for rel, ts in TEMPLATES.items() for t in ts]
    new = [_compile(rel, t) for rel, ts in HELDOUT_TEMPLATES_2.items() for t in ts]
    for task in ("quote", "lookup", "keyed", "latest", "compose", "count"):
        r = rec2(task, 2, length=2000, split="paraphrase2", depth="middle" if task in ("quote", "lookup", "keyed") else None)
        sents, _ = split_doc(r["prompt"])
        facts = [x for x in sents if any(p.fullmatch(x) for p in new)]
        assert facts, task
        assert not any(p.fullmatch(x) for x in sents for p in seen), task


def test_v21_steps_are_one_step():
    r = rec2("compose", 4, split="steps")
    assert r["prompt"].rstrip().endswith("Where does the teacher of " + r["prompt"].split("teacher of ")[-1].split(" live?")[0] + " live?\nAnswer:")
    assert r["prompt"].count("the teacher of the teacher") == 0
    d = rec2("deduce", 4, split="steps")
    assert d["meta"]["depth"] == 1 and not re.search(r"Every \w+ is a \w+\.", d["prompt"])


def test_v2_published_data_is_byte_identical():
    """Pins text-world-v2 (published on the Hub): exams of the id / harder / paraphrase splits and training
    documents (1 and 8 questions) on the test fixtures. Hash from the v2 generator of 2026-10-10, checked
    unchanged by the v2.1 additions (2026-10-13). A change here is a new data version, never an edit."""
    import hashlib
    import json

    h = hashlib.sha256()
    for split in ("id", "harder", "paraphrase"):
        for task in TASKS:
            for i in range(3):
                depth = ("early", "middle", "late")[i] if task in ("quote", "lookup", "keyed") else None
                h.update(json.dumps(make_eval_record(task, 7 * i + 1, split, 1500, depth, STORIES, ntok, version="v2"),
                                    sort_keys=True).encode())
    for q in (1, 8):
        for i in range(10):
            h.update(make_train_document(i, STORIES, 400 + 50 * i, ntok, version="v2", questions=q).encode())
    assert h.hexdigest() == "56bb1906d48ffd41d74157222c236094dac9b09ce8501b0471dab0ea01997ffc"


# ---------------------------------------------------------------- bridge questions (training only, 2026-10-16)
def _seed_for(task, start=0):
    for s in range(start, start + 500):
        if random.Random(s).choice(TASKS) == task:
            return s
    raise AssertionError(task)


@pytest.mark.parametrize("task", TASKS)
def test_bridge_off_and_other_tasks_keep_their_bytes(task):
    s = _seed_for(task)
    base = make_train_document(s, STORIES, 600, ntok, version="v2", questions=8)
    assert make_train_document(s, STORIES, 600, ntok, version="v2", questions=8, bridge=0) == base
    with_b = make_train_document(s, STORIES, 600, ntok, version="v2", questions=8, bridge=4)
    if task in ("compose", "deduce"):
        assert with_b != base and with_b.count("Question:") == base.count("Question:") + 4
    else:
        assert with_b == base


@pytest.mark.parametrize("task", ["compose", "deduce"])
def test_bridge_answers_are_stated_and_follow_the_task_questions(task):
    for k in range(5):
        s = _seed_for(task, 100 * k)
        doc = make_train_document(s, STORIES, 900, ntok, version="v2", questions=8, bridge=4)
        body, _, tail = doc.partition("\n\nQuestion:")
        qs = re.findall(r"Question: (.*?)\nAnswer: (.*?)\.(?=\n|$)", "Question:" + tail)
        own = [q for q, _ in qs if q.startswith("Where does") or q.endswith("like?") and not q.startswith("What is a ")]
        brid = [(q, a) for q, a in qs if q.startswith(("Who is the teacher", "What is a ")) or
                (q.startswith("What is ") and not q.endswith("like?"))]
        assert len(brid) == 4
        assert [q for q, _ in qs].index(brid[0][0]) >= len(own)  # bridges come after the task's own questions
        for q, a in brid:
            if task == "compose":
                p = re.match(r"Who is the teacher of (.*)\?", q).group(1)
                assert any(f in body for f in (f"{a} was the teacher of {p}.", f"{p} was taught by {a}.",
                                               f"{p}'s teacher was {a}.", f"{p} learned to read from {a}",
                                               f"{p} went to lessons with a teacher named {a}."))
            elif q.endswith("like?"):
                cat = re.match(r"What is a (.*) like\?", q).group(1)
                assert f"Every {cat} is {a}." in body
            elif q.startswith("What is a "):
                cat = re.match(r"What is a (.*)\?", q).group(1)
                assert f"Every {cat} is {a}." in body
            else:
                p = re.match(r"What is (.*)\?", q).group(1)
                assert f"{p} is {a}." in body


def test_bridge_never_reaches_the_exams():
    for task in ("compose", "deduce"):
        r = make_eval_record(task, 3, "id", 1500, None, STORIES, ntok, version="v2")
        assert r["prompt"].count("Question:") == 1
        assert "Who is the teacher" not in r["prompt"] and "What is a " not in r["prompt"]

@pytest.mark.parametrize("task", TASKS)
def test_bridge_only_asks_just_the_bridges(task):
    s = _seed_for(task)
    base = make_train_document(s, STORIES, 600, ntok, version="v2", questions=8)
    assert make_train_document(s, STORIES, 600, ntok, version="v2", questions=8, bridge=0, bridge_only=False) == base
    doc = make_train_document(s, STORIES, 900, ntok, version="v2", questions=8, bridge=8, bridge_only=True)
    if task not in ("compose", "deduce"):
        assert doc == make_train_document(s, STORIES, 900, ntok, version="v2", questions=8, bridge=8)
        return
    qs = re.findall(r"Question: (.*?)\nAnswer: (.*?)\.(?=\n|$)", doc)
    assert 1 <= len(qs) <= 8
    if task == "compose":
        assert all(q.startswith("Who is the teacher of ") for q, _ in qs)
    else:
        assert all(q.startswith("What is ") for q, _ in qs)
    body = doc.partition("\n\nQuestion:")[0]
    for q, a in qs:  # every bridge answer is stated in the document
        assert a in body


# ---------------------------------------------------------------- read targets (training only, 2026-10-17)
def _word_offsets(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+|\s+", text)]


@pytest.mark.parametrize("task", ["compose", "deduce", "lookup", "quote", "latest"])
def test_read_targets_point_at_the_chain_facts(task):
    from scripts.build_text_checks_data import read_targets

    for k in range(4):
        s = _seed_for(task, 100 * k)
        chains = []
        doc = make_train_document(s, STORIES, 900, ntok, version="v2", questions=8, bridge=4, chains=chains)
        assert doc == make_train_document(s, STORIES, 900, ntok, version="v2", questions=8, bridge=4)
        offs = _word_offsets(doc)
        q = read_targets(doc, offs, chains)
        quads = [q[i:i + 4] for i in range(0, len(q), 4)]
        answers = [x for x in quads if x[1] == 0]
        assert len(answers) == doc.count("\nAnswer:")
        for pos, _, _, _ in answers:  # the answer position is the "Answer:" token (BOS shifts by one)
            a, b = offs[pos - 1]
            assert doc[a:b].endswith("Answer:")
        chain_of = dict(chains)
        if task == "latest":
            assert not chain_of and all(x[1] == 0 for x in quads)
            continue
        assert chain_of
        for pos, r, t0, t1 in quads:
            if r == 0:
                continue
            a, b = offs[pos - 1]
            question = re.findall(r"Question: (.*?)\nAnswer:", doc[: b])[-1]
            fact, word = chain_of[question][r - 1]
            assert doc[offs[t0 - 1][0]: offs[t1 - 1][1]] == fact
            assert offs[t1 - 1][1] < a  # the fact sits before its question
            assert word in fact
        if task == "compose":  # a one-step question: teacher fact, then the teacher's home
            for qq, facts in chains:
                if qq.startswith("Where does the teacher of ") and "the teacher of the teacher" not in qq:
                    p = qq[len("Where does the teacher of "):-len(" live?")]
                    assert p in facts[0][0] and len(facts) == 2
        # output spans: the hop's output word inside its fact
        qo = read_targets(doc, offs, chains, span="output")
        for pos, r, t0, t1 in [qo[i:i + 4] for i in range(0, len(qo), 4)]:
            if r == 0:
                continue
            a, b = offs[pos - 1]
            question = re.findall(r"Question: (.*?)\nAnswer:", doc[: b])[-1]
            fact, word = chain_of[question][r - 1]
            assert doc[offs[t0 - 1][0]: offs[t1 - 1][1]].strip(" .:,") == word.strip(" .")


def test_read_targets_tensor_pads_and_cuts():
    from data.data_collators import read_targets_tensor

    t = read_targets_tensor([{"read_targets": [9, 0, -1, -1, 9, 1, 2, 4]}, {"read_targets": []},
                             {"read_targets": [30, 0, -1, -1]}], max_len=20)
    assert t.shape == (3, 2, 4)
    assert t[0].tolist() == [[9, 0, -1, -1], [9, 1, 2, 4]]
    assert (t[1] == -1).all() and (t[2] == -1).all()  # row 3's answer sits past the cut
