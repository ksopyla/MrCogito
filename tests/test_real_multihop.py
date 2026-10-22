"""E33s real multi-hop data: the chain rule, document assembly, target spans and scoring (offline, hand-built items)."""
import random

from data.real_multihop import (
    QAItem,
    assemble,
    chain_targets,
    f1_score,
    from_gsm8k,
    from_hotpot,
    from_musique,
    normalize_answer,
    parse_chain,
    written_answer,
)

S1 = "Green is the fourth studio album by British progressive rock musician Steve Hillage"
S2 = "Miquette Giraudy is a keyboard player and vocalist, best known for her work with her partner Steve Hillage"
Q = "Who is the spouse of the Green performer?"


def test_rule_finds_the_bridge_and_puts_the_answer_last():
    body = S1 + ". Some text. " + S2 + "."
    got = parse_chain([S1, S2], "Miquette Giraudy", Q, body)
    assert got == [(S1, "Steve Hillage"), (S2, "Miquette Giraudy")]
    # the answer's step listed first is moved to the end by the text rule
    assert parse_chain([S2, S1], "Miquette Giraudy", Q, body) == got


def test_rule_gives_up_on_a_comparison_without_a_shared_entity():
    s1 = "Lake Orvel was built in 1962"
    s2 = "The Quarry Line opened in 1891"
    assert parse_chain([s1, s2], "The Quarry Line", "Which is older, Lake Orvel or the Quarry Line?", s1 + " " + s2) == []


def test_rule_on_equations():
    got = parse_chain(["600*30/100=180", "600*10/100=60", "180+60=240", "600-240=360"], "360", "q", "", math=True)
    assert [o for _, o in got] == ["180", "60", "240", "360"]


def test_written_answer_modes():
    steps = [S1, S2]
    assert written_answer(steps, "Miquette Giraudy", "all") == f" {S1}. {S2}. So Miquette Giraudy."
    assert written_answer(steps, "Miquette Giraudy", "drop1") == f" {S2}. So Miquette Giraudy."
    assert written_answer(steps, "Miquette Giraudy", "none") == " Miquette Giraudy."


def _item():
    paras = [("Green (album)", S1 + ". It was recorded in 1977.", True), ("Miquette Giraudy", S2 + ".", True),
             ("Lake Orvel", "Lake Orvel is a reservoir.", False)]
    return QAItem("musique", "x", Q, "Miquette Giraudy", [], paras, [S1, S2])


def test_assemble_places_or_removes_evidence_and_ends_in_answer():
    ntok = lambda s: len(s.split())  # noqa: E731
    pool = [(f"T{i}", f"Distractor paragraph number {i} about nothing in particular.", False) for i in range(200)]
    it = _item()
    doc = assemble(it, 400, ntok, random.Random(0), pool)
    assert doc.endswith(f"\n\nQuestion: {Q}\nAnswer:") and S1 in doc and S2 in doc
    assert 250 <= ntok(doc) <= 400
    gone = assemble(it, 400, ntok, random.Random(0), pool, remove_evidence=True)
    assert S1 not in gone and S2 not in gone and "Lake Orvel" in gone
    early = assemble(it, 400, ntok, random.Random(1), pool, depth=0.0)
    assert early.index(S1) < early.index("Distractor")


def test_chain_targets_point_at_the_step_words():
    from transformers import AutoTokenizer  # noqa: F401  (only to mirror the real path; offsets come from a stub)

    it = _item()
    doc = "\n\n".join(f"{t}\n{x}" for t, x, _ in it.paragraphs) + f"\n\nQuestion: {Q}\nAnswer:"
    text = doc + written_answer(it.steps, it.answer, "none")
    derived = parse_chain(it.steps, it.answer, Q, doc)
    # whitespace "tokenizer" with character offsets
    offs, i = [], 0
    for tok in text.split(" "):
        offs.append((i, i + len(tok)))
        i += len(tok) + 1
    quads = chain_targets(text, offs, Q, derived)
    assert quads[:4][1] == 0 and len(quads) == 12
    words = text.split(" ")
    spans = [" ".join(words[a - 1:b]) for a, b in zip(quads[6::4], quads[7::4])]
    assert spans[0].startswith("Steve") and "Hillage" in spans[0]
    assert spans[1].startswith("Miquette") or "Miquette" in spans[1]


def test_math_targets_sit_in_the_written_answer():
    it = from_gsm8k({"question": "How many?", "steps": ["<<2*3=6>>", "<<6+1=7>>"], "answer": "7"}, 0)
    prompt = assemble(it, 0, len, random.Random(0), [])
    derived = parse_chain(it.steps, it.answer, it.question, prompt, math=True)
    text = "\n\n" + prompt + written_answer([s for s, _ in derived], it.answer, "all")
    offs, i = [], 0
    for tok in text.split(" "):
        offs.append((i, i + len(tok)))
        i += len(tok) + 1
    quads = chain_targets(text, offs, it.question, derived, steps_in_answer=True)
    assert len(quads) == 12  # answer position + two steps, both after it
    assert all(t > quads[0] for t in quads[6::4])


def test_loaders_build_items_from_hf_records():
    rec = {"id": "2hop__1", "question": Q, "answer": "Miquette Giraudy", "answer_aliases": [], "answerable": True,
           "paragraphs": [{"idx": 0, "title": "Green (album)", "paragraph_text": S1 + ". Recorded 1977.", "is_supporting": True},
                          {"idx": 1, "title": "MG", "paragraph_text": S2 + ". She lives in France.", "is_supporting": True}],
           "question_decomposition": [{"id": 1, "question": "Green >> performer", "answer": "Steve Hillage", "paragraph_support_idx": 0},
                                      {"id": 2, "question": "#1 >> spouse", "answer": "Miquette Giraudy", "paragraph_support_idx": 1}]}
    it = from_musique(rec)
    assert it.steps == [S1 + ".", S2 + "."] and it.qtype == "2hop"
    hp = {"id": "h1", "question": Q, "answer": "x", "type": "bridge",
          "context": {"title": ["A", "B"], "sentences": [["a0. ", "a1."], ["b0."]]},
          "supporting_facts": {"title": ["A", "B"], "sent_id": [1, 0]}}
    it2 = from_hotpot(hp)
    assert it2.steps == ["a1.", "b0."] and [p[2] for p in it2.paragraphs] == [True, True]


def test_scoring_normalisation():
    assert normalize_answer("The  Quarry-Line!") == "quarry line"
    assert f1_score("Steve Hillage", ["steve hillage"]) == 1.0
    assert 0 < f1_score("Steve", ["Steve Hillage"]) < 1
    assert f1_score("nothing", ["Steve Hillage", "nothing"]) == 1.0


def test_rule_reads_a_bridge_named_only_by_the_next_paragraphs_title():
    s1 = "Glass Harbour is the second album by the folk singer Ada Renn"
    s2 = "She has been married to the sound engineer Joel Marsh since 1983"
    q = "Who is the spouse of the singer of Glass Harbour?"
    assert parse_chain([s1, s2], "Joel Marsh", q, s1 + " " + s2) == []  # the second fact says "She"
    got = parse_chain([s1, s2], "Joel Marsh", q, s1 + " " + s2, titles=["Glass Harbour", "Ada Renn"])
    assert got == [(s1, "Ada Renn"), (s2, "Joel Marsh")]


def test_exam_reads_the_answer_after_so():
    from evaluation.real_multihop_eval import _answer_of

    assert _answer_of("Green is by Steve Hillage. Giraudy is his partner. So Miquette Giraudy.", True) == "Miquette Giraudy"
    assert _answer_of("Miquette Giraudy. Then more text", False) == "Miquette Giraudy"
    assert _answer_of("no chain at all.", True) == "no chain at all"
