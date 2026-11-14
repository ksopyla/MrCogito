"""scripts/build_reasoning_rows.py: multi-hop library rows and planted-fact lookup rows for the long-context mix.

No network: tiny generated fixtures and a word-level stand-in tokenizer; the tests marked "real" use the SmolLM2
tokenizer when it is in the local Hugging Face cache and are skipped otherwise.
"""
import argparse
import gzip
import io
import itertools
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_reasoning_rows as b  # noqa: E402

from data import bin_token_dataset as btd  # noqa: E402

REQUIRED_BANS = ["special magic", "magic number", "magic uuid", "pass key", "passkey"]
WORDS = ("river stone garden window market harbour candle winter morning letter captain village meadow bridge "
         "silver quiet journey kitchen ladder orchard shadow thunder velvet whisper yellow anchor basket copper "
         "donkey engine feather goblin hammer island jacket kettle lemon mirror needle orange pebble quarrel ribbon "
         "saddle timber umbrella valley wagon").split()


# ----------------------------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------------------------
class FakeTok:
    """A word-level stand-in with the one property the builder relies on: where a non-space character is followed by
    white space there is always a token boundary (a token never holds white space after its first character).
    Ids are given in order of first use and shared by every instance, so within one test run a text always has the
    same ids."""

    _RE = re.compile(re.escape(b.EOT_TOKEN) + r"| ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]|\s")
    words, index = [b.EOT_TOKEN], {b.EOT_TOKEN: 0}

    def __init__(self):
        self.eot, self.name, self.vocab = 0, "fake", 65536

    def _id(self, w: str) -> int:
        if w not in self.index:
            assert len(self.words) < 65536
            self.index[w] = len(self.words)
            self.words.append(w)
        return self.index[w]

    def encode(self, text):
        return [self._id(m.group()) for m in self._RE.finditer(text)]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def encode_offsets(self, texts):
        return [([self._id(m.group()) for m in self._RE.finditer(t)], [m.span() for m in self._RE.finditer(t)])
                for t in texts]

    def decode(self, ids):
        return "".join(self.words[int(i)] for i in ids)


def real_tok():
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download("HuggingFaceTB/SmolLM2-135M", "tokenizer.json", local_files_only=True)
        return b.Tok(Tokenizer.from_file(path), "HuggingFaceTB/SmolLM2-135M")
    except Exception as e:  # noqa: BLE001 - not cached, or a library is missing
        pytest.skip(f"SmolLM2 tokenizer not in the local cache ({type(e).__name__})")


def _sentence(rng, lo=6, hi=14):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi))).capitalize() + "."


def wiki_records(n, seed=0, sentences=(1, 4)):
    """2Wiki-shaped records: two supporting paragraphs (a film, its director) and eight others from a shared pool."""
    rng = random.Random(seed)
    pool = [(f"Place {k}", [_sentence(rng) for _ in range(rng.randint(*sentences))]) for k in range(3 * n + 20)]
    recs = []
    for i in range(n):
        film, person, year = f"Film {i} ({1900 + i % 90} film)", f"Director{i} Surname{i}", str(1800 + i)
        ctx = rng.sample(pool, 8) + [(film, [f"{film} is a film directed by {person}.", _sentence(rng)]),
                                     (person, [f"{person} was born in {year}.", _sentence(rng)])]
        rng.shuffle(ctx)
        yes_no = i % 3 == 0
        recs.append({"id": f"q{i}", "type": "compositional",
                     "question": f"Was the director of {film} born in {year}?" if yes_no
                     else f"When was the director of {film} born?",
                     "answer": "yes" if yes_no else year,
                     "evidences": [[film, "director", person], [person, "date of birth", year]],
                     "supporting_facts": {"title": [film, person], "sent_id": [0, 0]},
                     "context": {"title": [t for t, _ in ctx], "sentences": [s for _, s in ctx]}})
    return recs


def wiki_items(n, seed=0, **kw):
    items = [b.parse_wiki_record(r)[0] for r in wiki_records(n, seed, **kw)]
    assert all(items)
    return items


class FakeWiki:
    def __init__(self, items, n_val=30, group=25):
        self.val, rest = items[:n_val], items[n_val:]
        self.groups = [rest[i: i + group] for i in range(0, len(rest), group)]
        self.n_train_groups, self.asked, self.cleaned = len(self.groups), 0, False

    def val_items(self):
        return list(self.val)

    def train_items(self, n):
        self.asked = max(self.asked, n)
        return [it for g in self.groups[:n] for it in g]

    def describe(self):
        return {"repo": "fake-2wiki"}

    def cleanup(self):
        self.cleaned = True


def book_text(seed, n_words):
    """Gutenberg-like prose: sentences in paragraphs, lines wrapped, blank lines between paragraphs."""
    rng, paras, words = random.Random(seed), [], 0
    while words < n_words:
        sents = []
        for _ in range(rng.randint(3, 7)):
            s = _sentence(rng)
            if rng.random() < 0.15:
                s = "Mr. " + s[0].upper() + s[1:]
            sents.append(s)
        text = " ".join(sents)
        words += len(text.split())
        line, lines = "", []
        for w in text.split():
            if len(line) + len(w) > 68:
                lines.append(line.rstrip())
                line = ""
            line += w + " "
        paras.append("\n".join(lines + [line.rstrip()]))
    return "\n\n".join(paras) + "\n"


class FakeBooks:
    def __init__(self, books, mix_val=()):
        self.items, self.mix_val, self.read, self.cleaned = books, set(mix_val), 0, False

    def books(self):
        for x in self.items:
            self.read += 1
            yield (*x, x[0] in self.mix_val)   # the last field: validation in the training mix

    def describe(self):
        return {"repo": "fake-books"}

    def cleanup(self):
        self.cleaned = True


def some_books(n, words=5200, seed=0):
    return [(f"b{i}", f"Book {i}", "Public Domain", book_text(seed * 1000 + i, words)) for i in range(n)]


LOOKUP = {"kind": "lookup", "seq_len": 2048, "tokens": 2048 * 9, "facts": (4, 12), "spans_per_book": 3,
          "buckets": {"val": (0, 30), "train": (30, 100)}}
TINY_SETS = {"multihop_2k": {"kind": "multihop", "seq_len": 2048, "tokens": 2048 * 14}, "lookup_2k": LOOKUP}


def args_for(out, **kw):
    base = {"out": str(out), "only": "", "scale": 1.0, "seed": 0, "tokenizer": "fake", "val_rows": 6,
            "shard_tokens": 2048 * 5}
    return argparse.Namespace(**{**base, **kw})


def build(out, **kw):
    wiki, books = FakeWiki(wiki_items(260)), FakeBooks(some_books(40))
    res = b.run(args_for(out, **kw), wiki=wiki, books=books, tok=FakeTok(), sets=TINY_SETS)
    return res, wiki, books


def check_row(row, tok, seq_len):
    """What every row must be: [eot] + the text's own tokenization, eot-padded to exactly seq_len."""
    assert row.ids.dtype == np.uint16 and len(row.ids) == seq_len
    assert row.ids[0] == tok.eot and 1 < row.n <= seq_len
    body = row.ids[1: row.n].tolist()
    assert tok.eot not in body and (row.ids[row.n:] == tok.eot).all()
    assert tok.encode(row.text) == body          # the pieces, concatenated, are the tokenization of the text
    assert tok.decode(body) == row.text
    assert row.text[row.q_char:].startswith("\n\nQuestion: ") and row.text.endswith("\n\n")
    assert "Question:" not in row.text[: row.q_char]
    for q in row.qa:                              # answer_pos is where the answer's tokens are
        a = tok.encode(" " + q["answer"])
        assert row.ids[q["answer_pos"]: q["answer_pos"] + len(a)].tolist() == a


def blocks(text):
    """(context, [(question, evidence or None, answer)]) of a decoded row."""
    at = text.index("\n\nQuestion: ")
    found = re.findall(r"\n\nQuestion: (.*)\n(?:Evidence: (.*)\n)?Answer: (.*)", text[at:])
    return text[:at], [(q, e or None, a) for q, e, a in found]


# ----------------------------------------------------------------------------------------------------------------
# Templates and facts
# ----------------------------------------------------------------------------------------------------------------
def test_templates_are_many_and_avoid_the_eval_needles():
    statements, questions = b.statement_templates(), b.question_templates()
    assert len(statements) >= 40 and len(questions) >= 40
    assert len(b.KINDS) >= 7
    assert all(x in b.BANNED for x in REQUIRED_BANS)
    for s in statements + questions + b.REVISIONS + b.template_vocabulary():
        assert b.banned_in(s) is None, s
    assert b.banned_in("One of the special magic numbers for x is: 1.") == "special magic"
    assert b.banned_in("The Pass Key is 71432.") == "pass key"
    assert b.banned_in("VAR QPE = 12") == "VAR " and b.banned_in("Tolvar is here") is None


def test_eval_suite_needle_words_are_banned():
    """The needle wording of the RULER tasks our suite runs, read from lm-eval's source when it is installed."""
    import importlib.util

    suite = (Path(__file__).resolve().parents[1] / "evaluation" / "run_lsa_suite.py").read_text()
    assert "niah_single_1" in suite and "ruler_vt" in suite and "babilong_qa1" in suite  # the evals the list is for
    spec = importlib.util.find_spec("lm_eval")
    if spec is None:
        pytest.skip("lm-eval is not installed")
    ruler = Path(spec.origin).parent / "tasks" / "ruler"
    needle = re.search(r'NEEDLE = "(.*)"', (ruler / "prepare_niah.py").read_text()).group(1)
    ask = re.search(r'TEMPLATE = """(.*?)"""', (ruler / "niah_utils.py").read_text(), re.S).group(1)
    track = re.search(r'"template": """(.*?)"""', (ruler / "vt_utils.py").read_text(), re.S).group(1)
    for text in (needle, ask, track, "VAR ABCDE = 12345 "):
        assert b.banned_in(text), text
    opening = needle.split("{")[0].lower()                     # "one of the special magic "
    assert not any(t.lower().startswith(opening[:10]) for t in b.statement_templates())


def test_facts_are_varied_revised_and_answer_the_latest_value():
    rng = random.Random(5)
    names = b._Names(rng)
    facts = [b.make_fact(rng, names, revised=rng.random() < b.P_REVISED) for _ in range(3000)]
    assert {f.kind for f in facts} == set(b.KINDS)
    assert len({f.subject for f in facts}) == len(facts)
    share = sum(len(f.statements) == 2 for f in facts) / len(facts)
    assert 0.11 < share < 0.19
    assert len({s for f in facts for s in f.statements}) == sum(len(f.statements) for f in facts)
    for f in facts:
        assert f.answer == f.values[-1] and len(f.statements) == len(f.values)
        assert len(set(f.values)) == len(f.values)                      # a revision changes the value
        for v, s in zip(f.values, f.statements):
            assert v in s and f.subject in s
        assert f.subject in f.question and f.answer not in f.question
        assert not any(b.banned_in(x) for x in f.statements + [f.question])
        if f.kind == "identifier":
            assert all(6 <= len(v) <= 10 for v in f.values)
    planted = facts[:12]
    for m in (b.make_missing(rng, names, planted) for _ in range(200)):
        assert m.answer == b.NOT_STATED and not m.statements and not m.values
        assert m.question not in {f.question for f in planted}
    known = sum(m.subject in {f.subject for f in planted}
                for m in (b.make_missing(rng, names, planted) for _ in range(200)))
    assert 40 < known < 160   # both kinds: another attribute of a planted subject, and a subject never named


def test_row_facts_counts():
    rows = [b.make_row_facts(random.Random(i), (4, 12)) for i in range(600)]
    assert {len(p) for p, _ in rows} == set(range(4, 13))
    with_missing = [m for _, m in rows if m]
    assert 0.10 < len(with_missing) / len(rows) < 0.20 and {len(m) for m in with_missing} == {1, 2}


# ----------------------------------------------------------------------------------------------------------------
# Multi-hop rows
# ----------------------------------------------------------------------------------------------------------------
def test_parse_wiki_record_and_evidence():
    rec = wiki_records(4)[1]
    it, why = b.parse_wiki_record(rec)
    assert why == "" and it.answer == "1801" and len(it.paras) == 10 and sum(p[2] for p in it.paras) == 2
    assert it.evidence == ("The director of Film 1 (1901 film) is Director1 Surname1. "
                           "The date of birth of Director1 Surname1 is 1801.")
    assert b.evidence_sentence("X", "educated at", "Y") == "X was educated at Y"
    assert b.evidence_sentence("X", "narrated by", "Y") == "For X, narrated by: Y"
    no_triples = b.parse_wiki_record({**rec, "evidences": []})[0]  # falls back to the supporting sentences
    assert no_triples.evidence == ("Film 1 (1901 film) is a film directed by Director1 Surname1. "
                                   "Director1 Surname1 was born in 1801.")
    assert b.parse_wiki_record({**rec, "answer": "Atlantis"}) == (None, "answer_not_in_context")
    assert b.parse_wiki_record({**rec, "answer": " "})[1] == "empty_question_or_answer"
    gone = {**rec, "supporting_facts": {"title": ["Nowhere"], "sent_id": [0]}}
    assert b.parse_wiki_record(gone)[1] == "supporting_paragraph_missing"
    twice = {**rec, "context": {"title": rec["context"]["title"] * 2, "sentences": rec["context"]["sentences"] * 2}}
    assert len(b.parse_wiki_record(twice)[0].paras) == 10


def _library(tok, seq_len, items, n_rows, passes=2, seed="s"):
    stream = ((p, it) for p in range(passes) for it in random.Random(p).sample(items, len(items)))
    return list(itertools.islice(b.multihop_rows(b._Feed(stream, tok, block=16), tok, seq_len, seed), n_rows))


def test_multihop_rows_fake_tokenizer():
    tok, S, items = FakeTok(), 4096, wiki_items(120)
    rows = _library(tok, S, items, 100)
    assert len(rows) >= 12
    seen = set()
    for row in rows:
        check_row(row, tok, S)
        assert row.n >= 0.97 * S
        ctx, qs = blocks(row.text)
        assert len(qs) == len(row.qa) >= 3 and ctx.startswith("Title: ")
        titles = re.findall(r"(?:^|\n\n)Title: (.*)\n", ctx)
        assert len(titles) == len(set(titles)) == row.info["paragraphs"]
        by_id = {it.id: it for it in items}
        for (q, ev, a), meta in zip(qs, row.qa):
            it = by_id[meta["id"]]
            assert (q, ev, a) == (it.question, it.evidence, it.answer)
            assert (row.info["pass"], it.id) not in seen          # an item at most once per pass
            seen.add((row.info["pass"], it.id))
            if a not in ("yes", "no"):
                assert a in ctx                                   # the answer is in the library
            ends = []
            for title, text, sup in it.paras:
                if sup:                                           # every supporting paragraph is in the row
                    piece = f"Title: {title}\n{text}"
                    assert piece in ctx
                    ends.append(1 + len(tok.encode(ctx[: ctx.index(piece) + len(piece)])))
            assert meta["distance"] == meta["answer_pos"] - min(ends) > 0
    assert np.mean([r.n / S for r in rows]) > 0.985
    assert {r.info["pass"] for r in rows} == {0, 1}
    again = _library(FakeTok(), S, items, 100)
    assert all((x.ids == y.ids).all() for x, y in zip(rows, again)) and len(again) == len(rows)
    other = _library(FakeTok(), S, items, 100, seed="t")
    assert any(x.text != y.text for x, y in zip(rows, other))


def test_multihop_questions_close_to_their_paragraphs_are_asked_last(monkeypatch):
    monkeypatch.setattr(b, "MIN_BACK", 1500)
    tok, S = FakeTok(), 4096
    late_rows = 0
    for row in _library(tok, S, wiki_items(120), 8, passes=1):
        q0 = 1 + len(tok.encode(row.text[: row.q_char]))                       # where the question block starts
        far = [q["answer_pos"] - q["distance"] for q in row.qa]                # end of its farthest paragraph
        head = [len(tok.encode(f"\n\nQuestion: {q}\nEvidence: {ev}\nAnswer:")) for q, ev, _ in blocks(row.text)[1]]
        late = [q0 + h - f < 1500 for h, f in zip(head, far)]                  # too close if it were asked first
        assert late == sorted(late)                                            # those are asked after all the others
        close = [f for f, flag in zip(far, late) if flag]
        assert close == sorted(close)                                          # and the closest one last of all
        late_rows += any(late)
    assert late_rows >= 2


def test_wiki_item_stream_reads_more_groups_before_a_second_pass():
    items = wiki_items(130)
    src = FakeWiki(items, n_val=30, group=25)          # 4 train groups of 25
    got = list(itertools.islice(b.wiki_item_stream(src, "train", "k", est_items=10), 250))
    assert [p for p, _ in got[:100]] == [0] * 100 and len({it.id for _, it in got[:100]}) == 100
    assert [p for p, _ in got[100:200]] == [1] * 100 and len({it.id for _, it in got[100:200]}) == 100
    assert [it.id for _, it in got[:100]] != [it.id for _, it in got[100:200]]
    assert not {it.id for _, it in got} & {it.id for it in src.val_items()}
    val = list(itertools.islice(b.wiki_item_stream(src, "val", "k", est_items=10), 60))
    assert {it.id for _, it in val} == {it.id for it in items[:30]} and [p for p, _ in val] == [0] * 30 + [1] * 30


def test_wiki_source_splits_row_groups(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    recs = wiki_records(60)
    files = {}
    for k in range(2):
        path = tmp_path / f"train-{k}.parquet"
        pq.write_table(pa.Table.from_pylist(recs[k * 30: (k + 1) * 30]), path, row_group_size=5)
        files[f"train-{k}"] = path
    a = b.WikiSource(tmp_path, 0, files=files, n_val_groups=3)
    c = b.WikiSource(tmp_path, 1, files=files, n_val_groups=3)
    assert a.val_groups == c.val_groups and len(a.val_groups) == 3          # the val items do not depend on the seed
    assert a.train_groups != c.train_groups and a.n_train_groups == 9
    assert not set(a.val_groups) & set(a.train_groups)
    val, train = {it.id for it in a.val_items()}, {it.id for it in a.train_items(99)}
    assert len(val) == 15 and len(train) == 45 and not val & train
    assert [it.id for it in a.train_items(2)] == [it.id for it in a.train_items(4)][:10]
    d = a.describe()
    assert d["splits_read"] == ["train"] and d["records"]["kept"] == 60
    assert all("validation" not in n and "test" not in n for n in b.WIKI_TRAIN_FILES)


# ----------------------------------------------------------------------------------------------------------------
# Lookup rows
# ----------------------------------------------------------------------------------------------------------------
def _lookup(tok, cfg, text, doc_id="b7", seed="s"):
    (ids, offsets), = tok.encode_offsets([text])
    return b.lookup_rows(doc_id, text, ids, offsets, tok, cfg, seed)


def check_lookup_row(row, tok, cfg, book):
    check_row(row, tok, cfg["seq_len"])
    ctx, qs = blocks(row.text)
    assert [(q, a) for q, _, a in qs] == [(m["question"], m["answer"]) for m in row.qa]
    planted = [m for m in row.qa if m["values"]]
    missing = [m for m in row.qa if not m["values"]]
    assert cfg["facts"][0] <= len(planted) <= cfg["facts"][1] and len(missing) <= 2
    assert row.info["facts"] == len(planted) and row.info["not_stated"] == len(missing)
    plain = ctx
    for m in planted:
        assert m["answer"] == m["values"][-1] != b.NOT_STATED
        assert ctx.count(m["answer"]) >= 1 and len(m["statements"]) == len(m["values"]) <= 2
        at = [ctx.index(s) for s in m["statements"]]
        assert at == sorted(at)                                   # a revision comes after what it revises
        for s, v, end in zip(m["statements"], m["values"], m["statement_ends"]):
            assert ctx.count(s) == 1 and v in s
            assert 1 + len(tok.encode(ctx[: ctx.index(s) + len(s)])) == end
            before = ctx[: ctx.index(s)]
            sep = "\n\n" if before.endswith("\n\n") else " "
            assert before.endswith(sep) and not before.rstrip().endswith("Mr.")
            assert before.rstrip()[-1] in ".!?\"')"               # planted after a sentence or a paragraph
            plain = plain.replace(sep + s, "", 1)
        assert m["distance"] == m["answer_pos"] - m["statement_ends"][-1] > 0
        if len(m["values"]) == 2:                                 # the answer is the later value
            assert m["values"][0] != m["values"][1] and m["answer"] == m["values"][1]
            assert ctx.index(m["statements"][1]) > ctx.index(m["statements"][0])
    assert plain in book                                          # without the facts: one contiguous span of the book
    assert len(tok.encode(plain)) == row.info["span_tokens"]
    for m in missing:
        assert m["answer"] == b.NOT_STATED and m["distance"] is None
        same = [p for p in planted if p["subject"] == m["subject"]]
        if same:                                                  # the subject is known, this fact about it is not
            assert (same[0]["kind"], same[0]["attr"]) != (m["kind"], m["attr"])
        else:
            assert m["subject"] not in ctx
    assert not any(b.banned_in(s) for m in row.qa for s in m["statements"] + [m["question"]])
    assert b.banned_in(row.text) is None                          # the fixture book has no such words either


def test_lookup_rows_fake_tokenizer():
    tok, cfg = FakeTok(), LOOKUP
    revised = missing = total = 0
    for i in range(12):
        book = book_text(i, 6500)
        rows = _lookup(tok, cfg, book, doc_id=f"b{i}")
        assert len(rows) == 3
        spans = []
        for row in rows:
            check_lookup_row(row, tok, cfg, book)
            assert row.n >= cfg["seq_len"] - 70                    # filled to the end of a sentence
            plain = blocks(row.text)[0]
            for m in row.qa:
                for s in m["statements"]:
                    plain = plain.replace(("\n\n" if "\n\n" + s in plain else " ") + s, "", 1)
            spans.append((book.index(plain), book.index(plain) + len(plain)))
            revised += row.info["revised"]
            missing += row.info["not_stated"]
            total += row.info["facts"]
        assert all(a[1] <= c[0] for a, c in zip(spans, spans[1:]))  # the spans of a book do not overlap
        again = _lookup(FakeTok(), cfg, book, doc_id=f"b{i}")
        assert all((x.ids == y.ids).all() and x.text == y.text for x, y in zip(rows, again))
        assert [x.text for x in rows] != [y.text for y in _lookup(FakeTok(), cfg, book, doc_id=f"b{i}", seed="t")]
    assert revised > 0 and missing > 0 and 0.05 < revised / total < 0.3
    assert _lookup(tok, cfg, book_text(1, 900)) == []              # a book under seq_len tokens gives no row
    assert _lookup(tok, cfg, book_text(1, 6500) + b.EOT_TOKEN) == []


def test_book_split_rule():
    assert b.stable_hash("10002") == int.from_bytes(
        __import__("hashlib").blake2b(b"10002", digest_size=8).digest(), "big")
    ids = [str(i) for i in range(40000)]
    held = [i for i in ids if b.book_held_out(i, "text of " + i)]
    assert 0.012 < len(held) / len(ids) < 0.03                    # two 0.5 % rules, on the id and on the text
    assert all(b.book_held_out(i, "") for i in ids if b.stable_hash(i) % 200 == 0)
    assert all(b.book_held_out(i, "") for i in ids if b.stable_hash(i) % 1_000_000 < 5000)
    text = next(t for t in (f"body {k}" for k in range(10 ** 6)) if b.stable_hash(t) % 200 == 0)
    assert b.book_held_out("no-id-rule-hit" if not b.book_held_out("no-id-rule-hit", "") else "x1", text)
    buckets = np.bincount([b.book_bucket(i) for i in ids], minlength=100)
    assert buckets.min() > 300 and len(buckets) == 100
    spans = sorted(tuple(r) for n in ("lookup_16k", "lookup_64k") for r in b.SETS[n]["buckets"].values())
    assert all(a[1] <= c[0] for a, c in zip(spans, spans[1:]))    # no book feeds two sets or two splits


def test_longdocs_validation_rule_is_replayed():
    count = lambda texts: [len(t) // 4 for t in texts]            # a stand-in for the GPT-2 token count
    rng = random.Random(0)
    texts = ["x" * rng.choice([100, 40959, 40960, 65535, 65536, 90000, 300000]) for _ in range(900)]
    rule = b.LongdocsVal(count)
    marks = rule.mark(texts[:333]) + rule.mark(texts[333:334]) + rule.mark(texts[334:])   # any batching, same marks
    kept = [i for i, t in enumerate(texts) if len(t) >= 40960 and len(t) // 4 >= 16384]
    assert [i for i, m in enumerate(marks) if m] == kept[99::100] and len(kept[99::100]) >= 3
    assert rule.kept == len(kept) and not any(marks[i] for i, t in enumerate(texts) if len(t) < 65536)
    try:  # the mix builder's own replay of the rule, when that script is here: the two must agree
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "build_m21_mix", Path(__file__).resolve().parents[1] / "scripts" / "build_m21_mix.py")
        mix = importlib.util.module_from_spec(spec)
        sys.modules["build_m21_mix"] = mix
        spec.loader.exec_module(mix)
        theirs, hashes = mix.LongdocsLiteHeldout(count), [mix.stable_hash(k) for k in ("1", "10002", "\u00e9t\u00e9")]
        rule_val = mix.HashHeldout(0.005)
    except Exception as e:  # noqa: BLE001 - not there, renamed, or its imports are missing
        pytest.skip(f"scripts/build_m21_mix.py cannot be compared ({type(e).__name__})")
    assert theirs.mark([str(i) for i in range(len(texts))], texts) == marks
    assert hashes == [b.stable_hash(k) for k in ("1", "10002", "\u00e9t\u00e9")]
    ids = [str(i) for i in range(30000)]
    assert all(b.book_held_out(i, "") for i in ids if rule_val.is_val(i)) and any(rule_val.is_val(i) for i in ids)


def test_book_source_marks_the_stream_once(tmp_path, monkeypatch):
    docs = [{"id": str(i), "metadata": {"license": "Public Domain", "title": f"T{i}"},
             "text": "".join(random.Random(i).choices("abcdefgh \n", k=41000 if i % 3 else 500))} for i in range(70)]
    blob = gzip.compress(("\n".join(json.dumps(d) for d in docs) + "\n").encode(), 6)
    asked, counted = [], []

    def fake_range(url, start, end):
        asked.append((start, end))
        return blob[start:end]

    def count(texts):
        counted.extend(len(t) for t in texts)
        return [20000] * len(texts)

    monkeypatch.setattr(b, "http_range", fake_range)
    monkeypatch.setattr(b.HubPrefix, "CHUNK", 4096)
    monkeypatch.setattr(b.BookSource, "GROUP", 5)
    monkeypatch.setattr(b, "hub_file_url", lambda repo, name, revision: name)
    monkeypatch.setattr(b, "hub_info", lambda repo: {"sha": "feedbeef", "license_tag": None,
                                                    "sizes": {"README.md": 10, "books-0000.json.gz": len(blob)}})
    src = b.BookSource(tmp_path / "_raw", count=lambda texts: count(texts))
    first = list(itertools.islice(src.books(), 12))
    assert [x[0] for x in first] == [str(i) for i in range(12)] and first[3][1:3] == ("T3", "Public Domain")
    assert first[5][3] == docs[5]["text"] and all(len(x) == 5 for x in first)
    assert len(counted) == 10 and 0 < src.describe()["files"][0]["bytes_fetched"] < len(blob)   # 15 read, 10 long
    full = list(src.books())                                       # the second reader: the head is not counted again
    assert len(full) == 70 and len(counted) == sum(i % 3 > 0 for i in range(70)) and len(set(asked)) == len(asked)
    long = [i for i in range(70) if i % 3]
    assert [int(x[0]) for x in full if x[4]] == long[99::100] == []   # under 100 kept documents: none is validation
    monkeypatch.setattr(src._val, "every", 7)
    src._marks, src._val.kept = [], 0
    assert [int(x[0]) for x in src.books() if x[4]] == long[6::7]      # every 7th kept one, in file order
    assert [x[4] for x in src.books()] == [x[4] for x in src.books()]
    assert src.describe()["revision"] == "feedbeef" and (tmp_path / "_raw").exists()
    src.cleanup()
    assert not list((tmp_path / "_raw").rglob("*.gz"))


def test_lookup_set_skips_held_out_and_other_licence_books(tmp_path):
    held = [f"h{i}" for i in range(3000) if b.stable_hash(f"h{i}") % 200 == 0][:3]
    by_text = next(k for k in range(10 ** 6) if b.stable_hash(book_text(9, 5200) + str(k)) % 1_000_000 < 5000)
    extra = [(h, "held", "Public Domain", book_text(50 + k, 5200)) for k, h in enumerate(held)]
    extra += [("t0", "held by its text", "Public Domain", book_text(9, 5200) + str(by_text)),
              ("c0", "not free", "CC BY-NC", book_text(77, 5200)), ("s0", "short", "Public Domain", book_text(3, 300))]
    books = extra + some_books(40)
    (tmp_path / "free").mkdir()
    free = b.build_lookup_set("lk", LOOKUP, FakeBooks(books), FakeTok(), tmp_path / "free", 0, {"train": 9, "val": 6},
                              2048 * 5)
    mix_val = {free["train"]["book_ids"][0], free["val"]["book_ids"][0]}   # books the mix keeps for validation
    src, tok = FakeBooks(books, mix_val), FakeTok()
    res = b.build_lookup_set("lk", LOOKUP, src, tok, tmp_path, 0, {"train": 9, "val": 6}, 2048 * 5)
    assert res["train"]["n_rows"] == 9 and res["val"]["n_rows"] == 6 and res["train"]["target_met"]
    used = set(res["train"]["book_ids"]) | set(res["val"]["book_ids"])
    assert not used & (set(held) | {"t0", "c0", "s0"} | mix_val)
    assert res["books"]["books_held_out_hash_rule"] == 4 and res["books"]["books_held_out_longdocs_rule"] == 2
    assert res["books"]["books_other_licence"] == 1
    assert not set(res["train"]["book_ids"]) & set(res["val"]["book_ids"])
    assert all(30 <= b.book_bucket(i) < 100 for i in res["train"]["book_ids"])
    assert all(b.book_bucket(i) < 30 for i in res["val"]["book_ids"])
    assert src.read < len(books)                                   # it stops reading when both splits are full
    assert [p.name for p in sorted(tmp_path.glob("lk_*"))] == ["lk_train_000000.bin", "lk_train_000001.bin",
                                                               "lk_val_000000.bin"]


# ----------------------------------------------------------------------------------------------------------------
# The build: shards, done.json, the training reader, resume, determinism
# ----------------------------------------------------------------------------------------------------------------
def shard_rows(path, seq_len):
    arr = np.asarray(btd.open_shard(str(path)))
    assert len(arr) % seq_len == 0
    return arr.reshape(-1, seq_len)


def test_build_writes_whole_rows_the_trainer_can_read(tmp_path):
    tok = FakeTok()
    wiki, books = FakeWiki(wiki_items(260)), FakeBooks(some_books(40))
    (tmp_path / "_raw").mkdir()                      # the mix builder's own folder, when --out is the mix folder
    (tmp_path / "_raw" / "theirs.bin").write_bytes(b"x")
    (tmp_path / b.RAW_DIR).mkdir()                   # what an interrupted run of this builder leaves behind
    (tmp_path / b.RAW_DIR / "half.gz").write_bytes(b"x")
    res = b.run(args_for(tmp_path), wiki=wiki, books=books, tok=tok, sets=TINY_SETS)
    assert res == {"built": ["multihop_2k", "lookup_2k"], "skipped": []}
    assert wiki.cleaned and books.cleaned and not (tmp_path / b.RAW_DIR).exists()
    assert (tmp_path / "_raw" / "theirs.bin").exists() and b.RAW_DIR != "_raw"
    assert b.MAGIC == btd.MAGIC and b.HEADER_INTS == btd.HEADER_INTS
    for name, cfg in TINY_SETS.items():
        S = cfg["seq_len"]
        done = json.loads((tmp_path / f"{name}.done.json").read_text())
        train = sorted(tmp_path.glob(f"{name}_train_*.bin"))
        n_train = round(cfg["tokens"] / S)
        assert [p.name for p in train] == [f"{name}_train_{k:06d}.bin" for k in range(len(train))] == done["shards"]
        assert [(x["file"], x["tokens"]) for x in done["train"]["shards"]] == [
            (p.name, len(btd.open_shard(str(p)))) for p in train]
        assert done["val_shards"] == [f"{name}_val_000000.bin"]
        assert done["train"]["rows"] == {"stream": {str(S): n_train}}
        assert [p.name for p in tmp_path.glob(f"{name}_val_*.bin")] == [f"{name}_val_000000.bin"]
        assert done["rows"] == n_train and done["tokens"] == n_train * S and done["val"]["n_rows"] == 6
        assert done["val"]["rows"] == {"stream": {str(S): 6}} and done["train"]["n_rows"] == n_train
        assert done["seq_len"] == S and done["eot"] == 0 and done["rule"] == "stream" and done["seconds"] >= 0
        assert done["text_bytes"] > 2 * done["rows"] * S and done["items_per_row"]["mean"] >= 3
        assert done["answer_distance"]["n"] > 0 and "histogram" in done["answer_distance"]
        assert done["train"]["joint_check"] == {"rows": 1, "mismatched": 0}
        assert done["parameters"]["seed"] == 0 and done["hub"]["repo"].startswith("fake")
        assert done["train"]["fill"]["min"] > 0.96
        rows = np.concatenate([shard_rows(p, S) for p in train])
        assert len(rows) == n_train and all(len(shard_rows(p, S)) * S <= 2048 * 5 for p in train)
        for split, files in (("train", train), ("val", [tmp_path / f"{name}_val_000000.bin"])):
            ds = btd.BinRowDataset([{"name": name, "files": [str(p) for p in files]}], seq_len=S, rule="stream",
                                   eot=0, seed=0)
            assert len(ds) == done[split]["n_rows"]
            for i in range(len(ds)):
                item = ds[i]
                ids, doc = item["input_ids"].numpy(), item["doc_ids"].numpy()
                assert len(ids) == S and ids[0] == 0 and doc[0] == 0
                n = int((doc == 0).sum())                           # the row's one real document
                assert n > 0.96 * S and (ids[1:n] != 0).all() and (ids[n:] == 0).all()
                assert (np.diff(doc[n - 1:]) == 1).all()            # each padding eot is a document of its own
                ctx, qs = blocks(tok.decode(ids[1:n]))
                assert len(qs) >= 3
                for q, ev, a in qs:
                    assert (ev is not None) == (cfg["kind"] == "multihop")
                    if a not in ("yes", "no", b.NOT_STATED):
                        assert a in ctx, (q, a)                    # the answer is in the row before its question block
                assert b.banned_in(tok.decode(ids[1:n])) is None
    look = json.loads((tmp_path / "lookup_2k.done.json").read_text())
    assert all(x in look["parameters"]["banned_strings_checked"] for x in REQUIRED_BANS)
    assert look["parameters"]["statement_templates"] >= 40 and look["parameters"]["question_templates"] >= 40
    assert look["train"]["counts"]["facts"] >= 4 * look["rows"] and look["books"]["books_used"] >= 3
    multi = json.loads((tmp_path / "multihop_2k.done.json").read_text())
    assert multi["train"]["passes_used"] >= 1 and multi["train"]["counts"]["paragraphs"] > 0
    try:  # scripts/build_m21_mix.py lists these sets in its mix.json from their done.json: it must read the rows
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "build_m21_mix", Path(__file__).resolve().parents[1] / "scripts" / "build_m21_mix.py")
        mix = importlib.util.module_from_spec(spec)
        sys.modules["build_m21_mix"] = mix
        spec.loader.exec_module(mix)
        seen = {s: mix.rows_available(multi, s, "stream", 2048, tmp_path, 0) for s in ("train", "val")}
    except Exception as e:  # noqa: BLE001 - not there, renamed, or its imports are missing
        pytest.skip(f"scripts/build_m21_mix.py cannot be asked ({type(e).__name__})")
    assert seen == {"train": 14, "val": 6}


def test_build_is_deterministic_and_resumes(tmp_path):
    def files(d):
        return {p.name: p.read_bytes() for p in sorted(d.glob("*.bin"))}

    build(tmp_path / "a")
    build(tmp_path / "b")
    a = files(tmp_path / "a")
    assert len(a) == 7 and a == files(tmp_path / "b")
    build(tmp_path / "c", seed=1)
    c = files(tmp_path / "c")
    assert a.keys() == c.keys() and all(a[k] != c[k] for k in a)
    # resume: a set with done.json is skipped; one without is rebuilt, the same bytes
    stamp = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "a").iterdir()}
    res, wiki, books = build(tmp_path / "a")
    assert res == {"built": [], "skipped": ["multihop_2k", "lookup_2k"]} and wiki.asked == 0 and books.read == 0
    assert stamp == {p.name: p.stat().st_mtime_ns for p in (tmp_path / "a").iterdir()}
    (tmp_path / "a" / "lookup_2k.done.json").unlink()
    (tmp_path / "a" / "lookup_2k_train_000001.bin").write_bytes(b"half written")
    (tmp_path / "a" / "lookup_2k_train_000007.bin.tmp").write_bytes(b"stale")
    res, wiki, _ = build(tmp_path / "a")
    assert res == {"built": ["lookup_2k"], "skipped": ["multihop_2k"]} and wiki.asked == 0
    assert files(tmp_path / "a") == a and not list((tmp_path / "a").glob("*.tmp"))


def test_only_and_scale(tmp_path):
    res, _, books = build(tmp_path, only="multihop_2k", scale=0.5)
    assert res["built"] == ["multihop_2k"] and books.read == 0
    assert not list(tmp_path.glob("lookup_2k*"))
    assert json.loads((tmp_path / "multihop_2k.done.json").read_text())["rows"] == 7
    with pytest.raises(SystemExit):
        build(tmp_path, only="nope")
    assert b.target_rows(b.SETS["multihop_16k"], 1.0) == 12207 and b.target_rows(b.SETS["lookup_64k"], 0.002) == 1
    assert {n: (c["seq_len"], c["tokens"]) for n, c in b.SETS.items()} == {
        "multihop_16k": (16384, 0.20e9), "multihop_64k": (65536, 0.06e9),
        "lookup_16k": (16384, 0.10e9), "lookup_64k": (65536, 0.03e9)}
    assert b.SETS["lookup_16k"]["facts"] == (4, 12) and b.SETS["lookup_64k"]["facts"] == (8, 24)


def test_command_line(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")     # main() sets it; restored after the test
    seen = {}
    monkeypatch.setattr(b, "run", lambda args: seen.update(vars(args)))
    b.main(["--out", str(tmp_path), "--only", "lookup_16k,lookup_64k", "--scale", "0.002", "--seed", "3"])
    assert seen == {"out": str(tmp_path), "only": "lookup_16k,lookup_64k", "scale": 0.002, "seed": 3,
                    "tokenizer": "HuggingFaceTB/SmolLM2-135M", "val_rows": 64, "shard_tokens": 100_000_000}
    assert __import__("os").environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"   # anonymous: no stored token is sent


def test_shard_writer_and_summary(tmp_path):
    w = b.ShardWriter(tmp_path, "x_train", 8, shard_tokens=20)   # 2 rows per shard
    rows = [np.full(8, k, dtype=np.uint16) for k in range(5)]
    for r in rows:
        w.add(r)
    assert w.close() == ["x_train_000000.bin", "x_train_000001.bin", "x_train_000002.bin"]
    assert [len(btd.open_shard(str(tmp_path / f))) for f in w.files] == [16, 16, 8]
    assert (np.concatenate([shard_rows(tmp_path / f, 8) for f in w.files]) == np.stack(rows)).all()
    with pytest.raises(ValueError):
        w.add(np.zeros(7, dtype=np.uint16))
    s = b.summarize([100, 1999, 2000, 5000, 40000])
    assert s["share_ge_2000"] == 0.6 and s["histogram"]["<500"] == 1 and s["histogram"]["2000-4000"] == 1
    assert s["histogram"]["1000-2000"] == 1 and s["histogram"][">=32000"] == 1 and b.summarize([]) == {"n": 0}


# ----------------------------------------------------------------------------------------------------------------
# Hub access (no network: the range reader is replaced)
# ----------------------------------------------------------------------------------------------------------------
def test_retry_backs_off_and_gives_up():
    waits, calls = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("503")
        return "ok"

    assert b.retry(flaky, "x", sleep=waits.append) == "ok" and waits == [2.0, 4.0]
    with pytest.raises(OSError):
        b.retry(lambda: (_ for _ in ()).throw(OSError("down")), "x", tries=3, sleep=waits.append)
    assert waits == [2.0, 4.0, 2.0, 4.0]
    with pytest.raises(KeyError):                                   # not a network error: no retry
        b.retry(lambda: {}["x"], "x", sleep=waits.append)
    assert len(waits) == 4


def test_ranged_reads_resume_and_stop_early(tmp_path, monkeypatch):
    rows = [{"id": str(i), "metadata": {"license": "Public Domain", "title": f"T{i}"}, "text": "x" * 3000 + str(i)}
            for i in range(400)]
    half = "\n".join(json.dumps(r) for r in rows[:200]) + "\n"
    rest = "\n".join(json.dumps(r) for r in rows[200:]) + "\n"
    blob = gzip.compress(half.encode(), 1) + gzip.compress(rest.encode(), 1)   # two gzip members
    asked = []

    def fake_range(url, start, end):
        asked.append((start, end))
        return blob[start:end]

    monkeypatch.setattr(b, "http_range", fake_range)
    monkeypatch.setattr(b.HubPrefix, "CHUNK", 1024)
    p = b.HubPrefix("u", tmp_path / "raw" / "f.json.gz", len(blob))
    first = list(itertools.islice(b.gz_json_lines(p.chunks()), 5))
    assert [r["id"] for r in first] == ["0", "1", "2", "3", "4"]
    assert 0 < p.have() < len(blob) and asked == [(k, k + 1024) for k in range(0, p.have(), 1024)]
    got = list(b.gz_json_lines(b.HubPrefix("u", tmp_path / "raw" / "f.json.gz", len(blob)).chunks()))
    assert got == rows and asked[0] == (0, 1024)
    assert [a for a, _ in asked] == sorted({a for a, _ in asked})   # no byte was downloaded twice
    n = len(asked)
    assert list(b.gz_json_lines(p.chunks())) == rows and len(asked) == n and p.fetch_all().read_bytes() == blob

    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(wiki_records(600)), buf, row_group_size=50, compression="none")
    blob, asked[:] = buf.getvalue(), []
    f = b.HubRangeFile("u", len(blob), block=256)
    pf = pq.ParquetFile(f)
    assert pf.metadata.num_row_groups == 12
    assert [r["id"] for r in pf.read_row_group(7).to_pylist()] == [f"q{i}" for i in range(350, 400)]
    assert f.fetched == sum(e - s for s, e in asked) < len(blob) / 3   # the footer and one row group, not the file


def test_wiki_source_reads_the_hub_in_ranges_or_whole(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    recs, blobs = wiki_records(120), {}
    for k, name in enumerate(b.WIKI_TRAIN_FILES):
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(recs[k * 60: (k + 1) * 60]), buf, row_group_size=5, compression="none")
        blobs[name] = buf.getvalue()
    asked = []

    def fake_range(url, start, end):
        asked.append(end - start)
        return blobs[url][start:end]

    monkeypatch.setattr(b, "http_range", fake_range)
    monkeypatch.setattr(b.HubRangeFile, "BLOCK", 512)
    monkeypatch.setattr(b, "hub_file_url", lambda repo, name, revision: name)
    monkeypatch.setattr(b, "hub_info", lambda repo: {"sha": "abc123", "license_tag": None,
                                                    "sizes": {n: len(v) for n, v in blobs.items()}})
    total = sum(len(v) for v in blobs.values())
    src = b.WikiSource(tmp_path / "_raw", 0, n_val_groups=4)
    assert src.n_train_groups == 20 and len(src.val_items()) == 20 and len(src.train_items(2)) == 10
    d = src.describe()
    assert sum(len(f["row_groups_read"]) for f in d["files"]) == 6
    assert sum(f["bytes_fetched"] for f in d["files"]) == sum(asked) < total     # a few row groups, not the files
    assert d["revision"] == "abc123" and not (tmp_path / "_raw").exists()        # nothing is kept on disk
    whole = b.WikiSource(tmp_path / "_raw", 0, whole_files=True, n_val_groups=4)
    assert [it.id for it in whole.val_items()] == [it.id for it in src.val_items()]
    assert [it.id for it in whole.train_items(20)] == [it.id for it in src.train_items(20)]
    assert sorted(p.stat().st_size for p in (tmp_path / "_raw").rglob("*.parquet")) == sorted(map(len, blobs.values()))
    assert [f["bytes_fetched"] for f in whole.describe()["files"]] == [len(blobs[n]) for n in b.WIKI_TRAIN_FILES]
    whole.cleanup()
    assert not list((tmp_path / "_raw").rglob("*.parquet"))                      # the raw files go when they are done


# ----------------------------------------------------------------------------------------------------------------
# The real tokenizer (skipped when it is not in the local cache)
# ----------------------------------------------------------------------------------------------------------------
def test_real_tokenizer_multihop_16k():
    tok, S = real_tok(), 16384
    assert tok.eot == 0 and tok.vocab == 49152
    rows = _library(tok, S, wiki_items(260, sentences=(1, 5)), 3, passes=1)
    assert len(rows) == 3
    dist = []
    for row in rows:
        check_row(row, tok, S)
        assert row.n >= 0.99 * S
        ctx, qs = blocks(row.text)
        assert len(qs) == len(row.qa) > 20
        assert all(a in ctx for _, _, a in qs if a not in ("yes", "no"))
        dist += [q["distance"] for q in row.qa]
    assert np.mean(np.asarray(dist) >= 2000) > 0.9


def test_real_tokenizer_lookup_16k():
    tok, cfg = real_tok(), b.SETS["lookup_16k"]
    book = book_text(11, 45000)
    rows = _lookup(tok, cfg, book)
    assert len(rows) == 3
    for row in rows:
        check_lookup_row(row, tok, cfg, book)
        assert row.n >= 0.99 * cfg["seq_len"]
    cfg64 = b.SETS["lookup_64k"]
    assert _lookup(tok, cfg64, book) == []                          # under 65,536 tokens: not used at 64k
    long = _lookup(tok, cfg64, book_text(12, 70000))
    assert len(long) == 1 and 8 <= long[0].info["facts"] <= 24
    check_lookup_row(long[0], tok, cfg64, book_text(12, 70000))
