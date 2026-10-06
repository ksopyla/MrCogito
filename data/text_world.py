"""Text capability checks — the "world document" generator (draft, `text-world-v0`).

Spec: docs/engineering_specs/text_capability_checks.md (§8–§10).

A world document is simple-language text about a small invented cast: real stories (the
filler, with their characters renamed to cast members) with fact sentences woven in, and a
question about those facts at the end:

    ... Once upon a time Lumo found a red ball ... Lumo lived in Tavimere. ...
    ... Later that spring, Lumo moved to Kodaford. ...

    Question: Where does Lumo live now?
    Answer: Kodaford.

Everything is a pure function of the integer seed given to `make_item` plus the filler pool,
so the same inputs give the same bytes on any machine.

Tasks (levels T1–T7 of the spec): quote, lookup, keyed, latest, compose, count, deduce. Each
item carries its answer, the candidate answers planted in the document, its guessing floor,
and an evidence-removed twin (the same document with the sentences that answer the question
replaced by filler): a model that does the task must drop to the floor on the twin.

Text-level only: no tokenizer here. Lengths are measured with a caller-supplied `ntok`
(token counter) so the builder can target exact token lengths with the frozen tokenizer.
"""
from __future__ import annotations

import random
import re
import zlib
from dataclasses import dataclass, field
from typing import Callable, Optional

TEXT_WORLD_VERSION = "text-world-v0"

# --------------------------------------------------------------------------------------------
# Vocabulary of the invented world
# --------------------------------------------------------------------------------------------
# Name syllables: consonant-vowel pairs, so look-alike names differ by exactly one syllable.
_SYLLABLES = [
    "lu", "mo", "ta", "vi", "ri", "ka", "pe", "no", "sa", "mi", "do", "ve", "ra", "ti", "bo",
    "ze", "fa", "ku", "li", "ne", "go", "ha", "ju", "wo", "be", "si", "pu", "ro", "de", "ma",
]
_PLACE_SUFFIXES = ["ford", "hill", "dale", "wick", "moor", "bury", "ton", "mere"]
ANIMALS = ["cat", "dog", "fox", "owl", "frog", "duck", "goat", "hen", "pig", "cow", "bee",
           "mouse", "horse", "sheep", "bird", "fish"]
JOBS = ["baker", "farmer", "teacher", "doctor", "painter", "singer", "cook", "driver", "builder",
        "fisher", "gardener", "nurse", "tailor", "potter", "miller", "sailor"]
SIGN_WORDS = ["blue", "frogs", "sing", "at", "noon", "big", "red", "boats", "sleep", "under",
              "the", "moon", "happy", "cats", "jump", "over", "green", "hills", "little", "bears",
              "dance", "in", "rain", "old", "trees", "talk", "to", "stars", "warm", "bread"]
VISIT_PLACES = ["the market", "the river", "the library", "the bakery", "the pond", "the mill"]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six"]
PROPERTIES = ["shiny", "soft", "loud", "cold", "tiny", "sweet", "fast", "brave"]

# Character names that TinyStories / SimpleStories use for their protagonists; occurrences in
# filler stories are renamed to cast members, so a name alone never locates a fact.
COMMON_STORY_NAMES = {
    "Lily", "Tim", "Tom", "Sue", "Ben", "Max", "Mia", "Sam", "Anna", "Timmy", "Lucy", "Sara",
    "Sarah", "Jack", "Bob", "Mary", "Jane", "Amy", "Kim", "Joe", "Zoe", "Mom", "Dad", "Billy",
    "Molly", "Jill", "Spot", "Rex", "Emma", "Leo", "Ella", "Mila", "Nina", "Kai", "Ava", "Eli",
    "Maya", "Finn", "Luna", "Oliver", "Ollie", "Rosie", "Daisy", "Bella", "Charlie", "Lucas",
}

# Words that state a reserved relation. A filler sentence that mentions a cast member together
# with one of these is dropped, so filler never states (or contradicts) a world fact.
_RESERVED = re.compile(
    r"\b(live[sd]?|living|home|moved?|moving|sister|brother|sibling|friends?|work(?:s|ed)?|job|"
    r"sign|visit(?:s|ed)?|pet|called|named|market|river|library|bakery|pond|mill|every|is a|"
    r"how many|question|answer)\b",
    re.IGNORECASE,
)

TASKS = ("quote", "lookup", "keyed", "latest", "compose", "count", "deduce")
LEVEL_OF = {"quote": "T1", "lookup": "T2", "keyed": "T3", "latest": "T4", "compose": "T5",
            "count": "T6", "deduce": "T7"}

# In-distribution dials (training and the ID eval split) and the "harder" split (spec §8.5).
ID_DIALS = {"quote": {"signs": (2, 8)}, "lookup": {"cast": (2, 6)}, "keyed": {"cast": (4, 16)},
            "latest": {"moves": (1, 4), "cast": (3, 6)}, "compose": {"hops": (1, 2), "cast": (4, 8)},
            "count": {"cast": (3, 6)}, "deduce": {"depth": (1, 2)}}
EVAL_DIALS = {"quote": {"signs": 4}, "lookup": {"cast": 4}, "keyed": {"cast": 16},
              "latest": {"moves": 4, "cast": 5}, "compose": {"hops": 2, "cast": 8},
              "count": {"cast": 4}, "deduce": {"depth": 2}}
HARDER_DIALS = {"quote": {"signs": 16}, "lookup": {"cast": 4}, "keyed": {"cast": 64},
                "latest": {"moves": 8, "cast": 8}, "compose": {"hops": 3, "cast": 8},
                "count": {"cast": 8}, "deduce": {"depth": 3}}

# Sentence templates per relation. The last template of each relation is held out of training
# (the "paraphrase" split); the others are used for training and the ID split.
TEMPLATES = {
    "home": ["{p} lived in {v}.", "{v} was where {p} made a home.", "{p} had a little house in {v}.",
             "The town of {v} was home to {p}.", "{p} lived in a small house in the town of {v}."],
    "move": ["Later, {p} moved to {v}.", "After a while, {p} moved to {v}.",
             "One spring, {p} packed a bag and moved to {v}.", "Then {p} moved away to {v}.",
             "Soon after, {p} went to live in {v}."],
    "sibling": ["{q} was the sister of {p}.", "{p} had a sister called {q}.",
                "{p} and {p2} sister {q} were very close.", "{q} was {p2} sister.",
                "{p} grew up with a sister named {q}."],
    "pet": ["{p} had a {a} called {v}.", "{p} kept a {a} named {v}.",
            "{v} was the name of the {a} that {p} had.", "{p} loved a {a} called {v}.",
            "The {a} that lived with {p} was called {v}."],
    "job": ["{p} worked as a {v}.", "{p} was a {v} in the town.", "Every day {p} worked as a {v}.",
            "{p} had a job as a {v}.", "The town knew {p} as a good {v}."],
    "sign": ["The sign on {p2} door said: {v}.", "On the door of {p} there was a sign that said: {v}.",
             "{p} put up a sign that said: {v}.", "A sign by {p2} gate said: {v}.",
             "{p} painted a sign that said: {v}."],
    "visit": ["{p} visited {v}.", "That day {p} visited {v}.", "{p} went for a visit to {v}.",
              "In the morning {p} visited {v}.", "{p} paid a visit to {v}."],
}

QUESTIONS = {
    "quote": "What did the sign of {p} say?",
    "lookup": "Where does {p} live?",
    "keyed": "Where does {p} live?",
    "latest": "Where does {p} live now?",
    "compose1": "Where does the sister of {p} live?",
    "compose2": "Where does the sister of the sister of {p} live?",
    "compose3": "Where does the sister of the sister of the sister of {p} live?",
    "count": "How many times did {p} visit {v}?",
    "deduce": "Is {p} {v}?",
}


def _poss(name: str) -> str:
    return name + "'s"


# --------------------------------------------------------------------------------------------
# Names and places (deterministic; held-out names are a fixed hash class)
# --------------------------------------------------------------------------------------------
def is_heldout_name(name: str) -> bool:
    """1 in 5 names is reserved for evaluation (never appears in training documents)."""
    return zlib.crc32(name.lower().encode()) % 5 == 0


def _make_name(rng: random.Random, n_syll: int = 2) -> str:
    return "".join(rng.choice(_SYLLABLES) for _ in range(n_syll)).capitalize()


def sample_names(rng: random.Random, n: int, split: str, lookalike_frac: float = 0.0,
                 taken: Optional[set] = None) -> list[str]:
    """n distinct person names of the split; `lookalike_frac` of them differ from another cast
    member by one syllable (e.g. Lumo / Lumi)."""
    want_heldout = split != "train"
    taken = set() if taken is None else taken
    out: list[str] = []
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 100000:
            raise RuntimeError("name pool exhausted")
        if out and rng.random() < lookalike_frac:
            base = rng.choice(out).lower()
            sylls = [base[i:i + 2] for i in range(0, len(base), 2)]
            sylls[rng.randrange(len(sylls))] = rng.choice(_SYLLABLES)
            name = "".join(sylls).capitalize()
        else:
            name = _make_name(rng, rng.choice((2, 2, 3)))
        if name in taken or is_heldout_name(name) != want_heldout:
            continue
        taken.add(name)
        out.append(name)
    return out


def sample_places(rng: random.Random, n: int, taken: Optional[set] = None) -> list[str]:
    taken = set() if taken is None else taken
    out: list[str] = []
    while len(out) < n:
        place = (_make_name(rng, 2) + rng.choice(_PLACE_SUFFIXES)).capitalize()
        if place not in taken:
            taken.add(place)
            out.append(place)
    return out


# --------------------------------------------------------------------------------------------
# Filler: stories renamed to cast members, reserved-relation sentences dropped
# --------------------------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[A-Z][a-z]+\b")


def clean_story(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def renamed_filler(story: str, cast: list[str], rng: random.Random) -> list[str]:
    """Rename the story's common character names to cast members and return its sentences,
    minus any sentence that mentions a cast member with a reserved relation word."""
    names = sorted({w for w in _WORD.findall(story) if w in COMMON_STORY_NAMES})
    mapping = {n: rng.choice(cast) for n in names} if cast else {}
    text = story
    for old, new in mapping.items():
        text = re.sub(rf"\b{old}\b", new, text)
    cast_set = set(cast)
    out = []
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if not s:
            continue
        if _RESERVED.search(s) and (cast_set & set(_WORD.findall(s)) or "?" in s):
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------------
@dataclass
class Fact:
    text: str
    evidence: bool = False      # answers the question; removed in the twin
    order: Optional[float] = None  # fixed relative position in [0,1] (ordered events), else free


@dataclass
class WorldItem:
    task: str
    question: str
    answer: str                 # without the closing period
    candidates: list[str]       # answer values planted in the document (gold first)
    floor: float                # guessing floor of the picked-candidate / answer score
    facts: list[Fact]
    cast: list[str]
    meta: dict = field(default_factory=dict)


def _tmpl(rng: random.Random, rel: str, split: str, **kw) -> str:
    pool = TEMPLATES[rel]
    if split == "paraphrase":
        t = pool[-1]
    else:
        t = rng.choice(pool[:-1])
    p = kw.get("p", "")
    return t.format(p2=_poss(p), **kw)


def _dial(task: str, split: str, rng: random.Random, key: str) -> int:
    if split == "harder":
        return HARDER_DIALS[task][key]
    if split == "train":
        lo, hi = ID_DIALS[task][key]
        return rng.randint(lo, hi)
    return EVAL_DIALS[task][key]


def build_item(task: str, rng: random.Random, split: str = "train") -> WorldItem:
    """One question about a fresh world. `split` ∈ train | id | harder | paraphrase."""
    tsplit = "paraphrase" if split == "paraphrase" else ("train" if split == "train" else "id")
    names_split = "train" if split == "train" else "eval"
    if task == "quote":
        n = _dial(task, split, rng, "signs")
        cast = sample_names(rng, n, names_split)
        signs = []
        for _ in range(n):
            while True:
                s = " ".join(rng.sample(SIGN_WORDS, rng.randint(4, 6)))
                if s not in signs:
                    signs.append(s)
                    break
        facts = [Fact(_tmpl(rng, "sign", tsplit, p=c, v=s), evidence=(i == 0)) for i, (c, s) in enumerate(zip(cast, signs))]
        return WorldItem(task, QUESTIONS["quote"].format(p=cast[0]), signs[0], signs, 1.0 / n, facts, cast)

    if task in ("lookup", "keyed"):
        n = _dial(task, split, rng, "cast")
        cast = sample_names(rng, n, names_split, lookalike_frac=0.25 if task == "keyed" else 0.0)
        places = sample_places(rng, n)
        if task == "lookup":
            # only the asked person's home is stated; the others get unrelated facts (jobs)
            facts = [Fact(_tmpl(rng, "home", tsplit, p=cast[0], v=places[0]), evidence=True)]
            facts += [Fact(_tmpl(rng, "job", tsplit, p=c, v=rng.choice(JOBS))) for c in cast[1:]]
            return WorldItem(task, QUESTIONS["lookup"].format(p=cast[0]), places[0], places[:1], 0.0, facts, cast)
        facts = [Fact(_tmpl(rng, "home", tsplit, p=c, v=v), evidence=(i == 0)) for i, (c, v) in enumerate(zip(cast, places))]
        return WorldItem(task, QUESTIONS["keyed"].format(p=cast[0]), places[0], places, 1.0 / n, facts, cast)

    if task == "latest":
        k = _dial(task, split, rng, "moves")
        n = _dial(task, split, rng, "cast")
        cast = sample_names(rng, n, names_split)
        places = sample_places(rng, k + 1 + 2 * (n - 1))
        target = places[: k + 1]
        facts = [Fact(_tmpl(rng, "home", tsplit, p=cast[0], v=target[0]), evidence=True, order=0.02)]
        # the target's moves sit in the first 80 % of the document, in order …
        pos = sorted(rng.uniform(0.05, 0.8) for _ in range(k))
        facts += [Fact(_tmpl(rng, "move", tsplit, p=cast[0], v=v), evidence=True, order=o) for v, o in zip(target[1:], pos)]
        # … and every other person moves after the target's last move, so "the most recently
        # mentioned place" is never the answer.
        rest = places[k + 1:]
        for i, c in enumerate(cast[1:]):
            facts.append(Fact(_tmpl(rng, "home", tsplit, p=c, v=rest[2 * i]), order=rng.uniform(0.0, 0.5)))
            facts.append(Fact(_tmpl(rng, "move", tsplit, p=c, v=rest[2 * i + 1]), order=rng.uniform(pos[-1] + 0.01, 0.99)))
        return WorldItem(task, QUESTIONS["latest"].format(p=cast[0]), target[-1], list(reversed(target)),
                         1.0 / (k + 1), facts, cast, meta={"moves": k})

    if task == "compose":
        hops = _dial(task, split, rng, "hops")
        n = max(_dial(task, split, rng, "cast"), hops + 2)
        cast = sample_names(rng, n, names_split)
        places = sample_places(rng, n)
        # every person has a sister (a random permutation without fixed points) and a home:
        # the asked chain is one of n same-shaped chains.
        perm = list(range(n))
        while any(i == p for i, p in enumerate(perm)):
            rng.shuffle(perm)
        cur, chain = 0, [0]
        for _ in range(hops):
            cur = perm[cur]
            chain.append(cur)
        evidence_sib = {(chain[i], chain[i + 1]) for i in range(hops)}
        facts = []
        for i in range(n):
            facts.append(Fact(_tmpl(rng, "sibling", tsplit, p=cast[i], q=cast[perm[i]]), evidence=(i, perm[i]) in evidence_sib))
            facts.append(Fact(_tmpl(rng, "home", tsplit, p=cast[i], v=places[i]), evidence=(i == chain[-1])))
        ans = places[chain[-1]]
        cands = [ans] + [p for j, p in enumerate(places) if j != chain[-1]]
        return WorldItem(task, QUESTIONS[f"compose{hops}"].format(p=cast[0]), ans, cands, 1.0 / n, facts, cast,
                         meta={"hops": hops})

    if task == "count":
        n = _dial(task, split, rng, "cast")
        cast = sample_names(rng, n, names_split, lookalike_frac=0.3)
        where = rng.choice(VISIT_PLACES)
        k = rng.randrange(len(NUMBER_WORDS))  # balanced answers 0..6
        facts = [Fact(_tmpl(rng, "visit", tsplit, p=cast[0], v=where), evidence=True) for _ in range(k)]
        others = [p for p in VISIT_PLACES if p != where]
        facts += [Fact(_tmpl(rng, "visit", tsplit, p=cast[0], v=rng.choice(others))) for _ in range(rng.randint(1, 3))]
        for c in cast[1:]:
            for _ in range(rng.randint(0, 3)):
                facts.append(Fact(_tmpl(rng, "visit", tsplit, p=c, v=rng.choice(VISIT_PLACES))))
        return WorldItem(task, QUESTIONS["count"].format(p=cast[0], v=where), NUMBER_WORDS[k], list(NUMBER_WORDS),
                         1.0 / len(NUMBER_WORDS), facts, cast, meta={"count": k})

    if task == "deduce":
        depth = _dial(task, split, rng, "depth")
        cast = sample_names(rng, 2, names_split)
        cats = [w.lower() for w in sample_names(rng, 2 * depth + 2, "train" if split == "train" else "eval")]
        prop = rng.choice(PROPERTIES)
        yes = rng.random() < 0.5
        chain = cats[:depth]
        facts = [Fact(f"{cast[0]} is a {chain[0]}.", evidence=True)]
        facts += [Fact(f"Every {a} is a {b}.", evidence=True) for a, b in zip(chain, chain[1:])]
        facts.append(Fact(f"Every {chain[-1]} is {'' if yes else 'not '}{prop}.", evidence=True))
        # distractor chain with the opposite conclusion about the other person
        other = cats[depth:2 * depth]
        facts.append(Fact(f"{cast[1]} is a {other[0]}."))
        facts += [Fact(f"Every {a} is a {b}.") for a, b in zip(other, other[1:])]
        facts.append(Fact(f"Every {other[-1]} is {'not ' if yes else ''}{prop}."))
        return WorldItem(task, QUESTIONS["deduce"].format(p=cast[0], v=prop), "yes" if yes else "no",
                         ["yes", "no"] if yes else ["no", "yes"], 0.5, facts, cast, meta={"depth": depth})

    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


# --------------------------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------------------------
def qa_suffix(question: str) -> str:
    return f"\n\nQuestion: {question}\nAnswer:"


def assemble(item: WorldItem, stories: list[str], target_tokens: int, ntok: Callable[[str], int],
             rng: random.Random, depth: Optional[float] = None, remove_evidence: bool = False,
             filler_seed: Optional[int] = None) -> tuple[str, dict]:
    """Return (prompt text ending in 'Answer:', info). The prompt plus ' <answer>.' is about
    `target_tokens` tokens. `depth` ∈ [0,1] places the single evidence fact (tasks whose
    evidence is one sentence); `remove_evidence` swaps evidence sentences for filler sentences.

    `filler_seed` fixes the filler draw so an item and its evidence-removed twin share the
    same filler stories in the same places."""
    frng = random.Random(filler_seed if filler_seed is not None else rng.random())
    suffix = qa_suffix(item.question)
    budget = target_tokens - ntok(suffix) - ntok(" " + item.answer + ".") - 1
    facts = list(item.facts)
    fact_tok = sum(ntok(" " + f.text) for f in facts)
    # filler sentences (renamed stories), drawn until the budget is filled
    filler: list[str] = []
    used = fact_tok
    spare: list[str] = []
    guard = 0
    while used < budget and guard < 100000:
        guard += 1
        sents = renamed_filler(frng.choice(stories), item.cast, frng)
        for s in sents:
            t = ntok(" " + s)
            if used + t > budget:
                spare.append(s)
                continue
            filler.append(s)
            used += t
        if used >= budget - 4:
            break
    # positions: each fact gets a relative position in [0,1]
    n_slots = len(filler) + 1
    evid = [f for f in facts if f.evidence]
    single = len(evid) == 1 and depth is not None
    placed: list[tuple[float, int, str]] = []
    for i, f in enumerate(facts):
        if f.evidence and single:
            pos = depth
        elif f.order is not None:
            pos = f.order
        else:
            pos = rng.random()
        text = f.text
        if f.evidence and remove_evidence:
            # same length budget: swap in a filler sentence (from the spare pool if possible)
            text = spare.pop() if spare else frng.choice(filler) if filler else ""
        placed.append((pos, i, text))
    placed.sort()
    # interleave facts into the filler at their relative positions
    out: list[str] = []
    fi = 0
    for k in range(n_slots):
        frac = k / max(1, n_slots - 1)
        while fi < len(placed) and placed[fi][0] <= frac:
            if placed[fi][2]:
                out.append(placed[fi][2])
            fi += 1
        if k < len(filler):
            out.append(filler[k])
    out.extend(p[2] for p in placed[fi:] if p[2])
    body = " ".join(out)
    info = {"n_filler_sentences": len(filler), "n_facts": len(facts), "evidence_removed": remove_evidence}
    return body + suffix, info


def answer_text(item: WorldItem) -> str:
    """The scored continuation after 'Answer:' (leading space, closing period)."""
    return f" {item.answer}."


def make_eval_record(task: str, seed: int, split: str, length: int, depth_bin: Optional[str],
                     stories: list[str], ntok: Callable[[str], int]) -> dict:
    """A frozen evaluation record: prompt, answer, candidates, floor and the evidence-removed twin."""
    rng = random.Random(seed)
    item = build_item(task, rng, split)
    depth = {"early": 0.05, "middle": 0.5, "late": 0.95}.get(depth_bin) if depth_bin else None
    fseed = rng.randrange(2**31)
    place_seed = rng.randrange(2**31)
    prompt, info = assemble(item, stories, length, ntok, random.Random(place_seed), depth=depth, filler_seed=fseed)
    twin, _ = assemble(item, stories, length, ntok, random.Random(place_seed), depth=depth, remove_evidence=True,
                       filler_seed=fseed)
    return {
        "id": f"{task}-{split}-{length}-{depth_bin or 'spread'}-{seed}",
        "version": TEXT_WORLD_VERSION,
        "task": task, "level": LEVEL_OF[task], "split": split, "length": length,
        "depth": depth_bin or "spread", "seed": seed,
        "prompt": prompt, "prompt_removed": twin, "answer": answer_text(item),
        "candidates": [f" {c}." for c in item.candidates], "floor": item.floor,
        "meta": {**item.meta, **info},
    }


def make_train_document(seed: int, stories: list[str], length: int, ntok: Callable[[str], int],
                        tasks: tuple = TASKS) -> str:
    """One training world document: a random task at ID dials, question and answer included."""
    rng = random.Random(seed)
    task = rng.choice(tasks)
    item = build_item(task, rng, "train")
    prompt, _ = assemble(item, stories, length, ntok, rng)
    return prompt + answer_text(item)
