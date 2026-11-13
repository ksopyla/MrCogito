"""Real multi-hop QA and maths written as reasoning chains (E33s, 2026-10-21).

E33s trains on real data whose answers write their reasoning out:
    "\\n\\nQuestion: <q>\\nAnswer: <step>. <step>. So <answer>."
A rule then reads, off that text alone, what each loop round should name and where in the context it should read
(the per-round targets of `PerceiverARLM`: `message_loop_name_lens`, `message_loop_read_aux`). The written steps are
removed in three stages (iCoT): all → the first dropped → none.

The datasets' own annotations (supporting paragraphs and sentences, MuSiQue's per-hop sub-answers, GSM8K-Aug's
equations) are used only to WRITE the chain text, as a reasoning dataset would. No target comes from a generator.

Spec: docs/experiments_specs/ahead/E33s_written_chains_real_data.md · plan: E33s_written_chains_real_data_plan.md.
"""
from __future__ import annotations

import bisect
import random
import re
from dataclasses import dataclass, field

DATASETS = {
    # name: (HF repo, config, train split, eval split)
    "musique": ("dgslibisey/MuSiQue", None, "train", "validation"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "train", "validation"),
    "2wiki": ("framolfese/2WikiMultihopQA", None, "train", "validation"),
    "gsm8k": ("whynlp/gsm8k-aug", None, "train", "test"),
}

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on", "at", "to", "for", "from", "by",
    "with", "and", "or", "as", "that", "this", "it", "its", "his", "her", "their", "he", "she", "they", "s", "so",
    "who", "what", "which", "when", "where", "also", "has", "have", "had", "not", "but", "into", "than", "one",
}
_WORD = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class QAItem:
    dataset: str
    id: str
    question: str
    answer: str
    aliases: list = field(default_factory=list)
    paragraphs: list = field(default_factory=list)  # [(title, text, supporting)]
    steps: list = field(default_factory=list)       # the chain the dataset writes: sentences or equations
    kind: str = "multihop"                          # multihop | math
    qtype: str = ""                                 # bridge | comparison | … where the dataset says
    meta: dict = field(default_factory=dict)        # e.g. MuSiQue's per-hop sub-answers (for the rule check only)
    step_titles: list = field(default_factory=list) # the title of each step's paragraph (shown above it in the text)


# ------------------------------------------------------------------ loaders (HF records → QAItem)
def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def _find_ci(hay: str, needle: str) -> int:
    return hay.lower().find(needle.lower()) if needle else -1


def from_musique(rec: dict) -> QAItem | None:
    """Per hop: the sentence of its supporting paragraph that contains the hop's sub-answer."""
    paras = [(p["title"], p["paragraph_text"], bool(p["is_supporting"])) for p in rec["paragraphs"]]
    steps, titles = [], []
    for hop in rec["question_decomposition"]:
        idx = hop["paragraph_support_idx"]
        if idx is None or not (0 <= idx < len(paras)):
            return None
        sent = next((s for s in sentences(paras[idx][1]) if _find_ci(s, hop["answer"]) >= 0), None)
        if sent is None:
            return None
        steps.append(sent)
        titles.append(paras[idx][0])
    return QAItem("musique", rec["id"], rec["question"], rec["answer"], list(rec.get("answer_aliases") or []),
                  paras, steps, qtype=f"{len(steps)}hop",
                  meta={"sub_answers": [h["answer"] for h in rec["question_decomposition"]]}, step_titles=titles)


def _sf_sentences(context: dict, sf: dict) -> tuple[list, list[str], list[str]] | None:
    titles, sents = list(context["title"]), list(context["sentences"])
    sup_titles = set(sf["title"])
    paras = [(t, " ".join(s.strip() for s in ss), t in sup_titles) for t, ss in zip(titles, sents)]
    steps, step_titles = [], []
    for t, i in zip(sf["title"], sf["sent_id"]):
        if t not in titles:
            return None
        ss = sents[titles.index(t)]
        if not (0 <= i < len(ss)):
            return None
        steps.append(ss[i].strip())
        step_titles.append(t)
    return paras, steps, step_titles


def from_hotpot(rec: dict, dataset: str = "hotpotqa") -> QAItem | None:
    """HotpotQA / 2WikiMultihopQA (HotpotQA layout): the supporting-fact sentences, verbatim."""
    got = _sf_sentences(rec["context"], rec["supporting_facts"])
    if got is None:
        return None
    paras, steps, titles = got
    return QAItem(dataset, str(rec["id"]), rec["question"], rec["answer"], [], paras, steps,
                  qtype=str(rec.get("type", "")), step_titles=titles)


_EQ = re.compile(r"<<(.*?)>>")


def from_gsm8k(rec: dict, i: int) -> QAItem | None:
    steps = [m.strip() for s in rec["steps"] for m in (_EQ.findall(s) or [s]) if m.strip()]
    if not steps:
        return None
    return QAItem("gsm8k", f"gsm8k-{i}", rec["question"], str(rec["answer"]).strip(), [], [], steps, kind="math")


def load_items(name: str, split: str, limit: int | None = None) -> list[QAItem]:
    """Load one dataset split as QAItems (records the chain writer cannot place are dropped and counted)."""
    from datasets import load_dataset

    repo, cfg, tr, ev = DATASETS[name]
    ds = load_dataset(repo, cfg, split={"train": tr, "eval": ev}.get(split, split))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for i, rec in enumerate(ds):
        if name == "musique":
            if not rec.get("answerable", True):
                continue
            it = from_musique(rec)
        elif name in ("hotpotqa", "2wiki"):
            it = from_hotpot(rec, name)
        else:
            it = from_gsm8k(rec, i)
        if it is not None:
            out.append(it)
    return out


# ------------------------------------------------------------------ the rule (text only)
def _words(s: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(s)]


def _shared_run(a: str, b: str, qwords: set, body_low: str, max_n: int = 6) -> str | None:
    """The run of up to `max_n` consecutive words of `a` that also appears in `b` (as whole words), trimmed of
    function words at both ends and holding a word that is neither a function word nor in the question. The best
    run is the most name-like (every content word capitalised or a number), then the most capitalised words, then
    the longest, then the rarest in the body."""
    wa = _words(a)
    b_low = " " + " ".join(w.lower() for w, _, _ in _words(b)) + " "
    best, best_key = None, None
    for n in range(1, min(max_n, len(wa)) + 1):
        for i in range(len(wa) - n + 1):
            run = wa[i:i + n]
            while run and run[0][0].lower() in STOP:
                run = run[1:]
            while run and run[-1][0].lower() in STOP:
                run = run[:-1]
            if not run:
                continue
            low = [w.lower() for w, _, _ in run]
            content = [w for w, _, _ in run if w.lower() not in STOP]
            if not any(w not in STOP and w not in qwords for w in low):
                continue
            if f" {' '.join(low)} " not in b_low:
                continue
            caps = sum(w[0].isupper() or w[0].isdigit() for w in content)
            text = a[run[0][1]:run[-1][2]]
            key = (caps == len(content), caps, len(run), -body_low.count(text.lower()))
            if best_key is None or key > best_key:
                best, best_key = text, key
    return best


def _number_after_eq(step: str) -> str | None:
    tail = step.rsplit("=", 1)[-1] if "=" in step else ""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", tail)
    return m.group(0) if m else None


def parse_chain(steps: list[str], answer: str, question: str, body: str, math: bool = False,
                titles: list[str] | None = None) -> list[tuple[str, str]]:
    """Read a written chain back into (step, output) pairs by a rule on the text alone.
    - Equations: a step's output is the number after its '='; the last step's is the answer.
    - Sentences: the step that contains the answer moves to the end (if any does); a step's output is the run of
      words it shares with the next step (`_shared_run`), the next step read together with the title of its
      paragraph when `titles` are given (a later fact often names its subject only by the title above it, or by a
      pronoun); the last step's output is the answer.
    Returns [] when a step has no output (e.g. comparison questions); the answer target alone then remains."""
    pairs = [(s.strip().rstrip("."), (titles[i] if titles and i < len(titles) else "")) for i, s in enumerate(steps)
             if s.strip()]
    steps = [s for s, _ in pairs]
    if not steps:
        return []
    if math:
        out = [(s, _number_after_eq(s)) for s in steps[:-1]] + [(steps[-1], answer.strip())]
        return out if all(o for _, o in out) else []
    ans = re.sub(r"^(a|an|the) ", "", answer.strip(), flags=re.I)
    with_ans = [i for i, s in enumerate(steps) if _find_ci(s, ans) >= 0]
    if with_ans:
        i = with_ans[-1]
        pairs = pairs[:i] + pairs[i + 1:] + [pairs[i]]
        steps = [s for s, _ in pairs]
    qwords = {w.lower() for w, _, _ in _words(question)}
    body_low = body.lower()
    out = []
    for r, s in enumerate(steps):
        if r == len(steps) - 1:
            word = s[_find_ci(s, ans):_find_ci(s, ans) + len(ans)] if _find_ci(s, ans) >= 0 else ans
        else:
            nxt = (pairs[r + 1][1] + " " + steps[r + 1]).strip()
            word = _shared_run(s, nxt, qwords, body_low)
            if not word:
                return []
        out.append((s, word))
    return out


# ------------------------------------------------------------------ documents and written answers
def qa_suffix(question: str) -> str:
    return f"\n\nQuestion: {question}\nAnswer:"


def written_answer(steps: list[str], answer: str, keep: str) -> str:
    """keep = all | drop1 | none (| answers): the text after 'Answer:'."""
    shown = {"all": steps, "drop1": steps[1:], "none": [], "answers": []}[keep]
    clean = [s.strip().rstrip(".") for s in shown]
    return (" " + ". ".join(clean) + f". So {answer}.") if clean else f" {answer}."


def _para_text(p) -> str:
    return f"{p[0]}\n{p[1]}"


def assemble(item: QAItem, target_tokens: int, ntok, rng: random.Random, pool: list, depth: float | None = None,
             remove_evidence: bool = False) -> str:
    """The context (supporting + own distractor paragraphs, padded with `pool` paragraphs to about
    `target_tokens`) followed by the question; ends with 'Answer:'. `depth` ∈ [0, 1] places the supporting block
    (exams: early / middle / late); `remove_evidence` swaps the supporting paragraphs for pool paragraphs."""
    if item.kind == "math":
        return f"Question: {item.question}\nAnswer:"
    sup = [p for p in item.paragraphs if p[2]]
    own = [p for p in item.paragraphs if not p[2]]
    if remove_evidence:
        sup = [rng.choice(pool) for _ in sup]
    budget = target_tokens - ntok(qa_suffix(item.question)) - 32
    used = sum(ntok(_para_text(p)) + 1 for p in sup)
    paras = []
    for p in rng.sample(own, len(own)):  # the question's own distractors first, as far as they fit
        n = ntok(_para_text(p)) + 1
        if used + n <= budget:
            paras.append(p)
            used += n
    seen = {p[1] for p in sup + paras}
    tries = 0
    while used < budget and pool and tries < 20 * (len(pool) + 1):
        tries += 1
        p = rng.choice(pool)
        if p[1] in seen:
            continue
        n = ntok(_para_text(p)) + 1
        if used + n > budget:
            break
        paras.append(p)
        seen.add(p[1])
        used += n
    rng.shuffle(paras)
    if depth is None:
        for p in sup:
            paras.insert(rng.randrange(len(paras) + 1), p)
    else:
        at = int(round(depth * len(paras)))
        paras[at:at] = sup
    return "\n\n".join(_para_text(p) for p in paras) + qa_suffix(item.question)


def chain_targets(text: str, offsets: list, question: str, derived: list[tuple[str, str]],
                  steps_in_answer: bool = False) -> list[int]:
    """Flat (answer position, round, first token, last token) quadruples in row positions (BOS = 0), the
    `read_targets` format of build_text_checks_data.py. Round 0 marks the answer position. Round r points at step
    r's output inside step r: in the context (sentence chains), or, for equations, in the written answer after the
    answer position (a naming target only while the steps are written; a read target there is causally masked)."""
    starts = [a for a, _ in offsets]

    def tok_of(c: int) -> int:
        return max(0, bisect.bisect_right(starts, c) - 1)

    block = qa_suffix(question)
    i = text.find(block)
    if i < 0:
        i = text.find(block.lstrip("\n"))
        if i < 0:
            return []
        block = block.lstrip("\n")
    end = i + len(block)
    pos = tok_of(end - 1) + 1
    out = [pos, 0, -1, -1]
    for r, (step, word) in enumerate(derived, 1):
        c0 = text.find(step, end) if steps_in_answer else text.find(step, 0, i)
        if c0 < 0:
            continue
        w = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text[c0:c0 + len(step)])
        if not w:
            continue
        a, b = c0 + w.start(), c0 + w.end()
        out += [pos, r, tok_of(a) + 1, tok_of(b - 1) + 1]
    return out


# ------------------------------------------------------------------ scoring (SQuAD normalisation)
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, golds: list[str]) -> float:
    from collections import Counter

    best = 0.0
    p = normalize_answer(pred).split()
    for g in golds:
        gt = normalize_answer(g).split()
        common = Counter(p) & Counter(gt)
        same = sum(common.values())
        if not p or not gt:
            f = float(p == gt)
        elif same == 0:
            f = 0.0
        else:
            prec, rec = same / len(p), same / len(gt)
            f = 2 * prec * rec / (prec + rec)
        best = max(best, f)
    return best
