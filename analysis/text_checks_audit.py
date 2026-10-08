#!/usr/bin/env python
"""Audit the text-check exam items with no model at all (spec §2 "shortcut audit", §10.3).

Three readers go through every frozen item:
  rule reader   knows the generator's sentence templates, rebuilds the world from the document and
                answers. It must score 100 %: any miss means the item is ill-posed (the filler states
                or contradicts a fact, a template is unparseable, two answers fit).
  shortcuts     cheap readers that skip the skill (ignore the asked name, take the only place word,
                count every visit, ...). Each must sit at the task's guessing floor; one that scores
                clearly above it marks the task as flawed for that skill.
  twin check    the evidence-removed twin must not contain the answer (else its score is not a floor).
Plus: candidate first-token collisions under the exam tokenizer (makes `pick` uninformative) and
filler contamination counts (filler sentences that state something about the asked person that the
question depends on).

  uv run python analysis/text_checks_audit.py --items Cache/text_checks/v0_hub/eval \
      --tokenizer Cache/text_checks/v0_hub/tokenizer --out docs/3_Evaluations_and_Baselines/text_audit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.text_world import NUMBER_WORDS, PROPERTIES, TEMPLATES, VISIT_PLACES  # noqa: E402

NAME = r"[A-Z][a-z]+"
_SENT = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'])\s+")  # facts may follow a closing quote: '…!" Lumo lived in X.'
_PLACE = re.compile(r"\b[A-Z][a-z]*(?:ford|hill|dale|wick|moor|bury|ton|mere)\b")
_VALUE = {"home": NAME, "move": NAME, "sign": r"[a-z ]+", "job": r"[a-z]+", "visit": r"the [a-z]+",
          "pet": NAME, "sibling": NAME}


def _compile(rel: str, tmpl: str) -> re.Pattern:
    out, seen = [], set()
    for part in re.split(r"(\{[a-z0-9]+\})", tmpl):
        if not part.startswith("{"):
            out.append(re.escape(part))
            continue
        f = part[1:-1]
        key = "p" if f in ("p", "p2") else f
        grp = NAME if key in ("p", "q") else (r"[a-z]+" if key == "a" else _VALUE[rel])
        out.append(f"(?P={key})" if key in seen else f"(?P<{key}>{grp})")
        seen.add(key)
        if f == "p2":
            out.append("'s")
    return re.compile("".join(out))


PATTERNS = {rel: [_compile(rel, t) for t in ts] for rel, ts in TEMPLATES.items()}
_IS_A = re.compile(rf"({NAME}) is a ([a-z]+)\.")
_EVERY_A = re.compile(r"Every ([a-z]+) is a ([a-z]+)\.")
_EVERY_P = re.compile(rf"Every ([a-z]+) is (not )?({'|'.join(PROPERTIES)})\.")
_Q = {
    "quote": re.compile(rf"What did the sign of ({NAME}) say\?"),
    "lookup": re.compile(rf"Where does ({NAME}) live\?"),
    "latest": re.compile(rf"Where does ({NAME}) live now\?"),
    "compose": re.compile(rf"Where does ((?:the sister of )+)({NAME}) live\?"),
    "count": re.compile(rf"How many times did ({NAME}) visit (the [a-z]+)\?"),
    "deduce": re.compile(rf"Is ({NAME}) ({'|'.join(PROPERTIES)})\?"),
}
_Q["keyed"] = _Q["lookup"]


def split_doc(prompt: str) -> tuple[list[str], str]:
    body, _, q = prompt.rpartition("\n\nQuestion: ")
    return [s.strip() for s in _SENT.split(body) if s.strip()], q.split("\nAnswer:")[0].strip()


def parse(sents: list[str]) -> list[tuple[int, str, dict]]:
    """Every sentence that fully matches a template: (index, relation, fields)."""
    out = []
    for i, s in enumerate(sents):
        for rel, pats in PATTERNS.items():
            m = next((m for m in (p.fullmatch(s) for p in pats) if m), None)
            if m:
                out.append((i, rel, m.groupdict()))
                break
    return out


def rule_reader(task: str, sents: list[str], q: str) -> str | None:
    facts = parse(sents)
    m = _Q[task].fullmatch(q)
    if not m:
        return "<unparsed question>"
    if task == "quote":
        hits = [f["v"] for _, r, f in facts if r == "sign" and f["p"] == m[1]]
        return hits[0] if len(hits) == 1 else None
    if task in ("lookup", "keyed"):
        hits = {f["v"] for _, r, f in facts if r == "home" and f["p"] == m[1]}
        return hits.pop() if len(hits) == 1 else None
    if task == "latest":
        seq = [f["v"] for _, r, f in facts if r in ("home", "move") and f["p"] == m[1]]
        return seq[-1] if seq else None
    if task == "compose":
        sis = defaultdict(set)
        for _, r, f in facts:
            if r == "sibling":
                sis[f["p"]].add(f["q"])
        who = m[2]
        for _ in range(m[1].count("the sister of")):
            if len(sis[who]) != 1:
                return None
            who = next(iter(sis[who]))
        hits = {f["v"] for _, r, f in facts if r == "home" and f["p"] == who}
        return hits.pop() if len(hits) == 1 else None
    if task == "count":
        k = sum(1 for _, r, f in facts if r == "visit" and f["p"] == m[1] and f["v"] == m[2])
        return NUMBER_WORDS[k] if k < len(NUMBER_WORDS) else str(k)
    if task == "deduce":
        isa = {a: b for s in sents for a, b in _IS_A.findall(s)}
        up = {a: b for s in sents for a, b in _EVERY_A.findall(s)}
        prop = {c: (neg == "") for s in sents for c, neg, p in _EVERY_P.findall(s) if p == m[2]}
        c, guard = isa.get(m[1]), 0
        while c is not None and c not in prop and guard < 10:
            c, guard = up.get(c), guard + 1
        return None if c is None or c not in prop else ("yes" if prop[c] else "no")
    raise ValueError(task)


def shortcuts(task: str, sents: list[str], q: str) -> dict[str, str | None]:
    """Readers that skip part of the skill. Their accuracy must sit near the floor."""
    facts = parse(sents)
    m = _Q[task].fullmatch(q)
    places = [w for s in sents for w in _PLACE.findall(s)]
    out: dict[str, str | None] = {}
    if task == "quote":
        signs = [f["v"] for _, r, f in facts if r == "sign"]
        out["first sign"] = signs[0] if signs else None
        out["last sign"] = signs[-1] if signs else None
    elif task in ("lookup", "keyed"):
        uniq = list(dict.fromkeys(places))
        out["the only place word"] = uniq[0] if len(uniq) == 1 else None
        out["first place"] = places[0] if places else None
        out["last place"] = places[-1] if places else None
        if task == "keyed" and m:
            # look-alike trap: the home of the name that differs by one letter-pair
            homes = {f["p"]: f["v"] for _, r, f in facts if r == "home"}
            near = [p for p in homes if p != m[1] and len(p) == len(m[1]) and
                    sum(a != b for a, b in zip(p, m[1])) <= 2]
            out["look-alike's home"] = homes[near[0]] if near else None
    elif task == "latest":
        out["last place in the document"] = places[-1] if places else None
        per = defaultdict(list)
        for _, r, f in facts:
            if r in ("home", "move"):
                per[f["p"]].append(f["v"])
        busiest = max(per.values(), key=len) if per else []
        out["busiest mover's last place (ignores the name)"] = busiest[-1] if busiest else None
        if m:
            seq = per.get(m[1], [])
            out["first home (no update)"] = seq[0] if seq else None
    elif task == "compose" and m:
        sis = {f["p"]: f["q"] for _, r, f in facts if r == "sibling"}
        homes = {f["p"]: f["v"] for _, r, f in facts if r == "home"}
        hops = m[1].count("the sister of")
        out["own home (0 hops)"] = homes.get(m[2])
        who = m[2]
        for _ in range(max(1, hops - 1)):
            who = sis.get(who, who)
        out["one hop short" if hops > 1 else "one hop (= the skill)"] = homes.get(who)
        # symmetric reading of 'sister': whoever has the asked person as her sister is also a sister
        back = [p for p, s in sis.items() if s == m[2]]
        out["symmetric sister's home (1st hop reversed)"] = homes.get(back[0]) if back else None
    elif task == "count" and m:
        allv = sum(1 for _, r, f in facts if r == "visit" and f["v"] == m[2])
        named = sum(1 for _, r, f in facts if r == "visit" and f["p"] == m[1])
        out["all visits to the place (ignores the name)"] = NUMBER_WORDS[allv] if allv < 7 else None
        out["all visits by the person (ignores the place)"] = NUMBER_WORDS[named] if named < 7 else None
    elif task == "deduce" and m:
        rules = [(c, neg == "") for s in sents for c, neg, p in _EVERY_P.findall(s) if p == m[2]]
        out["first property rule"] = ("yes" if rules[0][1] else "no") if rules else None
        out["last property rule"] = ("yes" if rules[-1][1] else "no") if rules else None
        direct = [s for s in sents if m[1] in s and m[2] in s.lower() and not s.startswith("Every")
                  and not _IS_A.fullmatch(s)]
        if direct:
            neg = re.search(r"\b(not|n't|never)\b", direct[-1])
            out["filler says it directly"] = "no" if neg else "yes"
        else:
            out["filler says it directly"] = None
    return out


def contamination(task: str, sents: list[str], q: str) -> int:
    """Filler sentences (not template facts) about the asked person that touch what is asked."""
    m = _Q[task].fullmatch(q)
    if not m:
        return 0
    facts = {i for i, _, _ in parse(sents)}
    name = m[2] if task == "compose" else m[1]
    if task == "deduce":
        cue = re.compile(rf"\b{m[2]}\b", re.I)
    elif task == "count":
        cue = re.compile(rf"\b({m[2].split()[1]}|went to|go to)\b", re.I)
    elif task == "compose":
        cue = re.compile(r"\b(sister|brother|sibling)\b", re.I)
    else:
        cue = re.compile(r"\b(live|lived|home|house|town|moved|sign|door)\b", re.I)
    return sum(1 for i, s in enumerate(sents) if i not in facts and name in s and cue.search(s))


def audit_rows(rows: list[dict], tok_path: str | None) -> list[dict]:
    tok = None
    if tok_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_path)
    out = []
    for r in rows:
        gold = r["answer"].strip().rstrip(".")
        sents, q = split_doc(r["prompt"])
        rs, _ = split_doc(r["prompt_removed"])
        res = {k: r[k] for k in ("id", "split", "task", "length", "depth", "floor")}
        rr = rule_reader(r["task"], sents, q)
        res["rule_ok"] = rr == gold
        res["rule_answer"] = rr
        res["twin_rule_ok"] = rule_reader(r["task"], rs, q) == gold
        res["twin_has_answer"] = (gold in " ".join(rs)) if r["task"] not in ("count", "deduce") else None
        res["shortcuts"] = {k: (v == gold if v is not None else None) for k, v in shortcuts(r["task"], sents, q).items()}
        res["contamination"] = contamination(r["task"], sents, q)
        if tok is not None and len(r["candidates"]) > 1:
            firsts = [tok.encode(c, add_special_tokens=False)[0] for c in r["candidates"]]
            res["first_token_shared"] = firsts.count(firsts[0]) > 1
        out.append(res)
    return out


def _load(path: Path) -> list[dict]:
    cols = ["id", "split", "task", "length", "depth", "floor", "prompt", "prompt_removed", "answer", "candidates"]
    if path.suffix == ".jsonl":
        with path.open() as fh:
            return [{k: d[k] for k in cols} for d in map(json.loads, fh)]
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=cols).to_pylist()


def _work(args):
    path, tok = args
    return audit_rows(_load(Path(path)), tok)


def summarize(res: list[dict]) -> dict:
    cells = defaultdict(list)
    for r in res:
        cells[(r["split"], r["task"])].append(r)
    table = []
    for (split, task), rs in sorted(cells.items()):
        n = len(rs)
        sc = defaultdict(list)
        for r in rs:
            for k, v in r["shortcuts"].items():
                if v is not None:
                    sc[k].append(v)
        by_len = defaultdict(list)
        for r in rs:
            by_len[r["length"]].append(r)
        twin = [r["twin_has_answer"] for r in rs if r["twin_has_answer"] is not None]
        fts = [r["first_token_shared"] for r in rs if "first_token_shared" in r]
        table.append({
            "split": split, "task": task, "n": n, "floor": sum(r["floor"] for r in rs) / n,
            "rule_reader": sum(r["rule_ok"] for r in rs) / n,
            "rule_reader_by_length": {L: round(sum(x["rule_ok"] for x in v) / len(v), 4) for L, v in sorted(by_len.items())},
            "twin_rule_reader": sum(r["twin_rule_ok"] for r in rs) / n,
            "twin_has_answer": sum(twin) / len(twin) if twin else None,
            "shortcuts": {k: {"acc": round(sum(v) / len(v), 4), "coverage": round(len(v) / n, 4)} for k, v in sc.items()},
            "contaminated_items": sum(r["contamination"] > 0 for r in rs) / n,
            "contaminated_by_length": {L: round(sum(x["contamination"] > 0 for x in v) / len(v), 4) for L, v in sorted(by_len.items())},
            "first_token_shared": sum(fts) / len(fts) if fts else None,
            "rule_misses": [{"id": r["id"], "got": r["rule_answer"]} for r in rs if not r["rule_ok"]][:10],
        })
    return {"cells": table}


def render(summary: dict) -> str:
    lines = ["| split | task | n | floor | rule reader | twin: rule reader / answer in text | best shortcut | items with filler about the asked fact | candidates sharing 1st token |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in summary["cells"]:
        best = max(c["shortcuts"].items(), key=lambda kv: kv[1]["acc"] * kv[1]["coverage"], default=None)
        bs = f"{best[0]}: {best[1]['acc']:.0%} (on {best[1]['coverage']:.0%})" if best else "—"
        ta = "—" if c["twin_has_answer"] is None else f"{c['twin_has_answer']:.0%}"
        ft = "—" if c["first_token_shared"] is None else f"{c['first_token_shared']:.0%}"
        lines.append(f"| {c['split']} | {c['task']} | {c['n']} | {c['floor']:.2f} | {c['rule_reader']:.1%} | "
                     f"{c['twin_rule_reader']:.0%} / {ta} | {bs} | {c['contaminated_items']:.1%} | {ft} |")
    lines.append("")
    lines.append("All shortcuts (accuracy on the items where the shortcut gives an answer, and that coverage):")
    for c in summary["cells"]:
        for k, v in c["shortcuts"].items():
            lines.append(f"- {c['split']}/{c['task']} · {k}: {v['acc']:.0%} on {v['coverage']:.0%} of items")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", required=True, help="a folder of eval parquet/jsonl files (searched recursively) or one file")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8)
    args = p.parse_args()
    src = Path(args.items)
    files = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.suffix in (".parquet", ".jsonl"))
    with Pool(min(args.procs, len(files))) as pool:
        res = [r for part in pool.map(_work, [(str(f), args.tokenizer) for f in files]) for r in part]
    summary = summarize(res)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(summary, indent=1))
    (out / "audit.md").write_text(render(summary))
    print(render(summary))


if __name__ == "__main__":
    main()
