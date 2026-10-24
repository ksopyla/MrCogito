#!/usr/bin/env python
"""Append the decoded step to the memory (idea 4 of the 17 Oct 2026 T4–T7 plan): does writing a multi-step
question's intermediate answer into the document let a trained text-checks model finish the chain? Nothing is trained.

On the frozen compose / deduce exam items (each rebuilt from its seed, so the chain's facts and bridge words are
known), the final question is scored as the exam scores it (teacher forcing, exact = greedy) under these conditions:

  base          the item as examined
  gold_near     the chain's first fact (the gold sentence, as written in the document) appended right before the
                question — inside the raw window: the decoded step a model would append, read locally
  gold_far      the same sentence inserted about `--far_tokens` tokens before the question — beyond the raw window,
                so it reaches the question only through the notes (the memory path)
  gold_chain    every fact but the last appended before the question (only the final lookup is left)
  own_append    the model answers each step itself ("Who is the teacher of Lumo?", then "Who is the teacher of <its
                answer>?" …; deduce "What is Lumo?", "What is a <its answer>?"), greedy; each answer is written back as
                a sentence ("<t> was the teacher of Lumo.", "Lumo is a wump.", "Every wump is a tove.") appended
                before the next question; then the original question
  own_rewrite   the question rewritten with the model's own last bridge ("Where does <t> live?", "What is a <c> like?")
  gold_rewrite  the same with the gold bridge (the multi-hop note's D3 reference)

Also reported: each step's own-answer accuracy.

  uv run python analysis/append_oracle.py --checkpoint <ckpt> --items <data>/eval/id.jsonl \\
      --tasks compose deduce --lengths 4096 16384 --max_items 150 --out <out>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from data.text_world import QUESTIONS, build_item, qa_suffix  # noqa: E402
from evaluation.text_checks_eval import score_prompt  # noqa: E402

SEP = "\n\nQuestion:"


def rebuild(r: dict):
    version = r["version"].replace(".1", "")
    item = build_item(r["task"], random.Random(r["seed"]), r["split"], version)
    if f"{SEP} {item.question}" not in r["prompt"] or f" {item.answer}." != r["answer"]:
        raise ValueError(f"item {r['id']} did not rebuild from its seed")
    return item


@torch.no_grad()
def greedy(model, tok, prompt: str, device, max_new: int = 8) -> str:
    ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
    new: list[int] = []
    for _ in range(max_new):
        out = model(input_ids=torch.tensor([ids + new], device=device))
        logits = out.logits if hasattr(out, "logits") else out[0]
        nxt = int(logits[0, -1].argmax())
        new.append(nxt)
        text = tok.decode(new)
        if nxt == tok.eos_token_id or "." in text or "\n" in text:
            break
    return tok.decode(new).split("\n")[0].split(".")[0].strip()


def _strip_article(s: str) -> str:
    for a in ("a ", "an "):
        if s.startswith(a):
            return s[len(a):]
    return s


def steps_of(item) -> list:
    """(bridge question, gold answer, sentence builder, next subject) per bridge of the asked chain."""
    p = item.cast[0]
    chain = item.steps[item.question]  # [(fact index, output word)] in hop order; the last is the answer
    out = []
    if item.task == "compose":
        prev = p
        for _, word in chain[:-1]:
            out.append((QUESTIONS["bridge_teacher"].format(p=prev), word, prev))
            prev = word
    else:  # deduce: class of the person, then each class link
        prev = p
        for k, (_, word) in enumerate(chain[:-1]):
            q = QUESTIONS["bridge_isa"].format(p=prev) if k == 0 else QUESTIONS["bridge_cat"].format(p=prev)
            out.append((q, f"a {word}", prev))
            prev = word
    return out


def sentence(task: str, k: int, subject: str, answer: str) -> str:
    if task == "compose":
        return f"{answer} was the teacher of {subject}."
    return f"{subject} is {answer}." if k == 0 else f"Every {subject} is {answer}."


def rewrite(item, bridge: str) -> str:
    if item.task == "compose":
        return QUESTIONS["lookup"].format(p=bridge)
    return QUESTIONS["bridge_prop"].format(p=_strip_article(bridge))


def insert_far(body: str, text: str, tok, far_tokens: int) -> str:
    """Insert `text` at a sentence boundary about `far_tokens` tokens before the end of `body`."""
    ids = tok.encode(body, add_special_tokens=False)
    if len(ids) <= far_tokens + 64:
        return text + " " + body
    cut = len(tok.decode(ids[: len(ids) - far_tokens]))
    j = body.find(". ", cut)
    j = cut if j < 0 else j + 2
    return body[:j] + text + " " + body[j:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", default=["compose", "deduce"])
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--max_items", type=int, default=150, help="per (task, length)")
    p.add_argument("--far_tokens", type=int, default=2048)
    p.add_argument("--attn_backend", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_perceiver_lm(a.checkpoint, device, a.attn_backend or ("flex" if device == "cuda" else "sdpa")).eval()
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.checkpoint)

    items, seen = [], defaultdict(int)
    for path in a.items:
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                key = (r["task"], r["length"])
                if r["task"] in a.tasks and r["length"] in a.lengths and seen[key] < a.max_items:
                    seen[key] += 1
                    items.append(r)
    items.sort(key=lambda r: (r["length"], r["task"]))
    print(f"{len(items)} items: {dict(seen)}", flush=True)

    rows, t0 = [], time.time()
    for i, r in enumerate(items):
        item = rebuild(r)
        body = r["prompt"].partition(SEP)[0]
        q = qa_suffix(item.question)
        chain = item.steps[item.question]
        cands = r["candidates"]

        def sc(prompt: str) -> bool:
            return bool(score_prompt(model, tok, prompt, r["answer"], cands, device)["exact"])

        first = item.facts[chain[0][0]].text
        row = {"id": r["id"], "task": r["task"], "length": r["length"], "steps": len(chain),
               "base": sc(r["prompt"]),
               "gold_near": sc(body + " " + first + q),
               "gold_far": sc(insert_far(body, first, tok, a.far_tokens) + q),
               "gold_chain": sc(body + " " + " ".join(item.facts[j].text for j, _ in chain[:-1]) + q)}
        # own steps: answer each bridge question, write the answer back as a sentence, ask the next
        doc, own_ok, bridge = body, [], None
        for k, (bq, gold, subject) in enumerate(steps_of(item)):
            subj = subject if k == 0 or bridge is None else _strip_article(bridge)
            if k > 0:
                bq = (QUESTIONS["bridge_teacher"] if item.task == "compose" else QUESTIONS["bridge_cat"]).format(p=subj)
            ans = greedy(model, tok, doc + qa_suffix(bq), device)
            own_ok.append(ans == gold)
            doc = doc + " " + sentence(item.task, k, subj, ans)
            bridge = ans
        row["own_steps"] = own_ok
        row["own_append"] = sc(doc + q)
        row["own_rewrite"] = sc(body + qa_suffix(rewrite(item, bridge)))
        last_gold = steps_of(item)[-1][1]
        row["gold_rewrite"] = sc(body + qa_suffix(rewrite(item, last_gold)))
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            print(f"progress {i + 1}/{len(items)} · {time.time() - t0:.0f}s", flush=True)

    modes = ("base", "gold_near", "gold_far", "gold_chain", "own_append", "own_rewrite", "gold_rewrite")
    groups = defaultdict(list)
    for x in rows:
        groups[(x["task"], x["length"])].append(x)
    cells = []
    for (task, length), g in sorted(groups.items()):
        c = {"task": task, "length": length, "n": len(g), **{m: sum(x[m] for x in g) / len(g) for m in modes}}
        n_steps = max(len(x["own_steps"]) for x in g)
        c["own_step_acc"] = [sum(x["own_steps"][k] for x in g if len(x["own_steps"]) > k) /
                             max(1, sum(1 for x in g if len(x["own_steps"]) > k)) for k in range(n_steps)]
        cells.append(c)
    res = {"checkpoint": a.checkpoint, "far_tokens": a.far_tokens, "seconds": round(time.time() - t0, 1),
           "cells": cells, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{'task':<8} {'len':>6} {'n':>4} " + " ".join(f"{m:>12}" for m in modes) + "  own step accuracy")
    for c in cells:
        print(f"{c['task']:<8} {c['length']:>6} {c['n']:>4} " + " ".join(f"{c[m]:12.2f}" for m in modes) +
              "  " + " ".join(f"{x:.2f}" for x in c["own_step_acc"]))
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
