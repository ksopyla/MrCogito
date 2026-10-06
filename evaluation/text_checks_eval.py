#!/usr/bin/env python
"""Score a trained checkpoint on the text capability checks (draft `text-v0`).

Spec: docs/engineering_specs/text_capability_checks.md §10. Items come from
`scripts/build_text_checks_data.py` (eval/<split>.jsonl).

One forward pass per item over  BOS + prompt + answer  (teacher forcing):
  exact    the top token is the gold token at every answer position (incl. the closing period)
           — the same verdict as greedy decoding, without generating;
  pick     the candidate whose FIRST token is most probable at the answer position; counted right
           only when the gold candidate's first token wins and no other candidate shares it
           (shared first tokens are reported as `pick_ambiguous`);
  removed  `exact` on the evidence-removed twin — must sit at the guessing floor.

  uv run python evaluation/text_checks_eval.py --checkpoint <run>/final \
      --items Cache/text_checks/smoke/eval/id.jsonl --out <run>/text_checks_id.json \
      [--message_override none] [--max_items_per_cell 50] [--lengths 512 1024]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def _answer_ids(tok, prompt: str, answer: str, p_ids: list[int]) -> list[int]:
    full = tok.encode(prompt + answer, add_special_tokens=False)
    if full[: len(p_ids)] == p_ids:
        return full[len(p_ids):]
    return tok.encode(answer, add_special_tokens=False)  # BPE merged across the boundary


@torch.no_grad()
def score_prompt(model, tok, prompt: str, answer: str, candidates: list[str], device) -> dict:
    p_ids = tok.encode(prompt, add_special_tokens=False)
    a_ids = _answer_ids(tok, prompt, answer, p_ids)
    ids = torch.tensor([[tok.bos_token_id] + p_ids + a_ids], device=device)
    out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    start = len(p_ids)  # logits[start] predicts the first answer token (BOS shifts by one)
    pred = logits[0, start: start + len(a_ids)].argmax(-1).tolist()
    exact = pred == a_ids
    firsts = [(_answer_ids(tok, prompt, c, p_ids) or [-1])[0] for c in candidates]
    gold_first = a_ids[0]
    probs = logits[0, start].float()
    uniq = sorted(set(firsts))
    best = max(uniq, key=lambda t: probs[t].item()) if uniq else -1
    ambiguous = firsts.count(gold_first) > 1
    # one candidate (the single-fact lookup) has no pick to make: reported as None, not 100 %
    pick = None if len(candidates) < 2 else bool(best == gold_first and not ambiguous)
    return {"exact": bool(exact), "pick": pick, "pick_ambiguous": bool(ambiguous), "n_tokens": ids.shape[1]}


def summarize(rows: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["split"], r["task"], r["length"])].append(r)
    out = []
    for (split, task, length), rs in sorted(cells.items()):
        n = len(rs)

        def mean(k):
            v = [float(r[k]) for r in rs if r.get(k) is not None]
            return sum(v) / len(v) if v else None

        acc = mean("exact")
        out.append({
            "split": split, "task": task, "level": rs[0]["level"], "length": length, "n": n,
            "exact": acc, "exact_se": math.sqrt(acc * (1 - acc) / n) if acc is not None and n else None,
            "pick": mean("pick"), "pick_ambiguous": mean("pick_ambiguous"),
            "removed_exact": mean("removed_exact"), "floor": mean("floor"),
            "by_depth": {d: sum(r["exact"] for r in rs if r["depth"] == d) / max(1, sum(1 for r in rs if r["depth"] == d))
                         for d in sorted({r["depth"] for r in rs})},
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--items", nargs="+", required=True, help="eval/<split>.jsonl files")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=None, help="default: the checkpoint's own tokenizer")
    p.add_argument("--device", default=None)
    p.add_argument("--attn_backend", default="sdpa")
    p.add_argument("--message_override", default="real", choices=("real", "none"))
    p.add_argument("--max_items_per_cell", type=int, default=0)
    p.add_argument("--lengths", type=int, nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--no_removed", action="store_true", help="skip the evidence-removed twins")
    args = p.parse_args()

    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_perceiver_lm(args.checkpoint, device, args.attn_backend)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    msg = model.message_override(args.message_override) if hasattr(model, "message_override") else nullcontext()

    items = []
    for path in args.items:
        with open(path) as fh:
            items += [json.loads(line) for line in fh]
    if args.lengths:
        items = [r for r in items if r["length"] in set(args.lengths)]
    if args.tasks:
        items = [r for r in items if r["task"] in set(args.tasks)]
    if args.max_items_per_cell:
        seen = defaultdict(int)
        kept = []
        for r in items:
            key = (r["split"], r["task"], r["length"])
            if seen[key] < args.max_items_per_cell:
                seen[key] += 1
                kept.append(r)
        items = kept

    t0 = time.time()
    rows = []
    with msg:
        for r in items:
            s = score_prompt(model, tok, r["prompt"], r["answer"], r["candidates"], device)
            row = {k: r[k] for k in ("id", "split", "task", "level", "length", "depth", "floor")}
            row.update(s)
            if not args.no_removed:
                row["removed_exact"] = score_prompt(model, tok, r["prompt_removed"], r["answer"], r["candidates"], device)["exact"]
            rows.append(row)
    summary = summarize(rows)
    res = {"checkpoint": args.checkpoint, "message_override": args.message_override, "items": args.items,
           "n_items": len(rows), "seconds": round(time.time() - t0, 1), "cells": summary, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{'split':<10} {'task':<8} {'len':>6} {'n':>4} {'exact':>6} {'pick':>6} {'removed':>7} {'floor':>6}")
    for c in summary:
        f = lambda v: "  -  " if v is None else f"{v:6.2f}"  # noqa: E731
        print(f"{c['split']:<10} {c['task']:<8} {c['length']:>6} {c['n']:>4} {f(c['exact'])} {f(c['pick'])} "
              f"{f(c['removed_exact']):>7} {f(c['floor'])}")
    print(f"scored {len(rows)} items in {res['seconds']}s → {args.out}")


if __name__ == "__main__":
    main()
