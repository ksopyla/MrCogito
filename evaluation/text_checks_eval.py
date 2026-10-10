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
  pick_prob  the gold candidate's share of the probability over the candidates' first tokens (graded
           pick; guessing gives 1 / candidates) — moves before `exact` does, so calibration runs read it;
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


def _answer_ids(tok, prompt: str, answer: str, p_ids: list[int], tail_chars: int = 400) -> list[int]:
    """Token ids of `answer` as they appear after `prompt`. Only the prompt's tail is re-encoded: BPE
    merges are local, and re-encoding a 128k-token document per candidate made the scorer CPU-bound
    (hours per model on 2026-10-07)."""
    tail = prompt[-tail_chars:]
    t_ids = tok.encode(tail, add_special_tokens=False)
    full = tok.encode(tail + answer, add_special_tokens=False)
    if full[: len(t_ids)] == t_ids:
        return full[len(t_ids):]
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
    # graded signal: mean surprise (nats per token) on the gold answer, closing period excluded
    lp = torch.log_softmax(logits[0, start: start + len(a_ids)].float(), -1)
    n_ans = max(1, len(a_ids) - 1)
    answer_nll = float(-lp[torch.arange(n_ans), torch.tensor(a_ids[:n_ans], device=lp.device)].mean())
    firsts = [(_answer_ids(tok, prompt, c, p_ids) or [-1])[0] for c in candidates]
    gold_first = a_ids[0]
    probs = logits[0, start].float()
    uniq = sorted(set(firsts))
    best = max(uniq, key=lambda t: probs[t].item()) if uniq else -1
    ambiguous = firsts.count(gold_first) > 1
    # one candidate (the single-fact lookup) has no pick to make: reported as None, not 100 %
    pick = None if len(candidates) < 2 else bool(best == gold_first and not ambiguous)
    # graded pick (calibration signal before exact/pick move): the gold candidate's share of the probability
    # over the candidates' first tokens — guessing gives 1 / candidates; None when the gold first token is shared
    pick_prob = None
    if len(candidates) >= 2 and not ambiguous and gold_first in uniq:
        share = torch.softmax(probs[torch.tensor(uniq, device=probs.device)], -1)
        pick_prob = float(share[uniq.index(gold_first)])
    return {"exact": bool(exact), "pick": pick, "pick_ambiguous": bool(ambiguous), "n_tokens": ids.shape[1],
            "answer_nll": answer_nll, "pick_prob": pick_prob}


@torch.no_grad()
def story_ce(model, path: str, n_rows: int, device) -> float:
    """Mean next-token cross-entropy per token on held-out story rows (language quality, T0)."""
    from datasets import load_from_disk

    ds = load_from_disk(path)
    tot, cnt = 0.0, 0
    for i in range(min(n_rows, len(ds))):
        x = torch.tensor([ds[i]["input_ids"]], device=device, dtype=torch.long)
        _, per, valid = model(input_ids=x, labels=x.clone(), return_per_token_loss=True)
        tot += float(per.float().sum())
        cnt += int(valid.sum())
    return tot / max(1, cnt)


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
            "pick": mean("pick"), "pick_ambiguous": mean("pick_ambiguous"), "pick_prob": mean("pick_prob"),
            "answer_nll": mean("answer_nll"),
            "removed_answer_nll": mean("removed_answer_nll"),
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
    p.add_argument("--attn_backend", default=None, help="default: flex on CUDA (long documents), sdpa elsewhere")
    p.add_argument("--message_override", default="real", choices=("real", "none"))
    p.add_argument("--max_items_per_cell", type=int, default=0)
    p.add_argument("--lengths", type=int, nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--no_removed", action="store_true", help="skip the evidence-removed twins")
    p.add_argument("--story_eval", default=None, help="held-out story rows (<data>/stories/eval): language loss")
    p.add_argument("--story_rows", type=int, default=64)
    args = p.parse_args()

    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    backend = args.attn_backend or ("flex" if str(device).startswith("cuda") else "sdpa")
    model = load_perceiver_lm(args.checkpoint, device, backend)
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

    # shortest documents first; every scored item is appended to <out>.rows.jsonl, so an interrupted
    # run resumes where it stopped and progress is visible
    items.sort(key=lambda r: (r["length"], r["split"], r["task"], r["id"]))
    part = args.out + ".rows.jsonl"
    rows, done_ids = [], set()
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(row)
                done_ids.add(row["id"])
    todo = [r for r in items if r["id"] not in done_ids]
    print(f"{len(items)} items, {len(done_ids)} already scored, {len(todo)} to go", flush=True)
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with msg, open(part, "a") as pf:
        for i, r in enumerate(todo):
            s = score_prompt(model, tok, r["prompt"], r["answer"], r["candidates"], device)
            row = {k: r[k] for k in ("id", "split", "task", "level", "length", "depth", "floor")}
            row.update(s)
            if not args.no_removed:
                rm = score_prompt(model, tok, r["prompt_removed"], r["answer"], r["candidates"], device)
                row["removed_exact"], row["removed_answer_nll"] = rm["exact"], rm["answer_nll"]
            rows.append(row)
            pf.write(json.dumps(row) + "\n")
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                pf.flush()
                el = time.time() - t0
                print(f"progress {i + 1}/{len(todo)} · length {r['length']} · {el:.0f}s · "
                      f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)
        story_loss = story_ce(model, args.story_eval, args.story_rows, device) if args.story_eval else None
    summary = summarize(rows)
    res = {"checkpoint": args.checkpoint, "message_override": args.message_override, "items": args.items,
           "n_items": len(rows), "seconds": round(time.time() - t0, 1), "story_loss": story_loss,
           "cells": summary, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{'split':<10} {'task':<8} {'len':>6} {'n':>4} {'exact':>6} {'pick':>6} {'p_pick':>6} {'removed':>7} {'floor':>6} "
          f"{'ans_nll':>7} {'rm_nll':>7}")
    for c in summary:
        f = lambda v: "  -  " if v is None else f"{v:6.2f}"  # noqa: E731
        print(f"{c['split']:<10} {c['task']:<8} {c['length']:>6} {c['n']:>4} {f(c['exact'])} {f(c['pick'])} "
              f"{f(c['pick_prob'])} {f(c['removed_exact']):>7} {f(c['floor'])} {f(c['answer_nll']):>7} "
              f"{f(c['removed_answer_nll']):>7}")
    if story_loss is not None:
        print(f"held-out story loss: {story_loss:.4f} (next-token cross-entropy per token, lower is better)")
    print(f"scored {len(rows)} items in {res['seconds']}s → {args.out}")


if __name__ == "__main__":
    main()
