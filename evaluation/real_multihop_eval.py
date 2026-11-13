#!/usr/bin/env python
"""E33s exam on real multi-hop QA and maths (eval/realqa.jsonl from scripts/build_real_multihop_data.py --exams).

Per item, one forward pass over  BOS + prompt + " <answer>."  (teacher forcing, as evaluation/text_checks_eval.py):
  exact    the top token is the gold token at every answer position, for the answer or any alias — the same
           verdict as greedy decoding, without generating (any length, cheaply)
  removed  `exact` on the evidence-removed twin: the floor
On a subsample of each cell (--gen_items, lengths ≤ --gen_max_length), greedy generation of up to --gen_tokens tokens:
  gen_em / gen_f1  SQuAD-normalised exact match and token F1 against the answer and its aliases.

    uv run python evaluation/real_multihop_eval.py --checkpoint <ckpt> --items <data>/eval/realqa.jsonl \\
        --out final_realqa.json [--message_override none] [--lengths 4096 16384] [--max_items_per_cell 300]
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

from data.real_multihop import f1_score, normalize_answer  # noqa: E402
from evaluation.text_checks_eval import _answer_ids  # noqa: E402


@torch.no_grad()
def tf_exact(model, tok, prompt: str, answers: list[str], device) -> tuple[bool, float]:
    """Teacher-forced exact over the answers (the first that matches wins); NLL of the first (gold) answer."""
    p_ids = tok.encode(prompt, add_special_tokens=False)
    nll = float("nan")
    for k, a in enumerate(answers):
        a_ids = _answer_ids(tok, prompt, f" {a}.", p_ids)
        ids = torch.tensor([[tok.bos_token_id] + p_ids + a_ids], device=device)
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        start = len(p_ids)
        span = logits[0, start: start + len(a_ids)]
        if k == 0:
            lp = torch.log_softmax(span.float(), -1)
            n = max(1, len(a_ids) - 1)
            nll = float(-lp[torch.arange(n), torch.tensor(a_ids[:n], device=lp.device)].mean())
        if span.argmax(-1).tolist() == a_ids:
            return True, nll
    return False, nll


@torch.no_grad()
def greedy(model, tok, prompt: str, device, max_new: int, chain: bool = False) -> str:
    """Greedy answer. Default: up to the first period or newline. `chain`: the model may write its reasoning first
    ("<step>. <step>. So <answer>."); generation runs to a newline and the answer is the text after the last "So"."""
    ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
    if getattr(model, "_is_hf", False):  # Hub models: cached generation
        x = torch.tensor([ids], device=device)
        g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(g[0, len(ids):].tolist(), skip_special_tokens=True).split("\n")[0].strip()
        return _answer_of(text, chain)
    out_ids = []
    for _ in range(max_new):
        o = model(input_ids=torch.tensor([ids + out_ids], device=device))
        logits = o.logits if hasattr(o, "logits") else o[0]
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_token_id:
            break
        out_ids.append(nxt)
        text = tok.decode(out_ids)
        if "\n" in text or (not chain and text.rstrip().endswith(".")):
            break
    return _answer_of(tok.decode(out_ids).split("\n")[0].strip(), chain)


def _answer_of(text: str, chain: bool) -> str:
    if chain and " So " in f" {text}":
        text = f" {text}".rsplit(" So ", 1)[1]
    text = text.split(". ")[0].strip() if not chain else text.strip()
    return text[:-1] if text.endswith(".") else text


def summarize(rows: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["dataset"], r["length"])].append(r)
    out = []
    for (ds, L), rs in sorted(cells.items()):
        n = len(rs)
        ex = sum(r["exact"] for r in rs) / n
        rm = [r["removed_exact"] for r in rs if r.get("removed_exact") is not None]
        gen = [r for r in rs if "gen_f1" in r]
        by_depth = defaultdict(list)
        for r in rs:
            by_depth[r["depth"]].append(r["exact"])
        out.append({
            "dataset": ds, "task": ds, "split": "dev", "length": L, "n": n, "exact": ex,
            "exact_se": math.sqrt(ex * (1 - ex) / n) if n else None,
            "removed_exact": sum(rm) / len(rm) if rm else None,
            "answer_nll": sum(r["answer_nll"] for r in rs) / n,
            "gen_n": len(gen), "gen_em": sum(r["gen_em"] for r in gen) / len(gen) if gen else None,
            "gen_f1": sum(r["gen_f1"] for r in gen) / len(gen) if gen else None,
            "by_depth": {d: sum(v) / len(v) for d, v in sorted(by_depth.items())},
            "mean_tokens": sum(r.get("n_tokens", 0) for r in rs) / n,
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--attn_backend", default=None)
    p.add_argument("--message_override", default="real", choices=("real", "none"))
    p.add_argument("--lengths", type=int, nargs="*", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--max_items_per_cell", type=int, default=0)
    p.add_argument("--no_removed", action="store_true")
    p.add_argument("--gen_items", type=int, default=100, help="greedy-generation F1 on this many items per cell")
    p.add_argument("--gen_max_length", type=int, default=4096, help="generate only in cells up to this length")
    p.add_argument("--gen_tokens", type=int, default=24)
    p.add_argument("--chain", action="store_true",
                   help="the model writes its reasoning before the answer: score the text after the last 'So' "
                        "(greedy, --gen_tokens should allow the steps, e.g. 192); teacher-forced exact is skipped")
    args = p.parse_args()

    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    backend = args.attn_backend or ("flex" if str(device).startswith("cuda") else "sdpa")
    cfg = json.load(open(os.path.join(args.checkpoint, "config.json"))) if os.path.isdir(args.checkpoint) else {}
    if cfg.get("model_type") in ("llama", "qwen2", "mistral", "gpt2", "gemma", "phi3") or not os.path.isdir(args.checkpoint):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            attn_implementation="sdpa").to(device)
        model._is_hf = True
    else:
        model = load_perceiver_lm(args.checkpoint, device, backend)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    msg = model.message_override(args.message_override) if hasattr(model, "message_override") else nullcontext()

    items = []
    for path in args.items:
        with open(path) as fh:
            items += [json.loads(line) for line in fh]
    if args.lengths:
        items = [r for r in items if r["length"] in set(args.lengths)]
    if args.datasets:
        items = [r for r in items if r["dataset"] in set(args.datasets)]
    if args.max_items_per_cell:
        seen = defaultdict(int)
        keep = []
        for r in items:
            k = (r["dataset"], r["length"])
            if seen[k] < args.max_items_per_cell:
                keep.append(r)
                seen[k] += 1
        items = keep

    t0 = time.time()
    rows, gen_seen = [], defaultdict(int)
    with msg:
        for i, r in enumerate(items):
            answers = [r["answer"]] + [a for a in r.get("aliases") or [] if a and a != r["answer"]]
            ex, nll = (False, float("nan")) if args.chain else tf_exact(model, tok, r["prompt"], answers, device)
            row = {"id": r["id"], "dataset": r["dataset"], "length": r["length"], "depth": r.get("depth"),
                   "exact": bool(ex), "answer_nll": nll, "n_tokens": r.get("n_tokens")}
            if not args.no_removed and not args.chain and r.get("removed_prompt"):
                row["removed_exact"] = bool(tf_exact(model, tok, r["removed_prompt"], [r["answer"]], device)[0])
            k = (r["dataset"], r["length"])
            if args.gen_items and r["length"] <= args.gen_max_length and gen_seen[k] < args.gen_items:
                gen_seen[k] += 1
                pred = greedy(model, tok, r["prompt"], device, args.gen_tokens, chain=args.chain)
                row.update(pred=pred, gen_em=float(normalize_answer(pred) in {normalize_answer(a) for a in answers}),
                           gen_f1=f1_score(pred, answers))
            rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(items)} · {time.time() - t0:.0f}s", flush=True)

    cells = summarize(rows)
    res = {"checkpoint": args.checkpoint, "message_override": args.message_override, "items": args.items,
           "chain": bool(args.chain),
           "n_items": len(rows), "seconds": round(time.time() - t0, 1), "cells": cells}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    with open(args.out + ".rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    f = lambda x: "  -  " if x is None else f"{x:5.3f}"  # noqa: E731
    print(f"{'dataset':<10} {'len':>6} {'n':>5} {'exact':>6} {'removed':>7} {'gen_f1':>6} {'gen_em':>6}")
    for c in cells:
        print(f"{c['dataset']:<10} {c['length']:>6} {c['n']:>5} {f(c['exact'])} {f(c['removed_exact']):>7} "
              f"{f(c['gen_f1'])} {f(c['gen_em'])}")
    print(f"scored {len(rows)} items in {time.time() - t0:.1f}s → {args.out}")


if __name__ == "__main__":
    main()
