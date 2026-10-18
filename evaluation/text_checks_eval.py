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
  removed  `exact` on the evidence-removed twin — must sit at the guessing floor;
  far / near  `exact` on items whose fact sits early or middle (beyond any local window) vs late (spec §10.5).

  uv run python evaluation/text_checks_eval.py --checkpoint <run>/final \
      --items Cache/text_checks/smoke/eval/id.jsonl --out <run>/text_checks_id.json \
      [--message_override none] [--max_items_per_cell 50] [--lengths 512 1024]

Exam-time probes on a trained model (nothing retrained; recorded in the output):
  --loop_rounds r        a looped model (E33) stops after r rounds and decodes through its coda (rounds 1-4 exams)
  --note_order rank --note_strength s [--note_margin m --note_sharp k]
                         the notes' order among matching notes (E33e) with a fixed strength s for every head, put on a
                         model trained without it (E33, E33d: replaces E33d's page distance)
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

from evaluation.text_checks import DEPTH_FAR, DEPTH_NEAR  # noqa: E402


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
    firsts = [(_answer_ids(tok, prompt, c, p_ids) or [-1])[0] for c in candidates]
    return _score(logits, start, a_ids, firsts, candidates, ids.shape[1])


@torch.no_grad()
def score_prompt_a2a(model, tok, prompt: str, answer: str, candidates: list[str], device,
                     document: str | None = None) -> dict:
    """E35: agent A gets the document (the prompt before its question; `document` overrides it — "" = no message
    content, another item's document = the mismatched-message control), agent B the question. Row: BOS doc EOS BOS
    question answer; an empty document gives the receiver-only row."""
    body, _, q = prompt.partition("\n\nQuestion:")
    recv = "Question:" + q
    doc = body if document is None else document
    d_ids = tok.encode(doc, add_special_tokens=False) if doc else []
    r_ids = tok.encode(recv, add_special_tokens=False)
    a_ids = _answer_ids(tok, recv, answer, r_ids)
    head = ([tok.bos_token_id] + d_ids + [tok.eos_token_id]) if d_ids else []
    ids = torch.tensor([head + [tok.bos_token_id] + r_ids + a_ids], device=device)
    out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    start = ids.shape[1] - len(a_ids) - 1
    firsts = [(_answer_ids(tok, recv, c, r_ids) or [-1])[0] for c in candidates]
    return _score(logits, start, a_ids, firsts, candidates, ids.shape[1])


def _score(logits, start: int, a_ids: list[int], firsts: list[int], candidates: list[str], n_tokens: int) -> dict:
    pred = logits[0, start: start + len(a_ids)].argmax(-1).tolist()
    exact = pred == a_ids
    # graded signal: mean surprise (nats per token) on the gold answer, closing period excluded
    lp = torch.log_softmax(logits[0, start: start + len(a_ids)].float(), -1)
    n_ans = max(1, len(a_ids) - 1)
    answer_nll = float(-lp[torch.arange(n_ans), torch.tensor(a_ids[:n_ans], device=lp.device)].mean())
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
    # which candidate was picked (index into `candidates`; gold is 0): e.g. latest's earlier places of the same person
    picked = next((i for i, t in enumerate(firsts) if t == best), None) if len(candidates) >= 2 else None
    return {"exact": bool(exact), "pick": pick, "pick_ambiguous": bool(ambiguous), "n_tokens": n_tokens,
            "answer_nll": answer_nll, "pick_prob": pick_prob, "picked": picked}


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


def apply_note_order(model, mode: str, strength, margin: float, sharp: float, gate: str = "own") -> dict:
    """Exam-time E33e probe: switch the global read's note term to the rank count with fixed strengths (an E33d model's
    learned page-distance slopes are replaced; an E33 model gets the term added). `strength` is one number for every
    head or an "a,b,…" string with one value per head (E33g's order heads)."""
    import torch.nn as nn

    from nn.perceiver_ar_lm import note_age_init

    done = {}
    if getattr(model.config, "message_write", None) != "latent_memory":
        raise SystemExit("--note_order needs a latent-memory model (E31c / E33 / E33d)")
    for name, mod in model.named_modules():
        if not (hasattr(mod, "note_age_mode") and getattr(mod, "pattern", None) == "full"
                and getattr(mod, "compressor", "x") is None):
            continue
        cfg = model.config
        if getattr(mod, "note_age_slope", None) is None:
            mod.note_age_slope = nn.Parameter(torch.zeros(mod.h, device=next(mod.parameters()).device))
            mod.note_age_unit = float(cfg.lm_stride)
        mod.note_age_mode = mode
        mod.note_rank_rep = int(cfg.lm_latents) * int(cfg.lm_reader_tokens)
        mod.note_rank_margin, mod.note_rank_sharp = float(margin), float(sharp)
        mod.note_rank_gate = str(gate)
        with torch.no_grad():
            mod.note_age_slope.copy_(note_age_init(str(strength), mod.h).to(mod.note_age_slope.device))
        done[name] = {"mode": mode, "strength": strength, "margin": margin, "sharp": sharp, "gate": gate}
    if not done:
        raise SystemExit("--note_order: no latent-memory global read found in this model")
    return done


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
        depths = sorted({r["depth"] for r in rs})
        by_n = {d: sum(1 for r in rs if r["depth"] == d) for d in depths}
        cell = {
            "split": split, "task": task, "level": rs[0]["level"], "length": length, "n": n,
            "exact": acc, "exact_se": math.sqrt(acc * (1 - acc) / n) if acc is not None and n else None,
            "pick": mean("pick"), "pick_ambiguous": mean("pick_ambiguous"), "pick_prob": mean("pick_prob"),
            "answer_nll": mean("answer_nll"),
            "removed_answer_nll": mean("removed_answer_nll"),
            "removed_exact": mean("removed_exact"), "floor": mean("floor"),
            "by_depth": {d: sum(r["exact"] for r in rs if r["depth"] == d) / max(1, by_n[d]) for d in depths},
            "by_depth_n": by_n,
        }
        # where the picks went: candidate 0 is the gold answer, 1, 2 … the others in the item's order (latest: the
        # asked person's earlier places, newest first)
        picks = [r.get("picked") for r in rs if r.get("picked") is not None]
        if picks:
            hist = {"0": 0, "1": 0, "2": 0, "3+": 0}
            for k in picks:
                hist[str(k) if k < 3 else "3+"] += 1
            cell["picked_hist"] = {k: v / len(picks) for k, v in hist.items()}
        # far (early + middle facts) vs near (late facts, inside a local window up to 4k): spec §10.5
        far = [r for r in rs if r["depth"] in DEPTH_FAR]
        near = [r for r in rs if r["depth"] in DEPTH_NEAR]
        if far and near:
            cell.update(far=sum(r["exact"] for r in far) / len(far), far_n=len(far),
                        near=sum(r["exact"] for r in near) / len(near), near_n=len(near))
        out.append(cell)
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
    p.add_argument("--loop_rounds", type=int, default=0, help="looped model: stop after this many rounds (0 = trained)")
    p.add_argument("--a2a_message", default="real", choices=("real", "none", "swapped"),
                   help="E35 model: the document agent A reads — the item's own, none, or the next item's of the same "
                        "task and length (mismatched-message control)")
    p.add_argument("--note_order", default=None, choices=("rank",), help="exam-time E33e probe (see the docstring)")
    p.add_argument("--note_strength", default="0", help="one strength for every head, or a,b,… one per head")
    p.add_argument("--loop_feedback", default="as_trained", choices=("as_trained", "off"),
                   help="E33q probe: 'off' removes the round-to-round name feedback (gate) at exam time")
    p.add_argument("--note_gate", default="own", choices=("own", "content"),
                   help="E33g2: count matches by each head's own score, or by the strength-0 heads' mean score")
    p.add_argument("--loop_emb_map", default="as_trained",
                   help="looped-model probe (18 Oct): which trained round embedding each round uses, e.g. 0,1,1,2 "
                        "(round 3 acts as a bridge round, round 4 as the answer round), or 'zero' for none")
    p.add_argument("--note_margin", type=float, default=1.0)
    p.add_argument("--note_sharp", type=float, default=2.0)
    args = p.parse_args()

    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    backend = args.attn_backend or ("flex" if str(device).startswith("cuda") else "sdpa")
    model = load_perceiver_lm(args.checkpoint, device, backend)
    model.eval()
    probes = {}
    if args.loop_rounds:
        if int(getattr(model, "loop_rounds", 1) or 1) < 2:
            raise SystemExit("--loop_rounds needs a looped model")
        model._loop_rounds_override = int(args.loop_rounds)
        probes["loop_rounds"] = int(args.loop_rounds)
    if args.loop_emb_map != "as_trained":
        emb = getattr(model, "loop_emb", None)
        if emb is None:
            raise SystemExit("--loop_emb_map needs a looped model")
        with torch.no_grad():
            if args.loop_emb_map == "zero":
                emb.zero_()
            else:
                idx = [int(i) for i in args.loop_emb_map.split(",")]
                if len(idx) != emb.shape[0] or not all(0 <= i < emb.shape[0] for i in idx):
                    raise SystemExit(f"--loop_emb_map needs {emb.shape[0]} indices in 0..{emb.shape[0] - 1}")
                emb.copy_(emb[idx].clone())
        probes["loop_emb_map"] = args.loop_emb_map
    if args.loop_feedback == "off":
        if getattr(model, "loop_feedback_gate", None) is None:
            raise SystemExit("--loop_feedback off: this model has no round feedback")
        model.loop_feedback_gate = None
        probes["loop_feedback"] = "off"
    if args.note_order:
        probes["note_order"] = apply_note_order(model, args.note_order, args.note_strength, args.note_margin,
                                                args.note_sharp, args.note_gate)
    if probes:
        print("exam-time probes:", json.dumps(probes), flush=True)
    # E33d: the per-head page-distance slopes of the note read, logged with every exam (curve checkpoints included)
    slopes = {n: [round(float(x), 5) for x in p_.detach().float().cpu().flatten()]
              for n, p_ in model.named_parameters() if "note_age_slope" in n}
    if slopes:
        print("note_age_slope:", json.dumps(slopes), flush=True)
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
    is_a2a = getattr(model.config, "checkpoint_family", "") == "a2a"
    if is_a2a:
        probes["a2a_message"] = args.a2a_message
        if not args.no_removed:
            print("a2a: evidence-removed twins are skipped (the controls are --a2a_message none / swapped)", flush=True)
            args.no_removed = True
        nxt = {}
        groups = defaultdict(list)
        for r in items:
            groups[(r["split"], r["task"], r["length"])].append(r)
        for g in groups.values():
            for j, r in enumerate(g):
                nxt[r["id"]] = g[(j + 1) % len(g)]["prompt"].partition("\n\nQuestion:")[0]
    with msg, open(part, "a") as pf:
        for i, r in enumerate(todo):
            if is_a2a:
                doc = None if args.a2a_message == "real" else ("" if args.a2a_message == "none" else nxt[r["id"]])
                s = score_prompt_a2a(model, tok, r["prompt"], r["answer"], r["candidates"], device, document=doc)
            else:
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
           "note_age_slopes": slopes or None, "probes": probes or None, "cells": summary, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{'split':<10} {'task':<8} {'len':>6} {'n':>4} {'exact':>6} {'far':>6} {'pick':>6} {'p_pick':>6} {'removed':>7} "
          f"{'floor':>6} {'ans_nll':>7} {'rm_nll':>7}")
    for c in summary:
        f = lambda v: "  -  " if v is None else f"{v:6.2f}"  # noqa: E731
        print(f"{c['split']:<10} {c['task']:<8} {c['length']:>6} {c['n']:>4} {f(c['exact'])} {f(c.get('far'))} {f(c['pick'])} "
              f"{f(c['pick_prob'])} {f(c['removed_exact']):>7} {f(c['floor'])} {f(c['answer_nll']):>7} "
              f"{f(c['removed_answer_nll']):>7}")
    if story_loss is not None:
        print(f"held-out story loss: {story_loss:.4f} (next-token cross-entropy per token, lower is better)")
    print(f"scored {len(rows)} items in {res['seconds']}s → {args.out}")


if __name__ == "__main__":
    main()
