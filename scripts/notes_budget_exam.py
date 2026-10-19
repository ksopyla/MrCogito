#!/usr/bin/env python
"""Notes-budget exam: how many of agent A's notes does agent B need to answer, when B never sees A's text?

A latent-memory model (E31c / E33 / E33e) is split at its notebook, nothing retrained:
  agent A  reads the document and writes the notes (the writer sees page text only);
  agent B  reads ANOTHER document of exactly the same token length followed by the real question, and its global read
           sees A's notes instead of its own — so everything B knows about A's document crossed in the notes.

A closes its last page first: notes exist only for finished pages, so a neutral filler after the document runs to the
next page close (both rows get it; `--no_flush` turns it off and late facts cannot cross).

Per item (one teacher-forced forward each, scored like `evaluation/text_checks_eval.py`):
  own_plain       the normal exam row (sanity check against the run's exam)
  own_all         own text with the closing filler, all notes
  split_all       B's own text, all of A's notes
  split_none      B's own text, no notes (must sit at guessing for far facts)
  split_oracle_K  only the K notes the answer positions looked at most in `own_all` (all rounds, all heads): what the
                  answer needs — an upper bound for any selection
  split_request_K only the K notes B's FIRST read asks for: B sends its first-round query at the answer position (one
                  query vector per head, computed before it has read anything), A returns its K best-matching notes —
                  a one-message request protocol that B and A can actually run
  split_rounds_K  one request per read of the loop: B sends its round-r queries at the answer positions, A returns the
                  K notes that score best for them, B re-reads with every note received so far and asks for round r+1
                  (at most rounds × K notes; `sent` = how many)

Wire cost: a note is k̄ ‖ v̄ = kv_heads × head_dim × 2 numbers; the request is heads × head_dim numbers.

  uv run python scripts/notes_budget_exam.py --checkpoint <E33 checkpoint> --items <data>/eval/id.jsonl \\
      --tasks lookup keyed latest quote --lengths 4096 16384 --max_items_per_cell 40 \\
      --budgets 1 2 4 8 16 32 64 128 --out Cache/e35/notes_budget_e33.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from evaluation.text_checks_eval import _answer_ids, _score  # noqa: E402


def split_ids(tok, prompt: str) -> tuple[list[int], int]:
    """Prompt ids and the number of document tokens before the question."""
    p_ids = tok.encode(prompt, add_special_tokens=False)
    body = prompt.partition("\n\nQuestion:")[0]
    b_ids = tok.encode(body, add_special_tokens=False)
    n = 0
    while n < min(len(b_ids), len(p_ids)) and b_ids[n] == p_ids[n]:
        n += 1
    return p_ids, n


def other_text(ids: list[int], n: int) -> list[int]:
    """Exactly n tokens of another document (repeated if it is shorter)."""
    if not ids:
        raise ValueError("empty replacement document")
    return (ids * (n // len(ids) + 1))[:n]


@torch.no_grad()
def run(model, ids: list[int], start: int, a_ids: list[int], firsts: list[int], candidates: list[str], device, *,
        override=None, keep=None, capture_pos=None) -> tuple[dict, list]:
    model._notes_override, model._note_keep = override, keep
    model._note_capture = None if capture_pos is None else {"q_pos": torch.tensor([capture_pos], device=device)}
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x)
    logits = out.logits if hasattr(out, "logits") else out[0]
    s = _score(logits, start, a_ids, firsts, candidates, len(ids))
    mass = model._note_capture.get("mass", []) if model._note_capture is not None else []
    model._notes_override, model._note_keep, model._note_capture = None, None, None
    return s, mass


def summarize(rows: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        for cond, s in r["cond"].items():
            cells[(r["task"], r["length"], cond)].append(dict(s, late=r["depth"] == "late"))
    out = []
    for (task, length, cond), ss in sorted(cells.items()):
        def mean(k, sel=lambda s: True):
            v = [s[k] for s in ss if s.get(k) is not None and sel(s)]
            return sum(v) / len(v) if v else None
        out.append({"task": task, "length": length, "cond": cond, "n": len(ss), "exact": mean("exact"),
                    "far": mean("exact", lambda s: not s["late"]), "late": mean("exact", lambda s: s["late"]),
                    "answer_nll": mean("answer_nll"), "pick_prob": mean("pick_prob"), "covered": mean("covered"),
                    "sent": mean("sent")})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--lengths", type=int, nargs="*", default=None)
    p.add_argument("--max_items_per_cell", type=int, default=40)
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--round_budgets", type=int, nargs="*", default=[2, 4, 8, 16, 32],
                   help="notes per read for the per-round request protocol (split_rounds_K)")
    p.add_argument("--device", default=None)
    p.add_argument("--attn_backend", default=None)
    p.add_argument("--no_flush", action="store_true", help="A does not close its last page (late facts cannot cross)")
    a = p.parse_args()
    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend = a.attn_backend or ("flex" if device.startswith("cuda") else "sdpa")
    model = load_perceiver_lm(a.checkpoint, device, backend).eval()
    if getattr(model, "memory_writer", None) is None:
        raise SystemExit("the notes-budget exam needs a latent-memory model (E31c / E33 / E33e)")
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    cfg = model.config
    note_numbers = None  # k̄ ‖ v̄ of one note, read off the first forward
    request_numbers = int(cfg.num_attention_heads) * int(cfg.head_dim)

    items = []
    for path in a.items:
        with open(path) as fh:
            items += [json.loads(line) for line in fh]
    if a.tasks:
        items = [r for r in items if r["task"] in set(a.tasks)]
    if a.lengths:
        items = [r for r in items if r["length"] in set(a.lengths)]
    seen, kept = defaultdict(int), []
    for r in items:
        key = (r["task"], r["length"])
        if seen[key] < a.max_items_per_cell:
            seen[key] += 1
            kept.append(r)
    items = sorted(kept, key=lambda r: (r["length"], r["task"], r["id"]))
    groups = defaultdict(list)
    for r in items:
        groups[(r["task"], r["length"])].append(r)
    nxt = {}
    for gl in groups.values():
        for j, r in enumerate(gl):
            nxt[r["id"]] = gl[(j + 1) % len(gl)]

    part = a.out + ".rows.jsonl"
    rows, done = [], set()
    if os.path.exists(part):
        for line in open(part):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            done.add(row["id"])
    todo = [r for r in items if r["id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to go · budgets {a.budgets} · request "
          f"{request_numbers} numbers", flush=True)
    # A closes its last page: notes exist only for finished pages (window W, stride ST, from the row start), so the
    # document's tail would never reach B. A neutral filler after the document runs to the next page close.
    W, ST = int(cfg.lm_window), int(cfg.lm_stride)
    filler = tok.encode(" And that was all for the day." * 64, add_special_tokens=False)

    def flush_len(n_doc: int) -> int:  # the document's last token sits at row index n_doc (BOS at 0)
        if a.no_flush:
            return 0
        j = max(0, -(-(n_doc - (W - 1)) // ST))
        return ST * j + W - 1 - n_doc

    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(part, "a") as pf:
        for i, r in enumerate(todo):
            p_ids, n_doc = split_ids(tok, r["prompt"])
            a_ids = _answer_ids(tok, r["prompt"], r["answer"], p_ids)
            firsts = [(_answer_ids(tok, r["prompt"], c, p_ids) or [-1])[0] for c in r["candidates"]]
            F = flush_len(n_doc)
            fill = filler[:F]
            start = len(p_ids) + F
            own = [tok.bos_token_id] + p_ids[:n_doc] + fill + p_ids[n_doc:] + a_ids
            o_ids, o_n = split_ids(tok, nxt[r["id"]]["prompt"])
            o_doc = o_ids[:o_n]
            b_ids = [tok.bos_token_id] + other_text(o_doc, n_doc) + fill + p_ids[n_doc:] + a_ids
            ans_pos = list(range(start, start + len(a_ids)))
            cond = {}
            cond["own_plain"], _ = run(model, [tok.bos_token_id] + p_ids + a_ids, len(p_ids), a_ids, firsts,
                                       r["candidates"], device)
            cond["own_all"], mass_own = run(model, own, start, a_ids, firsts, r["candidates"], device,
                                            capture_pos=ans_pos)
            notes = model._last_notes
            nb = notes[0].shape[1]
            note_numbers = int(notes[0].shape[2] * notes[0].shape[3] + notes[1].shape[2] * notes[1].shape[3])
            oracle = torch.stack(mass_own).sum(0)[0].sum(0)                       # [nb]: all rounds, answer positions
            cond["split_all"], mass_b = run(model, b_ids, start, a_ids, firsts, r["candidates"], device,
                                            override=notes, capture_pos=ans_pos)
            request = mass_b[0][0, 0]                                              # [nb]: B's first read
            none = torch.zeros(1, nb, dtype=torch.bool, device=device)
            cond["split_none"], _ = run(model, b_ids, start, a_ids, firsts, r["candidates"], device,
                                        override=notes, keep=none)
            tot = float(oracle.sum()) or 1.0
            for K in a.budgets:
                if K >= nb:
                    continue
                for name, score in (("oracle", oracle), ("request", request)):
                    top = torch.topk(score, K).indices
                    keep = none.clone()
                    keep[0, top] = True
                    s, _ = run(model, b_ids, start, a_ids, firsts, r["candidates"], device, override=notes, keep=keep)
                    s["covered"] = float(oracle[top].sum()) / tot  # share of the answer's note attention kept
                    cond[f"split_{name}_{K}"] = s
            # one request per read: B sends its round-r queries (all answer positions), A returns its K best notes for
            # them; B re-reads with every note received so far and asks for the next round. At most rounds × K notes.
            R = len(mass_b)
            for K in a.round_budgets:
                if K * R >= nb:
                    continue
                keep = none.clone()
                keep[0, torch.topk(mass_b[0][0].sum(0), K).indices] = True
                for rr in range(1, R):
                    _, mm = run(model, b_ids, start, a_ids, firsts, r["candidates"], device, override=notes,
                                keep=keep, capture_pos=ans_pos)
                    keep[0, torch.topk(mm[rr][0].sum(0), K).indices] = True
                s, _ = run(model, b_ids, start, a_ids, firsts, r["candidates"], device, override=notes, keep=keep)
                s["covered"] = float(oracle[keep[0]].sum()) / tot
                s["sent"] = int(keep.sum())
                cond[f"split_rounds_{K}"] = s
            row = {k: r[k] for k in ("id", "split", "task", "length", "depth", "floor")}
            row.update({"flush": F, "n_notes": nb, "note_share": float(oracle.sum()) / (len(ans_pos) * len(mass_own) *
                                                                           int(cfg.num_attention_heads)),
                        "cond": {k: {kk: v[kk] for kk in ("exact", "answer_nll", "pick_prob", "covered", "sent")
                                     if kk in v}
                                 for k, v in cond.items()}})
            rows.append(row)
            pf.write(json.dumps(row) + "\n")
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                pf.flush()
                print(f"progress {i + 1}/{len(todo)} · length {r['length']} · {time.time() - t0:.0f}s · "
                      f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)
    summary = summarize(rows)
    res = {"checkpoint": a.checkpoint, "items": a.items, "budgets": a.budgets, "note_numbers": note_numbers,
           "request_numbers": request_numbers, "n_items": len(rows), "cells": summary}
    json.dump(res, open(a.out, "w"), indent=1)
    order = (["own_plain", "own_all", "split_all", "split_none"]
             + [f"split_{n}_{K}" for K in a.budgets for n in ("oracle", "request")]
             + [f"split_rounds_{K}" for K in a.round_budgets])
    by = {(c["task"], c["length"], c["cond"]): c for c in summary}
    for task, length in sorted({(c["task"], c["length"]) for c in summary}):
        print(f"\n{task} @ {length}  (exact: all / far facts / late facts · answer surprise · share of the answer's "
              f"note attention kept)")
        f = lambda v: " -  " if v is None else f"{v:.2f}"  # noqa: E731
        for cname in order:
            c = by.get((task, length, cname))
            if c:
                cov = "" if c["covered"] is None else f"  kept {c['covered']:.2f}"
                cov += "" if c.get("sent") is None else f"  sent {c['sent']:.1f} notes"
                print(f"  {cname:<20} {f(c['exact'])} / {f(c['far'])} / {f(c['late'])}  {c['answer_nll']:.3f}{cov}")


if __name__ == "__main__":
    main()
