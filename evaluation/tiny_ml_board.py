#!/usr/bin/env python
"""Score a model the way the Tiny-ML Leaderboard does (huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard).

The board (read 2026-11-14 from its index.html) takes three lm-evaluation-harness 0-shot numbers per model:
    blimp : BLiMP accuracy over its 67 sets, in percent
    arc   : ARC-Easy test accuracy (acc, not acc_norm), in percent
    wiki  : WikiText-2 byte perplexity (lower is better)
    params: unique parameters, a shared input/output table counted once; the board admits up to 150M
and computes
    wiki score = 100 · (1 - (ln min(wiki, 500) - lo) / (hi - lo)), clipped to [0, 100], lo/hi over the board
    overall    = (blimp + arc + wiki score) / 3
    efficiency = overall · (1 + 0.5 · t),  t = (log10 Pmax - log10 P) / (log10 Pmax - log10 Pmin) clipped to [0, 1]
Rankings are by efficiency. Both normalisations depend on the whole board, so a new entry shifts everyone a little;
score() puts the candidate on the board before normalising, as adding it would.

A snapshot of the board's entries is kept next to this file (leaderboards/tiny_ml_board_20261114.json: name, org,
params, the three numbers and the training tokens each entry reports).

    uv run python evaluation/tiny_ml_board.py --blimp 78.1 --arc 51.4 --wiki 2.04 --params 141.5M --name ours
    uv run python evaluation/tiny_ml_board.py --from_json Cache/Evaluation_reports/lm_eval/<tag>.json --params 141.5M
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

BOARD = Path(__file__).resolve().parent / "leaderboards" / "tiny_ml_board_20261114.json"
WIKI_PPL_CAP = 500.0
MAX_SIZE_BONUS = 0.5
PARAM_CAP = 150e6


def _js_float(u: str) -> float:
    """JavaScript parseFloat: the longest leading number, NaN when there is none ("~100M" -> NaN)."""
    m = re.match(r"\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", u)
    return float(m.group(0)) if m else math.nan


def parse_params(s) -> float:
    """The board's parseParams: a B/M/K suffix, NaN when unreadable (such entries get no size bonus)."""
    if isinstance(s, (int, float)):
        return float(s)
    u = str(s).upper().replace(",", "")
    for suf, mul in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if u.endswith(suf):
            return _js_float(u) * mul
    return _js_float(u)


def load_board(path: Path = BOARD) -> list[dict]:
    return json.loads(Path(path).read_text())


def score(board: list[dict], cand: dict | None = None) -> list[dict]:
    """Every entry (the candidate included, flagged 'candidate') with wiki_score, overall and efficiency, best first."""
    rows = [dict(r) for r in board] + ([{**cand, "candidate": True}] if cand else [])
    ps = [parse_params(r["params"]) for r in rows]
    logs_p = [math.log10(x) for x in ps if not math.isnan(x) and x > 0]
    pmin, pmax = min(logs_p), max(logs_p)
    logs_w = [math.log(min(r["wiki"], WIKI_PPL_CAP)) for r in rows if isinstance(r.get("wiki"), (int, float)) and r["wiki"] > 0]
    wlo, whi = min(logs_w), max(logs_w)
    for r in rows:
        w = r.get("wiki")
        if isinstance(w, (int, float)) and w > 0 and whi > wlo:
            r["wiki_score"] = max(0.0, min(1.0, 1 - (math.log(min(w, WIKI_PPL_CAP)) - wlo) / (whi - wlo))) * 100
        else:
            r["wiki_score"] = None
        parts = [r.get("blimp"), r.get("arc"), r["wiki_score"]]
        r["overall"] = sum(x if isinstance(x, (int, float)) else 0.0 for x in parts) / 3
        p = parse_params(r["params"])
        t = (pmax - math.log10(p)) / (pmax - pmin) if pmax > pmin and not math.isnan(p) and p > 0 else 0.0
        r["efficiency"] = r["overall"] * (1 + MAX_SIZE_BONUS * max(0.0, min(1.0, t))) if r["overall"] > 0 else -1
    rows.sort(key=lambda r: -r["efficiency"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def from_lm_eval(path: str) -> dict:
    """The three board numbers from an evaluation/run_lm_eval_suite.py JSON (leaderboard tier)."""
    res = json.loads(Path(path).read_text())["results"]

    def metric(task, name):
        for k, v in res.get(task, {}).items():
            if k.split(",")[0] == name and isinstance(v, (int, float)):
                return float(v)
        return None

    blimp = metric("blimp", "acc")
    if blimp is None:  # older harness: no group row, average the 67 sets (1,000 examples each)
        sets = [metric(t, "acc") for t in res if t.startswith("blimp_")]
        sets = [x for x in sets if x is not None]
        blimp = sum(sets) / len(sets) if sets else None
    arc = metric("arc_easy", "acc")
    return {"blimp": None if blimp is None else 100 * blimp, "arc": None if arc is None else 100 * arc,
            "wiki": metric("wikitext", "byte_perplexity")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from_json", help="a run_lm_eval_suite.py result (leaderboard tier)")
    p.add_argument("--blimp", type=float)
    p.add_argument("--arc", type=float)
    p.add_argument("--wiki", type=float)
    p.add_argument("--params", required=True, help="unique parameters, e.g. 141.5M")
    p.add_argument("--name", default="candidate")
    p.add_argument("--board", default=str(BOARD))
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args(argv)
    cand = from_lm_eval(a.from_json) if a.from_json else {}
    for k in ("blimp", "arc", "wiki"):
        if getattr(a, k) is not None:
            cand[k] = getattr(a, k)
    cand.update(name=a.name, params=a.params)
    if parse_params(a.params) > PARAM_CAP:
        print(f"note: {a.params} is over the board's 150M cap; scored anyway, not eligible")
    rows = score(load_board(a.board), cand)
    me = next(r for r in rows if r.get("candidate"))
    print(f"{me['name']}: BLiMP {me.get('blimp')}, ARC-Easy {me.get('arc')}, WikiText-2 byte ppl {me.get('wiki')}"
          f" -> wiki score {me['wiki_score']:.1f}, overall {me['overall']:.2f}, efficiency {me['efficiency']:.2f},"
          f" rank {me['rank']} of {len(rows)}")
    for r in rows[: a.top]:
        mark = " <==" if r.get("candidate") else ""
        print(f"  {r['rank']:3d}. {r['name'][:34]:34s} {str(r['params']):>10s}  BLiMP {r.get('blimp')!s:>6}  "
              f"ARC {r.get('arc')!s:>6}  wiki {r.get('wiki')!s:>7}  eff {r['efficiency']:.2f}{mark}")
    if me["rank"] > a.top:
        print(f"  ... {me['rank']:3d}. {me['name']}  eff {me['efficiency']:.2f} <==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
