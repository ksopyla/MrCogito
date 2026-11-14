#!/usr/bin/env python
"""Check the cached greedy decode against the per-letter loop on a real capability checkpoint (2026-11-14).

The length ladder's picked-candidate score decodes the answer greedily, one forward over the whole row per letter
(`verification/bapo_capability_probe._greedy_answers`); at 32k that costs ~33 s a row for the E33t catalogue model and
~5 min a row at 128k. `--decode cached` reads the row once and runs one step per later position (nn/incremental.py).
This script reads the ladder's own rows at one length (same seed, same batches) and reports:

* `--letters`: both decodes, letter by letter (rows with every letter equal, letters that differ, where), plus each
  decode's scores and cost;
* without it: the cached decode's scores only, to compare with a full decode already in a ladder JSON (`--ref`):
  first letter of the greedy answer, every letter right, picked candidate, the same rows.

    uv run python verification/check_cached_decode.py --ckpt <job dir> --length 32768 --letters --out check_32k.json
    uv run python verification/check_cached_decode.py --ckpt <job dir> --length 131072 --ref <job dir>/ladder.json \\
        --out check_128k.json

Evaluation only; it trains nothing and writes only `--out`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.bapo_ladder import generate_row_for  # noqa: E402
from verification.bapo_capability_probe import _answer_scores, _answer_summary, _greedy_answers  # noqa: E402
from verification.length_ladder import _amp_ctx, data_config, load  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="job dir with <arch>.pt (bapo_capability_probe --save_ckpt)")
    p.add_argument("--arch", default=None, help="which <arch>.pt (default: the only one)")
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--tokens_per_batch", type=int, default=65536, help="as the ladder: batch = max(1, this // length)")
    p.add_argument("--seed", type=int, default=12345, help="as the ladder: rows from seed + length")
    p.add_argument("--backend", default="auto", choices=("auto", "sdpa", "flex"))
    p.add_argument("--amp", default="auto")
    p.add_argument("--letters", action="store_true", help="also run the per-letter loop and compare letter by letter")
    p.add_argument("--ref", default=None, help="a ladder JSON whose full decode at --length the scores must match")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    paths = sorted(ckpt.glob("*.pt"))
    if args.arch:
        paths = [q for q in paths if q.stem == args.arch]
    if len(paths) != 1:
        raise SystemExit(f"need exactly one <arch>.pt in {ckpt}, found {[q.name for q in paths]}")
    L = int(args.length)
    backend = args.backend
    if backend == "auto":
        backend = "flex" if (device.type == "cuda" and L > 8192) else "sdpa"
    model, state = load(paths[0], seq_len=L, backend=backend, device=device)
    cfg = data_config(state, L)
    rng = np.random.default_rng(args.seed + L)
    batch = max(1, args.tokens_per_batch // L)
    from verification.heat_pause import heat_pause

    counts = {"cached": [], "full": []}
    secs = {"cached": 0.0, "full": 0.0}
    rows_same = letters_diff = letters_n = 0
    first_mismatch: list = []
    done = 0
    t_start = time.time()
    while done < args.rows:
        heat_pause("check_cached_decode")
        b = min(batch, args.rows - done)
        rr = [generate_row_for(cfg, rng) for _ in range(b)]
        ids = torch.from_numpy(np.stack([r.input_ids for r in rr])).long().to(device)
        labels = torch.from_numpy(np.stack([r.labels for r in rr])).long().to(device)
        m = labels != -100
        got = {}
        if args.letters:  # as the ladder: its teacher-forced read gives the letters up to the first wrong one
            _sync(device)
            t0 = time.time()
            with torch.no_grad(), _amp_ctx(device, args.amp):
                out = model(ids, return_logits=True)
                logits = out.logits if hasattr(out, "logits") else out
            got["full"] = _greedy_answers(model, ids, labels, amp=args.amp, device=device, first_logits=logits)
            _sync(device)
            secs["full"] += time.time() - t0
            del logits, out
        _sync(device)
        t0 = time.time()
        got["cached"] = _greedy_answers(model, ids, labels, amp=args.amp, device=device, decode="cached")
        _sync(device)
        secs["cached"] += time.time() - t0
        model.eval()
        for kind, cur in got.items():
            counts[kind].append(_answer_scores(ids, labels, cur, cfg))
        if args.letters:
            for r in range(b):
                pos = torch.nonzero(m[r]).flatten()
                a, c = got["full"][r, pos], got["cached"][r, pos]
                diff = (a != c)
                letters_n += int(pos.numel())
                letters_diff += int(diff.sum())
                if bool(diff.any()):
                    first_mismatch.append({"row": done + r, "letter": int(torch.nonzero(diff)[0, 0]),
                                           "of": int(pos.numel())})
                else:
                    rows_same += 1
        done += b
        print(f"  {done}/{args.rows} rows  cached {secs['cached'] / done:.2f} s/row"
              + (f"  full {secs['full'] / done:.2f} s/row  rows equal {rows_same}/{done}" if args.letters else ""),
              flush=True)

    report = {"ckpt": str(ckpt), "arch": paths[0].stem, "length": L, "rows": args.rows, "batch": batch,
              "backend": backend, "amp": args.amp, "seed": args.seed + L,
              "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "wall_s": time.time() - t_start}
    for kind in ("cached", "full"):
        if counts[kind]:
            report[kind] = {**_answer_summary(counts[kind]), "sec_per_row": secs[kind] / args.rows}
    if args.letters:
        report["letters"] = {"rows_equal": rows_same, "rows": args.rows, "letters_differ": letters_diff,
                             "letters": letters_n, "first_mismatch": first_mismatch}
    if args.ref:
        ref = json.loads(Path(args.ref).read_text())["results"][paths[0].stem].get(str(L), {})
        keys = ("exact", "first_greedy", "candidate", "candidate_rows")
        report["ref"] = {k: ref.get(k) for k in keys}
        report["ref"]["rows"] = ref.get("rows")
        report["ref_match"] = {k: (ref.get(k) is not None and abs(float(report["cached"].get(k, float("nan")))
                                                                 - float(ref[k])) < 1e-9) for k in keys}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "letters"} | (
        {"letters": {k: v for k, v in report["letters"].items() if k != "first_mismatch"}} if args.letters else {}),
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
