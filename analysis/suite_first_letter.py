#!/usr/bin/env python
"""Re-read capability-suite cells on first-letter accuracy (p0).

The suite scores the teacher-forced mean over the answer letters. On cells whose book holds
several candidate answers (lookalike, chain, shuffled, unique, Glyph story), the first given
letters identify the candidate and the rest are copied, so the mean overstates the capability.
The first answer letter is predicted with no answer letters in context: it is the honest score.

    uv run python analysis/suite_first_letter.py --in_dir Cache/capability/li_full_30m [more dirs]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

MULTI_CANDIDATE = ("lookalike", "chain", "shuffled", "unique", "story")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", nargs="+", required=True)
    p.add_argument("--all", action="store_true", help="also list single-candidate cells")
    args = p.parse_args()
    cells: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for root in args.in_dir:
        for jf in Path(root).rglob("job.json"):
            d = jf.parent
            if not (d / "DONE").exists():
                continue
            job = json.loads(jf.read_text())
            rung = [f for f in d.glob("*.json") if f.name not in ("job.json", "summary.json")]
            if not rung:
                continue
            res = json.loads(rung[0].read_text()).get("results") or {}
            for arch, r in res.items():
                fin = r.get("final") or {}
                ppa = fin.get("per_position_acc") or []
                if fin.get("acc") is None or not ppa:
                    continue
                cells[(job["size"], job["cell"])][arch].append((fin["acc"], ppa[0]))
    print("| size | cell | arch | seeds | mean acc (median) | **first letter** (median) | per seed (mean / first) |")
    print("|---|---|---|---|---|---|---|")
    for (size, cell), by_arch in sorted(cells.items()):
        if not args.all and not any(k in cell for k in MULTI_CANDIDATE):
            continue
        for arch, vals in sorted(by_arch.items()):
            m = statistics.median(v[0] for v in vals)
            f = statistics.median(v[1] for v in vals)
            per = ", ".join(f"{100 * a:.0f}/{100 * b:.0f}" for a, b in vals)
            print(f"| {size} | {cell} | {arch} | {len(vals)} | {100 * m:.1f} | **{100 * f:.1f}** | {per} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
