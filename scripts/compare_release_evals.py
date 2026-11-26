"""Before / after table of two release-test folders (scripts/release_evals.sh): RULER per task and length, the
long-document tests, the core tasks. `--before` may be a model card folder (docs/release/hf_card_*), whose files
have the same names.

Usage: python scripts/compare_release_evals.py --before docs/release/hf_card_concept_memory_150m \
           --after Cache/list_ft_evals/list_a
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def ruler(folder: Path) -> dict:
    """{(length, task): score} from every ruler/ruler_*.json of the folder (our own model's files: no prefix)."""
    out: dict = {}
    for f in sorted(glob.glob(str(folder / "ruler" / "ruler_*.json"))):
        d = json.loads(Path(f).read_text())
        res = d.get("results") or d.get("tasks") or d
        for length, tasks in (res.items() if isinstance(res, dict) else []):
            if not isinstance(tasks, dict):
                continue
            for task, v in tasks.items():
                score = v.get(f"{length},none", v.get("score")) if isinstance(v, dict) else v
                if isinstance(score, (int, float)):
                    out[(str(length), task)] = float(score)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    a = ap.parse_args()
    b, c = ruler(Path(a.before)), ruler(Path(a.after))
    if not c:
        f = sorted(glob.glob(str(Path(a.after) / "ruler" / "ruler_*.json")))
        print("no RULER scores read from", a.after, "- top-level keys of its first file:",
              list(json.loads(Path(f[0]).read_text()))[:12] if f else "no file")
        return
    lengths = sorted({k[0] for k in c}, key=int)
    tasks = sorted({k[1] for k in c})
    print(f"{'RULER task':22}" + "".join(f"{L + ' before':>14}{'after':>8}" for L in lengths))
    for t in tasks:
        print(f"{t:22}" + "".join(
            f"{b.get((L, t), float('nan')):14.2f}{c.get((L, t), float('nan')):8.2f}" for L in lengths))
    for name, pick in (("mean of all", lambda t: True),
                       ("mean without the two", lambda t: t not in ("niah_multivalue", "niah_multiquery"))):
        cells = []
        for L in lengths:
            ts = [t for t in tasks if pick(t) and (L, t) in b and (L, t) in c]
            cells.append(f"{sum(b[(L, t)] for t in ts) / max(1, len(ts)):14.3f}"
                         f"{sum(c[(L, t)] for t in ts) / max(1, len(ts)):8.3f}")
        print(f"{name:22}" + "".join(cells))


if __name__ == "__main__":
    main()
