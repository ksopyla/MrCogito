#!/usr/bin/env python
"""Extra exam items at short lengths (diagnostics: does a model solve the tasks when the document is
short?). Same generator, held-out stories and names as the frozen sets; written as an extra split file.

  uv run python scripts/make_text_checks_short_items.py --data Cache/text_checks/lab_v1 --lengths 256 512 --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.text_world import TASKS, make_eval_record  # noqa: E402
from scripts.build_text_checks_data import SINGLE_EVIDENCE_TASKS, load_stories  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 512])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--split", default="id")
    p.add_argument("--max_train_files", type=int, default=1)
    a = p.parse_args()
    data = Path(a.data)
    meta = json.loads((data / "text_checks_meta.json").read_text())
    version = meta["version"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(data / "tokenizer"))
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    _, held = load_stories("full", a.max_train_files, 0)
    out = data / "eval" / f"{a.split}_short.jsonl"
    with out.open("w") as fh:
        for task in TASKS:
            for length in a.lengths:
                base = zlib.crc32(f"short/{a.split}/{task}/{length}".encode()) * 10_000
                for i in range(a.n):
                    depth = ("early", "middle", "late")[i % 3] if task in SINGLE_EVIDENCE_TASKS else None
                    rec = make_eval_record(task, base + i, a.split, length, depth, held, ntok, version=version)
                    fh.write(json.dumps(rec) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
