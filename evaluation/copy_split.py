"""E33n diagnostic: split long-document loss by whether the next token could be COPIED from earlier text.

For each scored position p (token ids[p], predicted from ids[:p]) the (k+1)-gram ids[p-k..p] is looked up among the
earlier positions of the same row:
  near  its most recent earlier copy lies inside the last `window` raw tokens (p - q <= window - k), so a model that
        sees only the local window can copy it;
  far   it occurred earlier, but only outside that window: copying it needs long-range access to exact earlier text;
  none  it never occurred earlier in the row.
Loss sums and token counts per group (positions >= --min_pos), per source, notebook on / off, plus each row's sums
with its document key (for a bootstrap over documents). The window / dense gap on `far` tokens is the part of the
far-context value that is exact copying; a compressed notebook cannot be expected to recover it.

Usage: uv run python evaluation/copy_split.py --checkpoint RUN/final --data DATA/longdocs_lite_gpt2 \
         --length 16384 --rows 64 --ks 3,7 --notes_off --out copy_split.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CATS = ("none", "near", "far")


def copy_categories(ids, k: int, window: int) -> np.ndarray:
    """Per position: 0 none / 1 near / 2 far (see the module doc). Positions < k are 'none'."""
    return copy_sources(ids, k, window)[0]


def copy_sources(ids, k: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """(categories as `copy_categories`, source position q per position: the most recent earlier end of the same
    (k+1)-gram, so ids[q] == ids[p]; −1 where there is none)."""
    a = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
    cat = np.zeros(len(a), dtype=np.int8)
    src = np.full(len(a), -1, dtype=np.int64)
    last: dict[bytes, int] = {}
    for p in range(k, len(a)):
        key = a[p - k: p + 1].tobytes()
        q = last.get(key)
        if q is not None:
            cat[p] = 1 if p - q <= window - k else 2
            src[p] = q
        last[key] = p
    return cat, src


@torch.no_grad()
def score(model, ds, rows: int, ks: list[int], window: int, min_pos: int, device: str) -> dict:
    from evaluation.long_doc_loss import doc_key
    tot = {k: {c: [0.0, 0] for c in CATS} for k in ks}
    detail = []
    n = min(rows, len(ds))
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        ids = item["input_ids"]
        out, per, valid = model(input_ids=ids[None].to(device), labels=ids[None].to(device),
                                doc_ids=item["doc_ids"][None].to(device), return_per_token_loss=True)
        per, valid = per[0].float().cpu().numpy(), valid[0].cpu().numpy()  # per[t] is the loss of token t+1
        pos = np.arange(1, len(per) + 1)
        keep = valid & (pos >= min_pos)
        row = {}
        for k in ks:
            cat = copy_categories(ids.numpy(), k, window)[1: len(per) + 1]
            row[k] = {}
            for ci, c in enumerate(CATS):
                m = keep & (cat == ci)
                s, cnt = float(per[m].sum()), int(m.sum())
                tot[k][c][0] += s
                tot[k][c][1] += cnt
                row[k][c] = [s, cnt]
        detail.append({"doc": doc_key(ds, i), "k": {str(k): v for k, v in row.items()}})
    return {"rows": n, "seconds": round(time.time() - t0, 1),
            "k": {str(k): {c: {"sum": v[0], "tokens": v[1], "loss": v[0] / v[1] if v[1] else None}
                           for c, v in t.items()} for k, t in tot.items()},
            "per_row": detail}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="LongDocs-lite dir with <source>_val_*.bin and manifest.json")
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=64, help="rows per source")
    ap.add_argument("--ks", default="3,7", help="context lengths k of the (k+1)-gram copy test")
    ap.add_argument("--window", type=int, default=512, help="raw tokens every arm sees (the reading layer's window)")
    ap.add_argument("--min_pos", type=int, default=2048, help="score positions >= this (the far-context range)")
    ap.add_argument("--notes_off", action="store_true")
    ap.add_argument("--open_off", action="store_true", help="E33p: also score with the opened tokens off")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.bin_token_dataset import BinRowDataset
    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval
    from evaluation.long_doc_loss import opens

    model = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, attn_backend=a.backend)
    has_notes = getattr(model, "memory_writer", None) is not None and \
        str(getattr(model.config, "message_override", "real")) == "real"
    ks = [int(x) for x in a.ks.split(",")]
    res = {"checkpoint": a.checkpoint, "length": a.length, "window": a.window, "min_pos": a.min_pos,
           "has_notebook": has_notes, "sources": {}}
    for src in sorted(json.loads((Path(a.data) / "manifest.json").read_text())["sources"]):
        try:
            ds = BinRowDataset([{"name": src, "files": [str(Path(a.data) / f"{src}_val_*.bin")]}], a.length, "doc",
                               seed=7)  # the same rows as evaluation/long_doc_loss.py
        except ValueError:
            continue
        r = {"notes_on": score(model, ds, a.rows, ks, a.window, a.min_pos, a.device)}
        if a.notes_off and has_notes:
            with model.message_override("none"):  # restores the model's own setting afterwards
                r["notes_off"] = score(model, ds, a.rows, ks, a.window, a.min_pos, a.device)
        if a.open_off and opens(model):
            with model.open_override("off"):
                r["open_off"] = score(model, ds, a.rows, ks, a.window, a.min_pos, a.device)
        res["sources"][src] = r
        print(src, json.dumps({m: {k: {c: v["loss"] for c, v in t.items()} for k, t in x["k"].items()}
                               for m, x in r.items()}), flush=True)
        Path(a.out).write_text(json.dumps(res))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
