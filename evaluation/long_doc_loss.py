"""E33n evaluation 1–2 and 5: held-out long-document loss by token position, at several lengths, notebook on / off.

LSA (arXiv 2610.42147) compares long-context mechanisms by validation loss on held-out long documents; the far-context
value of a mechanism is where it differs from a window-only model, i.e. at positions beyond the local reach. Here a row
is one contiguous span of ONE held-out document (LongDocs-lite val shards, `rule="doc"`), and the per-token loss is
averaged in position buckets: 0–512 (inside every arm's raw window), 512–2k, 2k–8k, 8k–16k, 16k–32k, 32k–64k,
64k–128k. Lengths past the 16k training length are the length-transfer test. For notebook models `--notes_off` adds
the same rows with the notebook switched off (message_override "none"): the notebook's value inside one model.

Usage: uv run python evaluation/long_doc_loss.py --checkpoint RUN/final --data DATA/longdocs_lite_gpt2 \
         --lengths 4096,16384,65536 --rows 16 --notes_off --out results.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

BUCKETS = (0, 512, 2048, 8192, 16384, 32768, 65536, 131072)


def bucket_names(S: int) -> list[str]:
    return [f"{a}-{b}" for a, b in zip(BUCKETS[:-1], BUCKETS[1:]) if a < S]


@torch.no_grad()
def score(model, ds, S: int, rows: int, device: str) -> dict:
    sums = {b: 0.0 for b in bucket_names(S)}
    counts = {b: 0 for b in sums}
    n = min(rows, len(ds))
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        ids = item["input_ids"][None].to(device)
        doc = item["doc_ids"][None].to(device)
        out, per, valid = model(input_ids=ids, labels=ids, doc_ids=doc, return_per_token_loss=True)
        per, valid = per[0].float(), valid[0]
        pos = torch.arange(1, per.shape[0] + 1, device=per.device)  # per[t] predicts token t+1
        for (a, b), name in zip(zip(BUCKETS[:-1], BUCKETS[1:]), sums):
            m = valid & (pos >= a) & (pos < b)
            sums[name] += float(per[m].sum())
            counts[name] += int(m.sum())
    return {"rows": n, "seconds": round(time.time() - t0, 1),
            "loss": {b: (sums[b] / counts[b] if counts[b] else None) for b in sums},
            "tokens": counts,
            "all": sum(sums.values()) / max(1, sum(counts.values()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="LongDocs-lite dir with <source>_val_*.bin and manifest.json")
    ap.add_argument("--lengths", default="4096,16384,32768,65536")
    ap.add_argument("--rows", type=int, default=16, help="rows per source and length")
    ap.add_argument("--notes_off", action="store_true")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.bin_token_dataset import BinRowDataset
    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    model = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, attn_backend=a.backend)
    has_notes = getattr(model, "memory_writer", None) is not None and \
        str(getattr(model.config, "message_override", "real")) == "real"
    srcs = sorted(json.loads((Path(a.data) / "manifest.json").read_text())["sources"])
    res = {"checkpoint": a.checkpoint, "has_notebook": has_notes, "lengths": {}}
    for S in (int(x) for x in a.lengths.split(",")):
        per_len = {}
        for src in srcs:
            try:
                ds = BinRowDataset([{"name": src, "files": [str(Path(a.data) / f"{src}_val_*.bin")]}], S, "doc",
                                   seed=7)
            except ValueError:
                per_len[src] = {"rows": 0}
                continue
            per_len[src] = {"notes_on": score(model, ds, S, a.rows, a.device)}
            if a.notes_off and has_notes:
                model._message_override = "none"
                per_len[src]["notes_off"] = score(model, ds, S, a.rows, a.device)
                model._message_override = "real"
            print(S, src, json.dumps({k: v.get("loss") if isinstance(v, dict) else v
                                      for k, v in per_len[src].items()}), flush=True)
        res["lengths"][S] = per_len
        Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
