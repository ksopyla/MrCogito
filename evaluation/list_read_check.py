"""E33t stage C (plan §10.4): before any long evaluation, check that the list read (each query's choice kept as a list,
nn.exact_read.ChoiceList + perceiver_ar_lm.attend_list) gives a trained model the same per-token loss as the table read
it was trained with, on held-out LongDocs-lite rows (one document per row), in fp32; then time the list read at longer
lengths and report its peak memory.

Usage: uv run python evaluation/list_read_check.py --checkpoint RUN/decay/.../final --data DATA/longdocs_lite_gpt2 \
         --length 16384 --rows 2 --time_lengths 131072,262144 --out check.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def first_rows(data: str, S: int, n: int) -> list[torch.Tensor]:
    from data.bin_token_dataset import BinRowDataset

    srcs = sorted(json.loads((Path(data) / "manifest.json").read_text())["sources"])
    rows = []
    for src in srcs:
        try:
            ds = BinRowDataset([{"name": src, "files": [str(Path(data) / f"{src}_val_*.bin")]}], S, "doc", seed=7)
        except ValueError:
            continue
        rows.append(ds[0]["input_ids"])
        if len(rows) >= n:
            break
    return rows


@torch.no_grad()
def per_token(model, ids: torch.Tensor, impl: str) -> torch.Tensor:
    with model.open_impl(impl):
        x = ids[None].to("cuda")
        _, per, valid = model(input_ids=x, labels=x, return_per_token_loss=True)
    return per[0][valid[0]].float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--time_lengths", default="", help="also time the list read alone at these lengths (bf16)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    res = {"checkpoint": a.checkpoint, "length": a.length, "rows": []}
    model = load_perceiver_ar_for_eval(a.checkpoint, device="cuda", dtype=torch.float32, attn_backend="flex")
    for ids in first_rows(a.data, a.length, a.rows):
        t0 = time.time()
        ref = per_token(model, ids, "table")
        t1 = time.time()
        lst = per_token(model, ids, "list")
        t2 = time.time()
        d = (lst - ref).abs()
        r = {"tokens": int(ref.numel()), "loss_table": float(ref.mean()), "loss_list": float(lst.mean()),
             "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
             "seconds_table": round(t1 - t0, 1), "seconds_list": round(t2 - t1, 1)}
        res["rows"].append(r)
        print(json.dumps(r), flush=True)
    del model
    torch.cuda.empty_cache()
    if a.time_lengths:
        model = load_perceiver_ar_for_eval(a.checkpoint, device="cuda", attn_backend="flex")
        res["timing"] = {}
        for L in (int(x) for x in a.time_lengths.split(",")):
            rows = first_rows(a.data, L, 1)
            if not rows:
                res["timing"][L] = "no document this long"
                continue
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                per = per_token(model, rows[0], "list")
                res["timing"][L] = {"seconds": round(time.time() - t0, 1), "loss": float(per.mean()),
                                    "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            except torch.cuda.OutOfMemoryError as e:
                res["timing"][L] = f"OutOfMemoryError: {str(e)[:200]}"
                torch.cuda.empty_cache()
            print(L, json.dumps(res["timing"][L]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
