#!/usr/bin/env python
"""Write a re-weighted copy of a pretokenized manifest (new file ⇒ fresh length / packing caches).

The trainer's sequence-length and packing caches are keyed by the manifest path, so source
weights of a pretokenized mix must be changed by writing a *new* manifest, never by a runtime
override. Weights are per-row sampling probabilities; pass them as token-share targets and the
script converts with the mean row length of each source (from the manifest's token stats when
present, else by scanning up to `--sample_rows` rows).

Example (E22 long-document mix over the E18b sources):
  uv run python scripts/write_manifest_variant.py \
      --src  $TOK/e18b_lm_ret05_manifest.json \
      --dst  $TOK/e22_longmix_32k_manifest.json \
      --token_share '{"pg19":0.35,"finepdfs_100BT":0.35,"fineweb_edu":0.15,"stack_edu_py":0.10,"retrieval_keyed_recall":0.05}' \
      --mix_id e22_longmix_32k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _mean_row_tokens(src: dict, sample_rows: int) -> float:
    from datasets import load_from_disk

    ds = load_from_disk(src["train_path"])
    n = min(len(ds), sample_rows)
    idx = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    lens = [len(ds[i]["input_ids"]) for i in idx]
    return float(sum(lens)) / max(1, len(lens))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--token_share", required=True, help="JSON: source name -> target token share")
    p.add_argument("--mix_id", default=None)
    p.add_argument("--sample_rows", type=int, default=2000)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    man = json.loads(Path(args.src).read_text())
    share = {k: float(v) for k, v in json.loads(args.token_share).items()}
    names = [s["name"] for s in man["sources"]]
    unknown = sorted(set(share) - set(names))
    if unknown:
        raise SystemExit(f"unknown sources {unknown}; manifest has {names}")
    total = sum(share.values())
    share = {k: v / total for k, v in share.items()}

    rows = []
    for src in man["sources"]:
        name = src["name"]
        if name not in share:
            src["weight"] = 0.0
            rows.append((name, 0.0, None, 0.0))
            continue
        mean_tok = src.get("mean_row_tokens") or _mean_row_tokens(src, args.sample_rows)
        src["mean_row_tokens"] = mean_tok
        src["weight_token_share_target"] = share[name]
        w = share[name] / mean_tok
        src["weight"] = w
        rows.append((name, share[name], mean_tok, w))
    wsum = sum(r[3] for r in rows)
    for src in man["sources"]:
        src["weight"] = src["weight"] / wsum
    man["sources"] = [s for s in man["sources"] if s["weight"] > 0]
    man["mix_id"] = args.mix_id or (man.get("mix_id", "mix") + "+reweighted")
    man["derived_from"] = str(args.src)
    man["token_share_target"] = share

    print(f"{'source':26s} {'share':>7s} {'mean_tok':>9s} {'row_weight':>11s}")
    for name, sh, mt, w in rows:
        print(f"{name:26s} {sh:7.3f} {mt if mt is None else round(mt):>9} {w / wsum:11.6f}")
    if args.dry_run:
        return
    Path(args.dst).write_text(json.dumps(man, indent=2))
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
