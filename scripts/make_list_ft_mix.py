"""A small data mix for the list fine-tune test (root-cause work, 2026-11-23).

One phase of 16k rows: the 16k phase of a built mix as it is (replay, its sources in their own proportions) plus one
constructed set at `--weight` of the rows — a list set of scripts/build_reasoning_rows.py (questions that need
several far values), or, for the control run, a lookup set (every question one value). OUT gets mix.json (one phase,
"decay", read by scripts/run_e33n.py --mix), links to the shards, and a link to the replay mix's tokenizer.

Usage:
  python scripts/make_list_ft_mix.py --replay Cache/e38_check/m21_check --extra Cache/list_rows --set list_16k \
      --out Cache/list_ft/a/mix --weight 0.5 --tokens 6e7
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True, help="a built mix (scripts/build_m21_mix.py): its 16k phase is replayed")
    ap.add_argument("--extra", required=True, help="folder of the constructed set (build_reasoning_rows.py --out)")
    ap.add_argument("--set", default="list_16k", help="the constructed set's name (list_16k, or lookup_16k: control)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=0.5, help="share of the rows from the constructed set")
    ap.add_argument("--tokens", type=float, default=6e7)
    ap.add_argument("--phase", default="decay", help="the replay mix's phase to copy (its row length)")
    ap.add_argument("--seed", type=int, default=5042)
    a = ap.parse_args()
    # several constructed sets: --set a,b with --extra dirA,dirB (or one folder for all); the weight is split evenly
    sets = [s for s in a.set.split(",") if s]
    extras = [Path(e).resolve() for e in a.extra.split(",") if e]
    if len(extras) == 1:
        extras = extras * len(sets)
    if len(extras) != len(sets):
        raise SystemExit("--extra needs one folder, or one per --set")
    replay, out = Path(a.replay).resolve(), Path(a.out)
    mix = json.loads((replay / "mix.json").read_text())
    ph = mix["phases"][a.phase]
    out.mkdir(parents=True, exist_ok=True)

    def link(src: Path, dst: Path) -> None:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src, dst)

    # the replay mix's own supply report: a source with no row at this phase's length is left out (small builds)
    have = {s["name"]: s.get("rows_available") for s in ph.get("supply", [])}
    have_ev = {s["name"]: s.get("rows_available") for s in ph.get("eval_supply", [])}
    train = []
    for s in ph["train"]:
        files = sorted(glob.glob(str(replay / f"{s['name']}_train_*.bin")))
        if not files or s["name"] in sets:  # a constructed set of the same name comes from --extra, at --weight
            continue
        if s["name"] in have and not have[s["name"]]:
            continue
        for f in files:
            link(Path(f), out / Path(f).name)
        train.append({"name": s["name"], "weight": float(s["weight"]), "rule": s.get("rule", ph.get("rule", "stream"))})
    total = sum(t["weight"] for t in train) or 1.0
    for t in train:  # the replay sources keep their proportions and share 1 - weight
        t["weight"] = round(t["weight"] / total * (1.0 - a.weight), 6)
    n_replay = len(train)
    ev = []
    for s in ph.get("eval", []):
        files = sorted(glob.glob(str(replay / f"{s['name']}_val_*.bin")))
        if files and (s["name"] not in have_ev or have_ev[s["name"]]):
            for f in files:
                link(Path(f), out / Path(f).name)
            ev.append({"name": s["name"], "rows": int(s["rows"])})
    said = []
    for set_name, extra in zip(sets, extras):
        done = json.loads((extra / f"{set_name}.done.json").read_text())
        if int(done["seq_len"]) != int(ph["seq_len"]):
            raise SystemExit(f"{set_name} has rows of {done['seq_len']}, the replay phase of {ph['seq_len']}")
        name = f"{set_name}_ft"  # its own name: the replay mix may hold a set of the same name (lookup_16k)
        for f in sorted(glob.glob(str(extra / f"{set_name}_train_*.bin"))):
            link(Path(f), out / Path(f).name.replace(set_name, name, 1))
        train.append({"name": name, "weight": round(float(a.weight) / len(sets), 6), "rule": "stream"})
        for f in sorted(glob.glob(str(extra / f"{set_name}_val_*.bin"))):
            link(Path(f), out / Path(f).name.replace(set_name, name, 1))
        ev.append({"name": name, "rows": 16 if int(ph["seq_len"]) <= 16384 else 4, "rule": "stream"})
        rows = int(done["rows"])
        said.append(f"{name} at {a.weight / len(sets):.2f} ({rows:,} rows, {rows * int(ph['seq_len']) / 1e6:.0f}M tokens)")
    link(replay / mix.get("tokenizer", "tokenizer"), out / "tokenizer")
    new = {"name": f"list_ft:{mix.get('name')}+{a.set}", "tokenizer": "tokenizer",
           "tokenizer_source": mix.get("tokenizer_source"), "vocab_size": mix.get("vocab_size"), "eot": mix["eot"],
           "phases": {"decay": {"seq_len": int(ph["seq_len"]), "rule": ph.get("rule", "stream"), "seed": int(a.seed),
                                "tokens": float(a.tokens), "train": train, "eval": ev}}}
    (out / "mix.json").write_text(json.dumps(new, indent=1))
    print(f"{out}: rows of {ph['seq_len']}, {n_replay} replay sources at {1 - a.weight:.2f}, " + ", ".join(said)
          + f"; phase budget {a.tokens / 1e6:.0f}M tokens")


if __name__ == "__main__":
    main()
