"""Summarise long_context_use.py outputs: per source and pooled, target loss per variant (nats/token and bits/byte),
paired differences vs the full context with a 95% bootstrap interval over documents."""
import json
import math
import sys

import numpy as np

rng = np.random.default_rng(0)


def boot(x, n=10000):
    x = np.asarray(x)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(path):
    d = json.load(open(path))
    T = d["target"]
    L = d["span"]
    print(f"\n=== {path.split('/')[-1]}  span {L}, target = last {T} tokens")
    pooled = {}
    for src, s in d["sources"].items():
        rows = [r for r in s.get("rows", []) if "error" not in r]
        errs = [r for r in s.get("rows", []) if "error" in r]
        if not rows:
            print(f"{src}: no rows ({len(errs)} errors) {errs[:1]}")
            continue
        full = str(s["contexts"][-1])
        keys = list(rows[0]["cut"].keys()) + (["swap"] if "swap" in rows[0] else [])
        line = f"{src:10} n={len(rows):2}"
        for k in keys:
            v = [((r["swap"] if k == "swap" else r["cut"][k]) / T) for r in rows]
            line += f"  {k}={np.mean(v):.3f}"
        print(line + (f"  errors={len(errs)}" if errs else ""))
        for r in rows:
            f = r["cut"][full]
            for k in keys:
                if k == full:
                    continue
                v = r["swap"] if k == "swap" else r["cut"][k]
                pooled.setdefault(k, []).append(((v - f) / T, (v - f) / math.log(2) / r["target_bytes"], src))
        pooled.setdefault("_full", []).extend([(r["cut"][full] / T, r["cut"][full] / math.log(2) / r["target_bytes"], src) for r in rows])
    print("pooled: difference vs the full context on the same target tokens (positive = worse than full)")
    for k, v in pooled.items():
        if k == "_full":
            a = np.array([x[0] for x in v]); b = np.array([x[1] for x in v])
            print(f"  full context: {a.mean():.3f} nats/token, {b.mean():.4f} bits/byte (n={len(v)})")
            continue
        a = np.array([x[0] for x in v]); b = np.array([x[1] for x in v])
        lo, hi = boot(a)
        share = float((a > 0).mean())
        print(f"  {k:>7}: +{a.mean():.4f} nats/token [{lo:.4f}, {hi:.4f}]  +{b.mean():.4f} bits/byte  "
              f"worse on {share:.0%} of {len(a)} docs")
    # position buckets on the full span, on vs off / nosum
    on, off, nos = {}, {}, {}
    for src, s in d["sources"].items():
        for r in s.get("rows", []):
            if "error" in r:
                continue
            for name, tgt in (("on", on), ("off", off), ("nosum", nos)):
                for bk, (sm, c) in r["buckets"].get(name, {}).items():
                    t = tgt.setdefault(bk, [0.0, 0])
                    t[0] += sm; t[1] += c
    if on:
        print("by position (nats/token, pooled over documents): on / memory off / summaries off")
        for bk in on:
            o = on[bk][0] / on[bk][1]
            f = off[bk][0] / off[bk][1] if bk in off else float("nan")
            n = nos[bk][0] / nos[bk][1] if bk in nos else float("nan")
            print(f"  {bk:>12}: {o:.3f}  {f:.3f} (+{f - o:.3f})  {n:.3f} (+{n - o:.3f})")
        # positions >= 2k
        def ge(t):
            s = sum(v[0] for k, v in t.items() if int(k.split('-')[0]) >= 2048)
            c = sum(v[1] for k, v in t.items() if int(k.split('-')[0]) >= 2048)
            return s / c if c else float("nan")
        print(f"  positions >= 2k: on {ge(on):.3f}, off {ge(off):.3f} (+{ge(off) - ge(on):.3f}), "
              f"nosum {ge(nos):.3f} (+{ge(nos) - ge(on):.3f})")


for p in sys.argv[1:]:
    summarise(p)
