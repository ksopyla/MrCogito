"""E40: read evaluation/long_context_use.py outputs (one model, or two models paired on the same documents).

Measures, in nats per token of the target (the last 4,096 tokens of each span), mean over documents with a 95 %
bootstrap interval over documents:
  full            the whole document before the target
  gain_far        loss with the last 1,024 tokens only − loss with the whole document (what the far past is worth)
  cost_<mode>     loss with a memory switch on the whole document − loss as trained (off, nosum, sumonly)
  sum_alone_share (loss off − loss sumonly) / (loss off − loss full): the share of the memory read's benefit that the
                  concept summaries carry alone, nothing opened (P1: "are the concepts the main source?")

Usage:
  python scripts/e40_longctx_summary.py one  A.json [--json out.json]
  python scripts/e40_longctx_summary.py pair A.json B.json [--json out.json]   # B − A per measure, paired by document
"""
from __future__ import annotations

import argparse
import json

import numpy as np

RNG = np.random.default_rng(0)


def boot(x, n: int = 10000) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_doc(path: str) -> dict:
    """{(source, doc): {measure: value}} for one long_context_use.py output."""
    d = json.load(open(path))
    T = int(d["target"])
    out = {}
    for src, s in d["sources"].items():
        full_key = str(s["contexts"][-1]) if s.get("contexts") else None
        for r in s.get("rows", []):
            if "error" in r or full_key is None:
                continue
            c = {k: v / T for k, v in r["cut"].items()}
            m = {"full": c[full_key]}
            if "1024" in c:
                m["gain_far"] = c["1024"] - c[full_key]
                m["cut_1k"] = c["1024"]
            for mode in ("off", "nosum", "sumonly"):
                if mode in c:
                    m[f"cost_{mode}"] = c[mode] - c[full_key]
                    m[mode] = c[mode]
            out[(src, int(r["doc"]))] = m
    return out


def ratio_share(rows: list[dict]) -> tuple[float, float, float]:
    """sum_alone_share as a ratio of means, with a bootstrap interval over documents."""
    have = [r for r in rows if "cost_off" in r and "cost_sumonly" in r]
    if not have:
        return float("nan"), float("nan"), float("nan")
    num = np.array([r["cost_off"] - r["cost_sumonly"] for r in have])
    den = np.array([r["cost_off"] for r in have])
    val = float(num.mean() / den.mean())
    idx = RNG.integers(0, len(have), size=(10000, len(have)))
    bs = num[idx].mean(1) / den[idx].mean(1)
    return val, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def one(path: str) -> dict:
    docs = per_doc(path)
    rows = list(docs.values())
    res = {"file": path, "docs": len(rows), "measures": {}}
    keys = sorted({k for r in rows for k in r})
    for k in keys:
        v = [r[k] for r in rows if k in r]
        lo, hi = boot(v)
        res["measures"][k] = {"mean": float(np.mean(v)), "lo": lo, "hi": hi, "n": len(v),
                              "share_positive": float(np.mean(np.asarray(v) > 0))}
    s, lo, hi = ratio_share(rows)
    res["measures"]["sum_alone_share"] = {"mean": s, "lo": lo, "hi": hi, "n": len(rows)}
    return res


def pair(a: str, b: str) -> dict:
    da, db = per_doc(a), per_doc(b)
    common = sorted(set(da) & set(db))
    res = {"a": a, "b": b, "docs": len(common), "b_minus_a": {}}
    keys = sorted({k for key in common for k in da[key]} & {k for key in common for k in db[key]})
    for k in keys:
        v = [db[key][k] - da[key][k] for key in common if k in da[key] and k in db[key]]
        lo, hi = boot(v)
        res["b_minus_a"][k] = {"mean": float(np.mean(v)), "lo": lo, "hi": hi, "n": len(v),
                               "share_b_higher": float(np.mean(np.asarray(v) > 0))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    o = sub.add_parser("one")
    o.add_argument("a")
    o.add_argument("--json")
    p = sub.add_parser("pair")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--json")
    args = ap.parse_args()
    if args.what == "one":
        res = one(args.a)
        print(f"{args.a}: {res['docs']} documents")
        for k, m in res["measures"].items():
            print(f"  {k:16} {m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]"
                  + (f"  positive on {m['share_positive']:.0%}" if "share_positive" in m else ""))
    else:
        res = pair(args.a, args.b)
        print(f"B − A, paired on {res['docs']} documents (A {args.a}, B {args.b})")
        for k, m in res["b_minus_a"].items():
            print(f"  {k:16} {m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]  B higher on {m['share_b_higher']:.0%}")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
