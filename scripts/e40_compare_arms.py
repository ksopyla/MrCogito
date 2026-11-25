"""E40: judge the retrofit's arms against the control from their release-test folders (scripts/release_evals.sh).

Reads, per arm folder: long_context/real_docs_{64k,16k}.json (evaluation/long_context_use.py with modes off, nosum,
sumonly), ruler/ruler_{4k_8k,16k}.json, eval_results/<tag>_final.json (BLiMP, ARC-Easy, WikiText-2) and
<tag>_core7.json. Prints the measures and the spec's marks (docs/experiments_specs/done_failed/E40_depth_mixed_read.md):
  S1  depth-mixed read: far-context gain at 64k (loss with the last 1k only − whole document) ≥ control + 0.010,
      95 % interval above zero
  S2  depth-mixed read: the cost of switching the summaries off at 64k higher than the control's, and its
      summaries-only loss lower than the control's
  S3  2-DCA everywhere: whole-document loss at 16k and at 64k at least 0.010 below the control's
  S4  no harm (each arm): RULER 4k–16k 13-task mean within 0.02 of the control's; BLiMP and the 7-task core mean
      within 1 point
  K3  both arms within ±0.005 of the control on whole-document loss (64k) and far-context gain (64k)
Speed (S5) is P4's, recorded in the spec.

Usage: python scripts/e40_compare_arms.py --control DIR --read DIR --dca2 DIR [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e40_longctx_summary import pair  # noqa: E402

CORE7 = ("arc_challenge", "arc_easy", "hellaswag", "lambada_openai", "piqa", "sciq", "winogrande")


def ruler(folder: Path) -> dict:
    """{length: {task: score}} from ruler/ruler_*.json (our own run's files)."""
    out: dict = {}
    for f in sorted(glob.glob(str(folder / "ruler" / "ruler_*.json"))):
        res = json.loads(Path(f).read_text()).get("results", {})
        for length, tasks in res.items():
            for task, v in tasks.items():
                s = v.get(f"{length},none") if isinstance(v, dict) else None
                if isinstance(s, (int, float)) and s >= 0:
                    out.setdefault(int(length), {})[task] = float(s)
    return out


def board(folder: Path) -> dict:
    out = {}
    for f in glob.glob(str(folder / "eval_results" / "*_final.json")):
        b = json.loads(Path(f).read_text()).get("tiny_ml_board", {})
        out.update({k: b[k] for k in ("blimp", "arc", "wiki") if k in b})
    for f in glob.glob(str(folder / "eval_results" / "*_core7.json")):
        r = json.loads(Path(f).read_text()).get("results", {})
        accs = [100 * r[t]["acc,none"] for t in CORE7 if t in r]
        if len(accs) == len(CORE7):
            out["core7"] = sum(accs) / len(accs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True)
    ap.add_argument("--read", required=True)
    ap.add_argument("--dca2", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    arms = {"control": Path(a.control), "read": Path(a.read), "dca2": Path(a.dca2)}
    res: dict = {"long": {}, "ruler": {}, "board": {}, "marks": {}}
    for L in ("64k", "16k"):
        fc = arms["control"] / "long_context" / f"real_docs_{L}.json"
        for arm in ("read", "dca2"):
            fa = arms[arm] / "long_context" / f"real_docs_{L}.json"
            if fc.exists() and fa.exists():
                res["long"][f"{arm}_{L}"] = pair(str(fc), str(fa))["b_minus_a"]
    for arm, d in arms.items():
        r = ruler(d)
        means = {L: sum(v.values()) / len(v) for L, v in r.items() if len(v) == 13}
        res["ruler"][arm] = {"by_length": means,
                             "mean_4k_16k": (sum(means[L] for L in (4096, 8192, 16384)) / 3
                                             if all(L in means for L in (4096, 8192, 16384)) else None)}
        res["board"][arm] = board(d)

    def g(key, m):
        v = res["long"].get(key, {}).get(m)
        return v if v else None

    marks = res["marks"]
    s1 = g("read_64k", "gain_far")
    if s1:
        marks["S1"] = {"diff": s1["mean"], "lo": s1["lo"], "hi": s1["hi"],
                       "pass": s1["mean"] >= 0.010 and s1["lo"] > 0}
    nos, so = g("read_64k", "cost_nosum"), g("read_64k", "sumonly")
    if nos and so:
        marks["S2"] = {"cost_nosum_diff": nos["mean"], "sumonly_diff": so["mean"],
                       "pass": nos["mean"] > 0 and nos["lo"] > 0 and so["mean"] < 0 and so["hi"] < 0}
    f64, f16 = g("dca2_64k", "full"), g("dca2_16k", "full")
    if f64 and f16:
        marks["S3"] = {"full_64k_diff": f64["mean"], "full_16k_diff": f16["mean"],
                       "pass": f64["mean"] <= -0.010 and f16["mean"] <= -0.010}
    c = res["board"]["control"]
    for arm in ("read", "dca2"):
        b, rc, ra = res["board"][arm], res["ruler"]["control"]["mean_4k_16k"], res["ruler"][arm]["mean_4k_16k"]
        ok = {}
        if rc is not None and ra is not None:
            ok["ruler"] = abs(ra - rc) <= 0.02
        for k in ("blimp", "core7"):
            if k in b and k in c:
                ok[k] = abs(b[k] - c[k]) <= 1.0
        marks[f"S4_{arm}"] = {"ruler_diff": None if rc is None or ra is None else ra - rc,
                              "blimp_diff": b.get("blimp", float("nan")) - c.get("blimp", float("nan")),
                              "core7_diff": b.get("core7", float("nan")) - c.get("core7", float("nan")),
                              "pass": bool(ok) and all(ok.values()), "checked": ok}
    k3 = []
    for arm in ("read", "dca2"):
        for m in ("full", "gain_far"):
            v = g(f"{arm}_64k", m)
            if v:
                k3.append(abs(v["mean"]) <= 0.005)
    if len(k3) == 4:
        marks["K3"] = {"tripped": all(k3)}

    print("Long documents, arm − control, paired by document (nats per token; 95 % interval):")
    for key, ms in res["long"].items():
        for m in ("full", "gain_far", "cost_nosum", "cost_off", "cost_sumonly", "sumonly"):
            if m in ms:
                v = ms[m]
                print(f"  {key:10} {m:13} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    print("RULER 13-task mean (share found):")
    for arm, r in res["ruler"].items():
        print(f"  {arm:8} " + " ".join(f"{L // 1024}k {v:.3f}" for L, v in sorted(r["by_length"].items()))
              + (f"  mean 4k-16k {r['mean_4k_16k']:.3f}" if r["mean_4k_16k"] is not None else ""))
    print("Grammar (BLiMP, %), 7-task core mean (%), ARC-Easy (%), WikiText-2 (byte perplexity):")
    for arm, b in res["board"].items():
        print(f"  {arm:8} " + "  ".join(f"{k} {v:.3f}" for k, v in b.items()))
    print("Marks:", json.dumps(marks, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
