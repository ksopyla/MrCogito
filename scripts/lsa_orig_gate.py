"""E37 stage B, original LSA: the smoke-test gate (stdlib only; run on the node by lsa_orig_node.sh).

Reads the smoke run folder (lsa_orig_run.sh with LSA_SMOKE=1: trunk 200 steps, decay 100 steps, quick evals), prints a
JSON summary and exits 0 when the full run may start:
  * trunk and decay both finished, no NaN / inf loss, the trunk's loss fell (last three logs < 0.8 x the first), the
    decay's loss did not rise by more than 0.2;
  * the full run's estimate (tokens / measured tokens per second + the evals, scaled from the smoke evals) is at most
    MAX_HOURS (45) and its cost at most MAX_USD.
Eval failures are reported, not gating: the checkpoints go to S3 and the evals can be rerun.
Usage: python3 scripts/lsa_orig_gate.py RUN_DIR [PRICE_PER_HOUR]
"""
import json
import re
import statistics
import sys
from pathlib import Path

MAX_HOURS, MAX_USD = 45.0, 450.0
TRUNK_TOKENS, DECAY_TOKENS = 3.0e9, 0.4e9
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"step:\s*(\d+)\s+loss:\s*([-+\w.]+).*?memory:\s*([\d.]+)GiB.*?tps:\s*([\d,]+)")


def steps(log: Path) -> list[dict]:
    out = []
    if not log.exists():
        return out
    for raw in log.read_text(errors="replace").splitlines():
        m = LINE.search(ANSI.sub("", raw))
        if m:
            out.append({"step": int(m[1]), "loss": float(m[2]), "mem_gib": float(m[3]),
                        "tps": int(m[4].replace(",", ""))})
    return out


def main() -> int:
    R = Path(sys.argv[1])
    price = float(sys.argv[2]) if len(sys.argv) > 2 else 6.88
    res, ok, why = {"run": str(R)}, True, []
    for ph in ("trunk", "decay"):
        s = steps(R / f"{ph}.log")
        done = (R / ph / "DONE").exists()
        losses = [x["loss"] for x in s]
        tps = [x["tps"] for x in s[2:]] or [x["tps"] for x in s]
        res[ph] = {"done": done, "logged": len(s), "first_loss": losses[0] if losses else None,
                   "last_loss": losses[-1] if losses else None,
                   "tps_median": statistics.median(tps) if tps else None,
                   "peak_mem_gib": max((x["mem_gib"] for x in s), default=None), "last_step": s[-1]["step"] if s else None}
        if not done or len(s) < 3:
            ok = False
            why.append(f"{ph} did not finish ({len(s)} logged steps)")
            continue
        if any(not (abs(v) < 1e9) for v in losses):
            ok = False
            why.append(f"{ph} loss not finite")
        tail = statistics.mean(losses[-3:])
        if ph == "trunk" and not tail < 0.8 * losses[0]:
            ok = False
            why.append(f"trunk loss did not fall ({losses[0]:.3f} -> {tail:.3f})")
        if ph == "decay" and tail > losses[0] + 0.2:
            ok = False
            why.append(f"decay loss rose ({losses[0]:.3f} -> {tail:.3f})")
    # evals of the smoke (3 RULER documents per task, core 20 per task, one long-document row per source)
    ev = {}
    for name in ("eval_longdoc", "eval_val", "eval_core", "ruler_short", "ruler_long"):
        p = R / f"{name}.json"
        ev[name] = "ok" if p.exists() else "missing"
    res["evals"] = ev
    if any(v != "ok" for v in ev.values()):
        why.append("some smoke evals missing (not gating): " + ", ".join(k for k, v in ev.items() if v != "ok"))

    def secs(name, key="seconds", overhead=90.0):
        """An eval's seconds less its fixed part (model build and load, task setup), which does not scale."""
        try:
            return max(0.0, float(json.loads((R / f"{name}.json").read_text()).get(key) or 0) - overhead)
        except Exception:
            return 0.0

    # full evals from the smoke ones: RULER 100 / 3 documents (64k alone; 4k-32k ~ 15 x the 4k cost), core ~145 x
    # (23k requests / 160), long-document 8 rows x 5 lengths ~ 40 x; two lanes in parallel
    lane_long = secs("ruler_long") * 100 / 3
    lane_rest = secs("ruler_short") * 100 / 3 * 15 + secs("eval_core") * 145 + secs("eval_longdoc") * 20
    eval_h = max(lane_long, lane_rest) / 3600 if (lane_long or lane_rest) else 4.0
    tt, td = res["trunk"].get("tps_median"), res["decay"].get("tps_median")
    if tt and td:
        train_h = TRUNK_TOKENS / tt / 3600 + DECAY_TOKENS / td / 3600
        total = train_h + eval_h + 0.3
        res["estimate"] = {"train_hours": round(train_h, 2), "eval_hours": round(eval_h, 2),
                           "total_hours": round(total, 2), "usd": round(total * price, 1)}
        if total > MAX_HOURS or total * price > MAX_USD:
            ok = False
            why.append(f"estimate {total:.1f} h / ${total * price:.0f} over the limit ({MAX_HOURS} h, ${MAX_USD})")
    else:
        ok = False
        why.append("no tokens-per-second measurement")
    res["pass"], res["why"] = ok, why
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
