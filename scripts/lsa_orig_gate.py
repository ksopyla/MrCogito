"""E37 stage B, original LSA: the smoke-test gate (stdlib only; run on the node by lsa_orig_node.sh).

Reads the smoke run folder (lsa_orig_run.sh with LSA_SMOKE=1: trunk 200 steps, decay 100 steps, quick evals), prints a
JSON summary and exits 0 when the full run may start:
  * trunk and decay both finished, no NaN / inf loss, the trunk's loss fell (last three logs < 0.8 x the first), the
    decay's loss did not rise by more than 0.2;
  * the full run's estimate (tokens / the node's measured tokens per second, i.e. torchtitan's per-device rate x the
    GPUs in hw.json, + the evals, scaled per lane of lanes.json from the smoke evals) is at most MAX_HOURS (45) and its
    cost at most MAX_USD.
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
    # the hardware and the eval lanes the run script recorded (one lane per GPU, or two on one GPU)
    hw = json.loads((R / "hw.json").read_text()) if (R / "hw.json").exists() else {"gpus": 1}
    lanes = json.loads((R / "lanes.json").read_text())["lanes"] if (R / "lanes.json").exists() else []
    res["hw"] = hw
    # evals of the smoke (3 RULER documents per task, core 20 per task, one long-document row per source at 4k, 128k)
    def names(entry):   # a lane entry may be a glob (ruler_32k_*: one file per task)
        if "*" not in entry:
            return [entry]
        return sorted(p.name[:-5] for p in R.glob(f"{entry}.json")) or [entry]

    lanes = [[n for e in lane for n in names(e)] for lane in lanes]
    ev = {name: ("ok" if (R / f"{name}.json").exists() else "missing") for lane in lanes for name in lane}
    res["evals"] = ev
    if not ev or any(v != "ok" for v in ev.values()):
        why.append("some smoke evals missing (not gating): " + ", ".join(k for k, v in ev.items() if v != "ok"))

    def secs(name, key="seconds", overhead=90.0):
        """An eval's seconds less its fixed part (model build and load, task setup), which does not scale."""
        try:
            return max(0.0, float(json.loads((R / f"{name}.json").read_text()).get(key) or 0) - overhead)
        except Exception:
            return 0.0

    # full evals from the smoke ones, per lane (the lanes run side by side): RULER 100 / 3 documents per task, core
    # ~145 x (about 23k requests against 160), long-document ~15 x (8 rows at 4k-128k against 1 row at 4k + 128k)
    scale = {"eval_core": 145.0, "eval_longdoc": 15.0, "eval_val": 1.0}
    lane_s = [sum(secs(n) * scale.get(n, 100 / 3) for n in lane) for lane in lanes]
    eval_h = max(lane_s) / 3600 if lane_s and max(lane_s) > 0 else 4.0
    res["eval_lane_hours"] = [round(x / 3600, 2) for x in lane_s]
    # torchtitan logs tokens per second per device; the node's rate is that times the GPUs
    g = int(hw.get("gpus", 1))
    tt, td = res["trunk"].get("tps_median"), res["decay"].get("tps_median")
    if tt and td:
        train_h = TRUNK_TOKENS / (tt * g) / 3600 + DECAY_TOKENS / (td * g) / 3600
        total = train_h + eval_h + 0.3
        res["estimate"] = {"node_tps_trunk": tt * g, "node_tps_decay": td * g, "train_hours": round(train_h, 2),
                           "eval_hours": round(eval_h, 2), "total_hours": round(total, 2),
                           "usd": round(total * price, 1)}
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
