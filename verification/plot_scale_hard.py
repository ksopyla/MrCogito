#!/usr/bin/env python
"""Plot the exclusive-scope <10M harder-cell campaign (plus the seq128 easy end as a ghost).

Dedicated snake_case PNGs land in /opt/cursor/artifacts/. The r=32 D JSON is a
ratio-free reuse stub of seq256 D — it is not a second training curve.
In-progress chain D is parsed from its log if the JSON is not written yet.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HARD = Path("/opt/cursor/artifacts/scale_hard")
EASY_A = Path("/opt/cursor/artifacts/scale/long_data_span32.json")
EASY_3ARM = Path("/opt/cursor/artifacts/sym_probe_far_copy.json")
R16 = Path("/opt/cursor/artifacts/scale/r16_span32.json")
OUT_DIR = Path("/opt/cursor/artifacts")
CHANCE = 0.25
BAR = 0.95
BUDGET = 64_000

STEP_RE = re.compile(
    r"\[([ACD])\] step\s+(\d+)\s+examples\s+(\d+)\s+"
    r"train\s+([\d.]+)\s+eval CE\s+([\d.]+)\s+acc\s+([\d.]+)"
    r"(?:\s+lr\s+[\d.e+-]+\s+\(([\d.]+) s/step\))?"
)
META_SKIP = {"campaign_index.json", "campaign_meta.json", "winner_lr.json"}


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def traces(bundle: dict, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = bundle["results"][arm]["trace"]
    batch = bundle["config"].get("batch", 32)
    xs = np.array([p.get("examples", p["step"] * batch) for p in tr], dtype=float)
    steps = np.array([p["step"] for p in tr], dtype=float)
    ys = np.array([p["acc"] for p in tr], dtype=float)
    return xs, steps, ys


def wall_trace(bundle: dict, arm: str) -> tuple[np.ndarray, np.ndarray] | None:
    tr = bundle["results"][arm]["trace"]
    if not tr or "wall_s" not in tr[0]:
        return None
    walls = np.array([p["wall_s"] for p in tr], dtype=float)
    ys = np.array([p["acc"] for p in tr], dtype=float)
    return walls, ys


def acc_at(xs: np.ndarray, ys: np.ndarray, budget: float) -> tuple[float, bool]:
    if len(xs) == 0:
        return float("nan"), False
    idx = int(np.searchsorted(xs, budget, side="right") - 1)
    if idx < 0:
        return float(ys[0]), False
    return float(ys[idx]), bool(xs[idx] >= 0.9 * budget)


def examples_to_bar(xs: np.ndarray, ys: np.ndarray, bar: float = BAR) -> float | None:
    hit = np.where(ys >= bar)[0]
    if len(hit) == 0:
        return None
    return float(xs[hit[0]])


def is_reuse_stub(bundle: dict) -> bool:
    note = str(bundle.get("note") or "")
    if "reused" in note.lower():
        return True
    name = str(bundle.get("run_name") or "")
    cfg_r = bundle.get("config", {}).get("ratio")
    logged_r = None
    for arm, s in (bundle.get("summary") or {}).items():
        logged_r = s.get("r")
        break
    if name.endswith("_D") and cfg_r not in (None, logged_r) and logged_r is not None:
        return True
    return False


def is_lr_probe(bundle: dict, path: Path) -> bool:
    name = path.name if path else str(bundle.get("run_name") or "")
    return name.startswith("lr") and "cell_" not in name


def difficulty_key(bundle: dict) -> str:
    cfg = bundle["config"]
    name = str(bundle.get("run_name") or "")
    if cfg.get("task") == "chain":
        return f"chain h{cfg.get('hops')} seq{cfg['seq_len']}"
    key = f"seq{cfg['seq_len']} r={cfg['ratio']}"
    if "D9" in name:
        key += " D9"
    return key


def label_for(bundle: dict, arm: str, extra: str = "") -> str:
    cfg = bundle["config"]
    params = bundle["results"][arm]["params"] / 1e6
    stop = bundle["results"][arm].get("stop_reason", "")
    tag = f"{arm} {difficulty_key(bundle)} ({params:.2f}M)"
    if extra:
        tag += f" {extra}"
    if stop == "floor_patience":
        tag += " floor-kill"
    elif stop == "target_acc":
        tag += " ≥95%"
    return tag


def parse_log_bundle(log_path: Path, json_hint: Path | None = None) -> dict | None:
    if not log_path.exists():
        return None
    text = log_path.read_text()
    rows = []
    arm = None
    for m in STEP_RE.finditer(text):
        arm = m.group(1)
        rec = {
            "step": int(m.group(2)),
            "examples": int(m.group(3)),
            "ce_nats": float(m.group(5)),
            "acc": float(m.group(6)),
        }
        if m.group(7):
            rec["wall_s"] = float(m.group(2)) * float(m.group(7))
        rows.append(rec)
    if not rows or arm is None:
        return None
    hint = load(json_hint) if json_hint else None
    cfg = (hint or {}).get("config") or {}
    # Infer task/seq from the log header when JSON is missing.
    header = {}
    tm = re.search(r"task=(\w+) seq=(\d+)", text)
    if tm:
        header["task"] = tm.group(1)
        header["seq_len"] = int(tm.group(2))
    hm = re.search(r"slots=(\d+)", text)
    if hm:
        header["slots"] = int(hm.group(1))
    task = cfg.get("task") or header.get("task", "chain")
    seq = cfg.get("seq_len") or header.get("seq_len", 256)
    ratio = cfg.get("ratio", 8)
    hops = cfg.get("hops", 3 if task == "chain" else 2)
    last = rows[-1]
    floor_kill = "floor kill" in text
    stop = "floor_patience" if floor_kill else "in_progress"
    name = log_path.stem
    if arm == "A":
        params = 5_107_858
    elif "D9" in name:
        params = 4_974_227  # 9 decoder layers, param-matched to 5.11M A
    else:
        params = 2_267_977
    return {
        "run_name": log_path.stem,
        "task": task,
        "config": {
            "task": task,
            "seq_len": seq,
            "ratio": ratio,
            "hops": hops,
            "batch": 32,
            "steps": cfg.get("steps", 0),
        },
        "summary": {
            arm: {
                "arm": arm,
                "params": params,
                "examples_seen": last["examples"],
                "steps": last["step"],
                "acc": last["acc"],
                "ce": last["ce_nats"],
                "lr": 0.001,
                "seq": seq,
                "r": ratio,
                "task": task,
                "hops": hops,
                "stop_reason": stop,
            }
        },
        "results": {
            arm: {
                "arm": arm,
                "params": params,
                "examples_seen": last["examples"],
                "steps": last["step"],
                "acc": last["acc"],
                "stop_reason": stop,
                "final": last,
                "trace": rows,
            }
        },
        "in_progress": stop == "in_progress",
    }


def collect_hard() -> tuple[list[dict], list[dict], list[dict]]:
    cells: list[dict] = []
    probes: list[dict] = []
    stubs: list[dict] = []
    seen_names: set[str] = set()
    for p in sorted(HARD.glob("*.json")):
        if p.name in META_SKIP:
            continue
        if "stuck" in p.name or p.name.endswith("_slow.json"):
            continue
        b = load(p)
        if not b or "results" not in b:
            continue
        b["_path"] = str(p)
        if is_lr_probe(b, p):
            probes.append(b)
            continue
        if is_reuse_stub(b):
            stubs.append(b)
            continue
        cells.append(b)
        seen_names.add(p.stem)
    # Live logs without JSON yet (chain D, …).
    for log in sorted(HARD.glob("cell_*.log")):
        if log.stem in seen_names:
            continue
        if "stuck" in log.name or "slow" in log.name:
            continue
        parsed = parse_log_bundle(log)
        if parsed:
            parsed["_path"] = str(log)
            cells.append(parsed)
            seen_names.add(log.stem)
    return cells, probes, stubs


def style_for(arm: str, in_progress: bool = False) -> dict:
    base = {
        "A": {"color": "#1f4e79", "ls": "-", "lw": 2.2},
        "C": {"color": "#2ca02c", "ls": "--", "lw": 1.8},
        "D": {"color": "#d62728", "ls": ":", "lw": 2.2},
    }[arm]
    if in_progress:
        base = {**base, "lw": 1.6, "alpha": 0.85}
    return base


def cell_color(bundle: dict, arm: str) -> str:
    key = difficulty_key(bundle)
    palette = {
        ("A", "seq256 r=8"): "#1f77b4",
        ("D", "seq256 r=8"): "#d62728",
        ("C", "seq256 r=8"): "#2ca02c",
        ("A", "seq256 r=32"): "#08306b",
        ("D", "seq256 r=32"): "#843c39",
        ("A", "chain h3 seq256"): "#6a3d9a",
        ("D", "chain h3 seq256"): "#e6550d",
        ("C", "chain h3 seq256"): "#74c476",
        ("A", "seq512 r=8"): "#0b3d91",
        ("D", "seq512 r=8"): "#a50f15",
        ("C", "seq512 r=8"): "#41ab5d",
        ("D", "seq512 r=8 D9"): "#fb6a4a",
        ("A", "seq256 r=16"): "#2171b5",
        ("A", "seq1024 r=8"): "#081d58",
        ("A", "chain h2 seq512"): "#9e9ac8",
        ("D", "chain h2 seq512"): "#fd8d3c",
    }
    return palette.get((arm, key), style_for(arm)["color"])


def write_inventory(cells: list[dict], probes: list[dict], stubs: list[dict]) -> Path:
    rows = []
    for group, items in (("cell", cells), ("lr_probe", probes), ("reuse_stub", stubs)):
        for b in items:
            for arm, s in (b.get("summary") or b.get("results") or {}).items():
                rec = {
                    "group": group,
                    "file": Path(b.get("_path", "")).name,
                    "run_name": b.get("run_name"),
                    "in_progress": bool(b.get("in_progress")),
                    "arm": arm,
                    "params": s.get("params"),
                    "examples_seen": s.get("examples_seen", s.get("final", {}).get("examples")),
                    "steps": s.get("steps", s.get("final", {}).get("step")),
                    "acc": s.get("acc", s.get("final", {}).get("acc")),
                    "ce": s.get("ce", s.get("final", {}).get("ce_nats")),
                    "lr": s.get("lr", b.get("config", {}).get("lr")),
                    "seq": s.get("seq", b.get("config", {}).get("seq_len")),
                    "r": s.get("r", b.get("config", {}).get("ratio")),
                    "task": s.get("task", b.get("config", {}).get("task")),
                    "hops": s.get("hops", b.get("config", {}).get("hops")),
                    "stop_reason": s.get("stop_reason"),
                    "hit_95": float(s.get("acc", s.get("final", {}).get("acc") or 0)) >= BAR,
                }
                if "trace" in (b.get("results") or {}).get(arm, {}):
                    xs, _, ys = traces(b, arm)
                    rec["acc_at_64k"], rec["reached_64k"] = acc_at(xs, ys, BUDGET)
                    rec["examples_to_95"] = examples_to_bar(xs, ys)
                rows.append(rec)
    payload = {
        "bar": BAR,
        "matched_budget_examples": BUDGET,
        "model_target": "<10M",
        "device": "cpu",
        "runs": rows,
    }
    path = OUT_DIR / "harder_campaign_inventory.json"
    path.write_text(json.dumps(payload, indent=2))
    (HARD / "harder_campaign_inventory.json").write_text(json.dumps(payload, indent=2))
    return path


def plot_curves(cells: list[dict], easy_a, easy_3, r16, x_key: str, outfile: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0, label="95% bar")
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0, label="chance (1/4)")

    if easy_a and "A" in easy_a.get("results", {}):
        xs, steps, ys = traces(easy_a, "A")
        x = steps if x_key == "steps" else xs
        ax.plot(
            x,
            ys,
            color="0.55",
            ls="-.",
            lw=1.3,
            label="A seq128 r=8 easy (1.35M, ref)",
        )
    if r16 and "A" in r16.get("results", {}):
        xs, steps, ys = traces(r16, "A")
        x = steps if x_key == "steps" else xs
        ax.plot(x, ys, color="0.7", ls="-.", lw=1.0, label="A seq128 r=16 easy (1.35M, ref)")

    for b in cells:
        in_prog = bool(b.get("in_progress"))
        for arm in b["results"]:
            xs, steps, ys = traces(b, arm)
            x = steps if x_key == "steps" else xs
            extra = "(in progress)" if in_prog else ""
            ax.plot(
                x,
                ys,
                color=cell_color(b, arm),
                ls=style_for(arm, in_prog)["ls"],
                lw=style_for(arm, in_prog)["lw"],
                marker="o" if (not in_prog and len(x) <= 16) else None,
                ms=3.5,
                label=label_for(b, arm, extra),
            )

    ax.set_xlabel("optimizer steps" if x_key == "steps" else "examples seen")
    ax.set_ylabel("eval accuracy (supervised tokens)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    fig.savefig(HARD / outfile.name, dpi=140)
    plt.close(fig)


def plot_difficulty(cells: list[dict], stubs: list[dict], easy_a, easy_3, outfile: Path) -> None:
    """Matched-budget accuracy + examples-to-95%."""
    records = []  # (arm, diff, acc64, reached, examples_to_95, params, note)

    def add(bundle, arm, note=""):
        xs, _, ys = traces(bundle, arm)
        acc64, reached = acc_at(xs, ys, BUDGET)
        records.append(
            (
                arm,
                difficulty_key(bundle),
                acc64,
                reached,
                examples_to_bar(xs, ys),
                bundle["results"][arm]["params"],
                note,
            )
        )

    if easy_a and "A" in easy_a.get("results", {}):
        add(easy_a, "A", "easy")
    if easy_3:
        for arm in ("C", "D"):
            if arm in easy_3.get("results", {}):
                add(easy_3, arm, "easy")
    for b in cells:
        for arm in b["results"]:
            note = "in progress" if b.get("in_progress") else ""
            add(b, arm, note)
    # r=32 D reuse: same exam for D (ratio is unused). Show as a hatched D bar.
    r8_d = next(
        (b for b in cells if b.get("config", {}).get("ratio") == 8 and "D" in b.get("results", {})
         and b.get("config", {}).get("task") == "far_copy"),
        None,
    )
    if r8_d is not None:
        xs, _, ys = traces(r8_d, "D")
        acc64, reached = acc_at(xs, ys, BUDGET)
        records.append(
            (
                "D",
                "seq256 r=32",
                acc64,
                reached,
                examples_to_bar(xs, ys),
                r8_d["results"]["D"]["params"],
                "D reused (ratio-free)",
            )
        )

    diffs = []
    for _, d, *_ in records:
        if d not in diffs:
            diffs.append(d)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    ax = axes[0]
    x = np.arange(len(diffs))
    width = 0.26
    arm_bar_color = {"A": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
    for i, arm in enumerate("ACD"):
        vals = []
        hatches = []
        for d in diffs:
            hits = [(acc, reached, note) for a, dd, acc, reached, _, _, note in records if a == arm and dd == d]
            if not hits:
                vals.append(np.nan)
                hatches.append(None)
            else:
                acc, reached, note = hits[-1]
                vals.append(acc)
                hatches.append("///" if (not reached or "reused" in note or "progress" in note) else None)
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=f"arm {arm}",
            color=arm_bar_color[arm],
            zorder=3,
        )
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(diffs, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel(f"accuracy at {BUDGET:,} examples")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Matched budget (hatch = short of 64k, reuse, or in-progress)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    for i, arm in enumerate("AD"):
        vals = []
        for d in diffs:
            hits = [e95 for a, dd, _, _, e95, _, _ in records if a == arm and dd == d]
            if not hits or hits[-1] is None:
                vals.append(np.nan)
            else:
                vals.append(hits[-1] / 1000.0)
        ax2.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=f"arm {arm} hit ≥95%",
            color=arm_bar_color[arm],
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(diffs, rotation=22, ha="right", fontsize=8)
    ax2.set_ylabel("examples to ≥95% (thousands)")
    ax2.set_title("Time-to-bar (missing bar = never hit 95%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Harder cells  ·  exclusive-scope <10M  ·  95% bar  ·  CPU",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    fig.savefig(HARD / outfile.name, dpi=140)
    plt.close(fig)


def plot_lr(probes: list[dict], outfile: Path) -> None:
    if not probes:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0, label="chance")
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0, label="95% bar")
    order = sorted(probes, key=lambda b: b["config"].get("lr", 0))
    for b in order:
        lr = b["config"]["lr"]
        xs, _, ys = traces(b, "A")
        ax.plot(xs, ys, lw=2.0, marker="o", ms=4, label=f"A lr={lr:g}  stop={b['results']['A'].get('stop_reason')}")
    ax.set_xlabel("examples seen")
    ax.set_ylabel("eval accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("LR probe on seq256 r=8 far_copy (800 steps / 25.6k examples)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    fig.savefig(HARD / outfile.name, dpi=140)
    plt.close(fig)


def plot_params_vs_seq(cells: list[dict], easy_a, easy_3, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for arm, marker in (("A", "o"), ("D", "s")):
        pts = []
        if arm == "A" and easy_a:
            acc = easy_a["results"]["A"]["final"]["acc"]
            if acc >= BAR:
                pts.append((128, easy_a["results"]["A"]["params"] / 1e6))
        if arm == "D" and easy_3:
            acc = easy_3["results"]["D"]["final"]["acc"]
            if acc >= BAR:
                pts.append((128, easy_3["results"]["D"]["params"] / 1e6))
        for b in cells:
            if b["config"].get("task") != "far_copy":
                continue
            if arm not in b["results"]:
                continue
            rec = b["results"][arm]
            acc = rec.get("acc", rec.get("final", {}).get("acc", 0))
            if acc >= BAR:
                pts.append((b["config"]["seq_len"], rec["params"] / 1e6))
        if pts:
            pts = sorted(set(pts))
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker + "-",
                lw=1.8,
                label=f"arm {arm} ≥95% (measured cells only)",
            )
    ax.set_xlabel("seq_len (far_copy, span 32, r=8 cells that actually hit 95%)")
    ax.set_ylabel("params (M)")
    ax.set_title("Params vs seq at ≥95%  (r=32 A missed 95%; width-matched D missed seq512)")
    ax.set_ylim(0, 10)
    ax.axhline(10, color="0.5", ls="--", lw=0.8, label="<10M target")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    fig.savefig(HARD / outfile.name, dpi=140)
    plt.close(fig)


def plot_combined(cells, probes, stubs, easy_a, easy_3, r16, outfile: Path) -> None:
    """Two-panel poster used by the campaign runner."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    ax = axes[0]
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0, label="95% bar")
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0, label="chance")
    if easy_a and "A" in easy_a.get("results", {}):
        xs, _, ys = traces(easy_a, "A")
        ax.plot(xs, ys, color="0.55", ls="-.", lw=1.3, label="A seq128 r=8 easy (1.35M, ref)")
    for b in cells:
        for arm in b["results"]:
            xs, _, ys = traces(b, arm)
            extra = "(in progress)" if b.get("in_progress") else ""
            ax.plot(
                xs,
                ys,
                color=cell_color(b, arm),
                ls=style_for(arm, bool(b.get("in_progress")))["ls"],
                lw=style_for(arm, bool(b.get("in_progress")))["lw"],
                label=label_for(b, arm, extra),
            )
    ax.set_xlabel("examples seen")
    ax.set_ylabel("eval accuracy (supervised tokens)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Accuracy vs examples")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Right panel: 64k matched budget only (harder cells + easy ghost).
    records = []
    if easy_a:
        xs, _, ys = traces(easy_a, "A")
        acc, _ = acc_at(xs, ys, BUDGET)
        records.append(("A", "seq128 r=8", acc))
    if easy_3:
        for arm in ("C", "D"):
            if arm in easy_3.get("results", {}):
                xs, _, ys = traces(easy_3, arm)
                acc, _ = acc_at(xs, ys, BUDGET)
                records.append((arm, "seq128 r=8", acc))
    for b in cells:
        for arm in b["results"]:
            xs, _, ys = traces(b, arm)
            acc, _ = acc_at(xs, ys, BUDGET)
            records.append((arm, difficulty_key(b), acc))
    r8_d = next(
        (
            b
            for b in cells
            if b.get("config", {}).get("task") == "far_copy"
            and b.get("config", {}).get("ratio") == 8
            and "D" in b.get("results", {})
        ),
        None,
    )
    if r8_d is not None:
        xs, _, ys = traces(r8_d, "D")
        acc, _ = acc_at(xs, ys, BUDGET)
        records.append(("D", "seq256 r=32", acc))
    diffs = []
    for _, d, _ in records:
        if d not in diffs:
            diffs.append(d)
    ax2 = axes[1]
    x = np.arange(len(diffs))
    width = 0.24
    arm_bar_color = {"A": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
    for i, arm in enumerate("ACD"):
        vals = []
        for d in diffs:
            hits = [acc for a, dd, acc in records if a == arm and dd == d]
            vals.append(hits[-1] if hits else np.nan)
        ax2.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=f"arm {arm}",
            color=arm_bar_color[arm],
        )
    ax2.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax2.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(diffs, rotation=22, ha="right", fontsize=8)
    ax2.set_ylabel(f"accuracy at {BUDGET:,} examples (or last ckpt)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("Difficulty vs accuracy (matched example budget)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Exclusive-scope concept slots <10M  ·  95% bar  ·  CPU harder-cell campaign",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    fig.savefig(HARD / outfile.name, dpi=140)
    plt.close(fig)


def plot_steps_sizes_acc(cells: list[dict], easy_a, easy_3, outfile: Path) -> None:
    """The comparison the scaling-law goal asked for: steps, sizes, accuracies."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))

    ax = axes[0]
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    for b in cells:
        for arm in b["results"]:
            _, steps, ys = traces(b, arm)
            ax.plot(
                steps,
                ys,
                color=cell_color(b, arm),
                ls=style_for(arm, bool(b.get("in_progress")))["ls"],
                lw=1.8,
                label=label_for(b, arm, "(live)" if b.get("in_progress") else ""),
            )
    ax.set_xlabel("optimizer steps")
    ax.set_ylabel("eval accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Accuracy vs steps")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax2.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    for b in cells:
        for arm in b["results"]:
            rec = b["results"][arm]
            acc = rec.get("acc", rec.get("final", {}).get("acc", 0))
            params = rec["params"] / 1e6
            ax2.scatter(
                params,
                acc,
                s=90,
                c=cell_color(b, arm),
                marker={"A": "o", "C": "D", "D": "s"}[arm],
                zorder=3,
            )
            ax2.annotate(
                difficulty_key(b),
                (params, acc),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
    if easy_a:
        acc = easy_a["results"]["A"]["final"]["acc"]
        ax2.scatter(easy_a["results"]["A"]["params"] / 1e6, acc, s=70, c="0.5", marker="o")
        ax2.annotate("A seq128", (1.35, acc), textcoords="offset points", xytext=(5, 4), fontsize=7, color="0.4")
    ax2.set_xlabel("params (M)")
    ax2.set_ylabel("final eval accuracy")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("Size vs accuracy (<10M)")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for b in cells:
        if b["config"].get("task") != "far_copy":
            continue
        for arm in b["results"]:
            rec = b["results"][arm]
            acc = rec.get("acc", rec.get("final", {}).get("acc", 0))
            if acc < BAR:
                continue
            xs, _, ys = traces(b, arm)
            e95 = examples_to_bar(xs, ys)
            if e95 is None:
                continue
            ax3.scatter(
                b["config"]["seq_len"],
                e95 / 1000.0,
                s=90,
                c=cell_color(b, arm),
                marker={"A": "o", "C": "D", "D": "s"}[arm],
                zorder=3,
            )
            ax3.annotate(
                f"{arm} {rec['params']/1e6:.2f}M",
                (b["config"]["seq_len"], e95 / 1000.0),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
    ax3.set_xlabel("seq_len (far_copy cells that hit 95%)")
    ax3.set_ylabel("examples to ≥95% (thousands)")
    ax3.set_title("Data to 95% vs length")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "Exclusive-scope concept slots <10M · steps · sizes · accuracies · 95% bar",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    fig.savefig(HARD / outfile.name, dpi=140)
    plt.close(fig)


def plot_compute_matched(cells: list[dict], outfile: Path) -> None:
    """Same-compute view: accuracy vs wall-clock, plus params×data for 95% hits."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    ax = axes[0]
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    for b in cells:
        for arm in b["results"]:
            wt = wall_trace(b, arm)
            if wt is None:
                continue
            walls, ys = wt
            ax.plot(
                walls / 60.0,
                ys,
                color=cell_color(b, arm),
                ls=style_for(arm, bool(b.get("in_progress")))["ls"],
                lw=1.8,
                label=label_for(b, arm, "(live)" if b.get("in_progress") else ""),
            )
    ax.set_xlabel("wall-clock (minutes)")
    ax.set_ylabel("eval accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Accuracy vs compute (wall)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    rows = []
    for b in cells:
        for arm in b["results"]:
            rec = b["results"][arm]
            acc = rec.get("acc", rec.get("final", {}).get("acc", 0))
            xs, _, ys = traces(b, arm)
            e95 = examples_to_bar(xs, ys)
            wt = wall_trace(b, arm)
            wall95 = None
            if wt is not None and e95 is not None:
                walls, wys = wt
                hit = np.where(ys >= BAR)[0]
                if len(hit):
                    wall95 = float(walls[hit[0]])
            rows.append(
                {
                    "label": f"{arm} {difficulty_key(b)}",
                    "arm": arm,
                    "params_m": rec["params"] / 1e6,
                    "acc": acc,
                    "e95": e95,
                    "wall95_min": None if wall95 is None else wall95 / 60.0,
                }
            )
    labels = [r["label"] for r in rows if r["e95"] is not None]
    e95s = [r["e95"] / 1000.0 for r in rows if r["e95"] is not None]
    colors = [{"A": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}[r["arm"]] for r in rows if r["e95"] is not None]
    ax2.barh(range(len(labels)), e95s, color=colors)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("examples to ≥95% (thousands)")
    ax2.set_title("Data to 95% (missing = not yet)")
    ax2.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Same-compute / same-data view  ·  exclusive-scope <10M  ·  95% bar", fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    fig.savefig(HARD / outfile.name, dpi=140)
    plt.close(fig)
    (OUT_DIR / "harder_compute_matched.json").write_text(json.dumps(rows, indent=2))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HARD.mkdir(parents=True, exist_ok=True)
    cells, probes, stubs = collect_hard()
    easy_a = load(EASY_A)
    easy_3 = load(EASY_3ARM)
    r16 = load(R16)

    inv = write_inventory(cells, probes, stubs)
    print(f"wrote {inv} ({sum(len(b.get('results') or {}) for b in cells + probes + stubs)} rows)", flush=True)

    plot_curves(
        cells,
        easy_a,
        easy_3,
        r16,
        "examples",
        OUT_DIR / "harder_accuracy_vs_examples.png",
        "Harder cells: accuracy vs examples  (A 5.11M / C,D 2.27M, exclusive, CPU)",
    )
    plot_curves(
        cells,
        easy_a,
        easy_3,
        r16,
        "steps",
        OUT_DIR / "harder_accuracy_vs_steps.png",
        "Harder cells: accuracy vs steps  (batch 32, warmup-constant; LR searched per cell)",
    )
    plot_difficulty(cells, stubs, easy_a, easy_3, OUT_DIR / "harder_difficulty_vs_accuracy.png")
    plot_lr(probes, OUT_DIR / "harder_lr_probe.png")
    plot_params_vs_seq(cells, easy_a, easy_3, OUT_DIR / "harder_params_vs_max_seq.png")
    plot_combined(
        cells,
        probes,
        stubs,
        easy_a,
        easy_3,
        r16,
        OUT_DIR / "concept_slot_scaling_frontier.png",
    )
    # Keep the names the previous campaign script advertised.
    for src, dst in (
        ("harder_accuracy_vs_examples.png", "accuracy_vs_examples.png"),
        ("harder_accuracy_vs_examples.png", "concept_slot_harder_learning_curves.png"),
        ("harder_difficulty_vs_accuracy.png", "difficulty_vs_accuracy.png"),
        ("harder_params_vs_max_seq.png", "params_vs_max_seq.png"),
    ):
        data = (OUT_DIR / src).read_bytes()
        (OUT_DIR / dst).write_bytes(data)
        (HARD / dst).write_bytes(data)

    plot_steps_sizes_acc(cells, easy_a, easy_3, OUT_DIR / "harder_steps_sizes_accuracies.png")
    plot_compute_matched(cells, OUT_DIR / "harder_compute_matched.png")

    print(f"wrote {OUT_DIR / 'harder_accuracy_vs_examples.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'harder_accuracy_vs_steps.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'harder_difficulty_vs_accuracy.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'harder_lr_probe.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'harder_params_vs_max_seq.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'concept_slot_scaling_frontier.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'harder_steps_sizes_accuracies.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'harder_compute_matched.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
