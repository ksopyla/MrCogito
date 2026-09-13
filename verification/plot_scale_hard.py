#!/usr/bin/env python
"""Plot the exclusive-scope <10M HARDER concept-slot scaling campaign.

Writes:
  /opt/cursor/artifacts/concept_slot_harder_learning_curves.png
  /opt/cursor/artifacts/concept_slot_scaling_frontier.png
and copies both into /opt/cursor/artifacts/scale_hard/.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HARD = Path("/opt/cursor/artifacts/scale_hard")
EASY_A = Path("/opt/cursor/artifacts/scale/long_data_span32.json")
EASY_R16 = Path("/opt/cursor/artifacts/scale/r16_span32.json")
EASY_3ARM = Path("/opt/cursor/artifacts/sym_probe_far_copy.json")
OUT_DIR = Path("/opt/cursor/artifacts")
CHANCE = 0.25
BAR = 0.95
BATCH_DEFAULT = 32

ARM_COLOR = {"A": "#0072B2", "C": "#009E73", "D": "#D55E00"}
ARM_MARKER = {"A": "o", "C": "^", "D": "s"}

# Harder-cell linestyles (seq / r / task).
HARD_STYLE = {
    ("far_copy", 256, 8): "-",
    ("far_copy", 256, 32): "--",
    ("chain", 256, 8): "-.",
}

SKIP_JSON = {
    "campaign_index.json",
    "campaign_meta.json",
    "winner_lr.json",
}

STEP_RE = re.compile(
    r"\[([ACD])\] step\s+(\d+)\s+examples\s+(\d+).*acc\s+([0-9.]+)"
)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def examples_of(point: dict, batch: int) -> int:
    if "examples" in point:
        return int(point["examples"])
    return int(point["step"]) * batch


def traces(bundle: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    batch = int(bundle.get("config", {}).get("batch", BATCH_DEFAULT))
    tr = bundle["results"][arm]["trace"]
    xs = np.array([examples_of(p, batch) for p in tr], dtype=float)
    ys = np.array([p["acc"] for p in tr], dtype=float)
    return xs, ys


def first_hit_examples(xs: np.ndarray, ys: np.ndarray, bar: float = BAR) -> float | None:
    above = np.where(ys >= bar)[0]
    if above.size == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(xs[0])
    x0, y0 = float(xs[i - 1]), float(ys[i - 1])
    x1, y1 = float(xs[i]), float(ys[i])
    if y1 <= y0:
        return x1
    frac = (bar - y0) / (y1 - y0)
    return x0 + frac * (x1 - x0)


def parse_log_trace(path: Path) -> tuple[str | None, np.ndarray, np.ndarray]:
    if not path.exists():
        return None, np.array([]), np.array([])
    xs, ys, arm = [], [], None
    for line in path.read_text().splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        arm = m.group(1)
        xs.append(int(m.group(3)))
        ys.append(float(m.group(4)))
    return arm, np.array(xs, dtype=float), np.array(ys, dtype=float)


def is_lr_probe(name: str) -> bool:
    return name.startswith("lr") and "cell_" not in name


def is_r32_d_reuse(bundle: dict, path: Path) -> bool:
    if path.name != "cell_seq256_r32_D.json":
        return False
    note = str(bundle.get("note", "")).lower()
    r = bundle.get("results", {}).get("D", {}).get("r")
    return "reused" in note or r == 8


def collect_hard_cells() -> list[tuple[Path, dict]]:
    rows = []
    for p in sorted(HARD.glob("*.json")):
        if p.name in SKIP_JSON or is_lr_probe(p.name):
            continue
        b = load(p)
        if not b or "results" not in b:
            continue
        if is_r32_d_reuse(b, p):
            continue
        rows.append((p, b))
    return rows


def in_progress_logs() -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Return (run_name, arm, xs, ys) for logs whose JSON is missing or empty."""
    out = []
    for log in sorted(HARD.glob("cell_*.log")):
        json_path = log.with_suffix(".json")
        if json_path.exists():
            b = load(json_path)
            if b and b.get("results"):
                continue
        arm, xs, ys = parse_log_trace(log)
        if arm is None or xs.size == 0:
            continue
        out.append((log.stem, arm, xs, ys))
    return out


def cell_label(bundle: dict, arm: str) -> str:
    cfg = bundle["config"]
    task = cfg.get("task", "far_copy")
    seq = cfg.get("seq_len")
    r = cfg.get("ratio")
    hops = cfg.get("hops")
    params = bundle["results"][arm]["params"] / 1e6
    if task == "chain":
        return f"{arm} chain h={hops} seq{seq} ({params:.2f}M)"
    return f"{arm} seq{seq} r={r} ({params:.2f}M)"


def linestyle_for(bundle: dict) -> str:
    cfg = bundle["config"]
    key = (cfg.get("task", "far_copy"), int(cfg.get("seq_len", 0)), int(cfg.get("ratio", 0)))
    return HARD_STYLE.get(key, ":")


def difficulty_slot(task: str, seq: int, r: int, hops: int) -> tuple[float, str]:
    if task == "chain":
        return 4.0, f"chain h={hops}\nseq{seq} r={r}"
    if seq == 128 and r == 8:
        return 0.0, "seq128 r=8\n(easy)"
    if seq == 128 and r == 16:
        return 1.0, "seq128 r=16\n(easy)"
    if seq == 256 and r == 8:
        return 2.0, "seq256 r=8"
    if seq == 256 and r == 32:
        return 3.0, "seq256 r=32"
    if seq == 512:
        return 5.0, f"seq512 r={r}"
    return 6.0, f"{task} seq{seq} r={r}"


def plot_learning_curves(hard: list[tuple[Path, dict]], running: list) -> Path:
    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    ax.axhline(BAR, color="0.25", ls="--", lw=1.15, zorder=1)
    ax.axhline(CHANCE, color="0.45", ls=":", lw=1.15, zorder=1)
    ax.text(2000, BAR + 0.018, "95% bar", ha="left", va="bottom", fontsize=9, color="0.25")
    ax.text(2000, CHANCE + 0.018, "chance 25%", ha="left", va="bottom", fontsize=9, color="0.35")

    for _, b in hard:
        for arm, rec in b["results"].items():
            xs, ys = traces(b, arm)
            ax.plot(
                xs,
                ys,
                linestyle_for(b),
                color=ARM_COLOR[arm],
                lw=2.15,
                marker=ARM_MARKER[arm],
                ms=4.5,
                markevery=max(1, len(xs) // 10),
                label=cell_label(b, arm),
            )
            last_x, last_y = float(xs[-1]), float(ys[-1])
            stop = rec.get("stop_reason", "")
            if last_y >= BAR:
                ax.scatter([last_x], [last_y], marker=ARM_MARKER[arm], s=55, color=ARM_COLOR[arm], zorder=5)
            elif stop == "floor_patience":
                ax.scatter([last_x], [last_y], marker="x", s=70, color=ARM_COLOR[arm], zorder=5)
            else:
                ax.scatter([last_x], [last_y], marker=ARM_MARKER[arm], s=50, facecolors="white", edgecolors=ARM_COLOR[arm], linewidths=1.4, zorder=5)

    for name, arm, xs, ys in running:
        ax.plot(
            xs,
            ys,
            ":",
            color=ARM_COLOR[arm],
            lw=1.8,
            marker=ARM_MARKER[arm],
            ms=5,
            label=f"{arm} {name} (in progress)",
        )
        ax.scatter(
            [float(xs[-1])],
            [float(ys[-1])],
            marker="P",
            s=90,
            color=ARM_COLOR[arm],
            zorder=6,
        )
        ax.annotate(
            "in progress",
            (float(xs[-1]), float(ys[-1])),
            textcoords="offset points",
            xytext=(10, 8),
            fontsize=8,
            color=ARM_COLOR[arm],
        )

    ax.set_xlabel("unique examples seen")
    ax.set_ylabel("eval accuracy on supervised tokens")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(left=0)
    ax.set_title(
        "HARDER campaign · accuracy vs examples\n"
        "A exclusive slots (5.11M) vs C no-array vs D dense  ·  CPU, n_symbols=4, chance=25%",
        fontsize=12,
    )
    handles, labels = ax.get_legend_handles_labels()
    extra = [
        Line2D([0], [0], color="0.3", ls="-", lw=1.8, label="linestyle: seq256 r=8"),
        Line2D([0], [0], color="0.3", ls="--", lw=1.8, label="linestyle: seq256 r=32"),
        Line2D([0], [0], color="0.3", ls="-.", lw=1.8, label="linestyle: chain hops=3"),
    ]
    ax.legend(handles + extra, labels + [e.get_label() for e in extra], fontsize=8, loc="lower right", ncol=1, framealpha=0.92)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    dest = OUT_DIR / "concept_slot_harder_learning_curves.png"
    fig.savefig(dest, dpi=160)
    shutil.copy2(dest, HARD / dest.name)
    plt.close(fig)
    return dest


def frontier_points(hard: list[tuple[Path, dict]], running: list) -> list[dict]:
    pts: list[dict] = []

    easy_a = load(EASY_A)
    if easy_a and "A" in easy_a.get("results", {}):
        xs, ys = traces(easy_a, "A")
        hit = first_hit_examples(xs, ys)
        rec = easy_a["results"]["A"]
        pts.append(
            {
                "arm": "A",
                "task": "far_copy",
                "seq": 128,
                "r": 8,
                "hops": 2,
                "params": rec["params"],
                "hit": hit is not None,
                "examples": hit if hit is not None else float(xs[-1]),
                "acc": float(ys[-1]),
                "incomplete": False,
                "label": "A 1.35M",
            }
        )

    easy_r16 = load(EASY_R16)
    if easy_r16 and "A" in easy_r16.get("results", {}):
        xs, ys = traces(easy_r16, "A")
        hit = first_hit_examples(xs, ys)
        rec = easy_r16["results"]["A"]
        pts.append(
            {
                "arm": "A",
                "task": "far_copy",
                "seq": 128,
                "r": 16,
                "hops": 2,
                "params": rec["params"],
                "hit": hit is not None,
                "examples": hit if hit is not None else float(xs[-1]),
                "acc": float(ys[-1]),
                "incomplete": False,
                "label": "A 1.35M",
            }
        )

    easy3 = load(EASY_3ARM)
    if easy3:
        for arm in ("C", "D"):
            if arm not in easy3.get("results", {}):
                continue
            rec = easy3["results"][arm]
            xs, ys = traces(easy3, arm)
            hit = first_hit_examples(xs, ys)
            pts.append(
                {
                    "arm": arm,
                    "task": "far_copy",
                    "seq": 128,
                    "r": 8,
                    "hops": 2,
                    "params": rec["params"],
                    "hit": hit is not None,
                    "examples": hit if hit is not None else float(xs[-1]),
                    "acc": float(ys[-1]),
                    "incomplete": False,
                    "label": f"{arm} {rec['params']/1e6:.2f}M",
                }
            )

    for _, b in hard:
        cfg = b["config"]
        for arm, rec in b["results"].items():
            xs, ys = traces(b, arm)
            hit = first_hit_examples(xs, ys)
            pts.append(
                {
                    "arm": arm,
                    "task": cfg.get("task", "far_copy"),
                    "seq": int(cfg["seq_len"]),
                    "r": int(cfg.get("ratio", 0)),
                    "hops": int(cfg.get("hops", 2)),
                    "params": rec["params"],
                    "hit": hit is not None,
                    "examples": hit if hit is not None else float(xs[-1]),
                    "acc": float(ys[-1]),
                    "incomplete": False,
                    "label": f"{arm} {rec['params']/1e6:.2f}M",
                }
            )

    for name, arm, xs, ys in running:
        # chain h3 D in flight
        task = "chain" if "chain" in name else "far_copy"
        seq = 512 if "512" in name else 256
        r = 32 if "r32" in name else 8
        hops = 3 if "h3" in name or "chain" in name else 2
        hit = first_hit_examples(xs, ys) if xs.size else None
        pts.append(
            {
                "arm": arm,
                "task": task,
                "seq": seq,
                "r": r,
                "hops": hops,
                "params": 2267977 if arm in "CD" else 5107858,
                "hit": hit is not None,
                "examples": hit if hit is not None else float(xs[-1]),
                "acc": float(ys[-1]),
                "incomplete": True,
                "label": f"{arm} in progress",
            }
        )
    return pts


def plot_frontier(pts: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(11.6, 6.7))
    slots: dict[str, float] = {}
    for p in pts:
        x, lab = difficulty_slot(p["task"], p["seq"], p["r"], p["hops"])
        slots[lab] = x

    jitter = {"A": -0.14, "C": 0.0, "D": 0.14}
    for p in pts:
        x0, lab = difficulty_slot(p["task"], p["seq"], p["r"], p["hops"])
        x = x0 + jitter[p["arm"]]
        y = max(p["examples"], 1.0)
        face = ARM_COLOR[p["arm"]] if p["hit"] and not p["incomplete"] else "white"
        marker = "P" if p["incomplete"] else ARM_MARKER[p["arm"]]
        ax.scatter(
            [x],
            [y],
            marker=marker,
            s=120 if p["incomplete"] else 95,
            facecolors=face,
            edgecolors=ARM_COLOR[p["arm"]],
            linewidths=1.7,
            zorder=4,
        )
        acc_s = f"{100 * p['acc']:.0f}%"
        if p["incomplete"]:
            text = f"{p['label']}\n{acc_s} in flight"
        elif p["hit"]:
            text = f"{p['label']}\n95% @ {p['examples']/1000:.0f}k"
        else:
            text = f"{p['label']}\n{acc_s} no-95%"
        ax.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=(0, 9 if p["arm"] != "C" else -28),
            ha="center",
            fontsize=7.4,
            color=ARM_COLOR[p["arm"]],
        )

    ordered = sorted(slots.items(), key=lambda kv: kv[1])
    ax.set_xticks([v for _, v in ordered])
    ax.set_xticklabels([k for k, _ in ordered], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("examples to 95%  (open = never hit; y = examples spent)")
    ax.set_xlabel("difficulty cell  (seq_len / compression r / hops)")
    ax.set_ylim(4_000, 4.5e5)
    ax.set_xlim(-0.55, max(slots.values()) + 0.55)
    ax.set_title(
        "Concept-slot scaling frontier  ·  <10M exclusive slots vs C / D\n"
        "filled = hit 95%   ·   open = missed   ·   plus = still running",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.28)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_COLOR["A"], markeredgecolor=ARM_COLOR["A"], ms=9, label="A exclusive slots"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=ARM_COLOR["C"], markeredgecolor=ARM_COLOR["C"], ms=9, label="C no array (floor)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ARM_COLOR["D"], markeredgecolor=ARM_COLOR["D"], ms=9, label="D dense causal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="0.3", ms=8, label="open: 95% not hit"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="white", markeredgecolor="0.3", ms=9, label="plus: in progress"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper left", framealpha=0.92)
    ax.text(
        0.99,
        0.03,
        "Frontier so far: 5.11M exclusive-slot Arm A solves far_copy seq=256 r=8 at 95%.\n"
        "seq=256 r=32 still climbing (88% @ 256k). chain hops=3 is still at chance.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.2",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.9),
    )
    fig.tight_layout()
    dest = OUT_DIR / "concept_slot_scaling_frontier.png"
    fig.savefig(dest, dpi=160)
    shutil.copy2(dest, HARD / dest.name)
    plt.close(fig)
    return dest


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HARD.mkdir(parents=True, exist_ok=True)
    hard = collect_hard_cells()
    running = in_progress_logs()
    c1 = plot_learning_curves(hard, running)
    pts = frontier_points(hard, running)
    c2 = plot_frontier(pts)
    print(f"wrote {c1}", flush=True)
    print(f"wrote {c2}", flush=True)
    print(f"harder cells: {len(hard)}  in-progress traces: {len(running)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
