#!/usr/bin/env python
"""Plot the exclusive-scope <10M scaling campaign (plus the seq128 easy end)."""
from __future__ import annotations

import json
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


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def traces(bundle: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    tr = bundle["results"][arm]["trace"]
    xs = np.array([p.get("examples", p["step"] * bundle["config"].get("batch", 32)) for p in tr])
    ys = np.array([p["acc"] for p in tr], dtype=float)
    return xs, ys


def label_for(bundle: dict, arm: str) -> str:
    cfg = bundle["config"]
    task = cfg.get("task", "?")
    seq = cfg.get("seq_len")
    r = cfg.get("ratio")
    hops = cfg.get("hops")
    params = bundle["results"][arm]["params"] / 1e6
    if task == "chain":
        diff = f"chain h{hops} seq{seq}"
    else:
        diff = f"seq{seq} r={r}"
    return f"{arm} {diff} ({params:.2f}M)"


def collect_hard() -> list[dict]:
    rows = []
    for p in sorted(HARD.glob("*.json")):
        if p.name in {"campaign_index.json", "campaign_meta.json", "winner_lr.json"}:
            continue
        if p.name.startswith("lr") and "cell_" not in p.name:
            # LR probes: keep for a small inset, not the main A/C/D figure
            continue
        b = load(p)
        if not b or "results" not in b:
            continue
        rows.append(b)
    return rows


def difficulty_key(bundle: dict) -> str:
    cfg = bundle["config"]
    if cfg.get("task") == "chain":
        return f"chain/h{cfg.get('hops')}/seq{cfg['seq_len']}"
    return f"far_copy/seq{cfg['seq_len']}/r{cfg['ratio']}"


def main() -> int:
    hard = collect_hard()
    easy_a = load(EASY_A)
    easy_3 = load(EASY_3ARM)
    r16 = load(R16)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    ax = axes[0]
    ax.axhline(BAR, color="0.4", ls="--", lw=1, label="95% bar")
    ax.axhline(CHANCE, color="0.7", ls=":", lw=1, label="chance")

    style = {"A": "-", "C": "--", "D": ":"}
    # Easy end (seq128, 1.35M A / 0.60M C,D)
    if easy_a and "A" in easy_a.get("results", {}):
        xs, ys = traces(easy_a, "A")
        ax.plot(xs, ys, style["A"], color="#1f77b4", lw=2.2, label="A seq128 r=8 (1.35M, easy)")
    if r16 and "A" in r16.get("results", {}):
        xs, ys = traces(r16, "A")
        ax.plot(xs, ys, style["A"], color="#5fa8d3", lw=1.4, label="A seq128 r=16 (1.35M)")
    if easy_3:
        if "C" in easy_3.get("results", {}):
            xs, ys = traces(easy_3, "C")
            ax.plot(xs, ys, style["C"], color="#2ca02c", lw=1.6, label="C seq128 (0.60M, floor)")
        if "D" in easy_3.get("results", {}):
            xs, ys = traces(easy_3, "D")
            ax.plot(xs, ys, style["D"], color="#d62728", lw=1.8, label="D seq128 (0.60M)")

    colors = {
        "A": ["#1f77b4", "#4c78a8", "#08306b", "#6baed6"],
        "C": ["#2ca02c", "#74c476"],
        "D": ["#d62728", "#e45756", "#843c39"],
    }
    seen = {a: 0 for a in "ACD"}
    for b in hard:
        for arm in b["results"]:
            xs, ys = traces(b, arm)
            c = colors[arm][seen[arm] % len(colors[arm])]
            seen[arm] += 1
            ax.plot(xs, ys, style[arm], color=c, lw=2.0, label=label_for(b, arm))

    ax.set_xlabel("examples seen")
    ax.set_ylabel("eval accuracy (supervised tokens)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Accuracy vs examples  (<10M A vs C vs D)")
    ax.legend(fontsize=7, loc="lower right", ncol=1)
    ax.grid(True, alpha=0.3)

    # Panel 2: difficulty vs acc at a matched example budget.
    ax2 = axes[1]
    budget = 64_000
    # pick the largest budget that most full cells have reached, else last point
    records = []
    if easy_a:
        xs, ys = traces(easy_a, "A")
        idx = int(np.searchsorted(xs, budget, side="right") - 1)
        records.append(("A", "seq128 r=8", float(ys[max(idx, 0)]), easy_a["results"]["A"]["params"]))
    if easy_3:
        for arm in ("C", "D"):
            if arm not in easy_3["results"]:
                continue
            xs, ys = traces(easy_3, arm)
            idx = int(np.searchsorted(xs, budget, side="right") - 1)
            records.append((arm, "seq128 r=8", float(ys[max(idx, 0)]), easy_3["results"][arm]["params"]))
    for b in hard:
        cfg = b["config"]
        if cfg.get("steps", 0) <= 800 and str(b.get("run_name", "")).startswith("lr"):
            continue
        diff = difficulty_key(b)
        for arm, r in b["results"].items():
            xs, ys = traces(b, arm)
            idx = int(np.searchsorted(xs, budget, side="right") - 1)
            if idx < 0:
                acc = float(ys[0])
            else:
                acc = float(ys[idx])
            records.append((arm, diff, acc, r["params"]))

    # Group by difficulty, plot A/C/D bars
    diffs = []
    for _, d, _, _ in records:
        if d not in diffs:
            diffs.append(d)
    x = np.arange(len(diffs))
    width = 0.24
    for i, arm in enumerate("ACD"):
        vals = []
        for d in diffs:
            hits = [acc for a, dd, acc, _ in records if a == arm and dd == d]
            vals.append(hits[-1] if hits else np.nan)
        ax2.bar(x + (i - 1) * width, vals, width, label=f"arm {arm}")
    ax2.axhline(BAR, color="0.4", ls="--", lw=1)
    ax2.axhline(CHANCE, color="0.7", ls=":", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(diffs, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel(f"accuracy at {budget:,} examples (or last ckpt if fewer)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("Difficulty vs accuracy (matched example budget)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Exclusive-scope concept slots <10M  ·  95% bar  ·  CPU campaign",
        fontsize=12,
    )
    fig.tight_layout()
    combined = OUT_DIR / "concept_slot_scaling_frontier.png"
    fig.savefig(combined, dpi=140)
    fig.savefig(HARD / "concept_slot_scaling_frontier.png", dpi=140)

    # Also save the two panels as separate files.
    fig.savefig(OUT_DIR / "accuracy_vs_examples.png", dpi=140)

    # Optional: params vs max seq at ≥95% for far_copy.
    fig2, ax3 = plt.subplots(figsize=(7.2, 4.4))
    for arm, marker in (("A", "o"), ("D", "s")):
        pts = []
        # easy
        if arm == "A" and easy_a:
            acc = easy_a["results"]["A"]["final"]["acc"]
            if acc >= BAR:
                pts.append((128, easy_a["results"]["A"]["params"] / 1e6))
        if arm == "D" and easy_3:
            acc = easy_3["results"]["D"]["final"]["acc"]
            if acc >= BAR:
                pts.append((128, easy_3["results"]["D"]["params"] / 1e6))
        for b in hard:
            if b["config"].get("task") != "far_copy":
                continue
            if arm not in b["results"]:
                continue
            if b["results"][arm].get("acc", b["results"][arm]["final"]["acc"]) >= BAR:
                pts.append((b["config"]["seq_len"], b["results"][arm]["params"] / 1e6))
        if pts:
            pts = sorted(set(pts))
            ax3.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-", label=f"arm {arm} ≥95%")
    ax3.set_xlabel("seq_len (far_copy, span 32)")
    ax3.set_ylabel("params (M)")
    ax3.set_title("Params vs max seq at ≥95%")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "params_vs_max_seq.png", dpi=140)
    fig2.savefig(HARD / "params_vs_max_seq.png", dpi=140)

    # Dedicated difficulty bar (already in combined); copy panel file.
    fig.savefig(OUT_DIR / "difficulty_vs_accuracy.png", dpi=140)
    print(f"wrote {combined}", flush=True)
    print(f"wrote {OUT_DIR / 'accuracy_vs_examples.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'difficulty_vs_accuracy.png'}", flush=True)
    print(f"wrote {OUT_DIR / 'params_vs_max_seq.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
