#!/usr/bin/env python
"""Summary comparison plots from the durable exclusive-slot law inventory.

Does not require /opt/cursor/artifacts/scale_hard JSON (that store can wipe).
Reads docs/4_Research_Notes/exclusive_slot_law_inventory.json.

Legend (architecture, not letter soup):
  A  = exclusive-scope concept-slot model
  D  = dense full-causal transformer baseline (4-layer, width-matched)
  D9 = dense full-causal transformer baseline (9-layer, param-matched)
  C  = no-concept leak check (same decoder window; NOT a dense LM baseline)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
INV = REPO / "docs/4_Research_Notes/exclusive_slot_law_inventory.json"
OUTS = [
    REPO / "docs/4_Research_Notes/figures",
    Path("/workspace/Cache/scale_hard"),
    Path("/tmp/scale_hard"),
    Path("/opt/cursor/artifacts"),
]
BAR = 0.95
CHANCE = 0.25

# A = concepts, D/D9 = dense baseline, C = leak check.
STYLE = {
    "A": {
        "color": "#1f77b4",
        "marker": "o",
        "label": "A · concepts (exclusive slots)",
    },
    "D": {
        "color": "#d62728",
        "marker": "s",
        "label": "D · dense baseline (4L, width-matched)",
    },
    "D9": {
        "color": "#8c1515",
        "marker": "^",
        "label": "D9 · dense baseline (9L, param-matched)",
    },
    "C": {
        "color": "#2ca02c",
        "marker": "D",
        "label": "C · leak check (no concepts, not a baseline)",
    },
}
SERIES_ORDER = ("A", "D", "D9", "C")


def series_key(cell: dict) -> str:
    arm = cell["arm"]
    if arm == "A":
        return "A"
    if arm == "C":
        return "C"
    if cell.get("params_m", 0) >= 4.0 or "D9" in cell.get("run", ""):
        return "D9"
    return "D"


def legend_handles(keys: tuple[str, ...] = SERIES_ORDER, *, patches: bool = False):
    out = []
    for key in keys:
        st = STYLE[key]
        if patches:
            out.append(Patch(facecolor=st["color"], edgecolor="0.2", label=st["label"]))
        else:
            out.append(
                Line2D(
                    [0],
                    [0],
                    color=st["color"],
                    marker=st["marker"],
                    linestyle="none",
                    markersize=9,
                    label=st["label"],
                )
            )
    return out


def save_fig(fig, name: str) -> None:
    for d in OUTS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / name, dpi=140, bbox_inches="tight")
        except OSError as exc:
            print(f"skip {d}: {exc}", flush=True)


def pick(cells: list[dict], pred) -> dict | None:
    hits = [c for c in cells if pred(c)]
    return hits[0] if hits else None


def main() -> int:
    payload = json.loads(INV.read_text())
    cells = payload["cells"]
    in_flight = payload.get("in_flight") or []
    for d in OUTS:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Figure 1: steps / size / data-to-95%
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6))

    ax = axes[0]
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0, label="95% bar")
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0, label="chance 25%")
    used = set()
    for c in cells:
        if c["task"] != "far_copy" or not c["hit_95"]:
            continue
        key = series_key(c)
        st = STYLE[key]
        ax.scatter(
            c["steps"],
            c["acc"],
            s=120,
            c=st["color"],
            marker=st["marker"],
            zorder=3,
            edgecolors="0.15",
            linewidths=0.4,
        )
        used.add(key)
    ax.set_xlabel("optimizer steps at ≥95%")
    ax.set_ylabel("eval accuracy")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title("Steps to 95% (far_copy hits)")
    ax.legend(handles=legend_handles(tuple(k for k in SERIES_ORDER if k in used)), fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax2.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    ax2.axvline(10, color="0.5", ls="--", lw=0.8)
    used = set()
    for c in cells:
        key = series_key(c)
        st = STYLE[key]
        ax2.scatter(
            c["params_m"],
            c["acc"],
            s=100,
            c=st["color"],
            marker=st["marker"],
            zorder=3,
            edgecolors="0.15",
            linewidths=0.4,
        )
        used.add(key)
    ax2.set_xlabel("params (M)")
    ax2.set_ylabel("final accuracy")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.02, 1.08)
    ax2.set_title("Size vs accuracy (<10M)")
    ax2.legend(handles=legend_handles(tuple(k for k in SERIES_ORDER if k in used)), fontsize=7.5, loc="center right")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    used = set()
    for c in cells:
        if c["task"] != "far_copy" or c["e95"] is None:
            continue
        key = series_key(c)
        st = STYLE[key]
        ax3.scatter(
            c["seq"],
            c["e95"] / 1000.0,
            s=120,
            c=st["color"],
            marker=st["marker"],
            zorder=3,
            edgecolors="0.15",
            linewidths=0.4,
        )
        tag = f"{key} r={c['r']}"
        if c["min_gap"] != 32:
            tag += f" g{c['min_gap']}"
        ax3.annotate(tag, (c["seq"], c["e95"] / 1000.0), textcoords="offset points", xytext=(5, 4), fontsize=7)
        used.add(key)
    ax3.set_xlabel("seq_len (far_copy cells that hit 95%)")
    ax3.set_ylabel("examples to ≥95% (thousands)")
    ax3.set_title("Data to 95% vs length")
    ax3.legend(handles=legend_handles(tuple(k for k in SERIES_ORDER if k in used)), fontsize=7.5, loc="upper left")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "Exclusive-scope concept slots vs dense baseline  ·  <10M  ·  95% bar\n"
        "A = concepts  ·  D / D9 = dense LM baseline  ·  C = leak check (not a baseline)",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, "exclusive_slot_law_steps_sizes_acc.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: working law (length / compression / composition)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.4))
    ax = axes[0]
    used = set()
    for c in cells:
        if c["task"] != "far_copy" or c["r"] != 8 or series_key(c) == "C":
            continue
        if c["e95"] is None:
            continue
        key = series_key(c)
        st = STYLE[key]
        ax.scatter(
            c["seq"],
            c["e95"] / 1000.0,
            s=120,
            c=st["color"],
            marker=st["marker"],
            edgecolors="0.15",
            linewidths=0.4,
        )
        used.add(key)
    ax.set_xlabel("seq_len (r=8 far_copy)")
    ax.set_ylabel("examples to ≥95% (thousands)")
    ax.set_title("Length: E95 stays ~10^5 through 1024")
    ax.legend(handles=legend_handles(tuple(k for k in SERIES_ORDER if k in used)), fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axhline(BAR, color="0.35", ls="--", lw=1.0, label="95% bar")
    ax2.axhline(CHANCE, color="0.65", ls=":", lw=1.0, label="chance 25%")
    # D ignores r: show the width-matched dense copy at seq256 as a reference.
    d_ref = pick(cells, lambda c: c["run"] == "cell_seq256_r8_D")
    if d_ref is not None:
        ax2.axhline(
            d_ref["acc"],
            color=STYLE["D"]["color"],
            ls="--",
            lw=1.0,
            alpha=0.7,
            label="D dense @ seq256 (ignores r)",
        )
    compress_xy = {
        (128, 8): (-36, 8),
        (256, 8): (8, -16),
        (128, 16): (-40, 10),
        (256, 16): (8, -16),
        (256, 32): (8, 8),
        (512, 8): (8, 8),
        (1024, 8): (8, -16),
    }
    for c in cells:
        if c["task"] != "far_copy" or series_key(c) != "A" or c["seq"] != 256:
            continue
        ax2.scatter(
            c["r"],
            c["acc"],
            s=130,
            c=STYLE["A"]["color"],
            marker=STYLE["A"]["marker"],
            zorder=3,
            edgecolors="0.15",
            linewidths=0.4,
        )
        ax2.annotate(
            f"seq{c['seq']}",
            (c["r"], c["acc"]),
            textcoords="offset points",
            xytext=compress_xy.get((c["seq"], c["r"]), (8, 6)),
            fontsize=8,
        )
    ax2.set_xlabel("pooling ratio r  (concepts only; dense D ignores r)")
    ax2.set_ylabel("final accuracy")
    ax2.set_ylim(-0.02, 1.08)
    ax2.set_title("Compression at seq256: r=32 is the miss (A concepts)")
    ax2.legend(fontsize=7.5, loc="lower left")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax3.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    chain = [c for c in cells if c["task"] == "chain"]
    labels = []
    accs = []
    cols = []
    for c in chain:
        key = series_key(c)
        if c["hops"] == 3:
            labels.append(f"{key} hops=3")
        else:
            labels.append(f"{key} hops=2 packed")
        accs.append(c["acc"])
        cols.append(STYLE[key]["color"])
    ax3.bar(range(len(labels)), accs, color=cols, edgecolor="0.2")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax3.set_ylabel("final accuracy")
    ax3.set_ylim(-0.02, 1.08)
    ax3.set_title("Composition: hops=2 and hops=3 exam kills")
    ax3.legend(
        handles=legend_handles(tuple(dict.fromkeys(series_key(c) for c in chain))),
        fontsize=7.5,
        loc="upper right",
    )
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Working law · exclusive slots vs dense baseline · <10M · 95% bar\n"
        "A = concepts  ·  D / D9 = dense LM baseline  ·  C = leak check (not a baseline)",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, "exclusive_slot_working_law.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: per-task grouped comparisons (the one to read first)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.8))

    # Length: one bar group per exam.
    length_exams = [
        ("seq128", lambda c: c["task"] == "far_copy" and c["seq"] == 128 and c["r"] == 8 and c["min_gap"] == 32),
        ("seq256", lambda c: c["task"] == "far_copy" and c["seq"] == 256 and c["r"] == 8 and c["min_gap"] == 32),
        ("seq512\npadded", lambda c: c["task"] == "far_copy" and c["seq"] == 512 and c["r"] == 8 and c["min_gap"] == 32),
        ("seq512\ntrue-reach", lambda c: c["task"] == "far_copy" and c["seq"] == 512 and c["min_gap"] == 128),
        ("seq1024\ntrue-reach", lambda c: c["task"] == "far_copy" and c["seq"] == 1024 and c["min_gap"] == 256),
    ]
    ax = axes[0]
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    n_exams = len(length_exams)
    n_series = len(SERIES_ORDER)
    width = 0.18
    x0 = list(range(n_exams))
    used = set()
    for i, key in enumerate(SERIES_ORDER):
        xs, ys, hatches = [], [], []
        for j, (_, pred) in enumerate(length_exams):
            cell = pick(cells, lambda c, p=pred, k=key: p(c) and series_key(c) == k)
            flight = None
            if cell is None:
                flight = pick(in_flight, lambda c, p=pred, k=key: p(c) and series_key(c) == k)
            chosen = cell or flight
            if chosen is None:
                continue
            xs.append(j + (i - (n_series - 1) / 2) * width)
            ys.append(chosen["acc"])
            hatches.append("//" if flight is not None else "")
            used.add(key)
        if not xs:
            continue
        bars = ax.bar(
            xs,
            ys,
            width=width * 0.95,
            color=STYLE[key]["color"],
            edgecolor="0.15",
            label=STYLE[key]["label"],
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_alpha(0.65)
    ax.set_xticks(x0)
    ax.set_xticklabels([name for name, _ in length_exams], fontsize=8)
    ax.set_ylim(-0.02, 1.12)
    ax.set_ylabel("eval accuracy")
    ax.set_title("far_copy length  (span=32, r=8)")
    handles = legend_handles(tuple(k for k in SERIES_ORDER if k in used), patches=True)
    handles.append(Patch(facecolor="0.85", edgecolor="0.2", hatch="//", label="in flight (not done)"))
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Compression: A at r=8/16/32, plus D reference (ignores r).
    ax2 = axes[1]
    ax2.axhline(BAR, color="0.35", ls="--", lw=1.0, label="95% bar")
    ax2.axhline(CHANCE, color="0.65", ls=":", lw=1.0, label="chance 25%")
    if d_ref is not None:
        ax2.axhline(
            d_ref["acc"],
            color=STYLE["D"]["color"],
            ls="--",
            lw=1.2,
            label="D dense baseline @ seq256 (ignores r)",
        )
    a_comp = [
        c
        for c in cells
        if c["task"] == "far_copy" and series_key(c) == "A" and c["seq"] in (128, 256)
    ]
    a_comp_xy = {
        (128, 8): (-52, 10),
        (256, 8): (-52, -22),
        (128, 16): (10, 12),
        (256, 16): (10, -24),
        (256, 32): (10, 8),
    }
    for c in a_comp:
        ax2.scatter(
            c["r"],
            c["acc"],
            s=150,
            c=STYLE["A"]["color"],
            marker=STYLE["A"]["marker"],
            zorder=3,
            edgecolors="0.15",
            linewidths=0.5,
        )
        n_ex = int((c["e95"] or c["examples"]) / 1000)
        ax2.annotate(
            f"A seq{c['seq']}\n{c['acc']:.0%} @ {n_ex}k",
            (c["r"], c["acc"]),
            textcoords="offset points",
            xytext=a_comp_xy.get((c["seq"], c["r"]), (8, 6)),
            fontsize=7.5,
        )
    ax2.set_xticks([8, 16, 32])
    ax2.set_xlabel("pooling ratio r  (tokens per concept slot)")
    ax2.set_ylabel("final accuracy")
    ax2.set_ylim(-0.02, 1.12)
    ax2.set_xlim(4, 38)
    ax2.set_title("far_copy compression  (A concepts vs D dense)")
    ax2.legend(fontsize=7, loc="center right")
    ax2.grid(True, alpha=0.3)

    # Composition.
    ax3 = axes[2]
    ax3.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax3.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    chain_exams = [
        ("hops=3\nA concepts", pick(cells, lambda c: c["run"] == "cell_chain_h3_A")),
        ("hops=3\nD dense 4L", pick(cells, lambda c: c["run"] == "cell_chain_h3_D")),
        ("hops=3\nC leak", pick(cells, lambda c: c["run"] == "cell_chain_h3_C")),
        ("hops=2 packed\nD9 dense 9L", pick(cells, lambda c: c["run"] == "cell_chain_h2_D9")),
    ]
    xs = list(range(len(chain_exams)))
    ys = [c["acc"] if c else 0.0 for _, c in chain_exams]
    cols = []
    for name, cell in chain_exams:
        if cell is None:
            cols.append("0.7")
        else:
            cols.append(STYLE[series_key(cell)]["color"])
    bars = ax3.bar(xs, ys, color=cols, edgecolor="0.2")
    for bar, (_, cell) in zip(bars, chain_exams):
        if cell is None:
            continue
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{cell['acc']:.0%}\n{int(cell['examples']/1000)}k",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax3.set_xticks(xs)
    ax3.set_xticklabels([name for name, _ in chain_exams], fontsize=8)
    ax3.set_ylim(-0.02, 1.12)
    ax3.set_ylabel("final accuracy")
    ax3.set_title("chain composition  (exam kill at this budget)")
    ax3.legend(
        handles=[
            Patch(facecolor=STYLE["A"]["color"], edgecolor="0.2", label=STYLE["A"]["label"]),
            Patch(facecolor=STYLE["D"]["color"], edgecolor="0.2", label=STYLE["D"]["label"]),
            Patch(facecolor=STYLE["D9"]["color"], edgecolor="0.2", label=STYLE["D9"]["label"]),
            Patch(facecolor=STYLE["C"]["color"], edgecolor="0.2", label=STYLE["C"]["label"]),
        ],
        fontsize=7,
        loc="upper right",
    )
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Per-task comparison  ·  A = concepts   D/D9 = dense baseline   C = leak check\n"
        "95% success bar  ·  25% chance (n_symbols=4)  ·  decoder window = 32 for A and C",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, "exclusive_slot_task_comparisons.png")
    plt.close(fig)

    print(f"wrote plots next to {INV} and Cache/tmp/artifacts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
