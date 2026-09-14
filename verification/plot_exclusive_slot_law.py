#!/usr/bin/env python
"""Summary comparison plots from the durable exclusive-slot law inventory.

Does not require /opt/cursor/artifacts/scale_hard JSON (that store can wipe).
Reads docs/4_Research_Notes/exclusive_slot_law_inventory.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
INV = REPO / "docs/4_Research_Notes/exclusive_slot_law_inventory.json"
OUTS = [
    REPO / "docs/4_Research_Notes/figures",
    Path("/workspace/Cache/scale_hard"),
    Path("/tmp/scale_hard"),
]
BAR = 0.95
CHANCE = 0.25
ARM_COLOR = {"A": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}


def main() -> int:
    cells = json.loads(INV.read_text())["cells"]
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.2))

    ax = axes[0]
    ax.axhline(BAR, color="0.35", ls="--", lw=1.0, label="95% bar")
    ax.axhline(CHANCE, color="0.65", ls=":", lw=1.0, label="chance")
    for c in cells:
        if c["task"] != "far_copy" or not c["hit_95"]:
            continue
        ax.scatter(
            c["steps"],
            c["acc"],
            s=110,
            c=ARM_COLOR[c["arm"]],
            marker={"A": "o", "D": "s", "C": "D"}[c["arm"]],
            zorder=3,
        )
        label = f"{c['arm']} S={c['seq']} r={c['r']}"
        if c["min_gap"] != 32:
            label += f" g{c['min_gap']}"
        ax.annotate(label, (c["steps"], c["acc"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("optimizer steps at ≥95%")
    ax.set_ylabel("eval accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Steps to 95% (far_copy hits)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax2.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    ax2.axvline(10, color="0.5", ls="--", lw=0.8)
    for c in cells:
        ax2.scatter(
            c["params_m"],
            c["acc"],
            s=90,
            c=ARM_COLOR[c["arm"]],
            marker={"A": "o", "D": "s", "C": "D"}[c["arm"]],
            zorder=3,
        )
    ax2.set_xlabel("params (M)")
    ax2.set_ylabel("final accuracy")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("Size vs accuracy (<10M)")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for c in cells:
        if c["task"] != "far_copy" or c["e95"] is None:
            continue
        ax3.scatter(
            c["seq"],
            c["e95"] / 1000.0,
            s=110,
            c=ARM_COLOR[c["arm"]],
            marker={"A": "o", "D": "s"}[c["arm"]],
            zorder=3,
        )
        tag = f"{c['arm']} r={c['r']}"
        if c["min_gap"] != 32:
            tag += f" g{c['min_gap']}"
        ax3.annotate(tag, (c["seq"], c["e95"] / 1000.0), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax3.set_xlabel("seq_len (far_copy cells that hit 95%)")
    ax3.set_ylabel("examples to ≥95% (thousands)")
    ax3.set_title("Data to 95% vs length")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "Exclusive-scope concept slots <10M · 95% bar · steps · sizes · accuracies",
        fontsize=12,
    )
    fig.tight_layout()
    for d in OUTS:
        fig.savefig(d / "exclusive_slot_law_steps_sizes_acc.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0))
    ax = axes[0]
    for c in cells:
        if c["task"] != "far_copy" or c["r"] != 8 or c["arm"] == "C":
            continue
        if c["e95"] is None:
            continue
        ax.scatter(
            c["seq"],
            c["e95"] / 1000.0,
            s=110,
            c=ARM_COLOR[c["arm"]],
            marker={"A": "o", "D": "s"}[c["arm"]],
        )
    ax.set_xlabel("seq_len (r=8 far_copy)")
    ax.set_ylabel("examples to ≥95% (thousands)")
    ax.set_title("Length: E95 stays ~10^5 through 1024")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axhline(BAR, color="0.35", ls="--", lw=1.0)
    for c in cells:
        if c["task"] != "far_copy" or c["arm"] != "A":
            continue
        ax2.scatter(c["r"], c["acc"], s=120, zorder=3)
        ax2.annotate(f"seq{c['seq']}", (c["r"], c["acc"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xlabel("pooling ratio r")
    ax2.set_ylabel("final accuracy (Arm A)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("Compression: r=32 is the miss")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.axhline(BAR, color="0.35", ls="--", lw=1.0)
    ax3.axhline(CHANCE, color="0.65", ls=":", lw=1.0)
    chain = [c for c in cells if c["task"] == "chain"]
    labels = [f"{c['arm']} h{c['hops']}" for c in chain]
    accs = [c["acc"] for c in chain]
    cols = [ARM_COLOR[c["arm"]] for c in chain]
    ax3.bar(range(len(labels)), accs, color=cols)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("final accuracy")
    ax3.set_ylim(-0.02, 1.05)
    ax3.set_title("Composition: hops=2 and hops=3 exam kills")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Working law · exclusive slots <10M · 95% bar", fontsize=12)
    fig.tight_layout()
    for d in OUTS:
        fig.savefig(d / "exclusive_slot_working_law.png", dpi=140)
    plt.close(fig)

    print(f"wrote plots next to {INV} and Cache/tmp", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
