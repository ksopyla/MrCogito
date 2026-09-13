#!/usr/bin/env python
"""Plots for the BAPO capability ladder (E18 vs dense vs encoder-decoder).

Reads the JSON bundles written by `verification/bapo_capability_probe.py`.

  uv run python analysis/plot_bapo_capability.py --in_dir Cache/bapo_tiny --out_dir /opt/cursor/artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARCH_COLOR = {
    "dense": "#2166ac",
    "e18": "#b2182b",
    "encdec": "#4daf4a",
    "e18_local": "#999999",
}
ARCH_LABEL = {
    "dense": "dense decoder-only",
    "e18": "E18 (one global read)",
    "encdec": "encoder-decoder",
    "e18_local": "E18 local (no read)",
}


def _load(in_dir: Path) -> list[dict]:
    bundles = []
    for p in sorted(in_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        bundles.append(json.loads(p.read_text()))
    if not bundles:
        raise SystemExit(f"no rung JSON files in {in_dir}")
    return bundles


def _style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def plot_learning_curves(bundles: list[dict], out: Path) -> None:
    n = len(bundles)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, b in zip(axes, bundles):
        for arch, r in b["results"].items():
            steps = [t["step"] for t in r["trace"]]
            acc = [t["acc"] for t in r["trace"]]
            ax.plot(steps, acc, color=ARCH_COLOR.get(arch, "k"), label=ARCH_LABEL.get(arch, arch), lw=2)
        ax.axhline(0.75, color="0.4", ls="--", lw=1, label="solvability 75%")
        ax.axhline(b["card"]["chance_acc"], color="0.6", ls=":", lw=1, label="chance")
        ax.set_title(b["task"])
        ax.set_xlabel("step")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("held-out token accuracy")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Learning curves · {bundles[0]['scale']} · hidden={bundles[0]['hidden']}", y=1.03)
    fig.savefig(out / "learning_curves.png")
    fig.savefig(out / "learning_curves.svg")
    plt.close(fig)


def plot_accuracy_heatmap(bundles: list[dict], out: Path) -> None:
    arches = list(next(iter(bundles))["results"])
    tasks = [b["task"] for b in bundles]
    acc = np.array([[b["results"][a]["final"]["acc"] if a in b["results"] else np.nan for a in arches] for b in bundles])
    fig, ax = plt.subplots(figsize=(1.8 * len(arches) + 2.2, 0.7 * len(tasks) + 2.2))
    im = ax.imshow(acc, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(arches)), [ARCH_LABEL.get(a, a) for a in arches], rotation=25, ha="right")
    ax.set_yticks(range(len(tasks)), tasks)
    for i in range(len(tasks)):
        for j in range(len(arches)):
            ax.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="accuracy")
    ax.set_title("Final accuracy")
    fig.savefig(out / "accuracy_heatmap.png")
    fig.savefig(out / "accuracy_heatmap.svg")
    plt.close(fig)


def plot_information_flow(bundles: list[dict], out: Path) -> None:
    arches = list(next(iter(bundles))["results"])
    x = np.arange(len(bundles))
    width = 0.8 / max(len(arches), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(bundles) + 2, 3.8))
    for i, arch in enumerate(arches):
        vals = [b["results"][a]["info"]["information_flow"] if a in b["results"] else 0 for a, b in [(arch, b) for b in bundles]]
        ax.bar(x + i * width, vals, width, color=ARCH_COLOR.get(arch, "k"), label=ARCH_LABEL.get(arch, arch))
    ax.set_xticks(x + width * (len(arches) - 1) / 2, [b["task"] for b in bundles], rotation=20, ha="right")
    ax.set_ylabel("information flow (recovered / prize)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Effective prefix bandwidth as a fraction of the task prize")
    fig.savefig(out / "information_flow.png")
    fig.savefig(out / "information_flow.svg")
    plt.close(fig)


def plot_recovered_bits(bundles: list[dict], out: Path) -> None:
    arches = list(next(iter(bundles))["results"])
    x = np.arange(len(bundles))
    width = 0.8 / max(len(arches), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(bundles) + 2, 3.8))
    for i, arch in enumerate(arches):
        vals = [b["results"][arch]["info"]["recovered_bits"] for b in bundles]
        ax.bar(x + i * width, vals, width, color=ARCH_COLOR.get(arch, "k"), label=ARCH_LABEL.get(arch, arch))
    prizes = [b["card"]["prize_bits"] for b in bundles]
    ax.plot(x + width * (len(arches) - 1) / 2, prizes, "k^", label="prize (bits)")
    ax.set_xticks(x + width * (len(arches) - 1) / 2, [b["task"] for b in bundles], rotation=20, ha="right")
    ax.set_ylabel("recovered bits (span)")
    ax.legend(fontsize=8)
    ax.set_title("Recovered information vs task prize")
    fig.savefig(out / "recovered_bits.png")
    fig.savefig(out / "recovered_bits.svg")
    plt.close(fig)


def plot_bytes_per_token(bundles: list[dict], out: Path) -> None:
    """Two views: effective information bytes/token, and nominal KV cache bytes/token."""
    arches = list(next(iter(bundles))["results"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(bundles))
    width = 0.8 / max(len(arches), 1)
    for i, arch in enumerate(arches):
        info_b = [b["results"][arch]["info"]["bytes_per_input_token"] for b in bundles]
        cache_b = [b["results"][arch]["info"]["nominal_a_bytes"] for b in bundles]
        ax1.bar(x + i * width, info_b, width, color=ARCH_COLOR.get(arch, "k"), label=ARCH_LABEL.get(arch, arch))
        ax2.bar(x + i * width, cache_b, width, color=ARCH_COLOR.get(arch, "k"), label=ARCH_LABEL.get(arch, arch))
    ax1.set_xticks(x + width * (len(arches) - 1) / 2, [b["task"] for b in bundles], rotation=20, ha="right")
    ax2.set_xticks(x + width * (len(arches) - 1) / 2, [b["task"] for b in bundles], rotation=20, ha="right")
    ax1.set_ylabel("effective bytes / input token")
    ax2.set_ylabel("unbounded KV cache bytes / token")
    ax1.set_title("Information actually recovered")
    ax2.set_title("Nominal cache bandwidth (BAPO a)")
    ax2.legend(fontsize=8)
    fig.savefig(out / "bytes_per_token.png")
    fig.savefig(out / "bytes_per_token.svg")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    in_dir, out = Path(args.in_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundles = _load(in_dir)
    _style()
    plot_learning_curves(bundles, out)
    plot_accuracy_heatmap(bundles, out)
    plot_information_flow(bundles, out)
    plot_recovered_bits(bundles, out)
    plot_bytes_per_token(bundles, out)
    print(f"wrote plots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
