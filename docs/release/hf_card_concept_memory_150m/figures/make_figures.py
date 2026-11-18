"""Figures for the 150M concept-memory model card, drawn from the result files next to the card.

    python figures/make_figures.py          (from docs/release/hf_card_concept_memory_150m)

radar.png        ours, SmolLM2-135M at its earliest public checkpoint (~252B tokens) and Pythia-160M on six
                 standard benchmarks. Each axis is the model's score as a share of the best of the three (rim = the
                 best, centre = zero); for WikiText, where lower is better, the share is best ÷ score. The raw
                 numbers are printed next to each axis and in the card's table.
long_docs.png    bits per byte on the last pages of 55 held-out 64k-token documents against the context each model
                 is given, every model on the same text; a line ends at the most context the model can take.
data_eff.png     score against training tokens (log scale): Pythia-160M's and SmolLM2-135M's published checkpoints
                 (scored by us, same harness) and our single model at 11.5B tokens.
tokens_to_match  printed: the training tokens Pythia-160M needs to reach our score on each test (log-linear
                 interpolation between its checkpoints).

All numbers come from files in eval_results/ and long_context/ (our harness: lm-evaluation-harness 0-shot with our
adapter for our model, the stock hf model for the others; long_context_compare.py for the long documents).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ER, LC, OUT = ROOT / "eval_results", ROOT / "long_context", ROOT / "figures"
sys.dont_write_bytecode = True  # no __pycache__ next to the published files
sys.path.insert(0, str(LC))
from summarise_compare import load as load_long, pooled  # noqa: E402

# reference palette (dataviz skill), light mode; first three slots validate all-pairs (2026-11-18)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
OURS = dict(name="150M concept memory (ours)", board="e38_mem150m_final", core7="e38_mem150m_core7",
            long="compare_ours.json", tokens=11.5e9, color="#2a78d6", short="ours")
SMOL = dict(name="SmolLM2-135M, earliest public checkpoint", board="ref_smollm2_135m_step240k_board",
            core7="ref_smollm2_135m_step240k_core7", long="compare_smollm2_step240k.json", tokens=240_000 * 1_048_576,
            color="#eb6834", short="SmolLM2")
SMOL_FINAL = dict(name="SmolLM2-135M", board="ref_smollm2_135m_board", tokens=2.0e12, color="#eb6834")
PYTH = dict(name="Pythia-160M", board="ref_pythia160m_step143000_board", core7="ref_pythia160m_step143000_core7",
            long="compare_pythia.json", tokens=299.9e9, color="#1baf7a", short="Pythia")
# the long-document chart: (file, label, parameters, training tokens as reported); ours first, drawn in colour
LONG = [("compare_ours.json", "Ours", "141.5M", "11.5B"),
        ("compare_smollm2_step240k.json", "SmolLM2-135M, 252B checkpoint", "135M", "252B"),
        ("compare_smollm2.json", "SmolLM2-135M final", "135M", "~2T"),
        ("compare_pythia.json", "Pythia-160M", "162M", "300B"),
        ("compare_gemma.json", "Gemma 3 270M", "268M", "6T"),
        ("compare_granite.json", "Granite 4.0 350M", "352M", "~15T"),
        ("compare_lfm25.json", "LFM2.5-350M", "354M", "28T")]
CORE7 = {"arc_challenge": 25, "arc_easy": 25, "hellaswag": 25, "lambada_openai": 0, "piqa": 50, "sciq": 25,
         "winogrande": 50}
PYTHIA_GBS, SMOL_GBS = 2_097_152, 1_048_576  # tokens per step, from each model card

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def fmt_tokens(t: float) -> str:
    return f"{t / 1e12:g}T" if t >= 1e12 else f"{t / 1e9:.1f}B" if t < 9.5e9 or abs(t - 11.5e9) < 1e8 else f"{t / 1e9:.0f}B"


def board(tag: str) -> dict | None:
    f = ER / f"{tag}.json"
    return json.load(open(f))["tiny_ml_board"] if f.exists() else None


def core7_mean(tag: str) -> float | None:
    f = ER / f"{tag}.json"
    if not f.exists():
        return None
    res = json.load(open(f))["results"]
    return 100 * float(np.mean([res[t]["acc,none"] for t in CORE7]))


def long_measures(fname: str) -> tuple[float, float] | None:
    """(bits per byte with the most context the model takes, bits per byte saved from 1k to that)."""
    f = LC / fname
    if not f.exists():
        return None
    r, keys, by = load_long(str(f))
    docs = sorted(by)
    nb = np.array([d[1] for d in docs], dtype=float)
    cut = {k: np.array([by[d]["cut"][k] for d in docs]) for k in keys}
    most = next((k for k in keys if all(by[d]["windowed"][k] for d in docs)), keys[-1])
    return pooled(cut[most], nb), pooled(cut[keys[0]], nb) - pooled(cut[most], nb)


def core7(tag: str) -> dict | None:
    f = ER / f"{tag}.json"
    return json.load(open(f))["results"] if f.exists() else None


def measures(m: dict) -> dict | None:
    b, c = board(m["board"]), core7(m["core7"])
    if b is None or c is None:
        print(f"skip {m['name']}: missing results", file=sys.stderr)
        return None
    acc = lambda t: 100 * c[t]["acc,none"]
    return {"Grammar\n(BLiMP)": b["blimp"], "Science quiz\n(ARC-Easy)": b["arc"],
            "Hard science quiz\n(ARC-Challenge)": acc("arc_challenge"), "Science with a passage\n(SciQ)": acc("sciq"),
            "Sensible continuation\n(HellaSwag)": acc("hellaswag"),
            "Wikipedia text\n(WikiText-2)": math.log2(b["wiki"])}


# axis: higher is better?
AXES = {"Grammar\n(BLiMP)": True, "Science quiz\n(ARC-Easy)": True, "Hard science quiz\n(ARC-Challenge)": True,
        "Science with a passage\n(SciQ)": True, "Sensible continuation\n(HellaSwag)": True,
        "Wikipedia text\n(WikiText-2)": False}
UNITS = {a: "{:.1f} %" for a in AXES}
UNITS["Grammar\n(BLiMP)"] = UNITS["Science quiz\n(ARC-Easy)"] = "{:.2f} %"
UNITS["Wikipedia text\n(WikiText-2)"] = "{:.3f} bits/byte"


def radar(models: list[dict]) -> None:
    vals = [(m, measures(m)) for m in models]
    vals = [(m, v) for m, v in vals if v]
    axes = list(AXES)
    best = {a: (max if AXES[a] else min)(v[a] for _, v in vals) for a in axes}

    def norm(a, x):  # share of the best of the three: rim = best, centre = zero
        return x / best[a] if AXES[a] else best[a] / x

    ang = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    fig = plt.figure(figsize=(8.6, 8.4))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_offset(np.pi / 2)  # first axis at the top, then clockwise
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, 2 * np.pi)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    for a, t in zip(axes, ang):  # axis name outside the rim, each model's raw value next to it
        disp = np.pi / 2 - t  # screen angle of this axis
        c, sn = math.cos(disp), math.sin(disp)
        ha = "center" if abs(c) < 0.2 else ("left" if c > 0 else "right")
        vals_txt = "\n".join(f"{m['short']}  {UNITS[a].format(v[a])}" for m, v in vals)
        kw = dict(xy=(t, 1.04), textcoords="offset points", ha=ha, annotation_clip=False)
        if sn > 0.8:  # top: values just above the rim, name above them
            ax.annotate(vals_txt, xytext=(0, 4), va="bottom", fontsize=7.5, color=INK2, **kw)
            ax.annotate(a, xytext=(0, 42), va="bottom", fontsize=10, color=INK, fontweight="bold", **kw)
        elif sn < -0.8:  # bottom: name just below the rim, values under it
            ax.annotate(a, xytext=(0, -4), va="top", fontsize=10, color=INK, fontweight="bold", **kw)
            ax.annotate(vals_txt, xytext=(0, -34), va="top", fontsize=7.5, color=INK2, **kw)
        else:  # sides: name above the axis line, values below it
            dx = 6 if c > 0 else -6
            ax.annotate(a, xytext=(dx, 2), va="bottom", fontsize=10, color=INK, fontweight="bold", **kw)
            ax.annotate(vals_txt, xytext=(dx, -4), va="top", fontsize=7.5, color=INK2, **kw)
    for m, v in reversed(vals):  # ours drawn last, on top
        r = [norm(a, v[a]) for a in axes]
        ours = m is vals[0][0]
        ax.fill(np.append(ang, ang[0]), r + r[:1], color=m["color"], alpha=0.16 if ours else 0.08, zorder=2)
        ax.plot(np.append(ang, ang[0]), r + r[:1], color=m["color"], linewidth=2.4 if ours else 1.8, zorder=3)
        ax.scatter(ang, r, s=34, color=m["color"], edgecolor=SURFACE, linewidth=2, zorder=4)
    handles = [plt.Line2D([], [], color=m["color"], linewidth=2.4, marker="o", markersize=6,
                          label=f"{m['name']} · {fmt_tokens(m['tokens'])} training tokens")
               for m, _ in vals]
    fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.012, "Further out is better on every axis: each score as a share of the best of the three "
             "(rim = best, centre = zero;\nfor WikiText, lower is better, so best ÷ score). Accuracy in %, 0-shot; "
             "WikiText in bits per byte. Same harness for all three;\nPythia in fp32 (it loses ~6 ARC points in "
             "bf16), the others in bf16.", ha="center", fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.22, right=0.78)
    fig.savefig(OUT / "radar.png", dpi=160)
    print("wrote radar.png")


def curve(prefix: str, gbs: int, steps_tag) -> list[tuple[float, dict]]:
    pts = []
    for f in sorted(ER.glob(f"{prefix}_*_board.json")):
        step = steps_tag(f.stem)
        if step is not None:
            pts.append((step * gbs, json.load(open(f))["tiny_ml_board"]))
    return sorted(pts, key=lambda p: p[0])


def pythia_step(stem: str):
    s = stem.split("_")[2]
    return int(s[4:]) if s.startswith("step") else None


def smol_step(stem: str):
    s = stem.split("_")[3]
    return int(s[4:-1]) * 1000 if s.startswith("step") and s.endswith("k") else None


def data_efficiency() -> None:
    ours = board(OURS["board"])
    py = curve("ref_pythia160m", PYTHIA_GBS, pythia_step)
    sm = curve("ref_smollm2_135m", SMOL_GBS, smol_step)
    final = board(SMOL_FINAL["board"])
    if final:
        sm.append((SMOL_FINAL["tokens"], final))
    panels = [("blimp", "Grammar (BLiMP), % ↑", lambda b: b["blimp"]),
              ("arc", "Science quiz (ARC-Easy), % ↑", lambda b: b["arc"]),
              ("wiki", "WikiText-2, bits per byte ↓", lambda b: math.log2(b["wiki"]))]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, (key, title, get) in zip(axs, panels):
        for pts, m in ((py, PYTH), (sm, SMOL_FINAL)):
            if pts:
                ax.plot([p[0] for p in pts], [get(p[1]) for p in pts], color=m["color"], linewidth=2, marker="o",
                        markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=m["name"])
        ax.scatter([OURS["tokens"]], [get(ours)], s=90, color=OURS["color"], edgecolor=SURFACE, linewidth=2,
                   zorder=5, label=OURS["name"])
        ax.axvline(OURS["tokens"], color=GRID, linewidth=1, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(1.5e9, 3e12)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: fmt_tokens(x).replace(".0", "")))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("training tokens (log scale)")
        if key == "wiki":
            ax.invert_yaxis()  # up is better on every panel
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Up is better on every panel. Every point scored by us on one RTX 3090, same harness, 0-shot. "
             "Checkpoints: Pythia-160M (EleutherAI, scored in fp32), SmolLM2-135M (Hugging Face, from ~250B tokens).", fontsize=8,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT / "data_eff.png", dpi=160)
    print("wrote data_eff.png")
    # tokens Pythia-160M needs to match us, log-linear between its checkpoints
    for key, title, get in panels:
        up = key != "wiki"
        target = get(ours)
        hit = None
        for (t0, b0), (t1, b1) in zip(py, py[1:]):
            y0, y1 = get(b0), get(b1)
            if (y1 >= target if up else y1 <= target) and not (y0 >= target if up else y0 <= target):
                f = (target - y0) / (y1 - y0)
                hit = math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
                break
        if py and (get(py[0][1]) >= target if up else get(py[0][1]) <= target):
            hit = py[0][0]
        msg = (f"{hit / 1e9:.1f}B tokens ({hit / OURS['tokens']:.1f}× ours)" if hit else
               f"not reached by {py[-1][0] / 1e9:.0f}B tokens (> {py[-1][0] / OURS['tokens']:.0f}× ours)" if py else "n/a")
        print(f"Pythia-160M to match our {title.split(',')[0]} ({target:.3f}): {msg}")


def long_docs() -> None:
    """Bits per byte against context, one line per model on the same text; ours in colour, the others in grey."""
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    end_labels = []
    for i, (fname, label, params, tok) in enumerate(LONG):
        f = LC / fname
        if not f.exists():
            print(f"skip {label}: no {fname}", file=sys.stderr)
            continue
        r, keys, by = load_long(str(f))
        docs = sorted(by)
        nb = np.array([d[1] for d in docs], dtype=float)
        xs, ys = [], []
        for k in keys:  # each budget the model reads whole; the first budget it needs windows for is its limit
            win = all(by[d]["windowed"][k] for d in docs)
            # a windowed budget means the model read at most its own length: draw that point at its length
            xs.append(min(int(k), int(r["max_len"])) if win else int(k))
            ys.append(pooled(np.array([by[d]["cut"][k] for d in docs]), nb))
            if win:
                break
        ours = i == 0
        col = OURS["color"] if ours else MUTED
        ax.plot(xs, ys, color=col, linewidth=2.6 if ours else 1.4, marker="o", markersize=6 if ours else 4,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4 if ours else 3)
        end_labels.append((xs[-1], ys[-1], f"{label} · {params} · {tok} tokens   {ys[-1]:.3f}", col, ours))
    # direct labels at each line's end, nudged apart vertically
    end_labels.sort(key=lambda e: e[1])
    placed = []
    for x, y, txt, col, ours in end_labels:
        yy = y
        for py_ in placed:
            if abs(yy - py_) < 0.014:
                yy = py_ + 0.014
        placed.append(yy)
        ax.annotate(txt, xy=(x, y), xytext=(x * 1.12, yy), va="center", fontsize=8.5, color=INK if ours else INK2,
                    fontweight="bold" if ours else "normal", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 4096, 16384, 28672, 61439])
    ax.set_xticklabels(["1k", "4k", "16k", "28k", "61k\n(whole span)"])
    ax.set_xlim(800, 61439 * 1.05)
    ax.set_ylim(0.958, 0.768)  # inverted: up is better; room below for the short-context models' labels
    ax.set_ylabel("bits per byte on the last pages (lower is better)")
    ax.set_xlabel("context before the last pages, in tokens (log scale)")
    ax.grid(color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.01, "Up is better. 55 held-out documents (books, papers, government and parliament text), "
             "the same characters for every model. A line ends at the most\ncontext the model can take; models "
             "that cannot take a budget are scored with sliding windows there. For scale: gzip needs 2.49 bits per "
             "byte on\nthe same pages, xz 1.97, plain text 8.", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.58, top=0.95, bottom=0.23)
    fig.savefig(OUT / "long_docs.png", dpi=160)
    print("wrote long_docs.png")


if __name__ == "__main__":
    radar([OURS, SMOL, PYTH])
    long_docs()
    data_efficiency()
