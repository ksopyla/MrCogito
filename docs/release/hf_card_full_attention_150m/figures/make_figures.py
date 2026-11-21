"""Figures for the Meridian21 150M model cards (concept-memory-140m-11.5bt, topic-index-140m-11.5bt,
catalogue-150m-11.5bt, full-attention-140m-11.5bt), drawn from the result files next to the card. The same script
ships with every card; only CARD below differs.

    python figures/make_figures.py          (from the card's folder)

The four models trained on the same data, in the same order, for the same budget, with the same layers, and differ
only in the reading layer. In each card "this model" is drawn solid and the other models on the same data dashed;
a family model whose results are not in this folder is left out. Each model keeps one colour on every chart of every
card: concept memory blue, topic index orange, catalogue aqua (the dataviz skill's first three categorical slots,
validated all-pairs on the light surface), the full-attention model near-black ink (the ceiling inside its 64k
training length, not a categorical hue), every model from outside the family grey. No fourth hue clears the
palette's all-pairs floors next to these three (yellow and magenta fail against orange), so SmolLM2-135M, aqua on
the cards of 2026-11-18, is grey from the 2026-11-21 family update on.

radar.png        this model, full-attention-140m-11.5bt and SmolLM2-135M at its earliest public checkpoint (≈252B
                 tokens) on six standard benchmarks. Each axis is the model's score as a share of the best of the three
                 (rim = the best, centre = zero); for WikiText, where lower is better, the share is best ÷ score. The
                 raw numbers are printed next to each axis and in the card's table.
ruler_lengths.png  RULER by task family at 4k, 8k, 16k and 32k tokens: this model against the other models on the
                 same data, the same harness and settings (share of expected answers found, 0–1).
ruler_heat.png   this model's 13 RULER tasks at 4k, 8k, 16k and 32k.
ruler_radar.png  RULER at 8k by task family, this model and full-attention-140m-11.5bt against each reference, when the
                 reference runs are in ruler/ (ref_*.json).
long_docs.png    bits per byte on the last pages of 55 held-out 64k-token documents against the context each model
                 is given, every model on the same text; a line ends at the most context the model can take.
data_eff.png     score against training tokens (log scale): Pythia-160M's and SmolLM2-135M's published checkpoints
                 (scored by us, same harness) and the 150M models on the same data at 11.5B tokens.

All numbers come from eval_results/, long_context/ and ruler/: this card's model under plain names (ruler_*,
compare_ours.json), every other model on the same data under its prefix (cm_, ti_, ca_, fa_; compare_<line>.json;
e38_<tag>_final.json and e38_<tag>_core7.json in eval_results/).
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
ER, LC, RU, OUT = ROOT / "eval_results", ROOT / "long_context", ROOT / "ruler", ROOT / "figures"
sys.dont_write_bytecode = True  # no __pycache__ next to the published files
sys.path.insert(0, str(LC))
from summarise_compare import load as load_long, pooled  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CARD = "full_attention"  # this card's model, a key of FAMILY (the one line that differs between the cards)
FAMILY = {"concept_memory": dict(name="concept-memory-140m-11.5bt", board="e38_mem150m_final", core7="e38_mem150m_core7",
                                 color=BLUE, short="concept-memory", prefix="cm_", file="compare_concept_memory.json",
                                 ramp=["#f0f4fa", "#9ec5f4", "#3987e5", "#184f95"]),
          "topic_index": dict(name="topic-index-140m-11.5bt", board="e38_lsa150m_final", core7="e38_lsa150m_core7",
                              color=ORANGE, short="topic-index", prefix="ti_", file="compare_topic_index.json",
                              ramp=["#fdf0e9", "#f6b896", "#eb6834", "#a3401b"]),
          "catalogue": dict(name="catalogue-150m-11.5bt", board="e38_cat150m_final", core7="e38_cat150m_core7",
                            color=AQUA, short="catalogue", prefix="ca_", file="compare_catalogue.json",
                            ramp=["#eaf7f1", "#9fdcc4", "#1baf7a", "#0f7350"]),
          "full_attention": dict(name="full-attention-140m-11.5bt", board="e38_rope150m_final", core7="e38_rope150m_core7",
                                 color=INK, short="full-attention", prefix="fa_", file="compare_full_attention.json",
                                 ramp=["#f3f2ef", "#c3c2b7", "#6f6e69", "#2c2c2a"])}
for _k, _m in FAMILY.items():
    _m.update(key=_k, tokens=11.5e9, family=True)
OURS = dict(FAMILY[CARD], name=FAMILY[CARD]["name"] + " (this model)", long="compare_ours.json", ruler="ruler_")
# the other models on the same data whose results ship in this folder, in the family's order
OTHERS = [dict(m, long=m["file"], ruler=m["prefix"] + "ruler_") for k, m in FAMILY.items()
          if k != CARD and ((LC / m["file"]).exists() or (ER / f"{m['board']}.json").exists())]
# the second model on the radar and the RULER radars: the full-attention model on the same data (the ceiling inside
# its training length); on its own card, the first memory model
_ctrl = "full_attention" if CARD != "full_attention" else "concept_memory"
CTRL = next((m for m in OTHERS if m["key"] == _ctrl), None)
SMOL = dict(name="SmolLM2-135M, earliest public checkpoint", board="ref_smollm2_135m_step240k_board",
            core7="ref_smollm2_135m_step240k_core7", tokens=240_000 * 1_048_576, color=MUTED, short="SmolLM2")
SMOL_FINAL = dict(name="SmolLM2-135M", board="ref_smollm2_135m_board", tokens=2.0e12, color=INK2, marker="s")
PYTH = dict(name="Pythia-160M", tokens=299.9e9, color=MUTED, marker="o")
# the long-document chart: (file, label, parameters, training tokens as reported, colour or None for grey)
LONG = [(OURS["long"], OURS["name"], "", "", OURS["color"])] + \
       [(m["long"], m["name"], "", "", m["color"]) for m in OTHERS] + [  # the Hub id already names size and budget
        ("compare_smollm2_step240k.json", "SmolLM2-135M, 252B checkpoint", "135M", "252B", None),
        ("compare_smollm2.json", "SmolLM2-135M final", "135M", "≈2T", None),
        ("compare_pythia.json", "Pythia-160M", "162M", "300B", None),
        ("compare_gemma.json", "Gemma 3 270M", "268M", "6T", None),
        ("compare_granite.json", "Granite 4.0 350M", "352M", "≈15T", None),
        ("compare_lfm25.json", "LFM2.5-350M", "354M", "28T", None)]
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


AXES = {"Grammar\n(BLiMP)": True, "Science quiz\n(ARC-Easy)": True, "Hard science quiz\n(ARC-Challenge)": True,
        "Science with a passage\n(SciQ)": True, "Sensible continuation\n(HellaSwag)": True,
        "Wikipedia text\n(WikiText-2)": False}  # higher is better?
UNITS = {a: "{:.1f} %" for a in AXES}
UNITS["Grammar\n(BLiMP)"] = UNITS["Science quiz\n(ARC-Easy)"] = "{:.2f} %"
UNITS["Wikipedia text\n(WikiText-2)"] = "{:.3f} bits/byte"


def radar(models: list[dict]) -> None:
    vals = [(m, measures(m)) for m in models]
    vals = [(m, v) for m, v in vals if v]
    if len(vals) < 2:
        print("skip radar: missing results", file=sys.stderr)
        return
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
        disp = np.pi / 2 - t
        c, sn = math.cos(disp), math.sin(disp)
        ha = "center" if abs(c) < 0.2 else ("left" if c > 0 else "right")
        vals_txt = "\n".join(f"{m['short']}  {UNITS[a].format(v[a])}" for m, v in vals)
        kw = dict(xy=(t, 1.04), textcoords="offset points", ha=ha, annotation_clip=False)
        if sn > 0.8:
            ax.annotate(vals_txt, xytext=(0, 4), va="bottom", fontsize=7.5, color=INK2, **kw)
            ax.annotate(a, xytext=(0, 42), va="bottom", fontsize=10, color=INK, fontweight="bold", **kw)
        elif sn < -0.8:
            ax.annotate(a, xytext=(0, -4), va="top", fontsize=10, color=INK, fontweight="bold", **kw)
            ax.annotate(vals_txt, xytext=(0, -34), va="top", fontsize=7.5, color=INK2, **kw)
        else:
            dx = 6 if c > 0 else -6
            ax.annotate(a, xytext=(dx, 2), va="bottom", fontsize=10, color=INK, fontweight="bold", **kw)
            ax.annotate(vals_txt, xytext=(dx, -4), va="top", fontsize=7.5, color=INK2, **kw)
    for m, v in reversed(vals):  # ours drawn last, on top
        r = [norm(a, v[a]) for a in axes]
        ours = m is vals[0][0]
        ax.fill(np.append(ang, ang[0]), r + r[:1], color=m["color"], alpha=0.16 if ours else 0.08, zorder=2)
        ax.plot(np.append(ang, ang[0]), r + r[:1], color=m["color"], linewidth=2.4 if ours else 1.8, zorder=3,
                linestyle="-" if ours else (0, (5, 2)) if m.get("family") else "-")
        ax.scatter(ang, r, s=34, color=m["color"], edgecolor=SURFACE, linewidth=2, zorder=4)
    handles = [plt.Line2D([], [], color=m["color"], linewidth=2.4, marker="o", markersize=6,
                          linestyle=(0, (5, 2)) if m.get("family") and m is not vals[0][0] else "-",
                          label=f"{m['name']} · {fmt_tokens(m['tokens'])} training tokens")
               for m, _ in vals]
    fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.012, "Further out is better on every axis: each score as a share of the best of the three "
             "(rim = best, centre = zero;\nfor WikiText, lower is better, so best ÷ score). Accuracy in %, 0-shot; "
             "WikiText in bits per byte. Same harness for all three, bf16.", ha="center", fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.22, right=0.78)
    fig.savefig(OUT / "radar.png", dpi=160)
    print("wrote radar.png")


# ------------------------------------------------------------------------------------------------------------ RULER
RULER_FAMILIES = [("One needle", ["niah_single_1", "niah_single_2", "niah_single_3"]),
                  ("Needle among distractor keys", ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3"]),
                  ("Several values or keys at once", ["niah_multivalue", "niah_multiquery"]),
                  ("Count words", ["ruler_cwe", "ruler_fwe"]),
                  ("Follow variables", ["ruler_vt"]),
                  ("Answer questions", ["ruler_qa_squad", "ruler_qa_hotpot"])]
RULER_TASKS = [("niah_single_1", "single needle, noise filler"), ("niah_single_2", "single needle, essays"),
               ("niah_single_3", "single needle, UUID value"), ("niah_multikey_1", "multi-key 1"),
               ("niah_multikey_2", "multi-key 2"), ("niah_multikey_3", "multi-key 3 (UUIDs)"),
               ("niah_multivalue", "multi-value"), ("niah_multiquery", "multi-query"), ("ruler_vt", "variable tracking"),
               ("ruler_cwe", "common words"), ("ruler_fwe", "frequent words"), ("ruler_qa_squad", "QA (SQuAD)"),
               ("ruler_qa_hotpot", "QA (HotpotQA)")]
RULER_REFS = [("ref_smollm2_135m", "SmolLM2-135M", "≈2T"), ("ref_gemma3_270m", "Gemma 3 270M", "6T"),
              ("ref_granite4_350m", "Granite 4.0 350M", "≈15T"), ("ref_lfm25_350m", "LFM2.5-350M-Base", "28T")]


def ruler_scores(prefix: str) -> dict[int, dict[str, float]]:
    """{length: {task: score}} merged over the run files whose name starts with `prefix` (lengths may be split)."""
    res: dict[int, dict[str, float]] = {}
    for f in sorted(RU.glob(f"{prefix}*.json")):
        for L, r in json.load(open(f))["results"].items():
            res.setdefault(int(L), {}).update({t: v[f"{L},none"] for t, v in r.items()
                                               if isinstance(v, dict) and isinstance(v.get(f"{L},none"), (int, float))})
    return {L: r for L, r in res.items() if len(r) == 13}


def ruler_family(r: dict[str, float]) -> list[float]:
    return [float(np.mean([r[t] for t in ts])) for _, ts in RULER_FAMILIES]


def spread(ys: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Label positions for end values ys (any order): at least `gap` apart, inside [lo, hi], each as near its value."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [0.0] * len(ys)
    prev = lo - gap
    for i in order:  # bottom up, pushing each label above the last
        pos[i] = max(ys[i], prev + gap)
        prev = pos[i]
    over = max(pos) - hi
    if over > 0:  # off the top: shift every label down by the overflow, keeping the gaps
        top = None
        for i in reversed(order):
            pos[i] = min(pos[i] - over, top - gap) if top is not None else pos[i] - over
            top = pos[i]
    return pos


def ruler_lengths() -> None:
    """Small multiples, one per task family: score against length, this model and the other models on the same data."""
    runs = [(OURS, ruler_scores(OURS["ruler"]))] + [(m, ruler_scores(m["ruler"])) for m in OTHERS]
    runs = [(m, sc) for m, sc in runs if sc]
    Ls = sorted(set.intersection(*[set(sc) for _, sc in runs])) if len(runs) > 1 else []
    if not Ls:
        print("skip ruler_lengths: missing results", file=sys.stderr)
        return
    fig, axs = plt.subplots(2, 3, figsize=(10.6, 6.0), sharex=True, sharey=True)
    for ax, (fam, _) in zip(axs.flat, RULER_FAMILIES):
        i = [f for f, _ in RULER_FAMILIES].index(fam)
        ends = []
        for m, sc in reversed(runs):  # this model drawn last, on top
            ys = [ruler_family(sc[L])[i] for L in Ls]
            ax.plot(range(len(Ls)), ys, color=m["color"], linewidth=2.4 if m is OURS else 1.8,
                    linestyle="-" if m is OURS else (0, (5, 2)), marker="o", markersize=6 if m is OURS else 5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4 if m is OURS else 3)
            ends.append(ys[-1])
        for y, yy in zip(ends, spread(ends, 0.075, 0.02, 1.0)):  # end values, nudged apart
            ax.annotate(f"{y:.2f}", xy=(len(Ls) - 1, y), xytext=(len(Ls) - 1 + 0.12, yy), textcoords="data",
                        va="center", fontsize=7.5, color=INK2, annotation_clip=False)
        ax.set_title(fam, fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(len(Ls)))
        ax.set_xticklabels([f"{L // 1024}k" for L in Ls])
        ax.set_xlim(-0.25, len(Ls) - 0.35)
        ax.grid(color=GRID, linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("share of answers found ↑")
    mean = lambda sc, L: np.mean(list(sc[L].values()))
    handles = [plt.Line2D([], [], color=m["color"], linewidth=2.4 if m is OURS else 1.8, marker="o",
                          linestyle="-" if m is OURS else (0, (5, 2)),
                          label=f"{m['name']} · mean of 13 tasks: " +
                                " / ".join(f"{mean(sc, L):.2f}" for L in Ls) + " at " + " / ".join(f"{L // 1024}k" for L in Ls))
               for m, sc in runs]
    fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "RULER, 100 prompts per task and length, greedy, each task's own answer length; a family "
             "averages its tasks; same harness and settings for every model.\nAll trained on planted-fact lookups, "
             "so the needle tasks are not zero-shot. Run-to-run noise at 100 prompts is up to ±0.09 on a single task.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1 - 0.035 * (len(runs) + 1)))
    fig.savefig(OUT / "ruler_lengths.png", dpi=160)
    print("wrote ruler_lengths.png")


def ruler_heat() -> None:
    ours = ruler_scores(OURS["ruler"])
    if not ours:
        print("skip ruler_heat: missing results", file=sys.stderr)
        return
    Ls = sorted(ours)
    M = np.array([[ours[x][t] for x in Ls] for t, _ in RULER_TASKS])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ours", OURS["ramp"])  # one hue: this model's colour
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color=SURFACE if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([f"{x // 1024}k" for x in Ls])
    ax.set_yticks(range(len(RULER_TASKS)))
    ax.set_yticklabels([n for _, n in RULER_TASKS], fontsize=8.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"RULER, {FAMILY[CARD]['name']}:\nshare of expected answers found", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "ruler_heat.png", dpi=160)
    print("wrote ruler_heat.png")


def ruler_radar(L: int = 8192) -> None:
    ours, other = ruler_scores(OURS["ruler"]), (ruler_scores(CTRL["ruler"]) if CTRL else {})
    refs = [(n, lab, tok, ruler_scores(n)) for n, lab, tok in RULER_REFS]
    refs = [x for x in refs if L in x[3]]
    if L not in ours or not refs:
        print("skip ruler_radar: missing results", file=sys.stderr)
        return
    names = ["One needle", "Needle among\ndistractor keys", "Several values\nor keys at once", "Count words",
             "Follow\nvariables", "Answer\nquestions"]
    ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    fig, axs = plt.subplots(1, len(refs), figsize=(3.9 * len(refs), 4.6), subplot_kw={"projection": "polar"})
    for ax, (n, lab, tok, sc) in zip(np.atleast_1d(axs), refs):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetalim(0, 2 * np.pi)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1"], fontsize=7, color=MUTED)
        ax.set_rlabel_position(90)
        ax.grid(color=GRID, linewidth=0.7)
        ax.spines["polar"].set_color(GRID)
        ax.set_xticks(ang)
        ax.set_xticklabels(names, fontsize=7.5, color=INK2)
        ax.tick_params(axis="x", pad=2)
        lines = [(ruler_family(sc[L]), MUTED, 1.6, 2, "-")]
        if L in other:
            lines.append((ruler_family(other[L]), CTRL["color"], 1.8, 3, (0, (5, 2))))
        lines.append((ruler_family(ours[L]), OURS["color"], 2.4, 4, "-"))
        for vals, col, w, z, ls in lines:
            ax.fill(np.append(ang, ang[0]), vals + vals[:1], color=col, alpha=0.10, zorder=z - 1)
            ax.plot(np.append(ang, ang[0]), vals + vals[:1], color=col, linewidth=w, zorder=z, linestyle=ls)
            ax.scatter(ang, vals, s=18, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=z + 1)
        ax.set_title(f"vs {lab} · {tok} tokens\nmean of 13: this model {np.mean(list(ours[L].values())):.2f}, "
                     f"it {np.mean(list(sc[L].values())):.2f}", fontsize=8.5, color=INK, pad=28)
    handles = [plt.Line2D([], [], color=OURS["color"], linewidth=2.4, marker="o", label=f"{OURS['name']} · 11.5B tokens"),
               *([plt.Line2D([], [], color=CTRL["color"], linewidth=1.8, marker="o", linestyle=(0, (5, 2)),
                             label=CTRL["name"])] if L in other else []),
               plt.Line2D([], [], color=MUTED, linewidth=1.6, marker="o", label="the reference named above each chart")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.01, f"RULER at {L // 1024}k tokens, 100 prompts per task, greedy, each task's own answer length; "
             "score = share of expected answers found (0 centre, 1 rim). Families average their tasks. Our models "
             "trained on planted-fact lookups, so their needle scores are not zero-shot.", ha="center", fontsize=7.5,
             color=MUTED, wrap=True)
    fig.subplots_adjust(top=0.70, bottom=0.13, left=0.05, right=0.95, wspace=0.75)
    fig.savefig(OUT / "ruler_radar.png", dpi=160)
    print("wrote ruler_radar.png")


# ---------------------------------------------------------------------------------------------------- long documents
def long_docs() -> None:
    """Bits per byte against context, one line per model on the same text; our two models in colour, others grey."""
    fig, ax = plt.subplots(figsize=(10.4, 5.1))
    end_labels = []
    for fname, label, params, tok, col in LONG:
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
            xs.append(min(int(k), int(r["max_len"])) if win else int(k))
            ys.append(pooled(np.array([by[d]["cut"][k] for d in docs]), nb))
            if win:
                break
        hi = col is not None
        ax.plot(xs, ys, color=col or MUTED, linewidth=2.6 if hi else 1.4, marker="o", markersize=6 if hi else 4,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4 if hi else 3,
                linestyle=(0, (5, 2)) if any(fname == m["long"] for m in OTHERS) else "-")
        end_labels.append((xs[-1], ys[-1], f"{label}" + (f" · {params} · {tok} tokens" if params else "") + f"   {ys[-1]:.3f}",
                           col or MUTED, hi))
    end_labels.sort(key=lambda e: e[1])
    placed = []
    for x, y, txt, col, hi in end_labels:  # direct labels in one column right of the plot, nudged apart vertically
        yy = y
        for p in placed:
            if abs(yy - p) < 0.0125:
                yy = p + 0.0125
        placed.append(yy)
        ax.annotate(txt, xy=(x, y), xytext=(61439 * 1.3, yy), va="center", fontsize=8.5, color=INK if hi else INK2,
                    fontweight="bold" if hi else "normal", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8, shrinkB=3, relpos=(0, 0.5)))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 4096, 16384, 28672, 61439])
    ax.set_xticklabels(["1k", "4k", "16k", "28k", "61k\n(whole span)"])
    ax.set_xlim(800, 61439 * 1.05)
    ax.set_ylim(0.958, 0.765)  # inverted: up is better
    ax.set_ylabel("bits per byte on the last pages (lower is better)")
    ax.set_xlabel("context before the last pages, in tokens (log scale)")
    ax.grid(color=GRID, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.01, "Up is better. 55 held-out documents (books, papers, government and parliament text), "
             "the same characters for every model. A line ends at the most\ncontext the model can take; models "
             "that cannot take a budget are scored with sliding windows there. For scale: gzip needs 2.49 bits per "
             "byte on\nthe same pages, xz 1.97, plain text 8.", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.55, top=0.95, bottom=0.22)
    fig.savefig(OUT / "long_docs.png", dpi=160)
    print("wrote long_docs.png")


# --------------------------------------------------------------------------------------------------- data efficiency
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
    others = [(m, board(m["board"])) for m in OTHERS]
    others = [(m, b) for m, b in others if b]
    if ours is None:
        print("skip data_eff: missing results", file=sys.stderr)
        return
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
                ax.plot([p[0] for p in pts], [get(p[1]) for p in pts], color=m["color"], linewidth=2, marker=m["marker"],
                        markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=m["name"])
        # all at 11.5B tokens: drawn a little apart on the token axis so none hides another, this model rightmost
        n = len(others)
        for j, (m, b) in enumerate(others):
            ax.scatter([m["tokens"] * 1.1 ** (j - n)], [get(b)], s=70, color=m["color"], edgecolor=SURFACE,
                       linewidth=2, zorder=5, label=m["name"], marker="D")
        ax.scatter([OURS["tokens"] * 1.1], [get(ours)], s=100, color=OURS["color"], edgecolor=SURFACE, linewidth=2, zorder=6,
                   label=OURS["name"])
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
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, "Up is better on every panel. Every point scored by us on one RTX 3090, same harness, "
             "0-shot. Checkpoints: Pythia-160M (EleutherAI, scored in fp32), SmolLM2-135M (Hugging Face, from ≈250B "
             "tokens).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(OUT / "data_eff.png", dpi=160)
    print("wrote data_eff.png")
    for key, title, get in panels:  # tokens Pythia-160M needs to match us, log-linear between its checkpoints
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


if __name__ == "__main__":
    radar([OURS] + ([CTRL] if CTRL else []) + [SMOL])
    ruler_lengths()
    ruler_heat()
    ruler_radar()
    long_docs()
    data_efficiency()
