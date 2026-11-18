"""Tables for the cross-model real-document test (outputs of evaluation/long_context_compare.py score).

    python summarise_compare.py compare_ours.json compare_smollm2.json ...   (the first file is the reference)

Bits per byte on the target text (the last 4,096 SmolLM2 tokens of each 64k span), pooled over documents: summed nats
/ ln 2 / summed bytes, lower is better. Intervals: 95 % bootstrap over documents (2,000 resamples, seed 0).
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

LN2 = math.log(2)
NAMES = {"lsa_m120t_final": "150M topic index (ours)", "e37a_m120t_final": "150M concept memory (sibling)",
         "SmolLM2-135M": "SmolLM2-135M", "pythia-160m": "Pythia-160M",
         "gemma-3-270m": "Gemma 3 270M", "granite-4.0-350m-base": "Granite 4.0 350M base",
         "smollm2_135m_step240k": "SmolLM2-135M at 252B tokens (step 240,000)", "lfm25_350m_base": "LFM2.5-350M-Base"}


def load(path: str):
    r = json.load(open(path))
    rows = [x for src in r["sources"].values() for x in src if "cut" in x]
    keys = [str(b) for b in r["budgets"]]
    by = {(x["doc"], x["target_bytes"]): x for x in rows}
    return r, keys, by


def pooled(nats: np.ndarray, nbytes: np.ndarray) -> float:
    return float(nats.sum() / LN2 / nbytes.sum())


def boot(fn, n: int, reps: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    vals = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(paths: list[str]) -> None:
    runs = [load(p) for p in paths]
    common = set.intersection(*(set(by) for _, _, by in runs))
    docs = sorted(common)
    nb = np.array([d[1] for d in docs], dtype=float)
    print(f"{len(docs)} documents in common; target {nb.sum() / 1e6:.2f} MB of text\n")
    ref_full = None
    vs = []
    print("| Model | Params | Max length | 1k | 4k | 16k | 28k | whole document | saved, 1k → most context |"
          " docs better with more | another document's far past |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, (r, keys, by) in enumerate(runs):
        cut = {k: np.array([by[d]["cut"][k] for d in docs]) for k in keys}
        win = {k: [by[d]["windowed"][k] for d in docs] for k in keys}
        full = keys[-1]
        # the most context this model reads: the first windowed budget (all larger ones copy it), else the full span
        most = next((k for k in keys if all(win[k])), full)
        b = {k: pooled(cut[k], nb) for k in keys}
        gain = b[keys[0]] - b[most]
        lo, hi = boot(lambda ix: pooled(cut[keys[0]][ix], nb[ix]) - pooled(cut[most][ix], nb[ix]), len(docs))
        better = float(np.mean(cut[most] < cut[keys[0]])) * 100
        # budgets past the model's limit read the same last tokens as `most`: shown once, marked "(limit)"
        cells = [f"{b[k]:.3f}" + (" (limit)" if k == most and all(win[k]) else "") if keys.index(k) <= keys.index(most)
                 else "—" for k in keys]
        swap = ""
        if all("swap" in by[d] for d in docs):
            s = np.array([by[d]["swap"] for d in docs])
            swap = f"{pooled(s, nb):.3f}"
        name = NAMES.get(r["model"].rstrip("/").split("/")[-1], r["model"])
        if r.get("lead_mode", "default") != "default":
            name += f" (start: {r['lead_mode']})"
        print(f"| {name} | {r['params'] / 1e6:.1f}M | {r['max_len']:,} | " + " | ".join(cells) +
              f" | {gain:.3f} [{lo:.3f}, {hi:.3f}] | {better:.0f} % | {swap} |")
        if i == 0:
            ref_full, ref_name = cut[most], name
        else:
            d_ = pooled(cut[most], nb) - pooled(ref_full, nb)
            lo2, hi2 = boot(lambda ix: pooled(cut[most][ix], nb[ix]) - pooled(ref_full[ix], nb[ix]), len(docs))
            wins = float(np.mean(ref_full / nb < cut[most] / nb)) * 100
            vs.append(f"| {name} | {pooled(cut[most], nb):.3f} | {pooled(ref_full, nb):.3f} | {d_:+.3f} "
                      f"[{lo2:+.3f}, {hi2:+.3f}] | {wins:.0f} % |")
    if vs:
        print(f"\nEach model with the most of the document it can take, against {ref_name} with the whole document:\n")
        print(f"| Model | its bits per byte | {ref_name} | difference (+ = {ref_name} better) | documents where "
              f"{ref_name} is better |")
        print("|---|---:|---:|---:|---:|")
        print("\n".join(vs))


if __name__ == "__main__":
    main(sys.argv[1:])
