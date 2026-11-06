"""E33p instrumentation part B, the comparison (plan §8 B3/B4): linear CKA between arms and across loop rounds, plus a
table of the probe metrics, from evaluation/flow_probe.py outputs.

Usage: uv run python scripts/compare_flow.py --arm dense=dense.json:dense_acts.pt --arm notes_open=n.json:n_acts.pt \
          ... --out flow_compare.json
Linear CKA (Kornblith et al. 2019) of two layers' outputs X [N, d], Y [N, d'] on the same tokens, centred:
‖Yᵀ X‖²_F / (‖Xᵀ X‖_F ‖Yᵀ Y‖_F); 1 = the same representation up to rotation and scale.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    num = (y.T @ x).pow(2).sum()
    den = (x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt()
    return float(num / den.clamp_min(1e-30))


def _layer_sort(k: str):
    m = re.match(r"L(\d+)(?:/r(\d+))?", k)
    return (int(m.group(1)), int(m.group(2) or -1)) if m else (10**6, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=PROBE.json:ACTS.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {}
    for spec in a.arm:
        name, files = spec.split("=", 1)
        js, acts = files.split(":", 1)
        arms[name] = (json.loads(Path(js).read_text()), torch.load(acts, weights_only=True))
    names = list(arms)
    keys = sorted(set.intersection(*(set(arms[n][1]) for n in names)), key=_layer_sort)
    between = {}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            between[f"{n1}|{n2}"] = {k: linear_cka(arms[n1][1][k], arms[n2][1][k]) for k in keys}
    within = {}
    for n in names:
        acts = arms[n][1]
        rounds = {}
        for k in acts:
            m = re.match(r"L(\d+)/r(\d+)$", k)
            if m:
                rounds.setdefault(int(m.group(1)), {})[int(m.group(2))] = acts[k]
        within[n] = {f"L{l}": {f"r{r}|r{r + 1}": linear_cka(rs[r], rs[r + 1]) for r in sorted(rs) if r + 1 in rs}
                     for l, rs in sorted(rounds.items())}
    out = {"arms": names, "layers": keys, "cka_between": between, "cka_rounds": within,
           "metrics": {n: arms[n][0]["metrics"] for n in names},
           "lens": {n: {k: v for k, v in arms[n][0]["scalars"].items() if k.startswith("lens/")} for n in names},
           "dist_mass": {n: arms[n][0]["dist_mass"] for n in names}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for pair, d in between.items():
        print(pair, " ".join(f"{k}:{v:.2f}" for k, v in d.items()))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
