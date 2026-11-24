"""The depth wiring of a trained perceiver_ar checkpoint, read from its weights (E40 pre-flight P2, 2026-11-24).

Every block's input is  α · x + β · x0 + σ · skip  (nn/perceiver_ar_lm.py `Block.mix`): x the stream, x0 the
embedding, skip the U-net skip (layers [0, n/2) push their output, layers [n − n/2, n) pop them in reverse; inside the
reading layer's loop only the last pass's pushes survive). This prints α, β per layer, σ per skip pair (which layer's
output feeds which later layer) and the loop markers' sizes. With E40's depth mixes (`par_depth_mix`) it also prints
what each mix learned: per column, the mean and spread of its per-dimension weight b, and the size of its
input-dependent weight w (weights only; what w does per token needs activations).

Usage:
  uv run python scripts/depth_weights.py <checkpoint folder or model.safetensors> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(path: str) -> tuple[dict, dict]:
    from safetensors.torch import load_file

    p = Path(path)
    f = p if p.suffix == ".safetensors" else p / "model.safetensors"
    sd = load_file(str(f))
    cfg_f = f.parent / "config.json"
    cfg = json.loads(cfg_f.read_text()) if cfg_f.exists() else {}
    return sd, cfg


def _layer_key(sd: dict, i: int, name: str):
    for k in (f"layers.{i}.{name}", f"model.layers.{i}.{name}", f"perceiver_ar.layers.{i}.{name}"):
        if k in sd:
            return sd[k]
    pat = re.compile(rf"(^|\.)layers\.{i}\.{re.escape(name)}$")
    for k, v in sd.items():
        if pat.search(k):
            return v
    return None


def wiring(sd: dict, cfg: dict) -> dict:
    n = int(cfg.get("num_hidden_layers", 0)) + int(cfg.get("pre_layers", 0)) + int(cfg.get("global_layers", 0)) \
        if cfg else 0
    if not n:
        n = 1 + max(int(m.group(1)) for k in sd if (m := re.search(r"layers\.(\d+)\.alpha$", k)))
    n_skip = n // 2
    gi = int(cfg.get("pre_layers", 0)) if cfg else None
    loop_span = int(cfg.get("message_loop_span", 1) or 1) if cfg else 1
    rounds = int(cfg.get("message_loop_rounds", 1) or 1) if cfg else 1

    def role(i: int) -> str:
        if gi is not None and i == gi:
            return "reading layer"
        if gi is not None and rounds > 1 and gi < i <= gi + loop_span:
            return "local, looped with the reading layer"
        return "local"

    layers = []
    for i in range(n):
        a, b, s = (_layer_key(sd, i, x) for x in ("alpha", "beta", "sigma"))
        row = {"layer": i, "role": role(i), "alpha": float(a) if a is not None else None,
               "beta": float(b) if b is not None else None}
        if s is not None:
            row["sigma"] = float(s)
            row["skip_from"] = n - 1 - i
            row["skip_from_role"] = role(n - 1 - i)
        layers.append(row)
    loop = None
    for k, v in sd.items():
        if k.endswith("loop_emb"):
            loop = [float(r.norm()) for r in v.float()]
    return {"n_layers": n, "reading_layer": gi, "loop_rounds": rounds, "layers": layers, "loop_emb_norms": loop}


def mixes(sd: dict) -> dict:
    """E40 depth mixes (nn/depth_mix.py): per mix, per column, the mean and spread of b (= 1 + b') and |w|."""
    out = {}
    for k, v in sd.items():
        m = re.search(r"(depth_mix\.[\w.]+)\.b_delta$", k)
        if not m:
            continue
        base = k[: -len(".b_delta")]
        b = 1.0 + v.float()
        w = sd.get(base + ".w")
        out[base] = {"columns": int(b.shape[0]), "b_mean": [round(float(r.mean()), 4) for r in b],
                     "b_std": [round(float(r.std()), 4) for r in b],
                     "w_norm": round(float(w.float().norm()), 4) if w is not None else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--json", help="also write the numbers here")
    a = ap.parse_args()
    sd, cfg = load(a.checkpoint)
    w = wiring(sd, cfg)
    w["mixes"] = mixes(sd)
    print(f"{a.checkpoint}: {w['n_layers']} layers, reading layer {w['reading_layer']}, loop x{w['loop_rounds']}")
    print(f"{'layer':>5}  {'role':38} {'alpha':>7} {'beta':>7} {'sigma':>7}  skip from")
    for r in w["layers"]:
        s = f"{r['sigma']:+7.3f}  layer {r['skip_from']} ({r['skip_from_role']})" if "sigma" in r else ""
        print(f"{r['layer']:>5}  {r['role']:38} {r['alpha']:+7.3f} {r['beta']:+7.3f} {s}")
    if w["loop_emb_norms"]:
        print("loop markers (norm per pass):", ", ".join(f"{x:.3f}" for x in w["loop_emb_norms"]))
    for name, m in w["mixes"].items():
        print(f"{name}: {m['columns']} columns, b mean {m['b_mean']}, b spread {m['b_std']}, |w| {m['w_norm']}")
    if a.json:
        Path(a.json).write_text(json.dumps(w, indent=1))


if __name__ == "__main__":
    main()
