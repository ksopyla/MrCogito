#!/usr/bin/env python
"""True length axis: grow min_gap with seq, frozen 5.11M A.

seq=512 min_gap=32 (already measured) only adds slots of padding. This runner
places the 32-letter span farther back: min_gap = seq/4, r=8, exclusive
scope, LR searched per cell. D uses 9 layers (4.97M) so params match A.
C is the leak check. 95% bar. Do not shrink A.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path("/opt/cursor/artifacts/scale_hard")
WRAPPER = Path("/workspace/verification/run_scale_job.sh")
TARGET = 0.95

REF = {
    "hidden": 256,
    "token_embedding_dim": 32,
    "enc_layers": 2,
    "enc_window": 16,
    "latent_layers": 2,
    "dec_layers": 4,
    "scope": "exclusive",
    "xattn_wo_init": 0.02,
    "pooler_wo_init": 0.02,
    "batch": 32,
    "sched": "warmup_constant",
    "warmup_steps": 200,
    "eval_rows": 128,
    "threads": 4,
    "seed": 0,
    "n_symbols": 4,
}


def _argv(name: str, kw: dict) -> list[str]:
    args = ["bash", str(WRAPPER), name]
    for k, v in kw.items():
        flag = f"--{k}"
        if isinstance(v, (list, tuple)):
            args += [flag, *map(str, v)]
        else:
            args += [flag, str(v)]
    return args


def run(name: str, **kw) -> dict:
    path = OUT / f"{name}.json"
    if path.exists():
        print(f"SKIP {name} (json exists)", flush=True)
        return json.loads(path.read_text())
    merged = {**REF, **kw}
    argv = _argv(name, merged)
    env = os.environ.copy()
    env["SCALE_OUT_DIR"] = str(OUT)
    print(f"\n===== {name} =====\n{' '.join(argv[3:])}\n", flush=True)
    rc = subprocess.call(argv, env=env)
    if not path.exists():
        raise SystemExit(f"{name} produced no JSON (rc={rc})")
    return json.loads(path.read_text())


def acc(b: dict, arm: str) -> float:
    return float(b["summary"][arm]["acc"])


def train(name, arm, lr, steps, target, patience, **cell):
    return run(
        name,
        **cell,
        arms=[arm],
        lr=lr,
        steps=steps,
        eval_every=250,
        target_acc=target,
        floor_patience_steps=patience,
    )


def search_lr(prefix, arm, lrs, **cell):
    ranked = []
    for lr in lrs:
        tag = f"{lr:.0e}".replace("-", "")
        b = train(f"{prefix}_{arm}_lr{tag}", arm, lr, 800, TARGET, 800, **cell)
        print(f"  LR {lr:.0e}: acc={acc(b, arm):.3f}", flush=True)
        ranked.append((lr, b))
        if acc(b, arm) >= TARGET:
            return lr, b
    ranked.sort(key=lambda t: acc(t[1], arm), reverse=True)
    return ranked[0][0], ranked[0][1]


def reach_cell(seq: int) -> dict:
    return {
        "task": "far_copy",
        "seq_len": seq,
        "min_gap": max(32, seq // 4),
        "span_len": 32,
        "ratio": 8,
        "dec_segment": 32,
        "hops": 2,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for seq in (512, 1024):
        cell = reach_cell(seq)
        print(f"\n>>> REACH seq={seq} min_gap={cell['min_gap']} r=8 slots={seq // 8}", flush=True)
        lr, probe = search_lr(f"reach_seq{seq}", "A", (3e-4, 1e-4, 1e-3), **cell)
        if acc(probe, "A") < TARGET:
            a = train(f"reach_seq{seq}_A", "A", lr, 8000, TARGET, 2500, **cell)
        else:
            a = probe
        print("A", a["summary"]["A"], flush=True)
        d = train(
            f"reach_seq{seq}_D9",
            "D",
            lr,
            8000,
            TARGET,
            4000,
            **cell,
            dec_layers=9,
        )
        print("D9", d["summary"]["D"], flush=True)
        train(f"reach_seq{seq}_C", "C", lr, 1200, 0.0, 0, **cell)
        if acc(d, "D") < TARGET:
            print(f"KILL reach seq={seq}: param-matched D missed 95%", flush=True)
        if acc(a, "A") < TARGET:
            print(f"A missed 95% at seq={seq} min_gap={cell['min_gap']}", flush=True)
    return subprocess.call(["uv", "run", "python", "/workspace/verification/plot_scale_hard.py"])


if __name__ == "__main__":
    sys.exit(main())
