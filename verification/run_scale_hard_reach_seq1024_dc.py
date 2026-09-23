#!/usr/bin/env python
"""Matched C and param-matched D9 at seq=1024 true reach. Do not rerun A.

Arm A already hit 95.9% at 96k (lr=1e-4, min_gap=256, 128 slots). The JSON
write to /opt/cursor/artifacts failed after the artifact store wiped; the
numbers live in docs/4_Research_Notes/exclusive_slot_law_inventory.json.
This runner only fills the missing same-regime controls:

  D9  9 decoder layers, 4.97M, full-causal, lr=1e-4, 8000 steps / 95% bar
  C   4 decoder layers, 2.27M, segment-confined, leak check, 1200 steps

JSON lands on /workspace/Cache/scale_hard (workspace disk). Do not shrink A.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("SCALE_OUT_DIR", "/workspace/Cache/scale_hard"))
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
    "task": "far_copy",
    "seq_len": 1024,
    "min_gap": 256,
    "span_len": 32,
    "ratio": 8,
    "dec_segment": 32,
    "hops": 2,
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        ">>> REACH seq=1024 min_gap=256 r=8 slots=128 lr=1e-4  (A already 95.9% @ 96k; D9 then C)",
        flush=True,
    )
    d = run(
        "reach_seq1024_D9",
        arms=["D"],
        dec_layers=9,
        lr=1e-4,
        steps=8000,
        eval_every=250,
        target_acc=TARGET,
        floor_patience_steps=4000,
    )
    print("D9", d["summary"]["D"], flush=True)
    c = run(
        "reach_seq1024_C",
        arms=["C"],
        lr=1e-4,
        steps=1200,
        eval_every=250,
        target_acc=0.0,
        floor_patience_steps=0,
    )
    print("C", c["summary"]["C"], flush=True)
    if acc(d, "D") < TARGET:
        print("KILL reach seq=1024: param-matched D missed 95%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
