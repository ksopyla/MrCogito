#!/usr/bin/env python
"""Continue the <10M exclusive-slot law: param-matched D, then harder cells.

The closed grid was width-matched (A 5.11M vs C/D 2.27M). The scaling-law
goal requires the same parameter regime. D with 9 decoder layers is 4.97M
(closest to A's 5.11M without going over 10M). Then push r=16, hops=2 packed
chain, and seq=1024 at frozen hidden=256.

Does not rerun closed cells. LR is searched per new geometry.
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
A_PARAMS = 5_107_858
D9_LAYERS = 9  # 4.97M — closest width-matched-depth upgrade to A's 5.11M

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
    merged = {**REF, **kw}
    argv = _argv(name, merged)
    env = os.environ.copy()
    env["SCALE_OUT_DIR"] = str(OUT)
    print(f"\n===== {name} =====\n{' '.join(argv[3:])}\n", flush=True)
    rc = subprocess.call(argv, env=env)
    path = OUT / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{name} produced no JSON (rc={rc})")
    return json.loads(path.read_text())


def acc(bundle: dict, arm: str) -> float:
    return float(bundle["summary"][arm]["acc"])


def stop(bundle: dict, arm: str) -> str:
    return str(bundle["summary"][arm].get("stop_reason", ""))


def train(name: str, arm: str, lr: float, steps: int, target: float, patience: int, **cell) -> dict:
    return run(
        name,
        **cell,
        arms=[arm],
        lr=lr,
        steps=steps,
        eval_every=250 if steps > 800 else 200,
        target_acc=target,
        floor_patience_steps=patience,
    )


def search_lr(prefix: str, arm: str, lrs: tuple[float, ...], **cell) -> tuple[float, dict]:
    """Short probes, keep the LR that left chance; then a long run of the winner."""
    ranked: list[tuple[float, dict]] = []
    for lr in lrs:
        tag = f"{lr:.0e}".replace("-", "")
        b = train(
            f"{prefix}_{arm}_lr{tag}",
            arm,
            lr,
            steps=800,
            target=TARGET,
            patience=800,
            **cell,
        )
        a = acc(b, arm)
        print(f"  LR {lr:.0e}: acc={a:.3f} stop={stop(b, arm)}", flush=True)
        ranked.append((lr, b))
        if a >= TARGET:
            return lr, b
    ranked.sort(key=lambda t: acc(t[1], arm), reverse=True)
    return ranked[0][0], ranked[0][1]


def far_copy(seq: int, ratio: int) -> dict:
    return {
        "task": "far_copy",
        "seq_len": seq,
        "min_gap": 32,
        "span_len": 32,
        "ratio": ratio,
        "dec_segment": 32,
        "hops": 2,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "continue_meta.json").write_text(
        json.dumps(
            {
                "why": (
                    "Width-matched D is 2.27M vs A's 5.11M. Goal requires same "
                    "parameter regime. D9=4.97M. Then push r / hops / seq."
                ),
                "A_params": A_PARAMS,
                "D9_layers": D9_LAYERS,
                "bar": TARGET,
            },
            indent=2,
        )
    )

    # --- 1. Param-matched D on the cell where 4-layer D missed and A hit 97% ---
    # No 800-step LR probe: 4-layer D at seq512 was still near chance at 800 steps
    # under 3e-4. Start at A's seq512 LR and only retune if still at chance.
    d9_cell = {**far_copy(512, 8), "dec_layers": D9_LAYERS}
    d9 = train(
        "cell_seq512_r8_D9",
        "D",
        3e-4,
        steps=8000,
        target=TARGET,
        patience=4000,
        **d9_cell,
    )
    print("D9", d9["summary"]["D"], flush=True)
    if acc(d9, "D") < 0.40 and stop(d9, "D") == "floor_patience":
        d9 = train(
            "cell_seq512_r8_D9_lr1e4",
            "D",
            1e-4,
            steps=8000,
            target=TARGET,
            patience=4000,
            **d9_cell,
        )
        print("D9 lr=1e-4", d9["summary"]["D"], flush=True)

    # --- 2. Compression interpolation: r=16 at frozen 5.11M (r=8 hit, r=32 miss) ---
    r16_cell = far_copy(256, 16)
    a_r16 = train(
        "cell_seq256_r16_A",
        "A",
        1e-3,
        steps=8000,
        target=TARGET,
        patience=4000,
        **r16_cell,
    )
    print("r16 A", a_r16["summary"]["A"], flush=True)
    if acc(a_r16, "A") < 0.30:
        lr16, _ = search_lr("cell_seq256_r16", "A", (3e-4, 1e-4), **r16_cell)
        a_r16 = train(
            "cell_seq256_r16_A_lrretune",
            "A",
            lr16,
            steps=8000,
            target=TARGET,
            patience=4000,
            **r16_cell,
        )
        print("r16 A retune", a_r16["summary"]["A"], flush=True)

    # --- 3. Composition: hops=2, 32-token answers (pack the loss), D first ---
    chain = {
        "task": "chain",
        "seq_len": 512,
        "min_gap": 32,
        "span_len": 32,
        "ratio": 8,
        "dec_segment": 32,
        "hops": 2,
        "key_len": 32,
        "value_len": 32,
        "n_distractors": 2,
    }
    lr_ch, d_ch_probe = search_lr("cell_chain_h2", "D", (3e-4, 1e-3, 1e-4), **chain)
    if acc(d_ch_probe, "D") >= TARGET:
        d_ch = d_ch_probe
    else:
        d_ch = train(
            "cell_chain_h2_D",
            "D",
            lr_ch,
            steps=8000,
            target=TARGET,
            patience=4000,
            **chain,
        )
    print("chain h2 D", d_ch["summary"]["D"], flush=True)
    if acc(d_ch, "D") >= TARGET:
        a_ch = train(
            "cell_chain_h2_A",
            "A",
            lr_ch,
            steps=8000,
            target=TARGET,
            patience=4000,
            **chain,
        )
        print("chain h2 A", a_ch["summary"]["A"], flush=True)
        train("cell_chain_h2_C", "C", lr_ch, 1200, 0.0, 0, **chain)
    else:
        print("KILL chain h2: D missed 95% — exam still too hard", flush=True)

    # --- 4. Length: seq=1024 r=8 A, LR search (do not shrink) ---
    s1024 = far_copy(1024, 8)
    lr1024, a_probe = search_lr("cell_seq1024_r8", "A", (1e-4, 3e-4, 1e-3), **s1024)
    if acc(a_probe, "A") >= TARGET:
        print(f"seq1024 A hit 95% on LR probe lr={lr1024:.0e}", flush=True)
    else:
        a1024 = train(
            "cell_seq1024_r8_A",
            "A",
            lr1024,
            steps=8000,
            target=TARGET,
            patience=2500,
            **s1024,
        )
        print("seq1024 A", a1024["summary"]["A"], flush=True)

    print("\nCONTINUE RUNS DONE — plotting\n", flush=True)
    return subprocess.call(["uv", "run", "python", "/workspace/verification/plot_scale_hard.py"])


if __name__ == "__main__":
    sys.exit(main())
