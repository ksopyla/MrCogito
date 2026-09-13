#!/usr/bin/env python
"""Exclusive-scope <10M length×difficulty campaign. Configs over the shared probe.

Matching rule (width-and-decoder-depth, not param-matched):
  Arm A  hidden=256, enc=2, latent=2, dec=4, tok_emb=32  -> 5.11M
  Arm C  same hidden / dec_layers, concept_mode=none, dec_segment=32  -> 2.27M
  Arm D  same hidden / dec_layers, concept_mode=none, dec_segment=seq_len  -> 2.27M
C and D have no encoder/pooler/latent, so fewer params. Same rule as the seq128
1.35M vs 0.60M campaign. Kill a cell if Arm D cannot reach 95% — the exam is then
too hard for this budget, not a concept failure.
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


def write_index() -> None:
    rows = []
    for p in sorted(OUT.glob("*.json")):
        if p.name in {"campaign_index.json", "campaign_meta.json", "winner_lr.json"}:
            continue
        try:
            bundle = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        for arm, s in (bundle.get("summary") or {}).items():
            rows.append({"file": p.name, "run_name": bundle.get("run_name"), **s})
    (OUT / "campaign_index.json").write_text(json.dumps({"runs": rows}, indent=2))


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
    bundle = json.loads(path.read_text())
    write_index()
    return bundle


def arm_summary(bundle: dict, arm: str) -> dict:
    return bundle["summary"][arm]


def hit_target(bundle: dict, arm: str, bar: float = TARGET) -> bool:
    return arm_summary(bundle, arm)["acc"] >= bar


def pick_lr(results: list[tuple[float, dict]]) -> float:
    hit = [(lr, b) for lr, b in results if hit_target(b, "A")]
    if hit:
        hit.sort(key=lambda t: arm_summary(t[1], "A")["examples_seen"])
        return hit[0][0]
    results = sorted(results, key=lambda t: arm_summary(t[1], "A")["acc"], reverse=True)
    return results[0][0]


def far_copy_kwargs(seq: int, ratio: int, dec_segment: int = 32) -> dict:
    return {
        "task": "far_copy",
        "seq_len": seq,
        "min_gap": dec_segment,
        "span_len": 32,
        "ratio": ratio,
        "dec_segment": dec_segment,
        "hops": 2,
    }


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "campaign_meta.json").write_text(
        json.dumps(
            {
                "hypothesis": (
                    "The exclusive concept array has a measurable length×difficulty "
                    "frontier below 10M params (95% on supervised tokens)."
                ),
                "reference_arm_A": "hidden=256 enc=2 latent=2 dec=4 tok=32 -> 5.11M",
                "matching_rule": (
                    "same hidden and decoder depth; C/D drop encoder/latent/xattn"
                ),
                "bar": TARGET,
                "device": "cpu",
                "do_not_rerun": "seq128 span32 is the easy end; see /opt/cursor/artifacts/scale/",
            },
            indent=2,
        )
    )

    # --- LR search on the first hard cell: seq 256 far_copy r=8 ---
    lr_results: list[tuple[float, dict]] = []
    for lr in (3e-3, 6e-3, 1e-3):
        tag = f"{lr:.0e}".replace("-", "")
        b = train(
            f"lr{tag}_a_seq256_r8",
            "A",
            lr,
            steps=800,
            target=TARGET,
            patience=800,
            **far_copy_kwargs(256, 8),
        )
        s = arm_summary(b, "A")
        print(
            f"LR {lr:.0e}: acc={s['acc']:.3f} examples={s['examples_seen']} "
            f"stop={s['stop_reason']}",
            flush=True,
        )
        lr_results.append((lr, b))

    winner = pick_lr(lr_results)
    print(f"\n>>> frozen LR = {winner:.0e}\n", flush=True)
    (OUT / "winner_lr.json").write_text(json.dumps({"lr": winner}, indent=2))
    winner_bundle = next(b for lr, b in lr_results if lr == winner)

    def cell_ad(prefix: str, **cell) -> tuple[dict, dict]:
        a = train(f"{prefix}_A", "A", winner, 8000, TARGET, 4000, **cell)
        print(prefix, "A", arm_summary(a, "A"), flush=True)
        d = train(f"{prefix}_D", "D", winner, 8000, TARGET, 4000, **cell)
        print(prefix, "D", arm_summary(d, "D"), flush=True)
        return a, d

    def cell_c(prefix: str, steps: int, **cell) -> dict:
        c = train(f"{prefix}_C", "C", winner, steps, 0.0, 0, **cell)
        print(prefix, "C", arm_summary(c, "C"), flush=True)
        return c

    # Cell 1: seq 256 far_copy r=8 — A, D, C (new geometry).
    # If the LR probe already crossed 95%, reuse it instead of retraining A.
    if hit_target(winner_bundle, "A"):
        a256 = winner_bundle
        dest = OUT / "cell_seq256_r8_A.json"
        src = OUT / f"{winner_bundle.get('run_name', '')}.json"
        if src.exists() and not dest.exists():
            dest.write_text(src.read_text())
        print("cell_seq256_r8 A reused LR winner", arm_summary(a256, "A"), flush=True)
        d256 = train(
            "cell_seq256_r8_D", "D", winner, 8000, TARGET, 4000, **far_copy_kwargs(256, 8)
        )
        print("cell_seq256_r8 D", arm_summary(d256, "D"), flush=True)
    else:
        a256, d256 = cell_ad("cell_seq256_r8", **far_copy_kwargs(256, 8))
    if not hit_target(d256, "D"):
        print("KILL seq256: Arm D missed 95% — exam too hard for this budget", flush=True)
        subprocess.call(["uv", "run", "python", "/workspace/verification/plot_scale_hard.py"])
        return 0
    cell_c("cell_seq256_r8", 1500, **far_copy_kwargs(256, 8))

    # Cell 2: higher r (32 tokens/slot) at the same seq. C already proved this geometry.
    a_r32, d_r32 = cell_ad("cell_seq256_r32", **far_copy_kwargs(256, 32))
    if not hit_target(d_r32, "D"):
        print("KILL r=32: Arm D missed 95%", flush=True)

    # Cell 3: chain hops=3, key_len=8 (pack the loss; span=8 far_copy starved).
    chain = {
        "task": "chain",
        "seq_len": 256,
        "min_gap": 32,
        "span_len": 32,
        "ratio": 8,
        "dec_segment": 32,
        "hops": 3,
        "key_len": 8,
        "value_len": 8,
        "n_distractors": 3,
    }
    a_ch, d_ch = cell_ad("cell_chain_h3", **chain)
    cell_c("cell_chain_h3", 1500, **chain)
    if not hit_target(d_ch, "D"):
        print("KILL chain h3: Arm D missed 95% — exam too hard", flush=True)

    # Cell 4: longer seq (new geometry => C once) if D still solves seq256.
    a512, d512 = cell_ad("cell_seq512_r8", **far_copy_kwargs(512, 8))
    cell_c("cell_seq512_r8", 1200, **far_copy_kwargs(512, 8))
    if not hit_target(d512, "D"):
        print("KILL seq512: Arm D missed 95%", flush=True)

    print("\nCAMPAIGN RUNS DONE — plotting\n", flush=True)
    return subprocess.call(["uv", "run", "python", "/workspace/verification/plot_scale_hard.py"])


if __name__ == "__main__":
    sys.exit(main())
