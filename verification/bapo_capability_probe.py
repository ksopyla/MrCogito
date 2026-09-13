#!/usr/bin/env python
"""Train tiny E18 / dense / encoder-decoder models on one BAPO DNA rung.

This is the scale-up of `verification/symbolic_channel_probe.py` onto the E18 architecture
and matched dense transformers. The other agent owns the `perceiver_concept` Arm-A 100%
map at seq=128; this probe answers a different question: *what can E18 do, at matched params,
once each task is proven solvable?*

Protocol
--------
1. Train `dense` first. If held-out accuracy < 75%, the rung is uncalibrated — do not interpret
   E18 or encdec numbers (the task may be too small, too few steps, or a generator bug).
2. Train `e18_local` on retrieval rungs. It must sit near chance / the analytic floor; if it
   does not, the task leaks into the local window.
3. Train `e18` and `encdec` under the same budget.
4. Write a JSON bundle (learning traces + InfoReport) and optional plots.

  # solvability proof + architecture comparison on the tiny core ladder
  uv run python verification/bapo_capability_probe.py --scale tiny --arch dense e18 encdec e18_local

  # one task
  uv run python verification/bapo_capability_probe.py --scale tiny --task far_copy --steps 400 --out /tmp/bapo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.bapo_ladder import (  # noqa: E402
    SCALES,
    SOLVABLE_ACC,
    TINY_PROOF_TASKS,
    config_for,
    rung_card,
)
from data.symbolic_tasks import generate_row  # noqa: E402
from evaluation.bapo_metrics import info_report  # noqa: E402
from evaluation.bapo_models import ARCHES, ArchSpec, arch_cache, build_model, n_params  # noqa: E402


def make_batch(cfg, rng, batch: int, device):
    rows = [generate_row(cfg, rng) for _ in range(batch)]
    ids = torch.from_numpy(np.stack([r.input_ids for r in rows])).long().to(device)
    labels = torch.from_numpy(np.stack([r.labels for r in rows])).long().to(device)
    return ids, labels


@torch.no_grad()
def evaluate(model, batches) -> dict:
    model.eval()
    ce_sum, n, hits = 0.0, 0, 0
    for ids, labels in batches:
        out = model(ids, labels=labels, return_per_token_loss=True)
        if isinstance(out, tuple):
            _lm, per, valid = out
        else:
            raise RuntimeError("model did not return per-token loss")
        ce_sum += float(per[valid].sum())
        n += int(valid.sum())
        packed = model(ids, return_logits=True)
        logits = packed.logits if hasattr(packed, "logits") else packed
        pred = logits[:, :-1].argmax(-1)
        tgt = labels[:, 1:]
        m = tgt != -100
        hits += int((pred[m] == tgt[m]).sum())
    model.train()
    return {"ce_nats": ce_sum / max(n, 1), "acc": hits / max(n, 1), "tokens": n}


def train_one(arch: str, cfg, args, eval_batches, spec: ArchSpec, device) -> dict:
    model = build_model(
        arch,
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=spec,
        seed=args.seed,
    ).to(device)
    params = n_params(model)
    if params > args.max_params:
        raise SystemExit(f"{arch} has {params} params > --max_params {args.max_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = max(1, min(50, args.steps // 10))

    def lr_factor(step: int) -> float:
        return min(1.0, (step + 1) / warmup)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    rng = np.random.default_rng(args.seed + 1000 + sum(ord(c) for c in arch))
    t0, trace = time.time(), []
    best_acc = -1.0
    for step in range(1, args.steps + 1):
        ids, labels = make_batch(cfg, rng, args.batch, device)
        out = model(ids, labels=labels)
        loss = out.loss if hasattr(out, "loss") else out[0].loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, eval_batches)
            trace.append({"step": step, **ev, "sec": time.time() - t0})
            print(
                f"  [{arch}] step {step:5d}  train {float(loss.detach()):.4f}  "
                f"eval CE {ev['ce_nats']:.4f}  acc {ev['acc']:.3f}  "
                f"({(time.time() - t0) / step:.2f} s/step)",
                flush=True,
            )
            best_acc = max(best_acc, ev["acc"])
            if ev["acc"] >= args.early_stop_acc:
                print(f"  [{arch}] early stop at {step} (acc {ev['acc']:.3f})", flush=True)
                break
    cache = arch_cache(arch, spec, cfg.seq_len)
    final = trace[-1]
    report = info_report(
        ce_nats=final["ce_nats"],
        acc=final["acc"],
        n_supervised=final["tokens"],
        cfg=cfg,
        window=spec.local_window,
        nominal_b_tokens=cache["nominal_b_tokens"],
        nominal_a_bytes=cache["nominal_a_bytes"],
    )
    return {
        "arch": arch,
        "params": params,
        "cache": cache,
        "final": final,
        "best_acc": best_acc,
        "trace": trace,
        "info": report.as_dict(),
        "early_stopped": final["acc"] >= args.early_stop_acc,
    }


def run_rung(task: str, args) -> dict:
    scale = SCALES[args.scale]
    cfg = config_for(scale, task)
    spec = ArchSpec(
        name="shared",
        hidden=args.hidden,
        pre_layers=args.pre_layers,
        global_layers=args.global_layers,
        stack_layers=args.stack_layers,
        local_window=scale.local_window,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        head_dim=args.head_dim,
    )
    card = rung_card(scale, task)
    eval_rng = np.random.default_rng(args.seed + 99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_batches = [make_batch(cfg, eval_rng, args.batch, device) for _ in range(max(1, args.eval_rows // args.batch))]
    print(
        f"\n=== {task} @ {scale.name}  seq={cfg.seq_len} gap={cfg.min_gap} "
        f"window={scale.local_window} prize={card['prize_bits']:.2f} bits  "
        f"floor={card['floor_nats']:.4f} nats  chance={card['chance_acc']:.3f}  device={device} ===",
        flush=True,
    )
    results = {}
    for arch in args.arch:
        print(f"--- {arch} ---", flush=True)
        results[arch] = train_one(arch, cfg, args, eval_batches, spec, device)
        print(
            f"  {arch}: {results[arch]['params']/1e6:.3f}M  acc {results[arch]['final']['acc']:.3f}  "
            f"flow {results[arch]['info']['information_flow']:.3f}  "
            f"a_bits {results[arch]['info']['effective_a_bits']:.2f}  "
            f"B/tok {results[arch]['info']['bytes_per_input_token']:.4g}",
            flush=True,
        )
    calibrated = True
    notes = []
    if "dense" in results:
        dacc = results["dense"]["final"]["acc"]
        ok = dacc >= SOLVABLE_ACC
        notes.append(("dense >= 75% (task is solvable here)", ok, f"acc={dacc:.3f}"))
        calibrated = calibrated and ok
    if "e18_local" in results and task not in {"count", "majority"}:
        lacc = results["e18_local"]["final"]["acc"]
        # Local arm should not substantially beat chance on retrieval rungs.
        leak = lacc > card["chance_acc"] + 0.15
        notes.append(("e18_local near chance (task does not leak)", not leak, f"acc={lacc:.3f}"))
        if leak:
            calibrated = False
    print()
    for name, ok, detail in notes:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
    return {
        "task": task,
        "scale": scale.name,
        "card": card,
        "calibrated": calibrated,
        "results": results,
        "hidden": args.hidden,
        "steps": args.steps,
        "batch": args.batch,
        "seed": args.seed,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", default="tiny", choices=list(SCALES))
    p.add_argument("--task", nargs="+", default=list(TINY_PROOF_TASKS))
    p.add_argument("--arch", nargs="+", default=["dense", "e18", "encdec"], choices=list(ARCHES))
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--pre_layers", type=int, default=1)
    p.add_argument("--global_layers", type=int, default=1)
    p.add_argument("--stack_layers", type=int, default=2)
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--dec_layers", type=int, default=2)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_rows", type=int, default=64)
    p.add_argument("--early_stop_acc", type=float, default=0.99)
    p.add_argument("--max_params", type=int, default=100_000_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="directory for JSON bundles (one file per task)")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    bundles = []
    for task in args.task:
        bundle = run_rung(task, args)
        bundles.append(bundle)
        if out_dir:
            path = out_dir / f"{args.scale}_{task}.json"
            path.write_text(json.dumps(bundle, indent=2))
            print(f"wrote {path}")

    if out_dir:
        summary = {
            "scale": args.scale,
            "tasks": args.task,
            "arches": args.arch,
            "solvable_acc": SOLVABLE_ACC,
            "rungs": [
                {
                    "task": b["task"],
                    "calibrated": b["calibrated"],
                    "acc": {a: r["final"]["acc"] for a, r in b["results"].items()},
                    "information_flow": {a: r["info"]["information_flow"] for a, r in b["results"].items()},
                    "recovered_bits": {a: r["info"]["recovered_bits"] for a, r in b["results"].items()},
                    "bytes_per_input_token": {a: r["info"]["bytes_per_input_token"] for a, r in b["results"].items()},
                    "params": {a: r["params"] for a, r in b["results"].items()},
                }
                for b in bundles
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_dir / 'summary.json'}")
    return 0 if all(b["calibrated"] for b in bundles) else 2


if __name__ == "__main__":
    raise SystemExit(main())
