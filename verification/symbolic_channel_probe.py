#!/usr/bin/env python
"""Does the concept array carry information *at all*? A CPU-sized falsification probe.

Trains two tiny `perceiver_concept` models on one symbolic task from `data/symbolic_tasks.py`:

  arm A  concept_mode=full, concept_xattn_scope=exclusive  — the array is the ONLY route to the
         evidence, because the decoder's raw window cannot reach it and same-segment slots are
         masked out (E23's scope).
  arm C  concept_mode=none                                 — segment-confined decoder, no array.
         It is the *proof* that the task is unreachable locally: it must sit at the analytic
         floor, and if it does not, the task leaks and the generator is wrong.
  arm D  concept_mode=none, dec_segment=seq_len             — full raw access, the calibration
         control. It must drive the loss well below the floor; if it cannot, the setup is simply
         undertrained and NOTHING can be concluded about arm A. Always run it.

Reported against `floor_nats`, which is exact here, so "the channel carried information" is a
measurement rather than a comparison against a possibly-weak control. Arm A is also scored with
`concept_override("none")` — same weights, array removed — so any gain is attributable.

Read the three arms together:

  D below floor, C at floor, A below floor  -> the channel carries information.
  D below floor, C at floor, A at floor     -> the task is learnable and locally unreachable, so
                                               the *channel* is what failed. A real result.
  D at floor                                -> inconclusive: too small, too few steps, or a bug.

  uv run python verification/symbolic_channel_probe.py --task recall --steps 1500
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.symbolic_tasks import (  # noqa: E402
    SymbolicTaskConfig,
    chance_accuracy,
    floor_nats,
    generate_row,
)
from nn.perceiver_concept_lm import PerceiverConceptConfig, PerceiverConceptLM  # noqa: E402

ARMS = ("A", "C", "D")


def make_batch(cfg: SymbolicTaskConfig, rng: np.random.Generator, batch: int):
    rows = [generate_row(cfg, rng) for _ in range(batch)]
    ids = torch.from_numpy(np.stack([r.input_ids for r in rows])).long()
    labels = torch.from_numpy(np.stack([r.labels for r in rows])).long()
    return ids, labels


def build_model(arm: str, cfg: SymbolicTaskConfig, args) -> PerceiverConceptLM:
    # Arm D is the same decoder with its raw window opened to the whole row: a plain causal
    # transformer that can see the evidence directly. It calibrates the instrument.
    dec_segment = cfg.seq_len if arm == "D" else args.dec_segment
    conf = PerceiverConceptConfig(
        vocab_size=cfg.vocab.vocab_size,
        hidden_size=args.hidden,
        intermediate_size=2 * args.hidden,
        token_embedding_dim=32,
        enc_layers=args.enc_layers,
        enc_window=args.enc_window,
        concept_ratio=args.ratio,
        concept_slots=1,
        latent_layers=args.latent_layers,
        dec_layers=args.dec_layers,
        dec_segment=dec_segment,
        dec_local="block",
        concept_mode="full" if arm == "A" else "none",
        concept_xattn_scope=args.scope,
        num_attention_heads=args.hidden // 32,
        num_kv_heads=1,
        xattn_kv_heads=1,
        head_dim=32,
        ngram_orders=(2,),
        ngram_buckets=512,
        enc_value_embed_layers=(0,),
        dec_value_embed_layers=(0,),
        value_embed_dim=16,
        z_loss=1e-4,
        chunked_ce_block_size=256,
        use_liger=False,
        attn_backend="sdpa",
        attn_pad_multiple=min(args.dec_segment, dec_segment),
        xattn_wo_init_std=args.xattn_wo_init,
        pooler_wo_init_std=args.pooler_wo_init,
        pad_token_id=cfg.vocab.control("eos"),
        bos_token_id=cfg.vocab.control("bos"),
        eos_token_id=cfg.vocab.control("eos"),
    )
    torch.manual_seed(args.seed)
    return PerceiverConceptLM(conf)


@torch.no_grad()
def evaluate(model, batches, override: str | None = None) -> dict:
    model.eval()
    ce_sum, n, hits = 0.0, 0, 0
    ctx = model.concept_override(override) if override else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for ids, labels in batches:
            out, per, valid = model(ids, labels=labels, return_per_token_loss=True)
            ce_sum += float(per[valid].sum())
            n += int(valid.sum())
            logits = model(ids, return_logits=True).logits
            pred = logits[:, :-1].argmax(-1)
            tgt = labels[:, 1:]
            m = tgt != -100
            hits += int((pred[m] == tgt[m]).sum())
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    model.train()
    return {"ce_nats": ce_sum / max(n, 1), "acc": hits / max(n, 1), "tokens": n}


def train_arm(arm: str, cfg: SymbolicTaskConfig, args, eval_batches) -> dict:
    model = build_model(arm, cfg, args)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1, anneal_strategy="cos"
    )
    rng = np.random.default_rng(args.seed + 1000)
    t0, trace = time.time(), []
    for step in range(1, args.steps + 1):
        ids, labels = make_batch(cfg, rng, args.batch)
        out = model(ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, eval_batches)
            trace.append({"step": step, **ev})
            print(
                f"  [{arm}] step {step:5d}  train {float(out.loss.detach()):.4f}  "
                f"eval CE {ev['ce_nats']:.4f}  acc {ev['acc']:.3f}  "
                f"({(time.time() - t0) / step:.2f} s/step)",
                flush=True,
            )
    result = {"arm": arm, "params": n_params, "final": trace[-1], "trace": trace}
    if arm == "A":
        result["ablate_none"] = evaluate(model, eval_batches, override="none")
        result["ablate_far"] = evaluate(model, eval_batches, override="far")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="recall")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--n_symbols", type=int, default=8)
    p.add_argument("--min_gap", type=int, default=64)
    p.add_argument("--key_len", type=int, default=2)
    p.add_argument("--value_len", type=int, default=2)
    p.add_argument("--span_len", type=int, default=8)
    p.add_argument("--n_distractors", type=int, default=3)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--count_mod", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--enc_window", type=int, default=32)
    p.add_argument("--latent_layers", type=int, default=2)
    p.add_argument("--dec_layers", type=int, default=2)
    p.add_argument("--dec_segment", type=int, default=64)
    p.add_argument("--ratio", type=int, default=16)
    p.add_argument("--scope", default="exclusive", choices=["causal", "exclusive"])
    # The concept path has two zero-init residual gates in series between the evidence and the
    # loss — `pooler.wo` (the only order-sensitive part of the write; the rest is a mean over the
    # block) and `xattn.wo` (the read's output). The gradient into the read's query/key
    # projections and into `pooler.wo` is proportional to `xattn.wo`, so at zero the channel can
    # only learn to consume the order-free mean of the visible slots. These map straight onto the
    # model's config fields.
    p.add_argument("--xattn_wo_init", type=float, default=0.0,
                   help="config xattn_wo_init_std (0 = the family's zero-init)")
    p.add_argument("--pooler_wo_init", type=float, default=0.0,
                   help="config pooler_wo_init_std (0 = the family's zero-init)")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_rows", type=int, default=128)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the result bundle as JSON")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    cfg = SymbolicTaskConfig(
        task=args.task,
        seq_len=args.seq_len,
        n_symbols=args.n_symbols,
        min_gap=args.min_gap,
        key_len=args.key_len,
        value_len=args.value_len,
        span_len=args.span_len,
        n_distractors=args.n_distractors,
        hops=args.hops,
        count_mod=args.count_mod,
    )
    if args.min_gap < args.dec_segment:
        raise SystemExit(
            f"min_gap ({args.min_gap}) < dec_segment ({args.dec_segment}): the evidence could fall "
            "inside the decoder's raw window and the floor would not hold"
        )
    floor = floor_nats(cfg, args.dec_segment)
    chance = chance_accuracy(cfg)

    eval_rng = np.random.default_rng(args.seed + 99)
    eval_batches = [
        make_batch(cfg, eval_rng, args.batch) for _ in range(max(1, args.eval_rows // args.batch))
    ]

    print(
        f"task={args.task} seq={args.seq_len} alphabet={args.n_symbols} min_gap={args.min_gap} "
        f"dec_segment={args.dec_segment} slots={args.seq_len // args.ratio}\n"
        f"floor = {floor:.4f} nats/supervised token (= ln {args.n_symbols if args.task != 'count' else args.count_mod}), "
        f"chance acc = {chance:.3f}",
        flush=True,
    )

    results = {}
    for arm in args.arms:
        print(f"--- arm {arm} ---", flush=True)
        results[arm] = train_arm(arm, cfg, args, eval_batches)

    print("\n=== verdict ===")
    print(f"floor (no route to the evidence): {floor:.4f} nats, acc {chance:.3f}")
    for arm, r in results.items():
        f = r["final"]
        print(
            f"arm {arm} ({r['params'] / 1e6:.2f}M params): CE {f['ce_nats']:.4f} "
            f"({f['ce_nats'] - floor:+.4f} vs floor)  acc {f['acc']:.3f}"
        )
        if "ablate_none" in r:
            print(
                f"  same weights, array removed: CE {r['ablate_none']['ce_nats']:.4f} "
                f"acc {r['ablate_none']['acc']:.3f}   | far slots only: "
                f"CE {r['ablate_far']['ce_nats']:.4f} acc {r['ablate_far']['acc']:.3f}"
            )
    bundle = {
        "task": args.task,
        "floor_nats": floor,
        "chance_acc": chance,
        "config": vars(args),
        "results": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(bundle, indent=2))
        print(f"\nwrote {args.out}")

    # The instrument is only interpretable if both controls behave: C must not beat the floor
    # (the task does not leak) and D must beat it clearly (the task is learnable at this scale).
    checks = []
    if "C" in results:
        gap = results["C"]["final"]["ce_nats"] - floor
        checks.append(("task does not leak (arm C at the floor)", gap > -0.05, f"{gap:+.4f} nats"))
    if "D" in results:
        gap = results["D"]["final"]["ce_nats"] - floor
        checks.append(("task is learnable here (arm D below the floor)", gap < -0.10, f"{gap:+.4f} nats"))
    print()
    for name, ok, detail in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    calibrated = all(ok for _, ok, _ in checks) and len(checks) == 2
    if "A" in results and calibrated:
        gap = results["A"]["final"]["ce_nats"] - floor
        print(
            f"\n-> the channel {'CARRIES information' if gap < -0.10 else 'FAILED'}: "
            f"arm A is {gap:+.4f} nats vs the floor"
        )
    elif "A" in results:
        print("\n-> INCONCLUSIVE about arm A: run arms C and D and get both checks green first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
