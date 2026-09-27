#!/usr/bin/env python
"""Inference cost of the memory architectures vs book length (untrained weights; cost only).

For each arch and length: one forward pass at batch 1 (bf16, flex above 8k), median of
`--repeats` timed passes after one warm-up. Reports seconds per row, peak GPU memory, the
number of reader-visible memory entries, the writer's share of the forward time (CUDA events
around the writer module), and parameter counts (total / writer).

    uv run python verification/memory_cost_bench.py --arch e18_local e30_li e31_li e31_li_m1 dense \\
        --lengths 8192 32768 131072 262144 524288 1048576 --max_len dense=131072 --out bench.json

Answers "why is E30 cheaper, and is it only configuration?" in the E30 vs E31 limits study
(docs/experiments_specs/ahead/E31b_e30_vs_e31_limits.md). Accuracy is not measured here.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.bapo_ladder import SCALES, config_for, generate_row_for  # noqa: E402
from evaluation.bapo_models import ArchSpec, build_model, n_params  # noqa: E402
from verification.length_ladder import memory_slots  # noqa: E402

PLATFORM = dict(hidden=960, head_dim=64, n_kv_heads=1, pre_layers=1, global_layers=1, stack_layers=2,
                token_embedding_dim=128, ngram_orders=(), swp_n_heads=8, swp_query_dim=128,
                message_raw_window=256, zero_init_residuals=False)


def writer_modules(model) -> list[torch.nn.Module]:
    from nn.latent_memory import LatentMemoryWriter
    from nn.perceiver_ar_lm import SlidingWindowPerceiverCompressor

    return [m for m in model.modules() if isinstance(m, (LatentMemoryWriter, SlidingWindowPerceiverCompressor))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", nargs="+", default=["e18_local", "e30_li", "e31_li", "e31_li_m1", "dense"])
    p.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 131072, 262144, 524288, 1048576])
    p.add_argument("--max_len", default="dense=131072", help="per-arch cap, e.g. 'dense=131072'")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--spec", default="", help="extra ArchSpec fields, e.g. 'lm_latents=8,lm_reader_tokens=1'")
    p.add_argument("--out", default="Cache/study/e30_vs_e31/cost_bench.json")
    args = p.parse_args()

    device = torch.device("cuda")
    caps = {a: int(v) for a, v in (t.split("=") for t in filter(None, args.max_len.split(",")))}
    extra = {}
    for tok in filter(None, args.spec.split(",")):
        k, v = tok.split("=")
        extra[k] = type(getattr(ArchSpec(name="x"), k))(v)
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {"platform": PLATFORM, "results": {}}
    for arch in args.arch:
        res = report["results"].setdefault(arch + (f"[{args.spec}]" if args.spec else ""), {})
        for L in args.lengths:
            if str(L) in res and ("sec_per_row" in res[str(L)] or "error" in res[str(L)]):
                continue
            if arch in caps and L > caps[arch]:
                continue
            cfg = config_for(SCALES["bridge_1k"], "recall", seq_len=L)
            row = generate_row_for(cfg, np.random.default_rng(L))
            v = cfg.vocab
            spec = ArchSpec(name=arch, attn_backend="flex" if L > 8192 else "sdpa",
                            message_boundary_token_id=int(v.control("query")), **PLATFORM, **extra)
            torch.manual_seed(0)
            model = build_model(arch, vocab_size=v.vocab_size, seq_len=L, answer_start=int(row.answer_start),
                                pad_id=int(v.control("eos")), bos_id=int(v.control("bos")),
                                eos_id=int(v.control("eos")), spec=spec, seed=0).to(device).eval()
            wm = writer_modules(model)
            events: list[tuple] = []
            hooks = []
            for m in wm:
                hooks.append(m.register_forward_pre_hook(
                    lambda mod, inp: events.append([torch.cuda.Event(enable_timing=True)]) or events[-1][0].record()))
                hooks.append(m.register_forward_hook(
                    lambda mod, inp, o: events[-1].append(torch.cuda.Event(enable_timing=True)) or events[-1][1].record()))
            ids = torch.from_numpy(row.input_ids[None]).long().to(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times, wshare = [], []
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for rep in range(args.repeats + 1):
                        events.clear()
                        torch.cuda.synchronize()
                        t0 = time.time()
                        model(ids, return_logits=True)
                        torch.cuda.synchronize()
                        dt = time.time() - t0
                        if rep:
                            times.append(dt)
                            w = sum(a.elapsed_time(b) for a, b in events) / 1000.0
                            wshare.append(w / dt)
                r = {
                    "sec_per_row": statistics.median(times),
                    "writer_share": statistics.median(wshare) if wm else 0.0,
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "memory_entries": memory_slots(model, arch, L),
                    "params_m": n_params(model) / 1e6,
                    "writer_params_m": sum(n_params(m) for m in wm) / 1e6,
                }
            except torch.cuda.OutOfMemoryError as e:
                r = {"error": f"OOM: {str(e)[:160]}"}
            for h in hooks:
                h.remove()
            del model
            torch.cuda.empty_cache()
            res[str(L)] = r
            print(f"  [{arch}] {L:>8}  " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                                     for k, v in r.items()), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
