#!/usr/bin/env python
"""Long-context length ladder: evaluate trained probe weights at longer inputs.

The same exam (same task, answer packing and prize), only the haystack grows. A model
trained at 2,048 tokens by `bapo_capability_probe.py --save_ckpt DIR` is loaded and
evaluated, with no further training, at each `--lengths` value (e.g. 2k … 128k).
Per length it reports answer accuracy (± row SE), accuracy by fact depth (where the fact
sits in the book), the number of memory slots the reader sees, peak GPU memory and
seconds per row, so the cost curve shows whether the architecture is linear in length.

    uv run python verification/length_ladder.py --ckpt Cache/length_ladder/lookup2k_s1 \
        --lengths 2048 4096 8192 16384 32768 65536 131072 --rows 64 \
        --out Cache/length_ladder/lookup2k_s1/ladder.json

Stage B (curriculum): continue training at a longer length with
`bapo_capability_probe.py --init_ckpt DIR --seq_len 8192 ... --save_ckpt DIR2`, then run
this script on DIR2.

Long rows need the `flex` backend (the `sdpa` path builds a dense S x (S + C) mask:
~32 GB at 128k). `--backend auto` uses flex on CUDA above 8k tokens.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.bapo_ladder import SCALES, config_for, generate_row_for  # noqa: E402
from evaluation.bapo_models import E31_ARCHES, SWP_ARCHES, ArchSpec, build_model  # noqa: E402


def _amp_ctx(device: torch.device, amp: str):
    if device.type == "cuda" and amp in ("auto", "bf16"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


def memory_slots(model, arch: str, seq_len: int) -> int | None:
    """Reader-visible memory entries at this length (None for dense / local)."""
    cfg = getattr(model, "config", None)
    if arch in E31_ARCHES:
        from nn.latent_memory import lm_geometry

        return int(lm_geometry(cfg, seq_len).n_slots)
    if arch in SWP_ARCHES:
        from nn.perceiver_ar_lm import swp_geometry

        return int(swp_geometry(cfg, seq_len).n_slots)
    return None


def load(ckpt_path: Path, *, seq_len: int, backend: str, device: torch.device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    spec_d = dict(state["spec"])
    spec_d["attn_backend"] = backend
    for k in ("value_embed_layers", "ngram_orders", "message_anchor_token_ids"):
        if k in spec_d and isinstance(spec_d[k], list):
            spec_d[k] = tuple(spec_d[k])
    spec = ArchSpec(**spec_d)
    data = state["data"]
    model = build_model(
        state["arch"], vocab_size=int(data["vocab_size"]), seq_len=seq_len,
        answer_start=int(data["answer_start"]), pad_id=int(data["pad_id"]), bos_id=int(data["bos_id"]),
        eos_id=int(data["eos_id"]), spec=spec, seed=int(state.get("seed", 0)),
    )
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), state


def data_config(state: dict, seq_len: int):
    d = state["data"]
    over = dict(d["over"])
    over["seq_len"] = int(seq_len)
    return config_for(SCALES[d["scale"]], d["task"], **over)


@torch.no_grad()
def eval_length(model, cfg, *, rows: int, batch: int, seed: int, device, amp: str, depth_bins: int) -> dict:
    rng = np.random.default_rng(seed)
    row_acc: list[float] = []
    depths: list[float] = []
    ce_sum, ce_n = 0.0, 0
    t_rows = 0.0
    done = 0
    while done < rows:
        b = min(batch, rows - done)
        rr = [generate_row_for(cfg, rng) for _ in range(b)]
        ids = torch.from_numpy(np.stack([r.input_ids for r in rr])).long().to(device)
        labels = torch.from_numpy(np.stack([r.labels for r in rr])).long().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with _amp_ctx(device, amp):
            out = model(ids, return_logits=True)
            logits = out.logits if hasattr(out, "logits") else out
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_rows += time.time() - t0
        lg = logits[:, :-1].float()
        tgt = labels[:, 1:]
        m = tgt != -100
        pred = lg.argmax(-1)
        lp = torch.log_softmax(lg[m], dim=-1)
        ce_sum += float(-lp.gather(-1, tgt[m][:, None]).sum())
        ce_n += int(m.sum())
        correct = (pred == tgt) & m
        for i, r in enumerate(rr):
            k = int(m[i].sum())
            row_acc.append(float(correct[i].sum()) / max(k, 1))
            evidence_end = r.answer_start - r.gap
            depths.append(evidence_end / float(len(r.input_ids)))
        done += b
    acc = float(np.mean(row_acc))
    se = float(np.std(row_acc, ddof=1) / math.sqrt(len(row_acc))) if len(row_acc) > 1 else float("nan")
    edges = np.linspace(0.0, 1.0, depth_bins + 1)
    by_depth = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [a for a, dd in zip(row_acc, depths) if lo <= dd < hi or (hi == 1.0 and dd == 1.0)]
        by_depth.append({"lo": float(lo), "hi": float(hi), "n": len(sel),
                         "acc": float(np.mean(sel)) if sel else None})
    return {
        "acc": acc, "acc_se": se, "ce_nats": ce_sum / max(ce_n, 1), "rows": len(row_acc),
        "sec_per_row": t_rows / max(len(row_acc), 1), "by_depth": by_depth,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="dir with <arch>.pt from bapo_capability_probe --save_ckpt")
    p.add_argument("--arch", nargs="*", default=None, help="subset of arches (default: every <arch>.pt)")
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--rows", type=int, default=64, help="eval rows per length")
    p.add_argument("--tokens_per_batch", type=int, default=65536, help="batch = max(1, this // length)")
    p.add_argument("--max_len", default="", help="per-arch length cap, e.g. 'dense=32768' (quadratic cost)")
    p.add_argument("--backend", default="auto", choices=("auto", "sdpa", "flex"))
    p.add_argument("--amp", default="auto")
    p.add_argument("--depth_bins", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=None, help="JSON path (default <ckpt>/ladder.json)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt)
    paths = sorted(ckpt_dir.glob("*.pt"))
    if args.arch:
        paths = [q for q in paths if q.stem in args.arch]
    if not paths:
        raise SystemExit(f"no <arch>.pt in {ckpt_dir}")
    caps = {}
    for tok in filter(None, args.max_len.split(",")):
        a, v = tok.split("=")
        caps[a.strip()] = int(v)
    out_path = Path(args.out) if args.out else ckpt_dir / "ladder.json"
    report = {"ckpt": str(ckpt_dir), "lengths": args.lengths, "rows": args.rows, "results": {}}
    if out_path.exists():
        report = json.loads(out_path.read_text())  # resume: keep finished (arch, length) cells
    for path in paths:
        arch = path.stem
        res = report["results"].setdefault(arch, {})
        for L in args.lengths:
            if str(L) in res and "acc" in res[str(L)]:
                continue
            if arch in caps and L > caps[arch]:
                res[str(L)] = {"skipped": f"cap {caps[arch]}"}
                continue
            backend = args.backend
            if backend == "auto":
                backend = "flex" if (device.type == "cuda" and L > 8192) else "sdpa"
            model, state = load(path, seq_len=L, backend=backend, device=device)
            cfg = data_config(state, L)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            batch = max(1, args.tokens_per_batch // L)
            try:
                ev = eval_length(model, cfg, rows=args.rows, batch=batch, seed=args.seed + L,
                                 device=device, amp=args.amp, depth_bins=args.depth_bins)
            except torch.cuda.OutOfMemoryError as e:  # report, keep going
                res[str(L)] = {"error": f"OOM: {str(e)[:200]}", "backend": backend, "batch": batch}
                print(f"  [{arch}] {L:>7}  OOM (batch {batch}, {backend})", flush=True)
                del model
                torch.cuda.empty_cache()
                out_path.write_text(json.dumps(report, indent=2))
                continue
            ev["backend"] = backend
            ev["batch"] = batch
            ev["memory_slots"] = memory_slots(model, arch, L)
            ev["trained_seq_len"] = int(state["data"]["seq_len"])
            if device.type == "cuda":
                ev["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            res[str(L)] = ev
            depth = "  ".join(
                f"{b['lo']:.1f}-{b['hi']:.1f}:{b['acc']:.2f}" if b["acc"] is not None else f"{b['lo']:.1f}-{b['hi']:.1f}:-"
                for b in ev["by_depth"]
            )
            print(
                f"  [{arch}] {L:>7}  acc {ev['acc']:.3f} ±{ev['acc_se']:.3f}  CE {ev['ce_nats']:.3f}  "
                f"C={ev['memory_slots']}  {ev['sec_per_row']:.3f} s/row  "
                f"peak {ev.get('peak_gb', 0):.1f} GB  [{backend}, batch {batch}]  depth {depth}",
                flush=True,
            )
            out_path.write_text(json.dumps(report, indent=2))
            del model
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
