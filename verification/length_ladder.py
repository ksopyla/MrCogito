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

Exams with several same-shaped candidates (keyed lookups, parallel chains) are also scored on the picked
candidate (`--candidate auto`): the answer is decoded greedily and the nearest planted candidate must be the asked
one, the capability checks' score for those tasks (`answer_exact` in the probe); the first letter has a ~40 %
guessing floor there.

Floor stop (`--floor_stop auto`, the default; evaluation only, `capability_checks.md` rule 3): once two lengths in
a row, both at or past the training length, score at the floor (the task's own score <= its guessing floor + 5
points), the longer lengths are not read; they are recorded as `{"skipped": "floor stop"}` and count as the floor
on the board. The task comes from the checkpoint folder's job name (`task_of_job`); `--floor_stop off` reads them.

Decode cost cap (`--decode_cap auto`, the default: 131,072 tokens; evaluation only, rule 3): at and past the cap every
row is still read teacher-forced (first letter, every letter, CE), but the greedy decode behind the picked candidate
(one forward per answer letter) is skipped; the rung is recorded with `"candidate_mode": "cost cap"` and
`"decode_cap": <cap>`, and the board counts its picked score as the floor. `--decode_cap off` re-reads exactly
those rungs.

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


def data_config(state: dict, seq_len: int, recipe: str | None = None):
    """The checkpoint's training exam at `seq_len`, or another recipe (E33a: lookup no-harm ladder of a
    chain-trained checkpoint)."""
    d = state["data"]
    if recipe:
        from data.bapo_ladder import resolve_recipe

        rec = resolve_recipe(recipe)
        return config_for(SCALES[d["scale"]], rec.task, **{**rec.overrides, "seq_len": int(seq_len)})
    over = dict(d["over"])
    over["seq_len"] = int(seq_len)
    return config_for(SCALES[d["scale"]], d["task"], **over)


def floor_stop(res: dict, lengths, L: int, *, train_len: int, floor: float, key: str,
               margin: float | None = None) -> list[int] | None:
    """The two rungs that stop the ladder before reading `L`, or None. Only rungs at or past the training length
    count; a rung the stop already left unread counts as at the floor, so the stop holds for every longer length."""
    from evaluation.capability_tasks import FLOOR_STOP, LADDER_FLOOR_MARGIN

    margin = LADDER_FLOOR_MARGIN if margin is None else margin
    prev = [x for x in sorted(set(int(v) for v in lengths)) if train_len <= x < L]
    if len(prev) < 2:
        return None

    def at_floor(x: int) -> bool:
        v = res.get(str(x)) or {}
        if v.get("skipped") == FLOOR_STOP:
            return True
        y = v.get(key)
        return y is not None and y <= floor + margin + 1e-9

    return prev[-2:] if all(at_floor(x) for x in prev[-2:]) else None


def floor_rule(ckpt_dir: Path, mode: str) -> tuple[str, float, str] | None:
    """(task, floor, score key) of the floor stop for this checkpoint folder, or None (off / unknown task)."""
    if mode == "off":
        return None
    from evaluation.capability_tasks import ladder_floor, task_of_job

    task = task_of_job(ckpt_dir.name)
    if mode != "auto":
        return (task or "?", float(mode), ladder_floor(task)[1] if task else "first_acc")
    if task is None:
        print(f"  floor stop off: {ckpt_dir.name} is no v4 task folder", flush=True)
        return None
    return (task, *ladder_floor(task))


def decode_cap(mode: str) -> int | None:
    """The length from which the greedy decode is skipped (`--decode_cap`), or None (off)."""
    if mode == "off":
        return None
    from evaluation.capability_tasks import LADDER_DECODE_CAP

    return LADDER_DECODE_CAP if mode == "auto" else int(mode)


def has_candidates(cfg, row) -> bool:
    """Several same-shaped answers are planted in the book (the picked-candidate score applies)."""
    from verification.bapo_capability_probe import _candidates

    n = int(cfg.key_len) if getattr(cfg, "chain_answer_path", False) else int((row.labels != -100).sum())
    return n > 0 and len(_candidates(cfg, row.input_ids, n)) > 1


@torch.no_grad()
def eval_length(model, cfg, *, rows: int, batch: int, seed: int, device, amp: str, depth_bins: int,
                candidate: str = "auto", decode: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    cand_on: bool | None = None if candidate == "auto" else candidate == "on"
    cand_sum = cand_n = exact_sum = first_g_sum = 0.0
    dec_rows = 0
    t_dec = 0.0
    row_acc: list[float] = []
    row_first: list[float] = []   # first answer letter: the honest score on multi-candidate exams
    pos_hits: list[list[float]] = []  # per answer position (teacher-forced)
    depths: list[float] = []
    ce_sum, ce_n = 0.0, 0
    t_rows = 0.0
    done = 0
    from verification.heat_pause import heat_pause

    while done < rows:
        heat_pause("ladder")  # Polonez heat rule (MC_HEAT_*, off when unset); timings start after it
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
        if cand_on is None:
            cand_on = has_candidates(cfg, rr[0])
        if cand_on and decode:
            from verification.bapo_capability_probe import _answer_exact

            t1 = time.time()
            # the teacher-forced forward above gives every greedy letter up to the first wrong one
            ae = _answer_exact(model, [(ids, labels)], amp=amp, device=device, cfg=cfg, first_logits=logits)
            model.eval()
            t_dec += time.time() - t1
            exact_sum += ae["exact"] * ae["rows"]
            first_g_sum += ae["first_greedy"] * ae["rows"]
            dec_rows += ae["rows"]
            cand_sum += ae.get("candidate", 0.0) * ae.get("candidate_rows", 0)
            cand_n += ae.get("candidate_rows", 0)
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
            hits = correct[i][m[i]].float().tolist()
            row_first.append(hits[0] if hits else float("nan"))
            for j, h in enumerate(hits):
                if j >= len(pos_hits):
                    pos_hits.append([])
                pos_hits[j].append(h)
            evidence_end = r.answer_start - r.gap
            depths.append(evidence_end / float(len(r.input_ids)))
        done += b
    acc = float(np.mean(row_acc))
    se = float(np.std(row_acc, ddof=1) / math.sqrt(len(row_acc))) if len(row_acc) > 1 else float("nan")
    edges = np.linspace(0.0, 1.0, depth_bins + 1)
    by_depth = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [a for a, dd in zip(row_acc, depths) if lo <= dd < hi or (hi == 1.0 and dd == 1.0)]
        self_first = [f for f, dd in zip(row_first, depths) if lo <= dd < hi or (hi == 1.0 and dd == 1.0)]
        by_depth.append({"lo": float(lo), "hi": float(hi), "n": len(sel),
                         "acc": float(np.mean(sel)) if sel else None,
                         "first_acc": float(np.mean(self_first)) if self_first else None})
    cand = {"candidate_mode": "on" if cand_on else "off"}
    if cand_on and not decode:  # the decode cost cap: read, not decoded (the picked score counts as the floor)
        from evaluation.capability_tasks import DECODE_CAP

        cand = {"candidate_mode": DECODE_CAP}
    if cand_on and dec_rows:
        cand.update(exact=exact_sum / dec_rows, first_greedy=first_g_sum / dec_rows,
                    sec_per_row_decode=t_dec / dec_rows)
        if cand_n:
            cand.update(candidate=cand_sum / cand_n, candidate_rows=int(cand_n))
    return {
        **cand,
        "acc": acc, "acc_se": se, "ce_nats": ce_sum / max(ce_n, 1), "rows": len(row_acc),
        "first_acc": float(np.mean(row_first)),
        "first_acc_se": float(np.std(row_first, ddof=1) / math.sqrt(len(row_first))) if len(row_first) > 1 else float("nan"),
        "per_position_acc": [float(np.mean(h)) for h in pos_hits],
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
    p.add_argument("--need_first", action="store_true",
                   help="re-evaluate cells saved before first-letter accuracy was recorded")
    p.add_argument("--out", default=None, help="JSON path (default <ckpt>/ladder.json)")
    p.add_argument("--candidate", default="auto", choices=("auto", "on", "off"),
                   help="also decode greedily and score the picked candidate (auto: exams with several candidates)")
    p.add_argument("--floor_stop", default="auto",
                   help="auto: stop after two lengths in a row at the task's floor + 5 points (task from the folder name); "
                        "off: read every length; a number: that floor")
    p.add_argument("--decode_cap", default="auto",
                   help="auto: skip the greedy decode from 131,072 tokens on (picked score 'not read, cost cap'); "
                        "off: decode every length (re-reads the capped rungs); a number: that length")
    p.add_argument("--recipe", default=None, help="ladder this recipe instead of the checkpoint's own exam")
    p.add_argument("--loop_rounds", type=int, default=None,
                   help="E33a: evaluate looped checkpoints with this many loops (default: as trained)")
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
    rule = floor_rule(ckpt_dir, args.floor_stop)
    dcap = decode_cap(args.decode_cap)
    from evaluation.capability_tasks import DECODE_CAP, FLOOR_STOP, LADDER_FLOOR_MARGIN, TASK_BY_ID

    for path in paths:
        arch = path.stem
        res = report["results"].setdefault(arch, {})
        train_len = None
        for L in args.lengths:
            capped = L >= dcap if dcap is not None else False
            if (str(L) in res and "acc" in res[str(L)] and ("first_acc" in res[str(L)] or not args.need_first)
                    and ("candidate_mode" in res[str(L)] or args.candidate == "off")
                    and not ("decode_cap" in res[str(L)] and not capped)):  # a capped rung is re-read once uncapped
                continue
            if rule is not None:
                if train_len is None:  # the length it was trained at, never below the task's own
                    st = torch.load(path, map_location="cpu", weights_only=False)
                    train_len = max(int(st["data"]["seq_len"]),
                                    TASK_BY_ID[rule[0]].train_len if rule[0] in TASK_BY_ID else 0)
                    del st
                stop = floor_stop(res, args.lengths, L, train_len=train_len, floor=rule[1], key=rule[2])
                if stop:
                    res[str(L)] = {"skipped": FLOOR_STOP, "floor_stop": {
                        "task": rule[0], "floor": rule[1], "margin": LADDER_FLOOR_MARGIN, "score": rule[2],
                        "after": stop, "train_len": train_len}}
                    print(f"  [{arch}] {L:>7}  not read: floor stop ({' and '.join(str(x) for x in stop)} at or below "
                          f"{100 * (rule[1] + LADDER_FLOOR_MARGIN):.0f} % on {rule[2]})", flush=True)
                    out_path.write_text(json.dumps(report, indent=2))
                    continue
            if arch in caps and L > caps[arch]:
                res[str(L)] = {"skipped": f"cap {caps[arch]}"}
                continue
            backend = args.backend
            if backend == "auto":
                backend = "flex" if (device.type == "cuda" and L > 8192) else "sdpa"
            model, state = load(path, seq_len=L, backend=backend, device=device)
            if args.loop_rounds is not None and getattr(model, "loop_emb", None) is not None:
                model._loop_rounds_override = int(args.loop_rounds)
            cfg = data_config(state, L, args.recipe)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            batch = max(1, args.tokens_per_batch // L)
            try:
                ev = eval_length(model, cfg, rows=args.rows, batch=batch, seed=args.seed + L,
                                 device=device, amp=args.amp, depth_bins=args.depth_bins, candidate=args.candidate,
                                 decode=not capped)
            except torch.cuda.OutOfMemoryError as e:  # report, keep going
                res[str(L)] = {"error": f"OOM: {str(e)[:200]}", "backend": backend, "batch": batch}
                print(f"  [{arch}] {L:>7}  OOM (batch {batch}, {backend})", flush=True)
                del model
                torch.cuda.empty_cache()
                out_path.write_text(json.dumps(report, indent=2))
                continue
            if ev["candidate_mode"] == DECODE_CAP:
                ev["decode_cap"] = dcap
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
                f"  [{arch}] {L:>7}  acc {ev['acc']:.3f} ±{ev['acc_se']:.3f}  first {ev['first_acc']:.3f}  "
                + (f"picked {ev['candidate']:.3f}  " if "candidate" in ev else "")
                + (f"picked not read (cost cap from {dcap})  " if "decode_cap" in ev else "")
                + f"CE {ev['ce_nats']:.3f}  "
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
