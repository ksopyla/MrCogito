#!/usr/bin/env python
"""Teacher-forced CogitoProbe metrics on a perceiver_ar checkpoint.

Reports token-acc on packed labels, recovered bits vs the row `prize_bits`
column (`max(0, prize_bits + Σ log2 p(gold))`), and `message_override` ablations
(`real` / `none` / `swapped`). Does not launch 32k or NIAH.

Prefer Hub:

  uv run python evaluation/evaluate_cogito_probe.py \
    --checkpoint Cache/Training/<run>/checkpoint-<step> \
    --data ksopyla/cogito-probe-bits --seq_len 1024 --variant fixed --split test \
    --out Cache/Evaluation_reports/e28_bits_1024_fixed.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from torch.nn import functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_preprocess import filter_cogito_probe, resolve_cogito_probe_id  # noqa: E402
from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval  # noqa: E402

LN2 = math.log(2)


def _collate(rows, pad_id: int):
    n = max(len(r["input_ids"]) for r in rows)
    ids, labels, prize = [], [], []
    for r in rows:
        x = list(r["input_ids"])
        y = list(r["labels"])
        pad = n - len(x)
        ids.append(x + [pad_id] * pad)
        labels.append(y + [-100] * pad)
        prize.append(float(r.get("prize_bits") or 0.0))
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(prize, dtype=torch.float32),
        [str(r.get("task") or "") for r in rows],
        [str(r.get("variant") or "") for r in rows],
    )


@torch.no_grad()
def _score(model, data, *, batch_size: int, device, pad_id: int) -> dict:
    n_tok = 0
    n_ok = 0
    rec_sum = 0.0
    prize_sum = 0.0
    n_rows = 0
    by_task: dict[str, dict] = defaultdict(lambda: {"n": 0, "acc_ok": 0, "acc_n": 0, "rec": 0.0, "prize": 0.0})
    for start in range(0, len(data), batch_size):
        chunk = [data[i] for i in range(start, min(start + batch_size, len(data)))]
        ids, labels, prize, tasks, _variants = _collate(chunk, pad_id)
        ids, labels = ids.to(device), labels.to(device)
        out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        gold = labels[:, 1:]
        valid = gold != -100
        gather = gold.clamp(min=0)
        tok_logp = logp.gather(-1, gather.unsqueeze(-1)).squeeze(-1)
        tok_logp = tok_logp.masked_fill(~valid, 0.0)
        pred = logits[:, :-1].argmax(-1)
        n_ok += int(((pred == gold) & valid).sum().item())
        n_tok += int(valid.sum().item())
        row_log2 = (tok_logp.sum(-1) / LN2).cpu()
        rec = torch.clamp(prize + row_log2, min=0.0)
        rec_sum += float(rec.sum().item())
        prize_sum += float(prize.sum().item())
        n_rows += len(chunk)
        for i, task in enumerate(tasks):
            slot = by_task[task or "all"]
            slot["n"] += 1
            slot["rec"] += float(rec[i])
            slot["prize"] += float(prize[i])
            row_valid = int(valid[i].sum().item())
            slot["acc_n"] += row_valid
            slot["acc_ok"] += int(((pred[i] == gold[i]) & valid[i]).sum().item())
    acc = (n_ok / n_tok) if n_tok else 0.0
    tasks_out = {}
    for task, slot in by_task.items():
        tasks_out[task] = {
            "n": slot["n"],
            "acc": (slot["acc_ok"] / slot["acc_n"]) if slot["acc_n"] else 0.0,
            "recovered_bits": slot["rec"] / max(slot["n"], 1),
            "prize_bits": slot["prize"] / max(slot["n"], 1),
        }
    return {
        "n_rows": n_rows,
        "n_supervised": n_tok,
        "acc": acc,
        "recovered_bits": rec_sum / max(n_rows, 1),
        "prize_bits": prize_sum / max(n_rows, 1),
        "information_flow": (rec_sum / prize_sum) if prize_sum else 0.0,
        "by_task": tasks_out,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--data",
        required=True,
        help="Hub id (ksopyla/cogito-probe-bits) or a local save_to_disk dir.",
    )
    p.add_argument("--seq_len", type=int, default=None, help="Hub filter: packed length (1024 or 4096).")
    p.add_argument("--variant", default=None, help="Hub filter: fixed | scaled.")
    p.add_argument("--split", default="test", help="Hub split when --data is a Hub id.")
    p.add_argument("--task", default=None, help="Optional Hub task filter (hop_friend_place, subexpr, …).")
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--message_overrides",
        nargs="+",
        default=["real", "none", "swapped"],
        help="E21 channel ablations. Dense checkpoints ignore none/swapped.",
    )
    args = p.parse_args()
    model = load_perceiver_ar_for_eval(args.checkpoint, device=args.device)
    pad_id = int(getattr(model.config, "pad_token_id", 0) or 0)
    src = Path(args.data)
    if src.exists():
        data = load_from_disk(str(src))
        if args.seq_len is not None or args.variant or args.task:
            data = filter_cogito_probe(
                data,
                seq_len=args.seq_len or int(data[0]["seq_len"]),
                variant=args.variant,
                task=args.task,
            )
    else:
        hub_id = resolve_cogito_probe_id(args.data)
        seq = int(args.seq_len or 1024)
        bundle = load_dataset(hub_id)
        split = args.split if args.split in bundle else "test"
        data = filter_cogito_probe(
            bundle[split], seq_len=seq, variant=args.variant or "fixed", task=args.task,
        )
    report = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "n": len(data),
        "overrides": {},
    }
    for mode in args.message_overrides:
        cm = model.message_override(mode) if hasattr(model, "message_override") else torch.no_grad()
        with torch.no_grad(), cm:
            report["overrides"][mode] = _score(
                model, data, batch_size=args.batch_size, device=args.device, pad_id=pad_id
            )
        print(mode, report["overrides"][mode]["acc"], report["overrides"][mode]["recovered_bits"], flush=True)
    real = report["overrides"].get("real") or {}
    none = report["overrides"].get("none") or {}
    swapped = report["overrides"].get("swapped") or {}
    report["s2_load_bearing"] = {
        "real_minus_none_bits": (real.get("recovered_bits") or 0) - (none.get("recovered_bits") or 0),
        "swapped_le_none": (swapped.get("recovered_bits") or 0) <= (none.get("recovered_bits") or 0) + 1e-6,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
