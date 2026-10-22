#!/usr/bin/env python
"""E36 evaluation — a frozen reader with and without memory (docs/4_Research_Notes/e36_frozen_reader_writer_20261018.md).

Every measure is reported under the conditions that apply:
  none   : the reader sees only the target text (the floor);
  raw    : the relevant raw text in the reader's context (the ceiling);
  raw1   : only the last raw span before the target (how much of the gain is local);
  memory : the writer's entries of the earlier text (needs --checkpoint);
  other  : entries written from ANOTHER document (no-harm: must not be worse than none).

Measures:
  continuation : cross-entropy on span M (tokens 1..S-1) of held-out long-document rows, per source; the share of
                 the raw-text gain that memory recovers, and the bits the memory carries per entry;
  reconstruction (--checkpoint): the reader rebuilds a span from its own entry;
  squad        : SQuAD v1.1 validation, zero-shot completion, contains-match and exact match;
  cramming     : (--cramming) one OPTIMISED entry per text (an input vector, and a key/value prefix entry) for
                 128 / 256 / 512-token texts: the reader's channel ceiling before any writer is trained.

  uv run python evaluation/frozen_reader_eval.py --reader HuggingFaceTB/SmolLM2-360M \
      --manifest $DATASETS_TOK_DIR/e36_longdoc_4k6_v1_manifest.json --cramming --out s0_smollm2.json
  uv run python evaluation/frozen_reader_eval.py --checkpoint <run>/final --manifest ... --out s1.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from nn.frozen_reader_memory import load_frozen_reader_memory  # noqa: E402

LN2 = math.log(2.0)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _ce_per_row(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Mean CE over tgt tokens 1..T-1 per row -> (B,)."""
    lp = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), tgt[:, 1:], reduction="none")
    return lp.mean(1)


# ---------------------------------------------------------------------------------------------------------------
# continuation / reconstruction on held-out rows
# ---------------------------------------------------------------------------------------------------------------
def _eval_rows(manifest: str, per_source: int):
    from datasets import load_from_disk

    m = json.load(open(manifest))
    for src in m["sources"]:
        ds = load_from_disk(src["eval_path"])
        n = min(per_source, len(ds))
        yield src["name"], [ds[i]["input_ids"] for i in range(n)]


@torch.no_grad()
def continuation(reader, fm, rows: list[list[int]], span: int, mem_spans: int, bs: int, dev: str) -> dict:
    s, m = span, mem_spans
    need = (m + 1) * s
    ids = torch.tensor([r[1:1 + need] for r in rows if len(r) >= need + 1], device=dev)
    sums = Counter()
    n = 0
    for i in range(0, len(ids), bs):
        x = ids[i:i + bs]
        b = x.shape[0]
        mem, tgt = x[:, :m * s], x[:, m * s:]
        # ceilings / floor use the plain reader: absolute positions as in the document
        none = reader(input_ids=tgt, use_cache=False).logits
        sums["none"] += _ce_per_row(none, tgt).sum().item()
        raw = reader(input_ids=x, use_cache=False).logits[:, m * s:]
        sums["raw"] += _ce_per_row(raw, tgt).sum().item()
        raw1 = reader(input_ids=x[:, (m - 1) * s:], use_cache=False).logits[:, s:]
        sums["raw1"] += _ce_per_row(raw1, tgt).sum().item()
        if fm is not None:
            span_mid = fm.config.entry_positions == "span_mid"
            tpos = torch.arange(s, device=dev) + (m * s if span_mid else m)
            epos = fm.entry_pos(0, m, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                e = fm.write(mem)
                lm = fm.read_logits(e, tgt, tpos, epos)
                lo = fm.read_logits(e.roll(1, 0) if b > 1 else e.flip(1), tgt, tpos, epos)
                e_tgt = fm.write(tgt)  # reconstruction: the target span's own entry
                rep_pos, rpos = fm.recon_positions(m, dev)
                lr = fm.read_logits(e_tgt, tgt, rpos, rep_pos)
            sums["memory"] += _ce_per_row(lm, tgt).sum().item()
            sums["other"] += _ce_per_row(lo, tgt).sum().item()
            sums["recon"] += _ce_per_row(lr, tgt).sum().item()
        n += b
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["rows"] = n
    gain = out["none"] - out["raw"]
    if "memory" in out:
        out["recovered"] = (out["none"] - out["memory"]) / gain if gain > 0 else float("nan")
        out["bits_per_entry"] = (out["none"] - out["memory"]) * (s - 1) / LN2 / m
        out["recon_bits"] = (out["none"] - out["recon"]) * (s - 1) / LN2
    out["raw_bits"] = gain * (s - 1) / LN2
    return out


# ---------------------------------------------------------------------------------------------------------------
# SQuAD v1.1, zero-shot completion
# ---------------------------------------------------------------------------------------------------------------
def _norm(t: str) -> str:
    t = "".join(c for c in t.lower() if c not in set(string.punctuation))
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return " ".join(t.split())


@torch.no_grad()
def _greedy(step_logits, prompt: torch.Tensor, max_new: int, stop: set[int]) -> list[int]:
    cur, out = prompt, []
    for _ in range(max_new):
        nxt = int(step_logits(cur)[0, -1].argmax())
        if nxt in stop:
            break
        out.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], 1)
    return out


@torch.no_grad()
def squad(reader, fm, tok, n_items: int, dev: str, conditions: list[str]) -> dict:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="validation").shuffle(seed=0).select(range(n_items))
    stop = {tok.eos_token_id, *tok.encode("\n", add_special_tokens=False)}
    hits = {c: Counter() for c in conditions}
    for idx, ex in enumerate(ds):
        golds = [_norm(a) for a in ex["answers"]["text"]]
        q = tok.encode(f"Question: {ex['question']}\nAnswer:", add_special_tokens=False)
        ctx = tok.encode(f"{ex['context'].strip()}\n\n", add_special_tokens=False)
        for c in conditions:
            if c == "none":
                prompt = torch.tensor([q], device=dev)
                fn = lambda cur: reader(input_ids=cur, use_cache=False).logits  # noqa: E731
            elif c == "raw":
                prompt = torch.tensor([ctx + q], device=dev)
                fn = lambda cur: reader(input_ids=cur, use_cache=False).logits  # noqa: E731
            else:  # memory / other: the passage only through the writer's entries
                s = getattr(fm.config, "passage_span", 0) or fm.config.span  # passages: tokens per entry
                src = ctx if c == "memory" else tok.encode(ds[(idx + 1) % n_items]["context"], add_special_tokens=False)
                chunks = [src[i:i + s] for i in range(0, len(src), s)]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                    e = torch.cat([fm.writer(torch.tensor([ch], device=dev)) for ch in chunks]).unsqueeze(0)
                k = e.shape[1]
                span_mid = fm.config.entry_positions == "span_mid"
                idx_k = torch.arange(k, device=dev)
                epos = idx_k * s + s // 2 if span_mid else idx_k  # as data/data_collators.py:DataCollatorForMemoryRows
                base = k * s if span_mid else k
                prompt = torch.tensor([q], device=dev)

                def fn(cur, e=e, epos=epos, base=base):
                    tpos = torch.arange(cur.shape[1], device=dev) + base
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                        return fm.read_logits(e, cur, tpos, epos)
            pred = _norm(tok.decode(_greedy(fn, prompt, 16, stop)))
            hits[c]["exact"] += any(pred == g for g in golds)
            hits[c]["contains"] += any(g and g in pred for g in golds)
    return {c: {k: v / n_items for k, v in h.items()} | {"items": n_items} for c, h in hits.items()}


# ---------------------------------------------------------------------------------------------------------------
# Cramming probe: one optimised entry per text
# ---------------------------------------------------------------------------------------------------------------
def _cram(reader, ids: torch.Tensor, way: str, steps: int, lr: float) -> dict:
    rc = reader.config
    b, n = ids.shape
    dev, dt = ids.device, reader.dtype
    emb = reader.get_input_embeddings()
    with torch.no_grad():
        base_ce = _ce_tail(reader(input_ids=ids, use_cache=False).logits[:, :-1], ids)  # no entry
    if way == "input":
        scale = emb.weight.float().pow(2).mean().sqrt().item()
        p = torch.nn.Parameter(torch.randn(b, 1, rc.hidden_size, device=dev) * scale)
    else:
        hd = getattr(rc, "head_dim", None) or rc.hidden_size // rc.num_attention_heads
        p = torch.nn.Parameter(torch.randn(b, rc.num_hidden_layers, 2, rc.num_key_value_heads, 1, hd, device=dev) * 0.5)
    opt = torch.optim.Adam([p], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def logits_fn():
        if way == "input":
            x = torch.cat([p.to(dt), emb(ids)], 1)
            mask = torch.ones(b, n + 1, dtype=torch.long, device=dev)
            return reader(inputs_embeds=x, attention_mask=mask, use_cache=False).logits[:, 1:-1]  # predicts ids[1:]
        cache = DynamicCache()
        for i in range(rc.num_hidden_layers):
            cache.update(p[:, i, 0].to(dt), p[:, i, 1].to(dt), i)
        mask = torch.ones(b, n + 1, dtype=torch.long, device=dev)
        lg = reader(input_ids=ids, past_key_values=cache, cache_position=torch.arange(1, n + 1, device=dev),
                    position_ids=torch.arange(1, n + 1, device=dev).expand(b, -1), attention_mask=mask,
                    use_cache=True).logits
        return lg[:, :-1]  # predicts ids[1:]

    for _ in range(steps):
        lg = logits_fn()
        loss = _ce_tail(lg, ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        lg = logits_fn()
        ce = _ce_tail(lg, ids).item()
        ok = lg.argmax(-1) == ids[:, 1:]
        acc = ok.float().mean().item()
        prefix = (ok.cumprod(1).sum(1)).float().mean().item()
        full = ok.all(1).float().mean().item()
    scored = n - 1
    return {"ce": ce, "ce_no_entry": base_ce.item(), "token_acc": acc, "exact_prefix_tokens": prefix,
            "all_exact": full, "bits_gain_per_entry": (base_ce.item() - ce) * scored / LN2}


def _ce_tail(lg, ids):
    """lg (B, n-1, V) predicts ids[:, 1:]."""
    return F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))


def cramming(reader, rows, lengths, n_texts, steps, lr, dev) -> dict:
    out = {}
    for way in ("input", "kv"):
        for n in lengths:
            ids = torch.tensor([r[1:1 + n] for r in rows[:n_texts]], device=dev)
            t0 = time.time()
            res = _cram(reader, ids, way, steps, lr)
            res["seconds"] = round(time.time() - t0, 1)
            out[f"{way}_{n}"] = res
            print(f"[cramming] {way} {n}: {res}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", default=None, help="S0: a reader Hub id (no writer)")
    ap.add_argument("--checkpoint", default=None, help="S1+: a trained E36 checkpoint (writer + adapter)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rows_per_source", type=int, default=256)
    ap.add_argument("--span", type=int, default=512)
    ap.add_argument("--mem_spans", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--squad_items", type=int, default=500)
    ap.add_argument("--cramming", action="store_true")
    ap.add_argument("--cram_lengths", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--cram_texts", type=int, default=8)
    ap.add_argument("--cram_steps", type=int, default=800)
    ap.add_argument("--cram_lr", type=float, default=0.02)
    ap.add_argument("--skip", nargs="*", default=[], help="continuation | squad")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    fm = None
    if a.checkpoint:
        fm = load_frozen_reader_memory(a.checkpoint, dev)
        reader, name = fm.reader, fm.config.reader_name
        span, mem_spans = fm.config.span, fm.config.mem_spans
    else:
        name = a.reader
        reader = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if dev.startswith("cuda")
                                                      else torch.float32, attn_implementation="sdpa").to(dev).eval()
        reader.requires_grad_(False)
        span, mem_spans = a.span, a.mem_spans
    tok = AutoTokenizer.from_pretrained(name)
    res = {"reader": name, "checkpoint": a.checkpoint, "span": span, "mem_spans": mem_spans, "started": _now(),
           "config": fm.config.to_dict() if fm is not None else None}
    print(f"[e36-eval] {_now()} reader={name} checkpoint={a.checkpoint} span={span} M={mem_spans}", flush=True)
    rows_by_src = dict(_eval_rows(a.manifest, a.rows_per_source))
    if "continuation" not in a.skip:
        res["continuation"] = {}
        for src, rows in rows_by_src.items():
            r = continuation(reader, fm, rows, span, mem_spans, a.batch_size, dev)
            res["continuation"][src] = r
            print(f"[continuation] {src}: {r}", flush=True)
    if "squad" not in a.skip and a.squad_items > 0:
        conds = ["none", "raw"] + (["memory", "other"] if fm is not None else [])
        res["squad"] = squad(reader, fm, tok, a.squad_items, dev, conds)
        print(f"[squad] {res['squad']}", flush=True)
    if a.cramming:
        rows = next(iter(rows_by_src.values()))
        with torch.enable_grad():
            res["cramming"] = cramming(reader, rows, a.cram_lengths, a.cram_texts, a.cram_steps, a.cram_lr, dev)
    res["finished"] = _now()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[e36-eval] {_now()} wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
