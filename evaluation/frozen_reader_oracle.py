#!/usr/bin/env python
"""E36 oracle check: can the frozen reader answer SQuAD questions from key/value entries of a passage at all?

The entries of each passage (one per `passage_span`-token chunk, as in S2) are OPTIMISED directly instead of being
written by the writer; they start from the trained writer's entries, so an oracle can only improve on it:
  free     : every number of every per-layer key and value of every entry is free;
  expander : the writer-space entries (576 numbers each), read through the trained key/value expander.
Objectives, neither of which sees the question being scored:
  recon    : the reader rebuilds the passage from the entries (cross-entropy on the passage tokens);
  teacher  : KL to the reader with the passage in context, on the passage's OTHER questions; a passage's questions
             are split in two halves, each scored with entries optimised on the other half.
Every question is scored zero-shot ("Question: q\\nAnswer:", greedy, 16 tokens, as evaluation/frozen_reader_eval.py)
under: none, raw (the passage in context), writer (the trained entries), and the four oracles.

  uv run python evaluation/frozen_reader_oracle.py --checkpoint <E36 S2 run>/final --out oracle.json
  (split across GPUs with --offset 0/50/100/150 --passages 50; merge the JSONs by their item counts)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer, DynamicCache  # noqa: E402
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: E402

from evaluation.frozen_reader_eval import _greedy, _norm, _now  # noqa: E402
from nn.frozen_reader_memory import load_frozen_reader_memory  # noqa: E402

ORACLES = ("recon_free", "recon_expander", "teacher_free", "teacher_expander")


def _passages(n: int, max_q: int, seed: int):
    from datasets import load_dataset

    by_ctx: dict[str, list] = defaultdict(list)
    for ex in load_dataset("rajpurkar/squad", split="validation"):
        by_ctx[ex["context"]].append(ex)
    ctxs = [c for c, qs in by_ctx.items() if len(qs) >= 2]
    random.Random(seed).shuffle(ctxs)
    return [(c, by_ctx[c][:max_q]) for c in ctxs[:n]]


def _pad(seqs, pad, dev):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long, device=dev)
    mask = torch.zeros((len(seqs), n), dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, device=dev)
        mask[i, :len(s)] = 1
    return ids, mask


def _kv_logits(fm, x, epos, emask, ids, tpos, tmask):
    """Reader logits over ids (B, T) with a key/value prefix x (B, K, L, 2, h, hd) at positions epos (B, K)."""
    r, dt = fm.reader, fm.reader.dtype
    cos, sin = r.model.rotary_emb(x[:, :, 0, 0, 0, :].to(dt), epos)
    cache = DynamicCache()
    for i in range(x.shape[2]):
        k, _ = apply_rotary_pos_emb(x[:, :, i, 0].transpose(1, 2).to(dt), x[:, :, i, 0].transpose(1, 2).to(dt), cos, sin)
        cache.update(k, x[:, :, i, 1].transpose(1, 2).to(dt), i)
    m, t = x.shape[1], ids.shape[1]
    return r(input_ids=ids, position_ids=tpos, past_key_values=cache, cache_position=torch.arange(m, m + t, device=ids.device),
             attention_mask=torch.cat([emask.long(), tmask.long()], 1), use_cache=True).logits


class Batch:
    """A group of passages: chunks, entry positions, the writer's entries, and the targets of each objective."""

    def __init__(self, fm, tok, group, span, dev):
        self.dev, self.span = dev, span
        self.ctx = [tok.encode(c.strip() + "\n\n", add_special_tokens=False) for c, _ in group]
        self.qs = [qs for _, qs in group]
        chunks = [[c[j:j + span] for j in range(0, len(c), span)] for c in self.ctx]
        b, k = len(group), max(len(c) for c in chunks)
        ids = torch.zeros((b, k, span), dtype=torch.long, device=dev)
        mask = torch.zeros((b, k, span), dtype=torch.long, device=dev)
        for i, cs in enumerate(chunks):
            for j, c in enumerate(cs):
                ids[i, j, :len(c)] = torch.tensor(c, device=dev)
                mask[i, j, :len(c)] = 1
        self.emask = mask.any(-1)
        self.epos = (torch.arange(k, device=dev) * span + span // 2).expand(b, -1).contiguous()
        self.base = torch.tensor([len(c) * span for c in chunks], device=dev)  # targets start after the entries
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            self.w0 = fm.writer(ids.view(b * k, span), mask.view(b * k, span)).view(b, k, -1).float()
            self.x0 = fm.adapter(self.w0).float()

    def tpos(self, idx, t):
        return self.base[idx].view(-1, 1) + torch.arange(t, device=self.dev)


def _optimise(fm, batch, kind, objective, tok, half, steps, lr):
    """Return the optimised key/value entries (B, K, L, 2, h, hd) for one batch of passages."""
    dev = batch.dev
    p = torch.nn.Parameter((batch.x0 if kind == "free" else batch.w0).clone())
    opt = torch.optim.Adam([p], lr=lr)
    if objective == "recon":
        idx = torch.arange(len(batch.ctx), device=dev)
        ids, tmask = _pad(batch.ctx, 0, dev)
    else:  # teacher matching on the half of each passage's questions that will NOT be scored
        pairs = [(i, f"Question: {q['question'].strip()}\nAnswer: {q['answers']['text'][0].strip()}\n")
                 for i, qs in enumerate(batch.qs) for n, q in enumerate(qs) if n % 2 == half]
        idx = torch.tensor([i for i, _ in pairs], device=dev)
        tgts = [tok.encode(s, add_special_tokens=False) for _, s in pairs]
        ids, tmask = _pad(tgts, 0, dev)
        tea, tea_mask = _pad([batch.ctx[i] + t for (i, _), t in zip(pairs, tgts)], 0, dev)
        mem_len = torch.tensor([len(batch.ctx[i]) for i, _ in pairs], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            teacher = fm._teacher_logprobs(tea, tea_mask, mem_len, ids.shape[1])
    tpos = batch.tpos(idx, ids.shape[1])
    w = tmask[:, 1:].float()
    last = None
    for _ in range(steps):
        x = p if kind == "free" else fm.adapter(p)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            lg = _kv_logits(fm, x[idx], batch.epos[idx], batch.emask[idx], ids, tpos, tmask)[:, :-1].float()
        if objective == "recon":
            per = F.cross_entropy(lg.transpose(1, 2), ids[:, 1:], reduction="none")
        else:
            per = F.kl_div(lg.log_softmax(-1), teacher, log_target=True, reduction="none").sum(-1)
        loss = (per * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()
    with torch.no_grad():
        return (p if kind == "free" else fm.adapter(p)).detach(), last


@torch.no_grad()
def _answer(fm, tok, batch, i, q, x, stop, dev):
    """Greedy answer to question q of passage i, given key/value entries x (B, K, ...) or a condition name."""
    prompt = torch.tensor([tok.encode(f"Question: {q['question']}\nAnswer:", add_special_tokens=False)], device=dev)
    r = fm.reader
    if x == "none":
        fn = lambda cur: r(input_ids=cur, use_cache=False).logits  # noqa: E731
    elif x == "raw":
        prompt = torch.cat([torch.tensor([batch.ctx[i]], device=dev), prompt], 1)
        fn = lambda cur: r(input_ids=cur, use_cache=False).logits  # noqa: E731
    else:
        def fn(cur):
            t = cur.shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                return _kv_logits(fm, x[i:i + 1], batch.epos[i:i + 1], batch.emask[i:i + 1], cur,
                                  batch.tpos(torch.tensor([i], device=dev), t), torch.ones_like(cur))
    pred = _norm(tok.decode(_greedy(fn, prompt, 16, stop)))
    golds = [_norm(a) for a in q["answers"]["text"]]
    return any(pred == g for g in golds), any(g and g in pred for g in golds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--passages", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0, help="skip this many passages (to split a run across GPUs)")
    ap.add_argument("--max_questions", type=int, default=4)
    ap.add_argument("--batch_passages", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr_free", type=float, default=1e-2)
    ap.add_argument("--lr_expander", type=float, default=1e-2)
    ap.add_argument("--passage_span", type=int, default=0, help="0: the checkpoint's passage_span (or span)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    fm = load_frozen_reader_memory(a.checkpoint, dev)
    fm.requires_grad_(False)
    if fm.config.way != "kv":
        raise ValueError("the oracle check needs a key/value-way checkpoint")
    span = a.passage_span or getattr(fm.config, "passage_span", 0) or fm.config.span
    tok = AutoTokenizer.from_pretrained(fm.config.reader_name)
    stop = {tok.eos_token_id, *tok.encode("\n", add_special_tokens=False)}
    items = _passages(a.offset + a.passages, a.max_questions, a.seed)[a.offset:]
    conds = ("none", "raw", "writer") + ORACLES
    hits = {c: Counter() for c in conds}
    losses = defaultdict(list)
    t0 = time.time()
    print(f"[oracle] {_now()} {len(items)} passages, {sum(len(q) for _, q in items)} questions, span={span}", flush=True)
    for g in range(0, len(items), a.batch_passages):
        batch = Batch(fm, tok, items[g:g + a.batch_passages], span, dev)
        xs = {"writer": batch.x0}
        for kind in ("free", "expander"):
            lr = a.lr_free if kind == "free" else a.lr_expander
            x, last = _optimise(fm, batch, kind, "recon", tok, None, a.steps, lr)
            xs[f"recon_{kind}"] = x
            losses[f"recon_{kind}"].append(last)
            for half in (0, 1):
                x, last = _optimise(fm, batch, kind, "teacher", tok, half, a.steps, lr)
                xs[f"teacher_{kind}_h{half}"] = x
                losses[f"teacher_{kind}"].append(last)
        for i, qs in enumerate(batch.qs):
            for n, q in enumerate(qs):
                for c in conds:
                    if c.startswith("teacher_"):
                        x = xs[f"{c}_h{1 - n % 2}"]  # optimised on the other half of this passage's questions
                    else:
                        x = c if c in ("none", "raw") else xs[c]
                    ex, co = _answer(fm, tok, batch, i, q, x, stop, dev)
                    hits[c]["exact"] += ex
                    hits[c]["contains"] += co
                    hits[c]["items"] += 1
        done = min(g + a.batch_passages, len(items))
        print(f"[oracle] {done}/{len(items)} passages, {time.time() - t0:.0f}s: "
              + " ".join(f"{c}={hits[c]['contains'] / max(hits[c]['items'], 1):.3f}" for c in conds), flush=True)
    res = {"checkpoint": a.checkpoint, "span": span, "steps": a.steps, "passages": len(items), "offset": a.offset,
           "finished": _now(),
           "squad": {c: {"exact": h["exact"] / h["items"], "contains": h["contains"] / h["items"], "items": h["items"]}
                     for c, h in hits.items()},
           "final_loss": {k: sum(v) / len(v) for k, v in losses.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res["squad"]), json.dumps(res["final_loss"]), flush=True)


if __name__ == "__main__":
    main()
