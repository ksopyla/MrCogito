#!/usr/bin/env python
"""Can a trained model already work as two agents? An eval-only probe, no retraining (2026-11-13).

Agent A reads a document and keeps the reading layer's memory of it. Agent B gets only the question, never the
document, and reads A's memory at the reading layer (nn/agent_split.py). Documents and questions are RULER's needle
tasks, built by lm-eval's own generator and scored its way (the reference string found in the reply).

Per task, length and document, the same question is answered these ways (each costs one position per token B
processes, ~25 ms on an L40S whatever the length, so the conditions and --new set the run time):
  joint    one agent reads [document + question] and answers: the usual cached decoding, the reference
  relay    A reads, B steps through the question with its local layers still reading A's last tokens: must equal joint
  split    A reads; B has the question only and reads A's memory
  bitsN    warm, with A's memory on a grid of N bits per number (how much of the 16 bits the answer uses); on the
           first --bits_docs documents of a cell
  tailN    split, but B's local layers still read A's last N tokens (how much of A's raw ending B leans on)
  warm     split, and B's own text starts with a neutral preamble of its own (WARM) before the question, so its local
           layers do not start on an empty window; nothing more of A's crosses
  swap     warm, but B reads the memory of the previous document: the answer must come from the memory
  alone    B with no memory at all: the floor

It also records what each way costs:
  seconds  A reading the document once; B from question to answer with the memory; one agent doing both
  sizes    the document as text (UTF-8, zlib, zstd, token ids, and its length in bits under this very model for the
           first --nll_docs documents: the size no lossless code shared by both agents can beat), A's whole memory,
           the remote read's traffic (queries up, reads down, per token B processes), and for E37's concept read the
           distinct document tokens and summaries B's reads touched (what a fetch protocol would send).

Usage (GPU): uv run --with wonderwords python scripts/a2a_split_probe.py --checkpoint RUN/final \
    --tokenizer DATA/tokenizer_gpt2 --tasks niah_single_1,niah_single_2 --lengths 4096,16384 --limit 50 --out out.json
Laptop pipeline check: add --device cpu --backend sdpa --dtype float32 --lengths 1024 --limit 2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NIAH = {  # lm_eval/tasks/ruler/niah_utils.py
    "niah_single_1": dict(type_haystack="repeat", type_needle_k="words", type_needle_v="numbers"),
    "niah_single_2": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers"),
    "niah_single_3": dict(type_haystack="essay", type_needle_k="words", type_needle_v="uuids"),
    "niah_multikey_1": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=4),
}
QUESTION = "\nWhat "  # RULER's template: "...{context}\nWhat is the special magic ... for {query} mentioned ...?"
# B's own preamble for the `warm` condition: plain prose, longer than the local window (128 tokens), no document content
WARM = ("You are about to answer a question about a long text that another reader has already gone through for you. "
        "You do not have the text itself, only the memory that the other reader kept while reading it. Take your "
        "time and look carefully through that memory for the passage that matters, and then give the exact answer "
        "that the text states, copying any names and numbers exactly as they were written there. Do not guess, and "
        "do not add anything that the text did not say. Some passages will look similar to each other, so make sure "
        "that the one you pick is about the very thing the question asks for. When you have found the right "
        "passage, state the answer directly and then stop. Here is the question that you should answer now.")


def ruler_docs(task: str, length: int, n: int, tok) -> list[dict]:
    from lm_eval.tasks.ruler import prepare_niah as pn
    from lm_eval.tasks.ruler.niah_utils import TEMPLATE

    spec = NIAH[task]
    return pn.generate_samples(pn.get_haystack(type_haystack=spec["type_haystack"]), max_seq_length=length,
                               template=TEMPLATE, num_samples=n, TOKENIZER=tok, **spec)


def two_sides(doc: dict, tok) -> tuple[list[int], list[int], str, bool]:
    """A's document and B's text (question + the task's answer prefix) as token ids; A's text; whether the two halves
    tokenize to the one-row prompt the harness builds."""
    text = doc["input"]
    cut = text.rfind(QUESTION)
    if cut < 0:
        raise ValueError("no question found in the RULER prompt")
    a_text, b_text = text[:cut], text[cut:] + " " + doc["gen_prefix"]
    a = tok.encode(a_text, add_special_tokens=False)
    b = tok.encode(b_text, add_special_tokens=False)
    whole = tok.encode(text + " " + doc["gen_prefix"], add_special_tokens=False)
    return a, b, a_text, whole == a + b


def found(reply: str, refs: list[str]) -> float:
    reply = reply.strip().lower()
    return sum(1.0 for r in refs if r.lower() in reply) / len(refs)


def _sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def doc_bits(m, buf: torch.Tensor, doc: torch.Tensor, n: int) -> float:
    """Bits this model needs for the document's tokens 1..n−1 given the ones before (its summed surprise)."""
    h = m.hidden_states(buf, doc_ids=doc)
    W, cap, tot = m.lm_head.weight, m.config.logit_softcap, 0.0
    for s in range(0, n - 1, 2048):
        e = min(n - 1, s + 2048)
        z = F.linear(h[:, s:e], W).float()
        if cap:
            z = cap * torch.tanh(z / cap)
        tot += float(F.cross_entropy(z[0], buf[0, s + 1:e + 1], reduction="sum"))
    return tot / math.log(2)


def text_sizes(a_text: str) -> dict:
    raw = a_text.encode("utf-8")
    out = {"utf8": len(raw), "zlib9": len(zlib.compress(raw, 9))}
    try:
        import zstandard

        out["zstd19"] = len(zstandard.ZstdCompressor(level=19).compress(raw))
    except ImportError:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="", help="default: the checkpoint directory")
    ap.add_argument("--tasks", default="niah_single_1,niah_single_2,niah_single_3")
    ap.add_argument("--lengths", default="4096,16384")
    ap.add_argument("--limit", type=int, default=50, help="documents per task and length")
    ap.add_argument("--limit_long", type=int, default=0, help="documents at lengths past 16k (0 = --limit)")
    ap.add_argument("--bits", default="8,4")
    ap.add_argument("--bits_docs", type=int, default=15, help="documents per cell that also get the bitsN conditions")
    ap.add_argument("--tails", default="16", help="A's last tokens B's local layers still read (tailN conditions)")
    ap.add_argument("--new", type=int, default=12, help="answer tokens (a 7-digit number takes 5; UUIDs need 32)")
    ap.add_argument("--nll_docs", type=int, default=3, help="documents per cell sized in bits under the model")
    ap.add_argument("--seed", type=int, default=-1,
                    help="documents of a cell from this seed, so two checkpoints are examined on the same documents "
                         "(lm-eval's generator draws needles, keys and depths from the global random state; "
                         "-1 = unseeded, as the runs of 2026-11-13 and E37c's of 2026-11-14 were)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval
    from nn.agent_split import SplitDecoder, row_for
    from nn.incremental import IncrementalDecoder

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.checkpoint)
    m = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, dtype=getattr(torch, a.dtype),
                                   attn_backend=a.backend, attn_pad_multiple=1)
    eos = tok.eos_token_id
    bits = [int(b) for b in a.bits.split(",") if b]
    tails = [int(b) for b in a.tails.split(",") if b]
    warm = torch.tensor(tok.encode(WARM, add_special_tokens=False), device=a.device)
    probe = SplitDecoder(m)
    att = m.layers[probe.gi].attn
    res = {"checkpoint": a.checkpoint, "kind": probe.kind, "vocab": int(m.config.vocab_size),
           "hidden": int(m.config.hidden_size), "query_heads": int(att.h), "head_dim": int(att.dh),
           "new": a.new, "bits": bits, "tails": tails, "warm_tokens": int(warm.shape[0]), "dtype": a.dtype,
           "seed": a.seed,
           "local_window": int(m.config.pre_window), "cells": []}
    print(f"{a.checkpoint}: {probe.kind} read, {att.h} query heads of {att.dh}", flush=True)
    # shortest documents first, every task at a length before the next length: a run cut short keeps whole lengths
    for L in sorted(int(x) for x in a.lengths.split(",")):
        for task in a.tasks.split(","):
            n_docs = a.limit_long if (a.limit_long and L > 16384) else a.limit
            t_gen = time.time()
            if a.seed >= 0:
                import random

                import numpy as np

                random.seed(f"{a.seed}:{task}:{L}")
                np.random.seed((a.seed * 1_000_003 + L) % (2 ** 31))
            docs = ruler_docs(task, L, n_docs, tok)
            conds = (["joint", "relay", "split"] + [f"tail{w}" for w in tails] + ["warm"]
                     + [f"bits{b}" for b in bits] + ["alone", "swap"])
            score = {c: [] for c in conds}
            follows, rows, secs = [], [], {"read_A": [], "B_with_memory": [], "one_agent": []}
            sizes, same_tokens, prev = [], 0, None
            cell = {"task": task, "length": L, "docs_seconds": round(time.time() - t_gen, 1)}
            for di, d in enumerate(docs):
                ia, ib, a_text, same = two_sides(d, tok)
                same_tokens += int(same)
                ta = torch.tensor(ia, device=a.device)
                tb = torch.tensor(ib, device=a.device)
                refs = d["outputs"]
                rec = {"index": di, "tokens_A": len(ia), "tokens_B": len(ib), "ref": refs}
                say = lambda toks: tok.decode(toks)
                try:
                    # one agent, the usual way
                    buf, doc, L0 = row_for(torch.cat([ta, tb]), a.new, eos=eos)
                    _sync(a.device); t0 = time.time()
                    g = IncrementalDecoder(m).greedy(buf, doc, L0, a.new, eos=eos)[0, L0:].tolist()
                    _sync(a.device); secs["one_agent"].append(time.time() - t0)
                    rec["joint"] = say(g)
                    # agent A reads
                    dec = SplitDecoder(m)
                    bufa, doca, La = row_for(ta, len(ib) + a.new + 64 + int(warm.shape[0]), eos=eos)
                    _sync(a.device); t0 = time.time()
                    dec.read(bufa, doca, La)
                    _sync(a.device); secs["read_A"].append(time.time() - t0)
                    dec.listen(split=False)
                    rec["relay"] = say(dec.answer(tb, a.new, eos=eos))
                    # agent B, the question only
                    dec.listen(split=True)
                    seen_k, seen_s, calls = {}, {}, [0, 0]

                    def trace(r, t, keys, summ, La=La):
                        k = keys[0, :La]
                        seen_k[r] = k.clone() if r not in seen_k else seen_k[r] | k
                        calls[0] += 1
                        calls[1] += int(k.sum())
                        if summ is not None:
                            seen_s[r] = summ[0].clone() if r not in seen_s else seen_s[r] | summ[0]

                    dec.trace = trace if dec.kind == "concept" else None
                    _sync(a.device); t0 = time.time()
                    out = dec.answer(tb, a.new, eos=eos)
                    _sync(a.device); secs["B_with_memory"].append(time.time() - t0)
                    dec.trace = None
                    rec["split"], rec["answer_tokens"] = say(out), len(out)
                    for w in tails:
                        dec.listen(split=True, tail=w)
                        rec[f"tail{w}"] = say(dec.answer(tb, a.new, eos=eos))
                    twb = torch.cat([warm, tb])
                    dec.listen(split=True)
                    rec["warm"] = say(dec.answer(twb, a.new, eos=eos))
                    for b in (bits if di < a.bits_docs else []):  # the coarse memory, B starting on its preamble
                        dec.listen(split=True, bits=b)
                        rec[f"bits{b}"] = say(dec.answer(twb, a.new, eos=eos))
                    dec.listen(blank=True)
                    rec["alone"] = say(dec.answer(tb, a.new, eos=eos))
                    if prev is not None and prev[0].S >= prev[0].L0 + len(twb) + a.new:
                        prev[0].listen(split=True)
                        rec["swap"] = say(prev[0].answer(twb, a.new, eos=eos))
                        follows.append(found(rec["swap"], prev[1]))  # does B report the other document's needle?
                    # sizes
                    up, down = dec.remote_read_numbers()
                    sz = {"tokens_A": len(ia), "tokens_B": len(ib), "answer_tokens": len(out),
                          "memory_numbers": dec.memory_numbers(), "remote_up_per_token": up,
                          "remote_down_per_token": down, **text_sizes(a_text)}
                    if seen_k:
                        sz["fetch_tokens"] = sum(int(v.sum()) for v in seen_k.values())       # distinct, summed over passes
                        sz["fetch_summaries"] = sum(int(v.sum()) for v in seen_s.values())
                        sz["keys_per_read"] = calls[1] / max(calls[0], 1)
                    if di < a.nll_docs:
                        sz["model_bits"] = doc_bits(m, bufa, doca, La)
                    sizes.append(sz)
                    dec.listen(split=True)  # leave the previous document's memory clean for the next swap
                    prev = (dec, refs)
                except torch.cuda.OutOfMemoryError as e:
                    rec["error"] = f"out of memory ({str(e)[:80]})"
                    prev = None
                    torch.cuda.empty_cache()
                for c in conds:
                    if c in rec:
                        score[c].append(found(rec[c], refs))
                rows.append(rec)
                if di < 3 or (di + 1) % 10 == 0:
                    print(f"  {task} {L} doc {di}: " + " | ".join(f"{c} {rec.get(c, '-')!r:.40}" for c in
                                                                 ("joint", "split", "warm", "alone", "swap")), flush=True)
            mean = lambda xs: (sum(xs) / len(xs)) if xs else None
            med = lambda xs: statistics.median(xs) if xs else None
            keys = sorted({k for s in sizes for k in s})
            cell.update(
                n=len(rows), same_tokens_as_one_row=same_tokens,
                score={c: mean(v) for c, v in score.items()}, n_scored={c: len(v) for c, v in score.items()},
                swap_reports_other_needle=mean(follows),
                seconds={k: med(v) for k, v in secs.items()},
                seconds_first={k: (v[0] if v else None) for k, v in secs.items()},
                sizes={k: mean([s[k] for s in sizes if k in s]) for k in keys},
                rows=rows,
            )
            res["cells"].append(cell)
            print(json.dumps({k: cell[k] for k in ("task", "length", "n", "score", "seconds", "sizes")}), flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
            prev = None
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
