"""Same documents, same text, any model: how much does each model gain from the far past of real documents?

Companion to evaluation/long_context_use.py, which tests one perceiver_ar checkpoint in its own tokens. This script
compares models with different tokenizers and context limits on exactly the same text, scored in bits per byte
(nats summed over the target text / ln 2 / its UTF-8 bytes; lower is better), so a bigger vocabulary earns nothing.

  prepare  picks the documents, spans and targets of long_context_use.py (by default the 64k spans: up to 16 held-out
           LongDocs-lite documents per source, the text of their last 4,096 SmolLM2 tokens as the target, the same
           donor documents for the swapped-past test) and stores them as text with character offsets for each
           context budget. Budgets are counted in SmolLM2 tokens of text; every model reads the same characters.
  score    scores one model (a Hub id, or our perceiver_ar checkpoint folder) on those spans: the start the model's own
           tokenizer gives a text (see load_model), then the context text, then the target text. When the input
           does not fit the model's maximum length, the target is scored with sliding windows of that length (stride
           512 / 1,024 / 4,096 tokens for limits of 2k / 8k / longer), so every model gets as much of the document as
           it can take; larger budgets then give the same score and are copied, marked "windowed".

Usage (from a MrCogito checkout, one GPU per process):
  uv run python evaluation/long_context_compare.py prepare --tokenizer <our checkpoint folder> \
      --data <LongDocs-lite shards> --out spans_64k.json.gz
  CUDA_VISIBLE_DEVICES=0 uv run python evaluation/long_context_compare.py score \
      --spans spans_64k.json.gz --model HuggingFaceTB/SmolLM2-135M > smollm2.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BUDGETS = (1024, 4096, 16384, 28672)  # SmolLM2 tokens of context before the target; "full" is added per span


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------------- prepare
def cmd_prepare(a) -> None:
    import tiktoken
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path.cwd()))
    from data.bin_token_dataset import open_shard

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    g2 = tiktoken.get_encoding("gpt2")
    L, T, near = a.span, a.target, a.near
    full = L - 1 - T
    budgets = [c for c in BUDGETS if c < full] + [full]
    out = {"span": L, "target": T, "near": near, "budgets": budgets, "tokenizer": a.tokenizer, "sources": {}}
    for src in (s for s in a.sources.split(",") if s):
        texts = []
        for f in sorted(Path(a.data).glob(f"{src}_val_*.bin")):
            sh = np.asarray(open_shard(str(f)))
            starts = np.flatnonzero(sh == 50256)
            ends = np.append(starts[1:], len(sh))
            texts += [g2.decode(sh[s + 1:e].tolist()) for s, e in zip(starts, ends)]
        if not (Path(a.cache) / f"{src}_val_retok.npz").exists():  # build long_context_use.py's cache first
            from evaluation.long_context_use import load_docs

            eot = tok.convert_tokens_to_ids("<|endoftext|>")
            load_docs(Path(a.data), src, tok, eot if eot is not None else tok.eos_token_id, 0, Path(a.cache))
        z = np.load(Path(a.cache) / f"{src}_val_retok.npz")
        cached = [z[k] for k in sorted(z.files, key=lambda s: int(s[1:]))]
        assert len(cached) == len(texts), (src, len(cached), len(texts))
        lens = [len(d) for d in cached]  # eot + the document's tokens, as in long_context_use.py
        picked = [j for j, n in enumerate(lens) if n >= L][: a.docs]
        donors = [j for j, n in enumerate(lens) if n >= L - T - near]
        rng = random.Random(1234)  # the same draw order as long_context_use.py, so the same donors

        memo = {}

        def tok_ends(j: int) -> list[int]:
            """Character end of each of the document's first L - 1 SmolLM2 tokens."""
            if j not in memo:
                t = texts[j]
                for cut in (8 * L, None):
                    enc = tok(t if cut is None else t[:cut], add_special_tokens=False, return_offsets_mapping=True)
                    if len(enc["input_ids"]) >= L or cut is None:
                        break
                ids = enc["input_ids"][: L - 1]
                if ids != cached[j][1: len(ids) + 1].tolist():
                    log(f"warning: {src} doc {j}: retokenised ids differ from the cache")
                memo[j] = [e for _, e in enc["offset_mapping"][: L - 1]]
            return memo[j]

        docs = []
        for j in picked:
            e = tok_ends(j)
            ts = e[L - T - 2]
            d = {"doc": j, "text": texts[j][: e[L - 2]], "tgt_start": ts,
                 "ctx_start": {str(c): (e[L - T - 2 - c] if c < full else 0) for c in budgets}}
            pool = [k for k in donors if k != j]
            if pool:
                k = pool[rng.randrange(len(pool))]
                ek = tok_ends(k)
                d["swap"] = {"donor": k, "donor_text": texts[k][: ek[L - T - near - 2]], "near_start": e[L - T - 2 - near]}
            docs.append(d)
            log(f"{src} doc {j}: target {len(d['text']) - ts:,} chars, span {len(d['text']):,} chars")
        out["sources"][src] = docs
    with gzip.open(a.out, "wt") as f:
        json.dump(out, f)
    log(f"wrote {a.out}: {sum(len(v) for v in out['sources'].values())} documents")


# --------------------------------------------------------------------------------------------------------- score
def load_model(name: str, device: str, max_len: int | None, lead_mode: str, dtype: str = "bfloat16"):
    """(model, tokenizer, lead tokens, max length, ours?). Each input starts the way the model's own tokenizer starts
    a text (`default`: Gemma adds <bos>, SmolLM2, Pythia and Granite add nothing), or with its bos/eos token (`bos`),
    or with nothing (`none`). Our checkpoint always starts with <|endoftext|>, as in its training. A start token the
    model does not expect can wreck it at long lengths: Granite 4.0 350M with a leading <|end_of_text|> scored 2.12
    bits per byte on a book at 19.8k tokens and 1.07 without it (2026-11-18 probe, both attention kernels agree)."""
    from transformers import AutoTokenizer

    cfg = Path(name) / "config.json"
    if cfg.exists() and json.loads(cfg.read_text()).get("model_type") == "perceiver_ar":
        sys.path.insert(0, str(Path.cwd()))
        from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

        model = load_perceiver_ar_for_eval(name, device=device, attn_backend="flex")
        tok = AutoTokenizer.from_pretrained(name)
        return model, tok, [int(model.config.eos_token_id)], max_len or 65536, True
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), attn_implementation="sdpa")
    model = model.to(device).eval()
    with_sp, without = tok("a")["input_ids"], tok("a", add_special_tokens=False)["input_ids"]
    default = with_sp[: len(with_sp) - len(without)] if with_sp[len(with_sp) - len(without):] == without else []
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    lead = {"default": default, "bos": [int(bos)], "none": []}[lead_mode]
    return model, tok, lead, max_len or int(model.config.max_position_embeddings), False


@torch.no_grad()
def nll_last(model, ids: list[int], n: int, ours: bool, device: str) -> float:
    """Summed nats of predicting the last n tokens of ids."""
    x = torch.tensor([ids], device=device)
    if ours:
        doc = torch.zeros_like(x)
        try:
            _, per, _ = model(input_ids=x, labels=x, doc_ids=doc, return_per_token_loss=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            with model.open_impl("list"):
                _, per, _ = model(input_ids=x, labels=x, doc_ids=doc, return_per_token_loss=True)
        return float(per[0, -n:].float().sum())
    try:
        logits = model(input_ids=x, logits_to_keep=n + 1).logits[0, :-1]
    except TypeError:
        logits = model(input_ids=x).logits[0, -n - 1:-1]
    tgt = x[0, -n:]
    return sum(float(F.cross_entropy(logits[i:i + 1024].float(), tgt[i:i + 1024], reduction="sum"))
               for i in range(0, n, 1024))


def score_ids(model, lead: list[int], ctx: list[int], tgt: list[int], W: int, stride: int, ours: bool, device: str):
    """(summed nats over tgt, windowed?) with ctx before it; sliding windows of W tokens when it does not fit."""
    ids, t0, n, room = ctx + tgt, len(ctx), len(tgt), W - len(lead)
    if len(ids) <= room:
        return nll_last(model, lead + ids, n, ours, device), False
    tot = 0.0
    for k in range(0, n, stride):
        end = t0 + min(k + stride, n)
        tot += nll_last(model, lead + ids[max(0, end - room):end], end - (t0 + k), ours, device)
    return tot, True


def cmd_score(a) -> None:
    with gzip.open(a.spans, "rt") as f:
        sp = json.load(f)
    model, tok, lead, W, ours = load_model(a.model, a.device, a.max_len, a.lead, a.dtype)
    stride = a.stride or (512 if W <= 2048 else 1024 if W <= 8192 else 4096)
    params = sum(p.numel() for p in model.parameters())
    log(f"{a.model}: {params / 1e6:.1f}M parameters, max length {W}, stride {stride}, starts with {lead}")

    def enc(s: str) -> list[int]:
        return tok(s, add_special_tokens=False)["input_ids"]

    res = {"model": a.model, "params": params, "max_len": W, "stride": stride, "lead": lead, "lead_mode": a.lead,
           "dtype": a.dtype,
           "budgets": sp["budgets"], "span": sp["span"], "target": sp["target"], "sources": {}}
    t_start = time.time()
    for src, docs in sp["sources"].items():
        rows = []
        for d in docs[: a.max_docs or None]:
            text, ts = d["text"], d["tgt_start"]
            tgt = enc(text[ts:])
            r = {"doc": d["doc"], "target_bytes": len(text[ts:].encode("utf-8")), "target_tokens": len(tgt),
                 "cut": {}, "ctx_tokens": {}, "windowed": {}}
            last = None
            for c in sp["budgets"]:
                c = str(c)
                ctx = enc(text[d["ctx_start"][c]:ts])
                r["ctx_tokens"][c] = len(ctx)
                if last is not None and r["windowed"][last]:
                    r["cut"][c], r["windowed"][c] = r["cut"][last], True  # only the last W tokens are read
                else:
                    r["cut"][c], r["windowed"][c] = score_ids(model, lead, ctx, tgt, W, stride, ours, a.device)
                last = c
            if a.swap and "swap" in d:
                ctx = enc(d["swap"]["donor_text"] + text[d["swap"]["near_start"]:ts])
                r["swap"], r["swap_windowed"] = score_ids(model, lead, ctx, tgt, W, stride, ours, a.device)
                r["swap_donor"] = d["swap"]["donor"]
            rows.append(r)
            bpb = {k: v / math.log(2) / r["target_bytes"] for k, v in r["cut"].items()}
            log(f"{src} doc {d['doc']} ({time.time() - t_start:.0f}s): " + " ".join(f"{k}={v:.3f}" for k, v in bpb.items())
                + (f" swap={r['swap'] / math.log(2) / r['target_bytes']:.3f}" if "swap" in r else "") + " bits/byte")
        res["sources"][src] = rows
    print(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tokenizer", required=True, help="the SmolLM2 tokenizer that defines spans and budgets")
    p.add_argument("--data", required=True, help="LongDocs-lite dir (GPT-2 val shards <source>_val_*.bin)")
    p.add_argument("--cache", default="Cache/e38_board/longctx", help="long_context_use.py's retokenised cache")
    p.add_argument("--sources", default="gutenberg,arxiv,doab,usgpo,hansard")
    p.add_argument("--span", type=int, default=65536)
    p.add_argument("--target", type=int, default=4096)
    p.add_argument("--near", type=int, default=4096, help="own tokens kept before the target in the swap test")
    p.add_argument("--docs", type=int, default=16)
    p.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--spans", required=True)
    s.add_argument("--model", required=True, help="Hub id, local folder, or a perceiver_ar checkpoint folder")
    s.add_argument("--max_len", type=int, default=None, help="override the model's maximum length")
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--swap", action="store_true")
    s.add_argument("--lead", default="default", choices=["default", "bos", "none"], help="start of each input")
    s.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="weights of a Hub model; float32 for fp16-trained models such as Pythia, which break in bf16")
    s.add_argument("--max_docs", type=int, default=0, help="per source; 0 = all (smoke tests use 1)")
    s.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
