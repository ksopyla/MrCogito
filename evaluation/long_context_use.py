"""Does a long-context model actually use the far past of real documents? Three tests on held-out long documents.

Built for the 150M concept-memory model's Hugging Face card (2026-11-17, author: "is the long context real, evaluate
on real long context"); works for any `perceiver_ar` checkpoint. Documents: the LongDocs-lite validation shards
(Common Pile Gutenberg, arXiv, DOAB, US GPO, UK Hansard; the same held-out documents as m21mix's validation for these
five sources), stored as GPT-2 tokens. GPT-2's byte-level BPE decodes losslessly, so each document is decoded to text
and re-encoded with the checkpoint's own tokenizer.

1. **Context cut** (Sun et al. 2021, "Do long-range language models actually use long-range context?"): the last
   `--target` tokens of an L-token span are scored with C ∈ `--contexts` tokens of the document before them (C =
   L − target is the full span). Loss falling as C grows = the far past is used.
2. **Swapped far past**: the same target and the same nearest 4,096 tokens, but everything before them taken from a
   different document of the same source. If the own-document full context beats this, the gain is the document's
   content, not a generic "more text before" effect.
3. **Memory off** (concept models): the reading layer's `concept_off` switch (sinks + own and previous 512-token
   block only: no opened tokens, no summaries) and `summaries off` (opened tokens kept, the coarse summaries removed),
   on the full span, loss by position bucket.

Every variant starts with the document-start token, then the span; one document per row (the read is
document-masked). Positions: the reading layer has no position code (global_nope) and the local layers' rotary
window is 128 tokens, so moving the target from position 61k to position 4k does not change what the model sees
apart from the context itself.

Usage (from a MrCogito checkout, one GPU per process):
  uv run python evaluation/long_context_use.py --checkpoint <our checkpoint folder> \
      --data <LongDocs-lite shards> --sources gutenberg,arxiv --span 65536 > out.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch

BUCKETS = (0, 512, 2048, 8192, 16384, 32768, 65536, 131072)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_docs(data: Path, src: str, tok, eot: int, min_len: int, cache: Path) -> list[np.ndarray]:
    """Validation documents of `src` with at least `min_len` tokens in the checkpoint's tokenizer (eot first)."""
    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"{src}_val_retok.npz"
    if cf.exists():
        z = np.load(cf)
        docs = [z[k] for k in sorted(z.files, key=lambda s: int(s[1:]))]
    else:
        import tiktoken

        from data.bin_token_dataset import open_shard

        g2 = tiktoken.get_encoding("gpt2")
        docs = []
        for f in sorted(data.glob(f"{src}_val_*.bin")):
            sh = np.asarray(open_shard(str(f)))
            starts = np.flatnonzero(sh == 50256)
            ends = np.append(starts[1:], len(sh))
            for a, b in zip(starts, ends):
                text = g2.decode(sh[a + 1:b].tolist())
                ids = tok(text, add_special_tokens=False)["input_ids"]
                docs.append(np.asarray([eot] + [t for t in ids if t != eot], dtype=np.int64))
        np.savez(cf, **{f"d{i}": d for i, d in enumerate(docs)})
    return [d for d in docs if len(d) >= min_len]


@contextmanager
def memory_mode(model, mode: str):
    """'on' (the model as trained), 'off' (sinks + window only), 'nosum' (opened tokens, no summaries).

    Concept read: `concept_off` / `concept["summaries"]`. LSA read (`par_global_read: lsa`, 2026-11-18): `lsa_off`
    keeps its sinks + 512-token window; it has no summaries, so 'nosum' is refused rather than silently a no-op.
    Notebook read (a `memory_writer`, e.g. the E38 catalogue, 2026-11-21): 'off' is `message_override("none")` (no
    notes, so nothing opens: window only), 'nosum' is `open_override("gist_off")` (opened tokens, the notes masked);
    the catalogue's notes only choose what opens (`lm_note_read: index`, no summaries in the read), so its 'nosum' is
    refused.
    Living read (E39, `par_global_read: living`): 'off' is `living_off`, 'nosum' turns its summaries off, and 'flat'
    (readout R2, E39 only) lets every visible span compete instead of the spans of the 32 best concepts."""
    living = [l.attn for l in model.layers if getattr(l.attn, "living", None) is not None]
    if living:
        if mode not in ("on", "off", "nosum", "flat"):
            raise ValueError(f"memory mode {mode!r}: the living read knows on | off | nosum | flat")
        prev_lv = [(a.living_off, a.living["summaries"], a.living["search"]) for a in living]
        for a in living:
            a.living_off = mode == "off"
            a.living["summaries"] = a.living["summaries"] and mode != "nosum"
            a.living["search"] = "flat" if mode == "flat" else a.living["search"]
        try:
            yield
        finally:
            for a, (off, s, srch) in zip(living, prev_lv):
                a.living_off, a.living["summaries"], a.living["search"] = off, s, srch
        return
    if mode == "flat":
        raise ValueError("memory mode 'flat' is the living read's (E39)")
    attns = [l.attn for l in model.layers if getattr(l.attn, "concept", None) is not None]
    lsa = [l.attn for l in model.layers if getattr(l.attn, "lsa", None) is not None]
    notes = getattr(model, "memory_writer", None) is not None and \
        str(getattr(model.config, "message_override", "real")) == "real"
    if mode == "nosum" and lsa and not attns:
        raise ValueError("'nosum' needs a concept read; the LSA read has no summaries (use --modes off)")
    if mode == "nosum" and notes and not attns and str(getattr(model.config, "lm_note_read", "gist")) == "index":
        raise ValueError("'nosum': the catalogue's notes only choose what opens, there are no summaries to mask "
                         "(use --modes off)")
    if mode != "on" and not attns and not lsa and not notes:
        raise ValueError(f"memory mode {mode!r}: the checkpoint has no concept, LSA or notebook read to switch")
    prev = [(a.concept_off, a.concept["summaries"]) for a in attns]
    prev_lsa = [a.lsa_off for a in lsa]
    for a in attns:
        a.concept_off = mode == "off"
        a.concept["summaries"] = mode != "nosum"
    for a in lsa:
        a.lsa_off = mode == "off"
    stack = ExitStack()
    if notes and mode == "off":
        stack.enter_context(model.message_override("none"))
    elif notes and mode == "nosum":
        stack.enter_context(model.open_override("gist_off"))
    try:
        with stack:
            yield
    finally:
        for a, (off, s) in zip(attns, prev):
            a.concept_off, a.concept["summaries"] = off, s
        for a, v in zip(lsa, prev_lsa):
            a.lsa_off = v


@torch.no_grad()
def per_token(model, ids: np.ndarray, device: str) -> torch.Tensor:
    """Per-token loss in nats: out[t] is the loss of predicting ids[t + 1]."""
    x = torch.from_numpy(ids)[None].to(device)
    doc = torch.zeros_like(x)
    try:
        _, per, _ = model(input_ids=x, labels=x, doc_ids=doc, return_per_token_loss=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        with model.open_impl("list"):
            _, per, _ = model(input_ids=x, labels=x, doc_ids=doc, return_per_token_loss=True)
    return per[0].float().cpu()


def target_loss(per: torch.Tensor, T: int) -> float:
    """Summed nats over the last T predicted tokens of the row."""
    return float(per[-T:].sum())


def buckets(per: torch.Tensor) -> dict:
    pos = torch.arange(1, per.shape[0] + 1)  # per[t] predicts the token at position t + 1
    out = {}
    for a, b in zip(BUCKETS[:-1], BUCKETS[1:]):
        m = (pos >= a) & (pos < b)
        if int(m.sum()):
            out[f"{a}-{b}"] = [float(per[m].sum()), int(m.sum())]
    return out


def run_source(model, tok, all_docs, src, a, device) -> dict:
    L, T, near = a.span, a.target, 4096
    contexts = [c for c in (int(x) for x in a.contexts.split(",")) if c + T < L] + [L - 1 - T]
    rng = random.Random(1234)
    picked = [j for j, d in enumerate(all_docs) if len(d) >= L][: a.docs]
    donors = [j for j, d in enumerate(all_docs) if len(d) >= L - T - near]
    rows = []
    t0 = time.time()
    for i, j in enumerate(picked):
        try:
            rows.append(score_doc(model, tok, all_docs, j, donors, contexts, rng, a, device))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # a length past what fits: record it, go on
            torch.cuda.empty_cache()
            rows.append({"doc": j, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        r = rows[-1]
        log(f"{src} L={L} doc {i + 1}/{len(picked)} ({time.time() - t0:.0f}s): " +
            (r["error"][:120] if "error" in r else " ".join(f"{k}={v / T:.3f}" for k, v in r["cut"].items()) +
             (f" swap={r['swap'] / T:.3f}" if "swap" in r else "")))
    return {"docs": len(rows), "contexts": contexts, "rows": rows}


def score_doc(model, tok, all_docs, j, donors, contexts, rng, a, device) -> dict:
    L, T, near = a.span, a.target, 4096
    d = all_docs[j]
    row = d[:L]  # eot + the document's first L - 1 tokens
    tgt = row[L - T:]
    r = {"doc": j, "doc_tokens": int(len(d)), "target_bytes": len(tok.decode(tgt.tolist()).encode("utf-8")),
         "cut": {}, "buckets": {}}
    with memory_mode(model, "on"):
        for C in contexts:
            ids = np.concatenate([row[:1], row[L - T - C:]])
            per = per_token(model, ids, device)
            r["cut"][str(C)] = target_loss(per, T)
            if C == L - 1 - T:
                r["buckets"]["on"] = buckets(per)
        pool = [k for k in donors if k != j]
        if a.swap and pool:
            k = pool[rng.randrange(len(pool))]
            far = all_docs[k][1: L - T - near]  # a different document's text, its start token dropped
            ids = np.concatenate([row[:1], far, row[L - T - near:]])
            assert len(ids) == L, len(ids)
            r["swap"] = target_loss(per_token(model, ids, device), T)
            r["swap_donor"] = k
    for mode in a.modes.split(","):
        if mode and mode != "on":
            with memory_mode(model, mode):
                per = per_token(model, row, device)
            r["cut"][mode] = target_loss(per, T)
            r["buckets"][mode] = buckets(per)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="LongDocs-lite dir (GPT-2 val shards <source>_val_*.bin)")
    ap.add_argument("--sources", default="gutenberg,arxiv,doab,usgpo,hansard")
    ap.add_argument("--span", type=int, default=65536, help="row length L; the target is its last --target tokens")
    ap.add_argument("--target", type=int, default=4096)
    ap.add_argument("--contexts", default="1024,4096,16384", help="context lengths C before the target (+ full)")
    ap.add_argument("--docs", type=int, default=16, help="documents per source, at most")
    ap.add_argument("--swap", action="store_true", help="also score with a different document's far past")
    ap.add_argument("--modes", default="off,nosum", help="memory switches on the full span: off, nosum")
    ap.add_argument("--cache", default="Cache/e38_board/longctx")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    from transformers import AutoTokenizer

    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, attn_backend="flex")
    eot = int(model.config.eos_token_id)
    res = {"checkpoint": a.checkpoint, "span": a.span, "target": a.target, "args": vars(a), "sources": {}}
    for src in (s for s in a.sources.split(",") if s):
        all_docs = load_docs(Path(a.data), src, tok, eot, 0, Path(a.cache))
        n = sum(len(d) >= a.span for d in all_docs)
        log(f"{src}: {len(all_docs)} documents, {n} of >= {a.span} tokens")
        if not n:
            res["sources"][src] = {"docs": 0}
            continue
        res["sources"][src] = run_source(model, tok, all_docs, src, a, a.device)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
