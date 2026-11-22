"""E39 pre-flight checks P1–P3 (docs/experiments_specs/ahead/E39_living_concepts.md), before anything is built.

All three run on the released 150M concept-memory model (arm e37a, the author's own weights) and read the reading
layer's keys and queries; nothing is trained. The grouping rules below are plain reference implementations of the
spec's rules (one row at a time, loops over strides), written for these checks; the model's fast kernels come later
(nn/living_concepts.py) and are tested against them.

  spans     P1  Is there a span signal? Adjacent-key cosine of the first loop pass against lags 2 … 256 on held-out
                16k rows of the long sources, the cut threshold θ that gives a mean span of 8 tokens (with the
                spec's bounds: spans of 1–16 tokens, at most 128 per 512-token stride), the span-length histogram,
                and how often a cut falls on a word start.
  concepts  P3  How do document-wide concepts grow? The leader rule on held-out documents of 16k, 64k and 128k for
                join thresholds λ 0.5 … 0.9 and for the λ that makes 1 span in 8 found a concept: concepts against
                tokens, single-span concepts, the largest concept, growth from 64k to 128k.
  ruler     P2  Is the diagnosis right? RULER's multi-value and multi-query prompts through lm-eval, one prompt at
                a time with cached decoding. For every expected answer value: the tokens it occupies in the haystack,
                in how many concepts they sit, which of them each reader (the last prompt position and every answer
                position, each of the 4 loop passes) reads word for word, the attention they get with and without
                the summaries in the softmax, and the same coverage under E39's rules simulated on the same keys and
                queries (spans + choice by match, two-level and flat). Each prompt is also decoded with the
                summaries switched off (does crowding by summaries cost the answer?).
  summary       Re-print the tables of a saved `ruler` run.

Usage (a home GPU, from the checkout; RULER needs wonderwords):
  .venv/bin/python scripts/e39_preflight.py spans --checkpoint CK --bin DIR --sources books,arxiv,... --out p1.json
  .venv/bin/python scripts/e39_preflight.py concepts --checkpoint CK --retok DIR --theta 0.55 --out p3.json
  uv run --no-sync --with wonderwords python scripts/e39_preflight.py ruler --checkpoint CK --length 8192 \
      --theta 0.55 --lam 0.7 --out p2_8k
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NEG = float("-inf")


@dataclass
class Geo:
    """The spec's sizes (tiny values in the local smoke test)."""

    stride: int = 512     # living_stride
    span_max: int = 16    # living_span_max
    span_cap: int = 128   # living_span_cap: spans per stride, forced + natural
    keep: int = 32        # living_keep: concepts kept by the first level
    cap: int = 512        # concept_top_k: tokens opened per reader and pass
    width: int = 256      # spans opened at most
    sinks: int = 4        # concept_sinks


def log(*a) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]", *a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------ reference grouping
@torch.no_grad()
def cut_spans(khat: torch.Tensor, member: torch.Tensor, doc: torch.Tensor, theta: float, geo: Geo):
    """Spans of one row. khat [S, d] unit keys of the first pass, member [S] bool (no sinks, no padding), doc [S].

    A span starts at a member token when the token before it is not a member of the same document, or a stride
    starts, or cos(k̂_t, k̂_{t−1}) < θ (a natural cut). A stride keeps at most span_cap − stride / span_max natural
    cuts, its lowest-cosine ones; a run longer than span_max is then split every span_max.
    Returns span id per token [S] (−1 = not a member; ids in position order), the span count and the cut masks."""
    S = khat.shape[0]
    dev = khat.device
    t = torch.arange(S, device=dev)
    cos = torch.full((S,), -2.0, device=dev)
    cos[1:] = (khat[1:] * khat[:-1]).sum(-1)
    prev_ok = torch.zeros(S, dtype=torch.bool, device=dev)
    prev_ok[1:] = member[:-1] & (doc[1:] == doc[:-1])
    inner = member & prev_ok & (t % geo.stride != 0)          # tokens that may continue the span before them
    natural = inner & (cos < theta)
    room = geo.span_cap - geo.stride // geo.span_max
    ns = -(-S // geo.stride)
    c = torch.where(natural, cos, torch.full_like(cos, 3.0))
    c = F.pad(c, (0, ns * geo.stride - S), value=3.0).view(ns, geo.stride)
    keep = (c.argsort(1).argsort(1) < room).view(-1)[:S]
    capped = natural & ~keep
    natural = natural & keep
    start = member & (~inner | natural)
    run0 = torch.cummax(torch.where(start, t, torch.zeros_like(t)), 0).values
    forced = member & ~start & ((t - run0) % geo.span_max == 0)
    start = start | forced
    sid = torch.cumsum(start.long(), 0) - 1
    sid = torch.where(member, sid, torch.full_like(sid, -1))
    return sid, int(start.sum()), {"cos": cos, "inner": inner, "natural": natural, "forced": forced, "capped": capped}


@torch.no_grad()
def leader_concepts(khat: torch.Tensor, sid: torch.Tensor, n_spans: int, lam: float, geo: Geo,
                    c_max: int | None = None, rounds: int = 3):
    """Document-wide concepts of one row by the leader rule, stride by stride.

    A span's centre is the normalised sum of its tokens' unit keys; a concept's centre the normalised sum of its
    member tokens' unit keys. Every span of a stride is compared with the concepts as they stood after the stride
    before: it joins the best one when the cosine is at least λ, otherwise it founds. Founders of one stride are
    merged by `rounds` parallel rounds of the leader rule (an unresolved span with no earlier unresolved span
    within λ leads; unresolved spans within λ of a new leader follow the earliest one); spans still unresolved
    found alone. With `c_max` concepts allocated, further founders join their nearest concept whatever the cosine.
    Returns concept per span [n_spans], the concept count, concepts after each stride [n_strides], each span's
    stride, first position and token count."""
    S, d = khat.shape
    dev = khat.device
    m = sid >= 0
    pos = torch.arange(S, device=dev)
    U = torch.zeros(n_spans, d, device=dev).index_add_(0, sid[m], khat[m])
    n_tok = torch.zeros(n_spans, device=dev).index_add_(0, sid[m], torch.ones(int(m.sum()), device=dev))
    first = torch.full((n_spans,), S, dtype=torch.long, device=dev).scatter_reduce_(0, sid[m], pos[m], "amin")
    s_stride = first // geo.stride
    uhat = F.normalize(U, dim=-1)
    n_strides = -(-S // geo.stride)
    limit = int(c_max) if c_max else n_spans
    csum = torch.zeros(max(min(limit, n_spans), 1), d, device=dev)
    nC = 0
    cof = torch.full((n_spans,), -1, dtype=torch.long, device=dev)
    after = torch.zeros(n_strides, dtype=torch.long)
    bounds = torch.searchsorted(s_stride.contiguous(), torch.arange(n_strides + 1, device=dev)).tolist()
    for s in range(n_strides):
        a, b = bounds[s], bounds[s + 1]
        if b > a:
            u = uhat[a:b]
            mm = b - a
            if nC:
                best, arg = (u @ F.normalize(csum[:nC], dim=-1).T).max(1)
                join = best >= lam
            else:
                arg = torch.zeros(mm, dtype=torch.long, device=dev)
                join = torch.zeros(mm, dtype=torch.bool, device=dev)
            assign = torch.where(join, arg, torch.full_like(arg, -1))
            f = (~join).nonzero().flatten()
            mf = int(f.numel())
            if mf:
                uf = u[f]
                near = (uf @ uf.T) >= lam
                ar = torch.arange(mf, device=dev)
                lower = ar[None, :] < ar[:, None]
                unres = torch.ones(mf, dtype=torch.bool, device=dev)
                lead_of = ar.clone()
                for _ in range(rounds):
                    lead = unres & ~(near & lower & unres[None, :]).any(1)
                    fol = near & lead[None, :]
                    has = fol.any(1) & unres & ~lead
                    lead_of = torch.where(has, fol.float().argmax(1), lead_of)
                    unres = unres & ~lead & ~has
                is_lead = lead_of == ar
                new_id = torch.cumsum(is_lead.long(), 0) - 1
                room = limit - nC
                cid = (nC + new_id)[lead_of]
                over = (new_id >= room)[lead_of]
                if bool(over.any()):   # the slots are full: nearest concept of the strides before, whatever the cosine
                    cid = torch.where(over, arg[f], cid)
                assign[f] = cid
                nC += min(int(is_lead.sum()), room)
            csum.index_add_(0, assign, U[a:b])
            cof[a:b] = assign
        after[s] = nC
    return cof, nC, after, s_stride, first, n_tok


def span_stats(k: torch.Tensor, sid: torch.Tensor, n_spans: int):
    """Additive statistics of each span from one pass's keys k [S, d]: n, Σk, Σ|k|²."""
    m = sid >= 0
    d = k.shape[1]
    n = torch.zeros(n_spans, device=k.device).index_add_(0, sid[m], torch.ones(int(m.sum()), device=k.device))
    sk = torch.zeros(n_spans, d, device=k.device).index_add_(0, sid[m], k[m])
    sk2 = torch.zeros(n_spans, device=k.device).index_add_(0, sid[m], (k[m] ** 2).sum(-1))
    return n, sk, sk2


def match_score(qf: torch.Tensor, n: torch.Tensor, sk: torch.Tensor, sk2: torch.Tensor) -> torch.Tensor:
    """E39's choice score: the estimated attention weight of an average member, best query head,
    q·μ/√d + |q|² σ² / (2d) (the summary logit without log n). qf [h, d] → [G] (−inf where empty)."""
    d = qf.shape[-1]
    nn_ = n.clamp_min(1.0)
    mu = sk / nn_[:, None]
    var = (sk2 / nn_ - (mu ** 2).sum(-1)).clamp_min(0.0) / d
    s = (qf @ mu.T) / math.sqrt(d) + ((qf ** 2).sum(-1, keepdim=True) / (2 * d)) * var[None]
    return s.amax(0).masked_fill(n <= 0, NEG)


def e39_open(qf, vis, cof, nC, st, sid, first, geo: Geo, two_level: bool) -> torch.Tensor:
    """The tokens one reader opens under E39's rules: [S] bool. vis [n_spans]: the spans it may open (two strides
    back or more); st: this pass's span statistics. Two-level: the `keep` best concepts (their versions built from
    the visible spans), then their spans best first until `cap` tokens; flat: every visible span competes."""
    n, sk, sk2 = st
    n_spans = n.shape[0]
    dev = n.device
    S = sid.shape[0]
    cand = vis & (n > 0)
    if two_level and nC > 0 and bool(cand.any()):
        cn = torch.zeros(nC, device=dev).index_add_(0, cof[cand], n[cand])
        csk = torch.zeros(nC, sk.shape[1], device=dev).index_add_(0, cof[cand], sk[cand])
        csk2 = torch.zeros(nC, device=dev).index_add_(0, cof[cand], sk2[cand])
        top = match_score(qf, cn, csk, csk2).topk(min(geo.keep, nC))
        keepc = torch.zeros(nC, dtype=torch.bool, device=dev)
        keepc[top.indices[top.values > NEG]] = True
        cand = cand & keepc[cof.clamp(min=0)]
    mm = min(geo.width, int(cand.sum()))
    if mm == 0:
        return torch.zeros(S, dtype=torch.bool, device=dev)
    order = match_score(qf, n, sk, sk2).masked_fill(~cand, NEG).topk(mm).indices
    cnt = n[order].long()
    off = cnt.cumsum(0) - cnt
    T = torch.full((n_spans,), geo.cap, dtype=torch.long, device=dev)
    T[order] = torch.where(off < geo.cap, off, torch.full_like(off, geo.cap))
    s = sid.clamp(min=0)
    return (sid >= 0) & (T[s] + (torch.arange(S, device=dev) - first[s]) < geo.cap)


# ------------------------------------------------------------------------------------------ model access
class _Stop(Exception):
    pass


def load(a):
    from transformers import AutoTokenizer

    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    model = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, attn_backend=a.backend)
    return model, AutoTokenizer.from_pretrained(a.checkpoint)


@torch.no_grad()
def first_pass_keys(model, ids: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """Unit keys of the reading layer on its first loop pass [S, d] fp32 (after the layers below it, the pass marker,
    wk and the key norm). The forward stops there, so 128k rows cost only the local layers."""
    att = model.layers[model.config.global_layer_index].attn
    box = {}

    def grab(q, k, v, x, *, doc_ids, key_valid):
        box["k"] = k[0, :, 0].float()
        raise _Stop

    att._concept_read = grab
    try:
        model.hidden_states(ids, doc_ids=doc)
    except _Stop:
        pass
    finally:
        del att._concept_read
    return F.normalize(box["k"], dim=-1)


def members_of(doc: torch.Tensor, sinks: int) -> torch.Tensor:
    """[S] bool: tokens that can be memory (their document's first `sinks` tokens and padding are not)."""
    from nn.exact_read import doc_starts

    ds = doc_starts(doc[None])[0]
    return (doc >= 0) & (torch.arange(doc.shape[0], device=doc.device) - ds >= sinks)


def bin_rows(data: str, src: str, S: int, eot: int, rows: int):
    from data.bin_token_dataset import BinRowDataset

    try:
        ds = BinRowDataset([{"name": src, "files": [str(Path(data) / f"{src}_val_*.bin")]}], S, "doc", eot=eot, seed=7)
    except ValueError:
        return
    for i in range(min(rows, len(ds))):
        it = ds[i]
        yield it["input_ids"], it["doc_ids"]


def retok_docs(cache: str, src: str) -> list[np.ndarray]:
    z = np.load(Path(cache) / f"{src}_val_retok.npz")
    return [z[k] for k in sorted(z.files, key=lambda s: int(s[1:]))]


def bisect(f, lo: float, hi: float, target: float, rising: bool, steps: int = 24) -> float:
    """x in [lo, hi] with f(x) = target for a monotone f."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        if (f(mid) < target) == rising:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------------------------------ P1 spans
def cmd_spans(a) -> None:
    geo = Geo()
    model, tok = load(a)
    dev = a.device
    rows = []
    for src in a.sources.split(","):
        if a.bin:
            it = bin_rows(a.bin, src, a.length, a.eot, a.rows)
        else:
            it = ((torch.from_numpy(d[:a.length].astype(np.int64)), torch.zeros(a.length, dtype=torch.long))
                  for d in [d for d in retok_docs(a.retok, src) if len(d) >= a.length][:a.rows])
        n = 0
        for ids, doc in it:
            ids, doc = ids.to(dev), doc.to(dev).long()
            khat = first_pass_keys(model, ids[None], doc[None])
            rows.append({"src": src, "ids": ids, "doc": doc, "khat": khat, "member": members_of(doc, geo.sinks)})
            n += 1
        log(f"{src}: {n} rows of {a.length}")
    if not rows:
        raise SystemExit("no rows")

    def lag_cos(sel, lag):
        tot, cnt = 0.0, 0
        for r in sel:
            k, m = r["khat"], r["member"]
            ok = m[lag:] & m[:-lag]
            tot += float(((k[lag:] * k[:-lag]).sum(-1) * ok).sum())
            cnt += int(ok.sum())
        return tot / max(cnt, 1)

    def mean_span(sel, theta):
        toks = sum(int(r["member"].sum()) for r in sel)
        spans = sum(cut_spans(r["khat"], r["member"], r["doc"], theta, geo)[1] for r in sel)
        return toks / max(spans, 1)

    lags = (1, 2, 4, 8, 16, 32, 64, 256)
    out = {"checkpoint": a.checkpoint, "length": a.length, "rows": len(rows), "geo": vars(geo),
           "lag_cosine": {str(l): lag_cos(rows, l) for l in lags}}
    adj = torch.cat([cut_spans(r["khat"], r["member"], r["doc"], 0.0, geo)[2]["cos"][r["member"]][1:] for r in rows])
    qs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=adj.device)
    sub = adj[torch.randperm(adj.numel(), device=adj.device)[:2_000_000]]
    out["adjacent_cosine"] = {"mean": float(adj.mean()), "std": float(adj.std()),
                              "quantiles": dict(zip([str(float(x)) for x in qs], torch.quantile(sub, qs).tolist()))}
    out["signal_lag1_minus_lag8"] = out["lag_cosine"]["1"] - out["lag_cosine"]["8"]
    theta = bisect(lambda th: mean_span(rows, th), -1.0, 1.0, a.target, rising=False)
    out["theta"] = theta
    out["mean_span_at_theta"] = mean_span(rows, theta)
    out["mean_span_no_cuts"] = mean_span(rows, -1.5)
    out["mean_span_all_cuts"] = mean_span(rows, 1.5)
    # the grouping's own numbers at θ
    hist = torch.zeros(geo.span_max + 1, dtype=torch.long)
    per_stride, capped_strides, at_cap, n_strides = [], 0, 0, 0
    word = {"cut": 0, "cut_word": 0, "inner": 0, "inner_word": 0, "cut_after_punct": 0, "inner_after_punct": 0}
    examples = {}
    for r in rows:
        sid, n_spans, info = cut_spans(r["khat"], r["member"], r["doc"], theta, geo)
        ln = torch.bincount(sid[sid >= 0], minlength=n_spans)
        hist += torch.bincount(ln, minlength=geo.span_max + 1)[: geo.span_max + 1].cpu()
        S = sid.shape[0]
        full = S // geo.stride
        first = torch.full((n_spans,), S, dtype=torch.long, device=sid.device).scatter_reduce_(
            0, sid[sid >= 0], torch.arange(S, device=sid.device)[sid >= 0], "amin")
        cnt = torch.bincount(first // geo.stride, minlength=full)[:full]
        per_stride.append(cnt.float().cpu())
        at_cap += int((cnt >= geo.span_cap).sum())
        capped_strides += int(info["capped"][: full * geo.stride].view(full, geo.stride).any(1).sum())
        n_strides += full
        toks = tok.convert_ids_to_tokens(r["ids"].tolist())
        ws = torch.tensor([t.startswith(("Ġ", "Ċ")) for t in toks], device=sid.device)
        pu = torch.tensor([False] + [t.endswith((".", ",", ";", ":", "?", "!", "Ċ", ")", '"')) for t in toks[:-1]],
                          device=sid.device)
        cut, inner = info["natural"], info["inner"]
        word["cut"] += int(cut.sum())
        word["cut_word"] += int((cut & ws).sum())
        word["inner"] += int(inner.sum())
        word["inner_word"] += int((inner & ws).sum())
        word["cut_after_punct"] += int((cut & pu).sum())
        word["inner_after_punct"] += int((inner & pu).sum())
        if examples.setdefault(r["src"], None) is None:   # one stretch of text per source, spans separated by ┃
            lo = min(4 * geo.stride, S - 160)
            seg, cur, parts = sid[lo:lo + 160].tolist(), None, []
            for j, s in enumerate(seg):
                if s != cur:
                    parts.append([])
                    cur = s
                parts[-1].append(int(r["ids"][lo + j]))
            examples[r["src"]] = " ┃ ".join(tok.decode(p).replace("\n", "⏎") for p in parts)
    tot = int(hist.sum())
    ps = torch.cat(per_stride)
    out["span_length_share"] = {str(i): int(hist[i]) / max(tot, 1) for i in range(1, geo.span_max + 1)}
    out["share_spans_len1"] = int(hist[1]) / max(tot, 1)
    out["share_spans_len_max"] = int(hist[geo.span_max]) / max(tot, 1)
    out["spans_per_stride"] = {"mean": float(ps.mean()), "p95": float(ps.quantile(0.95)), "max": float(ps.max())}
    out["share_strides_at_cap"] = at_cap / max(n_strides, 1)
    out["share_strides_cuts_capped"] = capped_strides / max(n_strides, 1)
    out["cuts_on_word_start"] = word["cut_word"] / max(word["cut"], 1)
    out["tokens_on_word_start"] = word["inner_word"] / max(word["inner"], 1)
    out["cuts_after_punctuation"] = word["cut_after_punct"] / max(word["cut"], 1)
    out["tokens_after_punctuation"] = word["inner_after_punct"] / max(word["inner"], 1)
    out["per_source"] = {}
    for src in dict.fromkeys(r["src"] for r in rows):
        sel = [r for r in rows if r["src"] == src]
        out["per_source"][src] = {
            "rows": len(sel), "lag1": lag_cos(sel, 1), "lag8": lag_cos(sel, 8),
            "mean_span_at_theta": mean_span(sel, theta),
            "theta_own": bisect(lambda th: mean_span(sel, th), -1.0, 1.0, a.target, rising=False)}
    out["examples"] = examples
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))


# ------------------------------------------------------------------------------------------ P3 concepts
def cmd_concepts(a) -> None:
    geo = Geo()
    model, _ = load(a)
    dev = a.device
    marks = [4096, 16384, 65536, 131072]
    docs = []
    for src in a.sources.split(","):
        all_docs = retok_docs(a.retok, src)
        pick: list[int] = []
        for need, k in ((131072, a.docs_128k), (65536, a.docs_64k), (16384, a.docs_16k)):
            pick += [j for j, d in enumerate(all_docs) if len(d) >= need][:k]
        for j in sorted(set(pick)):
            d = all_docs[j]
            L = min(len(d) // geo.stride * geo.stride, marks[-1])
            ids = torch.from_numpy(d[:L].astype(np.int64)).to(dev)
            doc = torch.zeros(L, dtype=torch.long, device=dev)
            khat = first_pass_keys(model, ids[None], doc[None])
            member = members_of(doc, geo.sinks)
            sid, n_spans, _ = cut_spans(khat, member, doc, a.theta, geo)
            docs.append({"src": src, "doc": j, "L": L, "khat": khat, "sid": sid, "n_spans": n_spans})
        log(f"{src}: {sum(1 for d in docs if d['src'] == src)} documents")

    def run(d, lam, upto=None):
        """Concepts of the document's first `upto` tokens (no slot limit: counts past N / 16 are flagged later)."""
        L = upto or d["L"]
        sid = d["sid"][:L]
        n_spans = int(sid.max()) + 1
        return leader_concepts(d["khat"][:L], sid, n_spans, lam, geo, c_max=None)

    def at_marks(d, lam):
        cof, nC, after, s_stride, first, n_tok = run(d, lam)
        res = {}
        for N in marks:
            if N > d["L"]:
                continue
            st = N // geo.stride
            sel = s_stride < st
            c = int(after[st - 1])
            sizes = torch.zeros(max(c, 1), device=dev).index_add_(0, cof[sel], torch.ones(int(sel.sum()), device=dev))
            toks = torch.zeros(max(c, 1), device=dev).index_add_(0, cof[sel], n_tok[sel])
            res[N] = {"concepts": c, "spans": int(sel.sum()), "tokens": float(n_tok[sel].sum()),
                      "single_span_share": float((sizes == 1).sum()) / max(c, 1),
                      "largest_token_share": float(toks.max() / n_tok[sel].sum().clamp_min(1)),
                      "over_slots": c > N // 16}
        return res

    def found_rate(lam, N):
        c = s = 0
        for d in docs:
            if d["L"] >= N:
                _, nC, _, s_stride, _, _ = run(d, lam, N)
                c += nC
                s += int(s_stride.numel())
        return c / max(s, 1)

    def table(lam):
        per = [dict(src=d["src"], doc=d["doc"], L=d["L"], marks=at_marks(d, lam)) for d in docs]
        agg = {}
        for N in marks:
            xs = [p["marks"][N] for p in per if N in p["marks"]]
            if not xs:
                continue
            agg[N] = {"docs": len(xs),
                      "concepts_mean": float(np.mean([x["concepts"] for x in xs])),
                      "spans_mean": float(np.mean([x["spans"] for x in xs])),
                      "found_rate": float(sum(x["concepts"] for x in xs) / max(sum(x["spans"] for x in xs), 1)),
                      "tokens_per_concept": float(sum(x["tokens"] for x in xs) / max(sum(x["concepts"] for x in xs), 1)),
                      "single_span_share": float(np.mean([x["single_span_share"] for x in xs])),
                      "largest_token_share": float(np.mean([x["largest_token_share"] for x in xs])),
                      "docs_over_slots": int(sum(x["over_slots"] for x in xs))}
        growth = {}
        for lo, hi in ((16384, 65536), (65536, 131072), (4096, 16384)):
            r = [p["marks"][hi]["concepts"] / max(p["marks"][lo]["concepts"], 1) for p in per if hi in p["marks"]]
            if r:
                growth[f"{lo}->{hi}"] = {"docs": len(r), "mean": float(np.mean(r)), "min": float(min(r)),
                                         "max": float(max(r)), "median": float(np.median(r))}
        return {"lam": lam, "by_length": agg, "growth": growth, "per_doc": per}

    out = {"checkpoint": a.checkpoint, "theta": a.theta, "geo": vars(geo),
           "docs": [{"src": d["src"], "doc": d["doc"], "L": d["L"], "spans": d["n_spans"]} for d in docs],
           "grid": {}, "calibrated": {}}
    for lam in (0.5, 0.6, 0.7, 0.8, 0.9):
        out["grid"][str(lam)] = table(lam)
        log(f"λ {lam}: " + json.dumps({N: (round(v['concepts_mean'], 1), round(v['found_rate'], 3))
                                       for N, v in out['grid'][str(lam)]['by_length'].items()}))
    for N in (16384, 65536):
        if any(d["L"] >= N for d in docs):
            lam = bisect(lambda x: found_rate(x, N), 0.0, 1.0, a.found, rising=True, steps=14)
            out["calibrated"][str(N)] = table(lam)
            log(f"λ for 1 span in {1 / a.found:.0f} founding at {N}: {lam:.4f}; growth "
                + json.dumps(out["calibrated"][str(N)]["growth"]))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"grid": {k: {"by_length": v["by_length"], "growth": v["growth"]} for k, v in out["grid"].items()},
                      "calibrated": {k: {"lam": v["lam"], "by_length": v["by_length"], "growth": v["growth"]}
                                     for k, v in out["calibrated"].items()}}, indent=1))


# ------------------------------------------------------------------------------------------ P2 RULER probe
def _bits(mask: torch.Tensor) -> int:
    return sum(1 << i for i, x in enumerate(mask.tolist()) if x)


@torch.no_grad()
def probe_prompt(model, buf, doc, L0: int, new: int, *, eos: int, values: list[torch.Tensor], theta: float,
                 lam: float, geo: Geo, check=None) -> dict:
    """One prompt (batch 1) through the cached decoding, every reading-layer call recorded.

    buf / doc [1, Lb] as the harness lays the row out (prompt in [0, L0), eos and document −1 after); values: the
    token positions of each expected answer value in the prompt. Returns the written tokens and, per reader (the
    last prompt position, then every answer position) and loop pass, for each value: which of its tokens the read
    holds word for word (bit i = token i; concept memory as trained, E39 two-level, E39 flat), which sit in the
    raw window, the attention on its tokens (best head, head mean), the same with the summaries taken out of the
    softmax, and the attention on the summaries of the concepts holding its unread tokens."""
    from nn.concept_memory import summary_log_scale, summary_logit
    from nn.exact_read import ORD_MOD, choose, doc_starts
    from nn.incremental import IncrementalDecoder

    gi = model.config.global_layer_index
    att = model.layers[gi].attn
    c = att.concept
    blk, cap = int(c["block"]), int(c["top_k"])
    dec = IncrementalDecoder(model)
    q0: list = []
    orig = att._concept_read

    def grab(q, k, v, x, *, doc_ids, key_valid):
        q0.append(q[:, L0 - 1:L0].detach().float().clone())
        return orig(q, k, v, x, doc_ids=doc_ids, key_valid=key_valid)

    att._concept_read = grab
    try:
        h = dec.prefill(buf, doc, L0)
    finally:
        del att._concept_read
    dev = buf.device
    S = dec.S
    pos = torch.arange(S, device=dev)
    d0 = dec.doc[0].clone()
    member = members_of(d0, geo.sinks)
    khat = F.normalize(dec.K[(gi, 0)][0, :, 0].float(), dim=-1)
    sid, n_spans, _ = cut_spans(khat, member, d0, theta, geo)
    cof, nC, after, s_stride, first, n_tok = leader_concepts(khat, sid, n_spans, lam, geo, c_max=max(1, S // 16))
    stats = [span_stats(dec.K[(gi, r)][0, :, 0].float(), sid, n_spans) for r in range(dec.passes)]
    recs: list[dict] = []

    def detail(q, r, t):
        d = q.shape[-1]
        n = t + 1
        jpos = pos[:n]
        docr, doc_t = dec.doc[:, :n], dec.doc[:, t]
        dsr = doc_starts(dec.doc)[:, :n]
        own_lo = (t // blk) * blk
        same = (docr == doc_t[:, None]) & (doc_t >= 0)[:, None]
        sink = ((jpos[None] - dsr) < int(c["sinks"])) & (docr >= 0) & same
        win = (jpos >= own_lo - blk)[None] & same
        qf = q[:, 0]
        lg = torch.einsum("bhd,bkd->bhk", qf, dec.K[(gi, r)][:, :n, 0].float()) / math.sqrt(d)
        opened = torch.zeros_like(win)
        lg_s, cid_tok = None, None
        built = dec.cc[r]
        if built is not None:
            cidx = built[0]
            g_ = cidx.groups
            G = g_.n_groups
            score = summary_logit(q, cidx, variance=bool(c["variance"])).amax(2)
            visible = (g_.g_close[:, None, :] <= t) & torch.isfinite(score)
            ids, off = choose(torch.nan_to_num(score, neginf=-1e30), visible, g_, torch.tensor([t], device=dev),
                              m=int(c["width"]), cap=cap, raw_window=blk)
            row = torch.full((1, G), cap, dtype=torch.int32, device=dev)
            row.scatter_(1, ids[:, 0], off[:, 0].to(torch.int32))
            codes = g_.codes[0, :, :n].long()
            g = torch.div(codes, ORD_MOD, rounding_mode="floor")
            hit = (codes >= 0) & (row.gather(1, g.clamp(0, G - 1)) + (codes - g * ORD_MOD) < cap)
            opened = hit & ((t - jpos) >= blk)[None] & same & ~win & ~sink
            cid_tok = torch.where(codes >= 0, g, torch.full_like(g, -1))[0]
            C = cidx.n_concepts
            vis = ((t >= (cidx.block_of + 2) * blk)[None] & (cidx.c_doc == doc_t[:, None]) & (doc_t >= 0)[:, None]
                   & ~(row[:, :C] < cap))
            logn = torch.where(cidx.n > 0, cidx.n.clamp_min(1.0).log(), torch.zeros_like(cidx.n))
            vhalf = 0.5 * cidx.var if c["variance"] else torch.zeros_like(cidx.var)
            sb = logn[:, None, :].float() + ((qf ** 2).sum(-1) / d)[..., None] * vhalf[:, None, :].float()
            lr = summary_log_scale(dsr, n, blk, int(c.get("sum_len", 0) or 0))
            if lr is not None:
                sb = sb + lr[:, t][:, None, None]
            lg_s = torch.einsum("bhd,bcd->bhc", qf, cidx.mu_k.float()) / math.sqrt(d) + sb
            lg_s = lg_s.masked_fill(~vis[:, None, :], NEG)
        raw = win | sink | opened
        lg_r = lg.masked_fill(~raw[:, None, :], NEG)
        p_ns = torch.softmax(lg_r, -1)[0]                                               # [h, n]
        use_sum = lg_s is not None and bool(c["summaries"])
        p = torch.softmax(torch.cat([lg_r, lg_s], -1), -1)[0] if use_sum else p_ns      # [h, n (+ C)]
        tau = t // geo.stride
        vis_sp = s_stride <= tau - 2
        win39 = pos >= (tau - 1) * geo.stride
        e2 = e39_open(qf[0], vis_sp, cof, nC, stats[r], sid, first, geo, True) | win39
        ef = e39_open(qf[0], vis_sp, cof, nC, stats[r], sid, first, geo, False) | win39
        rec = {"t": t, "r": r, "raw": [], "win": [], "e2": [], "ef": [], "mass_max": [], "mass_mean": [],
               "mass_ns_max": [], "sum_mass": [], "groups": [],
               "src": [float((p[:, :n] * m[0]).sum(-1).mean()) for m in (win, sink & ~win, opened)]
               + [float(p[:, n:].sum(-1).mean()) if use_sum else 0.0],
               "n_open": int(opened.sum()), "n_sum": int(vis.sum()) if lg_s is not None else 0}
        for A in values:
            A = A[A < n]
            if A.numel() == 0:
                for k_ in ("raw", "win", "e2", "ef", "groups"):
                    rec[k_].append(0)
                for k_ in ("mass_max", "mass_mean", "mass_ns_max", "sum_mass"):
                    rec[k_].append(0.0)
                continue
            rd = raw[0, A]
            rec["raw"].append(_bits(rd))
            rec["win"].append(_bits(win[0, A]))
            rec["e2"].append(_bits(e2[A]))
            rec["ef"].append(_bits(ef[A]))
            ma = p[:, A].sum(-1)
            rec["mass_max"].append(float(ma.max()))
            rec["mass_mean"].append(float(ma.mean()))
            rec["mass_ns_max"].append(float(p_ns[:, A].sum(-1).max()))
            sm, grp = 0.0, 0
            if cid_tok is not None:
                ct = cid_tok[A]
                grp = int(ct[ct >= 0].unique().numel())
                un = ct[(~rd) & (ct >= 0)].unique()
                if use_sum and un.numel():
                    sm = float(p[:, n + un].sum(-1).max())
            rec["sum_mass"].append(sm)
            rec["groups"].append(grp)
        return rec

    for r in range(dec.passes):
        recs.append(detail(q0[r], r, L0 - 1))
    real = dec._concept_read

    def traced(q, K, V, r, t, att_):
        recs.append(detail(q.float(), r, t))
        return real(q, K, V, r, t, att_)

    dec._concept_read = traced
    t_end = dec._decode(h, new, eos, check)
    spans_of = [int(sid[A[A < S]][sid[A[A < S]] >= 0].unique().numel()) if A.numel() else 0 for A in values]
    return {"tokens": dec.buf[0, L0:t_end].tolist(), "recs": recs, "passes": dec.passes,
            "e39": {"spans": n_spans, "concepts": nC, "tokens": int(member.sum()), "value_spans": spans_of}}


@torch.no_grad()
def probe_prompt_lsa(model, buf, doc, L0: int, new: int, *, eos: int, values: list[torch.Tensor], check=None) -> dict:
    """The same record as `probe_prompt` for a model with the topic-index read (our port of LSA, `par_global_read:
    lsa`): per reader and loop pass, which tokens of each answer value the read holds word for word (its sliding
    512-token window, its sinks, the tokens of the buckets it chose) and the attention on them. It has no summaries
    and nothing is simulated, so the E39 columns repeat the model's own and the summary columns are zero."""
    from nn.exact_read import ORD_MOD, doc_starts
    from nn.incremental import IncrementalDecoder
    from nn.lsa_buckets import lsa_open_table

    gi = model.config.global_layer_index
    att = model.layers[gi].attn
    c = att.lsa
    rw = int(c["window"])
    dec = IncrementalDecoder(model)
    q0: list = []
    orig = att._lsa_read

    def grab(q, k, v, *, doc_ids, key_valid):
        q0.append(q[:, L0 - 1:L0].detach().float().clone())
        return orig(q, k, v, doc_ids=doc_ids, key_valid=key_valid)

    att._lsa_read = grab
    try:
        h = dec.prefill(buf, doc, L0)
    finally:
        del att._lsa_read
    dev = buf.device
    pos = torch.arange(dec.S, device=dev)
    recs: list[dict] = []

    def detail(q, r, t):
        d = q.shape[-1]
        n = t + 1
        jpos = pos[:n]
        docr, doc_t = dec.doc[:, :n], dec.doc[:, t]
        dsr = doc_starts(dec.doc)[:, :n]
        same = (docr == doc_t[:, None]) & (doc_t >= 0)[:, None]
        win = ((t - jpos) < rw)[None] & same
        sink = ((jpos[None] - dsr) < int(c["sinks"])) & (docr >= 0) & same & ~win
        opened = torch.zeros_like(win)
        bucket = None
        built = dec.lsa_groups[r]
        if built is not None:
            centroids, gidx, members = built
            ch = lsa_open_table(q.to(dec.K[(gi, r)].dtype), centroids, gidx, top_k=c["top_k"], width=c["width"],
                                raw_window=rw, impl="list", q_offset=t, members=members)
            p_, ok_ = ch.opened(t, t + 1, rw)
            ok_ = ok_[0, 0] & (p_[0, 0] < n)
            opened[0, p_[0, 0][ok_]] = True
            opened = opened & same & ~win & ~sink
            codes = gidx.codes[0, 0, :n].long()
            bucket = torch.where(codes >= 0, torch.div(codes, ORD_MOD, rounding_mode="floor"), torch.full_like(codes, -1))
        raw = win | sink | opened
        lg = torch.einsum("bhd,bkd->bhk", q[:, 0], dec.K[(gi, r)][:, :n, 0].float()) / math.sqrt(d)
        p = torch.softmax(lg.masked_fill(~raw[:, None, :], NEG), -1)[0]
        rec = {"t": t, "r": r, "raw": [], "win": [], "e2": [], "ef": [], "mass_max": [], "mass_mean": [],
               "mass_ns_max": [], "sum_mass": [], "groups": [],
               "src": [float((p * m[0]).sum(-1).mean()) for m in (win, sink, opened)] + [0.0],
               "n_open": int(opened.sum()), "n_sum": 0}
        for A in values:
            A = A[A < n]
            if A.numel() == 0:
                for k_ in ("raw", "win", "e2", "ef", "groups"):
                    rec[k_].append(0)
                for k_ in ("mass_max", "mass_mean", "mass_ns_max", "sum_mass"):
                    rec[k_].append(0.0)
                continue
            b = _bits(raw[0, A])
            for k_ in ("raw", "e2", "ef"):
                rec[k_].append(b)
            rec["win"].append(_bits(win[0, A]))
            ma = p[:, A].sum(-1)
            rec["mass_max"].append(float(ma.max()))
            rec["mass_mean"].append(float(ma.mean()))
            rec["mass_ns_max"].append(float(ma.max()))
            rec["sum_mass"].append(0.0)
            rec["groups"].append(int(bucket[A][bucket[A] >= 0].unique().numel()) if bucket is not None else 0)
        return rec

    for r in range(dec.passes):
        recs.append(detail(q0[r], r, L0 - 1))
    real = dec._lsa_read

    def traced(q, K, V, r, t, att_):
        recs.append(detail(q.float(), r, t))
        return real(q, K, V, r, t, att_)

    dec._lsa_read = traced
    t_end = dec._decode(h, new, eos, check)
    return {"tokens": dec.buf[0, L0:t_end].tolist(), "recs": recs, "passes": dec.passes,
            "e39": {"spans": 1, "concepts": 1, "tokens": 0, "value_spans": [0 for _ in values]}}


def cmd_ruler(a) -> None:
    import lm_eval
    from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs, postprocess_generated_text
    from lm_eval.tasks import TaskManager

    from evaluation.lm_eval_perceiver_ar import PerceiverARLMEval
    from nn.incremental import IncrementalDecoder

    geo = Geo()
    L = int(a.length)
    records: list[dict] = []
    raw_path = Path(str(a.out) + ".pt")

    class Probe(PerceiverARLMEval):
        def generate_until(self, requests, disable_tqdm: bool = False):
            res = []
            att = self.model.layers[self.model.config.global_layer_index].attn
            eos_s = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            eos = self.tokenizer.eos_token_id
            t0 = time.time()
            for i, req in enumerate(requests):
                ctx, gk = req.args
                kwargs = normalize_gen_kwargs(gk, self.max_gen_toks)
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos_s)
                new = int(kwargs.pop("max_gen_toks"))
                enc, _ = self.tok_batch_encode([ctx], left_truncate_len=self.max_length - new,
                                               truncation=self.truncation)
                enc = enc.to(self.device)
                L0 = enc.shape[1]
                ids = enc[0].tolist()
                pieces = [self.tokenizer.decode([x]) for x in ids]
                offs = np.cumsum([0] + [len(p) for p in pieces])
                text = "".join(pieces)
                outs = [str(o) for o in req.doc["outputs"]]
                values = []
                for ref in outs:
                    toks, at = set(), text.find(ref)
                    while at >= 0:
                        lo = int(np.searchsorted(offs, at, side="right")) - 1
                        hi = int(np.searchsorted(offs, at + len(ref), side="left"))
                        toks.update(range(lo, hi))
                        at = text.find(ref, at + 1)
                    values.append(torch.tensor(sorted(toks), dtype=torch.long, device=self.device))
                Lb = -(-(L0 + new) // self.GEN_PAD) * self.GEN_PAD
                buf = torch.full((1, Lb), eos, dtype=enc.dtype, device=self.device)
                buf[:, :L0] = enc
                doc = torch.full((1, Lb), -1, dtype=torch.long, device=self.device)
                doc[:, :L0] = 0
                stops = [s for s in until if s]

                def check(out, t, done):
                    if stops and not done[0]:
                        done[0] = any(s in self.tokenizer.decode(out[0, L0:t].tolist()) for s in stops)
                    return done

                if att.lsa is not None:   # the topic-index model: the same record, nothing simulated, no summaries
                    pr = probe_prompt_lsa(self.model, buf, doc, L0, new, eos=eos, values=values, check=check)
                    reply = reply2 = postprocess_generated_text(generation=self.tok_decode(pr["tokens"]), stop=until,
                                                                think_end_token=None)
                else:
                    pr = probe_prompt(self.model, buf, doc, L0, new, eos=eos, values=values, theta=a.theta,
                                      lam=a.lam, geo=geo, check=check)
                    reply = postprocess_generated_text(generation=self.tok_decode(pr["tokens"]), stop=until,
                                                       think_end_token=None)
                    att.concept["summaries"] = False   # the same prompt with the summaries out of the read
                    try:
                        out2 = IncrementalDecoder(self.model).greedy(buf.clone(), doc.clone(), L0, new, eos=eos,
                                                                     check=check)
                    finally:
                        att.concept["summaries"] = True
                    reply2 = postprocess_generated_text(generation=self.tok_decode(out2[0, L0:].tolist()),
                                                        stop=until, think_end_token=None)
                records.append({"task": req.task_name, "doc_id": req.doc_id, "length": L, "L0": L0,
                                "outputs": outs, "values": [v.tolist() for v in values], "reply": reply,
                                "reply_nosum": reply2, "found": [o.lower() in reply.lower() for o in outs],
                                "found_nosum": [o.lower() in reply2.lower() for o in outs], **pr})
                res.append(reply)
                if (i + 1) % 10 == 0 or i + 1 == len(requests):
                    torch.save(records, raw_path)
                    sc = np.mean([np.mean(r["found"]) for r in records])
                    sc2 = np.mean([np.mean(r["found_nosum"]) for r in records])
                    log(f"{i + 1}/{len(requests)} prompts, {time.time() - t0:.0f}s; found {sc:.3f}, "
                        f"summaries off {sc2:.3f}")
            return res

    lm = Probe(pretrained=a.checkpoint, tokenizer=a.checkpoint, attn_backend=a.backend, max_length=L + 2 * 128,
               batch_size=1, device=a.device)
    tasks = [t for t in a.tasks.split(",") if t]
    ev = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=a.limit if a.limit > 0 else None, log_samples=False,
                                 task_manager=TaskManager(metadata={"tokenizer": a.checkpoint, "max_seq_lengths": [L]}),
                                 bootstrap_iters=0)
    torch.save(records, raw_path)
    summ = summarize(records, lm.tokenizer)
    summ["lm_eval"] = {t: v.get(f"{L},none") for t, v in ev["results"].items() if isinstance(v, dict)}
    summ["args"] = {"theta": a.theta, "lam": a.lam, "length": L, "limit": a.limit, "checkpoint": a.checkpoint}
    Path(str(a.out) + ".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def _pc(x: int) -> int:
    return bin(int(x)).count("1")


def _by_pass(sel: list[dict], P: int) -> dict:
    """Far values at the last prompt position, loop pass by loop pass: the share whose tokens are all held word for
    word (concept memory as trained; E39 two-level; E39 flat), and the attention on the value's tokens (best head),
    with the summaries in the softmax and without."""
    if not sel:
        return {}
    out = {}
    for r in range(P):
        out[f"pass_{r + 1}"] = {
            "cm_full": float(np.mean([x["p_raw"][r] >= 0.999 for x in sel])),
            "e2_full": float(np.mean([x["p_e2"][r] >= 0.999 for x in sel])),
            "ef_full": float(np.mean([x["p_ef"][r] >= 0.999 for x in sel])),
            "mass_mean": float(np.mean([x["p_mass"][r] for x in sel])),
            "mass_median": float(np.median([x["p_mass"][r] for x in sel])),
            "mass_no_summaries_mean": float(np.mean([x["p_mass_ns"][r] for x in sel]))}
    for key in ("p_raw", "p_e2", "p_ef"):
        n_full = [sum(c >= 0.999 for c in x[key]) for x in sel]
        out[f"{key[2:]}_passes_full_hist"] = {str(k): float(np.mean([n == k for n in n_full])) for k in range(P + 1)}
    out["mass_best_pass_mean"] = float(np.mean([max(x["p_mass"]) for x in sel]))
    out["mass_best_pass_median"] = float(np.median([max(x["p_mass"]) for x in sel]))
    return out


def _emit(sel: list[dict], P: int) -> dict:
    """Values the model wrote, at the positions that wrote their digits: the reader of digit j, loop pass by loop
    pass: is digit j's source token held word for word, and the attention on the value's tokens (best head)."""
    sel = [x for x in sel if x.get("emit")]
    if not sel:
        return {"values": 0}
    out = {"values": len(sel)}
    for r in range(P):
        out[f"pass_{r + 1}"] = {
            "digit_token_held": float(np.mean([np.mean([e["held"][r] for e in x["emit"]]) for x in sel])),
            "mass_mean": float(np.mean([np.mean([e["mass"][r] for e in x["emit"]]) for x in sel]))}
    best = [np.mean([max(e["mass"]) for e in x["emit"]]) for x in sel]
    first = [max(x["emit"][0]["mass"]) for x in sel]
    out["mass_best_pass_mean"] = float(np.mean(best))
    out["first_digit_mass_best_pass_mean"] = float(np.mean(first))
    out["first_digit_mass_best_pass_median"] = float(np.median(first))
    out["first_digit_held_any_pass"] = float(np.mean([any(x["emit"][0]["held"]) for x in sel]))
    out["passes_holding_digit_mean"] = float(np.mean([np.mean([sum(e["held"]) for e in x["emit"]]) for x in sel]))
    return out


def summarize(records: list[dict], tok=None) -> dict:
    """Per task: scores, and for each answer value the coverage and attention numbers, far values split by found /
    missed (far = no token in the raw window of the first reader). With a tokenizer, also the positions that wrote
    each found value's digits (`_emit`)."""
    out = {}
    for task in dict.fromkeys(r["task"] for r in records):
        rs = [r for r in records if r["task"] == task]
        vals = []
        split = {"window": [], "sinks": [], "opened": [], "summaries": []}
        n_open, n_sum = [], []
        for r in rs:
            P = r["passes"]
            recs = r["recs"]
            readers = [recs[i:i + P] for i in range(0, len(recs) - len(recs) % P, P)]
            starts = {}
            if tok is not None:  # where each found value starts in the reply, in tokens
                pieces = [tok.decode([x]) for x in r["tokens"]]
                offs = np.cumsum([0] + [len(p) for p in pieces])
                text = "".join(pieces)
                for v, ref in enumerate(r["outputs"]):
                    at = text.find(ref)
                    if at >= 0:
                        starts[v] = int(np.searchsorted(offs, at, side="right")) - 1
            for j, name in enumerate(split):
                split[name].append(float(np.mean([x["src"][j] for x in readers[0]])))
            n_open.append(float(np.mean([x["n_open"] for x in readers[0]])))
            n_sum.append(float(np.mean([x["n_sum"] for x in readers[0]])))
            for v, A in enumerate(r["values"]):
                nt = len(A)
                if nt == 0:
                    continue
                full = (1 << nt) - 1

                def cov(key, rd, any_pass=True):
                    if any_pass:
                        b = 0
                        for x in rd:
                            b |= x[key][v]
                        return _pc(b) / nt
                    return max(_pc(x[key][v]) for x in rd) / nt

                first = readers[0]
                emit = []
                if v in starts:  # the reader of the value's j-th token: position L0 − 1 + (its place in the reply)
                    for j in range(nt):
                        i = starts[v] + j
                        if i < len(readers):
                            emit.append({"held": [bool(x["raw"][v] >> j & 1) for x in readers[i]],
                                         "mass": [x["mass_max"][v] for x in readers[i]]})
                vals.append({
                    "emit": emit,
                    "p_raw": [_pc(x["raw"][v]) / nt for x in first], "p_e2": [_pc(x["e2"][v]) / nt for x in first],
                    "p_ef": [_pc(x["ef"][v]) / nt for x in first], "p_mass": [x["mass_max"][v] for x in first],
                    "p_mass_ns": [x["mass_ns_max"][v] for x in first],
                    "found": bool(r["found"][v]), "found_nosum": bool(r["found_nosum"][v]), "n_tok": nt,
                    "far": all(x["win"][v] == 0 for x in first),
                    "groups": float(np.mean([x["groups"][v] for x in first])),
                    "e39_spans": r["e39"]["value_spans"][v],
                    "cm_first_any": cov("raw", first), "cm_first_pass": cov("raw", first, False),
                    "cm_peak_any": max(cov("raw", rd) for rd in readers),
                    "cm_peak_pass": max(cov("raw", rd, False) for rd in readers),
                    "e2_first_any": cov("e2", first), "e2_first_pass": cov("e2", first, False),
                    "e2_peak_pass": max(cov("e2", rd, False) for rd in readers),
                    "ef_first_any": cov("ef", first), "ef_first_pass": cov("ef", first, False),
                    "ef_peak_pass": max(cov("ef", rd, False) for rd in readers),
                    "mass_peak": max(x["mass_max"][v] for rd in readers for x in rd),
                    "mass_first": max(x["mass_max"][v] for x in first),
                    "mass_ns_peak": max(x["mass_ns_max"][v] for rd in readers for x in rd),
                    "sum_mass_peak": max(x["sum_mass"][v] for rd in readers for x in rd),
                    "full": full})

        def block(sel):
            if not sel:
                return {"values": 0}
            keys = [k for k in sel[0] if k not in ("found", "found_nosum", "far", "full", "emit") and k[:2] != "p_"]
            b = {"values": len(sel), "found_with_summaries_off": float(np.mean([x["found_nosum"] for x in sel]))}
            for k in keys:
                b[k] = float(np.mean([x[k] for x in sel]))
            for k in ("cm_first_pass", "cm_peak_pass", "e2_first_pass", "e2_peak_pass", "ef_first_pass", "ef_peak_pass"):
                b[k + "_share_full"] = float(np.mean([x[k] >= 0.999 for x in sel]))
                b[k + "_share_none"] = float(np.mean([x[k] <= 0.001 for x in sel]))
            for k in ("mass_peak", "mass_ns_peak", "sum_mass_peak"):
                b[k + "_median"] = float(np.median([x[k] for x in sel]))
            return b

        far = [x for x in vals if x["far"]]
        missed = [x for x in far if not x["found"]]
        out[task] = {
            "prompts": len(rs),
            "score": float(np.mean([np.mean(r["found"]) for r in rs])),
            "score_summaries_off": float(np.mean([np.mean(r["found_nosum"]) for r in rs])),
            "values": len(vals), "far_values": len(far),
            "first_reader_attention": {k: float(np.mean(v)) for k, v in split.items()},
            "first_reader_tokens_opened": float(np.mean(n_open)), "first_reader_summaries": float(np.mean(n_sum)),
            "e39_rows": {"tokens_per_span": float(np.mean([r["e39"]["tokens"] / max(r["e39"]["spans"], 1) for r in rs])),
                         "spans_per_concept": float(np.mean([r["e39"]["spans"] / max(r["e39"]["concepts"], 1) for r in rs])),
                         "concepts": float(np.mean([r["e39"]["concepts"] for r in rs]))},
            "far_found": block([x for x in far if x["found"]]),
            "far_missed": block(missed),
            "first_reader_by_pass_far_missed": _by_pass(missed, rs[0]["passes"]),
            "first_reader_by_pass_far_found": _by_pass([x for x in far if x["found"]], rs[0]["passes"]),
            "writing_positions_far_found": _emit([x for x in far if x["found"]], rs[0]["passes"]),
            "writing_positions_window_found": _emit([x for x in vals if x["found"] and not x["far"]], rs[0]["passes"]),
            # the three outcomes of the spec, on the far values the model missed
            "missed_never_fully_opened": float(np.mean([x["cm_peak_pass"] < 0.999 for x in missed])) if missed else None,
            "missed_fully_opened_in_some_pass": float(np.mean([x["cm_peak_pass"] >= 0.999 for x in missed])) if missed else None,
        }
    return out


def cmd_summary(a) -> None:
    tok = None
    if a.checkpoint:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.checkpoint)
    records = []
    for f in a.raw.split(","):
        records += torch.load(f, weights_only=True)   # plain lists and numbers
    if a.first > 0:  # the first prompts of each task only (a run stopped early)
        seen, keep = {}, []
        for r in records:
            seen[r["task"]] = seen.get(r["task"], 0) + 1
            if seen[r["task"]] <= a.first:
                keep.append(r)
        records = keep
    s = summarize(records, tok)
    if a.out:
        Path(a.out).write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)

    def common(p):
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--backend", default="flex")
        p.add_argument("--out", required=True)

    p = sub.add_parser("spans")
    common(p)
    p.add_argument("--bin", default="", help="a data mix folder with <source>_val_*.bin (rows of one document)")
    p.add_argument("--retok", default="", help="or a folder of <source>_val_retok.npz (evaluation/long_context_use.py)")
    p.add_argument("--sources", required=True)
    p.add_argument("--eot", type=int, default=0)
    p.add_argument("--length", type=int, default=16384)
    p.add_argument("--rows", type=int, default=16, help="rows per source")
    p.add_argument("--target", type=float, default=8.0, help="mean span length θ is set for")
    p = sub.add_parser("concepts")
    common(p)
    p.add_argument("--retok", required=True)
    p.add_argument("--sources", default="gutenberg,arxiv,doab,usgpo,hansard")
    p.add_argument("--theta", type=float, required=True)
    p.add_argument("--found", type=float, default=0.125, help="share of spans that found a concept (calibration)")
    p.add_argument("--docs_16k", type=int, default=16)
    p.add_argument("--docs_64k", type=int, default=16)
    p.add_argument("--docs_128k", type=int, default=8)
    p = sub.add_parser("ruler")
    common(p)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--tasks", default="niah_multivalue,niah_multiquery")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--theta", type=float, required=True)
    p.add_argument("--lam", type=float, required=True)
    p = sub.add_parser("summary")
    p.add_argument("--raw", required=True, help="one or more saved `ruler` record files (.pt), comma-separated")
    p.add_argument("--checkpoint", default="", help="for its tokenizer: adds the positions that wrote each found value")
    p.add_argument("--first", type=int, default=0, help="only the first N prompts of each task")
    p.add_argument("--out", default="")
    a = ap.parse_args()
    {"spans": cmd_spans, "concepts": cmd_concepts, "ruler": cmd_ruler, "summary": cmd_summary}[a.what](a)


if __name__ == "__main__":
    main()
