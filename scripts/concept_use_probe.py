"""Root-cause probe (2026-11-23): why do the 150M models read several far values and write one of them again?

RULER's "all values of one key" and "several keys asked at once" need four far values written one after another.
The saved replies (docs/release/hf_card_concept_memory_150m/ruler/*_answers.jsonl) show every model of the family
writing one value and then looping; the E39 pre-flight (scripts/e39_preflight.py ruler) showed that the missed values
are opened and attended. This probe separates the remaining causes by handing the model its answer so far and
looking at the one decision that matters, the next value:

  force    Per prompt (through lm-eval, cached decoding, one prompt at a time), with the prompt read once:
           * free answers as the board scores them, with switches on the far read: as trained (`base`), summaries
             out (`nosum`), sharper weights (`temp1.5`, `temp2`: the match part of every logit times 1.5 / 2), and
             `ior`: once the answer holds a value, that value's tokens leave the far read (an oracle "do not read
             again what you have copied");
           * the next value after j written ones, j = 0 … 3. The answer so far is forced: ":" then the j values and
             ", " — the model's own picks while they are new targets, otherwise the next target in document order.
             Conditions: `base`; `hide_written` (the written values' tokens leave the far read at this step);
             `only_new` (the far read holds only the unwritten values and the 16 tokens before each, no summaries);
             `nosum`; `temp2`; `far_off` (window and sinks only: what the local text alone would write).
             Recorded: the 7-digit number written (a new target / a repeat / something else), the probability of
             each value's first digit, and at the deciding position the far read's weights on each value (per loop
             pass: best head, head mean, on the first digit and on all digits) and its sources (window, sinks, opened
             tokens, summaries).
           Works for the concept-memory, full-attention and topic-index models; the switches other than
           `hide_written` exist for the concept read only.
  weights  The numbers of a checkpoint that set how sharp and how loud each attention is: the query and key norm
           gains (the largest match a query can reach), each layer's residual weights (alpha, beta, sigma), the
           loop's round offsets, the sizes of the read's projections.
  summary  Re-print the tables of saved `force` runs.

Usage (a home GPU, from the checkout; RULER needs wonderwords):
  uv run --no-sync --with wonderwords python scripts/concept_use_probe.py force --checkpoint CK --length 8192 \
      --tasks niah_multivalue,niah_multiquery --limit 50 --out Cache/use_probe/cm_8k
  .venv/bin/python scripts/concept_use_probe.py weights --checkpoints cm=CK1,fa=CK2
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nn.incremental import IncrementalDecoder  # noqa: E402

NEG = float("-inf")
FREE = ("base", "nosum", "temp1.5", "temp2", "ior")
FORCED = ("base", "hide_written", "only_new", "nosum", "temp2", "far_off")


def log(*a) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]", *a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------ the decoder with switches
class ProbeDecoder(IncrementalDecoder):
    """The cached decoder with switches on the far read of the answer positions (`kn`) and a recorder (`rec`).

    kn: hide [S] bool (keys the far read may not use), only [S] bool (the only far keys it may use; summaries
    out), nosum, far_off, temp (the match part of every logit × temp), rec_t (record only this position)."""

    _SHARED = IncrementalDecoder._SHARED + ("kn", "rec", "vals")

    def __init__(self, model):
        super().__init__(model)
        self.kn: dict = {}
        self.rec = None
        self.vals: list = []

    def _record(self, r, t, p, n, win, sink, opened):
        """p [h, n (+ C)] the read's weights of position t on pass r; the masks [n] bool."""
        if self.rec is None or (self.kn.get("rec_t") is not None and t != self.kn["rec_t"]):
            return
        raw = p[:, :n]
        top_p, top_i = raw.max(-1)
        rec = {"r": r, "t": t,
               "src": [float((raw * m).sum(-1).mean()) for m in (win, sink & ~win, opened)]
               + [float(p[:, n:].sum(-1).mean())],
               "top_i": top_i.tolist(), "top_p": [round(float(x), 4) for x in top_p],
               "all_max": [], "all_mean": [], "first_max": [], "first_mean": [], "first_head": [], "readable": []}
        for A in self.vals:
            A = A[A < n]
            if A.numel() == 0:
                for k_ in ("all_max", "all_mean", "first_max", "first_mean"):
                    rec[k_].append(0.0)
                rec["first_head"].append(-1)
                rec["readable"].append(False)
                continue
            ma, mf = raw[:, A].sum(-1), raw[:, A[0]]
            rec["all_max"].append(float(ma.max()))
            rec["all_mean"].append(float(ma.mean()))
            rec["first_max"].append(float(mf.max()))
            rec["first_mean"].append(float(mf.mean()))
            rec["first_head"].append(int(mf.argmax()))
            rec["readable"].append(bool((win | sink | opened)[A[0]]))
        for name, P in (self.kn.get("marks") or {}).items():  # named sets of positions: weight on them, share readable
            P = P[P < n]
            if P.numel():
                mm = raw[:, P].sum(-1)
                rec["mark_" + name] = {"max": float(mm.max()), "mean": float(mm.mean()),
                                       "readable": float((win | sink | opened)[P].float().mean())}
        self.rec.append(rec)

    def _concept_read(self, q, K, V, r, t, att):
        """`IncrementalDecoder._concept_read` with the switches; with no switch it computes the same read."""
        from nn.concept_memory import summary_log_scale, summary_logit
        from nn.exact_read import ORD_MOD, choose

        kn = self.kn
        c = att.concept
        blk, cap = int(c["block"]), int(c["top_k"])
        B, _, h, d = q.shape
        n = t + 1
        jpos = torch.arange(n, device=q.device)
        doc, doc_t, ds = self.doc[:, :n], self.doc[:, t], self._ds
        own_lo = (t // blk) * blk
        sink = ((jpos[None] - ds[:, :n]) < int(c["sinks"])) & (doc >= 0)
        win = (jpos >= own_lo - blk)[None].expand(B, n)
        keep = win | sink
        opened = torch.zeros_like(keep)
        vt = V[:, :n]
        ks, vs, sbias = None, None, None
        built = self.cc[r]
        if built is not None and not kn.get("far_off"):
            cidx, dv, dv_tok = built
            gi = cidx.groups
            G = gi.n_groups
            score = summary_logit(q, cidx, variance=bool(c["variance"])).amax(2)
            visible = (gi.g_close[:, None, :] <= t) & torch.isfinite(score)
            ids, off = choose(torch.nan_to_num(score, neginf=-1e30), visible, gi,
                              torch.tensor([t], device=q.device), m=int(c["width"]), cap=cap, raw_window=blk)
            row = torch.full((B, G), cap, dtype=torch.int32, device=q.device)
            row.scatter_(1, ids[:, 0], off[:, 0].to(torch.int32))
            codes = gi.codes[0, :, :n].long()
            g = torch.div(codes, ORD_MOD, rounding_mode="floor")
            hit = (codes >= 0) & (row.gather(1, g.clamp(0, G - 1)) + (codes - g * ORD_MOD) < cap)
            opened = hit & ((t - jpos) >= blk)[None] & ~keep
            keep = keep | opened
            if dv_tok is not None:
                vt = torch.where((jpos < own_lo)[None, :, None, None], vt + dv_tok[:, :n].to(vt.dtype), vt)
            if c["summaries"] and not kn.get("nosum") and kn.get("only") is None:
                C = cidx.n_concepts
                vis = ((t >= (cidx.block_of + 2) * blk)[None] & (cidx.c_doc == doc_t[:, None])
                       & (doc_t >= 0)[:, None] & ~(row[:, :C] < cap))
                logn = torch.where(cidx.n > 0, cidx.n.clamp_min(1.0).log(), torch.zeros_like(cidx.n))
                vhalf = 0.5 * cidx.var if c["variance"] else torch.zeros_like(cidx.var)
                qn2 = (q.float() ** 2).sum(-1)[:, 0] / d
                sb = logn[:, None, :].float() + qn2[..., None] * vhalf[:, None, :].float()
                lr = summary_log_scale(ds[:, :n], n, blk, int(c.get("sum_len", 0) or 0))
                if lr is not None:
                    sb = sb + lr[:, t][:, None, None]
                sbias = sb.masked_fill(~vis[:, None, :], NEG)
                ks = cidx.mu_k[:, :, None].to(K.dtype)
                vs = (cidx.mu_v if dv is None else cidx.mu_v + dv.to(cidx.mu_v.dtype))[:, :, None].to(V.dtype)
        same = (doc == doc_t[:, None]) & (doc_t >= 0)[:, None]
        far = ~(win | sink)
        if kn.get("only") is not None:  # the far read holds these keys and no others (opened by the choice or not)
            only = kn["only"][None, :n]
            keep = (keep & ~far) | (only & far)
            opened = only & far
        if kn.get("also") is not None:  # these keys are readable whatever the choice (their summaries stay as well)
            also = kn["also"][None, :n] & far
            keep, opened = keep | also, opened | also
        if kn.get("dense"):  # every earlier token of the document word for word, no summaries: full attention
            keep, opened, sbias = torch.ones_like(keep), far, None
        if kn.get("hide") is not None:
            keep = keep & ~(kn["hide"][None, :n] & far)
        mask = keep & same
        temp = float(kn.get("temp", 1.0))
        kk, vv = K[:, :n, 0].float(), vt[:, :, 0].float()
        lg = torch.einsum("bhd,bkd->bhk", q[:, 0].float(), kk) / math.sqrt(d) * temp
        lg = lg + torch.zeros(B, n, device=q.device).masked_fill(~mask, NEG)[:, None]
        if sbias is not None:
            lg = torch.cat([lg, torch.einsum("bhd,bkd->bhk", q[:, 0].float(), ks[:, :, 0].float())
                            / math.sqrt(d) * temp + sbias], -1)
            vv = torch.cat([vv, vs[:, :, 0].float()], 1)
        p = torch.softmax(lg, -1)
        self._record(r, t, p[0], n, (win & same)[0], (sink & same)[0], (opened & mask)[0])
        return torch.einsum("bhk,bkd->bhd", p, vv)[:, None].to(q.dtype)

    def _window_read(self, q, K, V, t, w):
        """Full attention's reading layer (w = t + 1) with `hide` and the recorder; local layers unchanged."""
        if self.kind != "dense" or w != t + 1 or (self.kn.get("hide") is None and self.rec is None):
            return super()._window_read(q, K, V, t, w)
        n = t + 1
        d = q.shape[-1]
        mask = self.doc[:, :n] == self.doc[:, t:t + 1]
        if self.kn.get("hide") is not None:
            mask = mask & ~self.kn["hide"][None, :n]
        lg = torch.einsum("bhd,bkd->bhk", q[:, 0].float(), K[:, :n, 0].float()) / math.sqrt(d)
        p = torch.softmax(lg + torch.zeros_like(lg[:, 0]).masked_fill(~mask, NEG)[:, None], -1)
        z = torch.zeros(n, dtype=torch.bool, device=q.device)
        self._record(0, t, p[0], n, z, z, mask[0])
        return torch.einsum("bhk,bkd->bhd", p, V[:, :n, 0].float())[:, None].to(q.dtype)

    def _lsa_read(self, q, K, V, r, t, att):
        """The topic-index read with `hide` (the hidden keys are not readable, chosen or not)."""
        if self.kn.get("hide") is None:
            return super()._lsa_read(q, K, V, r, t, att)
        from nn.lsa_buckets import lsa_open_table
        from nn.perceiver_ar_lm import attend_list

        c = att.lsa
        opened = None
        built = self.lsa_groups[r]
        if not att.lsa_off and built is not None:
            centroids, gidx, members = built
            opened = lsa_open_table(q, centroids, gidx, top_k=c["top_k"], width=c["width"], raw_window=c["window"],
                                    impl="list", q_offset=t, members=members)
        kv = (~self.kn["hide"])[None].expand(self.B, -1)
        return attend_list(q, K, V, opened=opened, raw_window=c["window"], tag=self.doc, key_valid=kv,
                           ds=self._ds, n_sinks=c["sinks"], valid_query=True, q_offset=t)


@torch.no_grad()
def _next_logits(model, h):
    logits = F.linear(h, model.lm_head.weight).float()
    cap = model.config.logit_softcap
    return cap * torch.tanh(logits / cap) if cap else logits


@torch.no_grad()
def free_answer(dec: ProbeDecoder, h, new: int, eos: int, tok, L0: int, stops, ior=None) -> str:
    """Greedy answer from the current state (as `IncrementalDecoder._decode`), stop strings as the harness. `ior`:
    (reference strings, their token positions) — once the answer holds a reference, its tokens leave the far read."""
    out = dec.buf
    t = int(dec.t)
    hidden: set = set()
    for it in range(int(new)):
        if it:
            h = dec.step(out[:, t - 1])
        nxt = _next_logits(dec.m, h).argmax(-1)
        out[:, t] = nxt
        t += 1
        text = tok.decode(out[0, L0:t].tolist())
        if int(nxt[0]) == eos or any(s in text for s in stops):
            break
        if ior is not None:
            refs, poss = ior
            for i, ref in enumerate(refs):
                if i not in hidden and ref in text:
                    hidden.add(i)
                    hide = dec.kn.get("hide")
                    if hide is None:
                        hide = torch.zeros(dec.S, dtype=torch.bool, device=out.device)
                    hide = hide.clone()
                    hide[poss[i]] = True
                    dec.kn["hide"] = hide
    return tok.decode(out[0, L0:t].tolist())


START = ("base", "nosum", "far_off", "open_needles", "dense", "only_needles")


@torch.no_grad()
def start_probe(model, tok, dec: ProbeDecoder, snap0: dict, L0: int, values: list) -> dict:
    """The first token of the answer (the reader is the prompt's last token): the probability of ":" and the three
    likeliest tokens, with the far read as trained and with switches: summaries out, the far read off, the needle
    sentences always readable (`open_needles`: each value and the 16 tokens before it), every earlier token readable
    word for word and no summaries (`dense`: full attention with these weights), only the needle sentences far
    (`only_needles`). Recorded per loop pass: the read's weight on the ":" before each value, on the values' first
    digits and on the needle sentences, and how much of each the read could see word for word."""
    S, dev = dec.S, dec.buf.device
    colon = tok(":", add_special_tokens=False)["input_ids"][0]
    need = torch.zeros(S, dtype=torch.bool, device=dev)
    for A in values:
        if A.numel():
            need[max(0, int(A[0]) - 16):int(A[-1]) + 1] = True
    has = [A for A in values if A.numel() and int(A[0]) >= 2]
    marks = {"needle": need.nonzero().flatten()}
    if has:
        marks["colon"] = torch.stack([A[0] - 2 for A in has])
        marks["digit1"] = torch.stack([A[0] for A in has])
    out = {"colon_tokens": [tok.decode([int(dec.buf[0, int(A[0]) - 2])]) for A in has]}
    for cond in START:
        if dec.kind != "concept" and cond != "base":
            continue
        dec.restore(snap0)
        kn: dict = {"rec_t": L0 - 1, "marks": marks}
        if cond == "nosum":
            kn["nosum"] = True
        elif cond == "far_off":
            kn["far_off"] = True
        elif cond == "open_needles":
            kn["also"] = need
        elif cond == "dense":
            kn["dense"] = True
        elif cond == "only_needles":
            kn["only"] = need
        dec.kn, dec.rec = kn, []
        h = dec._position(L0 - 1)
        pr = torch.softmax(_next_logits(model, h), -1)[0]
        top = pr.topk(3)
        out[cond] = {"p_colon": float(pr[colon]), "top": [(tok.decode([int(i)]), round(float(p), 3))
                                                           for p, i in zip(top.values, top.indices)],
                     "trace": dec.rec}
        dec.rec = None
    return out


def summarize_start(records: list[dict]) -> dict:
    out: dict = {}
    for task in sorted({r["task"] for r in records}):
        rs = [r for r in records if r["task"] == task and "start" in r]
        if not rs:
            continue
        t: dict = {"prompts": len(rs), "kind": rs[0].get("kind")}
        for cond in START:
            xs = [r["start"][cond] for r in rs if cond in r["start"]]
            if not xs:
                continue
            c = {"p_colon": round(float(np.mean([x["p_colon"] for x in xs])), 3),
                 "colon_first": round(float(np.mean([x["top"][0][0] == ":" for x in xs])), 3),
                 "top": collections_top([x["top"][0][0] for x in xs])}
            last = [x["trace"][-1] for x in xs if x.get("trace")]
            first = [x["trace"][0] for x in xs if x.get("trace")]
            for name in ("colon", "digit1", "needle"):
                k = "mark_" + name
                if last and all(k in e for e in last):
                    c[name] = {"weight_best_head_last_pass": round(float(np.mean([e[k]["max"] for e in last])), 4),
                               "weight_head_mean_last_pass": round(float(np.mean([e[k]["mean"] for e in last])), 4),
                               "weight_best_head_first_pass": round(float(np.mean([e[k]["max"] for e in first])), 4),
                               "readable": round(float(np.mean([e[k]["readable"] for e in last])), 3)}
            if last:
                c["src_window_sinks_opened_summaries"] = [round(float(z), 3) for z in np.mean([e["src"] for e in last], 0)]
            t[cond] = c
        out[task] = t
    return out


def collections_top(xs: list, n: int = 4) -> list:
    import collections

    return collections.Counter(xs).most_common(n)


def knobs(cond: str, S: int, dev, values, written, t_rec=None) -> dict:
    kn: dict = {"rec_t": t_rec}
    if cond == "nosum":
        kn["nosum"] = True
    elif cond.startswith("temp"):
        kn["temp"] = float(cond[4:])
    elif cond == "far_off":
        kn["far_off"] = True
    elif cond == "hide_written":
        hide = torch.zeros(S, dtype=torch.bool, device=dev)
        for i in written:
            hide[values[i]] = True
        kn["hide"] = hide
    elif cond == "only_new":
        only = torch.zeros(S, dtype=torch.bool, device=dev)
        for i, A in enumerate(values):
            if i not in written and A.numel():
                only[max(0, int(A[0]) - 16):int(A[-1]) + 1] = True
        kn["only"] = only
    return kn


# ------------------------------------------------------------------------------------------ force
def cmd_force(a) -> None:
    import lm_eval
    from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs, postprocess_generated_text
    from lm_eval.tasks import TaskManager

    from evaluation.lm_eval_perceiver_ar import PerceiverARLMEval

    L = int(a.length)
    records: list[dict] = []
    raw_path = Path(str(a.out) + ".pt")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    free_conds = [c for c in a.free.split(",") if c]
    forced_conds = [c for c in a.forced.split(",") if c]

    class Probe(PerceiverARLMEval):
        def generate_until(self, requests, disable_tqdm: bool = False):
            res = []
            tok = self.tokenizer
            eos_s = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            eos = tok.eos_token_id
            t_start = time.time()
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
                pieces = [tok.decode([x]) for x in ids]
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
                nv = len(outs)
                order = sorted(range(nv), key=lambda v_: int(values[v_][0]) if values[v_].numel() else 1 << 40)
                dec = ProbeDecoder(self.model)
                dec.vals = values
                h0 = dec.prefill(buf, doc, L0)
                S = dec.S
                snap0 = dec.snapshot()
                rec = {"task": req.task_name, "doc_id": req.doc_id, "length": L, "L0": L0, "outputs": outs,
                       "values": [v.tolist() for v in values], "order": order, "free": {}, "forced": [],
                       "kind": dec.kind}
                if a.mode == "start":  # the first token of the answer only
                    rec["start"] = start_probe(self.model, tok, dec, snap0, L0, values)
                    records.append(rec)
                    res.append("")
                    if (i + 1) % 10 == 0 or i + 1 == len(requests):
                        torch.save(records, raw_path)
                        log(f"{i + 1}/{len(requests)} prompts, {time.time() - t_start:.0f}s")
                    continue
                # free answers
                for cond in free_conds:
                    if dec.kind != "concept" and cond not in ("base", "ior"):
                        continue
                    dec.restore(snap0)
                    dec.rec = None
                    dec.kn = knobs(cond, S, self.device, values, [])
                    h = h0 if cond in ("base", "ior") else dec._position(L0 - 1)
                    reply = free_answer(dec, h, min(new, a.new), eos, tok, L0, stops,
                                        ior=(outs, values) if cond == "ior" else None)
                    reply = postprocess_generated_text(generation=reply, stop=until, think_end_token=None)
                    rec["free"][cond] = {"reply": reply, "found": [o in reply for o in outs]}
                # the next value after j written ones
                digit = [tok(o[0], add_special_tokens=False)["input_ids"][0] for o in outs]
                ten = [tok(str(x), add_special_tokens=False)["input_ids"][0] for x in range(10)]
                written: list[int] = []
                for j in range(nv):
                    prefix = ":" + "".join(f" {outs[v_]}," for v_ in written) + " "
                    ptoks = torch.tensor(tok(prefix, add_special_tokens=False)["input_ids"], device=self.device)
                    if i == 0 and j == 1:
                        log("forced prefix tokens:", [tok.decode([x]) for x in ptoks.tolist()])
                    dec.restore(snap0)
                    dec.kn, dec.rec = {}, None
                    if ptoks.numel() > 1:
                        dec.feed(ptoks[:-1])
                    snap_j = dec.snapshot()
                    t_dec = int(dec.t)
                    row = {"j": j, "written": list(written), "res": {}}
                    for cond in forced_conds:
                        if j == 0 and cond == "hide_written":
                            continue
                        if dec.kind != "concept" and cond not in ("base", "hide_written"):
                            continue
                        dec.restore(snap_j)
                        dec.kn = knobs(cond, S, self.device, values, written, t_rec=t_dec)
                        dec.rec = [] if cond in ("base", "hide_written") else None
                        h = dec.feed(ptoks[-1:])
                        pr = torch.softmax(_next_logits(self.model, h), -1)[0]
                        p_first = [float(pr[dg]) for dg in digit]
                        p_digit = float(pr[ten].sum())
                        trace = dec.rec
                        dec.rec = None
                        t0 = int(dec.t)
                        t1 = dec._decode(h, 9, eos)
                        cont = tok.decode(dec.buf[0, t0:t1].tolist())
                        m = re.match(r"\s*(\d+)", cont)
                        num = m.group(1) if m else ""
                        new_t = [outs[v_] for v_ in range(nv) if v_ not in written]
                        cls = ("new" if num in new_t else "repeat" if num in [outs[v_] for v_ in written]
                               else "other")
                        row["res"][cond] = {"num": num, "cls": cls, "cont": cont, "p_first": p_first,
                                            "p_digit": p_digit, "trace": trace}
                    rec["forced"].append(row)
                    base = row["res"]["base"]
                    nxt = outs.index(base["num"]) if base["cls"] == "new" else next(
                        v_ for v_ in order if v_ not in written)
                    row["own_pick"] = base["cls"] == "new"
                    written.append(nxt)
                records.append(rec)
                res.append(rec["free"].get("base", {}).get("reply", ""))
                if (i + 1) % 10 == 0 or i + 1 == len(requests):
                    torch.save(records, raw_path)
                    log(f"{i + 1}/{len(requests)} prompts, {time.time() - t_start:.0f}s")
            return res

    lm = Probe(pretrained=a.checkpoint, tokenizer=a.checkpoint, attn_backend=a.backend, max_length=L + 2 * 128,
               batch_size=1, device=a.device)
    if a.sum_len >= 0:  # the summaries' total weight held at the level of a document of this length (0 = off)
        from nn.concept_memory import set_concept_sum_len

        log(f"length scale of the summaries set to {a.sum_len} on {set_concept_sum_len(lm.model, a.sum_len)} layer(s)")
    tasks = [t for t in a.tasks.split(",") if t]
    lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=a.limit if a.limit > 0 else None, log_samples=False,
                            task_manager=TaskManager(metadata={"tokenizer": a.checkpoint, "max_seq_lengths": [L]}),
                            bootstrap_iters=0)
    torch.save(records, raw_path)
    summ = summarize_start(records) if a.mode == "start" else summarize(records)
    summ["args"] = {"length": L, "limit": a.limit, "checkpoint": a.checkpoint, "mode": a.mode}
    Path(str(a.out) + ".json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def summarize(records: list[dict]) -> dict:
    """Per task: the free answers' score per switch; per j the share of new / repeat / other per condition; the
    probability of the first digit of the written against the unwritten values; the far read's weight on the first
    digit of the value the model then wrote, of the other unwritten values and of the written ones (base)."""
    out: dict = {}
    for task in sorted({r["task"] for r in records}):
        rs = [r for r in records if r["task"] == task]
        t: dict = {"prompts": len(rs), "kind": rs[0].get("kind"), "free": {}, "forced": {}, "read": {}}
        for cond in rs[0]["free"]:
            f = [np.mean(r["free"][cond]["found"]) for r in rs]
            hist = np.bincount([int(sum(r["free"][cond]["found"])) for r in rs], minlength=5).tolist()
            t["free"][cond] = {"score": round(float(np.mean(f)), 3), "found_hist": hist}
        nv = max(len(r["forced"]) for r in rs)
        for j in range(nv):
            rows = [r["forced"][j] for r in rs if len(r["forced"]) > j]
            tj: dict = {"own_pick_before": round(float(np.mean([x.get("own_pick", False) for x in
                                                                [r["forced"][j - 1] for r in rs
                                                                 if len(r["forced"]) > j]])) if j else 1.0, 3)}
            for cond in rows[0]["res"]:
                xs = [x["res"][cond] for x in rows if cond in x["res"]]
                tj[cond] = {k: round(float(np.mean([x["cls"] == k for x in xs])), 3) for k in ("new", "repeat", "other")}
                tj[cond]["p_digit"] = round(float(np.mean([x["p_digit"] for x in xs])), 3)
            # first-digit probability: written against unwritten values (base)
            pw, pn = [], []
            for x in rows:
                pf = x["res"]["base"]["p_first"]
                pw += [pf[v] for v in x["written"]]
                pn += [pf[v] for v in range(len(pf)) if v not in x["written"]]
            tj["p_first_written"] = round(float(np.mean(pw)), 4) if pw else None
            tj["p_first_unwritten"] = round(float(np.mean(pn)), 4) if pn else None
            t["forced"][str(j)] = tj
            # the far read at the deciding position (base): weight on first digits, by loop pass
            rd: dict = {}
            for x, r in zip(rows, [r for r in rs if len(r["forced"]) > j]):
                tr = x["res"]["base"].get("trace") or []
                outs = r["outputs"]
                num = x["res"]["base"]["num"]
                for e in tr:
                    d = rd.setdefault(e["r"], {"written": [], "unwritten": [], "chosen": [], "src": [],
                                               "readable_unwritten": []})
                    for v in range(len(outs)):
                        key = "written" if v in x["written"] else "unwritten"
                        d[key].append(e["first_max"][v])
                        if key == "unwritten":
                            d["readable_unwritten"].append(e["readable"][v])
                        if outs[v] == num:
                            d["chosen"].append(e["first_max"][v])
                    d["src"].append(e["src"])
            t["read"][str(j)] = {str(rr): {"first_max_written": round(float(np.mean(d["written"])), 4) if d["written"] else None,
                                           "first_max_unwritten": round(float(np.mean(d["unwritten"])), 4) if d["unwritten"] else None,
                                           "first_max_chosen": round(float(np.mean(d["chosen"])), 4) if d["chosen"] else None,
                                           "readable_unwritten": round(float(np.mean(d["readable_unwritten"])), 3) if d["readable_unwritten"] else None,
                                           "src_window_sinks_opened_summaries": [round(float(z), 3) for z in np.mean(d["src"], 0)]}
                                 for rr, d in sorted(rd.items())}
        out[task] = t
    return out


# ------------------------------------------------------------------------------------------ rows exam
_MULTI_MARKS = ("in this order", "in the order asked", "in that order", "One after another", "keeping this order")


def question_kind(question: str, answer: str) -> str:
    """single / list (all the values of one thing) / multi (several facts in one question), from the wording of
    scripts/build_reasoning_rows.py's list sets."""
    if any(m in question for m in _MULTI_MARKS):
        return "multi"
    return "list" if ", " in answer else "single"


@torch.no_grad()
def cmd_rows(a) -> None:
    """The questions of held-out lookup / list rows (scripts/build_reasoning_rows.py val shards), answered one at a
    time: the row is read once up to the first `Answer:`, the model writes its answer (greedy, to the end of the
    line), then the row's own answer is put in its place and the next question follows. Per kind of question: exact
    answers, the share of the answer's values found, answers with every value, answers that write a value twice."""
    from transformers import AutoTokenizer

    from data.bin_token_dataset import open_shard
    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, attn_backend=a.backend)
    dev = next(model.parameters()).device
    S = int(a.seq_len)
    rows = np.asarray(open_shard(a.shard)).reshape(-1, S)[: a.limit if a.limit > 0 else None]
    eos = tok.eos_token_id
    out_rows, t_start = [], time.time()
    for ri, row in enumerate(rows):
        pad = np.flatnonzero(row[1:] == eos)
        n = int(pad[0]) + 1 if pad.size else S
        ids = row[:n].astype(np.int64)
        pieces = [tok.decode([int(x)]) for x in ids[1:]]
        offs = np.cumsum([0] + [len(p) for p in pieces])           # character offset of token 1 + i
        text = "".join(pieces)
        qas = []
        for m in re.finditer(r"\n\nQuestion: (.*)\nAnswer:( .*)", text):
            a0 = int(np.searchsorted(offs, m.start(2), side="left")) + 1     # first token of the answer
            a1 = int(np.searchsorted(offs, m.end(2), side="left")) + 1       # first token after it
            qas.append((m.group(1), m.group(2).strip(), a0, a1))
        if not qas:
            continue
        Lb = -(-(n + 64) // 256) * 256
        gold = torch.from_numpy(ids).to(dev)
        L0 = qas[0][2]
        buf = torch.full((1, Lb), eos, dtype=torch.long, device=dev)   # as the harness: nothing after the prompt
        buf[0, :L0] = gold[:L0]
        doc = torch.full((1, Lb), -1, dtype=torch.long, device=dev)
        doc[0, :L0] = 0
        dec = ProbeDecoder(model)
        h = dec.prefill(buf, doc, L0)
        res = []
        for qi, (question, answer, a0, a1) in enumerate(qas):
            if qi:
                h = dec.feed(gold[int(dec.t):a0])
            snap = dec.snapshot()
            t0 = int(dec.t)
            new = (a1 - a0) + 12
            t = t0
            for it in range(new):
                if it:
                    h = dec.step(dec.buf[:, t - 1])
                nxt = _next_logits(model, h).argmax(-1)
                dec.buf[:, t] = nxt
                t += 1
                if "\n" in tok.decode(dec.buf[0, t0:t].tolist()) or int(nxt[0]) == eos:
                    break
            pred = tok.decode(dec.buf[0, t0:t].tolist()).split("\n")[0].strip()
            dec.restore(snap)
            kind = question_kind(question, answer)
            vals = answer.split(", ") if kind != "single" else [answer]
            written = [w.strip() for w in pred.split(", ")]
            res.append({"kind": kind, "question": question, "answer": answer, "pred": pred, "exact": pred == answer,
                        "found": [v in pred for v in vals], "repeat": len(written) != len(set(written)),
                        "answer_pos": a0})
        out_rows.append(res)
        if (ri + 1) % 8 == 0 or ri + 1 == len(rows):
            log(f"{ri + 1}/{len(rows)} rows, {time.time() - t_start:.0f}s")
    summ: dict = {"checkpoint": a.checkpoint, "shard": a.shard, "rows": len(out_rows)}
    for kind in ("single", "list", "multi"):
        xs = [q for r in out_rows for q in r if q["kind"] == kind]
        if xs:
            summ[kind] = {"questions": len(xs), "exact": round(float(np.mean([q["exact"] for q in xs])), 3),
                          "values_found": round(float(np.mean([np.mean(q["found"]) for q in xs])), 3),
                          "all_values": round(float(np.mean([all(q["found"]) for q in xs])), 3),
                          "writes_a_value_twice": round(float(np.mean([q["repeat"] for q in xs])), 3)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": out_rows}, indent=1))
    print(json.dumps(summ, indent=1))


def cmd_table(a) -> None:
    """The summaries of several `force` runs side by side: one line per run and task for the free answers, one per
    run, task and number of written values for the next value."""
    runs = []
    for p in a.json.split(","):
        # a .pt file: the raw records of a run (also of one still running: saved every 10 prompts)
        d = summarize(torch.load(p, weights_only=True)) if p.endswith(".pt") else json.loads(Path(p).read_text())
        runs.append((Path(p).stem, {k: v for k, v in d.items() if k != "args"}))
    print("FREE ANSWERS: share of the four values found (targets found 0..4 for `base`)")
    for name, d in runs:
        for task, t in d.items():
            fr = t["free"]
            print(f"  {name:10} {task:16} " + "  ".join(f"{c} {v['score']:.2f}" for c, v in fr.items())
                  + f"   base hist {fr['base']['found_hist']}")
    print("NEXT VALUE after j written ones: share new target / repeat / other, per condition")
    for name, d in runs:
        for task, t in d.items():
            for j, tj in t["forced"].items():
                conds = [c for c in tj if isinstance(tj[c], dict)]
                line = "  ".join(f"{c} {tj[c]['new']:.2f}/{tj[c]['repeat']:.2f}/{tj[c]['other']:.2f}" for c in conds)
                pf = (f"  p(first digit) written {tj['p_first_written']} unwritten {tj['p_first_unwritten']}"
                      if tj.get("p_first_written") is not None else f"  p(first digit) {tj['p_first_unwritten']}")
                print(f"  {name:10} {task:16} j={j} own pick before {tj['own_pick_before']:.2f}  {line}{pf}")
    print("FAR READ at the deciding position (base), last loop pass: best head's weight on the first digit of the "
          "written / unwritten / chosen values; sources window, sinks, opened, summaries")
    for name, d in runs:
        for task, t in d.items():
            for j, rj in t["read"].items():
                if not rj:
                    continue
                last = rj[max(rj, key=int)]
                first = rj[min(rj, key=int)]
                print(f"  {name:10} {task:16} j={j} last pass: written {last['first_max_written']} unwritten "
                      f"{last['first_max_unwritten']} chosen {last['first_max_chosen']} readable "
                      f"{last['readable_unwritten']} src {last['src_window_sinks_opened_summaries']} | first pass: "
                      f"written {first['first_max_written']} unwritten {first['first_max_unwritten']}")


def cmd_examples(a) -> None:
    """What the model wrote at each forced step of the first prompts of a saved `force` run (per task)."""
    recs = torch.load(a.raw, weights_only=True)
    for task in sorted({r["task"] for r in recs}):
        print("==", task)
        for r in [x for x in recs if x["task"] == task][: a.n]:
            print(f"  targets {r['outputs']}  free: {r['free']['base']['reply'][:90]!r}")
            for row in r["forced"]:
                w = [r["outputs"][v] for v in row["written"]]
                line = "  ".join(f"{c}: {x['num'] or x['cont'][:12]!r} ({x['cls']})" for c, x in row["res"].items()
                                 if c in a.conds.split(","))
                print(f"    j={row['j']} written {w} -> {line}")


def cmd_summary(a) -> None:
    for p in a.raw.split(","):
        recs = torch.load(p, weights_only=True)
        print(p)
        print(json.dumps(summarize(recs), indent=1))


# ------------------------------------------------------------------------------------------ weights
def cmd_weights(a) -> None:
    from safetensors.torch import load_file

    for item in a.checkpoints.split(","):
        name, path = item.split("=", 1)
        sd = load_file(str(Path(path) / "model.safetensors"))
        cfg = json.loads((Path(path) / "config.json").read_text())
        print(f"== {name}: {path}")
        print("   read:", cfg.get("par_global_read"), "| nope:", cfg.get("global_nope", cfg.get("par_global_nope")),
              "| heads:", cfg.get("num_attention_heads"), "kv heads:", cfg.get("num_kv_heads"), "head dim:",
              cfg.get("head_dim"), "| logit scale:", cfg.get("global_logit_scale"))
        layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.")})
        print("   layer  alpha   beta  sigma | q gain mean/max  k gain mean/max  max match | |wq|   |wk|   |wv|   |wo|  "
              "|mlp.down|")
        for i in layers:
            g = lambda k: sd.get(f"layers.{i}.{k}")  # noqa: E731
            qg, kg = g("attn.q_norm.weight").float(), g("attn.k_norm.weight").float()
            d = qg.numel()
            # RMSNorm outputs have mean square gain²: the largest q·k/√d puts all of both on the best dimension
            reach = float((qg.abs() * kg.abs()).max()) * math.sqrt(d)
            typical = float((qg.abs() * kg.abs()).mean()) * math.sqrt(d)
            sig = g("sigma")
            print(f"   {i:5d} {float(g('alpha')):6.3f} {float(g('beta')):6.3f} "
                  f"{(f'{float(sig):6.3f}' if sig is not None else '     -')} | "
                  f"{float(qg.abs().mean()):5.2f}/{float(qg.abs().max()):5.2f}     "
                  f"{float(kg.abs().mean()):5.2f}/{float(kg.abs().max()):5.2f}     {typical:5.1f}/{reach:5.1f} | "
                  + " ".join(f"{float(g(f'attn.{w}.weight').float().norm()):6.1f}" for w in ("wq", "wk", "wv", "wo"))
                  + f" {float(g('mlp.down.weight').float().norm()):6.1f}")
        for k in ("loop_emb", "loop_gate"):
            if k in sd:
                v = sd[k].float()
                print(f"   {k}: norms {[round(float(x), 3) for x in (v.norm(dim=-1) if v.dim() > 1 else v.reshape(-1))]}")
        extra = [k for k in sd if not k.startswith(("layers.", "embed.", "lm_head.")) and sd[k].numel() <= 16]
        for k in extra:
            print(f"   {k}: {[round(float(x), 4) for x in sd[k].float().reshape(-1)]}")
        small = [k for k in sd if k.startswith("layers.") and sd[k].numel() <= 16
                 and not k.endswith((".alpha", ".beta", ".sigma"))]
        for k in small:
            print(f"   {k}: {[round(float(x), 4) for x in sd[k].float().reshape(-1)]}")


def cmd_diff(a) -> None:
    """Where two checkpoints of the same model differ (a fine-tune against its start): the relative change
    |after − before| / |before| of every weight matrix, per layer and part, and the scalars that moved."""
    from safetensors.torch import load_file

    w0 = load_file(str(Path(a.before) / "model.safetensors"))
    w1 = load_file(str(Path(a.after) / "model.safetensors"))
    rows = []
    for k, v0 in w0.items():
        if k not in w1 or w1[k].shape != v0.shape:
            continue
        v0f, v1f = v0.float(), w1[k].float()
        if v0f.numel() <= 16:
            if float((v1f - v0f).abs().max()) > 1e-4:
                rows.append((k, None, [round(float(x), 4) for x in v0f.reshape(-1)[:4]],
                             [round(float(x), 4) for x in v1f.reshape(-1)[:4]]))
            continue
        rows.append((k, float((v1f - v0f).norm() / v0f.norm().clamp_min(1e-12)), None, None))
    print("relative change of each layer's parts (|after - before| / |before|, per cent)")
    parts = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.q_norm", "attn.k_norm", "mlp.gate", "mlp.up", "mlp.down")
    print("   layer " + " ".join(f"{p.split('.')[-1]:>7}" for p in parts))
    layers = sorted({int(k.split(".")[1]) for k, *_ in rows if k.startswith("layers.")})
    rel = {k: r for k, r, *_ in rows if r is not None}
    for i in layers:
        print(f"   {i:5d} " + " ".join(f"{100 * rel.get(f'layers.{i}.{p}.weight', float('nan')):7.2f}" for p in parts))
    other = sorted(((r, k) for k, r in rel.items() if not k.startswith("layers.")), reverse=True)
    print("   other:", ", ".join(f"{k} {100 * r:.2f}" for r, k in other[:8]))
    print("scalars that moved (before -> after):")
    for k, r, b, c in rows:
        if r is None:
            print(f"   {k}: {b} -> {c}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("force")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--tasks", default="niah_multivalue,niah_multiquery")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--mode", default="force", choices=("force", "start"),
                   help="start: only the first token of the answer (does the model begin a list?), with switches")
    p.add_argument("--sum_len", type=int, default=-1,
                   help="concept read: hold the summaries' total weight at the level of a document of this many "
                        "tokens (the model's own setting when not given; 0 = no length scale)")
    p.add_argument("--new", type=int, default=64, help="tokens of a free answer (four values need 36)")
    p.add_argument("--free", default=",".join(FREE))
    p.add_argument("--forced", default=",".join(FORCED))
    p.add_argument("--backend", default="flex")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_force)
    p = sub.add_parser("summary")
    p.add_argument("--raw", required=True, help="comma-separated .pt files of force runs")
    p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("diff")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.set_defaults(fn=cmd_diff)
    p = sub.add_parser("examples")
    p.add_argument("--raw", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--conds", default="base,hide_written")
    p.set_defaults(fn=cmd_examples)
    p = sub.add_parser("rows")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shard", required=True, help="a val shard of scripts/build_reasoning_rows.py (lookup or list set)")
    p.add_argument("--seq_len", type=int, default=16384)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--backend", default="flex")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_rows)
    p = sub.add_parser("table")
    p.add_argument("--json", required=True, help="comma-separated .json files of force runs")
    p.set_defaults(fn=cmd_table)
    p = sub.add_parser("weights")
    p.add_argument("--checkpoints", required=True, help="name=path,name=path")
    p.set_defaults(fn=cmd_weights)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
