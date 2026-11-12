"""Cached greedy decoding for the Perceiver AR family (`nn/perceiver_ar_lm.py`): the row is read once, then each step
runs only the new position through the layers (2026-11-12).

Why. The harness decoding (`evaluation/lm_eval_perceiver_ar.PerceiverARLMEval._model_generate`) re-runs the whole row
for every answer token. E33t stage D measured 5.2 s per forward at 64k for the catalogue model on one L40S and 32
forwards per RULER prompt (180 s a prompt, ~16 h for the 13 tasks at 25 prompts each). Here the row is read once (the
prefill: the model's own forward over the same fixed-length row), while every attention call's keys and values are
kept; a step then costs one position through every layer, reading the cached keys.

Exactness. The row geometry is the uncached decoding's: one row of `Lb` tokens for the whole answer, the slots after
the current position written as eos with document id −1. A position's output depends only on earlier positions
(tests/test_note_open.py::test_no_future_leak, tests/test_lsa_read.py::test_lsa_no_future_leak), so the prefill's keys
of the prompt are the uncached decoding's at every step, and each step writes its own position's keys. What the
reading layer derives from many keys is rebuilt when it could change for the current query:

* LSA's read (E33p): the buckets of a closed 512-key block are visible to later blocks. When the decoding reaches a
  block end and that block holds answer tokens, every pass's buckets are rebuilt from the cache (`lsa_groups`).
* The catalogue read (E33t): the notes only choose what opens, and a note opens only members older than the raw
  window. A page holding answer tokens starts after L0 − lm_window, so it has no member that old while the answer is
  shorter than raw_window − lm_window (256 tokens for E33t n4): the prefill's notes are exact for such answers, and a
  longer request is refused up front.

The reading layer reads through the list read (`attend_list`, `ChoiceList`), which gives the table read's outputs
(tests/test_catalogue_read.py::test_list_read_matches_table, tests/test_lsa_read.py::test_lsa_list_read_matches_table).
The tokens equal the uncached decoding's and the logits agree to float rounding (tests/test_incremental_decode.py).

Supported; anything else raises NotImplementedError before any work, so callers keep the uncached path: perceiver
par_mode, causal, no message boundary, one global layer, the tied loop over [global read + span layers] with loop
markers (and the prelude gate), sliding-window local layers without sinks; the reading layer is full attention,
LSA's read, or the catalogue read (index notes on non-overlapping pages, no pass exclusion, one level, no note-age or
count heads). Not yet: E33p's notes read with gist (notes_open), E37's concept read, write-back hooks, loop control,
loop feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from nn.exact_read import ChoiceList, doc_starts, note_open_table, tied_note_keys
from nn.lsa_buckets import lsa_groups, lsa_open_table
from nn.perceiver_ar_lm import PerceiverARLM, apply_rope, attend_list, rope_cos_sin


def _refuse(why: str):
    raise NotImplementedError(f"cached decoding: {why}")


def decode_kind(model: PerceiverARLM) -> str:
    """'dense' | 'lsa' | 'catalogue': the reading layer the cached decoding reproduces; NotImplementedError otherwise."""
    cfg, m = model.config, model
    if cfg.par_mode != "perceiver":
        _refuse("par_mode must be 'perceiver'")
    if cfg.block_attention_mode != "causal":
        _refuse("causal blocks only")
    if int(cfg.message_boundary_token_id) >= 0:
        _refuse("rows with a message boundary")
    if int(cfg.global_layers) != 1:
        _refuse("one global layer")
    if m.loop_control is not None or m.loop_feedback_gate is not None:
        _refuse("loop control / loop feedback")
    if m._loop_rounds_override:
        _refuse("a loop-round override")
    if bool(getattr(cfg, "write_back_hook", False)) or bool(getattr(cfg, "swa_sink", False)):
        _refuse("write-back hook / swa sinks")
    if any(b.read_rounds != 1 for b in m.layers):
        _refuse("read rounds inside a block")
    if m._notes_override is not None or m._note_keep is not None or m._note_capture is not None:
        _refuse("a notes probe is armed")
    gi = cfg.global_layer_index
    for i, b in enumerate(m.layers):
        if b.flow is not None or b.attn.flow is not None:
            _refuse("flow instrumentation is armed")
        if i != gi and b.attn.pattern != "swa":
            _refuse("local layers must be sliding-window")
    att = m.layers[gi].attn
    if att.pattern != "full":
        _refuse("the global layer is not a full read")
    if att.lsa is not None:
        if m.memory_writer is not None:
            _refuse("LSA's read with a notebook writer")
        return "lsa"
    if m.memory_writer is None:
        if att.compressor is not None:
            _refuse("a compressed-message read")
        return "dense"
    if str(getattr(cfg, "lm_read", "exclusive")) != "closed":
        _refuse("latent memory without the closed read")
    if str(getattr(cfg, "lm_note_key", "writer")) != "tied" or str(getattr(cfg, "lm_note_read", "gist")) != "index":
        _refuse("only the catalogue read (tied note keys, index read); not E33p's gist read or E31's notes")
    if str(cfg.lm_open) not in ("notes", "none") or int(getattr(cfg, "lm_open_levels", 1) or 1) != 1:
        _refuse("catalogue: notes (or nothing) open, one level")
    if bool(getattr(cfg, "lm_open_new_per_pass", False)) and m._open_eval != "reuse":
        _refuse("catalogue with pass exclusion")
    if int(cfg.lm_window) != int(cfg.lm_stride):
        _refuse("catalogue on overlapping pages")
    if att.note_age_slope is not None or att.note_counter is not None:
        _refuse("note-age / count heads")
    if int(getattr(cfg, "message_raw_window", 0) or 0) <= 0:
        _refuse("catalogue without a raw window")
    if m._message_override not in ("real", "none"):
        _refuse(f"message override {m._message_override!r}")
    return "catalogue"


@dataclass
class _Catalogue:
    ni: object                       # nn.exact_read.NoteOpen of the prefill
    slot_close: torch.Tensor
    slot_doc: torch.Tensor
    doc: torch.Tensor                # the message context's doc [B, S], answer positions written as they come
    tag: torch.Tensor                # its (doc, side) tags [B, S] int32
    tag_stride: int
    rw: int
    open: bool
    kappa: list                      # per loop pass: tied note keys [B, C, 1, dh] in the keys' dtype
    members: Optional[torch.Tensor]
    ek: Optional[torch.Tensor]       # learned empty slots [n, g, dh]
    ev: Optional[torch.Tensor]


class IncrementalDecoder:
    """Prefill once, then one position per step. `greedy` is the drop-in for the uncached harness loop."""

    def __init__(self, model: PerceiverARLM):
        self.m = model
        self.cfg = model.config
        self.kind = decode_kind(model)
        self.gi = self.cfg.global_layer_index

    # ------------------------------------------------------------------ limits
    def max_new_tokens(self) -> Optional[int]:
        """Longest answer the cached state stays exact for (None = any)."""
        if self.kind == "catalogue":
            return int(self.cfg.message_raw_window) - int(self.cfg.lm_window)
        return None

    # ------------------------------------------------------------------ prefill
    @torch.no_grad()
    def prefill(self, buf: torch.Tensor, doc: torch.Tensor, L0: int) -> torch.Tensor:
        """buf [B, Lb]: the prompt in [0, L0), anything after; doc [B, Lb]: document ids, −1 = padding or not yet
        written. Runs the model's forward over the row and keeps every attention call's keys and values. Returns the
        final-norm hidden state at L0 − 1 [B, d]."""
        m, cfg = self.m, self.cfg
        mult = int(cfg.attn_pad_multiple)
        S0 = buf.shape[1]
        if mult > 1 and S0 % mult:  # the forward's own right padding (`_pad_inputs`), done once here
            pad = mult - S0 % mult
            buf = F.pad(buf, (0, pad), value=int(cfg.pad_token_id))
            doc = F.pad(doc, (0, pad), value=-1)
        self.buf, self.doc = buf.clone(), doc.clone().long()
        self.B, self.S = buf.shape
        self.L0, self.t = int(L0), int(L0)
        calls: list = []
        msgs: dict = {}

        def hook(mod, args, kwargs):
            x = args[0]
            k_raw, v = mod.kv_raw(x, kwargs.get("ids"))
            k = mod.k_norm(k_raw)
            if mod.use_rope:
                k = apply_rope(k, kwargs["cos"], kwargs["sin"])
            calls.append((mod.layer_idx, k, v))
            if kwargs.get("message") is not None:
                msgs[mod.layer_idx] = kwargs["message"]

        handles = [b.attn.register_forward_pre_hook(hook, with_kwargs=True) for b in m.layers]
        try:
            h = m.hidden_states(self.buf, doc_ids=self.doc)
        finally:
            for hd in handles:
                hd.remove()
        self.K: dict = {}
        self.V: dict = {}
        seen: dict = {}
        for li, k, v in calls:
            r = seen.get(li, 0)
            seen[li] = r + 1
            self.K[(li, r)], self.V[(li, r)] = k, v
        self.passes = seen.get(self.gi, 1)
        self._ds0 = doc_starts(self.doc)  # document starts of the prompt (answer slots are added per step)
        self._ds_cols: dict = {}
        if self.kind == "lsa":
            self._build_lsa()
        elif self.kind == "catalogue":
            self._build_catalogue(msgs.get(self.gi))
            if self.cat.open:
                self.cat.ni.groups.doc_start = self.cat.ni.groups.doc_start.clone()
        return h[:, self.L0 - 1]

    def _ds_col(self, j: int) -> torch.Tensor:
        """[B] document start of position j (prompt: from the prefill; answer slots: as written)."""
        return self._ds_cols[j] if j in self._ds_cols else self._ds0[:, j]

    def _build_lsa(self) -> None:
        c = self.m.layers[self.gi].attn.lsa
        self.lsa_groups = []
        for r in range(self.passes):
            built = lsa_groups(self.K[(self.gi, r)], self.doc, block=c["block"], per_block=c["per_block"],
                               pool=c["pool"], cap=c["cap"], sinks=c["sinks"])
            if built is None:
                self.lsa_groups.append(None)
                continue
            centroids, gidx = built
            members = ChoiceList(gidx, self.B, 1, cap=c["top_k"], m=1).members()
            self.lsa_groups.append((centroids, gidx, members))

    def _build_catalogue(self, msg) -> None:
        att = self.m.layers[self.gi].attn
        if msg is None or msg.index is None:
            raise RuntimeError("the catalogue prefill built no note index")
        ni = msg.index
        tag, _ = msg.tags()
        k0 = self.K[(self.gi, 0)]
        open_ = ni.mode != "none" and msg.override == "real" and getattr(msg, "open_eval", "on") != "off"
        ek = ev = None
        if att.sink_k is not None and int(att.sink_k.shape[0]) > 0:
            n, g, dh = int(att.sink_k.shape[0]), att.g, att.dh
            ek = att.k_norm(att.sink_k).to(k0.dtype)[:, None, :].expand(n, g, dh)
            ev = att.sink_v.to(self.V[(self.gi, 0)].dtype)[:, None, :].expand(n, g, att.sink_v.shape[-1])
        self.cat = _Catalogue(
            ni=ni, slot_close=msg.slot_close, slot_doc=msg.slot_doc, doc=msg.doc.clone(), tag=tag.clone(),
            tag_stride=int(msg.tag_stride()), rw=int(msg.raw_window or 0), open=bool(open_),
            kappa=[tied_note_keys(ni.assign, self.K[(self.gi, r)], ni.page_tok).to(k0.dtype)
                   for r in range(self.passes)],
            members=(ChoiceList(ni.groups, self.B, 1, cap=ni.cap, m=1).members() if open_ else None),
            ek=ek, ev=ev,
        )

    # ------------------------------------------------------------------ one step
    @torch.no_grad()
    def step(self, tok: torch.Tensor) -> torch.Tensor:
        """Write `tok` [B] at the next position and return that position's final-norm hidden state [B, d]."""
        m, cfg = self.m, self.cfg
        t = self.t
        if t >= self.S:
            raise RuntimeError("the row is full")
        B = self.B
        self.buf[:, t] = tok
        self.doc[:, t] = 0  # as the uncached decoding: every written answer slot is document 0
        # where the new position's document starts: its queries count the opened members from there
        # (`choose`, `ChoiceList.opened`); the prefill's indexes saw this slot as padding
        prev = self.doc[:, t - 1] if t else torch.full_like(self.doc[:, 0], -2)
        ds_t = torch.where(prev == self.doc[:, t], self._ds_col(t - 1), torch.full_like(prev, t))
        self._ds_cols[t] = ds_t
        if self.kind == "catalogue":
            self.cat.doc[:, t] = 0
            self.cat.tag[:, t] = 0
            if self.cat.open:
                self.cat.ni.groups.doc_start[:, t] = ds_t
        elif self.kind == "lsa":
            blk = int(m.layers[self.gi].attn.lsa["block"])
            if t % blk == 0 and t > self.L0:  # a block holding answer tokens just closed
                self._build_lsa()
            for built in self.lsa_groups:
                if built is not None:
                    built[1].doc_start[:, t] = ds_t
        dev = self.buf.device
        orders = tuple(int(o) for o in (cfg.ngram_orders or ()))
        lo = max(0, t - (max(orders) if orders else 1) + 1)
        x0 = m.embed(self.buf[:, lo:t + 1], self.doc[:, lo:t + 1])[:, -1:]
        pos = m._positions(t + 1, B, self.doc[:, :t + 1], dev)[:, t:t + 1]
        cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, x0.dtype)
        ids = self.buf[:, t:t + 1]
        self._ds = doc_starts(self.doc) if self.kind == "lsa" else None
        n = len(m.layers)
        n_skip = n // 2

        def run(i, x, skips, r):
            layer = m.layers[i]
            skip = skips.pop() if (i >= n - n_skip and skips) else None
            x = layer.mix(x, x0, skip)
            x = x + self._attn(i, r, layer.attn, layer.attn_norm(x), ids, cos, sin, pos, t)
            x = x + layer.mlp(layer.mlp_norm(x))
            if i < n_skip:
                skips.append(x)
            return x

        x, skips = x0, []
        if m.loop_emb is not None:
            core = range(self.gi, self.gi + 1 + int(cfg.message_loop_span))
            for i in range(self.gi):
                x = run(i, x, skips, 0)
            prelude, skips0 = x, list(skips)
            s = skips0
            for r in range(int(m.loop_rounds)):
                x = x + m.loop_emb[min(r, m.loop_emb.shape[0] - 1)].to(x.dtype)
                if r and m.loop_gate is not None:
                    x = x + m.loop_gate.to(x.dtype) * prelude
                s = list(skips0)
                for i in core:
                    x = run(i, x, s, r)
            skips = s
            for i in range(core.stop, n):
                x = run(i, x, skips, 0)
        else:
            for i in range(n):
                x = run(i, x, skips, 0)
        self.t = t + 1
        return m.final_norm(x)[:, 0]

    def _attn(self, i, r, att, h, ids, cos, sin, pos, t):
        B = h.shape[0]
        q = att.q_norm(att.wq(h).view(B, 1, att.h, att.dh))
        k_raw, v = att.kv_raw(h, ids)
        k = att.k_norm(k_raw)
        if att.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q = att._scale_q(q, pos)
        K, V = self.K[(i, r)], self.V[(i, r)]
        K[:, t] = k[:, 0]
        V[:, t] = v[:, 0]
        if att.pattern == "swa":
            o = self._window_read(q, K, V, t, int(att.window))
        elif self.kind == "dense":
            o = self._window_read(q, K, V, t, t + 1)
        elif self.kind == "lsa":
            o = self._lsa_read(q, K, V, r, t, att)
        else:
            o = self._catalogue_read(q, K, V, r, t)
        return att.wo(o.reshape(B, 1, att.h * att.dh))

    def _window_read(self, q, K, V, t, w):
        """Keys [t − w + 1, t] of the query's document (make_mask_pred: causal, window, same document)."""
        lo = max(0, t - w + 1)
        k, v = K[:, lo:t + 1], V[:, lo:t + 1]
        ok = self.doc[:, lo:t + 1] == self.doc[:, t:t + 1]                                  # [B, L]
        h, g = q.shape[2], k.shape[2]
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if g != h:
            kt = kt.repeat_interleave(h // g, dim=1)
            vt = vt.repeat_interleave(h // g, dim=1)
        o = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=ok[:, None, None, :])
        return o.transpose(1, 2)

    def _lsa_read(self, q, K, V, r, t, att):
        c = att.lsa
        opened = None
        built = self.lsa_groups[r]
        if not att.lsa_off and built is not None:
            centroids, gidx, members = built
            opened = lsa_open_table(q, centroids, gidx, top_k=c["top_k"], width=c["width"], raw_window=c["window"],
                                    impl="list", q_offset=t, members=members)
        return attend_list(q, K, V, opened=opened, raw_window=c["window"], tag=self.doc, key_valid=None,
                           ds=self._ds, n_sinks=c["sinks"], valid_query=True, q_offset=t)

    def _catalogue_read(self, q, K, V, r, t):
        st = self.cat
        opened = None
        if st.open:
            opened = note_open_table(q, st.kappa[r], st.ni, doc=st.doc, slot_close=st.slot_close,
                                     slot_doc=st.slot_doc, raw_window=st.rw, impl="list", q_offset=t,
                                     members=st.members)
        return attend_list(q, K, V, opened=opened, raw_window=st.rw, tag=st.tag, key_valid=None, extra_k=st.ek,
                           extra_v=st.ev, q_offset=t)

    # ------------------------------------------------------------------ greedy decoding
    @torch.no_grad()
    def greedy(self, buf: torch.Tensor, doc: torch.Tensor, L0: int, new: int, *, eos: int,
               check: Optional[Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        """The uncached harness loop with a cache: buf / doc [B, Lb] as `_model_generate` builds them (prompt in
        [0, L0), eos and document −1 after). `check(buf, t, done)` → done, after each written token (stop strings).
        Returns buf[:, :t] with the answer written, eos after a row's first eos."""
        limit = self.max_new_tokens()
        if limit is not None and new > limit:
            _refuse(f"answers longer than {limit} tokens for this read")
        cfg = self.cfg
        W = self.m.lm_head.weight
        h = self.prefill(buf, doc, L0)
        out = self.buf
        B = out.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=out.device)
        t = int(L0)
        for it in range(int(new)):
            if it:
                h = self.step(out[:, t - 1])
            logits = F.linear(h, W).float()
            if cfg.logit_softcap:
                logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            out[:, t] = nxt
            t += 1
            done |= nxt == eos
            if check is not None:
                done = check(out, t, done)
            if bool(done.all()):
                break
        return out[:, :t]


__all__ = ["IncrementalDecoder", "decode_kind"]
