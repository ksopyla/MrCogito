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
  shorter than raw_window − lm_window (256 tokens for E33t n4): the prefill's notes are exact for such answers (with
  the cosine score, which does not depend on the other visible notes), and a longer request is refused up front.
* E37's concept read (2026-11-13): concepts of a closed 512-key block are visible two blocks later, and with the
  reasoner the block before the query's own reads the reasoned update. When the decoding reaches a block end and that
  block holds answer tokens, every pass's concepts and updates are rebuilt from the cache (`build_concepts`).
* E33p's notes read (notes_open, 2026-11-13): the notes are keys of the read (values from the writer, tied keys from
  the reading layer's keys), and a page closes at its last written token. So a page holding the newest position is
  rewritten every step: the writer runs again on that page (it reads page text only, each page on its own) and its
  tied keys are redone per pass. Such a page is visible to the newest position only: when the next token is written,
  the page closes one position later and the previous position no longer reads it, so each step first redoes the
  previous position (two positions per step). Answers stay shorter than raw_window − lm_window, as for the catalogue.

The reading layer reads through the list read (`attend_list`, `ChoiceList`), which gives the table read's outputs
(tests/test_catalogue_read.py::test_list_read_matches_table, tests/test_lsa_read.py::test_lsa_list_read_matches_table),
or, for the concept and notes reads, one query's row of the table (the same `choose`) and one fp32 softmax.
The tokens equal the uncached decoding's and the logits agree to float rounding (tests/test_incremental_decode.py).

Read once, answer many (2026-11-14). `snapshot()` after a prefill is the cache of that row; `ask(snap, question,
new)` restores it, writes the question's tokens as steps (one position each, like answer tokens) and decodes greedily.
A document is then read once and every question costs only its own tokens and its answer's, with the tokens a fresh
prefill over document + question would write (tests/test_incremental_decode.py::test_read_once_answer_many). For the
catalogue and notes reads the question and the answer together stay under `max_new_tokens()`.

Supported; anything else raises NotImplementedError before any work, so callers keep the uncached path: perceiver
par_mode, causal, no message boundary, one global layer, the tied loop over [global read + span layers] with loop
markers (and the prelude gate), sliding-window local layers without sinks; the reading layer is full attention,
LSA's read, E37's concept read (with or without the reasoner and the length scale), the catalogue read (index notes
on non-overlapping pages, cosine score, no pass exclusion, one level), or E33p's notes read (tied keys, notes or
nothing opened, argmax filing, the writer reading the token table). Not yet: page-level opening, note-age or count
heads, write-back hooks, loop control, loop feedback.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from nn.exact_read import ORD_MOD, ChoiceList, choose, doc_starts, note_open_table, tied_note_keys
from nn.lsa_buckets import lsa_groups, lsa_open_table
from nn.perceiver_ar_lm import PerceiverARLM, apply_rope, attend_list, rope_cos_sin


def _clone_state(x, memo: Optional[dict] = None):
    """A copy of a decoder state: every tensor cloned once (two names of one tensor stay one tensor), containers and
    plain objects rebuilt, modules and functions shared."""
    memo = {} if memo is None else memo
    if id(x) in memo:
        return memo[id(x)]
    if x is None or isinstance(x, (bool, int, float, str, bytes, torch.nn.Module, type)) or callable(x):
        return x
    if torch.is_tensor(x):
        y = x.detach().clone()
    elif isinstance(x, dict):
        y = {}
        memo[id(x)] = y
        for k, v in x.items():
            y[k] = _clone_state(v, memo)
        return y
    elif isinstance(x, list):
        y = []
        memo[id(x)] = y
        y.extend(_clone_state(v, memo) for v in x)
        return y
    elif isinstance(x, tuple):
        vals = [_clone_state(v, memo) for v in x]
        y = type(x)(*vals) if hasattr(x, "_fields") else tuple(vals)
    elif hasattr(x, "__dict__"):
        y = copy.copy(x)
        memo[id(x)] = y
        for k, v in vars(x).items():
            object.__setattr__(y, k, _clone_state(v, memo))
        return y
    else:
        y = copy.deepcopy(x)
    memo[id(x)] = y
    return y


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
    if att.concept is not None:
        if m.memory_writer is not None or att.compressor is not None:
            _refuse("the concept read with a notebook or a compressor")
        if att.concept_stats is not None:
            _refuse("a concept-read probe is armed")
        return "concept"
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
    if str(getattr(cfg, "lm_note_key", "writer")) != "tied":
        _refuse("E31's writer-keyed notes (only tied note keys: E33p's notes read, E33t's catalogue)")
    if int(getattr(cfg, "message_extra_slot_attends", 0) or 0) or bool(getattr(cfg, "message_update_slot_kv", False)):
        _refuse("extra slot reads")
    if att.note_age_slope is not None or att.note_counter is not None:
        _refuse("note-age / count heads")
    if int(getattr(cfg, "message_raw_window", 0) or 0) <= 0:
        _refuse("notes without a raw window")
    note_read = str(getattr(cfg, "lm_note_read", "gist"))
    if note_read == "gist":  # E33p: the notes are keys of the read (values from the writer) and the index
        if str(cfg.lm_open) not in ("notes", "none") or int(getattr(cfg, "lm_open_levels", 1) or 1) != 1:
            _refuse("the notes read: notes (or nothing) open, one level")
        if bool(getattr(cfg, "lm_open_new_per_pass", False)):
            _refuse("the notes read with pass exclusion")
        if str(getattr(cfg, "lm_filing", "argmax")) != "argmax":
            _refuse("the notes read with balanced filing")
        if str(getattr(cfg, "lm_writer_input", "table")) != "table":
            _refuse("a writer that reads the decoder's input layer (lm_writer_input='embed')")
        if m._open_eval != "on" or m._message_override != "real":
            _refuse(f"eval switches (open {m._open_eval!r}, message {m._message_override!r})")
        return "notes"
    if note_read != "index":
        _refuse(f"lm_note_read {note_read!r}")
    if str(cfg.lm_open) not in ("notes", "none") or int(getattr(cfg, "lm_open_levels", 1) or 1) != 1:
        _refuse("catalogue: notes (or nothing) open, one level")
    if bool(getattr(cfg, "lm_open_new_per_pass", False)) and m._open_eval != "reuse":
        _refuse("catalogue with pass exclusion")
    if int(cfg.lm_window) != int(cfg.lm_stride):
        _refuse("catalogue on overlapping pages")
    if str(getattr(cfg, "lm_open_score", "sum_softmax")) != "max_cos":
        # a softmax over the visible notes would count the pages holding answer tokens, which the prefill's
        # notebook does not have; the cosine score of a note does not depend on the others
        _refuse("catalogue scored by a softmax over the visible notes")
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
        if self.kind in ("catalogue", "notes"):  # a page holding answer tokens has no member old enough to open
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
        gatt = m.layers[self.gi].attn
        keep_x = self.kind == "concept" and gatt.concept_reasoner is not None  # the reasoner reads the layer input

        def hook(mod, args, kwargs):
            x = args[0]
            k_raw, v = mod.kv_raw(x, kwargs.get("ids"))
            k = mod.k_norm(k_raw)
            if mod.use_rope:
                k = apply_rope(k, kwargs["cos"], kwargs["sin"])
            calls.append((mod.layer_idx, k, v, x if (keep_x and mod.layer_idx == self.gi) else None))
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
        self.X: dict = {}  # the reading layer's input per pass (the concept reasoner's states)
        seen: dict = {}
        for li, k, v, x in calls:
            r = seen.get(li, 0)
            seen[li] = r + 1
            self.K[(li, r)], self.V[(li, r)] = k, v
            if x is not None:
                self.X[(li, r)] = x.clone()
        self.passes = seen.get(self.gi, 1)
        self._ds0 = doc_starts(self.doc)  # document starts of the prompt (answer slots are added per step)
        self._ds_cols: dict = {}
        if self.kind == "lsa":
            self._build_lsa()
        elif self.kind == "concept":
            self._build_concepts()
        elif self.kind == "catalogue":
            self._build_catalogue(msgs.get(self.gi))
            if self.cat.open:
                self.cat.ni.groups.doc_start = self.cat.ni.groups.doc_start.clone()
        elif self.kind == "notes":
            self._build_notes(msgs.get(self.gi))
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

    def _build_concepts(self) -> None:
        """E37: each pass's concepts of the written row (nn.concept_memory.build_concepts, as `_concept_read`), the
        reasoner's value updates, per concept and spread to the member tokens."""
        from nn.concept_memory import build_concepts, token_update

        att = self.m.layers[self.gi].attn
        c = att.concept
        doc = self.doc
        ds = doc_starts(doc)
        pos = torch.arange(self.S, device=doc.device)[None]
        sink = (pos - ds < int(c["sinks"])) & (doc >= 0)
        self.cc = []
        for r in range(self.passes):
            if att.concept_off:
                self.cc.append(None)
                continue
            x = self.X[(self.gi, r)] if att.concept_reasoner is not None else None
            cidx = build_concepts(self.K[(self.gi, r)], self.V[(self.gi, r)], x, doc, sink, block=int(c["block"]),
                                  per_block=int(c["per_block"]), iters=int(c["iters"]), tau=float(c["tau"]))
            if cidx is None:
                self.cc.append(None)
                continue
            dv = dv_tok = None
            if att.concept_reasoner is not None:
                dv = att.concept_reasoner(cidx.z, cidx.block_of, cidx.c_doc)
                dv_tok = token_update(cidx, dv, self.S)
            self.cc.append((cidx, dv, dv_tok))

    def _build_notes(self, msg) -> None:
        """E33p's notes read: the prefill's notebook (the writer's note values, the tied keys' assignment, when each
        page closed), copied so the pages holding answer tokens can be rewritten as the answer grows."""
        if msg is None or msg.index is None or msg.slots is None:
            raise RuntimeError("the notes prefill built no notebook")
        ni = msg.index
        geo = self.m.memory_writer.geometry(self.S)
        k0 = self.K[(self.gi, 0)]
        self.nt = SimpleNamespace(
            ni=ni, W=int(geo.window), K=int(geo.latents), starts=list(geo.starts), rw=int(msg.raw_window or 0),
            v_bar=msg.slots[1].clone(), assign=ni.assign.clone(), slot_close=msg.slot_close.clone(),
            slot_doc=msg.slot_doc.clone(), open=ni.mode != "none",
            kappa=[tied_note_keys(ni.assign, self.K[(self.gi, r)], ni.page_tok).to(k0.dtype)
                   for r in range(self.passes)],
            dirty=[],
        )
        if self.nt.open:
            ni.groups.doc_start = ni.groups.doc_start.clone()

    def _rewrite_pages(self, t: int) -> None:
        """The pages holding position t gained a token: their notes again from the writer (it reads page text only,
        each page on its own), as the uncached forward writes them for this row. Their tied keys are redone per pass
        once the pass has written its key at t (`_notes_read`)."""
        m, st = self.m, self.nt
        W, K = st.W, st.K
        wins = [w for w, s in enumerate(st.starts) if s <= t < s + W]
        pos = m._positions(self.S, self.B, self.doc, self.buf.device)
        for w in wins:
            s = st.starts[w]
            ids, doc = self.buf[:, s:s + W], self.doc[:, s:s + W]
            out = m.memory_writer(m._writer_input(ids, doc), torch.zeros_like(doc), doc, None, pos[:, s:s + W],
                                  self.cfg.rope_theta)
            st.assign[:, w] = out["assign"].float().mean(1).reshape(self.B, K, W)
            st.v_bar[:, w * K:(w + 1) * K] = out["v"].to(st.v_bar.dtype)
            st.slot_doc[:, w * K:(w + 1) * K] = out["slot_doc"]
            tok = s + torch.arange(W, device=doc.device)
            last = torch.where(out["window_pick"][:, 0], tok[None], torch.full_like(doc, -1)).amax(-1)
            st.slot_close[:, w * K:(w + 1) * K] = last[:, None]
        # tied keys to redo per pass: pages holding t, and those holding t − 1 (its keys are redone this step)
        st.dirty = [w for w, s in enumerate(st.starts) if s <= t < s + W or s <= t - 1 < s + W]

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
        # A block holding answer tokens just closed: rebuild what the reading layer derives from the closed blocks,
        # from the tokens written so far (position t is still unwritten; its own block is not read this way)
        if self.kind in ("lsa", "concept") and t > self.L0:
            att = m.layers[self.gi].attn
            blk = int((att.lsa if self.kind == "lsa" else att.concept)["block"])
            if t % blk == 0:
                self._build_lsa() if self.kind == "lsa" else self._build_concepts()
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
            for built in self.lsa_groups:
                if built is not None:
                    built[1].doc_start[:, t] = ds_t
        elif self.kind == "concept":
            for built in self.cc:
                if built is not None:
                    built[0].groups.doc_start[:, t] = ds_t
        elif self.kind == "notes":
            if self.nt.open:
                self.nt.ni.groups.doc_start[:, t] = ds_t
            self._rewrite_pages(t)
            if t:
                # The pages holding t − 1 and t closed at t − 1 in the last row, so t − 1 read their notes; with t
                # written they close at t and t − 1 no longer reads them. The uncached forward over this row computes
                # t − 1 that way, and t reads its keys: redo t − 1 first (its output is not used).
                self._position(t - 1)
        h = self._position(t)
        self.t = t + 1
        return h

    def _position(self, t: int) -> torch.Tensor:
        """Run position t (already written) through the layers with the cached keys of the earlier positions, writing
        its own keys and values; its final-norm hidden state [B, d]."""
        m, cfg = self.m, self.cfg
        B = self.B
        dev = self.buf.device
        orders = tuple(int(o) for o in (cfg.ngram_orders or ()))
        lo = max(0, t - (max(orders) if orders else 1) + 1)
        x0 = m.embed(self.buf[:, lo:t + 1], self.doc[:, lo:t + 1])[:, -1:]
        pos = m._positions(t + 1, B, self.doc[:, :t + 1], dev)[:, t:t + 1]
        cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, x0.dtype)
        ids = self.buf[:, t:t + 1]
        self._ds = doc_starts(self.doc) if self.kind in ("lsa", "concept") else None
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
        if (i, r) in self.X:
            self.X[(i, r)][:, t] = h[:, 0]
        if att.pattern == "swa":
            o = self._window_read(q, K, V, t, int(att.window))
        elif self.kind == "dense":
            o = self._window_read(q, K, V, t, t + 1)
        elif self.kind == "lsa":
            o = self._lsa_read(q, K, V, r, t, att)
        elif self.kind == "concept":
            o = self._concept_read(q, K, V, r, t, att)
        elif self.kind == "notes":
            o = self._notes_read(q, K, V, r, t)
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

    def _concept_read(self, q, K, V, r, t, att):
        """E37 (nn.concept_memory.attend_concepts) for the query t: one softmax over its document's sinks, the
        block-aligned window (its block and the one before), the member tokens of the concepts it opens (largest
        estimated mass first, until top_k tokens) and one summary entry per other visible concept, with the summaries'
        bias log n + |q|² σ² / 2d (+ log ρ, the length scale). With the reasoner, keys before the query's block read
        v + the update, its own block the raw v (attend_concepts' two copies; their masks do not overlap). fp32."""
        from nn.concept_memory import summary_log_scale, summary_logit

        c = att.concept
        blk, cap = int(c["block"]), int(c["top_k"])
        B, _, h, d = q.shape
        n = t + 1
        jpos = torch.arange(n, device=q.device)
        doc, doc_t, ds = self.doc[:, :n], self.doc[:, t], self._ds
        own_lo = (t // blk) * blk
        sink = ((jpos[None] - ds[:, :n]) < int(c["sinks"])) & (doc >= 0)
        keep = (jpos >= own_lo - blk)[None] | sink
        vt = V[:, :n]
        ks, vs, sbias = None, None, None
        built = self.cc[r]
        if built is not None:
            cidx, dv, dv_tok = built
            gi = cidx.groups
            G = gi.n_groups
            score = summary_logit(q, cidx, variance=bool(c["variance"])).amax(2)          # [B, 1, C]
            visible = (gi.g_close[:, None, :] <= t) & torch.isfinite(score)
            ids, off = choose(torch.nan_to_num(score, neginf=-1e30), visible, gi,
                              torch.tensor([t], device=q.device), m=int(c["width"]), cap=cap, raw_window=blk)
            row = torch.full((B, G), cap, dtype=torch.int32, device=q.device)              # the table's row of t
            row.scatter_(1, ids[:, 0], off[:, 0].to(torch.int32))
            codes = gi.codes[0, :, :n].long()
            g = torch.div(codes, ORD_MOD, rounding_mode="floor")
            hit = (codes >= 0) & (row.gather(1, g.clamp(0, G - 1)) + (codes - g * ORD_MOD) < cap)
            keep = keep | (hit & ((t - jpos) >= blk)[None])
            if dv_tok is not None:
                vt = torch.where((jpos < own_lo)[None, :, None, None], vt + dv_tok[:, :n].to(vt.dtype), vt)
            if c["summaries"]:
                C = cidx.n_concepts
                vis = ((t >= (cidx.block_of + 2) * blk)[None] & (cidx.c_doc == doc_t[:, None])
                       & (doc_t >= 0)[:, None] & ~(row[:, :C] < cap))                       # [B, C]
                logn = torch.where(cidx.n > 0, cidx.n.clamp_min(1.0).log(), torch.zeros_like(cidx.n))
                vhalf = 0.5 * cidx.var if c["variance"] else torch.zeros_like(cidx.var)
                qn2 = (q.float() ** 2).sum(-1)[:, 0] / d                                    # [B, h]
                sb = logn[:, None, :].float() + qn2[..., None] * vhalf[:, None, :].float()  # [B, h, C]
                lr = summary_log_scale(ds[:, :n], n, blk, int(c.get("sum_len", 0) or 0))
                if lr is not None:
                    sb = sb + lr[:, t][:, None, None]
                sbias = sb.masked_fill(~vis[:, None, :], float("-inf"))
                ks = cidx.mu_k[:, :, None].to(K.dtype)
                vs = (cidx.mu_v if dv is None else cidx.mu_v + dv.to(cidx.mu_v.dtype))[:, :, None].to(V.dtype)
        mask = keep & (doc == doc_t[:, None]) & (doc_t >= 0)[:, None]                      # [B, n]
        kk, vv = K[:, :n, 0].float(), vt[:, :, 0].float()                                  # one key head
        bias = torch.zeros(B, n, device=q.device).masked_fill(~mask, float("-inf"))[:, None].expand(B, h, n)
        if sbias is not None:
            kk = torch.cat([kk, ks[:, :, 0].float()], 1)
            vv = torch.cat([vv, vs[:, :, 0].float()], 1)
            bias = torch.cat([bias, sbias], -1)
        lg = torch.einsum("bhd,bkd->bhk", q[:, 0].float(), kk) / math.sqrt(d) + bias
        o = torch.einsum("bhk,bkd->bhd", torch.softmax(lg, -1), vv)
        return o[:, None].to(q.dtype)

    def _notes_read(self, q, K, V, r, t):
        """E33p (`_attend_open` with the closed rule) for the query t: one softmax over its document's raw window, the
        raw tokens of the notes it opens (nn.exact_read.note_open_table: its k best visible notes holding a member
        older than the window, oldest members first, up to cap tokens) and every note of its document whose page
        closed at or before t (keys: the tied keys of this pass, values: the writer's). fp32."""
        st = self.nt
        ni = st.ni
        Kn = st.K
        for w in st.dirty:  # pages holding t − 1 or t: tied keys with this pass's latest keys there
            st.kappa[r][:, w * Kn:(w + 1) * Kn] = tied_note_keys(st.assign[:, w:w + 1], K,
                                                                 ni.page_tok[w:w + 1]).to(st.kappa[r].dtype)
        B, _, h, dh = q.shape
        g = K.shape[2]
        n = t + 1
        jpos = torch.arange(n, device=q.device)
        doc_t = self.doc[:, t]
        same = (self.doc[:, :n] == doc_t[:, None]) & (doc_t >= 0)[:, None]                 # one side per row
        raw = same & ((t - jpos) < st.rw)[None]
        kap = st.kappa[r]                                                                   # [B, C, g, dh]
        vis = ((st.slot_doc >= 0) & (st.slot_close >= 0) & (st.slot_doc == doc_t[:, None]) & (doc_t >= 0)[:, None]
               & (st.slot_close <= t))                                                      # [B, C]
        qf = q[:, 0].float()                                                                # [B, h, dh]
        rep = h // g
        if st.open:
            gi = ni.groups
            G, cap = gi.n_groups, int(ni.cap)
            kb = kap[:, :, 0].float()
            if ni.score == "max_cos":
                s_note = torch.einsum("bhd,bcd->bhc", F.normalize(qf, dim=-1), F.normalize(kb, dim=-1)).amax(1)
            else:
                lg = torch.einsum("bhd,bcd->bhc", qf, kb) * dh ** -0.5
                p = torch.softmax(lg.masked_fill(~vis[:, None], float("-inf")), dim=-1)
                s_note = torch.nan_to_num(p, nan=0.0).sum(1)                                # [B, C]
            cand = vis & (gi.g_minpos <= t - st.rw)
            ids, off = choose(s_note[:, None], cand[:, None], gi, torch.tensor([t], device=q.device), m=int(ni.k),
                              cap=cap, raw_window=st.rw)
            row = torch.full((B, G), cap, dtype=torch.int32, device=q.device)
            row.scatter_(1, ids[:, 0], off[:, 0].to(torch.int32))
            hit = torch.zeros(B, n, dtype=torch.bool, device=q.device)
            for li in range(int(gi.codes.shape[0])):  # overlapping pages: a token is a member once per layer
                c = gi.codes[li, :, :n].long()
                gg = torch.div(c, ORD_MOD, rounding_mode="floor")
                hit |= (c >= 0) & (row.gather(1, gg.clamp(0, G - 1)) + (c - gg * ORD_MOD) < cap)
            raw = raw | (hit & same & ((t - jpos) >= st.rw)[None])
        kk = torch.cat([K[:, :n], kap.to(K.dtype)], 1).float().repeat_interleave(rep, dim=2)  # [B, n + C, h, dh]
        vv = torch.cat([V[:, :n], st.v_bar.to(V.dtype)], 1).float().repeat_interleave(rep, dim=2)
        ok = torch.cat([raw, vis], 1)
        lg = torch.einsum("bhd,bkhd->bhk", qf, kk) * dh ** -0.5
        lg = lg.masked_fill(~ok[:, None], float("-inf"))
        o = torch.einsum("bhk,bkhd->bhd", torch.softmax(lg, -1), vv)
        return o[:, None].to(q.dtype)

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
    def _decode(self, h: torch.Tensor, new: int, eos: int,
                check: Optional[Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]] = None) -> int:
        """Greedy steps from the current position (`h`: the hidden state of the position before it). Writes the
        answer into the row and returns the position after its last token."""
        cfg = self.cfg
        W = self.m.lm_head.weight
        out = self.buf
        B = out.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=out.device)
        t = int(self.t)
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
        return t

    @torch.no_grad()
    def greedy(self, buf: torch.Tensor, doc: torch.Tensor, L0: int, new: int, *, eos: int,
               check: Optional[Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        """The uncached harness loop with a cache: buf / doc [B, Lb] as `_model_generate` builds them (prompt in
        [0, L0), eos and document −1 after). `check(buf, t, done)` → done, after each written token (stop strings).
        Returns buf[:, :t] with the answer written, eos after a row's first eos."""
        limit = self.max_new_tokens()
        if limit is not None and new > limit:
            _refuse(f"answers longer than {limit} tokens for this read")
        h = self.prefill(buf, doc, L0)
        t = self._decode(h, new, eos, check)
        return self.buf[:, :t]

    # ------------------------------------------------------------------ read once, answer many
    _SHARED = ("m", "cfg", "kind", "gi")

    def snapshot(self) -> dict:
        """A copy of everything a step changes: the cached keys and values, the row, the reading layer's indexes.
        Taken after `prefill` over a document it is that document's cache; `restore` (or `ask`) starts from it."""
        return _clone_state({k: v for k, v in self.__dict__.items() if k not in self._SHARED})

    def restore(self, snap: dict) -> None:
        """Back to a snapshot (copied again, so the snapshot can be restored any number of times)."""
        for k in [k for k in self.__dict__ if k not in self._SHARED]:
            del self.__dict__[k]
        self.__dict__.update(_clone_state(snap))

    @torch.no_grad()
    def feed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Write `tokens` [B, n] (or [n], the same for every row) at the next positions, one step each, and return
        the final-norm hidden state of the last one [B, d]: the question after a cached document."""
        if tokens.dim() == 1:
            tokens = tokens[None].expand(self.B, -1)
        if tokens.shape[1] == 0:
            raise ValueError("feed needs at least one token")
        tokens = tokens.to(self.buf.device)
        h = None
        for j in range(tokens.shape[1]):
            h = self.step(tokens[:, j])
        return h

    @torch.no_grad()
    def ask(self, snap: dict, question: torch.Tensor, new: int, *, eos: int,
            check: Optional[Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        """One question from a document's cache: restore `snap` (taken after `prefill`), write the question, decode
        up to `new` tokens. Returns the answer tokens [B, ≤ new]. The prefilled row needs free slots for both:
        L0 + len(question) + new ≤ its length."""
        n = int(question.shape[-1])
        limit = self.max_new_tokens()
        if limit is not None and n + int(new) > limit:
            _refuse(f"question + answer longer than {limit} tokens for this read")
        self.restore(snap)
        if self.t + n + int(new) > self.S:
            raise ValueError(f"the row has {self.S - self.t} free slots; the question and answer need {n + int(new)}")
        h = self.feed(question)
        t0 = int(self.t)
        t = self._decode(h, new, eos, check)
        return self.buf[:, t0:t].clone()


__all__ = ["IncrementalDecoder", "decode_kind"]
