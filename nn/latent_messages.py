"""E35 — latent messages between agents: memory, writing to it, and sending from it, kept apart.

Spec: docs/experiments_specs/ahead/E35_latent_messages.md.

    AGENT A (sender)                                                  AGENT B (receiver)
    text ─► WRITER ─► MEMORY ─► TRANSMITTER ══ message [K × d_B] ══► RECEIVER: [message | own text] → answer

* WRITER — the E31 page writer (`nn.latent_memory.LatentMemoryWriter`) with the sender's own token embedding:
  pages of `lm_window` tokens every `lm_stride`, each written independently into `lm_latents` latents.
* MEMORY — `LatentMemory`: one entry per latent (its state ‖ what it read, 2·D numbers), its source position, its
  page, a valid mask. Append-only and page-parallel, so it grows linearly with the text; nothing reads raw text
  after writing.
* TRANSMITTER — `Transmitter`: K learned slots; each block = cross-attention over every memory entry (plus a learned
  "nothing" entry), with an optional order term (E33e: each entry's score drops by s_h · log(1 + soft count of
  matching entries on newer pages), s_h starting at 0), then self-attention among the slots and a feed-forward
  layer. The output is RMS-normed like the receiver's input embeddings.
* RECEIVER — a `PerceiverARLM` decoder reading the message as its first K input positions (`prefix_embeds`), the
  way "Cramming 1568 tokens into a single vector" (arXiv 2502.13063) feeds its memory vectors: every layer sees them.

The only tensor that crosses from A to B is the message. `write`, `send` and `answer` are separate calls; `forward`
chains them for training on rows ``BOS document EOS BOS receiver-text EOS`` (a row without that split is
receiver-only: the sender's memory is empty and the message reads the "nothing" entry).

Shapes: document tokens [B, S_A] → memory entries [B, C = n_pages · K_lat, 2·D] → message [B, K, d_B];
receiver tokens [B, S_B] → logits [B, S_B, V].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from nn.latent_memory import LatentMemoryWriter
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM


class A2AConfig(PretrainedConfig):
    model_type = "a2a_latent_messages"

    def __init__(
        self,
        sender_config: Optional[dict] = None,    # PerceiverARConfig fields of the writer's model (lm_* geometry, token dim)
        receiver_config: Optional[dict] = None,  # PerceiverARConfig fields of the receiver decoder
        msg_slots: int = 16,                     # K: vectors per message
        tx_layers: int = 2,
        tx_heads: int = 8,
        tx_dim: int = 512,
        tx_ffn: int = 1024,
        tx_order: str = "rank",                  # "rank" (E33e order among matching entries) | "none"
        tx_rank_margin: float = 0.5,
        tx_rank_sharp: float = 4.0,
        tx_request: str = "none",               # "none" (broadcast) | "question" (B's question steers what A sends)
        question_token_id: int = -1,             # E35 request mode: the token that opens a question block ("Question")
        answer_token_id: int = -1,               # … and the one that opens its answer ("Answer")
        loss_on: str = "all",                    # "all" receiver tokens | "answers" (from "Answer" ":" to the period)
        end_token_id: int = -1,                  # the token that closes an answer ("."), for loss_on="answers"
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.sender_config = dict(sender_config or {})
        self.receiver_config = dict(receiver_config or {})
        self.msg_slots = int(msg_slots)
        self.tx_layers = int(tx_layers)
        self.tx_heads = int(tx_heads)
        self.tx_dim = int(tx_dim)
        self.tx_ffn = int(tx_ffn)
        self.tx_order = str(tx_order)
        self.tx_rank_margin = float(tx_rank_margin)
        self.tx_rank_sharp = float(tx_rank_sharp)
        self.tx_request = str(tx_request)
        self.question_token_id = int(question_token_id)
        self.answer_token_id = int(answer_token_id)
        self.loss_on = str(loss_on)
        self.end_token_id = int(end_token_id)
        if self.loss_on not in ("all", "answers"):
            raise ValueError(f"loss_on must be all / answers, got {self.loss_on!r}")
        if self.loss_on == "answers" and min(self.answer_token_id, self.end_token_id) < 0:
            raise ValueError("loss_on='answers' needs answer_token_id and end_token_id")
        if self.tx_request not in ("none", "question"):
            raise ValueError(f"tx_request must be none / question, got {self.tx_request!r}")
        if self.tx_request == "question" and min(self.question_token_id, self.answer_token_id) < 0:
            raise ValueError("tx_request='question' needs question_token_id and answer_token_id")
        if self.tx_order not in ("rank", "none"):
            raise ValueError(f"tx_order must be rank / none, got {self.tx_order!r}")
        if self.tx_dim % self.tx_heads:
            raise ValueError("tx_dim must be divisible by tx_heads")
        # Bookkeeping read by the shared entrypoint / W&B init / eval routing (the receiver's shape).
        self.checkpoint_family = "a2a"
        self.pretraining_objective = "causal_lm"
        self.concept_num = 0
        self.hidden_act = "silu"
        self.norm_type = "rmsnorm"
        for k in ("hidden_size", "num_hidden_layers", "token_embedding_dim", "intermediate_size", "num_attention_heads",
                  "vocab_size"):
            setattr(self, k, int(self.receiver_config.get(k, 0) or 0))
        self.max_sequence_length = kwargs.get("max_sequence_length", None)


@dataclass
class LatentMemory:
    """What agent A keeps after reading: one entry per written latent."""
    entries: torch.Tensor  # [B, C, 2·D]  latent state ‖ what it read
    pos: torch.Tensor      # [B, C] source position (expected position of what the latent read)
    valid: torch.Tensor    # [B, C] bool
    per_page: int          # entries per page (latents per page); entries are page-ordered, page by page

    @property
    def n_entries(self) -> int:
        return int(self.valid.sum())


class _Block(nn.Module):
    def __init__(self, D: int, H: int, F_: int, order: bool):
        super().__init__()
        self.H, self.dh = H, D // H
        self.q_norm, self.m_norm = nn.RMSNorm(D), nn.RMSNorm(D)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(D, D, bias=False) for _ in range(4))
        self.null_k = nn.Parameter(torch.zeros(H, self.dh))
        self.null_v = nn.Parameter(torch.zeros(H, self.dh))
        self.s_norm = nn.RMSNorm(D)
        self.wqkv = nn.Linear(D, 3 * D, bias=False)
        self.so = nn.Linear(D, D, bias=False)
        self.f_norm = nn.RMSNorm(D)
        self.f_up = nn.Linear(D, 2 * F_, bias=False)
        self.f_down = nn.Linear(F_, D, bias=False)
        self.order_strength = nn.Parameter(torch.zeros(H)) if order else None
        for lin in (self.wo, self.so, self.f_down):
            nn.init.zeros_(lin.weight)  # every block starts as the identity on the slots

    def forward(self, slots, mem, valid, per_page: int, margin: float, sharp: float):
        B, K, D = slots.shape
        H, dh = self.H, self.dh
        q = self.wq(self.q_norm(slots)).view(B, K, H, dh)
        m = self.m_norm(mem)
        k = self.wk(m).view(B, -1, H, dh)
        v = self.wv(m).view(B, -1, H, dh)
        logit = torch.einsum("bkhd,bchd->bhkc", q.float(), k.float()) * dh ** -0.5  # [B,H,K,C]
        logit = logit.masked_fill(~valid[:, None, None, :], float("-inf"))
        if self.order_strength is not None and logit.shape[-1]:
            logit = logit - self.order_strength.float()[None, :, None, None] * _newer_match_logcount(
                logit, valid, per_page, margin, sharp)
        null = torch.einsum("bkhd,hd->bhk", q.float(), self.null_k.float())[..., None] * dh ** -0.5
        w = torch.softmax(torch.cat([logit, null], -1), -1)
        out = torch.einsum("bhkc,bchd->bkhd", w[..., :-1], v.float()) + w[..., -1:].permute(0, 2, 1, 3) * self.null_v.float()
        slots = slots + self.wo(out.reshape(B, K, D).to(slots.dtype))
        qkv = self.wqkv(self.s_norm(slots)).view(B, K, 3, H, dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        slots = slots + self.so(a.transpose(1, 2).reshape(B, K, D))
        g, u = self.f_up(self.f_norm(slots)).chunk(2, -1)
        return slots + self.f_down(F.silu(g) * u)


@torch.no_grad()
def _newer_match_logcount(logit, valid, per_page: int, margin: float, sharp: float) -> torch.Tensor:
    """E33e's order term for the transmitter: log(1 + Σ gate over entries on strictly newer pages), where an entry
    matches when its score is within ~margin of the slot's best entry: gate = σ(sharp · (l − max + margin)).
    Entries are page-ordered (`per_page` per page), so "newer" is a reverse running sum over pages. [B,H,K,C]."""
    B, H, K, C = logit.shape
    n_p = C // per_page
    best = logit.amax(-1, keepdim=True)
    best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
    gate = torch.sigmoid(sharp * (logit - best + margin))  # 0 on invalid entries (score −inf)
    g = gate[..., : n_p * per_page].view(B, H, K, n_p, per_page).sum(-1)
    newer = g.flip(-1).cumsum(-1).flip(-1) - g  # pages strictly after p
    out = torch.log1p(newer).repeat_interleave(per_page, -1)
    if out.shape[-1] < C:
        out = F.pad(out, (0, C - out.shape[-1]))
    return out


class Transmitter(nn.Module):
    """Memory (+ nothing) → K message vectors scaled like the receiver's input embeddings."""

    def __init__(self, cfg: A2AConfig, d_mem: int, d_out: int):
        super().__init__()
        D = cfg.tx_dim
        self.K = cfg.msg_slots
        self.margin, self.sharp = cfg.tx_rank_margin, cfg.tx_rank_sharp
        self.slots = nn.Parameter(torch.randn(self.K, D) * 0.02)
        self.in_proj = nn.Linear(d_mem, D, bias=False)
        self.blocks = nn.ModuleList(_Block(D, cfg.tx_heads, cfg.tx_ffn, cfg.tx_order == "rank")
                                    for _ in range(cfg.tx_layers))
        self.out_norm_in = nn.RMSNorm(D)
        self.out = nn.Linear(D, d_out, bias=False)
        self.out_norm = nn.RMSNorm(d_out)
        # request mode: B's request (its question, pooled in B's own embedding space) is added to every slot;
        # zero at the start, so a fresh request-mode transmitter sends what a broadcast one would
        self.req_norm = nn.RMSNorm(d_out) if cfg.tx_request != "none" else None
        self.req_proj = nn.Linear(d_out, D, bias=False) if cfg.tx_request != "none" else None
        if self.req_proj is not None:
            nn.init.zeros_(self.req_proj.weight)

    def forward(self, memory: LatentMemory, request: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = memory.entries.shape[0]
        mem = self.in_proj(memory.entries.to(self.in_proj.weight.dtype))
        x = self.slots[None].to(mem.dtype).expand(B, -1, -1)
        if self.req_proj is not None:
            if request is None:  # keep the request path in the graph (DDP) when a batch has none
                x = x + 0.0 * self.req_proj.weight.sum() + 0.0 * self.req_norm.weight.sum()
            else:
                x = x + self.req_proj(self.req_norm(request.to(mem.dtype)))[:, None]
        for blk in self.blocks:
            x = blk(x, mem, memory.valid, memory.per_page, self.margin, self.sharp)
        return self.out_norm(self.out(self.out_norm_in(x)))


def split_rows(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], bos: int, eos: int, pad: int):
    """Rows `BOS doc… EOS BOS recv… EOS [pad]` → (doc ids, doc mask, receiver ids, receiver mask), each right-padded.
    A row without an EOS directly followed by BOS is receiver-only (empty document)."""
    B, S = input_ids.shape
    lens = attention_mask.sum(-1).tolist() if attention_mask is not None else [S] * B
    docs, recvs = [], []
    for b in range(B):
        row = input_ids[b, : int(lens[b])]
        hit = ((row[:-1] == eos) & (row[1:] == bos)).nonzero()
        if len(hit):
            e = int(hit[0, 0])
            docs.append(row[1:e])
            recvs.append(row[e + 1:])
        else:
            docs.append(row[:0])
            recvs.append(row)

    def pack(seqs):
        L = max([len(t) for t in seqs] + [1])
        ids = input_ids.new_full((B, L), pad)
        mask = torch.zeros(B, L, dtype=torch.long, device=input_ids.device)
        for b, t in enumerate(seqs):
            ids[b, : len(t)] = t
            mask[b, : len(t)] = 1
        return ids, mask

    return (*pack(docs), *pack(recvs))


def answer_targets(ids: torch.Tensor, mask: torch.Tensor, answer_tok: int, end_tok: int) -> torch.Tensor:
    """Targets on answer tokens only: after "Answer" ":" up to and including the closing token; −100 elsewhere."""
    tgt = torch.full_like(ids, -100)
    for b in range(ids.shape[0]):
        row = ids[b].tolist()
        n = int(mask[b].sum())
        i = 0
        while i < n:
            if row[i] == answer_tok:
                j = i + 2
                while j < n:
                    tgt[b, j] = row[j]
                    if row[j] == end_tok:
                        break
                    j += 1
                i = j
            i += 1
    return tgt


class A2AModel(PreTrainedModel):
    """Agent A (writer + memory + transmitter) and agent B (receiver), joined only by the message."""

    config_class = A2AConfig
    base_model_prefix = "a2a"
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = False
    _supports_sdpa = True

    def __init__(self, config: A2AConfig):
        super().__init__(config)
        scfg = PerceiverARConfig(**config.sender_config)
        rcfg = PerceiverARConfig(**config.receiver_config)
        if rcfg.message_enabled:
            raise ValueError("the receiver reads the message as input positions: build it without the E21 boundary")
        self.sender_cfg = scfg
        # agent A
        self.sender_tok = nn.Embedding(scfg.vocab_size, scfg.token_embedding_dim)
        self.writer = LatentMemoryWriter(scfg)
        d_mem = 2 * int(scfg.lm_latent_dim)
        self.transmitter = Transmitter(config, d_mem, rcfg.hidden_size)
        # agent B
        self.receiver = PerceiverARLM(rcfg)
        self.post_init()

    def _init_weights(self, module):  # sub-models initialise themselves; keep their init
        return

    # ------------------------------------------------------------------ agent A
    def write(self, doc_ids: torch.Tensor, doc_mask: Optional[torch.Tensor] = None,
              chunk_pages: int = 0) -> tuple[LatentMemory, dict]:
        """Document tokens [B, S_A] → memory. Pages are written independently (the writer's own geometry).

        `chunk_pages` > 0 (one unpadded document, e.g. 1M tokens): write `chunk_pages` pages at a time. Chunks cut
        at page starts (multiples of the stride) and overlap by window − stride tokens, so the pages and every
        entry are the same as one pass over the whole document; memory for the writer stays bounded."""
        B, S = doc_ids.shape
        W, st = int(self.sender_cfg.lm_window), int(self.sender_cfg.lm_stride)
        if chunk_pages and B == 1 and S > 2 * chunk_pages * st and (doc_mask is None or bool(doc_mask.all())):
            T = chunk_pages * st
            parts, s0 = [], 0
            while True:
                end = min(S, s0 + T + (W - st))
                if S - end < W + st:  # the last chunk absorbs a short tail (keeps the global page geometry)
                    end = S
                parts.append(self._write_span(doc_ids[:, s0:end], None, s0)[0])
                if end == S:
                    break
                s0 += T
            return LatentMemory(torch.cat([m.entries for m in parts], 1), torch.cat([m.pos for m in parts], 1),
                                torch.cat([m.valid for m in parts], 1), parts[0].per_page), {}
        return self._write_span(doc_ids, doc_mask, 0)

    def _write_span(self, doc_ids, doc_mask, offset: int):
        B, S = doc_ids.shape
        if doc_mask is None:
            doc_mask = torch.ones_like(doc_ids)
        keep = doc_mask.bool()
        side = torch.zeros_like(doc_ids)
        doc = torch.where(keep, torch.zeros_like(doc_ids), torch.full_like(doc_ids, -1))
        pos = torch.arange(S, device=doc_ids.device)[None].expand(B, S) + int(offset)
        out = self.writer(self.sender_tok(doc_ids), side, doc, keep, pos, self.sender_cfg.rope_theta)
        entries = torch.cat([out["z"], out["read"]], -1)
        valid = out["latent_doc"] >= 0
        return LatentMemory(entries, out["latent_pos"], valid, int(self.sender_cfg.lm_latents)), out

    def send(self, memory: LatentMemory, request: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory (+ B's request [B, d_B] in request mode) → message [B, K, d_B]."""
        return self.transmitter(memory, request)

    # ------------------------------------------------------------------ agent B
    def request(self, q_ids: torch.Tensor, q_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """B's request: its question's words, pooled in B's own input-embedding space → [B, d_B] (576 numbers)."""
        e = self.receiver.embed(q_ids).float()
        m = (torch.ones_like(q_ids) if q_mask is None else q_mask).float()[..., None]
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

    def _question_spans(self, r_ids: torch.Tensor, r_mask: torch.Tensor) -> list:
        """Per receiver row: [(block start, question start, question end, block end)] — a block opens at
        "Question", its question words run to the "Answer" token, the block to the next "Question"."""
        qt, at = self.config.question_token_id, self.config.answer_token_id
        out = []
        for b in range(r_ids.shape[0]):
            row = r_ids[b, : int(r_mask[b].sum())].tolist()
            starts = [i for i, t in enumerate(row) if t == qt]
            spans = []
            for j, s0 in enumerate(starts):
                end = starts[j + 1] if j + 1 < len(starts) else len(row)
                a = next((i for i in range(s0 + 1, end) if row[i] == at), end)
                spans.append((s0, s0 + 2, a, end))  # skip "Question" ":"
            out.append(spans)
        return out

    def answer(self, message: torch.Tensor, recv_ids: torch.Tensor, recv_mask: Optional[torch.Tensor] = None,
               labels: Optional[torch.Tensor] = None):
        return self.receiver(input_ids=recv_ids, attention_mask=recv_mask, labels=labels, prefix_embeds=message,
                             return_logits=labels is None)

    def _targets(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.config.loss_on == "answers":
            return answer_targets(ids, mask, self.config.answer_token_id, self.config.end_token_id)
        return ids.masked_fill(~mask.bool(), -100)

    def empty_memory(self, B: int, device) -> LatentMemory:
        d = 2 * int(self.sender_cfg.lm_latent_dim)
        z = torch.zeros(B, 0, d, device=device)
        return LatentMemory(z, torch.zeros(B, 0, dtype=torch.long, device=device),
                            torch.zeros(B, 0, dtype=torch.bool, device=device), int(self.sender_cfg.lm_latents))

    # ------------------------------------------------------------------ training / scoring on packed rows
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, doc_ids: Optional[torch.Tensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None, return_logits: bool = False):
        """Rows `BOS doc EOS BOS recv EOS` → loss on the receiver's tokens only (labels are rebuilt from the
        receiver part: the document is never a target). Without labels, logits come back aligned to the input row
        (zeros on the document part)."""
        del doc_ids, cu_seqlens  # one sender document and one receiver text per row
        rc = self.receiver.config
        d_ids, d_mask, r_ids, r_mask = split_rows(input_ids, attention_mask, rc.bos_token_id, rc.eos_token_id,
                                                  rc.pad_token_id)
        if bool(d_mask.any()):
            memory, wout = self.write(d_ids, d_mask)
            keep = 0.0 * (wout["k"].float().sum() + wout["v"].float().sum())  # unused notebook heads (DDP)
        else:  # no document in this batch: keep every sender parameter in the graph (DDP)
            memory = self.empty_memory(input_ids.shape[0], input_ids.device)
            keep = 0.0 * (self.writer.participation().float().sum() + self.sender_tok.weight.sum())
        if self.config.tx_request == "question":
            return self._forward_request(input_ids, attention_mask, labels, memory, d_mask, r_ids, r_mask, keep)
        message = self.send(memory)
        if labels is not None:
            tgt = self._targets(r_ids, r_mask)
            out = self.answer(message, r_ids, r_mask, labels=tgt)
            return CausalLMOutput(loss=out.loss + keep, logits=None)
        out = self.answer(message, r_ids, r_mask)
        B, S = input_ids.shape
        logits = out.logits.new_zeros(B, S, out.logits.shape[-1])
        lens = attention_mask.sum(-1).tolist() if attention_mask is not None else [S] * B
        for b in range(B):
            n_r = int(r_mask[b].sum())
            end = int(lens[b])
            logits[b, end - n_r: end] = out.logits[b, :n_r]
        return CausalLMOutput(loss=None, logits=logits)


RECEIVER_OVERRIDES = {"attn_backend": "sdpa", "attn_pad_multiple": 64}


def _pad(seqs: list, pad: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    L = max([len(t) for t in seqs] + [1])
    ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
    for i, t in enumerate(seqs):
        ids[i, : len(t)] = torch.as_tensor(t, device=device)
        mask[i, : len(t)] = 1
    return ids, mask


def _request_forward(self, input_ids, attention_mask, labels, memory, d_mask, r_ids, r_mask, keep):
    """Request mode: every question block of a document row gets its own message, sent for B's request (the block's
    question words). Blocks run as separate short receiver rows `BOS Question: … Answer: …`; rows without a document
    (stories) run as before with the empty memory and no request. Loss = token-weighted mean over both."""
    rc = self.receiver.config
    dev = input_ids.device
    spans = self._question_spans(r_ids, r_mask)
    blk_rows, blk_seqs, q_seqs = [], [], []
    for b, sp in enumerate(spans):
        if not bool(d_mask[b].any()):
            continue
        row = r_ids[b].tolist()
        for (s0, qs, qe, e) in sp:
            blk_rows.append(b)
            blk_seqs.append([rc.bos_token_id] + row[s0:e])
            q_seqs.append(row[qs:qe] or [row[s0]])
    story = [b for b in range(r_ids.shape[0]) if not bool(d_mask[b].any())]
    total, n_tok, logits_parts = 0.0, 0, {}
    if blk_rows:
        idx = torch.tensor(blk_rows, device=dev)
        mem_b = LatentMemory(memory.entries.index_select(0, idx), memory.pos.index_select(0, idx),
                             memory.valid.index_select(0, idx), memory.per_page)
        q_ids, q_mask = _pad(q_seqs, rc.pad_token_id, dev)
        msg = self.send(mem_b, self.request(q_ids, q_mask))
        b_ids, b_mask = _pad(blk_seqs, rc.pad_token_id, dev)
        if labels is not None:
            tgt = self._targets(b_ids, b_mask)
            n = int((tgt[:, 1:] != -100).sum())
            total = total + self.answer(msg, b_ids, b_mask, labels=tgt).loss * n
            n_tok += n
        else:
            logits_parts["blocks"] = (self.answer(msg, b_ids, b_mask).logits, b_mask, blk_rows)
    if story:
        idx = torch.tensor(story, device=dev)
        s_ids, s_mask = r_ids.index_select(0, idx), r_mask.index_select(0, idx)
        msg = self.send(self.empty_memory(len(story), dev))
        if labels is not None:
            tgt = s_ids.masked_fill(~s_mask.bool(), -100)
            n = int((tgt[:, 1:] != -100).sum())
            total = total + self.answer(msg, s_ids, s_mask, labels=tgt).loss * n
            n_tok += n
        else:
            logits_parts["story"] = (self.answer(msg, s_ids, s_mask).logits, s_mask, story)
    if labels is not None:
        return CausalLMOutput(loss=total / max(n_tok, 1) + keep, logits=None)
    # scoring (one question per row: the block is the whole receiver part, BOS included) — logits back to the tail
    B, S = input_ids.shape
    lens = attention_mask.sum(-1).tolist() if attention_mask is not None else [S] * B
    out = torch.zeros(B, S, rc.vocab_size, device=dev)
    for lg, m, rows in logits_parts.values():
        for i, b in enumerate(rows):
            n = int(m[i].sum())
            out[b, int(lens[b]) - n: int(lens[b])] = lg[i, :n].float()
    return CausalLMOutput(loss=None, logits=out)


A2AModel._forward_request = _request_forward


def build_a2a_from_checkpoints(sender_dir: str, receiver_dir: str, receiver_overrides: Optional[dict] = None,
                               **a2a_kwargs) -> A2AModel:
    """Warm start: agent A's token embedding and writer from a notebook checkpoint (E31c / E33), agent B from a
    decoder checkpoint (dense); the transmitter starts fresh.

    `receiver_overrides` (default `RECEIVER_OVERRIDES`): the text-checks decoders were trained with flex attention and
    inputs padded to a multiple of 2,048; B's inputs are short (K + a question block), so padding them to 2,048 would
    waste 10× the compute, and flex would recompile for every length — SDPA with a 64-token pad multiple instead."""
    import json
    import os

    from safetensors.torch import load_file

    scfg = json.load(open(os.path.join(sender_dir, "config.json")))
    rcfg = json.load(open(os.path.join(receiver_dir, "config.json")))
    for c in (scfg, rcfg):
        for k in ("architectures", "transformers_version", "_name_or_path", "auto_map"):
            c.pop(k, None)
    rcfg.update(RECEIVER_OVERRIDES if receiver_overrides is None else receiver_overrides)
    model = A2AModel(A2AConfig(sender_config=scfg, receiver_config=rcfg,
                               pad_token_id=rcfg.get("pad_token_id"), bos_token_id=rcfg.get("bos_token_id"),
                               eos_token_id=rcfg.get("eos_token_id"), **a2a_kwargs))
    s_state = load_file(os.path.join(sender_dir, "model.safetensors"))
    model.sender_tok.weight.data.copy_(s_state["embed.tok.weight"])
    w_state = {k[len("memory_writer."):]: v for k, v in s_state.items() if k.startswith("memory_writer.")}
    missing, unexpected = model.writer.load_state_dict(w_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"writer warm start mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    r_state = load_file(os.path.join(receiver_dir, "model.safetensors"))
    missing, unexpected = model.receiver.load_state_dict(r_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"receiver warm start mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model
