"""🤗 Transformers interface for the Perceiver AR family (`nn/perceiver_ar_lm.py`): loading with
`AutoModelForCausalLM` and 🤗 `generate` over the cached decoder (`nn/incremental.py`), 2026-11-18.

Engineering spec: docs/engineering_specs/hub_from_pretrained_loading.md. `scripts/export_hub_model.py` vendors this
file, with the modules a checkpoint uses, into the flat `modeling_perceiver_ar.py` of a Hub repository, so a reader
loads a release with

    AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)

and scores text with `labels=` or samples with `generate` (greedy, sampling, beam search, repetition penalty, ...).

Scoring (`labels`, 🤗 convention: the model shifts them, −100 is ignored): the mean next-token cross-entropy of the
trained distribution (tanh soft-capped logits), the per-token loss `load_perceiver_ar_for_eval` gives
(`return_per_token_loss`); the training z-loss is not added. Padding (attention_mask 0) becomes document id −1, as in
the harness: padded keys are never read, positions restart at the first real token, padded targets are ignored.

Generation: `PerceiverARCache` is the `past_key_values` 🤗 generate carries between steps. Its first forward is the
cached decoder's prefill over one fixed-length row (the prompt, then free slots written as eos with document id −1,
`GEN_PAD`-rounded: the harness geometry, `PerceiverARLMEval._model_generate`); every later forward runs only the new
position. The logits are the harness's (`F.linear(h, lm_head.weight)`, soft-capped), so 🤗 greedy writes the tokens
of `IncrementalDecoder.greedy`. When the row is full, or a notebook read reaches its exact horizon
(`IncrementalDecoder.max_new_tokens`), the cache prefills again over the tokens written so far: the state a fresh
prefill of those tokens has. Beam search reorders the cache's rows (`IncrementalDecoder.reorder`). Reads the cached
decoder does not cover (`decode_kind` refuses, and the catalogue / notes reads under beam search) fall back to the
harness's uncached loop: every step reads the whole row again.

`use_cache=False` (in `generate` or a plain call) is the model's own forward over the given ids (the scoring path).
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from nn.incremental import IncrementalDecoder
from nn.perceiver_ar_lm import PerceiverARLM, per_token_ce_chunked

GEN_PAD = 256  # row lengths are multiples of this (PerceiverARLMEval.GEN_PAD): flex attention compiles per shape


def _round_up(n: int, m: int) -> int:
    return -(-int(n) // m) * m


def _softcap(logits: torch.Tensor, cap) -> torch.Tensor:
    return cap * torch.tanh(logits / cap) if cap else logits


def _mask_to_doc(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """attention_mask [B, S] → document ids (0 = token, −1 = padding); None when nothing is padded (the model's own
    path for a plain row, so scoring is bit-identical to a call without a mask)."""
    if attention_mask is None or bool(attention_mask.bool().all()):
        return None
    return torch.where(attention_mask.bool(), 0, -1).long()


class PerceiverARCache:
    """The decoder state 🤗 `generate` carries in `past_key_values` (one per `generate` call).

    `dec` is the `IncrementalDecoder` after its prefill (None before it, or when `uncached`); `headroom` the free slots
    a prefill leaves after the prompt (`generate`'s max_new_tokens when known); `since` the tokens written since the
    last prefill; `W` the output matrix [V, H] the steps multiply by (computed once)."""

    is_compileable = False
    to_legacy_cache = None  # read by generate's `return_legacy_cache` check

    def __init__(self, headroom: int = GEN_PAD):
        self.headroom = max(1, int(headroom))
        self.dec: Optional[IncrementalDecoder] = None
        self.uncached = False
        self.row_len = 0  # uncached rows: the fixed row length (grown when the answer passes it)
        self.since = 0
        self.W: Optional[torch.Tensor] = None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0 if self.dec is None else int(self.dec.t)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        if self.dec is None:
            return
        try:
            self.dec.reorder(beam_idx)
        except NotImplementedError:  # a notebook read: the rest of this generate call reads the whole row per step
            self.dec, self.uncached = None, True


class PerceiverARForCausalLM(PerceiverARLM, GenerationMixin):
    """`PerceiverARLM` with the 🤗 causal-LM interface: the same modules and state-dict keys, so a `perceiver_ar`
    checkpoint loads as it is; `forward` takes `labels`, `attention_mask`, `past_key_values`, `use_cache` and
    `logits_to_keep`, and `generate` is 🤗's (the repo's own uncached probe sampler is `PerceiverARLM.generate`)."""

    _gen_headroom: Optional[int] = None

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False  # the cache is ours: made by prepare_inputs_for_generation / the first forward

    def generate(self, *args, **kwargs):
        """🤗 `GenerationMixin.generate`. Reads `max_new_tokens` only to size the cached row once (no re-prefill)."""
        gc = kwargs.get("generation_config")
        if gc is None and len(args) > 1:
            gc = args[1]
        new = kwargs.get("max_new_tokens", getattr(gc if gc is not None else self.generation_config,
                                                    "max_new_tokens", None))
        self._gen_headroom = int(new) if new else GEN_PAD
        try:
            return GenerationMixin.generate(self, *args, **kwargs)
        finally:
            self._gen_headroom = None

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True,
                                      logits_to_keep=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            raise ValueError("PerceiverARForCausalLM reads token ids; inputs_embeds is not supported")
        if use_cache and not isinstance(past_key_values, PerceiverARCache):
            if past_key_values is not None and int(getattr(past_key_values, "get_seq_length", lambda: 0)() or 0):
                raise ValueError(f"past_key_values must be a PerceiverARCache, got {type(past_key_values).__name__}")
            past_key_values = PerceiverARCache(self._gen_headroom or GEN_PAD)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values,
                "use_cache": use_cache, "logits_to_keep": logits_to_keep}

    # The parameters 🤗 generate may pass are listed by name: no **kwargs, because the 🤗 Trainer treats a forward with
    # **kwargs as one that normalises its own loss over the accumulation steps (PerceiverARLM.forward's note).
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[PerceiverARCache] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor, None] = 0,
        doc_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,  # ignored: positions restart at each document start
        cache_position: Optional[torch.Tensor] = None,  # ignored: the cache counts the written positions itself
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is not None or input_ids is None:
            raise ValueError("PerceiverARForCausalLM reads token ids (input_ids); inputs_embeds is not supported")
        if output_attentions or output_hidden_states:
            raise ValueError("output_attentions / output_hidden_states are not returned by this model")
        keep = 0 if logits_to_keep is None else logits_to_keep
        if use_cache is None:
            use_cache = past_key_values is not None
        cfg = self.config
        if use_cache and labels is None:
            if not isinstance(past_key_values, PerceiverARCache):
                if past_key_values is not None and int(past_key_values.get_seq_length() or 0):
                    raise ValueError("past_key_values must be a PerceiverARCache")
                past_key_values = PerceiverARCache()
            h = self._cached_hidden(input_ids, attention_mask, past_key_values, keep)
            if isinstance(keep, int) and keep > 0:
                h = h[:, -keep:]
            elif torch.is_tensor(keep):
                h = h[:, keep]
            if past_key_values.W is None:
                past_key_values.W = self.lm_head.weight.detach()
            logits = _softcap(F.linear(h, past_key_values.W).float(), cfg.logit_softcap)
            return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

        doc = doc_ids if doc_ids is not None else _mask_to_doc(attention_mask)
        h = self.hidden_states(input_ids, doc_ids=doc)
        head_w, head_in = self._head_parts()
        loss = None
        if labels is not None:
            tgt = labels.to(h.device)[:, 1:]
            if attention_mask is not None:  # no padded target, and no target read from a padded position
                m = attention_mask.to(h.device).bool()
                tgt = tgt.masked_fill(~(m[:, 1:] & m[:, :-1]), -100)
            per = per_token_ce_chunked(head_in(h[:, :-1]), head_w, tgt, cfg.chunked_ce_block_size, cfg.logit_softcap)
            loss = per.sum() / (tgt != -100).sum().clamp(min=1)
        if isinstance(keep, int) and keep > 0:
            h = h[:, -keep:]
        elif torch.is_tensor(keep):
            h = h[:, keep]
        logits = _softcap(F.linear(head_in(h), head_w).float(), cfg.logit_softcap)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ------------------------------------------------------------------ cached decoding
    def _eos(self) -> int:
        e = self.config.eos_token_id
        if isinstance(e, (list, tuple)):
            e = e[0] if e else None
        return int(self.config.pad_token_id if e is None else e)

    def _row(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], S: int):
        """The harness row: ids in [0, L), eos after; document ids 0 (−1 where the mask is 0), −1 after."""
        B, L = input_ids.shape
        buf = input_ids.new_full((B, S), self._eos())
        buf[:, :L] = input_ids
        doc = torch.full((B, S), -1, dtype=torch.long, device=input_ids.device)
        doc[:, :L] = 0 if attention_mask is None else torch.where(attention_mask.to(input_ids.device).bool(), 0, -1)
        return buf, doc

    def _cached_hidden(self, input_ids, attention_mask, cache: PerceiverARCache, keep) -> torch.Tensor:
        """Final-norm hidden states [B, n, d] of the positions this call adds (after a prefill: the prompt's last
        position when `keep` is 1, as in 🤗 generate, else every prompt position)."""
        B, L = input_ids.shape
        if cache.uncached:
            return self._uncached_hidden(input_ids, attention_mask, cache)
        dec = cache.dec
        n_new = L - (int(dec.t) if dec is not None else 0)
        limit = dec.max_new_tokens() if dec is not None else None
        if (dec is None or n_new <= 0 or L > dec.S or dec.B != B
                or (limit is not None and cache.since + n_new > limit)):
            try:
                dec = IncrementalDecoder(self)
            except NotImplementedError:
                cache.dec, cache.uncached = None, True
                return self._uncached_hidden(input_ids, attention_mask, cache)
            buf, doc = self._row(input_ids, attention_mask, _round_up(L + cache.headroom, GEN_PAD))
            all_positions = not (isinstance(keep, int) and keep == 1)
            h = dec.prefill(buf, doc, L, all_positions=all_positions)
            cache.dec, cache.since = dec, 0
            return h if all_positions else h[:, None]
        hs = [dec.step(input_ids[:, j]) for j in range(int(dec.t), L)]
        cache.since += n_new
        return torch.stack(hs, 1)

    @torch.no_grad()
    def _uncached_hidden(self, input_ids, attention_mask, cache: PerceiverARCache) -> torch.Tensor:
        """The harness's uncached step: the whole row again, one fixed length for the answer (grown by `GEN_PAD` when
        the answer passes it); the hidden state of the last position [B, 1, d]."""
        L = input_ids.shape[1]
        if cache.row_len < L:
            cache.row_len = _round_up(L + (cache.headroom if cache.row_len == 0 else GEN_PAD), GEN_PAD)
        buf, doc = self._row(input_ids, attention_mask, cache.row_len)
        return self.hidden_states(buf, doc_ids=doc)[:, L - 1:L]


__all__ = ["GEN_PAD", "PerceiverARCache", "PerceiverARForCausalLM"]
