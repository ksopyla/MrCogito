"""E36 — a trained writer for a frozen reader (docs/4_Research_Notes/e36_frozen_reader_writer_20261018.md).

Writer: the lower `writer_layers` decoder layers of a small pretrained LM (SmolLM2-135M by default), made
bidirectional, read each span of `span` tokens (512) and mean-pool it into ONE entry of the writer's width. The
borrowed token embedding stays frozen (it costs no compute and shares the reader's tokenizer).

Reader: a frozen pretrained decoder (SmolLM2-360M by default). Two ways in (`way`):
  input — each entry -> 2-layer MLP -> RMS-normalised x a learnable gain -> one input position of the reader
          (the gain starts at the reader's embedding scale and is free to grow);
  kv    — each entry -> a key and a value in every attention layer of the reader, RMS-normalised per head x a
          learnable per-layer gain (calibrated on the reader's own keys / values), keys rotated at the entry's
          position.
Entries carry the middle position of their span (`entry_positions='span_mid'`) or sit at 0..M-1 ('compact'); the
fetched entries are always given in source order.

Training rows: [BOS] + (mem_spans + 1) * span tokens + [EOS] from one long document.
  continuation   : the entries of spans 0..M-1 -> the reader predicts span M (tokens 1..S-1 scored);
  reconstruction : (a `recon_frac` share of training steps) the entry of one span -> the reader rebuilds it.
Shapes: B rows, M = mem_spans, S = span, D = writer width, d = reader width, L = reader layers, V = vocabulary.
  (B, M*S) -> writer -> (B, M, D) -> input: (B, M, d) | kv: L x [(B, h_kv, M, hd), (B, h_kv, M, hd)]
  -> reader over the target (B, S) -> logits (B, S, V).
Only the writer layers + output norm and the adapter train; the checkpoint stores only those (the reader and the
borrowed embedding are reloaded from their Hub ids).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

WAYS = ("input", "kv")
ENTRY_POSITIONS = ("span_mid", "compact")


class FrozenReaderMemoryConfig(PretrainedConfig):
    model_type = "frozen_reader_memory"

    def __init__(
        self,
        reader_name: str = "HuggingFaceTB/SmolLM2-360M",
        writer_init: str = "HuggingFaceTB/SmolLM2-135M",
        writer_layers: int = 3,
        span: int = 512,
        mem_spans: int = 8,
        way: str = "input",
        recon_frac: float = 0.2,
        entry_positions: str = "span_mid",
        proj_hidden: int = 2048,
        reader_dtype: str = "bfloat16",
        vocab_size: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if way not in WAYS:
            raise ValueError(f"way must be one of {WAYS}, got {way!r}")
        if entry_positions not in ENTRY_POSITIONS:
            raise ValueError(f"entry_positions must be one of {ENTRY_POSITIONS}, got {entry_positions!r}")
        if not 0.0 <= recon_frac < 1.0:
            raise ValueError("recon_frac must be in [0, 1)")
        self.reader_name, self.writer_init, self.writer_layers = reader_name, writer_init, int(writer_layers)
        self.span, self.mem_spans, self.way = int(span), int(mem_spans), way
        self.recon_frac, self.entry_positions = float(recon_frac), entry_positions
        self.proj_hidden, self.reader_dtype, self.vocab_size = int(proj_hidden), reader_dtype, int(vocab_size)
        self.checkpoint_family = "frozen_reader"
        self.pretraining_objective = "causal_lm"


class SpanWriter(nn.Module):
    """Lower layers of a pretrained decoder, bidirectional within a span, mean-pooled to one entry.

    forward: span ids (N, S) -> entries (N, D)."""

    def __init__(self, base, n_layers: int):
        super().__init__()
        if not 1 <= n_layers <= len(base.layers):
            raise ValueError(f"writer_layers={n_layers} outside 1..{len(base.layers)}")
        self.embed_tokens = base.embed_tokens
        self.embed_tokens.requires_grad_(False)  # borrowed, frozen: no compute, shared tokenizer
        self.layers = nn.ModuleList(base.layers[:n_layers])
        for layer in self.layers:
            layer.self_attn.is_causal = False  # SDPA with no mask -> full attention within the span
        self.rotary_emb = base.rotary_emb
        self.norm = nn.RMSNorm(base.config.hidden_size, eps=1e-6)
        self.hidden_size = base.config.hidden_size

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n, s = ids.shape
        h = self.embed_tokens(ids).to(self.norm.weight.dtype)
        pos = torch.arange(s, device=ids.device).unsqueeze(0).expand(n, -1)
        pe = self.rotary_emb(h, pos)
        for layer in self.layers:
            out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
            h = out[0] if isinstance(out, tuple) else out
        return self.norm(h.mean(dim=1))


class InputProjector(nn.Module):
    """Way 'input': entry (…, D) -> one reader input vector (…, d)."""

    def __init__(self, d_in: int, d_out: int, hidden: int, gain: float):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_in, hidden), nn.Linear(hidden, d_out)
        self.log_gain = nn.Parameter(torch.tensor(math.log(max(gain, 1e-6))))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        x = self.fc2(F.gelu(self.fc1(e)))
        return F.rms_norm(x, (x.shape[-1],)) * self.log_gain.exp()


class KVExpander(nn.Module):
    """Way 'kv': entry (B, M, D) -> per-layer keys and values (B, M, L, 2, h_kv, hd)."""

    def __init__(self, d_in: int, n_layers: int, n_kv: int, head_dim: int, gains: torch.Tensor):
        super().__init__()
        self.shape = (n_layers, 2, n_kv, head_dim)
        self.proj = nn.Linear(d_in, n_layers * 2 * n_kv * head_dim)
        self.log_gain = nn.Parameter(gains.clamp_min(1e-6).log().float())  # (L, 2)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        x = self.proj(e).view(*e.shape[:-1], *self.shape)
        return F.rms_norm(x, (x.shape[-1],)) * self.log_gain.exp()[..., None, None]


class FrozenReaderMemory(PreTrainedModel):
    config_class = FrozenReaderMemoryConfig
    base_model_prefix = "frozen_reader_memory"
    _SKIP = ("reader.", "writer.embed_tokens.")  # reloaded from the Hub, never saved

    def __init__(self, config: FrozenReaderMemoryConfig, reader=None, writer_base=None):
        super().__init__(config)
        if reader is None:
            reader = AutoModelForCausalLM.from_pretrained(
                config.reader_name, torch_dtype=getattr(torch, config.reader_dtype), attn_implementation="sdpa")
        if writer_base is None:
            writer_base = AutoModelForCausalLM.from_pretrained(
                config.writer_init, torch_dtype=torch.float32, attn_implementation="sdpa").model
        reader.requires_grad_(False)
        self.reader = reader.eval()
        self.writer = SpanWriter(writer_base, config.writer_layers)
        rc = reader.config
        if config.vocab_size and config.vocab_size != rc.vocab_size:
            raise ValueError(f"config vocab {config.vocab_size} != reader vocab {rc.vocab_size}")
        config.vocab_size = rc.vocab_size
        d_w = self.writer.hidden_size
        wc = writer_base.config  # shape fields read by the shared training logs / W&B config
        config.hidden_size, config.token_embedding_dim, config.num_hidden_layers = d_w, d_w, config.writer_layers
        config.intermediate_size, config.num_attention_heads = wc.intermediate_size, wc.num_attention_heads
        config.concept_num, config.max_sequence_length = config.mem_spans, (config.mem_spans + 1) * config.span
        config.hidden_act, config.norm_type = wc.hidden_act, "rmsnorm"
        if config.way == "input":
            emb = reader.get_input_embeddings().weight.detach().float()
            self.adapter = InputProjector(d_w, rc.hidden_size, config.proj_hidden, emb.pow(2).mean().sqrt().item())
        else:
            head_dim = getattr(rc, "head_dim", None) or rc.hidden_size // rc.num_attention_heads
            self.adapter = KVExpander(d_w, rc.num_hidden_layers, rc.num_key_value_heads, head_dim,
                                      self._calibrate_kv_rms())
        self._fwd = 0
        self.post_init()

    def _init_weights(self, module):  # the sub-models initialise themselves; keep their init
        pass

    def train(self, mode: bool = True):
        super().train(mode)
        self.reader.eval()  # frozen: always in inference mode
        return self

    @torch.no_grad()
    def _calibrate_kv_rms(self) -> torch.Tensor:
        """Per-layer RMS of the reader's own keys and values on random tokens -> (L, 2)."""
        rc = self.reader.config
        dev = next(self.reader.parameters()).device
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(100, max(rc.vocab_size - 100, 101), (2, 64), generator=g).to(dev)
        cache = self.reader(input_ids=ids, use_cache=True).past_key_values
        out = []
        for i in range(rc.num_hidden_layers):
            k, v = _cache_kv(cache, i)
            out.append([k.float().pow(2).mean().sqrt(), v.float().pow(2).mean().sqrt()])
        return torch.tensor(out).cpu()

    # -- save only what trains -------------------------------------------------------------------------------
    def state_dict(self, *args, destination=None, prefix: str = "", keep_vars: bool = False):
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        for k in [k for k in sd if k[len(prefix):].startswith(self._SKIP)]:
            del sd[k]
        return sd

    # -- the three operations --------------------------------------------------------------------------------
    def write(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, M*S) token ids -> (B, M, D) entries, one per span."""
        b, n = ids.shape
        s = self.config.span
        if n % s:
            raise ValueError(f"{n} tokens is not a whole number of {s}-token spans")
        return self.writer(ids.reshape(b * (n // s), s)).view(b, n // s, -1)

    def entry_pos(self, first_span: int, m: int, device) -> torch.Tensor:
        """Positions of entries for spans first_span..first_span+m-1 -> (m,)."""
        idx = torch.arange(first_span, first_span + m, device=device)
        return idx * self.config.span + self.config.span // 2 if self.config.entry_positions == "span_mid" else idx

    def recon_positions(self, j: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuilding span j from its own entry: (entry position (1,), target positions (S,))."""
        s = self.config.span
        if self.config.entry_positions == "span_mid":
            return self.entry_pos(j, 1, device), torch.arange(s, device=device) + j * s
        return torch.zeros(1, dtype=torch.long, device=device), torch.arange(s, device=device) + 1

    def kv_cache(self, entries: torch.Tensor, entry_pos: torch.Tensor) -> DynamicCache:
        x = self.adapter(entries)  # (B, M, L, 2, h, hd)
        b, m = x.shape[:2]
        dt = self.reader.dtype
        cos, sin = self.reader.model.rotary_emb(x[:, :, 0, 0, 0, :].to(dt), entry_pos.expand(b, -1))
        cache = DynamicCache()
        for i in range(x.shape[2]):
            k = x[:, :, i, 0].transpose(1, 2).to(dt)  # (B, h, M, hd)
            v = x[:, :, i, 1].transpose(1, 2).to(dt)
            k, _ = apply_rotary_pos_emb(k, k, cos, sin)
            cache.update(k, v, i)
        return cache

    def read_logits(self, entries: Optional[torch.Tensor], tgt: torch.Tensor, tgt_pos: torch.Tensor,
                    entry_pos: Optional[torch.Tensor] = None, raw_prefix: Optional[torch.Tensor] = None,
                    raw_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The frozen reader over `tgt` (B, T), given entries (B, M, D) or raw prefix tokens or nothing.
        Returns logits (B, T, V): logits[:, t] predicts tgt[:, t + 1]."""
        r, (b, t) = self.reader, tgt.shape
        if raw_prefix is not None:
            ids = torch.cat([raw_prefix, tgt], 1)
            pos = torch.cat([raw_pos, tgt_pos], 0)
            return r(input_ids=ids, position_ids=pos.expand(b, -1), attention_mask=torch.ones_like(ids),
                     use_cache=False).logits[:, raw_prefix.shape[1]:]
        if entries is None or entries.shape[1] == 0:
            return r(input_ids=tgt, position_ids=tgt_pos.expand(b, -1), attention_mask=torch.ones_like(tgt),
                     use_cache=False).logits
        m = entries.shape[1]
        if self.config.way == "input":
            x = torch.cat([self.adapter(entries).to(r.dtype), r.get_input_embeddings()(tgt)], 1)
            pos = torch.cat([entry_pos, tgt_pos], 0).expand(b, -1)
            # an explicit all-ones mask: without it transformers reads the non-contiguous positions as packed rows
            mask = torch.ones(b, m + t, dtype=torch.long, device=tgt.device)
            return r(inputs_embeds=x, position_ids=pos, attention_mask=mask, use_cache=False).logits[:, m:]
        cache = self.kv_cache(entries, entry_pos)
        mask = torch.ones(b, m + t, dtype=torch.long, device=tgt.device)
        cache_position = torch.arange(m, m + t, device=tgt.device)
        return r(input_ids=tgt, position_ids=tgt_pos.expand(b, -1), past_key_values=cache,
                 cache_position=cache_position, attention_mask=mask, use_cache=True).logits

    # -- training ---------------------------------------------------------------------------------------------
    def _recon_step(self) -> tuple[bool, int]:
        """Same draw on every rank (a counter of forward calls, identical across DDP ranks)."""
        self._fwd += 1
        u = ((self._fwd * 2654435761) % 2**32) / 2**32
        j = (self._fwd * 40503) % max(self.config.mem_spans, 1)
        return u < self.config.recon_frac, j

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **_) -> CausalLMOutput:
        s, m = self.config.span, self.config.mem_spans
        need = (m + 1) * s
        ids = input_ids[:, 1:1 + need]  # drop BOS
        if ids.shape[1] < need:
            raise ValueError(f"rows need {need} tokens after BOS, got {ids.shape[1]}")
        dev = ids.device
        recon, j = self._recon_step() if self.training else (False, 0)
        span_mid = self.config.entry_positions == "span_mid"
        if recon:
            tgt = ids[:, j * s:(j + 1) * s]
            entries = self.write(tgt)
            epos, tpos = self.recon_positions(j, dev)
        else:
            tgt = ids[:, m * s:]
            entries = self.write(ids[:, :m * s])
            epos = self.entry_pos(0, m, dev)
            tpos = torch.arange(s, device=dev) + (m * s if span_mid else m)
        logits = self.read_logits(entries, tgt, tpos, epos)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
        return CausalLMOutput(loss=loss, logits=None)


def _cache_kv(cache, i: int):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def load_frozen_reader_memory(path: str, device: str = "cpu") -> FrozenReaderMemory:
    """Rebuild from config (reader and writer base from the Hub) and load the trained weights."""
    cfg = FrozenReaderMemoryConfig.from_pretrained(path)
    model = FrozenReaderMemory(cfg)
    sf = os.path.join(path, "model.safetensors")
    if os.path.exists(sf):
        from safetensors.torch import load_file
        sd = load_file(sf)
    else:
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(FrozenReaderMemory._SKIP)]
    if bad or unexpected:
        raise ValueError(f"checkpoint mismatch: missing {bad[:5]} unexpected {list(unexpected)[:5]}")
    return model.to(device).eval()
