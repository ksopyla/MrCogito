"""Information-theoretic scores for the BAPO capability ladder.

BAPO (Schnabel et al., NeurIPS 2025) cannot *measure* effective prefix bandwidth `a` inside a
transformer; it only operationalizes it via task success. These quantities are the missing
measurement: recovered nats against a closed-form floor, converted to bits, bits/token, and
bytes/token, plus the nominal cache bandwidth of each architecture.

Effective prefix bandwidth `a` (bits) ≈ recovered_bits on a task whose evidence is invisible
to the raw window. Attention bandwidth `b` (tokens) is the raw window (E18 stack `block`,
dense = seq_len, encdec decoder self-attn = suffix only).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from data.symbolic_tasks import SymbolicTaskConfig, chance_accuracy, chance_entropy_nats, floor_nats, prize_bits

LN2 = math.log(2)
BYTES_PER_ELEM = 2  # bf16 cache, matching analysis/geometry_cost_model.py


@dataclass(frozen=True)
class InfoReport:
    ce_nats: float
    acc: float
    n_supervised: int
    floor_nats: float
    chance_acc: float
    prize_nats: float
    prize_bits: float
    recovered_nats: float
    recovered_bits: float
    information_flow: float          # recovered / prize in [0, 1]
    bits_per_supervised_token: float
    bits_per_input_token: float
    bytes_per_input_token: float
    seq_len: int
    answer_len: int
    nominal_b_tokens: int            # raw-token attention bandwidth
    nominal_a_bytes: float           # unbounded KV cache bytes / token
    effective_a_bits: float          # recovered_bits (measured prefix bandwidth)

    def as_dict(self) -> dict:
        return asdict(self)


def info_report(
    *,
    ce_nats: float,
    acc: float,
    n_supervised: int,
    cfg: SymbolicTaskConfig,
    window: int,
    nominal_b_tokens: int,
    nominal_a_bytes: float,
) -> InfoReport:
    floor = floor_nats(cfg, window)
    prize_n = cfg.answer_len * chance_entropy_nats(cfg)
    recovered = max(0.0, floor - ce_nats)
    # A model that beats the floor by going *below* 0 nats (perfect) recovers the whole prize.
    # recovered is per supervised token (same units as ce/floor). Scale to the span:
    recovered_span = recovered * cfg.answer_len
    prize_b = prize_bits(cfg)
    rec_bits = recovered_span / LN2
    flow = 0.0 if prize_n <= 0 else min(1.0, recovered_span / prize_n)
    return InfoReport(
        ce_nats=float(ce_nats),
        acc=float(acc),
        n_supervised=int(n_supervised),
        floor_nats=float(floor),
        chance_acc=float(chance_accuracy(cfg)),
        prize_nats=float(prize_n),
        prize_bits=float(prize_b),
        recovered_nats=float(recovered_span),
        recovered_bits=float(rec_bits),
        information_flow=float(flow),
        bits_per_supervised_token=float(recovered / LN2),
        bits_per_input_token=float(rec_bits / cfg.seq_len),
        bytes_per_input_token=float(rec_bits / 8.0 / cfg.seq_len),
        seq_len=cfg.seq_len,
        answer_len=cfg.answer_len,
        nominal_b_tokens=int(nominal_b_tokens),
        nominal_a_bytes=float(nominal_a_bytes),
        effective_a_bits=float(rec_bits),
    )


def kv_bytes_per_item(n_kv_heads: int, head_dim: int, bytes_per: int = BYTES_PER_ELEM) -> int:
    return 2 * n_kv_heads * head_dim * bytes_per


def cache_profile(arch: str, *, n_layers: int, global_layers: int, n_kv_heads: int, head_dim: int,
                  local_window: int, seq_len: int, enc_layers: int = 0, dec_layers: int = 0,
                  compress_ratio: int = 1) -> dict:
    """Nominal BAPO (a, b) for an architecture, in cache bytes and raw tokens.

    `a` is reported as unbounded KV bytes per token of context (what grows with S).
    `b` is the number of raw tokens the answer-emitting stream can attend to.
    """
    item = kv_bytes_per_item(n_kv_heads, head_dim)
    if arch == "dense":
        return {
            "nominal_b_tokens": seq_len,
            "nominal_a_bytes": n_layers * item,
            "bounded_kv_bytes": 0,
            "unbounded_layers": n_layers,
        }
    if arch == "e18":
        return {
            "nominal_b_tokens": seq_len,  # one full causal layer sees the whole prefix
            "nominal_a_bytes": global_layers * item,
            "bounded_kv_bytes": (n_layers - global_layers) * item * local_window,
            "unbounded_layers": global_layers,
        }
    if arch in {"e21", "e30", "e30_ctx", "e31_page", "e31_bixt"}:
        r = max(float(compress_ratio), 1e-6)
        return {
            "nominal_b_tokens": local_window,  # suffix self-attn is local; prefix is slots only
            "nominal_a_bytes": global_layers * item / r,
            "bounded_kv_bytes": (n_layers - global_layers) * item * local_window,
            "unbounded_layers": global_layers,
        }
    if arch == "e18_local":
        return {
            "nominal_b_tokens": local_window,
            "nominal_a_bytes": 0.0,
            "bounded_kv_bytes": n_layers * item * local_window,
            "unbounded_layers": 0,
        }
    if arch == "encdec":
        # Decoder self-attn is suffix-only (b ≈ answer_len); prefix crosses only as encoder
        # memory, whose KV is n_enc * item bytes per prefix token.
        return {
            "nominal_b_tokens": 0,  # decoder cannot attend raw prefix
            "nominal_a_bytes": enc_layers * item,
            "bounded_kv_bytes": dec_layers * item * local_window,
            "unbounded_layers": enc_layers,
        }
    raise ValueError(f"unknown arch {arch!r}")
