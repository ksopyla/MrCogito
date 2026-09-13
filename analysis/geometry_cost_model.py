"""Analytic cost model for concept-array geometries vs a matched dense baseline.

Answers, for a given context length S, the two questions every spec has to justify:
  1. compute  — forward FLOPs, split into the terms that scale as S^2 and the terms that scale as S;
  2. state    — bytes of decode-time cache per token, split into bounded (fixed-size windows) and
                unbounded (grows with S) so the 1M/10M claims can be checked rather than asserted.

The dense reference is `perceiver_ar --par_mode dense`: every layer full causal at the same width.
The concept reference is `perceiver_concept` (E22/E23): a windowed encoder, r:1 pooling, a causal
latent stack over slots, and a decoder whose self-attention is confined to a segment plus
cross-attention to the slots.

Run:
    uv run python analysis/geometry_cost_model.py
    uv run python analysis/geometry_cost_model.py --seq 1048576 --no-flops
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

BYTES_PER_ELEM = 2  # bf16 cache


@dataclass(frozen=True)
class Width:
    """Per-layer width, shared by both families in the E18/E22 pilot geometry."""

    d: int = 768
    n_q_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 128
    ff: int = 2048

    @property
    def q_dim(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def score_flops_per_pair(self) -> int:
        """QK^T then AV, over every query head: 2 MAC-chains x 2 FLOPs x q_dim."""
        return 4 * self.q_dim

    @property
    def proj_flops_per_item(self) -> int:
        """Q/K/V/O projections for one self-attention layer, per item."""
        w = self.d * self.q_dim + 2 * self.d * self.kv_dim + self.q_dim * self.d
        return 2 * w

    @property
    def mlp_flops_per_item(self) -> int:
        """SwiGLU: gate + up + down."""
        return 2 * 3 * self.d * self.ff

    @property
    def kv_bytes_per_item_per_layer(self) -> int:
        return 2 * self.kv_dim * BYTES_PER_ELEM


def _causal_pairs(n: int) -> float:
    return n * (n + 1) / 2


def _windowed_pairs(n: int, window: int) -> float:
    """Causal attention with a sliding window of `window` keys (including self)."""
    if window >= n:
        return _causal_pairs(n)
    return _causal_pairs(window) + (n - window) * window


def _segment_pairs(n: int, segment: int) -> float:
    """Causal attention that resets at every segment boundary (no chaining)."""
    full, rest = divmod(n, segment)
    return full * _causal_pairs(segment) + _causal_pairs(rest)


def dense_cost(seq: int, layers: int, w: Width) -> dict[str, float]:
    quad = layers * w.score_flops_per_pair * _causal_pairs(seq)
    lin = layers * seq * (w.proj_flops_per_item + w.mlp_flops_per_item)
    return {
        "flops_quadratic": quad,
        "flops_linear": lin,
        "flops": quad + lin,
        "state_unbounded_bytes_per_token": layers * w.kv_bytes_per_item_per_layer,
        "state_bounded_bytes": 0.0,
    }


def concept_cost(
    seq: int,
    w: Width,
    *,
    enc_layers: int = 6,
    enc_window: int = 512,
    ratio: int = 16,
    slots_per_block: int = 1,
    latent_layers: int = 4,
    latent_window: int | None = None,
    dec_layers: int = 8,
    dec_segment: int = 1024,
    exclusive: bool = False,
    cross_topk: int | None = None,
) -> dict[str, float]:
    n_slots = (seq // ratio) * slots_per_block

    enc_quad = enc_layers * w.score_flops_per_pair * _windowed_pairs(seq, enc_window)
    latent_pairs = (
        _causal_pairs(n_slots) if latent_window is None else _windowed_pairs(n_slots, latent_window)
    )
    latent_quad = latent_layers * w.score_flops_per_pair * latent_pairs
    dec_self_quad = dec_layers * w.score_flops_per_pair * _segment_pairs(seq, dec_segment)

    # Cross-attention: every token queries the slots it is allowed to see. Under `causal` scope that
    # is every slot ending at or before the token (~n_slots/2 on average); under `exclusive` scope it
    # is only the slots strictly before the token's own segment, which removes the in-segment slots.
    # A dense read over the visible slots leaves the term quadratic in S (constant r/2 smaller); a
    # selective read that scores only `cross_topk` slots per token makes it linear in S instead.
    visible = n_slots / 2 if not exclusive else max(n_slots / 2 - dec_segment / (2 * ratio), 0.0)
    if cross_topk is not None:
        visible = min(visible, float(cross_topk))
    cross_quad = dec_layers * w.score_flops_per_pair * seq * visible

    token_items = (enc_layers + dec_layers) * seq
    slot_items = latent_layers * n_slots
    lin = (token_items + slot_items) * (w.proj_flops_per_item + w.mlp_flops_per_item)
    # Cross-attention projections: Q per token, K/V per slot, O per token.
    lin += dec_layers * 2 * (seq * (w.d * w.q_dim + w.q_dim * w.d) + n_slots * 2 * w.d * w.kv_dim)

    quad = enc_quad + latent_quad + dec_self_quad + cross_quad
    # Only the array grows with S: the encoder window and decoder segment are fixed, and the latent
    # stack's K/V (and the decoder's cross K/V) are recomputable from the stored array.
    array_bytes_per_token = w.d * BYTES_PER_ELEM * slots_per_block / ratio
    latent_kv_per_token = latent_layers * w.kv_bytes_per_item_per_layer * slots_per_block / ratio
    cross_kv_per_token = dec_layers * w.kv_bytes_per_item_per_layer * slots_per_block / ratio
    return {
        "flops_quadratic": quad,
        "flops_enc": enc_quad,
        "flops_latent": latent_quad,
        "flops_dec_self": dec_self_quad,
        "flops_cross": cross_quad,
        "flops_linear": lin,
        "flops": quad + lin,
        "state_unbounded_bytes_per_token": array_bytes_per_token,
        "state_unbounded_cached_bytes_per_token": array_bytes_per_token
        + latent_kv_per_token
        + cross_kv_per_token,
        "state_bounded_bytes": (
            enc_layers * w.kv_bytes_per_item_per_layer * enc_window
            + dec_layers * w.kv_bytes_per_item_per_layer * dec_segment
        ),
        "n_slots": float(n_slots),
    }


def _fmt(x: float) -> str:
    for unit, scale in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.3g}{unit}"
    return f"{x:.3g}"


def _bytes(x: float) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if abs(x) >= scale:
            return f"{x / scale:.3g} {unit}"
    return f"{x:.0f} B"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", type=int, nargs="+", default=[32768, 1 << 20, 10 << 20])
    p.add_argument("--dense_layers", type=int, default=18)
    p.add_argument("--ratio", type=int, default=16)
    p.add_argument("--dec_segment", type=int, default=1024)
    p.add_argument("--enc_window", type=int, default=512)
    p.add_argument("--exclusive", action="store_true", help="E23 cross-attention scope")
    p.add_argument(
        "--cross_topk",
        type=int,
        default=None,
        help="hypothetical selective read: score at most K slots per token instead of all visible",
    )
    p.add_argument(
        "--latent_window",
        type=int,
        default=None,
        help="hypothetical windowed latent stack instead of full causal over all slots",
    )
    p.add_argument("--no-flops", dest="flops", action="store_false")
    args = p.parse_args()

    w = Width()
    print(
        f"width d={w.d} q={w.n_q_heads}x{w.head_dim} kv={w.n_kv_heads}x{w.head_dim} ff={w.ff}  "
        f"dense={args.dense_layers}L full-causal  concept=6L enc(w{args.enc_window}) + r{args.ratio}"
        f" + 4L latent + 8L dec(seg{args.dec_segment}"
        f"{', exclusive xattn' if args.exclusive else ''}"
        f"{f', top-{args.cross_topk} read' if args.cross_topk else ''}"
        f"{f', latent window {args.latent_window}' if args.latent_window else ''})\n"
    )
    for seq in args.seq:
        d = dense_cost(seq, args.dense_layers, w)
        c = concept_cost(
            seq,
            w,
            ratio=args.ratio,
            dec_segment=args.dec_segment,
            enc_window=args.enc_window,
            exclusive=args.exclusive,
            cross_topk=args.cross_topk,
            latent_window=args.latent_window,
        )
        print(f"S = {seq:,} ({int(c['n_slots']):,} slots)")
        if args.flops:
            print(
                f"  forward FLOPs   dense {_fmt(d['flops'])}   concept {_fmt(c['flops'])}"
                f"   ->  {d['flops'] / c['flops']:.2f}x cheaper"
            )
            print(
                f"    attention     dense {_fmt(d['flops_quadratic'])}"
                f" ({100 * d['flops_quadratic'] / d['flops']:.0f}% of total)"
                f"   concept {_fmt(c['flops_quadratic'])}"
                f" ({100 * c['flops_quadratic'] / c['flops']:.0f}%)"
                f"   ->  {d['flops_quadratic'] / c['flops_quadratic']:.1f}x"
            )
            print(
                f"      of which    enc {_fmt(c['flops_enc'])}  dec-self {_fmt(c['flops_dec_self'])}"
                f"  cross {_fmt(c['flops_cross'])}  latent {_fmt(c['flops_latent'])}"
            )
            print(
                f"    per-token     dense {_fmt(d['flops_linear'])}   concept {_fmt(c['flops_linear'])}"
                f"   ->  {d['flops_linear'] / c['flops_linear']:.2f}x"
            )
        du = d["state_unbounded_bytes_per_token"]
        cu = c["state_unbounded_bytes_per_token"]
        print(
            f"  decode state    dense {_bytes(du)}/tok = {_bytes(du * seq)}"
            f"   concept {_bytes(cu)}/tok = {_bytes(cu * seq)} + {_bytes(c['state_bounded_bytes'])} fixed"
            f"   ->  {du / cu:.0f}x smaller"
        )
        cc = c["state_unbounded_cached_bytes_per_token"]
        print(
            f"    if latent + cross K/V are cached too: {_bytes(cc)}/tok = {_bytes(cc * seq)}"
            f"   ->  {du / cc:.0f}x smaller\n"
        )


if __name__ == "__main__":
    main()
