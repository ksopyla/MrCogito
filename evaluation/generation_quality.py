"""Generation-quality metrics for `concept_ar` checkpoints.

Three complementary axes, all reusable from notebooks, CLI runners, and batch
evaluation:

1. **Token-level diversity of a generated sequence** (no torch needed):
   `distinct_n`, `repetition_rate`, `length_binned_diversity_profile`.
   These diagnose *repetition loops* — the E05/E02-long failure mode where the
   decoder emits fluent-local but semantically-empty cycles. Distinct-n is the
   fraction of unique n-grams among all n-grams; repetition-rate is its complement
   at n=1 (the share of *repeated* tokens). Binning by generated-length window
   answers the E09 Stage-0 question: does diversity collapse past the K-window?

2. **Suffix cross-entropy by suffix-position** (the E09 Stage-0 diagnostic):
   `compute_suffix_ce_by_position` returns intact / concept-zeroed / concept-shuffled
   suffix-CE curves binned by position. For a windowed decoder (K fixed), positions
   beyond `K` cannot reach tokens further back than `K` through local self-attention;
   a rising CE-intact curve past K, or a growing intact-vs-shuffled gap, quantifies
   the "frozen snapshot + K-window cannot sustain prediction" wall — the wall E09's
   gated recurrent memory is designed to remove. A *flat* curve falsifies E09's
   hypothesis at no training cost (Stage 0 kill gate).

3. **Free-running generation** with optional per-step log-probabilities:
   `generate_free_running` returns the decoded token ids (and the per-step
   top-token logprob when requested), so callers can compute any token-level
   metric downstream (entropy, distinct-n over a sliding window, etc.).

Conventions mirror `analysis/concept_generation_eval.py` and
`nn/concept_encoder_perceiver.py::_teacher_forced_ce_window` (same shift-right
convention via `model._shift_right`, same `labels == -100` padding mask, same
`encode_concepts` / `decode_logits` surfaces). No new shift is re-derived here
(re-deriving it is the bug class behind the E01 double-shift).
"""

from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


# ============================================================
# 1. Token-level diversity metrics (pure, no torch)
# ============================================================

def _ngrams(ids: Sequence[int], n: int) -> List[Tuple[int, ...]]:
    if n <= 0 or len(ids) < n:
        return []
    return [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]


def distinct_n(ids: Sequence[int], n: int) -> float:
    """Fraction of unique n-grams in `ids`. 1.0 = no repetition at granularity n.

    Standard distinct-n (Li et al. 2016). Returns 0.0 for sequences shorter than n.
    """
    grams = _ngrams(ids, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repetition_rate(ids: Sequence[int], n: int = 1) -> float:
    """Share of *repeated* n-grams (complement of distinct_n). 0.0 = fully diverse.

    A model stuck in a length-T loop has repetition-rate → 1 − 1/T as length grows.
    At n=1 this is the token-level repetition rate; n=2 captures phrase-level loops.
    """
    return 1.0 - distinct_n(ids, n)


def length_binned_diversity_profile(
    ids: Sequence[int],
    bin_size: int = 128,
    ns: Sequence[int] = (1, 2),
) -> Dict[str, object]:
    """Distinct-n computed in `bin_size`-token windows across the sequence.

    Returns `{"bin_size", "num_bins", "n_values", "bins": [{bin_start, bin_end,
    n_tokens, distinct_n_by_n}...]}`. A flat-or-rising distinct-n profile across
    bins means generation keeps introducing novelty; a profile that *falls* toward
    the end is the repetition-loop signature. Set `bin_size` to the decoder's K
    window (E05: 128) to align bins with the local-context boundary.

    The final bin is padded only with the tokens actually present (no synthetic
    padding) — its distinct-n is still well-defined.
    """
    if bin_size <= 0:
        raise ValueError(f"bin_size must be > 0, got {bin_size}")
    if not ns or any(n <= 0 for n in ns):
        raise ValueError(f"ns must be non-empty and positive, got {ns}")

    L = len(ids)
    num_bins = max(1, (L + bin_size - 1) // bin_size)
    bins: List[Dict[str, object]] = []
    for b in range(num_bins):
        start = b * bin_size
        end = min(start + bin_size, L)
        chunk = ids[start:end]
        bins.append({
            "bin_index": b,
            "bin_start": start,
            "bin_end": end,
            "n_tokens": len(chunk),
            "distinct_n_by_n": {n: distinct_n(chunk, n) for n in ns},
        })
    return {
        "bin_size": bin_size,
        "num_bins": num_bins,
        "n_values": list(ns),
        "bins": bins,
    }


def repetition_conditional(ids: Sequence[int], n: int = 3) -> float:
    """Conditional repetition rate at gram-size n (a.k.a. REP-3 from Welleck et al. 2020).

    For every (n+1)-gram, count the n-grams that are followed by a token completing
    a *previously seen* n-gram at that prefix. REP-3 (n=3) was specifically proposed
    to detect the long-form repetition loops that distinct-n undercounts at high n.

    Returns 0.0 if the sequence is shorter than n+1.
    """
    if len(ids) < n + 1:
        return 0.0
    seen_ngrams: Counter = Counter()
    repeated_continuations = 0
    total_continuations = 0
    for i in range(len(ids) - n):
        prefix = tuple(ids[i : i + n])
        cont = ids[i + n]
        total_continuations += 1
        # Has this (n+1)-gram been seen before? If so, the continuation is a repeat.
        full = prefix + (cont,)
        if seen_ngrams[full] > 0:
            repeated_continuations += 1
        seen_ngrams[full] += 1
    if total_continuations == 0:
        return 0.0
    return repeated_continuations / total_continuations


# ============================================================
# 2. Suffix cross-entropy by position (E09 Stage-0 diagnostic)
# ============================================================

def _ce_by_position_bin(
    logits: torch.Tensor, labels: torch.Tensor, bin_edges: Sequence[int],
) -> List[Dict[str, float]]:
    """Per-bin next-token CE. `logits[t]` predicts `labels[t]` (T5 convention).

    Returns a list (one entry per bin) of `{"bin_index", "bin_start", "bin_end",
    "ce", "n_tokens"}`. Padding (labels == -100) is honoured.
    """
    out: List[Dict[str, float]] = []
    T = labels.size(1)
    for b, start in enumerate(bin_edges):
        end = bin_edges[b + 1] if b + 1 < len(bin_edges) else T
        if end <= start:
            continue
        sl = slice(start, end)
        nll = torch.nn.functional.cross_entropy(
            logits[:, sl, :].reshape(-1, logits.size(-1)),
            labels[:, sl].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid = (labels[:, sl] != -100).sum().clamp_min(1).item()
        out.append({
            "bin_index": b,
            "bin_start": int(start),
            "bin_end": int(end),
            "ce": float(nll.item() / valid),
            "n_tokens": int(valid),
        })
    return out


def _make_bin_edges(T: int, bin_size: int, window_k: Optional[int]) -> List[int]:
    """Bin edges: [0, window_k, window_k+bin_size, ...] when window_k is set, else [0, bin_size, 2*bin_size, ...].

    The first bin always ends at `window_k` (when set) so the first reported CE is the
    within-window fluency baseline; subsequent bins are aligned to `bin_size` so the
    beyond-window *trend* is visible at the same granularity as the diversity profile.
    """
    if T <= 0:
        return [0]
    edges: List[int] = [0]
    if window_k is not None and 0 < window_k < T:
        edges.append(window_k)
    cursor = edges[-1]
    while cursor < T:
        cursor += bin_size
        edges.append(min(cursor, T))
    # Dedup (window_k could equal bin_size).
    return sorted(set(edges))


@torch.no_grad()
def compute_suffix_ce_by_position(
    model,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device,
    *,
    prefix_ratio: float = 0.4,
    bin_size: int = 128,
    window_k: Optional[int] = None,
    early_k: int = 16,
) -> Dict[str, object]:
    """Suffix-CE binned by suffix position — the E09 Stage-0 diagnostic.

    For each (input_ids, attention_mask) batch this:
      1. splits the sequence at `prefix_ratio` (encoder sees the prefix; decoder
         teacher-forces the suffix — the E02-long / E05 prefix→suffix training
         contract, replicated in eval to avoid the train/eval objective mismatch
         that the suffix-contract W&B ablation captures);
      2. runs intact / concept-zeroed / concept-shuffled forward passes;
      3. bins each pass's suffix-CE by position (first bin = within-window fluency,
         later bins = progressively beyond-window).

    Returns aggregated curves and the per-bin intact-vs-shuffled gap. A rising
    `ce_intact_by_bin` past the K-window, or a growing `delta_shuffle_by_bin`,
    is the "frozen snapshot + K-window cannot sustain prediction" wall — quantified
    before any writable-memory training is spent.

    `batches` is a list of `(input_ids, attention_mask)` CPU tensors, matching the
    `run_concept_analysis.py` / `concept_generation_eval.py` convention.
    """
    if not 0.0 < prefix_ratio < 1.0:
        raise ValueError(f"prefix_ratio must be in (0, 1), got {prefix_ratio}")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be > 0, got {bin_size}")

    if window_k is None:
        window_k = getattr(model.config, "decoder_context_window", None)

    # Accumulate per-bin sums across batches; bins may differ per batch (T varies),
    # so we collect per-batch results and aggregate by bin_index at the end.
    per_pass_bins: Dict[str, List[Dict[str, float]]] = {
        "intact": [], "zero": [], "shuffle": []
    }
    early_sum: Dict[str, Dict[str, float]] = {
        p: {"nll": 0.0, "n": 0} for p in per_pass_bins
    }

    for input_ids, attention_mask in batches:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        T = input_ids.size(1)
        if T < 4:  # nothing meaningful to split
            continue

        # Real-token length per row (right-padding assumed). The split point is a
        # fraction of the *real* token count; padded positions land in the suffix
        # but contribute zero loss (labels == -100 below).
        real_lens = attention_mask.sum(dim=1)  # [B]
        # Use the batch-median real length to pick a single split column (vectorized).
        split_col = int(real_lens.float().median().item()) if real_lens.numel() else T // 2
        split_col = max(2, min(T - 2, int(round(split_col * prefix_ratio))))

        prefix_ids = input_ids[:, :split_col].contiguous()
        prefix_mask = attention_mask[:, :split_col].contiguous()
        suffix_ids = input_ids[:, split_col:].contiguous()
        suffix_mask = attention_mask[:, split_col:].contiguous()
        if suffix_ids.size(1) < 2:
            continue

        labels = suffix_ids.clone()
        labels[suffix_mask == 0] = -100

        # Encode from the prefix only (the prefix→suffix training contract).
        concepts = model.encode_concepts(
            input_ids=prefix_ids, attention_mask=prefix_mask, return_dict=True,
        ).last_hidden_state  # [B, C, H]

        dec_key_padding = labels == -100
        dec_input = model._shift_right(suffix_ids)

        logits_intact = model.decode_logits(
            concepts, dec_input, key_padding_mask=dec_key_padding,
        )
        logits_zero = model.decode_logits(
            torch.zeros_like(concepts), dec_input, key_padding_mask=dec_key_padding,
        )
        perm = torch.randperm(concepts.size(0), device=concepts.device)
        logits_shuffle = model.decode_logits(
            concepts[perm], dec_input, key_padding_mask=dec_key_padding,
        )

        edges = _make_bin_edges(suffix_ids.size(1), bin_size, window_k)
        for pass_name, logits in (
            ("intact", logits_intact),
            ("zero", logits_zero),
            ("shuffle", logits_shuffle),
        ):
            per_pass_bins[pass_name].extend(_ce_by_position_bin(logits, labels, edges))

            # Early-positions (first `early_k` real suffix tokens) — where concept
            # reliance is strongest, mirrors `_teacher_forced_ce_early`.
            k = max(1, min(early_k, labels.size(1)))
            nll_early = torch.nn.functional.cross_entropy(
                logits[:, :k, :].reshape(-1, logits.size(-1)),
                labels[:, :k].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_early = (labels[:, :k] != -100).sum().clamp_min(1).item()
            early_sum[pass_name]["nll"] += float(nll_early.item())
            early_sum[pass_name]["n"] += int(n_early)

    # Aggregate: align bins by bin_index across batches (bins differ in count when
    # T varies; the common case is they share the same first 2-3 bins, which carry
    # the diagnostic — within-window fluency + the first beyond-window stretch).
    def _agg(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
        by_idx: Dict[int, Dict[str, float]] = {}
        for r in rows:
            d = by_idx.setdefault(r["bin_index"], {"nll_sum": 0.0, "n": 0})
            d["nll_sum"] += r["ce"] * r["n_tokens"]
            d["n"] += r["n_tokens"]
        out = []
        for idx in sorted(by_idx):
            d = by_idx[idx]
            out.append({
                "bin_index": idx,
                "ce": d["nll_sum"] / max(1, d["n"]),
                "n_tokens": d["n"],
            })
        return out

    intact_curve = _agg(per_pass_bins["intact"])
    zero_curve = _agg(per_pass_bins["zero"])
    shuffle_curve = _agg(per_pass_bins["shuffle"])

    # Per-bin delta_shuffle = ce_shuffle_bin - ce_intact_bin (matched by bin_index).
    intact_by_idx = {b["bin_index"]: b["ce"] for b in intact_curve}
    shuffle_by_idx = {b["bin_index"]: b["ce"] for b in shuffle_curve}
    zero_by_idx = {b["bin_index"]: b["ce"] for b in zero_curve}
    delta_curve = []
    for idx in sorted(set(intact_by_idx) & set(shuffle_by_idx)):
        delta_curve.append({
            "bin_index": idx,
            "delta_shuffle": shuffle_by_idx[idx] - intact_by_idx[idx],
            "delta_zero": zero_by_idx.get(idx, float("nan")) - intact_by_idx[idx],
        })

    def _early(pass_name: str) -> float:
        d = early_sum[pass_name]
        return d["nll"] / max(1, d["n"])

    return {
        "prefix_ratio": prefix_ratio,
        "bin_size": bin_size,
        "window_k": window_k,
        "early_k": early_k,
        "ce_intact_by_bin": intact_curve,
        "ce_zero_by_bin": zero_curve,
        "ce_shuffle_by_bin": shuffle_curve,
        "delta_by_bin": delta_curve,
        "ce_intact_early": _early("intact"),
        "delta_shuffle_early": _early("shuffle") - _early("intact"),
        "delta_zero_early": _early("zero") - _early("intact"),
        "n_batches": len(batches),
    }


# ============================================================
# 3. Free-running generation
# ============================================================

@torch.no_grad()
def generate_free_running(
    model,
    tokenizer: "PreTrainedTokenizerBase",
    prompt: str,
    device,
    *,
    max_new_tokens: int = 256,
    max_prompt_len: int = 2048,
    greedy: bool = True,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: Optional[int] = None,
    return_logprobs: bool = False,
) -> Dict[str, object]:
    """Concept-bottleneck free-running decode.

    Encodes `prompt` into C concepts (encoder sees the full prompt), then autoregressively
    decodes a fresh sequence from the start token, cross-attending only to those concepts
    plus its own recent tokens. The decoder *never sees* the prompt tokens directly.

    Mirrors the decode loop in `notebooks/e05_generation_comparison.ipynb` and
    `analysis/run_concept_analysis.py:generate_ar_samples`, refactored to return a
    structured result (token ids + decoded text) so callers can compute any
    token-level metric downstream. Set `return_logprobs=True` to also collect the
    chosen-token log-probability at each step (cost: an extra log-softmax per step).

    Returns `{"text", "ids", "n_tokens", "prompt_ids", "prompt_n_tokens"}`
    extended with `"step_logprobs"` when requested.

    Sampling (when `greedy=False`): temperature + top-k + top-p, mirroring the
    notebook's primitive. `seed` makes sampling reproducible.
    """
    cfg = model.config
    start_id = cfg.bos_token_id or cfg.eos_token_id or cfg.pad_token_id or 0
    eos_id = cfg.eos_token_id
    if seed is not None:
        torch.manual_seed(int(seed))

    enc = tokenizer(
        prompt, truncation=True, max_length=max_prompt_len, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)
    concepts = model.encode_concepts(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
    ).last_hidden_state  # [1, C, H]

    cur = torch.tensor([[start_id]], device=device, dtype=torch.long)
    step_logprobs: List[float] = []
    for _ in range(max_new_tokens):
        logits = model.decode_logits(concepts, cur)[:, -1, :]
        if greedy or not temperature or temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            if top_k and top_k > 0:
                k = min(int(top_k), scaled.size(-1))
                thresh = torch.topk(scaled, k).values[:, -1:]
                scaled = scaled.masked_fill(scaled < thresh, float("-inf"))
            if top_p and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                remove = (sorted_probs.cumsum(dim=-1) - sorted_probs) >= top_p
                sorted_logits[remove] = float("-inf")
                scaled = scaled.scatter(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(scaled, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        if return_logprobs:
            logp = torch.log_softmax(logits, dim=-1)
            step_logprobs.append(float(logp.gather(-1, nxt).item()))
        cur = torch.cat([cur, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break

    gen_ids = cur[0, 1:].tolist()
    out: Dict[str, object] = {
        "text": tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
        "ids": gen_ids,
        "n_tokens": len(gen_ids),
        "prompt_ids": input_ids[0].tolist(),
        "prompt_n_tokens": int(input_ids.size(1)),
    }
    if return_logprobs:
        out["step_logprobs"] = step_logprobs
    return out


# ============================================================
# 4. Convenience aggregator
# ============================================================

def summarize_generation(generated_ids: Sequence[int], *, decoder_window_k: int = 128) -> Dict[str, object]:
    """One-shot diversity summary of a generated sequence.

    Computes distinct-1/2/3, repetition-1/2, REP-3 (Welleck), and a length-binned
    diversity profile aligned to `decoder_window_k` (E05: 128). Reusable from the
    notebook (`notebooks/e05_generation_comparison.ipynb`) and the CLI runner.
    """
    return {
        "n_tokens": len(generated_ids),
        "distinct_1": distinct_n(generated_ids, 1),
        "distinct_2": distinct_n(generated_ids, 2),
        "distinct_3": distinct_n(generated_ids, 3),
        "repetition_1": repetition_rate(generated_ids, 1),
        "repetition_2": repetition_rate(generated_ids, 2),
        "rep_3": repetition_conditional(generated_ids, 3),
        "length_binned_diversity": length_binned_diversity_profile(
            generated_ids, bin_size=decoder_window_k, ns=(1, 2),
        ),
    }
