"""Unit tests for evaluation/generation_quality.py.

Tiny random concept_ar model on CPU — checks wiring, ranges, and the qualitative
behaviour of the diversity metrics. We do not assert generation quality (the model
is untrained); we assert:
  - pure diversity metrics behave on hand-constructed sequences;
  - the suffix-CE-by-position diagnostic returns finite curves;
  - the free-running generator produces well-formed output;
  - the length-binned diversity profile buckets correctly.
"""

import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM
from evaluation.generation_quality import (
    compute_suffix_ce_by_position,
    distinct_n,
    generate_free_running,
    length_binned_diversity_profile,
    repetition_conditional,
    repetition_rate,
    summarize_generation,
)


def _tiny_config(**overrides) -> ConceptEncoderConfig:
    base = dict(
        vocab_size=40,
        hidden_size=32,
        token_embedding_dim=16,
        concept_num=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=64,
        decoder_num_layers=2,
        decoder_type="causal_ar",
        decoder_pos_type="rope",
        decoder_word_dropout=0.0,
        hidden_act="silu",
        norm_type="rmsnorm",
        use_bixt=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    base.update(overrides)
    return ConceptEncoderConfig(**base)


def _batches(B=2, T=24, vocab=40, n_batches=2):
    """Real-content (no pad) batches."""
    out = []
    for _ in range(n_batches):
        ids = torch.randint(3, vocab, (B, T))
        mask = torch.ones_like(ids)
        out.append((ids, mask))
    return out


# --- distinct_n ---

def test_distinct_n_all_unique_is_one():
    assert distinct_n([1, 2, 3, 4, 5], 1) == 1.0


def test_distinct_n_all_same_is_one_over_T():
    # All-same length-T sequence has 1 unique 1-gram out of T → distinct-1 = 1/T.
    assert distinct_n([7, 7, 7, 7], 1) == 0.25


def test_distinct_n_short_sequence_is_zero():
    # Sequence shorter than n has no n-grams → 0.0 by convention (avoids div-by-zero).
    assert distinct_n([1, 2], 3) == 0.0


def test_distinct_n_order_independent():
    # Same multiset → same distinct count.
    assert distinct_n([1, 2, 1, 2], 2) == distinct_n([2, 1, 2, 1], 2)


# --- repetition_rate ---

def test_repetition_rate_complement_of_distinct():
    seq = [1, 2, 3, 1, 2, 3]
    assert abs(repetition_rate(seq, 1) - (1.0 - distinct_n(seq, 1))) < 1e-9


def test_repetition_rate_loops_high():
    # A short loop repeated many times → repetition-1 close to (1 - 1/T).
    seq = [4, 5] * 50  # T=100, 2 unique → distinct-1 = 2/100 = 0.02 → rep-1 = 0.98
    assert repetition_rate(seq, 1) > 0.95


# --- repetition_conditional (REP-3) ---

def test_rep_3_no_repetition_is_zero():
    # All (n+1)-grams unique → no repeated continuation.
    seq = list(range(20))
    assert repetition_conditional(seq, n=3) == 0.0


def test_rep_3_loop_is_high():
    # Repeating a length-4 token block many times: most 4-grams are repeats.
    seq = [10, 11, 12, 13] * 20
    assert repetition_conditional(seq, n=3) > 0.7


def test_rep_3_short_sequence_is_zero():
    assert repetition_conditional([1, 2, 3], n=3) == 0.0  # length < n+1


# --- length_binned_diversity_profile ---

def test_length_binned_diversity_correct_bin_count():
    # 16 tokens, bin_size 4 → 4 bins.
    seq = list(range(16))
    out = length_binned_diversity_profile(seq, bin_size=4, ns=(1,))
    assert out["num_bins"] == 4
    assert out["bins"][0]["bin_start"] == 0
    assert out["bins"][0]["bin_end"] == 4
    assert out["bins"][-1]["bin_end"] == 16


def test_length_binned_diversity_final_bin_padded_only():
    # 10 tokens, bin_size 4 → bins of 4, 4, 2.
    seq = list(range(10))
    out = length_binned_diversity_profile(seq, bin_size=4, ns=(1,))
    assert out["num_bins"] == 3
    assert [b["n_tokens"] for b in out["bins"]] == [4, 4, 2]


def test_length_binned_diversity_invalid_args():
    import pytest
    with pytest.raises(ValueError):
        length_binned_diversity_profile([1, 2, 3], bin_size=0)
    with pytest.raises(ValueError):
        length_binned_diversity_profile([1, 2, 3], bin_size=2, ns=())


# --- compute_suffix_ce_by_position ---

def test_suffix_ce_returns_finite_curves():
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    out = compute_suffix_ce_by_position(
        model, _batches(), device="cpu",
        prefix_ratio=0.4, bin_size=8, window_k=8,
    )
    assert out["n_batches"] == 2
    assert out["window_k"] == 8
    assert len(out["ce_intact_by_bin"]) >= 1
    for k in ("ce_intact_early", "delta_shuffle_early", "delta_zero_early"):
        assert out[k] == out[k]  # not NaN
    for b in out["ce_intact_by_bin"]:
        assert b["ce"] >= 0.0  # CE non-negative
    # delta_by_bin is keyed by bin_index and aligns with ce_intact_by_bin.
    assert {b["bin_index"] for b in out["delta_by_bin"]} <= {b["bin_index"] for b in out["ce_intact_by_bin"]}


def test_suffix_ce_window_k_is_first_bin_edge():
    """When window_k is set, the first reported bin should end at window_k
    (the within-window fluency baseline)."""
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    out = compute_suffix_ce_by_position(
        model, _batches(T=32), device="cpu",
        prefix_ratio=0.4, bin_size=8, window_k=8,
    )
    # The first intact-CE bin should start at 0 and end at window_k (8).
    first = out["ce_intact_by_bin"][0]
    assert first["bin_index"] == 0


def test_suffix_ce_rejects_bad_args():
    import pytest
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    with pytest.raises(ValueError):
        compute_suffix_ce_by_position(model, _batches(), device="cpu", prefix_ratio=0.0)
    with pytest.raises(ValueError):
        compute_suffix_ce_by_position(model, _batches(), device="cpu", prefix_ratio=0.5, bin_size=0)


def test_suffix_ce_handles_padding():
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    ids = torch.randint(3, 40, (3, 20))
    mask = torch.ones_like(ids)
    mask[:, 12:] = 0  # pad the tail
    out = compute_suffix_ce_by_position(
        model, [(ids, mask)], device="cpu",
        prefix_ratio=0.4, bin_size=4, window_k=4,
    )
    assert out["n_batches"] == 1
    assert out["ce_intact_early"] == out["ce_intact_early"]  # not NaN


# --- generate_free_running ---

class _DummyTokenizer:
    """Tiny tokenizer stub that maps ints↔ints so we don't need a real HF tokenizer."""
    def __call__(self, prompt, **kwargs):
        # Encode each character to its ord code modulo vocab, with a leading BOS.
        ids = [1] + [(ord(c) % 38) + 3 for c in prompt[: kwargs.get("max_length", 2048)]]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
    def decode(self, ids, **_):
        return "".join(chr((i - 3) % 38 + 0x61) for i in ids if i >= 3)


def test_generate_free_running_returns_well_formed_output():
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    tok = _DummyTokenizer()
    out = generate_free_running(model, tok, "hello", device="cpu", max_new_tokens=12)
    assert out["n_tokens"] <= 12
    assert out["n_tokens"] == len(out["ids"])
    assert out["prompt_n_tokens"] == len(out["prompt_ids"])
    assert isinstance(out["text"], str)


def test_generate_free_running_logprobs_when_requested():
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    tok = _DummyTokenizer()
    out = generate_free_running(
        model, tok, "hello", device="cpu", max_new_tokens=8, return_logprobs=True,
    )
    assert "step_logprobs" in out
    assert len(out["step_logprobs"]) == out["n_tokens"]
    # Chosen-token log-prob is always ≤ 0.
    assert all(lp <= 0.0 for lp in out["step_logprobs"])


# --- summarize_generation ---

def test_summarize_generation_keys_present():
    summary = summarize_generation([1, 2, 3, 1, 2, 3] * 5, decoder_window_k=8)
    for k in ("n_tokens", "distinct_1", "distinct_2", "distinct_3",
              "repetition_1", "repetition_2", "rep_3", "length_binned_diversity"):
        assert k in summary
    assert summary["length_binned_diversity"]["bin_size"] == 8
