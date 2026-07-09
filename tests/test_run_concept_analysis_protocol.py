"""Tests for the 2026-07-07 Tier-1 data-protocol upgrade in run_concept_analysis.

Covers: length-bucket parsing, the per-batch-std + per-bucket ablation aggregator,
the derangement shuffle in concept_ablation_ce, and the centered within-sample
RankMe variant (shared-offset vs genuine collapse disambiguation).
"""

import torch

from analysis.run_concept_analysis import parse_length_buckets, compute_ar_concept_ablation
from analysis.concept_analysis import compute_within_sample_concept_rank
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM
from tests.test_concept_ar_decoder import _tiny_config


def test_parse_length_buckets_covers_range():
    assert parse_length_buckets("256,512,1024", 2048) == [
        (0, 256), (256, 512), (512, 1024), (1024, 2048)]
    # empty spec -> single bucket; edges >= max are dropped
    assert parse_length_buckets("", 2048) == [(0, 2048)]
    assert parse_length_buckets("512", 512) == [(0, 512)]


def test_ablation_aggregator_reports_std_and_per_bucket():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = ConceptEncoderForConditionalLM(cfg).eval()
    batches = []
    for bucket, T in [("(0,8]", 8), ("(8,16]", 16)]:
        ids = torch.randint(3, cfg.vocab_size, (4, T))
        mask = torch.ones_like(ids)
        mask[0, -3:] = 0
        batches.append({"input_ids": ids, "attention_mask": mask, "bucket": bucket})

    m = compute_ar_concept_ablation(model, batches, "cpu")
    assert "delta_zero_std" in m and "delta_shuffle_std" in m
    assert set(m["per_bucket"]) == {"(0,8]", "(8,16]"}
    for v in m["per_bucket"].values():
        assert v["n"] == 1
        assert "delta_shuffle" in v


def test_shuffle_ablation_is_derangement():
    # With batch size 2 the only valid shift is 1 (swap): delta_shuffle must reflect a
    # genuinely swapped batch every run, never an identity permutation.
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = ConceptEncoderForConditionalLM(cfg).eval()
    ids = torch.randint(3, cfg.vocab_size, (2, 12))
    mask = torch.ones_like(ids)
    for _ in range(5):
        m = model.concept_ablation_ce(ids, mask, ids.clone())
        assert abs(m["delta_shuffle"]) > 1e-6


def test_centered_rankme_separates_offset_from_collapse():
    # Shared big offset + small diverse residuals: raw RankMe collapses toward 1,
    # centered RankMe stays high -> "offset, not collapse".
    torch.manual_seed(0)
    B, C, H = 4, 16, 64
    offset = torch.randn(1, 1, H) * 100.0
    residual = torch.randn(B, C, H)
    m = compute_within_sample_concept_rank(offset + residual)
    assert m["within_sample_rankme_mean"] < 3.0
    assert m["within_sample_rankme_centered_mean"] > 10.0

    # Genuine rank-1 collapse: low on BOTH variants.
    direction = torch.randn(1, 1, H)
    scale = torch.rand(B, C, 1)
    collapsed = direction * scale + torch.randn(B, C, H) * 1e-4
    mc = compute_within_sample_concept_rank(collapsed)
    assert mc["within_sample_rankme_mean"] < 3.0
    assert mc["within_sample_rankme_centered_mean"] < 3.0
