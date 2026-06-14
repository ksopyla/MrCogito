"""Unit tests for the Phase-0 concept-space diagnostics.

Covers the two metrics added to disambiguate "slot redundancy" (the old batch-mean
effective rank) from "representation-manifold dimensionality" (what downstream tasks
ride on), plus per-slot input activity (dead-register detection).
"""

import torch

from analysis.concept_analysis import (
    compute_concept_geometry_metrics,
    compute_representation_manifold_metrics,
)


def _slot_collapsed(B=64, C=128, H=768):
    """All C slots ~ the same per-sample direction → slot rank ~1, but per-sample
    pooled embeddings still span many dims (the E02-style 'paradox')."""
    base = torch.randn(B, 1, H)
    return base.expand(B, C, H) + 0.01 * torch.randn(B, C, H)


def _slot_diverse(B=64, C=128, H=768):
    return torch.randn(B, C, H)


def test_slot_rank_detects_redundancy_but_manifold_does_not_collapse():
    x = _slot_collapsed()
    g = compute_concept_geometry_metrics(x)
    pooled = x.mean(dim=1)
    m = compute_representation_manifold_metrics(pooled)
    # Redundant slots → low slot effective rank.
    assert g["effective_rank"] < 5.0
    # But different inputs still occupy many directions → manifold rank is NOT ~1.
    assert m["manifold_rankme"] > 5.0


def test_diverse_slots_have_high_slot_rank():
    g = compute_concept_geometry_metrics(_slot_diverse())
    assert g["effective_rank"] > 30.0


def test_dead_register_detection():
    B, C, H = 32, 16, 64
    x = torch.randn(B, C, H)
    # Freeze half the slots to a constant across inputs (dead registers).
    x[:, : C // 2, :] = torch.randn(1, C // 2, H)
    g = compute_concept_geometry_metrics(x)
    assert 0.4 <= g["active_slot_fraction"] <= 0.6


def test_anisotropy_high_for_narrow_cone():
    # All sentence embeddings near one direction → anisotropy -> 1.
    d = torch.randn(1, 64)
    pooled = d + 0.001 * torch.randn(512, 64)
    m = compute_representation_manifold_metrics(pooled)
    assert m["manifold_anisotropy"] > 0.9


def test_manifold_metrics_handle_tiny_input():
    m = compute_representation_manifold_metrics(torch.randn(1, 64))
    assert m["manifold_rankme"] != m["manifold_rankme"]  # NaN for n<2
