#!/usr/bin/env python
"""Paired, preregistered E10 concept-vs-control checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.concept_analysis import compute_within_sample_concept_rank
from nn.backbone_concept_lm import BackboneConceptLM


def load_eval_rows(manifest_path: Path, seq_len: int, num_docs: int, seed: int) -> torch.Tensor:
    manifest = json.loads(manifest_path.read_text())
    parts = [
        load_from_disk(src["eval_path"])
        for src in manifest["sources"]
        if src.get("eval_path") and Path(src["eval_path"]).exists()
    ]
    if not parts:
        raise ValueError(f"No eval datasets found in {manifest_path}")
    dataset = concatenate_datasets(parts).shuffle(seed=seed)
    rows = []
    for row in dataset:
        ids = row["input_ids"]
        if len(ids) >= seq_len:
            rows.append(ids[:seq_len])
        if len(rows) == num_docs:
            break
    if len(rows) < num_docs:
        raise ValueError(
            f"Only {len(rows)} rows of length >= {seq_len} in {manifest_path}; "
            f"need {num_docs}."
        )
    return torch.tensor(rows, dtype=torch.long)


def score_mode(model, input_ids, mode: str, batch_size: int, device: str) -> torch.Tensor:
    scores = []
    for start in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[start : start + batch_size].to(device)
        scores.append(
            model.per_position_ce(
                batch, mode="blockwise", concept_mode=mode
            ).cpu()
        )
    return torch.cat(scores)


def paired_summary(a: torch.Tensor, b: torch.Tensor, lo: int, hi: int, seed: int) -> dict:
    """Summarize paired per-document improvement a-b over a position region."""
    per_doc = a[:, lo:hi].nanmean(dim=1) - b[:, lo:hi].nanmean(dim=1)
    generator = torch.Generator().manual_seed(seed)
    boot = []
    for _ in range(2000):
        idx = torch.randint(len(per_doc), (len(per_doc),), generator=generator)
        boot.append(per_doc[idx].mean())
    boot = torch.stack(boot).sort().values
    return {
        "mean": float(per_doc.mean().item()),
        "ci95": [
            float(boot[int(0.025 * len(boot))].item()),
            float(boot[int(0.975 * len(boot))].item()),
        ],
        "n_docs": len(per_doc),
    }


@torch.no_grad()
def evaluate_length(
    concept_model,
    control_model,
    rows,
    seq_len: int,
    gap: float,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    boundary = 2 * concept_model.config.concept_block
    if seq_len <= boundary:
        raise ValueError(f"seq_len={seq_len} must exceed beyond-local boundary {boundary}.")
    concept_real = score_mode(concept_model, rows, "real", batch_size, device)
    control = score_mode(control_model, rows, "zero", batch_size, device)
    concept_static = score_mode(concept_model, rows, "static", batch_size, device)
    concept_one_block = score_mode(concept_model, rows, "one_block", batch_size, device)
    concept_shuffle = score_mode(concept_model, rows, "shuffle", batch_size, device)

    improvement = paired_summary(control, concept_real, boundary, seq_len, seed)
    local_regression = paired_summary(
        concept_real, control, 0, min(concept_model.config.concept_block, seq_len), seed
    )
    static_gain = paired_summary(concept_static, concept_real, boundary, seq_len, seed)
    one_block_gain = paired_summary(concept_one_block, concept_real, boundary, seq_len, seed)
    shuffle_gain = paired_summary(concept_shuffle, concept_real, boundary, seq_len, seed)

    concepts = []
    for start in range(0, rows.shape[0], batch_size):
        batch = rows[start : start + batch_size].to(device)
        concepts.append(
            concept_model.encode_concepts(batch, return_dict=True).last_hidden_state.cpu()
        )
    rank = compute_within_sample_concept_rank(torch.cat(concepts))
    return {
        "seq_len": seq_len,
        "beyond_local_start": boundary,
        "stage0_gap": gap,
        "control_minus_concept_beyond_1024": improvement,
        "recovery_fraction": improvement["mean"] / gap,
        "concept_minus_control_local_lt512": local_regression,
        "static_minus_recurrent_beyond_1024": static_gain,
        "one_block_minus_recurrent_beyond_1024": one_block_gain,
        "shuffle_minus_real_beyond_1024": shuffle_gain,
        "concept_rank": rank,
    }


def parse_eval_spec(value: str) -> tuple[int, Path, float]:
    seq, manifest, gap = value.split(":", 2)
    return int(seq), Path(manifest), float(gap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_checkpoint", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        metavar="SEQ:MANIFEST:G",
        help="Frozen eval manifest and Stage-0 gap, e.g. 2048:path.json:0.2840.",
    )
    parser.add_argument("--num_docs", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    concept = BackboneConceptLM.from_pretrained(
        args.concept_checkpoint, dtype=dtype
    ).to(device).eval()
    control = BackboneConceptLM.from_pretrained(
        args.control_checkpoint, dtype=dtype
    ).to(device).eval()
    if not concept.has_concepts or control.has_concepts:
        raise ValueError("Expected concept checkpoint C>0 and control checkpoint C=0.")
    tokenizer = AutoTokenizer.from_pretrained(args.concept_checkpoint)

    report = {
        "concept_checkpoint": args.concept_checkpoint,
        "control_checkpoint": args.control_checkpoint,
        "tokenizer": tokenizer.name_or_path,
        "seed": args.seed,
        "lengths": {},
    }
    eval_specs = list(map(parse_eval_spec, args.eval))
    max_len_by_manifest = {}
    for seq_len, manifest, _ in eval_specs:
        max_len_by_manifest[manifest] = max(
            seq_len, max_len_by_manifest.get(manifest, 0)
        )
    frozen_rows = {
        manifest: load_eval_rows(
            manifest, max_seq_len, args.num_docs, args.seed
        )
        for manifest, max_seq_len in max_len_by_manifest.items()
    }
    for seq_len, manifest, gap in eval_specs:
        rows = frozen_rows[manifest][:, :seq_len]
        report["lengths"][str(seq_len)] = evaluate_length(
            concept, control, rows, seq_len, gap, args.batch_size, device, args.seed
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
