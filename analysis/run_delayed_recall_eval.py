#!/usr/bin/env python
"""Evaluate sparse delayed recall under causal recurrent-memory interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from datasets import load_from_disk

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.delayed_recall import validate_delayed_recall_rows
from nn.backbone_concept_lm import BackboneConceptLM


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_pair_indices(dataset, num_pairs: int) -> list[tuple[int, int]]:
    grouped: OrderedDict[str, list[int]] = OrderedDict()
    for index, pair_id in enumerate(dataset["pair_id"]):
        grouped.setdefault(str(pair_id), []).append(index)
    pairs = []
    for pair_id, indices in grouped.items():
        if len(indices) != 2:
            raise ValueError(f"Pair {pair_id!r} has {len(indices)} rows; expected 2.")
        indices.sort(key=lambda index: int(dataset[index]["variant"]))
        pairs.append((indices[0], indices[1]))
        if num_pairs and len(pairs) == num_pairs:
            break
    if num_pairs and len(pairs) < num_pairs:
        raise ValueError(f"Manifest has only {len(pairs)} pairs; requested {num_pairs}.")
    return pairs


def _bootstrap_summary(values: torch.Tensor, seed: int, samples: int = 2000) -> dict:
    values = values.float().cpu()
    generator = torch.Generator().manual_seed(seed)
    draws = torch.empty(samples, dtype=torch.float32)
    for index in range(samples):
        sample_indices = torch.randint(
            len(values), (len(values),), generator=generator
        )
        draws[index] = values[sample_indices].mean()
    draws = draws.sort().values
    return {
        "mean": float(values.mean().item()),
        "ci95": [
            float(draws[int(0.025 * samples)].item()),
            float(draws[int(0.975 * samples)].item()),
        ],
        "n_pairs": len(values),
    }


@torch.inference_mode()
def evaluate(
    model: BackboneConceptLM,
    dataset,
    pair_indices: list[tuple[int, int]],
    *,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("--batch_size must be a positive even number of rows.")
    pairs_per_batch = batch_size // 2
    mode_ce = {mode: [] for mode in ("real", "static", "zero", "donor")}
    mode_predictions = {mode: [] for mode in mode_ce}
    targets = []
    donor_targets = []

    for start in range(0, len(pair_indices), pairs_per_batch):
        pair_batch = pair_indices[start : start + pairs_per_batch]
        flat_indices = [index for pair in pair_batch for index in pair]
        rows = [dataset[index] for index in flat_indices]
        validate_delayed_recall_rows(
            rows,
            sequence_length=model.config.concept_block
            * (len(rows[0]["input_ids"]) // model.config.concept_block),
            block_size=model.config.concept_block,
        )
        input_ids = torch.tensor(
            [row["input_ids"] for row in rows],
            dtype=torch.long,
            device=device,
        )
        labels = torch.tensor(
            [row["labels"] for row in rows],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        valid = labels != -100
        if not torch.equal(
            valid.sum(dim=1), torch.ones(len(rows), dtype=torch.long, device=device)
        ):
            raise ValueError("Every evaluation row must contain exactly one target.")

        permutation = torch.arange(len(rows), device=device) ^ 1
        for report_name, concept_mode in (
            ("real", "real"),
            ("static", "static"),
            ("zero", "zero"),
            ("donor", "permutation"),
        ):
            metrics = model.per_position_metrics(
                input_ids,
                attention_mask,
                labels,
                concept_mode=concept_mode,
                concept_permutation=(
                    permutation if concept_mode == "permutation" else None
                ),
            )
            mode_ce[report_name].append(metrics["ce"][valid].float().cpu())
            mode_predictions[report_name].append(
                metrics["predictions"][valid].cpu()
            )
        targets.append(labels[valid].cpu())
        donor_targets.append(
            torch.tensor(
                [row["donor_answer_token_id"] for row in rows],
                dtype=torch.long,
            )
        )
        completed = min(start + len(pair_batch), len(pair_indices))
        if completed == len(pair_indices) or completed % 128 == 0:
            print(f"Evaluated {completed}/{len(pair_indices)} counterfactual pairs.")

    mode_ce = {mode: torch.cat(parts) for mode, parts in mode_ce.items()}
    mode_predictions = {
        mode: torch.cat(parts) for mode, parts in mode_predictions.items()
    }
    targets = torch.cat(targets)
    donor_targets = torch.cat(donor_targets)

    modes = {}
    for mode in mode_ce:
        modes[mode] = {
            "answer_ce": float(mode_ce[mode].mean().item()),
            "answer_top1_accuracy": float(
                (mode_predictions[mode] == targets).float().mean().item()
            ),
        }
    modes["donor"]["donor_target_follow_accuracy"] = float(
        (mode_predictions["donor"] == donor_targets).float().mean().item()
    )

    margins = {}
    for offset, mode in enumerate(("static", "zero", "donor")):
        per_row = mode_ce[mode] - mode_ce["real"]
        per_pair = per_row.view(-1, 2).mean(dim=1)
        margins[f"{mode}_minus_real"] = _bootstrap_summary(
            per_pair, seed=seed + offset
        )

    success_margins = all(
        summary["mean"] >= 0.10 and summary["ci95"][0] > 0
        for summary in margins.values()
    )
    success_competence = modes["real"]["answer_top1_accuracy"] >= 0.50
    leakage_clear = all(
        modes[mode]["answer_top1_accuracy"] <= 0.20
        for mode in ("static", "zero", "donor")
    )
    return {
        "num_pairs": len(pair_indices),
        "num_rows": 2 * len(pair_indices),
        "modes": modes,
        "margins": margins,
        "kill_gate_triggered": all(
            summary["mean"] < 0.01 for summary in margins.values()
        ),
        "success_gates": {
            "all_margins_ge_0_10_and_ci_above_zero": success_margins,
            "real_accuracy_ge_0_50": success_competence,
            "ablated_accuracies_le_0_20": leakage_clear,
            "passed": success_margins and success_competence and leakage_clear,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--eval_view",
        choices=["block2", "block3", "block4"],
        default="block4",
    )
    parser.add_argument("--num_pairs", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    eval_views = manifest.get("eval_views", {})
    if args.eval_view in eval_views:
        eval_path = Path(eval_views[args.eval_view])
    elif args.eval_view == "block4":
        eval_path = Path(manifest["sources"][0]["eval_path"])
    else:
        raise ValueError(f"Manifest has no {args.eval_view!r} evaluation view.")
    dataset = load_from_disk(str(eval_path))
    pair_indices = _load_pair_indices(dataset, args.num_pairs)

    device = _resolve_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = BackboneConceptLM.from_pretrained(
        str(args.checkpoint), dtype=dtype
    ).to(device).eval()
    if not model.has_concepts:
        raise ValueError("Delayed-recall attribution requires a concept checkpoint.")
    if int(manifest["block_size"]) != model.config.concept_block:
        raise ValueError(
            f"Manifest block_size={manifest['block_size']} does not match checkpoint "
            f"K={model.config.concept_block}."
        )

    result = evaluate(
        model,
        dataset,
        pair_indices,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "eval_path": str(eval_path),
        "eval_view": args.eval_view,
        "device": device,
        "seed": args.seed,
        "concept_block": model.config.concept_block,
        "concept_num": model.config.concept_num,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
