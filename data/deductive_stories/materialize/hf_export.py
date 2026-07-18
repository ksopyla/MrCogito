"""HF / disk export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.deductive_stories.schema import DeductiveExample


def examples_to_rows(examples: list[DeductiveExample]) -> list[dict[str, Any]]:
    return [ex.to_public_row() for ex in examples if ex.accepted]


def write_jsonl(examples: list[DeductiveExample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = examples_to_rows(examples)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_hf_dataset(
    examples: list[DeductiveExample],
    *,
    output_dir: Path,
    push_repo: str | None = None,
    private: bool = True,
) -> Path:
    """Save DatasetDict to disk; optionally push to the Hub."""
    from datasets import Dataset, DatasetDict

    output_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in examples_to_rows(examples):
        by_split.setdefault(row["split"], []).append(row)

    dset = DatasetDict(
        {split: Dataset.from_list(rows) for split, rows in by_split.items() if rows}
    )
    dset.save_to_disk(str(output_dir))
    card = output_dir / "README.md"
    if not card.exists():
        card.write_text(_dataset_card_markdown(), encoding="utf-8")
    if push_repo:
        dset.push_to_hub(push_repo, private=private)
    return output_dir


def _dataset_card_markdown() -> str:
    return """# Deductive Stories

Graph-first synthetic long-context deductive narratives with solver-verified
exact-match answers (1–5 questions per story).

## Pipeline

1. Programmatic typed event graph + solver gold
2. Chaptered LLM narrative expansion (Azure OpenAI)
3. Fidelity judge (reject-only; never overrides answers)
4. Parametric-leak + dedup + template-id train/test firewall
5. Dual length views (2K/4K train, 8K/16K eval)

## Scoring

Exact-match on normalized short answers. Do not use free-form LLM judges for
the primary gate.

## Contamination

Train and test use disjoint `template_id` families and disjoint name banks.
See `docs/engineering_specs/deductive_stories_synthetic_dataset.md`.

## License

Code: Apache-2.0 / project MIT. Dataset: synthetic; see generation provenance
columns (`writer_model`, `judge_model`, `generation_version`).
"""
