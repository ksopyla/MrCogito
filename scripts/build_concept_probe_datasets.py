#!/usr/bin/env python
"""Build the CogitoProbe series (bits / bind / arith / props) at a fixed seed.

Deterministic. Rows are composed from a verified 1-token atom table of the E18/E22
tokenizer (default HuggingFaceTB/SmolLM3-3B = Llama-3 vocab). Does **not** upload
to the Hub; prints the exact commands for a later approved publish.

Pilot (small-but-real, default):

  uv run python scripts/build_concept_probe_datasets.py \\
      --scale pilot --seed 20260916 \\
      --out_dir Cache/concept_probes/pilot \\
      --stats_out docs/3_Evaluations_and_Baselines/dataset_cards/cogito-probe-stats.json \\
      --cards_out docs/3_Evaluations_and_Baselines/dataset_cards

Full production recipe (not generated unless you pass --scale full):

  uv run python scripts/build_concept_probe_datasets.py --scale full --seed 20260916 ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.concept_probes.atoms import build_atom_table  # noqa: E402
from data.concept_probes.generate import generate_split  # noqa: E402
from data.concept_probes.schema import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TOKENIZER,
    FAMILIES,
    HUB_IDS,
    LENGTH_LADDER,
    VARIANTS,
    split_counts,
)
from data.concept_probes.stats import render_card, summarize_family  # noqa: E402


HF_FEATURES = Features(
    {
        "id": Value("string"),
        "family": Value("string"),
        "task": Value("string"),
        "variant": Value("string"),
        "seq_len": Value("int32"),
        "rung": Value("string"),
        "split": Value("string"),
        "seed": Value("int32"),
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "text": Value("string"),
        "context": Value("string"),
        "query": Value("string"),
        "answer": Value("string"),
        "n_tokens": Value("int32"),
        "prize_bits": Value("float32"),
        "gap": Value("int32"),
        "answer_start": Value("int32"),
        "answer_end": Value("int32"),
        "evidence_end": Value("int32"),
        "meta": Value("string"),
    }
)


def _save_split(path: Path, rows: list[dict], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    Dataset.from_list(rows, features=HF_FEATURES).save_to_disk(str(path))
    parquet = path.parent / f"{path.name}.parquet"
    Dataset.from_list(rows, features=HF_FEATURES).to_parquet(str(parquet))


def _upload_commands(out_dir: Path, families: list[str], *, private: bool) -> list[str]:
    flag = " --private" if private else ""
    cmds = []
    for fam in families:
        hub = HUB_IDS[fam]
        src = out_dir / "hub" / fam
        cmds.append(f"hf repo create {hub} --repo-type dataset{flag}")
        cmds.append(
            f"hf upload {hub} {src} --repo-type dataset "
            f"--commit-message \"Add CogitoProbe {fam} pilot (seed {DEFAULT_SEED})\""
        )
    return cmds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", choices=("pilot", "full"), default="pilot")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--lengths", nargs="+", type=int, default=list(LENGTH_LADDER))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stats_out", default=None)
    p.add_argument("--cards_out", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_cards", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    table = build_atom_table(tok, tokenizer_name=args.tokenizer, seed=args.seed)

    atom_meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": table.vocab_size,
        "arith_ids": {k: v.token_id for k, v in table.arith.items()},
        "markers": {k: {"id": v.token_id, "surface": v.surface} for k, v in table.markers.items()},
        "pool_sizes": {k: len(v) for k, v in table.pools.items()},
        "pool_surfaces": {k: [a.surface for a in v] for k, v in table.pools.items()},
        "pad_id": table.pad_id,
        "bos_id": table.bos_id,
        "eos_id": table.eos_id,
    }
    (out / "atom_table.json").write_text(json.dumps(atom_meta, indent=2))

    sample_atoms = ["(", "1", "+", "2", ")", "*", "[", "3", "-", "{", "4", "}", "]"]
    composed = [table.arith[s].token_id for s in sample_atoms]
    glued = tok.encode("".join(sample_atoms), add_special_tokens=False)
    spaced = tok.encode(" ".join(sample_atoms), add_special_tokens=False)

    stats: dict = {
        "seed": args.seed,
        "scale": args.scale,
        "tokenizer": args.tokenizer,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/build_concept_probe_datasets.py",
        "families": {},
        "atom_table": {
            "pool_sizes": atom_meta["pool_sizes"],
            "markers": atom_meta["markers"],
        },
        "hub": {fam: HUB_IDS[fam] for fam in args.families},
        "arith_tokenization_check": {
            "sample_glued": "".join(sample_atoms),
            "n_atoms": len(sample_atoms),
            "n_composed_ids": len(composed),
            "n_glued_bpe": len(glued),
            "n_spaced_bpe": len(spaced),
            "composed_equals_glued_bpe": list(composed) == list(glued),
            "note": "composed ids are the instrument; glued BPE is the author's surface form and is not 1:1",
        },
        "publish": {
            "status": "not_uploaded",
            "reason": "awaiting explicit approval; do not hf upload without it",
            "commands_private": _upload_commands(out, list(args.families), private=True),
            "commands_public": _upload_commands(out, list(args.families), private=False),
        },
    }

    for family in args.families:
        fam_rows: dict[str, list] = {"train": [], "validation": [], "test": []}
        sources = []
        for seq_len in args.lengths:
            n_train, n_val, n_test = split_counts(args.scale, seq_len)
            # Split the budget across variants so a length rung stays small-but-real.
            n_var = max(len(args.variants), 1)
            counts = {
                "train": max(n_train // n_var, 1),
                "validation": max(n_val // n_var, 1),
                "test": max(n_test // n_var, 1),
            }
            for variant in args.variants:
                for split, n in counts.items():
                    rows = generate_split(
                        family,
                        table,
                        seq_len=seq_len,
                        variant=variant,
                        split=split,
                        seed=args.seed,
                        n_rows=n,
                    )
                    payload = [r.to_dict() for r in rows]
                    fam_rows[split].extend(rows)
                    dest = out / "hf" / family / f"seq{seq_len}" / variant / split
                    _save_split(dest, payload, overwrite=True)
                    sources.append(
                        {
                            "name": f"{family}_seq{seq_len}_{variant}_{split}",
                            "path": str(dest),
                            "num_rows": n,
                            "mean_row_tokens": seq_len,
                            "family": family,
                            "seq_len": seq_len,
                            "variant": variant,
                            "split": split,
                        }
                    )
        # Combined per-split parquet for Hub staging.
        hub_root = out / "hub" / family
        hub_root.mkdir(parents=True, exist_ok=True)
        for split, rows in fam_rows.items():
            payload = [r.to_dict() for r in rows]
            ds = Dataset.from_list(payload, features=HF_FEATURES)
            ds.to_parquet(str(hub_root / f"{split}.parquet"))
            ds.save_to_disk(str(out / "pretok" / family / split))
        fam_stats = summarize_family(fam_rows)
        stats["families"][family] = fam_stats
        if args.cards_out and not args.skip_cards:
            card_dir = Path(args.cards_out) / f"cogito-probe-{family}"
            card_dir.mkdir(parents=True, exist_ok=True)
            card = render_card(
                family,
                fam_stats,
                seed=args.seed,
                tokenizer=args.tokenizer,
                scale=args.scale,
                hub_id=HUB_IDS[family],
            )
            (card_dir / "README.md").write_text(card)
            (hub_root / "README.md").write_text(card)

        manifest = {
            "mix_id": f"cogito_probe_{family}_{args.scale}",
            "objective": "causal_lm",
            "max_seq_length": max(args.lengths),
            "seed": args.seed,
            "created": stats["created"],
            "label_policy": "answer_only",
            "generator": "data/concept_probes/",
            "tokenizer": args.tokenizer,
            "sources": [
                s
                for s in sources
                if s["split"] in {"train", "validation"}
            ],
        }
        (out / f"manifest_{family}.json").write_text(json.dumps(manifest, indent=2))

    if args.stats_out:
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps({
        "out_dir": str(out),
        "families": {k: v.get("n_rows") for k, v in stats["families"].items()},
        "publish_status": stats["publish"]["status"],
        "upload_commands_private": stats["publish"]["commands_private"],
    }, indent=2))


if __name__ == "__main__":
    main()
