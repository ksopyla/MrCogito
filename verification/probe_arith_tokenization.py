#!/usr/bin/env python
"""Prototype: can number+bracket arithmetic streams force rich compressible latents?

This is a *measurement*, not a generator. It tokenizes candidate surface forms with the
E18/E22 tokenizer (SmolLM3 = Llama-3 vocab), generates nested ``+ - * ( ) [ ] { }``
expressions, and reports whether BPE yields a 1-symbol-1-token instrument or an
algorithmic-shortcut task with an ill-defined token alphabet.

Run:
  uv run python verification/probe_arith_tokenization.py
  uv run python verification/probe_arith_tokenization.py --tokenizer HuggingFaceTB/SmolLM3-3B
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


OPS = ("+", "-", "*")
BRACKETS = (("(", ")"), ("[", "]"), ("{", "}"))
DIGITS = tuple(str(d) for d in range(10))


def _ngram_entropy(ids: list[int], n: int) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    total = len(grams)
    counts = Counter(grams)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def _gzip_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(gzip.compress(raw, compresslevel=9)) / len(raw)


def gen_expr(rng: np.random.Generator, depth: int, max_depth: int) -> tuple[list[str], int, dict]:
    """Return (atom list, integer value, tree). Internal nodes always use a bracket pair."""
    if depth >= max_depth or (depth > 0 and rng.random() < 0.25):
        d = int(rng.integers(0, 10))
        return [str(d)], d, {"kind": "num", "value": d}
    opener, closer = BRACKETS[int(rng.integers(0, len(BRACKETS)))]
    op = OPS[int(rng.integers(0, len(OPS)))]
    left_a, left_v, left_t = gen_expr(rng, depth + 1, max_depth)
    right_a, right_v, right_t = gen_expr(rng, depth + 1, max_depth)
    atoms = [opener, *left_a, op, *right_a, closer]
    if op == "+":
        value = left_v + right_v
    elif op == "-":
        value = left_v - right_v
    else:
        value = left_v * right_v
    tree = {
        "kind": "op",
        "op": op,
        "bracket": opener + closer,
        "left": left_t,
        "right": right_t,
        "value": value,
    }
    return atoms, value, tree


def collect_subvalues(tree: dict, out: list[int] | None = None) -> list[int]:
    if out is None:
        out = []
    if tree["kind"] == "op":
        collect_subvalues(tree["left"], out)
        collect_subvalues(tree["right"], out)
        out.append(int(tree["value"]))
    else:
        out.append(int(tree["value"]))
    return out


def encode_forms(tokenizer, atoms: list[str]) -> dict[str, list[int]]:
    glued = "".join(atoms)
    spaced = " ".join(atoms)
    digit_spaced = " ".join(atoms)  # already atomic digits
    return {
        "glued": tokenizer.encode(glued, add_special_tokens=False),
        "spaced": tokenizer.encode(spaced, add_special_tokens=False),
        "digit_spaced": tokenizer.encode(digit_spaced, add_special_tokens=False),
    }


def probe_single_symbols(tokenizer) -> dict:
    rows = []
    for raw in [*DIGITS, *OPS, "(", ")", "[", "]", "{", "}"]:
        for form, s in (("bare", raw), ("ls", " " + raw)):
            ids = tokenizer.encode(s, add_special_tokens=False)
            rows.append(
                {
                    "symbol": raw,
                    "form": form,
                    "surface": s,
                    "n_tokens": len(ids),
                    "ids": ids,
                    "atomic": len(ids) == 1,
                    "decoded": tokenizer.decode(ids),
                }
            )
    n_atomic = sum(1 for r in rows if r["atomic"])
    return {"n": len(rows), "n_atomic": n_atomic, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM3-3B")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--out", default="Cache/concept_probes/arith_tokenization_probe.json")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rng = np.random.default_rng(args.seed)

    single = probe_single_symbols(tok)

    # Author's glued form vs space-separated atoms.
    examples = []
    glued_extra = []  # tokens that are not 1:1 with atoms
    spaced_ratio = []
    values = []
    n_subvalues = []
    unique_subvalues = []
    gzip_glued = []
    gzip_spaced = []
    unigram = []
    bigram = []
    tokens_per_atom_glued = []
    tokens_per_atom_spaced = []
    max_abs_value = 0
    eval_bits = []
    tree_bits_proxy = []

    for _ in range(args.n):
        atoms, value, tree = gen_expr(rng, 0, args.max_depth)
        subs = collect_subvalues(tree)
        forms = encode_forms(tok, atoms)
        glued_ids, spaced_ids = forms["glued"], forms["spaced"]
        examples.append(
            {
                "atoms": atoms,
                "glued": "".join(atoms),
                "spaced": " ".join(atoms),
                "value": value,
                "n_atoms": len(atoms),
                "n_glued_tokens": len(glued_ids),
                "n_spaced_tokens": len(spaced_ids),
                "glued_ids": glued_ids,
                "spaced_ids": spaced_ids,
                "n_subvalues": len(subs),
                "unique_subvalues": len(set(subs)),
            }
        )
        values.append(value)
        n_subvalues.append(len(subs))
        unique_subvalues.append(len(set(subs)))
        gzip_glued.append(_gzip_ratio("".join(atoms)))
        gzip_spaced.append(_gzip_ratio(" ".join(atoms)))
        unigram.append(_ngram_entropy(spaced_ids, 1))
        bigram.append(_ngram_entropy(spaced_ids, 2))
        tokens_per_atom_glued.append(len(glued_ids) / max(len(atoms), 1))
        tokens_per_atom_spaced.append(len(spaced_ids) / max(len(atoms), 1))
        max_abs_value = max(max_abs_value, abs(value))
        # Shortcut ceiling: only the scalar is needed.
        eval_bits.append(math.log2(max(abs(value), 1) + 1) + (1.0 if value < 0 else 0.0))
        # Lower bound on storing every distinct intermediate: ~log2(|v|+1) each.
        tree_bits_proxy.append(
            sum(math.log2(max(abs(v), 1) + 1) + (1.0 if v < 0 else 0.0) for v in set(subs))
        )
        # Detect glued merges: fewer tokens than atoms ⇒ BPE merged symbols.
        glued_extra.append(len(atoms) - len(glued_ids))

    # DNA-like control: iid 4-symbol streams of the same atom-length.
    dna_gzip = []
    dna_h1 = []
    for ex in examples:
        n = ex["n_atoms"]
        dna = rng.choice(list("ACGT"), size=n)
        text = "".join(dna.tolist())
        ids = tok.encode(text, add_special_tokens=False)
        dna_gzip.append(_gzip_ratio(text))
        dna_h1.append(_ngram_entropy(ids, 1))

    # English-ish proposition control (same n as mean arith length, tiny vocab).
    words = ["the", "miner", "dropped", "a", "blue", "coin", "in", "rome", "baker", "found", "red", "lamp"]
    prop_gzip = []
    for ex in examples:
        n = max(ex["n_atoms"] // 3, 4)
        sent = " ".join(rng.choice(words, size=n))
        prop_gzip.append(_gzip_ratio(sent))

    atomic_bare = [r for r in single["rows"] if r["form"] == "bare"]
    atomic_ls = [r for r in single["rows"] if r["form"] == "ls"]
    summary = {
        "tokenizer": args.tokenizer,
        "vocab_size": len(tok),
        "n_expr": args.n,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "single_symbol_atomic_bare": {
            r["symbol"]: r["atomic"] for r in atomic_bare
        },
        "single_symbol_atomic_leading_space": {
            r["symbol"]: r["atomic"] for r in atomic_ls
        },
        "bare_atomic_rate": sum(r["atomic"] for r in atomic_bare) / len(atomic_bare),
        "leading_space_atomic_rate": sum(r["atomic"] for r in atomic_ls) / len(atomic_ls),
        "mean_tokens_per_atom_glued": float(np.mean(tokens_per_atom_glued)),
        "mean_tokens_per_atom_spaced": float(np.mean(tokens_per_atom_spaced)),
        "frac_glued_merged": float(np.mean([x > 0 for x in glued_extra])),
        "mean_atoms": float(np.mean([e["n_atoms"] for e in examples])),
        "mean_value": float(np.mean(values)),
        "max_abs_value": int(max_abs_value),
        "mean_n_subvalues": float(np.mean(n_subvalues)),
        "mean_unique_subvalues": float(np.mean(unique_subvalues)),
        "mean_eval_bits_shortcut": float(np.mean(eval_bits)),
        "mean_tree_bits_proxy": float(np.mean(tree_bits_proxy)),
        "gzip_glued": float(np.mean(gzip_glued)),
        "gzip_spaced": float(np.mean(gzip_spaced)),
        "gzip_dna_iid": float(np.mean(dna_gzip)),
        "gzip_prop_words": float(np.mean(prop_gzip)),
        "unigram_entropy_spaced": float(np.mean(unigram)),
        "bigram_entropy_spaced": float(np.mean(bigram)),
        "examples_head": examples[:8],
        "single_symbol_rows": single["rows"],
    }

    # Verdict fields (filled in prose by the spec; numbers live here).
    summary["verdict_numbers"] = {
        "eval_only_stores_one_integer": True,
        "eval_bits_vs_tree_bits": {
            "eval": summary["mean_eval_bits_shortcut"],
            "tree": summary["mean_tree_bits_proxy"],
            "ratio_tree_over_eval": (
                summary["mean_tree_bits_proxy"] / max(summary["mean_eval_bits_shortcut"], 1e-6)
            ),
        },
        "glued_is_not_1to1": summary["mean_tokens_per_atom_glued"] != 1.0
        or summary["frac_glued_merged"] > 0,
        "spaced_near_1to1": abs(summary["mean_tokens_per_atom_spaced"] - 1.0) < 0.15,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"examples_head", "single_symbol_rows"}}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
