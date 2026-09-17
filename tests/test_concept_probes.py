"""CogitoProbe generators: solvability, determinism, packing, leakage, atom table."""
from __future__ import annotations

import numpy as np
import pytest

from data.concept_probes.atoms import (
    ARITH_ATOMS,
    MARKER_CANDIDATES,
    PREFERRED_WORDS,
    Atom,
    build_atom_table,
    int_to_arith_atoms,
    prize_bits_uniform,
)
from data.concept_probes.generate import generate_row, generate_split
from data.concept_probes.schema import (
    FAMILIES,
    IGNORE_INDEX,
    LENGTH_LADDER,
    expected_split_totals,
    recipe_for,
)
from data.concept_probes.stats import (
    FAMILY_PRETTY,
    leakage_report,
    render_card,
    size_category,
    summarize_family,
)


TINY_PLAN = (
    ("colors", 8),
    ("places", 8),
    ("jobs", 8),
    ("agents", 12),
    ("objects", 10),
    ("values", 8),
    ("keys", 24),
    ("filler", 12),
)


class FakeProbeTokenizer:
    """Word/atom tokenizer that mimics SmolLM3: bare arith is 1-token; ' 0' is not."""

    def __init__(self):
        self._enc: dict[str, list[int]] = {}
        self._dec: dict[int, str] = {}
        i = 40
        for sym in ARITH_ATOMS:
            self._enc[sym] = [i]
            self._dec[i] = sym
            i += 1
            # leading-space digits split (two ids) like SmolLM3
            if sym.isdigit():
                self._enc[" " + sym] = [3, i]
                i += 1
            else:
                self._enc[" " + sym] = [i]
                self._dec[i] = " " + sym
                i += 1
        for _name, cands in MARKER_CANDIDATES.items():
            w = cands[0]
            if w not in self._enc:
                self._enc[w] = [i]
                self._dec[i] = w
                i += 1
            if " " + w not in self._enc:
                self._enc[" " + w] = [i]
                self._dec[i] = " " + w
                i += 1
        self.probe_word_atoms = []
        for w in list(PREFERRED_WORDS) + [f"zz{k:03d}" for k in range(80)]:
            surface = " " + w
            if surface in self._enc:
                continue
            self._enc[surface] = [i]
            self._dec[i] = surface
            self.probe_word_atoms.append(Atom(name=w, token_id=i, surface=w.lower()))
            i += 1
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.all_special_ids = [0, 1, 2]
        self._n = i + 10

    def __len__(self):
        return self._n

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self._enc:
            return list(self._enc[text])
        raise KeyError(text)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self._dec.get(int(t), "?") for t in token_ids)


@pytest.fixture(scope="module")
def table():
    return build_atom_table(
        FakeProbeTokenizer(),
        tokenizer_name="fake",
        seed=0,
        pool_plan=TINY_PLAN,
    )


def test_arith_atoms_are_one_token(table):
    tok = FakeProbeTokenizer()
    for sym in ARITH_ATOMS:
        assert tok.encode(sym) == [table.arith[sym].token_id]
        assert len(tok.encode(sym)) == 1
    # leading-space digits are NOT atomic (the SmolLM3 confound)
    assert len(tok.encode(" 7")) != 1


def test_recipe_scales_and_fixed_holds():
    s = recipe_for("bits", 32768, "scaled")
    f = recipe_for("bits", 32768, "fixed")
    one = recipe_for("bits", 1024, "scaled")
    assert s.n_items == one.n_items * 16
    assert f.n_items == one.n_items


@pytest.mark.parametrize("family", FAMILIES)
def test_row_is_packed_deterministic_and_answer_labeled(table, family):
    a = generate_row(family, table, seq_len=256, variant="fixed", split="train", seed=7, row_index=0)
    b = generate_row(family, table, seq_len=256, variant="fixed", split="train", seed=7, row_index=0)
    c = generate_row(family, table, seq_len=256, variant="fixed", split="train", seed=7, row_index=1)
    assert a.input_ids == b.input_ids and a.labels == b.labels
    assert a.input_ids != c.input_ids
    assert len(a.input_ids) == 256 == len(a.labels) == len(a.attention_mask)
    supervised = [i for i, y in enumerate(a.labels) if y != IGNORE_INDEX]
    assert supervised, f"{family} produced no supervised tokens"
    assert supervised == list(range(a.answer_start, a.answer_end))
    assert a.labels[a.answer_start : a.answer_end] == a.input_ids[a.answer_start : a.answer_end]
    assert a.gap >= 0
    assert a.prize_bits > 0
    assert a.answer_end <= 256


@pytest.mark.parametrize("family", FAMILIES)
def test_solver_recovers_answer_from_meta(table, family):
    row = generate_row(family, table, seq_len=256, variant="fixed", split="test", seed=11, row_index=3)
    meta = row.meta
    if family == "bits":
        assert meta["query_values"]
        surfaces = row.answer.split()
        assert surfaces == meta["query_values"]
    elif family == "bind":
        assert meta["entities"]
        assert len(row.answer.split()) == meta["n_query"]
    elif family == "arith":
        if row.task == "eval":
            assert int(row.meta["root_value"]) == row.meta["root_value"]
            atoms = int_to_arith_atoms(int(row.meta["root_value"]))
            assert [table.arith[s].token_id for s in atoms] == row.input_ids[row.answer_start : row.answer_end]
        elif row.task == "subexpr":
            assert row.meta["nodes"]
        elif row.task == "match":
            assert row.meta["closer_pos"] > row.meta["opener_pos"]
    elif family == "props":
        by_obj = {p["object"]: p["color"] for p in meta["propositions"]}
        for obj, col in zip(meta["query_objects"], row.answer.split(), strict=True):
            assert by_obj[obj] == col


def test_bits_prize_matches_formula(table):
    row = generate_row("bits", table, seq_len=256, variant="fixed", split="train", seed=1, row_index=0)
    n_q = row.meta["n_query"]
    n_v = row.meta["n_values"]
    assert abs(row.prize_bits - prize_bits_uniform(n_q, n_v)) < 1e-9


def test_arith_eval_is_shortcut_relative_to_subexpr(table):
    eval_bits = []
    sub_bits = []
    for i in range(24):
        row = generate_row("arith", table, seq_len=256, variant="fixed", split="train", seed=3, row_index=i)
        if row.task == "eval":
            eval_bits.append(row.prize_bits)
        elif row.task == "subexpr":
            sub_bits.append(row.prize_bits)
    assert eval_bits and sub_bits
    assert float(np.mean(sub_bits)) > float(np.mean(eval_bits))


def test_split_streams_do_not_clone_rows(table):
    rows = {
        "train": generate_split("bits", table, seq_len=128, variant="fixed", split="train", seed=20260916, n_rows=20),
        "validation": generate_split(
            "bits", table, seq_len=128, variant="fixed", split="validation", seed=20260916, n_rows=12
        ),
        "test": generate_split("bits", table, seq_len=128, variant="fixed", split="test", seed=20260916, n_rows=12),
    }
    leak = leakage_report(rows)
    for pair, v in leak["pairs"].items():
        assert v["input_ids"] == 0, pair
        assert v["fingerprint"] == 0, pair
    assert all(v == 0 for v in leak["within_split_duplicate_ids"].values())


def test_summarize_family_runs(table):
    rows = generate_split("props", table, seq_len=128, variant="fixed", split="train", seed=0, n_rows=8)
    summary = summarize_family({"train": rows})
    assert summary["n_rows"] == 8
    assert summary["gzip_ratio_text_mean"] > 0
    assert summary["rungs"]["seq128"] == 8


def test_length_ladder_constants():
    assert LENGTH_LADDER == (1024, 4096, 8192, 16384, 32768)
    assert set(FAMILIES) == {"bits", "bind", "arith", "props"}
    full = expected_split_totals("full")
    assert full == {"train": 8448, "validation": 896, "test": 896, "n_rows": 10240}
    pilot = expected_split_totals("pilot")
    assert pilot["n_rows"] == 464
    assert size_category(464) == "n<1K"
    assert size_category(10240) == "10K<n<100K"


def test_dataset_card_names_hub_and_author(table):
    rows = generate_split("bits", table, seq_len=128, variant="fixed", split="train", seed=0, n_rows=4)
    stats = summarize_family({"train": rows})
    card = render_card(
        "bits",
        stats,
        seed=20260916,
        tokenizer="HuggingFaceTB/SmolLM3-3B",
        scale="full",
        hub_id="ksopyla/cogito-probe-bits",
    )
    assert "https://huggingface.co/datasets/ksopyla/cogito-probe-bits" in card
    assert "Krzysztof Sopyła" in card
    assert "--scale full" in card
    assert "10K<n<100K" not in card  # tiny fixture is n<1K
    assert "n<1K" in card
    assert "path: train.parquet" in card
    assert "- split: train" in card
    assert "from datasets import load_dataset" in card
    assert "## In 60 seconds" in card
    assert "key–value recall in a long haystack" in card
    # External readers should not need internal experiment ids.
    assert "E18" not in card
    assert "E21" not in card
    assert "E25" not in card
    assert "DNA A=4" not in card


@pytest.mark.parametrize("family", ["bits", "bind", "arith", "props"])
def test_dataset_card_is_self_contained_for_each_family(table, family):
    rows = generate_split(family, table, seq_len=128, variant="fixed", split="train", seed=1, n_rows=3)
    stats = summarize_family({"train": rows})
    card = render_card(
        family,
        stats,
        seed=20260916,
        tokenizer="HuggingFaceTB/SmolLM3-3B",
        scale="pilot",
        hub_id=f"ksopyla/cogito-probe-{family}",
    )
    assert f'load_dataset("ksopyla/cogito-probe-{family}")' in card
    assert 'ds.filter(lambda r: r["seq_len"] == 1024' in card
    assert "E18" not in card and "E21" not in card
    assert FAMILY_PRETTY[family].split(":")[0] in card


def test_int_atoms_include_sign():
    assert int_to_arith_atoms(12) == ["1", "2"]
    assert int_to_arith_atoms(-7) == ["-", "7"]
    assert int_to_arith_atoms(0) == ["0"]


def test_smollm3_glued_arith_is_not_one_to_one():
    """The author's surface form is not the instrument under our tokenizer."""
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B", use_fast=True)
    for sym in ARITH_ATOMS:
        assert len(tok.encode(sym, add_special_tokens=False)) == 1
    glued = "(1+2)*[3-{4}]"
    n_atoms = 13
    n_glued = len(tok.encode(glued, add_special_tokens=False))
    n_spaced = len(tok.encode(" ".join(glued), add_special_tokens=False))
    assert n_glued != n_atoms
    assert n_spaced != n_atoms
    assert len(tok.encode(" 7", add_special_tokens=False)) != 1
