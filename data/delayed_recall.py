"""Deterministic synthetic delayed-recall data for recurrent-memory diagnostics.

The task places a key/value fact in block 1 and asks for the value in a later
block. Counterfactual twins share the same key and all tokens after block 1 up
to the answer, but carry different values and targets. Rows are already
tokenized and contain a sparse causal-LM label mask.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


IGNORE_INDEX = -100

PREFERRED_VALUE_WORDS = (
    "red", "blue", "green", "black", "white", "yellow", "purple", "brown",
    "apple", "banana", "orange", "grape", "lemon", "peach", "mango", "cherry",
    "cat", "dog", "horse", "bird", "fish", "tiger", "lion", "bear",
    "table", "chair", "house", "river", "mountain", "ocean", "forest", "garden",
    "north", "south", "east", "west", "spring", "summer", "autumn", "winter",
    "circle", "square", "triangle", "star", "moon", "sun", "earth", "cloud",
    "stone", "metal", "wood", "glass", "paper", "cotton", "silk", "gold",
    "happy", "quiet", "bright", "soft", "sharp", "warm", "cold", "fast",
    "music", "dance", "story", "dream", "light", "shadow", "water", "fire",
    "book", "train", "plane", "boat", "road", "field", "bread", "milk",
)


@dataclass(frozen=True)
class DelayedRecallTokenPools:
    value_ids: tuple[int, ...]
    key_ids: tuple[int, ...]
    noise_ids: tuple[int, ...]
    decoded_values: tuple[str, ...]


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in ids]


def _marker_sequences(tokenizer) -> dict[str, list[int]]:
    return {
        "fact": _encode(tokenizer, "Memory record. Key"),
        "value": _encode(tokenizer, "has value"),
        "end": _encode(tokenizer, "."),
        "query": _encode(tokenizer, "Recall the stored value for key"),
        "answer": _encode(tokenizer, ". Answer:"),
    }


def select_delayed_recall_token_pools(
    tokenizer,
    *,
    model_vocab_size: int,
    value_count: int = 64,
    key_count: int = 256,
    noise_count: int = 512,
    seed: int = 42,
) -> DelayedRecallTokenPools:
    """Select disjoint, readable, model-valid single-token pools.

    Rows inject these ids directly, so each answer is exactly one model token.
    The decoded-word filter keeps generated artifacts inspectable without
    relying on tokenizer-specific whitespace markers.
    """

    if value_count < 2 or value_count % 2:
        raise ValueError("value_count must be an even integer >= 2.")
    if key_count < 2 or noise_count < 2:
        raise ValueError("key_count and noise_count must both be >= 2.")

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    marker_ids = {
        token_id
        for sequence in _marker_sequences(tokenizer).values()
        for token_id in sequence
    }
    candidates: list[tuple[int, str]] = []
    for token_id in range(min(len(tokenizer), model_vocab_size)):
        if token_id in special_ids or token_id in marker_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        word = decoded.strip()
        if not re.fullmatch(r"[A-Za-z]{3,16}", word):
            continue
        candidates.append((token_id, word))

    required = value_count + key_count + noise_count
    if len(candidates) < required:
        raise ValueError(
            f"Tokenizer exposes only {len(candidates)} readable single-token words; "
            f"need {required} ({value_count} values + {key_count} keys + "
            f"{noise_count} noise tokens)."
        )

    candidate_by_id = dict(candidates)
    preferred_values: list[tuple[int, str]] = []
    for word in PREFERRED_VALUE_WORDS:
        try:
            encoded = _encode(tokenizer, f" {word}")
        except (KeyError, ValueError):
            continue
        if len(encoded) != 1:
            continue
        token_id = encoded[0]
        if token_id in candidate_by_id and all(
            existing_id != token_id for existing_id, _ in preferred_values
        ):
            preferred_values.append((token_id, candidate_by_id[token_id]))

    rng = random.Random(seed)
    rng.shuffle(preferred_values)
    common_candidates = candidates[: max(required * 16, required)]
    rng.shuffle(common_candidates)
    selected_ids = {token_id for token_id, _ in preferred_values[:value_count]}
    fallback_values = [
        candidate
        for candidate in common_candidates
        if candidate[0] not in selected_ids
    ]
    values = (preferred_values + fallback_values)[:value_count]
    selected_ids = {token_id for token_id, _ in values}
    remaining = [
        candidate for candidate in common_candidates if candidate[0] not in selected_ids
    ]
    if len(remaining) < key_count + noise_count:
        remaining_ids = {token_id for token_id, _ in remaining} | selected_ids
        remaining.extend(
            candidate
            for candidate in candidates
            if candidate[0] not in remaining_ids
        )
    keys = remaining[:key_count]
    noise = remaining[key_count : key_count + noise_count]
    return DelayedRecallTokenPools(
        value_ids=tuple(token_id for token_id, _ in values),
        key_ids=tuple(token_id for token_id, _ in keys),
        noise_ids=tuple(token_id for token_id, _ in noise),
        decoded_values=tuple(word for _, word in values),
    )


def _balanced_value_pairs(
    value_ids: Sequence[int],
    *,
    num_rows: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if num_rows % len(value_ids):
        raise ValueError(
            f"num_rows={num_rows} must be divisible by value_count={len(value_ids)} "
            "for exact balance."
        )
    ordered: list[int] = []
    for _ in range(num_rows // len(value_ids)):
        cycle = list(value_ids)
        rng.shuffle(cycle)
        ordered.extend(cycle)
    pairs = list(zip(ordered[::2], ordered[1::2], strict=True))
    if any(left == right for left, right in pairs):
        raise AssertionError("A counterfactual pair received identical values.")
    return pairs


def _noise_block(
    noise_ids: Sequence[int],
    *,
    block_size: int,
    rng: random.Random,
) -> list[int]:
    return [noise_ids[rng.randrange(len(noise_ids))] for _ in range(block_size)]


def _build_pair_rows(
    tokenizer,
    pools: DelayedRecallTokenPools,
    *,
    pair_id: str,
    key_id: int,
    value_pair: tuple[int, int],
    sequence_length: int,
    block_size: int,
    query_block: int,
    rng: random.Random,
) -> list[dict]:
    num_blocks = sequence_length // block_size
    if query_block < 2 or query_block > num_blocks:
        raise ValueError(f"query_block must be in [2, {num_blocks}], got {query_block}.")

    markers = _marker_sequences(tokenizer)
    fact_template = (
        markers["fact"]
        + [key_id]
        + markers["value"]
        + [value_pair[0]]
        + markers["end"]
    )
    if len(fact_template) >= block_size:
        raise ValueError("Fact template does not fit inside block 1.")
    block1_noise = _noise_block(
        pools.noise_ids,
        block_size=block_size - len(fact_template),
        rng=rng,
    )

    query_prefix = markers["query"] + [key_id] + markers["answer"]
    answer_offset = block_size // 2
    query_noise_len = answer_offset - len(query_prefix)
    if query_noise_len < 0:
        raise ValueError("Query template does not fit before the fixed answer offset.")

    shared_blocks = [
        _noise_block(pools.noise_ids, block_size=block_size, rng=rng)
        for _ in range(num_blocks)
    ]
    query_noise = _noise_block(
        pools.noise_ids,
        block_size=query_noise_len,
        rng=rng,
    )
    query_tail = _noise_block(
        pools.noise_ids,
        block_size=block_size - answer_offset - 1,
        rng=rng,
    )
    answer_index = (query_block - 1) * block_size + answer_offset

    rows = []
    for variant, answer_id in enumerate(value_pair):
        fact = (
            markers["fact"]
            + [key_id]
            + markers["value"]
            + [answer_id]
            + markers["end"]
            + block1_noise
        )
        blocks = [list(block) for block in shared_blocks]
        blocks[0] = fact
        blocks[query_block - 1] = query_noise + query_prefix + [answer_id] + query_tail
        input_ids = [token_id for block in blocks for token_id in block]
        labels = [IGNORE_INDEX] * sequence_length
        labels[answer_index] = answer_id
        rows.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "pair_id": pair_id,
                "variant": variant,
                "answer_index": answer_index,
                "answer_token_id": answer_id,
                "donor_answer_token_id": value_pair[1 - variant],
                "query_block": query_block,
            }
        )
    return rows


def build_delayed_recall_rows(
    tokenizer,
    pools: DelayedRecallTokenPools,
    *,
    split: str,
    num_rows: int,
    sequence_length: int = 2048,
    block_size: int = 512,
    query_block: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Build adjacent counterfactual twins with exactly balanced answer values."""

    if num_rows <= 0 or num_rows % 2:
        raise ValueError("num_rows must be a positive even integer.")
    if sequence_length % block_size:
        raise ValueError("sequence_length must be divisible by block_size.")

    rng = random.Random(seed)
    value_pairs = _balanced_value_pairs(pools.value_ids, num_rows=num_rows, rng=rng)
    rows: list[dict] = []
    for pair_index, value_pair in enumerate(value_pairs):
        key_id = pools.key_ids[rng.randrange(len(pools.key_ids))]
        pair_rng = random.Random(rng.getrandbits(64))
        rows.extend(
            _build_pair_rows(
                tokenizer,
                pools,
                pair_id=f"{split}-{pair_index:06d}",
                key_id=key_id,
                value_pair=value_pair,
                sequence_length=sequence_length,
                block_size=block_size,
                query_block=query_block,
                rng=pair_rng,
            )
        )
    validate_delayed_recall_rows(
        rows,
        sequence_length=sequence_length,
        block_size=block_size,
    )
    return rows


def validate_delayed_recall_rows(
    rows: Iterable[dict],
    *,
    sequence_length: int,
    block_size: int,
) -> None:
    """Fail fast on masking, pairing, and local-leakage contract violations."""

    rows = list(rows)
    if not rows or len(rows) % 2:
        raise ValueError("Delayed-recall rows must contain adjacent counterfactual pairs.")
    for pair_start in range(0, len(rows), 2):
        left, right = rows[pair_start : pair_start + 2]
        for row in (left, right):
            if len(row["input_ids"]) != sequence_length:
                raise ValueError("Every delayed-recall input must have the exact sequence length.")
            if len(row["labels"]) != sequence_length:
                raise ValueError("Every delayed-recall label mask must match input length.")
            supervised = [
                index for index, label in enumerate(row["labels"]) if label != IGNORE_INDEX
            ]
            if supervised != [row["answer_index"]]:
                raise ValueError("Every row must supervise exactly its declared answer position.")
            answer_index = row["answer_index"]
            if row["labels"][answer_index] != row["answer_token_id"]:
                raise ValueError("Answer label and answer_token_id disagree.")
            if row["input_ids"][answer_index] != row["answer_token_id"]:
                raise ValueError("Teacher-forced input does not contain the declared answer token.")
            if not (block_size <= answer_index < sequence_length):
                raise ValueError("Answer must be outside block 1.")

        if left["pair_id"] != right["pair_id"] or {left["variant"], right["variant"]} != {0, 1}:
            raise ValueError("Rows must be adjacent variant-0/variant-1 twins.")
        if left["answer_index"] != right["answer_index"]:
            raise ValueError("Counterfactual twins must share the answer position.")
        if left["answer_token_id"] == right["answer_token_id"]:
            raise ValueError("Counterfactual twins must have incompatible answers.")
        if left["donor_answer_token_id"] != right["answer_token_id"]:
            raise ValueError("Left donor target does not point to its twin.")
        if right["donor_answer_token_id"] != left["answer_token_id"]:
            raise ValueError("Right donor target does not point to its twin.")
        answer_index = left["answer_index"]
        if left["input_ids"][block_size:answer_index] != right["input_ids"][block_size:answer_index]:
            raise ValueError(
                "Counterfactual twins differ in local context before the answer."
            )
