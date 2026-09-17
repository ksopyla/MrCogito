"""Row generators for the four CogitoProbe families."""
from __future__ import annotations

from typing import Callable

import numpy as np

from data.concept_probes.atoms import Atom, AtomTable, int_to_arith_atoms, prize_bits_uniform
from data.concept_probes.schema import (
    ARITH_MAX_ABS,
    IGNORE_INDEX,
    ProbeRecipe,
    ProbeRow,
    recipe_for,
    rung_name,
)


def _rng_for(*keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(list(keys)))


def _pick(rng: np.random.Generator, pool: tuple[Atom, ...], n: int, *, replace: bool = False) -> list[Atom]:
    if n > len(pool) and not replace:
        raise ValueError(f"need {n} from pool of {len(pool)}")
    idx = rng.choice(len(pool), size=n, replace=replace)
    return [pool[int(i)] for i in np.atleast_1d(idx)]


def _int_ids(table: AtomTable, n: int) -> list[int]:
    return [table.arith[s].token_id for s in int_to_arith_atoms(n)]


def _finalize(
    *,
    table: AtomTable,
    family: str,
    task: str,
    variant: str,
    seq_len: int,
    split: str,
    seed: int,
    row_index: int,
    body: list[int],
    query: list[int],
    answer: list[int],
    evidence_end_in_body: int,
    prize_bits: float,
    meta: dict,
) -> ProbeRow:
    bos = [table.bos_id] if table.bos_id is not None else []
    ans_mark = [table.marker("answer")]
    end_mark = [table.marker("end")]
    tail = query + ans_mark + answer + end_mark
    overhead = len(bos) + len(tail)
    if overhead >= seq_len:
        raise ValueError(f"query+answer longer than seq_len={seq_len}")
    budget = seq_len - overhead
    evid = evidence_end_in_body
    if len(body) > budget:
        drop = len(body) - budget
        body = body[drop:]
        evid = max(evid - drop, 0)
    pad_n = budget - len(body)
    # Pad *inside* the haystack (before the query) so the answer stays at a fixed tail.
    if pad_n:
        filler_pool = table.pools["filler"]
        # Deterministic filler from remaining budget: cycle filler words, not pad_id.
        extra = [filler_pool[i % len(filler_pool)].token_id for i in range(pad_n)]
        body = body + extra
    ids = bos + body + tail
    if len(ids) != seq_len:
        # Last-resort pad/trunc (should not fire).
        if len(ids) < seq_len:
            ids = ids + [table.pad_id] * (seq_len - len(ids))
        else:
            ids = ids[:seq_len]
    answer_start = len(bos) + len(body) + len(query) + len(ans_mark)
    answer_end = answer_start + len(answer)
    if answer_end > seq_len:
        answer_end = seq_len
        answer_start = min(answer_start, seq_len)
    labels = [IGNORE_INDEX] * seq_len
    for i in range(answer_start, min(answer_end, seq_len)):
        labels[i] = ids[i]
    attention = [0 if tok == table.pad_id and i >= answer_end else 1 for i, tok in enumerate(ids)]
    evidence_end = len(bos) + evid
    gap = answer_start - evidence_end
    text = table.render(ids)
    context = table.render(bos + body)
    query_text = table.render(query)
    answer_text = table.render(answer)
    return ProbeRow(
        id=f"{family}/{rung_name(seq_len)}/{variant}/{split}/{row_index:05d}",
        family=family,
        task=task,
        variant=variant,
        seq_len=seq_len,
        split=split,
        seed=seed,
        input_ids=ids,
        labels=labels,
        attention_mask=attention,
        text=text,
        context=context,
        query=query_text,
        answer=answer_text,
        n_tokens=seq_len,
        prize_bits=float(prize_bits),
        gap=int(gap),
        answer_start=int(answer_start),
        answer_end=int(answer_end),
        evidence_end=int(evidence_end),
        meta=meta,
    )


# --------------------------------------------------------------------------------------
# bits
# --------------------------------------------------------------------------------------


def generate_bits_row(
    table: AtomTable,
    recipe: ProbeRecipe,
    *,
    seq_len: int,
    variant: str,
    split: str,
    seed: int,
    row_index: int,
    rng: np.random.Generator,
) -> ProbeRow:
    keys = table.pools["keys"]
    values = table.pools["values"]
    n_items = min(recipe.n_items, len(keys))
    n_query = min(recipe.n_query, n_items)
    redundancy = int(recipe.extra.get("redundancy", 1))
    key_atoms = _pick(rng, keys, n_items, replace=False)
    value_atoms = _pick(rng, values, n_items, replace=True)
    facts = list(zip(key_atoms, value_atoms, strict=True))
    body: list[int] = []
    for key_a, val_a in facts:
        for _ in range(redundancy):
            body.extend(
                [
                    table.marker("key"),
                    key_a.token_id,
                    table.marker("val"),
                    val_a.token_id,
                    table.marker("dot"),
                ]
            )
    q_idx = [int(i) for i in rng.permutation(n_items)[:n_query]]
    query = [table.marker("query")] + [facts[i][0].token_id for i in q_idx]
    answer = [facts[i][1].token_id for i in q_idx]
    # Each fact is `redundancy` copies of (key val key val dot) = 5 tokens.
    span_len = 5
    evid = max((i + 1) * span_len * redundancy for i in q_idx)
    prize = prize_bits_uniform(n_query, len(values))
    meta = {
        "n_items": n_items,
        "n_query": n_query,
        "redundancy": redundancy,
        "n_values": len(values),
        "query_keys": [facts[i][0].surface for i in q_idx],
        "query_values": [facts[i][1].surface for i in q_idx],
        "content_fingerprint": sorted(
            f"{facts[i][0].surface}={facts[i][1].surface}" for i in range(n_items)
        ),
    }
    return _finalize(
        table=table,
        family="bits",
        task="recall_packed",
        variant=variant,
        seq_len=seq_len,
        split=split,
        seed=seed,
        row_index=row_index,
        body=body,
        query=query,
        answer=answer,
        evidence_end_in_body=evid,
        prize_bits=prize,
        meta=meta,
    )


# --------------------------------------------------------------------------------------
# bind
# --------------------------------------------------------------------------------------


def generate_bind_row(
    table: AtomTable,
    recipe: ProbeRecipe,
    *,
    seq_len: int,
    variant: str,
    split: str,
    seed: int,
    row_index: int,
    rng: np.random.Generator,
) -> ProbeRow:
    agents = table.pools["agents"]
    colors = table.pools["colors"]
    places = table.pools["places"]
    jobs = table.pools["jobs"]
    n = min(recipe.n_items, len(agents), len(colors), len(places), len(jobs))
    names = _pick(rng, agents, n, replace=False)
    col = _pick(rng, colors, n, replace=n > len(colors))
    plc = _pick(rng, places, n, replace=n > len(places))
    job = _pick(rng, jobs, n, replace=n > len(jobs))
    # Ring of friends so a hop is always defined and not identity.
    perm = [int(i) for i in rng.permutation(n)]
    friend = [perm[(perm.index(i) + 1) % n] for i in range(n)]
    body: list[int] = []
    ends = []
    for i in range(n):
        start = len(body)
        body.extend(
            [
                names[i].token_id,
                table.marker("is"),
                job[i].token_id,
                table.marker("dot"),
                names[i].token_id,
                table.marker("in"),
                plc[i].token_id,
                table.marker("dot"),
                names[i].token_id,
                table.marker("color"),
                col[i].token_id,
                table.marker("dot"),
                names[i].token_id,
                table.marker("friend"),
                names[friend[i]].token_id,
                table.marker("dot"),
            ]
        )
        ends.append(len(body))
    task_roll = int(rng.integers(0, 3))
    n_query = min(recipe.n_query, n)
    q_idx = [int(i) for i in rng.permutation(n)[:n_query]]
    if task_roll == 0:
        task = "attr_color"
        query = [table.marker("query"), table.marker("color")] + [names[i].token_id for i in q_idx]
        answer = [col[i].token_id for i in q_idx]
        n_choices = len(colors)
        evid = max(ends[i] for i in q_idx)
    elif task_roll == 1:
        task = "who_place"
        query = [table.marker("query"), table.marker("who"), table.marker("in")] + [
            plc[i].token_id for i in q_idx
        ]
        answer = [names[i].token_id for i in q_idx]
        n_choices = n
        evid = max(ends[i] for i in q_idx)
    else:
        task = "hop_friend_place"
        query = [table.marker("query"), table.marker("hop"), table.marker("place")] + [
            names[i].token_id for i in q_idx
        ]
        answer = [plc[friend[i]].token_id for i in q_idx]
        n_choices = len(places)
        evid = max(ends[i] for i in q_idx)
        evid = max(evid, max(ends[friend[i]] for i in q_idx))
    prize = prize_bits_uniform(n_query, max(n_choices, 2))
    meta = {
        "n_items": n,
        "n_query": n_query,
        "task": task,
        "entities": [a.surface for a in names],
        "query_index": q_idx,
        "content_fingerprint": [
            f"{names[i].surface}|{job[i].surface}|{plc[i].surface}|{col[i].surface}|{names[friend[i]].surface}"
            for i in range(n)
        ],
    }
    return _finalize(
        table=table,
        family="bind",
        task=task,
        variant=variant,
        seq_len=seq_len,
        split=split,
        seed=seed,
        row_index=row_index,
        body=body,
        query=query,
        answer=answer,
        evidence_end_in_body=evid,
        prize_bits=prize,
        meta=meta,
    )


# --------------------------------------------------------------------------------------
# arith
# --------------------------------------------------------------------------------------


def _arith_eval(op: str, left: int, right: int) -> int:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right


def _gen_expr(
    rng: np.random.Generator,
    table: AtomTable,
    *,
    depth: int,
    max_depth: int,
    star_prob: float,
) -> tuple[list[int], int, dict]:
    """Return (token ids, value, tree). Retry-friendly: may exceed ARITH_MAX_ABS."""
    leaf = depth >= max_depth or (depth > 0 and float(rng.random()) < 0.3)
    if leaf:
        d = int(rng.integers(0, 10))
        tid = table.arith[str(d)].token_id
        return [tid], d, {"kind": "num", "value": d, "lo": 0, "hi": 1}
    brackets = (("(", ")"), ("[", "]"), ("{", "}"))
    opener, closer = brackets[int(rng.integers(0, 3))]
    op = "*" if float(rng.random()) < star_prob else ("+" if float(rng.random()) < 0.5 else "-")
    left_ids, left_v, left_t = _gen_expr(
        rng, table, depth=depth + 1, max_depth=max_depth, star_prob=star_prob
    )
    right_ids, right_v, right_t = _gen_expr(
        rng, table, depth=depth + 1, max_depth=max_depth, star_prob=star_prob
    )
    value = _arith_eval(op, left_v, right_v)
    ids = [table.arith[opener].token_id, *left_ids, table.arith[op].token_id, *right_ids, table.arith[closer].token_id]
    tree = {
        "kind": "op",
        "op": op,
        "open": opener,
        "close": closer,
        "left": left_t,
        "right": right_t,
        "value": value,
    }
    return ids, value, tree


def _annotate_spans(tree: dict, ids_len_offset: int = 0) -> int:
    """Fill lo/hi token spans on every node. Returns hi."""
    if tree["kind"] == "num":
        tree["lo"] = ids_len_offset
        tree["hi"] = ids_len_offset + 1
        return tree["hi"]
    # ids layout: open, left..., op, right..., close
    tree["lo"] = ids_len_offset
    cur = ids_len_offset + 1
    cur = _annotate_spans(tree["left"], cur)
    cur += 1  # operator
    cur = _annotate_spans(tree["right"], cur)
    cur += 1  # closer
    tree["hi"] = cur
    return cur


def _iter_internal(tree: dict, out: list[dict] | None = None) -> list[dict]:
    if out is None:
        out = []
    if tree["kind"] == "op":
        _iter_internal(tree["left"], out)
        _iter_internal(tree["right"], out)
        out.append(tree)
    return out


def _bounded_expr(rng: np.random.Generator, table: AtomTable, max_depth: int, star_prob: float):
    for _ in range(80):
        ids, value, tree = _gen_expr(
            rng, table, depth=0, max_depth=max_depth, star_prob=star_prob
        )
        _annotate_spans(tree, 0)
        internals = _iter_internal(tree)
        vals = [tree["value"], *[n["value"] for n in internals]]
        if all(abs(int(v)) <= ARITH_MAX_ABS for v in vals) and internals:
            return ids, value, tree, internals
    # Fallback: a tiny bracketed sum that always fits.
    ids, value, tree = _gen_expr(rng, table, depth=0, max_depth=1, star_prob=0.0)
    _annotate_spans(tree, 0)
    return ids, value, tree, _iter_internal(tree)


def generate_arith_row(
    table: AtomTable,
    recipe: ProbeRecipe,
    *,
    seq_len: int,
    variant: str,
    split: str,
    seed: int,
    row_index: int,
    rng: np.random.Generator,
) -> ProbeRow:
    n_items = max(recipe.n_items, 1)
    max_depth = int(recipe.extra.get("max_depth", 3))
    star_prob = float(recipe.extra.get("star_prob", 0.2))
    exprs = []
    body: list[int] = []
    starts = []
    for _ in range(n_items):
        e_ids, value, tree, internals = _bounded_expr(rng, table, max_depth, star_prob)
        starts.append(len(body))
        exprs.append((e_ids, value, tree, internals, len(body)))
        body.extend(e_ids)
        body.append(table.marker("dot"))
    # Query the earliest expression (farthest from the tail).
    target = 0
    e_ids, value, tree, internals, start = exprs[target]
    evid = start + len(e_ids)
    task_roll = int(rng.integers(0, 10))
    if task_roll < 2:
        task = "eval"
        query = [table.marker("query"), table.marker("eval")]
        answer = _int_ids(table, value)
        prize = float(len(answer)) * np.log2(10.0)  # digit-ish; documented as shortcut
        meta_extra = {"root_value": value, "shortcut": True}
    elif task_roll < 5:
        task = "match"
        # Pick the root opener (position 0 of the target expr).
        opener_pos = start
        closer_pos = start + len(e_ids) - 1
        query = [table.marker("query"), table.marker("match"), *_int_ids(table, opener_pos)]
        answer = _int_ids(table, closer_pos)
        prize = float(np.log2(max(seq_len, 2)))
        meta_extra = {"opener_pos": opener_pos, "closer_pos": closer_pos, "shortcut": False}
    else:
        task = "subexpr"
        k = min(int(recipe.extra.get("n_subexpr", 4)), len(internals))
        pick = [internals[int(i)] for i in rng.choice(len(internals), size=k, replace=False)]
        # Query preorder index among internals.
        index_of = {id(n): i for i, n in enumerate(internals)}
        query = [table.marker("query"), table.marker("sub")]
        for n in pick:
            query.extend(_int_ids(table, index_of[id(n)]))
            query.append(table.marker("dot"))
        answer = []
        for n in pick:
            answer.extend(_int_ids(table, int(n["value"])))
            answer.append(table.marker("dot"))
        if answer:
            answer = answer[:-1]
        prize = float(k) * np.log2(2 * ARITH_MAX_ABS + 1)
        meta_extra = {
            "nodes": [{"index": index_of[id(n)], "value": int(n["value"])} for n in pick],
            "shortcut": False,
        }
    meta = {
        "n_items": n_items,
        "root_value": value,
        "n_internal": len(internals),
        "max_depth": max_depth,
        "glued": table.render(e_ids).replace(" ", ""),
        **meta_extra,
    }
    return _finalize(
        table=table,
        family="arith",
        task=task,
        variant=variant,
        seq_len=seq_len,
        split=split,
        seed=seed,
        row_index=row_index,
        body=body,
        query=query,
        answer=answer,
        evidence_end_in_body=evid,
        prize_bits=float(prize),
        meta=meta,
    )


# --------------------------------------------------------------------------------------
# props
# --------------------------------------------------------------------------------------


def generate_props_row(
    table: AtomTable,
    recipe: ProbeRecipe,
    *,
    seq_len: int,
    variant: str,
    split: str,
    seed: int,
    row_index: int,
    rng: np.random.Generator,
) -> ProbeRow:
    agents = table.pools["agents"]
    colors = table.pools["colors"]
    places = table.pools["places"]
    objects = table.pools["objects"]
    n = min(recipe.n_items, len(objects), len(agents), len(colors), len(places))
    obj = _pick(rng, objects, n, replace=False)
    ag = _pick(rng, agents, n, replace=True)
    col = _pick(rng, colors, n, replace=True)
    plc = _pick(rng, places, n, replace=True)
    body: list[int] = []
    ends = []
    for i in range(n):
        body.extend(
            [
                table.marker("the"),
                ag[i].token_id,
                table.marker("dropped"),
                table.marker("the"),
                col[i].token_id,
                obj[i].token_id,
                table.marker("in"),
                plc[i].token_id,
                table.marker("dot"),
            ]
        )
        ends.append(len(body))
    n_query = min(recipe.n_query, n)
    q_idx = [int(i) for i in rng.permutation(n)[:n_query]]
    query = [table.marker("query"), table.marker("color")] + [obj[i].token_id for i in q_idx]
    answer = [col[i].token_id for i in q_idx]
    evid = max(ends[i] for i in q_idx)
    prize = prize_bits_uniform(n_query, len(colors))
    meta = {
        "n_items": n,
        "n_query": n_query,
        "propositions": [
            {
                "agent": ag[i].surface,
                "color": col[i].surface,
                "object": obj[i].surface,
                "place": plc[i].surface,
            }
            for i in range(n)
        ],
        "query_objects": [obj[i].surface for i in q_idx],
        "content_fingerprint": sorted(
            f"{obj[i].surface}={col[i].surface}@{plc[i].surface}" for i in range(n)
        ),
    }
    return _finalize(
        table=table,
        family="props",
        task="prop_color",
        variant=variant,
        seq_len=seq_len,
        split=split,
        seed=seed,
        row_index=row_index,
        body=body,
        query=query,
        answer=answer,
        evidence_end_in_body=evid,
        prize_bits=prize,
        meta=meta,
    )


GENERATORS: dict[str, Callable[..., ProbeRow]] = {
    "bits": generate_bits_row,
    "bind": generate_bind_row,
    "arith": generate_arith_row,
    "props": generate_props_row,
}

FAMILY_SEED_TAG = {"bits": 1, "bind": 2, "arith": 3, "props": 4}
SPLIT_SEED_TAG = {"train": 11, "validation": 22, "test": 33}
VARIANT_SEED_TAG = {"scaled": 101, "fixed": 202}


def generate_row(
    family: str,
    table: AtomTable,
    *,
    seq_len: int,
    variant: str,
    split: str,
    seed: int,
    row_index: int,
) -> ProbeRow:
    recipe = recipe_for(family, seq_len, variant)
    rng = _rng_for(
        seed,
        FAMILY_SEED_TAG[family],
        seq_len,
        SPLIT_SEED_TAG[split],
        VARIANT_SEED_TAG[variant],
        row_index,
    )
    return GENERATORS[family](
        table,
        recipe,
        seq_len=seq_len,
        variant=variant,
        split=split,
        seed=seed,
        row_index=row_index,
        rng=rng,
    )


def generate_split(
    family: str,
    table: AtomTable,
    *,
    seq_len: int,
    variant: str,
    split: str,
    seed: int,
    n_rows: int,
) -> list[ProbeRow]:
    return [
        generate_row(
            family,
            table,
            seq_len=seq_len,
            variant=variant,
            split=split,
            seed=seed,
            row_index=i,
        )
        for i in range(n_rows)
    ]
