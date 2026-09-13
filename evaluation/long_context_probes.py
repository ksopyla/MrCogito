#!/usr/bin/env python
"""Long-context probes for the Perceiver AR v2 family (E18 pilot gates P2/P3; E18b/E18c/E21 suite).

All probes are teacher-forced (no generation needed, so they run at 32k–128k on a 3090):

  * buckets   — position-bucketed CE on long documents: does context beyond 8k lower the loss?
  * passkey   — NIAH-single: argmax accuracy over the 5 answer digits (RULER `niah_single`)
  * multikey  — NIAH-multikey: the target key among n_keys distractor keys (RULER `niah_multikey`)
  * vt        — variable tracking: hops-long assignment chains scattered in the filler; the answer
                is the ordered list of variables holding the queried value (RULER `vt`)
  * fwe       — frequent-words extraction: the three most frequent coded words of a shuffled
                list (RULER `fwe`, the aggregation axis; no filler)
  * copy      — token accuracy on the second half of a copy row (E18 gate P2)
  * tasks     — greedy accuracy on the marked spans of held-out E18b retrieval rows
  * reach     — paired reach ablation: same rows, same weights, the global read restricted to
                swa(W) for several W — does the loss at far positions depend on DIRECT access to
                far keys? Confound-free; reports paired Δ with a standard error.
  * suite     — several of the above in ONE process (model loaded and compiled once), merged JSON

The synthetic probes are a teacher-forced RULER-lite for *base* LMs: scored by greedy argmax on
the answer tokens given the gold prefix, which equals greedy generation's exact match whenever the
first answer token is right and needs no decode loop, instruction following or chat template. The
generation-based `ruler` group of lm-eval becomes the drop-in once a checkpoint is instruction
tuned and `generate()` is KV-cached (E18 main run).

Any probe accepts `--reach_window W` (e.g. the P2 copy model with W below the copy offset is the
positive control: accuracy must collapse if the global read is the retrieval channel).

Usage:
  uv run python evaluation/long_context_probes.py --checkpoint <dir> --probe buckets \
      --manifest <eval manifest> --buckets 2048,8192,16384,32768
  uv run python evaluation/long_context_probes.py --checkpoint <dir> --probe passkey \
      --manifest <eval manifest> --context_lengths 4096,8192,16384,32768
  uv run python evaluation/long_context_probes.py --checkpoint <dir> --probe suite \
      --suite passkey,multikey,vt,fwe,buckets --manifest <eval manifest> \
      --context_lengths 8192,32768 --trials 4 --out Cache/eval/<tag>/longctx_suite.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM  # noqa: E402


DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM3-3B"


def resolve_tokenizer_name(checkpoint: str, tokenizer: str | None = None) -> str:
    """Explicit name wins; else the tokenizer saved next to the weights; else the family default."""
    if tokenizer:
        return tokenizer
    ck = Path(checkpoint)
    if (ck / "tokenizer.json").exists() or (ck / "tokenizer_config.json").exists():
        return str(ck)
    return DEFAULT_TOKENIZER


def load_model(checkpoint: str, device: str, attn_backend: str | None = None):
    """Loads `perceiver_ar` (E18) or `perceiver_concept` (E22) from the checkpoint's config."""
    from nn.perceiver_families import load_perceiver_lm

    return load_perceiver_lm(checkpoint, device, attn_backend)


def _load_tokenizer(args):
    from transformers import AutoTokenizer

    name = getattr(args, "tokenizer", None)
    explicit = name if name not in (None, "", DEFAULT_TOKENIZER) else None
    return AutoTokenizer.from_pretrained(resolve_tokenizer_name(args.checkpoint, explicit))


def load_eval_rows(manifest: str, min_len: int, max_rows: int) -> list[list[int]]:
    from datasets import load_from_disk

    man = json.loads(Path(manifest).read_text())
    rows: list[list[int]] = []
    for src in man["sources"]:
        ds = load_from_disk(src["eval_path"])
        for r in ds:
            ids = r["input_ids"]
            if len(ids) >= min_len:
                rows.append(list(ids))
            if len(rows) >= max_rows:
                return rows
    return rows


@torch.no_grad()
def argmax_tokens(model, x: torch.Tensor, start: int, end: int, chunk: int = 2048) -> torch.Tensor:
    """Greedy next-token predictions for hidden positions [start, end) of x [1, S].

    Uses `model.hidden_states` + a chunked head projection so memory stays O(chunk × V)
    instead of O(S × V): full logits at 32k are 16 GB, which is why the passkey / copy probes
    OOM'd next to a training run. Argmax is invariant to the tanh soft-cap, so it is skipped.
    """
    h = model.hidden_states(x)[0]  # [S, d]
    S = h.shape[0]
    start = start % S if start < 0 else start
    end = S if end is None else (end % S if end < 0 else end)
    weight = model.lm_head.weight
    preds = []
    for lo in range(start, end, chunk):
        hi = min(lo + chunk, end)
        preds.append(torch.nn.functional.linear(h[lo:hi], weight).argmax(-1))
    return torch.cat(preds) if preds else h.new_zeros(0, dtype=torch.long)


@torch.no_grad()
def per_token_ce(model, ids: list[int], device: str, max_len: int) -> torch.Tensor:
    x = torch.tensor(ids[:max_len], device=device)[None]
    _, per, valid = model(input_ids=x, labels=x.clone(), return_per_token_loss=True)
    return per[0][valid[0]].float().cpu()


def bucket_means(per: torch.Tensor, edges: list[int]) -> list[float]:
    """Mean per-token CE inside consecutive position buckets [0,e0), [e0,e1), ... ."""
    out, lo = [], 0
    for hi in edges:
        seg = per[lo:hi]
        out.append(float(seg.mean()) if seg.numel() else float("nan"))
        lo = hi
    return out


def _bucket_labels(edges: list[int]) -> list[str]:
    labels, lo = [], 0
    for hi in edges:
        labels.append(f"[{lo},{hi})")
        lo = hi
    return labels


def reach_tail_stats(delta: torch.Tensor, thresh: float = 0.1) -> dict:
    """Distribution of per-token Δ = CE(W) − CE(full). A retrieval channel used on a few tokens
    shows up here (fraction of tokens that got much worse, mean of the worst 1%) even when the
    bucket mean is ~0."""
    d = delta.float()
    n = int(d.numel())
    if n == 0:
        return {"n_tokens": 0}
    k = max(1, n // 100)
    worst = torch.topk(d, k).values
    return {
        "n_tokens": n,
        f"frac_worse_gt_{thresh}": float((d > thresh).float().mean()),
        f"frac_better_gt_{thresh}": float((d < -thresh).float().mean()),
        "top1pct_mean": float(worst.mean()),
        "max": float(d.max()),
        "min": float(d.min()),
    }


def probe_reach(model, args, device) -> dict:
    """Paired reach ablation over `--reach_windows` (comma list; 'full' = unrestricted).

    For every window W the same rows are scored with every `full` layer restricted to swa(W);
    Δ(W) = CE(W) − CE(full) per row and bucket, reported as mean ± standard error over rows.
    Buckets entirely below W are computed from exactly the same keys, so their Δ is the numerical
    noise floor of the backend (exactly 0 on the sdpa/fp32 path).
    """
    edges = [int(x) for x in args.buckets.split(",")]
    rows = load_eval_rows(args.manifest, min_len=edges[-1], max_rows=args.max_rows)
    if not rows:
        raise SystemExit(f"no eval rows with >= {edges[-1]} tokens in {args.manifest}")
    windows: list[int | None] = []
    for w in args.reach_windows.split(","):
        w = w.strip()
        windows.append(None if w == "full" else int(w))
    if None not in windows:
        windows.append(None)
    labels = _bucket_labels(edges)
    per_row: dict[str, list[list[float]]] = {}
    per_tok: dict[str, torch.Tensor] = {}   # concatenated per-token CE over all rows (tail stats)
    touched: list[int] = []
    for w in windows:
        key = "full" if w is None else str(w)
        with model.reach_override(w) as t:
            touched = t or touched
            toks = [per_token_ce(model, ids, device, edges[-1]) for ids in rows]
            per_row[key] = [bucket_means(pt, edges) for pt in toks]
            per_tok[key] = torch.cat(toks)
    base = torch.tensor(per_row["full"])  # [rows, buckets]
    out: dict = {"rows": len(rows), "buckets": labels, "windows": [k for k in per_row],
                 "touched_layers": touched, "ce": {}, "delta_vs_full": {}, "tail": {}}
    for key, pt in per_tok.items():
        if key == "full":
            continue
        out["tail"][key] = reach_tail_stats(pt - per_tok["full"])
    for key, vals in per_row.items():
        t = torch.tensor(vals)
        out["ce"][key] = {lab: float(t[:, b].mean()) for b, lab in enumerate(labels)}
        if key == "full":
            continue
        d = t - base
        n = d.shape[0]
        out["delta_vs_full"][key] = {
            lab: {
                "mean": float(d[:, b].mean()),
                "se": float(d[:, b].std(unbiased=True) / (n ** 0.5)) if n > 1 else float("nan"),
                "n": n,
            }
            for b, lab in enumerate(labels)
        }
    out["per_row"] = per_row
    return out


def probe_buckets(model, args, device) -> dict:
    edges = [int(x) for x in args.buckets.split(",")]
    rows = load_eval_rows(args.manifest, min_len=edges[-1], max_rows=args.max_rows)
    if not rows:
        raise SystemExit(f"no eval rows with >= {edges[-1]} tokens in {args.manifest}")
    sums = [0.0] * len(edges)
    counts = [0] * len(edges)
    for ids in rows:
        per = per_token_ce(model, ids, device, edges[-1])
        lo = 0
        for bi, hi in enumerate(edges):
            seg = per[lo:hi]
            sums[bi] += float(seg.sum())
            counts[bi] += int(seg.numel())
            lo = hi
    out = {"rows": len(rows)}
    lo = 0
    for bi, hi in enumerate(edges):
        out[f"ce[{lo},{hi})"] = sums[bi] / max(counts[bi], 1)
        lo = hi
    return out


def build_filler(rows: list[list[int]], start_row: int, budget: int) -> list[int]:
    """Concatenate eval rows cyclically from `start_row` until at least `budget` ids (lets the
    passkey probe run at lengths beyond the tokenized row length, e.g. 128k on a 32k tree)."""
    out: list[int] = []
    i = start_row
    while len(out) < budget:
        out.extend(rows[i % len(rows)])
        i += 1
    return out[:budget]


def build_passkey(tokenizer, filler_ids: list[int], context_len: int, depth: float, rng,
                  frame: list[int] | None = None) -> tuple[list[int], list[int]]:
    key = f"{rng.randint(0, 99999):05d}"
    needle = tokenizer.encode(f" The pass key is {key}. Remember it. ", add_special_tokens=False)
    question = tokenizer.encode(" What is the pass key? The pass key is", add_special_tokens=False)
    if frame:
        question = question + list(frame)   # diagnostic: end the question with the training frame token
    answer = tokenizer.encode(f" {key}", add_special_tokens=False)
    budget = context_len - len(needle) - len(question) - len(answer)
    filler = filler_ids[:budget]
    cut = int(len(filler) * depth)
    ids = filler[:cut] + needle + filler[cut:] + question + answer
    return ids, answer


def probe_passkey(model, args, device) -> dict:
    tok = _load_tokenizer(args)
    lengths = [int(x) for x in args.context_lengths.split(",")]
    # filler rows need not be as long as the context: pieces are concatenated (build_filler)
    rows = load_eval_rows(args.manifest, min_len=min(8192, max(lengths)), max_rows=args.max_rows)
    if not rows:
        raise SystemExit("no filler rows long enough")
    rng = random.Random(args.seed)
    frame = [int(args.frame_token)] if getattr(args, "frame_token", None) is not None else None
    results = {}
    for L in lengths:
        correct = total = 0
        for depth in (0.1, 0.3, 0.5, 0.7, 0.9):
            for trial in range(args.trials):
                filler = build_filler(rows, (trial * 7 + int(depth * 10)) % len(rows), L)
                ids, answer = build_passkey(tok, filler, L, depth, rng, frame=frame)
                x = torch.tensor(ids, device=device)[None]
                n = len(answer)
                pred = argmax_tokens(model, x, x.shape[1] - n - 1, x.shape[1] - 1).tolist()
                correct += int(pred == answer)
                total += 1
        results[f"passkey@{L}"] = correct / total
    return results


# --------------------------------------------------------------------------------------
# RULER-lite synthetic tasks (teacher-forced; see module docstring)
# --------------------------------------------------------------------------------------

_KEY_NAMES = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
]
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _insert_at_depths(filler: list[int], pieces: list[tuple[float, list[int]]]) -> list[int]:
    """Splice token pieces into `filler` at fractional depths (sorted; stable for equal depths)."""
    out: list[int] = []
    prev = 0
    for depth, piece in sorted(pieces, key=lambda p: p[0]):
        cut = int(len(filler) * depth)
        out.extend(filler[prev:cut])
        out.extend(piece)
        prev = cut
    out.extend(filler[prev:])
    return out


def _score_answer(model, ids: list[int], answer: list[int], device) -> dict:
    """Greedy predictions for the trailing `answer` span of `ids` given the gold prefix."""
    x = torch.tensor(ids, device=device)[None]
    n = len(answer)
    pred = argmax_tokens(model, x, x.shape[1] - n - 1, x.shape[1] - 1).tolist()
    return {
        "exact": int(pred == answer),
        "tok_correct": sum(int(a == b) for a, b in zip(pred, answer)),
        "tok_total": n,
        "first": int(pred[0] == answer[0]) if n else 0,
    }


def _aggregate(scores: list[dict]) -> dict:
    n = max(len(scores), 1)
    return {
        "exact": sum(s["exact"] for s in scores) / n,
        "token_acc": sum(s["tok_correct"] for s in scores) / max(sum(s["tok_total"] for s in scores), 1),
        "first_token_acc": sum(s["first"] for s in scores) / n,
        "n": len(scores),
    }


def build_multikey(tokenizer, filler_ids: list[int], context_len: int, depth: float, rng,
                   n_keys: int = 4) -> tuple[list[int], list[int]]:
    """NIAH-multikey: `n_keys` "The pass key for <name> is <key>." needles (one target at `depth`,
    distractors at random depths), question about the target name. Distractor keys share the
    format, so the read must address by *name*, not by the needle template (RULER `niah_multikey`)."""
    names = rng.sample(_KEY_NAMES, n_keys)
    keys = [f"{rng.randint(0, 99999):05d}" for _ in names]
    target = 0
    needles = [tokenizer.encode(f" The pass key for {nm} is {k}. Remember it. ", add_special_tokens=False)
               for nm, k in zip(names, keys)]
    question = tokenizer.encode(f" What is the pass key for {names[target]}? The pass key for {names[target]} is",
                                add_special_tokens=False)
    answer = tokenizer.encode(f" {keys[target]}", add_special_tokens=False)
    budget = context_len - sum(len(n) for n in needles) - len(question) - len(answer)
    filler = filler_ids[:budget]
    depths = [depth] + [rng.uniform(0.05, 0.95) for _ in range(n_keys - 1)]
    ids = _insert_at_depths(filler, list(zip(depths, needles))) + question + answer
    return ids, answer


def build_variable_tracking(tokenizer, filler_ids: list[int], context_len: int, depth: float, rng,
                            hops: int = 3, n_chains: int = 2) -> tuple[list[int], list[int]]:
    """RULER `vt`: `n_chains` assignment chains ("VAR ABC = 12345", "VAR DEF = ABC", ...) whose
    statements are scattered from `depth` toward the end of the filler; the question names a
    value and the answer lists every variable that holds it, in chain order (multi-hop retrieval)."""
    def var_name():
        return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3))

    used: set[str] = set()
    chains: list[tuple[str, list[str]]] = []
    for _ in range(n_chains):
        names = []
        while len(names) < hops:
            v = var_name()
            if v not in used:
                used.add(v)
                names.append(v)
        chains.append((f"{rng.randint(10000, 99999)}", names))
    pieces: list[tuple[float, list[int]]] = []
    for value, names in chains:
        stmts = [f" VAR {names[0]} = {value} "] + [f" VAR {names[i]} = {names[i-1]} " for i in range(1, hops)]
        # chain statements in order, from `depth` toward the end; a later hop never precedes its source
        span = max(0.02, (0.95 - depth) / max(hops, 1))
        for i, s in enumerate(stmts):
            d = min(0.95, depth + i * span + rng.uniform(0.0, span * 0.5))
            pieces.append((d, tokenizer.encode(s, add_special_tokens=False)))
    value, names = chains[0]
    question = tokenizer.encode(
        f" Question: Find all variables that are assigned the value {value}. Answer: the variables are",
        add_special_tokens=False)
    answer = tokenizer.encode(" " + " ".join(names), add_special_tokens=False)
    budget = context_len - sum(len(p) for _, p in pieces) - len(question) - len(answer)
    filler = filler_ids[:budget]
    ids = _insert_at_depths(filler, pieces) + question + answer
    return ids, answer


def build_frequent_words(tokenizer, context_len: int, rng, n_answer: int = 3,
                         vocab_words: int = 40) -> tuple[list[int], list[int]]:
    """RULER `fwe` (aggregation): a shuffled list of coded words whose frequencies follow a
    steep power law; the answer is the `n_answer` most frequent words in order. Counts of the top
    words are made strictly decreasing so the answer is well defined. No filler: the list itself
    fills `context_len`."""
    def coded_word():
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 5)))

    words: list[str] = []
    while len(words) < vocab_words:
        w = coded_word()
        if w not in words:
            words.append(w)
    header = tokenizer.encode(" Read the following coded text and track the frequency of each coded word."
                              " Coded text:", add_special_tokens=False)
    question = tokenizer.encode(
        f" Question: What are the {n_answer} most frequently appeared words in the above coded text?"
        f" Answer: the {n_answer} most frequently appeared words are", add_special_tokens=False)
    answer = tokenizer.encode(" " + " ".join(words[:n_answer]), add_special_tokens=False)
    budget = context_len - len(header) - len(question) - len(answer)
    pieces = [tokenizer.encode(" " + w, add_special_tokens=False) for w in words]
    # Zipf-like weights; the answer words get strictly decreasing, dominant counts. The scale is
    # chosen so the expected token count of the list equals the budget (Σ count_i · len_i).
    weights = [1.0 / ((i + 1) ** 2.0) for i in range(vocab_words)]
    total = sum(weights)
    expected_tokens_per_unit = sum(w / total * len(p) for w, p in zip(weights, pieces))
    n_items = max(vocab_words * 2, int(budget / max(expected_tokens_per_unit, 1e-6)))
    counts = [max(1, int(n_items * w / total)) for w in weights]
    for i in range(1, n_answer + 1):
        if counts[i] >= counts[i - 1]:
            counts[i] = counts[i - 1] - 1
    items: list[int] = []
    for i, c in enumerate(counts):
        items.extend([i] * c)
    rng.shuffle(items)
    body: list[int] = []
    for i in items:
        if len(body) + len(pieces[i]) > budget:
            break
        body.extend(pieces[i])
    ids = header + body + question + answer
    return ids, answer


def _synthetic_probe(model, args, device, builder, name: str, with_filler: bool = True) -> dict:
    tok = _load_tokenizer(args)
    lengths = [int(x) for x in args.context_lengths.split(",")]
    rows = None
    if with_filler:
        rows = load_eval_rows(args.manifest, min_len=min(8192, max(lengths)), max_rows=args.max_rows)
        if not rows:
            raise SystemExit("no filler rows long enough")
    rng = random.Random(args.seed)
    results: dict = {}
    for L in lengths:
        scores: list[dict] = []
        depths = DEPTHS if with_filler else (0.5,)
        for depth in depths:
            for trial in range(args.trials):
                if with_filler:
                    filler = build_filler(rows, (trial * 7 + int(depth * 10)) % len(rows), L)
                    ids, answer = builder(tok, filler, L, depth, rng)
                else:
                    ids, answer = builder(tok, L, rng)
                if len(ids) > L:
                    ids = ids[-L:]
                scores.append(_score_answer(model, ids, answer, device))
        agg = _aggregate(scores)
        results[f"{name}@{L}"] = agg["exact"]
        results[f"{name}_token_acc@{L}"] = agg["token_acc"]
        results[f"{name}_first_token_acc@{L}"] = agg["first_token_acc"]
        results[f"{name}_n@{L}"] = agg["n"]
    return results


def probe_multikey(model, args, device) -> dict:
    n_keys = int(getattr(args, "n_keys", 4))
    return _synthetic_probe(model, args, device,
                            lambda t, f, L, d, r: build_multikey(t, f, L, d, r, n_keys=n_keys), "multikey")


def probe_vt(model, args, device) -> dict:
    hops = int(getattr(args, "vt_hops", 3))
    chains = int(getattr(args, "vt_chains", 2))
    return _synthetic_probe(model, args, device,
                            lambda t, f, L, d, r: build_variable_tracking(t, f, L, d, r, hops=hops, n_chains=chains),
                            "vt")


def probe_fwe(model, args, device) -> dict:
    return _synthetic_probe(model, args, device,
                            lambda t, L, r: build_frequent_words(t, L, r), "fwe", with_filler=False)


def probe_suite(model, args, device) -> dict:
    """Run several probes with one loaded model; per-probe failures are recorded, not fatal."""
    import time
    import traceback

    names = [s.strip() for s in args.suite.split(",") if s.strip()]
    out: dict = {"suite": names, "results": {}, "errors": {}, "elapsed_s": {}}
    for name in names:
        fn = PROBES[name]
        t0 = time.time()
        try:
            out["results"][name] = fn(model, args, device)
        except Exception as e:  # noqa: BLE001 — one failing probe must not discard the others
            out["errors"][name] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        out["elapsed_s"][name] = round(time.time() - t0, 1)
    return out


def _accuracy_on_labels(model, ids: list[int], labels: list[int], device) -> tuple[int, int]:
    """(#correct, #labelled) greedy next-token accuracy on the positions whose label != -100."""
    x = torch.tensor(ids, device=device)[None]
    tgt = torch.tensor(labels, device=device)[1:]
    pred = argmax_tokens(model, x, 0, x.shape[1] - 1)
    m = tgt != -100
    return int((pred[m] == tgt[m]).sum()), int(m.sum())


def probe_copy(model, args, device) -> dict:
    from datasets import load_from_disk

    ds = load_from_disk(args.copy_dataset)
    correct = total = 0
    for r in ds:
        c, n = _accuracy_on_labels(model, list(r["input_ids"]), list(r["labels"]), device)
        correct += c
        total += n
    return {"copy_token_accuracy": correct / max(total, 1), "rows": len(ds)}


def probe_tasks(model, args, device) -> dict:
    """Greedy accuracy on the marked spans of held-out retrieval rows (E18b training-task accuracy).
    Labels are derived exactly as the training collator derives them (`labels_from_span_markers`),
    so the number is the training objective's accuracy, not a separate format."""
    from datasets import load_from_disk

    from data.data_collators import labels_from_span_markers

    start, end = (int(x) for x in args.markers.split(","))
    ds = load_from_disk(args.tasks_dataset)
    correct = total = rows = 0
    first_tok_correct = first_tok_total = 0
    for r in ds:
        ids = list(r["input_ids"])
        if args.max_rows and rows >= args.max_rows:
            break
        labels = labels_from_span_markers(ids, start, end)
        c, n = _accuracy_on_labels(model, ids, labels, device)
        correct += c
        total += n
        # the first token after START is the pure retrieval decision (the rest is copy-continuation)
        x = torch.tensor(ids, device=device)[None]
        pred = argmax_tokens(model, x, 0, x.shape[1] - 1)
        for i, t in enumerate(ids[:-1]):
            if t == start and labels[i + 1] != -100:
                first_tok_total += 1
                first_tok_correct += int(int(pred[i]) == labels[i + 1])
        rows += 1
    return {
        "tasks_token_accuracy": correct / max(total, 1),
        "tasks_first_token_accuracy": first_tok_correct / max(first_tok_total, 1),
        "labelled_tokens": total, "rows": rows,
    }


def probe_concept(model, args, device) -> dict:
    """Paired concept ablation for the `perceiver_concept` family (E22 S1 instrument).

    The same rows are scored with the concept array `real`, `none` (the decoder's cross-attention
    dropped: segment-local raw context only), `shuffled` (each row reads the array of another
    row — a generic prior survives this, real content does not), `near` (only slots inside the
    token's raw segment stay visible — the array as extra local capacity) and `far` (only slots
    ending before the raw segment — the array as the long-range channel). Δ = CE(mode) − CE(real)
    per position bucket, mean ± standard error over rows. Rows are scored two at a time so
    `shuffled` has a partner. Tokens in the first concept block (positions < concept_ratio) see no
    slot and are identical under every mode, which is the built-in noise-floor check. The E22
    diagnosis (2026-09-12) reads `near` vs `far`: Δ_far ≈ 0 with Δ_near ≈ Δ_none means the array is
    used as local depth, not as memory.
    """
    if not hasattr(model, "concept_override"):
        raise SystemExit("--probe concept needs a perceiver_concept checkpoint")
    edges = [int(x) for x in args.buckets.split(",")]
    rows = load_eval_rows(args.manifest, min_len=edges[-1], max_rows=args.max_rows)
    if len(rows) < 2:
        raise SystemExit(f"need >= 2 eval rows with >= {edges[-1]} tokens in {args.manifest}")
    if len(rows) % 2:
        rows = rows[:-1]
    labels = _bucket_labels(edges)
    modes = [m.strip() for m in args.concept_modes.split(",") if m.strip()]
    if modes[:1] != ["real"]:
        raise SystemExit("--concept_modes must start with 'real' (the paired reference)")
    per_row: dict[str, list[list[float]]] = {m: [] for m in modes}
    per_tok: dict[str, list[torch.Tensor]] = {m: [] for m in modes}
    for i in range(0, len(rows), 2):
        pair = [ids[: edges[-1]] for ids in rows[i : i + 2]]
        x = torch.tensor(pair, device=device)
        for mode in modes:
            with torch.no_grad(), model.concept_override(mode):
                _, per, valid = model(input_ids=x, labels=x.clone(), return_per_token_loss=True)
            for b in range(2):
                pt = per[b][valid[b]].float().cpu()
                per_row[mode].append(bucket_means(pt, edges))
                per_tok[mode].append(pt)
    base = torch.tensor(per_row["real"])
    out: dict = {"rows": len(rows), "buckets": labels, "modes": modes, "ce": {}, "delta_vs_real": {}, "tail": {}}
    for mode in modes:
        t = torch.tensor(per_row[mode])
        out["ce"][mode] = {lab: float(t[:, b].mean()) for b, lab in enumerate(labels)}
        if mode == "real":
            continue
        d = t - base
        n = d.shape[0]
        out["delta_vs_real"][mode] = {
            lab: {"mean": float(d[:, b].mean()),
                  "se": float(d[:, b].std(unbiased=True) / (n ** 0.5)) if n > 1 else float("nan"),
                  "n": n}
            for b, lab in enumerate(labels)
        }
        out["tail"][mode] = reach_tail_stats(torch.cat(per_tok[mode]) - torch.cat(per_tok["real"]))
    out["per_row"] = per_row
    return out


PROBES = {
    "buckets": probe_buckets, "passkey": probe_passkey, "multikey": probe_multikey, "vt": probe_vt,
    "fwe": probe_fwe, "copy": probe_copy, "reach": probe_reach, "tasks": probe_tasks,
    "concept": probe_concept,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe", choices=sorted(PROBES) + ["suite"], required=True)
    p.add_argument("--suite", default="passkey,multikey,vt,fwe,buckets",
                   help="--probe suite: comma list of probes to run with one loaded model")
    p.add_argument("--n_keys", type=int, default=4, help="multikey: needles per row (1 target)")
    p.add_argument("--vt_hops", type=int, default=3, help="vt: assignments per chain")
    p.add_argument("--vt_chains", type=int, default=2, help="vt: chains per row (1 queried)")
    p.add_argument("--tasks_dataset", default=None, help="--probe tasks: arrow dir of held-out retrieval rows")
    p.add_argument("--markers", default="128103,128104", help="--probe tasks: 'start_id,end_id'")
    p.add_argument("--frame_token", type=int, default=None,
                   help="passkey diagnostic: append this id to the question (the training frame token)")
    p.add_argument("--reach_window", type=int, default=None,
                   help="restrict every full layer to swa(W) for this probe (positive-control runs)")
    p.add_argument("--reach_windows", default="512,2048,8192,full",
                   help="--probe reach: comma list of windows to sweep ('full' = unrestricted)")
    p.add_argument("--concept_modes", default="real,none,shuffled,near,far",
                   help="--probe concept: comma list of concept_override modes (must start with 'real')")
    p.add_argument("--manifest", default=None)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="default: the tokenizer saved in the checkpoint dir, else the family default")
    p.add_argument("--buckets", default="8192,32768")
    p.add_argument("--context_lengths", default="4096,8192,16384,32768")
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--max_rows", type=int, default=64)
    p.add_argument("--copy_dataset", default=None)
    p.add_argument("--attn_backend", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON output path")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device, args.attn_backend)
    fn = probe_suite if args.probe == "suite" else PROBES[args.probe]
    with model.reach_override(args.reach_window) as touched:
        res = fn(model, args, device)
    res["checkpoint"] = args.checkpoint
    res["probe"] = args.probe
    res["context_lengths"] = args.context_lengths
    res["seed"] = args.seed
    if args.reach_window is not None:
        res["reach_window"] = args.reach_window
        res["touched_layers"] = touched
    if getattr(args, "frame_token", None) is not None:
        res["frame_token"] = args.frame_token
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
