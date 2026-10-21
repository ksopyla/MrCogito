#!/usr/bin/env python
"""E33s data: real multi-hop QA and maths written as reasoning chains, in the text-checks data layout.

Modes (combine in one call, or run one at a time into the same --out_dir):
  --rule_check            P0: chains for every training question, the rule's coverage per dataset and question type,
                          and (MuSiQue) its agreement with the per-hop sub-answers → rule_check.json. CPU only.
  --train_tokenizer       a BPE of --vocab tokens on Wikipedia + FineWeb-Edu + QA paragraphs + GSM8K-Aug text.
  --pretrain_tokens T     phase-P rows: Wikipedia + FineWeb-Edu streamed into rows of --seq_len → text/{train,eval}.
  --stage S --qa_tokens T phase-M rows for stage S ∈ all | drop1 | none | answers → qa/{train,eval}, text/ replay
                          (--replay_from a pretrain dir, --replay share of tokens), with the read_targets column.
  --exams                 frozen eval/realqa.jsonl from the dev / test splits at --lengths, with removed twins.

Layout (what scripts/run_text_checks.py reads): tokenizer/, manifest.json, text_checks_meta.json (kind, mix.
mean_row_tokens, hashes), eval/. Spec: docs/experiments_specs/ahead/E33s_written_chains_real_data.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_text_checks_data as btc  # noqa: E402  (row writer, tokenizer trainer, language rows)

from data.real_multihop import (  # noqa: E402
    DATASETS,
    QAItem,
    assemble,
    chain_targets,
    load_items,
    normalize_answer,
    parse_chain,
    written_answer,
)

MULTIHOP = ("musique", "hotpotqa", "2wiki")
ROUNDS = 4  # E33's loop rounds: per-round targets beyond them would never be read
# share of the QA tokens per dataset (MuSiQue is the main exam: upweighted against its size)
QA_SHARES = {"musique": 0.30, "hotpotqa": 0.25, "2wiki": 0.20, "gsm8k": 0.25}


def _body(it: QAItem) -> str:
    return "\n\n".join(f"{t}\n{x}" for t, x, _ in it.paragraphs)


def derive(it: QAItem) -> list[tuple[str, str]]:
    return parse_chain(it.steps, it.answer, it.question, _body(it) or it.question, math=it.kind == "math")


# ------------------------------------------------------------------ P0
def rule_check(names, limit, out: Path) -> dict:
    res = {}
    for name in names:
        t0 = time.time()
        items = load_items(name, "train", limit)
        by_type, agree, total = defaultdict(lambda: [0, 0]), 0, 0
        n_steps = Counter()
        for it in items:
            d = derive(it)
            by_type[it.qtype or name][0] += 1
            by_type[it.qtype or name][1] += bool(d)
            n_steps[len(it.steps)] += 1
            if name == "musique" and d and it.meta.get("sub_answers"):
                subs = it.meta["sub_answers"][:-1]
                outs = [o for _, o in d][:-1]
                for o, s in zip(outs, subs):
                    total += 1
                    no, ns = normalize_answer(o), normalize_answer(s)
                    agree += bool(no) and bool(ns) and (no in ns or ns in no)
        n = len(items)
        cov = sum(v[1] for v in by_type.values()) / max(1, n)
        res[name] = {"items": n, "coverage": round(cov, 4),
                     "by_type": {k: {"n": v[0], "coverage": round(v[1] / max(1, v[0]), 4)} for k, v in sorted(by_type.items())},
                     "steps": dict(sorted(n_steps.items())), "seconds": round(time.time() - t0, 1)}
        if name == "musique":
            res[name]["bridge_agreement_with_sub_answers"] = round(agree / max(1, total), 4)
            res[name]["bridges_checked"] = total
        print(name, json.dumps(res[name]), flush=True)
    (out / "rule_check.json").write_text(json.dumps(res, indent=2))
    return res


# ------------------------------------------------------------------ text sources
def stream_text(n_docs: int, seed: int, skip: int = 0):
    """Wikipedia and FineWeb-Edu documents, alternating (streamed; nothing downloaded in full)."""
    from datasets import load_dataset

    wiki = iter(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000))
    web = iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000))
    for _ in range(skip):
        next(wiki), next(web)
    k = 0
    while k < n_docs:
        for src in (wiki, web):
            try:
                yield next(src)["text"]
            except StopIteration:
                return
            k += 1


def text_rows(docs, n_rows: int, seq_len: int, tok, bos: int, eos: int, batch: int = 512):
    """Documents concatenated (blank line between) into rows of exactly seq_len tokens; long ones run on."""
    cur, made, buf = [], 0, []
    sep = tok.encode("\n\n", add_special_tokens=False)

    def flush_docs(texts):
        nonlocal cur, made
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            cur += (sep if cur else []) + ids
            while len(cur) >= seq_len - 2 and made < n_rows:
                yield btc._row(cur[: seq_len - 2], bos, eos, [] if btc._VERSION["read_targets"] else None)
                made += 1
                cur = cur[seq_len - 2:]
                if len(cur) < 256:
                    cur = []

    for d in docs:
        buf.append(d)
        if len(buf) >= batch:
            yield from flush_docs(buf)
            buf = []
            if made >= n_rows:
                return
    if buf and made < n_rows:
        yield from flush_docs(buf)


# ------------------------------------------------------------------ QA rows
def qa_rows(items_by: dict, shares: dict, n_tokens: float, seq_len: int, tok, bos: int, eos: int, keep: str,
            seed: int, pools: dict):
    """Rows until each dataset has its share of `n_tokens`. Multi-hop: one question per row, its context padded
    with same-dataset distractors to a log-uniform length up to seq_len. Maths: questions packed into a row."""
    rng = random.Random(seed)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    got, fails = Counter(), Counter()
    order = [n for n in shares if n in items_by and items_by[n]]
    while order:
        for n in [n for n in order if fails[n] >= 200]:  # 200 rows in a row that do not fit: give the dataset up
            print(f"warning: {n} rows do not fit in {seq_len} tokens; dropped from the mix", flush=True)
            order.remove(n)
        if not order:
            break
        name = min(order, key=lambda n: got[n] / (shares[n] * n_tokens))
        if got[name] >= shares[name] * n_tokens:
            order.remove(name)
            continue
        items = items_by[name]
        if name == "gsm8k":
            parts, used, chains = [], 0, []
            while used < seq_len - 400:
                it = items[rng.randrange(len(items))]
                d = derive(it)
                steps = [s for s, _ in d] if d else it.steps
                block = "\n\n" + assemble(it, 0, ntok, rng, []) + written_answer(steps, it.answer, keep)
                n = ntok(block)
                if used + n > seq_len - 4:
                    break
                parts.append(block)
                chains.append((it.question, d))
                used += n
            text = "".join(parts)
            in_answer = True
        else:
            it = items[rng.randrange(len(items))]
            d = derive(it)
            steps = [s for s, _ in d] if d else it.steps
            target = int(math.exp(rng.uniform(math.log(512), math.log(seq_len - 64))))
            text = None
            for shrink in (1.0, 0.85, 0.7, 0.5):
                doc = assemble(it, int(target * shrink), ntok, rng, pools[name])
                cand = doc + written_answer(steps, it.answer, keep)
                if ntok(cand) <= seq_len - 2:
                    text = cand
                    break
            if text is None:
                fails[name] += 1
                continue
            fails[name] = 0
            chains = [(it.question, d)]
            in_answer = False
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        if len(ids) > seq_len - 2:
            fails[name] += 1
            continue
        rt = []
        if keep != "answers":
            for q, d in chains:
                if len(d) > ROUNDS:  # the loop has 4 rounds: steps 1–3, then the answer on round 4
                    d = d[: ROUNDS - 1] + d[-1:]
                if d:
                    rt += chain_targets(text, enc["offset_mapping"], q, d, steps_in_answer=in_answer)
                else:  # no chain found: the answer position alone (round 0)
                    rt += chain_targets(text, enc["offset_mapping"], q, [], steps_in_answer=in_answer)
        got[name] += len(ids)
        yield btc._row(ids, bos, eos, rt)


def _pools(items: list[QAItem], k: int = 6) -> list:
    """Distractor paragraphs: every non-supporting paragraph of the items (title, text, False)."""
    return [p for it in items for p in it.paragraphs if not p[2] and len(p[1]) > 80]


# ------------------------------------------------------------------ exams
def build_exams(names, lengths, per_dataset: int, tok, out: Path, seed: int):
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    path = out / "eval" / "realqa.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    depths = (("early", 0.1), ("middle", 0.5), ("late", 0.9))
    n = 0
    with path.open("w") as fh:
        for name in names:
            items = load_items(name, "eval")
            rng = random.Random(seed)
            rng.shuffle(items)
            items = items[:per_dataset]
            pool = _pools(load_items(name, "eval")) if name != "gsm8k" else []
            for L in (lengths if name != "gsm8k" else lengths[:1]):
                for i, it in enumerate(items):
                    dname, depth = depths[i % 3]
                    s = seed * 1_000_003 + i * 131 + L
                    prompt = assemble(it, L, ntok, random.Random(s), pool, depth=depth)
                    removed = assemble(it, L, ntok, random.Random(s), pool, depth=depth, remove_evidence=True)
                    rec = {"id": f"{name}-{it.id}-{L}", "split": "dev", "task": name, "dataset": name, "length": L,
                           "depth": dname if name != "gsm8k" else "none", "prompt": prompt, "answer": it.answer,
                           "aliases": list(it.aliases), "removed_prompt": removed if name != "gsm8k" else None,
                           "qtype": it.qtype, "n_tokens": ntok(prompt)}
                    fh.write(json.dumps(rec) + "\n")
                    n += 1
    print(f"exams: {n} items → {path}", flush=True)
    return {"realqa.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()}, n


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--limit", type=int, default=None, help="per dataset and split (smoke builds)")
    p.add_argument("--rule_check", action="store_true")
    p.add_argument("--train_tokenizer", action="store_true")
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--tokenizer_docs", type=int, default=60_000)
    p.add_argument("--tokenizer", default=None, help="use this tokenizer folder (copied into the build)")
    p.add_argument("--pretrain_tokens", type=float, default=0)
    p.add_argument("--stage", default=None, choices=("all", "drop1", "none", "answers"))
    p.add_argument("--qa_tokens", type=float, default=0)
    p.add_argument("--replay_from", default=None, help="a pretrain build whose text rows are replayed")
    p.add_argument("--replay", type=float, default=0.3, help="share of the stage's tokens that are text replay")
    p.add_argument("--exams", action="store_true")
    p.add_argument("--exams_from", default=None, help="link eval/ to this build's frozen exams instead")
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--per_dataset", type=int, default=300)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.rule_check:
        rule_check(args.datasets, args.limit, out)

    from transformers import AutoTokenizer

    tok = None
    if args.train_tokenizer:
        texts = list(stream_text(args.tokenizer_docs, args.seed, skip=0))
        for name in MULTIHOP:
            if name in args.datasets:
                texts += [_body(it) + "\n\n" + it.question for it in load_items(name, "train", 20_000)]
        if "gsm8k" in args.datasets:
            texts += [it.question + " " + " ".join(it.steps) for it in load_items("gsm8k", "train", 20_000)]
        tok = btc.train_tokenizer(texts, args.vocab, out / "tokenizer")
        print(f"tokenizer: {len(tok)} tokens from {len(texts):,} texts ({time.time() - t0:.0f}s)", flush=True)
    elif args.tokenizer:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok.save_pretrained(str(out / "tokenizer"))
    elif (out / "tokenizer").exists():
        tok = AutoTokenizer.from_pretrained(str(out / "tokenizer"))
    if tok is None and (args.pretrain_tokens or args.stage or args.exams):
        raise SystemExit("no tokenizer: pass --train_tokenizer or --tokenizer")
    if tok is None:
        return
    bos, eos = tok.bos_token_id, tok.eos_token_id
    meta = {"version": "e33s-real-v1", "args": vars(args), "vocab": len(tok),
            "tokenizer_sha256": btc._sha(out / "tokenizer" / "tokenizer.json")}
    sources, mix = [], {}

    if args.pretrain_tokens:
        btc._VERSION["read_targets"] = False
        n_rows = math.ceil(args.pretrain_tokens / (args.seq_len - 2))
        btc._save(lambda: text_rows(stream_text(10**9, args.seed, skip=2000), n_rows, args.seq_len, tok, bos, eos),
                  out / "text" / "train")
        btc._save(text_rows(stream_text(1000, args.seed + 1), 256, args.seq_len, tok, bos, eos), out / "text" / "eval")
        sources.append({"name": "text", "train_path": str((out / "text" / "train").resolve()),
                        "eval_path": str((out / "text" / "eval").resolve()), "weight": 1.0, "in_eval": True})
        mix = {"kind": "real_text", "text_rows": n_rows, "mean_row_tokens": float(args.seq_len)}
        meta["kind"] = "real_text"
        print(f"pretrain rows: {n_rows:,} ({time.time() - t0:.0f}s)", flush=True)

    if args.stage:
        from datasets import load_from_disk

        btc._VERSION["read_targets"] = True
        items_by = {n: load_items(n, "train", args.limit) for n in args.datasets}
        held = {n: v[-max(1, len(v) // 200):] for n, v in items_by.items()}
        items_by = {n: v[: len(v) - len(held[n])] for n, v in items_by.items()}
        pools = {n: _pools(v) for n, v in items_by.items() if n != "gsm8k"}
        hpools = {n: _pools(v) for n, v in held.items() if n != "gsm8k"}
        shares = {n: QA_SHARES[n] for n in args.datasets}
        tot = sum(shares.values())
        shares = {n: s / tot for n, s in shares.items()}
        qa_t = args.qa_tokens * (1 - args.replay)
        btc._save(lambda: qa_rows(items_by, shares, qa_t, args.seq_len, tok, bos, eos, args.stage, args.seed, pools),
                  out / "qa" / "train")
        btc._save(qa_rows(held, shares, min(qa_t, 2e6), args.seq_len, tok, bos, eos, args.stage, args.seed + 1, hpools),
                  out / "qa" / "eval")
        qa_tr = load_from_disk(str(out / "qa" / "train"))
        m_qa = btc._mean_len(qa_tr)
        sources.append({"name": "qa", "train_path": str((out / "qa" / "train").resolve()),
                        "eval_path": str((out / "qa" / "eval").resolve()), "weight": None, "in_eval": True})
        m_text, w_text, w_qa = 0.0, 0.0, 1.0
        if args.replay_from and args.replay > 0:
            src = load_from_disk(str(Path(args.replay_from) / "text" / "train"))
            n_rep = min(len(src), math.ceil(args.qa_tokens * args.replay / (args.seq_len - 2)))
            rep = src.select(range(n_rep))
            btc._save(lambda: (btc._row(list(r["input_ids"])[1:-1], bos, eos, []) for r in rep), out / "text" / "train")
            ev = load_from_disk(str(Path(args.replay_from) / "text" / "eval"))
            btc._save((btc._row(list(r["input_ids"])[1:-1], bos, eos, []) for r in ev), out / "text" / "eval")
            m_text = float(args.seq_len)
            f = args.replay
            w_text, w_qa = f / m_text, (1 - f) / m_qa
            z = w_text + w_qa
            w_text, w_qa = w_text / z, w_qa / z
            sources.insert(0, {"name": "text", "train_path": str((out / "text" / "train").resolve()),
                               "eval_path": str((out / "text" / "eval").resolve()), "weight": w_text, "in_eval": True})
        for s in sources:
            if s["name"] == "qa":
                s["weight"] = w_qa
        mix = {"kind": "real_multihop", "stage": args.stage, "qa_rows": len(qa_tr), "mean_qa_row_tokens": m_qa,
               "mean_text_row_tokens": m_text, "row_weights": [w_text, w_qa],
               "mean_row_tokens": w_text * m_text + w_qa * m_qa if m_text else m_qa,
               "qa_shares": shares, "replay": args.replay if args.replay_from else 0.0}
        meta["kind"] = "real_multihop"
        print(f"stage {args.stage}: {len(qa_tr):,} qa rows (mean {m_qa:.0f} tokens) ({time.time() - t0:.0f}s)", flush=True)

    if args.exams_from:
        ev = out / "eval"
        if not ev.exists():
            ev.symlink_to(Path(args.exams_from).resolve() / "eval")
        meta["eval_sha256"] = json.loads((Path(args.exams_from) / "text_checks_meta.json").read_text()).get("eval_sha256")
    elif args.exams:
        hashes, n = build_exams(args.datasets, args.lengths, args.per_dataset, tok, out, args.seed)
        meta["eval_sha256"], meta["eval_items"] = hashes, n

    if sources:
        manifest = {"mix_id": f"e33s_{meta.get('kind')}_{args.stage or 'pretrain'}", "max_seq_length": args.seq_len,
                    "objective": "causal_lm", "seed": args.seed, "tokenizer": str((out / "tokenizer").resolve()),
                    "sources": sources}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    old = json.loads((out / "text_checks_meta.json").read_text()) if (out / "text_checks_meta.json").exists() else {}
    old.update({k: v for k, v in meta.items() if v is not None})
    if mix:
        old["mix"] = mix
    old["built_s"] = round(time.time() - t0, 1)
    (out / "text_checks_meta.json").write_text(json.dumps(old, indent=2, default=str))
    print(json.dumps({k: old[k] for k in ("version", "kind", "vocab") if k in old}), flush=True)


if __name__ == "__main__":
    main()
    # streamed HF datasets can leave background threads that keep the interpreter alive after main() returns
    # (seen 21 Oct: a finished build never exited); everything is written and flushed by now
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
