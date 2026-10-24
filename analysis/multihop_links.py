#!/usr/bin/env python
"""Multi-hop links: where does a compose / deduce answer break? Step 0 of the 17 Oct 2026 note
(docs/4_Research_Notes/e33_text_multihop_diagnosis_20261017.md). Nothing is trained.

A one-step compose answer ("Where does the teacher of Lumo live?") needs four links: read 1 finds the teacher fact,
the read carries the bridge ("Pim") out, the think step turns it into the next query, read 2 finds Pim's home. On a
trained text-checks checkpoint and the frozen exam items (each item rebuilt from its seed, so we know where every
fact sits), this measures:

  D1  latent-memory models: the share of each read round's attention (mean over heads) on the notes of the pages
      that hold fact r of the chain, at the answer position; the same on the single-step questions asked on the
      same document (what a read that works looks like), and the share a uniform read would give (`chance`)
  D2  linear probes on the state at the answer position after every layer call (a looped model: every round):
      which cast member is the bridge (and the second bridge on 2-step items), the asked person (a control the
      local layers see), the answer. Ridge map state → the name's mean token embedding, scored by the nearest
      candidate of the item; chance = 1 / candidates
  D3  the single steps on the same documents: "Who is the teacher of Lumo?" / "What is Lumo?" (scored over the
      whole candidate strings: names share their first token) and the bridge written in: "Where does Pim live?" /
      "What is a kavi like?"
  D4  compose: where the answer probability goes — the gold home, the asked person's own home (the lookup
      shortcut), the home of the person the asked one teaches (reverse direction), on 2-step items the 1-step
      answer (one hop done)
  D5  count: the predicted number (whole-candidate scoring) against the true one, by how many 192-token strides
      hold the visits (several visits in one stride share a page's notes)

  uv run python analysis/multihop_links.py --checkpoint <ckpt> --tokenizer <data>/tokenizer \\
      --items <data>/eval/steps.jsonl <data>/eval/id.jsonl --lengths 4096 --max_items 200 --out <out>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from data.text_world import build_item, qa_suffix  # noqa: E402
from evaluation.text_checks_eval import _answer_ids  # noqa: E402

TEACHER_Q = "Who is the teacher of "


# ------------------------------------------------------------------------------------------ items
def _has(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def chain_of(item, rec) -> dict | None:
    """The asked chain of a rebuilt item: its facts in hop order, the bridge(s), and the role answers."""
    if item.task == "compose":
        teacher = {q[len(TEACHER_Q):-1]: a for q, a in item.bridge}
        cast, places = item.cast, list(item.candidates)
        home, home_fact, teach_fact = {}, {}, {}
        for i, f in enumerate(item.facts):
            pl = [p for p in places if p in f.text]
            nm = [c for c in cast if _has(c, f.text)]
            if len(pl) == 1 and len(nm) == 1:
                home[nm[0]], home_fact[nm[0]] = pl[0], i
            elif not pl and len(nm) == 2:
                a, b = nm
                s = a if teacher.get(a) == b else b
                teach_fact[s] = i
        chain = [cast[0]]
        for _ in range(int(item.meta["hops"])):
            chain.append(teacher[chain[-1]])
        facts = [teach_fact[chain[r]] for r in range(len(chain) - 1)] + [home_fact[chain[-1]]]
        student = next((s for s, t in teacher.items() if t == cast[0]), None)
        groups = {"own_home": [home_fact[cast[0]]], "all_homes": sorted(home_fact.values()),
                  "all_teacher_facts": sorted(teach_fact.values())}
        return {"facts": facts, "groups": groups, "bridges": chain[1:], "asked": cast[0], "names": list(cast),
                "answer": home[chain[-1]], "answers": places, "own": home.get(cast[0]),
                "reverse": home.get(student) if student else None,
                "one_hop": home.get(chain[1]) if len(chain) > 2 else None,
                "steps": [(qa_suffix(f"{TEACHER_Q}{cast[0]}?"), chain[1], list(cast)),
                          (qa_suffix(f"Where does {chain[1]} live?"), home[chain[1]], places)]}
    if item.task == "deduce":
        texts = [f.text for f in item.facts]
        first = {}  # person → first category
        for t in texts:
            m = re.fullmatch(r"(\w+) is a (\w+)\.", t)
            if m and m.group(1) in item.cast:
                first[m.group(1)] = m.group(2)
        cat, facts, cats = first[item.cast[0]], [], [first[c] for c in item.cast]
        facts.append(texts.index(f"{item.cast[0]} is a {cat}."))
        chain, cur = [cat], cat
        while True:
            nxt = next((i for i, t in enumerate(texts) if t.startswith(f"Every {cur} is ")), None)
            if nxt is None:
                break
            facts.append(nxt)
            m = re.fullmatch(rf"Every {re.escape(cur)} is a (\w+)\.", texts[nxt])
            if not m:
                break
            cur = m.group(1)
            chain.append(cur)
        props = list(item.candidates)
        groups = {"all_isa": [i for i, t in enumerate(texts) if re.fullmatch(r"[A-Z]\w* is a \w+\.", t)],
                  "all_props": [i for i, t in enumerate(texts) if t.startswith("Every ") and " is a " not in t]}
        steps = [(qa_suffix(f"What is {item.cast[0]}?"), f"a {cat}", [f"a {c}" for c in cats])]
        if len(chain) == 1:
            steps.append((qa_suffix(f"What is a {cat} like?"), item.answer, props))
        return {"facts": facts, "groups": groups, "bridges": chain, "asked": item.cast[0], "names": list(item.cast),
                "bridge_pool": cats, "answer": item.answer, "answers": props, "steps": steps}
    if item.task == "count":
        return {"facts": [i for i, f in enumerate(item.facts) if f.evidence], "count": int(item.meta["count"])}
    return None


def token_spans(tok, prompt: str, texts: list[str]) -> list[list[int] | None]:
    """Row positions (BOS = 0) of the tokens of each sentence in the prompt (None when not found)."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    out, after = [], defaultdict(lambda: -1)  # a repeated sentence (count's visits) takes its next occurrence
    for t in texts:
        c0 = prompt.find(t, after[t] + 1)
        if c0 < 0:
            out.append(None)
            continue
        after[t] = c0
        c1 = c0 + len(t)
        out.append([j + 1 for j, (a, b) in enumerate(offs) if a < c1 and b > c0])
    return out


# ------------------------------------------------------------------------------------------ forwards
class Recorder:
    """Hooks every layer and keeps its output at the answer position, in call order (a looped model calls its core
    layers once per round)."""

    def __init__(self, model):
        self.calls, self.q, self.slot_pos = [], None, None
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(model.layers)]

    def _hook(self, i):
        def fn(_m, _inp, out):
            if self.q is not None:
                x = out[0] if isinstance(out, tuple) else out
                self.calls.append((i, x[0, self.q].float().cpu()))
        return fn


@torch.no_grad()
def forward(model, rec: Recorder, ids: list[int], q: int, device, capture: bool):
    rec.calls, rec.q = [], q
    memory = getattr(model, "memory_writer", None) is not None
    model._note_capture = {"q_pos": torch.tensor([[q]], device=device)} if (capture and memory) else None
    out = model(input_ids=torch.tensor([ids], device=device))
    logits = (out.logits if hasattr(out, "logits") else out[0])[0].float()
    mass = [m[0, 0].float().cpu() for m in model._note_capture.get("mass", [])] if model._note_capture else []
    if model._note_capture is not None and getattr(model, "_last_notes", None) is not None:
        rec.slot_pos = model._last_notes[2][0].cpu()  # each note's expected token position
    model._note_capture, rec.q = None, None
    return logits, mass, list(rec.calls)


def first_shares(logits, q: int, tok, prompt: str, p_ids, cands: list[str]) -> dict:
    """Each candidate's share of the probability over the candidates' first tokens (None when shared)."""
    firsts = [(_answer_ids(tok, prompt, f" {c}.", p_ids) or [-1])[0] for c in cands]
    uniq = sorted(set(firsts))
    share = torch.softmax(logits[q, torch.tensor(uniq)], -1)
    return {c: (float(share[uniq.index(t)]) if firsts.count(t) == 1 else None) for c, t in zip(cands, firsts)}


@torch.no_grad()
def whole_scores(model, rec, tok, prompt: str, p_ids, cands: list[str], gold: str, device, capture: bool):
    """Sum of log-probabilities of each whole candidate string; the gold one's share; the note mass at the answer."""
    lp, mass, calls = [], [], None
    q = len(p_ids)
    for c in cands:
        a = _answer_ids(tok, prompt, f" {c}.", p_ids)
        logits, m, cl = forward(model, rec, [tok.bos_token_id] + p_ids + a, q, device, capture and not mass)
        mass = mass or m
        calls = calls if calls is not None else cl  # the answer position precedes the candidate: same state
        lsm = torch.log_softmax(logits[q: q + len(a)], -1)
        lp.append(float(lsm[torch.arange(len(a)), torch.tensor(a)].sum()))
    t = torch.tensor(lp)
    share = torch.softmax(t, -1)
    g = cands.index(gold)
    return {"pick": int(t.argmax()) == g, "share": float(share[g]), "chance": 1.0 / len(cands),
            "pred": cands[int(t.argmax())], "_calls": calls}, mass


# ------------------------------------------------------------------------------------------ note mass
def output_spans(tok, prompt: str, fact_texts: list[str], words: list[str]) -> list[list[int] | None]:
    """Row positions of each hop's output word inside its fact (None when not found)."""
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    out = []
    for t, w in zip(fact_texts, words):
        c0 = prompt.find(t)
        m = re.search(rf"(?<!\w){re.escape(w)}(?!\w)", prompt[c0:c0 + len(t)]) if c0 >= 0 else None
        if not m:
            out.append(None)
            continue
        a, b = c0 + m.start(), c0 + m.end()
        out.append([j + 1 for j, (x, y) in enumerate(offs) if x < b and y > a])
    return out


def mass_on_notes(mass: list, slot_pos: torch.Tensor, spans: list, heads: int) -> list[dict]:
    """Per round: share (mean over heads) on the notes whose latent read each output word (expected position within
    2 tokens of it), and what a uniform read over the visible notes would give."""
    out = []
    for m in mass:
        vis = m > 0
        n_vis = int(vis.sum())
        tot = float(m.sum()) / heads
        row = {"out": [], "uniform": []}
        for sp in spans:
            if not sp:
                row["out"].append(None)
                row["uniform"].append(None)
                continue
            sel = vis & (slot_pos >= min(sp) - 2) & (slot_pos <= max(sp) + 2)
            row["out"].append(float(m[sel].sum()) / heads)
            row["uniform"].append(tot * int(sel.sum()) / max(1, n_vis))
        out.append(row)
    return out


def pages_of(positions: list[int] | None, starts, W: int) -> set:
    if not positions:
        return set()
    return {w for w, s in enumerate(starts) for p in positions if s <= p < s + W}


def mass_on(mass: list, page_sets: list[set], per_page: int, heads: int) -> list[dict]:
    """Per round: share (mean over heads) on each fact's pages, the uniform-read share, all notes, top note's fact."""
    out = []
    for m in mass:
        vis = m > 0
        n_vis = int(vis.sum())
        page = torch.arange(m.shape[0]) // per_page
        tot = float(m.sum()) / heads
        top = int(m.argmax()) // per_page if n_vis else -1
        row = {"notes": tot, "facts": [], "chance": [], "top": None}
        for j, ps in enumerate(page_sets):
            sel = torch.tensor([int(p) in ps for p in page]) & vis
            row["facts"].append(float(m[sel].sum()) / heads)
            row["chance"].append(tot * int(sel.sum()) / max(1, n_vis))
            if top in ps and row["top"] is None:
                row["top"] = j
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------ probes
def name_vec(model, tok, name: str) -> torch.Tensor:
    ids = tok.encode(" " + name, add_special_tokens=False)
    return model.embed.tok.weight[torch.tensor(ids, device=model.embed.tok.weight.device)].float().mean(0).cpu()


def ridge_probe(X: torch.Tensor, Y: torch.Tensor, pools: list[torch.Tensor], gold: list[int], seed: int = 0) -> dict:
    """Ridge state → target vector; accuracy = the nearest candidate (cosine) of each held-out item is the gold one.
    2/3 train (λ chosen on its last fifth), 1/3 test."""
    n = X.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, te = idx[: 2 * n // 3], idx[2 * n // 3:]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    Xs = (X - mu) / sd
    Xs = torch.cat([Xs, torch.ones(n, 1)], 1)

    def fit(rows, lam):
        A = Xs[rows]
        return torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ Y[rows])

    def acc(Wm, rows):
        P = Xs[rows] @ Wm
        ok = []
        for k, i in enumerate(rows.tolist()):
            c = torch.nn.functional.cosine_similarity(P[k][None], pools[i], dim=-1)
            ok.append(int(c.argmax()) == gold[i])
        return sum(ok) / max(1, len(ok))

    cut = int(len(tr) * 0.8)
    best = max((acc(fit(tr[:cut], lam), tr[cut:]), lam) for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0))[1]
    W_ = fit(tr, best)
    return {"acc": acc(W_, te), "n_test": len(te), "chance": sum(1.0 / pools[i].shape[0] for i in te.tolist()) / len(te),
            "lambda": best}


# ------------------------------------------------------------------------------------------ main
def call_labels(model, calls: list) -> list[str]:
    cfg = model.config
    R = int(getattr(model, "loop_rounds", 1) or 1)
    seen, out = defaultdict(int), []
    core = (range(cfg.global_layer_index, cfg.global_layer_index + 1 + cfg.message_loop_span)
            if R > 1 else range(0))
    for i, _ in calls:
        k = seen[i]
        seen[i] += 1
        out.append(f"r{k + 1}.L{i}" if i in core else f"L{i}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--tasks", nargs="*", default=["compose", "deduce", "count"])
    p.add_argument("--lengths", type=int, nargs="*", default=[4096])
    p.add_argument("--max_items", type=int, default=200, help="per (split, task, length) cell")
    p.add_argument("--version", default="v2")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--attn_backend", default=None)
    a = p.parse_args()

    from transformers import AutoTokenizer

    from nn.latent_memory import lm_geometry
    from nn.perceiver_families import load_perceiver_lm

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend = a.attn_backend or ("flex" if device.startswith("cuda") else "sdpa")
    model = load_perceiver_lm(a.checkpoint, device, backend).eval()
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.checkpoint)
    cfg = model.config
    memory = getattr(model, "memory_writer", None) is not None
    heads = int(cfg.num_attention_heads)
    per_page = int(getattr(cfg, "lm_latents", 0) or 0) * int(getattr(cfg, "lm_reader_tokens", 1) or 1)
    rec = Recorder(model)

    recs = []
    for path in a.items:
        with open(path) as fh:
            recs += [json.loads(line) for line in fh]
    seen, items = defaultdict(int), []
    for r in recs:
        key = (r["split"], r["task"], r["length"])
        if r["task"] in a.tasks and r["length"] in a.lengths and seen[key] < a.max_items:
            seen[key] += 1
            items.append(r)
    print(f"{len(items)} items: " + ", ".join(f"{k}={v}" for k, v in sorted(seen.items())), flush=True)

    rows, states, direct, skipped, t0 = [], [], [], defaultdict(int), time.time()
    labels = None
    for n_done, r in enumerate(items):
        item = build_item(r["task"], random.Random(r["seed"]), r["split"], a.version)
        if f" {item.answer}." != r["answer"]:
            skipped["rebuild"] += 1
            continue
        ch = chain_of(item, r)
        prompt = r["prompt"]
        p_ids = tok.encode(prompt, add_special_tokens=False)
        q = len(p_ids)
        spans = token_spans(tok, prompt, [item.facts[i].text for i in ch["facts"]])
        if any(s is None for s in spans):
            skipped["fact_not_found"] += 1
            continue
        row = {"id": r["id"], "task": r["task"], "split": r["split"], "length": r["length"],
               "hops": item.meta.get("hops", item.meta.get("depth"))}
        body = prompt.partition("\n\nQuestion:")[0]

        if r["task"] == "count":
            s, _ = whole_scores(model, rec, tok, prompt, p_ids, [c[1:-1] for c in r["candidates"]], item.answer,
                                device, False)
            s.pop("_calls", None)
            stride = int(getattr(cfg, "lm_stride", 192) or 192)
            strides = [sp[0] // stride for sp in spans]
            row.update(count=ch["count"], pred=s["pred"], right=s["pick"], strides=len(set(strides)),
                       shared=len(set(strides)) < len(strides))
            rows.append(row)
            continue

        a_ids = _answer_ids(tok, prompt, r["answer"], p_ids)
        ids = [tok.bos_token_id] + p_ids + a_ids
        logits, mass, calls = forward(model, rec, ids, q, device, True)
        labels = labels or call_labels(model, calls)
        cands = ch["answers"]
        sh = first_shares(logits, q, tok, prompt, p_ids, cands)
        row["gold_share"] = sh.get(ch["answer"])
        row["chance"] = 1.0 / len(cands)
        if r["task"] == "compose":
            for role in ("own", "reverse", "one_hop"):
                row[f"{role}_share"] = sh.get(ch[role]) if ch.get(role) else None
        if memory:
            geo = lm_geometry(cfg, len(ids))
            psets = [pages_of(s_, geo.starts, geo.window) for s_ in spans]
            row["d1"] = mass_on(mass, psets, per_page, heads)
            row["fact_pages"] = [sorted(ps) for ps in psets]
            if getattr(rec, "slot_pos", None) is not None:
                words = list(ch["bridges"]) + [ch["answer"]]
                osp = output_spans(tok, prompt, [item.facts[i].text for i in ch["facts"]], words)
                row["d1_out"] = mass_on_notes(mass, rec.slot_pos, osp, heads)
            gnames = list(ch.get("groups", {}))
            if gnames:
                gsp = [token_spans(tok, prompt, [item.facts[i].text for i in ch["groups"][g]]) for g in gnames]
                gsets = [set().union(*(pages_of(x, geo.starts, geo.window) for x in sp_ if x)) for sp_ in gsp]
                row["d1_groups"] = {g: [{"mass": rr["facts"][k], "uniform": rr["chance"][k]}
                                        for rr in mass_on(mass, gsets, per_page, heads)]
                                    for k, g in enumerate(gnames)}
        # D3: the single steps on the same document (one-step items only)
        if (r["task"] == "compose" and row["hops"] == 1) or (r["task"] == "deduce" and row["hops"] == 1):
            row["d3"] = []
            for j, (suffix, gold, pool) in enumerate(ch["steps"]):
                sp = body + suffix
                sp_ids = tok.encode(sp, add_special_tokens=False)
                s, m = whole_scores(model, rec, tok, sp, sp_ids, pool, gold, device, memory)
                cl = s.pop("_calls")
                if j == 0:  # positive control for D2: the bridge where the model is asked for it directly
                    pool_b = ch.get("bridge_pool", ch["names"])
                    direct.append({"task": r["task"], "X": torch.stack([x for _, x in cl]),
                                   "targets": {"bridge1": (torch.stack([name_vec(model, tok, c) for c in pool_b]),
                                                           pool_b.index(ch["bridges"][0]))}})
                if memory:  # the mass was captured on the first candidate's row
                    geo2 = lm_geometry(cfg, 1 + len(sp_ids) + len(_answer_ids(tok, sp, f" {pool[0]}.", sp_ids)))
                    ps2 = [pages_of(s_, geo2.starts, geo2.window) for s_ in spans]
                    s["d1"] = mass_on(m, ps2, per_page, heads)
                row["d3"].append(s)
        rows.append(row)
        # D2 inputs: the state per layer call, and the probe targets with their pools
        pool_b = ch.get("bridge_pool", ch["names"])
        tg = {"bridge1": (pool_b, ch["bridges"][0]), "asked": (ch["names"], ch["asked"]),
              "answer": (cands, ch["answer"])}
        if r["task"] == "compose" and len(ch["bridges"]) > 1:
            tg["bridge2"] = (ch["names"], ch["bridges"][1])
        states.append({"task": r["task"], "hops": row["hops"], "X": torch.stack([x for _, x in calls]),
                       "targets": {k: (torch.stack([name_vec(model, tok, c) for c in pool]), pool.index(g))
                                   for k, (pool, g) in tg.items()}})
        if (n_done + 1) % 25 == 0:
            print(f"progress {n_done + 1}/{len(items)} · {time.time() - t0:.0f}s · {time.strftime('%H:%M:%S %Z')}",
                  flush=True)

    # ---------------------------------------------------------------- summaries
    summ = {"checkpoint": a.checkpoint, "memory": memory, "loop_rounds": int(getattr(model, "loop_rounds", 1) or 1),
            "skipped": dict(skipped), "seconds": round(time.time() - t0, 1)}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["split"], row["length"], row["hops"])].append(row)

    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    cells = []
    for (task, split, length, hops), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        c = {"task": task, "split": split, "length": length, "hops": hops, "n": len(rs)}
        if task == "count":
            by = defaultdict(list)
            for x in rs:
                by[x["count"]].append(x["right"])
            c["exact"] = mean([x["right"] for x in rs])
            c["by_count"] = {k: mean(v) for k, v in sorted(by.items())}
            c["by_stride_share"] = {"all_apart": mean([x["right"] for x in rs if not x["shared"]]),
                                    "some_shared": mean([x["right"] for x in rs if x["shared"]])}
            c["abs_err"] = mean([abs(int(x["count"]) - _num(x["pred"])) for x in rs if _num(x["pred"]) is not None])
            cells.append(c)
            continue
        c["gold_share"], c["chance"] = mean([x["gold_share"] for x in rs]), mean([x["chance"] for x in rs])
        for role in ("own", "reverse", "one_hop"):
            v = mean([x.get(f"{role}_share") for x in rs])
            if v is not None:
                c[f"{role}_share"] = v
        if memory and rs and rs[0].get("d1"):
            nr, nf = len(rs[0]["d1"]), len(rs[0]["d1"][0]["facts"])
            c["d1"] = [{"notes": mean([x["d1"][k]["notes"] for x in rs]),
                        "facts": [mean([x["d1"][k]["facts"][j] for x in rs]) for j in range(nf)],
                        "chance": [mean([x["d1"][k]["chance"][j] for x in rs]) for j in range(nf)],
                        "top": [mean([float(x["d1"][k]["top"] == j) for x in rs]) for j in range(nf)]}
                       for k in range(nr)]
        if memory and rs and rs[0].get("d1_out"):
            nr, nf = len(rs[0]["d1_out"]), len(rs[0]["d1_out"][0]["out"])
            c["d1_out"] = [{"out": [mean([x["d1_out"][k]["out"][j] for x in rs]) for j in range(nf)],
                            "uniform": [mean([x["d1_out"][k]["uniform"][j] for x in rs]) for j in range(nf)]}
                           for k in range(nr)]
        if memory and rs and rs[0].get("d1_groups"):
            c["d1_groups"] = {g: [{"mass": mean([x["d1_groups"][g][k]["mass"] for x in rs]),
                                   "uniform": mean([x["d1_groups"][g][k]["uniform"] for x in rs])}
                                  for k in range(len(rs[0]["d1_groups"][g]))] for g in rs[0]["d1_groups"]}
        if rs and rs[0].get("d3"):
            c["d3"] = []
            for j in range(len(rs[0]["d3"])):
                d = {"pick": mean([float(x["d3"][j]["pick"]) for x in rs]),
                     "share": mean([x["d3"][j]["share"] for x in rs]), "chance": mean([x["d3"][j]["chance"] for x in rs])}
                if memory and rs[0]["d3"][j].get("d1"):
                    nr, nf = len(rs[0]["d3"][j]["d1"]), len(rs[0]["d3"][j]["d1"][0]["facts"])
                    d["d1"] = [{"facts": [mean([x["d3"][j]["d1"][k]["facts"][f] for x in rs]) for f in range(nf)],
                                "chance": [mean([x["d3"][j]["d1"][k]["chance"][f] for x in rs]) for f in range(nf)]}
                               for k in range(nr)]
                c["d3"].append(d)
        cells.append(c)

    probes = {}
    for task, st in [(t, [s for s in states if s["task"] == t]) for t in ("compose", "deduce")] + \
            [(f"{t}_asked_directly", [s for s in direct if s["task"] == t]) for t in ("compose", "deduce")]:
        if len(st) < 30:
            continue
        X = torch.stack([s["X"] for s in st])  # [N, calls, d]
        probes[task] = {}
        for tname in ("asked", "bridge1", "bridge2", "answer"):
            sub = [k for k, s in enumerate(st) if tname in s["targets"]]
            if len(sub) < 30:
                continue
            pools = [st[k]["targets"][tname][0] for k in sub]
            gold = [st[k]["targets"][tname][1] for k in sub]
            Y = torch.stack([pools[i][gold[i]] for i in range(len(sub))])
            probes[task][tname] = {labels[ci]: ridge_probe(X[sub, ci], Y, pools, gold) for ci in range(X.shape[1])}

    res = {"summary": summ, "cells": cells, "probes": probes, "call_labels": labels, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    _print(res)


_NUM = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                    "ten"])}


def _num(w):
    return _NUM.get(str(w).strip(" ."))


def _print(res):
    f = lambda v: "  -  " if v is None else f"{100 * v:5.1f}"  # noqa: E731
    print(f"\n{res['summary']}")
    for c in res["cells"]:
        head = f"{c['task']:<8} {c['split']:<6} {c['length']:>6} hops={c['hops']} n={c['n']}"
        if c["task"] == "count":
            print(head, "exact", f(c["exact"]), "abs_err", c["abs_err"], "by_count",
                  {k: f(v) for k, v in c["by_count"].items()}, "strides", {k: f(v) for k, v in c["by_stride_share"].items()})
            continue
        print(head, "gold", f(c["gold_share"]), "chance", f(c["chance"]),
              *(f"{r} {f(c[r + '_share'])}" for r in ("own", "reverse", "one_hop") if r + "_share" in c))
        for k, d in enumerate(c.get("d1") or []):
            print(f"   D1 round {k + 1}: notes {f(d['notes'])} | " + " | ".join(
                f"fact{j + 1} {f(d['facts'][j])} (uniform {f(d['chance'][j])}, top {f(d['top'][j])})"
                for j in range(len(d["facts"]))))
        for k, d in enumerate(c.get("d1_out") or []):
            print(f"   D1 output-word notes, round {k + 1}: " + " | ".join(
                f"hop{j + 1} {f(d['out'][j])} (uniform {f(d['uniform'][j])})" for j in range(len(d["out"]))))
        for g, ds in (c.get("d1_groups") or {}).items():
            print(f"   D1 {g}: " + " | ".join(f"r{k + 1} {f(d['mass'])} (uniform {f(d['uniform'])})" for k, d in enumerate(ds)))
        for j, d in enumerate(c.get("d3") or []):
            print(f"   D3 step {j + 1}: pick {f(d['pick'])} share {f(d['share'])} chance {f(d['chance'])}")
            for k, e in enumerate(d.get("d1") or []):
                print(f"      round {k + 1}: " + " | ".join(f"fact{i + 1} {f(e['facts'][i])} (uniform {f(e['chance'][i])})"
                                                   for i in range(len(e["facts"]))))
    for task, tp in res["probes"].items():
        for tname, by in tp.items():
            print(f"D2 {task} {tname}: " + " ".join(f"{k}={f(v['acc'])}" for k, v in by.items())
                  + f" (chance {f(next(iter(by.values()))['chance'])})")


if __name__ == "__main__":
    main()
