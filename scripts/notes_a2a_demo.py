#!/usr/bin/env python
"""Agent-to-agent demo on an E33-family model (E33f): agent A writes a notebook of a long document, agent B — which
never sees the document — answers questions by fetching a few of A's notes, one request per read of its loop.

The two agents share the model's weights but run separate parts of it and hold separate state:
  agent A  the writer (token table + latent-memory writer): document → notebook of notes (k̄ ‖ v̄, 128 numbers each,
           32 per page of 256 tokens, stride 192). Nothing of the reading stack runs on A's side.
  agent B  the reader (the whole stack) on a short row holding only its question and the answer so far; its global
           read sees only the notes A has sent it (an external notebook, readable from the row's first token).

Protocol, per answer token: for each read r of B's loop (4 rounds), B runs its row with the notes it holds and sends
its round-r note query (one 64-number vector per head) plus the log-total of its own raw attention per head; A scores
every note in its notebook against it (`score_notes`) and sends back the K best, with their addresses. After the
last round B reads with everything received and emits the token. Everything that crosses is counted, as 16-bit
numbers.

Controls per document: B with no notes at all, and B running the same protocol against the notebook of a different
document. Reference: B reading A's whole notebook (no bandwidth limit).

  uv run python scripts/notes_a2a_demo.py --checkpoint <E33f final> --data <text_checks data dir> \\
      --tasks lookup keyed quote --lengths 4096 16384 65536 262144 1048576 --docs 3 --out Cache/e35/notes_demo.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from nn.perceiver_ar_lm import score_notes  # noqa: E402
from scripts.a2a_demo import build_document  # noqa: E402

BYTES = 2  # 16-bit numbers on the wire


class AgentA:
    """Holds the document's notebook; answers requests with notes."""

    def __init__(self, model, tok, device, chunk_pages: int = 512):
        self.model, self.tok, self.device, self.chunk_pages = model, tok, device, chunk_pages
        cfg = model.config
        self.W, self.st = int(cfg.lm_window), int(cfg.lm_stride)
        self.per = int(cfg.lm_latents) * int(cfg.lm_reader_tokens)
        self.k = self.v = self.sp = self.ids = None

    def read(self, text: str) -> dict:
        t0 = time.time()
        ids = torch.tensor([self.tok.encode(text, add_special_tokens=False)], device=self.device)
        t1 = time.time()
        self.ids = ids
        self.k, self.v, self.sp = self.model.write_notes(ids, chunk_pages=self.chunk_pages)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return {"tokens": ids.shape[1], "notes": self.k.shape[1], "tokenize_s": round(t1 - t0, 3),
                "write_s": round(time.time() - t1, 3)}

    def serve(self, queries: torch.Tensor, raw_lse: torch.Tensor, K: int) -> torch.Tensor:
        """B's request → the addresses of A's K best notes for it."""
        score = score_notes(queries, raw_lse, self.k)[0].sum(0)
        return torch.topk(score, min(K, score.shape[0])).indices

    def notes(self, idx: torch.Tensor) -> tuple:
        return self.k[:, idx], self.v[:, idx], self.sp[:, idx]

    def page_text(self, note: int) -> str:
        """The page a note was written from (for the printed examples)."""
        S = self.ids.shape[1]
        w = note // self.per
        starts = list(range(0, max(S - self.W, 0) + 1, self.st))
        if starts[-1] != max(S - self.W, 0):
            starts.append(max(S - self.W, 0))
        s0 = starts[min(w, len(starts) - 1)]
        return self.tok.decode(self.ids[0, s0: s0 + self.W].tolist())


class AgentB:
    """Holds only its question, its answer so far and the notes it received."""

    def __init__(self, model, tok, device, rounds: int):
        self.model, self.tok, self.device, self.rounds = model, tok, device, rounds
        self.dot = tok.encode(".", add_special_tokens=False)[-1]
        cfg = model.config
        self.request_numbers = int(cfg.num_attention_heads) * (int(cfg.head_dim) + 1)  # query + raw log-total, per head
        self.held_idx: list[int] = []
        self.held = None
        self.blank = None  # one hidden placeholder note: the row then has a notebook even before anything arrived

    def _forward(self, ids: list[int], offset: int, capture: bool):
        m = self.model
        x = torch.tensor([ids], device=self.device)
        if self.held is None:
            m._notes_override = self.blank
            m._note_keep = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        else:
            m._notes_override, m._note_keep = self.held, None
        m._notes_external = True
        m._note_capture = {"q_pos": torch.tensor([[len(ids) - 1]], device=self.device)} if capture else None
        out = m(x, position_offset=offset)
        cap = m._note_capture
        m._notes_override, m._note_keep, m._note_capture, m._notes_external = None, None, None, False
        return out.logits[0, -1], cap

    def receive(self, idx: torch.Tensor, A: AgentA) -> int:
        new = [int(i) for i in idx.tolist() if int(i) not in self.held_idx]
        if new:
            self.held_idx += new
            t = torch.tensor(self.held_idx, device=self.device)
            self.held = A.notes(t)
        return len(new)

    @torch.no_grad()
    def ask(self, question: str, A: AgentA | None, K: int, offset: int, max_new: int = 16, log: list | None = None,
            request_tokens: int = 0):
        """Greedy answer. A = None: no notes at all. K = 0 with A: B reads A's whole notebook (no request).
        `request_tokens` > 0: B requests notes only for its first that many answer tokens, then answers from what it
        holds."""
        if self.blank is None:  # the shape of one note, from a one-page write
            k0, v0, s0 = self.model.write_notes(torch.full((1, int(self.model.config.lm_window)), self.dot,
                                                           device=self.device))
            self.blank = (torch.zeros_like(k0[:, :1]), torch.zeros_like(v0[:, :1]), torch.zeros_like(s0[:, :1]))
        self.held_idx, self.held = [], None
        if A is not None and K == 0:
            self.held = (A.k, A.v, A.sp)
        ids = [self.tok.bos_token_id] + self.tok.encode(f"Question: {question}\nAnswer:", add_special_tokens=False)
        out, sent_up, sent_down = [], 0, 0
        for _ in range(max_new):
            if A is not None and K > 0 and (not request_tokens or len(out) < request_tokens):
                for r in range(self.rounds):
                    _, cap = self._forward(ids + out, offset, capture=True)
                    qs, raw = cap["queries"][r]
                    sent_up += self.request_numbers * BYTES
                    idx = A.serve(qs, raw, K)
                    before = set(self.held_idx)
                    n_new = self.receive(idx, A)
                    sent_down += n_new * (A.k.shape[2] * A.k.shape[3] + A.v.shape[2] * A.v.shape[3]) * BYTES + 4 * n_new
                    if log is not None:  # only the notes that were new to B travel
                        log.append({"token": len(out), "round": r + 1,
                                    "new_notes": [int(i) for i in idx.tolist() if int(i) not in before]})
            logits, _ = self._forward(ids + out, offset, capture=False)
            t = int(logits.argmax())
            out.append(t)
            if t == self.dot or t == self.tok.eos_token_id:
                break
        return self.tok.decode(out).strip(), {"notes_held": len(self.held_idx), "bytes_up": sent_up,
                                              "bytes_down": sent_down}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="text-checks data dir (tokenizer)")
    p.add_argument("--tasks", nargs="+", default=["lookup", "keyed", "quote"])
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--docs", type=int, default=3)
    p.add_argument("--max_questions", type=int, default=12)
    p.add_argument("--k", type=int, default=4, help="notes A sends per request")
    p.add_argument("--request_tokens", type=int, default=0,
                   help="B requests notes only for its first N answer tokens (0 = every token)")
    p.add_argument("--whole_notebook_max", type=int, default=300_000,
                   help="skip the whole-notebook reference above this many notes")
    p.add_argument("--device", default=None)
    p.add_argument("--stories", choices=("full", "smoke"), default="full")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm
    from scripts.build_text_checks_data import load_stories

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_perceiver_lm(a.checkpoint, device, "sdpa").eval()
    model.config.attn_pad_multiple = 16  # B's rows are a question long
    for prm in model.parameters():
        prm.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(os.path.join(a.data, "tokenizer"))
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    _, held = load_stories(a.stories, 1, 0)
    rounds = int(getattr(model, "loop_rounds", 1) or 1)
    A, A2 = AgentA(model, tok, device), AgentA(model, tok, device)
    B = AgentB(model, tok, device, rounds)
    res = {"checkpoint": a.checkpoint, "k": a.k, "rounds": rounds, "request_tokens": a.request_tokens, "request_bytes": B.request_numbers * BYTES,
           "cells": [], "examples": []}
    t_all = time.time()
    for task in a.tasks:
        for length in a.lengths:
            base = zlib.crc32(f"notes-demo/{task}/{length}".encode()) * 10_000
            docs = [build_document(task, length, base + i, held, ntok) for i in range(a.docs)]
            score = {"fetched": [], "whole": [], "none": [], "other": []}
            stats = {"notes_held": [], "bytes_up": [], "bytes_down": [], "write_s": [], "notes": [], "tokens": [],
                     "text_bytes": [], "hit_page": []}
            for i, (body, qa) in enumerate(docs):
                r = A.read(body)
                A2.read(docs[(i + 1) % len(docs)][0])
                for k_ in ("write_s", "notes", "tokens"):
                    stats[k_].append(r[k_])
                stats["text_bytes"].append(len(body.encode()))
                off = r["tokens"]
                for j, (q, gold) in enumerate(qa[: a.max_questions]):
                    log = []
                    ans, st = B.ask(q, A, a.k, off, log=log, request_tokens=a.request_tokens)
                    ok = ans.rstrip(".").strip() == gold
                    score["fetched"].append(ok)
                    for k_ in ("notes_held", "bytes_up", "bytes_down"):
                        stats[k_].append(st[k_])
                    hit = any(gold in A.page_text(n) for n in B.held_idx)
                    stats["hit_page"].append(hit)
                    if j < 4:  # controls and the reference on a subset
                        if r["notes"] <= a.whole_notebook_max:
                            score["whole"].append(B.ask(q, A, 0, off)[0].rstrip(".").strip() == gold)
                        score["none"].append(B.ask(q, None, a.k, off)[0].rstrip(".").strip() == gold)
                        score["other"].append(B.ask(q, A2, a.k, off, request_tokens=a.request_tokens)[0]
                                              .rstrip(".").strip() == gold)
                    if len(res["examples"]) < 60 and j < 2:
                        first = log[0]["new_notes"][:2] if log and log[0]["new_notes"] else []
                        res["examples"].append({
                            "task": task, "length": length, "doc": i, "question": q, "gold": gold, "answer": ans,
                            "correct": ok, "notes_held": st["notes_held"], "notebook": r["notes"],
                            "bytes_up": st["bytes_up"], "bytes_down": st["bytes_down"], "fact_page_fetched": hit,
                            "first_round_pages": [A.page_text(n)[:240] for n in first]})
            mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
            cell = {"task": task, "length": length, "docs": a.docs,
                    **{f"acc_{k}": mean(v) for k, v in score.items()}, **{f"n_{k}": len(v) for k, v in score.items()},
                    **{k: mean(v) for k, v in stats.items()}}
            res["cells"].append(cell)
            print(f"{task:6s} {length:8d} tokens · notebook {cell['notes']:8.0f} notes, written in {cell['write_s']:.2f}s"
                  f" · B fetched {cell['notes_held']:.1f} notes ({cell['bytes_down'] / 1024:.1f} KB down, "
                  f"{cell['bytes_up'] / 1024:.1f} KB of requests up; the text is {cell['text_bytes'] / 1024:.0f} KB) · "
                  f"correct {cell['acc_fetched']:.0%} (whole notebook "
                  f"{'-' if cell['acc_whole'] is None else format(cell['acc_whole'], '.0%')}, no notes "
                  f"{cell['acc_none']:.0%}, another document's {cell['acc_other']:.0%}) · fact's page among the fetched "
                  f"{cell['hit_page']:.0%} ({time.time() - t_all:.0f}s)", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
