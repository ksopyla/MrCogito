#!/usr/bin/env python
"""E35 agent-to-agent demo: agent A reads a long document into its memory and sends K vectors; agent B, which never
sees the document, answers questions from the message and its own text alone.

Spec: docs/experiments_specs/ahead/E35_latent_messages.md. The two agents are separate objects: A holds the token
embedding, the writer, the memory and the transmitter; B holds the decoder. The message crosses as bytes (16-bit
floats), so what travels is explicit and its size is printed.

Documents come from the text-checks generator (held-out names and filler stories): every person's fact is stated
somewhere in the document, and B is asked about each of them. Controls per document: B with an empty message, and B
with the message of a different document (the mismatched-message check).

  uv run python scripts/a2a_demo.py --checkpoint Cache/e35/k16/train/<run>/final \\
      --data <text_checks data dir> --tasks lookup keyed quote latest --lengths 4096 16384 65536 262144 1048576 \\
      --docs 3 --out Cache/e35/demo.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
import zlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402


class AgentA:
    """The sender: reads text into its memory, sends a message on request.

    `refine_steps` > 0 (Cramming from the transmitter's output; assumes A holds a copy of B's decoder, as agents that
    share a model do): the message is then optimized for that many steps so that B, reading it, predicts the
    document's own text best — no knowledge of B's future questions; the words B cannot predict without the document
    (names, places) are where the loss is."""

    def __init__(self, model, tok, device, chunk_pages: int = 512, refine_steps: int = 0, refine_lr: float = 0.01,
                 refine_window: int = 1024):
        self.model, self.tok, self.device, self.chunk_pages = model, tok, device, chunk_pages
        self.refine_steps, self.refine_lr, self.refine_window = refine_steps, refine_lr, refine_window
        self.memory, self.doc_ids = None, None

    @torch.no_grad()
    def read(self, text: str) -> dict:
        t0 = time.time()
        ids = torch.tensor([self.tok.encode(text, add_special_tokens=False)], device=self.device)
        t1 = time.time()
        self.doc_ids = ids
        self.memory, _ = self.model.write(ids, chunk_pages=self.chunk_pages)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return {"tokens": ids.shape[1], "entries": self.memory.n_entries, "tokenize_s": round(t1 - t0, 3),
                "write_s": round(time.time() - t1, 3)}

    @torch.no_grad()
    def send(self, request: bytes | None = None) -> bytes:
        req = None
        if request is not None:
            req = torch.tensor(np.load(io.BytesIO(request)), device=self.device,
                               dtype=next(self.model.parameters()).dtype)[None]
        msg = self.model.send(self.memory, req)[0]  # [K, d]
        if self.refine_steps:
            msg = self.refine(msg)
        buf = io.BytesIO()
        np.save(buf, msg.float().cpu().numpy().astype(np.float16))
        return buf.getvalue()


    def refine(self, msg: torch.Tensor) -> torch.Tensor:
        rec = self.model.receiver
        vec = msg.detach().float().clone()[None].requires_grad_(True)
        opt = torch.optim.Adam([vec], lr=self.refine_lr)
        ids = self.doc_ids[0]
        W = min(self.refine_window, ids.shape[0])
        g = torch.Generator().manual_seed(0)
        pad_was = rec.config.attn_pad_multiple
        rec.config.attn_pad_multiple = 1
        with torch.enable_grad():
            for _ in range(self.refine_steps):
                s0 = int(torch.randint(0, ids.shape[0] - W + 1, (1,), generator=g))
                x = torch.cat([ids.new_tensor([self.tok.bos_token_id]), ids[s0:s0 + W]])[None]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                    loss = rec(input_ids=x, labels=x, prefix_embeds=vec.to(next(rec.parameters()).dtype)).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        rec.config.attn_pad_multiple = pad_was
        rec.zero_grad(set_to_none=True)
        return vec.detach()[0].to(msg.dtype)


class AgentB:
    """The receiver: never sees the document; reads the message as its first input positions."""

    def __init__(self, receiver, tok, device):
        self.receiver, self.tok, self.device = receiver, tok, device
        self.dot = tok.encode(".", add_special_tokens=False)[-1]

    @torch.no_grad()
    def request(self, question: str, model) -> bytes:
        """Request mode: B pools its question's words in its own embedding space and sends that (576 numbers)."""
        ids = torch.tensor([self.tok.encode(question, add_special_tokens=False)], device=self.device)
        buf = io.BytesIO()
        np.save(buf, model.request(ids)[0].float().cpu().numpy().astype(np.float16))
        return buf.getvalue()

    def receive(self, payload: bytes | None) -> torch.Tensor | None:
        if payload is None:
            return None
        arr = np.load(io.BytesIO(payload))
        return torch.tensor(arr, device=self.device, dtype=next(self.receiver.parameters()).dtype)[None]

    @torch.no_grad()
    def ask(self, message: torch.Tensor | None, question: str, empty: torch.Tensor, max_new: int = 16) -> str:
        ids = [self.tok.bos_token_id] + self.tok.encode(f"Question: {question}\nAnswer:", add_special_tokens=False)
        msg = message if message is not None else empty
        out = []
        for _ in range(max_new):
            x = torch.tensor([ids + out], device=self.device)
            logits = self.receiver(input_ids=x, prefix_embeds=msg, return_logits=True).logits[0, -1]
            t = int(logits.argmax())
            out.append(t)
            if t == self.dot or t == self.tok.eos_token_id:
                break
        return self.tok.decode(out).strip()


def build_document(task: str, length: int, seed: int, stories, ntok):
    from data.text_world import assemble, build_item

    rng = random.Random(seed)
    item = build_item(task, rng, "id", "v2")
    fseed, pseed = rng.randrange(2**31), rng.randrange(2**31)
    prompt, _ = assemble(item, stories, length, ntok, random.Random(pseed), filler_seed=fseed, version="v2")
    body = prompt.partition("\n\nQuestion:")[0]
    return body, item.qa


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="text-checks data dir (tokenizer)")
    p.add_argument("--tasks", nargs="+", default=["lookup", "keyed", "quote", "latest"])
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--docs", type=int, default=3, help="documents per (task, length)")
    p.add_argument("--max_questions", type=int, default=12)
    p.add_argument("--device", default=None)
    p.add_argument("--stories", choices=("full", "smoke"), default="full", help="filler pool (smoke: local checks)")
    p.add_argument("--refine_steps", type=int, default=0, help="Cramming-style refinement of each message (0 = off)")
    p.add_argument("--refine_lr", type=float, default=0.01)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer

    from nn.perceiver_families import load_perceiver_lm
    from scripts.build_text_checks_data import load_stories

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_perceiver_lm(a.checkpoint, device, "flex" if device.startswith("cuda") else "sdpa")
    tok = AutoTokenizer.from_pretrained(os.path.join(a.data, "tokenizer"))
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731
    _, held = load_stories(a.stories, 1, 0)
    A = AgentA(model, tok, device, refine_steps=a.refine_steps, refine_lr=a.refine_lr)
    for prm in model.parameters():
        prm.requires_grad_(False)
    B = AgentB(model.receiver, tok, device)
    with torch.no_grad():
        empty = model.send(model.empty_memory(1, device))
    K, d = empty.shape[1], empty.shape[2]
    res = {"checkpoint": a.checkpoint, "message_vectors": K, "vector_width": d, "refine_steps": a.refine_steps,
           "message_bytes": None, "cells": [], "examples": []}
    t_all = time.time()
    for task in a.tasks:
        for length in a.lengths:
            base = zlib.crc32(f"demo/{task}/{length}".encode()) * 10_000
            docs = [build_document(task, length, base + i, held, ntok) for i in range(a.docs)]
            payloads, reads, memories = [], [], []
            for body, _ in docs:
                reads.append(A.read(body))
                memories.append(A.memory)
                t0 = time.time()
                payloads.append(A.send())
                reads[-1]["send_s"] = round(time.time() - t0, 3)
            res["message_bytes"] = len(payloads[0])
            score = {"real": [], "none": [], "swapped": []}
            req_mode = getattr(model.config, "tx_request", "none") != "none"
            for i, (body, qa) in enumerate(docs):
                msgs = {"real": B.receive(payloads[i]), "none": None,
                        "swapped": B.receive(payloads[(i + 1) % len(docs)])}
                for q, gold in qa[: a.max_questions]:
                    if req_mode:  # B asks, A answers with a message for this question (other doc's: same request)
                        rq = B.request(q, model)
                        A.memory = memories[i]
                        msgs["real"] = B.receive(A.send(rq))
                        A.memory = memories[(i + 1) % len(docs)]
                        msgs["swapped"] = B.receive(A.send(rq))
                    for kind, msg in msgs.items():
                        if kind != "real" and len(score[kind]) >= 2 * a.max_questions:
                            continue  # controls on a subset
                        ans = B.ask(msg, q, empty)
                        ok = ans.rstrip(".").strip() == gold
                        score[kind].append(ok)
                        if kind == "real" and len(res["examples"]) < 400:
                            res["examples"].append({"task": task, "length": length, "doc": i, "question": q,
                                                    "gold": gold, "answer": ans, "correct": ok})
            tok_n = sum(r["tokens"] for r in reads) / len(reads)
            cell = {"task": task, "length": length, "docs": a.docs, "doc_tokens": tok_n,
                    "memory_entries": sum(r["entries"] for r in reads) / len(reads),
                    "tokens_per_vector": tok_n / K, "write_s": sum(r["write_s"] for r in reads) / len(reads),
                    "send_s": sum(r["send_s"] for r in reads) / len(reads),
                    "text_bytes_utf8": sum(len(b.encode()) for b, _ in docs) / len(docs),
                    **{f"acc_{k}": (sum(v) / len(v) if v else None) for k, v in score.items()},
                    **{f"n_{k}": len(v) for k, v in score.items()}}
            res["cells"].append(cell)
            print(f"{task:7s} {length:8d} tok · memory {cell['memory_entries']:8.0f} entries · message {K}×{d} "
                  f"({res['message_bytes']/1024:.1f} KB) = {cell['tokens_per_vector']:7.0f} tokens/vector · "
                  f"B correct {cell['acc_real']:.0%} (empty {cell['acc_none']:.0%}, other doc's {cell['acc_swapped']:.0%})"
                  f" · write {cell['write_s']:.2f}s ({time.time() - t_all:.0f}s)", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
