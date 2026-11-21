"""Read a document once, then answer many questions from its cache (2026-11-14).

The cached decoding (nn/incremental.py) reads a row once and then runs one position per written token. A question
after a long document costs the whole document again when it is decoded as a fresh prompt (5 s at 64k for the 100M
concept-memory model on one L40S, nearly all of it the read). Here the document is read once (`read`), the state is
kept (`IncrementalDecoder.snapshot`), and each `ask` restores it, writes the question's tokens as steps and decodes:
a question costs its own tokens plus the answer's, not the document's.

    s = DocumentSession(model, tokenizer)
    s.read(open("book.txt").read())            # once: seconds grow with the document
    answer, seconds = s.ask(" Who wrote the letter? Answer:")   # each: about a second

The tokens are the ones a fresh cached decoding of document + question writes
(tests/test_incremental_decode.py::test_read_once_answer_many). Models the cache does not cover raise
NotImplementedError in `read` (the caller keeps the plain loop). For the catalogue and notes reads the question and
answer together must stay under `IncrementalDecoder.max_new_tokens()`.

CLI: uv run python evaluation/doc_session.py --checkpoint RUN/final --doc book.txt --question " ... ?" --question ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nn.incremental import IncrementalDecoder  # noqa: E402


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


class DocumentSession:
    """One document, read once; any number of questions. `room`: free slots after the document for the longest
    question plus its answer; `pad`: the row length is rounded up to a multiple of it (the harness uses 256)."""

    def __init__(self, model, tokenizer, room: int = 256, pad: int = 256):
        self.m, self.tok = model, tokenizer
        self.room, self.pad = int(room), int(pad)
        self.dev = next(model.parameters()).device
        self.dec: Optional[IncrementalDecoder] = None
        self.snap: Optional[dict] = None
        self.doc_tokens = 0

    def _ids(self, x: Union[str, Sequence[int], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, str):
            x = self.tok(x, add_special_tokens=False)["input_ids"]
        return torch.as_tensor(x, dtype=torch.long, device=self.dev).reshape(1, -1)

    @torch.no_grad()
    def read(self, document: Union[str, Sequence[int], torch.Tensor]) -> dict:
        """Read the document (one forward over its row) and keep its cache. Returns its length and the seconds."""
        ids = self._ids(document)
        L0 = ids.shape[1]
        eos = int(self.tok.eos_token_id)
        Lb = -(-(L0 + self.room) // self.pad) * self.pad
        buf = torch.full((1, Lb), eos, dtype=torch.long, device=self.dev)
        buf[:, :L0] = ids
        doc = torch.full((1, Lb), -1, dtype=torch.long, device=self.dev)
        doc[:, :L0] = 0
        _sync(self.dev)
        t0 = time.time()
        dec = IncrementalDecoder(self.m)  # raises NotImplementedError for a read the cache does not cover
        dec.prefill(buf, doc, L0)
        snap = dec.snapshot()
        _sync(self.dev)
        self.dec, self.snap, self.doc_tokens = dec, snap, L0
        return dict(tokens=L0, row=Lb, seconds=time.time() - t0)

    @torch.no_grad()
    def ask(self, question: Union[str, Sequence[int], torch.Tensor], max_new: int = 32,
            stop: Optional[Sequence[str]] = None) -> tuple[str, float, list]:
        """(answer text, seconds, answer token ids). `stop`: strings that end the answer (kept in the text)."""
        if self.dec is None:
            raise RuntimeError("read a document first")
        q = self._ids(question)
        eos = int(self.tok.eos_token_id)
        stops = [s for s in (stop or []) if s]

        def check(out, t, done):
            if stops and not bool(done[0]):
                text = self.tok.decode(out[0, self.doc_tokens + q.shape[1]: t].tolist())
                done[0] = any(s in text for s in stops)
            return done

        _sync(self.dev)
        t0 = time.time()
        ans = self.dec.ask(self.snap, q, max_new, eos=eos, check=check if stops else None)
        _sync(self.dev)
        toks = ans[0].tolist()
        while toks and toks[-1] == eos:
            toks.pop()
        return self.tok.decode(toks), time.time() - t0, toks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="", help="default: the checkpoint directory")
    ap.add_argument("--doc", required=True, help="a text file")
    ap.add_argument("--question", action="append", required=True, help="repeat for several questions")
    ap.add_argument("--max_tokens", type=int, default=0, help="cut the document to this many tokens (0: all)")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="flex")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.checkpoint)
    m = load_perceiver_ar_for_eval(a.checkpoint, device=a.device, dtype=getattr(torch, a.dtype),
                                   attn_backend=a.backend, attn_pad_multiple=1)
    ids = tok(Path(a.doc).read_text(), add_special_tokens=False)["input_ids"]
    if a.max_tokens:
        ids = ids[: a.max_tokens]
    room = max(len(tok(q, add_special_tokens=False)["input_ids"]) for q in a.question) + a.new
    s = DocumentSession(m, tok, room=room)
    r = s.read(ids)
    print(f"read {r['tokens']} tokens in {r['seconds']:.2f} s")
    for q in a.question:
        text, sec, _ = s.ask(q, a.new)
        print(f"[{sec:.2f} s] {q.strip()} -> {text.strip()!r}")


if __name__ == "__main__":
    main()
