#!/usr/bin/env python
"""E35 sanity check: can agent A get a small fixed set of documents across to agent B at all?

Trains the warm-started A2A model on `--docs` world rows for `--steps` steps (no held-out data: this is an overfit
check), then scores the answer tokens of those same rows teacher-forced with (a) each row's own message and (b) the
message of another row. Memorising the answers in B's weights helps both equally; only information that crosses in
the message makes (a) beat (b).

  uv run python scripts/a2a_overfit_check.py --sender Cache/e35/init/sender --receiver Cache/e35/init/receiver \\
      --data <a2a data dir> --docs 64 --steps 400 --slots 16 --lr 1e-3 --out Cache/e35/overfit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from nn.latent_messages import build_a2a_from_checkpoints, split_rows  # noqa: E402


def answer_mask(r_ids: torch.Tensor, r_mask: torch.Tensor, answer_tok: int, dot_tok: int) -> torch.Tensor:
    """Targets inside answers: tokens after "Answer" ":" up to and including the closing period."""
    m = torch.zeros_like(r_ids, dtype=torch.bool)
    for b in range(r_ids.shape[0]):
        row = r_ids[b].tolist()
        n = int(r_mask[b].sum())
        i = 0
        while i < n:
            if row[i] == answer_tok:
                j = i + 2
                while j < n:
                    m[b, j] = True
                    if row[j] == dot_tok:
                        break
                    j += 1
                i = j
            i += 1
    return m


@torch.no_grad()
def score(model, d_ids, d_mask, r_ids, r_mask, amask, shift: int) -> dict:
    mem, _ = model.write(d_ids, d_mask)
    msg = model.send(mem)
    if shift:
        msg = msg.roll(shift, dims=0)  # each row gets another row's message
    lg = model.answer(msg, r_ids, r_mask).logits
    pred = lg[:, :-1].argmax(-1)
    tgt = r_ids[:, 1:]
    am = amask[:, 1:]
    tok_acc = float((pred == tgt)[am].float().mean())
    lp = torch.log_softmax(lg[:, :-1].float(), -1).gather(-1, tgt[..., None])[..., 0]
    return {"answer_token_acc": tok_acc, "answer_nll": float(-lp[am].mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sender", required=True)
    p.add_argument("--receiver", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--docs", type=int, default=64)
    p.add_argument("--max_doc_tokens", type=int, default=1024)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze_receiver", action="store_true", help="train A only (B frozen): no way to memorise in B")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(os.path.join(a.data, "tokenizer"))
    ans_tok = tok.encode("Answer", add_special_tokens=False)[0]
    dot_tok = tok.encode(".", add_special_tokens=False)[-1]
    ds = load_from_disk(os.path.join(a.data, "world", "train"))
    rows = []
    for i in range(len(ds)):
        ids = ds[i]["input_ids"]
        e = next((j for j in range(len(ids) - 1) if ids[j] == tok.eos_token_id and ids[j + 1] == tok.bos_token_id), None)
        if e is not None and e <= a.max_doc_tokens:
            rows.append(ids)
        if len(rows) == a.docs:
            break
    L = max(len(r) for r in rows)
    batch = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(rows), L, dtype=torch.long)
    for i, r in enumerate(rows):
        batch[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    d_ids, d_mask, r_ids, r_mask = (t.to(dev) for t in split_rows(batch, mask, tok.bos_token_id, tok.eos_token_id,
                                                                     tok.pad_token_id))
    amask = answer_mask(r_ids, r_mask, ans_tok, dot_tok)
    model = build_a2a_from_checkpoints(a.sender, a.receiver, msg_slots=a.slots).to(dev)
    if a.freeze_receiver:
        for q in model.receiver.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    res = {"args": vars(a), "n_docs": len(rows), "answer_tokens": int(amask.sum()), "log": []}
    model.eval()
    res["start"] = {"own": score(model, d_ids, d_mask, r_ids, r_mask, amask, 0),
                    "other": score(model, d_ids, d_mask, r_ids, r_mask, amask, 1)}
    print("start", res["start"], flush=True)
    t0 = time.time()
    g = torch.Generator().manual_seed(0)
    for step in range(1, a.steps + 1):
        model.train()
        idx = torch.randperm(len(rows), generator=g)[: a.batch].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            mem, w = model.write(d_ids[idx], d_mask[idx])
            msg = model.send(mem)
            tgt = r_ids[idx].masked_fill(~r_mask[idx].bool(), -100)
            loss = model.answer(msg, r_ids[idx], r_mask[idx], labels=tgt).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == a.steps:
            model.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                own = score(model, d_ids, d_mask, r_ids, r_mask, amask, 0)
                other = score(model, d_ids, d_mask, r_ids, r_mask, amask, 1)
            rec = {"step": step, "loss": float(loss), "own": own, "other": other, "s": round(time.time() - t0)}
            res["log"].append(rec)
            print(f"step {step:4d} loss {float(loss):.3f} · answer tokens right: own message {own['answer_token_acc']:.2f} "
                  f"vs another row's {other['answer_token_acc']:.2f} · surprise {own['answer_nll']:.2f} vs "
                  f"{other['answer_nll']:.2f} ({rec['s']}s)", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
