#!/usr/bin/env python
"""Optimizer convergence comparison on REAL text (wikitext-103) — AdamW vs Muon.

The memory benches use random tokens, which can't show learning. This script trains the
concept_ar model on cached wikitext-103 (SmolLM2 tokenizer) at a moderate context and
logs train loss every --log_every steps, so optimizers can be compared on CONVERGENCE
SPEED (loss vs step) and wall-clock — not just memory. Single GPU; same init + data for
all optimizers (deterministic), so differences are attributable to the optimizer.

Usage (Odra):
  uv run python scripts/bench_optimizer_convergence.py --optim adamw  --lr 2e-4 --steps 300
  uv run python scripts/bench_optimizer_convergence.py --optim muon   --lr 0.02 --steps 300
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM


def load_real_tokens(tokenizer_name: str, context: int, n_seqs: int, device):
    """Load cached wikitext-103, tokenize with SmolLM2 (streaming, bounded), chunk into
    fixed-length EOS-terminated sequences. Stops as soon as it has enough tokens."""
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    eos = tok.eos_token_id or 0
    target = n_seqs * context + context
    ids: list[int] = []
    for t in ds["text"]:
        if len(t) < 200:
            continue
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(eos)
        if len(ids) >= target:
            break
    ids = ids[: n_seqs * context]
    seqs = [ids[i * context:(i + 1) * context] for i in range(n_seqs)]
    return torch.tensor(seqs, dtype=torch.long, device=device)  # [n_seqs, context]


def build_model(context, window):
    cfg = ConceptEncoderConfig(
        vocab_size=49152, hidden_size=768, token_embedding_dim=256, concept_num=128,
        num_hidden_layers=6, num_attention_heads=8, intermediate_size=2048,
        max_sequence_length=context, decoder_num_layers=4, decoder_type="causal_ar",
        decoder_pos_type="rope", hidden_act="silu", norm_type="rmsnorm", use_bixt=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
        decoder_context_window=window, decoder_attn_impl="chunked_window",
        decoder_attn_chunk_size=2048, chunked_ce_block_size=2048,
    )
    torch.manual_seed(0)
    return ConceptEncoderForConditionalLM(cfg).to("cuda").to(torch.bfloat16)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optim", choices=["adamw", "muon"], required=True)
    p.add_argument("--lr", type=float, default=None, help="override default LR")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)
    n_seqs = args.steps * args.batch + 16
    data = load_real_tokens(args.tokenizer, args.context, n_seqs, "cuda")  # [n_seqs, ctx]

    model = build_model(args.context, args.window)
    model.train()
    model.gradient_checkpointing_enable()
    # reconstruction objective: encode input -> concepts -> AR-decode the same input
    if args.optim == "muon":
        from nn.muon import Muon
        lr = args.lr if args.lr is not None else 0.02
        opt = Muon(model.parameters(), lr=lr, adamw_lr=2e-3)
    else:
        lr = args.lr if args.lr is not None else 2e-4
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), fused=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"event": "config", "optim": args.optim, "lr": lr, "context": args.context,
                      "batch": args.batch, "steps": args.steps, "n_params_M": round(n_params / 1e6, 1)}),
          flush=True)
    t_start = time.time()
    for step in range(args.steps):
        idx = (step * args.batch) % (n_seqs - args.batch)
        ids = data[idx:idx + args.batch]                       # [B, ctx]
        mask = torch.ones_like(ids)
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            el = time.time() - t_start
            print(json.dumps({"event": "loss", "step": step, "loss": round(out.loss.item(), 4),
                              "elapsed_s": round(el, 1)}), flush=True)
    print(json.dumps({"event": "done", "optim": args.optim, "final_loss": round(out.loss.item(), 4),
                      "elapsed_s": round(time.time() - t_start, 1)}), flush=True)


if __name__ == "__main__":
    main()
