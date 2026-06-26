#!/usr/bin/env python
"""Memory benchmark for the Concept Encoder AR decoder (E02 / E05).

Runs a few forward+backward steps on ONE GPU (no DDP) at a given sequence length
and logs peak allocated / reserved CUDA memory per step, so each optimization
(fused windowed attention, chunked CE, gradient checkpointing, allocator tuning)
can be measured against the same baseline.

Usage:
  uv run python scripts/bench_memory.py --seq_len 2048 --steps 3 \
      --decoder_context_window 128    # E05 windowed
  uv run python scripts/bench_memory.py --seq_len 8192 --steps 3   # E02 full-causal

By default builds a tiny AR model (H768/T256/L6/C128/D4) matching E05's arch but
with a small vocab to keep the lm_head from dominating unless --full_vocab is set
(use --full_vocab to measure the real O(N*V) CE spike).

Outputs a JSON line per step to stdout and appends a row to --results_csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM


def build_config(args) -> ConceptEncoderConfig:
    vocab = 49152 if args.full_vocab else 1024
    cfg = dict(
        vocab_size=vocab,
        hidden_size=args.hidden_size,
        token_embedding_dim=args.token_emb,
        concept_num=args.concept_num,
        num_hidden_layers=args.enc_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate,
        max_sequence_length=args.seq_len,
        decoder_num_layers=args.dec_layers,
        decoder_type="causal_ar",
        decoder_pos_type="rope",
        decoder_word_dropout=0.0,
        hidden_act="silu",
        norm_type="rmsnorm",
        use_bixt=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    if args.decoder_context_window and args.decoder_context_window > 0:
        cfg["decoder_context_window"] = args.decoder_context_window
    if args.decoder_attn_impl and args.decoder_attn_impl != "sdpa":
        cfg["decoder_attn_impl"] = args.decoder_attn_impl
        cfg["decoder_attn_chunk_size"] = args.decoder_attn_chunk_size
    if args.chunked_ce_block_size and args.chunked_ce_block_size > 0:
        cfg["chunked_ce_block_size"] = args.chunked_ce_block_size
    return ConceptEncoderConfig(**cfg)


def fmt_mb(b: float) -> float:
    return round(b / (1024 * 1024), 1)


def run(args):
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    torch.manual_seed(args.seed)

    config = build_config(args)
    model = ConceptEncoderForConditionalLM(config).to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    model.train()

    # AdamW-fused matches the real training optimizer (fp32 states).
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        "event": "config", "seq_len": args.seq_len,
        "decoder_context_window": args.decoder_context_window,
        "hidden_size": args.hidden_size, "vocab": config.vocab_size,
        "batch_size": args.batch_size, "bf16": args.bf16,
        "decoder_layers": args.dec_layers, "encoder_layers": args.enc_layers,
        "n_params": n_params, "n_params_M": round(n_params / 1e6, 1),
        "gradient_checkpointing": args.gradient_checkpointing,
    }))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    B, N = args.batch_size, args.seq_len
    # prefix_suffix objective: prefix and suffix are both length N.
    prefix_ids = torch.randint(3, config.vocab_size, (B, N), device=device)
    prefix_mask = torch.ones(B, N, dtype=torch.long, device=device)
    suffix_ids = torch.randint(3, config.vocab_size, (B, N), device=device)
    suffix_mask = torch.ones(B, N, dtype=torch.long, device=device)
    labels = suffix_ids.clone()
    # leave labels all-real (no pad) to measure the worst-case N-length forward

    results = []
    for step in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        out = model(
            prefix_input_ids=prefix_ids, prefix_attention_mask=prefix_mask,
            suffix_input_ids=suffix_ids, suffix_attention_mask=suffix_mask,
            labels=labels,
        )
        loss = out.loss
        loss.backward()
        torch.cuda.synchronize()
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        row = {
            "event": "step", "step": step, "seq_len": N,
            "window": args.decoder_context_window or 0,
            "loss": round(loss.item(), 4), "dt_s": round(dt, 3),
            "peak_alloc_MB": fmt_mb(peak_alloc),
            "peak_reserved_MB": fmt_mb(peak_reserved),
            "bf16": args.bf16, "vocab": config.vocab_size,
            "ckpt": args.gradient_checkpointing,
            "batch": B,
        }
        print(json.dumps(row))
        results.append(row)
        if not torch.isfinite(loss):
            print(json.dumps({"event": "non_finite_loss", "step": step}))
            break

    if args.results_csv:
        exists = os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            if not exists:
                w.writeheader()
            for r in results:
                w.writerow(r)
    print(json.dumps({"event": "done", "seq_len": N, "steps": args.steps}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--decoder_context_window", type=int, default=128)
    p.add_argument("--decoder_attn_impl", default="sdpa", choices=["sdpa", "chunked_window"])
    p.add_argument("--decoder_attn_chunk_size", type=int, default=2048)
    p.add_argument("--chunked_ce_block_size", type=int, default=0)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--token_emb", type=int, default=256)
    p.add_argument("--concept_num", type=int, default=128)
    p.add_argument("--enc_layers", type=int, default=6)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", dest="bf16", action="store_false")
    p.add_argument("--full_vocab", action="store_true", help="Use V=49152 (real lm_head); default V=1024 to isolate attention mem")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_csv", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
