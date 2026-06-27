#!/usr/bin/env python
"""Sequence-parallel memory/throughput bench for the Concept Encoder (F6).

Launch with torchrun. Shards the token axis across GPUs so per-GPU activation memory
scales as O(N/P). Each rank holds a full model replica; the model narrows the sequence
to its shard internally (encode_decode_loss SP branch). After backward,
sync_seq_parallel_grads SUMs token-side and AVGs concept-side grads, so every rank's
optimizer step is identical and the replicas stay in sync.

The seq_len is rounded down to a multiple of world size (each rank gets N/P tokens).

Usage (Odra, 3x 3090):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=3 scripts/bench_seq_parallel.py \
      --seq_len 1048576 --decoder_context_window 128 --chunked_ce_block_size 2048 --steps 3
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM
from nn.sequence_parallel import sync_seq_parallel_grads


def build_config(args, seq_len):
    cfg = dict(
        vocab_size=49152, hidden_size=768, token_embedding_dim=256, concept_num=128,
        num_hidden_layers=6, num_attention_heads=8, intermediate_size=2048,
        max_sequence_length=seq_len, decoder_num_layers=4, decoder_type="causal_ar",
        decoder_pos_type="rope", hidden_act="silu", norm_type="rmsnorm", use_bixt=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
        decoder_context_window=args.decoder_context_window,
        decoder_attn_impl="chunked_window", decoder_attn_chunk_size=2048,
        chunked_ce_block_size=args.chunked_ce_block_size,
    )
    return ConceptEncoderConfig(**cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_len", type=int, default=1048576)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--decoder_context_window", type=int, default=128)
    p.add_argument("--chunked_ce_block_size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)

    N = (args.seq_len // world) * world  # divisible by world
    assert N > 0
    config = build_config(args, N)
    model = ConceptEncoderForConditionalLM(config).to(device).to(torch.bfloat16)
    model.set_sequence_parallel(dist.group.WORLD)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(json.dumps({
            "event": "config", "seq_len": N, "world": world, "shard_len": N // world,
            "batch": args.batch_size, "n_params_M": round(n_params / 1e6, 1),
            "vocab": config.vocab_size, "bf16": True,
        }), flush=True)

    B = args.batch_size
    torch.manual_seed(args.seed)  # identical inputs on every rank -> same sharded seq
    prefix = torch.randint(3, config.vocab_size, (B, N), device=device)
    pm = torch.ones(B, N, dtype=torch.long, device=device)
    suffix = torch.randint(3, config.vocab_size, (B, N), device=device)
    sm = torch.ones(B, N, dtype=torch.long, device=device)
    labels = suffix.clone()

    for step in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        out = model(prefix_input_ids=prefix, prefix_attention_mask=pm,
                    suffix_input_ids=suffix, suffix_attention_mask=sm, labels=labels)
        out.loss.backward()
        sync_seq_parallel_grads(model, dist.group.WORLD)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.tensor([torch.cuda.max_memory_allocated()], device=device)
        peaks = [torch.empty_like(peak) for _ in range(world)]
        dist.all_gather(peaks, peak)
        if rank == 0:
            pm_mb = {f"r{r}": round(float(peaks[r]) / 1048576, 1) for r in range(world)}
            print(json.dumps({
                "event": "step", "step": step, "seq_len": N, "world": world,
                "shard_len": N // world, "loss": round(out.loss.item(), 4),
                "dt_s": round(dt, 3), "peak_alloc_MB": pm_mb,
                "max_peak_MB": round(max(float(x) for x in peaks) / 1048576, 1),
            }), flush=True)
    if rank == 0:
        print(json.dumps({"event": "done", "seq_len": N, "world": world}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
