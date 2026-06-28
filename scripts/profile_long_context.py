#!/usr/bin/env python
"""Locate the memory wall in the long-context forward: per-PHASE peak allocation.

The round-1 note showed forward-only peak ~= full (fwd+bwd) peak — backward only
recomputes the forward transient under checkpointing, so the wall is the FORWARD
transient. This drives the three phases MANUALLY (encoder -> decoder -> lm_head+CE),
resetting the CUDA peak counter before each, so each phase reports its own peak
(including the retained state live at its start). The max across phases == the
global forward peak (phases are sequential), which must match bench_memory's
fwd_peak — and tells us whether the encoder, the decoder, or the output head is
the wall, and the per-phase slope in N.

Runs WITH grad (model.train()) but NO backward, matching the bench fwd_only reading.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM


def build_config(args) -> ConceptEncoderConfig:
    cfg = dict(
        vocab_size=49152, hidden_size=args.hidden_size, token_embedding_dim=args.token_emb,
        concept_num=args.concept_num, num_hidden_layers=args.enc_layers,
        num_attention_heads=args.num_heads, intermediate_size=args.intermediate,
        max_sequence_length=args.seq_len, decoder_num_layers=args.dec_layers,
        decoder_type="causal_ar", decoder_pos_type="rope", hidden_act="silu",
        norm_type="rmsnorm", use_bixt=True, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, tie_word_embeddings=False,
        decoder_attn_impl="chunked_window", decoder_attn_chunk_size=2048,
        chunked_ce_block_size=args.ce_block,
    )
    if args.window and args.window > 0:
        cfg["decoder_context_window"] = args.window
    return ConceptEncoderConfig(**cfg)


def mb(b: float) -> float:
    return round(b / (1024 * 1024), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_len", type=int, default=65536)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--token_emb", type=int, default=256)
    p.add_argument("--concept_num", type=int, default=128)
    p.add_argument("--enc_layers", type=int, default=6)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--ce_block", type=int, default=2048)
    p.add_argument("--no_ckpt", action="store_true", help="Disable gradient checkpointing.")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    torch.manual_seed(42)
    model = ConceptEncoderForConditionalLM(build_config(args)).to(device).to(torch.bfloat16)
    model.train()
    if not args.no_ckpt:
        model.gradient_checkpointing_enable()

    weights_alloc = torch.cuda.memory_allocated()

    B, N = args.batch_size, args.seq_len
    pid = torch.randint(3, 49152, (B, N), device=device)
    pm = torch.ones(B, N, dtype=torch.long, device=device)
    sid = torch.randint(3, 49152, (B, N), device=device)
    sm = torch.ones(B, N, dtype=torch.long, device=device)
    lab = sid.clone()
    dec_in = model._shift_right(sid)
    dec_kpm = (sm == 0)  # SDPA convention: True = ignore (all-real here -> all False)

    phases = []

    def run_phase(name, fn):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        out = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        alloc = torch.cuda.memory_allocated()
        phases.append({"name": name, "peak_MB": mb(peak), "alloc_after_MB": mb(alloc)})
        return out

    concepts = run_phase("encoder", lambda: model.encode_concepts(
        input_ids=pid, attention_mask=pm, return_dict=True).last_hidden_state)
    hidden = run_phase("decoder", lambda: model.decode_hidden(
        concepts, dec_in, key_padding_mask=dec_kpm))
    loss = run_phase("lm_head+CE", lambda: model._chunked_teacher_forced_ce(
        hidden, lab, args.ce_block))

    print(json.dumps({
        "seq_len": N, "batch": B, "window": args.window, "ckpt": not args.no_ckpt,
        "ce_block": args.ce_block,
        "weights_alloc_MB": mb(weights_alloc),
        "phases": phases,
        "global_peak_MB": round(max(ph["peak_MB"] for ph in phases), 1),
        "loss": round(loss.item(), 4),
    }))


if __name__ == "__main__":
    main()
