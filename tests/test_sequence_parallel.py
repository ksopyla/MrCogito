"""Correctness tests for nn.sequence_parallel (CPU/gloo, spawned ranks).

Two levels:
  1. DistLatTokAttention — sharded-KEY attention must equal single-process softmax(S)@v
     in forward and backward (dS, dv). The global token-axis softmax is reconstructed
     across ranks via max/sum/g all-reduces.
  2. ConceptEncoderForConditionalLM.set_sequence_parallel — a 2-rank seq-parallel
     FORWARD+BACKWARD must equal the single-GPU model (loss value + every parameter
     gradient), with the non-first-shard boundary positions masked identically in the
     reference. This validates token sharding, global positions (RoPE + token-pos
     embeddings via position_offset), replicated concepts, and the global-gradient CE.

Run under gloo so it runs on the local mac (no GPU) via torch.multiprocessing.spawn.
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM
from nn.sequence_parallel import DistLatTokAttention, sync_seq_parallel_grads

BACKEND = "gloo"


# --------------------------------------------------------------------------- #
# 1. DistLatTokAttention primitive
# --------------------------------------------------------------------------- #
def _reference_attn(r_lat, r_tok, v_tok, dout, scale):
    r_lat = r_lat.clone().requires_grad_(True)
    r_tok = r_tok.clone().requires_grad_(True)
    v_tok = v_tok.clone().requires_grad_(True)
    S = (r_lat @ r_tok.transpose(-2, -1)) * scale
    out = torch.matmul(torch.softmax(S, dim=-1), v_tok)
    out.backward(dout)
    return out.detach(), r_lat.grad, r_tok.grad, v_tok.grad


def _attn_worker(rank, world, port):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(BACKEND, rank=rank, world_size=world)
    pg = dist.group.WORLD
    try:
        torch.manual_seed(123)
        B, h, C, N, d = 2, 4, 8, 16, 8
        r_lat = torch.randn(B, h, C, d, dtype=torch.float64)          # replicated
        r_tok_full = torch.randn(B, h, N, d, dtype=torch.float64)     # sharded
        v_tok_full = torch.randn(B, h, N, d, dtype=torch.float64)
        scale = d ** -0.5
        torch.manual_seed(7)
        dout = torch.randn(B, h, C, d, dtype=torch.float64)

        ref_out, ref_drlat, ref_drtok, ref_dvtok = _reference_attn(r_lat, r_tok_full, v_tok_full, dout, scale)
        n = N // world
        s0, s1 = rank * n, (rank + 1) * n
        r_lat_sh = r_lat.clone().requires_grad_(True)
        r_tok_sh = r_tok_full[..., s0:s1, :].clone().requires_grad_(True)
        v_tok_sh = v_tok_full[..., s0:s1, :].clone().requires_grad_(True)
        out = DistLatTokAttention.apply(r_lat_sh, r_tok_sh, v_tok_sh, scale, None, pg)
        torch.testing.assert_close(out, ref_out, atol=1e-8, rtol=1e-6)
        out.backward(dout)
        # d(r_lat) is all-reduced INSIDE the Function -> full on every rank already.
        torch.testing.assert_close(r_lat_sh.grad, ref_drlat, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(r_tok_sh.grad, ref_drtok[..., s0:s1, :], atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(v_tok_sh.grad, ref_dvtok[..., s0:s1, :], atol=1e-8, rtol=1e-6)
    finally:
        dist.destroy_process_group()


def test_dist_lat_tok_attention_matches_global():
    port = str(29571 + os.getpid() % 1000)
    mp.spawn(_attn_worker, args=(2, port), nprocs=2, join=True)


# --------------------------------------------------------------------------- #
# 2. Full model sequence parallelism
# --------------------------------------------------------------------------- #
def _build_model(N, V=40, H=32, C=8, K=4, enc_layers=2, dec_layers=2):
    cfg = ConceptEncoderConfig(
        vocab_size=V, hidden_size=H, token_embedding_dim=16, concept_num=C,
        num_hidden_layers=enc_layers, num_attention_heads=4, intermediate_size=64,
        max_sequence_length=N, decoder_num_layers=dec_layers, decoder_type="causal_ar",
        decoder_pos_type="rope", hidden_act="silu", norm_type="rmsnorm", use_bixt=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
        decoder_context_window=K, decoder_attn_impl="chunked_window",
        decoder_attn_chunk_size=8, chunked_ce_block_size=8,
        attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
    )
    torch.manual_seed(0)
    return ConceptEncoderForConditionalLM(cfg).to(torch.float32)


def _model_worker(rank, world, port):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(BACKEND, rank=rank, world_size=world)
    pg = dist.group.WORLD
    try:
        N, B, K, dec_layers = 32, 2, 4, 2
        V = 40

        # --- seq-parallel model (identical weights on every rank via seed 0) ---
        m_sp = _build_model(N, V=V, K=K, dec_layers=dec_layers)
        m_sp.set_sequence_parallel(pg)
        m_sp.train()

        torch.manual_seed(100)
        prefix = torch.randint(3, V, (B, N))
        pm = torch.ones(B, N, dtype=torch.long)
        suffix = torch.randint(3, V, (B, N))
        sm = torch.ones(B, N, dtype=torch.long)
        labels = suffix.clone()

        out_sp = m_sp(prefix_input_ids=prefix, prefix_attention_mask=pm,
                      suffix_input_ids=suffix, suffix_attention_mask=sm, labels=labels)
        out_sp.loss.backward()
        # split gradient sync: SUM token/suffix/decoder/lm_head, AVG concept-side
        sync_seq_parallel_grads(m_sp, pg)

        # --- single-GPU reference, identical weights, with the SAME boundary mask ---
        m_rf = _build_model(N, V=V, K=K, dec_layers=dec_layers)
        m_rf.train()
        bm = dec_layers * K  # == model._sp_boundary_mask() default (L*window)
        n = N // world
        labels_ref = labels.clone()
        labels_ref[:, n:n + bm] = -100  # rank-1's boundary positions (global)
        out_rf = m_rf(prefix_input_ids=prefix, prefix_attention_mask=pm,
                      suffix_input_ids=suffix, suffix_attention_mask=sm, labels=labels_ref)
        out_rf.loss.backward()

        if rank == 0:
            torch.testing.assert_close(
                out_sp.loss.detach(), out_rf.loss.detach(), atol=1e-4, rtol=1e-4)
            sp_params = dict(m_sp.named_parameters())
            rf_params = dict(m_rf.named_parameters())
            assert set(sp_params) == set(rf_params)
            worst = 0.0
            worst_name = ""
            for name, p_rf in rf_params.items():
                p_sp = sp_params[name]
                assert p_sp.grad is not None and p_rf.grad is not None, name
                denom = p_rf.grad.abs().max().clamp(min=1e-8)
                rel = (p_sp.grad - p_rf.grad).abs().max().item() / denom.item()
                if rel > worst:
                    worst, worst_name = rel, name
            assert worst < 1e-3, f"worst rel Δgrad={worst:.2e} on {worst_name}"
            print(f"[sp-model] OK: loss Δ={abs(out_sp.loss.item()-out_rf.loss.item()):.2e} "
                  f"worst rel |Δgrad|={worst:.2e} ({worst_name})")
    finally:
        dist.destroy_process_group()


def test_seq_parallel_model_matches_single_gpu():
    port = str(29581 + os.getpid() % 1000)
    mp.spawn(_model_worker, args=(2, port), nprocs=2, join=True)
