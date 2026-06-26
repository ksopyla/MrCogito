"""Correctness tests for nn.sequence_parallel (CPU/gloo, spawned ranks).

DistLatTokAttention with sharded KEYS must equal single-process softmax(S) @ v in
forward AND backward (dS, dv): the global softmax over tokens is reconstructed across
ranks via the max/sum/g all-reduces. Run under gloo so it executes on the local mac
(no GPU needed) via torch.multiprocessing.spawn.
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nn.sequence_parallel import DistLatTokAttention

BACKEND = "gloo"
WORLD = 2


def _reference(S, v, dout):
    """Single-process global softmax attention + backward. S [B,h,C,N], v [B,h,N,d]."""
    S = S.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    A = torch.softmax(S, dim=-1)
    out = torch.matmul(A, v)
    out.backward(dout)
    return out.detach(), S.grad, v.grad


def _worker(rank):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ["MASTER_PORT"])
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD)
    pg = dist.group.WORLD
    try:
        torch.manual_seed(123)
        B, h, C, N, d = 2, 4, 8, 16, 8
        S_full = torch.randn(B, h, C, N, dtype=torch.float64)
        v_full = torch.randn(B, h, N, d, dtype=torch.float64)
        torch.manual_seed(7)
        dout = torch.randn(B, h, C, d, dtype=torch.float64)

        ref_out, ref_dS, ref_dv = _reference(S_full, v_full, dout)
        n = N // WORLD
        s0, s1 = rank * n, (rank + 1) * n
        S_sh = S_full[..., s0:s1].clone().requires_grad_(True)
        v_sh = v_full[..., s0:s1, :].clone().requires_grad_(True)
        out = DistLatTokAttention.apply(S_sh, v_sh, pg)        # [B,h,C,d] replicated
        torch.testing.assert_close(out, ref_out, atol=1e-8, rtol=1e-6)
        out.backward(dout)
        torch.testing.assert_close(S_sh.grad, ref_dS[..., s0:s1], atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(v_sh.grad, ref_dv[..., s0:s1, :], atol=1e-8, rtol=1e-6)
    finally:
        dist.destroy_process_group()


def test_dist_lat_tok_attention_matches_global():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29571 + os.getpid() % 1000)
    mp.spawn(_worker, args=(), nprocs=WORLD, join=True)
