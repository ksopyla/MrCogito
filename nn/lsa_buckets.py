# Copyright 2026 Prefrontal AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported into MrCogito (2026-11-06) from github.com/Prefrontal-AI/latent-sparse-attention @ 5ebea73:
#   src/lsa/core/ward.py                          pool_keys, ward_rounds (verbatim)
#   src/lsa/training/native/lsa.py                build_flat_buckets → flat_buckets (labels instead of member lists)
#   src/lsa/training/native/kernels/reference.py  scores, select_indices → lsa_open_table (via nn.exact_read)
#   src/lsa/training/native/lsa_indexer.py        query_head_reduction "max": unit per-head queries, unit keys
# Licence text: third_party/LSA_LICENSE. CHANGES (MrCogito, marked "MrCogito:" below):
#   * documents are packed in our rows with no cross-document attention: a query takes keys of its own document
#     only, and its sinks are the first `sinks` tokens of its document (LSA: of the row);
#   * the selected keys are returned as a per-query table over buckets (nn.exact_read.OpenTable) for a flex mask,
#     not as a list of positions for LSA's Triton gather kernel; the keys taken are the same;
#   * `ward_merge`'s refusal check (one host sync) is a config-time check instead (`lsa_bucket_cap ≥ 2·bucket`).
"""LSA's flat bucket read (the released native LSA arm, `kind: lsa`, `coarse: false`), for the E33p LSA-native arm.

Every forward, without gradient: the reading layer's keys (unit rows) are summed in pools of 4, and each closed
512-key block's 128 pools are merged by mutual-cheapest-pair Ward rounds into 64 buckets of at most 16 keys; a
bucket's centre is its normalised key sum. Each query scores every centre of the blocks closed before its own block
(cosine, best query head), keeps its `width` best and takes their keys best bucket first, in position order,
skipping its sinks and window, until `top_k` keys are held (the last bucket cut short).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from nn.exact_read import ChoiceList, GroupIndex, OpenTable, choose, doc_starts, groups_from_spans


def pool_keys(index_keys: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Sum each run of `pool_size` consecutive index keys into a pool key. [G, T, d] → [G, T / pool_size, d]."""
    blocks, key_count, index_dim = index_keys.shape
    return index_keys.reshape(blocks, key_count // pool_size, pool_size, index_dim).sum(2)


def ward_rounds(
    pool_keys: torch.Tensor,
    pool_sizes: torch.Tensor,
    bucket_count: int,
    bucket_cap: int,
    max_rounds: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge each block's pools down to `bucket_count` buckets and label every pool with its bucket (LSA, verbatim).

    Merging two buckets costs |a_i| + |a_j| − |a_i + a_j| on their key sums. Each round every bucket finds its
    cheapest partner among those the cap allows (a tie to the lower index), and every pair that is each other's
    cheapest merges, the cheapest pairs first in the round that reaches the block's count.

    Returns labels [G, P] int64 (numbered from 0 within its block in the order of its first pool) and whether each
    block reached its count [G] bool."""
    blocks, pools, _ = pool_keys.shape
    device = pool_keys.device
    key_sums = pool_keys.clone()
    bucket_sizes = pool_sizes.clone()
    alive = torch.ones(blocks, pools, dtype=torch.bool, device=device)
    parent = torch.arange(pools, device=device).expand(blocks, pools).clone()
    index = torch.arange(pools, device=device).expand(blocks, pools)
    diagonal = torch.eye(pools, dtype=torch.bool, device=device)
    for _ in range(max_rounds):
        merges_left = alive.sum(1) - bucket_count
        if not (merges_left > 0).any():
            break
        gram = torch.bmm(key_sums, key_sums.transpose(1, 2))
        squares = gram.diagonal(dim1=1, dim2=2)
        merged_alignments = (squares[:, :, None] + squares[:, None, :] + 2 * gram).clamp_min(0).sqrt()
        alignments = squares.clamp_min(0).sqrt()
        merge_costs = alignments[:, :, None] + alignments[:, None, :] - merged_alignments
        allowed = alive[:, :, None] & alive[:, None, :] & ~diagonal
        allowed &= bucket_sizes[:, :, None] + bucket_sizes[:, None, :] <= bucket_cap
        best_merge_costs, best = merge_costs.masked_fill(~allowed, float("inf")).min(2)
        del gram, merged_alignments, merge_costs, allowed
        # A pair that is each other's cheapest merges into its lower pool
        mutual = (best_merge_costs < float("inf")) & (best.gather(1, best) == index) & (index < best)
        # A block merges no further than its count, its cheapest pairs first
        rank = best_merge_costs.masked_fill(~mutual, float("inf")).argsort(1).argsort(1)
        mutual &= rank < merges_left.clamp_min(0)[:, None]
        if not mutual.any():
            break
        block_ids, kept = mutual.nonzero(as_tuple=True)
        absorbed = best[block_ids, kept]
        key_sums[block_ids, kept] += key_sums[block_ids, absorbed]
        bucket_sizes[block_ids, kept] += bucket_sizes[block_ids, absorbed]
        alive[block_ids, absorbed] = False
        parent[block_ids, absorbed] = kept
    # Every absorbed pool points at the pool it merged into - follow the chain to the surviving one
    for _ in range(math.ceil(math.log2(max(pools, 2))) + 1):
        parent = parent.gather(1, parent)
    number = torch.cumsum(alive.long(), 1) - 1
    return number.gather(1, parent), alive.sum(1) <= bucket_count


def closed_block_count(key_count: int, block_size: int) -> int:
    """Blocks some query of a sequence sees closed — those before the last query's own (LSA)."""
    return (key_count - 1) // block_size


@torch.no_grad()
def flat_buckets(index_keys: torch.Tensor, *, block: int, per_block: int, pool: int,
                 cap: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """LSA's `build_flat_buckets`: index keys [B, S, d] fp32 unit rows → (centroids [B, n_blk·per_block, d],
    key labels [B, n_blk, block] long, the bucket of every key within its block). None when no block is closed."""
    B, S, d = index_keys.shape
    blocks = closed_block_count(S, block)
    if blocks <= 0:
        return None
    rows = index_keys[:, : blocks * block].float().reshape(B * blocks, block, d)
    pools = pool_keys(rows, pool)
    pool_sizes = torch.full(pools.shape[:2], float(pool), device=index_keys.device)
    labels, _ = ward_rounds(pools, pool_sizes, per_block, cap)
    key_labels = labels.repeat_interleave(pool, dim=1)                                    # [B·blocks, block]
    key_sums = torch.zeros(rows.shape[0], per_block, d, device=index_keys.device)
    key_sums.scatter_add_(1, key_labels[..., None].expand(-1, -1, d), rows)
    centroids = key_sums / key_sums.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return centroids.reshape(B, blocks * per_block, d), key_labels.view(B, blocks, block)


@torch.no_grad()
def lsa_groups(k: torch.Tensor, doc: torch.Tensor, *, block: int, per_block: int, pool: int, cap: int,
               sinks: int) -> Optional[tuple[torch.Tensor, GroupIndex]]:
    """The buckets of one reading-layer call as groups. k [B, S, 1, dh] (post-norm, no rotation), doc [B, S]."""
    B, S = doc.shape
    iq_k = F.normalize(k[:, :, 0].detach().float(), dim=-1)
    out = flat_buckets(iq_k, block=block, per_block=per_block, pool=pool, cap=cap)
    if out is None:
        return None
    centroids, labels = out
    blocks = labels.shape[1]
    ds = doc_starts(doc)
    pos = torch.arange(S, device=doc.device)[None].expand(B, S)
    # MrCogito: sinks are the first tokens of each document; pads (doc −1) are never taken
    sink = (pos - ds < sinks) & (doc >= 0)
    excl = (sink | (doc < 0))[:, : blocks * block].reshape(B, blocks, block)
    member = torch.where(excl, torch.full_like(labels, -1), labels)
    # a bucket is visible to queries of later blocks (LSA: bucket block < query block)
    g_close = ((torch.arange(blocks, device=doc.device) + 1) * block).repeat_interleave(per_block)[None].expand(B, -1)
    g_doc = torch.zeros_like(g_close)
    gi = groups_from_spans(member, [b * block for b in range(blocks)], S=S, per_span=per_block, doc_start=ds,
                           g_close=g_close.contiguous(), g_doc=g_doc, any_doc=True, sink=sink)
    return centroids, gi


@torch.no_grad()
def lsa_open_table(q: torch.Tensor, centroids: torch.Tensor, gi: GroupIndex, *, top_k: int, width: int,
                   raw_window: int, chunk: int = 0, impl: str = "table"):
    """LSA's selection for every query: score = max over query heads of cos(q_h, centre) on the buckets of blocks
    closed before the query's block; the `width` best; keys best bucket first until `top_k` (nn.exact_read.choose:
    keys of the query's document, outside its sinks and window). q [B, S, h, dh]. `impl = "list"` (E33t stage C,
    evaluation): the choice as a ChoiceList instead of the table."""
    B, S, h, dh = q.shape
    G = centroids.shape[1]
    if chunk <= 0:
        chunk = max(128, min(1024, (1 << 25) // max(B * h * G, 1) // 128 * 128))
    iq = F.normalize(q.detach().float(), dim=-1)
    tab = (ChoiceList(gi, B, S, cap=top_k, m=min(int(width), G)) if impl == "list"
           else OpenTable(gi, B, S, cap=top_k))
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        qp = torch.arange(s0, s1, device=q.device)
        score = torch.einsum("bshd,bgd->bshg", iq[:, s0:s1], centroids).amax(2)            # [B, ch, G]
        visible = gi.g_close[:, None, :] <= qp[None, :, None]
        ids, off = choose(score, visible, gi, qp, m=width, cap=top_k, raw_window=raw_window)
        tab.add(s0, ids, off)
    return tab
