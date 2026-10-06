# E33: why the second hop is not learned, and what to try (2026-10-06)

**Status.** The loop keeps every E31 capability (Odra suite seeds 0–2: on par with `e31_li_m1` on every v3 cell;
E31 from scratch passes the new keyed exams: 1-step chain 97–99 %, recall 1 of 8 98 %, keyed 1 of 4 75–94 %).
Neither E31, the loop nor the 4-layer dense control learns the **second** hop of the parallel chains (C5): all sit
on the guessing floor (picked candidate 6–14 % vs 8.3 %) at 4× budget, with or without a 1 → 2 hop curriculum and
with the hop count in the question. Teacher-forced accuracy stays ~92 %: the rest of a node is copied once the
first letter is given; the node itself is never chosen.

## 1. What the model carries after round 1 (linear probe, local)

`e33cur_edge_loop_s1` (loop, trained on the 1-hop exam, 98 % picked). Hidden state at the decision position (the
answer marker), after the tied core's first round; one linear probe per letter offset, 1536 / 512 rows, chance 25 %:

| state after round 1 | letter 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8–16 |
|---|---|---|---|---|---|---|---|---|
| node 1 (what the read returned) | 0.99 | 0.91 | 0.69 | 0.54 | 0.45 | 0.43 | 0.36 | 0.26–0.30 |
| start node (what the question gave) | 0.38 | 0.42 | 0.35 | 0.40 | 0.49 | 0.67 | 0.78 | 0.70 → 1.00 |

The read returns the **head** of the target's name (the answer head needs only the first letter at this position;
the later letters are copied with teacher forcing). The question's start node is held by its **tail** (the local
layers see the last 16 tokens), so the memory learned to find an edge by the tail of its source name. The second
lookup needs node 1's tail, which round 1 does not carry: the query for hop 2 has nothing to match. The same
mismatch explains why a 4-layer dense model fails at 16-letter nodes, and why the curriculum's progress exit
(round 1 → node 1, first letter only) did not help.

Script: `probe_round1.py` (session scratchpad); to be promoted to `analysis/` if the diagnosis holds.

## 2. Literature that bears on it (verified by the research scout)

- **The plateau is the documented pre-transition state.** In-context 2-hop with distractor chains sits at a uniform
  guess over end nodes, then jumps (Guo et al. 2025, arXiv 2502.13913: ~800 steps, *single-token* nodes, 3 layers).
- **Looped models do multi-hop, but slowly and with atomic entities.** Saunshi et al. 2025 (2502.17416): one layer
  looped 6× solves p-hop (p = 16) at 99.9 %. Kohli et al. 2026 (2604.07822): a 4-layer recurrent-depth model groks
  up to 4 hops only after > 1.3 M steps with a 95 %-gated hop curriculum. Yao et al. 2025 (2505.17923): data grows
  exponentially in hops; a curriculum cuts it but does not remove it.
- **The returned bridge is misaligned with what the next hop needs.** DiscoLoop (Fu et al. 2026, 2607.00341): in a
  looped 2-hop model the first iteration decodes the bridge, but its hidden state is poorly aligned with the bridge
  token embedding; feeding back the decoded embedding plus the state reaches near-perfect accuracy. Biran et al.
  2024 (2406.12775): the second hop starts too late; back-patching a later state into earlier layers fixes 32–66 %.
- **Teacher forcing starves the decision.** Bachmann & Nagarajan 2024 (2403.06963), "Clever Hans" on path-star:
  once the first node is given the rest is copied, so the hard first choice gets little signal. Multi-token
  prediction helps induction at ~30M (Gloeckle et al. 2024, 2404.19737).
- **Written steps, then internalize.** Coconut (Hao et al. 2024, 2412.06769) and stepwise internalization (Deng et
  al. 2024, 2405.14838): train with the steps written, then replace them stage by stage; without the curriculum
  Coconut is no better than no chain of thought. Abbe et al. 2024 (2406.06467): a scratchpad breaks the globality
  barrier that blocks end-to-end learning.
- **RL does not create a missing skill.** Yue et al. 2025 (2504.13837): RL raises pass@1 but not pass@k beyond the
  base model; RL composes f(g(x)) only after f and g are learned (Yuan et al. 2025, 2509.25123). Our base model is
  at the guessing floor on hop 2, so RL now would be a noisier version of the supervised signal we already have.
  It becomes useful later: after the written-path stage, rewarding correct final answers while the written steps
  are removed (credit spread over the loop: RLTT, 2602.10520).
- **Memory networks.** MemN2N needed several hops with tied embeddings (output embedding of hop r = input of hop
  r+1): the returned value lives in the space the next query needs. Our reader has no such tie between what it
  returns (values) and what it looks up with (keys).

## 3. The options, judged

| option | what it tests | evidence for | cost | verdict |
|---|---|---|---|---|
| one-token names | removes the head/tail mismatch and teacher-forcing dilution | every positive multi-hop result above uses atomic entities | 1 run per arm, ~2–6 h | **running** (loop / single read / dense) |
| short names (4, 8 letters) | how much of the mismatch is name length | dilution 1/16 → 1/4; probe: letters 1–3 survive | ~6 h | **running** (loop, k4); dense k4/k8 on Odra (tracking) |
| written path, then the loop takes over | can the memory do two lookups when the bridge is written; can the loop replace the written step | Coconut, Deng, Abbe | ~2 h + ~1.5 h | **running** |
| feed the decoded bridge back into the next round | DiscoLoop's fix: the next query comes from the bridge's *embedding*, aligned with how names are addressed | DiscoLoop 2026 | small code change in the loop (decode → embed → add) | next, if one-token names work but 16-letter names do not |
| whole-name exit loss | round r must name node r completely (predict all letters of n_r from the decision position) | multi-token prediction; probe shows the tail is lost | small: extra exit targets | next, the multi-token analogue of the above |
| more budget (16×, 1M+ examples) | a late phase transition | Kohli, Guo, Yao | 6–20 h per arm | after the mismatch is removed; on the exam that moves first |
| more capacity (8 dense layers, wider KV head) | depth / bandwidth limit | Sanford: 4 layers suffice for 2 hops with single tokens | 8-layer dense running on Odra | reference only; E31's KV head is 64-dim, 1 head |
| RL post-training | sharpen an existing skill | Yue 2025: no new skill from RL | — | **not now**; after the written-path stage |
| flow log (per eval: picked candidate, exits vs node r, grad norm per module) | when a jump happens, whether the writer learns | — | done (`--flow_log`) | on in every diagnostic run |

A first smoke of the flow log already shows the memory writer receiving ~10³× less gradient than the reader layers
at initialisation; whether that persists in the 30M runs is the first thing to read from the logs.

## 4. Decision rules for the running diagnostics (Polonez, `Cache/study/e33_hop2_diag`)

- **One-token names, loop ≫ single read (≥ 50 % picked vs floor 8.7 %):** the loop composes when names are atomic;
  the remaining problem is multi-token names → feed the decoded bridge back / whole-name exits.
- **One-token names, all arms at the floor:** not a name problem; budget (16×) and the deeper dense reference next.
- **Written path passes, path → loop keeps it:** the internalization route works; extend to 3 hops.
- **Written path passes, path → loop collapses:** the loop cannot hold the bridge internally yet; add whole-name exits.
