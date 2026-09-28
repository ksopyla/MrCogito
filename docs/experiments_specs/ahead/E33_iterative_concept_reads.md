# E33 — Iterative concept reads (read → update → read) for multi-hop over the latent memory

- **Status:** draft 2026-09-28.
  - Q-loop implemented: `--message_read_rounds R` (tied global block, slots frozen after round 1,
    zero-init round embedding; tests in `tests/test_latent_memory.py`).
  - First probe queued: Odra GPU 0, study phase `e33`. It runs R ∈ {1, 4} × {e30_li, e31_li_m1},
    seed 1, lookup-2k → pchain2 → pchain3 at 1k, with ladders to 16k.
  - The L-loop is not implemented.
  - **First result (28 Sep, pchain2, 32-letter nodes, seed 1, first letter at 1k):**
    - e30_li: R1 37 %, R4 37 %;
    - e31_li_m1: R1 43 %, R4 41 %.
    - No gain from the loop.
  - Likely confound: each node is a 32-letter key spread over about 5 latents, so re-querying
    needs a 64-bit key rebuilt from a mixed read.
  - Queued `e33b`: the same test with **8-letter nodes** (16-bit prize), where a node fits in 1–2
    latents. If e33b also shows no gain, the kill criterion applies to the Q-loop, and the L-loop
    (reasoning among fetched latents) becomes the next test.
- **Serves:** vision priority 3, "reason in concept space". E31b showed that **one exclusive
  read does not do multi-hop** on the first-letter metric:
  - the 4-hop chain beyond training length: E30-LI solves it by position, E31-LI at ≈ 50 %;
  - parallel chains (the real REACHABILITY test): chance for both memories, even at training
    length. Dense (30M, 4 layers) is also only 39–44 % there.

  Multi-hop needs sequential reads whose queries depend on what the previous read returned.
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-28 · closed —
- **Builds on:**
  - [E31b](E31b_e30_vs_e31_limits.md): memories, first-letter metric, `chain_parallel` exam.
  - [E32](E32_two_level_concept_memory.md): the memory this loop will read.
  - E19 (looped slot refinement, never run).
  - E25 / E21 extra exclusive hops: one extra attend over frozen slots, mixed at 272 / 288 tokens,
    fail at 1024 select.

## Q-loop result (29 Sep, provisional; seed 1, 1k, first letter; chance 25 %)
| exam | e30_li R1 / R4 | e31_li_m1 R1 / R4 |
|---|---|---|
| pchain2, 32-letter nodes | 37 / 37 | 43 / 41 |
| pchain3, 32-letter nodes | 40 / 39 | 37 / 36 |
| pchain2, 8-letter nodes | 38 / 44 | 45 / 44 |
| pchain3, 8-letter nodes | 39 / 38 | (running) |

**The tied query-side loop does not unlock multi-hop.** Five of five completed pairs are within noise.

Likely reasons, in order of how testable they are:
1. **No per-hop supervision.** The answer is only the final node, so the loop gets no signal until all
   hops are right. That credit assignment is too sparse. A per-round auxiliary target (round r predicts
   node r of the chain) would test this directly.
2. **Frozen memory keys.** The slot key for "B as a source" is written without knowing that B will
   be looked up. Re-querying needs a key/query match that the writer was never trained for.
3. **Tied weights with R = 4 from lookup weights.** The loop starts as 4 copies of a lookup read,
   which may be a poor basin.

**Next (the L-loop, and a supervised Q-loop):**
- **L-loop:** the question scores pages and takes the top-k (k = 8). Then R rounds of self-attention
  run *among the fetched latents* (tied, question-conditioned) before the read. Relations are
  composed inside the memory, where E31's page encoder already links facts within a window (Glyph
  chain-512: 100 %; the ord chain transfers).
- **Supervised Q-loop:** the same loop, plus per-round node targets on pchain (an auxiliary CE on a
  small head). This separates "cannot" from "not trained to".
- The kill rule is unchanged, applied after both.

## Hypothesis
Give the reading tokens **R ≥ 3 weight-tied read rounds** over the same memory. Each round
attends to the slots with a query formed from the state the previous round returned, then
applies an FFN. Train with a lookup → pchain2 → pchain3 curriculum. Then pchain3 at 1k reaches
≥ 75 % **first-letter** accuracy, where R = 1 stays at chance (≈ 25–45 %), and the gain holds
to 16k with the length-invariant memory. **Because** a k-hop REACHABILITY answer needs k
dependent lookups (BAPO: bandwidth grows with depth). A tied loop provides that dependence at
the cost of R × one read (linear in N), not a wider model.

## Arms (30M E31b platform, `--message_raw_window 256`, both memories)
| arm | loop | where the "thinking" happens | cost per reading token |
|---|---|---|---|
| R1 (control) | one read (today) | — | 1 × N/6 |
| **Q-loop R3 / R4** | the global read block (attn over slots + FFN) applied R times, weights tied, residual carried; a small per-round embedding | the query state (token side) | R × N/6 |
| **L-loop R4** (E19-shaped) | the question's top-k pages (k = 16) are fetched, then R rounds of self-attention *among the fetched latents*, conditioned on the question, then one read | the latent set (concept side) | top-k scoring + R × (k·K)² |
| Q-loop R4 + ord | Q-loop with order-preserving slot positions (`scaled`, if the E31b `ord` arms pass) | query state, with order | R × N/6 |

Memories: `e31_li_m1` (stateful latents, the lean for reasoning) and `e30_li` (cheap slots).
The Q-loop is read-side only, so the same loop code serves both. The L-loop needs latents;
on E30 slots it runs over pooled K/V vectors, which is itself a test of "stateful latents matter".

## Exams (first-letter accuracy primary; free-running exact match as a check)
- `chain_parallel`, hops 2 / 3 / 4 (3 decoy chains) at 1k; ladder to 32k.
- `chain_ordered` 4 and 8 hops (does the loop replace position?).
- Controls: lookup and recall16 (the loop must not hurt retrieval).
- Dense R1 and dense with 2× depth as ceilings at 1k.

## Protocol
- Start from each memory's lookup-2k weights. Curriculum pchain2 → pchain3 → pchain4 at 1k,
  step 5e-5, 4800 steps each (extend once).
- Seeds 0 and 1 for every headline cell.
- Log per-round attention entropy and which slot each round reads. A working loop reads a
  different chain edge in each round.

## Success / kill
- **S1:** Q-loop R4 pchain3 ≥ 75 % first letter at 1k (2 seeds); R1 < 50 %.
- **S2:** at 8k / 16k the loop keeps ≥ 0.8× its 1k first-letter accuracy.
- **S3:** lookup / recall16 within 2 points of R1.
- **S4 (concept vs token):** the L-loop matches the Q-loop on pchain3 at ≤ 1/4 of its read cost at 32k.
  That is evidence for reasoning in latent space rather than in token space.
- **Kill:** no loop arm beats R1 by ≥ 20 points on pchain3 at 1k after the curriculum and
  2 seeds. That would mean the one-read memory is not the bottleneck; the next bet would be
  answer-side scratchpad / chain-of-thought tokens instead of latent loops.

## Implementation sketch
- `message_read_rounds: int` on the exclusive global layer. Receivers repeat
  `h ← h + Attn(q(h + e_r), slots) ; h ← h + FFN(h)`, tied, R times. Only receiver rows loop;
  sender rows are unchanged. The mask is unchanged: every round reads the same slots.
- `lm_refine_rounds`, `lm_refine_topk` in `nn/latent_memory.py` for the L-loop: question-scored
  page top-k (straight-through), then tied latent self-attention rounds.
- Tests: causality with R > 1; R = 1 identical to today; per-round read diagnostics.
