# E33a read–think–reread loop: why the extra reads never helped (diagnosis, 2026-10-04)

**Verdict.** The loop was never given a first hop to build on. Every model trained on the parallel chains, the
loop arms, the single-read control and the dense model alike, settled on a **link-target guess**: read all links
at once and answer with the commonest first letter among the links' targets. That guess scores 44 / 41 / 39 % for
2 / 3 / 4 hops; the arms scored 41–46 / 38–44 / 35–42 %. Decoding the full answers confirms it: the answers are
spread over all link targets, almost evenly, and land on the asked chain's end only 5–6 % of the time. The memory
read at the answer position spreads evenly over all edges and never looks up the start node, so rounds 2–4 had
nothing to refine. Kill criterion K1 is formally met, but the exam could not have told a working loop from a broken
one. **E33a is untested, not refuted.** No wiring bug was found.

The causes are upstream of the loop and are all small to fix:
1. the exam's guessing floor (44 / 41 / 39 %, and 43.75 % for a chain-end guess that no model found) sits where we
   read "partial reasoning";
2. the curriculum never taught *content addressing by key*: the lookup that every chain run starts from, and that
   is replayed during chain training, holds a single marked fact, so the key is never needed;
3. the capability checks have the same gap: C1 (Address) and C2 (Discriminate) are solvable from block markers
   without reading the key, so nothing before C3 shows that the memory can be addressed by key at all;
4. the spec's 75 % gate was derived from those key-free exams (a "ceiling" of 94–97 %), while the dense model on
   the same exam sat at 44–48 %;
5. budget: about 150 k training rows per stage, below where comparable chain tasks show their phase transition
   even with single-token nodes.

## 1. The plateau is a link-target guess

**What the models answer.** Greedy-decoding the full 8-letter answer of 128 fresh pchain3 rows (local, Apple GPU,
`decode_answers.py` in the session scratchpad) and classifying it:

| model (pchain3, 1k) | asked chain's end | another chain's end | asked chain's middle node | another chain's middle node | a start node | not a node |
|---|---|---|---|---|---|---|
| single read | 5.5 % | 32.0 % | 14.1 % | 43.8 % | 0 % | 4.7 % |
| loop seed 2, 4 rounds | 6.2 % | 23.4 % | 15.6 % | 51.6 % | 0 % | 3.1 % |
| loop seed 2, 3 rounds | 4.7 % | 23.4 % | 16.4 % | 52.3 % | 0 % | 3.1 % |
| loop seed 2, 1 or 2 rounds (exits) | 0 % | 0 % | 0 % | 0 % | 0 % | 100 % |
| uniform over the 12 link targets | 8.3 % | 25.0 % | 16.7 % | 50.0 % | 0 % | 0 % |

The answers are almost exactly a uniform draw over the targets of all 12 links. A start node (never a target) is
never produced, so the model knows the answer is a link target, and nothing more. The early exits' free-running
answers are not nodes at all.

**Why that scores ~40 % on the first letter.** The first letter is an argmax over the mix of the candidates' first
letters, so the commonest first letter among the link targets wins. Simulated over 200 k rows:

| hops | link targets | link-target guess (argmax of the mix) | observed arms (first letter, 1k) | chain-end guess |
|---|---|---|---|---|
| 2 | 8 | **44.1 %** | 41–46 % (dense from lookup weights 44.1 %) | 43.75 % |
| 3 | 12 | **40.7 %** | 38–44 % (dense 38.7 %) | 43.75 % |
| 4 | 16 | **38.7 %** | 35–42 % | 43.75 % |

The link-target guess predicts the level *and* the slow fall with more hops; the chain-end guess (pick one of the 4
pure targets, `1/4 + 3/4 · 1/4`) predicts a flat 43.75 % and is not what the models do, though it is open to them.
Teacher forcing makes the later letters easy once the first narrows the candidates, which is why token accuracy
sits at 0.81–0.85 everywhere. Dense from scratch stayed at 25 % (pchain2 / pchain3): it did not even find the
link-target guess.

The literature reports the same first stage: on in-context chains among distractor chains, transformers first put
"nearly uniform probabilities on all possible end tokens", then jump much later (Guo et al. 2025, arXiv 2502.13913:
~800 steps × batch 512 ≈ 400 k examples, single-token nodes, short context).

## 2. Hop 1 is never learned (the per-round exits)

With progress exits, round 1 is trained to output chain node 1, which is a plain keyed lookup of the start node.
The exit accuracies from the Odra logs (first letter, answer / progress node):

| run | round 1 | round 2 | round 3 | round 4 |
|---|---|---|---|---|
| pchain3 loop s2 | 0.42 / 0.41 | 0.43 / 0.44 | 0.41 / 0.41 | 0.40 / 0.40 |
| pchain3 loop fine-tuned from E31 | 0.38 / 0.40 | 0.39 / 0.45 | 0.38 / 0.38 | 0.38 / 0.38 |
| pchain2 loop s2 | 0.39 / 0.40 | 0.38 / 0.38 | 0.38 / 0.38 | 0.39 / 0.39 |
| pchain4 loop s1 | 0.41 / 0.39 | 0.40 / 0.37 | 0.40 / 0.35 | 0.40 / 0.40 |

Round 1 reaches node 1 no better than guessing among nodes of the same role. The model never did the first hop,
so the flat exits are expected: there was no hop for later rounds to continue.

## 3. What the read looks at (local trace on the trained checkpoints)

`analysis/loop_read_trace.py` records the global read's attention at the position that predicts the first
answer letter, maps each memory entry to the edge its latent read, and sums the attention per edge class
(32 pchain3 rows at 1k; the read has 160 memory entries).

| checkpoint | round | share on the hop-1 edge | hop-2 | hop-3 | other chains (9 edges) | filler | top entry is hop-1 |
|---|---|---|---|---|---|---|---|
| pchain3 loop s2 | 1 | 0.047 | 0.048 | 0.049 | 0.53 | 0.29 | 0 / 32 |
| pchain3 loop s2 | 4 | 0.048 | 0.046 | 0.049 | 0.53 | 0.26 | 0 / 32 |
| pchain3 single read | 1 | 0.030 | 0.031 | 0.026 | 0.26 | 0.63 | 2 / 32 |
| pchain3 loop fine-tuned | 1 → 4 | 0.020 → 0.029 | 0.018 → 0.023 | 0.016 → 0.023 | 0.18 → 0.25 | 0.67 → 0.63 | 0 / 32 |
| **positive control: recall 1 of 16 facts** | 1 | **0.463 on the asked fact** | — | — | 0.39 (15 facts) | 0.06 | **27 / 32** |

The positive control shows the trace works and that the same memory *can* be addressed by key (86 % first letter on
recall16). On the chains, the asked edge gets the same ~5 % as any other edge, and the four rounds read the same
distribution: the query never names the start node.

## 4. Wiring checks (local, tiny model): no bug

- progress targets: exit *r* is trained on node *r*+1, and node *r*+1 is the target of node *r*'s edge;
- a training exit after round *r* equals the forward with `R = r` (max |Δ| = 0);
- every round's marker and the writer receive gradient from the final loss and the exits;
- `R = 1` with a zero marker reproduces the plain E31 stack exactly.

Trained-weight scalars: the residual scale of the two looped layers is 0.947 × 0.996 per round (0.79 after four
rounds), and the per-round markers have norm 0.11–0.25 against a residual of norm ≈ 30, so rounds are barely
labelled. Neither blocks a hop; both are worth watching once hops are learned.

## 5. The curriculum never needed the key

- **Init.** Every chain run started from `len_lookup_*` (or the E33a lookup root), trained on `recall_single`:
  one fact, opened by a `keymark` token. The question's key never has to be matched.
- **Replay.** The no-harm replay during chain training is the same single-fact lookup, so it also never asks for
  the key.
- **Evidence that this matters.** From scratch at 1k, "1 of 8 facts" stays at chance for dense (27 %) and for most
  E31 configurations within 4800 steps. Started from the single-fact lookup, it is learned: dense 98 % in 300
  steps, `e31_li_m1` 95 %, recall16 86 %. Key addressing is learnable, but only from the right stepping stone, and
  in the chain runs nothing rewarded it. The final answer is the chain end, and round 1's read is shared between
  the final loss (which the shortcut satisfies) and the 0.3-weighted progress loss for node 1.

## 6. The capability checks have the same gap

| level | exam | can it be solved without the key? |
|---|---|---|
| C1 Address | `recall_single` (1 fact) | yes: there is one fact, found by its `keymark` |
| C2 Discriminate | `select` (fact + decoys) | yes: the fact block opens with `keymark`, decoys with `decoy` |
| C3 Hold many | `recall` (8 / 16 facts) | **no**: the first exam that needs the key |
| C4 Compose | `chain_ordered` | partly: the chain's edges are the first *k* hop blocks in reading order, so a reader that counts hop markers needs no key. The memory's slots carry no order across windows (`lm_slot_pos=boundary`), so it cannot use that route. Dense 100 % vs memory chance at 2k is therefore not a hop-following comparison |
| C5 Reason | `chain_parallel` | not solvable, but guessable: the link-target guess gives 44 / 41 / 39 % first letter for 2 / 3 / 4 hops, and an unused chain-end guess 43.75 % |

The ladder asks for composition (C4–C5) before it has shown key addressing (C3 is still calibrating), and the C5
curriculum candidate replays the key-free lookup.

## 7. What looping should improve, and what we can say

- **Improved:** nothing measurable. Suite seed 0 ties E31 on 15 of 19 cells. Copy-256 is lower (86 vs 100, one
  seed). The shuffled-chain win (69 vs 59) falls inside E31's own seed spread (59–87).
- **Should improve, not observed yet:** tasks that need two dependent reads — multi-hop chains, "the value of the fact
  whose key is the value of X". These were never tested, because no arm learned the first read.
- **Not expected to improve:** single-fact storage and retrieval (Ouro: loops add manipulation, not storage).

## 8. Small changes (no new architecture)

1. **Curriculum:** single-fact lookup → keyed lookup → 1-hop parallel chain (now legal: `hops=1` with
   `n_chains > 1`) → 2 → 3 → 4 hops. Advance a stage only when its exits pass (e.g. round-1 node accuracy ≥ 90 %).
   Replay a *keyed* exam (recall8 or 1-hop chains), not the single fact. A hop curriculum cut the data needed for
   3 / 4 hops from 10× / 100× to 2× / 5× of the 2-hop budget (Yao et al. 2025, arXiv 2505.17923).
2. **Data:** `chain_overhang` (new, default 0): every chain continues past the asked node, so the answer is not a
   pure target and the chain-end guess scores nothing. The link-target guess remains (the answer is still a link
   target), with a floor that falls as links are added (16 targets: 38.7 %). Default rows are byte-identical to
   today's. Every C5 report should print the guessing floor of its exact exam next to the score.
3. **Supervision:** keep progress exits as the diagnostic scaffold (MemN2N with strong supervision of the
   supporting facts reached 0 % error). At the 2-hop stage, round 1's target and the final target no longer
   compete, because round 1 already does the lookup. Randomised loop counts (Huginn) can come later.
4. **Exams (for the capability-checks owner):** add a keyed C1 (recall with distractors) or mark C1/C2 as
   "locate", not "address". Run C5 with an overhang, or report its floor as 43.75 %. Note C4's counting route.
   Gates on C5 must cite a dense or keyed ceiling measured on C5 itself.
5. **Budget:** expect 10⁵–10⁶ rows per hop stage. Read the exits as the progress signal, not only the final score.

## 9. Local small-model runs

The trace (on the trained 30M checkpoints, CPU) and the wiring checks ran locally in seconds to minutes. The Apple
GPU matches the CPU (max |Δlogit| 1e-3) and the memory changes the answer, so the following learning tests are
valid. On a shrunken E31 (hidden 128–256, 16 latents per 64-token window, 256-token books, `--device mps`), the
model learned nothing within budget: the single-fact lookup stayed at chance after 1500 steps, and so did the
1-hop parallel chain after 3000 steps. At 30M, E31 learns the single-fact lookup from scratch and key addressing
only from that stepping stone, after thousands of steps. So a curriculum test that can say whether the loop learns
hop 2 once hop 1 exists needs the real 30M model on a GPU: about 5 h for 3 stages × 4 arms on Polonez.
