# Strategy synthesis — the latent channel is the product, and it must be the objective (2026-09-11)

**Written:** 2026-09-11 · dated research note (append-only ledger; later notes may supersede the
*interpretation*, never the record) · author: Cursor cloud-agent session on branch
`cursor/strategy-sota-review-2026-09-e212`, for review by KS.

**Question asked.** How do we reach the Vision — an efficient latent-reasoning model usable for
latent agent-to-agent communication, at 200M–1B parameters with 1M-token context — given
everything E01–E18 taught us and what the frontier shipped in 2025–26? Where is the advantage
nobody else is taking?

**Answer in one paragraph.** Fifteen experiments over eight months say the same thing from
different angles: *a compressed latent channel carries nothing unless the training objective
can be satisfied only through it*. Every time the raw-token path (local window, n-gram tables,
teacher-forced decoder context) could satisfy next-token loss, the concept channel went dead —
regardless of init, gating, optimizer, depth placement, or backbone. The one objective that
de-collapsed concepts with scale (E02-long prefix→suffix) is precisely the one where the decoder
had **no raw access** to the prefix; the one probe that made the E18 global read work perfectly
(P2 copy, 99.9998% at a 16k offset) is precisely the one where labels **demand** long-range bits.
The frontier (Qwen3-Next/3.5, DeepSeek V3.2/V4, GLM-5, Gemma 3) has meanwhile shipped several
of our *mechanisms* — hashed n-gram memory, few-global/many-local layers, KV compression,
sparse indexers, Muon — but all of them still train under plain CE and still expose **text**
as the only interface between models. Nobody trains a model whose native interface is a
compressed latent message. That is the gap: **make the sender→receiver latent message the
pretraining objective**, on the E18 platform, so the concept channel is load-bearing from
step 0, and let long context, latent reasoning, and agent-to-agent communication fall out of
the same trained object. Details, evidence and the concrete bet follow.

---

## 1. What our own ledger says (evidence, with the numbers)

| Experiment (spec) | Objective | Raw-token bypass available? | Did the latent channel carry load? |
|---|---|---|---|
| E01 AR reconstruction | reconstruct input from concepts + teacher-forced tokens | yes (decoder context) | early yes (Δshuffle 1.50 @4k), then rank 14.6→4.6 collapse |
| **E02 / E02-long prefix→suffix** | predict the *suffix* from concepts of the *prefix* | **no** (prefix tokens invisible to decoder) | **yes — de-collapses with scale**: slot rank 5.9→16.7 over 0.3→5 ep, STS-B 0.714, early-position Δzero 1.43 |
| E03 anchor | reconstruction + frozen-teacher MSE anchor | yes | anchor helps relative (RankMe 167 vs 150) but absolute gates unmet |
| E04 parallel Perceiver-IO decoder | reconstruction, no token self-attn | partially removed | RankMe 107.8 but STS-B 0.532 ≪ E02 |
| E05 windowed decoder (K=128) | prefix→suffix with a local window | **yes (window)** | Δshuffle_beyond 0.39→0.23; Muon-long collapsed harder (RankMe 4.96, STS-B 0.062) — *more compute made the bypass better* |
| E10–E10e Gemma + concept memory | plain CE | yes (Gemma's own attention) | <0.002 nats every time; healthy geometry, zero use |
| E14 / E15 forced delayed recall | sparse recall labels (1 answer / 2k tokens) | yes | control at chance — supervision **too sparse** to learn |
| E16 / E16b shared-depth recurrent | plain CE at 2k / 4k | yes | 2k: 0.001 nats; 4k: 2.47 but causality later found leaky; free-run degenerate |
| E17–E17e per-layer banks | plain CE ± carry dropout / starved window | yes, unless dropped | forcing carry-drop gives use (Δperm 0.59) that does **not** transfer to ordinary CE (0.013) and collapses RankMe |
| **E18 Perceiver AR v2** | plain CE, every token, one global read | yes (SWA stack, n-gram tables) | LM value **≈ 0** (arm A 4.090 vs arm C 4.091) even when 62% of context is beyond the stack; but **P2 copy 99.9998%** when labels demand retrieval; dense control shows the 0.035–0.10 nat prize *exists* |

Three readings that are now stable enough to plan on:

1. **Natural-text next-token loss does not supervise long-range addressing at ≤1B.** The dense
   control extracts ~1–2.6% loss from 2k–8k reach *at every layer*; a single read recovers ~⅔ of
   that dependence and *none* of the value (arm C). Perceiver AR (2022) saw the same ("no gain
   past 2k"). This is not a bug in any of our designs; it is the objective. BAPO
   ([lit review](../literature_review/reasoning_bandwidth_information_flow.md)) says the
   same from theory: the model routes through channel `b` (raw tokens) whenever `b` suffices.
2. **Dense labels that can only be satisfied through the channel train it fast.** P2 converged in
   < 500 steps with ~16k supervised tokens per row; E14/E15's one-label-per-2k-tokens never left
   chance. Supervision *density* through the channel, not capacity, is the lever
   (E18b's design premise).
3. **The objective that worked (prefix→suffix) is a two-party communication problem.** E02's
   decoder is a receiver that must reconstruct the suffix from a message about the prefix. It
   de-collapsed with scale; it plateaued at STS-B 0.71 because 128 fixed slots over 512 tokens
   is a weak message and because there was no exact-recall pressure. The receiver framing is
   right; the message and the supervision were too small.

Corollary for measurement (learned the hard way): a *dependence* probe (ablate the channel,
loss goes up) is not a *value* probe (train without the channel, loss is the same). Always keep
the no-channel arm (E18 arm C). RankMe/slot-rank alone say nothing about use.

## 2. What the frontier shipped in 2025–26 and what that means for us

*(Filled from the 2026-09-11 scout sweep; per-model detail and URLs in
[`frontier_open_models_architecture.md`](../literature_review/frontier_open_models_architecture.md),
[`long_context_architectures_training.md`](../literature_review/long_context_architectures_training.md),
[`latent_reasoning_looped_depth.md`](../literature_review/latent_reasoning_looped_depth.md),
[`latent_agent_communication.md`](../literature_review/latent_agent_communication.md),
[`small_lm_training_recipes.md`](../literature_review/small_lm_training_recipes.md).)*

**The frontier converged on our skeleton — and stopped one step short of our thesis.**

- **DeepSeek-V4.1-Flash (2026-09-10)** is a *Causal Encoder-Decoder*: 20 causal encoder layers,
  20 decoder layers, and **the decoder's global KV cache is projected from the encoder's final hidden
  states** instead of from each decoder layer — K/V computed once per sequence and read by the whole
  stack, at **890 bytes per token**, 1M context, SWA caches not persisted at all
  ([card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)). That is E18's "pre-encoder → K/V
  once → read" identity at 552B. Our 1 KB/token number is now table stakes; the *direction* is
  ratified by the strongest possible source.
- **Qwen3.8-Flash-Next** (125B-A6B + **51B hashed n-gram tables** held in host memory; 3 GDN : 1 sparse
  QSA layer; Gated Residual; Muon; 1M) ratifies the other half of E18's input side, and reports two
  ablations we should act on: n-gram vocabulary lowers loss monotonically while downstream accuracy
  *saturates*, and an SWA-hybrid loses to a GDN-hybrid on 7/9 benchmarks
  ([report](https://arxiv.org/html/2608.30320)).
- **GLM-5.3-Flash** (34 KDA + 11 sparse-MLA layers, top-2048 of 1M, mHC, 1M) and **DeepSeek V4**
  (CSA/HCA, Muon, MTP) complete the picture: hybrid linear-attention mixers, few sparse/compressed
  global layers, hashed n-gram capacity, Muon, MTP — all trained under next-token CE and agentic RL,
  all communicating with other models in **text** (`reasoning_effort` scales *tokens*).
- **Long context, small scale:** nobody publishes a from-scratch ≤1B model at 1M with RULER-1M; the
  field's answer to "CE does not train long-range use" is *data* (synthetic retrieval in continued
  pretraining — Nemotron 3 Nano RULER 86.3 @1M; NExtLong; LongFilter, which is our reach probe turned
  into a data filter) and *trainable addressing* (DSA/QSA indexers). Compression to fewer slots is
  bracketed at 4–8× with retention (ICAE, Activation Beacon) and known to lose exact recall unless
  trained for it — E18c's premise.
- **Latent A2A:** LatentMAS (+14.6%, −71–84% tokens, ICML'26 spotlight), C2C (ICLR'26), Interlat
  (ACL'26; ~8-vector compressed messages, utilisation loss), StateBridge, ThoughtComm — every one bolts
  a channel onto a **frozen text-pretrained** model with a projector or aligner; audits show part of
  the gains are generic-prefix steering; every shipping protocol (MCP GA, A2A v1.0, Agents SDK) is
  text. No funded company sells a latent envelope.
- **Latent reasoning:** see §2b below (scout pending at the time of writing; the ledger's own reading
  — Ouro-style pretrained recurrence over post-hoc fine-tuning, task-selective gains, measurement as
  the bottleneck — stands).
- **Recipes:** Muon (NorMuon/Polar Express) + wd + RMS-matched scale ≈ 2× AdamW; data mix and stage
  decay beat every knob; MTP hurts below ~1B; z-loss in cooldown; WSD; NoPE on the read is not free
  (our tiny study and Qwen's ablation agree). A ~600M at 300B–1T tokens should land at HellaSwag
  58–65 / ARC-C 35–48 / MMLU 28–40.

### 2b. Latent reasoning (to be completed from the scout)

<!-- LATENT_REASONING_SECTION -->

## 3. The advantage others are not taking

Put §1 and §2 side by side. The frontier has the *mechanism* we bet on (contextualised prefix →
one projected global K/V → deep local stack, hashed n-gram capacity) and none of the *objective*.
They train that cache under CE, where our ledger says a compressed channel is bypassed the moment
a raw path exists, and they compress it in **bytes** (FP4, cross-layer sharing, top-k access), not in
**count** — so it is still one slot per token, unreadable to any other model, and semantically
unorganised. Meanwhile the latent-communication literature transmits exactly that object post hoc,
through projectors, and cannot prove content transfer.

The cell no one occupies is therefore precise: **a model pretrained so that its global-read cache is
a compressed, content-bearing message that a copy of itself can act on without the source text.**
Three consequences make this our lever rather than a feature:

1. **It fixes our training problem.** It is the only formulation in which CE *itself* supervises the
   concept channel densely at every receiver position — the receiver has no raw path (E02's
   mechanism, at 32k–1M with length-proportional slots and exact-recall pressure). No auxiliary
   anti-collapse loss, no forced carry-dropout, no hoping the read is used.
2. **It is one object with three uses.** The same `prefix_kv()` slots are the 1M-token cache
   (64 B/token at r = 16 → 64 MB at 1M), the state that latent reasoning steps refine (E19 write-back),
   and the agent message (E21's original purpose). Long context, reasoning bandwidth and A2A stop
   being three phases and become three probes on one trained artifact.
3. **It is hard to "scale past."** It is a pretraining objective, not a kernel: a frontier lab would
   have to re-pretrain to obtain a native message space, and their CE-trained caches are, by our
   own arm-C evidence, not messages. And it is auditable with the causal controls the field is asking
   for (no-message / swapped-message), which is the credibility currency for both reviewers and VCs.

Cross-domain grounding (why this should work, not just why it is unoccupied): BAPO's two-party
prefix/suffix model is literally Yao's communication complexity — bound the message, force the
protocol; the corpus callosum — two hemispheres with a bandwidth-limited bridge develop transmissible
codes because neither reads the other's raw input; rate–distortion — the deliverable is the curve of
receiver gain vs bytes/token across r, not one point; and Interlat's utilisation loss / DIAL's
differentiable channel say the receiver must be *trained to listen* — here it is, by CE.

## 4. The concrete bet and its training recipe

**Spec:** [E21 — Latent-message pretraining](../experiments_specs/ahead/E21_latent_message_pretraining.md)
(draft, awaiting go). In one line: on half the rows, place a *message boundary* P that severs every
sliding-window layer and the n-gram context, leave only the global read open across it, and let that
read see the prefix only as one slot per 16 tokens (E18c's compressor); same weights on both sides;
plain CE on every token plus E18b's dense retrieval rows built so that keys lie before P and queries
after. Arms: R (r = 16) and U (r = 1, ceiling). Probes: `none` / `swapped` / `raw` message.
Gates: ≥ 0.30 nats early-suffix gain, ≥ 80% retention vs raw, ≥ 90% recall through the message,
swapped ≤ none, ≤ 0.5% LM cost. Kill: r = 4 also dead → the concept-slot message is falsified here.

**The training recipe for a Perceiver-style latent model, as the ledger now states it:**

| Rule | Evidence | Where it lands |
|---|---|---|
| The channel must be the *only* path for something the loss demands | E02 vs E01/E05; E18 arm C; BAPO | message boundary (E21) |
| Supervise densely through the channel, never one label per 2k tokens | P2 (< 500 steps) vs E14/E15 (chance) | receiver CE at every position + E18b rows |
| Learn addressing on raw K/V first, compress second | P2 → E18b → E18c ladder | warm start E21 from E18b-R; r = 16 → 4 ladder |
| Keep the raw local channel healthy and *separate*; do not starve it | E17e starve, E05 K=128 | SWA stack untouched on the sender side; boundary only on q of rows |
| Always run the no-channel control; dependence ≠ value | reach ablation vs arm C | `none` probe pre-registered; U arm as ceiling |
| Causal content check | StateBridge random prefix; latent-MAS audits | `swapped` probe (S4) |
| Muon with wd 0.1 / adamw 2e-4 / RMS scale; refit LR up; no MTP; z-loss in cooldown | E05 Muon collapse + fix; Moonlight; Qwen3.8-Next | launcher defaults |
| Judge on tasks and message metrics, not CE | E05 Muon-long; Qwen n-gram ablation | RULER/HELMET from slots, bytes/token |

**Sequencing on the platform** (each a registered spec; no calendar): E18b (in flight) → E18c code
(compressor) → **E21 pilot at 125M/32k** → E19 write-back steps on E21's arm R (metric: S1 grows
with refinement steps K; BAPO Thm 8 with a re-readable tape) → 600M main run with the boundary
objective in stage 1 and the E21 probes as M2 → HF demo (reader/answerer on one 3090), paper,
SPRIND / Jean Zay applications ([positioning](../1_Strategy_and_Plans/positioning_and_funding.md)).

**Platform upgrades worth one arm each, not defaults:** a GDN local mixer in place of SWA (Qwen
ablation), a DSA-style indexer in front of the read (the 10M path and the cheapest "trainable
addressing" baseline for E18b), 4 pre-encoder layers (DeepSeek uses 20), NorMuon.

## 5. What we stop doing, what we keep

- **Stop** leading with "1 KB/token cache" or "single global read" as the differentiator — DeepSeek
  ships both. Keep them as platform facts.
- **Stop** treating latent A2A as the last phase. It is the objective (E21) and the demo.
- **Stop** judging concept channels by CE or RankMe alone; every future spec carries a no-channel
  control and a causal (swapped) control.
- **Keep** E18b exactly as specified — its result (does dense retrieval supervision generalise?)
  is E21's warm start and the retrieval-class claim.
- **Keep** E18c's compressor as code; its *run* is subsumed by E21 arm R unless E21 is not approved.
- **Keep** E19 (write-back reasoning) as the next bet after E21, redefined on the message: refinement
  steps that raise the receiver's gain are latent reasoning we can measure.
- **Park** E20 (block diffusion) and the Gemma-backbone line (E10–E17); revive only with a new
  ingredient. Historical evidence stays in the ledger.
- **Defer** the AWS main run until E21's pilot gates, as the AWS one-pager already requires for E18b.
