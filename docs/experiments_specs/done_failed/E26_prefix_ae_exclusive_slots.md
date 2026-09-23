# E26 — Weak prefix AE on exclusive r=16 slots (write objective, not extra hop)

- **Status:** ran 2026-09-16 (Polonez GPU 0, DNA MATCH Wave A) · **killed 2026-09-16 — S1 miss** → `done_failed`
- **Serves:** Vision compressed memory that is *content-addressable* at r=16, the
  serving ratio. Queue:
  [e21_improvement_queue.md](../../4_Research_Notes/e21_improvement_queue.md).
- **Implementation plan:** [E26_prefix_ae_exclusive_slots_plan.md](E26_prefix_ae_exclusive_slots_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-16 · closed 2026-09-16

> One coherent bet: the compressor is trained by a **weak reconstruction of each
> 16-token prefix block**, not by answer CE. E25 learned `u`/`delta` under answer CE
> and got a document mean (0 bits). Inplace exclusive geometry stays; this is not
> Beacon end-append vs concat (inplace already won that). Not an extra hop.

## Hypothesis
If exclusive E21 at seq=512 packed `recall_single`, r=16 inplace, is trained with
packed answer CE **plus** a weak linear head that reconstructs each complete sender
block’s token ids from that block’s slot, and the compressor (`u`/`delta`) receives
gradients **only from that AE** (frozen mean at step 0; never answer-CE pooling),
then MATCH recovered bits reach **≥ 0.75 × live E18** (E25 bar ≈ **36 bits**)
**because** Fine-KV / ICAE / Compressive-Transformer attn-recon make slots hold the
block’s identity, which MATCH needs and which answer CE destroyed.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (`KVCompressor`, exclusive inplace mask),
  probe `verification/bapo_capability_probe.py`, factory `evaluation/bapo_models.py`.
  No new training script.
- **Init / checkpoint:** random, H=256 SSMax log, `message_identity_slots` starts as
  frozen mean; AE head random. No SVD / pretrained token init.
- **Baseline to beat:** E25 r=16 frozen-mean MATCH @512 **0 bits**; identity r=1
  **43.08 bits**; live E18 **47.94 bits**. E25 r=16 *learned* pool **0 bits**.
- **Materially new:** prefix-block AE as the **write** loss. Not “train `u` longer”
  on y. Not E23’s far-repeat on perceiver_concept. Not E27’s extra raw keys.

## The architectural bet
```
for each complete sender block j of r=16 tokens:
    slot_j = pool(block_j)     # mean at step 0; u/delta may move under L_AE only
    L_AE  += CE(weak_head(slot_j) → token ids of block_j)
L = L_answer(y) + λ L_AE
QUERY severs SWA; global read exclusive slots only (no extra_slot_attends)
```
Weak head = linear (slot → r × vocab), no self-attn decoder (ICAE-style weak recon).
Inplace exclusive stays. **Out of scope:** hybrid anchors (E27); loops (E19);
language LM; 32k.

## Why this is not a safe retread
E25 already ran learned pooling on this exam and **zeroed** the channel. The missing
ingredient is the published repair: an auxiliary that pays for *every prefix token in
the block*, including keys the answer span never names. Surprising if it works: a
*local* 16-token AE, not more global CE, is what makes exclusive slots MATCH-class.

## Success criteria (set BEFORE running)
- **S0:** dense ≥ 75% (no AE needed on dense).
- **S1:** E21+AE MATCH bits ≥ **0.75 × live E18** at seq=512 r=16. `e18_local` chance.
- **S1b (AE actually writes):** after S1, key-token AE accuracy on held-out blocks
  ≥ 80% (the slot contains the key, not a gist). If S1 passes and S1b misses, the
  read found another route — log it, do not celebrate the compressor.
- **S2 (don’t regress INDEX):** `far_copy` seq=1024 r=16 ≥ 0.75 × E18, only if S1 passes.

## Kill criteria (set BEFORE running)
- **K1 / K2:** dense < 75% or `e18_local` leak — same as E25.
- **K3:** AE key-token accuracy ≥ 80% by 8k steps **and** MATCH flow < 0.05 — slots
  hold keys the read cannot address. Stop pooling-as-MATCH; E27/`r=1` owns `b`.
- **K4:** AE key-token accuracy still < 40% at 8k — the mean cannot store MATCH even
  for a linear decoder (count-starved). Do not train `u` longer under y.
- **K5:** MATCH at chance at 8k while identity r=1 still clears 0.75 × E18 — AE did
  not repair r=16. One extra-step to 16k if climbing; then kill.

## Plan
- **Data:** DNA `recall_single` seq=512, E25 recipe.
- **Compute:** Odra 1×3090. **~4 GPU-h** (AE overhead + 8k–16k extra-step).
- **Launch:**
  ```bash
  uv run python verification/bapo_capability_probe.py \
    --scale bridge --seq_len 512 --recipe recall_single \
    --arch dense e18 e21 e18_local \
    --hidden 256 --global_logit_scale log \
    --message_ratio 16 --message_slots_inplace \
    --message_prefix_ae --message_prefix_ae_weight 1.0 \
    --steps 800 --k1_mult 4 \
    --out Cache/Evaluation_reports/e26_prefix_ae_slots
  ```
  (`--message_identity_slots` off so `u`/`delta` can move **under AE only**; plan
  isolates compressor grads.)
- **New foundation code:** reusable `message_prefix_ae` head + compressor-grad
  isolation on `PerceiverARLM` / probe loss. Defaults off; E18 checkpoints load.

## Result
- Run id: `50t48aas` (hunt `e26_prefix_ae_slots`; ignore aborted `reyc0q2p`)
- WandB: [https://wandb.ai/ksopyla/MrCogito/runs/50t48aas](https://wandb.ai/ksopyla/MrCogito/runs/50t48aas) // pragma: allowlist secret
- Run report: [e26_prefix_ae_slots_20260916](../../2_Experiments_Registry/run_reports/e26_prefix_ae_slots_20260916.md)
- Verdict: **killed** — S1 miss: exclusive E21 recovered **1.03 bits** (acc 0.311) vs live E18 **47.32 bits** (gate 0.75× ≈ 35.5). AE key_acc **0.66** (between K4 40% and K3 80%); MATCH flow **0.021**. RankMe **14.6**. S0/K2 pass. INDEX skipped. Prefix AE wrote some keys the exclusive read could not use.
