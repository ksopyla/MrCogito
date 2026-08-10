# Master Experiment Log

Central **index** of Concept Encoder runs. Keep this file scannable: intent and criteria live in
experiment specs; deep metrics and interpretation live in run reports. Live focus:
[`agenda.md`](../1_Strategy_and_Plans/agenda.md).

| Need | Where |
|---|---|
| What was the bet / success·kill criteria? | `docs/experiments_specs/<lifecycle>/<ID>.md` |
| What happened on a specific run? | [Training Runs](#training-runs) row → run report |
| What are we doing now? | [`agenda.md`](../1_Strategy_and_Plans/agenda.md) |
| Full metric dump / fair baselines | `run_reports/` |

> **Focus note (2026-08-10) — E17 partially successful; E17b mid-init next.** E17
> per-layer + init 0.01 finished 1B with dead writes but **healthier free-run than E16b**
> (`real`@256 **0.21/0.59** vs **0.04/0.94**). Shared init-0.3 opens mechanism
> (Δbeyond **1.69**) yet free-run stays broken (**0.06/0.90**). Next: **E17b** per-layer +
> `WRITE_GATE_INIT=0.1`. Specs:
> [E17](../experiments_specs/done_success/E17_four_bank_concept_memory.md) ·
> [E17b draft](../experiments_specs/ahead/E17b_per_layer_mid_write_init.md) ·
> [init-0.3 report](run_reports/e16b_shared_init030_1b_20260810.md).

> **Prior focus (2026-07-25) — E16b success.** Shared depth-recurrent concepts on Gemma at
> seq **4096** + long-doc mix + Muon + **1B** tokens cleared the beyond-local causal-use gate
> (offline RankMe **101**, Δshuffle/Δstatic≥1024 **2.47/2.35**, Δone-block≥1024 **0.58**).
> Long-context shared-depth is **one validated path**; other directions (E08, diffusion,
> design-only alternatives) stay open. Spec:
> [E16b](../experiments_specs/done_success/E16b_longctx_muon_1b.md) · [run report](run_reports/e16b_longctx_muon_1b_20260725.md).


> **⚠️ Tier-1 metric protocol note (2026-07-07).** `run_concept_analysis.py` numbers recorded
> before 2026-07-07 (E01–E05 rows using old train-split / seq-512 protocol) are **not
> comparable** with post-upgrade numbers (held-out eval, seq 2048, seeded, Δ ± std). They remain
> internally comparable with each other. Training-time W&B `concept_ablation/*` and STS-B /
> SICK / PAWS / GLUE are unaffected.

---

## Experiment Index

Primary navigation for humans and agents. One row per experiment ID. Resolve a missing path by
searching all lifecycle folders under `docs/experiments_specs/` — never assume `ahead/`.

### Recent closed (2026-06 → 2026-08)

| ID | What | Lifecycle | Key result | Spec · report |
|---|---|---|---|---|
| E17 | 4-bank per-layer concept memory (init 0.01 vs E16b) | done_success (mixed) | gen `real`@256 **0.21/0.59** vs E16b **0.04/0.94** — partial free-run win; writes still dead | [E17](../experiments_specs/done_success/E17_four_bank_concept_memory.md) · [report](run_reports/e17_lowinit_1b_generation_20260810.md) |
| E16b | Long-ctx Muon scale-up of shared-depth workspace | done_success | RankMe 101 · Δ≥1024 **2.47/2.35** — causal-use gate cleared | [E16b](../experiments_specs/done_success/E16b_longctx_muon_1b.md) · [report](run_reports/e16b_longctx_muon_1b_20260725.md) |
| E16a | Shared-depth optimizer A/B at 100M / 2K | done_failed | Both arms failed ≥0.01; Muon best short-ctx (0.0028) | [E16a](../experiments_specs/done_failed/E16a_muon_optimizer_ab.md) |
| E16 | Shared depth-recurrent workspace (50M / 2K) | done_failed | Failed 2K gate; same arch later succeeded as E16b | [E16](../experiments_specs/done_failed/E16_shared_depth_recurrent_concepts.md) |
| E15 | Supervision-calibrated delayed recall | done_failed | 12k labels; block-2 still at chance — protocol kill | [E15](../experiments_specs/done_failed/E15_supervision_calibrated_delayed_recall.md) · [report](run_reports/e15_supervision_calibrated_delayed_recall_20260713.md) |
| E14 | Forced delayed-recall memory | done_failed | Killed at 2M; task not yet learned | [E14](../experiments_specs/done_failed/E14_forced_delayed_recall_memory.md) · [report](run_reports/e14_forced_delayed_recall_gate_20260713.md) |
| E10e | Calibrated concept memory @ 100M | done_failed | CE/RankMe↑ vs E10; Δbeyond still ~0 | [E10e](../experiments_specs/done_failed/E10e_calibrated_memory_100m.md) · [report](run_reports/e10e_calibrated_memory_100m_20260713.md) |
| E10d | 3× concept-memory LR | done_failed | Killed ~25M — update scale not enough | [E10d](../experiments_specs/done_failed/E10d_differential_concept_lr.md) |
| E10c | Nonzero memory gate init 0.01 | done_failed | Killed ~25M — gates alone not enough | [E10c](../experiments_specs/done_failed/E10c_nonzero_memory_gates.md) |
| E10b | Normalized concept read | done_failed | Killed ~25M — read RMSNorm alone not enough | [E10b](../experiments_specs/done_failed/E10b_normalized_concept_read.md) · [report](run_reports/e10b_normalized_concept_read_20260712.md) |
| E10 | Gemma-3-1B backbone concept memory | done_failed | Geometry OK; recurrent mechanism null @ 100M | [E10](../experiments_specs/done_failed/E10_gemma_backbone_concept_memory.md) · [report](run_reports/e10_100m_concept_pilot_20260711.md) |
| E05b | Adam@wd=0.1 confound control | done_success | RankMe 30.9 — **wd innocent**; Muon-specific collapse | [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md) · [report](run_reports/e05b_wd_confound_control_20260711.md) |
| E05 | Windowed decoder as cross-window memory | done_failed | Stable under Adam; Muon collapses bottleneck | [E05](../experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md) · [report](run_reports/e05_muon_long_2ep_collapsed_20260709.md) |
| E04 | Concept-only parallel decoder | done_success | RankMe↑; STS-B 0.532 << E02 | [E04](../experiments_specs/done_success/E04_concept_only_parallel_decoder.md) · [report](run_reports/e04_parallel_decoder_20260620.md) |
| E03 | Frozen-encoder anchor de-collapse | done_success | Anchor beats control relatively; absolute gates unmet @ 0.3ep | [E03](../experiments_specs/done_success/E03_concept_anchor_decollapse.md) · [report](run_reports/e03_control_anchor_off_20260618.md) |
| E02 | Prefix→suffix AR (+ E02-long 5ep) | done_success | STS-B **0.714**; longer training de-collapses | [E02](../experiments_specs/done_success/E02_ar_prefix_suffix.md) · [report](run_reports/e02_long_5epoch_20260618.md) |
| E01 | Concept-conditioned AR decoder | done_failed | AR OK; recon overfits and collapses | [E01](../experiments_specs/done_failed/E01_concept_ar_decoder.md) · [report](run_reports/e01_concept_ar_decoder_20260614.md) |

### Ahead / open

| ID | What | Status | Spec |
|---|---|---|---|
| E08 | Concept-Flow reasoner (encode→reason→decode) | draft | [E08](../experiments_specs/ahead/E08_concept_flow_reasoner.md) |
| E05c | Decoder word-dropout on suffix (anti-bypass) | on hold / unrun | [E05c](../experiments_specs/ahead/E05c_anticollapse_extension.md) |
| E05d | VICReg on concept matrix | on hold / design-only | [E05d](../experiments_specs/ahead/E05d_concept_vicreg.md) |
| E11 | In-sequence memory-token concepts (Design A) | design-only | [E11](../experiments_specs/ahead/E11_memtoken_concept_memory.md) |
| E12 | Per-layer KV-prefix concepts (Design B) | design-only | [E12](../experiments_specs/ahead/E12_perlayer_kv_prefix_concepts.md) |
| E13 | Layer-wise recurrent KV-memory | draft (gated on E12) | [E13](../experiments_specs/ahead/E13_layerwise_recurrent_kv_memory.md) |
| E17b | Per-layer banks + mid write init 0.1 | draft (awaiting launch approval) | [E17b](../experiments_specs/ahead/E17b_per_layer_mid_write_init.md) · [plan](../experiments_specs/ahead/E17b_per_layer_mid_write_init_plan.md) |
| E17a | Untied per-bank writers (4 writers) — counterfactual to E17 | draft (conditional on open-gate E17b) | [E17a](../experiments_specs/ahead/E17a_untied_per_bank_writers.md) |

### Canceled (no run)

| ID | What | Spec |
|---|---|---|
| E06 | Latent-space prediction as primary objective | [E06](../experiments_specs/canceled/E06_latent_space_prediction.md) |
| E07 | Sentence-gap / boundary-only infilling | [E07](../experiments_specs/canceled/E07_sentence_gap_infilling.md) |
| E09 | Gated recurrent concept memory (superseded by E10) | [E09](../experiments_specs/canceled/E09_recurrent_concept_memory.md) |

**Genealogy (one variable at a time):** E01 → E02 → E03 → E04 → E05 → E10…E16a (short-ctx) → **E16b** (long-ctx success) → **E17** (per-layer init-0.01 partial success) → **E17b** (per-layer mid-init draft). See [`agenda.md`](../1_Strategy_and_Plans/agenda.md) for the living reading.

---

## Training Runs

Append-only chronological ledger (oldest → newest). **One row per training run.** Details belong in the linked spec/report — keep Verdict to one short sentence.

| Date | Exp | Run ID | Setup | Key metrics | Verdict | Links |
|---|---|---|---|---|---|---|
| 2026-01-17 | — | `weighted_mlm_H512L2C128_20260117_153544` | weighted MLM · H512 L2 C128 · Minipile | MLM 4.09 · MRPC 82.2 · QQP 61.5 | Best L2 MRPC F1. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/weighted_mlm_H512L2C128_20260117_153544) |
| 2026-01-18 | — | `perceiver_mlm_H512L2C128_20260118_172328` | perceiver MLM · H512 L2 C128 · Minipile | MLM 4.01 · MRPC 80.6 · QQP 67.3 | Canonical L2 sparse-MLM baseline. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/glue-mrpc-perceiver-mlm-h512l2c128-20260118-172328-36M-20260119_2026) |
| 2026-01-19 | — | `perceiver_posonly_mlm_H512L2C128_20260119_204015` | pos-only MLM · H512 L2 C128 · Minipile | MLM 4.09 · MRPC 81.8 · QQP 69.2 | Position-only queries. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/glue-mrpc-perceiver-posonly-mlm-h512l2c128-20260119-204015-36M-20260204_1943) |
| 2026-02-07 | — | `weighted_mlm_H512L6C128_20260207_174251` | weighted MLM · H512 L6 C128 · Minipile 40ep | MLM 3.42 · MRPC 80.2 · QQP 66.3 | L6 scaling; worse MLM, decent inference. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/weighted_mlm_H512L6C128_20260207_174251) |
| 2026-02-08 | — | `perceiver_posonly_mlm_H512L6C128_20260208_102656` | pos-only MLM · H512 L6 C128 · Minipile 40ep | MLM 2.64 · MRPC 81.0 · QQP 72.3 | L6 scaling. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/perceiver_posonly_mlm_H512L6C128_20260208_102656) |
| 2026-02-08 | — | `perceiver_mlm_H512L6C128_20260208_211633` | perceiver MLM · H512 L6 C128 · Minipile 40ep | MLM 2.54 · rank **5/128** · MRPC 81.3 · STS-B 0.627 | Best L6 GLUE; severe concept collapse. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/perceiver_mlm_H512L6C128_20260208_211633) |
| 2026-02-19 | — | `perceiver_mlm_H512L6C128_20260219_105435` | MLM + combined/kendall_gal · L6 | MLM 4.31 · rank **122/128** · STS-B 0.341 | Collapse fixed; GLUE crashed. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/glue-mrpc-perceiver-mlm-h512l6c128-20260219-105435-61M-20260219_2027) |
| 2026-02-21 | — | `perceiver_mlm_H512L6C128_20260220_184029` | MLM + combined/fixed=0.1 · L6 | MLM 3.57 · rank 15.9/128 · STS-B 0.507 | Failed to fix collapse; abandon combined loss. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/perceiver_mlm_H512L6C128_20260220_184029) |
| 2026-02-21 | — | `diffusion_H512L2C128D2_20260221_195554` | diffusion L2 (self-attn decoder) | diverged (grad explosion ep12) | FAILED — architecture redesign. See CHANGELOG 2026-02-23. | [report](run_reports/diffusion_L2_failure_20260221.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/diffusion_H512L2C128D2_20260221_195554) |
| 2026-02-23 | — | `diffusion_H512L2C128D2_20260223_203349` | diffusion xattn-only · L2 | rank 10.1/128 · STS-B 0.138 · MRPC 80.0 | Diffusion alone ≠ concept semantics. | [report](run_reports/diffusion_L2_eval_20260225.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/diffusion_H512L2C128D2_20260223_203349) |
| 2026-02-26 | — | `diffusion_H512L6C128D2_20260226_155541` | diffusion xattn-only · L6 + ELBO | rank 5.74/128 · STS-B 0.174 | TODO 11 FAILED — depth+ELBO no fix. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/diffusion_H512L6C128D2_20260226_155541) |
| 2026-03-01 | — | `diffusion_H512L6C128D2_20260301_165308` | diffusion L6 + VICReg | rank 5.09/128 | TODO 11b FAILED — close self-recon diffusion. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/diffusion_H512L6C128D2_20260301_165308) |
| 2026-03-04 | — | `prefix_diff_H512L6C128D2_20260304_200437` | prefix→suffix diffusion · L6 | rank 6.19/128 · STS-B 0.337 | TODO 13a FAILED. | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/prefix_diff_H512L6C128D2_20260304_200437) |
| 2026-03-08 | — | `prefix_diffBiXT_T64_H512L6C128D2_20260308_065355` | prefix diffusion BiXT T64 | rank 5.74/128 | TODO 13b FAILED — BiXT/T64 no rescue. | [report](run_reports/prefix_diffusion_bixt_v2_20260308.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/prefix_diffBiXT_T64_H512L6C128D2_20260308_065355) |
| 2026-03-08 | — | `perceiver_denoise_H512L6C128D3_20260308_220324` | perceiver_denoise BiXT · L6 D3 | rank 10.6/128 · STS-B zs 0.607 | TODO 10A MIXED — best early zs semantic signal. | [report](run_reports/perceiver_denoise_reconstruction_20260311.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/perceiver_denoise_H512L6C128D3_20260308_220324) |
| 2026-03-11 | — | `prefix_diffBiXT_T64_H512L6C128D2_20260311_194729` | prefix diffusion · WikiText-103 | rank 3.91/128 · STS-B zs 0.574 | TODO 13c FAILED — stop random-init prefix. | [report](run_reports/prefix_diffusion_wikitext103_probe_20260314.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/prefix_diffBiXT_T64_H512L6C128D2_20260311_194729) |
| 2026-06-13 | E01 | `concept_ar_H768L6C128D4_20260613_185955` | concept_ar recon · H768 L6 C128 D4 · FineWeb 1ep | rank best 14.6→last 4.6 · STS-B 0.556 | MIXED — AR plumbing OK; recon overfits/collapses. | [spec](../experiments_specs/done_failed/E01_concept_ar_decoder.md) · [report](run_reports/e01_concept_ar_decoder_20260614.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_H768L6C128D4_20260613_185955) |
| 2026-06-13 | E02 | `concept_ar_prefix_H768L6C128D4_20260613_134159` | prefix→suffix AR · FineWeb 1ep | rank 11.6/128 · **STS-B 0.702** | MIXED/positive — new zs best; rank still low. | [spec](../experiments_specs/done_success/E02_ar_prefix_suffix.md) · [report](run_reports/e02_ar_prefix_suffix_20260614.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_prefix_H768L6C128D4_20260613_134159) |
| 2026-06-14 | E03 | `concept_ar_H768L6C128D4_20260614_164206` | anchor-ON warmup 0.3ep · recon | RankMe 167 · STS-B 0.556 · Δshuf_early 3.34 | INCONCLUSIVE alone — needs matched control. | [spec](../experiments_specs/done_success/E03_concept_anchor_decollapse.md) · [report](run_reports/e03a_anchor_on_warmup_20260615.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_H768L6C128D4_20260614_164206) |
| 2026-06-14 | E02 | `concept_ar_prefix_H768L6C128D4_20260614_101305` | E02-long prefix→suffix · FineWeb 5ep | rank 16.7 · RankMe 82 · **STS-B 0.714** | POSITIVE — longer prefix→suffix de-collapses. | [spec](../experiments_specs/done_success/E02_ar_prefix_suffix.md) · [report](run_reports/e02_long_5epoch_20260618.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_prefix_H768L6C128D4_20260614_101305) |
| 2026-06-15 | E03 | `concept_ar_H768L6C128D4_20260615_211458` | anchor-OFF control 0.3ep | rank 5.9 · STS-B 0.485 · gap_wd 1.68 | Control collapses; anchor wins relatively. | [spec](../experiments_specs/done_success/E03_concept_anchor_decollapse.md) · [report](run_reports/e03_control_anchor_off_20260618.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_H768L6C128D4_20260615_211458) |
| 2026-06-18 | E04 | `perceiver_denoise_H768L6C128D4_20260618_200645` | parallel Perceiver-IO decoder · FineWeb 1ep | RankMe 108 · STS-B 0.532 | MIXED — geometry↑ vs control; semantics << E02. | [spec](../experiments_specs/done_success/E04_concept_only_parallel_decoder.md) · [report](run_reports/e04_parallel_decoder_20260620.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/perceiver_denoise_H768L6C128D4_20260618_200645) |
| 2026-06-28 | E05 | `concept_ar_prefix_H768L6C128D4_20260627_192407` | windowed K=128 · attempt 2 (diverged) | RankMe was 59.8 then collapsed · Δshuf_beyond 0.35 | DIVERGED step 40k — opt failure, arch sound. | [spec](../experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md) · [report](run_reports/e05_attempt2_diverged_20260628.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_prefix_H768L6C128D4_20260627_192407) |
| 2026-06-30 | E05 | `concept_ar_prefix_H768L6C128D4_20260629_093840` | windowed K=128 · attempt 3 Adam 0.5ep | RankMe 37.7 · STS-B 0.452 · Δshuf_beyond 0.39 | Stage1 pass / Stage2 unmet — semantics weak. | [spec](../experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md) · [report](run_reports/e05_attempt3_completed_20260630.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_prefix_H768L6C128D4_20260629_093840) |
| 2026-07-01 | E05 | `concept_ar_prefix_H768L6C128D4_20260702_031956` | windowed + Muon 0.5ep | eval 2.61 · RankMe 10.6 · STS-B 0.518 | Opt win / concept regression (bypass). | [spec](../experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md) · [report](run_reports/e05_muon_divergence_rootcause_20260701.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_prefix_H768L6C128D4_20260702_031956) |
| 2026-07-04 | E05 | `concept_ar_prefix_H768L6C128D4_20260704_225659` | windowed + Muon 2ep | eval 2.58 · RankMe 5.0 · STS-B 0.062 | REGRESSION — more compute → harder collapse. | [spec](../experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md) · [report](run_reports/e05_muon_long_2ep_collapsed_20260709.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_prefix_H768L6C128D4_20260704_225659) |
| 2026-07-09 | E05b | `concept_ar_prefix_H768L6C128D4_20260709_214837` | Adam@wd=0.1 control 0.5ep | RankMe **30.9** · Δshuf_beyond **0.50** | DECISIVE — wd innocent; Muon-specific collapse. | [spec](../experiments_specs/done_success/E05b_wd_confound_control.md) · [report](run_reports/e05b_wd_confound_control_20260711.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/concept_ar_prefix_H768L6C128D4_20260709_214837) |
| 2026-07-11 | E10 | `backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847` | Gemma-3-1B + LoRA · C128/K512 · 100M | RankMe 77 · Δbeyond ≈0 | INCONCLUSIVE/neg mechanism — geometry OK, no persistence. | [spec](../experiments_specs/done_failed/E10_gemma_backbone_concept_memory.md) · [report](run_reports/e10_100m_concept_pilot_20260711.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260711_152847) |
| 2026-07-12 | E10b | `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_133258` | E10 + read RMSNorm · ~25M kill | RankMe 112 · Δbeyond ≈0 | KILLED — read norm alone insufficient. | [spec](../experiments_specs/done_failed/E10b_normalized_concept_read.md) · [report](run_reports/e10b_normalized_concept_read_20260712.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_133258) |
| 2026-07-12 | E10c | `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_153028` | E10b + gate init 0.01 · ~25M kill | RankMe 115 · Δbeyond ≈0 | KILLED — small-live gates alone insufficient. | [spec](../experiments_specs/done_failed/E10c_nonzero_memory_gates.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_153028) |
| 2026-07-12 | E10d | `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_173115` | E10c + 3× concept-mem LR · ~25M kill | RankMe 108 · Δbeyond ≈0.001 | KILLED — update scale not the missing ingredient. | [spec](../experiments_specs/done_failed/E10d_differential_concept_lr.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_173115) |
| 2026-07-12 | E10e | `backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506` | calibrated memory · 100M | RankMe 100 · Δbeyond ≈0.0016 | KILLED — more CE ≠ persistent memory. | [spec](../experiments_specs/done_failed/E10e_calibrated_memory_100m.md) · [report](run_reports/e10e_calibrated_memory_100m_20260713.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260712_215506) |
| 2026-07-13 | E14 | `backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219` | forced delayed recall · synthetic | acc ~chance · margins <0.01 | KILLED at 2M — task not learned; arch inconclusive. | [spec](../experiments_specs/done_failed/E14_forced_delayed_recall_memory.md) · [report](run_reports/e14_forced_delayed_recall_gate_20260713.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219) |
| 2026-07-13 | E15 | `backbone_concept_gemma_3_1b_pt_K512_concept_20260713_191759` | E14 continued · 12k labels | block-2 acc 0.98% (need ≥80%) | KILLED protocol — more labels ≠ learnable sparse task. | [spec](../experiments_specs/done_failed/E15_supervision_calibrated_delayed_recall.md) · [report](run_reports/e15_supervision_calibrated_delayed_recall_20260713.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260713_191759) |
| 2026-07-14 | E16 | `backbone_concept_gemma_3_1b_pt_K512_concept_20260714_075403` | shared_depth_recurrent · 50M / 2K | RankMe 62 · min-beyond 0.0005 | FAILED 2K gate — later cleared under E16b long-ctx. | [spec](../experiments_specs/done_failed/E16_shared_depth_recurrent_concepts.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260714_075403) |
| 2026-07-14 | E16a | `backbone_concept_gemma_3_1b_pt_K512_concept_20260714_211016` | shared-depth Adam · 100M / 2K | RankMe 59 · min-beyond ≈0.0009 | FAILED ≥0.01 at 100M/2K. | [spec](../experiments_specs/done_failed/E16a_muon_optimizer_ab.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260714_211016) |
| 2026-07-15 | E16a | `backbone_concept_gemma_3_1b_pt_K512_concept_20260715_034606` | shared-depth Muon · 100M / 2K | RankMe 97 · min-beyond 0.0028 | Best short-ctx but still failed ≥0.01; Muon used in E16b. | [spec](../experiments_specs/done_failed/E16a_muon_optimizer_ab.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260715_034606) |
| 2026-07-18 | E16b | `backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850` | shared-depth Muon · seq **4096** · 1B | RankMe **101** · Δshuf/static≥1024 **2.47/2.35** | SUCCESS — beyond-local causal use cleared (~235× gate). | [spec](../experiments_specs/done_success/E16b_longctx_muon_1b.md) · [report](run_reports/e16b_longctx_muon_1b_20260725.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850) |
| 2026-08-01 | E17 | `backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805` | per_layer_banks · init 0.01 · 100M | writes ≤0.011 · RankMe 123 | 100M report — cold-start confound; continued to 1B. | [spec](../experiments_specs/done_success/E17_four_bank_concept_memory.md) · [report](run_reports/e17_falsified_init_is_the_cause_20260802.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260801_211805) |
| 2026-08-07 | E17 | `backbone_concept_gemma_3_1b_pt_K512_concept_20260807_195730` | per_layer_banks · init 0.01 · 1B | RankMe 98 · Δbeyond 0.004 · gen **0.21/0.59** | MIXED / partial — free-run lift vs E16b; writes dead. | [spec](../experiments_specs/done_success/E17_four_bank_concept_memory.md) · [report](run_reports/e17_lowinit_1b_generation_20260810.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260807_195730) |
| 2026-08-07 | — | `backbone_concept_gemma_3_1b_pt_K512_concept_20260807_090248` | shared-depth · init **0.3** · 1B | Δshuf_beyond **1.69** · gen **0.06/0.90** | Mechanism open; free-run still E16b-broken. | [report](run_reports/e16b_shared_init030_1b_20260810.md) · [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260807_090248) |

---

## Evaluation Experiments (Zero Training Cost)

| Date | Eval | Source | Key scores | Verdict | Links |
|---|---|---|---|---|---|
| 2026-02-22 | ViaDecoder GLUE | `perceiver_mlm_H512L6C128_20260208_211633` | MRPC 82.7 · STS-B 0.650 · QQP 73.4 · MNLI-m 59.8 | ViaDecoder > CLS-Query; collapse remains | [report](run_reports/via_decoder_eval_20260222.md) |
| 2026-03-11 | STS-B zero-shot | `perceiver_denoise_…20260308_220324` ckpt-202000 | Pearson **0.607** | Best early zs semantic signal | [W&B](https://wandb.ai/ksopyla/[REDACTED]/runs/9p3cda65) |
| 2026-03-11 | Partial GLUE | same denoise ckpt | MRPC 78.7 · STS-B FT 0.102 | Mixed; stop broad sweep | — |
| 2026-03-14 | STS-B zero-shot | prefix-diff WikiText final | Pearson 0.574 | Below gate; collapse unrepaired | [report](run_reports/prefix_diffusion_wikitext103_probe_20260314.md) |
| 2026-06-14 | STS-B + geometry | E01 best/last | best P 0.556 / last 0.207 | Best early concept use; last collapsed | [E01](../experiments_specs/done_failed/E01_concept_ar_decoder.md) · [report](run_reports/e01_concept_ar_decoder_20260614.md) |
| 2026-06-14 | STS-B + geometry | E02 best/last | **P 0.702** · rank 11.6 | New project zs best | [E02](../experiments_specs/done_success/E02_ar_prefix_suffix.md) · [report](run_reports/e02_ar_prefix_suffix_20260614.md) |
| 2026-06-15 | STS-B + geometry | E03A anchor-ON | P 0.556 · RankMe 167 | Kill gates pass; needs control | [E03](../experiments_specs/done_success/E03_concept_anchor_decollapse.md) · [report](run_reports/e03a_anchor_on_warmup_20260615.md) |
| 2026-06-18 | STS-B + geometry | E02-long 5ep | **P 0.714** · RankMe 82 | Longer prefix→suffix de-collapses | [E02](../experiments_specs/done_success/E02_ar_prefix_suffix.md) · [report](run_reports/e02_long_5epoch_20260618.md) |
| 2026-06-18 | STS-B + geometry | E03 control | P 0.485 · rank 5.9 | Completes matched pair | [E03](../experiments_specs/done_success/E03_concept_anchor_decollapse.md) · [report](run_reports/e03_control_anchor_off_20260618.md) |
| 2026-06-20 | STS-B + geometry | E04 | P 0.532 · RankMe 108 | Geometry↑; semantics << E02 | [E04](../experiments_specs/done_success/E04_concept_only_parallel_decoder.md) · [report](run_reports/e04_parallel_decoder_20260620.md) |
| 2026-06-20 | Frozen pool probe | E04 | SICK ΔP +0.22 (attn≫mean) | Distributed info partially hidden from mean | [E04](../experiments_specs/done_success/E04_concept_only_parallel_decoder.md) |
| 2026-06-20 | Frozen pool probe | E02-long | SICK ΔP +0.336 (attn≫mean) | Confirms distributed geometry | [E02](../experiments_specs/done_success/E02_ar_prefix_suffix.md) |
| 2026-07-09 | E10 Stage-0 gap G | untrained Gemma-3-1B wrapper | G(2K)=0.284 (≥0.05 gate) | Decisive GO for E10 training | [report](run_reports/e10_stage0_gap_curve_20260709.md) |
| 2026-07-11 | E10 Stage-0 protocol audit | protocol only | prior G not train-disjoint | Recompute before launch | — |
| 2026-07-25 | E16b Tier-1 | E16b ckpt-7900/7905 | RankMe 101 · Δshuf/static 2.47/2.35 · Δone-block 0.58 | PASS — confirms training trajectory | [E16b](../experiments_specs/done_success/E16b_longctx_muon_1b.md) · [report](run_reports/e16b_longctx_muon_1b_20260725.md) |
| 2026-08-01 | E16b Tier-1.5 gen | E16b ckpt-7900 vs gemma-3-1b-pt | real@256 d1/r3 **0.04/0.94** · base **0.16/0.71** · sample base **0.49/0.03**; zero≫real; long prompt hurts E16b | Free-run FAIL — mechanism intact; chat SFT not the fix | [report](run_reports/e16b_generation_quality_assessment_20260801.md) |
| 2026-08-10 | E17 Tier-1.5 gen | E17 low-init ckpt-7900 vs base / E16b | real@256 **0.21/0.59** · zero **0.21/0.53** · base **0.16/0.71** · E16b **0.04/0.94**; long prompt helps E17 | Partial free-run win vs E16b; absolute bar open; writes dead | [report](run_reports/e17_lowinit_1b_generation_20260810.md) |
| 2026-08-10 | shared init-0.3 Tier-1.5 | `…20260807_090248` ckpt-7905 | Δbeyond **1.69** · real@256 **0.06/0.90** · zero≫real · digit attractors | Mechanism↑ free-run still FAIL on shared | [report](run_reports/e16b_shared_init030_1b_20260810.md) |

**ViaDecoder baselines (L6 canonical, 2026-02-22):** MRPC F1 82.73 · STS-B P 0.650 · QQP F1 73.35 · MNLI-m 59.75 · MNLI-mm 60.90 — full note in [report](run_reports/via_decoder_eval_20260222.md).

---

## Architecture notes (pointers only)

- **2026-02-21 — abandon MLM primary:** diagnosis in [`mlm_perceiver_diagnosis_20260221.md`](../4_Research_Notes/mlm_perceiver_diagnosis_20260221.md); BiXT + TSDAE path followed.
- **2026-02-23 — diffusion decoder redesign:** remove O(N²) self-attn → xattn-only + AdaLN-Zero. Detail in CHANGELOG `[2026-02-23]` and [report](run_reports/diffusion_L2_failure_20260221.md).

---

## Run reports

Newest first:

- [Shared init-0.3 @1B control (Aug 10)](run_reports/e16b_shared_init030_1b_20260810.md)
- [E17 low-init 1B generation vs E16b (Aug 10)](run_reports/e17_lowinit_1b_generation_20260810.md)
- [E17 / init-is-the-cause (Aug 2)](run_reports/e17_falsified_init_is_the_cause_20260802.md)
- [E16b free-run generation vs base Gemma (Aug 1)](run_reports/e16b_generation_quality_assessment_20260801.md)
- [E16b long-context Muon 1B (Jul 25)](run_reports/e16b_longctx_muon_1b_20260725.md)
- [E15 delayed recall (Jul 13)](run_reports/e15_supervision_calibrated_delayed_recall_20260713.md)
- [E14 delayed recall gate (Jul 13)](run_reports/e14_forced_delayed_recall_gate_20260713.md)
- [E10e calibrated memory (Jul 13)](run_reports/e10e_calibrated_memory_100m_20260713.md)
- [E10b normalized read (Jul 12)](run_reports/e10b_normalized_concept_read_20260712.md)
- [E10 100M pilot (Jul 11)](run_reports/e10_100m_concept_pilot_20260711.md)
- [E10 Stage 0 gap curve (Jul 9)](run_reports/e10_stage0_gap_curve_20260709.md)
- [E05b wd confound (Jul 11)](run_reports/e05b_wd_confound_control_20260711.md)
- [E05 Muon 2ep collapse (Jul 9)](run_reports/e05_muon_long_2ep_collapsed_20260709.md)
- [E05 Muon root-cause (Jul 1)](run_reports/e05_muon_divergence_rootcause_20260701.md)
- [E05 attempt 3 (Jun 30)](run_reports/e05_attempt3_completed_20260630.md)
- [E05 attempt 2 (Jun 28)](run_reports/e05_attempt2_diverged_20260628.md)
- [E04 parallel decoder (Jun 20)](run_reports/e04_parallel_decoder_20260620.md)
- [E03 control (Jun 18)](run_reports/e03_control_anchor_off_20260618.md)
- [E02-long 5ep (Jun 18)](run_reports/e02_long_5epoch_20260618.md)
- [E03A anchor-ON (Jun 15)](run_reports/e03a_anchor_on_warmup_20260615.md)
- [E02 prefix→suffix (Jun 14)](run_reports/e02_ar_prefix_suffix_20260614.md)
- [E01 AR decoder (Jun 14)](run_reports/e01_concept_ar_decoder_20260614.md)
- [Prefix diffusion WikiText (Mar 14)](run_reports/prefix_diffusion_wikitext103_probe_20260314.md)
- [Prefix diffusion BiXT (Mar 8)](run_reports/prefix_diffusion_bixt_v2_20260308.md)
- [Perceiver denoise (Mar 11)](run_reports/perceiver_denoise_reconstruction_20260311.md)
- [ViaDecoder eval (Feb 22)](run_reports/via_decoder_eval_20260222.md)
- [Diffusion L2 failure (Feb 21)](run_reports/diffusion_L2_failure_20260221.md)
- [Concept losses (Feb 19)](run_reports/concept_losses_20260219.md)
- Comparative: [L2 vs L6](../3_Evaluations_and_Baselines/comparative_studies/l2_vs_l6_scaling.md) · [canonical baselines](../3_Evaluations_and_Baselines/canonical_baselines.md)
