# E28 exclusive CogitoProbe-bits @1024 — dense K1, exclusive ~0 bits

**Date:** 2026-09-16 (train + probe); recorded 2026-09-17
**Machine:** Odra 3× RTX 3090
**Run ID:** `perceiver_ar_dense_H256L1g1s2N1024_20260916_185135` (dense `fixed`) · `perceiver_ar_perceiver_H256L1g1s2N1024_20260916_190840` (exclusive `fixed`) · `perceiver_ar_perceiver_H256L1g1s2N1024_20260916_192646` (exclusive `scaled`)
**WandB (training):** [dense](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_dense_H256L1g1s2N1024_20260916_185135) · [excl `fixed`](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_perceiver_H256L1g1s2N1024_20260916_190840) · [excl `scaled`](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_perceiver_H256L1g1s2N1024_20260916_192646) // pragma: allowlist secret
**Raw log:** Odra `Cache/logs/shell_perceiver_denoise_20260916_185120.log` · `_190826.log` · `_192530.log`
**Best checkpoint:** trainer `eval_loss` min at `checkpoint-200` on each arm; probe JSON scored **`checkpoint-1032`** (last)
**Git commit:** dense/`fixed` `5ff01e5` · `scaled` `501896f` (PR 42 launch)
**Git tag:** `arch/e18-perceiver-ar-v2-228-g5ff0` · `…-230-g5018`
**Related:** queue [e21_improvement_queue.md](../../4_Research_Notes/e21_improvement_queue.md) · Wave A [E27](e27_hybrid_key_anchors_20260916.md) / [E26](e26_prefix_ae_slots_20260916.md)

---

## Goal

Test whether exclusive r=16 slots recover unique prefix facts on **`ksopyla/cogito-probe-bits`** at seq=1024 (`fixed` = same 40-bit prize, longer haystack; `scaled` = capacity). Success required dense packed-answer acc ≥ 75% and exclusive recovered bits ≥ 0.75× that dense at **both** 1k and 4k.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` H=256 glob=1 stack=2 kv=1 SSMax `log`, 72.7M |
| Dense | `par_mode=dense`, message knobs off (`boundary=-1`) |
| Exclusive | inplace r=16, `identity_slots=True`, boundary token **48** (`Q`), no prefix AE, no `key_spans` (Wave A had already missed) |
| Dataset | Hub `ksopyla/cogito-probe-bits` seq=1024 · 2,048 train / 128 val · `preserve_precomputed_labels` · packing off |
| Objective | packed-answer CE · 12 epochs · 1,032 steps · batch 24 (3×4×accum 2) · lr 3e-4 · warmup 50 |
| Probe | `evaluation/evaluate_cogito_probe.py` test n=128 · `real`/`none`/`swapped` |
| Compute | dry-run audit (no W&B write-back): dense **0.811 GPU-h** / 0.238 kWh / 0.0254B max-tokens; excl `fixed` **0.818**; excl `scaled` **0.817**. `compute/audit_state=flagged` (`loss_fraction:unknown`; `gpu_hours:summary_vs_ts` 1.4–2.2%). World 3. Do not hand-estimate. |

## Training Outcome

All three arms finished 12 epochs. Last train CE **1.48 / 1.30 / 1.46** vs eval **4.11 / 4.49 / 4.51** (overfit). Throughput ~22k real tok/s. First dense start `…20260916_184906` was SIGTERM'd after a dense+boundary reject; **not** ledger evidence. Seq=4096 **not launched** (1k probe acc ≪ 0.5).

## Concept Health

No RankMe / STS-B (packed CogitoProbe, not LM). Channel ablations on last ckpt:

| arm | real acc | real bits | none bits | swapped bits | `real−none` |
|---|---:|---:|---:|---:|---:|
| dense `fixed` | 0.065 | **0.21** | 0.21 | 0.21 | 0 |
| exclusive `fixed` | 0.021 | **0** | 0 | 0 | 0 |
| exclusive `scaled` | 0.025 | **0.003** | 0.001 | 0.003 | 0.003 |

Prize **40 bits** (8 supervised tokens/row). 32-way chance ≈ 3.1%. Dense **6.5%** is ~2× chance, not 75%. Exclusive `fixed` sits **at or below** chance. Dense `real=none=swapped` is expected (no message channel). Exclusive `fixed` `none` acc 0.027 > real 0.021 — not load-bearing.

## Evaluation

Zero-shot STS-B / GLUE not in scope. JSON: Odra `Cache/Evaluation_reports/E28_s1024_{fixed_dense,fixed_excl,scaled_excl}_bits_*.json`. Uncompressed E18 / `e18_local` **not run** (K2 untested).

## Interpretation

**K1 hits:** dense token-acc **6.5% / 0.21 bits** on a 40-bit prize. The paying exam was not instantiated, so K3's "while dense saturates" clause does not fire. Scoring exclusive anyway is still an S1 miss (0 < 0.75×0.21). This is not mixed: exclusive recovered **0 bits** of unique prefix facts at 1k. 4k/8k/32k stay off. Do not read the train-CE drop as a solved probe.

## Decision

**killed / `done_failed`.** Do not launch 4k `fixed` or 32k. Do not treat `scaled`@1024 (same 40-bit prize) as a capacity result. E19 stays gated.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E28_exclusive_cogitoprobe_bits.md`, `agenda.md`*
