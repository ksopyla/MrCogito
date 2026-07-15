# E16b — Long-context Muon scale-up of the shared depth-recurrent workspace

- **Status:** approved — user-authorized bold compound scale-up 2026-07-15 (4K + long mix + Muon + 1B)
- **Serves:** test whether the E16 shared concept workspace becomes causally useful when given genuine multi-block documents, Muon optimization, and ~10× more tokens than E16a
- **Implementation plan:** [E16b_longctx_muon_1b_plan.md](E16b_longctx_muon_1b_plan.md)
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-15 · closed —

> This is an intentional multi-factor research bet, not a single-variable A/B.
> E10–E16a cautious calibrations kept geometry healthy but never unlocked persistent
> concept use under short 2K plain CE. E16b changes the operating regime: longer
> sequences, a long-document-heavy mix, Muon, and a 1B-token budget — while keeping
> the E16 architecture fixed.
>
> E16a's pre-registered "do not select Muon for E16b on mechanism grounds" rule is
> **user-overridden**: Muon clearly won CE and RankMe at 100M, and the research
> priority is a novel scale-up rather than another tiny matched control.

## Hypothesis

If the E16 shared depth-recurrent concept workspace is trained with Muon on a
4K long-document mix for 1B non-padding tokens, then
`min(delta_static_beyond, delta_shuffle_beyond)` at positions ≥2048 will reach at
least **0.01 nats**, because multi-block documents create real pressure for a
compact recurrent state that 2K short-context CE did not.

## Builds-on

- **Foundation:** E16 `concept_io_mode="shared_depth_recurrent"` in
  `nn/backbone_concept_lm.py`; Muon path in `nn/muon.py` +
  `training/concept_pretraining_trainer.py`; generic launcher + E10 protocol
  wrapper; recipe-driven `scripts/pretokenize_mix.py`.
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42
  (not a resume from E16a — sequence length and data change).
- **Baselines to beat:**
  - E16a Muon 100M (`…20260715_034606`): min-beyond **0.00284**, RankMe 97,
    eval CE 1.768 at 2K.
  - E16 Adam 50M (`…20260714_075403`): min-beyond **0.00050**.

## The change (compound, authorized)

Relative to E16a Muon:

| Factor | E16a Muon | E16b |
|---|---|---|
| Sequence length | 2048 | **4096** |
| Data mix | `smollm3_inspired_2k_e05` | **`smollm3_inspired_4k_e16b`** |
| Token budget | 100M | **1B** |
| Optimizer | Muon 0.01 / wd 0.1 / adamw_lr 2e-4 | same Muon recipe |
| Architecture | shared_depth_recurrent C=128 K=512 | unchanged |

Held fixed: Gemma-3-1B frozen + LoRA, C=128, K=512, four tied depth writes,
read RMSNorm, gate init 0.01, causal-LM objective, seed 42, effective batch 72
(microbatch calibrated for 4K VRAM).

**Why 4K not 8K first:** 4K already doubles the number of recurrent blocks
(8×512 vs 4×512) and is the largest length likely to fit Odra's 3×24 GB cards
with headroom after calibration. An 8K follow-up is deferred unless 4K
calibration leaves clear VRAM/throughput margin.

## Success criteria (set BEFORE running)

- **Primary causal-use gate:** at 1B tokens and positions ≥2048,
  `min(delta_static_beyond, delta_shuffle_beyond) >= 0.01` nats.
- **Scale evidence:** the same min-beyond at the ~250M and ~500M checkpoints is
  reported; success requires the 1B value, but a monotonic rise across those
  checkpoints is supporting evidence.
- **Geometry:** within-sample RankMe ≥38.4/128 at 1B; centered RankMe reported.
- **Depth utilization:** ≥3/4 write and ≥3/4 read gates have magnitude ≥0.005.
- **Optimization:** no non-finite loss/gradients; eval CE falls vs early
  checkpoints (not required to beat 2K E16a absolute CE, which is a different
  length regime).

## Kill criteria (set BEFORE running)

Only hard safety stops — no early mechanism kill that would abort a bold scale-up:

- Stop on non-finite loss/gradients.
- Stop if within-sample RankMe <19.2 at two consecutive evaluations after
  warmup.
- Stop if eval loss rises for three consecutive evaluations totaling >0.10 nats
  after the first 100M tokens.
- Do not stop solely because beyond-deltas are still small at 100M/250M; those
  are reported as checkpoints, not abort rules.

## Plan

- **Architecture:** E16 shared depth-recurrent; 4K ⇒ 8 concept blocks per sequence.
- **Data:** new recipe `data/mix_recipes/smollm3_inspired_4k_e16b.json`; Gemma
  pretokenize to `datasets_tok_gemma` at `max_seq_length=4096`.
- **Compute:** Odra 3× RTX 3090; calibrate microbatch at 4K before full launch;
  preserve effective batch 72. Expected wall time on the order of several days.
- **Steps:** `TARGET_TOKENS=1000000000`, warmup 500, Muon recipe from E16a,
  `AUTO_INTERVALS=1`, retain ~12 checkpoints.
- **Launch:**
  ```bash
  # After mix is pretokenized and 4K VRAM calibrated:
  bash scripts/launch_e16b.sh
  ```
- **New foundation code:** 4K mix recipe; thin `scripts/launch_e16b.sh` that
  overrides seq length / mix / Muon / 1B budget on top of `launch_e10.sh`;
  launcher contract tests. No model fork.

## Result

<Filled in AFTER, by experiment-track.>
- Run id: —
- WandB: —
- Run report: —
- Verdict: —
