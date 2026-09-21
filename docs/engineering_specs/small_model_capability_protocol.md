# Small-model (<10M) capability protocol

- **Type:** engineering foundation (how to train and score DNA/BAPO probes
  when the model is <10M). Not an `E0NN` by itself.
- **Status:** written 2026-09-20. Implements the E25 / exclusive-slot-law
  lessons so E30 (and later writes) are not killed by a false K1 or scored
  on a length that cannot test the claim.
- **Serves:** [E30](../experiments_specs/ahead/E30_sliding_window_perceiver.md),
  any exclusive-channel follow-up. Does **not** replace
  [bapo_capability_ladder.md](bapo_capability_ladder.md) (task set, scores,
  75% dense gate). This file is the **capacity / training-flow** layer on
  top of that ladder.
- **Owner:** Krzysztof Sopyla

The MLM-era note
[`docs/experiment_ideas/concept_encoder_training_protocol_v1.md`](../experiment_ideas/concept_encoder_training_protocol_v1.md)
is **not** this protocol (micro/tiny BERT, MRPC, WikiText). Do not use it
to score <10M concept-memory models.

## Why a separate protocol

At <10M the failure modes are not "the idea is false" vs "the exam leaked":

1. **Late takeoff.** Dense seq=512/1024 sat at chance then jumped
   ([exclusive-slot law](../2_Experiments_Registry/run_reports/exclusive_slot_under10m_law_20260915.md)).
   Killing at 800 steps while CE is still falling is a false K1.
2. **LR is not portable with length.** 3e-3 trains seq=128; 1e-3 trains
   seq=256; 3e-4 trains seq=512; 1e-4 trains seq=1024. 3e-3 floor-kills
   longer geometry.
3. **Width-matched dense is not param-matched.** 4-layer H=256 dense
   missed padded seq=512; 9-layer ~5M dense passed. Score compressed
   models against a **param-matched** dense when the width-matched one
   fails S0.
4. **Composition is an exam kill.** Packed 2–3 hop stayed chance for
   *dense* at this budget. Do not score hops as an architecture fail
   until dense ≥ 75% on that replica.
5. **Geometry must match the claim.** A sliding-window write with
   `W=256` is **one window** at seq=128. Tiny is a training-flow smoke,
   not the sliding claim. Require `n_windows≥2` (and log it) before S1.
6. **Packed answers.** Short y starves the channel (span=8 stayed chance;
   span=32 hit 99%). Use the ladder's pack overrides.

## The four students (always)

| arm | role | typical params |
|---|---|---|
| **dense** | solvability ceiling (S0) | width-matched; param-match if S0 fails |
| **e18** | uncompressed one-global-read ceiling | same H/depth as the compressed arm |
| **e18_local** | leak check (K2) | no global read |
| **compressed** | the bet (`e21` mean slots, `e30` SWP banks, …) | **< 10M** (`--max_params 10000000`) |

Do not quote a compressed score unless S0 and K2 pass.

## Training flow (what to log every eval)

The probe already logs CE, acc, bits, flow. For any *learned write* also log:

| signal | pass / fail |
|---|---|
| `n_windows`, `C`, `N/C` | sliding claim needs `n_windows≥2` |
| mean attention entropy / `log W` | pick if < 0.85; smear if ≈ 1 |
| RankMe of slot K | collapse if ~1 on a live INDEX channel |
| `message_override=none` | must return chance if the channel is load-bearing |
| `message_override=swapped` | should drop vs `real` (content-specific) |
| train CE vs eval CE | still-falling CE at step cap → extend, don't kill |
| `params` | hard fail if > 10M on this protocol |

## Optimiser / schedule

| seq | width | default LR | notes |
|---|---|---|---|
| 128 (`tiny`) | H=128 (~0.6M) | 3e-3 | E24/E25 tiny winner; E30 1e-3 stays chance at 1600 |
| 128 (`tiny`) | H≥384 (~5M+) | 1e-3 | 3e-3 RankMe-collapses E30 (0 bits); 1e-3 recovers INDEX |
| 256 (`tiny_wide`) | H=128 | 1e-3 | exclusive-slot law |
| 256 (`tiny_wide`) | H≥384 | 1e-3 | same length law; do not reuse 3e-3 |
| 512 (`bridge`) | any | 3e-4 | 3e-3 floor-kills |
| ≥1024 | any | 1e-4 | same law |

AdamW, wd=0.01, warmup `min(50, steps/10)`, clip 1.0 (existing probe).
`--warm_residuals` at seq≥512 (E18 `wo` stays shut at 1/S mass).

Param-matched 4-layer width (keep `stack_layers=2`; do not add SWA depth as the 5/10M knob):

| target | hidden | head_dim | dense / e30 |
|---|---|---|---|
| tiny smoke | 128 | 32 | 0.595M / 0.617M |
| ~5M | 384 | 64 | 5.108M / 5.163M (SWP 8 heads, qdim 128) |
| ~9M (<10M cap) | 512 | 64 | 8.969M / 9.041M |
| ~31M (capacity hunt) | 960 | 64 | 31.00M / 31.13M |

H=384 `stack_layers=6` is 10.03M — over the 10M cap. H=960 is **not** this
protocol's 10M hard fail: it is a separate width-matched 4-layer capacity hunt
(`--max_params 40000000`). Spec K3 still says do not widen `H` to rescue a
seq=128 S1 miss; run 30M only when the question is longer seq / harder rungs.

`--steps 800 --k1_mult 4` is the advertised budget. Non-dense arms get
`max(steps × k1_mult, dense_steps_used)` so a dense 99% early-stop cannot
starve a slower write. The probe **does not skip_rest** on a dense arm that is still below 75% **if eval CE has
dropped ≥ 0.2 nats in the last third of the run**, and it applies the same one-shot
extension to **every** arch (E18 / E21 / E30 included). The JSON records
`hunt.k1_extended`. Do not score a still-falling compressed write as a kill.

## Length ladder (when to score what)

| scale | seq | what it can prove | what it cannot |
|---|---|---|---|
| `tiny` | 128 | INDEX/MATCH solvability, training flow, exclusive cut | sliding (`W=256` does not fit; auto-fit K=8 is a *scaled analogue*) |
| `tiny_wide` | 256 | two-window analogue, compression 8× | E21's MATCH wall (that was 512) |
| `bridge` | 512 | E21 mean MATCH wall; `K=32` banks | 1M context |
| `bridge_1k` | 1024 | INDEX non-regression vs E21 53.8 bits | — |

`--evidence_align right` on INDEX at 512+ (spread 4k is K1 until dense
≥ 75%). `local_window < min_gap` always.

## Plots (required)

Reuse `analysis/plot_bapo_capability.py` (learning curves, heatmap,
information flow, recovered bits, bytes/token). Arch colour for `e30`
is required. A write-diagnostic figure (entropy vs `log W`, RankMe,
`n_windows`) belongs next to those plots when the JSON has
`slot_geometry` / `write_geometry`.

## Launch skeleton

Fair **write** comparison is concat on both arms: E21 frozen-mean
(`--message_identity_slots`) vs E30 CA banks. Do **not** pass
`--message_slots_inplace` as the E30 control — inplace is an E21-internal
`KV_LEN=S` flavour E30 cannot match (K slots per window). Width-matched
4-layer is the default scale path (raise `H`, keep `stack_layers=2`);
param-match by adding SWA depth only if that dense misses S0. At H=384
`head_dim=64` (6 Q heads) pin `--swp_n_heads 8 --swp_query_dim 128` so the
write does not gcd-drop; H=128/256/512 divide 128 naturally.

```bash
# smoke (training flow + exclusive cut). Not the MATCH claim.
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe far_copy recall_single \
  --arch dense e18 e18_local e21 e30 \
  --message_identity_slots \
  --hidden 128 --max_params 10000000 --lr 3e-3 --steps 800 \
  --out Cache/small_model_tiny

uv run python analysis/plot_bapo_capability.py \
  --in_dir Cache/small_model_tiny --out_dir Cache/small_model_tiny/plots
```

Claim runs copy the same flags with `--scale tiny_wide --lr 1e-3` then
`--scale bridge --recipe recall_single --evidence_align right --lr 3e-4 --warm_residuals`.

## Capacity hunt (~30M) — outside the 10M cap

Width-matched 4-layer H=960 (~31.1M e30). Same students as the <10M protocol
except skip `e18_local` unless K2 is in doubt. Fair write is still concat
frozen-mean e21 vs concat CA e30. Pin `--swp_n_heads 8 --swp_query_dim 128`.
A 30M hit is **not** the E30 S1 claim unless the rung has `n_windows≥2` at
the E21-mean MATCH wall (seq=512). A 30M miss is **not** a <10M K3.

```bash
uv run python verification/bapo_capability_probe.py \
  --scale tiny --recipe far_copy recall_single \
  --arch dense e18 e21 e30 \
  --message_identity_slots \
  --hidden 960 --head_dim 64 --stack_layers 2 \
  --swp_n_heads 8 --swp_query_dim 128 \
  --max_params 40000000 --lr 1e-3 --steps 800 \
  --out Cache/e30_30m_tiny
```

Then `--scale tiny_wide --lr 1e-3` (INDEX + MATCH; `select_1decoy` only if
dense MATCH ≥ 75%). Bridge MATCH: `--scale bridge --recipe recall_single \
--evidence_align right --lr 3e-4 --warm_residuals`.

## Non-goals

- Not GLUE / STS-B / lighteval (canonical
  [evaluation_protocol.md](../3_Evaluations_and_Baselines/evaluation_protocol.md)
  L2+ stays gated).
- Not a 32k language-model run.
- Not hops/SELECT as a first kill (dense must pass first).
