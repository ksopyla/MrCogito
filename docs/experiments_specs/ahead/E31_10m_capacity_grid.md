# E31 — 10M capacity-allocation grid (H vs C vs depth)

- **Status:** spec (Tier B grid; no GPU runs yet)
- **Serves:** the "hidden size vs concept size, how many layers" question inside a
  fixed parameter budget. First controlled H-vs-C data: the E30 width ladder
  (0.6M→50M) pinned SWP geometry, and the coverage sweep varied W at one width.
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-24 · closed —
- **Needs:** Step 1 (`examples_to_criterion`, throughput, pre-probe) and Step 2
  (`lookup_1key`, `lookalike`, `chain_4hop` recipes) — both landed.

## Hypothesis

At a fixed ~9M budget on the exams where E30 already shows signal
(`chain_4hop` + `lookup_1key` @ bridge_1k/1024), concept spend beats hidden
spend: a narrower model with bigger banks (H=384, K=64) reaches 75% on fewer
examples than a wider model with default banks (H=512, K=32), because the
1024-token chain is count-starved (BAPO), not width-starved. Depth (extra SWA
layers at matched params) is the control — expected flat, which would confirm
the bottleneck is write capacity, not stack depth.

## Grid (all cells < 10M, width-matched dense + e30 per cell)

Measured with the probe factory (vocab 24 smoke; DNA vocab shifts all cells
equally — verify per-cell `< 10M` at launch with `--max_params 10000000`).

| cell | H | stack | SWP | dense | e30 | question |
|---|---|---|---|---|---|---|
| A H-wide | 512 | 2 | K=32 cov 8 (default) | 8.98M | 9.05M | baseline: spend on hidden |
| B C-big | 384 | 2 | K=64 cov 8 | 5.11M* | 5.17M* | spend on concepts — needs H raise or K-cost to match A |
| C depth | 416 | 4 | K=32 cov 8 | 8.69M | 8.75M | spend on SWA depth at matched params |
| D heads | 384 | 2 | K=32, 4→8 write heads, qdim 128→256 | 5.11M | 5.17M | write-head expressivity at fixed H |

\* B/D sit at ~5.1M as specced — deliberately UNDER budget: they test whether
a 5M concept-heavy model matches the 9M H-wide one (a stronger claim than
param-matched parity). If B ≈ A on examples-to-75%, concept spend wins outright.
Optional B+ variant: raise H to 512 with K=64 to param-match A (~9M) if B wins.

## Fixed protocol per cell

- Exams: `--recipe chain_4hop lookup_1key --scale bridge_1k` (1024, the E30
  signal length). `lookalike` optional second rung.
- Students: `--arch dense e30` (+ e21 frozen-mean concat as reference if cheap).
- LR: 1e-4 + `--warm_residuals` (bridge_1k law). `--eval_every 100`.
- Pre-probe: `--dense_preprobe_steps 200 --dense_preprobe_min_acc 0.5` — skip
  broken cells before GPU burn (Tier A policy).
- Budget: `--steps 1200 --k1_mult 4` (4800 cap; chain@1024 needed 4800–9600
  at 31M — 9M may need the full cap; do not kill a falling CE).
- SWP overrides per cell: `--swp_bank_size 64` (B), default 32 (A/C/D);
  `--swp_n_heads 8 --swp_query_dim 256` (D).
- Score: `examples_to_criterion` (primary — samples to 75%), recovered bits,
  entropy/logW + RankMe (S2), `none` ablation, toks/sec + peak GB.

## Launch skeleton (one cell; Odra/Polonez, one probe per GPU)

```bash
# Cell A (H-wide baseline)
uv run python verification/bapo_capability_probe.py \
  --scale bridge_1k --recipe chain_4hop lookup_1key \
  --arch dense e30 --message_identity_slots \
  --hidden 512 --head_dim 64 --stack_layers 2 \
  --max_params 10000000 --lr 1e-4 --warm_residuals \
  --steps 1200 --k1_mult 4 --eval_every 100 \
  --dense_preprobe_steps 200 --dense_preprobe_min_acc 0.5 \
  --amp auto --out Cache/e31_cellA
```

Cells B/C/D copy with their H/stack/SWP flags (see table).

## Success / kill (per cell, then across)

- **Cell S0:** dense ≥ 75% on the exam (else the cell is uncalibrated — record,
  do not score e30).
- **Win:** B reaches 75% on fewer examples than A (concept spend > hidden spend).
- **Flat depth:** C ≈ A on examples-to-criterion (depth is not the bottleneck).
- **Kill:** B needs MORE examples than A at both 5M and param-matched 9M —
  record "wider H beats bigger C at 1024" and do not widen K further.
- **Wall policy:** any claimed wall gets one budget-doubled confirmation
  (the 50M lesson) before recording.

## Cost estimate

4 cells × 2 exams × 2 arches ≈ 16 arms at bridge_1k/1200–4800 steps.
At ~0.1–0.3 s/step on a 3090 (E30 31M measured 0.04–0.14 s/step at ≤512;
1024 is heavier): roughly 1–3 GPU-hours per arm → **one overnight wave on
3×3090 Odra**, one probe per GPU.
