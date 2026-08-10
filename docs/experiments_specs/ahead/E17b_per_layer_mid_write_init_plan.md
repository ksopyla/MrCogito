# E17b — Implementation plan (per-layer + WRITE_GATE_INIT=0.1)

- **Spec:** [E17b_per_layer_mid_write_init.md](E17b_per_layer_mid_write_init.md) · **Status:** active (launching)
- **HOW only:** config / launcher / eval protocol over the existing E17 foundation.

## Code / config
- Reuse `concept_io_mode=per_layer_banks` (already in `nn/backbone_concept_lm.py`).
- Self-contained launcher `scripts/launch_e17b.sh` sets `EXPERIMENT_ID=E17b` and
  `WRITE_GATE_INIT=0.1`, then execs `launch_e10.sh` (per-layer + Muon + 4k mix).
  Override-friendly: `WRITE_GATE_INIT=0.15 bash scripts/launch_e17b.sh` if we later
  want a slightly hotter prior.
- No nn changes, no new trainer.

## Launch (Polonez)
```bash
ssh polonez
cd /home/ksopyla/dev/<repo>
git fetch origin && git checkout <branch-or-dev> && git pull --ff-only
uv sync
byobu new-session -s E17b
bash scripts/launch_e17b.sh
```

Confirm before start:
- GPUs idle (`nvidia-smi`).
- Manifest present: `$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json`
  (`launch_e17b.sh` points `DATASETS_TOK_DIR` at `datasets_tok_gemma_4k`).
- W&B group/tags pick up `EXPERIMENT_ID=E17b`.

## Eval protocol
**100M = monitor only** (log write gates / Δbeyond / RankMe; optional light gen sniff).
**Do not stop** on flat write gates at 100M — past runs showed that budget is too short
for gates to bang open. Full Tier-1.5 + mechanism verdict at **1B**.

At 100M and 1B:
1. Compute audit once per run id (after finish for the final numbers).
2. Gate telemetry from trainer logs (`concept_gates/write_*`).
3. At 1B (mandatory): matched generation assessment via
   `analysis/run_e16b_generation_assessment.py`.
4. Optional: `run_concept_analysis.py` on pretokenized `e16b_long_4k_v1` manifest.

Compare to:
- E17 `…20260807_195730` (per-layer, init 0.01)
- Shared init-0.3 `…20260807_090248`
- E16b `…20260718_150850`

## Why 0.1 (not 0.3 or 0.05)
- 0.01: empirically dead on both topologies through 1B.
- 0.3: opens shared mechanism but free-run @1B still **0.06/0.90**; one write depth dies.
- 0.1: tanh≈0.1 = E17's own open-gate success floor; log-mid between 0.01 and 0.3.
  Escalate only after a full-budget signal, not from a 100M flat curve.

