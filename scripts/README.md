# Operational scripts

Training scripts have three distinct roles. Keep these layers separate so an experiment can pin
its protocol without copying the shared launcher.

## Training launchers

| Script | Role | Owns |
|---|---|---|
| `train_concept_pretraining_multigpu.sh` | Generic runner | GPU/DDP setup, shared defaults, optional pretokenization, canonical Python entrypoint, logging, and `accelerate launch` |
| `train_perceiver_denoise_multigpu.sh` | Compatibility wrapper | Delegates old commands to the generic runner; do not add behavior here |
| `launch_e05.sh` | Experiment protocol wrapper | E05 architecture, data, optimizer-arm, stability, and token-budget pins |
| `launch_e10.sh` | Experiment protocol wrapper | E10 backbone/concept-control arms, Gemma data paths, LoRA, and token-budget pins |
| `launch_e10_pipeline.sh` | Orchestration pipeline | Waits for prerequisites, runs the E10 gate and pretokenization, then invokes `launch_e10.sh` |
| `launch_e14.sh` | Experiment protocol wrapper | E10e architecture with sparse-label forced delayed recall and its 2M-token checkpoint gate |
| `launch_e17c.sh` | Experiment protocol wrapper | E17c depth-private gated cell + carry pressure |
| `launch_e17d.sh` | Experiment protocol wrapper | E17d attn-residual global concept layers, no token carry, 300M |
| `calibrate_e17d_batch.sh` | Throughput/VRAM sweep | E17d per-device batch on 4×3090; rank by real tok/s under `length_group`; VRAM is a constraint; restore eff. batch ~72 via accum |
| `test_perceiver_denoise_local.ps1` | Local smoke | Small reconstruction smoke against the canonical Python entrypoint |

Use the generic runner for ad hoc maintained-family training:

```bash
HIDDEN_SIZE=768 CONCEPT_NUM=128 \
  bash scripts/train_concept_pretraining_multigpu.sh
```

Use an experiment wrapper when reproducing or continuing its frozen protocol:

```bash
bash scripts/launch_e05.sh
CONCEPT_NUM=0 bash scripts/launch_e10.sh
bash scripts/launch_e14.sh
```

## Change policy

- Add reusable CLI/environment capabilities only to the generic runner and
  `training/train_concept_pretraining.py`.
- Keep `launch_eNN.sh` wrappers thin: they pin experiment-specific environment values and delegate.
- Keep multi-step prerequisite/evaluation flows in explicit `launch_eNN_pipeline.sh` orchestrators.
- Do not copy the generic launcher for a model variant.
- Preserve compatibility paths for one migration window and test that they generate identical
  training arguments.
- Parked diffusion and prefix-diffusion launchers remain under `parked/scripts/`; they are
  snapshots, not maintained runner templates. Weighted MLM retains only a parked Python trainer,
  and the superseded recursive-MLM fork is preserved in git history.

Evaluation, tokenization, upload, sync, and server-setup scripts are utilities rather than training
launchers. Their command-specific usage lives in each script and the relevant project skill.
