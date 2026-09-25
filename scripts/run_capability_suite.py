#!/usr/bin/env python
"""Plan and launch the MrCogito capability suite for one or more architectures.

Every job is one call of the shared probe (`verification/bapo_capability_probe.py`) on
one frozen exam cell at one model size and seed; the candidate architecture always trains
next to the controls (default: the dense model = the ceiling on the same replica).

  # what would run (no GPU needed)
  uv run python scripts/run_capability_suite.py --arch e30 --sizes 5m 10m --tier screen

  # write per-GPU launch scripts for Odra (3 GPUs), then start them in Byobu on the server
  uv run python scripts/run_capability_suite.py --arch e30 --sizes 30m --tier standard \\
      --mode scripts --gpus 0 1 2 --host odra --out Cache/capability/e30_standard

  # run sequentially here (CPU/GPU smoke; --budget_scale shrinks every step budget)
  uv run python scripts/run_capability_suite.py --arch e30 --sizes 5m --levels 0 \\
      --mode local --budget_scale 0.02 --out /tmp/capability_smoke

  # then score everything under the output folder
  uv run python analysis/capability_scorecard.py --in_dir Cache/capability/e30_standard

Output layout: <out>/<size>/<cell>/seed<k>[_lr<x>]/ with the probe's rung JSON plus a
`job.json` (suite version, cell, size, seed, lr, command) that the scorecard reads.
Spec: docs/engineering_specs/capability_suite.md
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.bapo_models import ARCHES  # noqa: E402
from evaluation.capability_suite import (  # noqa: E402
    ARCH_FLAGS,
    DEFAULT_CONTROLS,
    EVAL_ROWS,
    SIZES,
    SUITE_VERSION,
    TIERS,
    Cell,
    budget_for,
    cells_for,
    grad_accum_for,
    lr_for,
)

PROBE = "verification/bapo_capability_probe.py"


@dataclass
class Job:
    job_id: str
    size: str
    cell: str
    level: int
    seed: int
    lr: float
    warm_residuals: bool
    lr_measured: bool
    arches: list[str]
    out_dir: str
    cmd: list[str]
    cost: float  # rough relative GPU cost (params × tokens), for balancing GPUs

    def shell(self) -> str:
        return " ".join(shlex.quote(c) for c in self.cmd)


def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def plan(
    arches: list[str],
    sizes: list[str],
    tier: str,
    *,
    out: str,
    levels: tuple[int, ...] | None = None,
    cell_ids: tuple[str, ...] | None = None,
    seeds: int | None = None,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    lr_pair: bool = False,
    budget_scale: float = 1.0,
    eval_rows: int = EVAL_ROWS,
    extra: tuple[str, ...] = (),
) -> list[Job]:
    """Expand (arches × sizes × tier cells × seeds [× lr pair]) into probe jobs."""
    unknown = [a for a in list(arches) + list(controls) if a not in ARCHES]
    if unknown:
        raise SystemExit(
            f"unknown arch {unknown}; register it in evaluation/bapo_models.ARCHES/build_model first"
        )
    job_arches = list(dict.fromkeys(list(controls) + list(arches)))
    n_seeds = seeds if seeds is not None else TIERS[tier].seeds
    jobs: list[Job] = []
    for size_name in sizes:
        size = SIZES[size_name]
        for cell in cells_for(tier, levels, cell_ids):
            base_lr, warm, measured = lr_for(size_name, cell)
            lrs = [base_lr, base_lr / 2] if lr_pair else [base_lr]
            b = budget_for(cell)
            steps = max(20, int(round(b.steps * budget_scale)))
            eval_every = max(5, min(b.eval_every, steps // 4))
            for seed in range(n_seeds):
                for lr in lrs:
                    tag = f"seed{seed}" + (f"_lr{_fmt_lr(lr)}" if lr_pair else "")
                    out_dir = str(Path(out) / size_name / cell.id / tag)
                    cmd = ["uv", "run", "python", PROBE, *cell.probe_args(),
                           "--arch", *job_arches, "--no-skip_uncalibrated",
                           *size.probe_args(),
                           "--lr", f"{lr:g}", *(["--warm_residuals"] if warm else []),
                           "--steps", str(steps), "--k1_mult", str(b.k1_mult),
                           "--batch", str(b.batch), "--grad_accum", str(grad_accum_for(cell)),
                           "--eval_every", str(eval_every),
                           "--eval_rows", str(eval_rows), "--seed", str(seed),
                           "--amp", "auto", "--out", out_dir]
                    for a in job_arches:
                        cmd += list(ARCH_FLAGS.get(a, ()))
                    cmd += list(extra)
                    cost = size.dense_params_m * steps * b.k1_mult * b.batch * cell.seq_len * len(job_arches)
                    jobs.append(Job(
                        job_id=f"{size_name}/{cell.id}/{tag}", size=size_name, cell=cell.id,
                        level=cell.level, seed=seed, lr=lr, warm_residuals=warm, lr_measured=measured,
                        arches=job_arches, out_dir=out_dir, cmd=cmd, cost=float(cost),
                    ))
    return jobs


def assign_gpus(jobs: list[Job], gpus: list[str]) -> dict[str, list[Job]]:
    """Greedy longest-first balancing of estimated cost across GPUs."""
    load = {g: 0.0 for g in gpus}
    out: dict[str, list[Job]] = {g: [] for g in gpus}
    for j in sorted(jobs, key=lambda j: -j.cost):
        g = min(load, key=load.get)
        out[g].append(j)
        load[g] += j.cost
    return out


def write_job_meta(job: Job, tier: str) -> None:
    d = Path(job.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"suite_version": SUITE_VERSION, "tier": tier, **asdict(job)}
    (d / "job.json").write_text(json.dumps(meta, indent=2))


def write_scripts(by_gpu: dict[str, list[Job]], *, out: str, repo_dir: str, host: str, tier: str) -> list[Path]:
    launch = Path(out) / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    paths = []
    for g, jobs in by_gpu.items():
        lines = [
            "#!/usr/bin/env bash",
            f"# capability suite {SUITE_VERSION} · tier {tier} · host {host} · GPU {g} · {len(jobs)} jobs",
            "set -uo pipefail  # keep going when one cell fails; each job logs its own exit code",
            f"cd {shlex.quote(repo_dir)}",
            'export PATH="$HOME/.local/bin:$PATH"  # uv is not on PATH in non-interactive shells',
            "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f"export CUDA_VISIBLE_DEVICES={g}",
            f"mkdir -p {shlex.quote(str(launch))}",
            f"exec > >(tee -a {shlex.quote(str(launch / f'{host}_gpu{g}.log'))}) 2>&1  # progress log for monitoring",
        ]
        for j in jobs:
            log = Path(j.out_dir) / "probe.log"
            lines += [
                f'if [ -f {shlex.quote(str(Path(j.out_dir) / "DONE"))} ]; then echo "skip {j.job_id} (done)"; else',
                f'  echo "JOB {j.job_id} $(date \'+%Y-%m-%dT%H:%M:%S\')"',
                f"  mkdir -p {shlex.quote(j.out_dir)}",
                f"  {j.shell()} > {shlex.quote(str(log))} 2>&1",
                # probe exit 2 = results written but the dense ceiling missed 75 % (uncalibrated cell)
                f'  rc=$?; echo "EXIT {j.job_id} $rc"; if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then touch {shlex.quote(str(Path(j.out_dir) / "DONE"))}; fi',
                "fi",
            ]
        p = launch / f"{host}_gpu{g}.sh"
        p.write_text("\n".join(lines) + "\n")
        p.chmod(0o755)
        paths.append(p)
    # one-liner to start every GPU script in its own Byobu window
    starter = launch / f"{host}_start_all.sh"
    session = "capability"
    starter.write_text(
        "#!/usr/bin/env bash\n# start each GPU script in its own window of the byobu session "
        f"'{session}' (created if missing)\nset -euo pipefail\n"
        f"byobu has-session -t {session} 2>/dev/null || byobu new-session -d -s {session}\n"
        + "".join(
            f"byobu new-window -t {session} -n cap_gpu{g} 'bash {shlex.quote(str(p.resolve()))}; exec bash'\n"
            for g, p in zip(by_gpu, paths)
        )
        + f"echo 'started {len(paths)} GPU scripts in byobu session {session} (byobu attach -t {session})'\n"
    )
    starter.chmod(0o755)
    return paths + [starter]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", nargs="+", required=True, help="candidate architecture(s) (evaluation.bapo_models.ARCHES)")
    p.add_argument("--sizes", nargs="+", default=["5m", "10m", "30m", "50m"], choices=list(SIZES))
    p.add_argument("--tier", default="screen", choices=list(TIERS))
    p.add_argument("--levels", nargs="+", type=int, default=None, help="override the tier's levels")
    p.add_argument("--cells", nargs="+", default=None, help="explicit cell ids (e.g. L3.lookup-1k)")
    p.add_argument("--seeds", type=int, default=None, help="override the tier's seed count")
    p.add_argument("--controls", nargs="*", default=list(DEFAULT_CONTROLS),
                   help="arches trained next to the candidate in every job (default: dense = ceiling)")
    p.add_argument("--lr_pair", action="store_true", help="also run lr/2 (step-size cliffs are common)")
    p.add_argument("--budget_scale", type=float, default=1.0, help="shrink/grow every step budget (smoke: 0.02)")
    p.add_argument("--eval_rows", type=int, default=EVAL_ROWS)
    p.add_argument("--extra", default="", help="extra probe flags appended to every job (quoted string)")
    p.add_argument("--out", default="Cache/capability/run")
    p.add_argument("--mode", default="print", choices=("print", "local", "scripts"))
    p.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids for --mode scripts")
    p.add_argument("--host", default="odra", help="label for script names (odra / polonez)")
    p.add_argument("--repo_dir", default=None, help="repo path on the host (default: this checkout)")
    args = p.parse_args()

    jobs = plan(
        args.arch, args.sizes, args.tier, out=args.out,
        levels=tuple(args.levels) if args.levels else None,
        cell_ids=tuple(args.cells) if args.cells else None,
        seeds=args.seeds, controls=tuple(args.controls), lr_pair=args.lr_pair,
        budget_scale=args.budget_scale, eval_rows=args.eval_rows, extra=tuple(shlex.split(args.extra)),
    )
    untested = sorted({(j.size, j.cell) for j in jobs if not j.lr_measured})
    print(f"capability suite {SUITE_VERSION}: {len(jobs)} jobs · tier {args.tier} · "
          f"arches {jobs[0].arches if jobs else []} · sizes {args.sizes}")
    if untested:
        print(f"  note: {len(untested)} (size, cell) pairs use a step size not yet measured in the ledger "
              "— consider --lr_pair on the first run")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "plan.json").write_text(json.dumps(
        {"suite_version": SUITE_VERSION, "tier": args.tier, "jobs": [asdict(j) for j in jobs]}, indent=2))

    if args.mode == "print":
        for j in jobs:
            print(f"[{j.job_id}] {j.shell()}")
        return 0
    for j in jobs:
        write_job_meta(j, args.tier)
    if args.mode == "scripts":
        repo = args.repo_dir or str(Path(__file__).resolve().parents[1])
        by_gpu = assign_gpus(jobs, args.gpus)
        paths = write_scripts(by_gpu, out=args.out, repo_dir=repo, host=args.host, tier=args.tier)
        for g, js in by_gpu.items():
            print(f"  GPU {g}: {len(js)} jobs")
        print("  scripts:\n    " + "\n    ".join(str(x) for x in paths))
        return 0
    failed = []
    for j in jobs:
        print(f"--- {j.job_id}", flush=True)
        log = Path(j.out_dir) / "probe.log"
        with log.open("w") as fh:
            rc = subprocess.call(j.cmd, stdout=fh, stderr=subprocess.STDOUT)
        if rc in (0, 2):  # 2 = results written, dense ceiling < 75 % (uncalibrated here)
            (Path(j.out_dir) / "DONE").touch()
            if rc == 2:
                print("    done; dense model < 75 % on this cell (uncalibrated at this budget)", flush=True)
        else:
            failed.append(j.job_id)
            print(f"    FAILED (exit {rc}); see {log}", flush=True)
    print(f"done: {len(jobs) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
