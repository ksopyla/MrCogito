# Results

Every result file that a spec, run report, master-log row or agenda line cites lives here, in git, so
it can be read on any machine with the servers off. `Cache/` (gitignored) is scratch: the scripts
write there, and checkpoints and raw logs stay there or go to the NAS.

**Rule (since 2026-10-03):** a doc links to a file here or to a NAS path
(`/nas/ml_data/mrcogito/...`), never to a `Cache/` path. Links written before that date may still
point into `Cache/`; they are not rewritten.

| folder | what | how it gets here |
|---|---|---|
| [`capability/`](capability/README.md) | capability suite and length-battery results, compact JSON per (folder, host) | `scripts/pull_capability_results.sh <host> <folder>` (also archives the raw folder to the NAS) — skill `capability-checks` |
| `evaluations/<name>/` | final outputs of a checkpoint evaluation: concept-analysis and generation JSON, benchmark CSVs, lm-eval and long-context JSON, the suite summary, small plots | `scripts/save_eval_results.sh <host> <name> <files…>` — skill `experiment-evaluate` (handoff) |

- `<name>` is the experiment id plus a short run tag (`E22_pilot`, `E16b_ckpt7900`), one folder per
  evaluated run; several checkpoints of one run share it.
- Copy only what the write-up uses; at most 5 MB per file (checkpoints never). Bigger artefacts go to
  the NAS and are cited by their NAS path.
- Raw report folders from every server checkout (no checkpoints) are archived with
  `bash scripts/archive_reports_to_nas.sh <host>` to `/nas/ml_data/mrcogito/results/reports/<host>/`.
- Append-only, like the rest of `2_Experiments_Registry/`: add new files, never edit numbers by hand.
