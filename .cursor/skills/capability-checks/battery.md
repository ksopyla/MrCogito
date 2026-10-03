# Length battery — the E31b protocol for any variant

Reference for step 2 of the `capability-checks` process (`SKILL.md` next to this file).
The question it answers: does the variant keep finding, recalling and chaining facts as the input
grows from 1k to 128k tokens, and does it keep everything the champion can do?

- **Definition:** `battery_jobs()` + `BATTERY_VARIANTS` + `BATTERY_VERSION` in
  `scripts/study_plans/e30_vs_e31.py` (the E31b study plan; its `BASE` flags are the E31 platform:
  30M, H 960, 4 layers, 128-d token embedding, no n-grams, `--message_raw_window 256`)
- **Queue runner:** `scripts/run_study_queue.py` (per-GPU queues, parent → child curriculum chains
  through `DONE` markers, the probe then `verification/length_ladder.py`)
- **Origin and the champion's numbers:** `docs/experiments_specs/ahead/E31b_e30_vs_e31_limits.md`

## What it runs (per seed; seeds 1 and 2 by default)

| job | trains | then ladders |
|---|---|---|
| `{tag}_lookup_{arm}_s{seed}` | lookup at 2k from scratch (the root) | 2k → 128k |
| `…_b8k`, `…_b16k` | lookup, + 8k stage, + 16k stage (curriculum from the root) | 2k → 128k |
| `{tag}_chain_{arm}_s{seed}` (+ `_b8k`) | in-order 4-hop chain at 2k from the root, + 8k stage | 2k → 128k |
| `{tag}_hard_{exam}_{arm}_s{seed}` | at 1k from the root: recall8, recall16, decoy8, unique, match3, chain8 | 1k → 128k |
| `{tag}_hard_recall{8,16}_{arm}_s{seed}_b8k` | the recall exams, + 8k stage | 2k → 128k |

128 rows per ladder length; every ladder records first-letter accuracy (`first_acc`). Roughly
2–3 GPU-days per seed on Odra. Each job is resumable (`DONE` / `FAILED` markers).

## Running it

1. Register the variant (no new job function, ever):
   ```python
   BATTERY_VARIANTS["e34x"] = dict(tag="e34x", arm="core", arch="e31_li_m1",
                                   flags=["--my_flag", "4"], accum=1, train_root=True)
   ```
   - `tag_arm` must equal the variant's suite arch name (the board joins on it).
   - `l1k` / `l2k` / `accum` change only the micro-batch split when memory needs it (same
     effective batch); anything else is a different exam.
   - `train_root=False` only when an earlier phase already trained `{tag}_lookup_{arm}_s{seed}`.
2. Plan locally (seconds): `uv run python scripts/run_study_queue.py --plan e30_vs_e31 --phase battery_e34x --out Cache/study/e30_vs_e31 --host odra --gpus 0 1 2 --mode print`
3. Commit, push, ff-pull on the server (git only), then generate and start the queues there:
   ```bash
   ssh odra 'cd ~/dev/MrCogito && export PATH="$HOME/.local/bin:$PATH" && \
     uv run python scripts/run_study_queue.py --plan e30_vs_e31 --phase battery_e34x \
       --out Cache/study/e30_vs_e31 --host odra --gpus 0 1 2 --mode scripts && \
     bash Cache/study/e30_vs_e31/launch/odra_battery_e34x_start.sh'
   ```
   Queues open in the byobu session `study`. Use the study folder that already holds the
   variant's parents when it inits from past weights.
4. Monitor: `uv run python analysis/study_table.py --root Cache/study/e30_vs_e31 --prefix e34x_ --first`
   on the server (one row per job: accuracy, first letter, ladder per length). A job that extends
   once while eval loss falls is normal.

## Reading it

- Read first-letter accuracy; the mean over letters overstates multi-candidate exams.
- The board draws one chart per exam and stage, one line per variant, the champion first.
- Typical champion picture (`e31_li_m1`, E31b): lookup ≥ 94 % to 128k after the 16k stage;
  decoys 89 % at 128k; recall holds to 16k–32k then fades; the 4-hop chain does not transfer
  past its training length without the order code; parallel chains sit at chance.
