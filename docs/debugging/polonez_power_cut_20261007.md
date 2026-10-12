# Polonez power cut, 2026-10-07 — incident analysis

*Analysed 2026-10-12 after the author powered Polonez back on (08:16 UTC).*

## Verdict

A **hard power cut under full 4-GPU load**, not a software crash. The cause is most likely **heat** (a hardware
protective cut-off in the water loop, board or CPU) or the **power supply tripping** (hot PSU, 3090 power spikes),
and the logs cannot separate the two because nothing records temperature or power over time.

## Timeline (UTC)

| time | event |
|---|---|
| 2026-10-03 16:07 | boot (after coolant top-up); GPU power limit set to 350 W by `nvidia-power-limit.service` |
| 2026-10-07 03:52–09:20 | text-check screen training, back-to-back runs (dense, E31c, E31c loop) |
| 11:23 | text-check evals start on all 4 GPUs (4 models, exams up to 32k tokens) |
| 11:39:50 | last system-journal line: a routine ssh login, nothing unusual |
| 11:42:31 | last heartbeat of the text-check waiter (`waiting for GPU 0 …`, every 2 min) |
| 11:42:46 | last write to `eval_dense/final_id.json.rows.jsonl`; the file ends in ~3 KB of null bytes |
| ≤ 11:44:31 | machine off (the next 2-min heartbeat never came) |
| 2026-10-12 08:16 | powered on by hand (Wake-on-LAN did not wake it) |

At the same time Odra, in the same room (no air conditioning), ran all 3 GPUs at 80–83 °C.

## Evidence

- **No shutdown sequence** in the previous boot's journal (no "System is powering down", no power-off target), no
  kernel panic, no GPU Xid, no machine-check (MCE), no OOM, no thermal message. The journal stops mid-routine.
- **Null bytes at the end of a file being written**: blocks were allocated but their data never reached disk — the
  signature of an instant power loss.
- **The same pattern on every outage since August**: boots ending 2026-08-02 10:33, 08-07 06:58, 09-12 14:13,
  09-29 07:48 and 10-07 ~11:43 UTC all stop without a shutdown, all in daytime, the known ones under heavy load.
- **Load at the cut**: 4× RTX 3090 at the 350 W limit (~1.4 kW) plus a Threadripper 3970X (up to ~280 W) —
  roughly 1.7–1.8 kW sustained, with millisecond 3090 transients well above that.

## What cannot be checked

- No temperature or power history: nothing logs GPU / CPU temperature or power to disk.
- The board's main fan/pump controller is not readable (only the secondary `it8792` chip and the CPU sensor load), so
  the pump and radiator fans are invisible; no BMC / IPMI event log.
- The PSU model and rating are not recorded here.

## Causes, ranked

1. **Overheating → hardware cut-off** (CPU THERMTRIP, board VRM or water-loop protection). Fits the hot shared room,
   the history of liquid-cooling problems and the daytime pattern; these cut-offs leave no log.
2. **PSU over-current / over-temperature trip.** Fits the high sustained draw, 3090 spikes and Wake-on-LAN failing
   afterwards (a latched-off PSU).
3. Software / kernel fault — **ruled out** by the evidence above.

## Recommendations

- **Log temperatures and power** every 30 s to disk (GPU temp/power via `nvidia-smi`, CPU `Tctl`, `sensors`),
  fsync'd, so the next outage shows its last values. *Needs the author's OK (standing job on Polonez).*
- **Lower the GPU power limit to ~300 W** (edit `nvidia-power-limit.service`; sudo): ~5–10 % slower, less heat and
  fewer PSU spikes.
- **Physical check**: pump running, radiator fans, coolant level, PSU rating vs ~1.8 kW.
- **Keep the operating rules**: bursts < 10 h with a **20 min cooldown** after each 5–6 h
  (`scripts/run_bursts.sh --cooldown_min 20`); check Odra < 80 °C before long Polonez runs.
