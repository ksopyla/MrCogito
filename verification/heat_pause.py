"""Polonez heat rule for long capability queues: work in bursts with a rest after each one, every process of a
launch on ONE shared clock, so all its cards rest at the same time (power cuts under long multi-GPU load; rule:
bursts under 10 h, a 20 min cooldown after each 5–6 h).

A probe job can run longer than one burst (a 19.2k-step stage of the catalogue read takes ~7.5 h on a 3090, twice
that if it extends), so the pause sits inside the loops: the probe calls `heat_pause()` before every training step,
the length ladder before every batch of rows. Inside a rest window the call sleeps to the window's end (GPU idle,
weights and optimizer kept in memory) and returns the seconds slept, which callers leave out of their timings.
Nothing else changes: the same steps, rows and random draws in the same order.

Off unless all three are set (scripts/run_study_queue.py --heat BURST_MIN,COOL_MIN writes them into its scripts):
  MC_HEAT_EPOCH      unix time the schedule starts (one value for every GPU script of a launch)
  MC_HEAT_BURST_MIN  minutes of work per burst (330 = 5.5 h)
  MC_HEAT_COOL_MIN   minutes of rest after each burst (20)
Rest windows: [epoch + k·(burst + cool) + burst, epoch + (k + 1)·(burst + cool)), k = 0, 1, …
"""
from __future__ import annotations

import os
import time


def schedule(env=None) -> tuple[float, float, float] | None:
    env = os.environ if env is None else env
    try:
        e, b, c = float(env["MC_HEAT_EPOCH"]), float(env["MC_HEAT_BURST_MIN"]) * 60, float(env["MC_HEAT_COOL_MIN"]) * 60
    except (KeyError, ValueError):
        return None
    return (e, b, c) if b > 0 and c > 0 else None


def rest_left(now: float | None = None, sched: tuple[float, float, float] | None = None) -> float:
    """Seconds to the end of the current rest window (0 while working or when the schedule is off)."""
    s = schedule() if sched is None else sched
    if s is None:
        return 0.0
    e, b, c = s
    now = time.time() if now is None else now
    if now < e:
        return 0.0
    phase = (now - e) % (b + c)
    return b + c - phase if phase >= b else 0.0


def heat_pause(tag: str = "") -> float:
    """Sleep through a rest window if one is open now; returns the seconds slept."""
    left = rest_left()
    if left <= 0:
        return 0.0
    print(f"HEAT-PAUSE {left / 60:.1f} min {tag} from {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    time.sleep(left)
    print(f"HEAT-RESUME {tag} {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    return left


if __name__ == "__main__":  # a queue script can wait out a rest window between jobs
    heat_pause("queue")
