"""Capability board: honest match labels for suite step sizes, and one run per seed in a median."""
from analysis.capability_board import _one_per_seed, evidence


def _suite_ledger(lr: float, seed: int = 0, ladder: dict | None = None, collected: str = "2026-10-04") -> dict:
    job = {"job_id": f"30m/L3.lookup-2k/seed{seed}", "size": "30m", "cell": "L3.lookup-2k", "level": 3,
           "seed": seed, "lr": lr, "status": "done", "results": {"dense": {"p0": 0.9, "acc": 0.9}}}
    if ladder:
        job["ladders"] = {"ladder": {"dense": {str(L): {"first_acc": v} for L, v in ladder.items()}}}
    return {"kind": "suite", "name": "t", "host": "h", "collected": collected,
            "suite_version": "2026-09-25.v3", "jobs": [job]}


def test_non_default_step_size_is_settings_differ():
    ev, _ = evidence([_suite_ledger(1e-4), _suite_ledger(5e-5)])
    labels = sorted(e["label"] for e in ev)
    assert labels == ["same", "settings-differ"]
    assert any("step 5e-05" in e["via"] for e in ev)


def test_diagnostic_seeds_stay_out_of_the_medians():
    extra = evidence([_suite_ledger(1e-4, seed=3)])[0][0]
    assert _one_per_seed([extra]) == []


def test_seed_rerun_with_ladder_replaces_the_older_run():
    old = evidence([_suite_ledger(1e-4, collected="2026-10-03")])[0][0]
    new = evidence([_suite_ledger(1e-4, ladder={2048: 0.8}, collected="2026-10-04")])[0][0]
    kept = _one_per_seed([old, new])
    assert kept == [new] and kept[0]["ladder"] == {2048: 0.8}
