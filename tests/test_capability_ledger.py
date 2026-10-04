"""Capability checks plumbing: the length battery job list, the ledger collector and the board's
job-name parser (process: docs/engineering_specs/capability_checks.md)."""
import json

from analysis.capability_board import parse_job
from analysis.capability_ledger import collect, dumps, study_jobs, suite_rows
from scripts.study_plans import e30_vs_e31 as plan


def test_battery_matches_the_e31b_protocol():
    jobs = plan.battery_jobs("e34x", "core", "e31_li_m1", seeds=(1,), flags=["--x", "1"])
    names = {j["name"] for j in jobs}
    assert len(jobs) == 1 + 2 + 2 + 6 + 2  # root, lookup 8k/16k, chain 2k/8k, 6 hard, 2 recall 8k
    assert {"e34x_lookup_core_s1", "e34x_lookup_core_s1_b16k", "e34x_chain_core_s1_b8k",
            "e34x_hard_decoy8_core_s1", "e34x_hard_recall16_core_s1_b8k"} <= names
    for j in jobs:
        assert j["rows"] == 128 and j["args"][-2:] == ["--x", "1"]
        assert j["init"] is None or j["init"] in names
        assert max(j["ladder"]) == 131072


def test_registered_variants_run_as_phases():
    for tag, kw in plan.BATTERY_VARIANTS.items():
        assert plan.jobs(f"battery_{tag}") == plan.battery_jobs(**kw)
    e33a = plan.jobs("e33a_e31b")  # the battery plus the second seed of the answer-exit reasoning arm
    assert all(j in e33a for j in plan.jobs("battery_e33a"))


def test_parse_job_joins_e31b_and_battery_names():
    assert parse_job("len_lookup_e31_li_m1_s1_b16k") == ("e31_li_m1", "lookup_16k")
    assert parse_job("len_chain_e30_li_s0") == ("e30_li", "chain_2k")
    assert parse_job("hard_recall8_e31_li_s1_b8k") == ("e31_li", "recall8_8k")
    assert parse_job("dense_decoy8_s0") == ("dense", "decoy8_1k")
    assert parse_job("e33a_lookup_loop_s2_b8k") == ("e33a_loop", "lookup_8k")
    assert parse_job("e33a_hard_match3_loop_s1") == ("e33a_loop", "match3_1k")
    assert parse_job("ratio_lookup_e31_K8m1_s0") is None


def _rung(acc):
    return {"results": {"e31_li_m1": {
        "params": 31_000_000, "final": {"acc": acc, "acc_se": 0.01, "step": 100, "per_position_acc": [acc, acc]},
        "info": {"recovered_bits": 60.0, "prize_bits": 64.0}, "throughput": {"sec_per_step": 0.5}}}}


def test_collect_study_and_suite(tmp_path):
    study = tmp_path / "study"
    job = study / "len_lookup_e31_li_m1_s1"
    job.mkdir(parents=True)
    (job / "bridge_1k_recall_single.json").write_text(json.dumps(_rung(0.9)))
    (job / "ladder.json").write_text(json.dumps({"results": {"e31_li_m1": {
        "2048": {"acc": 0.9, "first_acc": 0.95, "by_depth": [1, 2]}}}}))
    (job / "DONE").write_text("")
    (study / "plan_length_odra.json").write_text(json.dumps({"phase": "length", "git": "abc", "jobs": [
        {"name": job.name, "args": ["--seed", "1", "--recipe", "recall_single"], "init": None}]}))
    led = json.loads(dumps(collect(study, "odra")))
    assert led["kind"] == "study" and led["git"] == ["abc"]
    row = study_jobs(led)[0]
    assert row["seed"] == "1" and row["p0"] == 0.9 and row["ladders"]["ladder"]["2048"]["first_acc"] == 0.95
    assert "by_depth" not in row["ladders"]["ladder"]["2048"]

    suite = tmp_path / "suite" / "30m" / "L3.lookup-1k" / "seed0"
    suite.mkdir(parents=True)
    (suite / "job.json").write_text(json.dumps({"suite_version": "v3", "tier": "full", "size": "30m",
                                                "cell": "L3.lookup-1k", "level": 3, "seed": 0, "lr": 1e-4}))
    (suite / "bridge_1k_recall_single.json").write_text(json.dumps(_rung(0.8)))
    (suite / "DONE").write_text("")
    led = collect(tmp_path / "suite", "odra")
    rows = suite_rows(led)
    assert led["kind"] == "suite" and led["suite_version"] == "v3"
    assert rows[0]["cell"] == "L3.lookup-1k" and rows[0]["bits"] == 60.0 and rows[0]["p0"] == 0.8


def test_board_on_the_committed_ledger_never_reruns_flawed_tasks():
    from analysis.capability_board import evidence, reruns, table
    from analysis.capability_ledger import load_ledgers
    from evaluation.capability_tasks import FLAWED

    ev, _ = evidence(load_ledgers())
    items = reruns(table(ev), ["e31_li_m1", "dense"])
    assert items and not any(t in FLAWED for i in items for t in i.get("tasks", [i["task"]]))
    assert {i["kind"] for i in items} <= {"calibrate", "train", "ladder"}
