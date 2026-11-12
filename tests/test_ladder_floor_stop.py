"""The ladder floor stop (evaluation only, capability_checks.md rule 3; author via the E33t session, 2026-11-12) and
the greedy decode that reuses one forward for the letters up to the first wrong one.

Floor stop: a ladder stops reading longer once two lengths in a row, both at or past the training length, score at
or below the task's guessing floor + 5 points; the unread lengths are marked "not read, floor stop" and count as the
floor in the no-harm scoring."""
import json
import sys

import numpy as np
import pytest
import torch

from evaluation.capability_tasks import FLOOR_STOP, LADDER_FLOOR_MARGIN, TASK_BY_ID, ladder_floor, task_of_job
from verification.length_ladder import floor_rule, floor_stop

LADDER = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def test_job_names_map_to_their_task():
    assert task_of_job("v4_C1_edge-1k_e33t_catalogue_n4_s1") == "C1.edge-1k"
    assert task_of_job("v4_C5_pchain2-1k_h2_e33t_catalogue_n4_s0") == "C5.pchain2-1k"
    assert task_of_job("v4_C1_lookup-16k_e31_li_m1_s2") == "C1.lookup-16k"
    assert task_of_job("len_lookup_e33t_catalogue_n4_s0_b16k") == "C1.lookup-16k"
    assert task_of_job("len_lookup_e33t_catalogue_n4_s0_b8k") == "C1.lookup-16k"  # a curriculum stage
    assert task_of_job("cal2_edge_k8_dense_s0") is None and task_of_job("e33a_full_30m") is None
    # the floor is the task's own score's guessing floor: the picked candidate on C1.edge (25 % since the audit)
    assert ladder_floor("C1.edge-1k") == (TASK_BY_ID["C1.edge-1k"].floor, "candidate")
    assert ladder_floor("C1.lookup-16k") == (TASK_BY_ID["C1.lookup-16k"].chance, "first_acc")


def _rungs(scores: dict, key="candidate") -> dict:
    return {str(L): {"acc": 0.5, "first_acc": 0.5, "candidate_mode": "on", key: v} for L, v in scores.items()}


def test_two_rungs_in_a_row_at_the_floor_stop_the_ladder():
    # the E33t edge seed that learned: 86 % at 1k, 7.8 % at 2k, 14.5 % at 4k (floor 25 %)
    res = _rungs({1024: 0.863, 2048: 0.078, 4096: 0.145})
    kw = dict(train_len=1024, floor=0.25, key="candidate")
    assert floor_stop(res, LADDER, 4096, **kw) is None            # only one rung at the floor so far
    assert floor_stop(res, LADDER, 8192, **kw) == [2048, 4096]
    res["8192"] = {"skipped": FLOOR_STOP}
    assert floor_stop(res, LADDER, 16384, **kw) == [4096, 8192]   # an unread rung counts as the floor
    # "at the floor" = score <= floor + 5 points (inclusive); below the floor stops too
    edge = 0.25 + LADDER_FLOOR_MARGIN
    assert floor_stop(_rungs({2048: edge, 4096: 0.0}), LADDER, 8192, **kw) == [2048, 4096]
    assert floor_stop(_rungs({2048: edge + 0.01, 4096: 0.0}), LADDER, 8192, **kw) is None
    # a rung without the task's own score (an OOM, a cap) never counts as the floor
    assert floor_stop({**_rungs({2048: 0.1}), "4096": {"error": "OOM"}}, LADDER, 8192, **kw) is None


def test_short_rungs_never_cut_off_the_training_length():
    """A 16k-trained task's ladder also reads 2k–8k: weak short rungs must not stop the 16k read or anything after."""
    res = _rungs({2048: 0.0, 4096: 0.0, 8192: 0.0}, key="first_acc")
    kw = dict(train_len=16384, floor=0.25, key="first_acc")
    assert floor_stop(res, LADDER, 16384, **kw) is None
    res.update(_rungs({16384: 0.9}, key="first_acc"))
    assert floor_stop(res, LADDER, 32768, **kw) is None
    res.update(_rungs({32768: 0.2, 65536: 0.1}, key="first_acc"))
    assert floor_stop(res, LADDER, 131072, **kw) == [32768, 65536]


def test_the_rule_comes_from_the_folder_name(tmp_path):
    d = tmp_path / "v4_C1_edge-1k_x_s0"
    assert floor_rule(d, "auto") == ("C1.edge-1k", 0.25, "candidate")
    assert floor_rule(d, "off") is None
    assert floor_rule(tmp_path / "some_suite_run", "auto") is None
    assert floor_rule(d, "0.1") == ("C1.edge-1k", 0.1, "candidate")


def test_ladder_marks_unread_rungs_without_loading_a_model(tmp_path, monkeypatch):
    """The stop is decided before a model is built: a resumed ladder whose last two rungs sit at the floor writes the
    marker for every longer length and reads nothing."""
    from verification import length_ladder

    d = tmp_path / "v4_C1_edge-1k_e33t_catalogue_n4_s1"
    d.mkdir()
    torch.save({"data": {"seq_len": 1024}}, d / "e33t_catalogue_n4.pt")
    (d / "ladder.json").write_text(json.dumps({"ckpt": str(d), "lengths": LADDER, "rows": 64, "results": {
        "e33t_catalogue_n4": _rungs({1024: 0.863, 2048: 0.078, 4096: 0.145, 8192: 0.052, 16384: 0.121,
                                     32768: 0.04, 65536: 0.12})}}))
    monkeypatch.setattr(length_ladder, "load", lambda *a, **k: pytest.fail("no model may be built"))
    monkeypatch.setattr(sys, "argv", ["length_ladder.py", "--ckpt", str(d), "--lengths", *map(str, LADDER),
                                      "--rows", "64"])
    assert length_ladder.main() == 0
    res = json.loads((d / "ladder.json").read_text())["results"]["e33t_catalogue_n4"]
    assert res["65536"]["candidate"] == 0.12                       # measured rungs stay measured
    assert res["131072"]["skipped"] == FLOOR_STOP
    assert res["131072"]["floor_stop"] == {"task": "C1.edge-1k", "floor": 0.25, "margin": LADDER_FLOOR_MARGIN,
                                           "score": "candidate", "after": [32768, 65536], "train_len": 1024}


def test_ledger_and_board_count_unread_rungs_as_the_floor(tmp_path):
    from analysis import capability_board as cb
    from analysis.capability_ledger import _ladder as ledger_ladder

    p = tmp_path / "ladder.json"
    p.write_text(json.dumps({"results": {"m": {
        "1024": {"acc": 0.9, "first_acc": 0.9, "candidate": 0.86, "candidate_mode": "on"},
        "2048": {"acc": 0.1, "first_acc": 0.2, "candidate": 0.08, "candidate_mode": "on"},
        "131072": {"skipped": FLOOR_STOP, "floor_stop": {"task": "C1.edge-1k", "floor": 0.25}}}}}))
    lad = ledger_ladder(p)["m"]
    assert lad["131072"]["skipped"] == FLOOR_STOP                   # the ledger keeps the marker
    job = {"ladders": {"ladder": lad}}
    out, kind = cb._ladder("C1.edge-1k", job)
    assert out == {1024: 0.86, 2048: 0.08, 131072: 0.25} and kind == "candidate"  # the floor, not a "mixed" ladder
    assert cb._unread(job) == [131072]

    def rec(model, seed, ladder, unread=()):
        return {"task": "C1.edge-1k", "model": model, "label": "same", "seed": seed, "score": ladder[1024],
                "mean": ladder[1024], "ladder": ladder, "ladder_kind": "candidate", "unread": list(unread),
                "via": "v4 task recipe", "source": "x", "collected": ""}
    champ = {1024: 0.98, 2048: 0.97, 131072: 0.88}
    ev = [rec("e31_li_m1", s, champ) for s in (0, 1, 2)]
    ev += [rec("v", 0, out, [131072]), rec("v", 1, out, [131072]), rec("v", 2, {**out, 131072: 0.9})]
    tbl = cb.table(ev)
    assert tbl["C1.edge-1k"]["v"]["floor_stop"] == {"131072": 2}    # 2 of 3 seeds not read at 128k
    lost = cb.no_harm(tbl, "e31_li_m1", ["v"])["v"]["lost"]
    assert any(x["where"] == "C1.edge-1k @ 128k" for x in lost)      # counted at the floor: lost where the champion passes


# ---- greedy decode: one forward for every letter up to the first wrong one ----

def _slow_greedy(model, ids, labels):
    """The decode before 2026-11-12: one forward per answer letter."""
    cur = ids.clone()
    m = labels != -100
    order = [torch.nonzero(m[r]).flatten() for r in range(m.shape[0])]
    for j in range(int(m.sum(-1).max())):
        logits = model(cur, return_logits=True)
        logits = logits.logits if hasattr(logits, "logits") else logits
        for r, pos in enumerate(order):
            if j < len(pos):
                cur[r, int(pos[j])] = logits[r, int(pos[j]) - 1].argmax(-1)
    return cur


@torch.no_grad()
def test_fast_greedy_decode_gives_the_same_letters():
    from tests.test_capability_catalogue import V, _build, _exam
    from verification.bapo_capability_probe import _answer_exact, _greedy_answers

    torch.manual_seed(0)
    c, ids, labels, q_id = _exam("chain_parallel", rows=4, hops=1, chain_overhang=1, key_len=16)
    model = _build(V, c, q_id).eval()
    kw = dict(amp="off", device=torch.device("cpu"))
    ref = _slow_greedy(model, ids, labels)
    assert torch.equal(_greedy_answers(model, ids, labels, **kw), ref)          # a random model: wrong at letter 0
    # rows whose true answer is the model's own greedy answer, fully (rows 0, 1) or for the first 5 letters (row 2):
    # the fast path decodes them in one forward, or one plus the letters after the first wrong one
    m = labels != -100
    ids2, lab2 = ids.clone(), labels.clone()
    for r in range(3):
        pos = torch.nonzero(m[r]).flatten()
        keep = pos if r < 2 else pos[:5]
        ids2[r, keep] = ref[r, keep]
        lab2[r, keep] = ref[r, keep]
    ref2 = _slow_greedy(model, ids2, lab2)
    assert torch.equal(_greedy_answers(model, ids2, lab2, **kw), ref2)

    def forwards(rows):
        calls, orig = [], model.forward
        model.forward = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        try:
            out = _greedy_answers(model, ids2[rows], lab2[rows], **kw)
        finally:
            model.forward = orig
        assert torch.equal(out, ref2[rows])
        return len(calls)
    k = int(m.sum(-1).max())
    assert forwards([0, 1]) == 1                    # right answers: one forward instead of 16
    assert forwards([2]) <= k - 5                   # right for 5 letters: the first forward covers them (and the 6th)
    assert forwards([3]) == k                       # wrong at the first letter: no saving, same letters
    # reusing the caller's forward (the ladder's teacher-forced read) changes nothing
    first = model(ids2, return_logits=True)
    first = first.logits if hasattr(first, "logits") else first
    assert torch.equal(_greedy_answers(model, ids2, lab2, first_logits=first, **kw), ref2)
    a = _answer_exact(model, [(ids2, lab2)], cfg=c, **kw)
    b = _answer_exact(model, [(ids2, lab2)], cfg=c, first_logits=first, **kw)
    assert a == b and a["exact"] >= 0.5                                          # rows 0 and 1 are right by construction
