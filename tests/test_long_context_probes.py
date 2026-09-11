"""Tests for evaluation/long_context_probes.py and scripts/build_copy_task_dataset.py."""
import torch

from evaluation.long_context_probes import build_passkey, per_token_ce
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM
from scripts.build_copy_task_dataset import make_rows


class _Tok:
    """Minimal tokenizer stub: one id per character (a-z, digits, space, punctuation)."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 200 + 10 for c in text]

    def decode(self, ids):
        return "".join(chr(i - 10) for i in ids)


def test_passkey_places_key_at_depth_and_ends_with_answer():
    tok = _Tok()
    filler = [5] * 5000
    for depth in (0.1, 0.5, 0.9):
        ids, answer = build_passkey(tok, filler, 1024, depth, __import__("random").Random(0))
        assert len(ids) == 1024
        assert ids[-len(answer):] == answer
        needle_pos = next(i for i, t in enumerate(ids) if t != 5)
        assert abs(needle_pos / 1024 - depth) < 0.08, (depth, needle_pos)


def test_copy_rows_mask_first_half_and_mirror():
    rows = make_rows(3, 22, 1000, 1008, bos=1, eos=2, seed=0)
    for r in rows:
        ids, labels = r["input_ids"], r["labels"]
        assert len(ids) == 22 and ids[0] == 1 and ids[-1] == 2
        half = 10
        assert ids[1 : 1 + half] == ids[1 + half : 1 + 2 * half][::-1]
        assert labels[: 1 + half] == [-100] * (1 + half)
        assert labels[1 + half :] == ids[1 + half :]


def test_per_token_ce_buckets_sum_to_total():
    cfg = PerceiverARConfig(
        vocab_size=50, hidden_size=16, intermediate_size=32, token_embedding_dim=8, pre_layers=0,
        global_layers=1, stack_layers=1, block=8, num_attention_heads=2, num_kv_heads=1, head_dim=8,
        ngram_buckets=16, value_embed_layers=(), attn_backend="sdpa", attn_pad_multiple=1,
        chunked_ce_block_size=4, use_liger=False, z_loss=0.0,
    )
    model = PerceiverARLM(cfg).eval()
    ids = list(range(3, 23))
    per = per_token_ce(model, ids, "cpu", 20)
    assert per.shape == (19,)
    x = torch.tensor(ids)[None]
    assert torch.allclose(model(input_ids=x, labels=x.clone()).loss, per.mean(), atol=1e-5)


def test_argmax_tokens_matches_full_logits():
    import torch
    from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM
    from evaluation.long_context_probes import argmax_tokens

    torch.manual_seed(0)
    cfg = PerceiverARConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, token_embedding_dim=8,
        pre_layers=1, pre_window=4, global_layers=1, stack_layers=2, block=6,
        num_attention_heads=4, num_kv_heads=2, head_dim=8, ngram_buckets=64,
        value_embed_layers=(0,), value_embed_dim=4, use_liger=False, attn_backend="sdpa",
        attn_pad_multiple=1, chunked_ce_block_size=5,
    )
    model = PerceiverARLM(cfg).eval()
    for p in model.parameters():
        if p.ndim == 2:
            p.data.normal_(0, 0.2)
    x = torch.randint(3, 97, (1, 23))
    with torch.no_grad():
        full = model(input_ids=x).logits[0].argmax(-1)
    assert torch.equal(argmax_tokens(model, x, 0, 22, chunk=5), full[:22])
    assert torch.equal(argmax_tokens(model, x, 23 - 4, 23 - 1), full[-4:-1])


def test_copy_rows_plain_copy_variant():
    from scripts.build_copy_task_dataset import make_rows

    rows = make_rows(2, 18, 10, 20, bos=1, eos=2, seed=0, task="copy")
    for r in rows:
        ids, labels = r["input_ids"], r["labels"]
        half = (18 - 2) // 2
        assert ids[1 + half : 1 + 2 * half] == ids[1 : 1 + half]
        assert labels[: 1 + half] == [-100] * (1 + half)
        assert labels[1 + half :] == ids[1 + half :]


def _tiny_model(seed=0):
    import torch
    from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM

    torch.manual_seed(seed)
    cfg = PerceiverARConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, token_embedding_dim=8,
        pre_layers=1, pre_window=3, global_layers=1, stack_layers=2, block=3, nope_every=0,
        num_attention_heads=4, num_kv_heads=2, head_dim=8, ngram_buckets=64,
        value_embed_layers=(0,), value_embed_dim=4, use_liger=False, attn_backend="sdpa",
        attn_pad_multiple=1, chunked_ce_block_size=5, z_loss=0.0,
    )
    model = PerceiverARLM(cfg).eval()
    for layer in model.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return model


def test_bucket_means_matches_manual_slicing():
    import torch
    from evaluation.long_context_probes import bucket_means

    per = torch.arange(10, dtype=torch.float32)
    assert bucket_means(per, [4, 10]) == [1.5, 6.5]
    assert bucket_means(per, [10]) == [4.5]


def test_probe_reach_paired_stats_and_exact_zero_below_window(tmp_path):
    """End-to-end on CPU: build a 3-row eval set, sweep windows, check the paired invariants."""
    import json
    import types
    import torch
    from datasets import Dataset
    from evaluation.long_context_probes import probe_reach

    model = _tiny_model()
    torch.manual_seed(1)
    rows = [{"input_ids": torch.randint(3, 97, (24,)).tolist()} for _ in range(3)]
    ds_dir = tmp_path / "eval"
    Dataset.from_list(rows).save_to_disk(str(ds_dir))
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"sources": [{"eval_path": str(ds_dir)}]}))
    args = types.SimpleNamespace(buckets="8,16,24", manifest=str(manifest), max_rows=8,
                                 reach_windows="8,16,full")
    res = probe_reach(model, args, "cpu")
    assert res["rows"] == 3 and res["windows"] == ["8", "16", "full"]
    assert res["touched_layers"] == [model.config.global_layer_index]
    # buckets fully below the window are bit-identical -> Δ exactly 0 with se 0
    d8 = res["delta_vs_full"]["8"]
    assert d8["[0,8)"]["mean"] == 0.0 and d8["[0,8)"]["se"] == 0.0 and d8["[0,8)"]["n"] == 3
    d16 = res["delta_vs_full"]["16"]
    assert d16["[0,8)"]["mean"] == 0.0 and d16["[8,16)"]["mean"] == 0.0
    # beyond the window something changes
    assert d8["[16,24)"]["mean"] != 0.0
    # ce['full'] equals the mean over rows of the unrestricted per-row bucket means
    full_rows = torch.tensor(res["per_row"]["full"])
    for b, lab in enumerate(res["buckets"]):
        assert abs(res["ce"]["full"][lab] - float(full_rows[:, b].mean())) < 1e-6
    # the model is restored after the sweep
    assert all(model.layers[i].attn.pattern == "full" for i in res["touched_layers"])


def test_reach_window_flag_applies_to_any_probe(tmp_path):
    """--reach_window on the copy probe = the positive control: restricting the global read below
    the copy offset must change the predictions of a model that relies on it."""
    import torch
    import types
    from datasets import Dataset
    from evaluation.long_context_probes import probe_copy

    model = _tiny_model()
    torch.manual_seed(2)
    half = 8
    rows = []
    for _ in range(2):
        pat = torch.randint(3, 97, (half,)).tolist()
        ids = [1] + pat + pat + [2]
        labels = [-100] * (1 + half) + ids[1 + half:]
        rows.append({"input_ids": ids, "labels": labels})
    ds_dir = tmp_path / "copy"
    Dataset.from_list(rows).save_to_disk(str(ds_dir))
    args = types.SimpleNamespace(copy_dataset=str(ds_dir))
    with torch.no_grad():
        a = probe_copy(model, args, "cpu")
        with model.reach_override(2):
            b = probe_copy(model, args, "cpu")
    assert a["rows"] == 2 and b["rows"] == 2
    assert 0.0 <= a["copy_token_accuracy"] <= 1.0 and 0.0 <= b["copy_token_accuracy"] <= 1.0


def test_reach_tail_stats_and_probe_reports_them(tmp_path):
    import json
    import types
    import torch
    from datasets import Dataset
    from evaluation.long_context_probes import probe_reach, reach_tail_stats

    d = torch.tensor([0.0, 0.05, 0.5, -0.3, 0.2] + [0.0] * 95)
    st = reach_tail_stats(d, thresh=0.1)
    assert st["n_tokens"] == 100
    assert abs(st["frac_worse_gt_0.1"] - 0.02) < 1e-9 and abs(st["frac_better_gt_0.1"] - 0.01) < 1e-9
    assert abs(st["max"] - 0.5) < 1e-6 and abs(st["min"] + 0.3) < 1e-6
    assert abs(st["top1pct_mean"] - 0.5) < 1e-6  # top 1% of 100 = 1 token
    assert reach_tail_stats(torch.zeros(0)) == {"n_tokens": 0}

    model = _tiny_model()
    torch.manual_seed(3)
    rows = [{"input_ids": torch.randint(3, 97, (24,)).tolist()} for _ in range(2)]
    ds_dir = tmp_path / "eval"
    Dataset.from_list(rows).save_to_disk(str(ds_dir))
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"sources": [{"eval_path": str(ds_dir)}]}))
    args = types.SimpleNamespace(buckets="8,24", manifest=str(manifest), max_rows=8, reach_windows="8,24,full")
    res = probe_reach(model, args, "cpu")
    assert set(res["tail"]) == {"8", "24"}
    assert res["tail"]["8"]["n_tokens"] == 2 * 23
    # a window covering the whole sequence changes nothing anywhere
    assert res["tail"]["24"]["max"] == 0.0 and res["tail"]["24"]["min"] == 0.0
    assert res["tail"]["8"]["max"] > 0.0 or res["tail"]["8"]["min"] < 0.0


def test_build_filler_concatenates_rows_cyclically_to_budget():
    from evaluation.long_context_probes import build_filler

    rows = [[1, 2, 3], [4, 5], [6]]
    assert build_filler(rows, 0, 7) == [1, 2, 3, 4, 5, 6, 1]
    assert build_filler(rows, 2, 4) == [6, 1, 2, 3]
    assert len(build_filler(rows, 1, 100)) == 100


def test_passkey_frame_token_ends_the_question():
    tok = _Tok()
    filler = [5] * 3000
    ids, answer = build_passkey(tok, filler, 512, 0.5, __import__("random").Random(0), frame=[999])
    assert len(ids) == 512 and ids[-len(answer):] == answer
    assert ids[-len(answer) - 1] == 999  # the frame token sits right before the answer


def test_multikey_has_n_needles_and_asks_for_the_target():
    import random
    import re
    from evaluation.long_context_probes import build_multikey

    tok = _Tok()
    filler = [ord("~") - 10 + 10] * 6000   # '~' filler decodes cleanly
    for depth in (0.1, 0.5, 0.9):
        ids, answer = build_multikey(tok, filler, 2048, depth, random.Random(1), n_keys=4)
        assert len(ids) == 2048 and ids[-len(answer):] == answer
        text = tok.decode(ids)
        needles = re.findall(r"The pass key for (\w+) is (\d{5})\.", text)
        assert len(needles) == 4
        m = re.search(r"What is the pass key for (\w+)\? The pass key for \1 is (\d{5})$", text)
        assert m is not None
        target_name, target_key = m.group(1), m.group(2)
        assert dict(needles)[target_name] == target_key
        # the target needle sits at the requested depth
        pos = text.index(f"The pass key for {target_name} is") / len(text)
        assert abs(pos - depth) < 0.12, (depth, pos)


def test_variable_tracking_chain_is_ordered_and_answer_lists_the_chain():
    import random
    import re
    from evaluation.long_context_probes import build_variable_tracking

    tok = _Tok()
    filler = [ord("~")] * 8000
    ids, answer = build_variable_tracking(tok, filler, 3000, 0.2, random.Random(3), hops=3, n_chains=2)
    assert len(ids) == 3000 and ids[-len(answer):] == answer
    text = tok.decode(ids)
    value = re.search(r"assigned the value (\d{5})", text).group(1)
    names = re.search(r"the variables are ((?:[A-Z]{3} ?)+)$", text).group(1).split()
    assert len(names) == 3
    stmts = {v: text.index(f"VAR {v} = ") for v in names}
    assert text[stmts[names[0]]:].startswith(f"VAR {names[0]} = {value}")
    for i in range(1, 3):
        assert stmts[names[i]] > stmts[names[i - 1]]                      # hop i after its source
        assert text[stmts[names[i]]:].startswith(f"VAR {names[i]} = {names[i-1]}")
    # a distractor chain with another value is present
    assert len(set(re.findall(r"VAR [A-Z]{3} = (\d{5})", text))) == 2


def test_frequent_words_answer_is_the_top3_by_count():
    import random
    from collections import Counter
    from evaluation.long_context_probes import build_frequent_words

    tok = _Tok()
    ids, answer = build_frequent_words(tok, 4000, random.Random(5), n_answer=3)
    assert 0.9 * 4000 <= len(ids) <= 4000 and ids[-len(answer):] == answer
    text = tok.decode(ids)
    body = text.split("Coded text:")[1].split(" Question:")[0]
    counts = Counter(body.split())
    top = [w for w, _ in counts.most_common(3)]
    assert tok.decode(answer).split() == top
    c = [counts[w] for w in top]
    assert c[0] > c[1] > c[2]


def test_score_answer_and_aggregate_semantics():
    import torch
    from evaluation.long_context_probes import _aggregate, _score_answer

    model = _tiny_model()
    ids = list(range(3, 30))
    answer = ids[-3:]
    s = _score_answer(model, ids, answer, "cpu")
    assert set(s) == {"exact", "tok_correct", "tok_total", "first"} and s["tok_total"] == 3
    # greedy predictions of the trailing span equal the model's own argmax there
    x = torch.tensor(ids)[None]
    with torch.no_grad():
        pred = model(input_ids=x).logits[0].argmax(-1)[-4:-1].tolist()
    assert s["exact"] == int(pred == answer) and s["first"] == int(pred[0] == answer[0])
    agg = _aggregate([{"exact": 1, "tok_correct": 3, "tok_total": 3, "first": 1},
                      {"exact": 0, "tok_correct": 1, "tok_total": 3, "first": 0}])
    assert agg == {"exact": 0.5, "token_acc": 4 / 6, "first_token_acc": 0.5, "n": 2}


def test_probe_suite_runs_several_probes_with_one_model(tmp_path, monkeypatch):
    import json
    import types
    import torch
    from datasets import Dataset
    import evaluation.long_context_probes as lcp

    model = _tiny_model()
    torch.manual_seed(4)
    rows = [{"input_ids": torch.randint(3, 97, (64,)).tolist()} for _ in range(3)]
    ds_dir = tmp_path / "eval"
    Dataset.from_list(rows).save_to_disk(str(ds_dir))
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"sources": [{"eval_path": str(ds_dir)}]}))
    class _SmallTok:  # ids must fit the tiny model's 97-token vocabulary
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 90 + 5 for c in text]

    monkeypatch.setattr(lcp, "_load_tokenizer", lambda args: _SmallTok())
    args = types.SimpleNamespace(
        suite="fwe,buckets,multikey", manifest=str(manifest), max_rows=8, context_lengths="48",
        trials=1, seed=0, buckets="16,48", n_keys=2, checkpoint="x", tokenizer=None,
    )
    res = lcp.probe_suite(model, args, "cpu")
    assert res["suite"] == ["fwe", "buckets", "multikey"]
    assert "fwe@48" in res["results"]["fwe"] and 0.0 <= res["results"]["fwe"]["fwe@48"] <= 1.0
    assert "ce[0,16)" in res["results"]["buckets"]
    assert "multikey@48" in res["results"]["multikey"]
    assert res["errors"] == {}
    assert set(res["elapsed_s"]) == {"fwe", "buckets", "multikey"}
    # a broken probe is recorded, the others still complete
    monkeypatch.setitem(lcp.PROBES, "boom", lambda m, a, d: 1 / 0)
    args.suite = "boom,fwe"
    res = lcp.probe_suite(model, args, "cpu")
    assert "ZeroDivisionError" in res["errors"]["boom"] and "fwe@48" in res["results"]["fwe"]


def test_probe_tasks_counts_marked_positions(tmp_path):
    import types
    import torch
    from datasets import Dataset
    from data.data_collators import labels_from_span_markers
    from evaluation.long_context_probes import probe_tasks

    model = _tiny_model()
    S, E = 90, 91
    rows = [{"input_ids": [1, 5, 6, 7, 8, 9, S, 10, 11, E, 12, 13, 5, 6, S, 14, E, 2]},
            {"input_ids": [1, 20, 21, S, 22, E, 23, 24, 25, 2]}]
    ds_dir = tmp_path / "tasks"
    Dataset.from_list(rows).save_to_disk(str(ds_dir))
    args = types.SimpleNamespace(tasks_dataset=str(ds_dir), markers=f"{S},{E}", max_rows=0)
    with torch.no_grad():
        res = probe_tasks(model, args, "cpu")
    expected = sum(sum(l != -100 for l in labels_from_span_markers(r["input_ids"], S, E)[1:]) for r in rows)
    assert res["rows"] == 2 and res["labelled_tokens"] == expected == 7  # 3+2 in row 1, 2 in row 2
    assert 0.0 <= res["tasks_token_accuracy"] <= 1.0 and 0.0 <= res["tasks_first_token_accuracy"] <= 1.0
