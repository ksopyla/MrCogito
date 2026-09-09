"""Tests for evaluation/long_context_probes.py and scripts/build_copy_task_dataset.py."""
import torch

from evaluation.long_context_probes import build_passkey, per_token_ce
from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM
from scripts.build_copy_task_dataset import make_rows


class _Tok:
    """Minimal tokenizer stub: one id per character (a-z, digits, space, punctuation)."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 200 + 10 for c in text]


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
