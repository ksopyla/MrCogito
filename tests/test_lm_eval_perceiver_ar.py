"""Contract tests for the lm-evaluation-harness adapter (`evaluation/lm_eval_perceiver_ar.py`)
and the suite runner's summary logic (`evaluation/run_lm_eval_suite.py`). CPU, tiny model, a
word-level tokenizer built on the fly (no network)."""
import json

import pytest
import torch

from nn.perceiver_ar_lm import PerceiverARConfig, PerceiverARLM

V = 97
WORDS = ["the", "cat", "sat", "on", "mat", "dog", "ran", "far", "hello", "world", "foo", "bar",
         "a", "b", "c", "d", "e", "f", "g", "h"]


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = PerceiverARConfig(
        vocab_size=V, hidden_size=32, intermediate_size=64, token_embedding_dim=8,
        pre_layers=1, pre_window=4, global_layers=1, stack_layers=2, block=6, nope_every=0,
        num_attention_heads=4, num_kv_heads=2, head_dim=8, ngram_buckets=64,
        value_embed_layers=(0,), value_embed_dim=4, use_liger=False, attn_backend="sdpa",
        attn_pad_multiple=2048, chunked_ce_block_size=5, z_loss=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = PerceiverARLM(cfg).eval()
    for layer in model.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return model


def _save_checkpoint(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model = _tiny_model()
    ck = tmp_path / "ck"
    model.save_pretrained(ck, safe_serialization=True)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for i, w in enumerate(WORDS):
        vocab[w] = 4 + i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<s>",
                                   eos_token="</s>", unk_token="<unk>")
    fast.save_pretrained(ck)
    return ck, model, fast


def test_adapter_registers_and_forces_eval_friendly_attention(tmp_path):
    from lm_eval.api.registry import MODEL_REGISTRY

    import evaluation.lm_eval_perceiver_ar as adapter

    assert MODEL_REGISTRY["perceiver_ar"] is adapter.PerceiverARLMEval
    ck, _, _ = _save_checkpoint(tmp_path)
    lm = adapter.PerceiverARLMEval(pretrained=str(ck), device="cpu", batch_size=2, max_length=24)
    assert lm.model.config.attn_pad_multiple == 1          # no padding to the 2048 training block
    assert lm.model.config.attn_backend == "sdpa"
    assert lm.model.config.use_liger is False
    assert lm.max_length == 24 and lm.backend == "causal"
    assert str(lm.tokenizer.name_or_path).rstrip("/").endswith("ck")   # tokenizer from the checkpoint dir
    out = lm._model_call(torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]]))
    assert out.shape == (2, 4, V)


def test_loglikelihood_matches_manual_softcapped_logprobs(tmp_path):
    """The harness must score exactly the distribution the model was trained under: tanh soft-cap
    then log-softmax, summed over the continuation tokens given the gold prefix."""
    from lm_eval.api.instance import Instance

    import evaluation.lm_eval_perceiver_ar as adapter

    ck, _, tok = _save_checkpoint(tmp_path)
    lm = adapter.PerceiverARLMEval(pretrained=str(ck), device="cpu", batch_size=2, max_length=24)
    ctx, cont = "the cat sat on", " the mat"
    req = Instance(request_type="loglikelihood", doc={}, arguments=(ctx, cont), idx=0)
    (ll, greedy), = lm.loglikelihood([req])

    ctx_ids = tok.encode(ctx, add_special_tokens=False)
    cont_ids = tok.encode(cont, add_special_tokens=False)
    ids = torch.tensor([ctx_ids + cont_ids])
    with torch.no_grad():
        logits = lm.model(input_ids=ids).logits[0].float()           # soft-capped, [S, V]
    logp = torch.log_softmax(logits, dim=-1)
    manual = sum(float(logp[len(ctx_ids) - 1 + i, t]) for i, t in enumerate(cont_ids))
    assert abs(ll - manual) < 1e-3, (ll, manual)
    assert isinstance(greedy, bool)


def test_generate_until_is_explicitly_unsupported(tmp_path):
    import evaluation.lm_eval_perceiver_ar as adapter

    ck, _, _ = _save_checkpoint(tmp_path)
    lm = adapter.PerceiverARLMEval(pretrained=str(ck), device="cpu", batch_size=1, max_length=16)
    with pytest.raises(NotImplementedError):
        lm._model_generate(torch.tensor([[4, 5]]), max_length=8, stop=[])


def test_resolve_tokenizer_prefers_checkpoint_dir(tmp_path):
    from evaluation.long_context_probes import DEFAULT_TOKENIZER, resolve_tokenizer_name

    assert resolve_tokenizer_name(str(tmp_path)) == DEFAULT_TOKENIZER
    (tmp_path / "tokenizer_config.json").write_text("{}")
    assert resolve_tokenizer_name(str(tmp_path)) == str(tmp_path)
    assert resolve_tokenizer_name(str(tmp_path), "gpt2") == "gpt2"


def test_suite_summary_picks_main_metric_and_averages_accuracies():
    from evaluation.run_lm_eval_suite import CORE_TASKS, FULL_EXTRA, _tasks_for, summarize

    results = {
        "hellaswag": {"acc,none": 0.30, "acc_norm,none": 0.35, "acc_norm_stderr,none": 0.005},
        "winogrande": {"acc,none": 0.52, "acc_stderr,none": 0.01},
        "wikitext": {"word_perplexity,none": 55.0, "byte_perplexity,none": 2.1, "bits_per_byte,none": 1.07},
    }
    row = summarize(results, ["hellaswag", "winogrande", "wikitext"])
    assert row["hellaswag/acc_norm"] == 0.35 and row["hellaswag/acc_norm_stderr"] == 0.005
    assert row["winogrande/acc"] == 0.52
    assert row["wikitext/word_perplexity"] == 55.0
    assert abs(row["avg_acc"] - (0.35 + 0.52) / 2) < 1e-9 and row["n_acc_tasks"] == 2
    assert _tasks_for("core", None) == CORE_TASKS
    assert _tasks_for("full", None) == CORE_TASKS + FULL_EXTRA
    assert _tasks_for("core", "piqa, boolq") == ["piqa", "boolq"]


def test_suite_runner_end_to_end_on_tiny_checkpoint(tmp_path, monkeypatch):
    """`simple_evaluate` is replaced by a stub so no dataset download happens; the runner must
    build the perceiver_ar model, write the JSON and the summary CSV row."""
    import evaluation.run_lm_eval_suite as runner

    ck, _, _ = _save_checkpoint(tmp_path)
    captured = {}

    def fake_simple_evaluate(model, tasks, **kw):
        captured["model"] = model
        captured["tasks"] = tasks
        return {"results": {"piqa": {"acc_norm,none": 0.5, "acc,none": 0.49}}, "versions": {"piqa": 1},
                "n-samples": {"piqa": {"original": 10, "effective": 10}}, "config": {"model": "perceiver_ar"}}

    import lm_eval

    monkeypatch.setattr(lm_eval, "simple_evaluate", fake_simple_evaluate)
    out_dir = tmp_path / "reports"
    rc = runner.main(["--checkpoint", str(ck), "--tag", "tiny", "--tasks", "piqa", "--device", "cpu",
                      "--batch_size", "2", "--max_length", "16", "--out_dir", str(out_dir)])
    assert rc == 0
    assert captured["tasks"] == ["piqa"]
    assert type(captured["model"]).__name__ == "PerceiverARLMEval"
    payload = json.loads((out_dir / "tiny.json").read_text())
    assert payload["results"]["piqa"]["acc_norm,none"] == 0.5 and payload["model_kind"] == "perceiver_ar"
    csv_text = (out_dir / "summary.csv").read_text()
    assert "piqa/acc_norm" in csv_text and "tiny" in csv_text
    # re-running the same tag replaces the row instead of appending a duplicate
    runner.main(["--checkpoint", str(ck), "--tag", "tiny", "--tasks", "piqa", "--device", "cpu",
                 "--batch_size", "2", "--max_length", "16", "--out_dir", str(out_dir)])
    assert (out_dir / "summary.csv").read_text().count("tiny") == 1
