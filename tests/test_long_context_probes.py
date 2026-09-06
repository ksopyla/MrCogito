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
