"""E36 — trained writer, frozen reader (nn/frozen_reader_memory.py). Tiny random Llama models, no downloads."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nn.frozen_reader_memory import FrozenReaderMemory, FrozenReaderMemoryConfig

V, S, M = 101, 8, 3


def _tiny(layers, hidden, heads, kv):
    cfg = LlamaConfig(vocab_size=V, hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv, max_position_embeddings=4096,
                      tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    return LlamaForCausalLM(cfg)


def _build(way="input", entry_positions="span_mid", recon_frac=0.0, **kw):
    torch.manual_seed(0)
    reader, writer = _tiny(2, 32, 4, 2), _tiny(3, 24, 4, 2).model
    cfg = FrozenReaderMemoryConfig(writer_layers=2, span=S, mem_spans=M, way=way, proj_hidden=16,
                                   reader_dtype="float32", recon_frac=recon_frac, entry_positions=entry_positions,
                                   **kw)
    return FrozenReaderMemory(cfg, reader=reader, writer_base=writer)


def _rows(b=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, V, (b, 1 + (M + 1) * S + 1), generator=g)


@pytest.mark.parametrize("way", ["input", "kv"])
@pytest.mark.parametrize("pos", ["span_mid", "compact"])
def test_loss_finite_and_only_writer_and_adapter_learn(way, pos):
    model = _build(way, pos).train()
    out = model(input_ids=_rows())
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is None for p in model.reader.parameters())
    assert model.writer.embed_tokens.weight.grad is None
    assert model.writer.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapter.parameters())


def test_writer_is_bidirectional_within_a_span():
    model = _build().eval()
    seen = {}
    model.writer.layers[-1].register_forward_hook(lambda m, i, o: seen.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    ids = _rows(1)[:, 1:1 + S]
    model.writer(ids)
    first = seen["h"][0, 0].clone()
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % V
    model.writer(ids2)
    assert not torch.allclose(first, seen["h"][0, 0])  # token 0 sees the last token


@pytest.mark.parametrize("way", ["input", "kv"])
def test_reader_stays_causal_over_the_target_and_reads_the_entries(way):
    model = _build(way).eval()
    rows = _rows(1)[:, 1:]
    mem, tgt = rows[:, :M * S], rows[:, M * S:(M + 1) * S]
    tpos = torch.arange(S) + M * S
    with torch.no_grad():
        e = model.write(mem)
        base = model.read_logits(e, tgt, tpos, model.entry_pos(0, M, "cpu"))
        tgt2 = tgt.clone()
        tgt2[0, -1] = (tgt2[0, -1] + 1) % V
        later = model.read_logits(e, tgt2, tpos, model.entry_pos(0, M, "cpu"))
        assert torch.allclose(base[:, :-1], later[:, :-1], atol=1e-5)  # no peeking at later target tokens
        other = model.read_logits(e + 1.0, tgt, tpos, model.entry_pos(0, M, "cpu"))
        assert not torch.allclose(base[:, 0], other[:, 0])  # the first target token sees the entries


def test_no_memory_and_raw_prefix_match_the_plain_reader():
    model = _build().eval()
    rows = _rows(1)[:, 1:]
    raw, tgt = rows[:, :M * S], rows[:, M * S:(M + 1) * S]
    with torch.no_grad():
        plain = model.reader(input_ids=tgt).logits
        none = model.read_logits(None, tgt, torch.arange(S))
        full = model.reader(input_ids=rows[:, :(M + 1) * S]).logits[:, M * S:]
        pre = model.read_logits(None, tgt, torch.arange(S) + M * S, raw_prefix=raw, raw_pos=torch.arange(M * S))
    assert torch.allclose(plain, none, atol=1e-5)
    assert torch.allclose(full, pre, atol=1e-5)


def test_checkpoint_holds_only_trained_parts_and_reloads():
    model = _build("kv")
    sd = model.state_dict()
    assert not any(k.startswith(("reader.", "writer.embed_tokens.")) for k in sd)
    assert any(k.startswith("adapter.") for k in sd) and any(k.startswith("writer.layers.") for k in sd)
    fresh = _build("kv")
    with torch.no_grad():
        for p in fresh.adapter.parameters():
            p.add_(1.0)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith(("reader.", "writer.embed_tokens.")) for k in missing)
    assert torch.equal(fresh.adapter.proj.weight, model.adapter.proj.weight)


def test_reconstruction_draw_is_the_same_on_every_rank():
    a, b = _build(recon_frac=0.5).train(), _build(recon_frac=0.5).train()
    draws_a = [a._recon_step() for _ in range(50)]
    draws_b = [b._recon_step() for _ in range(50)]
    assert draws_a == draws_b
    assert 10 < sum(r for r, _ in draws_a) < 40


def test_trainer_divides_the_loss_by_accumulation_steps():
    assert FrozenReaderMemory.accepts_loss_kwargs is False


def test_attention_pool_starts_as_the_mean_and_learns():
    mean, attn = _build("kv").eval(), _build("kv", writer_pool="attn").eval()
    attn.writer.load_state_dict(mean.writer.state_dict(), strict=False)  # same layers and norm
    ids = _rows(2)[:, 1:1 + S]
    with torch.no_grad():
        assert torch.allclose(mean.writer(ids), attn.writer(ids), atol=1e-5)
    attn.train()
    attn(input_ids=_rows()).loss.backward()
    assert attn.writer.pool.q.grad.abs().sum() > 0


def test_curriculum_runs_reconstruction_phases_by_optimizer_step():
    model = _build("kv", curriculum="2:2,4:1", grad_accum=2).train()
    spans = []
    for _ in range(8):
        model._fwd += 1
        spans.append(model._curriculum_span())
        model._fwd -= 1
        out = model(input_ids=_rows())  # advances the counter
        assert torch.isfinite(out.loss)
    # steps 0-1 (forward calls 1-4) at 2 tokens, step 2 (calls 5-6) at 4 tokens, then the main objective
    assert spans == [2, 2, 2, 2, 4, 4, None, None]


def test_curriculum_spec_is_validated():
    with pytest.raises(ValueError):
        FrozenReaderMemoryConfig(span=8, curriculum="16:10")


@pytest.mark.parametrize("pool", ["mean", "attn"])
def test_padding_does_not_change_an_entry(pool):
    model = _build("kv", writer_pool=pool).eval()
    if pool == "attn":
        with torch.no_grad():
            model.writer.pool.q.normal_()  # away from the uniform start
    ids = _rows(1)[:, 1:1 + S]
    n = 5
    padded = ids.clone()
    padded[:, n:] = 0
    mask = torch.zeros_like(ids)
    mask[:, :n] = 1
    with torch.no_grad():
        assert torch.allclose(model.writer(ids[:, :n]), model.writer(padded, mask), atol=1e-5)


def _s2_batch(model, b=2):
    from data.data_collators import DataCollatorForMemoryRows

    g = torch.Generator().manual_seed(3)
    feats = []
    for i, ml in enumerate([S + 3, 5][:b]):
        ids = torch.randint(3, V, (ml + 6 + i,), generator=g).tolist()
        feats.append({"input_ids": ids, "mem_len": ml})
    return DataCollatorForMemoryRows(0, span=S, max_mem_spans=M)(feats), feats


def test_memory_rows_collator_lines_up_memory_target_and_teacher():
    batch, feats = _s2_batch(_build("kv"))
    assert batch["mem_ids"].shape == (2, 2 * S)
    assert batch["mem_mask"][1].sum() == 5 and batch["mem_len"].tolist() == [S + 3, 5]
    for i, f in enumerate(feats):
        n = int(batch["teacher_mask"][i].sum())
        assert batch["teacher_ids"][i, :n].tolist() == f["input_ids"]
        tl = int(batch["tgt_mask"][i].sum())
        assert batch["tgt_ids"][i, :tl].tolist() == f["input_ids"][f["mem_len"]:]


def test_teacher_logprobs_are_the_reader_on_the_raw_text():
    model = _build("kv").eval()
    batch, feats = _s2_batch(model)
    t = batch["tgt_ids"].shape[1]
    got = model._teacher_logprobs(batch["teacher_ids"], batch["teacher_mask"], batch["mem_len"], t)
    for i, f in enumerate(feats):
        ids = torch.tensor([f["input_ids"]])
        ml = f["mem_len"]
        ref = model.reader(input_ids=ids).logits[0].float().log_softmax(-1)
        tl = len(f["input_ids"]) - ml
        assert torch.allclose(got[i, :tl - 1], ref[ml:ml + tl - 1], atol=1e-4)


@pytest.mark.parametrize("way", ["input", "kv"])
def test_distill_loss_trains_only_writer_and_adapter(way):
    model = _build(way, objective="distill").train()
    batch, _ = _s2_batch(model)
    out = model(**batch)
    assert torch.isfinite(out.loss) and out.loss >= 0
    out.loss.backward()
    assert all(p.grad is None for p in model.reader.parameters())
    assert model.writer.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
