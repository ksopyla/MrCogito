"""E33a read–think–reread loop: prelude → [global read + span layers] × R (tied) → answer layer."""
import numpy as np
import pytest
import torch

from tests.test_latent_memory import M, future_leaks, lm_cfg, row
from nn.perceiver_ar_lm import PerceiverARLM

# 4 layers: 0 local (prelude), 1 global read, 2 local (core), 3 local (answer layer)
BASE = dict(stack_layers=2, lm_addr="none", lm_slot_pos="boundary")


def make(seed=0, **kw):
    torch.manual_seed(seed)
    m = PerceiverARLM(lm_cfg(**{**BASE, **kw})).eval()
    for layer in m.layers:  # warm residuals so every path is visible at init
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def labels_for(ids, start=11):
    lab = torch.full_like(ids, -100)
    lab[0, start:] = ids[0, start:]
    return lab


def test_one_loop_is_todays_stack():
    """R = 1 (and R = 4 with zero loop markers, run once) computes exactly the default model."""
    a = make(seed=3)
    b = make(seed=3, message_loop_rounds=1)
    ids = row()
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)
    c = make(seed=3, message_loop_rounds=4)
    c.load_state_dict(a.state_dict(), strict=False)  # loop_emb stays zero
    c._loop_rounds_override = 1
    with torch.no_grad():
        assert torch.equal(a(ids).logits, c(ids).logits)
        c._loop_rounds_override = None
        assert not torch.equal(a(ids).logits, c(ids).logits)  # 4 loops do compute something else


def test_loop_trains_and_stays_causal():
    m = make(message_loop_rounds=4)
    m.loop_emb.data.normal_(0, 0.1)
    assert future_leaks(m, row(), 11) == []
    assert future_leaks(m, row(extra_q=20), 11) == []
    m.train()
    ids = row()
    out = m(ids, labels=labels_for(ids))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for p in (m.loop_emb, m.layers[1].attn.wq.weight, m.layers[2].mlp.down.weight, m.layers[3].mlp.down.weight):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.memory_writer.parameters())


def test_exit_loss_matches_the_shorter_forwards():
    """loss = final CE + aux · Σ_{r<R} CE(exit r), and exit r's logits equal the R = r forward."""
    m = make(seed=5, message_loop_rounds=3, message_loop_exit_aux=0.5, message_loop_exit_targets="answer")
    m.loop_emb.data.normal_(0, 0.1)
    ids = row()
    lab = labels_for(ids)
    with torch.no_grad():
        total = m(ids, labels=lab).loss
        ce = []
        for r in (1, 2, 3):
            m._loop_rounds_override = r
            ce.append(m(ids, labels=lab, return_per_token_loss=True)[0].loss)
        m._loop_rounds_override = None
    assert torch.allclose(total, ce[2] + 0.5 * (ce[0] + ce[1]), atol=1e-5)


def test_progress_targets_used_and_answer_fallback():
    m = make(seed=6, message_loop_rounds=3, message_loop_exit_aux=1.0).train()
    ids = row()
    lab = labels_for(ids)
    with torch.no_grad():
        no_tg = m(ids, labels=lab).loss  # no round targets → exits use the labels
        other = lab.clone()
        other[0, 11:] = (other[0, 11:] + 5) % 80 + 3
        m.set_round_targets(torch.stack([other, other]))
        with_tg = m(ids, labels=lab).loss
    assert not torch.allclose(no_tg, with_tg)


def test_prelude_inject_and_wide_core_build():
    m = make(message_loop_rounds=3, message_loop_inject="prelude")
    assert m.loop_gate is not None and float(m.loop_gate.detach()) == 0.0
    m.train()
    ids = row()
    m(ids, labels=labels_for(ids)).loss.backward()
    assert m.loop_gate.grad is not None
    w = make(message_loop_rounds=3, message_loop_span=2, message_loop_exit_aux=0.3)  # core = layers 1..3, head after
    w.train()
    loss = w(ids, labels=labels_for(ids)).loss
    assert torch.isfinite(loss)
    assert future_leaks(w.eval(), row(), 11) == []


@pytest.mark.parametrize("bad", [
    dict(message_loop_rounds=2, message_read_rounds=2),
    dict(message_loop_rounds=2, message_loop_span=3),
    dict(message_loop_rounds=2, message_loop_inject="bogus"),
])
def test_invalid_loop_configs(bad):
    with pytest.raises(ValueError):
        lm_cfg(**{**BASE, **bad})


def test_probe_replay_and_round_target_fallback():
    from data.bapo_ladder import config_for, resolve_recipe
    from verification.bapo_capability_probe import make_batch

    pc = resolve_recipe("chain_parallel")
    c = config_for("bridge_1k", pc.task, hops=3, key_len=8, **pc.overrides)
    rl = resolve_recipe("recall_single")
    r = config_for("bridge_1k", rl.task, **{**rl.overrides, "seq_len": c.seq_len})
    ids, labels, tg = make_batch(c, np.random.default_rng(0), 8, "cpu", round_targets=3, replay=(r, 0.25))
    assert ids.shape == (8, c.seq_len) and tg.shape == (3, 8, c.seq_len)
    # the last 2 rows are lookup replay rows: no chain nodes → every round target is the answer
    for b in (6, 7):
        assert torch.equal(tg[0, b], labels[b]) and torch.equal(tg[2, b], labels[b])
    # chain rows: round 0 targets differ from the answer (node 1 vs terminal)
    assert any(not torch.equal(tg[0, b], labels[b]) for b in range(6))


def test_named_variant_e33a_loop_builds_the_loop():
    from evaluation.bapo_models import ArchSpec, build_model

    spec = ArchSpec(name="shared", hidden=64, head_dim=16, n_kv_heads=1, pre_layers=1, global_layers=1,
                    stack_layers=2, local_window=16, message_boundary_token_id=10, token_embedding_dim=32,
                    ngram_orders=(), lm_latent_dim=64, lm_writer_dim=32, zero_init_residuals=False)
    m = build_model("e33a_loop", vocab_size=17, seq_len=512, answer_start=480, pad_id=12, bos_id=11,
                    eos_id=12, spec=spec, seed=0)
    assert m.loop_emb is not None and m.loop_emb.shape[0] == 4
    assert m.config.message_loop_exit_aux == 0.3 and m.config.lm_reader_tokens == 1
    assert m.config.lm_slot_pos == "boundary" and m.config.lm_addr == "none"
