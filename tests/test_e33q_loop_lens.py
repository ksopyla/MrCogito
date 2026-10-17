"""E33q — the loop asks its next question with the name it found: round r names hop r's output word through the shared
LM head at the decision position (training loss), and its LM-head distribution, re-embedded with the shared token table
and input layer, joins round r + 1 through a gate that starts at 0.

Roadmap: docs/4_Research_Notes/text_t4_t7_roadmap_20261017.md §4.3, §4.4 (the decode-and-append oracle).
"""
import math

import torch
import torch.nn.functional as F

from tests.test_text_memory import TEXT, _targets, make, text_row

LOOP4 = dict(message_raw_window=8, message_loop_rounds=4)


def test_zero_gate_is_the_model_without_feedback():
    torch.manual_seed(0)
    m = make(**TEXT, **LOOP4, message_loop_feedback="soft").eval()
    assert m.loop_feedback_gate is not None and not m.loop_feedback_gate.any()
    ids = text_row(S=48)
    with torch.no_grad():
        base = m(ids).logits
        gate = m.loop_feedback_gate
        m.loop_feedback_gate = None
        assert torch.allclose(m(ids).logits, base, atol=1e-6)
        m.loop_feedback_gate = gate
        with torch.no_grad():
            gate.fill_(0.5)
        assert not torch.allclose(m(ids).logits, base, atol=1e-4)


def test_name_lens_is_minus_mean_log_p_of_the_hop_word_tokens():
    """Round 1's state at the decision position, read through final norm + LM head, must name the span's tokens."""
    torch.manual_seed(1)
    ids = text_row(S=48)
    q, t0, t1 = 40, 10, 12
    m = make(**TEXT, **LOOP4, message_loop_name_lens=1.0).train()
    states = []
    orig = m._loop_lens

    def spy(x, r, input_ids):
        states.append(x.detach().clone())
        return orig(x, r, input_ids)

    m._loop_lens = spy
    with torch.no_grad():
        base = m(ids, labels=ids.clone()).loss
        states.clear()
        with_t = m(ids, labels=ids.clone(), read_targets=_targets([q, 1, t0, t1], [q, 0, -1, -1])).loss
    h = m.final_norm(states[0][0, q]).detach()
    z = F.linear(h, m.lm_head.weight).float()
    sc = float(m.config.logit_softcap or 0.0)
    if sc > 0:
        z = sc * torch.tanh(z / sc)
    logp = F.log_softmax(z, -1)
    want = -float(logp[ids[0, t0:t1 + 1]].mean())
    assert abs(float(with_t - base) - want) < 1e-4


def test_feedback_gate_and_name_loss_receive_gradient():
    torch.manual_seed(2)
    ids = text_row(S=48)
    m = make(**TEXT, **LOOP4, message_loop_name_lens=0.3, message_loop_feedback="soft").train()
    out = m(ids, labels=ids.clone(), read_targets=_targets([40, 1, 10, 12], [40, 2, 20, 21]))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert m.loop_feedback_gate.grad is not None and m.loop_feedback_gate.grad.abs().sum() > 0
    assert m.lm_head.weight.grad.abs().sum() > 0  # from the LM loss itself


def test_lens_does_not_train_the_head_or_the_token_table():
    """The naming loss reaches the loop's state, never the shared output head or final norm directly (the token table
    still learns through the backbone, as for any loss)."""
    torch.manual_seed(4)
    ids = text_row(S=48)
    m = make(**TEXT, **LOOP4, message_loop_name_lens=1.0, message_loop_feedback="soft").train()
    with torch.no_grad():
        m.loop_feedback_gate.fill_(0.5)
    tgt = _targets([40, 1, 10, 11], [40, 2, 20, 21])
    m(ids, labels=ids.clone(), read_targets=tgt).loss.backward()
    with_lens = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None and
                 n in ("lm_head.weight", "final_norm.weight")}
    m.zero_grad()
    m.config.message_loop_name_lens = 0.0
    gate = m.loop_feedback_gate
    # same forward, the lens contributions removed: the shared weights' gradients must not change by the naming loss
    m(ids, labels=ids.clone(), read_targets=tgt).loss.backward()
    for n, g in with_lens.items():
        assert torch.allclose(dict(m.named_parameters())[n].grad, g, atol=1e-6), n
    assert gate is m.loop_feedback_gate


def test_long_spans_are_not_named():
    torch.manual_seed(5)
    ids = text_row(S=48)
    m = make(**TEXT, **LOOP4, message_loop_name_lens=1.0).train()
    with torch.no_grad():
        base = m(ids, labels=ids.clone()).loss
        long = m(ids, labels=ids.clone(), read_targets=_targets([40, 1, 5, 12])).loss  # 8 tokens: a sentence
    assert torch.allclose(base, long)


def test_eval_runs_the_feedback_without_targets():
    m = make(**TEXT, **LOOP4, message_loop_name_lens=0.3, message_loop_feedback="soft").eval()
    with torch.no_grad():
        m.loop_feedback_gate.fill_(0.1)
        out = m(text_row(S=48))
    assert torch.isfinite(out.logits).all()


def test_text_checks_arch_e33q_is_e33r_plus_naming_and_feedback():
    from evaluation.text_checks import ARCHES, count_params, model_args

    r = model_args("e33r_plain", "screen", 1152)
    q = model_args("e33q_plain", "screen", 1152)
    assert {k for k in q if q[k] != r.get(k)} == {"message_loop_name_lens", "message_loop_feedback"}
    assert ARCHES["e33q_plain"].has_notebook and count_params(q) >= count_params(r)


def test_writer_can_read_the_decoders_whole_input_layer():
    """lm_writer_input="embed": the writer starts from the same encoding as the reader's tokens; default unchanged."""
    torch.manual_seed(3)
    m = make(**TEXT, **LOOP4, lm_writer_input="embed").train()
    assert m.memory_writer.in_proj.in_features == m.config.hidden_size
    ids = text_row(S=48)
    out = m(ids, labels=ids.clone())
    out.loss.backward()
    assert torch.isfinite(out.loss) and m.embed.up0.weight.grad is not None
    d = make(**TEXT, **LOOP4)
    assert d.memory_writer.in_proj.in_features == d.config.token_embedding_dim
