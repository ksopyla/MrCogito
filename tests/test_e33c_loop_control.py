"""E33c — each loop round reads its next relation from the question (nn/perceiver_ar_lm.py LoopControl, a MAC control
unit: Hudson & Manning 2018). Round r attends over the last W words, queried by the previous control, with a learned bias
on the offset from the previous round's pointer; the control vector joins round r's input through a gate from 0.

Note: docs/4_Research_Notes/e33_text_multihop_diagnosis_20261017.md §8, "Why 3-hop compose fails".
"""
from types import SimpleNamespace

import pytest
import torch

from nn.perceiver_ar_lm import analytic_param_count, loop_control_param_count
from tests.test_text_memory import TEXT, _targets, make, text_row

LOOP4 = dict(message_raw_window=8, message_loop_rounds=4)
QN = dict(**TEXT, **LOOP4, message_loop_read_aux=0.5, message_loop_name_lens=0.3)  # E33qn's recipe at test size
CTRL = dict(message_loop_control="question", message_loop_control_window=16, message_loop_control_dim=16,
            message_loop_control_heads=2)
TGT = ([40, 1, 10, 12], [40, 2, 20, 21], [40, 0, -1, -1])
OWN = ("to_k", "to_v", "q_from_c", "q_from_x", "out", "c0", "ptr_bias")


def _e33c_from_e33qn():
    qn = make(**QN)
    c = make(**QN, **CTRL)
    missing, unexpected = c.load_state_dict(qn.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("loop_control.") for k in missing)
    return qn, c


def _own_params(m):
    return {n: getattr(m.loop_control, n) if n in ("c0", "ptr_bias") else getattr(m.loop_control, n).weight for n in OWN}


# ------------------------------------------------------------------ default off
def test_default_is_off_and_builds_nothing():
    a, b = make(**QN), make(**QN, message_loop_control="none")
    assert a.config.message_loop_control == "none" and a.loop_control is None and b.loop_control is None
    assert a.state_dict().keys() == b.state_dict().keys()
    ids = text_row(S=48)
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)
    with pytest.raises(ValueError):  # a control needs the loop
        make(**TEXT, message_raw_window=8, message_loop_control="question")
    with pytest.raises(ValueError):
        make(**QN, message_loop_control="bogus")
    with pytest.raises(ValueError):
        make(**QN, **{**CTRL, "message_loop_control_heads": 3})


# ------------------------------------------------------------------ warm start = E33qn at step 0
def test_e33c_equals_e33qn_at_init():
    """Loaded with E33qn's weights, E33c gives the same logits and the same training loss (naming lens and read target
    on) at step 0: only the gate (zero) separates them."""
    qn, c = _e33c_from_e33qn()
    assert not c.loop_control.gate.any()
    ids = text_row(S=48)
    with torch.no_grad():
        assert torch.equal(c(ids).logits, qn(ids).logits)
        qn.train(), c.train()
        a = qn(ids, labels=ids.clone(), read_targets=_targets(*TGT)).loss
        b = c(ids, labels=ids.clone(), read_targets=_targets(*TGT)).loss
    assert torch.equal(a, b)


def test_exam_probe_off_is_the_model_without_control():
    """--loop_control off sets model.loop_control = None: an opened control changes the logits, removing it gives the
    E33qn model on the same weights back."""
    qn, c = _e33c_from_e33qn()
    ids = text_row(S=48)
    with torch.no_grad():
        c.loop_control.gate.fill_(0.3)
        assert not torch.allclose(c(ids).logits, qn(ids).logits, atol=1e-4)
        c.loop_control = None
        assert torch.equal(c(ids).logits, qn(ids).logits)


# ------------------------------------------------------------------ the control read learns
def test_control_read_gets_gradient():
    """Step 0: only the gate gets gradient (the control is multiplied by it). Once the gate has moved, the gradient
    reaches every part of the control read: keys, values, both query parts, the start state, the pointer bias, the
    output projection."""
    _, c = _e33c_from_e33qn()
    c.train()
    ids = text_row(S=48)
    c(ids, labels=ids.clone(), read_targets=_targets(*TGT)).loss.backward()
    g = c.loop_control.gate.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    for n, p in _own_params(c).items():
        assert p.grad is None or not p.grad.any(), n
    c.zero_grad()
    with torch.no_grad():
        c.loop_control.gate.fill_(0.1)  # as after the first optimizer steps
    c(ids, labels=ids.clone(), read_targets=_targets(*TGT)).loss.backward()
    for n, p in _own_params(c).items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n


def test_naming_loss_alone_reaches_the_control():
    """The per-round naming loss (E33qn's training signal) trains the control read: its gradient is not only the LM's."""
    _, c = _e33c_from_e33qn()
    c.train()
    with torch.no_grad():
        c.loop_control.gate.fill_(0.1)
    ids = text_row(S=48)
    term = (c(ids, labels=ids.clone(), read_targets=_targets(*TGT)).loss
            - c(ids, labels=ids.clone()).loss)  # naming lens + read target only
    g = torch.autograd.grad(term, [c.loop_control.to_k.weight, c.loop_control.q_from_c.weight])
    assert all(float(x.abs().sum()) > 0 for x in g)


# ------------------------------------------------------------------ what the control can read
def _capture(m, ids, q, **kw):
    m.eval()
    m._control_capture = {"q_pos": torch.tensor([[q]])}
    with torch.no_grad():
        m(ids, **kw)
    cap, m._control_capture = m._control_capture, None
    return cap


def test_control_reads_only_its_window_and_document():
    _, c = _e33c_from_e33qn()
    with torch.no_grad():
        c.loop_control.gate.fill_(0.3)
    W = c.loop_control.W
    ids = text_row(S=48)
    cap = _capture(c, ids, 5)  # slots before the row start (keys −10 … −1) are empty
    assert len(cap["control_attn"]) == 4
    for a in cap["control_attn"]:
        assert not a[0, 0, :, : W - 1 - 5].any() and a[0, 0, :, W - 1 - 5:].sum(-1).allclose(torch.ones(2))
    doc = torch.zeros(1, 48, dtype=torch.long)
    doc[:, 30:] = 1
    cap = _capture(c, ids, 36, doc_ids=doc)  # keys 21 … 36: only 30 … 36 are in the question's document
    for a, p in zip(cap["control_attn"], cap["control_ptr"]):
        assert not a[0, 0, :, : W - 1 - 6].any()
        assert (p >= 0).all() and (p <= 6 + 1e-5).all()


def test_control_keeps_the_model_causal():
    _, c = _e33c_from_e33qn()
    with torch.no_grad():
        c.loop_control.gate.fill_(0.5)
    ids = text_row(S=48)
    other = ids.clone()
    other[0, 30] = (other[0, 30] + 7) % 80 + 3
    with torch.no_grad():
        a, b = c(ids).logits, c(other).logits
    assert torch.allclose(a[:, :30], b[:, :30], atol=1e-5)
    assert not torch.allclose(a[:, 30:], b[:, 30:], atol=1e-4)


def test_pointer_steps_by_the_learned_offset_every_round():
    """One rule, applied once per round: with content scores off and a bias that picks "3 tokens left of the previous
    pointer", round r's pointer sits 3r tokens left of the question's end — the round count never enters the rule."""
    _, c = _e33c_from_e33qn()
    ctl = c.loop_control
    W = ctl.W
    with torch.no_grad():
        ctl.to_k.weight.zero_()  # no content: the scores are the pointer bias alone
        ctl.ptr_bias.fill_(-30.0)
        ctl.ptr_bias[:, W - 1 - 3] = 0.0  # δ = key − pointer = −3
        ctl.gate.fill_(0.3)
    c._loop_rounds_override = 5  # one round more than trained: the rule still applies
    cap = _capture(c, text_row(S=48), 40)
    ptr = torch.stack([p[0, 0] for p in cap["control_ptr"]])  # [rounds, heads]
    assert torch.allclose(ptr, torch.tensor([3.0, 6.0, 9.0, 12.0, 15.0])[:, None].expand(5, 2), atol=1e-3)


# ------------------------------------------------------------------ size, arch card, warm start path
def test_param_count_and_text_checks_arch():
    from evaluation.text_checks import ARCHES, ENV_OF, count_params, fit_params, model_args

    c = make(**QN, **CTRL)
    n_ctrl = sum(p.numel() for p in c.loop_control.parameters())
    assert n_ctrl == loop_control_param_count(c.config)
    assert analytic_param_count(c.config).total - analytic_param_count(make(**QN).config).total == n_ctrl
    qn = model_args("e33qn_plain", "screen", 1152)
    e33c = model_args("e33c_plain", "screen", 1152)
    keys = {k for k in e33c if e33c[k] != qn.get(k)}
    assert keys == {"message_loop_control", "message_loop_control_window", "message_loop_control_dim",
                    "message_loop_control_heads", "message_loop_control_words"}
    assert keys <= set(ENV_OF) and ARCHES["e33c_plain"].has_notebook
    # 30M screen: the FFN width stays 1152, so a warm start from E33q / E33qn loads every weight
    assert fit_params("e33c_plain", "screen")[0]["intermediate_size"] == 1152
    assert count_params(e33c) - count_params(qn) == 3 * 576 * 128 + 128 * 128 + 128 * 576 + 128 + 4 * 96 + 576


def test_warm_start_from_e33q_through_the_factory(tmp_path):
    """The run plan's path: an E33q checkpoint (round feedback on) into E33c with WARM_START_SKIP=loop_feedback_gate.
    Every other weight loads; at step 0 E33c is E33q with its feedback removed (the exam's --loop_feedback off)."""
    from safetensors.torch import save_file

    from evaluation.text_checks import model_args
    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    fields = set(ModelArguments.__dataclass_fields__)
    tok = type("T", (), {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "__len__": lambda self: 256})()
    data = SimpleNamespace(max_seq_length=512, tokenizer_name="text-checks")

    def build(arch, **kw):
        ma = ModelArguments(**{k: v for k, v in {**model_args(arch, "smoke"), **kw}.items() if k in fields})
        ma.attn_backend, ma.use_liger = "sdpa", False
        return _build_perceiver_ar_model(tok, ma, data)[0].eval()

    torch.manual_seed(0)
    src = build("e33q_plain")
    with torch.no_grad():
        src.loop_feedback_gate.fill_(0.05)  # a trained E33q's gate is not 0
        for layer in src.layers:
            layer.attn.wo.weight.normal_(0, 0.2)
            layer.mlp.down.weight.normal_(0, 0.2)
    (tmp_path / "final").mkdir()
    save_file({k: v.contiguous() for k, v in src.state_dict().items()}, str(tmp_path / "final" / "model.safetensors"))
    with pytest.raises(ValueError, match="warm start mismatch"):  # the feedback gate has no place in E33c
        build("e33c_plain", model_name_or_path=str(tmp_path / "final"))
    c = build("e33c_plain", model_name_or_path=str(tmp_path / "final"), warm_start_skip="loop_feedback_gate")
    assert c.loop_feedback_gate is None and c.loop_control is not None and not c.loop_control.gate.any()
    ids = torch.randint(3, 250, (1, 300))
    with torch.no_grad():
        src.loop_feedback_gate = None
        assert torch.equal(c(ids).logits, src(ids).logits)
