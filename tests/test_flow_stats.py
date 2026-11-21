"""E33p instrumentation (plan §8 A, T14): the recorder is passive, every key is finite, the callback fires on log steps."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from nn.flow_stats import FlowRecorder
from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg, lm_cfg
from tests.test_lsa_read import LSA, DENSE
from tests.test_note_open import NOTES, PAGES, TIED
from tests.test_text_memory import TEXT
from training.flow_callback import FlowCallback, param_group

WINDOW = dict(**TEXT, message_loop_rounds=4, global_nope=True, message_raw_window=32, message_override="none")
E33 = dict(**TEXT, message_loop_rounds=4, global_nope=True, message_raw_window=32)
ARMS = {"window": (lm_cfg, WINDOW), "e33": (lm_cfg, E33), "notes_open": (lm_cfg, NOTES),
        "pages_open": (lm_cfg, PAGES), "dense": (cfg, DENSE), "lsa": (cfg, LSA)}


def make(name, seed=0):
    build, kw = ARMS[name]
    torch.manual_seed(seed)
    m = PerceiverARLM(build(**kw)).train()
    for layer in m.layers:
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def ids(S=200, B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 80, (B, S), generator=g)


def loss_and_grads(m, x, armed: bool):
    m.zero_grad()
    rec = FlowRecorder(n_sample=8, lens_tokens=32, buckets=((0, 32), (32, 96), (96, 200)))
    if armed:
        m.flow_recorder = rec
        rec.armed = True
    out = m(x, labels=x)
    out.loss.backward()
    m.flow_recorder = None
    return out.loss.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}, rec


@pytest.mark.parametrize("name", sorted(ARMS))
def test_recorder_is_passive_and_its_keys_are_finite(name):
    m = make(name)
    x = ids()
    l0, g0, _ = loss_and_grads(m, x, armed=False)
    l1, g1, rec = loss_and_grads(m, x, armed=True)
    assert torch.equal(l0, l1)
    assert g0.keys() == g1.keys()
    for n in g0:
        assert torch.equal(g0[n], g1[n]), n
    assert not rec.armed and all(l.flow is None for l in m.layers)
    out = rec.finish(m)
    assert out and all(math.isfinite(v) for v in out.values())
    keys = set(out)
    assert any(k.startswith("flow/L") and k.endswith("attn_rel") for k in keys)
    assert any(k.startswith("lens/L") for k in keys)
    assert any(k.startswith("read/r0/") and "mass_window" in k for k in keys)
    assert any("/r3" in k for k in keys)                                                 # loop rounds are tagged
    if name in ("window", "e33", "notes_open", "pages_open"):
        assert "writer/assign_entropy" in keys and "writer/empty_notes" in keys
    if name in ("notes_open", "pages_open", "lsa"):
        assert any(k.startswith("select/r0/cap_full") for k in keys)
        assert any(k.startswith("select/r1/overlap") for k in keys)
        assert any("mass_opened" in k for k in keys)
    if name == "notes_open":
        assert any(k.startswith("writer/kappa_ratio/r0") for k in keys)
        assert any(k.startswith("select/r0/skipped") for k in keys)
    if name == "dense":
        assert any("mass_beyond" in k for k in keys)
    if name == "lsa":
        assert any("mass_sinks" in k for k in keys)
    # masses of one bucket sum to 1 (each source counted once)
    for k in keys:
        if k.endswith("/entropy") and k.startswith("read/"):
            base = k[: -len("entropy")]
            tot = sum(v for kk, v in out.items() if kk.startswith(base + "mass_"))
            assert abs(tot - 1.0) < 1e-4, (k, tot)


def test_lens_of_the_last_layer_is_the_model_loss():
    m = make("e33").eval()
    x = ids(B=1)
    rec = FlowRecorder(n_sample=4, lens_tokens=64, buckets=((0, 200),))
    m.flow_recorder, rec.armed = rec, True
    with torch.no_grad():
        m(x, labels=x)
    stats = rec.finish(m)
    last = f"lens/L{len(m.layers) - 1}"
    if last not in stats:  # the last layer sits inside the loop: its final output is the last round's
        last = f"{last}/r{m.loop_rounds - 1}"
    # the model's mean loss covers all positions; recompute the last 64 from logits
    with torch.no_grad():
        lg = m(x, return_logits=True).logits[0, -65:-1].float()
    ref = torch.nn.functional.cross_entropy(lg, x[0, -64:])
    assert abs(stats[last] - float(ref)) < 1e-4


def test_param_groups():
    assert param_group("layers.3.attn.wq.weight") == "L3.attn"
    assert param_group("layers.3.mlp.up.weight") == "L3.ffn"
    assert param_group("module.memory_writer.wq.weight") == "writer"
    assert param_group("lm_head.weight") == "head" and param_group("embed.tok.weight") == "embed"


def test_callback_logs_on_log_steps_only():
    m = make("notes_open")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    cb = FlowCallback(every=2, recorder=FlowRecorder(n_sample=8, lens_tokens=32, buckets=((0, 200),)))
    x = ids()
    seen = []
    for step in range(3):
        st = SimpleNamespace(global_step=step)
        cb.on_step_begin(None, st, None, model=m)
        m(x, labels=x).loss.backward()
        cb.on_pre_optimizer_step(None, st, None, model=m)
        opt.step()
        cb.on_optimizer_step(None, st, None, model=m)
        opt.zero_grad()
        seen.append(cb.pop_flow())
    assert seen[0] and seen[2] and not seen[1]
    for k in ("grad/L2.attn/norm", "upd/L2.attn/ratio", "grad/writer/norm", "upd/head/ratio"):
        assert k in seen[0] and math.isfinite(seen[0][k]) and seen[0][k] > 0, k
    assert any(k.startswith("lens/") for k in seen[0])
