"""Notes-budget exam hooks (scripts/notes_budget_exam.py): a latent-memory model split at its notebook, nothing retrained.

`_note_capture` records each read's attention share per note, `_note_keep` hides notes, `_notes_override` makes the
read use another forward's notes (agent B reads agent A's notebook over its own text).
"""
import torch

from nn.perceiver_ar_lm import capture_note_mass, dense_message_mask
from tests.test_e33d_note_age import DEV, LOOP, TOL, WIDE, _read_setup
from tests.test_text_memory import TEXT, make, text_row


def _model(seed=4):
    return make(seed=seed, **TEXT, **LOOP, **WIDE, message_raw_window=8).to(DEV).eval()


def test_captured_mass_is_the_reads_own_softmax():
    ctx, t, pos, (B, S, h, g, dh, nb) = _read_setup(S=40)
    qp = torch.tensor([[5, 20, 39]], device=DEV)
    ctx.capture = {"q_pos": qp}
    capture_note_mass(t["q"], t["k"], t["kb"], ctx, None, q_slot=t["q_un"])
    got = ctx.capture["mass"][0]
    rep = h // g
    raw = torch.einsum("bqhd,bkhd->bhqk", t["q"], t["k"].repeat_interleave(rep, 2)) / dh ** 0.5
    notes = torch.einsum("bqhd,bkhd->bhqk", t["q_un"], t["kb"].repeat_interleave(rep, 2)) / dh ** 0.5
    mask = dense_message_mask(S, ctx, None, DEV)
    w = torch.softmax(torch.cat([raw, notes], -1).masked_fill(~mask, float("-inf")), -1).nan_to_num(0.0)
    ref = w[..., S:].sum(1)[:, qp[0]]  # [B,P,nb]
    assert got.shape == (B, 3, nb)
    assert torch.allclose(got, ref, atol=TOL)


def test_capture_in_a_looped_forward_gives_one_entry_per_round():
    m = _model()
    ids = text_row(S=48).to(DEV)
    with torch.no_grad():
        base = m(ids).logits
        m._note_capture = {"q_pos": torch.tensor([[47]], device=DEV)}
        out = m(ids).logits
    mass = m._note_capture["mass"]
    assert len(mass) == m.loop_rounds
    assert torch.allclose(out, base, atol=TOL)  # capturing changes nothing
    assert all(float(x.min()) >= 0 and float(x.sum()) <= m.config.num_attention_heads + 1e-4 for x in mass)
    assert float(mass[0].sum()) > 0
    nb = m._last_notes[0].shape[1]
    assert mass[0].shape == (1, 1, nb)


def test_keep_all_is_the_model_and_keep_none_is_notebook_off():
    m = _model()
    ids = text_row(S=48).to(DEV)
    with torch.no_grad():
        base = m(ids).logits
        m._note_capture = {"q_pos": torch.tensor([[47]], device=DEV)}
        m(ids)
        nb = m._last_notes[0].shape[1]
        m._note_capture = None
        m._note_keep = torch.ones(1, nb, dtype=torch.bool, device=DEV)
        assert torch.allclose(m(ids).logits, base, atol=TOL)
        m._note_keep = torch.zeros(1, nb, dtype=torch.bool, device=DEV)
        off = m(ids).logits
        m._note_keep = None
        with m.message_override("none"):
            ref = m(ids).logits
    assert torch.allclose(off, ref, atol=TOL)
    assert not torch.allclose(off, base, atol=1e-3)


def test_a_hidden_note_is_never_read_but_still_scored():
    """A hidden note's content cannot reach the logits (garbage in it changes nothing), while a capture still scores
    it — the request score agent A computes over its whole notebook."""
    m = _model()
    ids = text_row(S=48).to(DEV)
    with torch.no_grad():
        m._note_capture = {"q_pos": torch.tensor([[47]], device=DEV)}
        m(ids)
        full = torch.stack(m._note_capture["mass"]).sum(0)[0, 0]
        k, v, sp = m._last_notes
        nb = full.shape[0]
        top = int(full.argmax())
        keep = torch.ones(1, nb, dtype=torch.bool, device=DEV)
        keep[0, top] = False
        m._note_keep = keep
        m._note_capture = {"q_pos": torch.tensor([[47]], device=DEV)}
        hidden = m(ids).logits
        scored = torch.stack(m._note_capture["mass"]).sum(0)[0, 0]
        m._note_capture = None
        v2 = v.clone()
        v2[:, top] = 100.0
        m._notes_override = (k, v2, sp)
        garbage = m(ids).logits
    m._note_keep, m._notes_override = None, None
    assert float(full[top]) > 0 and float(scored[top]) > 0
    assert torch.allclose(hidden, garbage, atol=TOL)


def test_b_reads_a_notes_over_its_own_text():
    """Own notes as an override reproduce the model; with A's notes, B's answer position depends on A's text only
    through the notes: two different texts for B, same notes → the notes' effect is the same object."""
    m = _model()
    a = text_row(S=48, seed=1).to(DEV)
    b = text_row(S=48, seed=2).to(DEV)
    with torch.no_grad():
        m._note_capture = {"q_pos": torch.tensor([[47]], device=DEV)}
        base_a = m(a).logits
        notes_a = m._last_notes
        m(b)
        notes_b = m._last_notes
        m._note_capture = None
        m._notes_override = notes_a
        again = m(a).logits
        b_with_a = m(b).logits
        m._notes_override = notes_b
        b_with_b = m(b).logits
        m._notes_override = None
        b_plain = m(b).logits
    assert torch.allclose(again, base_a, atol=TOL)
    assert torch.allclose(b_with_b, b_plain, atol=TOL)
    assert not torch.allclose(b_with_a[0, -1], b_plain[0, -1], atol=1e-3)


def test_override_with_the_wrong_geometry_is_refused():
    m = _model()
    with torch.no_grad():
        m._note_capture = {"q_pos": torch.tensor([[47]], device=DEV)}
        m(text_row(S=48).to(DEV))
        notes = m._last_notes
        m._note_capture = None
        m._notes_override = (notes[0][:, :-1], notes[1][:, :-1], notes[2][:, :-1])
        try:
            m(text_row(S=48).to(DEV))
            raised = False
        except ValueError:
            raised = True
    m._notes_override = None
    assert raised
