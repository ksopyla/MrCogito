"""E35 — latent messages between agents (nn/latent_messages.py): memory, writing and sending kept apart.

Spec: docs/experiments_specs/ahead/E35_latent_messages.md.
"""
import json
import os

import pytest
import torch

from nn.latent_messages import A2AConfig, A2AModel, LatentMemory, _newer_match_logcount, build_a2a_from_checkpoints, split_rows
from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_latent_memory import cfg, lm_cfg
from tests.test_text_memory import TEXT

BOS, EOS, PAD = 1, 2, 0


def sender_cfg():
    return lm_cfg(**TEXT).to_dict()


def receiver_cfg():
    return cfg(par_mode="dense", ngram_orders=(), value_embed_layers=()).to_dict()


def make(seed=0, **kw):
    torch.manual_seed(seed)
    m = A2AModel(A2AConfig(sender_config=sender_cfg(), receiver_config=receiver_cfg(), msg_slots=4, tx_layers=2,
                           tx_heads=4, tx_dim=32, tx_ffn=64, **kw))
    for blk in m.transmitter.blocks:  # warm the zero-initialised outputs so every path is visible
        for lin in (blk.wo, blk.so, blk.f_down):
            lin.weight.data.normal_(0, 0.2)
    return m.eval()


def a2a_row(doc_len=40, recv_len=10, seed=1):
    g = torch.Generator().manual_seed(seed)
    doc = torch.randint(3, 80, (doc_len,), generator=g)
    recv = torch.randint(3, 80, (recv_len,), generator=g)
    return torch.cat([torch.tensor([BOS]), doc, torch.tensor([EOS, BOS]), recv, torch.tensor([EOS])])[None]


def test_split_rows_finds_the_document_and_the_receiver_text():
    r = a2a_row(12, 5)
    story = torch.tensor([[BOS, 5, 6, 7, EOS, PAD]])
    ids = torch.cat([r, torch.cat([story, story.new_full((1, r.shape[1] - 6), PAD)], 1)])
    mask = (ids != PAD).long()
    d, dm, rv, rm = split_rows(ids, mask, BOS, EOS, PAD)
    assert dm[0].sum() == 12 and torch.equal(d[0, :12], r[0, 1:13])
    assert rm[0].sum() == 7 and rv[0, 0] == BOS            # BOS recv… EOS
    assert dm[1].sum() == 0 and rm[1].sum() == 5            # a story row is receiver-only


def test_prefix_embeds_are_read_by_the_receiver_and_get_no_logits():
    torch.manual_seed(0)
    r = PerceiverARLM(cfg(par_mode="dense", ngram_orders=(), value_embed_layers=())).eval()
    ids = torch.randint(3, 80, (1, 9))
    p1, p2 = torch.randn(1, 3, 32), torch.randn(1, 3, 32)
    with torch.no_grad():
        a, b = r(ids, prefix_embeds=p1).logits, r(ids, prefix_embeds=p2).logits
        plain = r(ids).logits
    assert a.shape == plain.shape == (1, 9, r.config.vocab_size)
    assert not torch.allclose(a, b)                         # the message changes what B says
    loss = r(ids, labels=ids, prefix_embeds=p1).loss
    assert torch.isfinite(loss)


def test_only_the_message_crosses_from_a_to_b():
    """The receiver's output depends on the document only through the message: the same message gives the same
    answer whatever the document, and a different document gives a different message."""
    m = make()
    r1, r2 = a2a_row(seed=1), a2a_row(seed=2)
    r2[0, -11:] = r1[0, -11:]  # same receiver text, different document
    with torch.no_grad():
        mem1, _ = m.write(r1[:, 1:41])
        mem2, _ = m.write(r2[:, 1:41])
        msg1, msg2 = m.send(mem1), m.send(mem2)
        assert not torch.allclose(msg1, msg2)
        recv = r1[:, 42:]
        assert torch.allclose(m.answer(msg1, recv).logits, m.answer(msg1, recv).logits)
        assert not torch.allclose(m.answer(msg1, recv).logits, m.answer(msg2, recv).logits)
        # the packed forward is the same chain
        full = m(r1).logits
        assert torch.allclose(full[0, 42:], m.answer(msg1, recv).logits[0], atol=1e-5)
        assert float(full[0, :42].abs().sum()) == 0.0


def test_memory_grows_with_the_document_and_the_message_does_not():
    m = make()
    with torch.no_grad():
        short, _ = m.write(a2a_row(24)[:, 1:25])
        long, _ = m.write(a2a_row(120)[:, 1:121])
    assert long.n_entries > 2 * short.n_entries
    assert m.send(short).shape == m.send(long).shape == (1, 4, 32)


def test_empty_memory_reads_the_nothing_entry():
    m = make()
    with torch.no_grad():
        msg = m.send(m.empty_memory(2, "cpu"))
    assert msg.shape == (2, 4, 32) and torch.isfinite(msg).all()


def test_order_count_charges_older_matches_only():
    # one slot, one head, 3 pages × 2 entries; pages 0 and 2 match (score 5), page 1 does not (score 0)
    logit = torch.tensor([[[[5.0, 5.0, 0.0, 0.0, 5.0, 5.0]]]])
    valid = torch.ones(1, 6, dtype=torch.bool)
    lc = _newer_match_logcount(logit, valid, per_page=2, margin=0.5, sharp=4.0)[0, 0, 0]
    assert float(lc[4]) == 0.0 and float(lc[5]) == 0.0      # newest match pays nothing
    assert float(lc[0]) > 0.7                                # older match pays for the 2 newer matches
    assert float(lc[2]) > 0.7                                # (non-matching entries are low anyway)


def test_training_loss_reaches_every_part():
    m = make().train()
    ids = torch.cat([a2a_row(40, 10, 1), a2a_row(40, 10, 3)])
    m(ids, labels=ids).loss.backward()
    for name, mod in (("writer", m.writer), ("transmitter", m.transmitter), ("receiver", m.receiver)):
        g = sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
        assert g > 0, name
    assert m.sender_tok.weight.grad is not None


def test_story_only_batch_still_trains(tmp_path):
    m = make().train()
    ids = torch.tensor([[BOS, 5, 6, 7, 8, EOS]])
    loss = m(ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)


def test_warm_start_from_a_notebook_and_a_dense_checkpoint(tmp_path):
    torch.manual_seed(0)
    s = PerceiverARLM(lm_cfg(**TEXT))
    r = PerceiverARLM(cfg(par_mode="dense", ngram_orders=(), value_embed_layers=()))
    s.save_pretrained(tmp_path / "s")
    r.save_pretrained(tmp_path / "r")
    m = build_a2a_from_checkpoints(str(tmp_path / "s"), str(tmp_path / "r"), msg_slots=4, tx_dim=32, tx_heads=4,
                                   tx_ffn=64)
    assert torch.equal(m.sender_tok.weight, s.embed.tok.weight)
    assert torch.equal(m.writer.q, s.memory_writer.q)
    assert torch.equal(m.receiver.lm_head.weight, r.lm_head.weight)
    m.save_pretrained(tmp_path / "a2a")
    back = A2AModel.from_pretrained(tmp_path / "a2a")
    ids = a2a_row()
    with torch.no_grad():
        assert torch.allclose(back.eval()(ids).logits, m.eval()(ids).logits, atol=1e-5)
    assert json.load(open(tmp_path / "a2a" / "config.json"))["checkpoint_family"] == "a2a"
