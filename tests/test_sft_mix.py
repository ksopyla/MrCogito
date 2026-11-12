"""E37 instruction tuning: the mix builder (scripts/build_sft_mix.py), loss masks in the token-shard dataset
(data/bin_token_dataset.py), the collator and model contract, scripts/lsa_orig.py's labels, scripts/run_e33n.py --phase sft.

Real conversations: tests/data/smol_smoltalk_slice.json (38 rows of smol-smoltalk's test split); LongAlign rows are fake.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from data.bin_token_dataset import BinRowDataset, load_bin_manifest, manifest_has_masks, open_shard, write_shard
from data.data_collators import DataCollatorForCausalLM

ROOT = Path(__file__).resolve().parent.parent
EOT = 50256
S = 512
SLICE = json.loads((ROOT / "tests/data/smol_smoltalk_slice.json").read_text())["rows"]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("openai-community/gpt2", local_files_only=True)
    except Exception as e:  # pragma: no cover - no cached GPT-2 tokenizer
        pytest.skip(f"GPT-2 tokenizer not in the HF cache: {e}")


@pytest.fixture(scope="module")
def bsm():
    from scripts import build_sft_mix

    return build_sft_mix


def _lsa_orig():
    """scripts/lsa_orig.py imports only the standard library at module level (their venv is not needed here)."""
    spec = importlib.util.spec_from_file_location("lsa_orig_under_test", ROOT / "scripts/lsa_orig.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _words(n, seed):
    rng = np.random.default_rng(seed)
    vocab = ["river", "stone", "lamp", "orbit", "violet", "seven", "harbor", "maple", "quartz", "echo"]
    return " ".join(vocab[i] for i in rng.integers(0, len(vocab), n))


def _fake_longalign(n_words, seed):
    return {"source": "longalign", "messages": [
        {"role": "user", "content": f"Read the document and answer.\n\n{_words(n_words, seed)}\n\nWhich word is first?"},
        {"role": "assistant", "content": f"The first word is {_words(1, seed)}."}]}


def _docs(lengths, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in lengths:
        out += [EOT] + rng.integers(0, 50000, n - 1).tolist()
    return out


def _segments(row):
    """Split a packed row at document starts (a run of <|endoftext|> is one document)."""
    row = list(row)
    starts = [t for t in range(len(row)) if row[t] == EOT and (t == 0 or row[t - 1] != EOT)]
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(row)])]


@pytest.fixture(scope="module")
def mix(tmp_path_factory, tok, bsm):
    """One build: 30 real train rows + fake LongAlign (4 fit, 2 too long) + one row with a literal <|endoftext|>;
    8 real test rows + 1 fake LongAlign; fake LongDocs-lite replay shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path_factory.mktemp("sft")
    raw = d / "raw"
    raw.mkdir()
    train = SLICE[:30] + [_fake_longalign(n, i) for i, n in enumerate((150, 220, 300, 380, 700, 900))]
    train.append({"source": "smol-magpie-ultra-short", "messages": [
        {"role": "user", "content": "What does the string <|endoftext|> mean?"},
        {"role": "assistant", "content": "It is GPT-2's <|endoftext|> marker, written out as text."}]})
    test = SLICE[30:] + [_fake_longalign(200, 99)]
    pq.write_table(pa.Table.from_pylist(train), raw / "train-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist(test), raw / "test-00000-of-00001.parquet")
    ld = d / "longdocs"
    ld.mkdir()
    write_shard(ld / "gutenberg_train_000000.bin", _docs([1500, 2100, 400, 1300], 1))
    write_shard(ld / "arxiv_train_000000.bin", _docs([1100, 2600], 2))
    tokdir = d / "tokenizer_gpt2"
    tok.save_pretrained(tokdir)
    out = d / "mix"
    argv = ["build_sft_mix.py", "--out", str(out), "--tokenizer", str(tokdir), "--seq_len", str(S),
            "--tokens", str(7 * S), "--seed", "42", "--parquet_dir", str(raw),
            "--replay_glob", str(ld / "*_train_*.bin"), "--replay_share", "0.3", "--test_rows", "256",
            "--shard_tokens", str(3 * S)]
    old = sys.argv
    sys.argv = argv
    try:
        bsm.main()
    finally:
        sys.argv = old
    return {"out": out, "train": train, "test": test, "tokdir": tokdir, "raw": raw, "argv": argv,
            "manifest": json.loads((out / "manifest_sft.json").read_text()),
            "stats": json.loads((out / "stats.json").read_text())}


# ---------------------------------------------------------------- format and packing


def test_chat_format_pieces_and_mask(tok, bsm):
    fmt = bsm.ChatFormat(tok)
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi <|endoftext|> there"},
            {"role": "assistant", "content": "Hello."}]
    (ids, mask), = fmt.encode_batch([msgs])
    t = lambda x: tok(x, add_special_tokens=False, split_special_tokens=True)["input_ids"]  # noqa: E731
    want, want_mask = [EOT], [0]
    for m in msgs:
        h, c, e = t(f"<|{m['role']}|>\n"), t(m["content"]), t("\n<|end|>\n")
        a = int(m["role"] == "assistant")
        want += h + c + e
        want_mask += [0] * len(h) + [a] * (len(c) + len(e))
    assert ids.tolist() == want and mask.tolist() == want_mask
    assert ids.tolist().count(EOT) == 1  # a literal <|endoftext|> in a message is text, not the separator
    assert tok.decode(ids[mask == 1].tolist()) == "Hello.\n<|end|>\n"
    assert fmt.encode_batch([[{"role": "tool", "content": "x"}]]) == ["bad_role"]
    assert fmt.encode_batch([msgs[:2]]) == ["no_answer"]
    p_ids, p_mask = fmt.encode(msgs[:2], generation_prompt=True)
    assert p_ids[-len(t("<|assistant|>\n")):] == t("<|assistant|>\n") and not any(p_mask)


def test_best_fit_packing_and_prefix(bsm):
    rng = np.random.default_rng(0)
    lens = rng.integers(1, 400, 300).tolist()
    rows = bsm.pack_rows(lens, 512)
    assert sorted(i for r in rows for i in r) == list(range(300))
    assert all(sum(lens[i] for i in r) <= 512 for r in rows)
    assert len(rows) <= int(np.ceil(sum(lens) / 512)) + 2  # best-fit decreasing packs nearly full
    with pytest.raises(ValueError):
        bsm.pack_rows([513], 512)
    fixed, cand = [500, 300], rng.integers(10, 300, 200).tolist()
    k = bsm.select_prefix(fixed, cand, 512, 20)
    assert len(bsm.pack_rows(fixed + cand[:k], 512)) <= 20 < len(bsm.pack_rows(fixed + cand[: k + 1], 512))
    assert bsm.select_prefix([400] * 5, cand, 512, 3) == 0  # the fixed items alone already overflow


# ---------------------------------------------------------------- the built mix


def test_shards_and_manifest(mix):
    m, out = mix["manifest"], mix["out"]
    assert (m["seq_len"], m["rule"], m["eot"], m["seed"]) == (S, "stream", EOT, 42)
    smol, replay = m["train"]
    assert smol["name"] == "smol_smoltalk" and smol["weight"] == pytest.approx(0.7)  # 0.3: replay rows at 7 rows
    assert replay["name"] == "longdocs_replay" and replay["weight"] == pytest.approx(0.3)
    assert replay["rule"] == "doc" and "masks" not in replay  # the decay's rows, loss on every token
    (ev,) = m["eval"]
    assert ev["name"] == "smol_smoltalk_test" and ev["rows"] == 256
    assert manifest_has_masks(m)
    for src in (smol, ev):
        assert len(src["files"]) == len(src["masks"])
        for f, k in zip(src["files"], src["masks"]):
            assert Path(f).parent == out and not Path(k).name.endswith(".bin")  # a *_*.bin glob never takes a mask
            n = len(open_shard(f))
            assert n % S == 0 and len(open_shard(k)) == n
    assert len(smol["files"]) >= 2  # several shards of whole rows (--shard_tokens 3 rows)


def _expected(fmt, convs):
    out = {}
    for c, r in zip(convs, fmt.encode_batch([c["messages"] for c in convs])):
        if not isinstance(r, str) and len(r[0]) <= S:
            out[tuple(r[0].tolist())] = (c, r[1])
    return out


def _check_rows(files, masks, expected, tok):
    seen = []
    for f, k in zip(files, masks):
        ids, msk = np.asarray(open_shard(f)).reshape(-1, S), np.asarray(open_shard(k)).reshape(-1, S)
        for row, mrow in zip(ids, msk):
            for a, b in _segments(row):
                seg = tuple(row[a:b].tolist())
                if all(x == EOT for x in seg):  # the padding tail
                    assert b == S and not mrow[a:b].any()
                    continue
                assert seg in expected, "a row holds something other than whole conversations"
                conv, want_mask = expected[seg]
                assert mrow[a:b].tolist() == want_mask.tolist()
                answer = "".join(m["content"] + "\n<|end|>\n" for m in conv["messages"] if m["role"] == "assistant")
                got = tok.decode(row[a:b][mrow[a:b] == 1].tolist(), clean_up_tokenization_spaces=False)
                assert got == answer  # mask 1 exactly on the assistant content and its \n<|end|>\n
                seen.append(seg)
    assert len(seen) == len(set(seen)), "a conversation was packed twice"
    return seen


def test_rows_hold_whole_conversations_with_aligned_masks(mix, tok, bsm):
    fmt = bsm.ChatFormat(tok)
    smol, ev = mix["manifest"]["train"][0], mix["manifest"]["eval"][0]
    exp_train = _expected(fmt, mix["train"])
    seen = _check_rows(smol["files"], smol["masks"], exp_train, tok)
    long_fit = [k for k, (c, _) in exp_train.items() if c["source"] == "longalign"]
    assert len(long_fit) == 4 and set(long_fit) <= set(seen)  # every LongAlign conversation that fits
    st = mix["stats"]["train"]
    assert st["rows"] == 7 and st["row_tokens"] == 7 * S and st["conversations"] == len(seen)
    assert st["sources"]["longalign"]["kept"] == 4 and st["sources"]["longalign"]["dropped_too_long"] == 2
    assert 0 < st["answer_share_of_content"] < 1 and st["fill"] > 0.9
    exp_test = _expected(fmt, mix["test"])
    seen_test = _check_rows(ev["files"], ev["masks"], exp_test, tok)
    assert set(seen_test) == set(exp_test)  # the test split: every conversation that fits


def test_dataset_items_carry_labels(mix):
    tr, ev, m = load_bin_manifest(str(mix["out"] / "manifest_sft.json"))
    per = tr.rows_per_source
    assert per == {"smol_smoltalk": 7, "longdocs_replay": 3}  # 0.7 / 0.3 of 10 rows
    assert mix["stats"]["dataset"]["train_rows_per_source"] == per
    assert len(ev) == mix["stats"]["test"]["rows"]  # fewer test rows than the 256 asked for: all of them
    smol = tr.sources[0]
    for i in range(len(tr)):
        it = tr[i]
        k, r = tr.index[i]
        ids, doc, lab = it["input_ids"], it["doc_ids"], it["labels"]
        assert lab.shape == ids.shape and lab.dtype == torch.long
        starts = torch.zeros_like(ids, dtype=torch.bool)
        starts[1:] = doc[1:] != doc[:-1]
        if int(k) == 0:
            si, off = smol.rows[int(r)]
            mask = torch.from_numpy(np.asarray(smol.masks[int(si)][int(off): int(off) + S], dtype=np.int64))
            assert torch.equal(lab, torch.where(mask == 1, ids, torch.full_like(ids, -100)))
            assert not (mask.bool() & starts).any()
            eot_run = (ids == EOT) & torch.cat([torch.tensor([False]), ids[:-1] == EOT])
            assert torch.equal(doc[1:] - doc[:-1], ((ids == EOT) & ~eot_run)[1:].long())  # pad tail = 1 doc
        else:  # replay: every token but a document's first is a target
            assert torch.equal(lab, ids.masked_fill(starts, -100))
            assert doc.max().item() == 0  # the decay's rule: one document per row
    assert all("labels" in ev[i] for i in range(len(ev)))


def test_collator_keeps_labels_and_still_masks_doc_starts(mix):
    tr, _, _ = load_bin_manifest(str(mix["out"] / "manifest_sft.json"))
    tok = type("T", (), {"pad_token_id": EOT, "unk_token_id": None, "__len__": lambda self: 50257})()
    col = DataCollatorForCausalLM(tok, max_length=S, model_vocab_size=50257, preserve_precomputed_labels=True)
    items = [tr[i] for i in range(min(4, len(tr)))]
    b = col(items)
    assert torch.equal(b["labels"], torch.stack([x["labels"] for x in items]))
    assert torch.equal(b["doc_ids"], torch.stack([x["doc_ids"] for x in items]))
    assert b["attention_mask"].all()
    # a label left at a document start by some other producer is still ignored
    x = dict(items[0])
    lab = x["input_ids"].clone()
    x["labels"] = lab
    starts = (x["doc_ids"][1:] != x["doc_ids"][:-1]).nonzero().flatten() + 1
    assert len(starts) and (col([x])["labels"][0, starts] == -100).all()


def test_replay_share_of_rows(tmp_path):
    """BinRowDataset's weights → rows: weight w gives floor(w * total) rows, total = min(rows_i / w_i), so 0.9 / 0.1
    with enough replay rows uses all SFT rows and adds a ninth of them from the replay."""
    write_shard(tmp_path / "a.bin", np.full(90 * 64, 7, dtype=np.uint16))
    write_shard(tmp_path / "a.mask", np.ones(90 * 64, dtype=np.uint16))
    write_shard(tmp_path / "r.bin", _docs([64 * 40] * 3, 3))
    ds = BinRowDataset([{"name": "smol", "files": [str(tmp_path / "a.bin")], "masks": [str(tmp_path / "a.mask")],
                         "weight": 0.9},
                        {"name": "replay", "files": [str(tmp_path / "r.bin")], "weight": 0.1, "rule": "doc"}],
                       seq_len=64, rule="stream")
    assert ds.rows_per_source == {"smol": 90, "replay": 10}


def test_builder_refuses_a_replay_too_small_for_its_share(mix, tmp_path, bsm):
    """With 1 replay row, weights 0.7 / 0.3 would cut the 7 instruction rows to 2: the build stops instead."""
    write_shard(tmp_path / "one_train_000.bin", _docs([S + 10], 9))
    argv = list(mix["argv"])
    argv[argv.index("--out") + 1] = str(tmp_path / "mix")
    argv[argv.index("--replay_glob") + 1] = str(tmp_path / "one_train_*.bin")
    old = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit, match="too small"):
            bsm.main()
    finally:
        sys.argv = old
    assert "error" in json.loads((tmp_path / "mix" / "stats.json").read_text())


# ---------------------------------------------------------------- old manifests are untouched


def _old_item(ds, i):
    """data/bin_token_dataset.py's item before masks existed."""
    k, r = ds.index[i]
    src = ds.sources[int(k)]
    si, off = src.rows[int(r)]
    ids = torch.from_numpy(np.asarray(src.shards[int(si)][int(off): int(off) + ds.S], dtype=np.int64))
    doc = torch.cumsum((ids == ds.eot).long(), dim=0)
    return {"input_ids": ids, "doc_ids": doc - doc[0]}


def test_mask_free_manifest_items_unchanged(tmp_path):
    toks = _docs([30, 1, 1, 50, 12, 70], 4) + [EOT, EOT, EOT] + _docs([40], 5)  # runs of <|endoftext|> too
    write_shard(tmp_path / "a.bin", toks)
    write_shard(tmp_path / "b.bin", _docs([90, 33], 6))
    man = {"seq_len": 16, "rule": "stream", "seed": 3,
           "train": [{"name": "a", "files": [str(tmp_path / "a.bin")], "weight": 2.0},
                     {"name": "b", "files": [str(tmp_path / "b.bin")], "weight": 1.0}],
           "eval": [{"name": "a", "files": [str(tmp_path / "a.bin")], "rows": 4}]}
    (tmp_path / "m.json").write_text(json.dumps(man))
    tr, ev, _ = load_bin_manifest(str(tmp_path / "m.json"))
    assert not manifest_has_masks(man) and not tr.emit_labels and not ev.emit_labels
    for ds in (tr, ev):
        for i in range(len(ds)):
            it, old = ds[i], _old_item(ds, i)
            assert set(it) == {"input_ids", "doc_ids"}
            assert all(torch.equal(it[k], old[k]) and it[k].dtype == old[k].dtype for k in old)


def test_mask_validation(tmp_path):
    write_shard(tmp_path / "a.bin", _docs([64]))
    write_shard(tmp_path / "b.bin", _docs([64]))
    write_shard(tmp_path / "a.mask", np.ones(64, dtype=np.uint16))
    write_shard(tmp_path / "short.mask", np.ones(32, dtype=np.uint16))
    write_shard(tmp_path / "two.mask", np.full(64, 2, dtype=np.uint16))
    write_shard(tmp_path / "odd.bin", _docs([40]))
    write_shard(tmp_path / "odd.mask", np.ones(40, dtype=np.uint16))
    p = lambda n: str(tmp_path / n)  # noqa: E731
    bad = [
        ({"files": [p("a.bin"), p("b.bin")], "masks": [p("a.mask")]}, "mask shards for"),
        ({"files": [p("a.bin")], "masks": [p("short.mask")]}, "has 32 tokens"),
        ({"files": [p("a.bin")], "masks": [p("two.mask")]}, "other than 0/1"),
        ({"files": [p("odd.bin")], "masks": [p("odd.mask")]}, "not whole rows"),
        ({"files": [p("a.bin")], "masks": [p("a.mask")], "rule": "doc"}, "'stream' rule"),
    ]
    for spec, msg in bad:
        with pytest.raises(ValueError, match=msg):
            BinRowDataset([{"name": "x", **spec}], seq_len=16, rule="stream")


# ---------------------------------------------------------------- the two trainers


def _old_rows_to_batch(ids, doc):
    R, Sq = ids.shape
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    labels[:, :-1] = labels[:, :-1].masked_fill(doc[:, 1:] != doc[:, :-1], -100)
    return labels.reshape(-1)


def test_lsa_orig_labels(mix):
    lo = _lsa_orig()
    tr, _, _ = load_bin_manifest(str(mix["out"] / "manifest_sft.json"))
    items = [tr[i] for i in range(len(tr))]
    ids, doc = torch.stack([x["input_ids"] for x in items]), torch.stack([x["doc_ids"] for x in items])
    lab = torch.stack([x["labels"] for x in items])
    plain = lo.rows_to_batch(ids, doc)
    assert torch.equal(plain["labels"], _old_rows_to_batch(ids, doc))  # without labels: as before
    b = lo.rows_to_batch(ids, doc, lab)
    got = b["labels"].reshape(ids.shape)
    want = torch.full_like(ids, -100)
    want[:, :-1] = lab[:, 1:].masked_fill(doc[:, 1:] != doc[:, :-1], -100)
    assert torch.equal(got, want)
    assert (got[:, -1] == -100).all()  # a row's last token predicts nothing
    assert b["num_valid_tokens"] == int((want != -100).sum()) < plain["num_valid_tokens"]
    assert torch.equal(b["positions"], plain["positions"]) and torch.equal(b["input"], plain["input"])
    # position t is trained exactly where the dataset's label of t + 1 is an answer token (or replay text)
    assert torch.equal(got[:, :-1] != -100, lab[:, 1:] != -100)
    # a padding tail is one document: its positions count up from 0
    row = items[[i for i, x in enumerate(items) if x["input_ids"][-1] == EOT][0]]["input_ids"]
    tail = int((row.flip(0) == EOT).cumprod(0).sum())
    pos = plain["positions"].reshape(ids.shape)[[i for i, x in enumerate(items) if x["input_ids"][-1] == EOT][0]]
    assert pos[-tail:].tolist() == list(range(tail))


def test_lsa_orig_phase_numbers():
    lo = _lsa_orig()
    t, d, s = lo.phase_numbers("trunk"), lo.phase_numbers("decay"), lo.phase_numbers("sft")
    assert (t["steps"], t["warmup"], t["decay_steps"], t["tokens_per_rank"]) == (5722, 57, 0, 8 * 4096)
    assert (d["steps"], d["warmup"], d["decay_steps"], d["tokens_per_rank"]) == (763, 22, 740, 2 * 16384)
    assert (s["steps"], s["warmup"], s["decay_steps"], s["tokens_per_rank"]) == (763, 22, 740, 16384)
    assert set(s) == set(d) and lo.step_tokens("sft") == 16 * 16384 and lo.step_tokens("decay") == 128 * 4096
    assert lo.phase_numbers("sft", tokens=20e6)["steps"] == 76


def test_run_e33n_sft_command(tmp_path):
    from scripts.run_e33n import _final_eval_every, train_cmd

    man = tmp_path / "manifest_sft.json"
    man.write_text(json.dumps({"seq_len": 16384, "rule": "stream", "train": [], "eval": []}))
    cmd = train_cmd("lsa", "m100", "sft", data=tmp_path, out=tmp_path / "run", init="/r/decay/final", lr=1.5e-4,
                    sft_manifest=str(man))
    a = dict(zip(cmd[::1], cmd[1::1]))
    assert a["--bin_manifest"] == str(man) and a["--max_seq_length"] == "16384"
    assert int(a["--per_device_train_batch_size"]) * int(a["--gradient_accumulation_steps"]) == 16
    assert a["--max_steps"] == "763" and a["--warmup_steps"] == "22" and a["--learning_rate"] == "0.00015"
    assert json.loads(a["--lr_scheduler_kwargs"]) == {"num_decay_steps": 740, "decay_type": "1-sqrt",
                                                      "min_lr_ratio": 0.1}
    assert a["--model_name_or_path"] == "/r/decay/final" and a["--output_dir"] == str(tmp_path / "run" / "sft")
    assert a["--eval_strategy"] == "steps" and 763 % int(a["--eval_steps"]) == 0 and a["--eval_on_start"] == "True"
    assert _final_eval_every(763) == 109 and _final_eval_every(76) == 19 and _final_eval_every(7) == 1
    assert not (tmp_path / "run" / "manifest_trunk.json").exists()
    with pytest.raises(ValueError, match="needs"):
        train_cmd("lsa", "m100", "sft", data=tmp_path, out=tmp_path / "run", init="/r", sft_manifest=str(man))
    man.write_text(json.dumps({"seq_len": 4096}))
    with pytest.raises(ValueError, match="row length"):
        train_cmd("lsa", "m100", "sft", data=tmp_path, out=tmp_path / "run", init="/r", lr=1e-4,
                  sft_manifest=str(man))


def test_model_loss_counts_only_label_positions(mix):
    """PerceiverARLM shifts the labels itself and ignores -100: the loss is the mean CE over exactly the positions
    whose next-token label survives the mask."""
    from types import SimpleNamespace

    import torch.nn.functional as F

    from scripts.run_e33n import EOT as _EOT, VOCAB, model_args
    from training.concept_pretraining_args import ModelArguments
    from training.concept_pretraining_factories import _build_perceiver_ar_model

    tr, _, _ = load_bin_manifest(str(mix["out"] / "manifest_sft.json"))
    tok = type("T", (), {"pad_token_id": _EOT, "bos_token_id": _EOT, "eos_token_id": _EOT, "unk_token_id": None,
                         "__len__": lambda self: VOCAB})()
    col = DataCollatorForCausalLM(tok, max_length=S, model_vocab_size=VOCAB, preserve_precomputed_labels=True)
    smol_rows = [i for i in range(len(tr)) if int(tr.index[i][0]) == 0][:2]
    batch = col([tr[i] for i in smol_rows])
    fields = set(ModelArguments.__dataclass_fields__)
    ma = ModelArguments(**{k: v for k, v in model_args("dense", "tiny").items() if k in fields})
    ma.attn_backend, ma.use_liger, ma.z_loss = "sdpa", False, 0.0
    torch.manual_seed(0)
    model, _, _ = _build_perceiver_ar_model(tok, ma, SimpleNamespace(max_seq_length=S, tokenizer_name="gpt2"))
    model.eval()
    with torch.no_grad():
        loss = model(**batch).loss
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       doc_ids=batch["doc_ids"]).logits
        _, per, valid = model(**batch, return_per_token_loss=True)
    tgt = batch["labels"][:, 1:]
    manual = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
    assert torch.isfinite(loss) and torch.allclose(loss, manual, atol=1e-4)
    assert torch.equal(valid, tgt != -100) and int(valid.sum()) < valid.numel()
    full = col([{**tr[i], "labels": tr[i]["input_ids"]} for i in smol_rows])
    with torch.no_grad():
        assert not torch.allclose(model(**full).loss, loss)  # the mask changes what is trained
