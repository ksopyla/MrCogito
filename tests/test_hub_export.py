"""Hub releases that load with `from_pretrained` (docs/engineering_specs/hub_from_pretrained_loading.md, 2026-11-18).

`scripts/export_hub_model.py` writes a folder whose flat `modeling_perceiver_ar.py` / `configuration_perceiver_ar.py`
load with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`; `nn/perceiver_ar_hub.py` is the
🤗 interface inside it (`labels=` scoring, 🤗 `generate` over the cached decoder `nn/incremental.py`).

On the CPU (tiny random models, every run): the export's layout and remote-code imports; the exported model, loaded
with `local_files_only` from the folder, gives the per-token loss of `load_perceiver_ar_for_eval`; 🤗 greedy writes the
tokens of our cached greedy (`IncrementalDecoder.greedy`, the harness geometry) and of 🤗's uncached path; seeded
sampling repeats; beam search with the cache equals beam search without it; a left-padded batch decodes as its rows
alone; the cached row grows past its end; a read the cache does not cover falls back to the uncached loop;
`IncrementalDecoder.reorder` continues as a fresh decoding of the gathered rows.

On CUDA with a real checkpoint (`HUB_EXPORT_CHECKPOINT=<dir>`, e.g. the 150M concept memory on Odra,
`Cache/e38_board/e37a_m120t_final`): the same checks in bf16 on a fixed text (bf16 tolerance for the loss, exact
tokens for 32 greedy tokens), seeded sampling, beam search, and the tokenizer saved in the folder.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch

from nn.incremental import IncrementalDecoder
from nn.perceiver_ar_hub import GEN_PAD, PerceiverARCache, PerceiverARForCausalLM
from nn.perceiver_ar_lm import PerceiverARLM
from tests.test_concept_memory import CONCEPT
from tests.test_latent_memory import cfg as base_cfg
from tests.test_lsa_read import DENSE, LSA
from tests.test_note_open import row

ROOT = Path(__file__).resolve().parents[1]
# the 150M concept memory's shape in miniature: concept read, factorised tied head, soft cap, no n-gram tables, one id
# for bos / eos / pad (🤗 pads finished rows with pad, the harness writes eos)
IDS = dict(bos_token_id=0, eos_token_id=0, pad_token_id=0)
ARMS = {
    "concept": dict(CONCEPT, lm_head_tied=True, logit_softcap=30.0, ngram_orders=(), **IDS),
    "dense": dict(DENSE, **IDS),
    "lsa": dict(LSA, logit_softcap=30.0, **IDS),
}


def _exporter():
    spec = importlib.util.spec_from_file_location("export_hub_model", ROOT / "scripts" / "export_hub_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny(arm: str, cls=PerceiverARLM, **extra) -> PerceiverARLM:
    torch.manual_seed(3)
    m = cls(base_cfg(**dict(ARMS[arm], **extra))).eval()
    for layer in m.layers:  # warm residuals so every layer shows in the output
        layer.attn.wo.weight.data.normal_(0, 0.2)
        layer.mlp.down.weight.data.normal_(0, 0.2)
    return m


def _prompt(L0: int = 150, B: int = 2) -> torch.Tensor:
    ids = row(L0, B=B)
    ids[ids <= 2] = 5  # no pad / bos / eos (0, 1, 2) in the prompt
    return ids


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """arm → (checkpoint dir, export dir, manifest), each arm exported once."""
    ex = _exporter()
    out = {}
    for arm in ARMS:
        base = tmp_path_factory.mktemp(f"hub_{arm}")
        ckpt = base / "ckpt"
        _tiny(arm).save_pretrained(ckpt)
        man = ex.export(ckpt, base / f"export_{arm}")
        out[arm] = (ckpt, base / f"export_{arm}", man)
    return out


def _load(path, dtype=torch.float32, device="cpu"):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True, local_files_only=True,
                                                dtype=dtype).to(device).eval()


def _harness_greedy(model, ids, new, eos):
    """Our cached greedy in the harness geometry (`PerceiverARLMEval._model_generate`)."""
    B, L0 = ids.shape
    Lb = -(-(L0 + new) // GEN_PAD) * GEN_PAD
    buf = torch.full((B, Lb), eos, dtype=ids.dtype, device=ids.device)
    buf[:, :L0] = ids
    doc = torch.full((B, Lb), -1, dtype=torch.long, device=ids.device)
    doc[:, :L0] = 0
    return IncrementalDecoder(model).greedy(buf, doc, L0, new, eos=eos)


# ---------------------------------------------------------------------------------------------- the folder
@pytest.mark.parametrize("arm", list(ARMS))
def test_export_folder(exported, arm):
    from transformers.dynamic_module_utils import get_imports

    ckpt, out, man = exported[arm]
    names = {p.name for p in out.iterdir()}
    assert {"config.json", "configuration_perceiver_ar.py", "modeling_perceiver_ar.py", "generation_config.json",
            "model.safetensors", "export_manifest.json"} <= names
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["auto_map"] == {"AutoConfig": "configuration_perceiver_ar.PerceiverARConfig",
                               "AutoModelForCausalLM": "modeling_perceiver_ar.PerceiverARForCausalLM"}
    assert cfg["architectures"] == ["PerceiverARForCausalLM"]
    assert cfg["use_liger"] is False and cfg["tie_word_embeddings"] is False
    assert cfg["tokenizer_name"] == "HuggingFaceTB/SmolLM2-135M"
    assert cfg["attn_backend"] == "sdpa" and cfg["attn_pad_multiple"] == 1
    for f in ("modeling_perceiver_ar.py", "configuration_perceiver_ar.py"):
        req = set(get_imports(str(out / f)))  # what trust_remote_code asks to be installed
        assert {"torch", "transformers"} >= {r for r in req if r not in sys.stdlib_module_names}
    text = (out / "modeling_perceiver_ar.py").read_text()
    assert "\nfrom nn." not in text and "\n    from nn." not in text
    want = {"concept": "nn/concept_memory.py", "lsa": "nn/lsa_buckets.py"}.get(arm)
    for opt in ("nn/concept_memory.py", "nn/lsa_buckets.py"):
        assert (opt in man["vendored_modules"]) == (opt == want)
    assert man["files"]["model.safetensors"]["sha256"]


@pytest.mark.parametrize("arm", list(ARMS))
def test_loads_from_the_folder_alone(exported, arm):
    ckpt, out, _ = exported[arm]
    model = _load(out)
    assert type(model).__name__ == "PerceiverARForCausalLM"
    assert type(model).__module__.startswith("transformers_modules.")  # the folder's code, not the repo's nn/
    ref = PerceiverARLM.from_pretrained(ckpt).eval()
    a, b = model.state_dict(), ref.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------------------------- scoring
@pytest.mark.parametrize("arm", list(ARMS))
def test_same_per_token_loss_as_the_eval_loader(exported, arm):
    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    ckpt, out, _ = exported[arm]
    model = _load(out)
    ref = load_perceiver_ar_for_eval(str(ckpt), device="cpu", dtype=None)
    ids = _prompt(96, B=2)
    with torch.no_grad():
        got = model(input_ids=ids, labels=ids)
        _, per, valid = ref(input_ids=ids, labels=ids, return_per_token_loss=True)
        mine = torch.nn.functional.cross_entropy(got.logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    torch.testing.assert_close(mine, per, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got.loss, per.sum() / valid.sum(), atol=1e-6, rtol=1e-6)
    # padding: a left-padded row scores its tokens as the unpadded row does (exactly for full attention; the LSA and
    # concept reads cut the row into blocks at fixed positions, so padding moves the block ends a little)
    pad = 7
    ids_p = torch.cat([ids.new_zeros((1, pad)), ids[:1, :-pad]], 1)
    mask = torch.ones_like(ids_p)
    mask[:, :pad] = 0
    with torch.no_grad():
        lp = model(input_ids=ids_p, attention_mask=mask, labels=ids_p).loss
        lu = model(input_ids=ids[:1, :-pad], labels=ids[:1, :-pad]).loss
    torch.testing.assert_close(lp, lu, atol=1e-5 if arm == "dense" else 5e-3, rtol=0)


# ---------------------------------------------------------------------------------------------- generation
@pytest.mark.parametrize("arm", list(ARMS))
def test_greedy_matches_our_cached_greedy(exported, arm):
    ckpt, out, _ = exported[arm]
    model = _load(out)
    ref = PerceiverARLM.from_pretrained(ckpt).eval()
    ids = _prompt()
    eos = int(model.config.eos_token_id)
    got = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=32, do_sample=False)
    want = _harness_greedy(ref, ids, 32, eos)
    assert torch.equal(got[:, :want.shape[1]], want)
    plain = model.generate(input_ids=ids, max_new_tokens=32, do_sample=False, use_cache=False)
    assert torch.equal(got, plain)


def test_generate_uses_the_cache(exported, monkeypatch):
    model = _load(exported["concept"][1])
    mod = __import__(type(model).__module__, fromlist=["IncrementalDecoder"])
    calls = {"prefill": 0, "step": 0}
    for name in calls:
        fn = getattr(mod.IncrementalDecoder, name)

        def wrap(*a, _fn=fn, _name=name, **k):
            calls[_name] += 1
            return _fn(*a, **k)

        monkeypatch.setattr(mod.IncrementalDecoder, name, wrap)
    ids = _prompt(B=1)
    model.generate(input_ids=ids, max_new_tokens=10, min_new_tokens=10, do_sample=False)
    assert calls == {"prefill": 1, "step": 9}


@pytest.mark.parametrize("arm", ["concept", "dense"])
def test_seeded_sampling_repeats(exported, arm):
    model = _load(exported[arm][1])
    ids = _prompt()
    kw = dict(max_new_tokens=20, min_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9,
              repetition_penalty=1.1)
    torch.manual_seed(0)
    a = model.generate(input_ids=ids, **kw)
    torch.manual_seed(0)
    b = model.generate(input_ids=ids, **kw)
    torch.manual_seed(1)
    c = model.generate(input_ids=ids, **kw)
    assert torch.equal(a, b) and a.shape == (2, 170)
    assert not torch.equal(a, c)
    assert int(a.max()) < model.config.vocab_size


@pytest.mark.parametrize("arm", list(ARMS))
def test_beam_search_cached_equals_uncached(exported, arm):
    model = _load(exported[arm][1])
    ids = _prompt()
    kw = dict(max_new_tokens=12, num_beams=3, do_sample=False, num_return_sequences=2)
    a = model.generate(input_ids=ids, **kw)
    b = model.generate(input_ids=ids, use_cache=False, **kw)
    assert a.shape == (4, 162) and torch.equal(a, b)


@pytest.mark.parametrize("arm", ["concept", "lsa"])
def test_left_padded_batch_decodes_as_its_rows(exported, arm):
    model = _load(exported[arm][1])
    long, short = _prompt(150, B=1), _prompt(131, B=1)
    pad = long.shape[1] - short.shape[1]
    ids = torch.cat([long, torch.cat([short.new_zeros((1, pad)), short], 1)])
    mask = torch.ones_like(ids)
    mask[1, :pad] = 0
    kw = dict(max_new_tokens=16, min_new_tokens=16, do_sample=False)
    both = model.generate(input_ids=ids, attention_mask=mask, **kw)
    alone = [model.generate(input_ids=x, **kw) for x in (long, short)]
    assert torch.equal(both[0, 150:], alone[0][0, 150:])
    assert torch.equal(both[1, 150:], alone[1][0, 131:])


@pytest.mark.parametrize("arm", ["concept", "lsa"])
def test_row_grows_past_its_end(exported, arm):
    """A cache with one free slot after the prompt (the row ends at 256) re-prefills when the answer reaches the end;
    the tokens are the uncached path's."""
    model = _load(exported[arm][1])
    ids = _prompt(200, B=1)
    kw = dict(max_new_tokens=80, min_new_tokens=80, do_sample=False)
    cache = sys.modules[type(model).__module__].PerceiverARCache(headroom=1)  # the folder's own cache class
    a = model.generate(input_ids=ids, past_key_values=cache, **kw)
    b = model.generate(input_ids=ids, use_cache=False, **kw)
    assert cache.dec.S == 512 and torch.equal(a, b)


def test_unsupported_read_falls_back_to_the_uncached_loop():
    # the cache covers one global layer (and the tied loop needs exactly one)
    model = _tiny("dense", cls=PerceiverARForCausalLM, global_layers=2, message_loop_rounds=1)
    ids = _prompt(B=1)
    cache = PerceiverARCache()
    a = model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=12, do_sample=False)
    b = model.generate(input_ids=ids, use_cache=False, max_new_tokens=12, do_sample=False)
    assert cache.uncached and torch.equal(a, b)


@pytest.mark.parametrize("arm", list(ARMS))
def test_reorder_continues_as_a_fresh_decoding(arm):
    """Prefill two rows, step 6 tokens, keep rows [1, 0, 1], step 20 more (the concept read crosses block ends of 16,
    LSA's a block end of 32): the hidden states equal a decoder that read the gathered rows from the start."""
    m = _tiny(arm)
    ids = _prompt(150, B=2)
    nxt = row(26, seed=7, B=2) + 3
    buf = torch.full((2, 256), 2)
    buf[:, :150] = ids
    doc = torch.full((2, 256), -1)
    doc[:, :150] = 0
    dec = IncrementalDecoder(m)
    dec.prefill(buf, doc, 150)
    for j in range(6):
        dec.step(nxt[:, j])
    rows = torch.tensor([1, 0, 1])
    dec.reorder(rows)
    fresh = IncrementalDecoder(m)
    fresh.prefill(buf[rows], doc[rows], 150)
    for j in range(6):
        fresh.step(nxt[rows, j])
    for j in range(6, 26):
        torch.testing.assert_close(dec.step(nxt[rows, j]), fresh.step(nxt[rows, j]), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------------------------- the real checkpoint
CKPT = os.environ.get("HUB_EXPORT_CHECKPOINT")
TEXT = (
    "The river rises in the hills above the town and runs for about forty miles before it reaches the sea. In the "
    "eighteenth century its banks were lined with mills that ground grain and later spun cotton, and the town grew "
    "around them. When the railway arrived in 1847, the mills lost much of their trade to larger factories in the "
    "north, and many of the buildings were left empty. Today the old mill houses are a museum, a library and a row "
    "of small shops, and the path along the river is used by walkers and cyclists. Every spring the town holds a "
    "festival in which boats decorated with flowers are carried down the river to the harbour, where a band plays "
    "until the evening. Visitors who stay for a few days can follow the river up into the hills, where the water is "
    "clear and cold, and where the ruins of the first mill can still be seen among the trees."
)


@pytest.mark.skipif(not (CKPT and torch.cuda.is_available()), reason="set HUB_EXPORT_CHECKPOINT on a CUDA machine")
def test_real_checkpoint_on_cuda(tmp_path):
    from transformers import AutoTokenizer

    from evaluation.lm_eval_perceiver_ar import load_perceiver_ar_for_eval

    out = Path(os.environ.get("HUB_EXPORT_OUT") or tmp_path / "export")
    t0 = time.time()
    man = _exporter().export(CKPT, out, overwrite=True)
    t_export = time.time() - t0
    tok = AutoTokenizer.from_pretrained(str(out), local_files_only=True)
    t0 = time.time()
    model = _load(out, dtype=torch.bfloat16, device="cuda")
    t_load = time.time() - t0
    ref = load_perceiver_ar_for_eval(CKPT, device="cuda", dtype=torch.bfloat16)
    enc = tok(TEXT, return_tensors="pt").to("cuda")
    ids = enc["input_ids"]

    # scoring: the eval loader's per-token loss, in bf16
    with torch.no_grad():
        got = model(**enc, labels=ids)
        _, per, valid = ref(input_ids=ids, labels=ids, return_per_token_loss=True)
        mine = torch.nn.functional.cross_entropy(got.logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    ref_loss = float(per.sum() / valid.sum())
    assert abs(float(got.loss) - ref_loss) < 1e-4
    assert float((mine - per).abs().max()) < 0.05 and float((mine - per).abs().mean()) < 2e-3

    # greedy: 32 tokens, the tokens of our cached greedy
    t0 = time.time()
    g = model.generate(**enc, max_new_tokens=32, min_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()
    t_greedy = time.time() - t0
    want = _harness_greedy(ref, ids, 32, int(model.config.eos_token_id))
    assert torch.equal(g[:, :want.shape[1]], want)

    # seeded sampling repeats; beam search runs
    kw = dict(max_new_tokens=20, min_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9)
    torch.manual_seed(0)
    s1 = model.generate(**enc, **kw)
    torch.manual_seed(0)
    s2 = model.generate(**enc, **kw)
    assert torch.equal(s1, s2)
    beams = model.generate(**enc, max_new_tokens=16, num_beams=4, do_sample=False, repetition_penalty=1.2)
    assert beams.shape == (1, ids.shape[1] + 16)

    vocab_same = None
    try:
        hub_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", local_files_only=True)
        vocab_same = hub_tok.get_vocab() == tok.get_vocab()
        assert vocab_same, "the checkpoint's tokenizer is not HuggingFaceTB/SmolLM2-135M's (config.tokenizer_name)"
    except OSError:
        pass  # not in the local Hub cache
    report = {
        "checkpoint": CKPT, "export": str(out), "vendored_modules": man["vendored_modules"],
        "config_changes": sorted(man["config_changes"]), "text_tokens": int(ids.shape[1]),
        "loss_export": round(float(got.loss), 5), "loss_eval_loader": round(ref_loss, 5),
        "per_token_max_abs_diff": float((mine - per).abs().max()),
        "greedy_32": tok.decode(g[0, ids.shape[1]:]), "sample_20_seed0": tok.decode(s1[0, ids.shape[1]:]),
        "beam_16": tok.decode(beams[0, ids.shape[1]:]), "tokenizer_is_smollm2_135m": vocab_same,
        "seconds": {"export": round(t_export, 1), "load": round(t_load, 1), "greedy_32": round(t_greedy, 2)},
        "gpu": torch.cuda.get_device_name(0),
    }
    print("HUB_EXPORT_REPORT " + json.dumps(report))
