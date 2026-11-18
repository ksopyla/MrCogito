# Hub releases that load with `from_pretrained` — engineering spec

- **Type:** engineering (packaging + interface). Not an experiment: no architecture, objective or metric change.
- **Status:** open (2026-11-18), not started. Asked for by the author during the review of the first public release
  (150M concept memory, `Meridian21Lab/concept-memory-140m-11.5bt`).
- **Owner:** Krzysztof Sopyła
- **Serves:** every model we publish should be usable by a reader in one line, and every release should be checked
  that way before it goes public. Today the release card has to say "clone the repo at commit X and call our
  loader", which most readers will not do.

## Goal

For every `perceiver_ar` release (and later the other families):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Meridian21Lab/<repo>")
model = AutoModelForCausalLM.from_pretrained("Meridian21Lab/<repo>", trust_remote_code=True, dtype="bfloat16")
out = model.generate(**tok("The capital of France is", return_tensors="pt"), max_new_tokens=40, do_sample=True,
                     temperature=0.7, top_p=0.9)
```

works on CUDA from the Hub id, scores text with `labels=`, and generates with 🤗 Transformers' own `generate`
(sampling, beam search, repetition penalty) using our cached decoder.

## What is missing today (from the 2026-11-17/18 release work)

1. **No modeling code in the Hub repo and no `auto_map`** in `config.json`, so `trust_remote_code=True` has nothing
   to run. `nn/perceiver_ar_lm.py` imports the rest of `nn/` (concept memory, attention backends, incremental
   decoder), so it cannot simply be copied as one file.
2. **No `GenerationMixin` interface.** `PerceiverARLM.generate` is our own uncached sampler (temperature, top-k);
   `nn/incremental.py` holds the cached decoder, but it is not exposed as `past_key_values` /
   `prepare_inputs_for_generation`.
3. **`config.json` carries training-only fields:** `tokenizer_name` is an AWS path, `use_liger: true` (Liger is a
   training kernel; our eval loader switches it off by hand).
4. **CPU path:** the reference attention builds a full length × length mask, so it suits short inputs only.
5. **Flex attention compiles per shape;** short sampling loops need `attn_backend="sdpa"`.
6. **Transformers v5:** the project pins `<5`; new reference models (LFM2.5) already ship v5-only tokenizer
   configs. Coordinate with `transformers_v5_pytorch_2_12_upgrade.md`.
7. **Weight tying and dtype:** `tie_word_embeddings: false` while `lm_head_tied: true` (the output reuses the input
   table through our own factorised head, so Transformers' tying must stay off); weights are saved in fp32.
8. **Size of the code:** `nn/perceiver_ar_lm.py` (4,603 lines) imports `nn/concept_memory.py`, `nn/exact_read.py`,
   `nn/latent_memory.py` and `nn/lsa_buckets.py` (about 7,000 lines with the cached decoder). Two routes: (a) a
   trimmed `modeling_perceiver_ar.py` with only the paths the checkpoint uses (the usual Hub practice); (b) a
   pip-installable `mrcogito` package the card installs first (less work, one source of truth). The export script
   below is route (a), generated from the source so there is still one source of truth.

Release-time notes with the same list: `docs/release/hf_card_concept_memory_150m/OPEN_QUESTIONS.md`, section 3.

## Plan

1. **Export script** `scripts/export_hub_model.py <checkpoint> <out_dir>`: writes `configuration_perceiver_ar.py` and
   `modeling_perceiver_ar.py` (the modules the checkpoint's config actually uses, vendored into a flat package),
   adds `auto_map` (`AutoConfig`, `AutoModelForCausalLM`), clears training-only config fields (`use_liger: false`,
   `tokenizer_name` → the Hub tokenizer id), copies weights and tokenizer.
2. **Generation interface:** subclass `GenerationMixin`; wrap `IncrementalDecoder` as a cache object so `generate`
   uses it for greedy, sampling and beam search; fall back to the uncached loop where the cache does not cover a
   read.
3. **Tests** (`tests/test_hub_export.py`): the exported model gives the same per-token loss as
   `load_perceiver_ar_for_eval` on a fixed text (bf16 tolerance); `generate` greedy matches our cached greedy for 32
   tokens; sampling runs with a seed; loads with `local_files_only` from the export folder.
4. **Release checklist:** `huggingface-project` and `model-release` add a smoke test from the Hub id after the
   private upload (load, score one sentence, sample 20 tokens), and the card's "How to use" shows the snippet only
   when that test passed.
5. **Backfill:** re-export `Meridian21Lab/concept-memory-140m-11.5bt` once 1–4 pass, add `library_name: transformers`
   and the `custom_code` tag to its card, and replace "How to use".

## Done when

- `AutoModelForCausalLM.from_pretrained("<hub id>", trust_remote_code=True)` loads the 150M concept memory on a clean
  machine with only `transformers` and `torch` installed, scores text identically to our loader, and samples with
  🤗 `generate`.
- The release checklist runs that smoke test for every new upload.
