# Hub releases that load with `from_pretrained` — engineering spec

- **Type:** engineering (packaging + interface). Not an experiment: no architecture, objective or metric change.
- **Status:** steps 1–4 done 2026-11-18 (code commit `4964d695`, branch `claude/inspiring-nightingale-290565`, not
  pushed; result below). Step 5, the backfill of the Hub repo, waits for the author. Asked for by the author
  during the review of the first public release (150M concept memory, `Meridian21Lab/concept-memory-140m-11.5bt`).
- **Owner:** Krzysztof Sopyła
- **Serves:** every model we publish should be usable by a reader in one line, and every release should be checked
  that way before it goes public. Today the release card has to say "clone the repo at commit X and call our
  loader", which most readers will not do.

## Goal

For every `perceiver_ar` release (and later the other families):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Meridian21Lab/<repo>", trust_remote_code=True)  # Transformers 5 needs it here too
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

## Result (2026-11-18)

**Code.** `scripts/export_hub_model.py` (the export), `nn/perceiver_ar_hub.py` (`PerceiverARForCausalLM` =
`PerceiverARLM` + `GenerationMixin`, same state-dict keys; `PerceiverARCache` over `IncrementalDecoder`),
`IncrementalDecoder.reorder` and `prefill(all_positions=...)` in `nn/incremental.py`, `tests/test_hub_export.py`,
`scripts/hub_smoke_test.py` (the release checklist's Hub smoke test; imports nothing from the repo).

**What the export holds** (150M concept memory, `Cache/e38_board/e37a_m120t_final` on Odra, exported 14:29 UTC to
`~/dev/MrCogito-hubexport/Cache/hub_export/concept-memory-140m-11.5bt`, not uploaded):

| File | What |
|---|---|
| `modeling_perceiver_ar.py` (6,621 lines) | nn/exact_read, latent_memory, concept_memory, perceiver_ar_lm, incremental, perceiver_ar_hub pasted in import order; `nn/lsa_buckets.py` left out (stubs raise if reached). Whole modules, not trimmed to functions. |
| `configuration_perceiver_ar.py` (710 lines) | `PerceiverARConfig` and the six constants it reads |
| `config.json` | `auto_map`, `architectures` → `PerceiverARForCausalLM`, `use_liger` true → false, `tokenizer_name` AWS path → `HuggingFaceTB/SmolLM2-135M`, `attn_backend` flex → sdpa; `tie_word_embeddings: false` kept |
| `generation_config.json` | bos = eos = pad = 0 |
| weights, tokenizer | `model.safetensors` (566 MB, fp32, unchanged), the checkpoint's SmolLM2 tokenizer files (vocabulary identical to `HuggingFaceTB/SmolLM2-135M`, checked) |
| `export_manifest.json` | source commit, vendored modules, every changed config field (old → new), sha256 per file |

🤗's remote-code check asks only for `torch` and `transformers` (the Liger, flash-attn and `data.text_world` imports
sit in `try` blocks).

**Tests** (`tests/test_hub_export.py`, 27 tests). Mac CPU, tiny concept / full-attention / LSA models: 26 passed. Odra,
RTX 3090, 2026-11-18 14:29–14:30 UTC, all 27 passed, the real checkpoint in bf16 included:

- scoring: the export's loss on a fixed 182-token text equals `load_perceiver_ar_for_eval`'s per-token loss mean
  (2.75207 both); per-token difference at most 0.0003 nats (bf16 rounding of a different matrix shape);
- 🤗 greedy, 32 tokens, equals our cached greedy (`IncrementalDecoder.greedy` in the harness geometry) token for
  token; seeded sampling repeats; beam search runs; loaded with `local_files_only` from the folder;
- export 1.9 s, load 0.7 s, 32 greedy tokens 0.63 s.

**Clean machine** (fresh venvs with only torch 2.11 + Transformers, `python -I`, run from `/tmp`, 14:31–14:33 UTC):
`scripts/hub_smoke_test.py --local` passes on Transformers **4.57.6** and **5.23.0** with the same sentence loss
(2.0551) and the same seed-0 sample (" Paris, the capital of the French monarchy. …"); greedy, 4-beam and sampled
continuations are token-identical across the two versions.

**Caveats found.**
- bf16: 🤗 greedy with the cache and without it (`use_cache=False`, the model's own forward) part at token 14 of 32 on
  a near-tie (top-2 margin 0.054 logits); in fp32 they agree for all 32. The cache is exact; bf16 rounding differs
  between the two paths.
- Left padding moves the block ends of the concept and LSA reads (blocks sit at fixed row positions), so a
  left-padded row scores slightly differently from the same row alone (5e-4 nats on the tiny LSA model). Full
  attention is exact. Batch prompts of equal length, or one at a time, for exact numbers.
- Beam search with the catalogue / notes reads (E33t, E33p) falls back to the uncached loop (their notebooks are
  not reordered). Not used by this checkpoint.
- Long inputs: the export defaults to `attn_backend="sdpa"` (dense mask, fine for short prompts); pass
  `attn_backend="flex"` to `from_pretrained` for long contexts on CUDA.
- The tokenizer files include SmolLM2's ChatML `chat_template.jinja`; this is a base model.

**Transformers 5** (5.23.0, tested as above): loading, scoring and all generation modes work unchanged. Differences:
`AutoTokenizer` reads the model config to pick the tokenizer, so it needs `trust_remote_code=True` too (without it an
interactive "run the custom code?" prompt and two config warnings, among them `message_boundary_token_id = -1` outside
the vocabulary: v5 checks `*_token_id` fields); the remote-code module path gains a hash folder; loading took 4.7 s
against 1.0 s. Our code relies on `PretrainedConfig` (a v5 alias) and on the private
`_supports_default_dynamic_cache` hook (if a future version ignores it, `prepare_inputs_for_generation` replaces an
empty 🤗 cache with ours).

**Card text once step 5 passes** (How to use):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Meridian21Lab/concept-memory-140m-11.5bt"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
print(tok.decode(out[0], skip_special_tokens=True))
# scoring: model(**inputs, labels=inputs["input_ids"]).loss is the mean next-token loss in nats
# long documents (up to 64k tokens): from_pretrained(..., attn_backend="flex")
```

**Step 5, what is left:** push this branch, re-export from the pushed commit (the Odra export's manifest names
`8ad24f8`, the same change applied by `git am` in Odra's scratch clone), upload the export folder next to the card,
run `python -I scripts/hub_smoke_test.py Meridian21Lab/concept-memory-140m-11.5bt` in the clean venv
(`~/dev/MrCogito-hubexport/Cache/hub_export/venv457` on Odra), then add `library_name: transformers`, the
`custom_code` tag and the How to use text above.
