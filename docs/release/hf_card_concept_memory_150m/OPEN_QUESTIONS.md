# 150M concept memory: open questions before the Hugging Face release

**Decided 2026-11-17, ~20:40 UTC (author: "post as meridian21 organization, as public with Apache-2.0"):**
organisation `Meridian21Lab` (the account's only meridian21 organisation, full name "Meridian21 Lab"), public,
Apache-2.0 for the weights and the card; the card describes the mechanism at the level of the public code.
Repository: `Meridian21Lab/concept-memory-140m-11.5bt`, code pinned at commit `1fddc505` (branch `eng-100m`).

Still open after the release: a LICENSE file in the GitHub repository (the card's "clone the code" path), the stale
W&B tags, and the loading gaps in section 3. The sections below are the pre-decision record.

Draft of 2026-11-17, for the card in `README.md` next to this file.

## 1. Decisions for the author (each with a recommendation)

### 1.1 Visibility: public or gated?
**Recommendation: gated, with automatic approval and a two-field form (affiliation, intended use), until the
licence (1.3) is chosen. Then public.** Gating keeps the standing rule (weights private or gated until the IP
decision) and still lets investors and researchers download in one click. It also tells us who is testing the
model. Flip the switch to public once 1.3 is settled; the repository id and links stay the same.

### 1.2 Namespace: `ksopyla` or a meridian21 organisation?
**Recommendation: a `meridian21` organisation, with `ksopyla` as admin.** The investor goal makes the model a
company asset; a separate organisation can be handed over or shared without touching the personal account. Earlier
research uploads (`ksopyla/gemma3-concepts-1b-c128-1bt`, the concept-encoder and tokenizer repos) can stay where they
are. Checks done 2026-11-17 (read-only): `ksopyla` has 8 model repos, none named concept-memory; the Hub shows no
public repo under `meridian21`, so the organisation may not exist or may be empty. Before creating it, check that
the name is free. Note a name-clash risk: the Tiny-ML Leaderboard already lists an unrelated org `meridianmrm`
(model Zedev-134M-Fudas).

Proposed repo id: `<namespace>/concept-memory-140m-11.5bt` (line · size · data budget). The usual `c{C}` concept-count
slot does not fit this model: its number of concepts grows with the document, at 64 per 512 tokens.

### 1.3 IP: how much of the architecture to describe, and which licence?
**Recommendation: decide the GitHub question first, then describe the mechanism at the level of the public code and
choose the licence after reading NVIDIA's data terms.**

- **The mechanism is already public.** `github.com/ksopyla/MrCogito` is a public repository. Its `dev` branch has
  held `nn/concept_memory.py` and `nn/perceiver_ar_lm.py` since at least 2026-11-14, and so does `eng-100m`. The
  repository has **no LICENSE file**, so the code is readable but, by default, all rights are reserved. The card's
  "How the concept memory works" section describes what that code already shows. If the author wants patent
  protection, this is the bigger issue: speak to a patent attorney before more disclosure. Some jurisdictions
  (Europe) give no grace period after public disclosure. If no patent is planned, the card's description costs
  nothing extra.
- **Licence of the weights.** About 45 % of the training tokens (5.2B of 11.5B) come from NVIDIA's Nemotron-CC-v2
  (High-Quality + Diverse-QA) and Nemotron-CC-Math-v1. Both are gated on the Hub under their own terms
  (`license: other`). Nemotron-Pretraining-Specialized-v1 lists CC BY-SA 4.0, CC BY 4.0, GFDL and "other".
  The rest of the data is CC BY 4.0 (DCLM-Edu), ODC-By (Dolma 3 Longmino), Apache-2.0 (smol-smoltalk) and openly
  licensed Common Pile sources. Read NVIDIA's terms for releasing models trained on that data before picking
  Apache-2.0. If they allow it, Apache-2.0 for the weights, with an attribution list in the card. If unclear, a
  research licence (`license: other`) and keep the gate.
- **Code licence.** Add a LICENSE to the GitHub repository in the same step, so the card's "clone the code" path is
  legally usable.

## 2. Pre-publish checklist

- [ ] Decisions 1.1–1.3 made; replace every `⟦DECISION⟧` in `README.md` and drop the DRAFT header.
- [ ] Pin the code: the model code is on `dev`, not `main` (`main` was last updated 2026-09-05). Tag the commit
      the card points to (for example `concept-memory-150m-v1`) and put it in "How to use".
- [ ] Fix `config.json` in the staged copy: `tokenizer_name` points at an AWS path
      (`/opt/dlami/nvme/data/m21mix_v2/tokenizer`); set it to `HuggingFaceTB/SmolLM2-135M`. Set `use_liger` to
      `false` (Liger is a Linux training kernel). Add `auto_map` once modeling files exist (section 3).
- [ ] Stage only `model.safetensors` (566 MB, fp32), `config.json` and the tokenizer files from
      `Cache/e38_board/e37a_m120t_final/` on Odra; optionally add a bf16 copy (283 MB). The folder has no optimizer
      state.
- [ ] W&B tags: the three training runs carry the stale default tags `JeanKaddour/minipile` and `minipile`, but the
      model never read MiniPile. Fix the tags before the card links them.
- [ ] Record the real-document long-context results (raw files in `long_context/` next to this file, summary by
      `long_context/summarise.py`) in the experiment's spec and results folder. They close the spec's open
      "far context still used" check (switching the far read off costs 0.255 nats past 2k at 16k; bar 0.10). The
      session that owns the long-run results does this; the card does not depend on it.
- [ ] Run the "How to use" snippet from a fresh clone against the staged folder (CUDA).
- [ ] Upload as **private** first (`hf upload <namespace>/<slug> <folder> --repo-type model --private`), check that
      the card renders, then switch to gated or public. The author runs this or gives an explicit go; agents never
      handle the HF token.
- [ ] Leaderboard submission: none of its tests (BLiMP, ARC, WikiText) is in the training data; disclose the
      planted-fact lookup rows anyway, since the board flags benchmark training.

**Optional, each needs the author's OK:**
- [ ] SmolLM2-135M on our harness: the leaderboard tier (about 6 minutes on one Odra card) plus the seven core tasks.
      It is the natural same-size reference and is not on the board. Until then the card quotes the board only.
- [ ] The fair full-attention control on the same data (already queued as leaderboard run 4), before any "matches
      full attention" claim, and a second seed before any claim about a 1–2 point difference.

## 3. What is missing for `from_pretrained` from the Hub

Today a Hub user needs the MrCogito code (`evaluation/lm_eval_perceiver_ar.load_perceiver_ar_for_eval`). For
`AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)` the repository needs:

1. **Modeling and configuration files** in the Hub repo, self-contained: `nn/perceiver_ar_lm.py` (4,603 lines)
   imports `nn/concept_memory.py`, `nn/exact_read.py`, `nn/latent_memory.py` and `nn/lsa_buckets.py` (about 7,000
   lines with the cached decoder). Two routes: (a) a trimmed `modeling_perceiver_ar.py` with only the concept-memory
   path; (b) a pip-installable `mrcogito` package that the card installs first. Route (a) is the usual Hub practice;
   route (b) is less work and keeps one source of truth.
2. **`auto_map` in `config.json`**: `AutoConfig → configuration_perceiver_ar.PerceiverARConfig`,
   `AutoModelForCausalLM → modeling_perceiver_ar.PerceiverARLM`. `model_type: perceiver_ar` is not a Transformers
   architecture.
3. **Generation.** `PerceiverARLM.generate()` is our own uncached loop (greedy or top-k), not Transformers'
   `GenerationMixin`: no `past_key_values`, so `pipeline("text-generation")` and standard `generate()` arguments do
   not work. The cached decoder (`nn/incremental.py`, read once then one position per token) is not wired into
   `generate`.
4. **Hardware.** The fast path is flex attention on CUDA (PyTorch ≥ 2.10 in our lock). The CPU / MPS reference
   path builds a dense mask of length × length, so it is only practical for short inputs.
5. **Config hygiene**: `use_liger: true` (needs `liger-kernel`, Linux only); `tokenizer_name` points at an AWS
   path; `tie_word_embeddings: false` while `lm_head_tied: true` (the output reuses the input table through our
   own factorised head, so Transformers' weight tying must stay off); weights saved in fp32.
6. **Benchmarks.** lm-eval's stock `hf` model type cannot run it; our adapter registers `--model perceiver_ar`.
