# 150M LSA: open questions before the Hugging Face release

Card: `README.md` next to this file (card standard v3), long tables in `EVALUATION.md`. Sibling release:
`Meridian21Lab/concept-memory-140m-11.5bt` (`docs/release/hf_card_concept_memory_150m/`).

**2026-11-18 ≈17:03 UTC: uploaded PRIVATE** (author: "Yes, upload private") to
<https://huggingface.co/Meridian21Lab/lsa-140m-11.5bt>, Hub commit `8066f778`, 99 files (export from MrCogito commit
`e380a125`, the card folder, `LICENSE`, `NOTICE`), uploaded from the Mac with the `.env` variable
`HF_TOKEN_MERIDAIN21_ORG`. Hub smoke test from the repo id passed (`hub_smoke_test.json`). Still private: the public
switch waits for the author's look at the rendered page and a yes.

## 1. Decisions for the author (each with a recommendation)

### 1.1 Repository id
**Recommendation: `Meridian21Lab/lsa-140m-11.5bt`.** It follows the sibling's pattern (line · size · data budget)
and the naming rule (a short family name people can say aloud). "LSA" is the name Kubicek et al. (Prefrontal AI,
ETH Zürich) gave their method, so a reader may take the repo for theirs; the card's first line says it is our
from-scratch training of a port of their released read and links their paper and code. Alternative:
`Meridian21Lab/latent-sparse-attention-140m-11.5bt` (longer, plainer). The staged export folder and the card use the
recommended id; renaming is a find-and-replace before upload.

### 1.2 Licence and visibility
**Recommendation: as the sibling (author, 2026-11-17: "post as meridian21 organization, as public with
Apache-2.0"): Meridian21Lab, Apache-2.0, private upload first, public after the Hub smoke test.** The same data mix
(about 45 % of tokens from NVIDIA's gated Nemotron-CC sets, as for the sibling). New here: the repository ships
Prefrontal AI's code (`nn/lsa_buckets.py`, Apache-2.0, ported) inside `modeling_perceiver_ar.py`, with their
copyright notice kept; the staged folder adds the Apache-2.0 text as `LICENSE`, as the licence asks.

### 1.3 Credit to the LSA authors
**Recommendation: as drafted.** The card names the paper, authors and code in its summary and in "How LSA works",
states that this is our port trained from scratch (not their weights, not their training recipe: they train a 0.5B
model natively and distil 27B models), and lists the port's two changes (document-masked reads, a flex-attention mask
in place of their Triton kernel). Optional: tell the authors before the public switch.

## 2. Pre-publish checklist

- [x] 1.1–1.3: the author approved the private upload as `Meridian21Lab/lsa-140m-11.5bt` (2026-11-18); licence and
      visibility as the sibling.
- [x] Results from the Odra run (2026-11-18 15:08–16:24 UTC, `scripts/release_evals.sh`, 13 jobs, none failed) in
      `eval_results/`, `long_context/`, `ruler/`, `samples/`; figures drawn (`python figures/make_figures.py`).
- [ ] Sibling card: add one line linking this card (its "What's next" or header), in the sibling's own session.
- [x] Reference RULER rows (SmolLM2-135M, Gemma 3 270M, Granite 4.0 350M, LFM2.5-350M-Base at 4k/8k) from the
      sibling's commit `7c1fd645`, in `ruler/ref_*.json`; `ruler_radar.png` draws from them.
- [ ] Export: `scripts/export_hub_model.py` from the final pushed commit; `tests/test_hub_export.py` with
      `HUB_EXPORT_CHECKPOINT` (27/27 passed on 2026-11-18 on an RTX 3090 for commit `452fc01e`); local smoke test in a
      clean venv passed on Transformers 4.57.6 and 5.23.0 (same loss 2.2892, same sample).
- [x] `grep -riE "odra|polonez|/home/|/opt/|dlami|ip-172|us-east"` over this folder (scripts and JSON included): no
      hits (2026-11-18). Repeat over the staged export folder before upload.
- [ ] README is 396 lines after the YAML (sibling 362, skill target ≈300). Candidates to move to EVALUATION.md if
      the author wants it shorter: the leaderboard rows 5–10, the data-efficiency bullets, the long-document table's
      reference rows.
- [x] Upload **private** (author's yes, 2026-11-18) and `python -I scripts/hub_smoke_test.py Meridian21Lab/lsa-140m-11.5bt`
      in a clean venv (torch 2.11 + Transformers 4.57.6 only): passed, revision `8066f778`, sentence loss 2.3029 (CPU,
      fp32), 20 sampled tokens.
- [ ] The author looks at the rendered page (figures, tables, links; no strikethrough), then switches it public.
- [ ] After the public switch: the sibling card's back-link (the sibling session adds it), `library_name` stays.
- [ ] W&B: the three training runs carry the stale default tags `JeanKaddour/minipile` and `minipile` (the model
      never read MiniPile) and a machine-name tag (`ip-…`); fix before the card links them.
- [ ] Tiny-ML Leaderboard: the LSA model may be submitted next to the concept memory; disclose the planted-fact
      lookup rows, as for the sibling.

## 3. Notes for the record

- **RULER at 32k, 8 of 13 tasks drew different needles** for this model and the sibling: none of their 100 expected
  answers match. Both 32k runs used the same three task groups (checked by the sibling session, 2026-11-18, which
  corrected my first guess); the needle tasks draw from Python's shared global random state and something in the
  runtime changed it; cause not found. Those tasks are two separate 100-prompt draws, compared unpaired; the other 5
  tasks at 32k and every task at 4k–16k match prompt by prompt. Finding the cause would let a rerun make 32k fully
  paired; not needed for the conclusion (+0.072 [0.052, 0.092] either way).
- **Polonez, 2026-11-18 14:39 UTC:** the first launch of this protocol on Polonez hit an NVIDIA Xid 62 (internal
  micro-controller halt) on the card at PCI 0000:49:00 six seconds after start; the driver hung (nvidia-smi and the
  three jobs blocked in the kernel). Polonez needs a reboot; the run moved to Odra. No result from Polonez is used.
- **Protocol fixes found while porting it to LSA** (committed on `claude/hf-card-lsa150m-041025`):
  `long_context_use.py` switched only the concept read off, so on an LSA checkpoint "memory off" silently scored the
  model as trained; it now sets LSA's own switch (sinks + 512-token window) and refuses switches a checkpoint does
  not have. `long_context_compare.py prepare` builds a missing document cache instead of failing.
  `scripts/release_evals.sh` runs the whole release protocol on one checkpoint over a shared GPU queue.
