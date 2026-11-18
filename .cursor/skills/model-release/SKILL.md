---
name: model-release
description: How to present a MrCogito / Meridian21 model to the public - the model card (README) and its evaluation page, which comparisons to run against which models, how to show results (one comparison table, the community leaderboard table, radar and long-context charts), how to explain units, how to show text samples, and the tone (lead with measured wins, report trade-offs without judging, honest about limits, written to attract collaborators and funding). Use this skill whenever a model is about to be published, a Hugging Face card is drafted, reworked or reviewed, results are being prepared "for the community", "for VCs" or "for a release", someone asks how our model compares with other small models, or asks whether we can claim something (data-efficient, long context, better than X) in public. Pair with huggingface-project for the Hub mechanics (naming, upload, licence gates) and experiment-evaluate for running the scripts. Not for internal run reports or the experiment registry (experiment-track).
---

# Model release

## Mission

A release card is the project's public face. Two kinds of readers open it: researchers deciding whether the model is
worth a download, and people deciding whether the project is worth backing. Both want the same three things in the
first screen: **what is new, where it wins, and how fairly that was measured.** The card is honest and blunt about
limits, because a single overclaim found by a reader costs more credibility than any number earns; but it is written
from the side of what the model *does*, because we are still training and improving the architecture and the card
should show the potential, not apologise for it.

This skill owns *what to measure, what to compare against, and how to write it down*. Hub naming, tags, licence gates
and upload steps live in `huggingface-project`; running the evaluation scripts lives in `experiment-evaluate`.

## The author's rules (2026-11-18 review of the first release), with the reason for each

1. **Lead with measured wins; report trade-offs without judging.** No "where it is weak", "weak spots" or
   "limitations" verdict language in the top half. Every model has trade-offs: state them as facts in a neutral
   *Scope and trade-offs* section ("ARC-Easy 49.0 % against 61.6 % for SmolLM2-135M at 252B tokens"). Where we
   clearly win, say so plainly and put it first. *Why:* judging our own model in its card reads as either false
   modesty or spin; facts let the reader judge, and wins deserve the headline.
2. **Short card, deep annex.** The README is read once, top to bottom, in a few minutes: aim for ≤ ~300 lines after
   the YAML (the reference card: 650 → ~320 with three more models and the leaderboard table added).
   Method, per-test tables, our own diagnostic switches and long result tables go to `EVALUATION.md` in the same Hub
   repo (linked at the top), or into `<details>` blocks. *Why:* the first card was 650 lines and the strengths were
   buried; detail is for the reader who asks for it.
3. **One place to compare.** One *Results at a glance* table (measures as rows, models as columns, ours first) holds
   every same-harness number for every reference; a second table puts our model, at its computed rank, **inside the
   community leaderboard** we report against (for the Tiny-ML Leaderboard: rank, model, parameters, training tokens,
   the board's three tests, its score). Never spread one model's numbers over several sections. *Why:* the author
   could not compare models when the numbers lived in five places.
4. **Fair, current references, chosen before running.** At least two same-size models (the strongest open one at that
   size included), a data-matched one when its training checkpoints are public, and **a modern reference** from the
   last ~12 months (search the Hub for base models up to ~3× our size). For a model trained on far more data than ours,
   use its **earliest public checkpoint** in the comparison (name the token count) and show the final model only in
   the leaderboard table. *Why:* old references (Pythia, 2023) make any new model look good; the earliest checkpoint is
   the closest to our budget a reader can check.
5. **Each model in its own precision, started the way its tokenizer starts a text, quoted numbers in brackets.** If a
   reference publishes its own scores, show ours and theirs in **one column**, theirs in brackets, with a footnote
   naming the source and date ("43.64 (43.52)"). *Why:* bf16 cost Pythia-160M 6 ARC-Easy points and a stray start
   token wrecked Granite at long lengths; brackets show the reader our harness reproduces the published numbers.
   When a reference behaves oddly (a long-context model that gets worse with context), rule out our side first
   (precision, attention kernel, library version, start token) on one document, then report what was measured and
   how it was checked.
6. **Explain every unit at first use, with a scale.** Bits per byte: "how many bits the model needs to store each
   byte of unseen text; 0 is perfect, 8 is plain text, lower is better; gzip needs ~2.5, xz ~2.0 on the same pages".
   Byte perplexity is the same quantity (bits per byte = log₂ byte perplexity). Accuracy: name chance. *Why:* the
   author asked what 0.795 means and whether the scale runs 0–1; a reader without the scale cannot tell a win from
   noise.
7. **Figures the author asked for.** The radar shows the six standard benchmarks (BLiMP, ARC-Easy, ARC-Challenge,
   SciQ, HellaSwag, WikiText), ours and two same-size references, each axis as a share of the best of the three, raw
   values printed; everything else goes in the table. Long context gets its own line chart (bits per byte against
   context length, one line per model, the line ending at the model's limit). Data efficiency gets score against
   training tokens. *Why:* a radar of mixed measures hid which axes were ours; separate charts show one idea each.
8. **Text samples with modern decoding.** Never show greedy text for prose (it loops and undersells a small base
   model). Use nucleus sampling (temperature 0.7, top-p 0.9, repetition penalty 1.1), show the seed-0 reply unpicked,
   state the settings, keep the rest in a file; beam search (4 beams, no repeated 3-gram) is the alternative.
   `evaluation/sample_text.py` does both for our checkpoints and for 🤗 models. Exact-answer benchmarks (RULER, needle
   tests) stay greedy, as their official setups are.
9. **Generation benchmarks use the task's own answer length.** Never hand-set a cap. Before trusting a generation
   score, count the reference answers in *this model's* tokenizer and save every reply. *Why:* a 32-token cap that
   held every RULER answer in GPT-2 tokens cut correct answers in SmolLM2's tokenizer (one digit per token: a UUID is
   35 tokens), and a card shipped wrong RULER numbers.
10. **Claims about efficiency or context need a stated measure and the confound.** "Data-efficient" = a better score
    from the same tokens, measured as same-budget checkpoint, tokens-to-match, or against the field; always say that
    data and architecture are not separated without a same-data control. "Long context" = gain from far context on
    held-out real documents of the same text for every model, plus a swapped-past control.
11. **Keep what backers and replicators need:** training budget in tokens, wall time, hardware, cost; the data mix by
    stage with Hub links to every dataset and their licences; where the raw results are. Name hardware **generically**
    ("one RTX 3090", "4 × NVIDIA L40S on AWS spot"): no server names (Odra, Polonez), hostnames, absolute paths
    (home folders, `/opt/...` build paths), instance types or regions in anything that goes public, including script
    docstrings, figure captions, `config.json` and the JSON files shipped with the card. Relative checkout paths
    inside raw result files (`Cache/...` model ids) may stay: raw results are shipped as produced. *Why:* the author does not publish internal infrastructure (2026-11-18).
12. **How to use goes at the bottom, and only with code that runs for the reader.** `perceiver_ar` releases are
    exported with `scripts/export_hub_model.py` and load with
    `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` (2026-11-18). The card shows that snippet
    only after the **Hub smoke test** passed from the Hub id (checklist below); a family without an export says so in
    two lines and points to the repo.
13. **Citation ready to copy:** a BibTeX `@misc` for the model and an `@online` for the project page
    (<https://ai.ksopyla.com/projects/concept-reasoning/>), plus the code at its commit.
14. **Show the road ahead, briefly.** A *What's next* section of 3–5 bullets naming work that is actually queued or
    running (controls, bigger runs, features switched off in this release). *Why:* the card should show potential,
    and the honest form of potential is the next experiments, not adjectives.

## Workflow

1. **Collect the facts** from the run report and the spec: parameters (with and without the token table), token
   budget by stage, hardware, wall time, cost, data mix with Hub ids, what is switched off, the pass marks.
2. **Choose the references** (rule 4) and write them down before running anything. Search the Hub
   (`hub_repo_search`, base models, sort by downloads and by date); check each one's config for its context length
   and whether it is a base model; check gated access from the server that will run it.
3. **Run the comparisons** with `experiment-evaluate` scripts on one machine, one harness: lm-eval core tasks +
   the leaderboard's tests for every model; `evaluation/long_context_compare.py` (same text, bits per byte, swapped
   past) for every model; our own diagnostic tests (context cut, memory off) for ours; RULER with task-default answer
   lengths and saved replies; `evaluation/sample_text.py` for text. Details and known gotchas:
   [references/evaluation_methods.md](references/evaluation_methods.md).
4. **Draw the figures** with a script next to the card (`figures/make_figures.py`) from the result files, following
   the `dataviz` skill (validated palette, light background, one idea per chart).
5. **Write the card** from [references/card_template.md](references/card_template.md) and the long material into
   `EVALUATION.md`. Reference filled card: `docs/release/hf_card_concept_memory_150m/` (150M concept memory).
6. **Export and check:** `uv run python scripts/export_hub_model.py <checkpoint> <export_dir> --card README.md`
   (modeling code, `auto_map`, weights, tokenizer, `export_manifest.json`), run `tests/test_hub_export.py` with
   `HUB_EXPORT_CHECKPOINT=<checkpoint>` on a CUDA machine, go through the list below, then hand both folders to
   `huggingface-project` for the upload (private first, Hub smoke test, rendered page, then public).
7. **Record** the Hub URL in the run report and the result files' location on the ledger (append-only).

## Before publishing — checklist

- [ ] First screen: what is new, the wins with numbers, one figure; no verdict language about our own model.
- [ ] README ≤ ~300 lines after the YAML; method and long tables in `EVALUATION.md` or `<details>`.
- [ ] One results table with every reference as a column; our row inside the leaderboard table.
- [ ] References include a same-size model, a modern one (≤ 12 months), and the earliest public checkpoint of any
      big-data reference; chosen before running; post-trained (chat) weights excluded.
- [ ] Every number has a unit, a direction and a comparison; every unit is explained once with its scale.
- [ ] Quoted numbers sit in brackets with a footnote (source, date); everything else is our harness, named above the
      table (harness version, shots, precision per model, machine, date).
- [ ] Generation tasks: task-default answer lengths, replies saved, answer lengths checked in our tokenizer.
- [ ] Samples: sampling settings stated, seed-0 reply shown unpicked, no greedy prose.
- [ ] Training budget, cost, hardware, data mix with links and licences present.
- [ ] No server names, hostnames, absolute paths, instance types or regions in any uploaded file
      (`grep -riE "odra|polonez|/home/|/opt/|~/dev"` over the upload folder and the scripts copied into it).
- [ ] How to use at the bottom; no code that fails for a reader; `from_pretrained` status stated.
- [ ] **Hub smoke test passed** after the private upload, before going public: in an environment with only `torch`
      and `transformers`, `python -I scripts/hub_smoke_test.py <org>/<repo>` loads the model from the Hub id with
      `trust_remote_code=True`, scores one sentence and samples 20 tokens (exit 0, its JSON line saved next to the
      release files). Only then: the `from_pretrained` snippet in How to use, `library_name: transformers` and the
      `custom_code` tag in the YAML.
- [ ] BibTeX for the model and the project page; changelog line with the date.
- [ ] Raw results, scripts and figures in the repo folder that gets uploaded.
- [ ] No `~` for "about" in the card: Hub markdown turns a pair of tildes into strikethrough; write ≈.
- [ ] After upload: open the rendered page and scroll it once (figures, tables, strikethrough, links).
