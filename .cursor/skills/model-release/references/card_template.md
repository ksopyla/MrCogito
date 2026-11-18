---
# MrCogito / Meridian21 model card — STANDARD v3 (2026-11-18). Every Hub release uses this structure.
# v3 = v2 (survey of SmolLM2/3, OLMo, Gemma 3, Qwen3, Granite 4.0, Pythia, LFM2 cards) + the author's review of the
# first release: short card + EVALUATION.md, wins first, trade-offs reported not judged, one comparison table, the
# leaderboard table with our rows, units explained, sampled text, how-to-use at the bottom, BibTeX + project page.
# Reference filled card: docs/release/hf_card_concept_memory_150m/ (README.md + EVALUATION.md).
# Fill every {{PLACEHOLDER}}; aim for ≤ ~300 lines after the YAML.
language:
  - en
license: {{LICENSE}}                 # apache-2.0 for from-scratch; gemma for Gemma-derived; other → license_name + license_link
# library_name: transformers         # ONLY when AutoModelForCausalLM.from_pretrained works from the Hub
pipeline_tag: text-generation
tags: [mrcogito, meridian21, {{LINE}}, size-{{SIZE}}, budget-{{BUDGET}}, base-model, research, e{{EXP}}]
datasets:                            # every Hub dataset that fed training
  - {{HF_DATASET_ID}}
# base_model: only for fine-tunes/grafts, never for from-scratch models
model-index:                         # the headline same-harness numbers
  - name: {{HUB_REPO_NAME}}
    results:
      - task: {type: text-generation, name: {{TASK_NAME}}}
        dataset: {type: {{DATASET_TYPE}}, name: {{DATASET_NAME}}}
        metrics: [{type: {{METRIC}}, value: {{VALUE}}, name: {{METRIC_LABEL_WITH_DIRECTION}}}]
---

# {{PLAIN_NAME}}: {{ONE_LINE_WHAT_IT_DOES}}

**`{{ORG}}/{{HUB_REPO_NAME}}`** · [Project page](https://ai.ksopyla.com/projects/concept-reasoning/) ·
[Code]({{CODE_URL_AT_COMMIT}}) · [Evaluation details](EVALUATION.md) · [Training on W&B]({{WANDB_URL}})

{{SUMMARY}}  <!-- 2 short paragraphs: size, data, cost; the one new part in plain words with the standing analogy;
                  which project it belongs to; "research release: base model, N seeds, not instruction-tuned". -->

## Highlights

- **{{WIN_1}}** {{NUMBER_UNIT_DIRECTION_AND_THE_MODELS_IT_BEATS}}
- **{{WIN_2}}** …  <!-- 3–5 bullets, wins only, each a measured result with its comparison. The model's distinctive
                       capability first. No "where it is weak" here: trade-offs go to "Scope and trade-offs". -->

<p align="center"><img src="figures/radar.png" width="600" alt="{{RADAR_ALT}}"></p>

*{{RADAR_CAPTION}}*  <!-- six standard benchmarks (BLiMP, ARC-Easy, ARC-Challenge, SciQ, HellaSwag, WikiText), ours +
                         two same-size references, each axis a share of the best of the three, training tokens named. -->

## Results at a glance

{{HARNESS_LINE}}  <!-- harness + version, shots, max length, precision per model, machine, dates -->

| Measure | Chance | **Ours** · {{TOKENS}} | {{SAME_SIZE_REF}} · {{T}} ¹ | {{DATA_MATCHED_REF}} · {{T}} ² | {{MODERN_REF}} · {{T}} ³ |
|---|---:|---:|---:|---:|---:|
| {{BENCHMARK_IN_PLAIN_WORDS}} ({{NAME}}) | {{CHANCE}} | {{V}} ± {{SE}} | {{V}} | {{V}} ({{PUBLISHED}}) | {{V}} |
| **Mean of the core tasks** | | | | | |
| {{TEXT_PREDICTION}}, bits per byte ↓ | — | | | | |
| **{{OUR_DISTINCTIVE_TEST}}** | | | | | |
| {{LEADERBOARD}} place | | | | | |

¹ ² ³ footnotes: which checkpoint and why (earliest public checkpoint for big-data references), precision, where the
bracketed published numbers come from (source, date), what is quoted.

**What "{{UNIT}}" means.** {{ONE_PARAGRAPH: definition, 0 = perfect, the no-skill value, direction, everyday
anchors measured on the same text (gzip / xz), why it is fair across tokenizers, how big a 0.01 difference is}}

## {{DISTINCTIVE_CAPABILITY}}   <!-- e.g. Long documents -->

<p align="center"><img src="figures/{{CHART}}.png" width="860" alt="{{ALT}}"></p>

{{HOW_THE_TEST_WORKS_IN_5_LINES}} · one table: model, size, training tokens (quoted), reach, score, gain with CI,
control · 3 bullets "what this shows" · one line + link to our own diagnostic tests in EVALUATION.md.

## {{LEADERBOARD_NAME}}

{{WHAT_THE_BOARD_MEASURES}} · the board's top ~10 and our neighbourhood, our model **bold** at its computed place,
our runs of the references in *italics* at the place each would get, board entries marked quoted · one line on
placement by quality alone and among entries with a similar training budget · link to the full table.

## Data efficiency

{{HOW_MEASURED}} · `figures/data_eff.png` · 3 bullets: same budget, tokens to match, what it does not separate.

## {{GENERATION_BENCHMARK}}   <!-- e.g. RULER: compact rows, the mean; full table + any correction in EVALUATION.md -->

## What its text looks like

{{SAMPLING_SETTINGS}} (nucleus sampling, seed 0, not picked; never greedy for prose) · 2–3 samples, prompt in bold,
cut at a sentence end · one sentence on the errors visible in them.

## How {{THE_NEW_PART}} works

{{ANALOGY}} · ASCII diagram · one paragraph · what is switched off in this release.

## Training

Identity table: parameters (with and without the token table), shape, tokenizer, budget by stage, optimizer and
seed, machine, time and cost, energy, finish date. Data table by stage with **Hub links to every dataset** and
shares. One paragraph on constructed rows and on which test sets are not in the training data. Per-stage settings in
EVALUATION.md.

## Scope and trade-offs

Facts, not verdicts: base model; tests where references lead, with the training-token ratio that explains them;
the tested context and what happens past it; capabilities that fall with length; controls not run yet (seeds,
same-data baseline); facts and safety.

## What's next

3–5 bullets naming queued or running work (controls, larger runs, features switched off here, `from_pretrained`).

## How to use

If `from_pretrained` works: the 5-line snippet, tested from the Hub id, with the date. If not: two sentences on why,
the engineering task, and the repo commit + loader name. Never code that fails for a reader.

## Citation

```bibtex
@misc{{{CITE_KEY}},
  author       = {Sopyła, Krzysztof},
  title        = {{{PLAIN_NAME}}: {{SUBTITLE}}},
  year         = {{{YEAR}}},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/{{ORG}}/{{HUB_REPO_NAME}}}},
  note         = {Meridian21 Lab, MrCogito project. Code: \url{https://github.com/ksopyla/MrCogito}}
}

@online{sopyla2026mrcogito,
  author  = {Sopyła, Krzysztof},
  title   = {MrCogito: Concept Reasoning Model},
  year    = {2026},
  url     = {https://ai.ksopyla.com/projects/concept-reasoning/},
  urldate = {{{DATE}}}
}
```

## Licence

Weights and card: {{LICENSE}}. Training data licences by source, with the share of tokens under non-standard terms.

## Changelog and contact

- {{DATE}}: first release (weights, card, evaluation page, raw results, figures).

Krzysztof Sopyła · [ai.ksopyla.com](https://ai.ksopyla.com) · [Meridian21 Lab](https://huggingface.co/Meridian21Lab)
· questions and errors: open a discussion on this page.

<!--
EVALUATION.md (same repo, linked at the top) holds: harness and settings (precision per model, start tokens, exclusions
and why); the full benchmark table with ± and every reference checkpoint; the full leaderboard table; the full
data-efficiency table; the cross-model test per budget + head-to-head; our own diagnostic tests; the full generation
benchmark table and any correction notice; sampling settings; what we have not measured; per-stage training
settings; reproducibility (W&B, files, scripts, spec, commit).

STANDARD RULES v3 (checked before every upload; reasons in ../SKILL.md)
1. Wins first, measured; trade-offs as facts in "Scope and trade-offs"; no self-judging labels.
2. README ≤ ~300 lines after the YAML; detail in EVALUATION.md or <details>.
3. One comparison table (models as columns); our row inside the leaderboard table.
4. References chosen before running: same-size strongest, data-matched with checkpoints, a modern one (≤ 12 months),
   earliest public checkpoint for big-data references; base models only.
5. Each model in its own precision; inputs start as its tokenizer starts a text; published numbers in brackets.
6. Every number: unit, direction, comparison; every unit explained once with a scale.
7. Figures by a script in the repo from the result files, light PNG, one idea per chart, numbers also in a table.
8. Text samples sampled (settings stated, seed 0 unpicked), never greedy prose.
9. Generation benchmarks with task-default answer lengths and saved replies.
10. Budget, cost, hardware (generic GPU names, no server names or paths), data mix with links, licences always present.
11. How to use at the bottom; no failing code. BibTeX for model + project page.
-->
