---
# MrCogito model card template
# Fill every {{PLACEHOLDER}}. Delete sections marked OPTIONAL if empty.
# Naming: ksopyla/{line}-{size}-c{C}-{budget}[-v{N}]  (see huggingface-project SKILL)
#
# Inspired by: HF annotated model card, SmolLM2 (TOC + early How-to-use),
# OLMo (identity size/tokens table), Pythia (research intended/out-of-scope honesty).
language:
  - en
license: {{LICENSE}}                    # apache-2.0 | gemma
base_model: {{BASE_MODEL_OR_NULL}}      # e.g. google/gemma-3-1b-pt — omit key if from-scratch
tags:
  - mrcogito
  - {{LINE}}                            # gemma3-concepts | concept-ar | perceiver | …
  - size-{{SIZE}}                       # size-1b
  - concepts-{{C}}                      # concepts-128
  - budget-{{BUDGET}}                   # budget-1bt
  - e{{EXP}}                            # e16b
  - research
  - pytorch
  # add family-specific tags below
library_name: transformers
pipeline_tag: {{PIPELINE_TAG}}          # text-generation | feature-extraction
# Gemma-derived only:
# extra_gated_heading: Access Gemma on Hugging Face
# extra_gated_prompt: >-
#   To access Gemma on Hugging Face, you’re required to review and agree to
#   Google’s usage license. To do this, please ensure you’re logged in to Hugging
#   Face and click below. Requests are processed immediately.
# extra_gated_button_content: Acknowledge license
---

# {{HUB_REPO_NAME}}

{{ONE_PARAGRAPH_SUMMARY}}

> **Claim level:** {{CLAIM_LEVEL}}
> <!-- e.g. mechanism success on registered gate X — not a chat/GLUE SOTA claim -->

## Table of contents

1. [Model summary](#model-summary)
2. [How to use](#how-to-use)
3. [Uses](#uses)
4. [Limitations](#limitations)
5. [Architecture](#architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Citation](#citation)
9. [License](#license)

---

## Model summary

| | |
|---|---|
| **Hub id** | `ksopyla/{{HUB_REPO_NAME}}` |
| **Line / size / C / budget** | `{{LINE}}` · `{{SIZE}}` · `c{{C}}` · `{{BUDGET}}` |
| **Experiment** | {{EXP_ID}} — {{EXP_TITLE}} |
| **Run id** | `{{RUN_ID}}` |
| **Checkpoint** | `{{CHECKPOINT}}` |
| **Parameters** | ~{{PARAMS}} |
| **Tokenizer** | {{TOKENIZER}} |
| **Developed by** | [Krzysztof Sopyła](https://huggingface.co/ksopyla) ([ai.ksopyla.com](https://ai.ksopyla.com)) |
| **Code** | [github.com/ksopyla/MrCogito](https://github.com/ksopyla/MrCogito) |
| **Blog / project** | [Concept Encoder](https://ai.ksopyla.com/projects/concept-encoder/) |
| **Spec** | {{SPEC_URL}} |
| **Run report** | {{REPORT_URL}} |
| **W&B** | {{WANDB_URL}} |
| **Base model** | {{BASE_MODEL_LINK_OR_from-scratch}} |

**What it is.** {{WHAT_IT_IS_2_3_BULLETS_OR_SHORT_PARA}}

**What it is not.** {{WHAT_IT_IS_NOT}}

---

## How to use

Requires the [MrCogito](https://github.com/ksopyla/MrCogito) codebase (custom `{{MODEL_CLASS}}`, not plain `AutoModel*`).

```bash
git clone https://github.com/ksopyla/MrCogito.git
cd MrCogito
uv sync
```

```python
import torch
from transformers import AutoTokenizer
from {{IMPORT_PATH}} import {{MODEL_CLASS}}

repo_id = "ksopyla/{{HUB_REPO_NAME}}"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # use float32 on Apple MPS; bf16/fp16 OK on CUDA if supported

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = {{MODEL_CLASS}}.from_pretrained(repo_id, torch_dtype=dtype).to(device).eval()

# {{USAGE_SNIPPET_COMMENT}}
{{USAGE_CODE}}
```

{{OPTIONAL_RESEARCH_API_SNIPPET}}

---

## Uses

### Direct use

{{DIRECT_USE}}

### Downstream use (optional)

{{DOWNSTREAM_USE_OR_DELETE}}

### Out of scope

{{OUT_OF_SCOPE}}
<!-- Always include: not a production chatbot; not safety-critical; English-centric unless stated -->

---

## Limitations

{{LIMITATIONS_NUMBERED}}
<!-- Always cover: custom code, not IT/chat, generation caveats, honesty about unmeasured probes, license -->

---

## Architecture

### Vision (why concepts)

{{VISION_SHORT}}
<!-- Point to vision_and_goals.md + ai.ksopyla.com. Keep beliefs honest and scoped. -->

Typical long-term stack (direction, not a claim that this checkpoint implements all of it):

```
long input → C concepts (C ≪ N) → reason / refine in concept space → decode to tokens
```

### Where this release sits

{{WHERE_THIS_RELEASE_SITS}}

### Forward pass (keep an ASCII diagram)

```
{{ASCII_DIAGRAM}}
```

{{ARCHITECTURE_MECHANICS_PARAS}}

| Property | Value |
|---|---|
| Family / class | `{{MODEL_CLASS}}` (`{{MODEL_TYPE}}`) |
| Hidden size / layers | {{H}} / {{L}} |
| Concepts (C) | {{C}} |
| Concept block (K) | {{K}} |
| Concept I/O | `{{CONCEPT_IO_MODE}}` |
| Train seq length | {{SEQ}} |
| Other | {{OTHER_ARCH}} |

Implementation: {{CODE_PATH_URL}}

---

## Training

### Data

{{TRAINING_DATA_SUMMARY}}

Required for MrCogito releases: mix id, link to `data/mix_recipes/<id>.json`,
tokenizer Hub id + vocab, seq length, packing policy, and a **proportions table**
(weight · hf_id · subset/config · split · text column · caps). Link launchers /
pretokenize script when relevant.

| Role | Weight | Hub dataset | Config / split | Notes |
|---|---:|---|---|---|
| {{ROLE}} | {{W}} | [{{HF_ID}}](https://huggingface.co/datasets/{{HF_ID}}) | {{CONFIG}} / {{SPLIT}} | {{NOTES}} |

### Tokenizer

| | |
|---|---|
| Name | {{TOKENIZER_HUB_ID}} |
| Vocab size | {{VOCAB}} |
| Shipped in repo | yes / no |

---

### Procedure

| Property | Value |
|---|---|
| Objective | {{OBJECTIVE}} |
| Optimizer | {{OPTIMIZER}} |
| Token / epoch budget | {{BUDGET_DETAIL}} |
| Steps / batch / seed | {{STEPS_BATCH_SEED}} |
| Best metric (train selection) | {{BEST_METRIC}} |

### Compute

| Property | Value |
|---|---|
| Hardware | {{HARDWARE}} |
| Wall time | {{WALL}} |
| GPU-h | {{GPUH}} |
| Energy (if known) | {{KWH_OR_unknown}} |

---

## Evaluation

**Protocol:** {{EVAL_PROTOCOL}}

**Summary:** {{RESULTS_SUMMARY_2_SENTENCES}}

### Primary metrics (registered / claimed)

| Metric | Value | Notes |
|---|---:|---|
| {{M1}} | {{V1}} | {{N1}} |
| {{M2}} | {{V2}} | {{N2}} |

### Not measured / not claimed

| Probe | Status |
|---|---|
| {{PROBE}} | **not run** / **not claimed** |

{{OPTIONAL_BASELINE_CONTEXT_TABLE}}

---

## Citation

```bibtex
@misc{{{CITE_KEY}},
  author       = {Sopyła, Krzysztof},
  title        = {{{HUB_REPO_NAME}} ({{EXP_ID}})},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/ksopyla/{{HUB_REPO_NAME}}},
  note         = {{{CITE_NOTE}}}
}
```

## License

{{LICENSE_BLURB}}

## Model card contact

Krzysztof Sopyła — https://ai.ksopyla.com · https://github.com/ksopyla
