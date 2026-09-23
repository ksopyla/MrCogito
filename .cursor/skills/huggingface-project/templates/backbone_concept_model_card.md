---
# TEMPLATE — BackboneConceptLM (Gemma graft) model card
# Copy into the uploaded repo as README.md. Replace {{PLACEHOLDERS}}.
# License for Gemma-derived weights must remain `gemma` (not Apache-2.0).
language:
  - en
license: gemma
base_model: google/gemma-3-1b-pt
tags:
  - mrcogito
  - concept-encoder
  - backbone-concept
  - concept-bottleneck
  - gemma3
  - gemma
  - lora
  - causal-lm
  - research
  - pytorch
library_name: transformers
pipeline_tag: text-generation
extra_gated_heading: Access Gemma on Hugging Face
extra_gated_prompt: >-
  To access Gemma on Hugging Face, you’re required to review and agree to
  Google’s usage license. To do this, please ensure you’re logged in to Hugging
  Face and click below. Requests are processed immediately.
extra_gated_button_content: Acknowledge license
---

# {{TITLE}}

Research checkpoint from **[MrCogito](https://github.com/ksopyla/MrCogito)** —
a concept-reasoning architecture that grafts a compact recurrent **concept memory**
onto a frozen pretrained decoder (here: Gemma 3).

| | |
|---|---|
| Experiment | {{EXPERIMENT_ID}} |
| Run | `{{RUN_ID}}` |
| Checkpoint | `{{CHECKPOINT}}` |
| Project blog | [ai.ksopyla.com](https://ai.ksopyla.com) · [Concept Encoder](https://ai.ksopyla.com/projects/concept-encoder/) |
| Code | [github.com/ksopyla/MrCogito](https://github.com/ksopyla/MrCogito) |
| Spec / report | {{SPEC_LINK}} · {{REPORT_LINK}} |
| W&B | {{WANDB_LINK}} |

## What this model is (and is not)

**Is:** a block-recurrent causal LM. Tokens are processed in `concept_block`-sized
chunks; after each global Gemma layer the model **reads** and **writes** a shared
concept state (`C` vectors). Concepts can carry information across blocks.

**Is not:** an instruction-tuned chat model, a drop-in `AutoModel` without the
MrCogito code, or a claim of GLUE/chat SOTA. Prefer continuation prompts
(`Once upon a time…`) over chat instructions.

## Architecture

```
prompt tokens ──► Gemma blocks (K={{CONCEPT_BLOCK}}) ──► next-token logits
                      │  read / write (shared_depth_recurrent)
                      ▼
               concept state [B, C={{CONCEPT_NUM}}, H]
```

| Property | Value |
|---|---|
| Backbone | `{{BACKBONE}}` (frozen) + LoRA r={{LORA_R}} |
| Hidden size | {{HIDDEN_SIZE}} |
| Layers | {{NUM_LAYERS}} |
| Concepts (C) | {{CONCEPT_NUM}} |
| Concept block (K) | {{CONCEPT_BLOCK}} |
| Concept I/O | `{{CONCEPT_IO_MODE}}` |
| Train seq length | {{TRAIN_SEQ}} |
| Checkpoint params | ~{{PARAMS}} |
| Tokenizer | same as backbone (shipped in this repo) |

## Training

| Property | Value |
|---|---|
| Objective | causal next-token CE (block-recurrent) |
| Mix | `{{MIX_ID}}` |
| Optimizer | {{OPTIMIZER}} |
| Budget | {{TOKEN_BUDGET}} |
| Seed | {{SEED}} |

## Evaluation (be honest)

Fill only metrics that were actually measured. Mark unmeasured probes as **not run**.

### Mechanism / concept health (primary for this family)

| Metric | Value | Notes |
|---|---:|---|
| within-sample RankMe | {{RANKME}} | geometry |
| Δshuffle beyond-local | {{DELTA_SHUFFLE}} | causal use |
| Δstatic beyond-local | {{DELTA_STATIC}} | causal use |
| Δone-block beyond-local | {{DELTA_ONE_BLOCK}} | recurrence vs prior-block |

### Downstream / generation

| Probe | Status / score |
|---|---|
| STS-B zero-shot | {{STSB}} |
| SICK / PAWS / GLUE | {{GLUE_FAMILY}} |
| Generation vibe-check | {{GEN_QUALITY}} |

## Known limitations

- Custom `BackboneConceptLM` — clone MrCogito; not plain `AutoModel.from_pretrained`.
- Research checkpoint; generation can be fluent locally and still fail long-range tasks.
- No KV cache in the shipped generate loop — keep `max_new_tokens` modest.
- On Apple MPS use `dtype=torch.float32` (fp16 often NaNs).
- Gemma Terms of Use apply to the backbone-derived weights.

## How to load (external researcher)

```bash
git clone https://github.com/ksopyla/MrCogito.git
cd MrCogito
uv sync   # or: pip install -e .
```

```python
import torch
from transformers import AutoTokenizer
from nn.backbone_concept_lm import BackboneConceptLM

repo_id = "{{HF_REPO_ID}}"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # use float32 on MPS

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = BackboneConceptLM.from_pretrained(repo_id, torch_dtype=dtype).to(device).eval()

prompt = "Once upon a time, in a quiet library,"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.95)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Concept ablation modes (research): `concept_mode="real"|"zero"|"shuffle"|"static"|"one_block"`
on `generate` / forward helpers — see `nn/backbone_concept_lm.py`.

## Citation

```bibtex
@misc{{{CITE_KEY}},
  author       = {Sopyła, Krzysztof},
  title        = {{{TITLE}}},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/{{HF_REPO_ID}}},
  note         = {MrCogito BackboneConceptLM research checkpoint ({{EXPERIMENT_ID}})}
}
```

## License

Gemma Terms of Use apply (`license: gemma`). See
[google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt) and
[Google's Gemma license](https://ai.google.dev/gemma/terms).
