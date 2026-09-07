---
name: huggingface-project
description: Publish and inspect Concept Encoder artifacts on the Hugging Face Hub. Use when uploading checkpoints, generating model cards, checking the ksopyla namespace, searching the Hub via MCP for related models, datasets, or papers, or comparing Hub baselines before release. Not for general research planning or run bookkeeping.
---

# HuggingFace Project Integration

Use this skill for Hugging Face Hub actions and Hub-specific discovery, not for experiment design or training/evaluation tracking.

## Auth & namespace

- Authenticated user: **`ksopyla`** — https://huggingface.co/ksopyla
- Token: `HF_TOKEN` / `HUGGINGFACE_TOKEN` / `HUGGINGFACE_HUB_TOKEN` in `.env`
- MCP: `plugin-huggingface-skills-huggingface-skills` (`hf_whoami`, `hub_repo_search`, `hub_repo_details`, `hf_fs`, …)
- CLI: `hf` (preferred) or `huggingface_hub` Python API

```bash
hf auth whoami
```

## When to publish

Only upload after evaluation results are **promising enough to share honestly**
(see `experiment-track`). Prefer `done_success` / clear mechanism wins. Never
overclaim unmeasured probes.

---

## Naming convention (canonical)

**Do not cram everything into the Hub slug.** Strong Hub practice (SmolLM,
Gemma, BERT-style names) puts **line + size (+ a few disambiguators)** in the
repo id, and puts **budget / compute / wall time / RankMe** in the model card
+ tags. Over-long slugs (every hyperparam in the name) are hard to remember and
still incomplete.

### Repo id template

```text
ksopyla/{line}-{size}-c{C}-{budget}[-v{N}]
```

| Slot | What | Format | Examples |
|---|---|---|---|
| `{line}` | Architecture family (memorable) | lowercase kebab | `gemma3-concepts`, `concept-ar`, `perceiver`, `diffusion` |
| `{size}` | **Weight scale** (params in checkpoint) | `{N}m` / `{N}b` | `61m`, `74m`, `1b`, `3b` |
| `c{C}` | **Concept count** | `c` + int | `c128`, `c256`, `c512` |
| `{budget}` | **Data budget** (primary scale axis) | tokens or epochs | `100mt`, `1bt`, `5ep` |
| `-v{N}` | Optional **release version** of the *same* line/size/C/budget | integer | `-v1`, `-v2` |

**Budget tokens:** `mt` = million tokens, `bt` = billion tokens (non-padding target when that is how the run was scoped). Prefer tokens over steps. Use `ep` only for epoch-scoped from-scratch runs where tokens are awkward.

**Size:** round to a human number from checkpoint param count (`~1.01B` → `1b`, `~74M` → `74m`). For grafted models, size is the **full checkpoint** (backbone + LoRA + concepts), not “trainable-only”.

### What stays OUT of the slug

| Factor | Where it lives |
|---|---|
| Experiment id (`E16b`) | card title + tag `e16b` + body |
| Training run folder / W&B id | card body |
| Seq length, mix id, optimizer | card Training table |
| Wall time, GPU-h, kWh | card Compute table |
| RankMe / ΔCE / STS-B | card Evaluation + optional `model-index` |
| Checkpoint step (`7900`) | card; git revision note if needed |

### Line vocabulary (extend carefully)

| `{line}` | Family |
|---|---|
| `gemma3-concepts` | `BackboneConceptLM` on Gemma 3 |
| `concept-ar` | from-scratch AR / prefix→suffix concept LM |
| `perceiver` | Perceiver / BiXT MLM–style encoder |
| `diffusion` | parked / revived diffusion concept models |

New lines need a one-word family name people can say aloud.

### Concrete examples

| Release | Hub repo id |
|---|---|
| E16b Gemma graft, C=128, 1B tok | **`ksopyla/gemma3-concepts-1b-c128-1bt`** |
| Same config, second public cut | `ksopyla/gemma3-concepts-1b-c128-1bt-v2` |
| E16a-scale pilot if ever published | `ksopyla/gemma3-concepts-1b-c128-100mt` |
| E02-style AR ~74M, C=128, 5 ep | `ksopyla/concept-ar-74m-c128-5ep` |
| Old Perceiver ~61M MLM | `ksopyla/perceiver-61m-c128-minipile` (or keep legacy id) |

Mnemonic: **line → how big → how many concepts → how much data → (version)**.

### Versioning policy

1. **Same weights, card/docs fix** → commit on the **same** repo (no new id).
2. **Same line/size/C/budget, new training** (rerun, better ckpt) → **`-v2`** (or bump) on a new repo, set `new_version:` on the old card pointing forward.
3. **Different C, size, or budget** → **new slug** (those are identity, not version).
4. Mid-run / “last vs best” → one repo; document which ckpt in the card (do not create `…-ckpt7900` repos).

Optional: group a line under an HF **Collection** (`MrCogito gemma3-concepts`).

### Required tags (machine-readable twin of the slug)

Always add:

```yaml
tags:
  - mrcogito
  - {line}              # e.g. gemma3-concepts
  - size-{size}         # size-1b
  - concepts-{C}        # concepts-128
  - budget-{budget}     # budget-1bt
  - e{Exp}              # e16b
  - research
```

Optional compute tags when known: `gpuh-115`, `seq-4096` (useful for Hub search; still duplicate in the card table).

### Required card tables (every upload)

**Identity**

| Field | Example |
|---|---|
| Hub id | `ksopyla/gemma3-concepts-1b-c128-1bt` |
| Experiment | E16b |
| Run id | `backbone_concept_…150850` |
| Checkpoint | `checkpoint-7900` |
| Params | ~1.01B |
| Concepts C / block K | 128 / 512 |

**Compute** (honest; “unknown” allowed)

| Field | Example |
|---|---|
| Token budget | 1B non-padding |
| Wall time | ~38.3 h |
| GPU-h | ~115 |
| Energy | ~34.4 kWh |
| Hardware | 3× RTX 3090 (Odra) |

Do **not** invent GPU-h. Pull from run report / audit when available.

---

## License rules

| Weights | HF `license:` | Notes |
|---|---|---|
| From-scratch MrCogito (no Gemma) | `apache-2.0` | |
| Gemma-derived (`google/gemma-3-*`) | **`gemma`** | `base_model:` + gated prompt; never Apache |

## Model card checklist

1. Links: GitHub, [ai.ksopyla.com](https://ai.ksopyla.com) (+ project page), W&B, spec, run report
2. Naming slots reflected in title + tags
3. Identity + Training + **Compute** + Evaluation tables
4. Honest quality: mark **not run** for unmeasured probes
5. Limitations + external load recipe (clone MrCogito)
6. Citation + correct license

### Templates

| File | Role |
|---|---|
| [templates/MODEL_CARD.md](templates/MODEL_CARD.md) | **Canonical** card structure for all MrCogito Hub releases |
| [templates/backbone_concept_model_card.md](templates/backbone_concept_model_card.md) | Older Gemma-oriented draft — prefer `MODEL_CARD.md` |

Reference filled card: `Cache/HF_staging/gemma3-concepts-1b-c128-1bt/README.md` (E16b draft).

Card structure follows HF’s annotated model card + patterns from SmolLM2
(TOC + early How-to-use), OLMo (identity tables), and Pythia (research
intended / out-of-scope honesty).

## Upload workflows

### A. Named release (all new uploads)

1. Choose Hub id with the template above **before** staging.
2. Stage weights (no optimizer / trainer_state) + filled README:

```bash
STAGE=Cache/HF_staging/<hub-repo-name>
mkdir -p "$STAGE"
rsync -a --exclude='optimizer*' --exclude='rng_state*' --exclude='scheduler*' \
  --exclude='trainer_state.json' --exclude='training_args.bin' \
  Cache/Training/<run>/checkpoint-<N>/ "$STAGE/"
```

3. Upload:

```bash
hf upload ksopyla/<hub-repo-name> "$STAGE" --repo-type model \
  --commit-message "Add <Exp> <hub-repo-name> + model card"
```

4. Verify card, tags, license gate; smoke-load from Hub id.

### B. Legacy interactive script (old Perceiver only)

```bash
uv run python scripts/upload_model_to_hf.py --list-only
uv run python scripts/upload_model_to_hf.py --run-name perceiver_mlm_H512L6C128_20260208
```

Do **not** use this for `backbone_concept` — naming + card must follow section Naming convention.

## External load (`BackboneConceptLM`)

```python
from transformers import AutoTokenizer
from nn.backbone_concept_lm import BackboneConceptLM

tok = AutoTokenizer.from_pretrained("ksopyla/gemma3-concepts-1b-c128-1bt")
model = BackboneConceptLM.from_pretrained("ksopyla/gemma3-concepts-1b-c128-1bt", torch_dtype=…)
```

Needs project deps (`peft`, …). MPS → `float32`. Not plain `AutoModelForCausalLM`.

## After publish

- Link Hub URL from run report / ledger note (append-only).
- Optionally note in `CHANGELOG.md` when doing engineering-change-tracking.
