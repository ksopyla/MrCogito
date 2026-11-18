---
name: huggingface-project
description: The Hugging Face Hub mechanics for Concept Encoder / Meridian21 artifacts - repo naming, tags, licence gates, the Meridian21Lab organisation, staging and uploading checkpoints (private first, then public), checking the namespace, and searching the Hub via MCP for related models, datasets or papers. Use when uploading or renaming a Hub repo, setting card metadata, or looking up Hub models. For what a model card says, which models to compare against and how to present results, use the model-release skill (it owns the card template). Not for general research planning or run bookkeeping.
---

# HuggingFace Project Integration

Use this skill for Hugging Face Hub actions and Hub-specific discovery, not for experiment design or training/evaluation tracking.

## Auth & namespace

- Authenticated user: **`ksopyla`** — https://huggingface.co/ksopyla
- **Community releases** go to the **`Meridian21Lab`** organisation (author's decision 2026-11-17: public, Apache-2.0
  for from-scratch models). Research-only uploads may stay under `ksopyla`. Agents never read, print or enter the
  HF token; uploads use the login already stored on the server, and its write rights are the author's to grant.
- Token: `HF_TOKEN` / `HUGGINGFACE_TOKEN` / `HUGGINGFACE_HUB_TOKEN` in `.env` (user-scoped: no write right on
  Meridian21Lab). **Meridian21Lab writes:** the fine-grained token `HF_TOKEN_MERIDAIN21_ORG` in the main checkout's
  `.env` on the Mac (not on Odra or Polonez, 2026-11-18): read it inside the upload script, never print or echo it;
  the weights may need to come to the Mac first (tar over ssh from the server, then check the export manifest's
  hashes). An expired cached login breaks even public reads (401): use `HfApi(token=False)` for those.
- **Rename while private:** `HfApi.move_repo(from_id, to_id)` keeps the history and redirects the old id; do it
  before the repo goes public (the author renamed `lsa-140m-11.5bt` to `topic-index-140m-11.5bt`, 2026-11-18).
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
{org}/{line}-{size}[-c{C}]-{budget}[-v{N}]      # org = Meridian21Lab (releases) or ksopyla (research)
```

| Slot | What | Format | Examples |
|---|---|---|---|
| `{line}` | Architecture family (memorable) | lowercase kebab | `gemma3-concepts`, `concept-ar`, `perceiver`, `diffusion` |
| `{size}` | **Weight scale** (params in checkpoint) | `{N}m` / `{N}b` | `61m`, `74m`, `1b`, `3b` |
| `c{C}` | **Concept count**, only when fixed (omit when concepts grow with the document, e.g. concept memory) | `c` + int | `c128`, `c256`, `c512` |
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
| `concept-memory` | `perceiver_ar` with the concept-memory read (Meridian21Lab, 2026-11) |
| `topic-index` | `perceiver_ar` with our port of Latent Sparse Attention (the method keeps its authors' name) |

New lines need a one-word family name people can say aloud, describing what the model does in plain words; a
borrowed method keeps its authors' name for the method, not for our model (model-release rule 16). Elsewhere a
model is referred to by its Hub id, never as a "sibling" or by an internal code.

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

## Model card content → `model-release` skill

What the card says, which references to run, how results are shown (one comparison table, the leaderboard table,
radar and long-context charts), the units, text samples, tone, and the pre-publish checklist all live in the
**`model-release`** skill, with the card template (standard v3) in
`.cursor/skills/model-release/references/card_template.md`. Reference filled card:
`docs/release/hf_card_concept_memory_150m/` (README.md + EVALUATION.md).

This skill keeps the Hub side of the checklist:

1. Hub id, tags and `license:` follow the naming and licence sections above; `library_name: transformers` and the
   `custom_code` tag only after the Hub smoke test (step 3) passed.
2. Upload the export folder (`scripts/export_hub_model.py`, section below: modeling and configuration code,
   `config.json` with `auto_map`, weights, tokenizer, `export_manifest.json`) together with the release folder
   (README.md, EVALUATION.md, figures/, eval_results/, long_context/, samples/, ruler/), **private first**; open the
   rendered page, check figures, links and tables.
3. **Hub smoke test, from the Hub id, before the repo goes public:** in an environment with only `torch` and
   `transformers` (a fresh venv; `-I` keeps the repo's `nn/` off the path), on the server whose stored login can read
   the private repo:

   ```bash
   python -I scripts/hub_smoke_test.py Meridian21Lab/<repo>
   ```

   It loads with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`, checks the class comes from the
   repo's own code, scores one sentence (finite loss) and samples 20 tokens (seed 0, temperature 0.7, top-p 0.9);
   exit 0 = pass. Save its JSON line with the release files. A failure blocks the public switch and the
   `from_pretrained` snippet in the card.
4. Make it public; record the Hub URL in the run report.

### Templates

| File | Role |
|---|---|
| [`model-release/references/card_template.md`](../model-release/references/card_template.md) | **Standard v3** card structure + rules for all MrCogito / Meridian21 Hub releases |
| [templates/backbone_concept_model_card.md](templates/backbone_concept_model_card.md) | Older Gemma-oriented draft — prefer the v3 template |

Older draft: `Cache/HF_staging/gemma3-concepts-1b-c128-1bt/README.md` (E16b).

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

4. Verify card, tags, license gate; run the Hub smoke test from the Hub id (`scripts/hub_smoke_test.py`, checklist
   step 3 above). For `perceiver_ar`, stage with `scripts/export_hub_model.py` instead of step 2's copy.

### B. Legacy interactive script (old Perceiver only)

```bash
uv run python scripts/upload_model_to_hf.py --list-only
uv run python scripts/upload_model_to_hf.py --run-name perceiver_mlm_H512L6C128_20260208
```

Do **not** use this for `backbone_concept` — naming + card must follow section Naming convention.

## Loading with `from_pretrained` (`perceiver_ar`, since 2026-11-18)

`scripts/export_hub_model.py <checkpoint> <out_dir> [--card README.md]` writes a Hub folder that loads with
`AutoModelForCausalLM.from_pretrained(<id>, trust_remote_code=True, dtype=torch.bfloat16)`: the `nn/` modules the
checkpoint's config uses, vendored into flat `modeling_perceiver_ar.py` / `configuration_perceiver_ar.py` (generated,
never edited by hand: change the source and export again), `auto_map`, `use_liger: false`, the Hub tokenizer id,
`attn_backend: sdpa` (pass `attn_backend="flex"` for long inputs on CUDA), weights and tokenizer, and
`export_manifest.json` (source commit, changed config fields, sha256). 🤗 `generate` (greedy, sampling, beam search)
runs over our cached decoder (`nn/perceiver_ar_hub.py`); `labels=` gives our eval loader's per-token loss.
Tests: `tests/test_hub_export.py` (CPU always; `HUB_EXPORT_CHECKPOINT=<dir>` adds the real checkpoint on CUDA).
Spec: `docs/engineering_specs/hub_from_pretrained_loading.md`. Other families (`perceiver_concept`, Gemma grafts)
have no export yet.

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
