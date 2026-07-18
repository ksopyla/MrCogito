# E0NN — Implementation Plan

- **Spec:** [E0NN_*.md](E0NN_*.md) · **Status:** draft | approved | implemented
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for the spec's **architectural bet**. Repo-rooted: cite real classes/paths you read.
> Reuse infra; invent mechanism when the bet needs it. New code only as a reusable,
> config-selectable foundation component. Do **not** derisk a bold spec into a safe retread.

## 1. Source & fit
- **Origin:** <prior result / paper / vision / cross-domain analogy / idea> — link `research-explain` walkthrough and/or `docs/literature_review/<file>.md#anchor`.
- **Synthesis verdict:** <Adopt / Adapt — one line from research-synthesis; what we take, what we drop>.
- **Architecture mapping:** this touches <encoder | concept bottleneck | reasoning | decoder/head | loss | data> (a coherent bet may span more than one — say so).
- **Boldness check:** <confirm this plan implements the novel claim, not a safer substitute>.

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `<ClassName>` | reuse as-is | `nn/…py` |
| `<ClassName>` | extend (add knob) | `nn/…py` |
| `<NewComponent>` | new — reusable, config-selectable | `nn/…py` |

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `N`=tokens, `C`=concepts, `H`=hidden, `V`=vocab.
```
(B, N)            → embed (+pos)                         → (B, N, H)
(B, N, H)         → encode: cross-attn(Q=concepts,KV=tokens) → (B, C, H)   # ConceptEncoder
(B, C, H)         → [reason: concept→concept, optional]  → (B, C, H)
(B, C, H)         → decode/head (concept-conditioned)    → (B, *, V)       # keep O(C·N)
```

## 4. Inputs & data
- **Dataset:** <id + size> · **Collator:** `data/data_collators.py:<...>` (reuse / change: …) · **Preprocessing / masking / split:** …

## 5. Loss & training objective
- **Loss:** `loss_manager.py` components <…> / new loss via `register_loss` · **Objective:** <next-token CE / denoise / …> · **Weighting:** <…>.

## 6. Config & launch
- **New config fields** (backward-compatible defaults): `ConceptEncoderConfig.<field> = <default>` / `LossConfig.<…>`.
- **Registry:** `MODEL_REGISTRY["<type>"]` (+ eval routing / `run_concept_analysis.py` if trainable+evaluated).
- **Launch:** `<VAR=… bash scripts/<launcher>.sh>` (env-var overrides; new knobs added to the existing launcher).

## 7. Tests & smoke
- Unit test in `tests/test_<…>.py`: assert shapes + loss finite (tiny random tensors).
- Local MPS smoke: `<command>` — assert it runs a few steps; sanity via `run_concept_analysis.py` if concept geometry matters.

## 8. Risks & tradeoffs
- **Risk:** <what could fail / not transfer>. **Cheapest signal:** <metric tied to the spec's success/kill>. **Fallback:** <…>.

## 9. Code sketches (optional, `# sketch` — decisions, not demos)
```python
# sketch: interface only — signatures / shapes / config fields that pin a decision
```
