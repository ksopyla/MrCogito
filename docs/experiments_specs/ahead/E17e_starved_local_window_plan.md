# E17e — Starve the local window (K=256) — Implementation Plan

- **Spec:** [E17e_starved_local_window.md](E17e_starved_local_window.md) · **Status:** implemented
- **Authored by:** `implementation-plan` · for → `research-implement`
- **Token budget:** **300M** non-padding tokens (`TARGET_TOKENS=300000000`).
- **ID:** **E17e**. `E18` is already claimed on an unmerged addressable-RAM branch.

> Implement the spec's starve: E17d cell unchanged; `concept_block=256` is the
> local window and the write cadence. Patch loaded Gemma `sliding_window` copies
> so construction does not raise. Do not drop token–token attention at global
> layers, do not warm-start E17d, do not launch 1B.

## 1. Source & fit
- **Origin:** E17d
  (`backbone_concept_gemma_3_1b_pt_K512_concept_20260817_141227`, `checkpoint-2660`)
  proved attn-residual + no token carry keeps RankMe healthy (43–77) and spreads
  the block-start gist (first-64 Δperm 0.75, multi-bank) but late-half Δperm
  stays **0.044**. Five-whys: after ~64 tokens the K=512 local softmax already
  solves FinePDFs CE. See
  [E17d spec](../ahead/E17d_global_concept_assimilation.md) and
  [five-whys](../../4_Research_Notes/e17c_failure_five_whys_20260815.md).
- **Synthesis verdict:** **Adapt** the SWA-hybrid starve (Infini / NHA w shrink):
  memory is unused until the window cannot cover next-token CE. Take E17d's cell;
  drop "leave W=512". Do not Adopt concept-only global layers or MQAR in this
  spec — those are the follow-up if 256 still suffices.
- **Architecture mapping:** local token mask + write-loop block size. Frozen
  Gemma, LoRA, tokenizer, collator, mix, LM head, attn-residual read, additive
  writes, and no-carry policy stay.
- **Boldness check:** this plan implements K=256 starve, not a gate-init A/B and
  not a quieter "maybe also try 384".

## 2. Reuse map (read the modules first)

| Component | Action | Where |
|---|---|---|
| E17d cell (`attn_residual`, drop carry, untied additive) | reuse as-is | `nn/backbone_concept_lm.py` |
| `align_backbone_sliding_window` | **new** helper: patch config + each `Gemma3Attention.sliding_window` | `nn/backbone_concept_lm.py` |
| `BackboneConceptLM.__init__` | call align **before** the `concept_block == sliding_window` check | `nn/backbone_concept_lm.py` |
| `_windowed_causal_mask` / `_forward_blocks` | reuse: they already key off `config.concept_block` | `nn/backbone_concept_lm.py` |
| `INTRA_BLOCK_BINS` | reuse: frac 0.5–1.0 is the late-half key `256_512` | `nn/backbone_concept_lm.py` |
| `ModelArguments.concept_block` | help text: authority for window; Gemma is aligned | `training/concept_pretraining_args.py` |
| `scripts/launch_e10.sh` | `export CONCEPT_BLOCK="${CONCEPT_BLOCK:-512}"` so wrappers can override | `scripts/launch_e10.sh` |
| `scripts/launch_e17d.sh` | pin `CONCEPT_BLOCK=512` explicitly | `scripts/launch_e17d.sh` |
| Thin wrapper | new `scripts/launch_e17e.sh` = E17d pins + `CONCEPT_BLOCK=256` | `scripts/launch_e17e.sh` |
| Unit tests | extend backbone + launcher flow | `tests/test_backbone_concept_lm.py`, `tests/test_training_launcher_parameter_flow.py` |

**No new model class, training entrypoint, collator, or `LossManager` loss.**
Defaults stay K=512 so E10–E17d load paths are unchanged.

### Backward-compatibility boundary
```python
concept_block = 512          # default; E17e sets 256
# after align_backbone_sliding_window:
backbone.config.sliding_window == concept_block
layer.self_attn.sliding_window == concept_block
```
Old K=512 checkpoints omit nothing new; they still construct. Do not rename
`concept_block`.

## 3. Forward pass (tensor shapes)

Symbols: `B`=microbatch, `N=4096`, **`K=256`**, `G=4`, `C=128`, `H=1152`, `V`=Gemma vocab.
Global layers `g ∈ {5,11,17,23}` (tiny tests: two globals at 5 and 11 of 12).

```text
z = concept_init                                      [B, G, C, H]
for window b = 0 .. 15:                               # 4096/256 = 16, not 8
    tokens = current K (carry masked on every b>0)
    h = embed(tokens)                                 [B, Q, H], Q ≤ 2K
    for each Gemma layer:
        if layer is global g:
            read  z[:, g] from window b-1             [B, C, H]
            mix into attention residual               [B, Q, H]
            FFN on that mix
            write z[:, g] from h[:, -K:]              [B, C, H]
        else:
            sliding-window token attn with W=K=256
    score only current-window next-token targets
```

Layer 11 in window `b` still reads **bank 11 from window `b-1`**, never bank 5's
same-window write. That schedule is unchanged; only `K` changes.

## 4. Inputs & data
- **Dataset:** `e16b_long_4k_v1` Gemma 4k mix · seq 4096 · `SKIP_PRETOKENIZE=1`.
- **Collator:** existing causal LM collator; no change.
- **Split:** same immutable held-out as E17d.

## 5. Loss & training objective
- **Loss:** uniform next-token CE. `MEMORY_PRESSURE_TOKENS=0`.
- **Objective:** same as E17d. The starve is architectural, not a new loss.

## 6. Config & launch
- **No new config fields.** `concept_block=256` plus the existing E17d knobs.
- **Launch:** `SKIP_PRETOKENIZE=1 bash scripts/launch_e17e.sh`

```bash
EXPERIMENT_ID=E17e
CONCEPT_BLOCK=256
# remainder identical to launch_e17d.sh
```

`launch_e10.sh` currently **overwrites** `CONCEPT_BLOCK=512`. That must become a
default so E17e's export survives `exec`.

## 7. Tests & smoke
Extend `tests/test_backbone_concept_lm.py` (tiny random Gemma). Assert:

- Building with `backbone_config.sliding_window=8` and `concept_block=4` **does
  not raise**; both `backbone.config.sliding_window` and each
  `Gemma3Attention.sliding_window` equal 4.
- `_windowed_causal_mask` allows dist `< 4` and blocks dist `≥ 4`.
- On S=24, K=4 vs K=8, write-head `forward` is called **2× as often** (6 blocks
  vs 3, times G=2 banks).
- Matched E17d K=8 still constructs (align is a no-op when already equal).
- `concept_ablation_ce` still emits `delta_permutation_block_256_512` (frac
  0.5–1.0 of K).

Launcher: `tests/test_training_launcher_parameter_flow.py`

- `launch_e10.sh` with no override still passes `--concept_block 512`.
- `CONCEPT_BLOCK=256 bash scripts/launch_e10.sh` forwards 256.
- `launch_e17e.sh` pins `--concept_block 256` and the E17d cell knobs.
- `launch_e17d.sh` still pins `--concept_block 512`.

Local smoke: `uv run pytest tests/test_backbone_concept_lm.py tests/test_training_launcher_parameter_flow.py -q`.
Polonez launch is `SKIP_PRETOKENIZE=1 bash scripts/launch_e17e.sh` after approval
of this plan (do not start 1B).

## 8. Risks & tradeoffs
- **Risk:** 256 tokens of local context still covers FinePDFs CE (E17d's 128–256
  bin was already 0.10). **Cheapest signal:** 100M late-half Δperm `< 0.03` —
  kill; next is concept-only global layers or Bet B, not K=128 in this ID.
- **Risk:** pretrained Gemma SWA was trained at 512; LoRA may not adapt the
  shorter mask. **Signal:** eval_loss vs E17d 2.365; crash bar is 2.70.
- **Risk:** 16 windows vs 8 changes activation memory / tok/s. **Fallback:**
  retune microbatch by real tok/s only; do not change K.
- **Do not fall back to:** raising write/read gates, VICReg, or keeping W=512
  while only changing the write loop (that would not starve the local computer).

## 9. Code sketches (decisions, not demos)

```python
# sketch: concept_block is the authority
def align_backbone_sliding_window(backbone, concept_block: int) -> None:
    k = int(concept_block)
    cfg = backbone.config
    cfg.sliding_window = k
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "sliding_window"):
        text_cfg.sliding_window = k
    for layer in _backbone_layers(backbone):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "sliding_window"):
            attn.sliding_window = k

# sketch: launch_e10.sh
# export CONCEPT_BLOCK="${CONCEPT_BLOCK:-512}"   # was: CONCEPT_BLOCK=512
```
