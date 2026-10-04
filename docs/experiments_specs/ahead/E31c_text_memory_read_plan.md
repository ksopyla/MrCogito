# E31c — Implementation Plan

- **Spec:** [E31c_text_memory_read.md](E31c_text_memory_read.md) · **Status:** implemented (code + CPU tests); GPU kernel check and runs pending
- **Authored by:** `implementation-plan` · for → `research-implement` · branch `e31-text-memory` (off `dev` @ 457c826)

> Change tracker for the E31 → text step. Rule for every row: **defaults reproduce E31 / E33a bit
> for bit**; text behaviour is opt-in through two config values.

## 1. Source & fit
- **Origin:**
  - the E31 spec's own "memory visibility (text vs probe)" note: text reads the windows that closed
    before `t`;
  - E31b's length-invariant champion (`e31_li_m1`: slot keys at QUERY = content-only matching);
  - the author's roadmap (2026-10-03): text at about 256k first, no two-level notebook.
- **Verdict:** Adapt. The read rule and slot positions are translated from exam to text; everything
  else is kept.
- **Mapping:** decoder-side read (mask + key positions) and the trainer wiring. The writer, the main
  path and the loss are unchanged.
- **Boldness check:** the bet is "E31 as tested works on text". Not changing the architecture
  beyond the translation is the point.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| `LatentMemoryWriter` | extend: `lm_slot_pos="reader"` returns slot keys **un-rotated** | `nn/latent_memory.py` |
| `window_pick` | reuse as-is (all-zero `side` in text = "one document per window") | `nn/latent_memory.py` |
| `PerceiverARConfig` | new `lm_read ∈ {exclusive, closed}`; `lm_slot_pos` gains `reader`; `message_enabled` true for closed latent memory without a boundary id | `nn/perceiver_ar_lm.py` |
| `MessageCtx` | new fields `read_rule`, `slot_close` [B,nb], `slot_nope` | `nn/perceiver_ar_lm.py` |
| `_message_context` / `_lm_slot_tensors` | closed mode: build ctx with no QUERY in the batch (side = 0); slot close = last picked token of the window | `nn/perceiver_ar_lm.py` |
| `make_message_mask_pred` / `dense_message_mask` | closed rule: same doc ∧ close ≤ q (flex: packed into the existing int32 slot tag) | `nn/perceiver_ar_lm.py` |
| `_message_extra_blocks` | closed mode: slot blocks whose earliest close ≤ the query block's last token | `nn/perceiver_ar_lm.py` |
| `attend_message` + `Attention.forward` | `slot_nope`: one softmax over `[R(p)q, q]·[R(j)k,0] ‖ [0,k_s]`, scale `1/√dh` | `nn/perceiver_ar_lm.py` |
| text trainer | pass-through args (memory, read rule, raw window, loop, override) | `training/concept_pretraining_args.py`, `training/concept_pretraining_factories.py`, `scripts/train_concept_pretraining_multigpu.sh` |
| suite variants | `e31c_m1`, `e31c_loop` (+ `ArchSpec.lm_read`, ARCH_FLAGS) | `evaluation/bapo_models.py`, `evaluation/capability_suite.py` |
| long-context probes | `--message_override none` (notebook removed at eval) | `evaluation/long_context_probes.py` |

Not touched:
- the `e33a_loop` entry (another branch edits it);
- `BATTERY_VARIANTS` (it lives on the capability-process branch; `e31c` is added there after
  merge);
- the writer's maths, the loop code and the loss.

## 3. Forward pass (shapes, 30M suite shape at 16k)
Symbols: B batch, N = 16 384 tokens, H = 960, h = 15 heads, g = 1 KV head, dh = 64; windows
n_w = 85 (W 256 / stride 192), K = 32 latents, m = 1 → C = 2 720 notes.
```
ids (B,N) → embed (B,N,H) → layer 0 local(16)
writer: tok_emb (B,N,128) → windows (B·85,256,256) → latents (B·85,32,512) → k_s,v_s (B,2720,1,64)
        slot_doc (B,2720), slot_close (B,2720) = window start + last picked index      [new]
        reader mode: k_s NOT rotated                                                     [new]
global read: q (B,N,15,64) rotated → Q2 = [R(p)q, q] (B,N,15,128)                       [new]
             K2 = [R(j)k,0] (B,N,1,128) ‖ [0,k_s] (B,2720,1,128);  V = v ‖ v_s (…,64)
             mask: raw  j ≤ q, q−j < 256, same doc & side      (E31, unchanged)
                   slot same doc ∧ slot_close ≤ q                                         [new, closed]
→ layers 2, 3 local → head
```
Per token: ≤ 256 raw keys + the closed notes, which number N/6 at most and about half that on
average. At 256k that is about 21k notes on average per token at eval.

## 4. Inputs & data
- **Text:** no QUERY token. `message_boundary_token_id = -1` and `lm_read = closed`, so side is 0
  everywhere and the notebook is always on.
- **Packed rows:** windows sit at row positions. A window that straddles two documents writes only
  from the document of its last token (`window_pick`). The earlier document's tail stays readable
  raw, and no later token of that document exists.
- **Datasets (run phase, not code):**
  - E22 long-document sources rebuilt at 16k;
  - E18b keyed-recall rows from `scripts/build_retrieval_mix_dataset.py --context 16384` at about
    25 % of tokens;
  - a fluency tier; the manifest is written by `scripts/write_manifest_variant.py`.

## 5. Loss & objective
Unchanged next-token CE (chunked soft-cap). Keyed-recall rows carry dense labels from their markers
(the E18b collator path). With the loop arm, the answer exits are general (next-token targets).

## 6. Config & launch
New config (defaults = E31):
```python
# sketch
lm_read: str = "exclusive"       # "closed": every token reads the notes of windows closed at or before it
lm_slot_pos: str = "read"        # + "reader": slot keys un-rotated, met by the un-rotated query (NoPE slots)
```
- **Validation:**
  - closed needs `message_write = latent_memory`;
  - `message_boundary_token_id` may be −1 only in closed mode;
  - the flex slot tag needs `n_doc · S < 2³¹`.
- **Trainer args:** `message_write`, `message_raw_window`, `message_override`, `lm_read`,
  `lm_window`, `lm_stride`, `lm_latents`, `lm_latent_dim`, `lm_heads`, `lm_writer_dim`,
  `lm_enc_layers`, `lm_rounds`, `lm_reader_tokens`, `lm_addr`, `lm_slot_pos`, `lm_context`,
  `message_loop_rounds`, `message_loop_exit_aux`, `message_loop_exit_targets`. Each maps to a
  `PAR_*` env var on the launcher.
- **Suite variants:**
  - `e31c_m1 = e31_page + {lm_addr none, lm_slot_pos reader, lm_reader_tokens 1, lm_read closed}`;
  - `e31c_loop = e31c_m1 + the E33a loop {rounds 4, exit_aux 0.3, exit_targets answer}`.

## 7. Tests (CPU, `tests/test_text_memory.py`)
1. **Defaults bit-identical:** an explicit `lm_read = "exclusive"` gives the same logits as today's
   config. Existing `test_latent_memory.py` and `test_reread_loop.py` stay green.
2. **Closed = exclusive on exam answers:** answer-position logits are equal on suite-recipe rows
   (DNA lookup, lookalike, chain; Glyph fact), for the E31 champion and the E33a loop config.
3. **Text causality:** with no QUERY, perturbing token t moves no logit before t (`future_leaks` over
   every position), with several documents in one row.
4. **The notebook is live in text:** removing it (override none) changes logits only at positions
   where a window has closed, and the writer gets a gradient from a text-only loss.
5. **Closed mask semantics:** slot j is visible to q iff same doc ∧ close ≤ q (dense vs flex
   predicate on CPU forward; the band block lists cover it, with `_covers`).
6. **Reader mode exactness:** the global read equals a reference with RoPE logits on raw keys and
   NoPE logits on slots in one softmax. Moving slot keys by any rotation makes no difference.
7. **Trainer factory:** builds the text E31c model from model_args, with a forward on random ids
   (no QUERY).
8. **Suite variants build:** `e31c_m1` and `e31c_loop` build with their params within ±5 % of
   `e31_li_m1`.
- **GPU kernel check (Odra, under 1 min):** a flex forward + backward with query–key 128 / value 64
  at 16k, comparing flex and sdpa on 2k.

## 8. Risks & tradeoffs
- **Reader slot keys change exam behaviour.**
  - Signal: the S0 no-harm ladder.
  - Fallback: `lm_slot_pos = "read"` (true distance), or keep `boundary` on exams and `reader`
    only in text. A test shows both modes are exact.
- **Flex kernel with query–key 128 on sm86.**
  - Signal: the GPU check.
  - Fallback: split the read in two flex calls with LSE merge. This needs `lse` gradients; they
    were not verifiable on CPU.
- **The notebook ignored on text (the E22 failure).**
  - Signal: K1 at 50 % of the budget.
  - Fix the objective share (more keyed-recall rows), not the mask.
- **Cost:** book tokens on exams and every token in text read notes.
  - Signal: K2 in the smoke.
  - Block lists keep the read about triangular.
- **Packed-row window straddle:** a short document's tail may never be written. That is harmless
  for causality and recall within that document (it is still read raw), but it is noted.

## Change log (filled while implementing)
| # | change | file | test |
|---|---|---|---|
| 1 | `lm_read` / `lm_slot_pos=reader` config + validation; `message_enabled` for closed | `nn/perceiver_ar_lm.py` | 1, 7 |
| 2 | closed ctx without QUERY; `slot_close` | `nn/perceiver_ar_lm.py` | 3, 4 |
| 3 | closed rule in dense + flex masks, extra blocks | `nn/perceiver_ar_lm.py` | 2, 5 |
| 4 | reader mode: un-rotated slot keys + concatenated-dim read | `nn/latent_memory.py`, `nn/perceiver_ar_lm.py` | 6 |
| 5 | trainer pass-through + launcher env | `training/*`, `scripts/train_concept_pretraining_multigpu.sh` | 7 |
| 6 | suite variants `e31c_m1`, `e31c_loop` | `evaluation/bapo_models.py`, `evaluation/capability_suite.py` | 8 |
| 7 | probes `--message_override`; `message_override` saved in the config (a no-notebook control stays without one when reloaded) | `evaluation/long_context_probes.py`, `nn/perceiver_ar_lm.py` | 7 |
| 8 | probe CLI `--lm_read`, `--lm_slot_pos reader` | `verification/bapo_capability_probe.py` | 8 |

**Text-trainer defaults.** The `lm_*` args default to the E31 champion + the E31c read (one entry
per latent, `lm_addr none`, `lm_slot_pos reader`, `lm_read closed`). They are passed only when
`PAR_MESSAGE_WRITE=latent_memory`, so every older launch is byte-identical.

**Verification so far.**
- Old vs new code on 7 arches: bit-identical.
- `tests/test_text_memory.py`: 24 tests pass.
- Related suites: 200 tests pass.
- Pre-existing collator/tokenizer failures are unchanged from the base branch.
- Pending: the GPU flex check, once an Odra GPU is free (do not run it beside E33a jobs).
