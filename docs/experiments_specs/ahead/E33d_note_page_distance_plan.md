# E33d — Implementation Plan

Spec: [E33d_note_page_distance.md](E33d_note_page_distance.md). The bet, unwatered: a learned,
signed, per-head page-distance term added to every note's score in the global read, outside the
key, starting at zero.

## 1. Source & fit
- Origin: E33 run 1 on the 8-questions recipe (latest 95 % at 1k → 22 % at 4k → about 5 % beyond)
  and the code reading of 14 Oct: text notes carry no position (`lm_slot_pos = reader`, `lm_addr = none`).
- Literature: ALiBi (Press et al. 2022, https://arxiv.org/abs/2108.12409), a linear per-head
  distance penalty that holds beyond the training length; here on a log scale and only on the note
  columns. DNA evidence (E31b): order inside the key costs long-range lookup.
- Touches the **read** only (decode side of the notebook). Writer, data and loss unchanged.

## 2. Reuse map
| piece | status |
|---|---|
| `LatentMemoryWriter` (`nn/latent_memory.py`) | as is; its `slot_pos` (expected read position per note) is already in `MessageCtx.slot_pos` |
| `MessageCtx`, `dense_message_mask`, flex block masks | as is |
| `attend_message` (`nn/perceiver_ar_lm.py`) | extended: optional `note_bias=(slope, pos, slot_pos, unit, mode)` → flex `score_mod` / SDPA float mask on the note columns |
| `Attention` (global layer) | extended: `note_age_slope` [h], zeros, only when `lm_note_age != "none"` on the global read of a latent-memory model |
| `PerceiverARConfig` | two fields: `lm_note_age="none"`, `lm_note_age_unit=0` |
| trainer args / factory / launcher | pass-through `lm_note_age`, `lm_note_age_unit` (`PAR_LM_NOTE_AGE`, `PAR_LM_NOTE_AGE_UNIT`) |
| `evaluation/text_checks.py` | arches `e33d`, `e33d_plain`; `ENV_OF` entries; recipe cards borrow E33's |

## 3. Forward pass (30M text, B rows, S = 4096, C = 672 notes, h = 9, dh = 64)
1. Writer (unchanged): notes k̄, v̄ [B, C, 1, 64]; `slot_pos` [B, C] = expected read position.
2. Global read: Q = [RoPE(q), q] [B, S, 9, 128]; K = [raw ‖ notes]; one softmax.
3. **New:** for note columns, `bias[b, h, t, c] = −s_h · f(max(0, pos[b, t] − slot_pos[b, c]) / unit)`,
   f = log1p (or identity for "linear"); raw columns get 0. Flex: `score_mod` with captured
   `pos`, `slot_pos`, `slope` (gradient flows to `slope`). SDPA: a float mask [B, h, S, S + C].
4. Everything after the read is unchanged.

## 4. Inputs & data
None changed. Text World v2, 8 questions per training document.

## 5. Loss & training objective
Unchanged (next-token CE + 0.3 × each of the 3 loop exits during training). The slopes learn from
the same loss.

## 6. Config & launch
`PAR_LM_NOTE_AGE=log` (and optionally `PAR_LM_NOTE_AGE_UNIT=192`) on top of E33's settings; the
text-checks arch `e33d_plain` sets it. Old checkpoints load with the defaults "none" / 0.

## 7. Tests & smoke
`tests/test_e33d_note_age.py`:
- default config builds no slope and gives the same output as before;
- with `lm_note_age=log` and zero slopes, the output equals the plain E33 model's (same weights);
- with a positive slope, two notes with identical content but different positions: the newer one
  gets more read weight (SDPA path), and flex equals SDPA with the bias (CPU, eager flex);
- the slope receives a gradient through the full model.
Plus the existing text-checks runner tests (parameter band, launcher settings) with the new arches.
Laptop smoke: a few training steps of the smoke tier on MPS.

## 8. Risks & tradeoffs
- **Length transfer:** a head with a large positive slope penalises 85-page-old notes more than
  anything seen in training. Guard: log scale; heads can stay at 0. Kill: far lookup at 16k −5 points.
- **Slopes never move:** if the gradient on s is weak, E33d = E33. Diagnostic: log the slopes.
- **Compile:** a new `score_mod` closure per call; captured tensors are graph inputs, so lengths
  (not calls) trigger recompiles.

## 9. Code sketch
```python
# sketch
def note_age_bias(pos, slot_pos, slope, unit, mode):   # [B,S], [B,C], [h] -> [B,h,S,C]
    age = (pos[:, :, None] - slot_pos[:, None, :]).clamp(min=0).float() / unit
    f = torch.log1p(age) if mode == "log" else age
    return -slope.float()[None, :, None, None] * f[:, None]
```
