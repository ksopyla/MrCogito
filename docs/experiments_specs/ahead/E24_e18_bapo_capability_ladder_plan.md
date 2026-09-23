# E24 — Implementation Plan

- **Spec:** [E24_e18_bapo_capability_ladder.md](E24_e18_bapo_capability_ladder.md) · **Status:** implemented (tiny measured; medium launching)
- **Authored by:** `implementation-plan` · for → `research-implement`

> The HOW for a **measurement** bet: E18's capability surface on a calibrated BAPO DNA
> ladder vs matched dense decoder-only and a symmetric encoder-decoder. Does not replace E18
> with a safer architecture.

## 1. Source & fit
- **Origin:** DNA symbolic suite 2026-09-13; E18/E18b verdict (positional copy works, content
  addressing does not); BAPO (Schnabel et al. 2025, `docs/literature_review/reasoning_bandwidth_information_flow.md`);
  sibling Arm-A 100% map (do not duplicate).
- **Synthesis verdict:** Adapt BAPO's easy/hard split as *our* eval, where we can actually
  measure `a` (the paper cannot).
- **Architecture mapping:** data + eval + a reusable encoder-decoder baseline. E18 weights
  untouched. Objective remains teacher-forced CE on the supervised span.
- **Boldness check:** the claim is that E18 is a constant-`a` BAPO that fails exactly the
  hard tasks. Measuring that is the bet; shrinking it to "tune LR on far_copy" would be the
  retread.

## 2. Reuse map (read the modules first)
| Component | Action | Where |
|---|---|---|
| `PerceiverARLM` / `PerceiverARConfig` | reuse as-is (`perceiver` and `dense`) | `nn/perceiver_ar_lm.py` |
| `generate_row` / `floor_nats` | extend with BAPO tasks | `data/symbolic_tasks.py` |
| `build_symbolic_dataset.py` | pass new fields | `scripts/build_symbolic_dataset.py` |
| `symbolic_channel_probe.py` | leave alone (sibling agent's exam) | `verification/` |
| `EncoderDecoderLM` | new reusable baseline | `nn/encdec_lm.py` |
| ladder + metrics + factory + probe + plots | new | `data/bapo_ladder.py`, `evaluation/bapo_metrics.py`, `evaluation/bapo_models.py`, `verification/bapo_capability_probe.py`, `analysis/plot_bapo_capability.py` |

## 3. Forward pass (tensor shapes)
Symbols: `B`=batch, `N`=`seq_len`, `H`=hidden, `V`=`n_symbols+9` controls.
```
dense / e18:
  (B, N) → hashed embed → (B, N, H)
  → L layers (full or SWA+one full) → (B, N, H)
  → lm_head CE on labels != -100  (the answer span)

encdec:
  prefix = tokens[0 : answer_start-1]     # bidirectional encoder  (B, P, H)
  suffix = tokens[answer_start-1 :]       # causal decoder          (B, Q, H)
  decoder self-attn cannot see prefix; cross-attn to encoder memory
  CE on the same answer span
```

## 4. Inputs & data
- **Dataset:** on-the-fly `generate_row` (tiny). Medium/large: `scripts/build_symbolic_dataset.py --task … --seq_len 4096|16384|32768|131072`.
- **Collator:** probe stacks `input_ids` / `labels` directly; the shared collator is unused at tiny.
- **Split:** fresh rows every step (nothing memorisable); held-out eval seed offset +99.

## 5. Loss & training objective
- Teacher-forced CE on the answer span only (`labels == -100` elsewhere), same as
  `build_copy_task_dataset.py`. Constant LR after 50-step warmup (the OneCycle-horizon
  artefact from the Arm-A 100% map). Early-stop at 99% acc.

## 6. Config & launch
- **New config fields:** none on `PerceiverARConfig`. `EncDecConfig` is probe-only.
- **Registry:** encdec is **not** in MODEL_REGISTRY (not a training family).
- **Launch:** `uv run python verification/bapo_capability_probe.py --scale tiny --arch dense e18 encdec e18_local --out Cache/bapo_tiny`
- **Max params:** `--max_params 100000000`.

## 7. Tests & smoke
- `tests/test_symbolic_tasks.py` — solvable / not-local / floor for every new task.
- `tests/test_bapo_ladder.py` — every (scale, task) constructs; info_report; encdec shapes;
  all four arches forward+backward under 100M.
- Tiny probe smoke: a few dozen steps on `far_copy` (instrument check, not the 75% gate).

## 8. Risks & tradeoffs
- **Risk:** tiny budget too small → dense misses 75% → false "unsolvable". **Signal:** packed
  answers (`value_len`/`key_len`/`span_len` toward 16–32 tokens) and 4× steps before killing
  a rung (K1). Dense is trained first; other arches are not scored on an uncalibrated rung.
- **Risk:** E18 looks good on tiny far_copy (positional) and we over-generalise. **Signal:**
  S2/S3 on recall/select/chain are the load-bearing gates.
- **Fallback:** none architectural. If E18 matches dense on content tasks (K3), the E18b
  diagnosis is scale-specific and we stop scaling.

## 9. Code sketches
```python
# sketch: every rung cites these numbers
card = rung_card("tiny", "far_copy")
# card["prize_bits"], card["floor_nats"], card["solvable_acc"] == 0.75

# sketch: architecture factory, same hidden/depth, different pattern
model = build_model("e18"|"dense"|"e18_local"|"encdec", vocab_size=V, ...)
```
