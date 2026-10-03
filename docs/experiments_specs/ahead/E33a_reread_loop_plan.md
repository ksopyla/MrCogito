# E33a — Implementation Plan

- **Spec:** [E33a_reread_loop.md](E33a_reread_loop.md) · **Status:** approved (author asked to plan, implement and run, 2026-10-03)
- **Authored by:** `implementation-plan` · for → `research-implement`

## 1. Source & fit
- **Origin:** the author's proposal (2 Oct): read the memory, try to answer, reread with the current state, 4 static
  loops, then formulate the answer. This plan grounds it in the E33 negative result (the read-only loop) and the
  [latent reasoning review](../../4_Research_Notes/latent_reasoning_review_20261001.html).
- **Synthesis verdict:** Adapt.
  - Taken from Huginn / Mixture-of-Recursions: the prelude → tied core → untied coda layout.
  - Taken from Ouro / TRM: the loss at every exit (deep supervision).
  - Taken from End-to-end Memory Networks: hops over one fixed memory.
  - Dropped for now: random loop count and halting (static R = 4, as the author asked).
- **Architecture mapping:** reasoning (question-side recurrence over the E31 memory) + loss (exit losses) + data
  (lookup replay). The writer and the memory are unchanged.
- **Boldness check:** the loop contains the answer-forming local layer, and every loop decodes through the untied
  answer layer. That is the claim. It is not E33 retuned.

## 2. Reuse map
| Component | Action | Where |
|---|---|---|
| `PerceiverARConfig` | extend: `message_loop_rounds` (1), `message_loop_span` (1), `message_loop_inject` ("none"), `message_loop_exit_aux` (0.0), `message_loop_exit_targets` ("progress") | `nn/perceiver_ar_lm.py` (next to `message_read_rounds`, l.173; validation l.343) |
| `PerceiverARLM._run_layers` | extend: when `message_loop_rounds > 1`, run prelude → core × R → coda; collect per-loop states for exits | `nn/perceiver_ar_lm.py` l.2251 |
| `PerceiverARLM.forward` | extend: exit losses through the coda + `final_norm` + `lm_head`; targets from `set_round_targets` (progress) or labels (answer) | `nn/perceiver_ar_lm.py` l.2325 |
| `PerceiverARLM.__init__` | new params: `loop_emb [R, H]` (zero init), `loop_gate` scalar (zero init, only for inject="prelude") | `nn/perceiver_ar_lm.py` |
| `Block` (E33 `read_rounds`) | reuse unchanged; mutually exclusive with the new loop (config error) | l.1765 |
| global attention + E31 writer | reuse as is: slots come from `ctx.slots`, set once by the writer (l.2272); repeated calls are side-effect free (l.1575) | `nn/perceiver_ar_lm.py`, `nn/latent_memory.py` |
| `ArchSpec` / `build_model` | extend: 5 fields threaded like `message_read_rounds` (l.125, l.260) | `evaluation/bapo_models.py` |
| probe | extend: `--loop_rounds/--loop_span/--loop_inject/--loop_exit_aux/--loop_exit_targets`, `--replay_recipe/--replay_frac`, `--freeze_writer`, per-exit final eval, round targets for the loop | `verification/bapo_capability_probe.py` |
| `round_target_labels` | extend: rows without `nodes` fall back to their answer labels (E33 rows all had nodes, so no change for E33) | probe l.143 |
| length ladder | extend: `--loop_rounds` (eval-time R) and `--recipe` (ladder a different exam than the one trained, for the lookup no-harm check) | `verification/length_ladder.py` |
| study plan | extend: phase `e33a` | `scripts/study_plans/e30_vs_e31.py` |

## 3. Forward pass (tensor shapes)
Symbols: `B` batch (32), `N` tokens (1024), `H` 960, `C` memory entries (N/6 ≈ 160 at 1k), `V` 13 (DNA vocab),
`R` loops (4). Layers: 0 = local (prelude), 1 = global read, 2 = local, 3 = local. U-net skips: layers 0 and 1
push; layer 2 pops layer 1's output, layer 3 pops layer 0's output (`n_skip = 2`, l.2279).
```
(B, N)            → embed                                   → x0 (B, N, H)
(B, N)            → E31 writer (once)                       → slots K, V (B, C, 1, 64) + slot_pos   # read-only from here on
x0                → layer 0 (prelude, local 16)             → p (B, N, H); skips = [p]
for r in 0..R-1:                                            # core, tied weights
    x = (p if r == 0 else x_r) + loop_emb[r] (+ loop_gate·p if inject="prelude" and r > 0)
    s = skips.copy()                                        # [p]
    x = layer 1(x, x0, None; message=slots) → push x to s   # global read: own side + memory (one softmax)
    x = layer 2(x, x0, skip=s.pop())                        # local "think"; skip = this loop's read output
    x_r, s_r = x, s                                         # s_r = [p]
    if training and exit_aux > 0 and r < R-1:
        e_r = final_norm(layer 3(x_r, x0, skip=p))          # exit through the untied answer layer
x_R               → layer 3 (answer layer, skip = p)        → final_norm → lm_head → logits (B, N, V)
```
- **R = 1 is exact.** `loop_emb[0]` is zero at init, so the computation is `layer0 → layer1 → layer2 → layer3`
  with the same skips as today. Test: logits equal to the R = 1 model on a loaded checkpoint (atol 0).
- **span = 2 (the wide-core arm):** core = layers 1..3 and coda = none (head only). Layer 3's skip is `p` in every
  loop, the same as today. The exits are `final_norm(x_r)`.
- **Book side:** loops too. That is simple and exact; question-side outputs never read book-side main-path states
  after QUERY. Cost at 1k: about 1.6× per step. Restricting the loop to the question side is a later optimisation.
- **Gradient checkpointing:** the existing per-layer `torch_checkpoint` call wraps each layer call inside the loop
  unchanged.

## 4. Inputs & data
- **Main exam:** `--recipe chain_parallel --hops {2,3,4} --key_len 8` at `--seq_len 1024`, built by `run_rung` →
  `config_for` (probe l.687). Rows carry `meta["nodes"]` (`data/symbolic_tasks.py`).
- **Replay:** `--replay_recipe recall_single --replay_frac 0.25`. A second `cfg_rep = config_for(scale,
  rep.task, **rep.overrides, seq_len=args.seq_len)` with the same vocab and the same QUERY control id. Each batch
  takes `round(B·frac)` rows from `cfg_rep`. The rows are stacked; every row keeps its own labels.
- **Round targets (progress):** `round_target_labels(rows, labels, R-1)`. Exit r gets node r+1 (the terminal once
  r+1 ≥ hops). Replay rows get their own answer labels.
- **Eval:** the main exam's 256 rows as today, plus a final eval of the replay exam (256 rows). At the end:
  per-exit first-letter accuracy, against the answer and against the progress targets. Both run by setting
  `model._loop_rounds_override = r` (the exit-r state equals the R = r forward exactly).

## 5. Loss & training objective
- `loss = CE(final) + exit_aux · Σ_{r=1}^{R-1} CE(exit r; targets_r)`, with `exit_aux = 0.3`. Each exit CE is
  token-mean over its valid targets, computed with the existing `chunked_softcap_ce` (z-loss 0 on exits).
- `targets_r`: progress → `self._round_targets[r]`. Answer → `labels` for every exit.
- `--freeze_writer`: `requires_grad_(False)` on `model.memory_writer` before the optimizer is built. The optimizer
  takes `[p for p in model.parameters() if p.requires_grad]`.

## 6. Config & launch
- **Config defaults** keep every existing model and checkpoint identical: `message_loop_rounds=1`,
  `message_loop_span=1`, `message_loop_inject="none"`, `message_loop_exit_aux=0.0`,
  `message_loop_exit_targets="progress"`.
- **Validation:** loop R > 1 needs `message_enabled`, `par_mode="perceiver"`, exactly 1 global layer, and
  `global index + span < n_layers` (span 1 at minimum; span = n − 1 − gi is allowed with the head as the coda).
  Not combinable with `message_read_rounds > 1`.
- **Checkpoint loading:** the probe's strict `--init_ckpt` allows missing `round_emb` today. Extend it to
  `loop_emb` and `loop_gate`.
- **Study phase `e33a`** (Odra, `scripts/run_study_queue.py`). Every job uses `L1K_FT` + `--key_len 8` +
  `--replay_recipe recall_single --replay_frac 0.25` + its arm flags, in a chain pchain2 → pchain3 → pchain4, each
  stage starting from the previous stage. Ladder `LADDER_1K` on pchain3/pchain4. A lookup ladder on the final
  checkpoint covers no-harm.
  ```
  ★ loop from step 0 : lookup-2k with --loop_rounds 4 --loop_exit_aux 0.3 from random init, then the chain (seeds 1, 2)
  loop fine-tuned    : the same flags from len_lookup_e31_li_m1_s1 (past single-read checkpoint)
  control            : --loop_rounds 1 from len_lookup_e31_li_m1_s1
  answer exits       : ★ seed-1 lookup weights, --loop_exit_targets answer
  writer frozen      : ★ seed-1 lookup weights + --freeze_writer
  re-injection       : ★ seed-1 lookup weights + --loop_inject prelude
  wide core          : its own loop lookup-2k with --loop_span 2, then the chain
  ```
  Ladders: pchain at 1k / 4k / 16k (128 rows); lookup stages at 2k / 8k / 32k; each final pchain4 checkpoint also
  gets a lookup no-harm ladder (`--recipe recall_single`, 1k–32k) via the runner's new `ladders` job field.
  Launch on Odra: `uv run python scripts/run_study_queue.py --plan e30_vs_e31 --phase e33a --out Cache/study/e30_vs_e31
  --host odra --gpus 0 1 2 --mode scripts`, then start the generated queue scripts in byobu session `e33a`.

## 7. Tests & smoke
New `tests/test_reread_loop.py`, CPU, tiny config (H 64, 4 layers, DNA 128 tokens, E31 writer):
1. R = 1 with the loop code path equals the default model bit for bit; a loaded state dict gives the same logits.
2. R = 4: logits finite, loss finite, backward reaches `loop_emb`, layers 1 and 2, and the writer (none with
   `freeze_writer`).
3. Exit losses: with `exit_aux > 0` and round targets, the loss is greater than the final CE alone and equals final
   + aux·Σ exits (recomputed by hand from `_loop_rounds_override` forwards).
4. Override: `_loop_rounds_override = r` gives the same logits as the exit-r path.
5. Causality: changing tokens after a position does not change its logits (R = 4), and changing book tokens changes
   the answer only through the slots (exclusive read kept).
6. Span 2 builds and runs; invalid configs raise (loop + `message_read_rounds`, span too large).
7. The probe's `round_target_labels` falls back to labels for rows without nodes; the replay batch mixes two
   configs.

Smoke (local CPU, under a minute): the probe at `--seq_len 256`, 30 steps, `--loop_rounds 4 --loop_exit_aux 0.3
--replay_recipe recall_single`, hidden 128. Then 200 steps on Odra GPU 0 from the real init checkpoint, before the
queue starts.

## 8. Risks & tradeoffs
- **Same wall as E33** if the memory keys cannot be queried "as a source". Cheapest signal: pchain2 at 1200 steps.
  Per-exit accuracy shows whether exit 1 gets B and exit 2 does not. Fallback: latent thoughts (sequence recurrence,
  its own alternative page) or 2 memory key heads.
- **Retrieval drift** from training the writer on chains: replay plus the writer-frozen arm cover it.
- **Cost:** about 1.6× per step at 1k because the book side loops too. Acceptable for 18 jobs. Optimise only if this
  goes to 128k training.
- **The R = 1 → R = 4 jump** from a single-read checkpoint: `loop_emb` starts at zero, so loop 1 is the old model and
  loops 2–4 start as repeats. If pchain2 does not move in the first 1200 steps, a curriculum on R (2 → 4) is the
  first fix. That is a run setting, not a new spec.

## 9. Code sketches
```python
# sketch: PerceiverARLM._run_layers, loop branch (span=1, coda=[layer 3])
gi = cfg.global_layer_index; span = cfg.message_loop_span
R = self._loop_rounds_override or cfg.message_loop_rounds
pre, core, coda = range(0, gi), range(gi, gi + 1 + span), range(gi + 1 + span, n)
x, skips = run(pre, x, skips)                 # same skip/push rule as today
p, skips0 = x, list(skips)
states = []
for r in range(R):
    x = x + self.loop_emb[min(r, len(self.loop_emb) - 1)]
    if r and inject: x = x + self.loop_gate * p
    s = list(skips0); x, s = run(core, x, s)
    states.append((x, s))
self._loop_states = states[:-1] if collect else None
x, _ = run(coda, x, states[-1][1])
return x
```
