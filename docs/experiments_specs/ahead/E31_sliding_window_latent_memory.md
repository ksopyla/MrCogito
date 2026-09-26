# E31 — Sliding-window latent memory (real latents with addresses, written from two-way context)

- **Status:** draft — awaiting go-ahead (no runs)
- **Serves:** Vision priorities 1–2 — a length-scaling concept memory that **picks the
  actual information (facts) out of natural text and separates it from noise**. The claim
  is about text. DNA and Glyph are controlled gates on the way (exact bits, known noise),
  not the target: passing DNA alone is not success. This is the design the E30 idea note asked for;
  E30 built a narrower write (see [E30 · Built vs intended](E30_sliding_window_perceiver.md#built-vs-intended-review-2026-09-24)).
- **Implementation plan:** `E31_sliding_window_latent_memory_plan.md` *(to be written by `implementation-plan` after approval)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-24 · closed —
- **Revision:** replaces the same-day draft *"E31 — 10M capacity-allocation grid (H vs C
  vs depth)"* (commit `53f2f9c`, never run). That grid spent parameters on knobs of the
  E30 write, whose 1024 plateau looks salience-bound, not count-bound, so more slots
  (K=64) would not test the question. The H-vs-C grid is kept as a follow-up run of
  **this** ID on the winning write (see *Follow-ups*).

> One coherent bet: turn E30's per-window *weighted average of token K/V* into a
> **per-window latent array** — each latent a real vector with its own state, width and
> address — written from **two-way context** over the window. Two ways of supplying
> that context are the two arms of the same bet. The reader, the exclusive mask and
> the DNA exams stay as in E30.

## Hypothesis
If each sliding window (W = 256, stride 192) is written by **32 addressed latent
vectors** (width 4× the token embedding, query + cross-attention + FFN, per-head
attention, a learned position prior per latent) from tokens that already carry
**two-way context over the whole window** — either a 2-layer bidirectional page encoder
or latent↔token iteration (BiXT / Slot-Attention style) with no encoder — then on the
1024-token exams the exclusive notebook recovers **≥ 40 of 64 bits on the single lookup
and ≥ 0.75 × the full read on the lookalike and the in-order chain** (3 seeds), where
E30 plateaus at ~26 bits. **Because** the plateau is set by what a static query can
recognise in a 16-token causal state (13 value letters × 2 bits), and a latent that sees
the whole fact in context can bind and keep all of it (Tishby: keep I(Z;Y) — the
sufficient statistic of a fact is the fact *in context*, not a token filtered by a
blind query).

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (`PerceiverARLM`, exclusive `MessageCtx` /
  `attend_message` concat path, `swp_geometry`), `evaluation/bapo_models.py`,
  `verification/bapo_capability_probe.py`, `data/symbolic_tasks.py` +
  `data/bapo_ladder.py`. New write is a config-selectable compressor
  (`message_write="latent_memory"`), not a fork.
- **Init / checkpoint:** random init. Latent queries warm (`init_std`), never
  zero-initialised (cold-start law).
- **Baseline to beat (31M, H = 960, 4 layers, step 1e-4, warm residuals, bridge_1k):**

  | exam (64-bit prize, fact = `keymark · 2 key · 32 value`) | full read | E21 average | **E30 (built)** |
  |---|---|---|---|
  | single lookup (`recall_single`) | 0 | 0 | **25.7 bits** (54 %) |
  | lookalike (`select_1decoy`) | 62.6 | 18.7 | **25.9 bits** (56 %) |
  | in-order 4-hop chain (`chain_ordered`) | 63.3 | 4.0 | **40.1 bits** (77 %) |
  | single lookup @ 2048 | 0 | 0 | 8.3 (pg 256) / 25 (pg 128, extended) |

  Source: [limits](../../2_Experiments_Registry/run_reports/e30_length_hardness_limits_20260922.md) ·
  [coverage](../../2_Experiments_Registry/run_reports/e30_coverage_and_breadth_20260922.md).
- **Materially new:** (1) slots are **latent vectors with their own state** (residual +
  FFN, width 4×e), not convex mixes of token K/V; (2) each latent has an **address**
  (latent ID + window start + a learned position prior over its part of the window, and a
  RoPE position at the *expected* position of what it read); (3) writing from **two-way
  context** — never done on the exclusive platform (E18–E30 all wrote from a 16-token
  causal state); (4) per-head attention pooling (no head averaging). Vs the old
  BiXT/perceiver_denoise line (Mar 2026): that was a fixed C = 128 over the whole
  sequence with no windows, no addresses, no exclusive read — the bandwidth story the
  idea note rejected.

## The architectural bet
```
ids ─► tiny token embedding e = 128 (no hashed n-grams)
   ├──► main causal path (unchanged E18 platform: up-proj → SWA pre → global read → SWA stack → CE)
   └──► WRITER, per window w_i = sender tokens [192·i, 192·i + 256)   (fixed; no auto-shrink)
         context (the two arms):
           A · page encoder : 2 bidirectional full-attention layers inside the window (width d_w = 256)
           B · no encoder   : 3 rounds BiXT — z ← z + CA(z→x);  x ← x + CA(x→z)
                              with Slot-Attention competition (softmax over latents) in z←x
         latents:  z_k⁰ = q_k + addr(i, k)                       k = 1..32, D_lat = 4e = 512
                   z_k ← z_k + MHCA(z_k → window, bias b_k(t)) ; z_k ← z_k + FFN(z_k)   (×2 rounds in A)
                   b_k(t) learned per latent, init = soft sub-page prior around t = (k+½)·W/K
         address:  RoPE position of slot = window start + Σ_t w_k(t)·t   (where it actually read)
         to reader: k̄ = W_K z, v̄ = W_V z  in the global read's K/V space (C = 32 · n_windows)
   exclusive read after QUERY (E21/E30 mask): receivers see slots of earlier sides, never raw prefix
```
Geometry at the claim lengths (W = 256, stride 192, K = 32, coverage 8, compression ≈ 6×):

| seq | windows | slots C | N / C |
|---|---|---|---|
| 1024 | 5 (0, 192, 384, 576, 768) | 160 | 6.4 |
| 2048 | 11 (last flush-right at 1792) | 352 | 5.8 |
| 4096 | 21 | 672 | 6.1 |

**Pooling rule (fixes E30 leak):** a window pools only tokens on its *own* earliest
side, computed per window. Two-way attention runs among sender tokens only; the
receiver never sees a latent built from a later side. Causality test required (below).

**Memory visibility (text vs probe).** In the probes, the question side reads the memory
only after QUERY (the exclusive cut), and the book side of the main path never reaches the
answer. That is a property of the *exam*, not of the model. In text (rung 3 below), the main
causal path predicts **every** token, and token `t` may read the latents of every window
that has **closed** before `t` (window end ≤ t), plus its own local context. So the memory
is read from any position, not only after one QUERY. The writer must therefore also work
without a question boundary; the pooling rule becomes "closed windows only".

**In scope:** the writer (both context arms), latent addresses, the per-window pooling
rule, per-position answer accuracy, the DNA protocol below.
**Out of scope (follow-ups after a positive signal):** question-driven page fetch /
indexer on the read side, multi-hop read loops, streaming with a carried state,
delta-rule matrix latents, latent↔latent consolidation, the H-vs-C grid, TinyHashed
on/off, hierarchy of chapter/volume latents.

## Why this is not a safe retread
E30 = static filter over causal 16-token states; E22 = one learned query + mean + latent
transformer; E26/E27 = AE / anchors on averages; Mar-2026 BiXT = fixed 128 latents over
the whole text. E31 is the first write where a *windowed, addressed latent* reads a fact
in two-way context and is the only channel. Analogy: a hippocampal episode — the event
is encoded as a whole (context first), stored at an address (place/time), recalled by
content. If it fails, "real latents do not beat a static filter at 6× on the exclusive
read" is recorded and the next bet moves to the read side.

## Pre-flight (before the arms; hours, not days)
1. **Leak fix + causality test** in the E30 write: perturb any receiver token → logits at
   earlier positions unchanged (sdpa, several geometries incl. receiver-only windows).
2. **Per-position answer accuracy** in the probe JSON (accuracy per answer offset).
3. **Salience check on the E30 build** (lookup @ 1024, 31M, 1 seed): E30 as built vs E30
   with a 64-token causal reach on the pre-encoder only (`--pre_window 64`†; not
   `--local_window`, which also widens the answer-side stack and must stay below `min_gap = 64`). Prediction:
   E30 positions 1–13 ≥ 75 %, 14–32 near chance (25 %); the 64-reach control lifts
   positions 14–32. If E30 does **not** show the 13-letter split, the salience story is
   wrong: record it and keep E31 (the design is still the one the note asked for), but
   drop the "≥ 40 bits because of context" rationale from the claim.
4. **Key length of the 1024 runs — settled (2026-09-24).** The saved probe JSON on Odra
   (`pack` block) shows `recall_single` / `select_1decoy` at `bridge_1k` used a 2-letter
   key and a 32-letter value, 1 decoy; the chain used `--hops 4` with 32-letter keys. The
   26-bit salience prediction applies as stated. Eval sets were 32 rows (16 at 4096).
5. **Dense pre-probe:** now warns by default (`--dense_preprobe_action warn`); do not pass
   `skip` on the lookup rung, where the full read is 0 and E30 learns.
6. **Exam hygiene:** `lookup_1key`, `lookalike`, `chain_4hop` now reproduce the recorded
   64-bit exams (fixed 2026-09-24; checked by `EXPECTED_PRIZE_BITS`). Use them or the
   calibrated names; both are the same exam.

## Exam ladder (DNA → structured noise → text)
| rung | what it is | why | gate to move on |
|---|---|---|---|
| **1 · DNA** (`bridge_1k`: `recall_single`, `select_1decoy`, `chain_ordered`; lookup also at 2048) | one planted fact in random letters; exact floor and prize | measures how many bits the memory carries, with no language prior to hide behind | S1–S3 below |
| **2 · Glyph** (`fact_markov_single`, `story_fact`; 512 then 1024) | keyed facts inside **language-like** Markov filler; typed vocab | noise now looks plausible, which is the text situation; tests "separate signal from noise" | best latent arm ≥ 0.75 × full read, and ≥ E30 + 8 bits, where the full read passes |
| **3 · Text** (a) planted facts in natural prose, BABILong-style (PG19 / TinyStories paragraphs + inserted facts + a question); (b) a next-token LM smoke at 512 tokens with memory visibility by closed windows | the actual goal: facts in real language, and memory as part of ordinary next-token prediction | (a) the latent arm beats E30 on answer accuracy; (b) memory lowers loss on tokens whose evidence is outside the local window, vs the same model with memory cut |

Rung 2 needs no new data code (Glyph exists). Rung 3a needs a small generator (≈ one
file under `data/`); 3b reuses the shared LM entrypoint at 512 tokens, the longest length
where copying already passes for every arm (see the E30 training inventory). Rung 3 is a
smoke to check that the mechanism is not a trick of the tiny alphabet, not a pretrain.

## Coarse-geometry rung (the 1M setting, measured small)
The claim geometry (W 256, stride 192, 32 latents ≈ 6 tokens per latent, C ≈ N/6) is a
short-context test setting. At 1M tokens the intended memory is coarser: **512-token
windows with 8 latents** (stride 384 → C ≈ N/48; no overlap → N/64). That is ~24× cheaper
to read than full attention at 1M (≈ 23 G vs 550 G dot products per layer; ≈ 22K slots
per generated token), but 48–64 tokens per latent is the regime where trained compressors
keep gist, not verbatim content (DeepSeek-OCR ≈ 97 % at 10×, ≈ 60 % at 20×). E30's
512-token windows wrote nothing, but with the flawed writer, so this is open.

Run the **leading latent arm** from wave 1 at four geometries, same exams, 1 seed:

| geometry | stride | latents / window | slots at 1024 / 2048 / 4096 | tokens per latent |
|---|---|---|---|---|
| claim (fine) | 192 | 32 | 160 / 352 / 672 | ≈ 6 |
| half | 192 | 16 | 80 / 176 / 336 | ≈ 12 |
| **coarse, overlapped** (1M setting) | 384 (W = 512) | 8 | 24 / 40 / 88 | ≈ 48 |
| coarse, no overlap | 512 (W = 512) | 8 | 16 / 32 / 64 | 64 |

Exams: DNA lookup and lookalike at 1024 / 2048 / 4096, Glyph `story_fact` at 1024, with
per-position accuracy. Output: **recovered bits vs tokens per latent**.
**Decision rule** (feeds the 1M design, not E31's success/kill): if the coarse overlapped
geometry keeps ≥ 0.75 × the fine geometry's bits, a single coarse level may carry exact
facts at 1M. If it keeps < 0.5 × (typically: finds the fact but loses value letters), the
1M design needs the **two-level memory** in *Follow-ups*.

## Length ladder (long-context confirmation: 2k → 128k)
The final check that the architecture suits long inputs. The exam stays the same (task,
answer packing, 64-bit prize); only the haystack grows. Added 2026-09-25, after the
full-tier suite.

**Linear cost first.** Three things made the platform quadratic at long lengths. All
three are fixed on `e31-latent-memory`:
1. In the global layer, tokens before QUERY attended to every earlier token. New
   `message_raw_window` (default 0 = unchanged) makes those raw keys a causal window, so
   the memory is the only long path. The answer path never used them: receivers read only
   the slots.
2. `create_block_mask` materialised the full query × key grid for the memory read (29 GiB
   at 128k).
3. The sliding-window layers rebuilt their masks by checking every (query, key) pair.

Both mask problems are replaced by block lists computed from the band geometry
(`_band_block_mask`), used only above 2³⁰ pairs, so 2k training is unchanged. The new
path matches the old one at 36k to within bf16 noise.

Measured forward cost, 30M e31_page, one row, RTX 3090:

| length | fine memory (W 256, K 32, m 5) | coarse memory (W 512, K 8, m 1) | peak GPU memory |
|---|---|---|---|
| 16k | 0.30 s | 0.22 s | 0.9 GB |
| 32k | 0.44 s | 0.38 s | 1.0 GB |
| 64k | 0.86 s | 0.63 s | 1.8 GB |
| 128k | 1.35 s | 1.25 s | 3.4 GB |

Memory size: the fine setting has ≈ 0.83 reader entries per token (32 latents × 5 entries
per 192-token stride), so it is ≈ 109k slots at 128k. It hides raw tokens from the
receiver, but it does not shrink the memory. The coarse setting (N/48 ≈ 2.7k slots at
128k) is the real compression claim.

**Stage A: does it generalise to longer inputs?** Train at 2,048 tokens with
`--message_raw_window 256 --save_ckpt DIR`. Then evaluate the saved weights, with no
more training, at 2k / 4k / 8k / 16k / 32k / 64k / 128k (`verification/length_ladder.py`,
64 rows per length). Each length reports:
- accuracy ± row SE;
- accuracy by **fact depth** (5 bins, fact position / length);
- memory slots, peak GPU memory and seconds per row.

Arms:
- e31_page, fine: lookup-2k at 2 seeds, chain-2k at 1 seed;
- e31_page, coarse: lookup-2k at 2 seeds;
- dense, the full-attention reference;
- e18_local, the no-memory floor (it must sit at chance).

**Stage B: can it learn at long lengths?** A curriculum from the Stage A weights
(`--init_ckpt`): a short run at 8k, then 16k, then the ladder again. This separates
"cannot extrapolate" from "cannot hold a fact over a long input". Training at 128k does
not fit a 3090: the writer's activations are ≈ 50 GB per row. 64k–128k stay
evaluation-only.

**How to read a failure.** Each diagnosis comes with a prepared variant:
- Accuracy falls with the **distance** between fact and question → a position problem.
  RoPE on the memory read sees distances never seen in training; with θ = 500k and head
  dim 64, about 12 of 32 frequency pairs are out of range beyond 2k. Variant: a
  content-only (no-RoPE) memory read.
- Accuracy falls with **length at every depth** → attention spread over 100k+ slots
  instead of the 1.7k seen in training. Variant: `--global_logit_scale log` (SSMax).
- Coarse falls far below fine → the two-level memory in *Follow-ups*.

**Pass line (set before running):**
- e31_page, median of seeds, ≥ 75 % at 16k and 32k after Stage B;
- time and memory grow linearly with length (the table above);
- e18_local at chance;
- 64k and 128k reported as stretch results, not gating;
- dense is reported, not gating: it is the quadratic reference.

Commands:
```bash
# Stage A (per seed; the suite's lookup-2k budget at step size 5e-5)
uv run python verification/bapo_capability_probe.py --scale bridge_1k --recipe recall_single \
  --seq_len 2048 --arch dense e31_page e18_local --no-skip_uncalibrated --hidden 960 --head_dim 64 \
  --kv_heads 1 --pre_layers 1 --global_layers 1 --stack_layers 2 --max_params 40000000 --lr 5e-5 \
  --warm_residuals --steps 1200 --k1_mult 6 --batch 32 --grad_accum 2 --eval_every 100 --eval_rows 256 \
  --seed 1 --amp auto --swp_n_heads 8 --swp_query_dim 128 --token_embedding_dim 128 --ngram_orders none \
  --message_raw_window 256 --save_ckpt Cache/length_ladder/lookup2k_s1 --out Cache/length_ladder/lookup2k_s1
uv run python verification/length_ladder.py --ckpt Cache/length_ladder/lookup2k_s1 \
  --lengths 2048 4096 8192 16384 32768 65536 131072 --rows 64
```

## Arms (one probe per GPU)
| arm | what it is | role |
|---|---|---|
| full model (dense) | every layer full attention | exam is solvable here (S0) |
| full read (E18) | uncompressed prefix on the global read | ceiling on the same platform |
| E30 as built | static-query average, 16-token causal state | the baseline to beat |
| context-only control | E30 + 64-token causal pre-encoder reach (`--pre_window 64`†) | separates "more context" from "real latents" |
| **latent memory · page encoder** (A) | 2-layer bidirectional page encoder → 32 addressed latents | the bet |
| **latent memory · latent↔token iteration** (B) | no encoder; 3 BiXT rounds with slot competition | the bet |

Parameter note: A and B add a writer (~3–4 M at d_w = 256, D_lat = 512) to a ~31 M
model. Report params per arm; if the best arm wins by < 8 bits, rerun the context-only
control with matched extra depth before crediting the latents.

## Success criteria (set BEFORE running; median of 3 seeds)
Ceiling evidence: at 1024 the full read recovers **62.6** (lookalike) and **63.3**
(chain) of 64 bits on this fact format, width and budget — copying a 32-letter value
through one read is reachable here. The lookup itself has no full-read ceiling at 1024
(0 bits: trainability), so its gate is set against E30's measured plateau
(25.7 ± 1 bits over five runs), not against the prize.
- **S1 (the claim):** at least one latent-memory arm reaches **lookup ≥ 40 bits** and
  **lookalike ≥ 47 bits** (0.75 × 62.6) at 1024.
- **S2 (chain):** in-order chain ≥ **47 bits** (0.75 × 63.3; E30 40.1).
- **S3 (length):** lookup @ 2048 ≥ **32 bits** (E30 8–25).
- **S4 (latents, not just context):** best latent arm ≥ context-only control + 8 bits
  on lookalike @ 1024.
- **S6 (not a trick of the alphabet):** the rung-2 (Glyph) and rung-3 (text) gates in the
  exam ladder. S1–S3 without S6 is recorded as "works on DNA only".
- **S5 (write health):** `message_override=none` → chance; latent RankMe ≥ 8 per
  window (of 32); per-position accuracy flat across answer offsets 1–32 (no 13-letter
  step).

## Kill criteria (set BEFORE running)
- **K0:** causality test fails → stop, fix, rerun.
- **K1:** full read < 75 % on lookalike @ 1024 on this replica → exam not calibrated;
  do not score.
- **K2:** both latent arms < **32 bits** on lookup and < E30 + 6 bits on lookalike @ 1024
  after a doubled budget (9600 steps), 3 seeds → record "addressed latents from two-way
  context do not break the plateau"; next bet moves to the read side.
- **K3 (arm B only):** RankMe collapse (< 2) or divergence in ≥ 2/3 seeds at both 1e-4
  and 5e-5 → drop arm B, keep A.
- **Not a kill:** S4 miss (the context-only control matches). Record "context was the
  whole story", keep the cheaper write.

## Plan
- **Data:** rung 1 on-the-fly DNA rows (A = 4), `bridge_1k` 1024 tokens, fact spread
  (`--evidence_align spread`); lookup also at `--seq_len 2048`. Rung 2 Glyph
  (`--recipe fact_markov_single story_fact`, `--noise markov`). Rung 3 as in the ladder.
  `--eval_rows 512`.
- **Model:** 31M class (H = 960, 4 layers, head_dim 64), token embedding 128, hashed
  n-grams off, `--swp_auto_fit false --swp_window 256 --swp_stride 192 --swp_bank_size 32`.
- **Optimiser:** step 1e-4 with warm residuals (1024 law); 5e-5 retry for any arm that
  stalls; `--k1_mult 4`; do not kill a falling CE.
- **Budget:** 4800 steps, extend to 9600 when CE is still falling.
- **Compute:** Odra 3×3090 + Polonez 4×3090, one probe per GPU. Wave 1: 6 arms × 3 exams
  × 1 seed (≈ 18 probes, one night). Wave 2: 3 seeds for the full read, E30, the control
  and the leading latent arm (≈ 1–2 days). Wave 3: coarse-geometry rung, leading arm × 3
  more geometries × (2 DNA exams × 3 lengths + 1 Glyph) ≈ 21 probes (≈ 1 night; coarse
  geometries are cheaper to read).
- **Launch (flags marked † are added by research-implement):**
  ```bash
  uv run python verification/bapo_capability_probe.py \
    --scale bridge_1k --recipe recall_single select_1decoy chain_ordered \
    --arch dense e18 e30 e30_ctx64† e31_page† e31_bixt† \
    --hidden 960 --head_dim 64 --stack_layers 2 --max_params 40000000 \
    --token_embedding_dim 128† --no_ngram† \
    --swp_auto_fit false --swp_window 256 --swp_stride 192 --swp_bank_size 32 \
    --lm_latent_dim 512† --lm_enc_layers 2† --lm_rounds 3† \
    --lr 1e-4 --warm_residuals --steps 1200 --k1_mult 4 \
    --eval_rows 512 --per_position_acc† --seed 0 \
    --amp auto --out Cache/e31_wave1
  ```
- **New foundation code (via research-implement, reusable):**
  `LatentMemoryWriter` behind `message_write="latent_memory"` with
  `lm_context ∈ {page_bidir, bixt}`, `lm_latent_dim`, `lm_enc_layers`, `lm_rounds`, and
  its own geometry (`lm_window`, `lm_stride`, `lm_latents`; no auto-shrink);
  per-window pooling rule (also fixes `sw_perceiver`); memory visibility by closed windows
  (for the text rung); per-position accuracy in the probe; a small planted-facts-in-prose
  generator (rung 3a); `token_embedding_dim` / n-gram switches on `ArchSpec`; arch names in the BAPO
  factory. `block_mean` stays the default so E18/E21/E30 checkpoints load.

## Open design decisions (pending the alignment checklist)
Proposed after the spec was drafted; they are **not** part of the bet until the author
confirms them on the alignment page
([e31_architecture.html](../../3_Evaluations_and_Baselines/e31_architecture.html)):
1. **Competition in both arms:** softmax over latents per token (per head), then each
   latent renormalises over tokens, so two latents don't store the same thing.
2. **Null latent (optional, default off):** a latent the reader never sees, absorbing what no
   real latent claims. Useful for random filler; **risky in text**, where "noise" depends on
   the question asked later and may be thrown away. Test it as an ablation, not a default.
3. **8 heads per latent, no averaging**, plus head-diversity and per-head entropy diagnostics.
4. **Reader key/value width ≥ 320 per latent** (5 KV heads × 64; must divide the 15 query
   heads) in every arm. With 1 KV head a 512-dim latent is squeezed to 64 at the read.
5. **Book side of the main path in probes:** it never reaches the answer on the DNA/Glyph
   exams. Keep it (it is the text model's main path) but do not read probe results as
   evidence about it.

## Follow-ups (only after S1)
- **E31a — read the memory in every upper layer** (planned flavour; run after E31's write
  result so the write and the read are not changed at once). Each main-path layer above
  the first local layer = local self-attention + a memory read (its own K/V projection of
  the same latents) + FFN, the encoder–decoder / RETRO pattern. Fairness: the full-read
  control gets the same number of raw global reads. Why: a later read's query already holds
  what the first read fetched, so it can follow a chain. Evidence: E25 (averaged notebook,
  ~2.5 M) 2-hop chain at 256 tokens, 0 bits with one read → 99.6 % with two. Limits: extra
  reads did **not** rescue the 1024 lookalike in E25 (a read that never learns to find the
  fact is not fixed by more reads); the gradient gain is ~linear in the number of reads (each
  still spread over C slots at init). Targets: in-order chain at 2048 (0 today), shuffled
  2-hop chain. Log per-layer attention mass on the memory (layers may learn to ignore it).
- **1M plan — two-level memory** (design target; sized by the coarse rung):
  *coarse* level = W 512 · K 8 (≈ N/48), read densely by every token in every reading layer
  (gist + index; ≈ 22K slots per token at 1M); *fine* level = W 128–256 (≈ N/4–N/8), never
  read densely — the question takes the top ~16 pages from the coarse level and reads only
  their fine latents or raw text (≈ 500 extra slots per token). Per reading layer at 1M:
  ≈ 23 G dot products vs 550 G for full attention. 10M: add a level per ~8× (three levels ≈
  0.3 T per layer at 10M vs 55 T full). Same split as DeepSeek-V4 CSA (4× + top-k) / HCA
  (128× dense); ours differs by addressed, trained latents.

| context | full attention / layer | flat N/6 | flat N/48 (coarse) | two-level (N/48 + 16 fine pages) | three-level |
|---|---|---|---|---|---|
| 128K | 8.6 G | 2.9 G | 0.36 G | 0.43 G | 0.13 G |
| 1M | 550 G | 183 G | 23 G | 23 G | 3.5 G |
| 10M | 55 T | 18 T | 2.3 T | 2.3 T | 0.29 T |
- **H-vs-C grid** (the superseded draft): H vs latents-per-window vs depth at ~9M on the
  winning write, scored by examples-to-75 %.
- **Read side:** question-driven page scoring + fetch (HiLS-style score-weighted fusion,
  teacher KL), 2-hop reads for the shuffled chain.
- **Streaming:** carried state across windows, questions at random points (needs the
  pooling-rule fix), append-only multi-level archive.

## Result
**Capability suite, full tier, 30M, 3 seeds (suite 2026-09-25.v3), 2026-09-25.**
- Run id: `Cache/capability/e31_full_30m_{odra,polonez,polonez_2k}` (merged scorecard); targeted reruns in
  `Cache/capability/e31_reruns`. Branch `e31-latent-memory`.
- Verdicts:
  - **e31_page: scale up.** Frontier L5 at 30M: every gating level L0–L5 passes, plus both L6 stretch cells.
  - e30: promising, fix before scaling. Frontier L1: it fails the 1k lookalike, the long lookups and chain-2k.
  - e31_bixt: not ready. Frontier L0: slow learner; it passes L1 cells only with 8× budget.

| cell (bits / 64) | e31_page | e30 | e31_bixt | dense | pre-set criterion |
|---|---|---|---|---|---|
| lookalike-1k | **57.5** (94 %) | 25.7 | 24.0 | 63.3 | S1 ≥ 47 ✓ |
| lookup-1k | **57.6** (94 %) | 25.7 | 17.7 | 62.9 | S1 ≥ 40 ✓ |
| chain-1k | **60.6** (97 %) | 44.2 | 21.8 | 63.6 | S2 ≥ 47 ✓ |
| lookup-2k (step 5e-5) | **52.2** (89 %) | 25.7 | 0 | 62.9 | S3 ≥ 32 ✓ |
| chain-2k | **61.8** (98 %) | 15.0 | 30.3 | 63.1 | — |
| fact-1k (Glyph, /96) | **94.6** (99 %) | 81.6 | 80.0 | 95.2 | S6 (rung 2) ✓ |
| shuffled-1k (stretch) | **61.2** (98 %) | 1.3 | 0.1 | 63.0 | — |

- **S5 (write health) ✓.** Per-letter accuracy is flat across the 32 answer letters; E30's
  13-letter salience wall is gone. With the memory off, accuracy drops to chance.
- **S4 (latents vs context) is open.** In wave 1, the context-only control (`e30_ctx`: the
  E30 writer with a 64-token pre-encoder reach) reached 56.6 bits on lookup-1k, against 57.6
  for e31_page. On lookup, most of the gain over E30 is *context reach*, not the latent
  design. The S4 cell (lookalike-1k) plus lookup-2k, chain-2k and shuffled-1k are queued for
  `e30_ctx` (`Cache/capability/e30ctx_control_30m*`).
- **Caveats:**
  1. E31 has 35.7M parameters vs 31.2M for dense and e30 (+14 %; the suite rule is ±5 %).
  2. The platform changed for every arm (token embedding 128, no n-grams); e30 lost bits on
     some short cells compared with its ledger numbers.
  3. At 2,048 tokens, takeoff is stochastic:
     - every arm needed step size 5e-5 (at 1e-4 all but e30 stayed at chance);
     - dense seed 0 and e31_page seed 2 were slow or failed even at 5e-5;
     - reruns on another server flipped outcomes.
  4. The extension rule misses slow, still-rising curves (bixt, e31_page seed 2).
     Suggested fix: also extend when accuracy rose ≥ 5 points in the last third.
  5. The three lookup-2k 5e-5 `job.json` files were reconstructed from their 1e-4 siblings;
     the metadata was lost when the folders were renamed.
- **Coarse rung, lookup-2k, 1 seed:** W 512, K 8 reached 39 % with m = 1 and chance with
  m = 5 (fine geometry: 93 %). Under 0.5× the fine bits, so by the decision rule the 1M design
  needs the two-level memory.
- **Length ladder (2026-09-26): passes on lookup.**
  - Setup: e31_page, fine memory, `--message_raw_window 256`, seed 1.
  - Evaluation: 64 rows per length. The fact position is uniform in the book; SE ≈ ±0.3–2 points.
  - Directories: `Cache/length_ladder/*` on both servers.

  | lookup accuracy | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
  |---|---|---|---|---|---|---|---|
  | baseline memory, trained at 2k | 92 | 81 | 56 | 30 | 29 | 25 | 27 |
  | baseline, curriculum 8k → 16k → 32k → 64k | 85 | 85 | 86 | 84 | 84 | 84 | 53 |
  | length-invariant (`lm_addr none`, `lm_slot_pos boundary`), trained at 2k | 98.5 | 97 | 94 | 87 | 69 | 52 | 36 |
  | **length-invariant + one 8k stage** | **98.6** | **98.9** | **98.5** | **98.4** | **97.2** | **90.7** | **75.6** |
  | length-invariant, seed 2, trained at 2k | 65 | 65 | 65 | 58 | 49 | 40 | 32 |
  | no-memory control (e18_local) | 24 | 24 | 25 | 25 | 25 | 26 | 26 |

  - **Pass line met:** ≥ 75 % at 16k and 32k, and at 128k as well, after a single 8k
    stage.
  - **Baseline memory:** it learns an *absolute-position* read. Trained at length L, it
    reaches about 2L, and far facts fail first (depth effect at 8k: 42 % vs 70 %).
  - **Length-invariant memory:** accuracy is flat across fact depth at every length; the
    residual loss is uniform, which points to dilution. On seed 2 it keeps the same fraction
    of its 2k bits as seed 1 at 16k and 32k.
  - **SSMax:**
    - with the absolute address, it never took off (chance);
    - with the length-invariant memory, it held 40 % flat from 2k to 128k but was undertrained
      (took off late).
  - **Cost** is linear in length (see *Length ladder*: 1.35 s / 3.4 GB per 128k row).
    Training fits a 3090 up to 64k with flex: 22 GB, 5.3 s per 2 rows. sdpa runs out of
    memory at 32k.
- **Chain does not transfer yet.**
  - Dense and e31_page trained at 2k both drop to chance at 4k.
  - The baseline memory trained at 8k or 16k works *only at the training length*: 92 % at
    8k, 29 % at 16k; then 93 % at 16k, 37 % at 8k. The 4-hop solution is a
    length-specific position shortcut.
  - The length-invariant memory trained from scratch did not take off on chain (chance).
  - Pending: chain from the length-invariant lookup weights, and a longer from-scratch run.
- **S4 fails (latents vs context).** e30_ctx, the E30 writer with a 64-token pre-encoder,
  matches or beats e31_page:
  - lookalike-1k: 63.0 vs 57.5 bits (3 seeds);
  - lookup-2k: 84 / 99 % on seeds 0 and 1;
  - chain-2k: 97.5 %;
  - shuffled-1k: 96.5 %.

  E30's deficit was its 16-token context (the salience wall), not the window writer. What
  E31 adds that is still untested against e30_ctx is the length-invariant address; the
  e30_ctx ladder is queued.
