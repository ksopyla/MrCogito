# E18 / E21 vs dense — what actually failed (2026-09-16)

**Date:** 2026-09-16 · **Kind:** dated root-cause diagnosis (append-only ledger entry)
**Specs:** [E18](../experiments_specs/done_failed/E18_perceiver_ar_v2_baseline.md) · [E18b](../experiments_specs/done_failed/E18b_retrieval_trained_read.md) · [E18c](../experiments_specs/ahead/E18c_concept_compressed_read.md) (blocked) · E21 original (unmerged) `docs/experiments_specs/ahead/E21_latent_message_pretraining.md` on `origin/cursor/strategy-sota-review-2026-09-e212` · [E25](../experiments_specs/done_success/E25_e21_bapo_capability_ladder.md) (the E21 capability exam that actually closed)
**Prior reports (not rewritten):** [E18 family verdict](../2_Experiments_Registry/run_reports/e18_family_verdict_20260912.md) · [reach ablation](../2_Experiments_Registry/run_reports/e18_reach_ablation_20260909.md) · [E25 DNA capability](../2_Experiments_Registry/run_reports/e25_e21_dna_capability_report_20260916.md) · [E22 root cause](e22_root_cause_20260912.md) · [perceiver revisit](perceiver_revisit_synthesis_20260912.md)
**Live re-check (this note):** W&B (entity from env), Polonez `Cache/eval/e18{,b}` JSON + E21 train logs, Odra leftover Byobu (idle), local DNA leftover dump from `data/bapo_ladder.py` seed 0. No E18/E21 125M weights remain; DNA probe models were discarded by design.

> **Verdict.** The architecture — not eval, not DNA solvability, not “we never trained long enough on language” in the E18b sense — is the cause of the retrieval failures. A matched dense transformer learns the same tasks. What we are missing is not “more concepts” as a first-order story, and not “add a latent reasoner on top of the current channel.” The load-bearing defect is that **the long-range object is not a content-addressable memory**: a 1-layer encoder produces keys the read cannot look up by content; mean-pooling then smears the keys that copy still survives on; exclusive QUERY + leftover packing adds a short-length cliff that dense and uncompressed E18 do not have. Natural-text CE never pays for far content (~0.05 nats), so the language runs look like length successes (CE falling with position, 32k copy) while being memory failures. E21’s original 32k LM run does **not** support a “potential only at 8k–32k” claim: it died at ~2k steps with mean sequence length ~3.2k, and the DNA walls that *did* finish sit at 696–1536 tokens.

This note does not propose a next architecture. It ranks the failure modes and the measurements that would actually decide them.

---

## 1. Fairness: what was matched, what was not

| comparison | params | data / objective | compute / maturity | what the control decides |
|---|---|---|---|---|
| E18 vs dense (stage A) | both ~125M, H=768, 14 layers | same longdoc mix, causal LM, seq **8k**, **1.0B** tokens | same 9,030-step budget, Polonez 4×3090, ~28k tok/s both | **architecture is free on short-ctx LM** (P1) |
| E18 arms A/B/C (N=256) | matched ~125M | same LM, 0.5B tokens | same | **the read is used ≠ useful** |
| E18b T vs DT | matched 125M | **identical** 100% keyed-recall rows, seq 32k | T plateaued; DT solved | **task and eval are learnable; the read is not a content retriever** |
| E21 125M R vs (missing U / dense) | R launched; U died at init; no dense | 95% LM + 5% boundary-aware retrieval, advertised 32k | R SIGTERM ~step 2084 / ~3% of 0.5B; weights deleted | **does not decide the 32k message bet** |
| E25 DNA E21 vs E18 vs dense vs `e18_local` | ≤10.8M (typical H=256 ≈ 2.3M) | on-the-fly DNA, exact ~64-bit prize | 800–8k probe steps; models discarded | **capability of exclusive compressed read**, not LM quality |

**Architecture vs everything else.**

- **Not eval.** Dense on the same DNA JSON hits ~64 bits; `e18_local` stays at chance (leak check). Teacher-forced passkey / keyed-recall on E18b use the same collator as training. lm-eval at 1B tokens is near-chance for *both* E18 and dense (`avg_acc` 0.348 vs SmolLM2-135M 0.447) — uninformative, not a hidden E18 win or loss.
- **Not DNA “too hard.”** Dense solves every scored USER_CORE row we ran. E18 solves INDEX and (at the E25 recipe) MATCH identity out to 1280. The walls are E21-specific or one-global-read-specific.
- **Not checkpoint maturity for E18b.** 20× more task rows moved first-token accuracy 4.2% → 4.49%. Dense reached 99.33% on the same protocol.
- **Yes, maturity for the E21 125M LM.** That run is incomplete (below). Do not read its eval 3.57 as a 32k result.
- **Compute unaudited.** None of the E18/E18b/E21 W&B summaries have `compute/*`. Qualitative matched-budget claims above still hold (same launcher, GPUs, token budget); GPU-hours/kWh are a gap, not a confounder for the accuracy cliffs.

---

## 2. Numbers (re-checked, not restated from memory)

### 2.1 Language modelling — E18 vs dense (125M)

W&B live: perceiver `eval/loss = 3.78957` (run `…20260907_080943`, summary `_step` 10300 / `train/global_step` 10300, state `crashed` after the designed 1.0B stop; eval JSON is the ckpt-9030 number). Dense `eval/loss = 3.78557` (`…20260907_193351`). Documented 3.790 vs 3.786.

| | E18 perceiver | dense |
|---|---|---|
| eval CE @ 1.0B / 8k | **3.78957** | **3.78557** (−0.11%) |
| throughput | 27.7–28.0k tok/s | ~27.4k (1.02×) |
| passkey @32k (LM-only) | 0.0 | 0.0 |
| lm-eval 0-shot avg | 0.348 | 0.348 |
| wikitext ppl (suite) | 109.03 (stage A ckpt) | 110.17 |

Reach ablation (same weights, only the read’s window changes; Polonez `Cache/eval/e18/reach`):

| model | Δ CE, read/all-layers cut to 2k, at [2k, 8k) |
|---|---|
| stage A, N=2048 | **+0.0002** nats (symmetric tail) |
| arm A, N=256 (read needed) | **+0.024** nats, 25σ |
| arm C, **no read** | eval **4.091** vs arm A **4.090** |
| arm B, read at layer 7 | eval **4.090**, depended on 5.7× less |
| dense, all layers cut | **+0.035** nats — the prize exists |

Plain copy @32k offset 16k: **99.9998%**. Window 16,382 (two tokens short) → **0.4%**. Positional retrieval through the read is exact. Content retrieval is not.

### 2.2 Retrieval supervision — E18b (32k, 125M)

Polonez `Cache/eval/e18b/*_tasks.json` and `*_passkey.json` (re-read 2026-09-16):

| arm | task share | first-token acc | token acc | passkey @32k | task loss / ppl |
|---|---|---|---|---|---|
| 0 | none | 2.36% | 30.7% | 0.0 | — |
| R | 5% | 4.16% | 31.6% | 0.0 | — |
| R2 | 5% + value embed on the read | 4.38% | 31.7% | 0.0 | LM matched R to 5 dp |
| **T** | **100%** (~20×) | **4.49%** | 31.9% | **0.0** | **3.904 / 49.6** |
| **DT dense** | 100%, same protocol | **99.33%** | **99.27%** | **0.725** | **0.070 / 1.07** |

DT passkey @8k = 0.825 (never trained on that format). The transfer exists for dense and not for the one-read model.

### 2.3 E21 125M LM — launched, then lost

W&B + Polonez `Cache/logs/e21_arm_R_20260912_093654.log` / `e21_arm_U_20260912_125342.log`. `Cache/eval/e21/` is **empty**. No `model.safetensors` under `Cache/Training` or the hot Training NAS for these runs (cleanup after the 2026-09-12 disk event). Surviving weights on both servers are E22 only.

| arm | run | what happened | last numbers |
|---|---|---|---|
| **R** (r=16, the claim) | `perceiver_ar_perceiver_H768L1g1s12N2048_20260912_093714` | SIGTERM (signal 15) ~12:53 UTC; W&B `crashed` | eval **3.603 → 3.572** at gstep 521 / 2084; train loss ~3.39; `mean_sequence_length` **3203**; `mean_batch_max_length` 3305; `message_rows_frac` **0.047**; `receiver_token_frac` **0.178** |
| **U** (r=1 ceiling) | `…20260912_125403` | started, then died during init / first steps | no eval |

`message_rows_frac` ≈ 5% is **as designed** for the strategy-branch collator (only rows ≥ 4096 get a boundary, and then only with frac 0.5). `receiver_token_frac` ≈ 0.18 matches ~19% token-level mass on those rows. The silent failure is **`mean_sequence_length` ~ 3.2k under `length_group` packing of a 32k manifest** — most tokens never saw 8k, let alone 32k. No `--probe message` (`real / none / swapped / raw`) was run. S1–S6 of the original E21 spec are **unmeasured**.

### 2.4 E21 exclusive compressed read — DNA (E25; the actual evidence)

Scored cells from [e25_plots/metrics.md](../2_Experiments_Registry/run_reports/e25_plots/metrics.md). Pass = E21 ≥ 0.75 × live E18 bits (or 0.75 × dense when E18 ≈ 0). Dense must itself be ≥ 75%. `e18_local` is 0 bits on every scored cell.

| wall | dense bits | E18 bits | **E21 bits** | note |
|---|---|---|---|---|
| INDEX r=16 mean @1024 (8k steps) | 63.95 | 63.12 | **53.82 PASS** | slow: 32 bits @3.2k → 53.82 @8k |
| INDEX r=16 mean @1536 | 62.41 | **0** | **0 FAIL** | shared with uncompressed E18 |
| INDEX @2048 / 4096 | dense copies | — | E21 chance | do not 16k |
| MATCH identity r=1 @1280 | 63.97 | 63.95 | **63.94 PASS** | **no KV saving** vs E18 |
| MATCH identity r=1 @1536 | 63.95 | 63.95 | **1.50 FAIL** | extra hop also 0 |
| MATCH r=8 mean @1024 | 62.90 | 62.61 | **60.35 PASS** | |
| MATCH r=8 mean @1280 | 63.85 | 63.95 | **0 FAIL** | extra hop 0.01 |
| MATCH r=4 mean @1280 | — | 63.05 | **0 FAIL** | leftover-keep does not rescue |
| MATCH2 n_dist=1 identity @1024 | 58.81 | 0 | **61.81 PASS** | E21 beats a *dead* E18 |
| MATCH2 n_dist=1 identity @1280 | 62.94 | 0 | **0.01 FAIL** | extra hop 0.01 |
| SELECT identity @692 | 63.97 | 63.89 | **62.76 PASS** | |
| SELECT identity @696 | 63.95 | 63.93 | **0 FAIL** | pack32 / spread / w32 do not rescue |
| hops glob=2 @264 | — | 9.87 | **25.61 PASS** | exclusive needs 2 global layers |
| hops glob=2 @272 | — | 23.64 | **0.02 FAIL** | extra hop PASS 25.48 @272; FAIL @288 |
| shuffled hops n_dist=1 | 21.54 | 25.44 | 6.28 FAIL | extra hop PASS 24.08 |

Learned `u`/`delta` pooling at r=16 is chance (answer CE zeros the channel). Concatenating slots *beside* raw KV (early 512 recipe) is chance; in-place scatter is fine.

### 2.5 Independent leftover dump (this note, CPU, seed 0)

`config_for(scale, task, pack=True, seq_len=…, evidence_align="right")` reproduces the hunt geometry:

| recipe | seq | QUERY | QUERY % 32 | QUERY % 16 |
|---|---|---|---|---|
| INDEX `far_copy` | 1024 / **1536** / 2048 / 4096 | 988 / 1500 / 2012 / 4060 | **28 / 28 / 28 / 28** | 12 / 12 / 12 / 12 |
| MATCH `recall_single` | 1024 / **1280** / **1536** | 986 / 1242 / 1498 | **26 / 26 / 26** | 10 / 10 / 10 |
| SELECT `select_1decoy` | **692 / 696** | 654 / 658 | **14 / 18** | **14 / 2** |

INDEX 1536 and MATCH 1280/1536 walls are **not** leftover-modulo changes. SELECT 692→696 is the one wall that moves QUERY across a 16- and 32-pack boundary. Pack-stride 32 equalized exclusive sender count to 640 at both lengths and still failed at 8k ([pack32 report](../2_Experiments_Registry/run_reports/e25_bridge1k_696_pack32_ip_id_h256_select_20260914.md)).

---

## 3. Four author hypotheses

### H1 — the concept / latent bottleneck is too narrow (too few slots, dim too small, rank collapse)

**Verdict: contradicts as the E18 story; partial for E21 lookup-under-mean; cannot decide RankMe.**

Evidence against “too few / too small” as the primary failure:

- E18’s global read is **uncompressed raw KV** (1 KB/token). There is no slot bottleneck. Content-addressed retrieval still dies (T 4.49% vs DT 99.33%).
- E25 MATCH identity (`r=1`, every sender token is a slot, **same prefix KV as E18**) still dies at 1536 while dense and E18 hold 64 bits. Width was not the scarce resource.
- SELECT 696 is identity slots, H=256. Four extra filler tokens after BOS kill it; adding slots (r=1 already has all of them) does not.
- INDEX at r=16 / seq=1024 **passes** (53.82 bits): ~64 mean-pooled slots *can* carry a marked span. The compression bet is not “zero bandwidth.”

Evidence that **pooling width** does bite, but as *key smearing*, not as “128 was too few”:

- MATCH at 1280: `r=1` PASS 63.94 bits, `r=4` FAIL 0, `r=8` FAIL 0. Copy at 1024 survives `r=16`. Lookup needs the keys to stay unpooled; copy does not.
- That is the opposite of “add more free latents.” It is “do not average the keys of different facts.”

Rank collapse: **cannot decide for E18/E21.** `analysis/run_concept_analysis.py` has no RankMe path for `perceiver_ar` (no concept array). DNA models were discarded. The nearest number in the family is E22 RankMe **265/768** — *diverse* slots that were still used as a document embedding (far marginal 0.05 nats). Diversity without addressability is already in the ledger; collapse is not the default explanation.

**What would decide:** RankMe / pairwise cosine on frozen-mean vs identity slots at MATCH 1024 (pass) vs 1280 (fail), plus a linear probe: can a slot’s key recover the planted key string? If probe accuracy tracks r and the 1280 cliff, smearing is confirmed. If slots remain linearly separable and the read still fails, the defect is the query/read, not capacity.

### H2 — we need an explicit reasoning step on top of the latents

**Verdict: contradicts as the explanation of the current walls; undecided as a later bet once a lookup channel exists.**

- E25 extra exclusive slot-attend (`--message_extra_slot_attends`) is the cheapest “reason over slots” we actually ran. It **rescues shuffled hops** (n_dist=1/2 PASS) and ordered hops at 272, and **does not** rescue MATCH r=8 @1280, MATCH identity @1536, MATCH2 @1280, or SELECT @696. Composition ≠ lookup.
- E22 already had a 4-layer transformer *over* the slots. The array was load-bearing and diverse; far slots were still worth 0.05 nats and interchangeable with a document mean. Refinement without a lookup loss produced a gist.
- E18 arm B (read at layer 7 — deeper queries *and* keys, no extra cache) matched arm A on LM loss and was depended on 5.7× *less*. Depth of the read is not the LM lever. The mid-depth *retrieval* rerun (`Cache/jobs/e18b_mid_taskonly.sh`, ~1.4 GPU-h) was **never launched**, so we cannot say whether a deeper encoder would fix E18b. That is a missing measurement, not support for H2.

A latent reasoner cannot retrieve a key that was never written as a key. H2 becomes live **after** MATCH-under-compression passes.

**What would decide:** extra hop (or a weight-tied slot transformer) on a rung where identity MATCH already passes (e.g. 1280 r=1) *and* the compressor is the only change. If hops-style compute lifts r=4 MATCH, H2 is back. If it does not, the write is the bottleneck.

### H3 — pretraining tasks cannot produce semantically rich concepts, especially on DNA

**Verdict: split. Natural-text CE cannot; DNA can, and that is why DNA is the honest exam.**

| claim | verdict | evidence |
|---|---|---|
| Natural CE at this scale does not supervise far *content* | **supported** | E18 A = C (4.090 vs 4.091); stage A reach Δ ≈ 0.0002 nats; E22 far-slot marginal **0.05 nats**, flat 1k→32k; E18b 5% mix barely moves the needle |
| Therefore the 125M models have no long-range *semantics* | **supported as “no far memory,” not as “LM is broken”** | P1 parity with dense; lm-eval 0.348 = dense; later-position CE falls because books get easier, not because the read retrieves |
| DNA cannot produce addressable bits / “rich concepts” | **contradicted** | dense 99% / ~64 bits on INDEX, MATCH, SELECT; E21 identity MATCH 63.94 bits @1280; E21 INDEX r=16 53.82 bits @1024. The prize is exact and the ceiling is hit |
| DNA is the wrong proxy for language semantics | **supported, and already in the E25 report** | 4-symbol alphabet, planted keys, no lexical polysemy. Passing DNA is necessary-not-sufficient for language. Failing DNA while dense passes is still an architecture fail |

The interesting reading is the opposite of “DNA is too weak a pretraining task”: **DNA is the only objective that paid enough for us to see the circuit.** Language CE paid ~0.05 nats and the models delivered ~0.05 nats of gist. DNA pays ~64 bits and then we see copy vs lookup vs leftover vs hops split cleanly.

**What would decide a language-semantics version of H3:** exclusive-channel training on natural text *with a far-copyable token mass that is a real fraction of the loss* (E23’s planned far-repeat ×8 + 30% dense-label rows), plus the E22 `near`/`far` instrument. If far marginal stays ~0.05 with exclusivity and a paying objective, H3 upgrades from “CE doesn’t pay” to “the write cannot store language.” If far marginal rises and passkey/keyed-recall follow, H3 as currently stated is done.

### H4 — something systematically overlooked

**Verdict: supported. Several, ranked in §4.** The ones that are *not* the story: information leakage (`e18_local` chance on every scored DNA cell); “E21 never ran” (the 125M LM barely ran; E25 ran 122 hunts); “eval tasks don’t exist” (RULER-lite + DNA + reach exist; generation-based lm-eval does not, by spec).

---

## 4. Failure modes, ranked by how much they explain

1. **One-layer keys are not content-discriminative (E18/E18b, and E25 MATCH once you stop pooling).** Structural reading: E18 is a 1-layer encoder + 1 cross-attend + 12-layer SWA decoder. P2 shows the read can *copy by position* (RoPE). E18b T shows it cannot *look up by key* even with 100% labels. E24/E25 MATCH at a *fixed* offset is 0 bits for E18 in the tiny/512 recipes — “find the mark” is not the missing skill; binding key→value is. This single mode explains E18b vs DT, E18 tiny MATCH vs dense, and why E18c stayed blocked. **Weight: highest for the 125M family.**

2. **Natural-text CE does not pay for far content, and the SWA stack already covers the pilot context (E18 P3, E22).** N=2048 × 12 ≈ 24k chained reach, so at 8k–32k the global read is optional for LM loss (arm C). The dense prize for 2k–8k reach is only +0.035 nats. E22’s exclusive-ish array captured a 0.17-nat document embedding + 0.05-nat far marginal. This mode explains why language runs look “fine” (P1, falling CE with position) while retrieval is dead. It does **not** explain DNA walls (the prize is 64 bits and dense takes it). **Weight: highest for any “just train longer on FineWeb” proposal.**

3. **Mean-pool / learned-pool smears keys (E21 MATCH, learned `u`/`delta`).** Frozen mean copies a marked span (INDEX r=16 @1024 PASS) and destroys lookup (MATCH r=4 @1280 FAIL). Training the pooler with answer CE produced a document mean (chance). Identity (`r=1`) restores MATCH until the one-read length wall. This is the compression bet’s actual kill, and it is **orthogonal** to H2. **Weight: highest for E21-as-compressor.**

4. **Exclusive QUERY + leftover / packing geometry (SELECT 692 vs 696).** Four left-filler tokens after BOS move QUERY 654→658, leftover32 14→18, leftover16 14→2. Dense and E18 still score 64 bits. Pack-stride 32, remainder, spread, window 32, extra hop: no rescue. Independent seed-0 dump confirms the hunt table. This is not “SELECT is BAPO-hard”; it is an exclusive-mask / position-grid interaction at one length. **Weight: high for any exclusive-boundary design; local to SELECT-like type cues.**

5. **One global read has a short tape even with raw keys (INDEX 1536, MATCH identity 1536).** Leftover modulo is identical at 1024 and 1536 for INDEX/MATCH. Dense still copies. E18 goes to 0 on INDEX @1536 in the E25 recipe (and already @1024 in the earlier E24 right-align recipe — recipe-dependent, but the existence of a sub-2k copy wall for one read is real). Compression cannot help a tape that already dropped the span. **Weight: high for the 1M-context serving story; idle until (1)+(3) move.**

6. **E21 125M packing never instantiated 32k (length_group → mean_seq ~3.2k; q=0.5 boundary only on long rows → 5% message rows).** The original E21 hypothesis is untested, not falsified, at language scale. Treating eval 3.57 as “E21 LM ≈ E18” would be a mistake: different packing, 4% of the token budget, no message probe, no arm U. **Weight: high for the 32k *claim*; zero for the DNA walls.**

7. **Learned pooling + answer-only CE → document mean.** Same family as (3); listed separately because it is an *objective* bug on an otherwise live compressor. Needs an auxiliary write loss (key classification / slot reconstruction), not more CE.

8. **No KV-cache `generate`; lm-eval generation tasks out of scope; teacher-forced RULER-lite only.** This biases us toward CE and argmax-on-labels. It does **not** create the E18b / DNA cliffs (those are teacher-forced too, and dense passes). It *does* block a 32k generation vibe-check and any HF-style long-gen eval. **Weight: eval-gap, not architecture-kill.**

Ruled out or demoted: leakage (`e18_local`); “value embeddings missing on the read” (R2 null); “not enough retrieval rows” (T vs DT); RankMe collapse (unmeasured, E22 counterexample); “need 32k to see the architecture work” (DNA walls inside 1.5k; see §6).

---

## 5. Decisive measurements still missing

Cheap, no new spec required (do not launch full training to get these — they need *weights* or a probe JSON that already almost exists):

| measurement | why it decides | blocker today |
|---|---|---|
| Linear probe / attention peak at `START` or DNA `query` on E18b T vs DT | “no circuit” vs “wrong circuit” (E18 verdict item 4, never run) | **T/DT checkpoints deleted** |
| RankMe + key-string probe on E21 frozen-mean vs identity at MATCH 1024 vs 1280 | H1 smearing vs read defect | DNA models discarded; re-run is a short GPU probe, not a 125M LM |
| `--probe message` `real/none/swapped/raw` on any exclusive 125M checkpoint | original E21 S1/S4 | no checkpoint; 125M R died; packing would still be ~3k unless `length_group` is fixed |
| E18b mid-depth read (`PAR_GLOBAL_POSITIONS=7`) on 100% task data | is H1 “one layer of keys” or “the read pattern itself” | job staged, ~1.4 GPU-h, **not launched**; warm-start from layer-1 weights makes a *loss* ambiguous |
| `compute/audit_state` on the W&B runs | registry precondition; not a scientific confounder | never run; do not hand-wave GPU-hours |
| SELECT 696 exclusive-mask unit test: which sender positions are visible at the first answer token, 692 vs 696, r=1 | leftover bug vs RoPE-grid vs Flex pad | code-only, CPU; hunt packing knobs failed but no mask dump was saved |

Do **not** treat “run DNA at 8k/16k/32k” as decisive until MATCH-under-`r≥4` at 1280 passes. Those lengths will reproduce the same cliffs at higher wall-clock.

---

## 6. Does E21 have a length story? 8k / 16k / 32k blockers

**Claim to test:** “E21 only shows its potential at long sequences (up to 32k).”

**What supports a length story**

- Serving cost is a long-context claim: r=16 → ~16× fewer global slots; E18 cache is already 23× under dense (1 KB/token vs ~23 KB). That *geometry* only matters at 32k–1M.
- E18 P2 copy at 32k is real: the one read *can* be the retrieval channel when the task is positional and the offset is inside the window.
- E18 arm 0 (protocol-fixed 32k continuation) lowered far buckets ~12% vs stage A (e.g. [16k,32k) 2.611 → 2.279). That is optimizer/length-extrapolation of *local* CE, not proof the read is doing memory (arm C was never measured at 32k).

**What undermines it**

- DNA capability walls sit at **696, 1280, 1536** — inside one SWA window of the 125M stack. The exclusive compressed read fails *before* long context starts.
- INDEX r=16 @1024 is a **budget** story (slow climb to 53.82 bits @8k steps), not a sequence-length story. MATCH/SELECT at the next rung stay chance at 8k extra-step.
- The 125M E21 run’s actual `mean_sequence_length` was **~3.2k**, not 32k. A length-potential claim needs long rows in the batch.
- Language CE at 32k still does not need the read (E18 A=C at N=256; stage A reach Δ~0). Training longer at 32k under plain CE will reproduce E22’s gist, not a memory.

**Hard blockers to evaluating at 8k / 16k / 32k (ordered)**

1. **No E18/E21 125M checkpoints.** Polonez/Odra `Cache/Training` for those run ids is gone; the hot Training NAS has no `perceiver_ar` leftovers. Teacher-forced RULER-lite, reach, message probe, health check: all blocked until a re-train.
2. **E21 original packing.** `BATCH_PACKING_MODE=length_group` on a 32k manifest produced ~3.2k mean length. Re-training without fixing packing will not be a 32k eval.
3. **DNA 8k/16k/32k hunts were correctly skipped.** Walls already inside 1.5k; 16k-not-run is in the hunt reports. Memory on a 3090 is not why — H=256 DNA at 4k already ran (INDEX E21 chance, dense copies).
4. **No HF-compatible KV-cache `generate`.** `PerceiverARLM.generate` is full recompute per step “for probes only.” lm-eval generation tasks raise. 32k *generation* eval is an engineering gap ([eval layer spec](../engineering_specs/long_context_reasoning_eval_layer.md)), independent of the architecture kill.
5. **lm-eval loglikelihood at 1B tokens is chance for both E18 and dense.** Scaling the same suite to 8k-token items will not separate them until the model is past the SmolLM2-135M reference, or the task is retrieval (where DT already separates at 32k).
6. **Positional scheme.** RoPE θ=5e5 + P2 shows *positional* copy at 32k works. Content addressing failed at 32k *and* at 512. YaRN/NoPE is a 1M/10M training issue ([10M blockers](e18_10m_context_blockers.md)), not the reason MATCH dies at 1280.
7. **Arm U / dense 32k message controls were never trained.** Even a finished arm R could not score S2 (retention vs raw message) or S5 (LM cost vs E18b).

Memory/FLOPs at 125M / 32k are **not** the blocker: E18 and E18b already trained there on 4×3090. The 10M-context O(M²) global layer is a different regime and is idle until the channel stores keys.

---

## 7. Cheap analysis this note ran

| attempted | result |
|---|---|
| W&B curves for E18 stage A/dense, E18b T/DT, E21 R | confirms documented CE / first-token / packing stats; `compute/*` absent |
| SSH Polonez eval JSON | T 4.49% / DT 99.33% / DT passkey 0.725 / copy 0.999998; `Cache/eval/e21` empty |
| SSH both servers for weights | E18/E21 125M gone; E22 `final/model.safetensors` present (out of scope to re-eval here) |
| `run_concept_analysis.py` / health / AR ΔCE | **not run** — no perceiver_ar RankMe path; no E18/E21 weights; DNA discarded |
| Local DNA leftover dump | §2.5; SELECT-only leftover change; INDEX/MATCH walls are length/pooling |
| Compute audit writeback | **not run** (would mutate W&B summaries; gap recorded instead) |

---

## Pointers

- Do not treat this file as a spec or an agenda rewrite. Improvement proposals belong in a later note / experiment-design pass.
- E18c remains **blocked** on a functional retrieval channel ([spec](../experiments_specs/ahead/E18c_concept_compressed_read.md)).
- E23 (exclusive concept channel + paying objective) is the language-scale exclusivity bet; it does not by itself fix mean-pooled keys.
- Related laws already in the ledger, which this diagnosis uses rather than replaces: *used ≠ useful* (E18 arm C); *objective must pay for the channel* (E22); *exclusivity is two-sided* (E22); *copy ≠ lookup* (E25).

*Git: written against `dev` @ `4a6cae0`. Live W&B/SSH 2026-09-16.*
