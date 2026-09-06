# E18 — Perceiver AR v2: one global read, deep local stack, every token trained

- **Status:** approved (design discussed 2026-09-06; pilot authorized on Polonez; AWS main run pending explicit go)
- **Serves:** the **VC-facing long-context family** — a from-scratch < 600M-dense-parameter language model that trains with 256k causal context on every token and demonstrates 1M-token context at inference, as the platform for latent reasoning (E19), block-diffusion decoding (E20), and latent agent-to-agent messages (E21). Feasibility and literature: [perceiver_ar_modern_reproduction_feasibility.md](../../4_Research_Notes/perceiver_ar_modern_reproduction_feasibility.md).
- **Implementation plan:** [E18_perceiver_ar_v2_baseline_plan.md](E18_perceiver_ar_v2_baseline_plan.md) *(authored by `implementation-plan`; the HOW)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-06 · closed —

> **Family note.** E18 is the platform run. It carries one architectural bet (below) and one
> within-experiment control (a dense decoder-only transformer at matched parameters, tokens and
> data). Follow-ups are separate specs: **E19** latent write-back reasoning steps, **E20** block-diffusion
> decoder adaptation (SDAR recipe), **E21** latent agent-to-agent messages. E18 must land two
> hooks so those need no retraining: a *write-back projection* from latent states into the prefix
> K/V space (zero-init, unused by the E18 loss) and a *block attention mode* switch
> (`causal` now, `bidirectional` for E20).

## Hypothesis
If a language model reads its entire causal history through **one** cross-attention layer whose keys and values come from a **2-layer sliding-window pre-encoder over tiny hashed n-gram embeddings**, and processes only the current 4096-token block with a deep stack, **sweeping the stack over consecutive blocks so every token receives a loss**, then at ≈600M dense parameters it will (a) match a dense transformer of the same size on short-context loss within 1% at equal tokens, (b) beat it by ≥3% loss on tokens beyond position 8k in long documents, and (c) reach ≥95% needle retrieval at 256k and ≥80% at 1M, **because** the history channel is what long-range prediction needs (a BAPO "attention bandwidth `b`" made global at O(M) cost) while depth is only needed locally; the original Perceiver AR failed to show context gains because it read *raw* embeddings with *no* loss on the history, and both defects are removed here.

## Builds-on
- **Foundation:** `training/train_concept_pretraining.py` + `PerceiverDenoiseTrainer` (HF Trainer, `accelerate launch`), `scripts/train_concept_pretraining_multigpu.sh` (env-var knobs), `scripts/pretokenize_mix.py` + `data/mix_recipes/*.json` (manifest flow), `data/data_collators.py:DataCollatorForCausalLM`, `nn/muon.py` (Muon + AdamW split), `nn/concept_encoder_perceiver.py:ChunkedLMHeadCE` (chunked output head), `nn/sequence_parallel.py` (1M-context path on 24 GB cards), W&B identity helpers. **New reusable family** `perceiver_ar` (`nn/perceiver_ar_lm.py`) selected by config; objective `causal_lm`.
- **Init / checkpoint:** random init, from scratch (no pretrained backbone: the deliverable is a model we own end to end; E10–E17 evidence about frozen Gemma does not transfer).
- **Baseline to beat:** (1) the **matched dense control** trained inside this experiment (same tokenizer, data, tokens, optimizer, parameter count; plain causal transformer at 8k context) — the number that shows the architecture, not the data, is the win; (2) the literature anchors: Perceiver AR 974.6M PG-19 test word-ppl **28.9** / WikiText-103 **18.35**; LLP-214M PG-19 **18.83**.
- **Materially new vs the ledger:** nothing in E01–E17 is a from-scratch decoder-only LM; E16b/E17 are Gemma-backbone memory experiments at 2–4k. New here: (i) single global read over 256k–1M with K/V from a contextualized pre-encoder rather than raw embeddings; (ii) block-swept training (loss on 100% of tokens, prefix K/V computed once per sequence); (iii) tiny factorized input embeddings + hashed 2/3-gram tables feeding the reader; (iv) FlashAttention-3 / FlexAttention bottom-right causal cross-attend as the only long-range kernel. Not a retune of any prior spec.

## The architectural bet
```
tokens (M) ─► tiny embed e=256 (+ hashed 2-gram, 3-gram tables) ─► MLP up-proj to d ─►
  pre-encoder: 2 × causal sliding-window blocks (window 1024)      # O(M·w·d)
  ─► RMSNorm ─► K,V projection (GQA, 2 kv-heads)                    # computed ONCE per sequence
for each block b of N=4096 tokens (swept left→right, causal):
  Q_b = pre-encoded tokens of block b
  X_b = CrossAttend(Q_b → K,V[0 : end of block b], causal bottom-right)  + FFN   # the ONE global read
  X_b = LatentStack_L20(X_b)   # causal self-attn over the last N tokens, GQA, QK-norm, RoPE, SwiGLU,
                               # value-embeddings in 3 layers, U-net skip lambdas, zero-init out-proj
  logits_b = lm_head(X_b)      # untied, chunked / fused CE, logit softcap 30
loss = mean CE over all M tokens
```
**Training-time identity (what is actually implemented).** Sweeping the block over the whole
sequence makes the single cross-attend *exactly one full-causal attention layer* over the
sequence, and the latent stack *exactly L sliding-window layers of width N*. So E18 at train time
is a transformer with a per-layer attention pattern
`[SWA 1024] × 2 → [FULL causal] × 1 → [SWA 4096] × 20`, run with standard kernels
(FlashAttention window/causal on Hopper, FlexAttention on Ampere, SDPA reference). At inference
the computation is identical to training: the global layer caches K/V for the whole prefix
(**the only unbounded cache**, ≈1 GB at 1M tokens with 2 kv-heads × 128 dims in bf16) and every
window layer keeps a ring buffer of its last N keys (20 layers × 4096 ≈ 84 MB). Because each
window layer's inputs already depend on earlier tokens, the stack's receptive field grows with
depth (up to L·N) — the paper's "latents = the last N tokens recomputed" is an approximation of
this; the accurate description is one global read at the bottom plus bounded per-layer windows.
This is *not* a Gemma-style interleaved local/global model (those pay unbounded K/V at every
global layer and put globals throughout; here there is exactly one, at the bottom, and the bet is
that it suffices).

**Sizes (E18 main):** d=1280, 10 heads × 128, 2 kv-heads, SwiGLU 3456, L=20 latent layers +
2 pre-encoder layers + 1 cross-attend layer; vocab 128,256 (Llama-3 vocabulary via the ungated
`HuggingFaceTB/SmolLM3-3B` tokenizer, verified id-identical to Llama-3.2 so ProLong-512K shards
load directly). ≈ **594M dense** (stack 344M, pre-encoder 34M, cross-attend 17M, input 35M,
lm_head 164M) + **79M sparse lookup** (2×2^17×256 n-gram tables 67M, value embeddings 12M).
Strict < 600M total is reachable by dropping n-gram buckets to 2^16; decided at the pilot.

**Hooks for the family (in E18, unused by its loss):** `write_back_proj` (latent state → prefix
K/V space, zero-init) and `block_attention_mode ∈ {causal, bidirectional}`.

## Why this is not a safe retread
Perceiver AR (2022) is an encoder-decoder with zero encoder layers; the paper itself reports no
gain past 2k context on text. E18 changes the mechanism, not the knobs: the history is
contextualized before it is read (pre-encoder), every position is trained (block sweep), and the
input code is lexically disambiguated at O(1) cost (hashed n-grams, the Over-Tokenized-Transformer
result). Under the BAPO lens already adopted in this repo, E18 maximizes raw-token reach `b` to the
full context with a single layer and keeps depth local — the opposite allocation from E10–E17,
which spent depth on a compressed summary channel `a` and found it unused under plain CE.

## Success criteria (set BEFORE running)
**Pilot (Polonez, 4×3090, ~125M dense, M=8k then 32k) — gates for spending AWS:**
- P1 tests: FlexAttention/FA/naive cross-attend equivalence (bottom-right causal, doc masks), block-sweep loss ≡ single-block loss, KV-cache decode ≡ full recompute — all pass.
- P2 copy task: mirrored copy at 32k context, ≥ **99%** token accuracy on held-out sequences.
- P3 long-context use: at equal tokens, the 32k-context stage's eval loss on PG-19 validation tokens at positions ≥ 8k is ≥ **2%** lower (relative) than the 8k-context run's; passkey retrieval at 32k ≥ **90%**.
- P4 architecture tax: training throughput (tokens/s/GPU) at M=8k ≥ **60%** of the matched dense 125M control on the same cards.

**Main (AWS, 594M dense):**
- M1 short context: eval loss at equal tokens (300B) within **1%** of the dense control; lm-eval-harness (HellaSwag, ARC-E/C, PIQA, MMLU) average within 2 points of SmolLM2-360M.
- M2 long context: loss on positions ≥ 8k of long documents ≥ **3%** lower than the dense control at equal tokens; RULER average ≥ **80** at 128k; needle-in-a-haystack ≥ **95%** at 256k and ≥ **80%** at 1M.
- M3 paper parity: PG-19 test word-ppl ≤ **25** (Perceiver AR 28.9); WikiText-103 zero-shot word-ppl ≤ **20**.
- M4 efficiency: 1M-token prefill on one 80 GB GPU; prefix KV cache ≤ 1.5 GB at 1M.

## Kill criteria (set BEFORE running)
- Pilot: P3 fails after one fix iteration (pre-encoder 2→4 layers) → the single-read design is falsified at this scale; **do not launch AWS**; write the report.
- Pilot: P4 < 40% → kernel path is wrong; stop and fix before any AWS spend.
- Main: at 10% of the token budget, E18 loss > dense control by > 3% at equal tokens → stop (architecture tax).
- Main: at 100B tokens, passkey at 32k < 80% → stop stage 1, run the 32k diagnostic before continuing.
- Any: eval loss rising over 3 consecutive evals (the E05 divergence signature) → stop, retune per the Muon protocol (wd 0.1, adamw_lr 2e-4, constant-with-warmup calibration).

## Plan
- **Data (pilot):** new recipe `data/mix_recipes/e18_pilot_longdoc_v1.json` over the existing pretokenize spine, SmolLM3 tokenizer: PG-19 40% (long books), FinePDFs 20%, FineWeb-Edu 30%, stack-edu python 10%; ~3B tokens; packed with document masks; eval = PG-19 validation (positions ≥ 8k reported separately) + passkey set + copy-task set.
- **Data (main, stage 1, M=8k, 600B tokens):** Nemotron-CC-v2.1 High-Quality 26B + Medium-High 16.9B + High-Quality-Synthetic 93.5B + HQ-Translated 39.6B; Nemotron-CC-Code-v1 60B; Nemotron-Pretraining-Code-v2 raw 60B; Nemotron-Pretraining-Specialized-v1 (RQA, Math-Textbooks, STEM-SFT) 40B; PG-19 train 3B; ProLong-64K 31B. **Stage 2 (M=256k, 50B):** 34% long (ProLong-512K cut to 256k, PG-19, repo-level code) / 66% stage-1 mix. **Stage 3 (M=512k, 15B):** ProLong-512K native, YaRN on RoPE. Decontaminate PG-19 test/val and WikiText-103 test from all CC-derived shards. ClimbMix excluded (CC-BY-NC).
- **Compute:** pilot ≈ 12 GPU-h on Polonez (4×3090; watch the cooling issue — abort on thermal throttling, fall back to Odra). Main ≈ 3.0k H100-h stage 1 + 0.3k stage 2 + 0.2k stage 3 + 0.8k dense control (300B) + 0.5k evals ≈ **5k H100-h** ≈ $26k on p5 capacity blocks (6 nodes × ~4.5 days). Jean Zay reserved for E19–E21.
- **Steps / epochs:** pilot 2B tokens @ 8k + 0.5B @ 32k; main as above, WSD schedule, Muon lr 0.01 / adamw 2e-4 / wd 0.1 (E05-calibrated), batch ≈ 4M tokens.
- **Launch (pilot, Polonez):** `bash scripts/launch_e18.sh` → pins `MODEL_FAMILY=perceiver_ar OBJECTIVE_VARIANT=causal_lm HIDDEN_SIZE=768 NUM_LAYERS=12 PAR_BLOCK=2048 PAR_PRE_LAYERS=1 PAR_WINDOW=512 TOKEN_EMBEDDING_DIM=256 PAR_NGRAM_BUCKETS=65536 MAX_SEQ_LENGTH=8192 OPTIMIZER=muon ATTN_BACKEND=flex TOKENIZER_NAME=HuggingFaceTB/SmolLM3-3B PRETOKENIZED_MANIFEST=…` and delegates to the generic launcher; `E18_STAGE=32k` switches `MAX_SEQ_LENGTH=32768` and resumes. Dense control: `PAR_MODE=dense`.
- **New foundation code:** `nn/perceiver_ar_lm.py` (config-selectable family: tiny hashed embeddings, pre-encoder, cross-attend with FA3/Flex/SDPA backends, block-swept latent stack, write-back hook, block-attention-mode hook, KV-cache generate), `data/packing.py` (document-masked packing + block-mask cache), `evaluation/long_context_probes.py` (copy task, passkey, per-position loss), tests. Registered in `training/concept_pretraining_factories.py`; knobs in the generic launcher.

## Result
<Filled in AFTER, by experiment-track. Link out; do not paste full results here.>
- Run id: `<run_id>`
- WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
