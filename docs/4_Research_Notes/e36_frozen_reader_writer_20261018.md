# E36 — a trained writer for a frozen reader (18 Oct 2026)

## Brief
- **Reader:** SmolLM2-360M, frozen. It is a plain Llama model with attention in every layer, Apache-licensed and
  independently re-scored. It is the only model under 500M already measured reading injected vectors (61 tokens
  per optimised vector). Qwen3-0.6B-Base (just over 500M) is the second reader, to test transfer. Small diffusion
  models are not yet coherent enough.
- **Writer:** no other family meets the seven needs together. Recurrent states and test-time training cannot hold
  100M tokens; weights-as-memory is too slow to write. The compressors that work with frozen readers have
  converged on our design: chunks in parallel, a bidirectional encoder, pooled vectors.
- **Three changes to our writer:**
  - each entry covers a span of 8 tokens and carries its position, which gives order for free;
  - it starts from SmolLM2-135M's lower layers, sharing the reader's tokenizer (3 / 8 / 28 layers = 10.6M /
    28.3M / 99.1M);
  - keys for the fetch and values for the reader, trained apart.
- **Training without labels:**
  - continuation plus 20 % reconstruction;
  - then teacher matching, where the teacher is the same reader with the raw text in context;
  - then keys trained over a 100M-token memory.
- **Evaluation:** four conditions per item (no text / memory / text in context / fetched raw text). The tasks:
  - perplexity, SQuAD-type extraction, needle-in-a-haystack;
  - our text checks;
  - no-harm on the harness tasks, with an irrelevant memory present.
- **Main risk:** no frozen-reader compressor under 1B is published. The first step (S0, no training) measures
  what SmolLM2 can do with and without text.

The question from the author, 18 Oct:
- Train the memory (the writer) and keep a frozen reader: a modern, open-weight, small decoder (under 500M
  parameters, autoregressive or diffusion) that writes coherent text on its own and is already well evaluated.
- Task 1: pick that decoder; it must run on Odra or Polonez (RTX 3090, 24 GB).
- Task 2: is there a better writer than ours for seven needs:
  1. 100M tokens in memory;
  2. 1M tokens written in under 1 s;
  3. good training behaviour;
  4. good training tasks and public data;
  5. proven as a semantic compressor that keeps facts;
  6. a notion of time or order;
  7. small (10M, then 30M, at most 100M).
- Task 3: train the writer on available data (plain text, instruction following, reasoning texts, no extra labels)
  and score the decoder on known benchmarks with and without the memory.

Builds on (not repeated here):
- our review of the multi-hop campaign, [e33_multihop_review_label_free_20261018.md](e33_multihop_review_label_free_20261018.md)
  (facts F1–F7; §4: train values task-free, keys from the reader);
- [notebook_writer_compression_20261016.md](notebook_writer_compression_20261016.md) (compression objectives,
  the reader's bits-per-vector ceiling);
- [agent_memory_systems.md](../literature_review/agent_memory_systems.md) (plug-in memories for a frozen model,
  latent channels between agents, time, benchmarks);
- [learned_kv_context_compression.md](../literature_review/learned_kv_context_compression.md);
- the E35 and E33f specs (vector messages vs fetched notes).

## 1. What our own runs already say about a frozen reader

| our evidence | what it means for E36 |
|---|---|
| E35 overfit check (64 documents): a **trainable** reader memorised the answers and ignored the message; a **frozen** reader used it (answer surprise 1.97 with its own message vs 2.47 with another document's) | Freezing the reader is what forces the writer to carry the content. |
| E35, three runs: 16 broadcast vectors per document carried nothing (0 % on every fact task). E33f: the question fetches its best ~20 notes and gets far lookup 97–100 % on documents of 16k–1M tokens | The read must be a fetch driven by the question, not a summary pushed to the reader. |
| Note-content probe (18 Oct, parent session): with the split read (questions see the document only through the notes, E33f) the notes hold the property word 64–69 % and places 74–76 %; every model with a raw window over the document holds the property 8–13 % | With no raw-text shortcut, the writer stores what is needed **without labels**. A frozen reader that sees only the memory has exactly this property. |
| F2: the writer stores only what some downstream loss asks for | The loss on the frozen reader must cover what we want kept: continuation and reconstruction for general text, teacher-matched answers for facts. |
| F3: addressing and content need separate gradient paths | Two outputs per entry: a value (what the reader reads) and a key (what the fetch matches). Keys learn from the reader's likelihood, values from continuation and distillation. |
| E35 Cramming probe: our 30M reader takes about 85 bits per input vector; Pythia-160M 396, Llama-3.2-1B 2,120 (per-text optimised vectors, Cramming) | The reader sets the channel. A 0.3–0.5B reader should take several times more per vector than our 30M one. A trained writer stays far below the optimised ceiling (the "amortisation gap"). |
| E35 demo: our page writer writes 1M tokens into 174,752 entries in 0.44 s on one 3090 | Need 2 is already met at our writer's size. |
| F1 / F4: each hop must be paid for, and naming the hop's output is the strongest lever | A frozen reader cannot learn our in-model naming. Hops move to text: the reader writes the bridge in words and fetches again. This needs no labels, because the reader names the bridge itself. |

## 2. The budget the seven needs set

**Storage at 100M tokens** (bf16; Polonez has 256 GB of RAM and 4 × 24 GB of GPU memory):

| format | per token | at 100M tokens |
|---|---|---|
| the text itself (token ids) | 2–4 B | 0.2–0.4 GB |
| E33 notes (32 per 192 tokens, 64-d key + 64-d value) | 43 B | 4.3 GB (keys 2.1 GB) |
| entries in the reader's input space, 16 tokens each, d = 960 | 120 B | 12 GB, plus 0.8 GB of 64-d keys |
| the same, 32 tokens each | 60 B | 6 GB, plus 0.4 GB of keys |
| the reader's full key/value cache (SmolLM2-360M: 32 layers × 5 heads × 64 × 2 × 2 B) | 41 KB | 4.1 TB |

- A 100M-token memory fits in RAM only in compressed form. The reader reads a few dozen fetched entries per
  question, never the whole memory.
- The fetch is cheap: an exact top-k over 6.25M keys of 64 numbers is 0.8 GFLOP and 0.8 GB of memory traffic per
  question, under 1 ms on a 3090. No approximate index is needed below about 50M entries.

**Speed.** Writing costs about 2 × (non-embedding parameters) × tokens operations, plus attention within a page
(small at 256 tokens). A 3090 peaks at about 71 TFLOPS in bf16; assume 30 TFLOPS sustained for small layers:

| writer (non-embedding parameters) | 1M tokens, one 3090 | 1M tokens, Polonez's four |
|---|---|---|
| 10M | ≈ 0.7 s | ≈ 0.2 s |
| 30M | ≈ 2 s | ≈ 0.5 s |
| 100M | ≈ 7 s | ≈ 1.7 s |

- Our measured 0.44 s fits this estimate.
- Under 1 s on one GPU caps a dense writer at about 15M non-embedding parameters. On four GPUs the cap is about
  60M.
- Parameters that cost no operations are free here: a borrowed, frozen embedding table (the reader's own), or a
  sparse mixture of experts with ≈ 10M active parameters out of 100M.

**Positions.** The reader's cost is set by the positions it attends to. Fetching 32 entries of 16 tokens each puts
512 tokens of evidence into 32 positions.

## 3. Task 1 — the frozen reader

Checked on 18 Oct from model cards and papers. Scores: HS = HellaSwag, ARC = AI2 reasoning challenge, PIQA =
physical commonsense, MMLU = multitask knowledge, GSM8K = grade-school maths, IFEval = instruction following. No
card under 500M reports a long-context or reading score (RULER, needle-in-a-haystack, SQuAD), so those have to be
measured by us.

| reader | params (non-embedding) | design | context, vocabulary | scores | as a frozen reader of input vectors | verdict |
|---|---|---|---|---|---|---|
| **SmolLM2-360M** (`HuggingFaceTB/SmolLM2-360M`, `-Instruct`) | 362M (≈315M) | plain Llama, 32 layers, width 960, attention in every layer | 8k, 49,152 | HS 54.5, ARC 53.0, PIQA 71.7, MMLU-cloze 35.8, GSM8K 3.2; instruct IFEval 41.0; re-scored independently (Ettin paper) | measured: 61 tokens per optimised vector, 61–66 % of attention on the injected vector ([Progressive Cramming 2607.21231](https://arxiv.org/html/2607.21231)) | **main reader** |
| SmolLM2-135M | 135M (≈106M) | same family, 30 layers, width 576 | 8k, same tokenizer | HS 42.1, PIQA 68.4 | 38.5 tokens per vector | laptop checks; its lower layers start the writer (§4) |
| **Qwen3-0.6B-Base** | 0.6B (0.44B) — just over the limit | transformer, 28 layers, attention in every layer | 32k, 151,936 | MMLU 52.8, GSM8K 59.6, BBH 41.5 | no frozen-reader study | **second reader**: does the writer transfer to a stronger reader? |
| Ettin-decoder-400m | 400M | ModernBERT layout: 2 of 3 layers see only the last 128 tokens | 8k, 50,368 | 10-task average 53.1, tied with SmolLM2-360M; base only | none | alternative: the only reader with a twin encoder family (same data, same tokenizer) |
| Gemma 3 270M | 270M (≈100M) | 5 of 6 layers see only 512 tokens | 32k, 262k | HS 40.9, PIQA 67.7 | only 13 % of attention on the injected vector | reject: barely looks at its input vectors; gated licence |
| LFM2.5-350M, Granite 4.0 H-350M, Falcon-H1 | 350M–0.5B | hybrids: attention in only 4–6 layers, the rest convolution or Mamba | 32k | strong instruction scores (LFM2.5 IFEval 77.0) | none | watch: a key/value prefix reaches only the few attention layers |
| small diffusion LMs (BD3-LM 110M, Duo 170M, MDLM, DiffuGPT, dLLM conversions) | 110M–0.6B | masked or block diffusion | ≤ 1k for the from-scratch ones | best from-scratch sample perplexity 25.7 vs 14.1 for a same-size autoregressive model; conversions lose 5–14 points against their autoregressive base | none | reject for now: less coherent, barely evaluated |

Sources: [SmolLM2 card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), [SmolLM2 paper 2502.02737](https://arxiv.org/abs/2502.02737),
[Qwen3 report 2505.09388](https://arxiv.org/html/2505.09388), [Ettin 2507.11412](https://arxiv.org/html/2507.11412v2),
[Gemma 3 270M](https://huggingface.co/google/gemma-3-270m), [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M),
[Granite 4.0 350M](https://huggingface.co/ibm-granite/granite-4.0-350m-base), [BD3-LM 2503.09573](https://arxiv.org/html/2503.09573),
[DiffuGPT 2410.17891](https://arxiv.org/html/2410.17891), [dLLM](https://github.com/ZHZisZZ/dllm).

Why SmolLM2-360M:
- It is the only reader under 500M already measured as a frozen reader of injected vectors, and it attends to them.
- It has attention in every layer, so both ways in work: vectors as input positions, or a key/value prefix per layer.
- Apache licence, a published data recipe, scores re-measured by others, about 0.7 GB in bf16: it fits a 3090
  with room to back-propagate through it.
- Weak points: 8k context, so the "whole text in context" ceiling stops at 8k; almost no maths.

Two warnings from the same literature, both shaping E36:
- **Interference.** A crammed vector drives the reader's 5-shot MMLU to near zero (Progressive Cramming). So the
  no-harm test must be run with an *irrelevant* memory present, not only with none.
- **Norm.** Trained compressors emit vectors about 100× the norm of ordinary token embeddings, and the frozen
  reader needs that to decode them ([ACL 2026](https://aclanthology.org/2026.acl-long.1717/)). E35 initialised
  its output norm like the reader's embeddings; the E36 writer must be free to grow it.

## 4. Task 2 — the writer: is there a better design than ours?

Short answer: no other family meets the seven needs together. The compressors that work with a frozen reader have
converged on a design close to ours: chunks encoded in parallel by a bidirectional encoder, pooled into a few
vectors each, with a small projection into the reader. Three changes from that literature apply to us.

**The alternatives against the seven needs:**

| family | 100M tokens | 1M tokens < 1 s | small | keeps facts | order | verdict |
|---|---|---|---|---|---|---|
| **chunk encoder + pooled vectors** (ours; ARC-Encoder, REFRAG, LCLM, SAC, GMSA) | yes, with a fetch | yes (chunks in parallel) | yes | yes at 4–8 tokens per vector; −7 F1 on multi-hop at 16× | only if each vector carries its span's position | **keep, with three changes** |
| linear recurrent / state-space writer (Gated DeltaNet, Mamba) | no: a fixed state | yes | yes | exact recall falls from 84 at 2k to 28 at 4k (1.3B model) | built in | reject as the store |
| test-time training (TTT-E2E, LaCT, Titans) | no evidence | no: sequential over the text | — | needle-in-a-haystack 0.06 at 128k vs 0.99 for full attention | built in | reject |
| product-key memory layers | parameters, not a store | written by training, not by reading | — | gains on factual tasks | none | reject as the writer (the addressing idea stays) |
| text → adapter weights (Doc-to-LoRA, Generative Adapter) | one adapter per document; no top-k over many | one pass per document | — | good per document | none | reject at 100M tokens |
| **plain retrieval over raw text** (RETRO-style) | yes, trivially | no writer at all | — | the reader reads the text itself | by sorting | **the baseline to beat**: same answers, but 4–8× more reader positions |

Sources: [ARC-Encoder 2510.20535](https://arxiv.org/html/2510.20535), [REFRAG 2509.01092](https://arxiv.org/html/2509.01092),
[LCLM 2606.09659](https://arxiv.org/html/2606.09659v1), [SAC 2510.08907](https://arxiv.org/html/2510.08907v4),
[GMSA 2505.12215](https://arxiv.org/html/2505.12215v2), [No Mean Feat 2510.20797](https://arxiv.org/html/2510.20797),
[CLaRa 2511.18659](https://arxiv.org/html/2511.18659), [LatentPress 2609.01507](https://arxiv.org/html/2609.01507),
[Gated DeltaNet 2412.06464](https://arxiv.org/html/2412.06464), [TTT-E2E 2512.23675](https://arxiv.org/html/2512.23675v1),
[Memory Layers at Scale 2412.09764](https://arxiv.org/abs/2412.09764), [RETRO 2112.04426](https://arxiv.org/abs/2112.04426),
[EPL 2409.14364](https://arxiv.org/html/2409.14364).

**What the frozen-reader compressors report** (all with 7–8B readers; none under 1B has been published):

| method | writer | tokens per vector | frozen reader? | result vs the full text |
|---|---|---|---|---|
| ARC-Encoder | Llama-3.2-3B minus 2 layers, bidirectional; pairs of vectors averaged; 2-layer projector | 4–8 | yes (8B, 7B; a third reader added by training only a new projector) | HotpotQA 45.1 vs 48.3, SQuAD 71.1 vs 77.7; a shared encoder loses < 1 point across readers |
| No Mean Feat | bidirectional LLM, mean over spans | 4 / 16 / 128 | reader LoRA (frozen reader: −3.6) | HotpotQA F1 83.3 / 77.7 / 63.9 vs 84.7; a fixed (untrained) encoder −13.3 |
| LCLM | Qwen3-Embedding-0.6B, 1,024-token windows | 4 / 16 | reader trained | RULER 16k 90.0 / 65.9 vs 93.7; starting from an embedding model beat starting from a language model |
| SAC | Llama-3.2-1B LoRA, the middle token of each span as anchor | 5 / 15 / 51 | yes | HotpotQA F1 67.4 / 58.8 / 48.3; bidirectional +3.4 |
| REFRAG | RoBERTa 125M or 355M, one vector per 16 tokens | 16 | frozen in stage 1 only | 16.5× faster first token; the 355M writer is no better than the 125M one; the cold start needs a reconstruction curriculum |
| LatentPress | the reader's bottom 2 layers, frozen, + a 4–26M adapter | 4–16 | yes | LongMemEval 0.48–0.50 vs 0.49 with the raw text; 43 ms per conversation |

**What this changes in our writer:**
1. **Each entry is a contiguous span with a known position.** Entries are pooled over spans of 8 tokens and carry
   the middle position of their span. In ICAE, giving each vector its span's middle position raised
   reconstruction BLEU from 31.8 to 96.0 (EPL). Our E33 notes have no position and fail "latest" (22 % at 4k);
   E33d had to add one. With spans, order and time (need 6) come for free. The fetched entries are placed in the
   reader in source order, so the reader's own sense of "later in the text" does the rest. Timestamps and
   document ids go into the keys, kept out of the values (LongMemEval: +7–11 % on temporal recall).
2. **Start from a pretrained encoder that shares the reader's tokenizer.** We take the lower layers of
   SmolLM2-135M, made bidirectional within a page. The reader is SmolLM2-360M, and the two share a tokenizer.
   The size ladder falls out of the layer count:
   - 3 layers: 10.6M non-embedding parameters;
   - 8 layers: 28.3M;
   - 28 layers: 99.1M.

   The 49k-row embedding table is borrowed and frozen, so it costs no compute.

   ARC truncated a 3B model the same way, and LatentPress used the reader's own bottom layers. The alternative is
   Ettin's 17M / 32M / 68M encoders paired with the Ettin 400M decoder (one tokenizer, one dataset).
3. **Two outputs per entry, kept apart:**
   - a value: 576 numbers, stored at writer width and projected into the reader only when fetched;
   - a key: 64 numbers, for the fetch.

   This is F3. Values learn from continuation and distillation. Keys learn from how much a fetched entry lowers
   the reader's loss on the answer (Atlas/REPLUG style, or CLaRa's straight-through top-k: no relevance labels).

**The seven needs, met by this writer:**

| need | how | number |
|---|---|---|
| 100M tokens | 12.5M entries at 8 tokens each; values 14.4 GB and keys 1.6 GB in RAM; exact top-k fetch | under 1 ms per question |
| 1M tokens < 1 s | pages written in parallel | 10.6M: 0.7 s on one 3090; 28.3M: 0.5 s on four; 99M: about 1.7 s on four (misses; a mixture of experts with ≈ 10M active would meet it) |
| training | frozen reader, staged (§5); literature recipes exist | — |
| tasks and data | continuation, reconstruction, teacher matching on public text | §5 |
| semantic compressor | 4–8 tokens per vector is where frozen-reader compressors stay within 0–3 points of the full text | — |
| time / order | span positions, entries placed in source order, timestamps in keys | — |
| size | 10M → 30M → 100M by layer count | — |

**How the reader takes the entries.** There are two ways in:
- **A. Input positions.** The fetched entries become input vectors, placed in front of the question. All the
  frozen-reader papers above do this.
- **B. Key/value prefix.** A small expander (about 12M parameters) maps each fetched entry to a key and a value in
  every one of the reader's 32 layers.

For small frozen models, input-only prompts are weak: prompt tuning matches full fine-tuning only near 10B
parameters ([2104.08691](https://arxiv.org/abs/2104.08691)). Per-layer prefixes worked on GPT-2 medium
([2101.00190](https://arxiv.org/abs/2101.00190)). So E36 runs both as its first two arms.

## 5. Task 3 — training without labels, and the evaluation

**Training, four stages, no hand-made labels.** The teacher is always the same frozen reader with the raw text in
its context.

| stage | trains | loss | data | evidence |
|---|---|---|---|---|
| S0 qualify the reader | nothing | — | the evaluation suite below; a Cramming probe (tokens per optimised vector) | readers differ 38–71 tokens per vector; floors and ceilings under 1B are unpublished |
| S1 task-free writer | writer values + projector (or expander) | the reader continues the next page from the entries of earlier pages (80 %), and rebuilds a page from its own entries (20 %) | FineWeb-Edu; long documents from PG-19 train and ProLong (books, arXiv) | ARC: 20 % reconstruction best (41.6 vs 39.8 at 0 % and 37.5 at 100 %); REFRAG: the cold start needs a curriculum (1 page → many). Curriculum: 4 tokens per entry, then 8 |
| S2 teacher matching | writer values | KL between the reader's output given the entries and given the raw text; mixed forward/reverse (λ ≈ 0.6) | continuations of plain text (no labels at all); SYNTH (a Wikipedia passage, a question, a reasoning trace, an answer; CC-BY); passages and questions of SQuAD, TriviaQA, HotpotQA (answers written by the teacher, PISCO style); SmolTalk2 conversations (earlier turns become memory, the reply is the target) | Cartridges: distillation beat next-token training (+8.6 chrF); xRAG: −3.1 without the KL term; KV-Distill: mixed KL 86.0 vs 79.1 for reconstruction + continuation (SQuAD) |
| S3 the fetch at 100M tokens | keys only (values frozen) | the reader's answer loss over the fetched entries (each entry's share of lowering it) | a 100M-token memory: Wikipedia and FineWeb text with the evaluation passages planted inside | CLaRa: fetch trained in the compressed space, recall@5 96 % on HotpotQA at 4×; F3 |

Multi-hop is not trained. The reader writes the bridge entity in words, and the question fetches again (iterative
retrieval). This tests F1/F4 with no labels: the hop is paid for in text.

Compute: S1 at 1B target tokens, with a frozen 360M reader and a 28M writer, is about 4 hours on Polonez's four
GPUs (my estimate).

**Evaluation: one suite, four conditions per item.** The four conditions are:
- no text: the floor;
- memory: fetched entries;
- the relevant raw text in context: the ceiling;
- raw text fetched by the same keys, at equal reader positions: the retrieval baseline.

| measure | what it shows | tool | known numbers for small readers |
|---|---|---|---|
| continuation perplexity on PG-19 test, arXiv, ProofPile | general memory of what came before, beyond the 8k window | own code, CEPE / AutoCompressors protocol (score the last 256 tokens of ≥ 32k-token documents) | OPT-1.3B: 13.21 with no past → 12.47 with 28k tokens compressed |
| SQuAD, SWDE, FDA (extraction from a passage) | whether facts survive | lm-eval-harness `squad_completion`, `swde`, `fda` | a 360M model with the passage: 36.2 / 44.7 / 74.8; without it near 0 |
| TriviaQA, NQ, PopQA with a passage (completion format) | facts plus the reader's own knowledge | lm-eval-harness / own code | 360M with the passage: TriviaQA 46.5, NQ 27.8; SmolLM2-360M closed-book TriviaQA 16.9 |
| needle-in-a-haystack (RULER) at 1k–8k, then 32k → 1M → 100M from memory only | addressing at scale | RULER repo, harness `ruler` | nothing published under 1B; answers cannot be guessed |
| our text checks T1–T7 (lookup, keyed, latest, compose, deduce), re-tokenised for the reader | direct comparison with E33 / E33f; latest = time | `evaluation/text_checks_eval.py` (needs a reader adapter) | E33f: far lookup 97–100 % on 16k–1M documents |
| HotpotQA, 2Wiki, MuSiQue (distractor setting) | multi-hop, with iterative fetch | own code | no evidence under 1B; ceiling first |
| no-harm: HellaSwag, ARC, PIQA, Winogrande, MMLU, GSM8K, IFEval | the reader is untouched | lm-eval-harness | with no memory: identical by construction (assert it). With an irrelevant memory present: within 1 point |
| system | write speed, bytes per token, fetch time at 100M, reader positions per answer | own code | E35 demo: 1M tokens in 0.44 s |

Dropped for small readers:
- LongMemEval: the smallest reader tested (4B) scores 0.34 even with the full history.
- LoCoMo: a 1B reader with the full context gets 7–13 F1.
- BoolQ: a 270M reader scores at the majority baseline.
- BABILong: a frozen 1.2B reader gets 41 % even at 0k (Llama-3.2-1B, LM2 paper), so the gap is small.

Sources: [Cartridges 2506.06266](https://arxiv.org/html/2506.06266v3), [xRAG 2405.13792](https://arxiv.org/html/2405.13792),
[PISCO 2501.16075](https://arxiv.org/html/2501.16075), [KV-Distill 2503.10337](https://arxiv.org/html/2503.10337v1),
[ICAE 2307.06945](https://arxiv.org/html/2307.06945), [AutoCompressors 2305.14788](https://arxiv.org/html/2305.14788),
[CEPE 2402.16617](https://arxiv.org/html/2402.16617), [JRT 2407.05483](https://arxiv.org/html/2407.05483),
[RULER](https://github.com/NVIDIA/RULER), [LongMemEval 2410.10813](https://arxiv.org/html/2410.10813v2),
[A-MEM 2502.12110](https://arxiv.org/html/2502.12110), [SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH),
[SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[ProLong data](https://huggingface.co/datasets/princeton-nlp/prolong-data-64K), [lm-eval tasks](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md).

## 6. The E36 bet and its first runs

**The bet.** A 10–30M writer, trained only on public text against a frozen SmolLM2-360M, makes the reader answer
from a 100M-token memory at ≥ 80 % of its own score with the relevant text in context. It does this:
- at 8 tokens per reader position;
- writing 1M tokens in under 1 s;
- without moving the reader's own scores.

What would be new: every published frozen-reader compressor uses a 7–8B reader and a writer of 1B or more. A small
writer for a small frozen reader, with a fetch at 100M tokens, has no published instance.

| step | runs | pass | stop |
|---|---|---|---|
| S0 (no training, ~2 h) | SmolLM2-360M, its instruct version, Qwen3-0.6B-Base: floor and ceiling on the suite; Cramming probe; share of attention on injected vectors | ≥ 3 suite tasks with a ceiling − floor gap ≥ 15 points | a reader with < 2 such tasks is dropped |
| S1 (≈ 4 h each) | 10.6M writer, 1B tokens; arm A input positions vs arm B key/value prefix | the entries recover ≥ 50 % of the perplexity gain that the raw earlier text gives, at 8 tokens per entry | < 20 % on both arms after 1B tokens: the small reader's channel is too narrow |
| S2 | the better arm; then 28.3M | SQuAD / SWDE / FDA from memory ≥ 80 % of the in-context score; irrelevant memory moves MMLU and HellaSwag < 1 point | < 50 % of in-context |
| S3 | keys trained over a 100M-token memory | needle and lookup at 100M ≥ 80 % of the single-document score; equal to the raw-text baseline at 4–8× fewer reader positions | raw-text fetch beats it at equal positions by > 10 points |
| writer ladder | 10.6M → 28.3M → 99.1M | each size adds ≥ 3 points on the S2 suite | — |

**How this generalises the E33 lessons** (review F1–F7):
- The frozen reader with memory only removes the raw-window shortcut, which is what made the writer drop facts
  (note-content probe).
- Values and keys train apart (F3).
- Hops are named in text by the reader itself, not by generator labels (F1, F4, F7).
- Order is the span position, not a learned rank (F5, F6).

**Agent to agent.** The fetched entries are the wire between agents that share the reader; that is E33f's
protocol. A different reader needs only its own projector: ARC added a reader by training a new projector, with a
shared writer losing < 1 point.

## 7. What is uncertain
- **Nothing under 1B is published** for frozen-reader compression, reading scores with and without text, or RULER.
  Every number for SmolLM2-360M in §5 except no-harm and closed-book TriviaQA comes from a different 360M model
  (JRT) or must be measured in S0.
- A preprint finds SmolLM2-360M-Instruct near 0 % on NQ even with the gold passage, in free-form format
  ([2603.11513](https://arxiv.org/html/2603.11513)). Prompt format decides small-reader QA, so S0 fixes the format
  first.
- My own estimates: the speed table (2 × parameters × tokens at 30 TFLOPS sustained on a 3090), the storage table
  and the S1 compute. The 0.44 s E35 measurement is the only measured speed.
- The literature disagrees on bidirectional vs causal writers (SAC: bidirectional +3.4 F1; LCLM: causal better).
  Our page writer is bidirectional within a page.
- PCC's architecture and REFRAG's curriculum ablation could not be read; REFRAG's code is not released.
- Teacher matching with a 360M teacher is only as good as the teacher's own answers. On plain-text continuations
  this does not matter; on QA it caps the student at the teacher.
