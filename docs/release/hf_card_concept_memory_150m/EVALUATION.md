# 150M concept memory: evaluation details

The method and the full tables behind the [model card](README.md). Every number here has its raw file in this
repository (`eval_results/`, `long_context/`, `ruler/`, `samples/`), and `figures/make_figures.py` draws the card's
figures from those files.

## Contents
1. [Harness and settings](#harness-and-settings)
2. [Standard benchmarks](#standard-benchmarks)
3. [Tiny-ML Leaderboard](#tiny-ml-leaderboard)
4. [Data efficiency](#data-efficiency)
5. [Long documents across models](#long-documents-across-models)
6. [Our own long-context tests](#our-own-long-context-tests)
7. [RULER](#ruler)
8. [Text samples](#text-samples)
9. [What we have not measured](#what-we-have-not-measured)
10. [Training procedure](#training-procedure)
11. [Reproducibility](#reproducibility)

## Harness and settings

- **Machine:** one NVIDIA RTX 3090 (24 GB) per job; runs between 2026-11-17 19:58 UTC and 2026-11-18
  13:51 UTC. The AWS tests quoted at the end of section 6 ran on the training machine (4 × NVIDIA L40S) on 2026-11-16.
- **Benchmarks:** EleutherAI lm-evaluation-harness 0.4.11, 0-shot, all test items, max length 2,048, `acc` for
  every task. Our model through our adapter (`evaluation/lm_eval_perceiver_ar.py`, `--model perceiver_ar`), the
  others through lm-eval's stock `hf` model (`evaluation/run_lm_eval_suite.py`).
- **Precision:** each model in its training precision. Ours, SmolLM2-135M, Gemma 3 270M, Granite 4.0 350M and
  LFM2.5-350M-Base in bf16; Pythia-160M (trained in fp16) in fp32, because bf16 costs it about 6 points of
  ARC-Easy (37.46 % against 43.64 %; EleutherAI publish 43.52 %).
- **Start of each input:** the way each model's own tokenizer starts a text. Gemma and LFM2.5 add a start token;
  SmolLM2, Pythia and Granite add nothing; ours starts with `<|endoftext|>`, as every training document did. With a
  leading `<|end_of_text|>` Granite 4.0 350M scored 2.12 bits per byte on a book at 19.8k tokens, against 1.07
  without it.
- **Excluded:** LFM2-350M (its released weights are post-trained for chat; LFM2.5-350M-Base replaces it);
  MobileLLM-R1-140M-base (gated; the evaluation machine had no access on 2026-11-18); state-space models such as
  Mamba-130M (without their CUDA kernels the 🤗 path is too slow at 61k tokens).
- **LFM2.5-350M-Base loading:** its tokenizer config names a Transformers 5 class; on Transformers 4.57 we load the
  same `tokenizer.json` as a `PreTrainedTokenizerFast` (it still adds `<|startoftext|>`). Its long-document numbers
  are identical under Transformers 5.23.

## Standard benchmarks

Accuracy in %, higher is better, ± standard error from the harness. In brackets: EleutherAI's own published
zero-shot results for Pythia-160M (step 143000, `evals/pythia-v1/pythia-160m` in their GitHub repository, quoted;
the remaining gaps of 1–2.5 points fit the change of harness version).

| Benchmark | Chance | **Ours** · 11.5B | SmolLM2-135M · 252B | SmolLM2-135M · ≈2T (final) | Pythia-160M · 300B | LFM2.5-350M-Base · 28T |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP, 67 sets | 50 | **80.56** ± 0.14 | 79.72 ± 0.14 | 80.15 ± 0.14 | 77.48 | 80.17 |
| ARC-Easy | 25 | 48.95 ± 1.03 | 61.62 ± 1.0 | 64.27 ± 0.98 | 43.64 ± 1.0 (43.52) | **70.03** ± 0.9 |
| ARC-Challenge | 25 | 21.2 ± 1.2 | 27.1 ± 1.3 | 28.0 ± 1.3 | 19.5 ± 1.2 (18.8) | **38.2** ± 1.4 |
| HellaSwag | 25 | 29.9 ± 0.5 | 32.6 ± 0.5 | 35.5 ± 0.5 | 28.4 ± 0.4 | **37.1** ± 0.5 |
| LAMBADA (OpenAI) | ≈0 | 33.2 ± 0.7 | 35.0 ± 0.7 | **43.0** ± 0.7 | 35.4 ± 0.7 (32.8) | 35.8 ± 0.7 |
| PIQA | 50 | 62.8 ± 1.1 | 65.9 ± 1.1 | **68.6** ± 1.1 | 62.3 ± 1.1 (62.7) | 66.6 ± 1.1 |
| SciQ | 25 | 77.0 ± 1.3 | 80.4 ± 1.3 | 84.1 ± 1.2 | 75.4 ± 1.4 (74.1) | **90.1** ± 0.9 |
| WinoGrande | 50 | 50.4 ± 1.4 | 50.9 ± 1.4 | 53.6 ± 1.4 | 51.2 ± 1.4 (53.1) | **55.6** ± 1.4 |
| **Mean of the 7 above** | 28.6 | 46.2 | 50.5 | 53.9 | 45.1 | **56.2** |
| WikiText-2 byte perplexity ↓ | — | 1.950 | 1.873 | **1.799** | 1.924 | 2.112 |
| WikiText-2 bits per byte ↓ | — | 0.964 | 0.905 | **0.847** | 0.944 | 1.078 |

The 7-task mean uses ARC-Easy from the 7-task run (49.0 %; 48.95 % in the leaderboard run). The previous generation
of our model (100M concept memory, 3.4B tokens, GPT-2 tokenizer) had a 7-task mean of 41.6 %.

## Tiny-ML Leaderboard

The [board](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard) takes three lm-eval 0-shot numbers
(BLiMP over its 67 sets, ARC-Easy `acc`, WikiText-2 byte perplexity) and ranks by
`quality × (1 + 0.5 · t)`, where quality is the mean of BLiMP, ARC-Easy and a WikiText score normalised over the
board, and t is a size bonus from 0 (the largest entry) to 1 (the smallest). `evaluation/tiny_ml_board.py`
reproduces the formula from the board's page (five entries match to two decimals) on the snapshot of 2026-11-14
(`evaluation/leaderboards/tiny_ml_board_20261114.json`). Board entries are the board's own runs (quoted); rows in
*italics* are our runs of reference models, each placed on its own (adding a row shifts the normalisation a little).

| Place | Model | Organisation | Size | Training tokens | BLiMP % ↑ | ARC-Easy % ↑ | WikiText-2 byte ppl ↓ | Quality | Board score |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Tiny Chick 10M V3 | kurcontko.dev | 9.6M | 24.6B | 77.76 | 59.60 | 2.335 | 77.8 | 87.7 |
| 2 | Tiny Chick 10M V2.1 | kurcontko.dev | 9.6M | 16.3B | 76.79 | 53.79 | 2.342 | 75.5 | 85.2 |
| 3 | Tiny Chick 10M V2 | kurcontko.dev | 9.6M | 16.3B | 76.99 | 50.21 | 2.364 | 74.3 | 83.8 |
| 4 | Slayer139 1.02 | Fabryka AI | 139.3M | 24.0B | 79.22 | 65.91 | 1.862 | 81.7 | 83.4 |
| (4) | *SmolLM2-135M, final (our run)* |  | 134.5M | ≈2T | 80.15 | 64.27 | 1.799 | 81.5 | 83.2 |
| 5 | Boris-2-Think | opencerebral | 143.7M | 200B | 82.09 | 63.93 | 1.967 | 81.7 | 83.2 |
| 6 | Slayer149-ARC | Fabryka AI | 149.3M | 31B | 78.38 | 66.58 | 1.907 | 81.5 | 82.9 |
| 7 | Tiny Chick 10M | kurcontko.dev | 9.6M | 8.3B | 75.12 | 48.53 | 2.287 | 73.3 | 82.7 |
| 8 | Slayer139 1.01 | Fabryka AI | 139.3M | 23.7B | 79.13 | 63.09 | 1.878 | 80.7 | 82.3 |
| (9) | *SmolLM2-135M, 252B checkpoint (our run)* |  | 134.5M | 252B | 79.72 | 61.62 | 1.873 | 80.4 | 82.1 |
| 9 | Glint-2 | glintresearch | 1.1M | ≈300B | 74.54 | 36.20 | 3.127 | 67.2 | 81.7 |
| 10 | Boris-2 | opencerebral | 143.7M | 200B | 81.13 | 59.76 | 1.895 | 80.2 | 81.7 |
| (11) | *SmolLM2-135M, 503B checkpoint (our run)* |  | 134.5M | 503B | 79.62 | 59.97 | 1.861 | 79.9 | 81.6 |
| 11 | JugnuLM-110M-R2+ | altslate | 110.0M | 25B | 82.52 | 55.13 | 1.873 | 79.2 | 81.5 |
| 12 | GPT-X2-125M | axiomiclabs | 125.0M | 75B | 81.28 | 57.07 | 1.860 | 79.5 | 81.4 |
| 13 | Haidass-143M-v1 | DALab | 143.0M | 100B | 79.05 | 60.23 | 1.889 | 79.7 | 81.2 |
| 14 | Slayer149 Balanced ARC-E fine-tune | Fabryka AI | 149.3M | — | 79.98 | 59.34 | 1.897 | 79.7 | 81.0 |
| 15 | Slayer139 base | Fabryka AI | 139.3M | 23.7B | 79.41 | 57.58 | 1.875 | 78.9 | 80.5 |
| 16 | JugnuLM-53M | altslate | 53.0M | 12B | 78.14 | 51.43 | 2.040 | 76.0 | 80.5 |
| 17 | Supra2.1-Ultra-Preview-001 | supralabs | 230.0M | 40.4B | 82.24 | 57.11 | 1.930 | 79.6 | 79.6 |
| 18 | Supra-50M-Instruct | supralabs | 51.8M | 20B | 76.30 | 52.20 | 2.560 | 74.3 | 78.7 |
| 19 | GoLLeM-v5 128M | Fabryka AI | 122.8M | 24.9B | 79.09 | 53.24 | 2.254 | 76.3 | 78.2 |
| 20 | Glint-1.3 (merged) | glintresearch | 1.0M | 100B | 68.70 | 32.50 | 3.080 | 64.1 | 78.2 |
| 21 | Ivme-Conversate-v3-Base | ivmelabs | 24.8M | ≈15B | 78.49 | 39.02 | 2.136 | 71.7 | 78.1 |
| 22 | Supra-50M-Base | supralabs | 51.8M | 20B | 76.30 | 46.00 | 2.040 | 73.5 | 78.0 |
| 23 | GPT-S-5M | axiomiclabs | 5.2M | 25B | 72.27 | 35.69 | 2.570 | 67.4 | 77.7 |
| 24 | **150M concept memory (ours)** | Meridian21 Lab | 141.5M | 11.5B | 80.56 | 48.95 | 1.950 | 76.2 | 77.7 |
| 25 | Glint-2.3 | glintresearch | 1.7M | 10B | 67.30 | 36.00 | 3.051 | 64.8 | 77.6 |
| 26 | Zedev-134M-Fudas | meridianmrm | 134.0M | 20.7b | 81.69 | 46.38 | 1.880 | 76.0 | 77.6 |
| 27 | Hydrion-v1-Base | opengcm | 114.1M | ≈2.5B | 80.08 | 47.26 | 2.040 | 75.2 | 77.4 |
| 28 | GoLLeM-v5 64M | Fabryka AI | 62.9M | 24.9B | 76.16 | 48.19 | 2.347 | 73.4 | 77.2 |
| 29 | Ivme-Conversate-v2-Base | ivmelabs | 23.9M | ≈12.85B | 75.09 | 39.98 | 2.225 | 70.6 | 77.1 |
| 30 | GoLLeM-v5 64M (v1 recipe) | Fabryka AI | 62.9M | 13.1B | 75.83 | 47.94 | 2.372 | 73.1 | 77.0 |
| 31 | Pollock 1.5 Mini LM 128M | Fabryka AI | 127.4M | 21.50B | 78.50 | 47.73 | 1.944 | 75.1 | 76.9 |
| 32 | Glint-2.3-Mini | glintresearch | 1.1M | 10B | 62.27 | 36.20 | 3.122 | 63.1 | 76.8 |
| 33 | Qielbas Tiny 6M | labqoat | 6.1M | 3B | 73.25 | 36.24 | 3.069 | 66.8 | 76.6 |
| 34 | Hummingbird-V2 | juinron | 9.6M | 10B | 72.90 | 38.85 | 2.955 | 67.8 | 76.5 |
| 35 | TextModel-v1 | benchlabs | 122.7M | 1.64B | 80.31 | 49.71 | 2.628 | 74.6 | 76.5 |
| 36 | GoLLeM-v5 32M | Fabryka AI | 31.6M | 16B | 73.48 | 44.44 | 2.539 | 70.8 | 76.5 |
| 37 | Gros-Michel-90m-Base-v2 | finnianx | 95.0M | 9B | 80.20 | 43.18 | 2.080 | 73.8 | 76.4 |
| 38 | Michel-Nano-v2 | finnianx | 9.9M | 6.5B | 72.52 | 35.90 | 2.460 | 67.8 | 76.4 |
| 39 | MicroSupra-1k | supralabs | 0.0M | — | 58.61 | 26.39 | 11.270 | 50.9 | 76.3 |
| 40 | min-spark | minimalabs | 5.8M | 10B | 69.19 | 37.08 | 2.775 | 66.4 | 76.3 |
| 41 | ObsidianSmall-Base | dreamw | 8.7M | 4B | 71.89 | 33.29 | 2.467 | 66.7 | 75.5 |
| 42 | Gros-Michel-90m-Base | finnianx | 91.1M | 6.5B | 78.35 | 41.50 | 2.070 | 72.6 | 75.4 |
| 43 | GoLLeM-v5 16M | Fabryka AI | 17.4M | 16B | 70.08 | 40.91 | 2.675 | 68.2 | 75.3 |
| 44 | Cagliari-114M | meridianmrm | 114.0M | 5.24B | 80.22 | 38.30 | 1.960 | 72.5 | 74.6 |
| (44) | *Pythia-160M, final (our run, fp32; over the 150M cap)* |  | 162.3M | 300B | 77.48 | 43.64 | 1.924 | 73.5 | 74.5 |
| 45 | KeyLM-75M-Instruct | minimalabs | 75.0M | ≈18B | 76.02 | 39.10 | 2.170 | 70.8 | 74.0 |

The first 45 places of 101 shown; the full snapshot is in the file above.

By quality alone (no size bonus) our model is 14th of 101; on grammar alone, 8th. Of the 55 entries that report
15B training tokens or fewer (ours included), three place above it. ARC-Easy is the test that moves the rank most:
at 51.4 % the model would place 19th, at 60 % 12th.

## Data efficiency

| Test | **Ours** at 11.5B | Pythia-160M at 10.5B | at 12.6B | at 300B (final) | SmolLM2-135M at 252B | at 503B | at ≈2T (final) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grammar (BLiMP) % ↑ | **80.56** | 77.94 | 78.28 | 77.48 | 79.72 | 79.62 | 80.15 |
| Science quiz (ARC-Easy) % ↑ | 48.95 | 42.59 | 42.80 | 43.64 | 61.62 | — | **64.27** |
| WikiText-2 byte perplexity ↓ | 1.950 | 2.044 | 2.024 | 1.924 | 1.873 | — | **1.799** |
| Mean of 7 core tasks % ↑ | 46.2 | 42.1 | 43.2 | 45.1 | 50.5 | — | **53.9** |

Pythia-160M checkpoints scored (fp32): steps 1,000, 2,000, 4,000, 5,000, 6,000, 10,000, 20,000 and 143,000, at
2,097,152 tokens per step (2.1B to 300B tokens). SmolLM2-135M checkpoints: `HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints`
branches `step-240000` and `step-480000` (1,048,576 tokens per step) and the final model.

- **Same budget:** ahead of Pythia-160M at 10.5–12.6B tokens on every test.
- **Tokens to match** (log-linear between Pythia's checkpoints): not reached by 300B tokens for grammar (its best
  79.23 % at 42B) or the science quiz (its best 45.12 % at 42B); 36.2B tokens (3.1× ours) for WikiText, which is
  Wikipedia text, part of Pythia's training data (the Pile).
- **What it shows and what it does not:** the model as trained (architecture and data together) is more
  data-efficient than Pythia-160M on every test and at least as efficient as SmolLM2-135M on grammar; on knowledge,
  SmolLM2-135M's first public checkpoint is already far ahead. Most of the gap to Pythia is probably data (Pythia read
  the Pile, assembled in 2020; we read 2024–2025 filtered web and educational text). Grammar also levels off early
  (Pythia-160M wanders between 77.5 % and 79.2 % from 10B to 300B tokens), so it is weak evidence alone. A standard
  transformer of the same size on our exact data and budget will separate the architecture's share; it is queued.

## Long documents across models

`evaluation/long_context_compare.py` (a copy is in `long_context/`), one RTX 3090, 2026-11-18. **Documents:** held-out
LongDocs-lite validation documents from five Common Pile sources (Gutenberg books, arXiv papers, DOAB open-access
books, US GPO publications, UK Hansard debates), up to 16 per source at 64k tokens, 55 in all. **Target:** the last
≈16 KB of each span (its last 4,096 tokens in our tokenizer), 0.94 MB in all; the context before it is cut to each
budget, counted in our tokens, so every model reads the same characters. **Unit:** bits per byte of the target
(nats ÷ ln 2 ÷ UTF-8 bytes). A model that cannot take a budget is scored with sliding windows of its own maximum
length ("limit").

| Model | Size | Reads at most | 1k | 4k | 16k | 28k | 61k | Gain, 1k → most | Documents better with more | Another document's far past |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **150M concept memory** | 141.5M | 65,536 | 0.846 | 0.828 | 0.816 | 0.808 | **0.795** | **0.051** [0.037, 0.067] | 96 % | 0.832 |
| SmolLM2-135M, 252B | 134.5M | 2,048 | 0.933 (limit) | — | — | — | — | — | — | 0.932 |
| SmolLM2-135M, final | 134.5M | 8,192 | 0.851 | 0.835 | 0.832 (limit) | — | — | 0.018 [0.014, 0.024] | 95 % | 0.836 |
| Pythia-160M (fp32) | 162.3M | 2,048 | 0.942 (limit) | — | — | — | — | — | — | 0.941 |
| Gemma 3 270M | 268.1M | 32,768 | 0.843 | 0.825 | 0.812 | 0.805 | 0.805 (limit) | 0.038 [0.029, 0.049] | 96 % | 0.827 |
| Granite 4.0 350M base | 352.4M | 32,768 | **0.817** | **0.797** | **0.783** | **0.776** | 0.776 (limit) | 0.041 [0.033, 0.051] | 100 % | **0.798** |
| LFM2.5-350M-Base | 354.5M | 32,768 | 0.893 | 0.879 | 0.912 | 0.928 | 0.928 (limit) | −0.035 [−0.051, −0.020] | 22 % | 0.933 |

Head to head, each model with the most of the document it reads (LFM2.5 at its best, 4k) against ours with the
whole span; + = ours better, 95 % bootstrap interval over documents:

| Against | Difference, bits per byte | Documents where ours is better |
|---|---:|---:|
| SmolLM2-135M, 252B | **+0.138** [+0.114, +0.164] | 98 % |
| SmolLM2-135M, final | **+0.038** [+0.020, +0.058] | 75 % |
| Pythia-160M | **+0.147** [+0.125, +0.171] | 96 % |
| Gemma 3 270M | +0.010 [−0.006, +0.025]: a tie | 55 % |
| LFM2.5-350M-Base (at its best, 4k) | **+0.084** [+0.060, +0.108] | 89 % |
| Granite 4.0 350M base | −0.019 [−0.034, −0.003]: Granite better | 29 % |

- **Scale.** On the same targets, each compressor given the rest of the document first: gzip -9 2.49 bits per byte,
  xz -9e 1.97, bzip2 -9 1.77; plain text 8.
- **LFM2.5-350M-Base** predicts best with 4k of context; with its full 32k it is worse than with 4k on 93 % of
  documents and worse than with 1k on 78 %. The result is the same in fp32 and bf16, with SDPA and eager attention,
  and under Transformers 4.57 and 5.23 (checked on one book). Its card states a 32,768-token context; we report what
  we measured.
- **The swapped past** helps no model. Ours, Gemma 3 270M, Granite 4.0 350M and SmolLM2-135M score within 0.004 of
  their own 4k cut, so every model that reads far is reading the content; LFM2.5-350M-Base scores like its own long
  input (0.933), worse than its 4k cut.
- **Contamination:** our model never trained on these documents, but did on other documents of the same five
  sources; the reference models may have trained on these very documents (Gutenberg and arXiv are in most
  web-scale corpora, and Pythia's Pile includes both). The "gain" column compares each model with itself, so it is
  the measure least affected by this.

## Our own long-context tests

The same 64k documents in our own tokens, with switches no other model has (`long_context/long_context_use.py`,
one RTX 3090, 2026-11-17 19:58–20:35 UTC). Always the same target, the **last 4,096 tokens** of a 16k, 64k or 128k span.
Loss in nats per token (lower is better; nats per token depend on the tokenizer, so these compare the model only
with itself); 95 % intervals by bootstrap over documents.

1. **Context cut:** predict the target after 1k, 4k, 16k … or all of the document before it.
2. **Swapped far past:** the document's own nearest 4k kept, everything older from a different document of the same
   source.
3. **Memory off:** the memory-reading layer limited to its recent window (the last 512–1,024 tokens), or only the
   coarse summaries removed (word-for-word reads kept).

**At 64k** (55 documents):

| Context before the target | Loss | Worse than full context by | Documents worse |
|---|---:|---:|---:|
| **the whole document before it (61,439 tokens)** | **2.293** (0.791 bits per byte) | — | — |
| last 16,384 tokens only | 2.354 | +0.061 [0.036, 0.088] | 84 % |
| last 4,096 tokens only | 2.390 | +0.097 [0.065, 0.133] | 91 % |
| last 1,024 tokens only | 2.441 | +0.148 [0.109, 0.193] | 96 % |
| own last 4,096 + **another document's** 57k before them | 2.399 | +0.106 [0.078, 0.139] | 100 % |
| whole document, **summaries off** | 2.361 | +0.068 [0.062, 0.073] | 100 % |
| whole document, **memory off** | 2.580 | +0.287 [0.220, 0.364] | 100 % |

**At 16k** (120 documents, 12,287 tokens of full context): full 2.126; cut to 4k +0.042 [0.034, 0.050]; cut to 1k
+0.094 [0.080, 0.110] (99 % of documents worse); another document's far past +0.052 [0.043, 0.060]; summaries off
+0.023; memory off +0.271.

**At 128k, twice the training length** (15 documents: 4 books, 8 DOAB, 2 US GPO, 1 arXiv):

| Context before the last 4,096 tokens | Loss | Against the whole document | Documents worse than whole |
|---|---:|---:|---:|
| the whole document before it (126,975 tokens) | 2.451 | — | — |
| last 61,440 tokens only | 2.391 | −0.060 [−0.077, −0.045]: better | 0 % |
| last 16,384 tokens only | 2.409 | −0.042: better | 13 % |
| last 4,096 tokens only | 2.441 | −0.010 [−0.047, +0.031]: a tie | 40 % |
| own last 4,096 + another document's 123k before them | 2.533 | +0.082 [0.050, 0.115] | 93 % (of 14) |
| whole document, memory off | 2.645 | +0.194 | 100 % |

- **The far past is used:** the last 45k tokens of a 64k document still lower the loss on its final pages, on 84 %
  of documents; from 1k of context to all of it, 0.15 nats per token (about 7 % fewer bits per byte).
- **It is the content:** another document's far past scores like a 4k cut, worse than the real past on 55 of 55.
- **Both halves of the memory work:** removing only the summaries costs 0.068 at 64k, three times its cost at 16k
  (0.023); most of the gain comes from the word-for-word reads.
- **The gain depends on the text:** largest where documents repeat their own terms (US government documents 2.021 →
  1.642 nats per token from 1k to full context, arXiv 1.221 → 0.930), small for debates (2.593 → 2.533) and books
  (2.987 → 2.916).
- **Past 64k:** the summaries' length scale keeps the read from collapsing (memory off is still far worse), but text
  beyond 64k does not improve predictions, and a long wrong past distracts more than at 64k.
- **"Memory off" overstates the memory's value:** the model never trained without it; the context cut is the cleaner
  measure. Against the bar set before training (switching the far read off should cost ≥ 0.10 nats on positions past
  2k at 16k): 0.255, met.

**From the AWS tests (2026-11-16):** the 64k training stage lowered the loss on tokens 32k–64k of 64k documents by
0.11–0.16 nats per token on all seven long-document sources, against the model before that stage. Loss on positions
past 2k, in bits per byte, was 0.785 at 4k, 0.750 at 16k, 0.737 at 32k, 0.717 at 64k and 0.663 at 128k, but each
length used different documents; at 128k one arXiv paper drives the drop (without it, 0.735 → 0.733). The same-document
test above is the one to trust.

## RULER

13 tasks, 100 prompts per task and length, greedy cached decoding, each task's own answer length (lm-eval's RULER
configs, up to 128 tokens), every reply saved (`ruler/*_answers.jsonl`). RTX 3090 cards, 2026-11-18 12:40–13:51 UTC (4k and 8k, 16k on one card each; 32k split by task over three cards).
Score = share of the reference answers found in the reply (higher is better). **Not zero-shot:** training stages 2
and 3 included planted-fact lookups of a similar shape (different wording).

| RULER task | What it asks | 4k | 8k | 16k | 32k |
|---|---|---:|---:|---:|---:|
| single needle, noise filler | one number hidden in repeated filler | 1.00 | 1.00 | 1.00 | 1.00 |
| single needle, essays | one number hidden in real essays | 1.00 | 1.00 | 1.00 | 0.98 |
| single needle, UUID value | one 32-character ID hidden in essays | 0.29 | 0.27 | 0.07 | 0.07 |
| multi-key 1 | one number among distractor keys | 0.99 | 0.90 | 0.91 | 0.77 |
| multi-key 2 | the haystack is all similar needles | 0.92 | 0.84 | 0.76 | 0.41 |
| multi-key 3 | as 2, with UUID keys and values | 0.28 | 0.11 | 0.06 | 0.04 |
| multi-value | all values of one key | 0.31 | 0.34 | 0.24 | 0.24 |
| multi-query | several keys asked at once | 0.40 | 0.32 | 0.11 | 0.09 |
| variable tracking | follow a chain of assignments | 0.01 | 0.01 | 0.01 | 0.01 |
| common words | list the most common words | 0.52 | 0.42 | 0.27 | 0.13 |
| frequent words | list the most frequent words | 0.15 | 0.16 | 0.02 | 0.06 |
| QA (SQuAD) | answer from one of many paragraphs | 0.33 | 0.20 | 0.21 | 0.20 |
| QA (HotpotQA) | two-hop answer from many paragraphs | 0.15 | 0.15 | 0.19 | 0.15 |
| **Mean of 13** | | **0.49** | **0.44** | **0.37** | **0.32** |

**Why an earlier run scored lower.** Our first RULER run of this model (AWS, 2026-11-16) cut every answer at 32
tokens. That cap held every answer for our earlier models' GPT-2 tokenizer but not for SmolLM2's,
which writes every digit as its own token (one UUID is 35 tokens, four 7-digit numbers 36, ten numbered words 41).
The capped run's means were 0.43 / 0.40 / 0.34 / 0.30 at 4k / 8k / 16k / 32k; its UUID needle scored 0.02 at 4k and
the UUID multi-key 0.00. The table above replaces it. Tasks the cap could not affect (short answers) moved by up to ±0.09 between the two runs, which differ in machine and in the prompts drawn: that is the run-to-run noise at 100 prompts per task. The tasks with long answers moved more: the UUID needle +0.27 at 4k, the UUID multi-key +0.28, common words +0.08 to +0.18 at every length. Previous generation (100M, GPT-2 tokenizer, no lookup
training, capped run, unaffected): mean 0.185 at 4k and 0.136 at 16k; the essay needle 0.58 at 4k and 0.14 at 16k.

## Text samples

`evaluation/sample_text.py` (a copy is in `samples/`), one RTX 3090, 2026-11-18 12:49 UTC; prompts start with
`<|endoftext|>` as in training. Nucleus sampling: temperature 0.7, top-p 0.9, repetition penalty 1.1 over prompt
and reply, seeds 0, 1 and 2. Beam search: 4 beams, no repeated 3-gram, best by mean log-probability. 60 new tokens.
All outputs, unedited: `samples/samples_ours.json`. The card shows seed 0 for each prompt.

## What we have not measured

- **A full-attention model of the same size on the same data.** Without it, no result separates the memory from the
  data. In the previous generation (100M, 3.4B tokens) this read kept 99 % of the gap between a window-only model and
  a full-attention model at 16k; that control had no position code in its reading layer, and a fairer one is
  queued.
- **Long-document benchmarks with real questions** (LongBench, BABILong) and needle tests past 32k.
- **A second seed.** Every number here is seed 42; differences of a point or two are not established.
- **Energy use.**

## Training procedure

| | Stage 1 (main) | Stage 2 (decay) | Stage 3 (long) |
|---|---|---|---|
| Row length | 4,096 | 16,384 | 65,536 |
| Steps × rows per step | 15,259 × 128 | 5,722 × 32 | 954 × 8 |
| Tokens per step | 524,288 | 524,288 | 524,288 |
| Learning rate | Muon peak 0.01, constant after 1 % warm-up | fresh optimizer, 3 % warm-up, 1 − √ decay to 0.1 × | a tenth of the peak, decaying |
| Held-out loss (nats per token, lower is better) | 2.732 → 2.134 (4k web rows) | 2.295 → 1.996 (16k long-document rows) | 1.960 → 1.936 (64k rows) |
| Speed | 105.6k tokens/s | 69.7k tokens/s | 21.6k tokens/s (activations recomputed) |
| Wall time | 21.2 h | 12.2 h | 6.5 h (+ ≈1 h lost to a spot interruption) |

Muon updates the weight matrices; norms, embeddings and other vector parameters use AdamW (betas 0.9 / 0.95, weight
decay 0.1). Gradient clipping at 1.0, bf16, flex attention. A paid two-arm test (2,000 steps each, $15) chose Muon
over AdamW before the run: held-out loss 2.283 against 2.323 at step 2,000 (lower is better; the pass mark was 0.03).

**Long documents** in training are whole documents of at least 16,384 tokens; the five Common Pile long-document
sources keep a held-out set (the LongDocs-lite validation documents), which the long-context tests read. **Lookups:**
15 % ask for a fact that was never stated, 15 % of facts are revised later.

## Reproducibility

- **W&B (public project `ksopyla/MrCogito`):**
  [stage 1](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_perceiver_H896L3g1s9N128_20261115_052641_e37a_m120t_trunk) ·
  [stage 2](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_perceiver_H896L3g1s9N128_20261116_024010_e37a_m120t_decay) ·
  [stage 3](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_perceiver_H896L3g1s9N128_20261116_155301_e37a_m120t_long) ·
  report [150M concept memory vs earlier 100M runs](https://wandb.ai/ksopyla/MrCogito/reports/150M-concept-memory-vs-earlier-100M-runs--VmlldzoxODMxMjMwMw)
- **In this repository:** `eval_results/` (full lm-eval outputs for every model and checkpoint above),
  `long_context/` (both long-document scripts, per-document results, the summarisers `summarise.py` and
  `summarise_compare.py`), `ruler/` (scores and every reply), `samples/` (the sampling script and its outputs),
  `figures/make_figures.py`.
- **Code** at [commit `1fddc505`](https://github.com/ksopyla/MrCogito/tree/1fddc505): experiment spec
  `docs/experiments_specs/ahead/E38_long_budget_mix.md` (hypothesis and the pass marks set before training); data
  recipe `data/mix_recipes/m21_mix_v2.json` and builders `scripts/build_m21_mix.py`, `scripts/build_reasoning_rows.py`;
  training `scripts/run_e33n.py` (size `m120t`, arm `e37a`) and job `infra/aws/jobs/e38_board.sh`; architecture
  `nn/perceiver_ar_lm.py`, `nn/concept_memory.py`. The fixed RULER runner and the sampling script are newer than
  that commit; this repository has them as `ruler/run_lsa_suite.py` and `samples/sample_text.py`.
- **Intermediate checkpoints** of this model are not published yet.
