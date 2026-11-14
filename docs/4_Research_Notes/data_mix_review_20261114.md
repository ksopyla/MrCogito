# Data mixes and training protocols: 2026 open models, our past mixes, and the E38 mix (2026-11-14)

Written for the author's request of 2026-11-14: "Review the recent data mix and training protocols for opensource
SOTA 2026 models: kimi, qwen, mistral etc. Analyse our past data mixes … Try to reuse it, modify, and tailor it to
modern standards. Think about data serving in AWS; I don't want to be a bottleneck. … try to find some multi-hop or
reasoning data in Hugging Face". Three reads, each done by a separate agent on 2026-11-14 and kept below as
delivered (claims marked "unverified" were not read at a primary source). The mix they led to is specified in
`docs/experiments_specs/ahead/E38_long_budget_mix.md`.

## What we took from it
1. **One shape repeats in every published recipe:** a web-heavy main phase, a tail richer in code, math and curated
   or synthetic text while the learning rate decays, then a long-context stage that costs under 1 % of the tokens
   and mixes 34–40 % long documents with the previous mix (SmolLM2, SmolLM3, Olmo 3, Nemotron 3 Nano, Qwen3). The
   2026 flagships (DeepSeek-V4, Kimi K3) grow the context inside pretraining instead (4k → 16k → 64k → 1M).
2. **Nobody trains a small model at 16k–64k on a 10–30B-token budget.** Our long stages are 30 % of the tokens, not
   1 %: that part of E38 is ours, because far context is what the architecture is for.
3. **Instruction and reasoning text now sits inside pretraining** (Nemotron: SFT-style data throughout; Qwen3:
   synthetic QA and textbooks; IMU-1 and Olmo 3: a final instruction-rich stage). E38 puts 4 % chat and 7 %
   reasoning text into its 16k stage rather than leaving it all to a separate tuning stage, which in E37 cost 0.10
   nats of plain-text prediction.
4. **Our past mixes never used Nemotron data.** Two recipes name Nemotron sources (`e18_main_stage1_v1`, thirteen
   subsets; `nemotron_nano_v3_inspired_2k`) but neither was built: the big run was blocked and the packages are
   gated, while the AWS nodes carry no Hugging Face token. The mixes that were built are SmolLM3-style (FineWeb-Edu,
   DCLM, Stack-Edu, FineMath, FinePDFs, reasoning traces), tokenized with SmolLM2, Gemma or SmolLM3 tokenizers; the
   100M runs of E33n–E37 used a third party's GPT-2 tokenization of FineWeb-Edu.
5. **What E38 reuses:** the SmolLM2 tokenizer; the SmolLM3-style proportions for web, code and math; the open
   Nemotron "Specialized" subsets (Wiki-Rewrite, Math-Textbooks, RQA, InfiniByte-Reasoning; CC BY 4.0, no gate) in
   place of the gated ones; LongDocs-lite's five Common Pile sources, now joined by long PDFs and code packed per
   repository.
6. **Data serving:** every E38 source is public and ungated, so one CPU spot node downloads, tokenizes and writes
   the shards to S3 with no token, no laptop and no home server; training nodes copy the shards from S3 at boot
   (about 27 GB, a few minutes). The gated Nemotron and Stack v2 packages would need a token on the node, which
   is the author's to place, so they are left out of v1.
7. **Multi-hop and reasoning data:** the classic sets are small (2WikiMultihopQA ≈ 0.2B tokens) and three of them
   are in our tests (HotpotQA and SQuAD inside RULER, MuSiQue, BABILong). E38 uses 2WikiMultihopQA's training split
   packed into "libraries", invented-fact lookups in real books, and Nemotron's InfiniByte-Reasoning text.

## 1. Training recipes of open models (survey)

Published recipes share one shape: a web-heavy main phase, a tail richer in code, math and curated data during learning-rate decay, then a long-context stage costing under 1% of tokens. The 2026 flagships (DeepSeek-V4, Kimi K3) instead grow context inside pretraining. Nothing published trains at 16k–64k on a 10–30B-token budget. Rows were read at the primary source unless marked "unverified"; "@4k" means sequence length.

### Kimi (Moonshot)

| Model | Data, tokens | Curriculum, context | Tokenizer, optimizer | Post-training |
|---|---|---|---|---|
| K2 | 15.5T; web, code, math, knowledge; proportions not disclosed; knowledge text LLM-rephrased ≤2×, math rewritten as "learning notes", inside pretraining | 4k; anneal 400B @4k + 60B @32k; YaRN to 128k | 160k vocab (per K3 report); MuonClip; constant LR 10T, cosine 5.5T | SFT; synthetic tool-use trajectories; RL with verifiable + self-critique rewards |
| K3 (Jul 2026) | Tokens, proportions not disclosed; same domains + vision | 8k→64k during pretraining, 256k→1M in cooldown; per-stage tokens not disclosed | 160k; per-head Muon; cosine | SFT; 9 RL experts merged by multi-teacher on-policy distillation |

K3.1 Pro (weights Nov 2026): no training details found; unverified, secondary coverage only.

### Qwen

| Model | Data, tokens | Curriculum, context | Tokenizer | Post-training |
|---|---|---|---|---|
| Qwen3 | 36T, 119 languages; PDF text via Qwen2.5-VL; synthetic textbooks, QA, instructions, code inside pretraining; proportions not disclosed | S1 >30T @4k; S2 ~5T higher-quality STEM/code/reasoning/synthetic @4k; S3 "hundreds of billions" @32k (75% of text 16–32k long, 25% 4–16k) | BBPE 151,669 | Long-CoT cold start, GRPO reasoning RL (3,995 query-verifier pairs), thinking-mode fusion, general RL. 0.6B–14B: distilled from Qwen3-32B/235B |
| Qwen3.5-0.8B (Feb 2026) | Not disclosed | 262,144 native context | Not disclosed | Card says only "Pre-training & Post-training" |

Qwen3.6/3.8: no technical report found (unverified).

### DeepSeek

| Model | Data, tokens | Curriculum, context | Tokenizer, optimizer | Post-training |
|---|---|---|---|---|
| V3 | 14.8T; more math/code than V2; proportions not disclosed | 4k; then 1,000 steps @32k + 1,000 @128k (YaRN) | BBPE 128k; AdamW | SFT, R1 distillation, GRPO (section not read: unverified) |
| V4 (Jun 2026) | Flash 32T, Pro 33T; math, code, web, long documents (papers, technical reports prioritised); proportions not disclosed | 4k→16k→64k→1M in pretraining; per-stage tokens not disclosed; sample-level attention masking | 128k; Muon (AdamW for embeddings, head, norms); cosine | Per-domain experts (SFT + GRPO), then on-policy distillation from >10 teachers |

### Mistral

| Model | Pretraining | Post-training |
|---|---|---|
| Magistral Medium / Small | Not disclosed for the Medium 3 and Small 3 bases | Medium: RL only (GRPO without KL), 38k math + 35k code problems. Small: SFT on Medium's traces (10% general instruction), 4 epochs, then RL |

Small/Medium 3.x, Devstral, Large 4: no pretraining details found (unverified).

### Other flagships

| Model | Data, tokens | Curriculum, context | Post-training |
|---|---|---|---|
| GLM-4.5 | 23T; general web, then upsampled code/math/science; proportions not disclosed; Muon | 4k; mid-training: repo-level code 4k→32k, synthetic reasoning, long docs + agent trajectories 32k→128k | Experts (cold-start SFT + RL), then self-distillation |
| Olmo 3 | 5.93T: web 76.1%, science PDFs 13.6%, code 6.89%, math 2.56%, arXiv 0.86% | 8k; midtrain 100B (my sums: web 27.5%, code 20%, math 19%, synthetic QA 14%, thinking traces 8%, instruction 6%, PDFs 5%); long-context 50B/100B to 65k: 34% long PDFs + 66% midtraining mix | SFT, DPO, RL (counts not read) |
| Nemotron 3 Nano | 25T, 15 categories, SFT-style data inside pretraining; percentages only in Figure 3 | 8k; 23.5T diverse, then 1.5T high-quality (switch at 94%); long-context 121B at 512k + 4k: 20% long-document QA, 1% synthetic retrieval, 79% phase-2 data | SFT >18M samples; GRPO; reasoning stripped from 10% of samples |
| Ripple-65B Preview (Nov 2026) | 12T, mostly English + code, "minimal synthetic data"; mix not disclosed | 8k; midtrain 200B @64k (synthetic data added here); long-context 6.5B @2M | SFT 183B @128k; long-context SFT 13.5B @2M; RL 700 steps |
| gpt-oss | "Trillions of tokens", text-only; mix not disclosed | Not disclosed; YaRN to 131k; o200k_harmony, 201,088 vocab | CoT RL "similar to o3" |
| Gemma 3 / 4 | Gemma 3: 14T/12T/4T/2T (27B/12B/4B/1B), 270M: 6T; mix not disclosed. Gemma 4: tokens not disclosed | 32k pretraining; 128k at the end via RoPE rescale; 262k SentencePiece | Distillation + RL; Gemma 4 adds thinking mode |

MiniMax, Llama: not checked.

### Small: HuggingFaceTB

| Model | Data mix | Context, tokenizer | Post-training |
|---|---|---|---|
| SmolLM2-1.7B | 11T. S1 0–6T: web 90% (implied; FineWeb-Edu:DCLM 60:40), code 10%. S2 6–8T: web 75, code 20, math 5. S3 8–10T: math ~10, Stack-Edu replaces StarCoder. S4 10–11T (decay): web 58, Stack-Edu 24, math 14, Cosmopedia v2 4 | 2k; 8k extension in the last 75B: 40% long docs (Dolma books 20, DCLM 10, FineWeb-Edu 10) + 60% S4 mix; vocab 49,152; WSD | SmolTalk SFT, 2 epochs @8k |
| SmolLM2-135M/360M | 2T / 4T, single stage; edu-filtered DCLM plus Stack-Edu, FineMath, Cosmopedia from the start; percentages not in text | 49,152; WSD, 20% decay | SmolTalk |
| SmolLM3-3B | 11.2T. Web/code/math: S1 0–8T 85/12/3; S2 8–10T 75/15/10; S3 10–11.1T (decay) 63/24/13 | 4k; +50B @32k, +50B @64k (math, code, reasoning upsampled); YaRN 128k; Llama 3.2 tokenizer (author comment) | Reasoning mid-training 35B × 4 epochs; SFT 1.8B × 4 epochs (1B no-think, 0.8B think); APO; merge |

### Small: others

| Model | Data mix, tokens | Context, stages | Post-training |
|---|---|---|---|
| MobileLLM-R1 (140M–950M) | 4.2T resampled from ~2T. P1: FineWeb-Edu 63.75, StarCoder 10.66, OpenWebMath 6.93, arXiv 6.36, Wiki 5.03, StackExchange 5.03. P2: FineWeb-Edu 54.83, OpenWebMath 23.33, FineMath 8.01, Natural Reasoning 3.18 | 2k; mid-training 2 × 100B @4k, distilled from Llama-3.1-8B | Tulu 3 SFT (866k), then reasoning SFT ~6.2M samples @32k |
| MobileLLM-Pro (1B) | 1.4T: FineWeb-Edu 89.75, StarCoder 4.66, OpenWebMath 1.92, arXiv 1.35 | 2k; 20B of positional distillation to 128k with no long data | 7.64M-sample SFT |
| IMU-1 (430M) | 72B; pools of DCLM-edu, Python-edu, FineWiki, FinePDFs-edu, FineMath, MegaMath, Cosmopedia | 29B @768; 28B @896 (decay; adds arXiv, SYNTH); 14B @1,152 instruction-formatted + synthetic reasoning; NorMuon, WSD | None |
| LFM2 (350M–2.6B) | 10T: 75% English, 20% multilingual, 5% code; sources not disclosed. LFM2.5-230M: 19T | 4k, then 1T higher-quality @32k; BBPE 65,536; distilled from LFM1-7B | SFT 5.39M samples; alignment; merging |
| OPEN-1B (Sep 2026) | 400B, fixed mix: DCLM 66.7, FineWeb-Edu 13.3, Stack v2 12.0, Proof-Pile-2 8.0 | 4k, no stages; vocab 128,256 | None |
| nanochat d24 | FineWeb-Edu ~8.8B; ClimbMix since 4 Mar 2026 | 2k; Muon + AdamW | SmolTalk SFT (details not read) |

Gemma 3 270M: only the 6T figure. Granite 4.0 350M: pretraining not disclosed.

### Nemotron datasets

| Dataset | Content, tokens | License |
|---|---|---|
| Pre-Training-Dataset-v1 (CC-v2, Code-v1, SFT-v1) | 6,585.8B: English crawl 3,359.8; synthetic rephrase 1,257.3; Diverse QA 692.9; translated QA 558.2; math 206.2; synthetic code 174.9; SFT-style math 190.6, code 58.5, MMLU 81.6, general 5.7 | NVIDIA Data Agreement for Model Training, gated |
| Nemotron-CC-Math-v1 | 3+: 133B; 4+: 52B; MIND rewrite: 73B | NVIDIA Open Data License Agreement |
| Nemotron-CC-v2.1 | 2,544.8B: organic 26.0 / 16.9 / 53.5 by quality tier; rephrases 93.5 + 2,122.8; translated-to-English 39.6 + 26.8 (+157.8 rephrased); STEM QA 8.0 | NVIDIA Data Access Agreement (training only, gated) |
| CC-Code-v1; Pretraining-Code-v2 | 427.9B code pages; ~340B GitHub (metadata) plus synthetic QA, review, rewriting, transpilation | Same |
| Pretraining-Specialized-v1 | RQA 134.6B; STEM-SFT 82.5B; Math-Textbooks 25.1B; InfiniByte-Reasoning 19.4B; Wiki-Rewrite 7.9B; Scientific-Coding 1.2B | CC BY 4.0 (last two CC BY-SA / GFDL) |
| Specialized-v1.2 (May 2026) | Fact-Seeking 35.0B; Multiple-Choice 6.1B; Generative 0.7B | CC BY 4.0 |

- **Licence caveat:** cards for the synthetic subsets warn that Qwen, DeepSeek and Phi-4 licence terms may apply to trained models.
- **NVIDIA's mixing:** Nano 3 used 15 categories with weights shown only in Figure 3. The open 3.5 Lightning recipe lists 23 subsets, all at weight 1.0, with no curriculum.
- **Others' mixing:** Olmo 3 midtraining has 5% Nemotron Synth QA. MobileLLM-R1 mid-training uses Nemotron-Pretraining-Code at 20.69% then 29.10%, and Nemotron-CC-Math at 19.40%.
- **Not read:** the original Nemotron-CC card (HTTP 401) and Specialized-v1.1.

### What transfers to a 100M from-scratch model

Candidate datasets:
- **Web:** `HuggingFaceFW/fineweb-edu` (1.3T per MobileLLM-Pro; 10BT/100BT samples; ODC-By); `mlfoundations/dclm-baseline-1.0` (4T; CC-BY-4.0). `nvidia/ClimbMix` (400B) is CC BY-NC.
- **Long documents:** `allenai/dolma3_longmino_pool` (639B, ODC-By; 8–16k 144B, 16–32k 115B, 32–64k 106B, 64–128k 96B); `HuggingFaceFW/finepdfs` English (1,190B; ODC-By; card advises PDFs below 25%); `common-pile` books (Gutenberg 20.1 GB, pre-1929 46.3 GB; tokens not stated); `deepmind/pg19` (28,752 books; Apache 2.0). No ready repo-level code set verified.
- **Math:** `HuggingFaceTB/finemath` (4+ 9.6B, 3+ 34B; ODC-By); Nemotron-CC-Math 4+ (52B).
- **Code:** `HuggingFaceTB/stack-edu` (125B; Python 21.8B; per-file licences).
- **Synthetic/rewritten:** Nemotron Wiki-Rewrite (7.9B), Math-Textbooks (25.1B); Cosmopedia v2 (39.1M docs; ODC-By); `PleIAs/SYNTH` (~75B; CC-BY-4.0).
- **Instruction:** `HuggingFaceTB/smoltalk2` (SFT 3.38M rows; Apache 2.0 for new parts).
- **Reasoning:** `open-thoughts/OpenThoughts3-1.2M` (Apache 2.0); Nemotron InfiniByte-Reasoning (19.4B).

Published proportions:
- **Main phase:** web 85–90, code 10–12, math 0–3 (SmolLM2/3); MobileLLM-R1 used 64 / 11 / 9 plus 16 papers, wiki, QA.
- **Decay tail (last ~10%):** web 58–63, code 24, math 13–14, synthetic textbooks 4.
- **Long-context stage:** 0.4–0.9% of tokens; long-document share 40% (SmolLM2), 34% (Olmo 3), 21% (Nemotron), rest the previous high-quality mix.
- **Instruction/reasoning before SFT:** IMU-1's last 14B of 72B; Olmo 3 midtraining ~28%.

The Nemotron 3 Nano report (Figure 3 holds the category percentages) was saved but not readable here: `/Users/ksirg/.claude/projects/-Users-ksirg-devel-MrCogito/c3e9ab3f-713e-498a-91b5-9b3a75f1a41c/tool-results/webfetch-1794637376090-zmfdv1.pdf`

### Sources

- [Kimi K2](https://arxiv.org/html/2507.20534) · [Kimi K3](https://arxiv.org/html/2607.24653)
- [Qwen3](https://arxiv.org/html/2505.09388) · [Qwen3.5-0.8B card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [DeepSeek-V3](https://arxiv.org/html/2412.19437) · [DeepSeek-V4](https://arxiv.org/html/2606.19348v1)
- [Magistral](https://arxiv.org/html/2506.10910)
- [GLM-4.5](https://arxiv.org/html/2508.06471) · [Olmo 3](https://arxiv.org/html/2512.13961) · [Ripple-65B Preview card](https://huggingface.co/ricursiveAI/Ripple-65B-Preview)
- [gpt-oss model card paper](https://arxiv.org/html/2508.10925) · [gpt-oss-120b card](https://huggingface.co/openai/gpt-oss-120b)
- [Gemma 3 report](https://arxiv.org/html/2503.19786) · [Gemma 4 report](https://huggingface.co/papers/2607.02770.md) · [Gemma 3 270M card](https://huggingface.co/google/gemma-3-270m)
- [Nemotron 3 Nano paper](https://arxiv.org/html/2512.20848v1) · [Nano 3 pretraining recipe](https://docs.nvidia.com/nemotron/nightly/nemotron/nano3/pretrain.html) · [3.5 Lightning recipe](https://docs.nvidia.com/nemotron/0.2.0/nemotron/lightning35/pretrain.html) · [Nemotron Nano 2 page](https://research.nvidia.com/labs/adlr/NVIDIA-Nemotron-Nano-2)
- [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) · [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) · [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) · [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) · [Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) · [Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) · [Specialized-v1.2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.2)
- [SmolLM2](https://arxiv.org/html/2502.02737v1) · [SmolLM3 blog](https://huggingface.co/blog/smollm3) · [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)
- [MobileLLM-R1](https://arxiv.org/html/2509.24945) · [MobileLLM-Pro](https://arxiv.org/html/2511.06719v1) · [IMU-1](https://arxiv.org/html/2602.02522) · [OPEN-1B](https://arxiv.org/html/2609.17380v1)
- [LFM2 report](https://arxiv.org/html/2511.23404) · [LFM2.5-230M blog](https://www.liquid.ai/blog/lfm2-5-230m) · [Granite 4.0 350M card](https://huggingface.co/ibm-granite/granite-4.0-350m)
- [nanochat README](https://github.com/karpathy/nanochat) · [nanochat discussion 481](https://github.com/karpathy/nanochat/discussions/481) · [ClimbMix](https://huggingface.co/datasets/nvidia/ClimbMix)
- [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) · [dclm-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) · [finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) · [dolma3_longmino_pool](https://huggingface.co/datasets/allenai/dolma3_longmino_pool) · [Common Pile](https://arxiv.org/html/2506.05209v1) · [pg19](https://huggingface.co/datasets/deepmind/pg19)
- [finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) · [stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) · [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) · [SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) · [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)

## 2. Candidate datasets on the Hub: multi-hop, reasoning, long documents

Checked live on the Hub on 2026-11-14 (HF MCP tools plus the Hub API and dataset-viewer statistics); nothing was downloaded. Token figures marked "est." are my estimates from character-length statistics on partial viewer samples at 4 chars/token.

### Group A: multi-hop and reasoning (ranked)

| # | Repo | Row and size | License, gated, dates | Fit and risk |
|---|---|---|---|---|
| 1 | [nvidia/Nemotron-Pretraining-Specialized-v1](https://hf.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) | `text, license, metadata, uuid`; InfiniByte-Reasoning 1.5M rows / 19.4B tokens, RQA 17.5M / 134.6B; ~8–13k tokens per row | CC BY 4.0; open; 2025-12-14, updated 2026-10-28 | Plain-text reasoning at pretraining scale. STEM-heavy and hard for 100M; no eval overlap found. |
| 2 | [framolfese/2WikiMultihopQA](https://hf.co/datasets/framolfese/2WikiMultihopQA) | `id, question, answer, type, evidences, supporting_facts, context`; 167.5k / 12.6k / 12.6k | No tag (mirror xanhho: Apache-2.0); open; 2025-07-25 | Evidence triples give the chain; not one of our evals. Hold out validation and test. |
| 3 | [hotpotqa/hotpot_qa](https://hf.co/datasets/hotpotqa/hotpot_qa) | `id, question, answer, type, level, supporting_facts, context`; train 90.4k, validation 7.4k, fullwiki test 7.4k | CC BY-SA 4.0; open; updated 2025-08-11 | **Eval overlap:** hold out validation (both configs) and fullwiki test. |
| 4 | [dgslibisey/MuSiQue](https://hf.co/datasets/dgslibisey/MuSiQue) | `id, paragraphs, question, question_decomposition, answer, answer_aliases, answerable`; 19,938 / 2,417 | No tag (upstream CC BY 4.0, unverified); open; 2023-06 | **Eval overlap:** hold out validation; training on train makes the eval in-distribution. |
| 5 | [kilian-group/phantom-wiki-v1](https://hf.co/datasets/kilian-group/phantom-wiki-v1) | `id, question, intermediate_answers, answer, prolog, template, type, difficulty`; 500 QA per universe, plus text-corpus and database configs | MIT; open; 2025-03-06 | Fictional universes from a generator, so leak-free. Templated language. |
| 6 | [tasksource/proofwriter](https://hf.co/datasets/tasksource/proofwriter) | `theory, question, answer, allProofs, QDep…`; 585.6k / 85.5k / 174.5k | No tag; open; 2024-01-08 | Proof chains, short items. Same family: [tasksource/ruletaker](https://hf.co/datasets/tasksource/ruletaker) (480k train, Apache-2.0). |
| 7 | [hitachi-nlp/FLDx2](https://hf.co/datasets/hitachi-nlp/FLDx2) | `facts, hypothesis, proofs` plus formula versions; 100k / 5k / 5k | No tag; open; 2024-12-05 | Multi-step deduction corpus with proofs; card gives no detail (points to GitHub). |
| 8 | [RMT-team/babilong-train-5k-samples](https://hf.co/datasets/RMT-team/babilong-train-5k-samples) | `input, question, target`; 6 length configs (0k–32k) × qa1–qa10 × 5k = 300k | No tag; open; 2024-09-18 | **Eval overlap:** same templates as BABILong. Hold out RMT-team/babilong entirely. |
| 9 | [zai-org/LongAlign-10k](https://hf.co/datasets/zai-org/LongAlign-10k) (was THUDM) | `dataset, id, messages, length`; 9,888 rows; 8k–64k tokens; en + zh | No tag; open; 2024-01-29 | Long instructions. English-filtered copy in [HuggingFaceTB/smoltalk2](https://hf.co/datasets/HuggingFaceTB/smoltalk2) (6,249 rows, ~15k tokens average). |
| 10 | [nvidia/Nemotron-SFT-Long-Context-Transduction-v1](https://hf.co/datasets/nvidia/Nemotron-SFT-Long-Context-Transduction-v1) | `messages, metadata`; 10k rows; targets 2k–131k tokens | OpenMDW-1.1; open; 2026-10-14 | Programmatic lookups and sorting. Variable expansion resembles RULER variable tracking. |
| 11 | [open-thoughts/OpenThoughts3-1.2M](https://hf.co/datasets/open-thoughts/OpenThoughts3-1.2M) | `difficulty, source, domain, conversations`; 1.2M rows, 28.2 GB | Apache-2.0; open; 2025-05-28 | Thinking traces, no far dependency. Likewise [nvidia/OpenMathReasoning](https://hf.co/datasets/nvidia/OpenMathReasoning) (3.2M CoT, CC BY 4.0). |
| 12 | [deepmind/narrativeqa](https://hf.co/datasets/deepmind/narrativeqa) | `document, question, answers`; 14.7k / 3.5k / 10.6k (viewer, partial) | Apache-2.0; open; updated 2024-03 | Whole books and scripts as context; document length unverified. |

Checked and set aside:
- **Too small or weak:** IIRC (only the voidful/IIRC mirror, no card), yixuantt/MultiHopRAG (2.6k), allenai/qasper (888 train), emozilla/quality (articles ~3–9k tokens), CLUTRR/v1 (license unknown), LogiQA, FOLIO, PrOntoQA mirrors. tasksource/stepgame (300k train, MIT) is usable but was not ranked.
- **Newer, with caveats:** Daeyongkwon98/SAFE-Verified-MultiHopQA… (286k cleaned items, 2026-08, no license, contains eval dev splits); nace-ai/MegaWikiQA-v1-multihop (5.29M closed-book QA, ODbL, 2026-07, Wikipedia-derived facts may leak answers); caskcsg/LongMagpie_64k_dataset (no license).

### Group B: long documents (ranked)

| # | Repo | Row and size | Typical length | License, gated, dates | Fit and risk |
|---|---|---|---|---|---|
| 1 | [common-pile/project_gutenberg](https://hf.co/datasets/common-pile/project_gutenberg) | 5 columns incl. `text` (viewer timed out); 71,810 docs, 26.2 GB (~6.5B tokens est.) | Median ~68k tokens | Public domain; open; updated 2025-06-06 | Clean books. Contains PG-19, so dedupe and hold out its validation/test books. |
| 2 | [deepmind/pg19](https://hf.co/datasets/deepmind/pg19) | Card: `book_id, short_book_title, publication_date`, url, text (viewer unsupported); 28,602 / 50 / 100 books | Median ~75k tokens (from the [emozilla/pg19](https://hf.co/datasets/emozilla/pg19) Parquet mirror) | Apache-2.0; open; updated 2024-01 | Script loader. BABILong filler text comes from PG-19. |
| 3 | [HuggingFaceFW/finepdfs](https://hf.co/datasets/HuggingFaceFW/finepdfs) `eng_Latn` | 23 columns incl. `text, token_count, page_ends, is_truncated`; 206.9M docs, 1,190.65B tokens | Mean 5.75k tokens, heavy tail | ODC-By; open; 2025-09-05, updated 2026-04-03 | Largest source; filter on `token_count`. Edu subset: [finepdfs-edu](https://hf.co/datasets/HuggingFaceFW/finepdfs-edu). |
| 4 | [common-pile/pre_1929_books](https://hf.co/datasets/common-pile/pre_1929_books) | 137,127 docs, 73.8 GB (~18B tokens est.) | Median ~89k tokens | Public domain; open; updated 2025-06 | OCR noise. |
| 5 | [common-pile/library_of_congress](https://hf.co/datasets/common-pile/library_of_congress) | 135,500 docs, 47.8 GB (~12B est.) | Median ~47k tokens | CC BY 4.0 tag; open; updated 2025-11 | OCR noise. |
| 6 | [common-pile/arxiv_papers](https://hf.co/datasets/common-pile/arxiv_papers) | `id, text, source, created, added, metadata`; 321,336 docs, 21 GB (~5B est.) | Mean ~16k tokens | CC BY / BY-SA / CC0; open; updated 2025-06 | Clean LaTeX-derived text. |
| 7 | [placeholderlabs/pretrain-academic-mix-long-context](https://hf.co/datasets/placeholderlabs/pretrain-academic-mix-long-context) | `document_id, text, part_ends, part_trainable, must_not_split`; 1,001,557 docs, 43.7B tokens | Median ~27k tokens | **No license**; open; 2026-09-10 | Pre-filtered long papers; provenance unknown. |
| 8 | [placeholderlabs/pretrain-repository-v2-mix-long-context](https://hf.co/datasets/placeholderlabs/pretrain-repository-v2-mix-long-context) | Same schema; 2,578,578 repo docs, 130.2B tokens | Mean ~50k tokens | **No license**; open; 2026-09-17 | Only ready-made repo-packed code found. |
| 9 | [institutional/institutional-books-hl](https://hf.co/datasets/institutional/institutional-books-hl) | `barcode_src, title_src, token_count_o200k_base_gen, language_gen, text_by_page_*`; 242B tokens (paper title) | Whole books | Gated (auto), noncommercial, no redistribution; updated 2026-08-19 | Terms conflict with a VC demo. Volume count unverified. |
| 10 | [allenai/peS2o](https://hf.co/datasets/allenai/peS2o) | v2 full text: 8,242,162 papers, 36.09B whitespace tokens | Mean ~4.4k words | ODC-By; open; updated 2024-10 | Few papers reach 16k tokens. |
| 11 | [common-pile/usgpo](https://hf.co/datasets/common-pile/usgpo) | 2,732,677 docs, 74.5 GB | Median ~1k tokens, long tail | Public domain; open | Long hearings in the tail. [common-pile/uk_hansard](https://hf.co/datasets/common-pile/uk_hansard): 51,552 docs, 10 GB, ~48k-token mean. |
| 12 | [HuggingFaceFW/finewiki](https://hf.co/datasets/HuggingFaceFW/finewiki) `en` | 14 columns; sample median ~0.5k tokens | Very few ≥16k | CC BY-SA 4.0 / GFDL; open; 2025-10 | Wikipedia is the source text behind HotpotQA and MuSiQue. |

Code sources that did not make the table:
- **bigcode/the-stack-v2-train-full-ids:** gated and access was denied, so its structure is unverified.
- **nvidia/Nemotron-Pretraining-Code-v1/v2:** manually gated. v3 is open but holds metadata only (`commit_id, rel_path, language`).
- **bigcode/commitpack:** MIT, 4 TB, but per commit rather than per repo.

### Recommendation 1: top 5 Group A sets for 5–10 % of tokens

The classic sets are small (about 0.2B tokens for 2Wiki, less for the others, est.), so only the Nemotron text can fill volume.

1. **2WikiMultihopQA train.** Render as titled paragraphs, then `Question / Evidence chain / Answer`. Pool the paragraphs of 20–40 items into one shuffled 16k–64k "library" and put all questions at the end, so the evidence sits 8k–60k tokens back.
2. **PhantomWiki.** Generate new universes sized so the corpus is 16k–64k tokens. Render articles first, then questions with `intermediate_answers` as the chain.
3. **MuSiQue and HotpotQA train.** Pack the same way, with `question_decomposition` as the steps. Use them only if those evals may be in-distribution; otherwise skip both.
4. **ProofWriter, RuleTaker and FLDx2.** Render as `Facts and rules / Question / Proof / Answer`. Rename entities per theory, merge many theories, interleave book text, and ask at the end. This also replaces BABILong-train with our own templates.
5. **Nemotron InfiniByte-Reasoning and RQA.** Use the `text` column as-is at 2–3 % for reasoning style. Concatenating rows adds length but no far dependency.

### Recommendation 2: top 5 Group B sources

All ≥16k and ≥64k volumes below are estimates.

| Source | Tokens at ≥16k | Tokens at ≥64k |
|---|---|---|
| Gutenberg (common-pile, superset of PG-19) | ~6B | ~5B |
| FinePDFs English | ~0.4–0.6T (unverified, lognormal guess) | ~0.1–0.2T (unverified) |
| Pre-1929 books + Library of Congress | ~29B | ~25B |
| arXiv full text (common-pile) | ~3B | ~0.5B |
| Repo-packed code (Locus repository-v2) | most of 130B | roughly half |

- **FinePDFs:** measure the real share with `token_count` before planning around it.
- **Pre-1929 and LoC:** need an OCR-quality filter.
- **arXiv:** the Locus academic mix would add ~35–40B at ≥16k if its license clears.
- **Repo code:** the Locus mix has no license; the licensed alternative is building from Stack v2 once access is granted.

## 3. Audit of our own data mixes, tokenizers and data serving

## MrCogito data audit (read-only), 2026-11-14 ~06:45 UTC

Scope: checkouts `/Users/ksirg/devel/MrCogito` (dev), `-arch` (arch-10m), `-sft` (e37-sft), `-e33` (e33-multihop),
`-lsa`, `-kvcache`, plus `origin/e37-long`, `origin/e37-needles`, `origin/kv-cache` via `git show`.
Paths below are relative to `/Users/ksirg/devel/MrCogito-arch` unless a checkout or branch is named.

Limits of this audit
- **S3 could not be listed**: the AWS tool returned "expired or invalid AWS credentials". Every S3 size below is
  derived from the scripts and the shard format, not from a listing.
- **Servers and NAS were not opened** (no ssh). Server paths are the ones the scripts and docs name.
- `docs/experiments_specs/ahead/E37_concept_memory.md` is being edited by another session (614 → 636 lines while I
  read); its line numbers are as of ~06:40 UTC.
- The AWS / token-shard pipeline (`infra/aws/`, `data/bin_token_dataset.py`, `scripts/build_longdocs_lite.py`,
  `scripts/run_e33n.py`) exists on arch-10m, e33n-lsa-testbed and the e37-* branches only; **not on dev or main**.

---

### 1. Pretraining data mixes (used or defined)

#### 1a. Headline finding on "SmolLM tokenizer + Nemotron packages"
- **No Nemotron subset was ever tokenized or trained on, as far as the repo records.** Nemotron appears in two
  recipes and one plan table, none of which has a run report, a registry row, a manifest path or a launcher:
  - `data/mix_recipes/nemotron_nano_v3_inspired_2k.json` (its length profile is dated 2026-06-20; source 1 is gated, `"gated": true` at :25);
  - `data/mix_recipes/e18_main_stage1_v1.json` (2026-09-07; the run is **BLOCKED, never launched**:
    `docs/experiments_specs/ahead/E18_main_run_aws_plan.md:12`);
  - E33s Phase C table, `nemotron_cc_v21_hq` 30 % (branch e33-multihop,
    `docs/experiments_specs/ahead/E33s_written_chains_real_data.md:300-309`; the recipe file
    `e33s_realtext_4k_v1.json` it names does not exist on any branch; Phase C is deferred).
  - `docs/2_Experiments_Registry/master_experiment_log.md` has zero "Nemotron" hits.
- The mixes that **were** built are "SmolLM3-inspired" (FineWeb-Edu, DCLM, Stack-Edu, FineMath, FinePDFs, reasoning
  traces). The only NVIDIA source in a built mix is `nvidia/OpenMathReasoning` (cot split, weight 0.05) —
  not a Nemotron package.
- Tokenizers of the built mixes: **SmolLM2-135M** (E05, E36), **Gemma-3-1b** (E10–E17), **SmolLM3-3B = Llama-3
  vocabulary, 128,256** (E18, E22), a 4,096-token BPE (text checks), GPT-2 (E33n → E37).
- Hugging Face, namespace `ksopyla` (listed live): datasets `cogito-text-world`, `cogito-probe-{bits,bind,arith,props}`,
  `long-story-short-pilot` (private, one 146 KB parquet). **No tokenized pretraining mix is on the Hub.**
  Namespace `Meridian21Lab`: no datasets.

#### 1b. Table of mixes

| mix id | status | components (weight) | tokenizer / seq | size recorded | defined in | tokenized output | builder |
|---|---|---|---|---|---|---|---|
| `JeanKaddour/minipile` (default single dataset) | used, early encoder runs | MiniPile | `answerdotai/ModernBERT-base` default | not recorded | `scripts/train_concept_pretraining_multigpu.sh:67,91` | HF datasets cache | online tokenize |
| `long_2k_base_v1` | defined in code | FinePDFs 0.5, FineWeb-Edu sample-10BT 0.3, FineMath-3+ 0.2 | caller's | not recorded | `data/dataset_preprocess.py:382-417` | — | online |
| `smollm3_inspired_2k` | recipe (parent of the E05 ones) | FineWeb-Edu 0.30, DCLM 0.20, Stack-Edu py 0.15, FineMath-3+ 0.15, FinePDFs 0.10, big-reasoning-traces 0.05, OpenMathReasoning cot 0.05 | — / 2048 | not recorded | `data/mix_recipes/smollm3_inspired_2k.json` | — | `scripts/pretokenize_mix.py` |
| `smollm3_inspired_2k_e05warmup` | recipe (no DCLM) | FineWeb-Edu 0.35, FinePDFs 0.25, Stack-Edu 0.15, FineMath 0.15, traces 0.05, OpenMath 0.05 | SmolLM2 / 2048 | not recorded | `data/mix_recipes/smollm3_inspired_2k_e05warmup.json` | not recorded | same |
| **`smollm3_inspired_2k_e05`** | **used: E05, E05b** | FineWeb-Edu 0.30 (6 shards), DCLM 0.20 (35), Stack-Edu py/js/ts/md 0.15, FinePDFs 0.15 (8), FineMath-3+ 0.10 (10), traces 0.05, OpenMath cot 0.05 (`...e05.json:17-90`) | **`HuggingFaceTB/SmolLM2-135M`** (`scripts/launch_e05.sh:36`) / 2048 | 9,956,348 train rows (first build 6.92M) — `docs/2_Experiments_Registry/run_reports/e05b_wd_confound_control_20260711.md:25`, `e05_attempt2_diverged_20260628.md:27`; **tokens not recorded** | `data/mix_recipes/smollm3_inspired_2k_e05.json`, `scripts/launch_e05.sh:42` | server `…/dev/hf_home/datasets_tok/` (HF Arrow `save_to_disk` + `<mix>_manifest.json`) | `scripts/pretokenize_mix.py` (default tokenizer SmolLM2, :525) |
| same recipe, Gemma | used: E10–E16a | as above | `google/gemma-3-1b-pt` / 2048 (`scripts/launch_e10.sh:49-55`) | not recorded | same recipe | `…/hf_home/datasets_tok_gemma/smollm3_inspired_2k_e05_gemma_manifest.json` (`launch_e10.sh:27,55`) | same; raw read from NAS `/nas/ml_data/mrcogito/hf_datasets/raw` (`launch_e10.sh:31-32`) |
| `concept_flow_reasoning_2k` | recipe for E08 (spec in `ahead/`) | OpenMath cot 0.30, traces 0.20, FineWeb-Edu 0.25, DCLM 0.15, Stack-Edu py 0.10 | — / 2048 | not recorded | `data/mix_recipes/concept_flow_reasoning_2k.json` | not recorded | same |
| `smollm3_inspired_4k_e16b` | recipe | FinePDFs 0.28, DCLM 0.22, FineWeb-Edu 0.18, Stack-Edu 0.12, FineMath 0.10, traces 0.05, OpenMath 0.05 | Gemma / 4096 | not recorded | `data/mix_recipes/smollm3_inspired_4k_e16b.json` | not recorded | same |
| **`e16b_long_4k_v1`** | **used: E16b, E17, E17b–e** | FinePDFs 0.30, PG-19 0.18, DCLM 0.15, Wikipedia-en 0.12, FineWeb-Edu 0.10, Stack-Edu 0.10, FineMath 0.05 | `google/gemma-3-1b-pt` / 4096 | tokens not recorded in docs (a `.token_stats.json` sidecar sits next to the manifest on the server) | `data/mix_recipes/e16b_long_4k_v1.json`; `scripts/launch_e16b.sh:22,36-37` | `…/hf_home/datasets_tok_gemma_4k/e16b_long_4k_v1_gemma_manifest.json` (Odra: `docs/experiments_specs/done_success/E16b_longctx_muon_1b_plan.md:46-58`) | same |
| **`e18_pilot_longdoc_v1`** | **used: E18 pilot, E18b** | per row: FineWeb-Edu 0.55, FinePDFs 0.24, Stack-Edu py 0.15, PG-19 0.06 (≈ 24 % PG-19 / 36 % FinePDFs / 28 % FineWeb / 11 % code by tokens) | **`HuggingFaceTB/SmolLM3-3B`** (128,256; `scripts/launch_e18.sh:20`) / 8192 and 32768 | "~3B tokens" planned (`docs/experiments_specs/done_failed/E18_perceiver_ar_v2_baseline.md:110`); measured total not recorded | `data/mix_recipes/e18_pilot_longdoc_v1.json` | Polonez `/home/ksopyla/dev/hf_home/datasets_tok_smollm3_8k/` and `…_32k/` (`scripts/launch_e18.sh:83-88`; `docs/experiments_specs/done_failed/E18b_retrieval_trained_read_plan.md:58,81`) | same |
| `e22_longdoc_32k_v1` (+ merged `e22_longmix_32k_manifest.json`) | used: E22 | PG-19 chunked into 32k rows 0.5, FinePDFs ≥ 4,096 tokens 0.5; merged with the 8k fluency tier and 5 % keyed-recall rows | SmolLM3-3B / 32768 | not recorded | `data/mix_recipes/e22_longdoc_32k_v1.json`; `scripts/launch_e22.sh:22,88-91` | `…/hf_home/datasets_tok_smollm3_32k/` | `pretokenize_mix.py` (`chunk_long_docs`) + `scripts/write_manifest_variant.py` |
| **`e18_main_stage1_v1`** (Nemotron-first) | **defined only; run blocked 2026-09-12** | see 1c | planned `HuggingFaceTB/SmolLM3-3B` / 32768 (`E18_main_run_aws_plan.md:68`) | pool ~350B tokens, target 300B (recipe :3; plan :46) | `data/mix_recipes/e18_main_stage1_v1.json` | **none** (planned `/data/hf_home/datasets_tok_smollm3_32k` on the AWS node, then S3) | `pretokenize_mix.py`, est. 6–10 h on 96 vCPU (plan :67) |
| `nemotron_nano_v3_inspired_2k` | defined only | `nvidia/Nemotron-CC-v2` High-Quality 0.35 (gated), FinePDFs 0.25, DCLM 0.15, FineMath-4+ 0.10, traces 0.10, Stack-Edu py 0.05 | — / 2048 | none | `data/mix_recipes/nemotron_nano_v3_inspired_2k.json:15-84`; `docs/engineering_specs/long_context_data_mix_loader_architecture.md:85-88` | none | — |
| `e36_longdoc_4k6_v1` | used: E36 (Oct 18) | PG-19 chunked 0.5 (6 shards), FinePDFs long 0.5 (4 shards); rows of exactly 4,610 | **SmolLM2** (reader's tokenizer; `scripts/launch_e36.sh:38,44`) | not recorded (S1 used ≈ 0.59B row tokens per run) | `data/mix_recipes/e36_longdoc_4k6_v1.json` | Polonez `$DATASETS_TOK_DIR/e36_longdoc_4k6_v1_manifest.json` (`launch_e36.sh:17`) | `pretokenize_mix.py` |
| text-world v0 / v1 / v2 (text checks) | used: all 30M text checks, E33 line | TinyStories/SimpleStories rows of 4,096 + generated "world documents" (~1.4k tokens) with questions | own 4,096-token BPE (branch e33-multihop `evaluation/text_checks.py:29`) | v0: 2.1B tokens = 230,713 story rows + 836,072 world rows (`docs/engineering_specs/text_capability_checks.md:457-459`) | `scripts/build_text_checks_data.py`, `data/text_world.py` | servers `…/hf_home/datasets_tok/text_checks_v*`; Hub `ksopyla/cogito-text-world` (parquet: train/stories 6 × 0.9 GB, train/world ≥ 7 × 0.33 GB, eval splits, tokenizer, generator) | same script |
| **E33n / E37 token shards** | **current** | see section 2 | GPT-2 (50,257) | 6.0B + 1.5B tokens in S3 | `scripts/run_e33n.py`, `infra/aws/jobs/data_build.sh` | S3 `data/`; home servers `~/dev/MrCogito-lsa/Cache/e33n_data` | `data_build.sh`, `scripts/build_longdocs_lite.py` |
| E37 SFT mix (branch e37-sft) | used, home servers | `HuggingFaceTB/smol-smoltalk` (~412M GPT-2 tokens; ≤ 180M sampled + every LongAlign conversation that fits 16k) + 10 % LongDocs-lite replay; ~200M tokens per arm | GPT-2 | `docs/experiments_specs/ahead/E37_sft_sensitivity.md:39-57` (e37-sft) | `scripts/build_sft_mix.py` (e37-sft) | per server `Cache/e37_sft/data/sft_mix/` (.bin + .mask shards, `manifest_sft.json`) | `build_sft_mix.py` |

Raw archive: `pretokenize_mix.py --raw_archive_dir` moves raw parquet/zst to the NAS,
`/nas/ml_data/mrcogito/hf_datasets/raw/<source>/` (`scripts/train_concept_pretraining_multigpu.sh:332-334`). What it
holds today: not recorded in the repo.

#### 1c. Nemotron subsets in `e18_main_stage1_v1.json` (planned shares; nothing downloaded on record)

| source name (line) | Hub id / subset | shard cap | size note in recipe | token share (:13-28) |
|---|---|---|---|---|
| nemotron_cc_v21_hq (:48) | `nvidia/Nemotron-CC-v2.1` / High-Quality | 20 (all) | ~26B tokens | 25 % |
| nemotron_cc_v21_mhq (:58) | … / Medium-High-Quality | 12 (all) | ~16.9B | 13 % |
| nemotron_cc_v21_hq_synth (:68) | … / High-Quality-Synthetic | 40 of 86 | ~43B | 12 % |
| nemotron_cc_v21_hq_translated (:78) | … / High-Quality-Translated-To-English | 10 of 42 | ~9B | 3 % |
| nemotron_cc_v21_dqa (:88) | … / High-Quality-DQA | 6 | ~6B | 2 % |
| nemotron_code_v1_synthetic (:98) | `nvidia/Nemotron-Pretraining-Code-v1` / Synthetic-Code | 80 of 316 | ~28B | 10 % |
| nemotron_cc_math_4plus (:108) | `nvidia/Nemotron-CC-Math-v1` / 4plus | 46 (all) | ~62 GB | 5 % |
| nemotron_math_textbooks (:118) | `nvidia/Nemotron-Pretraining-Specialized-v1` / Nemotron-Pretraining-Math-Textbooks | 13 (all) | ~25B | 3 % |
| nemotron_rqa (:128) | … / Nemotron-Pretraining-RQA | 40 of 175 | not given | 3 % |
| nemotron_stem_sft (:138) | … / Nemotron-Pretraining-STEM-SFT | 30 of 104 | not given | 2 % |
| nemotron_wiki_rewrite (:148) | … / Nemotron-Pretraining-Wiki-Rewrite | 7 (all) | not given | 1 % |
| nemotron_fact_seeking (:158) | `nvidia/Nemotron-Pretraining-Specialized-v1.2` / Nemotron-Pretraining-Fact-Seeking | 10 of 72 | ~300 tokens/row | 1 % |
| nemotron_sft_general (:168) | `nvidia/Nemotron-Pretraining-SFT-v1` / Nemotron-SFT-General | 20 of 68 | ~7B | 2 % |
| finepdfs_100BT (:178) | `HuggingFaceFW/finepdfs_100BT` | 60 of 100 | ~60B | 14 % |
| pg19 (:189) | `emozilla/pg19` | 23 | ~3B | 4 % |

Access (as of 2026-09-07, `E18_main_run_aws_plan.md:17`): CC-v2.1, CC-v2, CC-Math-v1, Code-v1, SFT-v1, Specialized
approved for the Meridian21Lab account; `Nemotron-CC-Code-v1` and `Nemotron-Pretraining-Code-v2` requested, pending.
These repos are gated: they need `HF_TOKEN` (recipe :3), which today's AWS nodes do not carry (section 3).
Earlier E18 spec figures (600B tokens, other subset sizes): `docs/experiments_specs/done_failed/E18_perceiver_ar_v2_baseline.md:111`.

---

### 2. Data of the current 100M runs (E37 stage B, `scripts/run_e33n.py` size `m100`)

Budget: `SIZES["m100"]` trunk 3.0e9, decay 0.4e9, 128 rows of 4k per step (`scripts/run_e33n.py:44`);
`infra/aws/jobs/e37_full.sh:10` sets `E37_TRUNK_TOKENS=3.0e9 E37_DECAY_TOKENS=4.0e8`.

| stage | source | how built | row rule | tokens used | on disk | size |
|---|---|---|---|---|---|---|
| trunk, 4k | `kjj0/finewebedu10B-gpt2` — **a third party's GPT-2 tokenization of FineWeb-Edu sample-10BT; downloaded, not built by us; no raw text** | `infra/aws/jobs/data_build.sh:17-24`: `snapshot_download` of `finewebedu_val_000000.bin` + train shards 1..N (N = 60 in S3: `data_prep.sh:9`) | `"rule": "stream"`, seq 4096, seed 42 (`run_e33n.py:168-170`): consecutive 4,096-token windows of each shard, documents packed and cut at row edges, per-token doc ids from `<\|endoftext\|>` stop attention across documents (`data/bin_token_dataset.py:71-73,121-128`) | 5,722 steps × 128 rows × 4,096 = 3.0B of the 3.2B in the 32 shards a node copies (781,248 rows) | modded-nanogpt `.bin`: 256 int32 header `[20240520, 1, n]`, then uint16 ids | 200,001,024 bytes per shard (Hub listing): 60 + 1 val = **12.2 GB in S3**; **6.6 GB per stage-B node** (32 + val) |
| decay, 16k: "LongDocs-lite" | Common Pile "filtered" sets: `common-pile/project_gutenberg_filtered`, `arxiv_papers_filtered`, `doab_filtered`, `usgpo_filtered`, `uk_hansard_filtered` (`scripts/build_longdocs_lite.py:26-33`; default sources :127; `wikimedia` optional; LSA's code source, The Stack, left out because gated: :3-4) | whole Hub rows, tiktoken `"gpt2"` (:135-137); kept if ≥ 16,384 GPT-2 tokens (:93, :129; pre-filter `len(text) ≥ 2.5 × min_tokens` :106); `[50256] + ids` (:95); every 100th kept doc to val (:96); stop at 300M train tokens per source (:98, :128); shards of 100M (:130) | `"rule": "doc"`, seq 16384, seed 1042 (`run_e33n.py:172-174`): one contiguous span of ONE document per row, consecutive non-overlapping spans, tail dropped (`bin_token_dataset.py:76-84`); sources interleaved at equal weight, each capped at the smallest source's row count (:102-116) | 763 steps × 32 rows × 16,384 = 0.4B (about a quarter of the 1.5B built; rows per source not recorded) | same `.bin` format, `<source>_{train,val}_NNNNNN.bin` + `manifest.json` | 5 × ~300M train tokens × 2 bytes ≈ **3.0 GB** + val 2.2–2.9M tokens per source ≈ 26 MB (derived; not listed) |
| 64k stage (branch e37-long) | **the same LongDocs-lite shards; no separate build** | manifest `long = {**decay, "seq_len": 65536, "seed": 2042}` (`origin/e37-long:scripts/run_e33n.py:208`) | `doc` rule at 65,536: only documents ≥ 65,536 tokens give rows; "train rows at 64k: arxiv 693 is the smallest source" (`origin/e37-long:docs/experiments_specs/ahead/E33t_catalogue_read.md:355-359`) → 5 × 693 = 3,465 rows ≈ 227M tokens available | 191 steps × 8 rows × 65,536 = 0.1B (`origin/e37-long:infra/aws/jobs/e37_long.sh:4-7,18`) = 44 % of what exists | — | 0 extra |

LongDocs-lite build record: 2026-11-04, 11 min on a c7i.4xlarge; documents kept 2,906 / 8,911 / 3,217 / 3,092 / 5,088
(gutenberg / arxiv / doab / usgpo / hansard) (`docs/experiments_specs/done_failed/E33n_plan.md:11`;
`infra/aws/jobs/data_prep.sh:5`). Home-server copies rebuilt by `data_build.sh` are "identical to the S3 copy (same
documents kept and tokens per source, same shard hash)" (`E33n_plan.md:55-56`).

Tokenizer: GPT-2, saved with `pad_token = eos_token` as `tokenizer_gpt2/` (`data_build.sh:27-34`).
Eval rows: FineWeb val 256 rows; LongDocs val 16 rows per source (`run_e33n.py:170,174`).

Consequences worth knowing
- S3 holds 6.0B FineWeb-Edu tokens; the Hub release has ~10B. A trunk above ~6B tokens needs more shards pulled.
- LongDocs-lite is small for long rows: 227M tokens of 64k rows in total; rows ≥ 128k not counted anywhere.
- No code, no agent traces, no Wikipedia in the long tier (The Stack is gated; `wikimedia` is optional and off).

---

### 3. How data reaches an AWS node today

No baked AMI, no EBS snapshot, no streaming. Each node boots a stock image and **copies from S3 to local NVMe**.

| step | where | what |
|---|---|---|
| 0 (once) | `infra/aws/setup.sh` | private bucket `mrcogito-train-<account>` in us-east-1 (`env.sh:8,24`), node role limited to that bucket + SSM (:31-36), security group SSH from the laptop's current IP (:54-63), key pair |
| 0 (once, 2026-11-04) | `infra/aws/jobs/data_prep.sh` on a c7i.4xlarge CPU node | `data_build.sh DATA 60 300e6 16` pulls public Hub data **anonymously** (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, `data_build.sh:12`), then `aws s3 sync` to `s3://BUCKET/data/{fineweb_edu_gpt2,tokenizer_gpt2,longdocs_lite_gpt2}/` (:10-12). 11 min (:5) ≈ $0.13 at $0.714/h (derived) |
| 1 | `infra/aws/launch.sh` | AMI = latest AWS Deep Learning base GPU AMI from an SSM parameter (:29-32); 200 GB gp3 root (:82, `env.sh:20`); user data: hard `shutdown -h +MAX_HOURS` (:41), instance NVMe at `/opt/dlami/nvme` (:43-49), install `uv`, `git clone --branch BRANCH --single-branch https://github.com/ksopyla/MrCogito.git` (:62-65), tmux `job` running `node_run.sh JOB` (:67) |
| 2 | `infra/aws/node_run.sh` | `HF_HOME`, uv cache on NVMe (:10); spend heartbeat + job log to S3 every 5 min (:21-31); `uv sync --frozen` (:39); the job (:41); power-off = terminate (:46-48) |
| 3 | `infra/aws/jobs/e33n_run.sh:97-107` | `aws s3 cp` the val shard and the first `tokens // 100M + 2` train shards (32 for 3.0B) one by one (:99-104), `aws s3 sync` tokenizer and LongDocs-lite (:105-106). ≈ 9.6 GB per stage-B node |
| 4 | same, :46-62 | run folder synced to `s3://BUCKET/runs/<exp>/<run>/` every 15 min; with `E33N_CKPT_S3=1` a relaunched spot node first pulls the run folder back (:56-58) |
| eval-only nodes | `infra/aws/jobs/e37_ruler_later_node.sh:45`, `lsa_orig_node.sh:62` | tokenizer from S3, checkpoints from the run folder |

Staging time and cost
- **Not timed separately anywhere.** Indirect record: instance "running" → first training step is 2–4 minutes on
  three nodes, which includes clone, `uv sync` and the S3 copy: `e37-b-l40s` launched 22:26:24Z
  (`infra/aws/runs.log:2`), trunk from 22:28 (`E37_concept_memory.md:343`); 64k-stage nodes launched 19:12 /
  21:54, training from 19:16 / 21:57 (`E37_concept_memory.md:570,580`). At ≤ $7.6/h for the 4 × L40S spot node that
  is well under $1 per node (derived).
- S3 storage and transfer cost: not recorded.

Human steps and limits
- A valid AWS login on the laptop (profile `m21-research`, `env.sh:5`); it expired at least once mid-campaign
  (`E37_concept_memory.md:289`) and is expired now.
- `setup.sh` must be re-run when the laptop's IP changes (`infra/aws/README.md:9`).
- Code must be pushed to GitHub first: the node clones a branch of the public repo (`launch.sh:65`).
- **No Hugging Face token on nodes** ("Hugging Face downloads are anonymous", `README.md:28-29`). Gated data
  (Nemotron, The Stack) cannot be built on a node as things stand; it needs a token in Secrets Manager or a build
  elsewhere followed by an upload.
- W&B: key read from Secrets Manager secret `mrcogito/wandb-api-key` if the author stored it and granted the role
  read access, else offline and synced later from the laptop (`infra/aws/jobs/wandb_env.sh:2-8`). Branch e37-long
  forces offline.
- A `launch.sh` job cannot pass environment variables to the node, hence wrapper jobs (`e37_full.sh:7`).
- Quota: one 4-GPU spot slot (g6e.12xlarge) at a time (`E37_concept_memory.md:562-563`); no on-demand H100 since
  2026-11-04 (`infra/aws/README.md:19-21`).
- Checkpoints go from S3 to the home servers by one-hour presigned links (`E37_concept_memory.md:428`).
- A data rebuild or a new mix is a manual one-off: launch `data_prep.sh` (or a new job) on a CPU node.

---

### 4. Dataloader path of these runs

| item | value | evidence |
|---|---|---|
| entrypoint | `training/train_concept_pretraining.py` via `uv run python` or `torchrun --standalone --nproc_per_node=G` | `scripts/run_e33n.py:205-206` |
| command builder | `scripts/run_e33n.py cmd --arm --size --phase --data --out [--tokens --micro --lr --seed --gpus --init --eval_every]` | :244-276 |
| data route | `--bin_manifest` → `data.bin_token_dataset.load_bin_manifest`; "replaces every other data route when set" | `training/train_concept_pretraining.py:182-193`; `training/concept_pretraining_args.py:554-560` |
| dataset class | `BinRowDataset` (map-style): `np.memmap` uint16 per shard; an item is one slice → int64 `input_ids [S]` + `doc_ids [S]` | `data/bin_token_dataset.py:46-51,87-128` |
| row index | built once at start: every stream window, or every document span (the `doc` rule scans each whole shard for `eot`); a seeded permutation per source, sources interleaved by weight | :71-84, :102-116 |
| sampling | `--batch_packing_mode none` → HF Trainer's random sampler over the fixed index; Accelerate/DDP shards it across ranks; training stops at `--max_steps` | `run_e33n.py:219`; `training/concept_pretraining_trainer.py:82-87` |
| collator | `DataCollatorForCausalLM`: labels = inputs; label −100 at each document start (from `doc_ids`, not from a token id); out-of-vocab ids clamped | `data/data_collators.py:243-300`; `training/concept_pretraining_factories.py:826-835` |
| workers | `--dataloader_num_workers 6` per rank (24 on a 4-GPU node); other DataLoader settings are HF defaults | `run_e33n.py:227` |
| batch | trunk micro 4 × accum 8 × 4 GPUs = 128 rows (L40S); decay micro 1; 64k micro 1 × accum 2 × 4 | `infra/aws/jobs/e37_multi.sh:19`; `e37_full.sh:11-12`; `run_e33n.py:190-203` |
| data flags / env | `--bin_manifest`, `--tokenizer_name`, `--max_seq_length`, `--per_device_train_batch_size`, `--gradient_accumulation_steps`, `--max_steps`; env `E37_TRUNK_TOKENS`, `E37_DECAY_TOKENS`, `E37_MICRO_TRUNK`, `E33N_DECAY_TOKENS`, `E33N_LONG_TOKENS`, `E33N_GPUS`, `E33N_SEED`, `E33N_DATA` | `run_e33n.py:216-229`; `infra/aws/jobs/e33n_run.sh:8-28` |

Throughput
- **No dataloader bottleneck is recorded for this path.** The runs are GPU-bound: e37a's 3.0B-token trunk took
  22:28 → 09:00 on 4 × L40S (`E37_concept_memory.md:343`) ≈ 79k tokens/s; the 64k stage runs at 19–32 s per step
  (:570-581).
- Recorded data-side costs belong to the older HF-Arrow path (`--pretokenized_manifest`): padding waste with
  per-document rows (`docs/engineering_specs/pad_free_variable_length_training.md:10`), 35–37 min single-process
  length-cache / token-stats passes on a 32k manifest (`docs/experiments_specs/done_failed/E18b_retrieval_trained_read_plan.md:35`).
- Not measured: start-up cost of the `doc`-rule scan (15 shards read in full per rank and per phase), and whether
  6 workers keep up on an H100 at 4k rows.

---

### 5. Tokenizer handling, and what a switch to SmolLM2 (49,152) would touch

What the code supports
- The trainer is tokenizer-agnostic: `AutoTokenizer` from `--tokenizer_name`; pad set to eos when missing
  (`training/train_concept_pretraining.py:153-159`).
- Vocabulary size enters the model in one place: `vocab_size=len(tokenizer)`
  (`training/concept_pretraining_factories.py:544`), with `token_embedding_dim` (:547) and pad/bos/eos ids (:590-592).
- 100M model: input table `nn.Embedding(V, e=384)` (`nn/perceiver_ar_lm.py:2288`; `run_e33n.py:44,101`) and an
  **untied** output layer `Linear(896, V)` (`nn/perceiver_ar_lm.py:3002`). GPT-2: 50,257 × (384 + 896) ≈ 64.3M of the
  ~162–166M total. SmolLM2: 49,152 × 1,280 ≈ 62.9M (−1.4M). Sizes are named by the non-embedding count
  (`run_e33n.py:123-140`), so the FFN fit does not move.
- SmolLM2's tokenizer (Hub, read live): `<|endoftext|>` = id 0 and is bos = eos = unk; no pad token; `<|im_start|>` = 1,
  `<|im_end|>` = 2. 49,152 < 65,536, so **uint16 shards still work** (SmolLM3's 128,256 would not).
- Tokenizers used so far: ModernBERT, SmolLM2-135M, Gemma-3-1b, SmolLM3-3B, 4,096 / 16,384 BPE, GPT-2.
  `scripts/pretokenize_mix.py:525` already defaults to SmolLM2 but writes HF Arrow, not `.bin` shards.

What breaks or must be rebuilt

| area | item | evidence |
|---|---|---|
| data | **FineWeb-Edu trunk shards**: the Hub release is GPT-2 ids with no text; re-tokenize `HuggingFaceFW/fineweb-edu` sample-10BT from raw text. No `.bin` writer for an arbitrary tokenizer exists | `infra/aws/jobs/data_build.sh:17-24` |
| data | **LongDocs-lite**: the builder is tiktoken-GPT-2 only; the ≥ 16,384-token filter is counted in GPT-2 tokens, so the kept set shifts | `scripts/build_longdocs_lite.py:34,93,95,135-137` |
| data | S3 `data/` prefixes and home copies (`*_gpt2/`); upload of the new shards | `infra/aws/jobs/data_prep.sh:10-12` |
| constants | `VOCAB = 50257`, `EOT = 50256` → manifests' `"eot"` and the parameter-count stub | `scripts/run_e33n.py:38-39,136,168,172` |
| constants | `eot` default 50256 in the dataset and the manifest loader | `data/bin_token_dataset.py:90,133` |
| evals that open val shards with the **default** eot (would mis-split documents at id 0) | long-document loss, copy split, flow probe, password probe | `evaluation/long_doc_loss.py:137-138`; `evaluation/copy_split.py:112`; `evaluation/flow_probe.py:151`; `origin/e37-needles:evaluation/long_passkey.py:36` |
| job scripts | `tokenizer_gpt2` path hard-coded | `scripts/run_e33n.py:218`; `infra/aws/jobs/e33n_run.sh:105,144,151`; `e37_evals.sh:37,50`; `e37_ruler_later.sh:40`; `e37_ruler_later_node.sh:45`; `stage0.sh:15`; `e37_stage0_node.sh:12`; `lsa_orig_node.sh:62`; `data_build.sh:30-32` |
| checkpoints | every trained 100M model (e37a, lsa, catalogue_reuse_n4, dense, window, notes_open, dense_rope, seeds, 64k finals, SFT finals) is GPT-2 only; all arms and controls retrain | — |
| RULER needles | runs through `--tokenizer` (generic), but prompts are "sized by the model's own tokenizer": 4k / 16k / 64k haystacks hold different text; numbers stop being comparable with our GPT-2 runs and with LSA's table | `evaluation/run_lsa_suite.py:4-8,65,89-91` |
| "core" quizzes | log-likelihood tasks through lm-eval with `--tokenizer`: work unchanged | `evaluation/run_lsa_suite.py:9,25`; `evaluation/lm_eval_perceiver_ar.py:118-128` |
| BLiMP / SWDE / BABILong (branch e37-sft) | generic; BABILong picks a dataset config by nominal token length, so real lengths shift | `MrCogito-sft/evaluation/run_lsa_suite.py:35-41,51-70` |
| long-document loss | needs the rebuilt val shards; nats per token are not comparable across tokenizers (compare per byte or per document) | `evaluation/long_doc_loss.py:101,133-138` |
| greedy decoding | uses `tokenizer.eos_token_id` only: generic | `evaluation/lm_eval_perceiver_ar.py:175` |
| cached-decoding tests | stub tokenizer with eos 0, tiny random models: unaffected | `origin/kv-cache:tests/test_incremental_decode.py:34` |
| unit tests | `EOT = 50256`, vocabulary 50257 | `tests/test_bin_token_dataset.py:10,67-68`; `tests/test_long_doc_loss.py:8` |
| original-LSA baseline | their preset is GPT-2 (`qwen35_0_8b_gpt2`, V = 50,304, eot 50256): it stays GPT-2 unless their config is edited, so that comparison is lost or becomes cross-tokenizer | `evaluation/lsa_orig_eval.py:115,185,206`; `scripts/lsa_orig.py:55` |
| SFT (branch e37-sft) | the builder refuses any tokenizer whose `<\|endoftext\|>` is not 50256; role markers are plain text. With SmolLM2 the native `<\|im_start\|>` / `<\|im_end\|>` could replace them; the mix and its ~412M-token count are rebuilt | `MrCogito-sft/scripts/build_sft_mix.py:55,65-66` |
| `message_boundary_token_id` | −1 (off) in every current arm; nothing to change | `scripts/run_e33n.py:51,58,62,69,75` |
| default tokenizer of the long-context probes | SmolLM3-3B when none is passed and none is saved with the checkpoint | `evaluation/long_context_probes.py:55-65` |
| nn-builder designs | `"vocab": 50257` (drawings only) | `tools/nn-builder/designs/*.json:22` |

Prior decision on record: SmolLM2 tokenizer with **tied** embeddings, 16k rows, ~35 % long documents was decided on
2026-10-13 for the (never launched) ~500M AWS plan (`docs/1_Strategy_and_Plans/aws_small_pretraining_plan.md:95-100`);
E33n then took GPT-2 to copy LSA's protocol (`docs/experiments_specs/done_failed/E33n_lsa_testbed_topk_notes.md:35,73`).

---

### 6. Past multi-hop / reasoning data work

| work | data built | size | where | recorded lesson |
|---|---|---|---|---|
| E18b keyed-recall rows (Sep) | synthetic dense-label retrieval rows at 32k, merged 5 % → 100 % into the E18 pilot manifest (`scripts/build_retrieval_mix_dataset.py`) | 6,000 train rows × 32k ≈ 200M tokens (`docs/experiments_specs/done_failed/E18b_retrieval_trained_read.md:102-103`) | Polonez `datasets_tok_smollm3_32k/e18b_lm_ret05_manifest.json` | failed for the architecture, not the data: first-token 4.49 % vs the dense control's 99.33 % on the same rows (`docs/2_Experiments_Registry/master_experiment_log.md:143,258-262`) |
| text-world v0–v2 (Oct), text checks | stories + generated world documents with questions; exams 1k–128k | 2.1B tokens; later builds with 8 questions + 4 bridge questions per document | servers `datasets_tok/text_checks_v*`; Hub `ksopyla/cogito-text-world` | question density decides whether the notebook switches on: 1 question per document switched none on, 8 switched every one on (`MrCogito/docs/4_Research_Notes/hierarchy_lessons_hnet_ncm_mai_20261030.md:303-309`) |
| compose / deduce diagnosis (17 Oct) | none new | — | — | "the join between two lookups gets about 0.04 % of the training loss and no curriculum"; every model incl. dense at the guessing floor (`docs/1_Strategy_and_Plans/agenda.md:132-140`) |
| written chains on the text world (R2, 19 Oct; branch e33-multihop) | world documents whose answers write out their reasoning; steps removed in stages all → drop1 → none (`scripts/build_text_checks_data.py --written_chains`) | 0.15B-token stage builds, 0.9 GB each | Odra / Polonez `datasets_tok/text_checks_v2written_*` | **helped at 30M on synthetic text**: compose 92–93 % / deduce 99–100 % at 4k without generator labels, latest 78 %; count 17 %, 3-hop 0 %, new sentence shapes 5 %. One warm-started run; never shown from scratch (`MrCogito-e33/docs/1_Strategy_and_Plans/agenda.md:65-72`; spec `E33s_written_chains_real_data.md:17-27`) |
| E33s real multi-hop (21–22 Oct) | `data/real_multihop.py` + `scripts/build_real_multihop_data.py`: MuSiQue-Ans 19.9k, HotpotQA distractor 90.4k, 2WikiMultiHopQA 167.5k, GSM8K-Aug 385.6k as written chains, padded with distractor paragraphs to ≤ 4,096 tokens; frozen dev exams at 4k / 16k / 64k | pilot only: 0.3B pretraining tokens (Wikipedia + FineWeb-Edu) + 3 × 0.05B QA stages (17.5k / 17.8k / 18.4k rows of ~2k tokens), 16,384-token BPE | Polonez (exact folder not recorded) | rule coverage: MuSiQue 83 %, HotpotQA 67 %, 2Wiki 74 %, GSM8K 100 % (plan :189-200). 30M pilot: every score on its no-evidence floor (:204-224). **Ceiling probe: pretrained SmolLM2-135M fine-tuned on the same stages reaches MuSiQue 4.7 % exact (floor 1.0 %), HotpotQA 18.3 % (7.0 %); written chains did not help at that size** (:226-257). Verdict: "As specced, E33s would not be informative"; put on hold, re-scoped to the text world, real data deferred (spec :3-6, :17-19) |
| E33s re-scope, Phases A / B (24 Oct) | planned builds `text_checks_v2written_s_{all,drop1,none}` | 1.6B + 0.2B + 0.2B tokens, ~12 GB per server (plan :350-357) | planned | draft awaiting go-ahead; spec "Result" section empty |
| long-story-short / deductive stories | spec `docs/engineering_specs/deductive_stories_synthetic_dataset.md` | Hub `ksopyla/long-story-short-pilot`: one 146 KB parquet (private) | Hub | no result recorded |
| `cogito-probe-*` | concept-compression probes (bits, bind, arith, props) | small, on the Hub | Hub | probes, not training data |

---

### 7. What is already written about a better mix or the next data run

| where | what it says |
|---|---|
| `docs/experiments_specs/ahead/E37_concept_memory.md:93-94` and `:619` | **E37 "Stage D (data)"**: 5 % synthetic retrieval and multi-hop documents in the long decay (HiLS's recipe), a length curriculum 4k → 16k → 64k (SkyLadder), related-document packing. Two lines; no design, budget or go-ahead |
| same file `:564-592` | the 64k stage result at 100M: 0.1B tokens of 64k rows lowers loss at 64k by 0.03–0.04 nats and at 128k by 0.07–0.12, leaves ≤ 16k alone, costs $13–18 per model |
| `MrCogito/docs/4_Research_Notes/hierarchy_lessons_hnet_ncm_mai_20261030.md:69-74, 303-309` | "Far context only pays on long documents": LSA's variants sit within 0.004 at 4k and separate only in a 12B-token decay on 128k-token books, papers, code repositories and agent traces (window-only 1.154 vs full attention 1.085). "Long natural documents are a data lever we have not used." LSA's LongDocs recipe (public sources, equal share per domain, extra weight at the longest lengths) named as the template |
| `docs/literature_review/learned_kv_context_compression.md:382-388` | LSA's own data: 100B FineWeb-Edu at 4k, then 12B at 128k; LongDocs of 16k–1M tokens weighted towards 512k–1M |
| `docs/1_Strategy_and_Plans/aws_small_pretraining_plan.md:95-124` | the 2026-10-13 plan for a ~500M run: SmolLM2 tokenizer, tied embeddings, 16k packed rows, ~35 % long documents; proposed mix FineWeb-Edu 30 / DCLM 15 / Stack-Edu py 10 / FineMath-3+ 10 / PG-19 17.5 / FinePDFs ≥ 4k 17.5; phase 2 adds synthetic far-recall documents. Open: synthetic share, PG-19 decontamination ("missing"), 2-byte storage check. Never launched |
| `docs/experiments_specs/ahead/E18_main_run_aws_plan.md:44-51` | the blocked Nemotron plan's context schedule: 300B at 8k → 32k, then 50B at 256k with the long tier re-weighted to 35 % (+ ProLong-64K / 512K), then 20B at 512k with repo-level code |
| `MrCogito-e33/docs/experiments_specs/ahead/E33s_written_chains_real_data.md:300-309` | Phase C pretraining mix (deferred): Wikipedia 40 / Nemotron-CC-v2.1 HQ 30 / PG-19 15 / FinePDFs long 15, 1.5B tokens |
| `MrCogito-sft/docs/experiments_specs/ahead/E37_sft_sensitivity.md:39-66` | the instruction-tuning data (smol-smoltalk + 10 % LongDocs replay) |

Not found on arch-10m or e37-sft: a spec for a new pretraining mix, a books / papers / code-repository tier beyond
LongDocs-lite, or a data plan for the meridian21 demo.
