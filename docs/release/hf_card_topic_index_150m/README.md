---
language:
  - en
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
tags:
  - mrcogito
  - meridian21
  - custom_code
  - topic-index
  - latent-sparse-attention
  - lsa
  - size-140m
  - budget-11.5bt
  - long-context
  - base-model
  - research
  - pytorch
  - e38                        # internal experiment code (secondary label)
datasets:
  - nvidia/Nemotron-CC-v2
  - nvidia/Nemotron-CC-Math-v1
  - nvidia/Nemotron-Pretraining-Specialized-v1
  - HuggingFaceTB/dclm-edu
  - common-pile/stackv2_edu_filtered
  - common-pile/project_gutenberg_filtered
  - common-pile/arxiv_papers_filtered
  - common-pile/doab_filtered
  - common-pile/usgpo_filtered
  - common-pile/uk_hansard_filtered
  - allenai/dolma3_longmino_pool
  - HuggingFaceTB/smol-smoltalk
model-index:
  - name: topic-index-140m-11.5bt
    results:
      - task: {type: text-generation, name: Grammar (BLiMP, 67 sets, 0-shot)}
        dataset: {type: blimp, name: BLiMP}
        metrics: [{type: accuracy, value: 80.11, name: accuracy (%)}]
      - task: {type: text-generation, name: Science quiz (ARC-Easy, 0-shot)}
        dataset: {type: allenai/ai2_arc, name: ARC-Easy, config: ARC-Easy, split: test}
        metrics: [{type: accuracy, value: 48.40, name: accuracy (%)}]
      - task: {type: text-generation, name: Language modelling (WikiText-2)}
        dataset: {type: wikitext, name: WikiText-2}
        metrics: [{type: perplexity, value: 1.950, name: byte perplexity (lower is better)}]
---

# topic-index-140m-11.5bt: a small base model that reads 64k tokens with Latent Sparse Attention

**`Meridian21Lab/topic-index-140m-11.5bt`** · [Project page](https://ai.ksopyla.com/projects/concept-reasoning/) ·
[Code](https://github.com/ksopyla/MrCogito/tree/1fddc505) · [Evaluation details](EVALUATION.md) ·
[Training on W&B](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_perceiver_H896L3g1s9N128_20261118_083324_lsa_m120t_long) ·
Same data, other reads: [concept-memory-140m-11.5bt](https://huggingface.co/Meridian21Lab/concept-memory-140m-11.5bt) ·
[catalogue-150m-11.5bt](https://huggingface.co/Meridian21Lab/catalogue-150m-11.5bt) ·
[full-attention-140m-11.5bt](https://huggingface.co/Meridian21Lab/full-attention-140m-11.5bt)

A 141.5M-parameter English base language model, trained from scratch on 11.5B tokens of public data for about
$290. Its long-range read is **Latent Sparse Attention (LSA)**, the method of Kubicek et al. (Prefrontal AI and
ETH Zürich, [arXiv 2610.42147](https://arxiv.org/abs/2610.42147),
[code](https://github.com/Prefrontal-AI/latent-sparse-attention)): it sorts the text it has already read into small
groups of tokens with related content, and for every next word it looks up the groups that match best and rereads
those tokens word for word. This model is our port of their released read, trained from scratch in our own
architecture, and we name it after what the read does: a **topic index**. It is not a model from the LSA authors;
the method and its name are theirs.

It is a public model of [MrCogito](https://ai.ksopyla.com/projects/concept-reasoning/), a project building models that
compress long text into concepts and reason over them. A research release: base model, one training seed, not
instruction-tuned.

Meridian21 trains several models on the same data, in the same order, for the same budget, with the same layers,
and changes only the reading layer, so each card compares them on the same tests. Released so far: this model,
[concept-memory-140m-11.5bt](https://huggingface.co/Meridian21Lab/concept-memory-140m-11.5bt) (a concept memory),
[catalogue-150m-11.5bt](https://huggingface.co/Meridian21Lab/catalogue-150m-11.5bt) (a learned catalogue) and
[full-attention-140m-11.5bt](https://huggingface.co/Meridian21Lab/full-attention-140m-11.5bt), the same layers with
a reading layer that rereads every earlier token: the reference inside the 64k training length, which the memory
reads are built to go past. All four tie on the seven standard tasks (mean 46.2 to 46.5 %); this model finds the
most facts far back of the four (RULER at 8k 0.54, against 0.41–0.44) and reads long documents level with the
concept memory (0.797 vs 0.795 bits per byte). Inside 64k it costs nothing against full-attention-140m-11.5bt on
long documents (0.797 against 0.812) or on the standard tasks (46.5 against 46.2 %).

## Highlights

- **Finds facts far back as well as a 350M model trained on 1,300 times more text.** RULER, 13 tasks that hide
  facts in a long text and ask for them (score = share of expected answers found, 1 = all), at 8k tokens: **0.54**,
  level with Granite 4.0 350M (0.54; ≈15T training tokens; it leads at 4k, 0.62 against 0.54) and ahead of
  LFM2.5-350M-Base (0.42; 28T), Gemma 3 270M (0.33; 6T) and SmolLM2-135M (0.30; ≈2T), all scored by us with the same
  harness.
- **Ahead of the other memory models on the same data at every length.** 0.54 / 0.54 / 0.44 / 0.39 at 4k / 8k /
  16k / 32k tokens against concept-memory-140m-11.5bt's 0.49 / 0.44 / 0.37 / 0.32 (+0.05 to +0.10) and
  catalogue-150m-11.5bt's 0.43 / 0.41 / 0.36 / 0.29 (+0.08 to +0.13), with 95 % intervals above zero at every
  length. Against the concept memory the gain is where several facts are asked at once (+0.12 to +0.34) and in
  counting words across the text (+0.11 to +0.25); on single hidden facts the two tie.
- **Reads long documents as well as concept-memory-140m-11.5bt, and holds up better past them.** On the last pages of 55 held-out
  64k-token documents it needs **0.797 bits per byte** (lower is better; scale below): level with the concept memory
  (0.795) and Gemma 3 270M (0.805, twice our size), ahead of every same-size model (SmolLM2-135M 0.832, Pythia-160M
  0.942). At 128k tokens, twice its training length, it beats the concept memory by 0.025 nats per token on 13 of 15
  documents.
- **The new read costs nothing on short text.** Mean of seven standard tasks 46.5 % (full-attention-140m-11.5bt
  46.2 %, concept-memory-140m-11.5bt 46.2 %), WikiText byte perplexity 1.950 (1.955 and 1.950), grammar (BLiMP)
  80.11 % (80.13 % and 80.56 %); against Pythia-160M at our training budget it is ahead on every test.
- **Cheap, open, and one line to load.** ≈36 hours on one 4-GPU node (≈ $290). It loads with
  `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and scores a whole 64k document in one pass on a
  single 24 GB RTX 3090 (with `logits_to_keep=1`, about 4 s); every training source is a public Hugging Face
  dataset, every training stage is on public W&B, and every number on this card has its raw result file in this
  repository.

<p align="center"><img src="figures/radar.png" width="560" alt="Radar chart of six standard benchmarks for topic-index-140m-11.5bt (this model), full-attention-140m-11.5bt and SmolLM2-135M at its 252B-token checkpoint. The two 11.5B-token models overlap almost everywhere; SmolLM2 is further out on the science quizzes, HellaSwag and WikiText, level on grammar and SciQ."></p>

*Six standard benchmarks, same harness for all three, each score as a share of the best of the three (rim = best).
This model and full-attention-140m-11.5bt trained on 11.5B tokens each, SmolLM2-135M on 252B at this checkpoint.
Every number, the other models on the same data, Pythia-160M and LFM2.5-350M-Base included, in the table below.*

## Results at a glance

All numbers in this table come from **one harness**, run by us on RTX 3090 cards (2026-11-17/18; full attention
2026-11-21): EleutherAI's lm-evaluation-harness 0.4.11, 0-shot, all test items, max length 2,048; each model in its
training precision (our models, SmolLM2 and LFM2.5 in bf16, Pythia in fp32). Long documents: our cross-model test
(below). Accuracy in %, higher is better; best in each row in **bold**.

| Measure | Chance | **topic-index-140m-11.5bt** · 11.5B tokens | full-attention-140m-11.5bt ¹ · 11.5B | SmolLM2-135M · 252B ² | Pythia-160M · 300B ³ | LFM2.5-350M-Base · 28T ⁴ |
|---|---:|---:|---:|---:|---:|---:|
| Grammar (BLiMP, 67 sets) | 50 | 80.11 ± 0.14 | 80.13 | 79.72 | 77.48 | **80.17** |
| Science quiz (ARC-Easy) | 25 | 48.40 ± 1.03 | 49.75 | 61.62 | 43.64 (43.52) | **70.03** |
| Hard science quiz (ARC-Challenge) | 25 | 21.8 | 21.0 | 27.1 | 19.5 (18.8) | **38.2** |
| Sensible continuation (HellaSwag) | 25 | 30.1 | 30.3 | 32.6 | 28.4 | **37.1** |
| Science with a passage (SciQ) | 25 | 77.6 | 78.9 | 80.4 | 75.4 (74.1) | **90.1** |
| Last word of a passage (LAMBADA) | ≈0 | 33.3 | 31.3 | 35.0 | 35.4 (32.8) | **35.8** |
| Physical common sense (PIQA) | 50 | 63.4 | 62.1 | 65.9 | 62.3 (62.7) | **66.6** |
| Pronoun resolution (WinoGrande) | 50 | 50.7 | 50.0 | 50.9 | 51.2 (53.1) | **55.6** |
| **Mean of the 7 tasks above** | 28.6 | 46.5 | 46.2 | 50.5 | 45.1 | **56.2** |
| Wikipedia text (WikiText-2), bits per byte ↓ | — | 0.964 | 0.967 | **0.905** | 0.944 | 1.078 |
| **Long documents**, bits per byte ↓ (most context the model reads) | — | **0.797** (61k) | 0.812 (61k) | 0.933 (2k) | 0.942 (2k) | 0.879 (4k, its best) |
| **Gain from far context**, 1k → most, bits per byte ↑ | 0 | **0.049** | 0.037 | 0 (reads 2k) | 0 (reads 2k) | −0.035 (worse with more) |
| **Finding facts far back** (RULER at 8k, mean of 13) ↑ | ≈0 | **0.54** | 0.41 | — (reads 2k) | — (reads 2k) | 0.42 |
| Tiny-ML Leaderboard place (of 101) ⁵ | | 26th | 23rd | 9th | 44th (over the 150M cap) | 12th (over the 150M cap) |

¹ [full-attention-140m-11.5bt](https://huggingface.co/Meridian21Lab/full-attention-140m-11.5bt), Meridian21's
full-attention model on the same data: same data, order, budget and layers, with a reading layer that rereads every
earlier token (full attention, rotary positions). Inside its 64k training length it is the reference for the memory
reads, which are built for the lengths beyond it; differences inside 64k show what a read costs, not a contest. The
other two memory models are in the paragraph below and in EVALUATION.md.
² SmolLM2-135M's **earliest public training checkpoint** (step 240,000, ≈252B tokens, 22× ours), the closest to our
budget a reader can check; its final ≈2T-token model is in the leaderboard table below. These early checkpoints read
2,048 tokens; SmolLM2's 8k context came in its last training phase.
³ Pythia-160M's final checkpoint, scored by us in fp32 (it loses ≈6 ARC-Easy points in bf16). In brackets:
EleutherAI's own published zero-shot results (`evals/pythia-v1/pythia-160m` in their GitHub repository, quoted).
⁴ LFM2.5-350M-Base (Liquid AI, March 2026), 2.5× our size; training tokens as its card reports (quoted). Its card
states a 32k context; on these documents it predicts best with 4k and worse with more (checked in fp32 and bf16, two
attention kernels, Transformers 4.57 and 5.23).
⁵ The [Tiny-ML Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard) formula reproduced from
its page (snapshot of 2026-11-14), each of our runs placed on its own. WikiText-2 is reported there as byte perplexity:
1.950 for this model (bits per byte = log₂ of it).

**What "bits per byte" means.** The number of bits a model needs, on average, to store each byte (about one
character) of text it has not seen, using its own predictions: 0 would be perfect prediction, 8 is plain text with no
prediction at all, and lower is better. There is no 0-to-1 scale. On the same pages, gzip needs 2.49 bits per byte,
xz 1.97 and bzip2 1.77; this model needs 0.797, so it could store those pages in about a tenth of their
size. It counts bytes rather than tokens, so models with different tokenizers compare fairly. At these values a
difference of 0.01 is about 1.2 %.

**Against the other models on the same data** (same tests, documents and settings): RULER 0.05–0.10 above
concept-memory-140m-11.5bt (several facts at once, counting words) and 0.08–0.13 above catalogue-150m-11.5bt
(distractor keys, several facts at once) at every length; level with both on the seven standard tasks (46.5 % against 46.2 % and
46.5 %) and WikiText (0.964 against 0.964 and 0.966 bits per byte); long documents level with the concept memory
(0.797 against 0.795 bits per byte, concept-memory-140m-11.5bt better on 62 % of documents) and 0.012 ahead of the
catalogue (0.808, this model better on 96 %); grammar 0.45 points below the concept memory (80.11 % against 80.56 %)
and 0.53 below the catalogue (80.64 %). Against full-attention-140m-11.5bt, the reference inside 64k: level on the
seven standard tasks (46.5 % against 46.2 %), grammar (80.11 % against 80.13 %) and WikiText (0.964 against 0.967);
no cost on long documents (0.797 against 0.812 bits per byte, this model lower on 84 % of documents); RULER 0.08
higher at 4k and 0.13 at 8k, and at 16k and 32k full attention's score falls further (0.25 and 0.20 against 0.44 and
0.39). Paired differences with intervals: [EVALUATION.md](EVALUATION.md#models-on-the-same-data).

## Long documents

<p align="center"><img src="figures/long_docs.png" width="860" alt="Bits per byte on the last pages of 55 held-out 64k-token documents against the context each model is given. This model and concept-memory-140m-11.5bt lie on top of each other and improve steadily up to 61k tokens, ending slightly ahead of catalogue-150m-11.5bt and full-attention-140m-11.5bt, which gains little past 28k; same-size models stop at 2k and 8k; Gemma 3 270M is level up to its 32k limit; Granite 4.0 350M is ahead; LFM2.5-350M-Base gets worse past 4k."></p>

Every model reads the same characters: the last ≈16 KB of each 64k-token span is the target, and the context before
it is cut to the budget on the x-axis (sliding windows past a model's maximum length). Held-out documents from five
public sources (Gutenberg books, arXiv papers, open-access academic books, US government publications, UK
parliamentary debates); our models trained on other documents of these sources, the others may have seen these very
documents. One RTX 3090; 2026-11-18, catalogue-150m-11.5bt and full-attention-140m-11.5bt 2026-11-21.

| Model | Size | Training tokens | Reads at most | Bits per byte ↓ | Gain, 1k → most ↑ | Another document's far past ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **topic-index-140m-11.5bt** | 141.5M | 11.5B | 64k | 0.797 | 0.049 [0.035, 0.065] | 0.834 |
| concept-memory-140m-11.5bt | 141.5M | 11.5B | 64k | 0.795 | **0.051** [0.037, 0.067] | 0.832 |
| catalogue-150m-11.5bt | 145.6M | 11.5B | 64k | 0.808 | 0.042 [0.030, 0.056] | 0.842 |
| full-attention-140m-11.5bt | 141.5M | 11.5B | 64k | 0.812 | 0.037 [0.027, 0.049] | 0.835 |
| SmolLM2-135M, 252B checkpoint | 134.5M | 252B | 2k | 0.933 | — | 0.932 |
| SmolLM2-135M, final | 134.5M | ≈2T | 8k | 0.832 | 0.018 [0.014, 0.024] | 0.836 |
| Pythia-160M | 162.3M | 300B | 2k | 0.942 | — | 0.941 |
| Gemma 3 270M | 268.1M | 6T | 32k | 0.805 | 0.038 [0.029, 0.049] | 0.827 |
| Granite 4.0 350M base | 352.4M | ≈15T | 32k | **0.776** | 0.041 [0.033, 0.051] | **0.798** |
| LFM2.5-350M-Base | 354.5M | 28T | 32k | 0.879 at 4k; 0.928 at 28k | −0.035 [−0.051, −0.020] | 0.933 |

Training tokens as each card reports them (quoted); 95 % bootstrap intervals over documents. "Another document's
far past": the own last 4k tokens kept, everything older taken from a different document of the same source.

- **Ahead of every same-size model,** on 75 % (SmolLM2-135M final, +0.036) to
  98 % (SmolLM2-135M at 252B, +0.136) of documents, because it reads 8–30 times further back;
  level with Gemma 3 270M (+0.008 [−0.007, +0.023]) and ahead of LFM2.5-350M-Base at its best (+0.082 [0.059, 0.105]).
- **It gains from far context like the 32k models trained on 500–1,300 times more text, and keeps going:**
  0.036 bits per byte from 1k to 28k (Gemma 3 270M 0.038, Granite 4.0 350M 0.041), then 0.013
  more from 28k to 61k.
- **It reads the content.** With another document's far past it scores like a plain 4k cut, worse than the real past
  on 54 of 55 documents.
- **Next to the other models on the same data:** level with concept-memory-140m-11.5bt (0.002 bits per byte apart,
  the concept memory better on 62 % of documents) with the same gain from far context; 0.012 ahead of
  catalogue-150m-11.5bt (better on 96 % of documents). No cost against full-attention-140m-11.5bt, the reference inside
  64k (0.797 against 0.812, this model lower on 84 % of documents), which gains 0.003 from 28k to 61k against this
  model's 0.013. The reads differ in how they find facts (RULER below) more than in how well they predict text.

Our own tests of the read (context cut in our tokens, the far read switched off, 16k and 128k documents) are
in [EVALUATION.md](EVALUATION.md#our-own-long-context-tests). In short: switching the far read off costs
0.31 nats per token at 64k; past 64k, the training length, extra context no longer helps, so feed the
last ≈64k tokens; past 64k, this model holds up better than concept-memory-140m-11.5bt (at 128k, 0.025 nats per
token better with the whole document, on 13 of 15 documents) and catalogue-150m-11.5bt (0.015), and level with
full-attention-140m-11.5bt (0.005 apart, the interval includes zero).

## Finding facts far back (RULER)

The standard long-context suite: 13 tasks, 100 prompts per task and length, greedy decoding, each task's own answer
length; score = share of the expected answers found in the reply (0 to 1, higher is better). Every model runs through
the same harness, its prompts sized in its own tokens; RTX 3090 cards, 2026-11-18 (catalogue-150m-11.5bt and
full-attention-140m-11.5bt 2026-11-21). Not zero-shot for our models: two training stages included planted-fact
lookups of a similar shape.

<p align="center"><img src="figures/ruler_radar.png" width="900" alt="Four small radar charts of RULER at 8k tokens by task family, this model and full-attention-140m-11.5bt each against one reference model: SmolLM2-135M, Gemma 3 270M, Granite 4.0 350M and LFM2.5-350M-Base."></p>

| RULER at 8k, by task family | **topic-index-140m-11.5bt** | full-attention-140m-11.5bt | concept-memory-140m-11.5bt | catalogue-150m-11.5bt | SmolLM2-135M | Gemma 3 270M | Granite 4.0 350M | LFM2.5-350M-Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| One needle | 0.85 | 0.90 | 0.76 | 0.77 | 0.72 | 0.55 | **1.00** | 0.83 |
| Needle among distractor keys | 0.59 | 0.43 | **0.62** | 0.41 | 0.20 | 0.34 | 0.42 | 0.36 |
| Several values or keys at once | 0.67 | 0.35 | 0.33 | 0.28 | 0.28 | 0.29 | **0.70** | 0.51 |
| Count words | **0.45** | 0.14 | 0.29 | 0.42 | 0.11 | 0.34 | 0.31 | 0.05 |
| Follow variables | 0.00 | 0.00 | 0.01 | 0.03 | 0.06 | 0.00 | **0.17** | 0.13 |
| Answer questions | 0.23 | 0.16 | 0.17 | 0.17 | 0.15 | 0.15 | 0.27 | **0.31** |
| **Mean of the 13 tasks, 8k** | **0.54** | 0.41 | 0.44 | 0.41 | 0.30 | 0.33 | **0.54** | 0.42 |
| Mean of the 13 tasks, 4k | 0.54 | 0.46 | 0.49 | 0.43 | 0.40 | 0.48 | **0.62** | 0.47 |
| Mean of the 13 tasks, 16k | **0.44** | 0.25 | 0.37 | 0.36 | — | — | — | — |
| Mean of the 13 tasks, 32k | **0.39** | 0.20 | 0.32 | 0.29 | — | — | — | — |

- **Level with Granite 4.0 350M at 8k** (0.54 each; 2.5× our size, ≈1,300× our training text) and ahead of
  LFM2.5-350M-Base, Gemma 3 270M and SmolLM2-135M; at 4k Granite leads (0.62 against 0.54) and finds 32-character
  IDs that this model rarely writes out.
- **Against concept-memory-140m-11.5bt:** +0.05 [0.03, 0.07], +0.10 [0.08, 0.12], +0.07 [0.05, 0.08] and +0.07 [0.05, 0.09] at
  4k, 8k, 16k and 32k (95 % bootstrap over prompts; paired from 4k to 16k; at 32k eight of the 13 tasks are two
  separate 100-prompt draws, compared unpaired). The gain is in several facts at once and in counting words; concept-memory-140m-11.5bt
  leads on needles among distractor keys (0.62 against 0.59).
- **Against catalogue-150m-11.5bt:** +0.11 [0.09, 0.13], +0.13 [0.11, 0.15], +0.08 [0.06, 0.10] and +0.10 [0.08, 0.12]
  at 4k, 8k, 16k and 32k (paired at every length), most on distractor keys and several facts at once; the two count
  words about equally. **Against full-attention-140m-11.5bt** (the no-harm check inside 64k): 0.08 [0.06, 0.10] higher
  at 4k and 0.13 [0.11, 0.15] at 8k; at 16k and 32k full attention's score falls further (0.25 and 0.20 against 0.44
  and 0.39).
- **It keeps going past 8k:** mean 0.44 at 16k and 0.39 at 32k, with the essay needle at 1.00 and 0.98 (the reference
  models were run at 4k and 8k only). Every task at every length: [EVALUATION.md](EVALUATION.md#ruler).

<p align="center"><img src="figures/ruler_lengths.png" width="860" alt="Six small charts, one per RULER task family, share of answers found against context length from 4k to 32k tokens for this model and the three other models on the same data. topic-index-140m-11.5bt is highest for several values or keys at once and for counting words, level with concept-memory-140m-11.5bt on distractor keys; full-attention-140m-11.5bt falls fastest past 8k."></p>

*RULER by task family, 4k to 32k tokens: this model (orange) and the other models on the same data (dashed), same
harness and settings, 100 prompts per task and length (the same prompts for all four from 4k to 16k).*

## Tiny-ML Leaderboard

The board ranks models up to 150M parameters on grammar (BLiMP), the easy science quiz (ARC-Easy) and Wikipedia text
(WikiText-2 byte perplexity, lower is better), with a bonus of up to 50 % for small size. Board entries are the
board's own runs (quoted, snapshot 2026-11-14); rows in *italics* are our runs of other models, each placed on its own.

| Place | Model | Size | Training tokens | BLiMP % ↑ | ARC-Easy % ↑ | WikiText-2 byte ppl ↓ | Quality ⁶ | Board score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Tiny Chick 10M V3 | 9.6M | 24.6B | 77.76 | 59.60 | 2.335 | 77.8 | 87.7 |
| 2 | Tiny Chick 10M V2.1 | 9.6M | 16.3B | 76.79 | 53.79 | 2.342 | 75.5 | 85.2 |
| 3 | Tiny Chick 10M V2 | 9.6M | 16.3B | 76.99 | 50.21 | 2.364 | 74.3 | 83.8 |
| 4 | Slayer139 1.02 | 139.3M | 24.0B | 79.22 | 65.91 | 1.862 | 81.7 | 83.4 |
| *(4)* | *SmolLM2-135M, final* | *134.5M* | *≈2T* | *80.15* | *64.27* | *1.799* | *81.5* | *83.2* |
| 5 | Boris-2-Think | 143.7M | 200B | 82.09 | 63.93 | 1.967 | 81.7 | 83.2 |
| 6 | Slayer149-ARC | 149.3M | 31B | 78.38 | 66.58 | 1.907 | 81.5 | 82.9 |
| 7 | Tiny Chick 10M | 9.6M | 8.3B | 75.12 | 48.53 | 2.287 | 73.3 | 82.7 |
| 8 | Slayer139 1.01 | 139.3M | 23.7B | 79.13 | 63.09 | 1.878 | 80.7 | 82.3 |
| *(9)* | *SmolLM2-135M, 252B checkpoint* | *134.5M* | *252B* | *79.72* | *61.62* | *1.873* | *80.4* | *82.1* |
| 9 | Glint-2 | 1.1M | ≈300B | 74.54 | 36.20 | 3.127 | 67.2 | 81.7 |
| 10 | Boris-2 | 143.7M | 200B | 81.13 | 59.76 | 1.895 | 80.2 | 81.7 |
| … | | | | | | | | |
| *(21)* | *catalogue-150m-11.5bt* | *145.6M* | *11.5B* | *80.64* | *50.42* | *1.954* | *76.7* | *78.1* |
| *(23)* | *full-attention-140m-11.5bt* | *141.5M* | *11.5B* | *80.13* | *49.75* | *1.955* | *76.3* | *77.8* |
| 23 | GPT-S-5M | 5.2M | 25B | 72.27 | 35.69 | 2.570 | 67.4 | 77.7 |
| *(24)* | *concept-memory-140m-11.5bt* | *141.5M* | *11.5B* | *80.56* | *48.95* | *1.950* | *76.2* | *77.7* |
| 24 | Glint-2.3 | 1.7M | 10B | 67.30 | 36.00 | 3.051 | 64.8 | 77.6 |
| 25 | Zedev-134M-Fudas | 134M | 20.7B | 81.69 | 46.38 | 1.880 | 76.0 | 77.6 |
| **26** | **topic-index-140m-11.5bt** | **141.5M** | **11.5B** | **80.11** | **48.40** | **1.950** | **75.9** | **77.4** |
| 27 | Hydrion-v1-Base | 114.1M | ≈2.5B | 80.08 | 47.26 | 2.040 | 75.2 | 77.4 |
| … | | | | | | | | |
| *(44)* | *Pythia-160M (over the 150M cap)* | *162.3M* | *300B* | *77.48* | *43.64* | *1.924* | *73.5* | *74.5* |

⁶ Quality: the mean of BLiMP, ARC-Easy and the board's WikiText score, before the size bonus. By quality alone the
model is 16th of 101; on grammar alone, 11th. Of the 55 board entries that report 15B
training tokens or fewer, four place above it. The other models on the same data place 21st (catalogue-150m-11.5bt),
23rd (full-attention-140m-11.5bt) and 24th (concept-memory-140m-11.5bt).
Full board with our rows: [EVALUATION.md](EVALUATION.md#tiny-ml-leaderboard).

## Data efficiency

"Data-efficient" here means a better score from the same number of training tokens, measured against the published
training checkpoints of Pythia-160M (one every 2.1B tokens) and SmolLM2-135M (from 252B tokens), all scored by us
with the same harness.

<p align="center"><img src="figures/data_eff.png" width="860" alt="Three panels, score against training tokens on a log scale: Pythia-160M's checkpoints, SmolLM2-135M's checkpoints, and the four models on the same data as points at 11.5B tokens."></p>

- **Same budget:** against Pythia-160M at 10.5B and 12.6B tokens, this model is ahead on every test: +1.8 to +2.2 points of grammar, +5.6 to +5.8 on the science quiz, +3.3 to +4.4 on the 7-task mean,
  and 0.07–0.09 lower WikiText byte perplexity.
- **Tokens to match:** Pythia-160M does not reach this model's grammar or science-quiz score at any checkpoint up to
  300B tokens (26× ours); it reaches its WikiText score after ≈36B tokens (3.2×). SmolLM2-135M stays below
  its grammar score at 252B and 503B tokens and is level at ≈2T (80.15 %); it is ahead on the knowledge tests from its first
  checkpoint.
- **What it does not separate:** architecture from data. Part of the gap to Pythia is newer, cleaner data:
  full-attention-140m-11.5bt, the same layers with full attention on the same data, lands within 1.4 points of this
  model on every test here (and concept-memory-140m-11.5bt and catalogue-150m-11.5bt within 2.1), so on these tests
  the gap comes from the data and the shared training recipe, not from the read.

## What its text looks like

Nucleus sampling (temperature 0.7, top-p 0.9, repetition penalty 1.1), 60 new tokens, seed 0, not picked; prompt in
bold, unedited, cut at a sentence end (…). More seeds and beam-search outputs: `samples/samples_ours.json`.

> **The capital of France is** Paris, and the capital of France is also the capital of the French Republic. As a
> result of the international treaty with Germany, France's capital was renamed to Paris in 1914. …
>
> **In 1905, Albert Einstein published** his famous work on the theory of relativity. His paper provides a new
> understanding of the relationship between matter and energy in a massive universe and the theory of general
> relativity. …
>
> **def fibonacci(n):**
> ```python
>  return (n + 1) // 2
>
> def fibonacci(n):
> return n + 1
> ```

Fluent, on-topic English and code-shaped output; the facts are often wrong (Paris was never renamed; the 1905 papers
were special, not general, relativity) and the code is wrong, as expected from a 140M base model at 11.5B tokens. The
samples of the other models on the same data read the same.

## How the topic index (LSA) works

Think of a student reading a long book for an exam. Rereading every page for every answer is what full attention
does: exact, but the cost grows with the square of the book's length. LSA gives the student a **library index by
topic**: every finished chapter is sorted into small piles of passages about the same thing, each pile gets a label,
and for each new question the student reads the labels, pulls the few best piles and rereads those passages word for
word.

```
 tokens ─► 3 local layers ─► ┌──── looped 4 times, shared weights ────┐ ─► 8 local layers ─► next token
          (last 128 tokens)  │ LSA reading layer ─► 1 local layer      │    (last 128 tokens)
                             └────────────────────────────────────────┘
 LSA reading layer, for each new token, one softmax over:
   the document's first 4 tokens + its last 512 tokens, word for word
   + up to 512 earlier tokens from the best-matching buckets, word for word
 buckets: each closed 512-token block's keys are summed in fours and merged, most similar pair first,
          into 64 buckets of at most 16 tokens; a bucket's label is the direction of its summed keys
```

A query compares itself with the label of every bucket in the blocks before its own, keeps the 128 best buckets and
takes their tokens, best bucket first, until it holds 512. The buckets are built without gradients and add no
parameters: the model learns what to look up only through its next-token loss. Our port keeps the authors' bucket
construction and selection and changes two things: a token reads only its own document (we pack several documents
into one training row), and the selected tokens become a mask for PyTorch's flex attention instead of the authors'
Triton kernel (a third, a runtime check moved to configuration time, changes nothing the model computes). The design
differences from the other memory reads on the same data: concept-memory-140m-11.5bt also reads a one-line summary of
every group it did not open, which LSA ignores; catalogue-150m-11.5bt files its groups with a small learned writer
instead of building them from the keys without gradients.

## Training

| | |
|---|---|
| **Parameters** | 141,506,208 (input and output share one token table, counted once); 122.6M without the 49,152 × 384 table |
| **Shape** | width 896; 13 distinct layers; one LSA reading layer; local layers see the last 128 tokens |
| **Tokenizer** | SmolLM2's (`HuggingFaceTB/SmolLM2-135M`), 49,152 tokens; every document starts with `<|endoftext|>` |
| **Budget** | 11.5B tokens in three length stages: 8.0B at 4k-token rows → 3.0B at 16k → 0.5B at 64k |
| **Optimizer** | Muon (peak 0.01) for the weight matrices, AdamW for the rest; bf16; seed 42 |
| **Machine** | AWS spot nodes with 4 × NVIDIA L40S (48 GB); one replacement after a spot interruption |
| **Time and cost** | ≈36 hours of node time (≈145 GPU-hours) including ≈0.6 h of tests and ≈40 min lost to the interruption; ≈ $290 at ≈ $8/h spot; energy not measured |
| **Finished** | 2026-11-18 13:06 UTC |

**Data (`m21mix_v2`), all public Hugging Face datasets**, the same shards in the same order as concept-memory-140m-11.5bt. Shares are
of each stage's tokens.

| Stage | Tokens | What it reads |
|---|---:|---|
| 1. Main, 4k rows, documents packed | 8.0B | [Nemotron-CC v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) high-quality web 30 % and its Diverse-QA rewrites 22 %, [DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu) (score ≥ 3) 12 %, educational code ([Stack v2 Edu](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered)) 10 %, [Nemotron](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) Wikipedia rewrites 8 %, [Nemotron-CC-Math](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) (4+) 8 %, Nemotron reading-comprehension QA 6 %, Nemotron math textbooks 4 % |
| 2. Decay, 16k rows, one document each | 3.0B | **long documents 60 %**: science PDFs ([Dolma 3 Longmino](https://huggingface.co/datasets/allenai/dolma3_longmino_pool)) 22 %, [Gutenberg](https://huggingface.co/datasets/common-pile/project_gutenberg_filtered) books 12 %, code packed per repository 12 %, [arXiv](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered) 5 %, open-access books ([DOAB](https://huggingface.co/datasets/common-pile/doab_filtered)) 3 %, [US government documents](https://huggingface.co/datasets/common-pile/usgpo_filtered) 3 %, [UK Hansard](https://huggingface.co/datasets/common-pile/uk_hansard_filtered) 3 %; web replay 20 %; Nemotron InfiniByte reasoning 7 %; multi-hop "libraries" 6 %; planted-fact lookups 3 %; chat ([smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)) 4 % |
| 3. Long, 64k rows, one document each | 0.5B | science PDFs 30 %, books 25 %, code repositories 15 %, multi-hop libraries 10 %, lookups 5 %, web replay 5 %, arXiv 4 %, DOAB 3 %, US GPO 3 % |

The two constructed row types (about 3 % of all tokens): *multi-hop libraries*, the paragraphs of 20–160
2WikiMultihopQA training questions shuffled, with the questions and answers at the end; and *lookups*, 4–24 invented
facts planted in a real book passage and asked at the end. The test sets on this card (BLiMP, ARC, WikiText,
HotpotQA, SQuAD, the held-out long documents) are not in the training data. Stage-by-stage settings and losses:
[EVALUATION.md](EVALUATION.md#training-procedure).

## Scope and trade-offs

- **Base model.** No instruction or preference tuning; about 1 % of training text was chat transcripts rendered as
  plain text (the tokenizer folder carries a ChatML template for that reason). English, plus code.
- **Knowledge tests** (ARC, SciQ, HellaSwag, LAMBADA) are below SmolLM2-135M, which trained on 22× (at 252B) to
  ≈170× (final) more text, and LFM2.5-350M-Base (2.5× our size, ≈2,400× our text); at this size these tests track
  training tokens closely.
- **Next to the other memory models on the same data:** grammar 0.45 points below concept-memory-140m-11.5bt (80.11 %
  against 80.56 %, about two standard errors) and 0.53 below catalogue-150m-11.5bt; level on the other standard
  tests; long documents level with the concept memory (0.002 bits per byte behind) and 0.012 ahead of the catalogue;
  RULER 0.05–0.13 higher at every length; past 64k this model holds up better than both (0.025 and 0.015 nats per
  token at 128k).
- **Next to full-attention-140m-11.5bt** (inside 64k, where full attention is the reference): level on the standard
  tests; no cost on long documents (0.797 against 0.812 bits per byte); RULER 0.54 / 0.54 / 0.44 / 0.39 at 4k / 8k /
  16k / 32k against full attention's 0.46 / 0.41 / 0.25 / 0.20. At 128k, past both models' training length, the two
  are level (0.005 nats per token apart, the interval includes zero).
- **Speed of the read:** the 64k training stage (0.5B tokens) took 4.5 hours, against 1.7 hours for
  full-attention-140m-11.5bt on the same tokens, and scoring one 64k-token text takes about 4 s against 1.2 s (one
  RTX 3090); the read scores one bucket per eight earlier tokens, so its search still grows with length.
- **Context:** trained and tested to 64k tokens. Past 64k, extra context no longer improves predictions (at 128k, the
  whole document is 0.031 nats per token worse than its last 61k, on 14 of 15 documents); feed
  the last ≈64k.
- **Hard RULER tasks:** variable tracking stays at 0.00–0.01 at every length, the UUID tasks at 0.02–0.55, and most
  tasks fall with length.
- **One training seed;** differences of a point or two between the models on the same data are within what a second
  seed could move.
- **Facts and safety:** it makes things up, and has no safety tuning or output filtering; not for high-stakes or
  user-facing use.

## What's next

- **Past 64k:** the lengths the memory reads are built for and full attention cannot afford.
- **Longer budgets and larger models** with the best of the reads on the same data.
- **Reasoning over concepts** (built, switched off in all releases so far).

## How to use

Needs `torch` and `transformers` (tested with 4.57.6 and 5.23.0). The modeling code ships in this repository, so
loading takes `trust_remote_code=True`. Runs on one CUDA GPU in bf16.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Meridian21Lab/topic-index-140m-11.5bt"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")

inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9,
                     repetition_penalty=1.1)
print(tok.decode(out[0], skip_special_tokens=True))

# scoring: mean next-token loss in nats per token
loss = model(**inputs, labels=inputs["input_ids"]).loss
```

- `generate` is 🤗 Transformers' own (sampling, greedy, beam search), running over the model's cached decoder. This
  is a base model, not a chat model.
- **Long documents** (up to the 64k training length): add `attn_backend="flex"` to `from_pretrained` (PyTorch flex
  attention on CUDA). The default, `sdpa`, builds a dense attention mask and suits short inputs. To score a long
  document, pass `labels` and `logits_to_keep=1`: the loss is computed in chunks, and a whole 64k-token document
  takes one pass, about 4 s and 2.8 GB of GPU memory on an RTX 3090 (the logits of all 64k positions alone would
  take 12 GB, more than the card holds next to the model).
- For exact scores, batch prompts of one length or score them one at a time: left padding moves the reading layer's
  block edges and changes scores slightly.
- `modeling_perceiver_ar.py` is generated from the MrCogito source (`export_manifest.json` names the commit) and
  includes the LSA authors' bucket code with its licence notice. Evaluation code:
  [`832f4eeb`](https://github.com/ksopyla/MrCogito/tree/832f4eeb); training: [`1fddc505`](https://github.com/ksopyla/MrCogito/tree/1fddc505).

## Citation

```bibtex
@misc{sopyla2026topicindex150m,
  author       = {Sopyła, Krzysztof},
  title        = {topic-index-140m-11.5bt: a small base language model with Latent Sparse Attention for long context},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Meridian21Lab/topic-index-140m-11.5bt}},
  note         = {Meridian21 Lab, MrCogito project. Code: \url{https://github.com/ksopyla/MrCogito}}
}

@misc{kubicek2026lsa,
  author        = {Kubicek, Ales and Caciolai, Andrea and Peng, Liangzu and Sumant, Sukrit and Hoover, Benjamin and
                   Greenlee, Fraser and Battleday, Ruairidh M. and Urban, Radim and Hoefler, Torsten and
                   Whittington, James C. R.},
  title         = {Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention},
  year          = {2026},
  eprint        = {2610.42147},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

@online{sopyla2026mrcogito,
  author  = {Sopyła, Krzysztof},
  title   = {MrCogito: Concept Reasoning Model},
  year    = {2026},
  url     = {https://ai.ksopyla.com/projects/concept-reasoning/},
  urldate = {2026-11-18}
}
```

## Licence

Weights, card and code: **Apache License 2.0** (text in `LICENSE`). `modeling_perceiver_ar.py` contains Prefrontal
AI's LSA bucket code (Apache-2.0, copyright 2026 Prefrontal AI), ported with the changes listed in its header. The
training data come from public Hub datasets under their own licences: DCLM-Edu (CC BY 4.0), Dolma 3 Longmino (ODC-By),
smol-smoltalk (Apache-2.0), the openly licensed Common Pile sources, Nemotron-Pretraining-Specialized (CC BY 4.0, CC
BY-SA 4.0, GFDL and NVIDIA terms per subset), and NVIDIA's Nemotron-CC-v2 and Nemotron-CC-Math-v1 (gated on the Hub
under NVIDIA's terms; about 45 % of the training tokens). See each dataset's card for its terms.

## Changelog and contact

- 2026-11-18: first release (weights, modeling code, card, evaluation page, raw results, figures).
- 2026-11-18: aligned with the concept-memory-140m-11.5bt card: concept-memory-140m-11.5bt in every results table and chart, one colour per
  model across both cards, a head-to-head section in EVALUATION.md.
- 2026-11-21: catalogue-150m-11.5bt and full-attention-140m-11.5bt added to every comparison (tables, figures,
  EVALUATION.md); figures redrawn with one colour per model across the four cards, SmolLM2 now grey like the other
  outside models; long-document scoring tip (`logits_to_keep=1`) in How to use.

Krzysztof Sopyła · [ai.ksopyla.com](https://ai.ksopyla.com) · [Meridian21 Lab](https://huggingface.co/Meridian21Lab)
· questions and errors: open a discussion on this page.
