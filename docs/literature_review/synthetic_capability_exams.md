# Synthetic capability exams

Controlled tasks researchers actually train or score models on when they want to
*measure a mechanism*, not “language”. Home for RULER / NIAH, BABILong / bAbI,
induction-head copy & reverse, the Chomsky-hierarchy suite, MAD / MQAR, TinyStories,
SCAN / CFQ, and Dyck / RASP. The BAPO *theory* review stays in
[`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md).
32k compressor-eval protocols (HELMET, LongBench v2, retrieval-head knockout,
no-context controls) are appended at the end of this file. Learned KV
compressors those exams score live in
[`learned_kv_context_compression.md`](learned_kv_context_compression.md).

Project mapping: DNA A=4 (`data/symbolic_tasks.py`) is the exact-floor bandwidth
instrument. Glyph (`data/glyph_tasks.py`) is the typed-vocab / structured-noise family
this review motivates. Spec:
[`docs/4_Research_Notes/glyph_capability_ladder.md`](../4_Research_Notes/glyph_capability_ladder.md).

---

## Lost in Transmission: When and Why LLMs Fail to Reason Globally (BAPO)

NeurIPS 2025 · [arXiv:2505.08140](https://arxiv.org/abs/2505.08140) ·
code [github.com/microsoft/bapo](https://github.com/microsoft/bapo) ·
Schnabel, Tomlinson, Swaminathan, Neville.

### TL;DR
Global reasoning fails when prefix→suffix bandwidth is a small constant. INDEX and
MATCH2 are BAPO-easy; REACHABILITY, MAJORITY, MATCH3, UNIQUE are hard. The paper
cannot measure `a` inside a model — we can, against a closed-form floor.

### The exam they actually ran
API models (GPT-4o, Claude, Gemini) on synthetic instances of size `n ∈ {6, 50, 100, 200}`:
INDEX (pointer into a list), MATCH2 / MATCH3 over `Z_m`, graph REACHABILITY, majority
over a bitstring, UNIQUE / SETDIFF (hardness scales with `|Σ|`). Easy tasks stay near
100%; hard tasks fall to chance by `n=200`. CoT turns hard tasks easy at the cost of
1k–100k tokens. They do **not** use a 4-symbol DNA alphabet — they use integer lists
and English-ish prompts. Our DNA re-implementation is an Adapt of the *hardness
classes*, not a clone of their prompts.

### Verdict
**Adopt** the easy/hard split and the `(a, b)` scoring. **Adapt** INDEX / MATCH2 /
REACHABILITY onto a closed alphabet we can train from scratch. Full review:
[`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md).

---

## RULER: What’s the Real Context Size of Your Long-Context Language Models?

COLM 2024 · [arXiv:2404.06654](https://arxiv.org/abs/2404.06654) ·
code [github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER) ·
Hsieh et al. (NVIDIA). Extends Kamradt’s NIAH
([github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)).

### TL;DR
Vanilla needle-in-a-haystack is a shallow retrieval exam. RULER adds multi-key /
multi-value / UUID needles, variable-tracking (coreference chains), and
common/frequent-word extraction, with haystacks that are *not* iid: repeated noise
sentences, Paul Graham essays, or distractor needles.

### The exam they actually ran
- **S-NIAH:** `"the special magic number for XXX is: YYY"` planted in a haystack.
  Haystack types: `repeat` (fixed noise sentences), `essay` (PG essays), `needle`
  (other needles as distractors). Keys/values: words, 7-digit numbers, UUIDs.
- **Variable tracking:** `X1 = V; X2 = X1; …` with `num_chains` and `num_hops`,
  inserted into the same haystacks. This is REACHABILITY with English bindings —
  BAPO names it as a special case.
- **CWE / FWE:** aggregate over Zipf-distributed word bags (summarisation proxy).

Almost every advertised 32k model is near-perfect on vanilla NIAH and collapses as
length *and* needle-type complexity grow.

### Verdict
**Adapt** the haystack axis (structured / language-like vs iid), not the English
essays. Glyph’s `noise=markov|dyck|arith|mixed` is the closed-vocab version of
`type_haystack: essay` vs `noise`. **Watch** CWE/FWE until we have a compressive
channel that might win at aggregation (DNA `count` / `majority` already exist).
**Reject** scoring E18 on English RULER as the *first* instrument — unknown prize,
no exact floor.

---

## BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack

NeurIPS 2024 · [arXiv:2406.10149](https://arxiv.org/abs/2406.10149) ·
code [github.com/booydar/babilong](https://github.com/booydar/babilong) ·
Kuratov, Bulatov, Anokhin, et al. Facts from Weston et al. **bAbI**
([arXiv:1502.05698](https://arxiv.org/abs/1502.05698)).

### TL;DR
Hide bAbI facts (`Mary travelled to the office`) inside PG-19 books. The model must
*distinguish* task sentences from a closely related narrative distribution, then
chain them. That is “structured noise” as a first-class exam, not iid filler.

### The exam they actually ran
20 bAbI skills (single-fact QA, two-argument relations, three-argument, two-fact
coreference, three-fact, yes-no, counting, lists/sets, negation, indefinite
knowledge, basic deduction/induction, positional reasoning, …) with facts inserted
between PG-19 sentences in natural order until the sample hits a target length
(splits to 10M tokens; they evaluate to 50M with RMT). Popular LLMs use ~10–20% of
the context; RAG tops out ~60% on single-fact QA independent of length.

### Verdict
**Adapt** the “plant a uniquely determined fact in a local-LM-plausible haystack”
recipe. Do **not** ingest English BPE or PG-19 into the E18 probe. Glyph
`fact_markov` / `story_fact` is the closed-vocab port: Markov letters/words, a
`mark` + key/value, verifier recovers the payload. **Watch** the full 20-skill
bAbI set — too many micro-A/Bs; pick the mechanisms (single-fact, chain, count)
we already isolate.

---

## In-context Learning and Induction Heads

[transformer-circuits.pub/2022/in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads)
· [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) · Olsson et al.

### TL;DR
A two-head circuit copies `[A][B] … [A] → [B]`. It needs *two layers* (previous-token
head composed into an induction head). One-layer attention-only models cannot form it.
They also study **in-context reverse** and random-token **copy** as mechanistic
unit tests.

### The exam they actually ran
Small attention-only transformers on synthetic token sequences: copy the previous
occurrence’s successor; reverse a repeated substring. The induction circuit is
content-addressed (`A` matches `A`), not positional. Formation coincides with a
loss-curve bump and a jump in in-context learning.

### Verdict
**Adapt** copy vs reverse as two rungs of one family (positional INDEX vs permute).
E18’s one global read is structurally close to “one layer that can see the prefix”
— Olsson says induction needs a *previous-token composition* below that read.
That is a mechanistic reason to expect E18 to copy and fail reverse or keyed
recall unless the pre-encoder writes usable keys. **Watch** fuzzy induction
(`[A*][B*]`) until the exact reverse rung is calibrated.

---

## Neural Networks and the Chomsky Hierarchy

ICLR 2023 · [arXiv:2207.02098](https://arxiv.org/abs/2207.02098) ·
code [github.com/deepmind/neural_networks_chomsky_hierarchy](https://github.com/deepmind/neural_networks_chomsky_hierarchy)
· Delétang, Ruoss, Grau-Moya, et al.

### TL;DR
Group synthetic transduction tasks by Chomsky level and you can *forecast* which
architectures generalise. RNNs solve regular (parity, modular arithmetic simple,
cycle navigation). Stack-RNNs solve DCF (reverse string, bracketed modular
arithmetic, stack manipulation). Tape-RNNs solve CS (duplicate string, odds-first,
binary addition). Transformers in their protocol fail to *length-generalise* on
non-regular tasks (reverse 62%, modular-with-brackets 32.5%, duplicate ~53%).

### The exam they actually ran
Closed alphabets `{a, b}` or digits plus brackets. Reverse String: read `w`, emit
`w` reversed after a separator. Modular Arithmetic (DCF): evaluate an expression
with nested brackets using a stack. Duplicate String / Odds First: context-sensitive
copying and selection. They train on short lengths and test longer — the metric is
OOD length generalisation, not teacher-forced CE on a packed span.

### Verdict
**Adapt** Reverse and Dyck/bracketed arithmetic as *in-distribution* packed-span
exams first (we need dense ≥ 75% before any E18 number). Length-generalisation is a
later kill, not the first gate. **Watch** PARITY / modular-simple: a compressive
channel *should* win (O(1) state) — DNA `count` already occupies that slot. **Reject**
binary multiplication / sqrt as the next rung (CS, uncalibrated, not E18-specific).

---

## Zoology / MQAR and MAD (Mechanistic Architecture Design)

- Arora, Eyuboglu, et al., *Zoology*: [arXiv:2302.06612](https://arxiv.org/abs/2302.06612)
  / ICLR 2024 paper
  [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/448fc91f669c15d10364ee01d512cc10-Paper-Conference.pdf);
  blog [hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis).
- Poli, Thomas, et al., *MAD*: [arXiv:2403.17844](https://arxiv.org/abs/2403.17844) ·
  code [github.com/athms/mad-lab](https://github.com/athms/mad-lab).

### TL;DR
Tiny-vocab associative recall (MQAR) and a six-task MAD suite (in-context recall,
fuzzy recall, noisy recall, selective copying, compression, memorization) are how
SSM / Hyena / Mamba papers decide whether an architecture can *look something up*.
Attention solves MQAR at constant width; gated convs need width ∝ sequence length.
MAD’s **selective copying** is “copy the data tokens in order, ignore inserted
noise” — exactly “filter / every-k-th” with a known noise process.
**Noisy recall** inserts noise tokens between KV pairs.

### The exam they actually ran
The MAD CLI default is `--vocab-size 16` ([`train.py`](https://github.com/athms/mad-lab/blob/0f49a452/train.py));
the paper sweeps **16 / 32 / 64 / 128**. That is the precedent for Glyph-16/32 —
not 32 arbitrary ids, but the same closed sizes MAD used as architecture unit tests.
Sequences are `(key, value)` pairs then queries; the mapping is fresh per row.
Selective copy: a subset of positions marked as data, the rest noise; emit the data
in order (seq 256–1024, 16–96 tokens copied). Noisy recall has a separate
`--noise-vocab-size` and `--frac-noise` (iid inserted tokens). Compression:
autoencoder reconstruction from a short latent. They train for minutes at width
128 and treat the score as a unit test predictive of LM scaling (MAD’s claim).

### Verdict
**Adapt** selective copy (Glyph `filter_mod`, `every_k`) and noisy recall (Glyph
`fact_markov` with Markov/Dyck filler). **Adopt** the “unit-test then scale” protocol
— we already have it as dense ≥ 75% then score E18. **Watch** MAD compression as a
future concept-array exam (sibling `perceiver_concept`, not E18). **Reject**
memorization (hash the training set into weights) — it tests parameter memory, not
the long-range channel.

---

## TinyStories: How Small Can Language Models Be and Still Speak Coherent English?

[arXiv:2305.07759](https://arxiv.org/abs/2305.07759) · Eldan & Li · dataset
[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

### TL;DR
A closed *child-level* English vocabulary (~1500 words, GPT-3.5/4 generated stories)
lets <10M-parameter models produce grammatical multi-paragraph stories. Depth helps
consistency; width helps facts. It is a *language* exam at tiny scale, not a
long-range retrieval exam.

### Verdict
**Adapt** the closed-word idea (`cat dog red blue`) as *planted-fact* keys inside
Glyph, not English BPE. **Reject** training E18 on TinyStories itself — unknown
prize, no min_gap, not a channel instrument. **Watch** GPT-4 grading of free-run
stories; that is E05/E17 territory, not E24.

---

## MiniPile

Kaddour, [arXiv:2304.08442](https://arxiv.org/abs/2304.08442). A 6 GB / 1M-document
subset of The Pile, clustered and quality-filtered. **Not** a tiny-vocab synthetic
and **not** a capability exam. **Reject** as an E24 instrument. Useful later as a
cheap LM mix, not as a closed-form floor.

---

## SCAN and CFQ (composition)

- Lake & Baroni, *SCAN*, ICML 2018,
  [arxiv via anthology](https://mlanthology.org/icml/2018/lake2018icml-generalization/).
- Keysers et al., *CFQ*, [arXiv:1912.09713](https://arxiv.org/abs/1912.09713).

### TL;DR
SCAN: navigate from commands (`jump twice`, `turn left and walk`) with
algebraic composition; seq2seq RNNs fail systematic recombination. CFQ: SPARQL
from compositional questions with maximum compound divergence (MCD) splits.
Both test *train/test compound shift*, not long-range channels.

### Verdict
**Watch** as a *split* idea (hold out hop-length or k) after Glyph is dense-solvable
in-distribution. **Reject** as the next E24 rung — we do not yet have a channel that
content-addresses. Composition in-context (shuffled chain) is the BAPO-hard exam we
already own; SCAN’s contribution is OOD compounds, which is a later ablation.

---

## Dyck languages, RASP, and stack exams

- Weiss, Goldberg, Yahav, *Thinking Like Transformers* (RASP),
  [arXiv:2106.06981](https://arxiv.org/abs/2106.06981).
- Ebrahimi, Hermann, et al., *How Can Self-Attention Networks Recognize Dyck-n?*,
  [ACL anthology](https://aclanthology.org/2020.findings-emnlp.384.pdf).
- Suzgun, Belinkov, et al. (LSTM / stack-RNN Dyck); Yao et al. bounded-depth Dyck.

### TL;DR
Dyck-k (balanced brackets of k types) is the canonical DCF exam. RASP programs a
fixed-head transformer to label each prefix P/T/F (possible / balanced / failed) for
*any* k. Ebrahimi: a BOS token helps (empty-stack base). Bounded-depth Dyck is what
transformers actually learn; unbounded depth is the theoretical wall (Hahn 2020).

### Verdict
**Adapt** bounded Dyck-2 as Glyph `dyck_close`: plant `span_len` unmatched openers,
emit the unique closers, stack verifier is the proof. Depth = answer_len, so the
exam is bounded by construction. **Watch** Dyck-PTF per-position labels (richer
supervision) after the span-level rung calibrates. **Reject** Shuffle-Dyck as a
separate recipe (it is a different language; one family).

---

## ListOps / Long Range Arena (brief)

Nangia & Bowman nested list operations; Tay et al. LRA. Nested `MAX/MIN/SUM` over
digit lists — a stack/arithmetic cousin of Delétang modular-with-brackets. **Watch**
as a possible `arith_eval` generator on Glyph-32 (`d + d` with brackets). Not in the
thin slice: needs a dense S0 of its own and is easy to make K1 at 0.6M.

---

## What this review does *not* recommend

- English BPE TinyStories / MiniPile / PG-19 as the E18 probe (unknown floor).
- Ten micro-A/Bs of haystack type × k × modulus × width.
- Duplicating the sibling `perceiver_concept` Arm-A 100% `far_copy` exam.
- Scoring E18 on Glyph before a matched dense control hits 75%.
- Vanilla NIAH / PPL-on-last-256 / LongBench-v1 ROUGE as the *first* 32k
  gate for a compressed exclusive channel (see HELMET / LongBench v2 below).

---

## HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly

[arXiv:2410.02694](https://arxiv.org/abs/2410.02694) · Yen, Gao, Hou, Ding,
Fleischer, Izsak, Wasserblat, Chen (Princeton / Intel). Code:
[github.com/princeton-nlp/HELMET](https://github.com/princeton-nlp/HELMET).

### TL;DR
NIAH / PPL / ROUGE and short LongBench slices are noisy. A 7-category,
length-controlled, few-shot, model-graded suite (8k–128k) ranks long-context
LMs more consistently. At 128k, NIAH is ~100 for almost everyone; HELMET
separates GPT-4o/Gemini from Llama-3.1. Spearman of synthetic vs real tasks
**< 0.8–0.85**. Recall/RAG hold with length; re-rank and citation collapse.
Numbered-label many-shot ICL kills verbal prior (a swap-control cousin).

### The exam they actually ran
Categories: RAG (NQ/TQA/PopQA/HotpotQA with *retrieved* distractors, gold at
6 depths), citation (ALCE), re-rank (MS MARCO NDCG@10), long QA (NarrativeQA
model-judge; ∞Bench QA/MC with entity replace), summ (model-based
F1×fluency), many-shot ICL with **numbered labels**, synthetic recall
(JSON-KV + RULER MK/MV). Lengths 8k/16k/32k/64k/128k, 100 ex/dataset,
greedy, 2-shot so base models work. 51 models.

### Verdict
**Adopt** the length ladder + the "NIAH saturates, HELMET does not" rule.
**Adapt** numbered-label ICL and RAG-with-retrieved-distractors onto a closed
alphabet (Glyph) rather than English essays. **Reject** NIAH as the 32k
success criterion for E21. Recommend RAG as a cheap proxy; still require
synthetic recall + a swapped-message control.

---

## LongBench v2

[arXiv:2412.15204](https://arxiv.org/abs/2412.15204) ·
[github.com/THUDM/LongBench](https://github.com/THUDM/LongBench).

### TL;DR
Human-reviewed multiple-choice, contexts 8k–2M words. Humans 53.7%, best LLM
50.1%. **No-context control ≈ chance (25%)** — the questions are not
solvable from parametric memory. RAG with 512-token chunks often **stops
helping past 32k** (not retrieval-solvable). Length bins are not comparable
(task mix shifts).

### Verdict
**Adopt** the no-context / chance-floor control as a standard arm (our
`message_override=none`). **Watch** the English MCQ set until a dense control
clears it at 32k. **Reject** treating LongBench-v1 ROUGE as evidence of
long-range slot use.

---

## 100-LongBench: Are de facto Long-Context Benchmarks Literally Evaluating Long-Context Ability?

[arXiv:2505.19293](https://arxiv.org/abs/2505.19293).

### TL;DR
Controllable input lengths plus a **disentanglement metric** that separates
"the model is good at the task" from "the model uses the extra context."
Many de facto long-context scores are baseline-task skill, not length skill.

### Verdict
**Adapt** the disentanglement idea: report E21 at 4k *and* 32k on the *same*
prize, and require that the 32k score is not explained by the 4k score plus
a local window. DNA already does this (seq ladder). Do not skip the 4k
anchor when claiming 32k.

---

## Is It Really Long Context if All You Need Is Retrieval?

Position paper, 2024 · [arXiv:2407.00402](https://arxiv.org/abs/2407.00402) ·
Levy, Bogin, Berant.

### TL;DR
Taxonomy of long-context tasks by **diffusion** (how spread out the necessary
information is) and **scope**. Highly diffused, lengthy information is
under-explored; most advertised "long context" is retrieval of a local span.

### Verdict
**Adopt** as a design constraint on the 32k suite: INDEX/NIAH are low
diffusion (one span); MATCH3 / majority / multi-hop VT are high diffusion.
E21 at r=16 can look strong on low-diffusion and dead on high-diffusion —
that is Deng's RAG vs synthetic-recall split, and E25's INDEX vs MATCH.

---

## Retrieval Head Mechanistically Explains Long-Context Factuality

[arXiv:2404.15574](https://arxiv.org/abs/2404.15574) · Wu et al.

### TL;DR
<5% of heads implement long-context lookup. Knock them out → NIAH/CoT fail;
knock out random heads → little effect. Heads persist after 32–128k
continued pretrain. The causal test that the compressed channel is *used as
memory*, not that PPL moved.

### Verdict
**Adapt** as an E21 probe: ablate / zero the exclusive-slot read
(`message_override=none` already) *and*, if a dense/E18 control exists,
knock out retrieval-like heads vs random. DuoAttention operationalizes the
split ([`learned_kv_context_compression.md`](learned_kv_context_compression.md)).

---

## NExtLong: Toward Effective Long-Context Training without Long Documents

[arXiv:2501.12766](https://arxiv.org/abs/2501.12766).

### TL;DR
Interleave **hard-negative distractors** when packing synthetic long
contexts, so next-token loss actually depends on far tokens. Addresses the
fact that many "long" documents contain no long-range dependency.

### Verdict
**Adapt** into E21's 32k mix (q-fraction of rows with a message boundary is
not enough if those rows have no prefix→suffix dependency). Not an exam;
a training-data recipe. Related: token weighting of long-range-dependent
tokens ([arXiv:2503.09202](https://arxiv.org/html/2503.09202)); ProLong
warns that naively mixing long SFT can hurt
([arXiv:2410.02660](https://arxiv.org/abs/2410.02660)).
