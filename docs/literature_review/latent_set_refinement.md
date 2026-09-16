# Latent-set refinement — when an extra reasoning stage over latents pays off

Reviews of **latent / continuous-space reasoning** with one load-bearing
question for E21: does a compressed *set* of slots benefit from an explicit
reasoning / refinement stage on top, and in which form?

Forms, from cheapest to richest:

1. One extra exclusive attend over **frozen** slot K/V (E21
   `message_extra_slot_attends`, Goyal pause tokens).
2. A second unique global layer (`global_layers=2`).
3. Rewrite slot K/V between hops (`message_update_slot_kv`, Slot Attention GRU).
4. A small Transformer / self-attn over the slot set.
5. Many **weight-tied loops** over the set with per-slot targets (LOTUS,
   Saunshi `(k ⊗ L)`, Slot Attention `T>1`).
6. Sequential latent CoT (Coconut / CODI / SIM-CoT) — a *chain*, not a set.

Related reviews (do not duplicate):
- Coconut, Huginn, Ouro, LoopFormer, RMT (writable-memory family):
  [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)
- Pfau filler-token barrier, BAPO Thm 8 (CoT turns hard problems easy):
  [`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md)
- Objectives that force a non-bypassable bottleneck (TSDAE, VICReg, JEPA):
  [`concept_bottleneck_collapse_mitigation.md`](concept_bottleneck_collapse_mitigation.md)

**Standing claim from this file:** extra compute over a latent set pays off
when the task is iterative / multi-hop / set-binding **and** each slot is
step-supervised. **One extra unique layer is not a substitute for many tied
loops.** One extra attend over frozen K/V helps extractive QA, not MATCH-class
lookup, and is what E25 already killed for SELECT/MATCH. Sequential latent
tokens without step CE collapse as `K` grows.

---

## LOTUS — Looped Transformers with parallel supervision on latents

Preprint 2026 · [arXiv:2606.31779](https://arxiv.org/abs/2606.31779) ·
Fan, Svete, Lee (Microsoft Research / ETH / KRAFTON). Code:
[github.com/yingfan-bot/lotus](https://github.com/yingfan-bot/lotus).
HF: [yingfanbot/gsm-lotus-llama3b](https://huggingface.co/yingfanbot/gsm-lotus-llama3b).

### TL;DR
Pad `K` latent blocks × `c` tokens, loop the backbone `R` times over that
set while the question KV is frozen, and apply **parallel gold-CoT CE
through the base LM head** on the post-loop latents. First latent-CoT method
to match explicit CoT at 3B (GSM8K 70.0 vs CoT 71.5) with 2.5×–6.9× lower
thought latency. **`R=2` is 14.6%; `R=6` is 70.0%. `c=1` is 49.7%; `c=25`
is 70.0%. Answer-only (no `L_step`) is 63.3%.** One extra loop is not
enough; one token per step is too narrow; loops without per-slot CE are not
the 70%.

### Forward pass (shapes)
Symbols: question `Q`, `K=6` blocks, `c=25` tokens/block, `R=6` loops, hidden
`d`.

```
x = [Q, <BoT>, (K blocks of c shared <lat> tokens), <EoT>, A]
C_pre = KV of Q (frozen across loops)
h^{0} = embed(latents)
for t = 1..R:
  h^{t} = f_θ(E + h^{t-1} | C_pre)     # weight-tied backbone over the SET
L_step = CE( f_head(h^{R})_{i,j} , gold CoT token T_{i,j} )   # all positions in parallel
then a final forward: answer CE on A conditioned on frozen h^{R}
```

Latents are **never decoded at inference**. LOTUS-aux routes the same
supervision through a training-only auxiliary decoder (SIM-CoT-style).

### Evaluation / results
Llama-3.2-3B-Instruct, GSM8k-Aug: LOTUS 70.0±0.9 vs Explicit CoT 71.5 vs
CODI+SIM-CoT 62.3. OOD avg 63.9 vs CoT 62.1. Thought latency 133 vs 339 ms
(2.5×); NL-CoT stress 6.9×.

| train `R` | GSM8K |
|---|---|
| 2 | 14.6% |
| 3 | 23.2% |
| 4 | 52.6% |
| 5 | 68.1% |
| 6 | 70.0% |

Infer extra `R=7` on the `R=6` ckpt: 69.3% (no gain). Infer `R=1`: 22.7%.

| train `c` (K=6) | positions | GSM8K |
|---|---|---|
| 1 | 6 | 49.7% |
| 5 | 30 | 67.5% |
| 25 | 150 | 70.0% |
| 30 | 180 | 70.0% |

No `L_step` (answer-only): 63.3%. `L_step`-only gold NLL 9.29 / top-1 9.1%;
`L_ans`-only 5.97 / 9.4%; both 3.07 / 70.9%. Direct LM-head readout is
robust at GPT-2/1B; aux decoder only matches at 3B.

### Limitations
Math-only. Fixed `(K,c,R)`. Preprint, 3 seeds. Closest paper to "one extra
global layer vs many loops over a slot set" — and the answer is **many
loops + per-slot CE**, not one layer.

### Gradient flow
- Question KV is frozen across loops (like E21's frozen exclusive slots).
- Gradients from `L_step` flow through `R` tied applications into the
  latent embeddings.
- Answer CE is a second forward with stop on the post-loop latents in the
  default recipe (condition on frozen `h^{R}`).

---

## SIM-CoT: Supervised Implicit Chain-of-Thought

ICLR 2026 · [arXiv:2509.20317](https://arxiv.org/abs/2509.20317) ·
Wei, Liu, Zang, Dong, Cao, Wang, Qiu, Lin. Code:
[github.com/InternLM/SIM-CoT](https://github.com/InternLM/SIM-CoT).

### TL;DR
Coconut-style sequential latents `z_k` collapse as `K` grows because
answer-level loss does not pin each `z_k` to a step. An auxiliary decoder
that reconstructs gold step `s_k` from `z_k` (discarded at inference)
stabilizes the set. Coconut at 5 latents → 12.5%; SIM-CoT stays stable to
8–16. **Oversized aux decoder hurts** (1B backbone + 8B decoder 50.0 vs
matched 56.1).

### Evaluation / results
Coconut GPT-2 GSM8k-Aug 36.6→44.8. CODI+SIM-CoT LLaMA-3.1-8B +3.0
(61.1→64.1). Inter-latent distance 4.21 (failed) vs 32.81 (after SIM-CoT).
Curriculum still used for the Coconut backbone; SIM-CoT reports curriculum
forgetting at LLaMA-1B+.

### Why it matters here
The forcing term is **per-slot CE**, not "add a decoder." A reasoner that
only sees answer CE will homogenize the set — the same geometry as E02's
low slot-rank with high STS-B (one loaded direction).

---

## CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation

EMNLP 2025 · [arXiv:2502.21074](https://arxiv.org/abs/2502.21074) ·
[ACL](https://aclanthology.org/2025.emnlp-main.36/) · Shen, Yan, Zhang, Hu,
Du, He. Code: [github.com/zhenyi4/codi](https://github.com/zhenyi4/codi).

### TL;DR
One-stage implicit CoT: student emits 6 continuous thoughts; teacher is
standard CoT; L1-align hidden states at a designated pre-answer token
(`:`), **excluding the last CoT step** so the teacher cannot copy the
answer. Peak at 6 thoughts (dataset step count). w/o L1 43.7→24.5; keep
last CoT step 31.7 (shortcut); independent teacher 27.1.

### Evaluation / results
GPT-2 GSM8k 43.7 vs Coconut ~34, ~99% of CoT-SFT, 3.1× compression. OOD
(SVAMP/GSM-Hard/MultiArith) beats CoT-SFT on GPT-2. Student CE alone ≈
pause-token extra compute.

### Why it matters here
**Outcome / answer CE is not a reasoner objective.** Distillation has to
block the last-step shortcut. E21's receiver CE on natural text is
outcome-like unless the targets are determined by far content (the E23/E25
lesson).

---

## CoLaR: Think Silently, Think Fast

NeurIPS 2025 · [arXiv:2505.16552](https://arxiv.org/abs/2505.16552) ·
Tan, Li, Ju, Luo, Luan, Song (Renmin / Xiaomi). Code:
[github.com/xiaomi-research/colar](https://github.com/xiaomi-research/colar).

### TL;DR
Compress CoT by predicting the next `c`-token-merged embedding from a
probabilistic latent head; random `c` at train time; GRPO to shorten.
**Mean-pool (`-MP`) −3.4 pts; answer-only CE (`-OC`) −1.6 pts.** Sum/`√c`
merge, not mean. Per-token-averaged reward is load-bearing for RL (else
length hits the cap).

### Evaluation / results
Llama-3.2-1B: CoLaR-5 Acc 41.7 vs Coconut 27.6; CoLaR-2 48.8 vs CoT 53.6
with 53.3% shorter chains. MATH + GRPO: +5.36% Acc, −82.8% length on
DeepSeek-R1-Distill-Qwen-1.5B. Does not beat explicit CoT.

---

## Think Before You Speak: Pause Tokens

ICLR 2024 · [arXiv:2310.02226](https://arxiv.org/abs/2310.02226) ·
Goyal, Ji, Rawat, Menon, Kumar, Nagarajan (CMU / Google).

### TL;DR
Append a learned `<pause>` sequence to widen per-layer compute **iff** the
model is pause-pretrained *and* pause-finetuned. **SQuAD +18 EM, CSQA +8,
GSM8k +1.** Pause-FT-only mixed; GSM8k pause-FT can hurt. Periods as
fillers do not help (Pfau). `M_inf=0` breaks PausePT_PauseFT. +10 pauses ≉
+2 unique layers in the SQuAD gain. Optimal pause count is task-specific
(GSM8k ~10, SQuAD prefers 50).

### Why it matters here
This is the published version of E21 `message_extra_slot_attends=1`: extra
attends over a frozen prefix with no rewrite. Pays on **extractive QA**,
not on multi-hop lookup / math. E25's extra hop helping DNA *hops* and
failing MATCH/SELECT at 1280 is the same split.

---

## Quiet-STaR

[arXiv:2403.09629](https://arxiv.org/abs/2403.09629) · Zelikman et al.
Code: [github.com/ezelikman/quiet-star](https://github.com/ezelikman/quiet-star).

### TL;DR
Continue-pretrain an LM to emit internal rationales that help predict
*future* tokens (REINFORCE on mix of thought vs no-thought). Mistral-7B:
GSM8K 5.9→10.9, CSQA 36.3→47.2. **Verbalized multi-token thoughts beat a
single pause** (pause-FT harmed GSM8K in their comparison). Thoughts are
discrete language, not a compact KV set. TTFT ~10×.

---

## Reasoning with Latent Thoughts: On the Power of Looped Transformers

ICLR 2025 · [arXiv:2502.17416](https://arxiv.org/abs/2502.17416) ·
Saunshi, Dikkala, Li, Kumar, Reddi (Google / TTIC).

### TL;DR
Many reasoning problems need **depth not parameters**: `(k ⊗ L)` ≈
`(kL ⊗ 1)` on addition / p-hop / i-GSM. On LM pretraining, loops trade
PPL / closed-book QA (memorization) for reasoning. 1B Pile: loops cover
~34–50% of the PPL gap but **77–282% of math word problems** and can
*exceed* a 24-layer baseline on reasoning primitives. `(12 ⊗ 2)` math 34.3
vs 24-layer 29.3 despite worse PPL. Cosine-tying (`λ=10`, k=4) keeps PPL
and lifts math 29.3→36.4.

### Why it matters here
**Reject "add `global_layers=2`" as the reasoning bet.** Extra *unique*
depth helps PPL/memorization; extra *tied* depth helps multi-hop. E25's
second exclusive global layer failing SELECT is predicted: SELECT is
addressing, not iterative composition, and one unique layer is the wrong
axis.

---

## Mixture-of-Recursions (MoR)

NeurIPS 2025 · [arXiv:2507.10524](https://arxiv.org/abs/2507.10524) ·
Bae et al. (KAIST / Mila / Google). Code:
[github.com/raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions).

### TL;DR
Shared recursion block plus routers that send hard tokens through more
loops. IsoFLOP 360M: NLL 2.75 vs vanilla 2.78, few-shot 43.1 vs 42.3.
**Standard LM loss only — no CoT/step CE. Helps PPL / few-shot, not a
GSM-style reasoner.** `N_r=2` is the quality/efficiency sweet spot.

---

## ThoughtBubbles

[arXiv:2510.00219](https://arxiv.org/abs/2510.00219) · Liu, Murty, Manning,
Csordás (Stanford). Code:
[github.com/stanfordnlp/thoughtbubbles](https://github.com/stanfordnlp/thoughtbubbles).

### TL;DR
Unsupervised parallel residual forks during pretraining from LM loss.
`κ=4L` beats `κ=2L` and copy-N filler residuals on PPL. 319M ThoughtBubbles
< 772M baseline PPL on OpenWebText. No GSM8K (authors: scale too small).
Helps PPL / LAMBADA / HellaSwag; BLiMP mixed.

---

## Object-Centric Learning with Slot Attention

NeurIPS 2020 · [arXiv:2006.15055](https://arxiv.org/abs/2006.15055) ·
Locatello et al. Capacity/`K` reading:
[`information_bottleneck_latent_capacity.md`](information_bottleneck_latent_capacity.md).

### TL;DR
A **set** of slots binds to objects by iterative cross-attention with
softmax **over slots** (competition) and a GRU rewrite. **`T=1` is much
worse than `T=3`**; extra test-time `T` helps; applying loss every
iteration lets you train `T>3`. Without competition, slots collapse.
Optional slot–slot self-attn for communication beyond competition.

### Forward pass (shapes)
```
slots [B, K, D] random init
for t = 1..T:
  attn = softmax_over_K( slots W_q · inputs W_k )     # competition on slots
  updates = attn · inputs
  slots ← GRU(slots, updates)
```

### Why it matters here
Rewriting slot contents is *required* for binding (`T=1` fails). E21's
default extra hop attends without rewriting K/V (`message_update_slot_kv`
off). LOTUS freezes *input* KV and rewrites *latent embeddings*. Pause
tokens rewrite nothing. The three forms are not interchangeable.

---

## What Makes Effective Supervision in Latent Chain-of-Thought

ICML 2026 · [arXiv:2606.20075](https://arxiv.org/abs/2606.20075) ·
Chen et al. Code promised:
[EIT-NLP/Supervision-in-Latent-CoT](https://github.com/EIT-NLP/Supervision-in-Latent-CoT).

### TL;DR
Outcome supervision ≈ No-CoT because of **gradient attenuation** along the
latent chain and **representational drift**. Process supervision must cover
Trajectory (dense suffix CE) and Space (`I(L_t; S_t)`). **Generative
reconstruction of steps from latents beats rigid geometric compression**
(L2/cosine onto a frozen encoding collapses the manifold). Accuracy tracks
recoverable CoT information in the latents.

---

## Already reviewed (pointers)

- **Coconut** ([arXiv:2412.06769](https://arxiv.org/abs/2412.06769)):
  continuous thought as last hidden state fed back; curriculum from explicit
  CoT; wins on ProsQA/ProntoQA search, **flat/loses on GSM8K** at larger
  scale; audits find shortcuts
  ([arXiv:2512.21711](https://arxiv.org/abs/2512.21711)).
  [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md) §B.
- **Huginn** ([arXiv:2502.05171](https://arxiv.org/abs/2502.05171)) +
  skeptical analysis ([arXiv:2507.02199](https://arxiv.org/abs/2507.02199)):
  depth-recurrent; mixed evidence that genuine latent CoT emerges.
- **Ouro** ([arXiv:2510.25741](https://arxiv.org/abs/2510.25741)) + STARS
  stability follow-ups: looped latent state is unstable by default.
- **LoopFormer** ([arXiv:2602.11451](https://arxiv.org/abs/2602.11451)):
  elastic depth via shortcut modulation; "bare recursion amplifies collapse."
- **Pfau filler tokens** ([arXiv:2404.15758](https://arxiv.org/abs/2404.15758)):
  hidden CoT via dots works only with **dense supervision of the delay**;
  off-the-shelf fillers do not transfer. BAPO Thm 8 needs a *re-readable
  tape*, not a Markov `z_k → z_{k+1}`.
- **Perceiver iterative cross-attn**: more iterations help vision; Perceiver
  AR underperforms on text LM PPL
  ([`recurrent_memory_transformers.md`](recurrent_memory_transformers.md) §A).

---

## Cross-cutting (refinement form × objective, not verdicts)

1. **Form ranking by published ablations, task-selective:**
   - Many tied loops + per-position gold CE (LOTUS, Saunshi alg/math):
     strongest on planning / multi-hop / math.
   - Transformer / self-attn over the set (Perceiver latent TF, LOTUS
     parallel blocks, Slot Attention + slot–slot attn): strongest for
     set-binding.
   - Rewrite slot K/V (Slot Attention GRU): required for object binding;
     `T=1` fails.
   - One extra exclusive attend / pause: large on extractive QA, ~+1 pt
     GSM8k, does not save MATCH/SELECT.
   - Sequential latents without step CE (Coconut `K↑`): collapses.
   - Filler / pause off-the-shelf: no transfer.
2. **Objectives that force semantically rich latents:**
   - Per-slot / per-step CE (SIM-CoT, LOTUS `L_step`).
   - Trajectory distillation that **excludes the last answer-copy step**
     (CODI).
   - Next-compressed-embedding prediction, not mean-pool (CoLaR).
   - Generative reconstruction of steps (Chen et al. 2026), not L2-to-frozen
     encodings.
   - Exclusive decoder channel / word dropout (TSDAE; this repo's E21
     message boundary; E02 prefix→suffix).
   - Reconstruction of the *prefix from the slots* (Deng Fine-AE;
     Compressive Transformer attn-recon) — a *memory* objective, not a
     *reasoner* objective.
3. **Objectives that let the decoder shortcut:**
   - Answer-only / next-token CE with a raw path open (Broken ELBO; E05;
     E18 arm C).
   - More `K` without step supervision.
   - Independent teacher KD; oversized aux decoder.
   - Inference-only pauses/fillers.
   - Mean-pooling embeddings across `c` tokens.
   - Including the last CoT step in teacher KD.
4. **PPL vs reasoning split.** MoR / ThoughtBubbles / unique extra depth
   help PPL. Tied loops / LOTUS / pause-on-SQuAD help different axes. A
   slot-refinement stage that helps GSM may hurt NIAH/copy or vice versa.
5. **Curriculum at scale.** Coconut curriculum forgets at LLaMA-1B+;
   CODI / SIM-CoT / LOTUS avoid it. A one-step refinement module should
   not depend on a CoT curriculum.
