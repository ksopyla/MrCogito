# Addressable memory — RAM slots, sparse write/read, location vs content

Reviews of architectures that treat a small memory as a **set of locations
(addresses)** rather than as a permutation-equivariant soup that every token
densely mixes. The load-bearing question for [REDACTED] after E17c:

**Do concept slots have positions, and does a write/read touch only some of
them, leaving the rest unchanged?**

E17c showed that causal carry dropout can force concept *use* (carryless
Δpermutation 0.59 nats) while RankMe collapsed to 6.7 and almost all of the
signal sat in bank 0. That cell still **wrote every slot, every block**
(`gated_replace` interpolates all C rows). This file is the lineage for the
counterfactual: an addressable RAM. Writable-but-dense recurrence (RMT,
Block-Recurrent, Infini-attention, E17c itself) stays in
[`recurrent_memory_transformers.md`](recurrent_memory_transformers.md).

## How to read this file

Five addressing primitives appear again and again. They are not interchangeable:

| Primitive | What an "address" is | Unaddressed slots unchanged? |
|---|---|---|
| **Location** (NTM shift / learned address embeddings / Landmark) | An index or a pointer | Yes if write weight ≈ 0 |
| **Content** (NTM cosine, MemNets, Hopfield, DeltaNet) | Match a key against stored rows | Only if unmatched weights vanish |
| **Allocation** (DNC usage, SAM LRU) | Prefer unused / stale locations | Yes by construction |
| **Hash / product key** (Kanerva SDM, Lample product keys) | Cue → sparse hard locations | Yes (top-k only) |
| **Dense mix** (softmax over all C, BiXT, Infini Hebbian, GSA convex combo) | There is no unused location | **No** |

E17/E17b/E17c sit in the last row. E17d is a bet on the first four.

---

## Neural Turing Machines

arXiv 2014 · https://arxiv.org/abs/1410.5401 · Alex Graves, Greg Wayne, Ivo
Danihelka (DeepMind) · no official code; community PyTorch
https://github.com/ixaxaar/pytorch-dnc

### TL;DR
A controller plus an N×W memory *matrix of locations*, accessed by
differentiable read/write heads with hybrid content+location addressing.
Unaddressed rows stay put when their write weight is ~0.

### The problem that authors want to solve
LSTMs store working memory in compressed hidden state and cannot learn simple
programs (copy, sort, associative recall) that need a RAM.

### The solution, main idea on the intuition level and strong points
Memory is explicitly addressed. Read is `r = Σ_i w(i) M(i)`. Write is LSTM-style
**erase then add**: `M(i) ← M(i) ⊙ (1 − w(i) e) + w(i) a`. If `w(i)≈0`, location
`i` is invariant — the unused-slot rule. Addressing is hybrid: content
(cosine-softmax of a key against rows) composed with location (blend with the
previous weighting, circular shift, sharpening γ). Three modes: pure content,
content-then-shift (find a block, then index inside it), pure location
iteration.

### The detailed solution, training process, data preparation
Soft weights over *all* N locations; sparsity is only a bias from γ, not hard
top-k. Controllers are feedforward or LSTM. Trained on synthetic binary
sequences with BPTT. Softmax addressing therefore *leaks* a little onto every
slot.

### The evaluation procedure, evaluation datasets and results
Copy of 8-bit vectors (train length 1–20, generalizes to 100), repeat-copy,
associative recall, priority sort. LSTM baselines fail to generalize. Scale is
tiny; no language modeling.

### Previous attempts / related publications
Precursor to DNC (https://www.nature.com/articles/nature20101). Contrasted with
Memory Networks, which write sequentially and only content-read
(https://arxiv.org/abs/1503.08895). SAM later argued the dense softmax is why
NTMs do not scale (https://arxiv.org/abs/1610.09027).

### Relevance to [REDACTED]
This is the canonical "concepts are RAM" paper. The erase-add rule and the
location/content split are the mechanism E17c lacked. Do **not** clone the full
NTM controller onto Gemma: it never left synthetic scale, and soft addressing
still updates every row a little. Port the *invariants* (addresses exist;
unaddressed rows unchanged; hybrid location+content), then harden sparsity.

---

## Hybrid computing using a neural network with dynamic external memory (DNC)

Nature 2016 · https://www.nature.com/articles/nature20101 · Graves, Wayne,
Reynolds, Harley, Danihelka et al. (DeepMind) · official TF
https://github.com/google-deepmind/dnc · PyTorch port
https://github.com/ixaxaar/pytorch-dnc

### TL;DR
NTM plus two extra addressors: a temporal link matrix (walk write-order) and
**usage-based allocation** (write into unused locations). Allocation is
independent of memory contents, so empty RAM is writable.

### The problem that authors want to solve
NTM content addressing cannot target a *blank* row (all zeros look the same),
and NTM has no way to recover write order independently of content.

### The solution, main idea on the intuition level and strong points
Same erase-add write. Three attentions: (1) content lookup; (2) temporal links
`L[i,j]` recording that `j` was written after `i`; (3) a usage scalar in [0,1]
per location. The write head is given a weighting over *unused* locations.
Usage increases on write and can decrease after a free. A DNC trained at one N
can be upgraded to a larger memory without retraining.

### The detailed solution, training process, data preparation
Reads/writes still use dense weightings over all N rows. Official code is
TensorFlow 1 + Sonnet. Tasks: bAbI, graph curriculum (traversal, shortest path,
relation inference), Mini-SHRDLU via RL.

### The evaluation procedure, evaluation datasets and results
Joint bAbI: 3.8% mean error vs worse NTM/LSTM. Graph curriculum then zero-shot
transfer to the London Underground and a family tree; LSTM failed the first
lesson. Still not an LM.

### Relevance to [REDACTED]
E17c's banks start as learned constants, so they are never "empty," but after a
gated replace they can become a rank-1 blob — content addressing then cannot
find a fresh slot. A **usage / allocation bias** (prefer low-usage addresses) is
the published fix for "write to unused positions." Skip the temporal link
matrix on a first experiment; it is the expensive DNC-specific piece.

---

## Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes (SAM)

NeurIPS 2016 · https://arxiv.org/abs/1610.09027 · Rae, Hunt, Harley, Danihelka,
Senior, Wayne, Graves, Lillicrap (DeepMind) · community PyTorch in
https://github.com/ixaxaar/pytorch-dnc

### TL;DR
Soft NTM/DNC access is O(N) per step and O(NT) for BPTT. Hard top-K reads/writes
plus ANN lookup recover the same tasks at ~1000× speed. Unaddressed words are
**exactly** unchanged.

### The problem that authors want to solve
Dense softmax over memory does not scale, and BPTT duplicates the whole tape.

### The solution, main idea on the intuition level and strong points
Sparse read: keep K∈{4,8} largest content-weights. Sparse write is **not**
content-write (authors reject that for empty-memory problems). Instead write to
(a) previously read locations (update) or (b) the least-recently-accessed
location (overwrite stale), interpolated by gates. BPTT stores only the sparse
diffs.

### The evaluation procedure, evaluation datasets and results
Matches NTM data efficiency on algorithmic tasks, bAbI, Omniglot. 1M memories:
NTM 12s vs SAM 7ms per fwd+bwd. 64k words over T=100: NTM 29 GiB vs SAM 7.8 MiB.
SDNC ~400× faster than dense DNC at 2,000 slots.

### Relevance to [REDACTED]
The only paper in this lineage that *literally* implements "write to certain
positions, not all." For C=128 we do not need ANN indexes — exact top-k over
128 is cheap. The warning that matters: **do not content-address empty slots**;
combine top-k with location and/or LRU/usage. Softmax-then-top-k-renormalize is
the practical port (straight-through on the mask).

---

## Object-Centric Learning with Slot Attention

NeurIPS 2020 · https://arxiv.org/abs/2006.15055 · Locatello, Weissenborn,
Unterthiner, Mahendran, Heigold, Uszkoreit, Dosovitskiy, Kipf · official TF
https://github.com/google-research/google-research/tree/master/slot_attention
· PyTorch https://github.com/lucidrains/slot-attention

### TL;DR
Slots bind to objects because they **compete**: softmax is normalized over the
slot axis, so each token's mass sums to 1 across slots. No slot has an address;
slots are exchangeable.

### The solution, main idea on the intuition level and strong points
`attn = softmax(K(inputs) Q(slots)ᵀ, axis='slots')`. Updates are a weighted
mean of values, then a shared GRU per slot. Permutation-equivariant in slots;
slot count can change at test time. Slot-MLP without competition collapses
(CLEVR6 ARI 60 vs 99 with competition).

### The evaluation procedure, evaluation datasets and results
Unsupervised object discovery on CLEVR6 / Multi-dSprites / Tetrominoes. Vision,
K typically 7–11, not language.

### Relevance to [REDACTED]
Two opposite lessons. (1) Competition can *prevent* every slot from storing the
same thing. (2) Softmax-over-slots is also a winner-take-all engine: if one
slot outcompetes the others, you get E17c's bank-0 absorption. Slot Attention
has **no addresses** and **dense** token→slot assignment. It is not the RAM
template. Use it as the collapse diagnostic: if write mass concentrates on one
index, we reproduced Slot-MLP/WTA, not NTM.

---

## Hopfield Networks is All You Need

ICLR 2021 · https://arxiv.org/abs/2008.02217 · Ramsauer, Schäfl, Lehner, …
Hochreiter · official PyTorch https://github.com/ml-jku/hopfield-layers

### TL;DR
Transformer attention is one step of a continuous modern Hopfield net. The
energy has three kinds of minima: a **global average of all patterns**,
metastable subset averages, and single-pattern fixed points. Inverse
temperature β picks which.

### The solution, main idea on the intuition level and strong points
Update `ξ ← X softmax(β Xᵀ ξ)` ≡ attention. Storage capacity is exponential in
d. Small β (or an unsharpened softmax over all keys) is exactly the
global-average attractor — dense mixing collapses to one direction.

### The evaluation procedure, evaluation datasets and results
MIL, immune-repertoire, UCI, drug design. Theoretical, not an LM scaling paper.

### Relevance to [REDACTED]
This is the math of E17c's RankMe 6.7. Dense BiXT / dedicated softmax over 128
slots with a pressure objective that *must* dump information into z is a
Hopfield retrieval with too-small β: the energy minimizer is the average, not
128 distinct files. Sparse top-k + sharpening raises β for the kept addresses
and leaves other patterns unperturbed. RankMe of the full bank is a legitimate
collapse detector only if unused slots are not frozen random init (which would
inflate rank). Measure RankMe of **written** slots separately.

---

## Large Memory Layers with Product Keys

NeurIPS 2019 · https://arxiv.org/abs/1907.05242 · Lample, Sablayrolles,
Ranzato, Denoyer, Jégou (FAIR) · code inside
https://github.com/facebookresearch/XLM

### TL;DR
A key-value memory with product-quantized keys gives ~√|K| exact nearest
neighbor lookup, so a transformer FFN can hold ~10⁸–10⁹ extra parameters while
**updating only top-k value slots** per token.

### The solution, main idea on the intuition level and strong points
Keys are never materialized: two sub-key tables induce |K| = n₁×n₂ product
keys. Split the query, take top-k in each half, search k×k candidates, keep k.
Only those k value slots get gradients. Batch-norm on queries is load-bearing
for key coverage. Drop-in FFN replacement.

### The evaluation procedure, evaluation datasets and results
Up to 30B-word LM. 12-layer + 1 memory layer outperforms a 24-layer transformer
and is ~2× faster at inference. Strongest published "addressable KV with sparse
top-k write" *inside a real transformer*.

### Relevance to [REDACTED]
Proof that sparse slot updates work at LM scale. Caveat: the memory is a
**static codebook**, not a recurrent concept bank that holds the current
sequence. Addresses are learned product keys, not token positions. Steal
top-k-only gradients and the coverage diagnostic (are all keys used?), not the
FFN-as-memory architecture.

---

## Gated Slot Attention (GSA) and ABC

GSA: NeurIPS 2024 · https://arxiv.org/abs/2409.07146 · Zhang, Yang, Zhu, … Fu
· kernels in https://github.com/fla-org/flash-linear-attention

ABC: ACL 2022 · https://arxiv.org/abs/2110.02488 · Peng et al.

### TL;DR
ABC treats attention as RAM with a **bounded** number of slots m ≪ T and a
learned write intensity φ over those slots. GSA adds per-slot forget gates.
Writes are still a convex combination into *all* m slots unless the forget gate
is exactly 1.

### The solution, main idea on the intuition level and strong points
ABC write: `K̃_t = K̃_{t−1} + φ_t ⊗ k_t` with φ a cumulative softmax over m
slots — this *is* learned "where to write," but token 1 paints into every slot
(φ=1) and there is no forget (documented primacy bias). GSA:
`(K̃_t)_i = α_i (K̃_{t−1})_i + (1−α_i) k_t`. Softmax is kept so a pretrained
transformer can be distilled into a recurrent GSA (T2R).

### The evaluation procedure, evaluation datasets and results
GSA beats RetNet/GLA on in-context recall with a smaller state. T2R: finetune
Mistral-7B → GSA beats RWKV6/Mamba/T2R-to-GLA on a few billion tokens.

### Relevance to [REDACTED]
Closest 2024 "slots as RAM" inside LMs, and a cautionary tale: **learned write
intensity without an unused-slot invariant is still dense**. E17c's per-slot
sigmoid is GSA-like; it opened the gates and still collapsed. Do not Adapt GSA
as the E17d cell. Keep it as the "gated dense slot write already failed here"
baseline.

---

## MemoryLLM: Towards Self-Updatable Large Language Models

ICML 2024 · https://arxiv.org/abs/2402.04624 · Wang, Gao, Chen, Jiang, …
McAuley · official PyTorch https://github.com/wangyu-ustc/MemoryLLM

### TL;DR
Graft a fixed pool of memory tokens into every layer of Llama-2. Reads are
dense (every token attends all N memory tokens). Writes drop a random K and
append K new ones — location-like FIFO, not learned addresses.

### The solution, main idea on the intuition level and strong points
Llama-2 is the static backbone. Memory θ_l ∈ R^{N×d} per layer. Self-update:
concat last K memory tokens with new hidden states, run the layer, take last K
outputs as new memory, randomly drop K from the pool, append on the right.
Forgetting of a token decays exponentially. Follow-up M+
(https://arxiv.org/abs/2502.00592) scales this further.

### The evaluation procedure, evaluation datasets and results
Llama-2 7B + ~1B memory parameters. Model-editing and long-context QA improve
vs editing/retrieval baselines. ~1e6 successive updates without measured
integrity collapse.

### Relevance to [REDACTED]
Best published "graft a memory pool onto a frozen pretrained LM" recipe — the
engineering analogue of our Gemma+LoRA graft. But it does **not** learn WHERE
to write, and it reads every slot. K≪N is the only sparsity. E11 (mem-tokens in
sequence) is closer to MemoryLLM/RMT than to NTM. E17d should not copy the
random-drop write.

---

## Landmark Attention

NeurIPS 2023 · https://arxiv.org/abs/2305.16300 · Mohtashami, Jaggi (EPFL) ·
official PyTorch https://github.com/epfml/landmark-attention

### TL;DR
Insert one landmark token per block; attention learns to use landmarks as
**read addresses** for retrieving blocks. Successfully fine-tuned onto LLaMA-7B.

### Relevance to [REDACTED]
Rare successful *addressing graft onto a pretrained LM*. Addresses index *input
blocks*, not a persistent C-slot RAM, and the write is "append a landmark every
B tokens." Evidence that pretrained models can learn an addressor on the read
side; not a write-side RAM.

Full review: [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md)
(family A).

---

## Dense associative writes (Infini, DeltaNet, Titans) — contrast, not template

These 2024–2026 methods sparsify in **association / weight space**, not in slot
index. They are the wrong address primitive for "write to position i":

- **Infini-attention** (https://arxiv.org/abs/2404.07143): every token writes
  into a d×d matrix. No unused slot. HF reproduction reports iterated
  compression degrades. Reviewed in
  [`recurrent_memory_transformers.md`](recurrent_memory_transformers.md) §D.
- **DeltaNet / Gated DeltaNet** (https://arxiv.org/abs/2406.06484,
  https://arxiv.org/abs/2412.06464, https://github.com/NVlabs/GatedDeltaNet):
  overwrite the *matched association*. Sparse in content, dense in "which
  matrix cell."
- **Titans / Nested Learning** (https://arxiv.org/abs/2501.00663,
  https://arxiv.org/abs/2512.24695): surprise-gated test-time gradient into an
  MLP. "Where to write" = which parameters the gradient hits. No official
  Google PyTorch. Watch for later; not E17d.

**Memory Mosaics** (ICLR 2025 https://arxiv.org/abs/2405.06394; v2
https://arxiv.org/abs/2507.03285; archived
https://github.com/facebookresearch/MemoryMosaics) replace attention with
kernel-regression associative memories at Llama-8B scale. Content-addressed,
trained from scratch, no integer write. Watch.

---

## Cross-cutting findings

1. **Unused-slot invariance is the missing E17c ingredient.** Every method that
   actually leaves locations unchanged uses erase-add with w≈0 (NTM/DNC) or
   hard top-k (SAM, product keys). Gated dense writes (GSA, E17c
   `gated_replace`) do not have this invariant.
2. **Empty or collapsed memory cannot be content-addressed.** DNC allocation
   and SAM LRU exist because cosine-to-zeros is undefined as a discriminator.
   A learned address embedding (location) plus a usage bias is the portable
   pair.
3. **Softmax over all C is a Hopfield global-average.** Pressure objectives
   that force information through a dense mixer predict RankMe collapse, which
   is what E17c measured. Sparse top-k + sharpening is the published
   temperature control.
4. **Grafting onto a frozen ~1B LM has precedent for reads and for memory
   *pools*, not for NTM write heads.** Landmark (read addresses on LLaMA-7B),
   MemoryLLM (FIFO pool on Llama-2), Infini/GSA-T2R (convert the attention
   kernel). Nobody has shown a frozen 1B growing a sparse write decoder onto
   C=128 recurrent banks. That gap is the experiment, not a reason to retreat
   to dense BiXT.
5. **Diagnostics that distinguish RAM from dense mix:** write-mass entropy /
   occupancy histogram, RankMe of written vs unwritten slots, allocation
   (writes land on low-usage indices), and the unused-slot invariance test
   (`Δz[unaddressed] == 0`). RankMe-of-all-128 and Δpermutation alone cannot
   tell "learned addressing" from "one slot ate the dropout signal."
