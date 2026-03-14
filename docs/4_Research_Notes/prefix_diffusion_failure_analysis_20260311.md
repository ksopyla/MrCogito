# Prefix Diffusion Failure Analysis

**Date:** 2026-03-11  
**Author:** Krzysztof Sopyla / AI research analysis  
**Status:** Permanent research note  
**Related experiments:** `prefix_diff_H512L6C128D2_20260304_200437`, `prefix_diffBiXT_T64_H512L6C128D2_20260308_065355`, TODO 13c WikiText-103 probe  
**Related files:** `nn/concept_encoder_diffusion.py`, `nn/concept_encoder.py`, `data/data_collators.py`, `training/train_prefix_diffusion.py`, `scripts/train_prefix_diffusion_multigpu.sh`  
**Related reports:** [prefix_diffusion_bixt_v2_20260308.md](../2_Experiments_Registry/run_reports/prefix_diffusion_bixt_v2_20260308.md), [diffusion_diagnosis_20260226.md](diffusion_diagnosis_20260226.md)

---

## Executive Summary

The prefix diffusion track failed twice on MiniPile with concept effective rank stuck in the `5-6 / 128` regime. The original intuition was that `prefix -> suffix` should be an easier semantic task than full masked diffusion. After comparing the implementation against masked diffusion LMs such as MDLM and LLaDA, that intuition does not hold for the current MrCogito architecture.

The core issue is not a single bug. It is a mismatch between:

1. a **semantic compression objective**,
2. a **tiny cross-attention-only diffusion decoder** with no suffix token coordination,
3. a **continuous concept interface** that is not actually tight in raw scalar capacity,
4. and **random-init training** on a broad-text continuation problem.

In short:

- Prefix diffusion is harder than masked diffusion in the current setup, not easier.
- `128 x 512` concepts are not a small bottleneck in raw information capacity, especially when `token_embedding_dim=64`.
- The decoder is too weak to turn concept states into coherent suffixes.
- The architecture gives the model no good path toward learning semantics before it learns residual shortcuts and degenerate low-rank concept use.

This note extends the earlier diagnosis with information-theoretic analysis, decoder self-attention trade-offs, AdaLN-Zero analysis, pretrained warm-start options, and lower-budget research alternatives.

---

## 1. Base Diagnosis

### 1.1 Why "prefix is simpler" is false here

The reasoning behind the prefix experiment was:

- self-reconstruction diffusion can hash surface form,
- prefix-to-suffix removes the trivial copy path,
- therefore it should force semantic concepts.

That logic is only valid if the decoder is strong enough to solve the conditional generation problem once it has semantics.

In the current implementation:

- MDLM and LLaDA operate on the **full token sequence** with **full bidirectional self-attention**.
- MrCogito prefix diffusion encodes only the prefix into concepts, then asks a **2-layer cross-attention-only decoder** to generate the suffix.
- Suffix tokens do **not** attend to each other.
- A masked suffix position therefore receives concept information, but not local lexical context from neighboring suffix positions.

So the actual optimization problem is:

`prefix semantics -> concept bank -> many independent token predictions`

That is much harder than masked token prediction with rich local context.

### 1.2 Why MDLM and LLaDA work

The successful masked diffusion LMs share four properties:

1. No bottleneck between input and denoiser.
2. Full self-attention over the entire corrupted sequence.
3. ELBO-consistent loss weighting across masking levels.
4. Much larger unique-data and compute scale.

They are effectively "BERT-style denoisers with principled generation", not bottlenecked encoder-decoder models.

### 1.3 What the MiniPile failures already proved

The two MiniPile prefix runs already ruled out several shallow explanations:

- It is not just "missing BiXT".
- It is not just "token embedding width too large".
- It is not just a bad random split strategy.
- It is not just evaluation-path corruption.

The v2 run was cleaner engineering, but concept geometry stayed collapsed.

---

## 2. Architecture-Level Root Causes

### 2.1 No suffix self-attention means conditional independence pressure

The decoder layer in `nn/concept_encoder_diffusion.py` performs:

- cross-attention from suffix queries to concepts,
- gated FFN,
- no suffix token self-attention.

That means two masked suffix positions cannot coordinate except indirectly through the same concept bank. This is acceptable for simple denoising only when the input already contains strong local context. It is much worse for continuation generation, where many suffix tokens are simultaneously uncertain.

In practical terms:

- predicting `"mat"` after `"the cat sat on the"` is easy with local self-attention,
- predicting `"mat"` from only concept vectors is much harder,
- predicting the whole phrase coherently without token-token communication is harder still.

This pushes the model toward unigram-like or topic-level predictions rather than coherent semantic continuation.

### 2.2 The concept interface is mismatched, not simply too small

At first glance, `512 tokens -> 128 concepts` looks like a `4:1` compression bottleneck.

But in the hardened prefix run:

- token stream lives in `token_embedding_dim = 64`,
- concept stream lives in `hidden_size = 512`,
- raw token-state scalars for a 400-token prefix are about `400 x 64 = 25,600`,
- raw concept-state scalars are `128 x 512 = 65,536`.

So in ambient continuous dimensionality, the model is **expanding**, not compressing.

This means:

- the bottleneck is not tight enough to force semantic abstraction,
- but it is still the wrong interface for specifying a detailed suffix,
- so it is simultaneously too loose for representation learning and too weak for generation control.

This is a key reason the model can still collapse into a few dominant directions: it is not being forced into a clean semantic compression regime.

### 2.3 Random-init continuation is a worse problem than random-init denoising

A random-init denoiser can learn from local token statistics almost immediately.

A random-init prefix generator must learn all of the following at once:

1. how to encode the prefix,
2. how to organize concepts,
3. how to map concepts to suffix token distributions,
4. how suffix positions relate to each other,
5. and how a continuation objective differs from simple reconstruction.

This is exactly the kind of problem where warm-start and curriculum usually matter more than yet another clean random-init baseline.

---

## 3. Training-Regime and Scale Comparison

### 3.1 MrCogito vs masked diffusion SoTA

| Dimension | MrCogito prefix diffusion | MDLM | LLaDA 8B |
|---|---|---|---|
| Model type | bottleneck encoder + diffusion decoder | encoder-only masked diffusion | bidirectional transformer diffusion |
| Bottleneck | yes | no | no |
| Self-attention on generated tokens | no | yes | yes |
| Parameters | about `98.3M` in the current hardened config | around `0.2B` released OWT model, larger in paper sweeps | 8B |
| Unique pretraining data | MiniPile ~0.6B tokens | OpenWebText / LM1B scale | 2.3T tokens reported publicly |
| Decoder start point | random init | large denoiser backbone | large denoiser backbone |
| Objective difficulty | semantic continuation through concepts | masked token denoising | masked token denoising at large scale |

### 3.2 Compute comparison must be read carefully

There are two different compute comparisons:

1. **Token-exposure compute**
   MrCogito sees roughly `0.6B x 20 = 12B` token presentations on MiniPile, so the raw number of training token exposures is not tiny.

2. **Unique-data + model-capacity compute**
   MrCogito repeatedly revisits the same small corpus with a weaker architecture. MDLM and LLaDA train on much larger unique corpora with stronger denoisers. For language modeling, repeated passes over the same `0.6B` tokens are not equivalent to training on `9B+` or `2.3T` unique tokens.

The fair conclusion is:

- relative to a small MDLM setup, MrCogito is not orders of magnitude smaller in raw token exposures,
- but it is much smaller in **unique data diversity**, **model capacity**, and **effective decoder capacity**,
- and compared to LLaDA-scale training it is smaller by an overwhelming margin.

### 3.3 Rough FLOPs framing

Using the usual dense-transformer rule of thumb `train_FLOPs ~= 6 * P * D`:

- a `98M`-parameter model over `12B` token exposures is on the order of `7e18` FLOPs,
- a `0.2B` model over `9B` tokens is on the order of `1e19` FLOPs,
- an `8B` model over `2.3T` tokens is on the order of `1e23` FLOPs.

These are rough estimates, not apples-to-apples measurements, but they make the key point clear:

- the small masked diffusion literature is already heavier than the current prefix track by rough FLOPs,
- the bigger difference is in **unique data scale** and **decoder quality**, not just in scalar FLOPs,
- the frontier diffusion LM literature is many orders of magnitude heavier.

---

## 4. Why SODA Transfers Poorly to Text

SODA works because novel-view prediction in vision has three favorable properties:

1. The two views share the same underlying scene.
2. The target is continuous and spatially smooth.
3. The decoder is strong enough to turn compact latent state into a new image.

Prefix-to-suffix text continuation lacks all three:

1. Prefix and suffix share topic, but not surface realization.
2. The mapping from meaning to exact next tokens is discrete and multi-modal.
3. The decoder has no strong generative prior of its own.

So the SODA analogy is directionally insightful, but too optimistic for text when paired with a small random-init decoder.

---

## 5. Initial Verdict

The prefix diffusion failure is best understood as an **architecture-objective mismatch**, not as a single implementation bug.

The current stack asks a random, low-capacity decoder to solve a continuation problem that in practice needs at least one of:

- strong suffix token coordination,
- strong pretrained language priors,
- a much tighter semantic bottleneck with an easier output target,
- or a training curriculum that first learns semantics and only then learns generation.

The remaining sections analyze the most important design questions raised after the initial diagnosis.

---

## 6. What The WikiText-103 Rescue Probe Changes

TODO 13c moves the training setup to:

- `Salesforce/wikitext`, `wikitext-103-v1`
- prefix ratio `0.7-0.8`
- `sentence_boundary` splits
- same diffusion noise range `t in [0.3, 1.0]`
- longer training horizon (`40` epochs by default)

This is a meaningful improvement in trainability because:

1. the suffix becomes shorter,
2. the prefix contains more context,
3. Wikipedia is much less multimodal than MiniPile,
4. and the continuation distribution is narrower.

For `max_seq_length = 512`, the new setup implies roughly:

- prefix length about `358-410` tokens,
- suffix length about `102-154` tokens.

That is much easier than the MiniPile `0.3-0.5` setup, where the suffix often occupied half or more of the document.

However, the rescue probe does **not** fix the main structural problems:

- the decoder still lacks suffix self-attention,
- the decoder is still random init,
- the concept interface is still broad in scalar capacity but weak as a sequence model,
- and the model still tries to learn semantic compression and generation jointly.

So the most realistic expectation is:

- suffix loss may improve,
- generation may look less chaotic,
- but concept rank may still remain far below project gates.

The WikiText probe is therefore still worth running as a trainability probe, but it should be interpreted as:

`Can a cleaner corpus partially rescue the current architecture?`

not as:

`Does prefix diffusion as a research direction fundamentally work?`

---

## 7. Information-Theoretic Analysis Of The Concept Bottleneck

### 7.1 The bottleneck is wider than it looks

For the new WikiText setting:

- prefix length: about `350-400` content tokens,
- concept bank: `128 x 512 = 65,536` continuous activations.

For the hardened MiniPile run with `token_embedding_dim = 64`:

- raw prefix token states: about `400 x 64 = 25,600` scalars,
- concept states: `65,536` scalars.

So in ambient state size, the concept layer is **2.5x wider** than the raw token embedding stream.

This means the current bottleneck is not an information bottleneck in the ordinary linear-algebra sense. It is only a bottleneck in:

- token count,
- attention topology,
- and optimization path.

That distinction matters. A model can fail to learn semantics even when it has more than enough raw channel capacity.

### 7.2 Exact lexical information vs semantic information

If we only count token identities, one token from a `50k` vocabulary carries about:

`log2(50,000) ~= 15.6 bits`

So a `400`-token prefix contains on the order of:

`400 x 15.6 ~= 6,240 bits`

of raw lexical identity, ignoring higher-order syntax and semantics.

The concept bank stores continuous activations. In principle, `128 x 512` floating values can carry far more raw bits than the lexical sequence. So the failure is not that the model lacks raw representational capacity. The failure is that:

- the decoder cannot use that capacity coherently,
- the optimization does not reward semantic factorization,
- and the learned solution collapses to a low-rank shortcut.

### 7.3 Effective capacity is much smaller than nominal capacity

The measured concept geometry in the failed runs is:

- effective rank about `5-6 / 128`,
- global effective rank about `5`,
- very high mean and max similarity.

So the optimization effectively uses only a tiny low-dimensional subspace.

This gives a more precise statement:

- **nominal capacity** is too wide to force semantic abstraction,
- **effective learned capacity** is too small to support generation.

That is why the current system can be both "too big" and "not enough" at the same time.

### 7.4 Is `128 x 512` too big for the WikiText problem?

For semantic abstraction alone, probably yes.

A Wikipedia prefix of `350-400` tokens usually contains only a handful of major semantic units:

- topic identity,
- discourse frame,
- 5-15 salient entities,
- a few relations,
- sentence-level discourse transitions.

That suggests a semantic bottleneck closer to:

- `16-64` concept slots,
- and possibly lower hidden size (`256-384`)

would create stronger pressure toward abstraction.

But with the **current decoder**, reducing concept count alone is risky because the decoder already relies on the concept bank as a substitute for missing suffix self-attention.

So the right conclusion is:

- **Do not** run more random-init `C` sweeps on the current decoder.
- Revisit `C=32/64` only after adding decoder coordination or pretrained language priors.

### 7.5 What compression ratio is more plausible?

If the objective is **semantic continuation**, the bottleneck should probably compress toward semantic units, not token units.

A practical heuristic for this project:

- `32-64` concepts for a `350-400` token prefix gives a semantic compression of about `6:1` to `12:1` in token count,
- but only if the decoder has its own sequence modeling power.

Without that stronger decoder, tighter compression likely just makes optimization fail faster.

---

## 8. Should The Decoder Get Self-Attention?

### 8.1 Yes, as a trainability probe

Adding suffix self-attention is the single most plausible architectural change if the goal is to test whether the current prefix objective is failing mainly because output tokens cannot coordinate.

The key question is not whether it preserves the pure `O(C*N)` story. It does not. The real question is whether the additional `O(S^2)` cost is acceptable at current sequence lengths.

### 8.2 Complexity at current prefix-diffusion lengths

Let:

- `C = 128` concepts,
- `S = suffix length`.

Current decoder cost per attention layer is dominated by cross-attention:

`O(S * C)`

If we add full suffix self-attention, the new cost becomes:

`O(S * C + S^2)`

For the current tasks:

- WikiText probe: `S ~= 102-154`
- MiniPile prefix runs: `S ~= 256-358`

Pairwise attention counts:

| Setup | `S*C` | `S^2` | Self-attn overhead |
|---|---:|---:|---:|
| WikiText, `S=128` | `16,384` | `16,384` | about `1x` cross-attn |
| WikiText, `S=154` | `19,712` | `23,716` | about `1.2x` cross-attn |
| MiniPile, `S=256` | `32,768` | `65,536` | about `2x` cross-attn |
| MiniPile, `S=358` | `45,824` | `128,164` | about `2.8x` cross-attn |

So for the current 512-token experiments, adding one or two suffix self-attention layers is **not** a research-vision violation. It is a reasonable trainability ablation.

### 8.3 Does this break the long-context vision?

Asymptotically, yes: full suffix self-attention loses the strict `O(C*N)` decoder story.

Practically, not necessarily:

1. the encoder already has `O(C^2)` concept self-attention, so the whole model is not purely `O(C*N)` today,
2. one or two decoder self-attention layers at `S <= 154` are cheap,
3. and if self-attention proves necessary, it can later be replaced by:
   - local-window self-attention `O(S*w)`,
   - block attention,
   - slot-level latent refinement,
   - or AR decoding.

### 8.4 Recommendation

If the prefix line continues, add **one small decoder self-attention block** as a diagnostic:

- self-attn over suffix tokens,
- then cross-attn to concepts,
- then FFN,
- keep decoder depth small (`1-2` blocks),
- run only one clean WikiText probe.

If this does not materially improve rank or continuation quality, the issue is deeper than token coordination alone.

---

## 9. Is The Current Encoder/Decoder Attention Pattern Conceptually Right?

The current high-level pattern is:

- **Encoder:** concepts attend to tokens.
- **Decoder:** output queries attend to concepts.

This is conceptually sound and matches the Perceiver IO idea:

- latent queries compress an input sequence into a fixed latent bank,
- output queries read back task-relevant information from that bank.

So the pattern itself is **not** the mistake.

The actual problem is that this pattern is incomplete for generation.

### 9.1 Why the encoder side makes sense

In `ConceptEncoder`:

- concept slots act as learned latent queries,
- they pull information from the token sequence,
- concept self-attention mixes and refines those summaries,
- BiXT improves this by also updating token states.

This is a good way to build a fixed-size latent representation of a variable-length prefix.

### 9.2 Why the decoder side is insufficient by itself

In `ConceptEncoderForPrefixDiffusion`:

- masked suffix positions query the concept bank,
- but if a suffix token is masked, its query content is mostly position plus `[MASK]`,
- and there is no query-query interaction.

So token-to-concept attention alone does not give enough structure for:

- phrase formation,
- syntax repair,
- agreement across nearby tokens,
- or multi-token disambiguation.

### 9.3 Conclusion

Keep the encoder/decoder asymmetry:

- **concepts read tokens** in the encoder,
- **output positions read concepts** in the decoder.

But treat it as only the outer skeleton. For generation, the decoder still needs one of:

- self-attention,
- recurrence,
- latent slots,
- or autoregressive state.

---

## 10. AdaLN-Zero: Helpful Stabilizer Or Blocker?

### 10.1 What happens mathematically

The decoder layer uses AdaLN-Zero with zero-initialized modulation weights:

- `gate_ca = 0` at initialization,
- `gate_ff = 0` at initialization.

If a residual branch has the form:

`y = x + g * f(x, c, t)`

then the gradients at initialization are:

- `dL/df = g * dL/dy`
- `dL/dg = <dL/dy, f>`

So when `g = 0`:

- branch parameters inside `f` get **zero gradient**,
- only the gate parameters get updated first.

This means the decoder starts by learning gate opening before the concept-attention path itself can learn.

### 10.2 Is AdaLN-Zero the root cause?

No. It is probably **not** the fundamental reason the experiments failed.

But in this specific architecture it is a plausible amplifier of failure because:

1. the concept path is already the hardest path in the network,
2. the decoder has no self-attention to compensate,
3. and the residual input already provides a shortcut through noisy suffix embeddings.

So AdaLN-Zero likely encourages an early regime where the model learns to do as much as possible without concepts.

### 10.3 What makes this architecture unusually sensitive

DiT-style models tolerate AdaLN-Zero because they usually have:

- many layers,
- a strong denoising backbone,
- and direct sequence modeling power.

Here the decoder has only:

- 2 layers,
- no suffix self-attention,
- and concepts as its main hard path.

That makes the zero-gated start much more damaging.

### 10.4 Recommendation

Treat AdaLN-Zero as a **secondary blocker**, not the primary one.

If the prefix track continues, test one of these:

1. initialize `gate_ca` bias to a small positive value (`0.05-0.1`);
2. keep zero init for FFN, but not for concept cross-attention;
3. replace AdaLN-Zero with standard FiLM/AdaLN on the cross-attention block;
4. warm-start the decoder so the gated path is already useful.

Do not test AdaLN-Zero changes in isolation on the old decoder. Pair them with either decoder self-attention or pretrained warm-start.

---

## 11. Warm-Starting From A Pretrained Small LM

This is the most credible path if prefix generation remains a live research direction.

### 11.1 Why warm-start matters

Random init forces the model to learn:

- language understanding,
- semantic abstraction,
- latent compression,
- and continuation generation

all at once.

Warm-start lets the model inherit language priors first, which is exactly how LLaDA 2.0 and many latent-reasoning systems become trainable.

### 11.2 Best candidate types

There are two distinct warm-start targets:

1. **Encoder warm-start**
   Use a strong bidirectional model to contextualize prefix tokens before concept compression.

2. **Decoder warm-start**
   Use a small generative model so suffix formation is not learned from scratch.

These are not equally easy.

### 11.3 ModernBERT as encoder warm-start

`answerdotai/ModernBERT-base` is a particularly strong fit for the encoder side:

- bidirectional,
- has a real `[MASK]` token,
- hidden size `768`,
- 22 layers,
- about `149M` parameters,
- trained on `2T` tokens.

This matches the prefix encoding problem much better than a decoder-only LM.

Practical implementation path:

1. tokenize with ModernBERT,
2. run a frozen or LoRA-tuned ModernBERT prefix encoder,
3. project its contextual token states into the concept encoder,
4. let concepts compress those contextualized tokens,
5. keep the concept bank as the shared interface.

This is the lowest-risk warm-start path for semantics.

### 11.4 SmolLM2-135M as decoder or hybrid backbone

`HuggingFaceTB/SmolLM2-135M` is more suitable for the decoder side:

- hidden size `576`,
- 30 layers,
- 9 attention heads,
- vocab size `49,152`,
- trained on `2T` tokens using `64 x H100`,
- decoder-only Llama-style architecture.

This gives a real language generation prior.

But there are two complications:

1. it has no native `[MASK]` token,
2. its architecture does not match the current cross-attention-only diffusion decoder.

So if SmolLM2 is used, the cleaner design is:

- add a self-attention decoder,
- initialize self-attn + FFN from SmolLM2,
- add random-init cross-attention from decoder states to concepts,
- optionally freeze most LM layers and train only projections + cross-attn first.

### 11.5 Tokenizer implications

Tokenizer choice is not a detail here. It determines whether embedding warm-start is even valid.

If you use a pretrained model, the easiest rule is:

- **Use the pretrained model's tokenizer.**

Why:

- embedding matrices transfer directly,
- token frequency statistics match the pretrained model,
- and special-token semantics remain intact.

If you keep ModernBERT tokenization but import SmolLM2 weights, or vice versa, you lose most of the benefit unless you learn a vocabulary-alignment projection.

### 11.6 Recommended warm-start paths

Ordered by practicality:

1. **Best semantic probe:** ModernBERT encoder warm-start + current concept encoder + improved decoder.
2. **Best generation probe:** SmolLM2-style self-attention decoder + cross-attn to concepts.
3. **Best full system path:** pretrained encoder + pretrained decoder, with concepts as a learned interface in between.

What not to do:

- do not only copy token embeddings into the current model and call it warm-start,
- do not keep the current cross-attention-only decoder and expect a causal LM warm-start to transfer cleanly,
- do not ignore tokenizer mismatch.

---

## 12. What If The Target Only Needs To Be Semantically Similar?

This question is extremely important.

A document suffix is not deterministic given the prefix. Many continuations are semantically valid. Exact token cross-entropy punishes all but one of them.

For a standard strong LM, this is manageable because the model can learn the full conditional distribution. For the current bottleneck model, that exact-token pressure may be too sharp.

### 12.1 Why semantic targets change the problem

If the bottleneck only needs to preserve:

- discourse topic,
- entity relations,
- intent,
- sentence-level meaning,

then the continuation problem becomes much closer to representation learning and much less like lexical memorization.

That aligns better with the goal of concept tokens.

### 12.2 Existing precedents

There is real precedent for this idea:

- **Skip-thought vectors** train sentence encoders by predicting surrounding sentence content.
- **Large Concept Models (LCM)** operate in sentence representation space instead of token space.
- **TSDAE / SimCSE / sentence embedding methods** show that semantic spaces can be trained with denoising and contrastive targets.

The most relevant of these is LCM:

- they predict future sentence-level concepts rather than exact token sequences at the latent stage,
- then decode back to text,
- which is much closer to the spirit of MrCogito than token-level prefix diffusion is.

### 12.3 Teacher-student semantic auxiliary loss

A practical low-budget version for MrCogito:

1. choose a strong frozen sentence/paragraph encoder,
2. encode the true suffix into a semantic target vector,
3. predict that vector from prefix concepts,
4. train with contrastive or cosine loss,
5. optionally combine with token loss.

Possible teachers:

- ModernBERT pooled sentence embeddings,
- Sentence-BERT family,
- E5 / GTE style embedding models,
- SONAR-like multilingual sentence encoders if cross-lingual transfer matters later.

Useful losses:

- cosine similarity,
- InfoNCE with in-batch negatives,
- MSE in normalized embedding space,
- or teacher-student distillation over a small concept predictor head.

### 12.4 Does this make sense scientifically?

Yes. It changes the problem from:

`predict this exact lexical suffix`

to:

`predict a suffix that is semantically appropriate`

That is far closer to what concept tokens are supposed to carry.

### 12.5 Caveat

A pure semantic target is not enough if the end goal is generation. Eventually a decoder must still turn meaning into fluent text.

So the best use of this idea is:

- first train concepts to predict suffix semantics,
- then train or warm-start a decoder to realize those semantics as text.

This is much closer to a staged curriculum than the current one-shot setup.

---

## 13. Budget-Aware Research Ideas

The key constraint is that current compute is not enough for repeated random-init generative bets. So every new idea should either:

- reuse pretrained language knowledge,
- reduce target entropy,
- or improve gradient quality.

### 13.1 Most promising low-budget ideas

1. **One decoder self-attention probe on WikiText**
   Small code change, highly diagnostic.

2. **Prefix -> suffix semantic prediction**
   Use a frozen sentence encoder as teacher. Cheap and directly aligned with concept learning.

3. **ModernBERT encoder warm-start**
   Strongest low-risk semantic upgrade.

4. **AR suffix decoder conditioned on concepts**
   Replace diffusion with a much easier and better understood sequence model.

5. **Two-stage curriculum**
   Stage 1: learn semantic concepts.
   Stage 2: learn generation from concepts.

### 13.2 Topology / manifold view

The topology framing is useful as an intuition, even if not literal algebraic topology.

Think of:

- the prefix as one high-dimensional observation of meaning,
- the suffix as another high-dimensional observation of related meaning,
- and the concept bank as coordinates on a lower-dimensional semantic manifold.

The current model tries to learn:

`prefix surface form -> manifold coordinates -> suffix surface form`

in one shot.

That is too hard because the decoder is weak. In manifold terms, the model has not learned a stable chart for lifting semantic coordinates back into token sequences.

A better decomposition is:

1. learn good coordinates first,
2. ensure nearby semantic coordinates map to nearby semantic suffixes,
3. then learn a strong chart from semantic space back to text.

This is exactly why semantic-teacher losses and pretrained decoders are attractive.

### 13.3 Concrete small-compute alternatives

- **Next-sentence concept prediction**
  Predict the next sentence embedding instead of its exact tokens.

- **Hierarchical continuation**
  Predict only the next sentence first, not a 100+ token suffix.

- **Local self-attention decoder**
  Use windowed attention over suffix tokens to keep near-linear cost.

- **Latent slot bridge**
  Decode concepts into `16-32` suffix-plan slots, then text from slots.

- **Curriculum on suffix length**
  Start with 1 sentence, then 2, then paragraph-length suffixes.

- **Contrastive semantic continuation**
  Match true suffix semantics against in-batch negatives instead of exact token CE only.

---

## 14. Recommendations

### 14.1 What to stop

- Stop additional random-init MiniPile prefix-diffusion clean baselines.
- Do not spend time on concept-count sweeps with the current decoder.
- Do not interpret the current failure as evidence that "concepts for generation do not work".

### 14.2 What to finish

- Let the WikiText-103 rescue probe finish.
- Evaluate it mainly as a **trainability probe**, not as a final verdict.

Decision gate:

- if concept rank remains `< 10 / 128`, random-init diffusion prefix training should be considered closed.

### 14.3 Highest-value next experiments

1. **WikiText + one decoder self-attention ablation**
   Most diagnostic architecture fix.

2. **Semantic suffix teacher loss**
   Best fit to the concept-learning goal.

3. **ModernBERT encoder warm-start**
   Strongest practical warm-start for semantics.

4. **Concept-conditioned AR decoder**
   Best fit to generation once semantics exist.

### 14.4 Strategic recommendation

The strongest strategic conclusion is:

**Do not keep scaling the current random-init diffusion prefix architecture.**

If the prefix line continues, pivot to:

- pretrained backbone initialization,
- better decoder coordination,
- and likely a staged semantic-to-text curriculum.

Otherwise, focus effort on the denoising / TSDAE-style track, which is much better aligned with the current compute budget and architecture maturity.

---

## 15. Key Comparative Findings From Repositories And Papers

### MDLM

Repository configuration and paper together indicate:

- DiT-style denoiser backbone,
- `hidden_size=768`, `12` blocks in the small config,
- sequence length `1024`,
- global batch size `512`,
- learning rate `3e-4`,
- bf16 training,
- `sampling_eps=1e-3`,
- `parameterization=subs`,
- `max_steps=1,000,000`,
- OpenWebText training config in the public repo,
- and roughly two weeks on `8 x A100` for the largest OpenWebText models reported in the paper checklist.

Their core advantage is not just ELBO weighting. It is that the denoiser is a real sequence model with global self-attention and no bottleneck.

### LLaDA

Publicly reported details indicate:

- 8B bidirectional transformer,
- masking rate sampled per sequence,
- loss scaled by inverse mask probability `1 / p_mask`,
- and about `2.3T` training tokens for the 8B model.

Again, the crucial point is that the model denoises the full sequence directly.

### Current MrCogito prefix model

A local instantiation of the current hardened prefix setup (`H512`, `L6`, `C128`, `token_dim=64`, `D2`) gives about:

- **98.3M parameters**

So the current model is not tiny in parameter count. The problem is where those parameters are allocated and what they are being asked to do.

---

## Final Conclusion

The current prefix diffusion setup fails because it combines:

- a nominally wide but semantically weak concept interface,
- a decoder with no suffix token coordination,
- a hard continuation objective,
- and random-init training.

That combination is poor for both semantics and generation.

The next step should not be "one more clean random-init diffusion run". The next step should be a more staged and better-conditioned design:

`pretrained language understanding -> semantic concept bottleneck -> strong coordinated decoder`

That is the shortest path from the current failures to a research result that actually tests the concept-token hypothesis rather than the limits of an underpowered decoder.
