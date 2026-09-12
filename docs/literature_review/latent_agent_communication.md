# Latent (non-textual) communication between LM agents

Topical review (2024 → Sep 2026) for the Vision's Stage-2 headline — cooperating model instances
exchange latent vectors instead of text — and for the 2026-09-11 strategy proposal that the
sender→receiver latent message should be a **pretraining objective**, not a post-hoc channel
(see [`../4_Research_Notes/strategy_synthesis_latent_channel_20260911.md`](../4_Research_Notes/strategy_synthesis_latent_channel_20260911.md)
and spec [E21](../experiments_specs/ahead/E21_latent_message_pretraining.md)).

Collected 2026-09-11 (research-scout sweep). Per paper: what is transmitted · how the receiver
ingests it · trained or training-free · evidence · limitation. Related:
[`recurrent_memory_transformers.md`](recurrent_memory_transformers.md) (Coconut and writable memory),
[`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md) (BAPO — the
two-party prefix/suffix model that *is* a communication-complexity model).

---

## LatentMAS — Latent Collaboration in Multi-Agent Systems
[arXiv:2511.20639](https://arxiv.org/abs/2511.20639) · [code](https://github.com/Gen-Verse/LatentMAS) · ICML 2026 spotlight, HF Paper of the Day.

### TL;DR
Training-free MAS in continuous space. Each agent (same backbone, role prompts) produces ~10
autoregressive **latent thoughts** (last-layer hidden states fed back as `inputs_embeds`) and
hands its **full per-layer KV cache** to the next agent, whose cache is restored/concatenated
layer-wise. On 9 benchmarks (GSM8K, MATH, AIME, commonsense, code): **+14.6% accuracy,
−70.8–83.7% output tokens, 4–4.3× wall-clock** vs text MAS; scales to 14B+ with an HF+vLLM
hybrid. **Limits:** same checkpoint only; the message is the *entire* working memory (nothing is
compressed); causal audits ([arXiv:2608.04893](https://arxiv.org/html/2608.04893)) find part of
the gain is interface perturbation rather than transmitted content.

## Cache-to-Cache (C2C) — direct KV communication between LLMs
[arXiv:2510.03215](https://arxiv.org/abs/2510.03215) · [code](https://github.com/thu-nics/C2C) · [fuser weights](https://huggingface.co/nics-efc/C2C_Fuser) · ICLR 2026.

### TL;DR
Cross-*model* transfer: a "sharer" model's per-layer prefix **K/V** are mapped by a learned
projector with per-layer Gumbel gates and **fused into the receiver's running KV**. Frozen LLMs;
only the fuser is trained (NTP on OpenHermes-2.5). Qwen3-0.6B ← Qwen2.5-0.5B: **+8.5–10.5%** vs
solo, **+3–5%** vs text relay, ~2× lower latency. **Limits:** pair-specific fusers; message size =
full KV; multi-sharer preliminary.

## ThoughtComm — Thought Communication in multi-agent collaboration
[arXiv:2510.20733](https://arxiv.org/abs/2510.20733) · NeurIPS 2025 spotlight.

### TL;DR
Agents' **last-token hidden states** are concatenated and encoded by a sparse autoencoder into
identifiable *shared* and *private* latent thoughts (Jacobian-sparsity routing); each agent gets its
share back through a **prefix adapter**. **+19.06%** average over multi-agent fine-tuning on math
reasoning. Theory-first; small agent counts; AE + adapter training required.

## Interlat — inter-LLM latent communication
[arXiv:2511.09149](https://arxiv.org/abs/2511.09149) · [ACL 2026](https://aclanthology.org/2026.acl-long.1248/) · [code](https://github.com/XiaoDu-flying/Interlat)

### TL;DR
The closest to a *trained, compressed* channel: the sender's **last-layer hidden states** per step
are compressed to **~8 vectors (1–3% of the trajectory length)**; the receiver is trained jointly
(token↔latent mixing curriculum) with a **utilisation loss** so it actually attends the message.
Cross-family (Qwen → LLaMA) works. ALFWorld + symbolic tasks: up to **24×** lower communication
latency, **+8–10 pts** over text CoT. Feasibility study; 2 agents.

## StateBridge — training-free hidden-state handoff
[arXiv:2608.13317](https://arxiv.org/pdf/2608.13317) · [code](https://github.com/YanwenPeng/StateBridge)

### TL;DR
Sender final-layer states → **Procrustes alignment + norm calibration + vocabulary anchoring** →
receiver `inputs_embeds` prefix. Best or tied on 22/26 model pairs (math/code/QA). **Key
ablation:** a *random* aligned prefix still gives 48.8% (Qwen3-4B) — a warning that "the message
helps" can be a generic steering effect, not content.

## State Delta Encoding (SDE)
[EMNLP 2025](https://aclanthology.org/2025.emnlp-main.518/) · [code](https://github.com/LittleDinoC/StateDelta/)

### TL;DR
Transmit natural-language tokens **plus** per-token hidden-state *deltas* (hᵢ − hᵢ₋₁) at top layers,
added to the receiver's states during encoding. +0.3–17.3% on multi-hop QA / debate / workflows;
strongest on logic. Hybrid, same model, training-free.

## Activation grafting (Ramesh et al.)
[arXiv:2501.14082](https://arxiv.org/abs/2501.14082) · [ICML 2025](https://proceedings.mlr.press/v267/ramesh25a.html)

### TL;DR
One **last-token activation at layer k** of model A is summed/averaged/replaced into layer j of
model B, which resumes its forward. Coordination games and reasoning: **+27% vs natural language
at < ¼ compute**, 1–7B. Single vector; layer choice is hand-tuned.

## DroidSpeak — KV-cache sharing for multi-LLM serving
[arXiv:2411.02820](https://arxiv.org/abs/2411.02820) · [NSDI 2026](https://www.microsoft.com/en-us/research/publication/droidspeak-kv-cache-sharing-for-efficient-multi-llm-serving/)

### TL;DR
Serving-time reuse of a prefix KV across fine-tuned variants of one architecture, recomputing only
"critical layers": 4× throughput, 3.1× faster prefill, negligible quality loss. Efficiency, not
new capability — but the `store/fetch/partial_prefill` API is what a latent envelope would look like.

## Latent Agents / IMAD — distilling debate into weights
[ACL 2026](https://doi.org/10.18653/v1/2026.acl-long.709)

### TL;DR
Internalise multi-agent debate via SFT + RL so no runtime message exists: 5–16× fewer tokens than
debate; LLaMA-3.1-8B beats debate on GSM8K / MMLU-Pro / BBH. Not an A2A channel; a reminder that
the *cheapest* way to beat text MAS is to not communicate at all — a latent channel must beat this.

## Surveys and audits
- **Beyond Tokens** — taxonomy of 18 latent-communication methods. https://arxiv.org/abs/2606.05711
- **Causal audits** — aggregate gains can come from wrong-example KV or generic prefixes; proposes
  swap-the-message controls. https://arxiv.org/html/2607.26773v1 · https://arxiv.org/html/2608.04893
- **Classics:** DIAL (backprop through a noisy discretised channel) https://arxiv.org/abs/1605.06676 ·
  Lazaridou referential games (REINFORCE on communication success) https://arxiv.org/pdf/1804.03984 —
  the recipe is *differentiable channel + task reward*, which frozen-LLM methods cannot use.

## Industry / protocol status (Sep 2026)

| Layer | Ships | Latent? |
|---|---|---|
| MCP (GA 2026-07-28) | JSON-RPC tools/resources/prompts https://modelcontextprotocol.io/specification/2026-07-28 | no |
| Google A2A v1.0 (Mar 2026) | agent cards, task delegation https://a2a-protocol.org/v1.0.0/specification/ | no — payloads textual/structured |
| OpenAI Agents SDK / Anthropic agent teams | handoffs, shared git/files | no |
| SGLang / vLLM prefix caching | KV reuse as an optimisation https://mintlify.wiki/sgl-project/sglang/concepts/prefix-caching | KV, not a protocol |
| Speculative decoding (LongSpec) | draft cross-attends target KV https://arxiv.org/abs/2502.17421 | internal only |

No vendor exposes a "latent message" API. No funded company productises cross-agent latent
transfer (see [`../1_Strategy_and_Plans/positioning_and_funding.md`](../1_Strategy_and_Plans/positioning_and_funding.md)).

---

## Synthesis for [REDACTED]

**(a) What every working method has in common.** Same-checkpoint copies (LatentMAS, SDE) get a
shared geometry for free; cross-model needs a projector (C2C, Interlat) or closed-form aligner
(StateBridge). The transmitted object is either the **whole KV** (lossless, huge, un-compressed)
or a **small hidden-state prefix** (compressible to ~8–128 vectors but needs a *utilisation loss*
or the receiver ignores it). Every method bolts the channel onto a text-pretrained model.

**(b) Failure modes we must design against.** Receiver ignores the message (attention dilution);
message collapses into a generic steering vector (StateBridge random-prefix ablation); no gradient
to the sender when the sender is frozen (no adaptive protocol); headline gains that vanish under a
swap-the-message causal control.

**(c) The gap, stated precisely.** Nobody has *pretrained* a model whose global read K/V — the
object every method above transmits post hoc — is trained from step 0 to be a **compressed,
sender-independent message** with a receiver that has *no other access* to the source. That is what
our ledger says is required for a latent channel to carry load (E02 prefix→suffix worked because the
decoder had no raw access; every CE-trained memory went dead). It is also exactly BAPO's two-party
model turned into a training objective. E21 does this on the E18 platform: sender and receiver are
the *same weights*, the message is the (slot-compressed) `prefix_kv()` of the global read, and the
receiver's suffix CE / retrieval accuracy is the loss. Causal controls (no-message arm,
swapped-message arm) are pre-registered, per the audit literature.

**(d) A VC-legible ≤1B demo (from the survey).** Two copies of one ≤600M model: the *reader* ingests
a 32k–128k document and emits a 2k–8k-slot message (≈ 1–4 MB); the *answerer* sees only the message
plus the question. Metrics: task accuracy (RULER-style multi-needle + a long-doc QA set) vs (i)
text summary relay at equal bytes, (ii) full-document prompt (upper bound), (iii) swapped-message
control (must collapse ≥ 10 pp); bytes per message and p50 latency; the same artifact must also
serve as the model's own 1M-token cache. Win line: ≥ 90% of the full-document accuracy at ≤ 1/16 of
the bytes, positive causal gain, reproducible on one GPU.
