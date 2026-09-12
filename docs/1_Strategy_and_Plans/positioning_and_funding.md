# Positioning and funding — how a two-person architecture lab becomes legible (2026-09-11)

**Status:** proposal from the 2026-09-11 strategy review (branch `cursor/strategy-sota-review-2026-09-e212`);
mutable, correct in place. Evidence: [`../literature_review/frontier_open_models_architecture.md`](../literature_review/frontier_open_models_architecture.md),
[`../literature_review/latent_agent_communication.md`](../literature_review/latent_agent_communication.md),
and the 2026-09-11 funding-landscape scout (URLs inline). Research direction it serves:
[`../4_Research_Notes/strategy_synthesis_latent_channel_20260911.md`](../4_Research_Notes/strategy_synthesis_latent_channel_20260911.md).

## 1. What the landscape says (Sep 2026)

**Architecture bets do get funded, but on one of two proofs.** Either *pedigree* (Mistral's
€105M seed four weeks in with no product, [TechCrunch](https://techcrunch.com/2023/06/13/frances-mistral-ai-blows-in-with-a-113m-seed-round-at-a-260m-valuation-to-take-on-openai/);
Thinking Machines' $2B seed, [Reuters](https://www.reuters.com/technology/mira-muratis-ai-startup-thinking-machines-raises-2-billion-a16z-led-round-2025-07-15/))
or **one auditable number plus an open artifact**: Magic's 100M-token context
([blog](https://magic.dev/blog/100m-token-context-windows)), Inception's 1000+ tok/s diffusion LM
([$50M seed](https://techcrunch.com/2025/11/06/inception-raises-50-million-to-build-diffusion-models-for-code-and-text/)),
Pathway's BDH 29.5% ARC-AGI-1 at ~$0.0007/task with an independent replication
([CNBC-TV18](https://www.cnbctv18.com/business/startup/ai-startup-pathway-raises-funding-at-500-million-valuation-says-its-model-can-reason-at-lower-cost-19968804.htm)),
Manifest's "Brumby-14B retrained for $4k" ([Forbes](https://www.forbes.com/sites/the-prompt/2025/09/23/this-research-lab-is-giving-ai-a-better-memory/)),
Sapient's 27M-param HRM at 40% ARC-AGI-1 (~$22M seed). We have no pedigree lever; we have the
second route, and it runs through an artifact, not a deck.

**What VCs say they buy in 2026:** inference economics and tokens-per-watt over raw capability
([a16z](https://a16z.com/how-to-win-the-largest-market-in-ai/)); differentiation in post-training,
workflow and control plane, not architecture alone ([Opulentia](https://www.opulentia.vc/cheap-intelligence-the-new-genai-investment-thesis/),
[Turing VC](https://turingventurecapital.com/thesis)); and the standard reason to pass is
"the frontier will scale past it" ([a16z 2023](https://a16z.com/who-owns-the-generative-ai-platform/)).
A pure "our attention is cheaper" pitch is now a commodity claim: DeepSeek-V4.1-Flash ships an
encoder-projected 890 B/token global cache ([card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)),
Qwen3.8-Flash-Next runs 1M with QSA + 51B host-memory n-gram tables
([report](https://arxiv.org/html/2608.30320)). Our 1 KB/token cache is table stakes.

**Gaps nobody occupies as a company** (funding scout, confirmed against the literature review):
- *Latent agent-to-agent communication* — LatentMAS (ICML'26 spotlight), C2C (ICLR'26), Interlat
  (ACL'26) are papers and repos; every shipping protocol (MCP GA 2026-07-28, A2A v1.0, OpenAI
  Agents SDK, Anthropic agent teams) is text/JSON. No funded company sells a latent envelope.
- *Long-context small model with a portable, compressed context object* — Magic never shipped;
  Manifest sells kernels. Nobody owns "send the document's concepts, not the document."
- *Concept compression + latent reasoning as a product* — Pathway/BDH is the nearest public bet
  and it is a different mechanism.
- *Polish / CEE architecture-native seed* — CEE funds (Inovo, Market One, Kadmos, Łukasz Kaiser as
  angel) backed Pathway's infra story ([Nordic9](https://nordic9.com/news/pathway-raised-a-10-million-seed-round-led-by-tq-ventures-with-participation-from-kadmos-inovo-market-one-capital-id4-and-angel-investors/)),
  not a model lab; SPRIND's Next Frontier AI (€125M, 10 teams, explicit non-incremental
  architecture mandate, Jul 2026) is the new non-dilutive lane ([SPRIND](https://www.sprind.org/en/actions/challenges/next-frontier-ai)).

## 2. The positioning sentence

> **We train small language models whose native interface is a compressed latent message, not
> text.** One ≤1B model reads a million tokens into a few megabytes of concept slots; a second
> copy answers from the slots alone; agents built on it exchange concepts instead of prompts.
> Open weights, open recipe, reproducible on one GPU.

Three properties make this hard to "scale past": (i) it is an *objective*, not a kernel — the
frontier trains on next-token CE and would have to re-pretrain to get a native message space;
(ii) it is measurable with causal controls (no-message / swapped-message), which the latent-MAS
literature is currently missing and asking for; (iii) it composes with everything the frontier
ships (GDN mixers, sparse indexers, Engram) rather than competing with them.

## 3. The one number and the demo (what the artifact must show)

Pick **one headline metric** and make everything else supporting evidence:

**Headline:** *"A 600M model answers long-document questions from a 1/16-size latent message at
≥ 90% of full-context accuracy — and two copies cooperate through that message with zero text."*
Concretely: RULER-style multi-needle + a natural long-document QA set at 32k → 128k; accuracy of
answerer(slots only) / answerer(full document) ≥ 0.9; bytes(message) ≤ 1/16 bytes(bf16 K/V) and
≤ 1/4 of a text summary at equal accuracy; swapped-message control collapses ≥ 10 pp.

**Supporting numbers on the same checkpoint:** RULER @128k ≥ 80 (SmolLM3-3B: 61); NIAH @1M;
prefix cache bytes at 1M on one 24 GB GPU; short-context parity with SmolLM2-360M / Qwen3-0.6B
(HellaSwag / ARC / MMLU); throughput vs the matched dense control. Latent-reasoning steps
(E19) add a second curve: accuracy vs number of message-refinement steps.

**Demo (HF Space, one 3090):** drop a 500-page PDF → watch it become N slots (show bytes) →
chat with the *answerer* copy that never sees the text → toggle "swap message" to show the
answer collapse → toggle "text summary at equal bytes" to show it losing. This is the
Magic/Inception/Pathway pattern: one memorable mechanic, self-verifiable in the browser.

## 4. Recognisability levers (ranked by cost-effectiveness for us)

1. **Open weights + open recipe on HF** with the demo Space and a from-scratch training log (the
   modded-nanogpt / nanochat credibility pattern) — trending is earned by reproducibility.
2. **A pre-registered result with causal controls**, written as a paper for an ICLR/NeurIPS
   workshop first, main track after the 600M run; the latent-MAS audit papers are the reviewers
   to satisfy ([2607.26773](https://arxiv.org/html/2607.26773v1), [2608.04893](https://arxiv.org/html/2608.04893)).
3. **Leaderboard entries where small models are absent:** RULER-1M and HELMET have almost no
   ≤1B entries; a 600M entry with a compressed cache is visible by scarcity.
4. **A public "reach ablation" instrument**: release `--probe reach` / the no-channel-control
   protocol as a standalone tool ("does your long-context layer carry information?"). Tools travel
   further than models at our size.
5. **Independent replication invitation**: publish the tiny CPU copy study and the 125M arms so a
   third party can reproduce the negative (CE does not train the read) and the positive (P2).
6. **European sovereignty framing** (Mistral/Kyutai/ElevenLabs-Warsaw pattern): a Polish lab with
   Jean Zay / EuroHPC compute and an open model — policy press is cheap and compounding.
7. Blog + X/LinkedIn threads keyed to each gate result, never to plans.

## 5. Funding lanes (non-dilutive first, then seed)

| Lane | What | Fit | Action |
|---|---|---|---|
| **SPRIND Next Frontier AI** | €125M challenge, 10 teams, non-incremental architectures | Exact mandate; team brief already exists (`sprind_frontier_ai/`, local) | Track the next cohort call; the E21 artifact is the application's core |
| **Jean Zay Dynamic Access** | ≤ 50k normalised GPU-h rolling ([GENCI](https://www.genci.fr/en/news/new-allocation-campaign)) | E19–E21 runs | Apply as soon as the E21 pilot passes its first gate |
| **EuroHPC AI Factories / large-scale access** | H100/GH200 allocations for AI | 600M main run alternative to AWS | Prepare the compute request from the AWS one-pager |
| **AWS credits ($100K)** | already held | E18/E21 main run | Spend only after the pilot gates (unchanged policy) |
| **HF / NVIDIA Inception, Google TPU Research Cloud** | credits, visibility | demo Space, evals | Apply at first open release |
| **CEE pre-seed / seed** (Inovo, Market One, Movens, SMOK, Kaya; angels from Pathway's round) | $1–3M pre-seed | After the artifact; lead with the demo and the SPRIND/Jean Zay signal | Warm intros via the Polish AI research community |
| **US architecture-friendly seeds** (the funds behind Inception, Manifest, Sapient, Pathway) | $5–20M seed | Needs the public artifact + a paper | After a HF-trending release |

## 6. Gate-ordered path (no calendar; each step is a registered spec)

1. **E18b** (in flight): does dense retrieval supervision make the read a general retriever?
   → the retrieval-class long-context claim; unblocks the compressed read.
2. **E18c**: slot-count compression of the read (r=16) with retrieval retention as the metric.
3. **E21 latent-message pretraining** (new spec): sender/receiver split of one model over the
   slot message; no-message and swapped-message controls. *This is the artifact.*
4. **E19 write-back reasoning steps** on the same platform: accuracy vs refinement steps.
5. **600M main run** (AWS or Jean Zay) with the E21 objective in the mix from stage 1, the
   dense control, RULER/HELMET/LongBench-v2 at 128k–1M, the Space, the paper.

Kill the positioning if E21's pilot cannot beat a text summary at equal bytes with a positive
causal gain: then the honest product is "cheap long-context retrieval at 1 KB/token", which is a
feature the frontier already sells, and the lab should pivot to the reasoning axis (E19) alone.
