# Latent / continuous / recurrent-depth reasoning in language models

Topical review (2025 → Sep 2026) for the Vision's priority 1/3 (reason in concept space; recursion
depth as a compute axis) and for E19 (write-back refinement steps on the E18 platform). The frame:
**what has actually trained and won at 100M–3B, and what fails.** Collected 2026-09-11
(research-scout sweep); items not re-read directly are the scout's summary.

Related, do not duplicate: Coconut / Huginn / Ouro are also reviewed as *writable memory* in
[`recurrent_memory_transformers.md`](recurrent_memory_transformers.md) §B; the theory side (BAPO
Thm 8: chain-of-thought turns any problem constant-bandwidth; Pfau's hidden-computation barrier) is in
[`reasoning_bandwidth_information_flow.md`](reasoning_bandwidth_information_flow.md); the agenda's
2026-06-13 scan ("use Ouro not TRM; gains are task-selective; measurement is the bottleneck") stands.

---

## Pretrained recurrence (the family that wins at scale)

- **Ouro / LoopLM** (ByteDance) — 24 layers × 4 recurrent steps, entropy-regularised adaptive exit,
  **7.7T tokens** multi-stage pretraining; 1.4B/2.6B match 3–12B transformers; ablations attribute
  gains to knowledge *manipulation*, not capacity. The only open-weight pretrained looped LM.
  https://arxiv.org/abs/2510.25741 · https://huggingface.co/ByteDance/Ouro-2.6B
- **Huginn** — prelude / recurrent block / coda, from scratch 3.5B / 800B tokens, no CoT data;
  math/code improve with loops (≈ 50B-param effective compute) but probes find **no CoT-like latent
  structure** (https://arxiv.org/abs/2507.02199). https://arxiv.org/abs/2502.05171 · https://github.com/seal-rg/recurrent-pretraining
- **Mixture-of-Recursions** — per-token recursion depth via a router, recursion-wise KV cache;
  Pareto-better ppl/few-shot at **135M–1.7B**, ~2× throughput; an efficiency story more than a
  reasoning one. https://arxiv.org/abs/2507.10524 · https://github.com/raymin0223/mixture_of_recursions
- **PonderLM / PonderLM-2** — a latent "thought" refinement before each token during pretraining
  (300B Pile tokens); 1.4B ≈ 2.8B vanilla. https://arxiv.org/abs/2505.20674 · https://arxiv.org/abs/2509.23184
- **Thoughtbubbles** — pretraining learns to *fork/delete* residual streams mid-network (parallel
  latent paths), LM loss only, 150M–772M; 319M beats 772M on ppl/HellaSwag/LAMBADA; no GSM8K/ARC yet.
  https://arxiv.org/abs/2510.00219 · https://github.com/stanfordnlp/thoughtbubbles
- **Saunshi et al., Reasoning with Latent Thoughts** — k layers looped L times simulate T CoT steps;
  loop regularisation. https://arxiv.org/abs/2502.17416

## Continuous chain-of-thought (fine-tuned, CoT-supervised)

- **Coconut** (Meta FAIR, ICLR'25) — last hidden state fed back as the next input embedding; staged
  curriculum replacing token steps with k continuous thoughts; wins on ProsQA/ProntoQA-style planning,
  **flat/loses on GSM8K**; audits find shortcut dependence (https://arxiv.org/abs/2512.21711).
  https://arxiv.org/abs/2412.06769 · https://github.com/facebookresearch/coconut
- **CODI** (EMNLP'25) — self-distillation with shared weights aligning the answer-token hidden state;
  first implicit CoT to **match explicit CoT on GSM8K** (GPT-2), 3.1× compression.
  https://arxiv.org/abs/2502.21074 · https://github.com/zhenyi4/codi
- **SIM-CoT** (ICLR'26) — an auxiliary decoder supervises *each* latent against its gold CoT step
  (removed at inference); fixes **latent collapse** when scaling to 8–16 latents; +8.2% on Coconut.
  https://arxiv.org/abs/2509.20317 · https://github.com/InternLM/SIM-CoT
- **CoLaR** (NeurIPS'25) — merge c consecutive CoT embeddings, a latent head predicts the next
  compressed embedding; SFT + GRPO; −53% chain length at −4.8% vs CoT (7B class).
  https://arxiv.org/abs/2505.16552 · https://github.com/xiaomi-research/CoLaR
- **LOTUS** (Jun 2026) — K padded latent blocks, backbone looped R times, **parallel CE on each latent
  vs the gold CoT token**; first latent method at **GSM8K parity with Llama-3.2-3B CoT**, 2.5–6.9× faster
  thinking; latents decode to CoT steps. https://arxiv.org/abs/2606.31779 · https://github.com/yingfan-bot/lotus
- **LRT (ICLR'26)** — frozen LLM + small module mapping hidden states to a 256-vector latent prefix;
  SFT + GRPO; +~7% GSM8K at 4B. https://github.com/MobiusDai/LRT
- **Latent Recurrent Thoughts** (Sep 2026) — proposer + tiny recurrent reasoner with residual updates
  in front of a *frozen* decoder; answer-only training on Countdown/Sudoku/HumanEval/StrategyQA.
  https://arxiv.org/abs/2609.01117
- **Soft Thinking** — training-free probability-weighted "concept tokens"; marginal (+2.5 pass@1),
  shown to be largely single-threaded (https://arxiv.org/abs/2508.03440). https://arxiv.org/abs/2505.15778

## Puzzle specialists and audits

- **HRM / TRM** — 7–27M-param recursive nets with deep supervision and ACT; Sudoku-Extreme 87%,
  ARC-AGI-1 45% (TRM). Audits: the hierarchical module is ≈ useless vs an L-only transformer
  (https://arxiv.org/abs/2510.00355); ACT best at max steps; heavy task-specific refinement
  (https://arcprize.org/blog/hrm-analysis). Grid puzzles only; does not transfer to text LMs.
  https://arxiv.org/abs/2506.21734 · https://arxiv.org/abs/2510.04871
- **Diffusion-of-Thought / LaDiR** — non-AR latent refinement; small scale.
  https://arxiv.org/abs/2402.07754 · https://arxiv.org/abs/2510.04573
- **H-Net** (ICLR'26) — dynamic byte chunking as latent *compression*, not iterative reasoning; byte
  H-Net ≥ BPE transformer at > 1B. https://arxiv.org/abs/2507.07955
- **Pause / thinking tokens** — DIT learns [PAUSE] at low-confidence sites (+4.7% GSM8K,
  https://arxiv.org/abs/2506.03616); R1 "Wait" tokens are textual and suppressible
  (https://arxiv.org/abs/2506.08343).
- **2026 audits** — decodable ≠ causal (https://arxiv.org/abs/2606.12689); Coconut shortcuts
  (https://arxiv.org/abs/2512.21711); explicit CoT often bypassed internally (https://arxiv.org/abs/2602.03994).
- **Memory + latent (2026)** — RiM fixed memory blocks with a two-stage curriculum
  (https://arxiv.org/abs/2605.30343); G-MemLLM gated latent bank on a frozen LLM (https://arxiv.org/abs/2602.00015);
  LatentGraphMem (https://arxiv.org/abs/2601.03417). Long-context and latent reasoning are still
  mostly **orthogonal stacks**.

## Frontier status (Sep 2026)
Only **ByteDance (Ouro)** has released a pretrained looped LM. Meta, Google: research only.
DeepSeek, Qwen, GLM: all reasoning is **text** CoT with `reasoning_effort` scaling tokens.

---

## Synthesis for [REDACTED]

**(a) Ingredients that recur in the winners.** Pretrain *with* recurrence (Ouro, Huginn, PonderLM,
Thoughtbubbles) — post-hoc Coconut/CODI rarely beats CoT at scale; **per-step or parallel supervision
on latents** (SIM-CoT, LOTUS) prevents collapse; curriculum token-CoT → latent; random depth /
adaptive exit during training; test-time depth as a first-class axis (buys math/code, not ppl).

**(b) Recurring failures.** Parity-not-victory on GSM8K/MATH until step supervision at the right
scale; latent collapse without per-step signal; shortcut solutions without OOD robustness;
depth helps while probes find no CoT-like mechanism; puzzle SOTA does not transfer to text.

**(c) Benchmarks with reliable latent signal at ≤ 3B.** ProsQA / ProntoQA / graph planning (yes);
GSM8K (mixed; parity is the realistic target); Sudoku/Maze/ARC (yes, but for specialist nets);
multi-hop QA (often flat or artifact-driven); broad LM ppl (efficiency, not reasoning leaps).
BAPO-hard synthetic tasks (majority, reachability, variable tracking) remain the cleanest probe of
"does recursion add bandwidth" and are absent from every paper above.

**(d) What this means for E19 on the E18/E21 platform.** Two design choices follow directly:
1. **Reasoning steps should refine the *message*, not the token stream.** The frontier's failure
   mode is a latent that nothing supervises densely; in E21 the slots are already load-bearing (the
   receiver's CE depends on them at every position), so a write-back step that rewrites the slots
   gets dense gradient for free. E19's metric is then unambiguous: the receiver's gain (E21 S1) and
   recall-through-message must **increase with the number of refinement steps K**, at fixed
   parameters — the recurrent-depth claim, measured on a channel that cannot be bypassed.
2. **Re-readable tape, per-step signal.** Steps write into the global read's K/V (a tape the model
   re-reads — the BAPO Thm 8 / Pfau resolution), and each step's slots can be scored by the same
   receiver probe (SIM-CoT / LOTUS's lesson without CoT labels). Random K at training, adaptive exit
   later. Ouro's 7.7T budget is not available; the bet is that dense message supervision substitutes
   for scale — a falsifiable claim (flat gain-vs-K kills it).

Open: whether parallel latent paths (Thoughtbubbles / Coconut's superposition) add anything over
sequential refinement on the message; whether BAPO-hard tasks separate the two.
