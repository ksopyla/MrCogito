# E08 — Concept-Flow reasoner (encode→reason→decode via reasoning-trace distillation)

- **Status:** draft
- **Serves:** **Paradigm A** — encode→reason→decode with **decoding-as-thought-crystallization**. The new training-paradigm + architecture direction (replaces the incremental SG1-first path). Attacks the E05 failure (empty concepts, STS-B 0.452) with a strong *reasoning-trace* signal rather than another auxiliary objective.
- **Implementation plan:** [E08_concept_flow_reasoner_plan.md](E08_concept_flow_reasoner_plan.md) *(NOT yet written — authored by `implementation-plan`; it owns the HOW: reasoner module internals, tensor shapes, forward pass, losses, collator, config knobs, tests, risks).*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-07-02 · closed —

> **This is a paradigm-introduction experiment (E01-style new-component), not a clean single-variable A/B.** It adds one new *reusable* foundation module — the **Concept-Flow reasoner** — plus the flow+crystallization objective that requires it. The reasoner, its objective, and the reasoning-trace data are one coordinated paradigm (not three independent knobs), so splitting them would dissolve the hypothesis under test. Attribution for the "does flow add value?" sub-claim comes from a **no-flow AR ablation** run on the identical data (the within-experiment control), not from a one-variable frame the paradigm cannot satisfy.

## Hypothesis
If we insert a **Concept-Flow reasoner** (iterative refinement over the C concept slots) between the E02-long encoder and its AR decoder, and train it so the reasoner **flows question-concepts → the concept-states of existing strong-teacher reasoning traces** (the trace's steps + final answer, encoded by the student's own encoder), while the decoder **crystallizes** the final state into the answer — then the strong reasoning signal will **(a) de-collapse concepts** (within-sample RankMe ≫ 38, STS-B recovering toward 0.714 and far above E05's 0.452) **and (b) yield crystallizable reasoning** (math/reasoning accuracy above a no-flow ablation; non-degenerate generation), **because the flow target cannot be bypassed** the way weak objectives (reconstruction, MLM, plain prefix→suffix) let a strong AR decoder route around the bottleneck.

## Builds-on
- **Foundation:** `nn/concept_encoder_perceiver.py` (BiXT encoder + the concept-conditioned AR decoder — both **unchanged**) + the shared `scripts/train_perceiver_denoise_multigpu.sh` entrypoint and the existing pretokenize→manifest→train spine. The **single new component** is a config-selectable **Concept-Flow reasoner** (`REASONER_TYPE=concept_flow`) placed between encoder and decoder — reusable by every future experiment, never a fork.
- **Init / checkpoint:** **warm-start encoder + AR decoder from E02-long `concept_ar_prefix_H768L6C128D4_20260614_101305/checkpoint-296000`** (STS-B 0.714, the semantic leader). The reasoner is random-init with a **zero-initialised modulator** so that at step 0 the unmodulated backbone reproduces E02-long's behaviour (clean stability property). The exact modulation scheme is an implementation-plan detail.
- **Baseline to beat:**
  - *De-collapse:* E05-attempt3 within-sample RankMe **37.67**, STS-B **0.452**, Δshuffle_beyond **0.39**, free-running token-F1 **0.149** (the failure to fix).
  - *Ceiling (E02-long, the warm-start source):* STS-B **0.714**, within-sample RankMe **82.28** (the PRIMARY de-collapse metric; canonical record in the [E02-long run report](../2_Experiments_Registry/run_reports/e02_long_5epoch_20260618.md)). E08 warm-starts here, so the bar is to **preserve ~82** (zero-init modulator ⇒ step-0 = E02-long) while adding crystallizable reasoning.
  - *Reasoning attribution:* a **no-flow AR ablation** (same data, same warm-start, `REASONER_TYPE=none`) — the flow must beat it on reasoning accuracy or the paradigm is not justified.

> **Prerequisite (cheap, before training):** E02-long's run report records only *cross-sample* RankMe (245.9), not the *within-sample* RankMe that is now the primary de-collapse metric. Re-run `analysis/run_concept_analysis.py` on E02-long `checkpoint-296000` to get its within-sample RankMe, so the de-collapse baseline is on the same metric as E05's 37.67.

## The single change (paradigm-level)
Insert the Concept-Flow reasoner and train with the flow+crystallization objective on **existing reasoning-trace datasets** (not custom teacher generation — see Data). **Held fixed:** encoder/decoder architecture (H768 / L6 / C128 / D4, token_embedding_dim 256, BiXT, RoPE), SmolLM2-135M tokenizer, and the E02-long warm-start.

## Success criteria (set BEFORE running)
- **De-collapse (primary):** within-sample RankMe **> 60** (vs E05 37.67) **AND** STS-B zero-shot **≥ 0.65** (recovering toward E02-long 0.714) **AND** Δshuffle_beyond **≥ 0.5** (E05's unmet Stage-2 target).
- **Crystallization:** free-running token-F1 **> 0.3** (vs E05 0.149) **AND** no repetition loops on held-out prompts.
- **Reasoning:** reasoning-task accuracy (GSM8K / OpenMathReasoning held-out) **> no-flow AR ablation by ≥ 5 pp** — the attribution test.
- **Test-time compute (stretch, SG4):** accuracy at inference K=8 **> train K=4 by ≥ 2 pp**.

## Kill criteria (set BEFORE running)
- STS-B **< 0.45** (flow destroyed semantics) **OR** within-sample RankMe **< 20** (collapsed worse than E05) → stop; the paradigm fails its core claim.
- **Divergence:** eval_loss rising over 3 consecutive eval points (the E05-attempt2 signature) → stop.
- Reasoning accuracy **≤ no-flow ablation** → flow adds nothing; revert the run to the ablation arm as the reported result.

## Plan
- **Data:** a new **reasoning-trace-heavy mix** `data/mix_recipes/concept_flow_reasoning_2k.json` over the pretokenize→manifest→train spine — **~50% reasoning traces / ~50% fluency replay** (a ~5× boost in reasoning share vs `smollm3_inspired_2k_e05`):
  | Source | Weight | Role |
  |---|---|---|
  | `nvidia/OpenMathReasoning` (cot, problem+generated_solution) | 30% | primary trace source — clean problem/solution columns let the collator encode question and trace separately → the flow's concept-space targets |
  | `allenai/big-reasoning-traces` (DeepSeek CoT) | 20% | diverse long CoT; also stresses cross-window concept memory |
  | `HuggingFaceFW/fineweb-edu` | 25% | web fluency / STS-B retention |
  | `mlfoundations/dclm-baseline-1.0` | 15% | broad web long-context |
  | stack-edu (python) | 10% | code fluency |
  **These datasets ARE strong-teacher (DeepSeek-class) CoT traces — the distillation signal — so no on-the-fly teacher generation is needed for E08.** At seq 2048, traces exceeding ~2K (problem+solution) are filtered so the full answer stays visible (a 4K long-trace variant is a CF-1 option, noted in the recipe's `variants`). A reasoning-heavier 60/40 variant is banked in the recipe if the 50/50 keystone holds STS-B ≥ 0.65 and we want a stronger signal.
- **Teacher model (banked, NOT the E08 data path):** a local **Qwen3-30B-A3B-Thinking-2507** (AWQ-4bit, vLLM, fits one 3090, ~21 GB at 32K) is the validated option for *future* custom trace generation, a held-out eval set, or the later RL stage — kept as a Phase-4/5 asset, out of scope for E08.
- **Compute:** Odra 2–3× 3090; est. **~30–50 GPU-h** (warm-started fine-tune — far cheaper than E02-long's 72 h from scratch).
- **Steps / epochs:** ~3–5 epochs on the reasoning mix (warm-start).
- **Launch:** config over the shared entrypoint (`REASONER_TYPE=concept_flow`, the reasoning mix, warm-start from E02-long); the exact command + the no-flow ablation command are specified in `_plan.md` once the module exists.
- **New foundation code (reusable, config-selectable — via `research-implement`):**
  - `nn/concept_flow_reasoner.py` — the Concept-Flow reasoner (a weight-tied iterative refinement block + its flow-matching loss head).
  - A `reasoning_distillation` tokenize mode / collator on the existing spine (encodes question and trace separately so the trace becomes the flow's concept-space targets).
  - **No teacher-generation script for E08** (data is existing datasets).

## Result
<Filled in AFTER, by experiment-track. Link out; do not paste full results here.>
- Run id: `<run_id>`
- WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
