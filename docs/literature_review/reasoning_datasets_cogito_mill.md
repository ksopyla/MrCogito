# Literature review — reasoning datasets & LLM reasoning eval (cogito-mill)

**Date:** 2026-07-23  
**Question:** Is there still a need for [cogito-mill](https://github.com/meridian21lab/cogito-mill) / Long Story Short given the 2024–2026 flood of reasoning benchmarks? Should MrCogito abandon this work?  
**Serves:** train+eval signal for concept-bottleneck long-context models (windowed-AR, Gemma concept memory, Concept-Flow, etc.) — see [deductive_stories_synthetic_dataset.md](../engineering_specs/deductive_stories_synthetic_dataset.md) and cogito-mill [`docs/vision.md`](https://github.com/meridian21lab/cogito-mill/blob/main/docs/vision.md).  
**Scout:** [Scout reasoning datasets](d75cfadc-8359-468f-a80f-aa0342af99b8) + primary arXiv/ACL/HF checks.

> **Verdict up front:** The *generic* “another synthetic reasoning dataset” space is overcrowded. The *specific* cogito-mill contract (solver/world-model first → long narrative → evidence-grounded multi-query gold + structured traces + counterfactuals/minimal falsifiers, at 4K–32K, with train↔eval firewall) is **still largely unfilled**. Do **not** abandon the niche; do **abandon vanity scale** and any path that cannot ship inspectable, solver-verified items. Prefer a **small hard eval pack first**, then train mix weight — not a second MuSR with prettier agents.

---

## 0. What cogito-mill claims (reference frame)

| Axis | Cogito-mill / Long Story Short |
|------|--------------------------------|
| World authority | Typed formal world **before** prose; Z3 + causal sim certify consistency / uniqueness / interventions |
| Surface | Long **narrative** (pilot ~1.5–6K tok; target distribution ~4K–32K+, weighted long) |
| Gold | Exact-match answers + ordered typed deduction steps (sentence IDs, inference type) + counterfactual + minimal falsifier |
| Anti-shortcut | Visible-theory uniqueness; later: same-logic/diff-surface and same-surface/diff-logic variants; parametric-leak + distractor gates |
| Use | Both **train** (mix ingredient) and **frozen eval** for MrCogito concept ablations |

In-repo shallow prototype was killed (2026-07-19) for narrative/hardness quality; work lives in the separate mill.

---

## 1. Comparison matrix (blunt)

| Work | Verified gold? | Long narrative? | Train-scale? | Multi-query / process? | Closest overlap with mill | Differentiator mill still owns |
|------|----------------|-----------------|--------------|------------------------|---------------------------|--------------------------------|
| **LongProc** | Rule-based on structured **output** | Long **generation**, not story world | Eval | Procedure follow | Long-horizon coherence | Narrative deduction + latent world |
| **MCST** | Simulation / HEIL | Short action text | Large Q count | State QA | Negation / multi-constraint state | 4K–32K prose + proof traces |
| **LSP probes** | Probabilistic/logical probes | No (dialogs) | No | Multi-turn | Motivates latent state | Does not supply data |
| **ActionReasoningBench** | Planner/ASP | Short paraphrased PDDL | Train exists | RAC dims + ramifications | Formal→NL | Detective-length narrative + distractors |
| **CC-RAG** | No (GPT-4o triples) | Corpus chunks | Method, not corpus | Causal chains | Causal provenance *idea* | Solver-first synthetic worlds |
| **CausalDR / CDCR-SFT** | Teacher+checks on CLADDER | Short QA; **graph explicit** | ~25k | DAG + trace | Structured causal gold | Hide logic in prose; long context |
| **ZebraLogic** | Z3 CSP | Puzzle clues (short) | Eval (~1k) | Grid fill | Solver-verified logic | Narrative dispersion + evidence IDs |
| **Cross-Query Contradiction** | Z3/CaDiCaL | Short premises | Small (390 cases) | Bundle consistency | Multi-query SAT + depends_on | Long story + event-graph traces |
| **MuSR** | Soft / symbolic tree | ~1k-word mysteries | Tiny (756) | Soft multi-step | **Nearest neighbor** | Hardness+solver+scale+length |
| **DetectiveQA** | Human | Novel-length (~100k) | Eval-only | Step refs | Long detective QA | Synthetic, trainable, machine-checkable |

**Reading the matrix:** Crowding is real on *short* diagnostics and *puzzle* CSPs. The empty cell intersection is **long prose × solver-first gold × train+eval × concept-ablation-ready**.

---

## 2. Per-source notes (tracked findings)

### A. Procedural state tracking

#### A1. LongProc — Benchmarking Long-Context LMs on Long Procedural Generation

- **Cite:** Ye, Yin, He, Zhang, Yen, Gao, Durrett, Chen · **COLM 2025** · [arXiv:2501.05414](https://arxiv.org/abs/2501.05414) · [OpenReview](https://openreview.net/forum?id=ruWC5LIMSo) · [GitHub princeton-pli/LongProc](https://github.com/princeton-pli/LongProc) · [Project](https://princeton-pli.github.io/LongProc/) · [HF PrincetonPLI/LongProc](https://huggingface.co/datasets/PrincetonPLI/LongProc)
- **Thesis:** Long-context eval should require **integrating dispersed info and producing long structured outputs** (≤8K tokens), not only short-answer needle recall.
- **Construction:** Six deterministic procedural tasks (HTML→TSV, pseudocode↔code, path traversal, ToM object/belief tracking, Countdown, travel planning); difficulty by output length (0.5K / 2K / 8K); rule-based scoring.
- **Evidence:** 23 LCLMs; open models often fail at 2K output; GPT-4o degrades at 8K; reasoning models benefit from long-CoT training; long-range coherence fails.
- **Limitations:** Hardness leans on **output length and procedure**, not hidden narrative worlds; ToM tracking is closest to “state” but still a fixed procedure.
- **vs mill:** Tests **long generation under a recipe**, not **compressing a long story into a checkable latent world**. Useful as a *neighbor diagnostic* for coherence, useless as a substitute train mix for concept deduction.
- **Verdict:** **Watch** (eval tool) / **Reject** as mill replacement.

#### A2. MCST — Multi-Constraint State Tracking with Negation

- **Cite:** Sar, Singh Puri, Aich, Kaushish, Choudhury, Abraham · **ACL 2026 SRW** · [Anthology 2026.acl-srw.119](https://aclanthology.org/2026.acl-srw.119/) · [PDF](https://aclanthology.org/2026.acl-srw.119.pdf)
- **Thesis:** LLMs fail to maintain evolving world models under **interacting constraints + negated actions**.
- **Construction:** HEIL pipeline (expert ontologies → stochastic simulation → NL realization); **100,847** questions, 12 domains, 5 difficulty levels, 9 question types; culturally diverse names.
- **Evidence:** 14 SOTAs; accuracy &lt;35% at top difficulty; negation hurts **−23 to −32%**.
- **Limitations:** Diagnostic / SRW grade; short sequences; gold is final-state answers, not long evidence-localized proofs; GitHub promised, license/URL murky from paper alone.
- **vs mill:** Steal the **negation stress** and multi-constraint difficulty ladder. Do not confuse short state-tracking with long deduction.
- **Verdict:** **Adapt** (negation + calibrated difficulty) / not a substitute.

#### A3. On the Failure of Latent State Persistence in LLMs

- **Cite:** Huang, Sun, Wang, Dredze · preprint · [arXiv:2505.10571](https://arxiv.org/abs/2505.10571) · [GitHub penguinnnnn/LLM-Working-Memory](https://github.com/penguinnnnn/LLM-Working-Memory)
- **Thesis:** LLMs are **reactive post-hoc solvers**, not agents with persistent unwritten latent state (LSP).
- **Method:** Three probes — Number Guessing (probability-mass identity), Yes-No Game (concept drift → contradiction), Mathematical Mentalism (hidden variable binding / evolution).
- **Relevance:** Direct philosophical fuel for **concept bottlenecks** and “hidden world never written in context” anti-shortcuts. **Not a dataset.**
- **Limitations:** Tiny interactive probes; scaling/CoT externalizes computation without proving LSP.
- **vs mill:** Mill should *operationalize* LSP over **long disclosed evidence → compressed concepts**, with ablations (shuffle / zero concepts). LSP probes do not replace that.
- **Verdict:** **Adopt** as motivation / eval *idea*; **Reject** as data source.

#### A4. ActionReasoningBench

- **Cite:** Handa, Dolin, Kumbhar, Son, Baral · **ICLR 2025** · [arXiv:2406.04046](https://arxiv.org/abs/2406.04046) · [OpenReview NUD03NBDOE](https://openreview.net/forum?id=NUD03NBDOE) · [GitHub izuminka/reasoning_about_actions](https://github.com/izuminka/reasoning_about_actions)
- **Thesis:** Diagnostic for **Reasoning about Actions & Change**, including **ramification** (indirect effects).
- **Construction:** 8 PDDL domains → planner/ASP fluents → templated then LLM-paraphrased NL; up to 19 actions; six RAC dimensions; **149,237 train / 3,498 test** ([GDrive](https://drive.google.com/drive/folders/1v8yhRmd2IhLLNpiJhoh4fyiaKEcaI9_B)).
- **Evidence:** Decent on classical fluent/state tracking; sharp drop on numerical/composite; ramifications: GPT-4o **0%**, o1-preview **18.4%**.
- **Limitations:** Planning-domain English, not novels; short horizons; paraphrase surface can leak template cues.
- **vs mill:** Same *spirit* (formal world → language). Different *surface* and length. Ramification stress is worth importing into causal sim questions.
- **Verdict:** **Adapt** (ramifications) / **Reject** as long-context substitute.

---

### B. Provenance / causal DAGs

#### B1. CC-RAG — Structured Multi-Hop Reasoning via Theme-Based Causal Graphs

- **Cite:** Parekh, Jiang, Han (et al.) · ACL ARR 2025 May · [arXiv:2506.08364](https://arxiv.org/abs/2506.08364) · [OpenReview daSiBuVRHH](https://openreview.net/forum?id=daSiBuVRHH)
- **Thesis:** Flat RAG misses why/how multi-hop; build a **causal DAG of ⟨cause, relation, effect⟩** and chain forward/backward.
- **Construction:** Zero-shot GPT-4o triple extraction over domain corpora (Bitcoin, Gaucher) → DAG → retrieval-guided generation. **Method paper**, not a public train corpus.
- **Evidence:** Beats flat RAG on chain similarity / LLM-judge / human preference on two domains.
- **Limitations:** Extraction quality = teacher; **no solver-verified world**; small specialized eval; not synthetic long stories.
- **vs mill:** Shares “causal provenance paths” rhetoric. Opposite epistemology: mill’s graph is **ground truth**; CC-RAG’s graph is **extracted belief**.
- **Verdict:** **Watch** (RAG systems) / **Reject** for MrCogito data.

#### B2. CausalDR / CDCR-SFT

- **Cite:** Li, Shen, Nian, et al. · **AAAI 2026** (arXiv lists AAAI; user note “2025” is off) · [arXiv:2508.12495](https://arxiv.org/abs/2508.12495) · [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40454) · [GitHub MrLYG/CDCR-SFT](https://github.com/MrLYG/CDCR-SFT)
- **Thesis:** SFT models to **construct a variable-level causal DAG then reason over it** reduces logical hallucinations vs token-level CoT/GoT.
- **Construction:** CausalDR = **25,368** samples from CLADDER × permutation aug; DeepSeek-R1 emits JSON DAG + path + answer; graph/story-id splits to limit leakage.
- **Evidence:** Claims **95.33%** on CLADDER (above human 94.8%); ~10% HaluEval improvement.
- **Limitations:** Anchored to CLADDER templates; **graph is shown, not hidden in narrative**; short context; teacher-generated traces ≠ Z3 proof; permutation aug may not add real hardness.
- **vs mill:** Closest on **structured gold + DAG discipline**. Mill must keep the DAG **latent** and force recovery from prose — that is the concept-encoder stress test CausalDR avoids.
- **Verdict:** **Adapt** (DAG-shaped supervision targets for optional process heads) / **Reject** as narrative long-context corpus.

---

### C. Constraint satisfaction (solver-verified)

#### C1. ZebraLogic

- **Cite:** Lin et al. (AI2 et al.) · **ICML 2025** · [arXiv:2502.01100](https://arxiv.org/abs/2502.01100) · [PMLR](https://proceedings.mlr.press/v267/lin25i.html) · [HF WildEval/ZebraLogic](https://huggingface.co/datasets/WildEval/ZebraLogic) (also allenai mirrors)
- **Thesis:** Logic-grid CSPs with **controllable complexity** expose a **curse of complexity** that scale and test-time compute barely fix.
- **Construction:** ~1,000 programmatic puzzles; metrics = search-space size + Z3 conflict count; designed against parametric knowledge / leakage.
- **Evidence:** Accuracy collapses with complexity; Best-of-N / backtracking / self-verify help only modestly.
- **Limitations:** Narrow CSP family; hardness partly “search computation,” not evidence scattering; short inputs.
- **vs mill:** Gold standard for **solver-verified logic**. Mill should **steal complexity knobs and leakage discipline**, then bury constraints in narrative with sentence-level provenance — ZebraLogic deliberately does not.
- **Verdict:** **Adopt** design lessons (complexity metrics, unique-solution prune) / **Reject** as substitute.

#### C2. Cross-Query Contradiction Benchmark

- **Cite:** Salla, Amancherla, Saravanan · **ICLR 2026 Workshop on Logical Reasoning of LLMs** · [arXiv:2604.14525](https://arxiv.org/abs/2604.14525) · [OpenReview](https://openreview.net/forum?id=v9jBDyc72l) · [HF rohitspider/cross_query_benchmark](https://huggingface.co/datasets/rohitspider/cross_query_benchmark) · **MIT**
- **Thesis:** Per-query accuracy can hide **jointly unsatisfiable** belief states across interdependent queries on one case file.
- **Construction:** 390 cases / 2,515 queries; domains Relational/SAT, Temporal/SMT, Policy/Rules, Underspecified/Abductive; every label **Z3/CaDiCaL-verified**; SMT-LIB/CNF + `depends_on`; case-level stratified splits.
- **Evidence:** SetCons 0.56→0.94 with Check+Repair at 1.55× overhead vs self-consistency K=20 at 2.75×.
- **Limitations:** Workshop scale; short premises; solver defines hardness; underspecified “investigation” is not novel-length.
- **vs mill:** Directly steal **bundle-level satisfiability metrics**, `depends_on` provenance, and ENTAILED/CONTRADICTED/UNKNOWN vocabulary (already aligned with mill’s logical statuses). Apply them **on top of long stories**, not instead of them.
- **Verdict:** **Adopt** (eval metrics + multi-query contract) / incomplete alone.

---

### D. Narrative priors already cited by the mill (gap positioning)

| Work | Cite | One-line | Why not enough for MrCogito |
|------|------|----------|------------------------------|
| **MuSR** | [arXiv:2310.16049](https://arxiv.org/abs/2310.16049), ICLR 2024 | Neurosymbolic tree → LLM mystery (~1k words, 756 ex.) | Tiny; soft verification; length too short for concept-memory stress |
| **DetectiveQA** | [arXiv:2409.02465](https://arxiv.org/abs/2409.02465) | Human QAs on real novels (~100k tok) | Eval-only; copyright; no machine world model; cannot train at scale |
| **BRAINTEASER** | [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.885/) | Lateral puzzles + reconstructions | Commonsense/lateral, not deductive; contamination risk |
| **Shortcut Suite** | [arXiv:2410.13343](https://arxiv.org/abs/2410.13343) | Shortcut stress on existing NLI/QA | Diagnostic, not generative world→story |
| **Premise Order Matters** | [ICML 2024](https://proceedings.mlr.press/v235/chen24i.html) | Premise reorder / R-GSM | Presentation sensitivity; motivates variants, not a corpus |
| **PRM800K** | [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) | Human step labels on MATH | Process supervision for math CoT, not event-graph narrative |

**MuSR is the honest nearest neighbor.** If mill ships “MuSR but agentic” without **harder verification, longer context, and train scale**, it is a waste. If mill ships **ZebraLogic-grade verification inside DetectiveQA-length (or 8–16K) synthetic prose**, it is not redundant.

---

## 3. Cross-cutting lessons (what to steal / refuse)

### Steal

1. **Solver-first unique solution** (ZebraLogic, Cross-Query) — never let an LLM certify gold.
2. **Set-level consistency metrics** (Cross-Query) — multi-query bundles must be jointly satisfiable; score SetCons / contradiction density, not only EM.
3. **Negation + multi-constraint ladders** (MCST) — calibrate difficulty beyond hop count.
4. **Ramification / indirect effects** (ActionReasoningBench) — causal sim questions should include non-local effects.
5. **Leakage-aware splits** (CausalDR graph/story ids; Cross-Query case stratification; ZebraLogic anti-leak design).
6. **Process targets from solver deps** (mill already) — PRM800K shows step labels matter; free-form CoT does not.
7. **LSP / shortcut motivations** (Huang et al.; Shortcut Suite; Premise Order) — justify concept ablations and controlled variants.

### Refuse

1. **Extracted causal graphs as ground truth** (CC-RAG) — teacher hallucination becomes “structure.”
2. **Graph-explicit SFT as the main long-context story** (CausalDR) — trains the wrong skill for concept compression.
3. **Hardness = output length alone** (LongProc) — does not force latent world binding.
4. **Puzzle-only CSP as the only logic gate** (ZebraLogic alone) — misses narrative evidence localization that concept models claim to solve.
5. **Publishing another soft mystery set** (MuSR clone) — the field does not need it.

---

## 4. Do we need another dataset? Should we abandon?

### Honest overcrowding call

**Yes, overcrowded** for: short state tracking, RAC diagnostics, logic puzzles, causal QA with visible DAGs, and “LLM-written mysteries.” Leaderboard tourism here is a trap.

**No, not overcrowded** for the intersection MrCogito actually needs:

1. **Long-horizon narrative** (8K–16K+), not puzzle stubs  
2. **Solver/world-model is source of truth** before any prose  
3. **Scattered evidence + distractors** so parametric shortcuts fail  
4. **Multi-query gold** including counterfactual + minimal falsifier + (ideally) set consistency  
5. **Train-scale** with frozen, published eval and template holdout  
6. **Architecture-facing eval** — concept shuffle / zero / no-context leak gates on the *same* items  

No row in §1 hits all six. That is the only justification that survives blunt review.

### Abandon if…

- Generation stays “fact dump” or soft MuSR (already happened once in-repo).  
- Cost/latency of agentic mill cannot produce **≥100 accepted hard eval items** with Luna (or peer) ≤30% EM under blind protocol.  
- You only want a paper, not a **MrCogito falsification instrument**.  
- You are willing to redefine success as “consume ZebraLogic + MCST + DetectiveQA” and **drop** the claim that concepts must bind long narrative evidence.

### Do not abandon if…

- You keep the mill **narrow**: ship **frozen hard eval first** (hundreds, not tens of thousands), wire MrCogito concept ablations, then decide whether train mix is worth the generation bill.  
- You **import** Cross-Query metrics, ZebraLogic complexity knobs, MCST negation, ActionReasoningBench ramifications — instead of reinventing diagnostics that already exist.  
- You treat existing packs as **complementary gates** (puzzle CSP, short state tracking) while mill owns the **long narrative compression** gate.

### Recommendation for MrCogito (actionable)

| Priority | Action | Rationale |
|----------|--------|-----------|
| P0 | Continue cogito-mill **only** toward a **small, frozen, solver-verified eval pack** + concept-ablation harness | Unfilled niche; blocks architecture claims |
| P0 | Kill any batch that fails parametric-leak / distractor / solver-roundtrip (already in eng spec) | Avoid MuSR-2 soft trash |
| P1 | **Adopt** Cross-Query set metrics + ZebraLogic unique-solution/complexity reporting into mill validation | Free rigor |
| P1 | Use **ZebraLogic / MCST / ActionReasoningBench** as cheap *orthogonal* evals — do not rebuild them | Crowded space; consume |
| P2 | Train mix (≥5%) only after eval hardness is real | Train without a hard gate wastes GPU |
| Kill | “Full novel-length public leaderboard” and agentic complexity theater before P0 ships | Vanity |

**Bottom line:** Do **not** abandon the work. Do **abandon the fantasy that the world lacks reasoning datasets**. Build the **one missing instrument**: solver-verified long narrative deduction for concept models — or stop and admit MrCogito will only be tested on puzzles and short state tracking, which do **not** exercise the claimed architecture.

---

## 5. Source index (URLs)

| ID | URL |
|----|-----|
| LongProc | https://arxiv.org/abs/2501.05414 · https://github.com/princeton-pli/LongProc |
| MCST | https://aclanthology.org/2026.acl-srw.119/ |
| LSP | https://arxiv.org/abs/2505.10571 · https://github.com/penguinnnnn/LLM-Working-Memory |
| ActionReasoningBench | https://arxiv.org/abs/2406.04046 · https://openreview.net/forum?id=NUD03NBDOE |
| CC-RAG | https://arxiv.org/abs/2506.08364 |
| CausalDR / CDCR-SFT | https://arxiv.org/abs/2508.12495 · https://github.com/MrLYG/CDCR-SFT |
| ZebraLogic | https://arxiv.org/abs/2502.01100 · https://huggingface.co/datasets/WildEval/ZebraLogic |
| Cross-Query | https://arxiv.org/abs/2604.14525 · https://huggingface.co/datasets/rohitspider/cross_query_benchmark |
| MuSR | https://arxiv.org/abs/2310.16049 |
| DetectiveQA | https://arxiv.org/abs/2409.02465 |
| cogito-mill | https://github.com/meridian21lab/cogito-mill |
| Eng spec | `docs/engineering_specs/deductive_stories_synthetic_dataset.md` |

---

## 6. Changelog of this note

| Date | Change |
|------|--------|
| 2026-07-23 | Initial review from user brief + [Scout reasoning datasets](d75cfadc-8359-468f-a80f-aa0342af99b8); venue corrections (CausalDR→AAAI 2026; ActionReasoningBench→ICLR 2025 / arXiv 2406.04046); blunt abandon/continue recommendation; folded scout artifact URLs (LongProc project, ActionReasoningBench sizes/GDrive, CausalDR AAAI link, Cross-Query OpenReview). |
