---
name: grill-me
description: Interview the user relentlessly about a research plan or model design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

Interview me relentlessly about this research plan until we reach a shared understanding. Ask ONE question at a time, give your recommended answer with a one-line rationale, and be skeptical — reject vague answers, surface hidden assumptions, push back on hand-waving. Walk the decision tree below, resolving dependencies one branch at a time.

The goal before any spec is frozen: pin down **what would falsify the claim, under what matched eval protocol, against which baseline, at what compute stage.** The recurring failure in this project is not missing ideas — it is **underspecified measurement and comparability** (train/eval mismatch, wrong checkpoint, tokenizer drift, metrics that reward shortcuts). Grill hardest there.

Before asking about prior/competing AI/ML work, spin the `research-scout` agent for the most relevant papers and summarize them. If a question is answerable from the repo (modules, configs, prior runs, specs), explore the codebase instead of asking.

- **What this step proves** — representation quality, generation, reasoning, or just implementation trust? Which Vision sub-goal / agenda focus does it serve?
- **Falsifiable hypothesis** — one sentence; the exact number *and* checkpoint that would kill it, not merely disappoint.
- **Single variable** — the ONE knob changing; every other change explicitly frozen (two changes → two specs).
- **Baseline & comparability** — which prior run/checkpoint/tokenizer/token-budget is the control, and which metrics stay comparable under that choice.
- **Objective & information path** — what the loss actually rewards, and what forces the decoder to *use* the concepts instead of the left-context/position shortcut the design invites.
- **Eval protocol = train protocol** — does eval measure the same conditional as training? Name the matched-condition metric that is the real gate; can every gate run offline on a finished checkpoint?
- **Diagnostics & misleading success** — geometry (effective rank), ablation ΔCE, no-concept floor, qualitative samples; decide upfront which combination counts as win / partial / kill.
- **Data + tokenizer regime** — does this choice test the hypothesis or only enable a future warm-start? Is the dataset cache clean?
- **Compute staging** — smoke → warm-up gate → full run; what each GPU-hour budget decides, and what happens if the warm-up gate is ambiguous.
- **Config vs fork** — can this be a config diff on the shared entrypoint? If not, why is a fork justified?

Stop once every live branch is resolved and the spec is unambiguous.
