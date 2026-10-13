# Small-size pretraining on AWS — preparation plan (draft 2026-10-13)

Living plan for the first real-text pretraining of our memory architecture at ~500M parameters on
a few AWS GPUs, compared on real evaluations. Built top-down: groups first, then each group is drilled
down and its decisions are recorded here. Nothing is launched or bought until the author says go.

Supersedes nothing yet; reuses parts of the blocked E18 AWS plan
(`docs/experiments_specs/ahead/E18_main_run_aws_plan.md`: launch sequence, decontamination, risks).

Status keys: **exists** (in the repo, tested), **partial**, **missing**, **decide** (an open choice).

## 0. Goal, arms and the go gate

What question the run answers, which models are trained, for how many tokens, and what must be true
before money is spent.

**Decided (author, 2026-10-13):** the memory model is **E33** (`e31c_loop`: E31c latent notebook +
read–think–reread loop, 4 rounds).

Question the run answers: *at ~500M parameters on real text, does E33's notebook carry information from
beyond its local window (lower loss on far positions, retrieval from long documents) without costing
standard short-text ability, compared with same-size models trained on the same tokens?*

**Decided (author, 2026-10-13):**
- **Only E33 is trained.** No trained controls in this round.
- **Staged tokens:** 5B → evaluate and review → 10B → evaluate and review → the author decides whether
  to scale. With a WSD schedule (constant LR, then a short decay), each review point is a decayed
  *branch*; the main run continues from the pre-decay checkpoint with its optimizer state, so no
  tokens are wasted on restarts.
- **Batch size and learning rate are tuned before the main launch, on AWS at the target size (~500M)**
  in the dedicated environment, not on small proxies (author, 2026-10-13). Proposed: a grid of ~2 batch
  sizes × 3–4 learning rates, ~0.5B tokens each, each ending with a short decay (rankings can flip during
  the decay); a 500M model fits one H100, so the ~8 runs go one per GPU in parallel on one node
  (~4–6 h wall, ~4–6 node-h ≈ $250–350). The sweep doubles as the H100 speed test.
- **Go:** no automatic gate. Everything is prepared now; the author decides on launch from the state of
  the text capability board once its running tests finish.

Consequences for the evidence (no trained controls):
- The notebook's value is measured **inside E33**: the same checkpoint scored with the notebook on vs off
  (`message_override none`) on far positions of held-out long books, and on retrieval. A model never
  trained without the notebook may lean on it differently, so this is an ablation, not a twin.
- Standard ability is compared with **public models at matched tokens**: Pythia-410M intermediate
  checkpoints (Pile, scored at 5B / 10B tokens with our harness), and published matched-token rows
  (Titans' 400M Transformer++ at 15B, Samba-421M at 20B, from their papers). SmolLM2-360M and
  Qwen3-0.6B (4T / 36T tokens) are shown as a ceiling only. Different data → a loose comparison.
- A trained dense or no-memory control can be added later if E33 looks promising (same pipeline).

Proposed review criteria at each stage (to confirm):
- Training health: loss falls smoothly, no spikes or divergence; throughput as measured.
- Notebook used: switching it off raises E33's loss on positions beyond its window (held-out PG-19,
  decontaminated); the gap grows from 5B to 10B. Kill signal: a gap below ~0.005 nats at 5B.
- Retrieval: needle / RULER-lite in long documents above the notebook-off score.
- Standard ability: short-benchmark scores at or above Pythia-410M at the same token count.

Rough compute (estimate, replaced by the H100 speed test): a 500M dense model runs ~1M tokens/s per
8×H100 node in the ideal case; assume ~60 % of that for our kernels and 2–3× per token for the loop →
~0.2–0.3M tokens/s for E33 ≈ 0.9–1.4 node-h per 1B tokens → 5B ≈ 5–7 node-h (~$250–400 on-demand
at $55/node-h), 10B ≈ double. Evals and the decay branches add perhaps 20–30 %.

Inputs pending for the model group: the symbolic "E33 without exit losses" control (2026-10-14), which
decides whether the loop's per-round exit losses are needed; on web text they would apply to every
next token (more compute) rather than to answers only.

## 1. AWS infrastructure (Terraform)

**Decided (author, 2026-10-13):** on-demand p5.48xlarge (8×H100 80 GB, ~$55/h, per-second billing);
login through SSM Session Manager (no open ports); Terraform state in an S3 backend. The account exists
but has no GPU quota yet.

**Author action now (lead time ~1–3 days, may need a justification):** Service Quotas → Amazon EC2 →
"Running On-Demand P instances" → request **192 vCPUs** (one p5.48xlarge) in the chosen region.
Proposed region: us-east-1 (largest p5 pool, lowest price); confirm p5 is offered in the chosen
availability zone before the request.

Proposed layout (`infra/aws/`, written and tested by the agent; every `apply` that costs money is run
by the author):
- `bootstrap/`: the state bucket (versioned, encrypted, S3-native lock file, Terraform ≥ 1.10).
- `main/`:
  - network: a small VPC, one public subnet in the AZ with p5 capacity (variable), no inbound rules;
  - S3 bucket `mrcogito-train-<suffix>`: `data/` (pretokenized shards), `runs/` (checkpoints,
    logs), `results/` (evals);
  - IAM instance role: that bucket only, the two secrets, `AmazonSSMManagedInstanceCore`;
  - Secrets Manager: `HF_TOKEN`, `WANDB_API_KEY` (values set by the author outside Terraform);
  - launch template: AWS Deep Learning AMI (Ubuntu, CUDA 12.8+), root EBS ~200 GB, local NVMe for
    data and checkpoints, user-data bootstrap (clone repo at a tag, `uv sync`, pull secrets and data,
    start a byobu session), instance-initiated shutdown = terminate;
  - safety nets: AWS Budgets alarm (monthly cap, email), a hard maximum lifetime per instance, a
    CloudWatch alarm that stops an instance whose GPUs are idle for 30 min.
- Access: `aws ssm start-session`, plus an `ssh` ProxyCommand through SSM so `ssh aws-train` and
  byobu work as on Odra (needs the Session Manager plugin locally).
- Checkpoints sync to S3 every 30 min; results are copied back to Odra/NAS for the ledger.

## 2. Training data

- Tokenizer: SmolLM3 128k vocab (current) vs a ~49k vocab (smaller output layer). **decide**
- Mix: web (FineWeb-Edu / DCLM), code, math, plus a long-document tier (PG-19, FinePDFs) so the
  memory has something to do. Recipes exist (`data/mix_recipes/`); the long-document share is **decide**.
- Pretokenize on Odra/Polonez and upload to S3 vs on a cheap AWS CPU instance. Pipeline **exists**
  (`scripts/pretokenize_mix.py`, manifest stats).
- Decontamination of PG-19 and benchmark test sets. **missing**

## 3. Model changes

- ~500M configs for the three arms (memory writer sizes scaled, parameter-matched). **missing**
- Tied input/output embeddings (the untied 128k head is ~164M params). **decide**
- Kernels on H100: FlexAttention for the memory read, `torch.compile` graph breaks (host syncs). **partial**
- Memory-off switch usable during training evals (exists for eval only). **partial**
- KV cache / incremental memory writes (needed only for generation evals). **missing**, **decide** if needed

## 4. Training recipe and plumbing

- Optimizer and LR: Muon / AdamW. **partial**
- Batch size × learning rate sweep before launch (decided 2026-10-13): at 500M on AWS, one run per GPU
  (see group 0). The text checks already have a per-model
  tuning step (`tune_runs`, `tune_tokens` in `evaluation/text_checks.py`) to reuse. **partial**
- Schedule: WSD (constant then decay, branchable checkpoints). **missing**
- Sequence length, batch size, tokens per step; packed documents. Packing **exists**.
- Checkpointing: save cadence, retention, resume tested, S3 sync; safe defaults for long runs. **partial**

## 5. Evaluation

- No-harm short suite (HellaSwag, ARC-e, PIQA, LAMBADA, SciQ, …) via lm-eval-harness. **exists**
  (`evaluation/lm_eval_perceiver_ar.py`). Note: short items never close a memory window, so these
  measure the backbone only.
- Memory evidence: PG-19 per-position loss with memory on vs off; needle / RULER-lite; BABILong;
  our text exams. **partial**
- Reference rows: Pythia-410M intermediate checkpoints at matched tokens; published matched-token
  baselines (Titans 400M @15B, Samba-421M @20B); SmolLM2-360M / Qwen3-0.6B as a ceiling only. **partial**
- Where evals run: during training (callbacks) vs after, on Odra from S3 checkpoints. **decide**

## 6. Observability and cost tracking

- W&B: loss, grad norm, LR (exists); tokens seen, throughput, MFU (**missing**); memory health
  (writer entropy, slot usage, memory-off loss gap) (**missing** in the trainer).
- Cost: GPU-hours and $ per run, logged and capped. **missing**

## 7. Rehearsal, runbook and decision points

- End-to-end dress rehearsal on Odra/Polonez at ~125M on the same pipeline (doubles as the go gate).
- One-hour H100 throughput test of each arm before the full run.
- Launch runbook, in-run decision points (when to stop), risks.
