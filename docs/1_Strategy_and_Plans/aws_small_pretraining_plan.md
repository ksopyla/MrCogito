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

**Author action (deferred 2026-10-13: no AWS login yet; lead time ~1–3 days once filed):** Service Quotas → Amazon EC2 →
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

**Decided (author, 2026-10-13):**
- Tokenizer: **SmolLM2** (49,152 vocabulary), **tied input/output embeddings** (~63M of ~500M at width
  1280). Data is re-tokenized (CPU only); token ids fit in 2 bytes.
- **Long documents ~35 %** of training tokens (books and long PDFs).
- **Sequence length 16k**, documents packed into rows with document masks; books and long PDFs chunked
  into consecutive 16k rows. Extrapolation read at eval time to 128k.
- **Curriculum:** phase 1 (to the 5B review) natural text only; phase 2 (5B → 10B) adds synthetic text
  that forces far-back recall (Text World-style documents with questions about early facts; design in
  this group's drill-down). Side benefit: the notebook-off gap at 5B shows what natural text alone
  teaches; the change by 10B shows what the synthetic push adds (confounded with 5B more tokens).

Proposed phase-1 mix (to confirm; weights are token shares, checked with `scripts/manifest_token_stats.py`):

| source | share | role |
|---|---|---|
| FineWeb-Edu | 30 % | general knowledge (best early benchmark signal) |
| DCLM-baseline | 15 % | broader web |
| Stack-Edu (Python) | 10 % | code |
| FineMath-3+ | 10 % | math |
| PG-19 books (train split) | 17.5 % | long narrative (~0.6–0.7 epoch by 10B) |
| FinePDFs, long only (≥ 4k tokens) | 17.5 % | long expository text |

Open items:
- Phase-2 synthetic data: share, document length (16k), held-out names and sentence shapes so our
  text exams stay unseen. **decide**
- Pretokenize on Odra/Polonez (AWS deferred) into the NAS archive, upload to S3 later; the pipeline
  exists (`scripts/pretokenize_mix.py`); the 2-byte storage and the SmolLM2 tokenizer path need a check.
- Decontamination: PG-19 test/validation books removed from every training source (title / Gutenberg id
  match, including copies inside web and PDF data); held-out sets for evaluation: PG-19 test, a held-out
  slice of long PDFs, a held-out web slice. **missing**

## 3. Model changes

E33 on text as tested at 30M (`evaluation/text_checks.py` `e31c_loop`): local layers see the last 16
tokens; one reading layer sees the last 256 raw tokens plus the notebook (notes from every closed
256-token page); the loop repeats [reading layer + 1 layer] 4 times with a next-token loss at every
round exit (weight 0.3). Beyond the raw read, only the notebook carries information.

**Decided (author, 2026-10-13):**
- **Windows: local layers 128 tokens, reading layer 512 raw tokens** (+ notebook). At 16k rows ~97 %
  of positions have most of their context reachable only through the notebook.
- **Loop: reading layer + 2 layers, 4 rounds.** Follow-up after the first results (keep): the tested
  span (reading layer + 1 layer) and a large looped core (reading layer + 4 layers).

Defaults (taken unless the author objects):
- Width 1280, ~22 layers, FFN width fitted to ~500M parameters (as `fit_params` does for the text tiers).
  In a 22-layer model the loop adds ~+30 % compute, not 2–3× (the DNA models had 4 layers).
- Plain input (no word-pair/triple hash tables, no value embeddings), as the text checks now use.
- Per-round exit losses kept as tested; revisit after the symbolic "E33 without exit losses" control
  (2026-10-14). On web text they apply to every next token (3 extra output-layer passes per step).
- No generation cache now: likelihood-based benchmarks only; revisit for BABILong-style generation.

Open:
- Notebook size at 500M (page length, notes per page, writer width). **decide** (explained to the
  author first)
- Kernels on H100: FlexAttention for the reading layer with notes, `torch.compile` graph breaks (host
  syncs). **partial**
- Notebook-off switch usable in training-time evals (exists for eval: `message_override none`). **partial**

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
