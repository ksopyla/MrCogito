# Evaluation methods for a public release

How the comparisons on a release card are produced, and the traps found so far. Every item here cost a wrong number
once. Scripts are in `evaluation/`; the two 150M releases on the same data (`docs/release/hf_card_concept_memory_150m/`,
`docs/release/hf_card_topic_index_150m/`) are the worked examples, with every raw result file next to its card.

## Contents
1. Choosing references
2. Standard benchmarks (lm-eval) and the leaderboard table
3. Long documents across models (bits per byte)
4. Explaining bits per byte
5. Data efficiency
6. Generation benchmarks (RULER, needles)
7. Text samples
8. Known traps, by model
9. Running on the home servers
10. Comparing our models trained on the same data
11. Export, upload and the shell

## 1. Choosing references

| Role | What qualifies | 2026-11 example |
|---|---|---|
| Same size, strongest open | best open base model within ±20 % of our parameters | SmolLM2-135M |
| Same size, data-matched | public training checkpoints every few B tokens, so it can be scored at our budget | Pythia-160M (every 2.1B tokens) |
| Modern | a base model from the last ~12 months, up to ~3× our size | LFM2.5-350M-Base (2026-03), Gemma 3 270M, Granite 4.0 350M |
| Earliest checkpoint | a big-data reference at its first public checkpoint, closest to our budget | SmolLM2-135M step-240000 (~252B tokens) |

- Write the list down before running. A reference dropped after its result is in is cherry-picking.
- Base models only: check the card for "post-trained", "SFT", "DPO", "instruct". LFM2-350M's released weights are
  post-trained (excluded); LFM2.5-350M-Base is the base.
- Note each model's training tokens **as its card reports them** (quoted), its context length from the card (not only
  `max_position_embeddings`: LFM2.5 configs say 128,000, its card 32,768), and its licence.
- Hub search: `hub_repo_search` with `filters=["text-generation"]`, queries like "base 350M", "140M base", sorted by
  downloads and by `createdAt`.

## 2. Standard benchmarks and the leaderboard table

- `evaluation/run_lm_eval_suite.py`: `--tier leaderboard` (BLiMP 67 sets, ARC-Easy, WikiText-2 byte perplexity) and
  `--tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,sciq,winogrande,wikitext` (the 7 core tasks +
  WikiText). lm-eval 0-shot, max length 2,048, all test items. Ours through `--model perceiver_ar` (our adapter),
  references through the stock `hf` model with `--hf_model <id or local dir>`.
- Report `acc` for every task (the 7-task mean is the plain mean of `acc`; the suite's own `avg_acc` mixes in
  `acc_norm`). Show ± standard error where the harness gives one.
- Precision: `--dtype` (default bf16 on CUDA). Use each model's training precision; fp16-trained models (Pythia) in
  fp32.
- Reproduction check: where the reference publishes its own zero-shot numbers, put them in brackets in the same
  column; ours should agree within ~1–2.5 points (harness versions differ).
- **Leaderboard placement**: `evaluation/tiny_ml_board.py` reproduces the Tiny-ML Leaderboard formula from its
  snapshot (`evaluation/leaderboards/tiny_ml_board_<date>.json`). The normalisation depends on the whole board, so
  score each of our rows **on its own** against the snapshot (as submitting it alone would), and list the board's
  entries with the ranks they get with our model added. Show the top ~10, the neighbourhood of our rank, and our
  reference runs at the rank each would get; mark which rows are the board's runs and which are ours.

## 3. Long documents across models

`evaluation/long_context_compare.py`, two steps:

```bash
python evaluation/long_context_compare.py prepare --tokenizer <our checkpoint> --data <LongDocs-lite shards> \
    --out spans_64k.json.gz                     # once: 55 held-out 64k-token spans, targets fixed in characters
python evaluation/long_context_compare.py score --spans spans_64k.json.gz --swap --model <hub id or folder> \
    [--max_len 32768] [--lead default|bos|none] [--dtype bfloat16]   > compare_<name>.json
```

- Every model scores **the same characters**: the last ~16 KB of each span (the last 4,096 tokens in our tokenizer),
  after context cut to 1k / 4k / 16k / 28k / the whole span, counted in our tokens so every model sees the same text.
- Past a model's maximum length it is scored with **sliding windows** of that length (stride 512); the table marks
  that value "(limit)" and the chart ends the line at the model's length.
- `--swap` adds the control: own nearest 4k kept, older context replaced by another document of the same source.
  A model that reads the content gets no gain from the swapped past.
- `--lead default` starts each input the way the model's tokenizer starts a text (Gemma and LFM2.5 add a start
  token; SmolLM2, Pythia and Granite add none; ours `<|endoftext|>` as in training). Keep this default.
- Summarise with `long_context/summarise_compare.py ours.json ref1.json …`: pooled bits per byte per budget, "saved,
  1k → most context" with a bootstrap 95 % interval over documents, head-to-head differences, share of documents
  where ours is better. The "saved" column compares each model with itself, so it is the measure least affected by
  which model has seen more of these sources.

## 4. Explaining bits per byte

Bits per byte = the average number of bits the model needs to store each byte (about one character of English) of
text it has not seen, using its own predictions: total loss in nats ÷ ln 2 ÷ number of UTF-8 bytes. It counts bytes,
not tokens, so models with different tokenizers compare fairly.

- Scale: 0 = perfect prediction; 8 = plain text (no prediction); lower is better; no upper bound in principle.
- Anchors on the same 55 targets (each compressor given the rest of the document first, 2026-11-18): gzip -9 2.49,
  xz -9e 1.97, bzip2 -9 1.77; the 150M concept memory 0.795, i.e. it could store those pages in about a tenth of
  their size. Recompute anchors for a new document set (a short script over the spans file: compressed size of
  context + target minus compressed size of context).
- A difference of 0.01 bits per byte is about 1.2 % at ~0.8.
- Byte perplexity (lm-eval WikiText) is the same quantity: bits per byte = log₂(byte perplexity); 1.950 → 0.963.
- Nats per token (our own diagnostics) depend on the tokenizer: use them only to compare a model with itself.

## 5. Data efficiency

Claim only with one of these measures, strongest first, and always name the confound (data and architecture are not
separated without a same-data control):
1. **Same budget**: a data-matched reference's checkpoints bracketing our token count, same harness
   (Pythia: tokens = step × 2,097,152).
2. **Tokens to match**: the tokens the reference needs to reach our score, log-linear between its checkpoints, as a
   multiple of ours ("not reached by 300B" is a valid answer).
3. **Against the field** (quoted): our leaderboard score against the training tokens other entries report.

`figures/make_figures.py` in the reference release draws score against training tokens and prints tokens-to-match.
SmolLM2-135M intermediate checkpoints: `HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints`, branches
`step-240000` … `step-1920000`, 1,048,576 tokens per step (first ~252B tokens).

## 6. Generation benchmarks (RULER, needles)

- `evaluation/run_lsa_suite.py --suite ruler`: the 13 RULER tasks, 100 prompts each, one lm-eval call per length.
  **Answer length: leave `--max_gen_toks` at 0** (each task's own length, up to 128 tokens in lm-eval's RULER
  configs). Until 2026-11-18 the default was 32, which held every answer in GPT-2 tokens but not in the SmolLM2
  tokenizer (every digit is a token: one UUID 35 tokens, four 7-digit numbers 36, ten numbered words 41).
- `--answers_out replies.jsonl` saves every reply with its references and score: read a few before trusting a zero.
  (The 2026-11-18 rerun showed the UUID tasks fail because the model writes prose, not a UUID: the cap was not the
  only cause.)
- Greedy decoding, as in the official RULER setup; beam search would be a separate, labelled column.
- Say whether the test is zero-shot: training on planted-fact lookups makes the needle tests "learned", not
  transfer.
- Run lengths on separate GPUs (`--lengths 4096,8192`, `16384`, `32768`) with batch 8 / 4 / 2 on 24 GB cards.
- **Reference models on RULER:** `--hf_model <id or folder> --tokenizer <same>` runs them through lm-eval's stock
  `hf` model with the same tasks, lengths and answer lengths; each model's prompts are sized with its own tokenizer.
  Give every model room for prompt + answer (`--max_length` 0 = length + 256): RULER's 8k prompts run slightly over
  8,192 tokens (8,205 seen), and at an 8,192 cap lm-eval cuts the start of the prompt, which is the instruction.
  A radar of RULER only makes sense with references on it; for our model alone across lengths use a heatmap.
- **Comparing two of our models on RULER prompt by prompt:** check first that the saved expected answers match per
  (length, task, prompt id). The needle tasks draw from Python's shared random state, and two runs with the same
  settings and the same task groups drew different needles for 8 of 13 tasks at 32k (2026-11-18; 4k–16k matched);
  the cause was not found. Pair where the answers match, compare the rest as two separate draws (unpaired bootstrap),
  and say which is which.
- **lm-eval and start tokens:** with `add_bos_token` unset, lm-eval encodes with the tokenizer's defaults, so models
  whose tokenizer adds a start token (LFM2.5, Gemma) get it; on whole-document tasks (WikiText) such a model may see
  it twice (prefix + encoding). Checked for LFM2.5-350M-Base on 2026-11-18: byte perplexity 2.1116 vs 2.1115 with a
  single start token, no difference.

## 7. Text samples

```bash
python evaluation/sample_text.py --model <our checkpoint> --out samples_ours.json
python evaluation/sample_text.py --model HuggingFaceTB/SmolLM2-135M --hf --out samples_ref.json
```

Nucleus sampling (temperature 0.7, top-p 0.9, repetition penalty 1.1) for seeds 0, 1, 2 and beam search (4 beams, no
repeated 3-gram), 60 new tokens. Show seed 0 for each prompt, unedited (cut at a sentence end and mark the cut), name
the settings, ship the JSON. Point out factual errors in one sentence; do not hide them.

## 8. Known traps, by model

| Model | Trap | Fix |
|---|---|---|
| Pythia (any size) | fp16-trained; bf16 costs ~6 ARC-Easy points (37.5 vs 43.6) | `--dtype float32` |
| Granite 4.0 350M | a leading `<|end_of_text|>` breaks long inputs (2.12 vs 1.07 bits per byte at 19.8k) | `--lead default` |
| LFM2-350M | released weights are post-trained for chat | use LFM2.5-350M-Base |
| LFM2.5-350M-Base | `tokenizer_class: TokenizersBackend` needs Transformers 5 | on 4.x: download to a local folder, set `PreTrainedTokenizerFast` in its `tokenizer_config.json` (same `tokenizer.json`, still adds `<|startoftext|>`) |
| SmolLM2-135M checkpoints | intermediate checkpoints read 2,048 tokens; the 8k came in the final phase | report the earliest checkpoint's context as 2k |
| LFM2.5-350M-Base | card says 32k context; on our documents it predicts best at 4k and gets worse with more (0.879 → 0.928 bits per byte at 28k) | checked fp32 vs bf16, SDPA vs eager, Transformers 4.57 vs 5.23 on one book: identical; reported as measured |
| MobileLLM-R1-140M-base | gated (FAIR licence); the server's account had no access on 2026-11-18 | accept the licence on that account first, or leave it out and say so |
| Mamba / hybrid SSMs | without `mamba_ssm` + `causal_conv1d` the 🤗 path scans token by token: too slow at 61k | install the kernels in a separate environment, or leave out |
| Our perceiver_ar | not a stock 🤗 model; flex attention recompiles per new length | our lm-eval adapter for benchmarks; `attn_backend="sdpa"` for short sampling loops; the cached decoder for long generation |
| Our perceiver_ar, a new read | a diagnostic switch written for one read is a silent no-op on another: `long_context_use.py`'s "memory off" only knew the concept read, so on the LSA read it scored the model as trained (fixed 2026-11-18: it sets `lsa_off` and refuses a switch the checkpoint lacks) | before a full run, score one document with the switch on and off and check the loss changes; make switches refuse what a checkpoint does not have |

## 9. Running on the home servers

- Code reaches Odra/Polonez by git only; to run a script from a worktree without touching the server checkout, pipe
  it: `ssh odra 'cd ~/dev/MrCogito-eng && .venv/bin/python - <args>' < evaluation/<script>.py`.
- Long jobs must survive the ssh session: `export MRC_SRC="$(cat)"; cd …; setsid nohup <python> -c "import os;
  exec(compile(os.environ['MRC_SRC'], '<name>', 'exec'), {'__name__': '__main__', '__file__': '<stdin>'})" <args>
  > log 2>&1 < /dev/null &` (a `__future__` import rules out prepending code to the script).
- RULER needs `wonderwords`: `~/.local/bin/uv run --no-sync --with wonderwords python …` adds it without changing
  the server's environment.
- The Bash tool's shell is zsh: `set -- $var` does not split words; pass arguments separately.
- Stamp every run with UTC start and end times; pull result files into the release folder as soon as they land.
- **One command for the whole protocol:** `scripts/release_evals.sh <checkpoint> <out> <longdocs dir> 0,1,2
  <memory switches> <tag>` (set `HF_EXPORT=<export folder>` for the board tests through the stock `hf` model, `LONGCTX`
  to reuse a document cache, `ONLY="job …"` to rerun a subset). One shared queue: each card takes the next job when it
  has no compute process, other sessions' jobs included, longest jobs first; every job stamped in UTC in
  `logs/events.log`. The 150M models take ≈1 h 15 on three RTX 3090 once the cards are free. Launch it with
  `setsid -f` so the ssh call returns at once.
- **Check the cards before a long run:** `timeout 20 nvidia-smi` must answer. On 2026-11-18 a Polonez card hit Xid 62
  (micro-controller halt) six seconds into a run; the driver hung, every job and `nvidia-smi` blocked in the kernel
  (killed processes stay in D state), and the server needed a reboot. Look at `journalctl -k` for `Xid` when anything
  hangs, stop only your own PIDs, and move the run.
- **Shared servers:** another session's jobs may hold the cards; queue behind them (the script waits) and tell that
  session which cards you will take.

## 10. Comparing our models trained on the same data

- Copy the other model's raw result files into this card's folder under a prefix (`cm_ruler_*`, `cm_real_docs_*`,
  `compare_concept_memory.json`, `e38_mem150m_*` in `eval_results/`), so every comparison can be recomputed from one
  Hub repo, and add this model's files to the other card the same way.
- Our models share the SmolLM2 tokenizer, so nats per token compare directly between them, document by document:
  report paired differences (bootstrap over documents) for the context cut, the full context, the swapped past and
  the switches, plus the cross-model test in bits per byte. RULER: paired over prompts where the expected answers match
  (section 6).
- One table per card, "Models on the same data", with every test, both models, the difference with its interval and
  a verdict (ahead / level / behind; "0.2 %" when a real difference is tiny). The training losses at the end of each
  stage go in it too: two models on the same rows should land within a few thousandths unless the read matters there.
- Generate every table from the result files with one script; never type a number by hand.

## 11. Export, upload and the shell

- **Export into an empty folder** after the last card edit: `export_manifest.json` records each file's hash, the
  README included, and an export into a folder from an earlier run lists files from that run. Copy the card's other
  files (EVALUATION.md, figures, results, LICENSE, NOTICE) in after the export.
- **A borrowed method ships its licence notice:** the export keeps the original copyright header of a vendored module
  (the LSA buckets: Apache-2.0, Prefrontal AI); add the licence text as `LICENSE` and a `NOTICE` naming their code.
- **Private-name check:** `find <folder> -type f ! -name model.safetensors -print0 | xargs -0 grep -liE …`; grepping
  the weights for text stalls.
- **Hub reads with an expired login:** a cached token that has expired makes even public reads fail with 401; use
  `HfApi(token=False)` for anonymous reads of public repos, and the organisation's write token (huggingface-project
  skill) only for writes.
- **Writing card text from the shell:** quote heredocs (`<<'EOF'`). An unquoted heredoc runs every backticked phrase
  in the text (`hf`, `from_pretrained`, file names) as a command and pastes its output into the card; prefer writing
  a script file for long edits.
