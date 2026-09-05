# Phase-1 plan — LengthGroupedSampler + length cache (pad-free eng)

Companion to [`pad_free_variable_length_training.md`](./pad_free_variable_length_training.md).
**Scope:** Phase 0–1 implementation contract and retained design rationale.

- **Status:** implemented and locally verified (2026-08-14; `388 passed, 9 skipped`);
  Polonez throughput benchmark pending
- **Phase covered:** Phase 0 (pad_ratio metrics) + Phase 1 (length_group sampler)
- **Non-goals here:** token-budget sampler (Phase 2), FA varlen / packing (Phase 3)

> **Implementation note — 2026-08-14:** production uses
> `CachedLengthGroupedSampler` with a deterministic 20-window default. It deliberately
> emits the same full index stream on every process and leaves disjoint rank sharding to
> Hugging Face Accelerate, matching Trainer's native sampler contract. The earlier sketches
> below in which the custom sampler owns `world_size`/`rank` are superseded; double-sharding
> would be incorrect. Length grouping is enabled for E17c, while all historical launchers
> retain `BATCH_PACKING_MODE=none`.

---

## 0. Repo facts that constrain the design

| Fact | Where | Implication |
|---|---|---|
| Pretok train is **map-style**, not `IterableDataset` | `data/dataset_preprocess.py` `_fast_weighted_all_exhausted_interleave` → `concatenate_datasets` + `select(indices)` | Random-access `Sampler` is legal on interleaved pretok |
| Interleave seed lives in the manifest | `load_pretokenized_mix` uses `manifest["seed"]` | Length cache must key on **interleaved** index space (= manifest bytes + that seed) |
| Sidecar cache pattern already exists | `scripts/manifest_token_stats.py` → `manifest.json.token_stats.json` | Mirror for lengths: `manifest.json.lengths/` (HF Arrow via `save_to_disk`) |
| Collator already pad-to-batch-max | `data/data_collators.py` `DataCollatorForCausalLM` | Phase 1 leaves collator **unchanged** |
| Trainer has a gated sampler override | `training/concept_pretraining_trainer.py` `PerceiverDenoiseTrainer` | `none` delegates to HF; `length_group` supplies one shared sortish stream and Accelerate shards it |
| Launcher is env → CLI | `scripts/train_concept_pretraining_multigpu.sh` | New `BATCH_PACKING_MODE` env → `--batch_packing_mode` |
| HF ships `LengthGroupedSampler` + `group_by_length` | `transformers.trainer_pt_utils` (pin `transformers>=4.57.6,<5`) | Reuse algorithm; **do not** enable bare `--group_by_length` (it re-scans lengths every launch) |

---

## 1. Exact API — length cache sidecar

### 1.1 On-disk contract

Next to a pretok manifest (same directory as today):

```text
$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json
$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json.token_stats.json   # existing
$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json.lengths/            # HF Arrow dataset
$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json.lengths.meta.json
```

**`.lengths/` contents** — a one-column Hugging Face dataset (`save_to_disk`):

| column | dtype | meaning |
|---|---|---|
| `length` | `int32` | `len(input_ids)` for interleaved train row `i ∈ [0, N)` |

**`.lengths.meta.json`** holds invalidation fields: `manifest_sha256`, `n_rows`, `seed`, `max_seq_length`, `format=hf_datasets_arrow`.

The first npz sidecar (`*.lengths.npz`) is obsolete; a successful rewrite deletes it.

**Invalidation:** recompute if `manifest_sha256` mismatches `sha256(manifest bytes)` OR `n_rows != len(train_ds)` OR `seed` mismatch. Same atomic write as token_stats (`*.tmp` → `replace`).

### 1.2 Python API (`data/length_cache.py` — new)

```python
# data/length_cache.py
from pathlib import Path
import numpy as np

def length_cache_paths(manifest_path: Path) -> tuple[Path, Path]:
    """Return (dataset_dir, meta_path) beside the manifest."""
    ...

def compute_or_load_interleaved_lengths(
    manifest_path: str | Path,
    *,
    train_ds=None,          # optional; if None, call load_pretokenized_mix
    num_proc: int | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """Return int32[N] lengths aligned to load_pretokenized_mix(train) indices.

    Implementation sketch:
      1. Load/validate meta + Arrow length dataset.
      2. On miss: Dataset.map over train_ds.select_columns(["input_ids"]) with
         batched len() (same worker pattern as scripts/manifest_token_stats.py / pretokenize).
      3. Atomic save_to_disk + meta.
    """
    ...

def assert_lengths_match_dataset(lengths: np.ndarray, dataset) -> None:
    """Raise if len(lengths) != len(dataset) or any length < 1."""
    ...
```

**CLI** (thin wrapper, parallel to token_stats):

```bash
# scripts/manifest_length_cache.py
uv run python scripts/manifest_length_cache.py \
  --manifest "$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json" \
  [--force] [--num_proc 32]
# prints path + n_rows + length histogram summary (p50/p90/p99/max)
```

Optional: call from `scripts/train_concept_pretraining_multigpu.sh` when
`BATCH_PACKING_MODE=length_group` and cache missing (rank-0 / once), analogous to
`manifest_token_stats.py` epoch estimation.

### 1.3 Why interleaved lengths (not per-source)

`load_pretokenized_mix` reorders rows via weighted all-exhausted interleave. A
per-source Arrow length column would require replaying the same RNG choices to
map source-local → interleaved indices. Caching **after** interleave (exactly
what `manifest_token_stats` already iterates) is simpler, deterministic, and
guarantees `lengths[i]` matches `train_ds[i]`.

---

## 2. Exact API — LengthGroupedSampler

### 2.1 Module (`data/length_grouped_sampler.py`)

Prefer wrapping HF’s algorithm with an explicit lengths array (avoid dataset walk):

```python
# data/length_grouped_sampler.py
from torch.utils.data import Sampler
import numpy as np

class CachedLengthGroupedSampler(Sampler[int]):
    """HF-style length grouping over a precomputed lengths array.

    Mega-batch size = batch_size * mega_batch_mult. Production defaults to 20:
    enough local similarity without globally sorting an epoch.

    Algorithm (same as transformers.trainer_pt_utils.LengthGroupedSampler):
      1. permutation of [0, N) with generator seeded by `seed + epoch`
      2. split into mega-batches of size mega
      3. sort each mega-batch by lengths[i] descending
      4. flatten → list of indices
      5. emit the full deterministic stream; Accelerate shards dataloader batches
         across ranks using its standard Trainer path
    """

    def __init__(
        self,
        lengths: np.ndarray | list[int],
        batch_size: int,
        *,
        seed: int = 0,
        mega_batch_mult: int = 20,
    ): ...

    def set_epoch(self, epoch: int) -> None:
        """Required for DDP reshuffle across epochs (HF Trainer calls this)."""
        ...

    def __iter__(self): ...
    def __len__(self) -> int: ...
```

**Reuse note:** implementation may call
`transformers.trainer_pt_utils.get_length_grouped_indices(lengths, batch_size, ...)`
with an explicit generator seeded by `seed + epoch`. Keep our class so cached
`np.ndarray` lengths avoid a dataset rescan and epoch reshuffling is deterministic.

### 2.2 Config surface

**`DataTrainingArguments`** (`training/concept_pretraining_args.py`):

```python
batch_packing_mode: str = field(
    default="none",
    metadata={
        "help": "Pad-reduction sampler: 'none' | 'length_group' | 'token_budget' "
        "(token_budget = Phase 2; reject until implemented)."
    },
)
length_group_mega_batch_mult: int = field(
    default=20,
    metadata={"help": "Mega-batch multiplier for length_group (HF-compatible)."},
)
```

**Launcher** (`scripts/train_concept_pretraining_multigpu.sh`):

```bash
BATCH_PACKING_MODE="${BATCH_PACKING_MODE:-none}"   # none|length_group|token_budget
LENGTH_GROUP_MEGA_BATCH_MULT="${LENGTH_GROUP_MEGA_BATCH_MULT:-20}"

# pass through:
#   --batch_packing_mode "$BATCH_PACKING_MODE"
#   --length_group_mega_batch_mult "$LENGTH_GROUP_MEGA_BATCH_MULT"

# when length_group + PRETOKENIZED_MANIFEST set:
#   ensure lengths cache exists (uv run scripts/manifest_length_cache.py ...)
```

Default **`none`** preserves E17b checkpoint / curve comparability.

---

## 3. HF Trainer / PerceiverDenoiseTrainer wiring (DDP-safe)

### 3.1 Where to hook

| Step | File | Change |
|---|---|---|
| Load lengths after datasets | `training/concept_pretraining_factories.py` `load_pretraining_datasets` **or** `training/train_concept_pretraining.py` after `load_pretraining_datasets` | If `batch_packing_mode=="length_group"`: require `pretokenized_manifest`; `lengths = compute_or_load_interleaved_lengths(...)`; assert match |
| Pass lengths into trainer | `training/train_concept_pretraining.py` | `PerceiverDenoiseTrainer(..., train_lengths=lengths, batch_packing_mode=...)` |
| Override sampler | `training/concept_pretraining_trainer.py` | Implement `_get_train_sampler` |
| Leave collator alone | `data/data_collators.py` | No Phase-1 change |
| Eval path | unchanged | Eval keeps default sequential / distributed eval sampler (no length grouping required) |

### 3.2 Sampler override sketch

```python
# training/concept_pretraining_trainer.py  (sketch only)
class PerceiverDenoiseTrainer(Trainer):
    def __init__(self, *args, batch_packing_mode: str = "none",
                 train_lengths=None, length_group_mega_batch_mult: int = 20, **kwargs):
        self.batch_packing_mode = batch_packing_mode
        self.train_lengths = train_lengths
        self.length_group_mega_batch_mult = length_group_mega_batch_mult
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.batch_packing_mode == "none":
            return super()._get_train_sampler(train_dataset)

        if self.batch_packing_mode == "token_budget":
            raise NotImplementedError("Phase 2 — use BATCH_PACKING_MODE=length_group")

        if self.batch_packing_mode != "length_group":
            raise ValueError(f"Unknown batch_packing_mode={self.batch_packing_mode!r}")

        # Hard fail closed on streaming / iterable
        if isinstance(dataset, torch.utils.data.IterableDataset):
            raise ValueError(
                "batch_packing_mode=length_group requires a map-style dataset; "
                "got IterableDataset. Use a pretok manifest (load_pretokenized_mix)."
            )
        if self.train_lengths is None:
            raise ValueError("length_group requires train_lengths from the length cache.")

        if self.args.group_by_length:
            raise ValueError(
                "Do not combine --group_by_length with --batch_packing_mode=length_group; "
                "the cached sampler owns length grouping."
            )

        return CachedLengthGroupedSampler(
            lengths=self.train_lengths,
            batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
            seed=self.args.seed,
            mega_batch_mult=self.length_group_mega_batch_mult,
        )
```

### 3.3 DistributedSampler interaction (do / don’t)

| Do | Don’t |
|---|---|
| Emit one identical deterministic index stream and let Accelerate shard prepared dataloader batches | Pre-shard in the sampler and let Accelerate shard again — that double-shards |
| Implement `set_epoch` so Trainer/Accelerate reshuffles each epoch | Rely on process-local RNG without a shared seed — ranks could disagree |
| Keep identical `lengths` arrays on every rank after a main-process-first cache barrier | Let several ranks race to rewrite a missing cache |
| Keep `DataLoader(shuffle=False)` when a custom sampler is set (HF already does this) | Enable HF `--group_by_length` in parallel |
| Simulate 4-rank batch assignment and verify disjoint union + similar concurrent maxima | Assume single-process sampler tests prove distributed alignment |

**Accelerate / DDP:** `accelerate launch` + HF Trainer already sets `world_size` / `process_index`. Our override replaces only the train sampler; DDP gradient sync is unchanged. Uneven **compute** per step (one rank’s bucket still longer) remains possible — Phase 0 metrics expose it; Phase 2 token-budget addresses it.

### 3.4 Factory gate

In `load_pretraining_datasets` / train entrypoint:

- `length_group` **requires** `data_args.pretokenized_manifest` (online mix also map-style, but no sidecar path — fail with message to pretok first).
- `token_budget` → `NotImplementedError` until Phase 2.
- `none` → zero behavior change.

---

## 4. Unit tests to add under `tests/`

| File | Cases |
|---|---|
| `tests/test_length_cache.py` | write/load roundtrip; sha256 miss forces recompute; `n_rows` mismatch forces recompute; atomic `.tmp` cleaned; lengths equal `len(row["input_ids"])` on interleaved `load_pretokenized_mix` (reuse tiny manifest fixture from `tests/test_manifest_token_stats.py`) |
| `tests/test_length_grouped_sampler.py` | single-process: indices are permutation of `range(N)` (or drop_last subset); within each microbatch of size `B`, `max(L)-min(L)` ≪ global range on a bimodal length fixture; `set_epoch` changes order; `seed` reproducibility |
| `tests/test_length_grouped_sampler_ddp.py` | fake `world_size=2/4`: rank index sets disjoint; union covers full (non-drop) epoch; all ranks same mega-batch sort given same lengths+seed+epoch |
| `tests/test_batch_packing_mode_gates.py` | `IterableDataset` + `length_group` raises; missing `train_lengths` raises; `group_by_length=True` conflict raises; `token_budget` raises NotImplementedError; `none` calls through (mock `super()._get_train_sampler`) |
| `tests/test_pad_ratio_metrics.py` | helper: given `attention_mask`, `pad_ratio = 1 - sum(mask)/(B*L)`; all-real → 0; half-pad → 0.5; all-reduce stub / rank aggregation shape |

Do **not** require GPU. Collator regression already covered by `tests/test_data_collators.py` — add one optional case that a length-similar feature list yields near-zero pad under existing `DataCollatorForCausalLM` (documents Phase-1 invariant: collator unchanged).

---

## 5. W&B metrics for `pad_ratio` (Phase 0)

Log under a stable namespace (rank0 after local compute; all-reduce sums for globals):

| Key | Definition | Notes |
|---|---|---|
| `data/pad_ratio` | `1 - real_tokens / (B * L_max)` | Primary success metric; median ≤ 0.10 under length_group |
| `data/real_tokens_per_batch` | globally summed real tokens / microbatches | Logging-window mean |
| `data/padded_tokens_per_batch` | globally summed pad slots / microbatches | Logging-window mean |
| `data/mean_sequence_length` | real tokens / rows | |
| `data/mean_batch_max_length` | sum of local batch maxima / microbatches | Exposes rank shape |
| `perf/real_tokens_per_second` | global real tokens / wall delta | Contrast with padded throughput |

**Implemented hook:** `PerceiverDenoiseTrainer.compute_loss` accumulates detached mask
counts and the regular all-rank training `log({"loss": ...})` reduces and flushes them.
Rank-zero-only evaluation extension logs never enter the collective.

All-reduce: `real_tokens` and `padded_tokens` sum across ranks; `pad_ratio` from globals. Do not average per-rank pad_ratios (biased if length skew across ranks).

---

## 6. Conflict with interleaved pretok / IterableDataset

| Path | Type today | Phase-1 length_group |
|---|---|---|
| `load_pretokenized_mix` | **map-style** interleaved (concat+select) | **Supported** — primary path |
| `load_and_preprocess_dataset_mix` (live tokenize) | map-style interleaved | Technically sampleable, but **no sidecar**; gate: require pretok manifest |
| Single Hub dataset path | map-style | Same gate / or compute ephemeral lengths in RAM for tiny smokes only |
| True `datasets.IterableDataset` / streaming | not used by pretok today | **Hard error** if `batch_packing_mode=length_group` |
| HF `interleave_datasets(..., streaming=True)` | N/A today | If introduced later, length_group stays disabled |

**Subtlety:** interleave changes the **order** of rows vs per-source Arrow files. Length cache must never be built from raw `train_path` shards alone without applying `_fast_weighted_all_exhausted_interleave`. Building via `load_pretokenized_mix` then mapping lengths is the only supported construction.

**Weight / seed changes:** editing manifest weights or `seed` changes interleaved order → sha256 changes → cache miss → recompute. Good.

---

## 7. Phase-1 implementation checklist (paste into eng spec)

```markdown
### Phase 1 — implementation checklist (length_group)

**Phase 0 (metrics, can land first / same PR)**
- [x] Add stable W&B padding/length/useful-throughput keys
- [x] Wire trainer hook; all-reduce token sums before ratio
- [ ] Confirm E17b-like `bs=8` baseline logs pad_ratio ≳ 0.5 on `e16b_long_4k_v1`

**Length cache**
- [x] Add `data/length_cache.py` (`compute_or_load_interleaved_lengths`)
- [x] Add `scripts/manifest_length_cache.py` CLI
- [x] Sidecar: `*.lengths/` (HF Arrow) + `*.lengths.meta.json` next to pretok manifest
- [x] Invalidate on manifest sha256 / n_rows / seed mismatch (atomic write)
- [x] Tests: `tests/test_length_cache.py`

**Sampler + Trainer**
- [x] Add `data/length_grouped_sampler.py` (`CachedLengthGroupedSampler`)
- [x] Extend `DataTrainingArguments`: `batch_packing_mode`, `length_group_mega_batch_mult`
- [x] Override `PerceiverDenoiseTrainer._get_train_sampler`; Accelerate owns rank sharding
- [x] Gate: pretok manifest required; reject non-map datasets and HF `--group_by_length`
- [x] Pass `--batch_packing_mode` from `scripts/train_concept_pretraining_multigpu.sh`
      via `BATCH_PACKING_MODE` (default `none`)
- [x] Leave `DataCollatorForCausalLM` unchanged
- [x] Tests cover padding reduction, deterministic epochs, simulated 4-rank assignment,
      trainer wiring/metrics, cache gates, and launcher flow

**Polonez smoke (falsify)**
- [ ] Same E17b config `bs=8 accum=1` seq4k, `BATCH_PACKING_MODE=length_group`
- [ ] pad_ratio median ≤ 0.10; real tok/s ≥ 1.25× unbucketed `bs=8`; early loss within noise
- [ ] Kill → keep default `none` if metrics miss or DDP hangs
```

---

## 8. Suggested file touch list (implement PR)

| Path | Action |
|---|---|
| `data/length_cache.py` | **new** |
| `data/length_grouped_sampler.py` | **new** |
| `scripts/manifest_length_cache.py` | **new** |
| `training/concept_pretraining_args.py` | add fields |
| `training/concept_pretraining_trainer.py` | `_get_train_sampler` + optional pad metrics |
| `training/train_concept_pretraining.py` | load lengths; pass kwargs |
| `training/concept_pretraining_factories.py` | optional gate helper |
| `scripts/train_concept_pretraining_multigpu.sh` | `BATCH_PACKING_MODE` env |
| `tests/test_length_cache.py` | **new** |
| `tests/test_length_grouped_sampler.py` | **new** |
| `tests/test_length_grouped_sampler_ddp.py` | **new** |
| `tests/test_batch_packing_mode_gates.py` | **new** |
| `tests/test_pad_ratio_metrics.py` | **new** |
| `docs/engineering_specs/pad_free_variable_length_training.md` | tick checklist / link this plan |
| `CHANGELOG.md` | when implementing (not this design-only handoff) |

---

## 9. Decision recap for implementer

1. Phase 1 = **sampler + sidecar only**; collator and BackboneConceptLM forward stay put.
2. Lengths are for the **interleaved** pretok train index space, cached beside the manifest like token_stats.
3. DDP: custom sampler emits one shared stream; Accelerate owns rank sharding.
4. Interleaved pretok is already map-style — no IterableDataset conflict today; still fail closed if streaming appears.
5. Default `BATCH_PACKING_MODE=none` until Polonez smoke clears success criteria.
