# Phase-1 plan — LengthGroupedSampler + length cache (pad-free eng)

Companion to [`pad_free_variable_length_training.md`](./pad_free_variable_length_training.md).
**Scope:** design + concrete API / wiring sketch only. No production sampler yet.

- **Status:** design handoff (2026-08-10)
- **Phase covered:** Phase 0 (pad_ratio metrics) + Phase 1 (length_group sampler)
- **Non-goals here:** token-budget sampler (Phase 2), FA varlen / packing (Phase 3)

---

## 0. Repo facts that constrain the design

| Fact | Where | Implication |
|---|---|---|
| Pretok train is **map-style**, not `IterableDataset` | `data/dataset_preprocess.py` `_fast_weighted_all_exhausted_interleave` → `concatenate_datasets` + `select(indices)` | Random-access `Sampler` is legal on interleaved pretok |
| Interleave seed lives in the manifest | `load_pretokenized_mix` uses `manifest["seed"]` | Length cache must key on **interleaved** index space (= manifest bytes + that seed) |
| Sidecar cache pattern already exists | `scripts/manifest_token_stats.py` → `manifest.json.token_stats.json` | Mirror for lengths: `manifest.json.lengths.npz` |
| Collator already pad-to-batch-max | `data/data_collators.py` `DataCollatorForCausalLM` | Phase 1 leaves collator **unchanged** |
| Trainer does not override samplers today | `training/concept_pretraining_trainer.py` `PerceiverDenoiseTrainer` | Add `_get_train_sampler` override; default path stays HF `DistributedSampler` |
| Launcher is env → CLI | `scripts/train_concept_pretraining_multigpu.sh` | New `BATCH_PACKING_MODE` env → `--batch_packing_mode` |
| HF ships `LengthGroupedSampler` + `group_by_length` | `transformers.trainer_pt_utils` (pin `transformers>=4.57.6,<5`) | Reuse algorithm; **do not** enable bare `--group_by_length` (it re-scans lengths every launch) |

---

## 1. Exact API — length cache sidecar

### 1.1 On-disk contract

Next to a pretok manifest (same directory as today):

```text
$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json
$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json.token_stats.json   # existing
$DATASETS_TOK_DIR/e16b_long_4k_v1_gemma_manifest.json.lengths.npz         # NEW
```

**`.lengths.npz` contents**

| key | dtype | meaning |
|---|---|---|
| `lengths` | `int32[N]` | `len(input_ids)` for interleaved train row `i ∈ [0, N)` |
| `manifest_sha256` | `U64` (or stored in sidecar JSON; prefer JSON meta — see below) | invalidation |
| `n_rows` | scalar | must equal `len(train_ds)` after `load_pretokenized_mix` |
| `max_seq_length` | scalar | from manifest (sanity) |
| `seed` | scalar | manifest interleave seed |

Prefer a **two-file** pattern (matches token_stats readability + npz density):

```text
manifest.json.lengths.npz          # only array: lengths
manifest.json.lengths.meta.json    # sha256, n_rows, seed, max_seq_length, created_at
```

Or single JSON meta embedding a path — keep npz for the array (N can be millions).

**Invalidation:** recompute if `manifest_sha256` mismatches `sha256(manifest bytes)` OR `n_rows != len(train_ds)` OR `seed` mismatch. Same atomic write as token_stats (`*.tmp` → `replace`).

### 1.2 Python API (`data/length_cache.py` — new)

```python
# data/length_cache.py
from pathlib import Path
import numpy as np

def length_cache_paths(manifest_path: Path) -> tuple[Path, Path]:
    """Return (npz_path, meta_path) beside the manifest."""
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
      1. Load/validate meta + npz.
      2. On miss: map over train_ds.select_columns(["input_ids"]) with
         batched len() reduce (same worker pattern as scripts/manifest_token_stats.py).
      3. Atomic write npz + meta.
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
  [--force] [--num_proc 8]
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

### 2.1 Module (`data/length_grouped_sampler.py` — new)

Prefer wrapping HF’s algorithm with an explicit lengths array (avoid dataset walk):

```python
# data/length_grouped_sampler.py
from torch.utils.data import Sampler
import numpy as np

class ManifestLengthGroupedSampler(Sampler[int]):
    """HF-style length grouping over a precomputed lengths array.

    Mega-batch size = batch_size * world_size * mega_batch_mult
    (default mega_batch_mult=1 matches HF LengthGroupedSampler defaults).

    Algorithm (same as transformers.trainer_pt_utils.LengthGroupedSampler):
      1. permutation of [0, N) with generator seeded by `seed + epoch`
      2. split into mega-batches of size mega
      3. sort each mega-batch by lengths[i] descending
      4. flatten → list of indices
      5. if distributed: emit only indices for this rank's microbatches
         (stride by world_size over consecutive batch_size chunks)

    CRITICAL: do NOT wrap with DistributedSampler. This class owns rank slicing.
    """

    def __init__(
        self,
        lengths: np.ndarray | list[int],
        batch_size: int,
        *,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        mega_batch_mult: int = 1,
        drop_last: bool = False,
    ): ...

    def set_epoch(self, epoch: int) -> None:
        """Required for DDP reshuffle across epochs (HF Trainer calls this)."""
        ...

    def __iter__(self): ...
    def __len__(self) -> int: ...
```

**Reuse note:** implementation may call
`transformers.trainer_pt_utils.get_length_grouped_indices(lengths, batch_size, ...)`
then apply rank slicing — keep our class so we inject `np.ndarray` lengths and
own DDP semantics explicitly.

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
    default=1,
    metadata={"help": "Mega-batch multiplier for length_group (HF-compatible)."},
)
```

**Launcher** (`scripts/train_concept_pretraining_multigpu.sh`):

```bash
BATCH_PACKING_MODE="${BATCH_PACKING_MODE:-none}"   # none|length_group|token_budget
LENGTH_GROUP_MEGA_BATCH_MULT="${LENGTH_GROUP_MEGA_BATCH_MULT:-1}"

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
                 train_lengths=None, length_group_mega_batch_mult: int = 1, **kwargs):
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
                "our sampler owns length grouping + DDP rank slicing."
            )

        world_size = 1 if self.args.world_size is None else self.args.world_size
        rank = 0 if self.args.process_index is None else self.args.process_index
        return ManifestLengthGroupedSampler(
            lengths=self.train_lengths,
            batch_size=self.args.per_device_train_batch_size,
            world_size=world_size,
            rank=rank,
            seed=self.args.seed,
            mega_batch_mult=self.length_group_mega_batch_mult,
            drop_last=self.args.dataloader_drop_last,
        )
```

### 3.3 DistributedSampler interaction (do / don’t)

| Do | Don’t |
|---|---|
| Let `ManifestLengthGroupedSampler` take `world_size` / `rank` and emit **disjoint** index streams | Nest `DistributedSampler(LengthGroupedSampler(...))` — double-sharding, hang / skew |
| Implement `set_epoch` so Trainer reshuffles each epoch | Rely on process-local RNG without a shared seed — ranks would disagree on mega-batches |
| Keep identical `lengths` array on every rank (load same npz) | Rank-0-only length compute without barrier / broadcast of cache path |
| Keep `DataLoader(shuffle=False)` when a custom sampler is set (HF already does this) | Enable HF `--group_by_length` in parallel |
| Verify `__len__` per rank ≈ `ceil(N / (batch * world)) * batch` (or drop_last policy) | Allow unequal step counts across ranks without `dataloader_drop_last` / accelerator join |

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
| `data/real_tokens` | `attention_mask.sum()` (or `labels != -100`) | Per microbatch, pre-accum |
| `data/padded_tokens` | `B * L_max - real_tokens` | |
| `data/mean_L` | `real_tokens / B` | Mean true length in microbatch |
| `data/max_L` | `L_max` (= `input_ids.size(1)`) | |
| `data/batch_rows` | `B` | Useful when Phase 2 varies B |
| `perf/tokens_per_sec_real` | `global_real_tokens / wall_delta` | Contrast with any padded tok/s |
| `data/pad_ratio_running_median` | optional window median | Stabilise vs step noise |

**Hook options (pick one in implement):**

1. **Preferred:** small `PadRatioCallback(TrainerCallback)` in `training/` — `on_log` / custom `on_step_end` reading last batch stats stashed by collator or trainer.
2. **Alt:** thin wrapper around `DataCollatorForCausalLM` that records last-batch stats (collator stays pad-to-batch-max; wrapper only measures). Avoid mutating collator class API if possible.
3. **Alt:** inside `PerceiverDenoiseTrainer.compute_loss` when `attention_mask` present — simplest, but mixes metrics into loss path.

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
- [ ] Add pad-ratio helper + W&B keys: `data/pad_ratio`, `data/real_tokens`,
      `data/padded_tokens`, `data/mean_L`, `data/max_L`, `perf/tokens_per_sec_real`
- [ ] Wire callback or trainer hook; all-reduce token sums before ratio
- [ ] Confirm E17b-like `bs=8` baseline logs pad_ratio ≳ 0.5 on `e16b_long_4k_v1`

**Length cache**
- [ ] Add `data/length_cache.py` (`compute_or_load_interleaved_lengths`)
- [ ] Add `scripts/manifest_length_cache.py` CLI
- [ ] Sidecar: `*.lengths.npz` + `*.lengths.meta.json` next to pretok manifest
- [ ] Invalidate on manifest sha256 / n_rows / seed mismatch (atomic write)
- [ ] Tests: `tests/test_length_cache.py`

**Sampler + Trainer**
- [ ] Add `data/length_grouped_sampler.py` (`ManifestLengthGroupedSampler`)
- [ ] Extend `DataTrainingArguments`: `batch_packing_mode`, `length_group_mega_batch_mult`
- [ ] Override `PerceiverDenoiseTrainer._get_train_sampler` (no DistributedSampler nest)
- [ ] Gate: pretok manifest required; reject IterableDataset; reject HF `--group_by_length` combo
- [ ] Pass `--batch_packing_mode` from `scripts/train_concept_pretraining_multigpu.sh`
      via `BATCH_PACKING_MODE` (default `none`)
- [ ] Leave `DataCollatorForCausalLM` unchanged
- [ ] Tests: `tests/test_length_grouped_sampler.py`, `tests/test_length_grouped_sampler_ddp.py`,
      `tests/test_batch_packing_mode_gates.py`

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
3. DDP: custom sampler owns rank split — **never** wrap with `DistributedSampler`.
4. Interleaved pretok is already map-style — no IterableDataset conflict today; still fail closed if streaming appears.
5. Default `BATCH_PACKING_MODE=none` until Polonez smoke clears success criteria.
