import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import DataCollatorForWholeWordMask
from transformers.utils import logging


logger = logging.get_logger(__name__)
MIX_RECIPES_DIR = Path(__file__).resolve().parent / "mix_recipes"


def _fast_weighted_all_exhausted_interleave(datasets, probabilities, seed):
    """Vectorized equivalent of HF's weighted map-style ``all_exhausted`` interleave.

    ``datasets.interleave_datasets`` generates one source choice at a time in Python. At
    ~10M rows that can take hours before training even starts. HF draws choices in chunks
    of 1,000 from ``np.random.default_rng(seed)``; process each chunk source-by-source while
    preserving the same cyclic row indices and exact stop position.
    """
    if not datasets:
        raise ValueError("Cannot interleave an empty dataset list.")
    lengths = np.asarray([len(dataset) for dataset in datasets], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("Weighted interleave requires every source dataset to be non-empty.")

    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
    current = np.zeros(len(datasets), dtype=np.int64)
    exhausted = np.zeros(len(datasets), dtype=bool)
    rng = np.random.default_rng(seed)
    index_chunks = []

    while not exhausted.all():
        choices = rng.choice(len(datasets), size=1000, p=probabilities)

        # HF stops immediately after the last never-exhausted source reaches its end.
        remaining = np.flatnonzero(~exhausted)
        final_positions = []
        for source_idx in remaining:
            positions = np.flatnonzero(choices == source_idx)
            needed = lengths[source_idx] - current[source_idx]
            if len(positions) < needed:
                break
            final_positions.append(positions[needed - 1])
        else:
            choices = choices[: max(final_positions) + 1]

        global_indices = np.empty(len(choices), dtype=np.int64)
        for source_idx in range(len(datasets)):
            positions = np.flatnonzero(choices == source_idx)
            if not len(positions):
                continue
            source_rows = current[source_idx] + np.arange(len(positions), dtype=np.int64)
            global_indices[positions] = offsets[source_idx] + source_rows % lengths[source_idx]
            if source_rows[-1] >= lengths[source_idx] - 1:
                exhausted[source_idx] = True
            current[source_idx] = (source_rows[-1] + 1) % lengths[source_idx]
        index_chunks.append(global_indices)

    combined = concatenate_datasets(datasets)
    return combined.select(np.concatenate(index_chunks))


def configure_text_tokenizer_for_model_vocab(tokenizer, model_vocab_size: int) -> bool:
    """Prevent tokenizer-only multimodal tokens from exceeding text-model embeddings.

    Gemma-3's tokenizer includes ``<image_soft_token>`` at id 262144 while
    ``gemma-3-1b-pt`` has text embeddings for ids 0..262143. A literal occurrence in web
    text would otherwise become an invalid embedding index. When the tokenizer is larger
    than the model, split literal special-token strings into ordinary text.
    """
    if len(tokenizer) <= model_vocab_size:
        return False
    if not hasattr(tokenizer, "split_special_tokens"):
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} ids but model supports {model_vocab_size}, "
            "and this tokenizer cannot split out-of-range special tokens."
        )
    tokenizer.split_special_tokens = True
    return True


def _make_tokenize_fn(tokenizer, max_seq_length, append_eos_token_id, max_chars=None):
    """Build the batched tokenize function shared by the single-source and mix loaders.

    append_eos_token_id is None  -> legacy pad-to-max_length path (perceiver MLM etc.).
    append_eos_token_id is set    -> variable-length rows with one EOS appended (AR / long-context),
                                     padding deferred to the data collator.
    max_chars -> if set, truncate each raw text to this many characters BEFORE tokenizing.
        Guards against gigantic web/PDF docs OOM-ing or crashing a tokenize worker: the Fast
        tokenizer scans the *whole* input string even though ``truncation=max_seq_length``
        discards all but ~8k chars, so a 100MB DCLM web page can kill a ``num_proc`` worker
        before truncation runs. Lossless for the kept tokens when ``max_chars`` >> max_seq_length.
        Default ``None`` preserves the original behaviour (no pre-truncation).
    """
    def tokenize_batch_function(examples):
        text_batch = examples["text"]
        if max_chars:
            text_batch = [t[:max_chars] if t and len(t) > max_chars else t for t in text_batch]
        if append_eos_token_id is None:
            return tokenizer(
                text_batch,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )
        out = tokenizer(
            text_batch,
            padding=False,
            truncation=True,
            max_length=max_seq_length - 1,
            return_special_tokens_mask=True,
        )
        out["input_ids"] = [ids + [append_eos_token_id] for ids in out["input_ids"]]
        if "attention_mask" in out:
            out["attention_mask"] = [m + [1] for m in out["attention_mask"]]
        if "special_tokens_mask" in out:
            out["special_tokens_mask"] = [s + [1] for s in out["special_tokens_mask"]]
        return out

    return tokenize_batch_function



def _select_train_eval_splits(dataset, test_size_percent, seed=42):
    if "train" not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(
            f"Dataset must expose a 'train' split. Available splits: {available}"
        )

    train_ds = dataset["train"]

    if "validation" in dataset:
        logger.info("Using built-in validation split for evaluation.")
        return train_ds, dataset["validation"]

    if "test" in dataset:
        logger.info("Using built-in test split for evaluation.")
        return train_ds, dataset["test"]

    if len(train_ds) < 2:
        raise ValueError(
            "Dataset must contain at least 2 training examples when no built-in "
            "validation/test split is available."
        )

    test_size = max(1, min(int(len(train_ds) * test_size_percent), len(train_ds) - 1, 100000))
    logger.info(
        "Dataset has no validation/test split; creating holdout split from train "
        f"(size={test_size}, seed={seed})."
    )
    # A fixed seed makes the split deterministic: the resulting train_ds keeps a
    # stable fingerprint, so the downstream tokenization .map() cache is reused
    # across runs (no full re-tokenization on every launch) and every DDP rank
    # sees the SAME train/eval split.
    split_ds = train_ds.train_test_split(test_size=test_size, seed=seed)
    return split_ds["train"], split_ds["test"]


def _ensure_text_column(dataset_split, text_column_name, split_name):
    if "text" in dataset_split.column_names:
        return dataset_split

    if text_column_name not in dataset_split.column_names:
        available = ", ".join(dataset_split.column_names)
        raise ValueError(
            f"Split '{split_name}' does not contain the requested text column "
            f"'{text_column_name}'. Available columns: {available}"
        )

    logger.info(
        f"Renaming column '{text_column_name}' -> 'text' for split '{split_name}'."
    )
    return dataset_split.rename_column(text_column_name, "text")


def load_and_preprocess_text_dataset(
    tokenizer,
    dataset_hf_path,
    dataset_name_subset,
    text_column_name,
    test_size_percent=0.1,
    max_seq_length=512,
    dataset_cache_dir=None,
    train_num_proc=8,
    test_num_proc=4,
    append_eos_token_id=None,
    split_seed=42,
):
    """
    Loads and preprocesses the text dataset that fits to memory.
    
    * BookCorpus (bookcorpus/bookcorpus): Small (~1GB full), clean narrative text - https://huggingface.co/datasets/bookcorpus/bookcorpus
    * WikiMedia (wikimedia/wikipedia): Wikipedia articles with math/science concepts - https://huggingface.co/datasets/wikimedia/wikipedia
    * WikiText (Salesforce/wikitext): Preprocessed Wikipedia with math/science concepts - https://huggingface.co/datasets/Salesforce/wikitext
    
    Args:
        dataset_cache_dir: Optional path to cache directory. If None, uses ./Cache/Datasets relative to this file.
        split_seed: Seed for the train/holdout split when the dataset has no built-in
            validation/test split. Fixing it keeps the tokenization cache reusable and
            the split identical across runs and DDP ranks.
    """
    if dataset_cache_dir is None:
        DATASET_CACHE_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets")
        )
    else:
        DATASET_CACHE_DIR = os.path.abspath(dataset_cache_dir)

    if dataset_name_subset == "":
        dataset_name_subset = None

    # Load dataset - remove trust_remote_code=True as it's no longer supported/needed for most datasets
    # Minipile is a standard dataset, doesn't need it
    logger.info(
        f"Loading dataset '{dataset_hf_path}'"
        + (f" subset '{dataset_name_subset}'" if dataset_name_subset else "")
        + f" with cache_dir='{DATASET_CACHE_DIR}'."
    )
    dataset = load_dataset(dataset_hf_path, dataset_name_subset, cache_dir=DATASET_CACHE_DIR)

    train_ds, test_ds = _select_train_eval_splits(dataset, test_size_percent, seed=split_seed)
  
    # Rename column to match processing
    # do a collumn rename based on the mapping provided below
    #check if the text_column_name is in the dataset
    train_ds = _ensure_text_column(train_ds, text_column_name, "train")
    test_ds = _ensure_text_column(test_ds, text_column_name, "eval")


    # Tokenization function (shared with the dataset-mix loader)
    tokenize_batch_function = _make_tokenize_fn(tokenizer, max_seq_length, append_eos_token_id)

    # Process train dataset
    # Disable multiprocessing to avoid OOM or reduce num_proc significantly
    # Using a smaller number of processes (e.g., 4 or 8) is usually safe
    train_num_proc = max(1, min(train_num_proc, len(train_ds)))
    train_ds = train_ds.map(
        tokenize_batch_function,
        batched=True,
        num_proc=train_num_proc, # os.cpu_count()-2 can be too high (62 processes!) causing OOM
        remove_columns=["text"]
    )

    
    # Process test dataset
    test_num_proc = max(1, min(test_num_proc, len(test_ds)))
    test_ds = test_ds.map(
        tokenize_batch_function,
        batched=True,
        num_proc=test_num_proc, # Lower for test set
        remove_columns=["text"]
    )
    
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Multi-dataset mixes for long-context pretraining
# ---------------------------------------------------------------------------
# A "mix" is a list of source specs interleaved by sampling probability (weight).
# Each spec mirrors analysis/long_dataset_candidates.json so the mixes stay in sync
# with the measured sequence-length catalog. Fields:
#   hf_id        HF dataset id (ignored when data_files is given; use "parquet" loader)
#   subset       config/subset name (or None)
#   split        split name (default "train")
#   data_files   optional list of parquet URLs (FinePDFs-style refs/convert branch)
#   text_columns single/multiple columns concatenated with "\n\n" into a "text" column
#   weight       interleave sampling probability (normalised across the mix)
#   max_samples  per-source row cap (downloads only train[:N]) to bound disk/compute
#
# long_2k_base_v1 rationale (1k-row seqlen sample, SmolLM2-135M tokenizer):
#   FinePDFs  34.2% docs > 2k  -> the long-range backbone (real coherent documents)
#   FineWeb-Edu 8.6% > 2k      -> quality web + continuity with the E01-E04 baseline
#   FineMath-3+ 14.7% > 2k     -> coherent math/reasoning structure
# Short docs are NOT packed (no fake long-range signal); they simply don't exercise the
# window difference. Cross-window dependencies come from the long tail (FinePDFs-dominant).
DATASET_MIXES = {
    "long_2k_base_v1": [
        {
            "name": "finepdfs_100BT",
            "hf_id": "HuggingFaceFW/finepdfs_100BT",
            "subset": "default",
            "split": "train",
            "data_files": [
                f"https://huggingface.co/datasets/HuggingFaceFW/finepdfs_100BT/"
                f"resolve/refs%2Fconvert%2Fparquet/default/train/{shard:04d}.parquet"
                for shard in range(8)
            ],
            "text_columns": ["text"],
            "weight": 0.5,
            "max_samples": 2_000_000,
        },
        {
            "name": "fineweb_edu",
            "hf_id": "HuggingFaceFW/fineweb-edu",
            "subset": "sample-10BT",
            "split": "train",
            "text_columns": ["text"],
            "weight": 0.3,
            "max_samples": 2_000_000,
        },
        {
            "name": "finemath_3plus",
            "hf_id": "HuggingFaceTB/finemath",
            "subset": "finemath-3plus",
            "split": "train",
            "text_columns": ["text"],
            "weight": 0.2,
            "max_samples": 1_500_000,
        },
    ],
}


def _normalize_mix_source_spec(spec: dict[str, Any], *, source_idx: int, mix_label: str) -> dict[str, Any]:
    """Normalize one source spec into the internal loader contract.

    Supports both `hf_id` and `dataset` keys for HF compatibility.
    Supports either `text_columns` or a single `text_column`.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"{mix_label}: source #{source_idx} must be an object, got {type(spec)!r}")

    out = dict(spec)

    if "hf_id" not in out and "dataset" in out:
        out["hf_id"] = out["dataset"]

    if "text_columns" not in out and out.get("text_column"):
        out["text_columns"] = [out["text_column"]]

    text_columns = out.get("text_columns")
    if text_columns is None:
        out["text_columns"] = ["text"]
    elif isinstance(text_columns, str):
        out["text_columns"] = [text_columns]
    elif isinstance(text_columns, list):
        out["text_columns"] = [str(c) for c in text_columns if c]
        if not out["text_columns"]:
            raise ValueError(f"{mix_label}: source #{source_idx} has empty text_columns")
    else:
        raise ValueError(
            f"{mix_label}: source #{source_idx} has invalid text_columns type: {type(text_columns)!r}"
        )

    data_files = out.get("data_files")
    if data_files is not None:
        if isinstance(data_files, str):
            data_files = [data_files]
        if not isinstance(data_files, list):
            raise ValueError(f"{mix_label}: source #{source_idx} data_files must be string or list")
        data_files = [str(p) for p in data_files if p]
        if not data_files:
            raise ValueError(f"{mix_label}: source #{source_idx} has empty data_files")
        out["data_files"] = data_files

    if not out.get("hf_id") and not out.get("data_files"):
        raise ValueError(
            f"{mix_label}: source #{source_idx} must define either hf_id/dataset or data_files"
        )

    subset = out.get("subset")
    out["subset"] = None if subset in ("", None) else subset
    out["split"] = out.get("split", "train")

    out["weight"] = float(out.get("weight", 1.0))
    if out["weight"] <= 0:
        raise ValueError(f"{mix_label}: source #{source_idx} has non-positive weight={out['weight']}")

    if out.get("max_samples") is not None:
        out["max_samples"] = int(out["max_samples"])
        if out["max_samples"] <= 0:
            raise ValueError(
                f"{mix_label}: source #{source_idx} has non-positive max_samples={out['max_samples']}"
            )

    return out


def _resolve_mix_recipe_path(mix_recipe: str) -> Path:
    """Resolve a mix recipe path or short id.

    Accepted forms:
      - absolute/relative path to a JSON file
      - short id, resolved in data/mix_recipes/<id>.json
    """
    candidate = Path(mix_recipe).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

    stem = mix_recipe[:-5] if mix_recipe.endswith(".json") else mix_recipe
    packaged = MIX_RECIPES_DIR / f"{stem}.json"
    if packaged.exists():
        return packaged.resolve()

    raise ValueError(
        f"Could not resolve dataset mix recipe '{mix_recipe}'. "
        f"Checked direct path, cwd-relative path, and {MIX_RECIPES_DIR / (stem + '.json')}."
    )


def load_mix_recipe(mix_recipe: str) -> dict[str, Any]:
    """Load a JSON recipe with top-level metadata + sources list.

    Backward compatible with "list only" JSON (treated as top-level sources array).
    """
    recipe_path = _resolve_mix_recipe_path(mix_recipe)
    with recipe_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        payload = {"mix_id": recipe_path.stem, "sources": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"Mix recipe at {recipe_path} must be an object or list.")
    if "sources" not in payload:
        raise ValueError(f"Mix recipe at {recipe_path} is missing required top-level key 'sources'.")
    if not isinstance(payload["sources"], list) or not payload["sources"]:
        raise ValueError(f"Mix recipe at {recipe_path} must provide a non-empty 'sources' list.")

    mix_id = payload.get("mix_id", recipe_path.stem)
    sources = [
        _normalize_mix_source_spec(spec, source_idx=i, mix_label=f"mix_recipe:{mix_id}")
        for i, spec in enumerate(payload["sources"])
    ]

    recipe = dict(payload)
    recipe["mix_id"] = mix_id
    recipe["sources"] = sources
    recipe["_recipe_path"] = str(recipe_path)
    return recipe


def _resolve_mix_sources(mix: str | list[dict[str, Any]] | dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve mix input into normalized source specs + metadata.

    Supports:
      - registered DATASET_MIXES key
      - recipe id/path string
      - inline list of source specs
      - dict with top-level "sources"
    """
    if isinstance(mix, str):
        if mix in DATASET_MIXES:
            sources = [
                _normalize_mix_source_spec(spec, source_idx=i, mix_label=f"mix:{mix}")
                for i, spec in enumerate(DATASET_MIXES[mix])
            ]
            return sources, {"mix_origin": "registry", "mix_id": mix}

        recipe = load_mix_recipe(mix)
        return recipe["sources"], {
            "mix_origin": "recipe",
            "mix_id": recipe["mix_id"],
            "mix_recipe_path": recipe["_recipe_path"],
        }

    if isinstance(mix, list):
        sources = [
            _normalize_mix_source_spec(spec, source_idx=i, mix_label="mix:inline")
            for i, spec in enumerate(mix)
        ]
        return sources, {"mix_origin": "inline", "mix_id": "inline_mix"}

    if isinstance(mix, dict) and "sources" in mix:
        mix_id = mix.get("mix_id", "inline_mix")
        sources = [
            _normalize_mix_source_spec(spec, source_idx=i, mix_label=f"mix:{mix_id}")
            for i, spec in enumerate(mix["sources"])
        ]
        return sources, {"mix_origin": "inline_recipe", "mix_id": mix_id}

    raise ValueError(
        "mix must be a registered mix name, recipe path/id, source list, "
        "or dict with top-level 'sources'."
    )


def resolve_mix_sources(mix: str | list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Public helper for tooling/tests: resolve any supported mix input into source specs."""
    sources, _ = _resolve_mix_sources(mix)
    return sources


def _parse_mix_weight_override(mix_weight_override: str | dict[str, float] | None) -> dict[str, float]:
    if mix_weight_override is None:
        return {}
    if isinstance(mix_weight_override, dict):
        override = mix_weight_override
    elif isinstance(mix_weight_override, str):
        raw = mix_weight_override.strip()
        if not raw:
            return {}
        override = json.loads(raw)
    else:
        raise ValueError(
            f"mix_weight_override must be dict/JSON string/None, got {type(mix_weight_override)!r}"
        )
    if not isinstance(override, dict):
        raise ValueError("mix_weight_override must decode to an object mapping source->weight.")
    parsed = {str(k): float(v) for k, v in override.items()}
    for key, weight in parsed.items():
        if weight <= 0:
            raise ValueError(f"mix_weight_override has non-positive weight for '{key}': {weight}")
    return parsed


def _apply_mix_weight_override(
    sources: list[dict[str, Any]],
    mix_weight_override: str | dict[str, float] | None,
) -> list[dict[str, Any]]:
    override = _parse_mix_weight_override(mix_weight_override)
    if not override:
        return [dict(spec) for spec in sources]

    updated = [dict(spec) for spec in sources]
    used_keys = set()
    available_keys = set()

    for spec in updated:
        candidate_keys = []
        if spec.get("name"):
            candidate_keys.append(str(spec["name"]))
        if spec.get("hf_id"):
            candidate_keys.append(str(spec["hf_id"]))
        available_keys.update(candidate_keys)
        for key in candidate_keys:
            if key in override:
                spec["weight"] = float(override[key])
                used_keys.add(key)
                break

    unknown = set(override.keys()) - used_keys
    if unknown:
        raise ValueError(
            f"mix_weight_override contains unknown source key(s): {sorted(unknown)}. "
            f"Available source keys: {sorted(available_keys)}"
        )
    return updated


def apply_mix_weight_override(
    sources: list[dict[str, Any]],
    mix_weight_override: str | dict[str, float] | None,
) -> list[dict[str, Any]]:
    """Public helper for tooling/tests: apply runtime weight overrides to source specs."""
    return _apply_mix_weight_override(sources, mix_weight_override)


def _stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if content is not None:
            role = value.get("role")
            return (f"{role}: " if role else "") + _stringify(content)
        return str(value)
    if isinstance(value, list):
        return "\n".join(_stringify(v) for v in value if v is not None)
    return str(value)


def _normalize_to_text_column(ds, text_columns):
    """Collapse the requested text column(s) into a single 'text' column, dropping the rest."""
    cols = [c for c in (text_columns or ["text"]) if c in ds.column_names]
    if not cols:
        raise ValueError(
            f"None of text_columns={text_columns} present. Available: {ds.column_names}"
        )
    if cols == ["text"]:
        return ds.select_columns(["text"])

    def _join(example):
        parts = [_stringify(example[c]) for c in cols]
        return {"text": "\n\n".join(p for p in parts if p)}

    return ds.map(_join, remove_columns=ds.column_names)


def _load_mix_source(spec, cache_dir):
    """Load one mix source as a map-style dataset normalised to a 'text' column."""
    split = spec.get("split", "train")
    max_samples = spec.get("max_samples")
    if max_samples:
        split = f"{split}[:{int(max_samples)}]"
    data_files = spec.get("data_files")

    token = spec.get("hf_token")
    token_env = spec.get("hf_token_env")
    if token is None and token_env:
        token = os.environ.get(str(token_env))

    def _load_one(source_hf_id, source_subset, source_data_files):
        common_kwargs = {"split": split, "cache_dir": cache_dir}
        if token is not None:
            common_kwargs["token"] = token
        if spec.get("revision"):
            common_kwargs["revision"] = spec["revision"]

        if source_data_files:
            return load_dataset(
                "parquet",
                data_files=source_data_files,
                **common_kwargs,
            )
        dataset_kwargs = {}
        if spec.get("data_dir"):
            dataset_kwargs["data_dir"] = spec["data_dir"]
        if spec.get("trust_remote_code") is not None:
            dataset_kwargs["trust_remote_code"] = bool(spec["trust_remote_code"])
        return load_dataset(
            source_hf_id,
            source_subset,
            **dataset_kwargs,
            **common_kwargs,
        )

    try:
        ds = _load_one(spec.get("hf_id"), spec.get("subset") or None, data_files)
    except Exception as exc:
        fallback_hf_id = spec.get("fallback_hf_id")
        fallback_data_files = spec.get("fallback_data_files")
        fallback_subset = spec.get("fallback_subset", spec.get("subset"))
        if not fallback_hf_id and not fallback_data_files:
            raise
        source_name = spec.get("name", spec.get("hf_id", "unknown_source"))
        logger.warning(
            f"[mix] source '{source_name}' failed to load ({exc}); falling back to "
            f"{fallback_hf_id or 'parquet:data_files'}"
        )
        ds = _load_one(fallback_hf_id, fallback_subset, fallback_data_files)
    return _normalize_to_text_column(ds, spec.get("text_columns"))


def load_and_preprocess_dataset_mix(
    tokenizer,
    mix,
    mix_weight_override=None,
    test_size_percent=0.1,
    max_seq_length=2048,
    dataset_cache_dir=None,
    train_num_proc=8,
    test_num_proc=4,
    append_eos_token_id=None,
    split_seed=42,
    interleave_seed=42,
):
    """Load + tokenize a weighted mix of text datasets for long-context pretraining.

    `mix` can be:
      - a registered name in DATASET_MIXES
      - a recipe id/path (JSON in data/mix_recipes/)
      - an inline source list
      - an inline recipe dict {"sources":[...]}

    `mix_weight_override` optionally overrides source weights at runtime (dict or JSON string)
    using source `name` (preferred) or `hf_id` keys.

    Per source: load (capped), normalise to 'text', hold out a small eval slice, tokenize.
    Train parts are interleaved by normalised weight (stopping_strategy='all_exhausted');
    eval parts are concatenated into one representative multi-source holdout.

    Returns (train_ds, test_ds) — map-style, the same contract the Trainer expects.
    """
    if dataset_cache_dir is None:
        cache_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets")
        )
    else:
        cache_dir = os.path.abspath(dataset_cache_dir)

    sources, mix_meta = _resolve_mix_sources(mix)
    sources = _apply_mix_weight_override(sources, mix_weight_override)
    if not sources:
        raise ValueError(f"Empty dataset mix: {mix!r}")
    logger.info(
        f"[mix] resolved '{mix_meta.get('mix_id')}' from {mix_meta.get('mix_origin')}"
        + (
            f" ({mix_meta.get('mix_recipe_path')})"
            if mix_meta.get("mix_recipe_path")
            else ""
        )
    )

    tokenize_fn = _make_tokenize_fn(tokenizer, max_seq_length, append_eos_token_id)

    train_parts, eval_parts, weights = [], [], []
    for spec in sources:
        name = spec.get("name", spec.get("hf_id", "?"))
        logger.info(f"[mix] loading source '{name}' (weight={spec.get('weight')})")
        ds = _load_mix_source(spec, cache_dir)
        n = len(ds)
        eval_size = max(1, min(int(n * test_size_percent), n - 1, 5000))
        split_ds = ds.train_test_split(test_size=eval_size, seed=split_seed)
        src_train, src_eval = split_ds["train"], split_ds["test"]

        ntr = max(1, min(train_num_proc, len(src_train)))
        nte = max(1, min(test_num_proc, len(src_eval)))
        train_parts.append(src_train.map(tokenize_fn, batched=True, num_proc=ntr, remove_columns=["text"]))
        eval_parts.append(src_eval.map(tokenize_fn, batched=True, num_proc=nte, remove_columns=["text"]))
        weights.append(float(spec.get("weight", 1.0)))
        logger.info(f"[mix]   '{name}': {len(src_train):,} train / {len(src_eval):,} eval rows")

    total_w = sum(weights)
    probabilities = [w / total_w for w in weights]
    train_ds = _fast_weighted_all_exhausted_interleave(
        train_parts, probabilities, interleave_seed
    )
    test_ds = concatenate_datasets(eval_parts)
    logger.info(
        f"[mix] interleaved train={len(train_ds):,} (probs={[round(p, 3) for p in probabilities]}) "
        f"| eval={len(test_ds):,}"
    )
    return train_ds, test_ds


def load_pretokenized_mix(manifest_path):
    """Load a pre-tokenized mix produced by `scripts/pretokenize_mix.py`.

    Reads the manifest JSON, `load_from_disk`s each source's train/eval dirs,
    interleaves train parts by the manifest weights, and concatenates eval parts.
    Instant — no download, no tokenization. Returns (train_ds, test_ds).
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pretokenized manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    logger.info(
        f"[pretokenized] loading mix '{manifest.get('mix_id')}' from {manifest_path}"
        f" (seq={manifest.get('max_seq_length')}, obj={manifest.get('objective')})"
    )

    train_parts, eval_parts, weights = [], [], []
    for src in manifest["sources"]:
        name = src["name"]
        tr = load_from_disk(src["train_path"])
        ev = load_from_disk(src["eval_path"])
        train_parts.append(tr)
        eval_parts.append(ev)
        weights.append(float(src.get("weight", 1.0)))
        logger.info(f"[pretokenized]   '{name}': {len(tr):,} train / {len(ev):,} eval rows")

    total_w = sum(weights)
    probabilities = [w / total_w for w in weights]
    train_ds = _fast_weighted_all_exhausted_interleave(
        train_parts, probabilities, manifest.get("seed", 42)
    )
    test_ds = concatenate_datasets(eval_parts)
    logger.info(
        f"[pretokenized] interleaved train={len(train_ds):,}"
        f" (probs={[round(p, 3) for p in probabilities]}) | eval={len(test_ds):,}"
    )
    return train_ds, test_ds


if __name__ == "__main__":
    pass