import os
import torch
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import DataCollatorForWholeWordMask
from transformers.utils import logging


logger = logging.get_logger(__name__)


def _make_tokenize_fn(tokenizer, max_seq_length, append_eos_token_id):
    """Build the batched tokenize function shared by the single-source and mix loaders.

    append_eos_token_id is None  -> legacy pad-to-max_length path (perceiver MLM etc.).
    append_eos_token_id is set    -> variable-length rows with one EOS appended (AR / E05),
                                     padding deferred to the data collator.
    """
    def tokenize_batch_function(examples):
        text_batch = examples["text"]
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
# E05 — multi-dataset mixes for long-context training
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
# E05 "long_2k" rationale (1k-row seqlen sample, SmolLM2-135M tokenizer):
#   FinePDFs  34.2% docs > 2k  -> the long-range backbone (real coherent documents)
#   FineWeb-Edu 8.6% > 2k      -> quality web + continuity with the E01-E04 baseline
#   FineMath-3+ 14.7% > 2k     -> coherent math/reasoning structure
# Short docs are NOT packed (no fake long-range signal); they simply don't exercise the
# window difference. Cross-window dependencies come from the long tail (FinePDFs-dominant).
DATASET_MIXES = {
    "e05_long_2k": [
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
    if data_files:
        ds = load_dataset("parquet", data_files=data_files, split=split, cache_dir=cache_dir)
    else:
        subset = spec.get("subset") or None
        ds = load_dataset(spec["hf_id"], subset, split=split, cache_dir=cache_dir)
    return _normalize_to_text_column(ds, spec.get("text_columns"))


def load_and_preprocess_dataset_mix(
    tokenizer,
    mix,
    test_size_percent=0.1,
    max_seq_length=2048,
    dataset_cache_dir=None,
    train_num_proc=8,
    test_num_proc=4,
    append_eos_token_id=None,
    split_seed=42,
    interleave_seed=42,
):
    """Load + tokenize a weighted mix of text datasets for long-context training (E05).

    `mix` is either a registered name in DATASET_MIXES or a list of source specs.
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

    sources = DATASET_MIXES[mix] if isinstance(mix, str) else list(mix)
    if not sources:
        raise ValueError(f"Empty dataset mix: {mix!r}")

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
    train_ds = interleave_datasets(
        train_parts,
        probabilities=probabilities,
        seed=interleave_seed,
        stopping_strategy="all_exhausted",
    )
    test_ds = concatenate_datasets(eval_parts)
    logger.info(
        f"[mix] interleaved train={len(train_ds):,} (probs={[round(p, 3) for p in probabilities]}) "
        f"| eval={len(test_ds):,}"
    )
    return train_ds, test_ds


if __name__ == "__main__":
    pass