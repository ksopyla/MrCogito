import os
import torch
from datasets import load_dataset
from transformers import DataCollatorForWholeWordMask
from transformers.utils import logging


logger = logging.get_logger(__name__)



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


    # Tokenization function
    def tokenize_batch_function(examples):

        text_batch = examples["text"]

        # Default path (unchanged for all existing callers): pad to max_length.
        if append_eos_token_id is None:
            return tokenizer(
                text_batch,  # Note different column name
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True
            )

        # EOS-append path (AR decoder, e.g. SmolLM2 tokenizer that does not add EOS):
        # leave room for one EOS, append it, and defer padding to the data collator.
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

    
if __name__ == "__main__":
    pass