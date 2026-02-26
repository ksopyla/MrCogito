# %%
"""
This is script that analyse various datasets that could be used to train MLM and endoder decoder chat model.
We will load, view the dataset structure and fields, compute some dataset statistics like: number of rows, min-max row lenght,

All datasets will be downloaded rom HuggingFace hub

BookCorpus (bookcorpus/bookcorpus): Small (~1GB full), clean narrative text - https://huggingface.co/datasets/bookcorpus/bookcorpus

WikiMedia (wikimedia/wikipedia): Wikipedia articles  - https://huggingface.co/datasets/wikimedia/wikipedia
WikiText (Salesforce/wikitext): Preprocessed Wikipedia - https://huggingface.co/datasets/Salesforce/wikitext

For future reference:

* https://huggingface.co/datasets/allenai/olmo-mix-1124 -
* https://huggingface.co/datasets/allenai/dolmino-mix-1124



"""

# imports
from datasets import load_dataset, disable_caching, enable_caching, get_dataset_config_names, load_dataset_builder
from rich import print
from rich.table import Table
from rich.console import Console

# from rich.progress import track
import numpy as np
import os

# cache folder for all downloaded datasets, /Datasets
# set the absolute path to this repo, the structure is:
#  /Datasets
#  /docs
#  /playground

DATASET_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets")
)
MODEL_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Cache", "Models")
)
TOKENIZER_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers")
)
print(DATASET_CACHE_DIR)


# %% [Analysis Helper Function]
def analyze_dataset(dataset, name="Dataset", batched=False):
    """Compute text length statistics for a dataset. In order to visualize them we need to use the display_stats function.

    Note: If batched is set to True, the dataset will be processed in batches of 100 examples. This is more memory efficient than processing the entire dataset at once.
    On Windows batched multiprocessing should be run in the main function
    if __name__ == "__main__":
    """

    print(f"\n[bold magenta]Analyzing {name}:[/bold magenta]")

    enable_caching()  # Enable disk caching for map operations
    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)  # Ensure cache directory exists

    # Batch processing optimization
    if batched:
        dataset_lengths = dataset.map(
            lambda examples: {"text_length": [len(text) for text in examples["text"]]},
            batched=True,  # Process examples in batches
            batch_size=1000,  # Adjust based on available memory
            num_proc=4,  # Use 4 parallel processes
            desc="Calculating text lengths",
            cache_file_name=os.path.join(
                DATASET_CACHE_DIR, f"{name}_text_length.arrow"
            ),
        )
    else:
        dataset_lengths = dataset.map(
            lambda example: {"text_length": len(example["text"])},
            batched=False,
            desc="Calculating text lengths",
            # cache_file_name=os.path.join(DATASET_CACHE_DIR, f"{name}_text_length.arrow"),
            num_proc=1,
        )

    # Direct array access from the dataset
    lengths = np.array(dataset_lengths["text_length"])

    # count the number of examples within the ranges
    # Define bin edges (inclusive lower, exclusive upper)
    bins = [
        0,
        100,
        200,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        48000,
        64000,
        128000,
        1024000,
    ]

    # Use vectorized operations to count in bins
    bin_indices = np.digitize(lengths, bins, right=False)
    counts = np.bincount(bin_indices, minlength=len(bins) + 1)

    # Create dictionary of counts with range labels
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)] + [
        f"{bins[-1]}+"
    ]

    # Skip counts[0] which corresponds to x < bins[0]
    range_counts = {
        label: count
        for label, count in zip(bin_labels, counts[1 : len(bin_labels) + 1])
    }

    # Calculate statistics
    stats = {
        "total_examples": len(lengths),
        "lengths": lengths,
        "total_chars": lengths.sum(),
        "min_length": lengths.min(),
        "max_length": lengths.max(),
        "mean_length": lengths.mean(),
        "std_length": lengths.std(),
        "percentiles": np.percentile(lengths, [25, 50, 75, 90, 95, 99]),
        "range_counts": range_counts,
    }

    return stats


def display_stats(stats, name="Dataset"):
    """Visualize the text length statistics for a dataset. Shows the histogram of the text lengths and prints the basic statistics, based on the stats dictionary returned by the analyze_dataset function."""

    lengths = np.array(stats["lengths"])
    # Print statistics
    print(f"[bold]Basic Statistics:[/bold]")
    print(f"Examples: {stats['total_examples']:,}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Length range: {stats['min_length']} - {stats['max_length']}")
    print(f"Mean ± std: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")

    print("[bold]Range counts:[/bold]")
    for label, count in stats["range_counts"].items():
        print(f"{label}: {count:,}")

    print("[bold]Percentiles:[/bold]")
    print(
        f"25th: {stats['percentiles'][0]:.0f} | 50th: {stats['percentiles'][1]:.0f} | "
        f"75th: {stats['percentiles'][2]:.0f} | 90th: {stats['percentiles'][3]:.0f} | "
        f"95th: {stats['percentiles'][4]:.0f} | 99th: {stats['percentiles'][5]:.0f}"
    )

    # Plot histogram
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=1000, log=False)
    plt.title(f"Text Length Distribution - {name} subset {subset_size}")
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Count (log scale)")
    plt.show()


def get_dataset_metadata(dataset_name, config_name=None):
    """
    Get metadata about a dataset without downloading it.

    Args:
        dataset_name (str): The name of the dataset on HuggingFace Hub

    Returns:
        dict: A dictionary containing dataset metadata (configs, info, splits, features)
    """
    # Get available configs (if any)
    configs = get_dataset_config_names(dataset_name, cache_dir=DATASET_CACHE_DIR)
    print("Configs:", configs)

    # Load the dataset builder (does NOT download the data)
    builder = load_dataset_builder(
        dataset_name,
        name=config_name,
        cache_dir=DATASET_CACHE_DIR,
    )
    print("Dataset info:", builder.info)

    # Access size, features, splits, etc.
    print("Splits:", builder.info.splits)
    print("Features:", builder.info.features)

    # Print splits summary as a rich table
    splits = builder.info.splits
    if splits:
        table = Table(title=f"[bold magenta]Dataset Splits Summary for {dataset_name}[/bold magenta]", show_lines=True)
        table.add_column("Split", style="cyan", justify="left")
        table.add_column("Num Examples", style="green", justify="right")
        table.add_column("Size (GB)", style="yellow", justify="right")
        for split_name, split_info in splits.items():
            num_examples = f"{split_info.num_examples:,}" if hasattr(split_info, 'num_examples') else "-"
            num_bytes = split_info.num_bytes if hasattr(split_info, 'num_bytes') else 0
            size_gb = num_bytes / (1024 ** 3)
            table.add_row(
                str(split_name),
                str(num_examples),
                f"{size_gb:.3f}"
            )
        console = Console()
        console.print(table)
    else:
        print("[yellow]No split information available.[/yellow]")

    return {
        "configs": configs,
        "info": builder.info,
        "splits": builder.info.splits,
        "features": builder.info.features,
    }


# ---- Place the meta summary function here ----
def print_datasets_meta_summary(dataset_names):
    """
    Print a meta-summary table for a list of datasets.
    Args:
        dataset_names (list of str): List of dataset names (str)
    """
    from rich.table import Table
    from rich.console import Console

    table = Table(title="[bold magenta]Meta Summary of Datasets[/bold magenta]", show_lines=True)
    table.add_column("Dataset Name", style="cyan", justify="left")
    table.add_column("Config Name", style="green", justify="left")
    table.add_column("Splits", style="yellow", justify="left")

    for dataset_name in dataset_names:
        config_names = get_dataset_config_names(dataset_name, cache_dir=DATASET_CACHE_DIR)
        # If no configs, still show the dataset with config_name=None
        if not config_names:
            config_names = [None]
        for config_name in config_names:
            meta = get_dataset_metadata(dataset_name, config_name)
            splits = meta["splits"]
            split_names = ", ".join(splits.keys()) if splits else "-"
            table.add_row(
                dataset_name,
                config_name if config_name else "-",
                split_names
            )
    console = Console()
    console.print(table)



# %% load the bookcorpus dataset

bookcorpus_metadata = get_dataset_metadata("bookcorpus/bookcorpus")

# %%


bookcorpus = load_dataset(
    "bookcorpus/bookcorpus", cache_dir=DATASET_CACHE_DIR, trust_remote_code=True
)

# get the train split
bookcorpus_train = bookcorpus["train"]

# %% run the bookcorpus analysis - full train set contains 74,004,228 rows
subset_size = 50000000  # bookcorpus_train.num_rows
bookcorpus_train = bookcorpus_train.select(range(subset_size))

stats = analyze_dataset(bookcorpus_train, f"BookCorpus_{subset_size}")

display_stats(stats, f"BookCorpus_{subset_size}")


print("BookCorpus Analysis Complete")


# %% salesforce/wikitext-103-v1

wikitext_metadata = get_dataset_metadata("Salesforce/wikitext", "wikitext-103-v1")

# %% Salesforce/wikitext-103-v1 - 1.8M rows, clean text
wikitext = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-v1",
    cache_dir=DATASET_CACHE_DIR,
)

# %% get the train split
wikitext_train = wikitext["train"]

# %% wikitext_train.num_rows = 1.8M

subset_size = wikitext_train.num_rows

# wikitext_train = wikitext_train.select(range(subset_size))

stats = analyze_dataset(wikitext_train, f"WikiText_{subset_size}")

display_stats(stats, f"WikiText_{subset_size}")

# %% [Wikipedia Analysis]
# Updated to use 20231101.en subset for English Wikipedia
wikipedia = load_dataset(
    "wikimedia/wikipedia", "20231101.en", cache_dir=DATASET_CACHE_DIR
)


# %%
wikipedia_train = wikipedia["train"]

# %% get the train split
subset_size = wikipedia_train.num_rows

# wikipedia_train_subset = wikipedia_train.select(range(subset_size))

stats = analyze_dataset(wikipedia_train, f"Wikipedia_{subset_size}")

display_stats(stats, f"Wikipedia_{subset_size}")


#%% MiniPile dataset metadata
minipile_metadata = get_dataset_metadata("JeanKaddour/minipile")

# %% JeanKaddour/minipile

minipile = load_dataset("JeanKaddour/minipile", cache_dir=DATASET_CACHE_DIR)
minipile_train = minipile["train"]

subset_size = minipile_train.num_rows

stats = analyze_dataset(minipile_train, f"MiniPile_{subset_size}")

display_stats(stats, f"MiniPile_{subset_size}")



# %% load the allenai/olmo-mix-1124 dataset

from datasets import get_dataset_config_names, load_dataset_builder

# For example, for dolmino-mix-1124
dataset_name = "allenai/dolmino-mix-1124"

# Get metadata for the dataset
dolmino_mix_metadata = get_dataset_metadata(dataset_name, "wiki")

# %% load the allenai/olmo-mix-1124 dataset

olmo_mix_metadata = get_dataset_metadata("allenai/olmo-mix-1124")

#%% load dolma 

dolma_metadata = get_dataset_metadata("allenai/dolma", 'v1_7')
# %% load the allenai/dolmino-mix-1124 dataset


olmo_mix = load_dataset("allenai/olmo-mix-1124", cache_dir=DATASET_CACHE_DIR)


# dolmino_mix = load_dataset(
#     "allenai/dolmino-mix-1124",
#     cache_dir=DATASET_CACHE_DIR,
# )


#%%
# Example usage (run as a cell):
datasets_to_summarize = [
    "bookcorpus/bookcorpus",
    "Salesforce/wikitext",
    "wikimedia/wikipedia",
    "allenai/olmo-mix-1124",
    "allenai/dolmino-mix-1124",
]
print_datasets_meta_summary(datasets_to_summarize)
# %%
