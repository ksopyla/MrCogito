#%%
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

#imports
from datasets import load_dataset, disable_caching, enable_caching
from rich import print
#from rich.progress import track
import numpy as np
import os

# cache folder for all downloaded datasets, /Datasets
# set the absolute path to this repo, the structure is:
#  /Datasets
#  /docs
#  /playground
cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
print(cache_dir)

#%% [Analysis Helper Function]
def analyze_dataset(dataset, name="Dataset", batched=False):
    """Compute text length statistics for a dataset. In order to visualize them we need to use the display_stats function.
    
    Note: If batched is set to True, the dataset will be processed in batches of 100 examples. This is more memory efficient than processing the entire dataset at once.
    On Windows batched multiprocessing should be run in the main function 
    if __name__ == "__main__":
    """
    
    print(f"\n[bold magenta]Analyzing {name}:[/bold magenta]")
    
    enable_caching()  # Enable disk caching for map operations
    os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists
    
    # Batch processing optimization
    if batched:
        dataset_lengths = dataset.map(
            lambda examples: {"text_length": [len(text) for text in examples["text"]]},
            batched=True,  # Process examples in batches
            batch_size=1000,  # Adjust based on available memory
            num_proc=4,  # Use 4 parallel processes
            desc="Calculating text lengths",
            cache_file_name=os.path.join(cache_dir, f"{name}_text_length.arrow")
        ) 
    else:
        dataset_lengths = dataset.map(
            lambda example: {"text_length": len(example["text"])}, 
            batched=False,
            desc="Calculating text lengths",
            #cache_file_name=os.path.join(cache_dir, f"{name}_text_length.arrow"),
            num_proc=1
        )
    
    # Direct array access from the dataset
    lengths = np.array(dataset_lengths["text_length"])
    
    # count the number of examples within the ranges
        # Define bin edges (inclusive lower, exclusive upper)
    bins = [0,100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 48000, 64000, 128000, 1024000]
    
    # Use vectorized operations to count in bins
    bin_indices = np.digitize(lengths, bins, right=False)
    counts = np.bincount(bin_indices, minlength=len(bins)+1)
    
    # Create dictionary of counts with range labels
    bin_labels = [
        f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)
    ] + [f"{bins[-1]}+"]
    
    # Skip counts[0] which corresponds to x < bins[0]
    range_counts = {
        label: count 
        for label, count in zip(bin_labels, counts[1:len(bin_labels) + 1])
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
        "range_counts": range_counts
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
    for label, count in stats['range_counts'].items():
        print(f"{label}: {count:,}")
    

    print("[bold]Percentiles:[/bold]")
    print(f"25th: {stats['percentiles'][0]:.0f} | 50th: {stats['percentiles'][1]:.0f} | "
          f"75th: {stats['percentiles'][2]:.0f} | 90th: {stats['percentiles'][3]:.0f} | "
          f"95th: {stats['percentiles'][4]:.0f} | 99th: {stats['percentiles'][5]:.0f}")
    
    # Plot histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=1000, log=False)
    plt.title(f"Text Length Distribution - {name} subset {subset_size}")
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Count (log scale)")
    plt.show()

#%% load the bookcorpus dataset
bookcorpus = load_dataset("bookcorpus/bookcorpus", 
                         cache_dir=cache_dir, 
                         trust_remote_code=True)

# get the train split
bookcorpus_train = bookcorpus["train"]

#%% run the bookcorpus analysis - full train set contains 74,004,228 rows
subset_size = 50000000 #bookcorpus_train.num_rows
bookcorpus_train = bookcorpus_train.select(range(subset_size))

stats = analyze_dataset(bookcorpus_train, f"BookCorpus_{subset_size}")

display_stats(stats, f"BookCorpus_{subset_size}")



print("BookCorpus Analysis Complete")

#%% Salesforce/wikitext-103-v1 - 1.8M rows, clean text
wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1",
                       cache_dir=cache_dir,
                       )

#%% get the train split
wikitext_train = wikitext["train"]

#%% wikitext_train.num_rows = 1.8M

subset_size = wikitext_train.num_rows

#wikitext_train = wikitext_train.select(range(subset_size))

stats = analyze_dataset(wikitext_train, f"WikiText_{subset_size}")

display_stats(stats, f"WikiText_{subset_size}")

#%% [Wikipedia Analysis]
# Updated to use 20231101.en subset for English Wikipedia
wikipedia = load_dataset("wikimedia/wikipedia", "20231101.en",
                         cache_dir=cache_dir)


#%%
wikipedia_train = wikipedia["train"]

#%% get the train split
subset_size = wikipedia_train.num_rows

#wikipedia_train_subset = wikipedia_train.select(range(subset_size))

stats = analyze_dataset(wikipedia_train, f"Wikipedia_{subset_size}")

display_stats(stats, f"Wikipedia_{subset_size}")
# %%