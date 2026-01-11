import os
import torch
from datasets import load_dataset
from transformers import DataCollatorForWholeWordMask



def load_and_preprocess_text_dataset(tokenizer, dataset_hf_path, dataset_name_subset, text_column_name, test_size_percent=0.1, max_seq_length=512, dataset_cache_dir=None):
    """
    Loads and preprocesses the text dataset that fits to memory.
    
    * BookCorpus (bookcorpus/bookcorpus): Small (~1GB full), clean narrative text - https://huggingface.co/datasets/bookcorpus/bookcorpus
    * WikiMedia (wikimedia/wikipedia): Wikipedia articles with math/science concepts - https://huggingface.co/datasets/wikimedia/wikipedia
    * WikiText (Salesforce/wikitext): Preprocessed Wikipedia with math/science concepts - https://huggingface.co/datasets/Salesforce/wikitext
    
    Args:
        dataset_cache_dir: Optional path to cache directory. If None, uses ./Cache/Datasets relative to this file.
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
    dataset = load_dataset(dataset_hf_path, dataset_name_subset, cache_dir=DATASET_CACHE_DIR)

    # check if the dataset contains a train and test split
       
    train_ds = dataset["train"]
        
    if "test" in dataset:
        test_ds = dataset["test"]

    else:
        # test size is 10% of the train set not more than 100000 examples
        test_size = min(int(len(train_ds) * test_size_percent), 100000)
        # train_test_split returns a dictionary with 'train' and 'test' keys
        split_ds = train_ds.train_test_split(test_size=test_size)
        train_ds = split_ds['train']
        test_ds = split_ds['test']
  
    # Rename column to match processing
    # do a collumn rename based on the mapping provided below
    #check if the text_column_name is in the dataset
    if "text" not in train_ds.column_names:
        train_ds = train_ds.rename_column(text_column_name, "text")
    
    if "text" not in test_ds.column_names:
        test_ds = test_ds.rename_column(text_column_name, "text")


    # Tokenization function
    def tokenize_batch_function(examples):

        text_batch = examples["text"]

        
        return tokenizer(
            text_batch,  # Note different column name
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True
        )

   
    # Process train dataset
    train_ds = train_ds.map(
        tokenize_batch_function,
        batched=True,
        num_proc=os.cpu_count()-2,
        remove_columns=["text"]
    )

    
    # Process test dataset
    test_ds = test_ds.map(
        tokenize_batch_function,
        batched=True,
        num_proc=os.cpu_count()-2,
        remove_columns=["text"]
    )
    
    return train_ds, test_ds

    
if __name__ == "__main__":
    pass