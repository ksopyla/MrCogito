import os
import torch
from datasets import load_dataset
from transformers import DataCollatorForWholeWordMask



def load_and_preprocess_text_dataset(tokenizer, dataset_hf_path, dataset_name, text_column_name, test_size_percent=0.1, max_seq_length=512):
    """
    Loads and preprocesses the text dataset that fits to memory.
    
    * BookCorpus (bookcorpus/bookcorpus): Small (~1GB full), clean narrative text - https://huggingface.co/datasets/bookcorpus/bookcorpus
    * WikiMedia (wikimedia/wikipedia): Wikipedia articles with math/science concepts - https://huggingface.co/datasets/wikimedia/wikipedia
    * WikiText (Salesforce/wikitext): Preprocessed Wikipedia with math/science concepts - https://huggingface.co/datasets/Salesforce/wikitext
    """
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets"))

    dataset = load_dataset(dataset_hf_path, dataset_name, cache_dir=cache_dir, trust_remote_code=True)

    # check if the dataset contains a train and test split
       
    train_ds = dataset["train"]
    
    train_ds = train_ds.select(range(100000))
    
    if "test" in dataset:
        test_ds = dataset["test"]

    else:
        # test size is 10% of the train set not more than 100000 examples
        test_size = min(int(len(train_ds) * test_size_percent), 100000)
        train_ds, test_ds = train_ds.train_test_split(test_size=test_size)
  
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


class NeighborWordMaskCollator(DataCollatorForWholeWordMask):
    """
    This class mask the few nearby whole word, the intuition is that concepts contain multiple nearby words.
    """ 
    def __init__(self, *args, window_size=3, **kwargs):
        super().__init__(*args, **kwargs)
        
        # The window size for masking, defines how many whole words to mask
        self.window_size = window_size

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        This function masks the tokens in the input sequence.
        """
        # First apply whole word masking
        masked_inputs, mask_labels = super().torch_mask_tokens(inputs, special_tokens_mask)
        
        # Expand masks to neighbors
        batch_size, seq_len = inputs.shape
        expanded_mask = torch.zeros_like(masked_inputs, dtype=torch.bool)
        
        for b in range(batch_size):
            # Get original masked positions
            masked_indices = torch.where(mask_labels[b])[0].tolist()
            
            # Expand each mask position
            for idx in masked_indices:
                start = max(0, idx - self.window_size)
                end = min(seq_len, idx + self.window_size + 1)
                expanded_mask[b, start:end] = True
                
        # Apply expanded masking
        random_mask = torch.rand(expanded_mask.shape, device=inputs.device) < self.mlm_probability
        final_mask = expanded_mask & random_mask
        
        # Replace with [MASK] or random token
        masked_inputs = torch.where(
            final_mask,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token),
            inputs
        )
        
        # Optional: Replace 10% of masked tokens with random words
        random_words = torch.randint(
            len(self.tokenizer), 
            inputs.shape, 
            dtype=torch.long, 
            device=inputs.device
        )
        random_replace = (torch.rand(final_mask.shape, device=inputs.device) < 0.1) & final_mask
        masked_inputs[random_replace] = random_words[random_replace]

        return masked_inputs, final_mask
    
    
    
    
if __name__ == "__main__":
    pass
    """
    This class mask the few nearby whole word, the intuition is that concepts contain multiple nearby words.
    """ 
    def __init__(self, *args, window_size=3, **kwargs):
        super().__init__(*args, **kwargs)
        
        # The window size for masking, defines how many whole words to mask
        self.window_size = window_size

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        This function masks the tokens in the input sequence.
        """
        # First apply whole word masking
        masked_inputs, mask_labels = super().torch_mask_tokens(inputs, special_tokens_mask)
        
        # Expand masks to neighbors
        batch_size, seq_len = inputs.shape
        expanded_mask = torch.zeros_like(masked_inputs, dtype=torch.bool)
        
        for b in range(batch_size):
            # Get original masked positions
            masked_indices = torch.where(mask_labels[b])[0].tolist()
            
            # Expand each mask position
            for idx in masked_indices:
                start = max(0, idx - self.window_size)
                end = min(seq_len, idx + self.window_size + 1)
                expanded_mask[b, start:end] = True
                
        # Apply expanded masking
        random_mask = torch.rand(expanded_mask.shape, device=inputs.device) < self.mlm_probability
        final_mask = expanded_mask & random_mask
        
        # Replace with [MASK] or random token
        masked_inputs = torch.where(
            final_mask,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token),
            inputs
        )
        
        # Optional: Replace 10% of masked tokens with random words
        random_words = torch.randint(
            len(self.tokenizer), 
            inputs.shape, 
            dtype=torch.long, 
            device=inputs.device
        )
        random_replace = (torch.rand(final_mask.shape, device=inputs.device) < 0.1) & final_mask
        masked_inputs[random_replace] = random_words[random_replace]

        return masked_inputs, final_mask