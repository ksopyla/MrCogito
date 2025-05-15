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
    
    train_ds = train_ds.select(range(20000))
    
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
    This class masks neighboring whole words to capture concept-level information.
    The intuition is that concepts often contain multiple adjacent words.
    Inspired by LLaDA (Large Language Diffusion Models) that uses higher masking rates
    and by the concept encoder hypothesis that adjacent tokens form meaningful concepts.
    
    Unlike standard masking which randomly selects individual tokens, this collator:
    1. Masks at a higher rate (default 25% vs BERT's 15%)
    2. Expands masks to neighboring words using a window approach
    3. Respects word boundaries so words aren't broken in the middle
    4. Creates clusters of adjacent masked words to capture concept-level information
    """
    def __init__(self, tokenizer, mlm_probability=0.25, window_size=3, pad_to_multiple_of=None):
        """
        Initialize the NeighborWordMaskCollator.
        
        Args:
            tokenizer: The tokenizer to use for masking
            mlm_probability: Probability of masking a word (defaults to 0.25, higher than BERT's 0.15)
            window_size: Size of the window around each masked word to potentially mask
            pad_to_multiple_of: Pad sequences to multiples of this value
        """
        super().__init__(
            tokenizer=tokenizer, 
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of
        )
        self.window_size = window_size
    
    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 
        80% MASK, 10% random, 10% original.
        
        This overrides the parent method to implement concept-level masking by:
        1. First applying whole word masking to select initial words
        2. Then expanding the mask to neighboring words
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length)
            special_tokens_mask: Optional mask for special tokens
        
        Returns:
            Tuple of (masked_inputs, labels) where:
                - masked_inputs is the input with masked tokens
                - labels is tensor of the same shape as inputs with -100 for non-masked tokens
        """
        # Get base masking from parent class
        inputs_clone = inputs.clone()
        masked_inputs, labels = super().torch_mask_tokens(inputs_clone, special_tokens_mask)
        
        # Expand masks to neighboring words
        batch_size, seq_length = inputs.size()
        
        # Get word IDs for each token in the batch
        word_ids_batch = []
        for batch_idx in range(batch_size):
            encoding = self.tokenizer.encode_plus(
                self.tokenizer.decode(inputs[batch_idx].tolist(), skip_special_tokens=False),
                return_tensors="pt",
                add_special_tokens=False
            )
            word_ids = encoding.word_ids()
            if len(word_ids) < seq_length:
                word_ids = word_ids + [None] * (seq_length - len(word_ids))
            elif len(word_ids) > seq_length:
                word_ids = word_ids[:seq_length]
            word_ids_batch.append(word_ids)
        
        # Create new expanded mask
        expanded_mask = torch.zeros_like(inputs, dtype=torch.bool)
        final_labels = torch.ones_like(inputs) * -100  # Default: don't predict
        
        for batch_idx in range(batch_size):
            # Find originally masked words (not just tokens)
            masked_word_ids = set()
            for token_idx in range(seq_length):
                if labels[batch_idx, token_idx] != -100:  # If this token was originally masked
                    word_id = word_ids_batch[batch_idx][token_idx]
                    if word_id is not None:  # Skip special tokens
                        masked_word_ids.add(word_id)
            
            # For each masked word, find neighbors to mask
            for masked_word_id in masked_word_ids:
                # Find tokens in the window around this word
                for token_idx in range(seq_length):
                    current_word_id = word_ids_batch[batch_idx][token_idx]
                    
                    # Skip special tokens and None word_ids
                    if current_word_id is None:
                        continue
                    
                    # Check if this token belongs to a word within the window
                    if (current_word_id in masked_word_ids or 
                        any(abs(current_word_id - other_id) <= self.window_size 
                            for other_id in masked_word_ids if current_word_id is not None and other_id is not None)):
                        expanded_mask[batch_idx, token_idx] = True
                        final_labels[batch_idx, token_idx] = inputs[batch_idx, token_idx]
            
            # Apply additional random masking to ensure we reach desired masking rate
            actual_mask_rate = expanded_mask[batch_idx].float().mean().item()
            if actual_mask_rate < self.mlm_probability:
                additional_mask_needed = self.mlm_probability - actual_mask_rate
                additional_mask_probs = torch.rand(seq_length, device=inputs.device)
                
                # Only consider unmasked, non-special tokens for additional masking
                valid_tokens = ~expanded_mask[batch_idx]
                if special_tokens_mask is not None:
                    valid_tokens = valid_tokens & ~special_tokens_mask[batch_idx]
                
                # Calculate how many more tokens need to be masked
                n_valid = valid_tokens.sum().item()
                n_to_mask = int(n_valid * additional_mask_needed / (1 - actual_mask_rate))
                
                # Get indices of valid tokens
                valid_indices = torch.where(valid_tokens)[0]
                
                # Randomly select indices to mask
                if n_to_mask > 0 and len(valid_indices) > 0:
                    perm = torch.randperm(len(valid_indices), device=inputs.device)
                    additional_indices = valid_indices[perm[:n_to_mask]]
                    
                    # Apply additional masking
                    expanded_mask[batch_idx, additional_indices] = True
                    final_labels[batch_idx, additional_indices] = inputs[batch_idx, additional_indices]
        
        # Replace masked tokens with [MASK] or random tokens
        masked_inputs = inputs.clone()
        
        # 80% of the time, replace with [MASK]
        mask_indices = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & expanded_mask
        masked_inputs[mask_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace with random token
        random_indices = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & expanded_mask & ~mask_indices
        random_tokens = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long, device=inputs.device)
        masked_inputs[random_indices] = random_tokens[random_indices]
        
        # The rest of the time (10%), keep original tokens unchanged
        
        return masked_inputs, final_labels
    
    def __call__(self, examples):
        """
        Apply masking to a batch of examples.
        
        Args:
            examples: List of dictionaries with input_ids, token_type_ids, etc.
            
        Returns:
            Dictionary with masked inputs and corresponding labels
        """
        batch = super().__call__(examples)
        
        # Ensure "special_tokens_mask" is available or compute it
        if "special_tokens_mask" not in batch:
            batch["special_tokens_mask"] = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in batch["input_ids"]
            ]
            batch["special_tokens_mask"] = torch.tensor(batch["special_tokens_mask"], dtype=torch.bool)
        
        # Apply neighbor word masking
        inputs, labels = self.torch_mask_tokens(
            batch["input_ids"], batch["special_tokens_mask"]
        )
        
        batch["input_ids"] = inputs
        batch["labels"] = labels
        
        return batch
    
if __name__ == "__main__":
    pass