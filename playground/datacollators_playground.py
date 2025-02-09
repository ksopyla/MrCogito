#%%
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding, DataCollatorForWholeWordMask
from torch.utils.data import DataLoader
import torch
import os
from rich import print
import numpy as np



# Create a console instance with specific settings
# from rich.console import Console
# console = Console(force_jupyter=True, soft_wrap=True)


# 2. Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#%% tokenize the simple two sentences list and compare the results with different parameters: max_length, padding, return_special_tokens_mask, return_overflowing_tokens etc.

text = ["1 bureaucratic bureaucrats bureaucratical one two three four five six seven eight nine ten one two three four five six seven eight nine ten","2 One two three four five six seven eight nine ten", "3  One two"]


tokenized_text = tokenizer(text, padding=False, 
                           max_length=12, 
                           truncation=True, 
                           return_special_tokens_mask=True, 
                           return_overflowing_tokens=True)

# Fix the token conversion by iterating through batches
print(f"Tokenized text structure")

for i, tokenized_sentence in enumerate(zip(tokenized_text['input_ids'], tokenized_text['attention_mask'])):
    input_ids = tokenized_sentence[0]
    attention_mask = tokenized_sentence[1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Use a single print statement with Rich's formatting
    print(
        f"\nText {i+1} tokens:",
        tokens,
        input_ids,
        f"Attn msk: {attention_mask}",
        sep='\n'
    )

# prepare the data for the dataloader
# Convert tokenized text to Hugging Face Dataset format
formatted_data = {
    'input_ids': [enc.ids for enc in tokenized_text.encodings],
    'attention_mask': [enc.attention_mask for enc in tokenized_text.encodings]
}


# Create Hugging Face Dataset
tokenized_dataset = Dataset.from_dict(formatted_data).with_format('torch')


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.3  # 15% of tokens will be masked
)

data_collator_whole_word= DataCollatorForWholeWordMask(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.3  # 15% of tokens will be masked
)

def concept_masking_collate_fn(examples):
    """
    Function takes a list of already tokenized examples with special_tokens_mask and returns a batch of masked examples with labels.
    We want to mask neighboring words without breaking the words and not relay on the subword bert special hash tokens prefixed with "##" like the  DataCollatorForWholeWordMask does.
    
    
    """
    
    


    print(examples)
    

# Create DataLoader with correct dataset
dataloader = DataLoader(
    tokenized_dataset,  # Use formatted dataset instead of raw tokenizer output
    batch_size=2,
    collate_fn=data_collator_whole_word,
    shuffle=False
)



#%%

for i, batch in enumerate(dataloader):

 
    # Decode the input IDs to see original text
    original_text = tokenizer.batch_decode(batch['input_ids'])
    original_text = '\n'.join(original_text)
    # print("\nOriginal text snippets:")
    # print('\n'.join([t[:100] + '...' for t in original_text]))
    print(f"=== Batch {i+1} ===",
        f"Orig text:",
        original_text,
        f"In:",
        batch['input_ids'],
        f"Atn:",
        batch['attention_mask'],
        f"Lab:",
        batch['labels'],
        sep='\n'
    )


    # Show masked versions
    masked_text = tokenizer.batch_decode(batch['input_ids'] * (batch['labels'] != -100).long())
    print("Masked text snippets:",
          '\n'.join([t[:100] + '...' for t in masked_text]),sep='\n')
    

    # Show tensor shapes
    print("Tensor shapes:",
          f"Input IDs: {batch['input_ids'].shape}",
          f"Attn mask: {batch['attention_mask'].shape}",
          f"Labels: {batch['labels'].shape}",sep='\n')






#%% 3. Tokenization function


cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Datasets"))
print(cache_dir)


dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1",
                       cache_dir=cache_dir)

dataset_test = dataset["test"]

def tokenize_function(examples):

    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',  # We'll test both with and without this
        max_length=50,
        return_special_tokens_mask=True,
        return_overflowing_tokens=True
    )

# 4. Process the dataset
tokenized_dataset = dataset_test.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
).with_format('torch')



#%% 5. Create Data Collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # 15% of tokens will be masked
)

# 6. Create DataLoader
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=2,
    collate_fn=data_collator,
    shuffle=False
)

# 7. Let's inspect a few batches
for i, batch in enumerate(dataloader):
    print(f"\n=== Batch {i+1} === {batch}")
    
    # Decode the input IDs to see original text
    original_text = tokenizer.batch_decode(batch['input_ids'])
    print("\nOriginal text snippets:")
    print('\n'.join([t[:100] + '...' for t in original_text]))
    
    # Show masked versions
    masked_text = tokenizer.batch_decode(batch['input_ids'] * (batch['labels'] != -100).long())
    print("\nMasked text snippets:")
    print('\n'.join([t[:100] + '...' for t in masked_text]))
    
    # Show tensor shapes
    print("\nTensor shapes:")
    print(f"Input IDs: {batch['input_ids'].shape}")
    print(f"Attention mask: {batch['attention_mask'].shape}")
    print(f"Labels: {batch['labels'].shape}")
    
    # Show sample masked tokens
    print("\nSample masked tokens:")
    for seq_idx in range(2):
        mask_positions = (batch['labels'][seq_idx] != -100).nonzero().squeeze()
        if mask_positions.numel() > 0:
            print(f"Sequence {seq_idx+1}:")
            for pos in mask_positions[:3]:  # Show first 3 masked tokens
                original_token = tokenizer.decode(batch['labels'][seq_idx][pos])
                masked_token = tokenizer.decode(batch['input_ids'][seq_idx][pos])
                print(f"  Position {pos}: {original_token} (was {masked_token})")
    
    # Only show first 2 batches for demonstration
    if i == 1:
        break

# 8. Experiment with different collator settings
print("\n\n=== Experiment with Different Collator Settings ===")

# Try with dynamic padding (no pre-padding)
data_collator_dynamic = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    pad_to_multiple_of=8  # For better GPU utilization
)

dataloader_dynamic = DataLoader(
    tokenized_dataset,
    batch_size=2,
    collate_fn=data_collator_dynamic,
    shuffle=True
)

# Inspect dynamic padding results
batch = next(iter(dataloader_dynamic))
print("\nDynamic padding batch shapes:")
print(f"Input IDs: {batch['input_ids'].shape}")
print(f"Attention mask: {batch['attention_mask'].shape}")
print(f"Labels: {batch['labels'].shape}")

# %%
