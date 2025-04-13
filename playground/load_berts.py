#%% - load the bert model
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import BertModel, BertTokenizer
import torch
import os
from rich import print



cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Models"))
print(cache_dir)

#%% load the bert model for comparistion with the modern bert model
bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa", cache_dir=cache_dir)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)


#%% load the ModernBERT model

modern_bert_model = AutoModelForMaskedLM.from_pretrained('answerdotai/ModernBERT-base', cache_dir=cache_dir)
modern_bert_tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base', cache_dir=cache_dir)


#%%
model = modern_bert_model
tokenizer = modern_bert_tokenizer

model = bert_model
tokenizer = bert_tokenizer

#%% test the both of the models on example sentences with masked tokens



#  "Beautifull and shy girl with long hair collects the ripe sweet strawberries from the bush."

sentence = "[MASK] girl with long hair collects the ripe sweet strawberries from the bush."

# sentence with multiple masked tokens
# [MASK] == beautifull
sentence = "She was the most [MASK] girl I have ever seen, she was so gourgeous I couldn't believe my eyes."

# sentence with masked the  "bureaucratic" word, this is a word that is not in the vocab of the modern bert model
sentence = "Bureaucracy is everywhere, when applying for new ID, I have to fill out a ton of papers, and clerks were not helpful, those process are so [MASK]. Bureacracy is all over the place."

# "Her perspicacious analysis elucidated the labyrinthine complexities of geopolitical machinations"
sentence = "Her perspicacious analysis elucidated the labyrinthine [MASK][MASK] of geopolitical machinations"


#The control over oil-rich regions and strategic waterways has always been a key geopolitical factor in shaping relationships between nations
sentence = "Geopolitics. The control over oil-rich regions has always been a key [MASK] [MASK] factor in shaping relationships between nations."


tokenize_sentence = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(tokenize_sentence["input_ids"][0])

# Find all masked positions
masked_indices = [i for i, token_id in enumerate(tokenize_sentence["input_ids"][0]) 
                 if token_id == tokenizer.mask_token_id]

# Get predictions for all masks at once
outputs = model(**tokenize_sentence)
logits = outputs.logits[0]  # Get logits for first (and only) sequence
k=5


# Process each masked position
for mask_idx, masked_index in enumerate(masked_indices):
    predicted_tokens_ids = torch.topk(logits[masked_index], k=k).indices.tolist()
    predicted_tokens = [tokenizer.decode(token_id) for token_id in predicted_tokens_ids]
    predicted_logits = torch.topk(logits[masked_index], k=k).values.tolist()
    
    print(f"\nPredictions for Mask {mask_idx + 1} (position {masked_index}):")
    
    # pythonic way to print the token id, token and logit score with three decimal places, together
    print(list(zip(predicted_tokens_ids, predicted_tokens, predicted_logits)))
    
    # for token, logit in zip(predicted_tokens, predicted_logits):
    #     print(f"Token: {token}, Logit: {logit:.2f}")

# %% load bert from SequenceClassification

bert_seq_class_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir=cache_dir)







# %%
