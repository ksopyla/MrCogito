#%% - load the bert model
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import BertModel, BertTokenizer
import torch
import os
from rich import print



MODEL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Models"))
print(MODEL_CACHE_DIR)

#%% load the bert model for comparistion with the modern bert model
bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa", cache_dir=MODEL_CACHE_DIR)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=MODEL_CACHE_DIR)


model = bert_model
tokenizer = bert_tokenizer

#%% load the ModernBERT model

# modern_bert_model = AutoModelForMaskedLM.from_pretrained('answerdotai/ModernBERT-base', cache_dir=MODEL_CACHE_DIR)
# modern_bert_tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base', cache_dir=MODEL_CACHE_DIR)


# model = modern_bert_model
# tokenizer = modern_bert_tokenizer



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

sentence = ["Cats [MASK] to drink milk.", "Dogs drink [MASK] ."]

#%%
tokenize_sentence = tokenizer(sentence, return_tensors="pt", padding=True)
tokens = tokenizer.convert_ids_to_tokens(tokenize_sentence["input_ids"][0])

# Find all masked positions
masked_indices = [i for i, token_id in enumerate(tokenize_sentence["input_ids"][0]) 
                 if token_id == tokenizer.mask_token_id]

##priint tokens with respective token ids in next line
print([(token, token_id) for token, token_id in zip(tokens, tokenize_sentence["input_ids"][0])])

#%% Get predictions for all masks at once
outputs = model(**tokenize_sentence)


#%%
# Print the k top predicted tokens for each token in the input,
# showing the original token and their top k predictions/logits


sent_idx = 1

logits = outputs.logits[sent_idx]  # Get logits for first (and only) sequence
tokens = tokenizer.convert_ids_to_tokens(tokenize_sentence["input_ids"][sent_idx])
k=4

for idx, (token, token_id) in enumerate(zip(tokens, tokenize_sentence["input_ids"][sent_idx])):
    # Identify whether this token is a mask (to highlight, if needed)
    is_mask = (token_id == tokenizer.mask_token_id)
    print(f"Token {idx}: '{token}' (id {token_id})")

    predicted_tokens_ids = torch.topk(logits[idx], k=k).indices.tolist()
    predicted_tokens = [tokenizer.decode(ptid).strip() for ptid in predicted_tokens_ids]
    predicted_logits = torch.topk(logits[idx], k=k).values.tolist()

    predicted_string = [f"Top {rank}: '{ptoken:<5}' id={ptid:<4} logit={logit:.3f}" for rank, (ptid, ptoken, logit) in enumerate(zip(predicted_tokens_ids, predicted_tokens, predicted_logits), 1)]
    print(predicted_string)


# %% load bert from SequenceClassification

bert_seq_class_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", cache_dir=MODEL_CACHE_DIR, num_labels=5)
print(bert_seq_class_model)

#load the bert for token classification
bert_token_class_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", cache_dir=MODEL_CACHE_DIR)
print(bert_token_class_model)

#load the bert for question answering
bert_qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased", cache_dir=MODEL_CACHE_DIR)
print(bert_qa_model)


#%% load the modern bert for sequence classification

modern_bert_seq_class_model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", cache_dir=MODEL_CACHE_DIR, num_labels=5)
print(modern_bert_seq_class_model)

#load the modern bert for token classification
modern_bert_token_class_model = AutoModelForTokenClassification.from_pretrained("answerdotai/ModernBERT-base", cache_dir=MODEL_CACHE_DIR)
print(modern_bert_token_class_model)    

#load the modern bert for question answering
modern_bert_qa_model = AutoModelForQuestionAnswering.from_pretrained("answerdotai/ModernBERT-base", cache_dir=MODEL_CACHE_DIR)
print(modern_bert_qa_model)


#%% load the XLNet model

xlnet_model = AutoModelForMaskedLM.from_pretrained("xlnet-base-cased", cache_dir=MODEL_CACHE_DIR)


#%% load the XLNet for sequence classification
xlnet_seq_class_model = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased", cache_dir=MODEL_CACHE_DIR) 

#load the XLNet for token classification
xlnet_token_class_model = AutoModelForTokenClassification.from_pretrained("xlnet-base-cased", cache_dir=MODEL_CACHE_DIR)

#load the XLNet for question answering
xlnet_qa_model = AutoModelForQuestionAnswering.from_pretrained("xlnet-base-cased", cache_dir=MODEL_CACHE_DIR) 

#%% load the DeBERTa model

deberta_model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-base", cache_dir=MODEL_CACHE_DIR)


#%% load the DeBERTa for sequence classification
deberta_seq_class_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", cache_dir=MODEL_CACHE_DIR)

#load the DeBERTa for token classification
deberta_token_class_model = AutoModelForTokenClassification.from_pretrained("microsoft/deberta-base", cache_dir=MODEL_CACHE_DIR)

#load the DeBERTa for question answering
deberta_qa_model = AutoModelForQuestionAnswering.from_pretrained("microsoft/deberta-base", cache_dir=MODEL_CACHE_DIR)


#%% load the Albert model

albert_model = AutoModelForMaskedLM.from_pretrained("albert-base-v2", cache_dir=MODEL_CACHE_DIR)

#%% load the Albert for sequence classification
albert_seq_class_model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", cache_dir=MODEL_CACHE_DIR)

#load the Albert for token classification
albert_token_class_model = AutoModelForTokenClassification.from_pretrained("albert-base-v2", cache_dir=MODEL_CACHE_DIR)











































# %%
