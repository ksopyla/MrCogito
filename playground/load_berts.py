

#%% - load the bert model
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertModel, BertTokenizer
import torch


bert_model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#%% load the ModernBERT model

modern_bert_model = AutoModelForMaskedLM.from_pretrained('answerdotai/ModernBERT-base')
modern_bert_tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')

#%% test the both of the models on example sentences with masked tokens


#  "Beautifull and shy girl with long hair collects the ripe sweet strawberries from the bush."
sentence = "Beautifull and [MASK] [MASK] with long hair collects the ripe sweet strawberries from the bush."

mb_inputs = modern_bert_tokenizer(sentence, return_tensors="pt")
mb_outputs = modern_bert_model(**mb_inputs)

masked_index = mb_inputs["input_ids"][0].tolist().index(modern_bert_tokenizer.mask_token_id)
predicted_token_id = mb_outputs.logits[0, masked_index].argmax(axis=-1)
predicted_token = modern_bert_tokenizer.decode(predicted_token_id)
print("Predicted token:", predicted_token)

# print 5 best predictions with logits scores
predicted_tokens = torch.topk(mb_outputs.logits[0, masked_index], k=5).indices.tolist()
predicted_tokens = [modern_bert_tokenizer.decode(token_id) for token_id in predicted_tokens]
predicted_logits = torch.topk(mb_outputs.logits[0, masked_index], k=5).values.tolist()

# print the tokens and logits scores together
for token, logit in zip(predicted_tokens, predicted_logits):
    print(f"Token: {token}, Logit: {logit}")






# %%
