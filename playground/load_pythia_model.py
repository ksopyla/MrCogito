
#%%
import os
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from rich import print


#%%

model_name = "EleutherAI/pythia-410m-deduped"
model_cache_dir = os.path.join("../models", model_name)

model = GPTNeoXForCausalLM.from_pretrained(
  model_name,
  cache_dir=model_cache_dir,
  torch_dtype=torch.float16,
  attn_implementation="sdpa",
  device_map="auto",
)

#%%
print(model)
#
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  cache_dir=model_cache_dir,
)

print(tokenizer)

# Get individual special tokens
print("\nIndividual special tokens:")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"BOS token: {tokenizer.bos_token}") 
print(f"UNK token: {tokenizer.unk_token}")
print(f"SEP token: {tokenizer.sep_token}")
print(f"CLS token: {tokenizer.cls_token}")
print(f"MASK token: {tokenizer.mask_token}")

# Get special token IDs
print("\nSpecial token IDs:")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"BOS token ID: {tokenizer.bos_token_id}")

#%%
inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])