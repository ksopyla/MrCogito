#%%
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
from rich import print

#%%
model_name = "meta-llama/Llama-3.2-1B-instruct"
model_cache_dir = os.path.join("../models", model_name)

# Load model with mixed precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=model_cache_dir,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",  # Use Flash Attention 2
    device_map="auto",
)

#%%
print("\nModel info:")
print(model)

print("\nModel configuration:")
print(model.config)

#%%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=model_cache_dir,
)

print("\nTokenizer info:")
print(tokenizer)

# Print special tokens
print("\nSpecial tokens:")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"BOS token: {tokenizer.bos_token}")
print(f"UNK token: {tokenizer.unk_token}")

# Print special token IDs
print("\nSpecial token IDs:")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"BOS token ID: {tokenizer.bos_token_id}")

#%%
# Test the model
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print("\nModel output:")
print(tokenizer.decode(outputs[0])) 
# %%


# %%
llama_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>
What is the capital of France?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
inputs = tokenizer(llama_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=500)
print("\nModel output:")
print(tokenizer.decode(outputs[0])) 
# %%


processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_cache_dir)

model_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

chat_ml_prompt = processor.apply_chat_template(model_messages, add_generation_prompt=True, tokenize=False)

print(chat_ml_prompt)

#%%

chatml_inputs = processor.apply_chat_template(model_messages, add_generation_prompt=True, tokenize=False)
chatml_inputs = processor(chatml_inputs,add_special_tokens=False, return_tensors="pt").to(model.device)
outputs = model.generate(**chatml_inputs, max_length=500)
print("\nModel output:")
print(tokenizer.decode(outputs[0])) 

print(processor.decode(outputs[0][chatml_inputs["input_ids"].shape[-1]:]))
# %%
