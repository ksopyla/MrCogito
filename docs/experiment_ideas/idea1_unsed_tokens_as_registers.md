
### Idea 1 - use the unsued BERT or ModernBERT tokens as concept tokens

Do not crate new architecture, just add additional concept tokens at the begining of the sequence.


Publications: 

* [Memory Transformer](https://arxiv.org/abs/2006.11527) 
* [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
* [Large Concept Models]()
* [LLaDA diffusion model]()




## How to replace unused tokens in a model

I want to replace the name of the unused tokens in a model with a new name, the idea is to use those unused tokens as concept tokens. 

Vocab unused tokens replacement: 

* https://github.com/huggingface/transformers/issues/31475 
* https://github.com/huggingface/transformers/issues/27974
* https://discuss.huggingface.co/t/change-gemma-tokenizer-unused-token/80867/2 
 

Untested code, by copilot:

```python
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define the mapping of old tokens to new tokens
token_mapping = {
    "<unused0>": "<NEW_TOKEN1>",
    "<unused1>": "<NEW_TOKEN2>"
}

# Update the tokenizer's vocabulary

for old_token, new_token in token_mapping.items():
    if old_token in tokenizer.get_vocab():
        token_id = tokenizer.convert_tokens_to_ids(old_token)
        tokenizer.add_tokens([new_token])
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        tokenizer.vocab[new_token] = new_token_id
        del tokenizer.vocab[old_token]

# Save the updated tokenizer
tokenizer.save_pretrained("./updated_tokenizer")
```