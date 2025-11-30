import os
import argparse
from typing import List, Optional
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

# Define the README Template as a constant
README_TEMPLATE = """---
language:
- en
license: apache-2.0
tags:
- tokenizer
- unigram
- minipile
- concept-encoder
- chatml
- morphology
---

# Custom Unigram Tokenizer for Minipile ({vocab_size_k}k Vocab)

This is a Unigram tokenizer (SentencePiece-style) trained on the [JeanKaddour/minipile](https://huggingface.co/datasets/JeanKaddour/minipile) dataset. 

**Language**: English (en). 
*Note: While the Unigram algorithm handles unicode characters, the vocabulary is optimized for English text, code, and common technical terms found in Minipile.*

It was developed for the **[MrCogito](https://github.com/ksopyla/MrCogito)** project, which explores novel transformer architectures like the **Concept Encoder**.

## Motivation for Concept Encoders

This tokenizer is specifically optimized for **Concept Encoder** and **Concept Decoder** architectures (e.g., Perceiver IO, Latent Transformers).

### Why Unigram for Concept Encoding?
Concept Encoders work by compressing a sequence of tokens $T$ into a smaller set of abstract latent vectors (concepts) $C$. The efficiency of this compression depends heavily on the input quality:

*   **The Problem with BPE (BERT/GPT)**: BPE is a greedy compression algorithm. It often splits words into arbitrary frequent chunks (e.g., `unbe` + `liev` + `able`). A Concept Encoder must waste model capacity "repairing" these arbitrary splits to understand the word before it can even begin extracting the higher-level concept.
*   **The Unigram Advantage**: The Unigram algorithm is probabilistic and tends to preserve linguistically meaningful morphological units (e.g., `un` + `believ` + `able`). This acts as a "soft pre-compression", feeding the encoder units that already carry semantic weight. This allows the Concept Encoder to focus its limited latent capacity on *semantic aggregation* rather than *morphological repair*.

### Why Minipile?
While XLNet uses the Unigram algorithm, its vocabulary (circa 2019) lacks modern terms. By training a fresh Unigram model on **Minipile** (2023), we combine the superior morphological segmentation of Unigram with the vocabulary coverage of modern LLMs (code, Python, ChatML, technical terms).

## Features
- **Algorithm**: Unigram (SentencePiece).
- **Vocab Size**: {vocab_size} tokens.
- **Normalization**: NFKC (Cased).
- **Pre-tokenization**: Metaspace (reversible).
- **Chat Support**: Includes standard ChatML special tokens (`<|im_start|>`, `<|im_end|>`) and a pre-configured chat template.

## Usage

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Basic encoding
text = "Hello, world! This is a test."
tokens = tokenizer.tokenize(text)
print(tokens) 
# Output: [' Hello', ',', ' world', '!', ' This', ' is', ' a', ' test', '.']

# Chat Template
messages = [
    {{"role": "user", "content": "What is the Concept Encoder?"}},
    {{"role": "assistant", "content": "It is a novel transformer architecture..."}}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
# Output: <|im_start|>user\\nWhat is the Concept Encoder?<|im_end|>\\n<|im_start|>assistant\\n...
```

## Special Tokens
- PAD: `<pad>`
- UNK: `<unk>`
- CLS: `<cls>`
- SEP: `<sep>`
- MASK: `<mask>`
- Chat: `<|im_start|>`, `<|im_end|>`, `<|user|>`, `<|assistant|>`, `<|system|>`, `<|endoftext|>`
"""

def get_special_tokens(add_chat_tokens: bool = True) -> List[str]:
    """
    Define the set of special tokens for the tokenizer.
    Includes standard structural tokens and optional chat template tokens.
    """
    # Standard structural tokens (XLNet/RoBERTa style)
    tokens = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]
    
    if add_chat_tokens:
        # Modern Chat / Instruction Tuning tokens
        # Using standard formats often seen in open models
        chat_tokens = [
            "<|system|>", 
            "<|user|>", 
            "<|assistant|>", 
            "<|endoftext|>",  # EOS for generation
            "<|im_start|>",   # ChatML style
            "<|im_end|>",     # ChatML style
            
            # Tool Calling / Function Execution
            "<|tool_call|>",
            "<|tool_response|>",
            
            # Reasoning / Chain of Thought (DeepSeek style)
            "<|thought|>",
            
            # Multimodality Placeholders (Future proofing)
            "<|image|>",
            "<|audio|>"
        ]
        tokens.extend(chat_tokens)
        
    return tokens

def train_custom_tokenizer(
    dataset_name: str,
    vocab_size: int,
    repo_id: str,
    sample_size: int = 1_000_000,
    push_to_hub: bool = False,
    local_dir: str = "./tokenizers"
):
    """
    Trains a Unigram tokenizer (XLNet-style) robust for modern use cases.
    
    Features:
    - Unigram algorithm (best for morphology)
    - Metaspace pre-tokenization (reversible, handles whitespace)
    - NFKC normalization (standard Unicode)
    - Chat template support (reserved tokens)
    """
    print(f"\n{'='*60}")
    print(f"Training Custom Unigram Tokenizer")
    print(f"Dataset: {dataset_name}")
    print(f"Vocab Size: {vocab_size}")
    print(f"Target Repo: {repo_id}")
    print(f"{'='*60}\n")

    # 1. Initialize Tokenizer with Unigram model
    tokenizer = Tokenizer(models.Unigram())

    # 2. Normalization: NFKC
    # Using Cased training (no Lowercase) is better for code, names, and modern tasks.
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])

    # 3. Pre-tokenization: Metaspace
    # Replaces spaces with _ (U+2581). Critical for reversibility.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    # 4. Define Special Tokens
    special_tokens = get_special_tokens(add_chat_tokens=True)
    print(f"Special tokens: {special_tokens}")

    # 5. Trainer Configuration
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token="<unk>",
        shrinking_factor=0.75
    )

    # 6. Load Data
    print(f"Loading dataset {dataset_name}...")
    # Polonez Optimization: Load to RAM (streaming=False) for speed
    # 1M samples is small for 256GB RAM (~1-2GB text)
    dataset = load_dataset(dataset_name, split="train", streaming=False)
    
    # Select samples
    print(f"Selecting {sample_size} samples...")
    if len(dataset) > sample_size:
        # Randomly sample to ensure diverse coverage (code, web, academic)
        # instead of taking the first N which might be biased by source
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
    
    # Pre-extract text to list to remove Python iterator overhead during Rust training
    print("Extracting text to memory...")
    # Handle different column names if needed
    text_column = "text"
    for col in ["content", "body", "sentence"]:
        if col in dataset.column_names:
            text_column = col
            break
            
    # Using dataset['text'] returns a list directly in Arrow/HF datasets
    # This is extremely fast and keeps data contiguous in memory
    raw_corpus = dataset[text_column]
    
    # Polonez Optimization: Chunking
    # The Unigram trainer crashes on extremely long documents ("likelihood is NAN").
    # We must split long documents into manageable chunks for the trainer.
    # Reduced to 32768 to be absolutely safe against float underflow in lattice.
    MAX_CHUNK_SIZE = 2**15 # 32768
    corpus = []
    print(f"Chunking documents to {MAX_CHUNK_SIZE} chars to prevent trainer crash...")
    for text in raw_corpus:
        if len(text) > MAX_CHUNK_SIZE:
            # Split into chunks
            for i in range(0, len(text), MAX_CHUNK_SIZE):
                corpus.append(text[i : i + MAX_CHUNK_SIZE])
        else:
            corpus.append(text)
            
    print(f"Final training corpus size (after chunking): {len(corpus)} segments")

    # 7. Train
    print(f"Starting training on {len(corpus)} samples...")
    # Passing a list is much faster than a generator for the Rust backend
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # 8. Post-Processing (Decoder & Template)
    tokenizer.decoder = decoders.Metaspace()
    
    # Default template for single sequence: <cls> seq <sep>
    # This ensures compatibility with standard BERT-like pipelines
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<cls> $A <sep>",
        pair="<cls> $A <sep> $B <sep>",
        special_tokens=[
            ("<cls>", tokenizer.token_to_id("<cls>")),
            ("<sep>", tokenizer.token_to_id("<sep>")),
        ],
    )

    # 9. Wrap in Transformers & Save
    print("Training complete. wrapping...")
    
    # Convert special tokens list to a map for the wrapper
    # We need to explicitly tell the wrapper which token does what
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=2048,
        pad_token="<pad>",
        unk_token="<unk>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        # Additional special tokens are handled by the tokenizer_object, 
        # but we can register them here for easy access property
        additional_special_tokens=[t for t in special_tokens if t not in ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]]
    )
    
    # Define a default chat template (ChatML style as an example, popular and robust)
    # This allows tokenizer.apply_chat_template() to work out of the box
    fast_tokenizer.chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    if push_to_hub:
        print(f"Pushing to Hub: {repo_id}...")
        try:
            fast_tokenizer.push_to_hub(repo_id)
            
            # Create and upload a detailed README.md (Model Card)
            readme_content = README_TEMPLATE.format(
                vocab_size=vocab_size,
                vocab_size_k=vocab_size//1000,
                repo_id=repo_id
            )
            
            # Use HfApi to upload the README
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=readme_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model"
            )
            
            print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Upload failed: {e}")
            print(f"Saving locally instead to {local_dir}/{vocab_size}")
            fast_tokenizer.save_pretrained(f"{local_dir}/{vocab_size}")
    else:
        output_path = f"{local_dir}/{vocab_size}"
        print(f"Saving locally to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        fast_tokenizer.save_pretrained(output_path)

def main():
    parser = argparse.ArgumentParser(description="Train custom Unigram tokenizer")
    parser.add_argument("--dataset", type=str, default="JeanKaddour/minipile", help="HuggingFace dataset name")
    parser.add_argument("--sample_size", type=int, default=1_000_000, help="Number of samples to train on")
    parser.add_argument("--vocab_sizes", type=int, nargs="+", default=[32000, 64000], help="List of vocab sizes to train")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--user_handle", type=str, default="ksopyla", help="HF username for repo creation")
    
    args = parser.parse_args()

    for vocab in args.vocab_sizes:
        repo_name = f"{args.user_handle}/minipile-english-unigram-{vocab//1000}k"
        train_custom_tokenizer(
            dataset_name=args.dataset,
            vocab_size=vocab,
            repo_id=repo_name,
            sample_size=args.sample_size,
            push_to_hub=args.push_to_hub
        )

if __name__ == "__main__":
    main()

