import os
import argparse
from typing import List, Optional
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
from dotenv import dotenv_values
from huggingface_hub import login, HfApi

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

## Training Details

- **Dataset**: [{dataset_name}](https://huggingface.co/datasets/{dataset_name})
- **Sample Size**: {sample_size} documents
- **Preprocessing**: Documents were truncated to a maximum length of 4096 characters to ensure training stability while preserving local context (code blocks, latex, paragraphs).
- **Algorithm**: Unigram (SentencePiece)
- **Vocab Size**: {vocab_size}
- **Normalization**: NFKC (Cased)
- **Pre-tokenization**: Metaspace

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
- Unused: `<|unused0|>` ... `<|unused99|>` (100 reserved tokens)
"""

def get_special_tokens(add_chat_tokens: bool = True, num_unused_tokens: int = 100) -> List[str]:
    """
    Define the set of special tokens for the tokenizer.
    Includes standard structural tokens, optional chat template tokens, and unused tokens for future expansion.
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
        
    # Add unused tokens for future extensions (like ModernBERT)
    # We use the <|unusedN|> format to be consistent with other special tokens
    if num_unused_tokens > 0:
        unused_tokens = [f"<|unused{i}|>" for i in range(num_unused_tokens)]
        tokens.extend(unused_tokens)
        
    return tokens

def setup_HF_environment():
    """Setup environment variables and directories"""
    # Load .env from project root (parent of training dir)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        envs = dotenv_values(env_path)
        if "HF_TOKEN" in envs:
            login(token=envs["HF_TOKEN"])
    
    # Enable parallelism for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

def prepare_training_corpus(dataset_name: str, sample_size: int = 1_000_000) -> List[str]:
    """
    Loads the dataset and prepares a list of strings for training.
    CRITICAL: Splits long documents into lines/chunks to avoid Unigram trainer panics ("likelihood is NAN").
    """
    print(f"\n{'='*60}")
    print(f"Preparing Training Corpus")
    print(f"Dataset: {dataset_name}")
    print(f"Target Samples: {sample_size}")
    print(f"{'='*60}\n")

    # 1. Load Data
    print(f"Loading dataset {dataset_name}...")
    # Load to RAM (streaming=False) for speed on high-RAM machines
    dataset = load_dataset(dataset_name, split="train", streaming=False)
    
    # 2. Identify Text Column (Moved up for optimization)
    text_column = "text"
    available_cols = dataset.column_names
    print(f"Available columns: {available_cols}")
    
    for col in ["text", "content", "body", "sentence"]:
        if col in available_cols:
            text_column = col
            break
    print(f"Using text column: '{text_column}'")

    # 3. Optimize: Select only the text column BEFORE shuffle/select
    # This drastically reduces memory usage by dropping metadata columns early
    # and making the shuffle operation lighter.
    if len(available_cols) > 1:
        print("Removing non-text columns to save memory...")
        dataset = dataset.select_columns([text_column])

    # 4. Select Samples
    if len(dataset) > sample_size:
        print(f"Shuffling and selecting {sample_size} random samples...")
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
    
    # 5. Process into manageable chunks (Corpus Creation)
    # Use dataset.map to truncate documents to a fixed length (e.g., 4096 chars).
    # This preserves structure (code, latex) while keeping sequence lengths safe for Unigram.
    
    MAX_DOC_LENGTH = 4096
    print(f"Truncating documents to max {MAX_DOC_LENGTH} chars using dataset.map...")
    
    def truncate_batch(batch):
        return {text_column: [text[:MAX_DOC_LENGTH] for text in batch[text_column]]}

    # Determine optimized process count
    import multiprocessing
    num_proc = max(1, multiprocessing.cpu_count() // 2) # Use half cores to be safe/nice
    
    dataset = dataset.map(
        truncate_batch,
        batched=True,
        num_proc=num_proc,
        desc="Truncating documents"
    )
    
    print("Extracting text to memory...")
    corpus = dataset[text_column]
    
    print(f"Corpus preparation complete.")
    print(f"Total training sequences: {len(corpus)}")
    avg_len = sum(len(s) for s in corpus) / len(corpus) if corpus else 0
    print(f"Average sequence length: {avg_len:.1f} chars")
    
    return corpus

def train_and_save_tokenizer(
    corpus: List[str],
    vocab_size: int,
    repo_id: str,
    dataset_name: str,
    push_to_hub: bool = False,
    local_dir: str = "./tokenizers"
):
    """
    Trains a Unigram tokenizer on a pre-loaded corpus.
    """
    print(f"\n{'='*60}")
    print(f"Training Tokenizer: {repo_id}")
    print(f"Vocab Size: {vocab_size}")
    print(f"{'='*60}\n")

    # 1. Initialize Tokenizer
    tokenizer = Tokenizer(models.Unigram())

    # 2. Normalization: NFKC
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])

    # 3. Pre-tokenization: Metaspace
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    # 4. Define Special Tokens
    special_tokens = get_special_tokens(add_chat_tokens=True)

    # 5. Trainer Configuration
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token="<unk>",
        shrinking_factor=0.75,
        show_progress=True,
        max_piece_length=10,
        n_sub_iterations=5
    )

    # 6. Train
    print("Starting training...")
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # 7. Post-Processing
    tokenizer.decoder = decoders.Metaspace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<cls> $A <sep>",
        pair="<cls> $A <sep> $B <sep>",
        special_tokens=[
            ("<cls>", tokenizer.token_to_id("<cls>")),
            ("<sep>", tokenizer.token_to_id("<sep>")),
        ],
    )

    # 8. Wrap & Save
    print("Wrapping in Transformers format...")
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=8192,
        pad_token="<pad>",
        unk_token="<unk>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        additional_special_tokens=[t for t in special_tokens if t not in ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]]
    )
    
    # Set Chat Template
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
            
            # Upload README
            readme_content = README_TEMPLATE.format(
                vocab_size=vocab_size,
                vocab_size_k=vocab_size//1000,
                repo_id=repo_id,
                dataset_name=dataset_name,
                sample_size=len(corpus)
            )
            
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
            print(f"Saving locally instead...")
            output_path = f"{local_dir}/{vocab_size}"
            fast_tokenizer.save_pretrained(output_path)
    else:
        output_path = f"{local_dir}/{vocab_size}"
        print(f"Saving locally to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        fast_tokenizer.save_pretrained(output_path)

def main():
    parser = argparse.ArgumentParser(description="Train custom Unigram tokenizer")
    parser.add_argument("--dataset", type=str, default="JeanKaddour/minipile", help="HuggingFace dataset name")
    parser.add_argument("--sample_size", type=int, default=1_000_000, help="Number of samples to load")
    parser.add_argument("--vocab_sizes", type=int, nargs="+", default=[32000, 64000], help="List of vocab sizes to train")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--user_handle", type=str, default="ksopyla", help="HF username for repo creation")
    
    args = parser.parse_args()

    setup_HF_environment()

    # 1. Load and prepare corpus ONCE
    corpus = prepare_training_corpus(args.dataset, args.sample_size)
    
    # 2. Train multiple vocab sizes on the same corpus
    for vocab in args.vocab_sizes:
        # Naming Convention: {dataset_type}-{task}-{model_name}-{params}-{date}
        # e.g. minipile-unigram-tokenizer-32k-1M
        # But user asked for: minipile-english-unigram-{vocab}k
        # And requested to add vocab size and number of training documents to the name
        
        # Format samples count (e.g. 1000000 -> 1M, 500000 -> 500k)
        samples_count = len(corpus)
        if samples_count >= 1_000_000:
            samples_str = f"{samples_count//1_000_000}M"
        else:
            samples_str = f"{samples_count//1000}k"
            
        repo_name = f"{args.user_handle}/minipile-unigram-{vocab//1000}k-{samples_str}"
        
        train_and_save_tokenizer(
            corpus=corpus,
            vocab_size=vocab,
            repo_id=repo_name,
            dataset_name=args.dataset,
            push_to_hub=args.push_to_hub
        )

if __name__ == "__main__":
    main()
