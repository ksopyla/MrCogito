import os
import argparse
from typing import List, Optional
from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
from dotenv import dotenv_values
from huggingface_hub import login, HfApi

# Define README Templates for different algorithms
README_TEMPLATE_UNIGRAM = """---
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
- **Preprocessing**: Documents were chunked into 8192-character segments to ensure training stability while preserving local context (code blocks, latex, paragraphs). This prevents numerical instability in the training algorithm.
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

README_TEMPLATE_BPE = """---
language:
- en
license: apache-2.0
tags:
- tokenizer
- bpe
- minipile
- concept-encoder
- chatml
- code
---

# Custom BPE Tokenizer for Minipile ({vocab_size_k}k Vocab)

This is a Byte Pair Encoding (BPE) tokenizer trained on the [JeanKaddour/minipile](https://huggingface.co/datasets/JeanKaddour/minipile) dataset. 

**Language**: English (en). 
*Note: BPE is the industry standard for code models (GPT-4, CodeLlama, StarCoder). This tokenizer is optimized for English text, code, and common technical terms found in Minipile.*

It was developed for the **[MrCogito](https://github.com/ksopyla/MrCogito)** project, which explores novel transformer architectures like the **Concept Encoder**.

## Training Details

- **Dataset**: [{dataset_name}](https://huggingface.co/datasets/{dataset_name})
- **Sample Size**: {sample_size} documents
- **Preprocessing**: Documents were chunked into 8192-character segments to ensure training stability while preserving local context (code blocks, latex, paragraphs). This prevents numerical instability in the training algorithm.
- **Algorithm**: BPE (Byte Pair Encoding)
- **Vocab Size**: {vocab_size}
- **Normalization**: NFKC (Cased)
- **Pre-tokenization**: ByteLevel (standard for code)

## Motivation for Concept Encoders

This tokenizer is trained for **Concept Encoder** and **Concept Decoder** architectures (e.g., Perceiver IO, Latent Transformers).

### Why BPE?
BPE is the de-facto standard for modern language models, especially those handling code:
*   **Industry Standard**: Used by GPT-2, GPT-3, GPT-4, CodeLlama, StarCoder, and most modern LLMs
*   **Code-Friendly**: ByteLevel pre-tokenization handles whitespace and special characters robustly, crucial for programming languages
*   **Proven Performance**: Extensive research shows BPE works well for both natural language and code

### Why Minipile?
By training a fresh BPE model on **Minipile** (2023), we combine the proven BPE algorithm with modern vocabulary coverage (code, Python, ChatML, technical terms) while maintaining compatibility with existing model architectures.

## Features
- **Algorithm**: BPE (Byte Pair Encoding).
- **Vocab Size**: {vocab_size} tokens.
- **Normalization**: NFKC (Cased).
- **Pre-tokenization**: ByteLevel (reversible, handles all Unicode).
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
# Output: ['Hello', ',', 'Ġworld', '!', 'ĠThis', 'Ġis', 'Ġa', 'Ġtest', '.']

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
    Includes standard structural tokens, optional chat template tokens, unused tokens.
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
    # CRITICAL FIX: Unigram trainer panics on very long sequences (likelihood NAN).
    # We must split documents into smaller chunks (e.g., 4KB-8KB) instead of one huge string.
    # This preserves local context (paragraphs/functions) while keeping the trainer stable.
    
    CHUNK_SIZE = 8192
    print(f"Chunking documents to max {CHUNK_SIZE} chars...")
    
    def chunk_examples(batch):
        chunks = []
        for text in batch[text_column]:
            # Split text into chunks of CHUNK_SIZE
            for i in range(0, len(text), CHUNK_SIZE):
                chunk = text[i : i + CHUNK_SIZE]
                if len(chunk) > 100:  # Skip tiny fragments
                    chunks.append(chunk)
        return {text_column: chunks}

    # Determine optimized process count
    import multiprocessing
    num_proc = max(1, multiprocessing.cpu_count() // 2)
    
    dataset = dataset.map(
        chunk_examples,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names, # We only need the new chunks
        desc="Chunking corpus"
    )
    
    print(f"Corpus preparation complete.")
    print(f"Total training sequences: {len(dataset)}")
    
    return dataset[text_column]

def _create_post_processor(tokenizer: Tokenizer) -> processors.TemplateProcessing:
    """Create post-processor with CLS/SEP template (common for both algorithms)."""
    return processors.TemplateProcessing(
        single="<cls> $A <sep>",
        pair="<cls> $A <sep> $B <sep>",
        special_tokens=[
            ("<cls>", tokenizer.token_to_id("<cls>")),
            ("<sep>", tokenizer.token_to_id("<sep>")),
        ],
    )


def _train_bpe_tokenizer(corpus: List[str], vocab_size: int, special_tokens: List[str]) -> Tokenizer:
    """Train a BPE tokenizer (GPT-2/CodeLlama style)."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )
    
    print("Starting BPE training...")
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = _create_post_processor(tokenizer)
    
    return tokenizer


def _train_unigram_tokenizer(corpus: List[str], vocab_size: int, special_tokens: List[str]) -> Tokenizer:
    """Train a Unigram tokenizer (XLNet/SentencePiece style)."""
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    
    # Get initial alphabet to minimize <unk> tokens
    try:
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    except AttributeError:
        initial_alphabet = []
    
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token="<unk>",
        initial_alphabet=initial_alphabet,
        shrinking_factor=0.75,
        show_progress=True,
        max_piece_length=24,
        n_sub_iterations=5
    )
    
    print("Starting Unigram training...")
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    tokenizer.decoder = decoders.Metaspace()
    tokenizer.post_processor = _create_post_processor(tokenizer)
    
    return tokenizer


def _wrap_tokenizer(tokenizer: Tokenizer, special_tokens: List[str]) -> PreTrainedTokenizerFast:
    """Wrap tokenizer in Transformers format with chat template."""
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
    
    # Set ChatML template
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
    
    return fast_tokenizer


def _generate_readme(algorithm: str, vocab_size: int, repo_id: str, dataset_name: str, sample_size: int) -> str:
    """Generate README content for the tokenizer."""
    template = README_TEMPLATE_BPE if algorithm.lower() == "bpe" else README_TEMPLATE_UNIGRAM
    return template.format(
        vocab_size=vocab_size,
        vocab_size_k=vocab_size//1000,
        repo_id=repo_id,
        dataset_name=dataset_name,
        sample_size=sample_size
    )


def _save_tokenizer(
    fast_tokenizer: PreTrainedTokenizerFast,
    repo_id: str,
    vocab_size: int,
    dataset_name: str,
    corpus_size: int,
    algorithm: str,
    push_to_hub: bool,
    local_dir: str = "./tokenizers"
):
    """Save tokenizer locally or push to Hugging Face Hub."""
    if push_to_hub:
        print(f"Pushing to Hub: {repo_id}...")
        try:
            fast_tokenizer.push_to_hub(repo_id)
            
            readme_content = _generate_readme(algorithm, vocab_size, repo_id, dataset_name, corpus_size)
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
            print("Saving locally instead...")
            _save_locally(fast_tokenizer, vocab_size, local_dir)
    else:
        _save_locally(fast_tokenizer, vocab_size, local_dir)


def _save_locally(fast_tokenizer: PreTrainedTokenizerFast, vocab_size: int, local_dir: str):
    """Save tokenizer to local directory."""
    output_path = os.path.join(local_dir, str(vocab_size))
    print(f"Saving locally to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    fast_tokenizer.save_pretrained(output_path)


def train_and_save_tokenizer(
    corpus: List[str],
    vocab_size: int,
    repo_id: str,
    dataset_name: str,
    algorithm: str = "unigram",
    push_to_hub: bool = False,
    local_dir: str = "./tokenizers"
):
    """
    Train and save a tokenizer (Unigram or BPE) on a pre-loaded corpus.
    
    Args:
        corpus: List of training text strings
        vocab_size: Target vocabulary size
        repo_id: HuggingFace repository ID
        dataset_name: Name of the dataset used for training
        algorithm: Either "unigram" or "bpe"
        push_to_hub: Whether to push to HuggingFace Hub
        local_dir: Local directory for saving if not pushing to hub
    """
    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()} Tokenizer: {repo_id}")
    print(f"Vocab Size: {vocab_size}")
    print(f"{'='*60}\n")

    special_tokens = get_special_tokens(add_chat_tokens=True)
    
    # Train tokenizer based on algorithm
    if algorithm.lower() == "bpe":
        tokenizer = _train_bpe_tokenizer(corpus, vocab_size, special_tokens)
    else:
        tokenizer = _train_unigram_tokenizer(corpus, vocab_size, special_tokens)
    
    # Wrap in Transformers format
    print("Wrapping in Transformers format...")
    fast_tokenizer = _wrap_tokenizer(tokenizer, special_tokens)
    
    # Save or push to hub
    _save_tokenizer(
        fast_tokenizer=fast_tokenizer,
        repo_id=repo_id,
        vocab_size=vocab_size,
        dataset_name=dataset_name,
        corpus_size=len(corpus),
        algorithm=algorithm,
        push_to_hub=push_to_hub,
        local_dir=local_dir
    )

def _format_sample_size(sample_size: int) -> str:
    """Format sample size for repo naming (e.g., 1000000 -> '1M', 500000 -> '500k')."""
    if sample_size >= 1_000_000:
        return f"{sample_size//1_000_000}M"
    return f"{sample_size//1000}k"


def _build_repo_name(user_handle: str, algorithm: str, vocab_size: int, sample_size: int) -> str:
    """Build HuggingFace repository name."""
    algo_short = "bpe" if algorithm == "bpe" else "unigram"
    samples_str = _format_sample_size(sample_size)
    return f"{user_handle}/minipile-{algo_short}-{vocab_size//1000}k-{samples_str}"


def _get_algorithms_to_train(algorithm_arg: str) -> List[str]:
    """Parse algorithm argument and return list of algorithms to train."""
    if algorithm_arg == "both":
        return ["unigram", "bpe"]
    return [algorithm_arg]


def main():
    parser = argparse.ArgumentParser(description="Train custom tokenizer (Unigram or BPE)")
    parser.add_argument("--dataset", type=str, default="JeanKaddour/minipile", help="HuggingFace dataset name")
    parser.add_argument("--sample_size", type=int, default=1_000_000, help="Number of samples to load")
    parser.add_argument("--vocab_sizes", type=int, nargs="+", default=[32000, 64000], help="List of vocab sizes to train")
    parser.add_argument("--algorithm", type=str, choices=["unigram", "bpe", "both"], default="unigram", 
                        help="Tokenization algorithm: 'unigram', 'bpe', or 'both' (trains both for comparison)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--user_handle", type=str, default="ksopyla", help="HF username for repo creation")
    
    args = parser.parse_args()
    setup_HF_environment()

    # Load and prepare corpus once (shared across all training runs)
    corpus = prepare_training_corpus(args.dataset, args.sample_size)
    
    # Train tokenizers for each algorithm and vocab size
    algorithms_to_train = _get_algorithms_to_train(args.algorithm)
    
    for algorithm in algorithms_to_train:
        for vocab_size in args.vocab_sizes:
            repo_name = _build_repo_name(args.user_handle, algorithm, vocab_size, args.sample_size)
            
            train_and_save_tokenizer(
                corpus=corpus,
                vocab_size=vocab_size,
                repo_id=repo_name,
                dataset_name=args.dataset,
                algorithm=algorithm,
                push_to_hub=args.push_to_hub
            )

if __name__ == "__main__":
    main()
