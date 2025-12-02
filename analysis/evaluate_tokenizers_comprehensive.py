
import os
import sys
import argparse
import math
import torch
import evaluate
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)
from rich.console import Console
from rich.table import Table
from rich import box

# Add project root to path to import ground_truth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from playground.data.ground_truth import GROUND_TRUTH_MORPHEMS
except ImportError:
    print("Warning: Could not import GROUND_TRUTH_MORPHEMS. Please check the path.")
    GROUND_TRUTH_MORPHEMS = {}

DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
TOKENIZER_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def get_tokenizer_predictions(tokenizer, word):
    """Get tokenizer predictions for a word"""
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if hasattr(tokenizer, "convert_ids_to_tokens"):
             tokens = tokenizer.convert_ids_to_tokens(tokens)
        elif hasattr(tokenizer, "decode"):
             # Fallback for some tokenizers
             tokens = [tokenizer.decode([t]) for t in tokens]
    elif hasattr(tokenizer, "tokenize"):
        tokens = tokenizer.tokenize(word)
    else:
        return word
    
    cleaned_tokens = []
    for token in tokens:
        # Clean up common subword markers
        token = token.replace('Ġ', '').replace(' ', '').replace('##', '').replace(' ', '')
        if token.strip():
            cleaned_tokens.append(token)
    
    return ' '.join(cleaned_tokens)

def evaluate_morphology_bleu(tokenizers_dict, ground_truth_morphems):
    """Evaluate tokenizers using BLEU score on morphological segmentation"""
    bleu = evaluate.load("bleu")
    bleu_scores = {}
    
    for tokenizer_name, tokenizer in tokenizers_dict.items():
        predictions = []
        references = []
        
        for word, refs in ground_truth_morphems.items():
            pred = get_tokenizer_predictions(tokenizer, word)
            predictions.append(pred)
            references.append(refs)
        
        score = bleu.compute(predictions=predictions, references=references, max_order=3)
        bleu_scores[tokenizer_name] = score
    
    return bleu_scores

def calculate_compression_ratio(tokenizer, dataset, num_samples=10000):
    """
    Calculate compression ratio: Total Characters / Total Tokens.
    Higher is generally better (more information per token).
    """
    total_chars = 0
    total_tokens = 0
    
    # Take a subset
    subset = dataset.select(range(min(len(dataset), num_samples)))
    
    for item in subset:
        text = item['text']
        total_chars += len(text)
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
        
    if total_tokens == 0:
        return 0.0
        
    return total_chars / total_tokens

def train_small_model_and_get_perplexity(tokenizer, dataset_name="JeanKaddour/minipile", subset_size=100000, max_steps=1000):
    """
    Train a small Transformer model from scratch and calculate perplexity.
    This is a proxy for how 'learnable' the tokenization is.
    """
    print(f"Training small model for tokenizer with subset_size={subset_size}, max_steps={max_steps}")
    
    # 1. Load Dataset
    dataset = load_dataset(dataset_name, split="train", cache_dir=DATASET_CACHE_DIR)
    if subset_size and subset_size < len(dataset):
        dataset = dataset.select(range(subset_size))
        
    # 2. Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 3. Config Small Model (Tiny GPT-2 style)
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 2
    config.n_head = 4
    config.n_embd = 128
    config.vocab_size = len(tokenizer)
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    
    model = AutoModelForCausalLM.from_config(config)
    
    # 4. Training Arguments
    output_dir = os.path.join(RESULTS_DIR, "temp_trainer")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        max_steps=max_steps,
        logging_steps=max_steps//10,
        save_steps=max_steps + 1, # Don't save checkpoints
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        prediction_loss_only=True,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # 5. Evaluate Perplexity
    # Use a small validation set from minipile test
    eval_dataset = load_dataset(dataset_name, split="test", cache_dir=DATASET_CACHE_DIR)
    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), 1000))) # Small eval set for speed
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    eval_results = trainer.evaluate(eval_tokenized)
    perplexity = math.exp(eval_results['eval_loss'])
    
    # Cleanup
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    return perplexity

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tokenizers: BLEU, Compression, Perplexity")
    parser.add_argument("--minipile_samples", type=int, default=10000, help="Number of samples for compression ratio")
    parser.add_argument("--ppl_samples", type=int, default=10000, help="Number of samples for perplexity training")
    parser.add_argument("--ppl_steps", type=int, default=100, help="Training steps for perplexity")
    parser.add_argument("--skip_ppl", action="store_true", help="Skip perplexity evaluation (slow)")
    args = parser.parse_args()
    
    console = Console()
    console.print("[bold green]Starting Comprehensive Tokenizer Evaluation[/bold green]")
    
    # 1. Load Tokenizers
    # You can expand this list
    tokenizer_names = [
        "bert-base-cased",
        "xlnet-base-cased",
        "gpt2",
        "meta-llama/Llama-3.2-1B-instruct",
        "ksopyla/minipile-english-unigram-32k",
        "ksopyla/minipile-english-unigram-64k"
    ]
    
    tokenizers = {}
    for name in tokenizer_names:
        try:
            console.print(f"Loading {name}...")
            tokenizers[name] = AutoTokenizer.from_pretrained(name, cache_dir=TOKENIZER_CACHE_DIR, token=hf_token)
                
            # Ensure pad token for training
            if tokenizers[name].pad_token is None:
                tokenizers[name].pad_token = tokenizers[name].eos_token
                
        except Exception as e:
            console.print(f"[red]Failed to load {name}: {e}[/red]")

    # 2. Morphology BLEU
    console.print("\n[bold]Evaluating Morphology (BLEU)[/bold]")
    bleu_scores = evaluate_morphology_bleu(tokenizers, GROUND_TRUTH_MORPHEMS)
    
    # 3. Compression Ratio
    console.print(f"\n[bold]Evaluating Compression Ratio (on {args.minipile_samples} samples)[/bold]")
    try:
        dataset = load_dataset("JeanKaddour/minipile", split="train", cache_dir=DATASET_CACHE_DIR)
    except Exception as e:
        console.print(f"[red]Could not load MiniPile: {e}. Using dummy data?[/red]")
        # Fallback logic could go here
        dataset = []

    compression_ratios = {}
    if len(dataset) > 0:
        for name, tok in tokenizers.items():
            console.print(f"Calculating compression for {name}...")
            ratio = calculate_compression_ratio(tok, dataset, num_samples=args.minipile_samples)
            compression_ratios[name] = ratio
    
    # 4. Perplexity (Optional)
    perplexities = {}
    if not args.skip_ppl and len(dataset) > 0:
        console.print(f"\n[bold]Evaluating Downstream Perplexity (Train {args.ppl_steps} steps on {args.ppl_samples} samples)[/bold]")
        for name, tok in tokenizers.items():
            try:
                console.print(f"Training small model for {name}...")
                ppl = train_small_model_and_get_perplexity(
                    tok, 
                    dataset_name="JeanKaddour/minipile", 
                    subset_size=args.ppl_samples, 
                    max_steps=args.ppl_steps
                )
                perplexities[name] = ppl
            except Exception as e:
                console.print(f"[red]Failed PPL for {name}: {e}[/red]")
                perplexities[name] = float('nan')
    else:
        for name in tokenizers:
            perplexities[name] = 0.0

    # 5. Report
    table = Table(title="Tokenizer Evaluation Results")
    table.add_column("Tokenizer", style="cyan")
    table.add_column("BLEU", justify="right", style="green")
    table.add_column("1-gram", justify="right", style="magenta")
    table.add_column("Compression", justify="right", style="blue")
    table.add_column("Perplexity", justify="right", style="red")

    for name in tokenizer_names:
        if name not in tokenizers: continue
        
        bleu = bleu_scores.get(name, {'bleu': 0, 'precisions': [0,0,0]})
        comp = compression_ratios.get(name, 0)
        ppl = perplexities.get(name, 0)
        
        table.add_row(
            name,
            f"{bleu['bleu']:.4f}",
            f"{bleu['precisions'][0]:.4f}",
            f"{comp:.2f}",
            f"{ppl:.2f}" if ppl > 0 else "N/A"
        )
        
    console.print(table)

if __name__ == "__main__":
    main()

