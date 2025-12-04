
import os
import sys
import argparse
import math
import torch
import evaluate
import numpy as np
import wandb
import platform
import socket
from datetime import datetime
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

def calculate_compression_ratio(tokenizer, dataset):
    """
    Calculate compression ratio: Total Characters / Total Tokens.
    Higher is generally better (more information per token).
    Uses pre-truncated text from dataset.
    
    IMPORTANT: Does NOT use padding or special tokens in tokenization to get accurate 
    compression metrics based on actual content tokens only.
    """
    
    def tokenize_and_count(examples):
        """Batch processing function for efficient tokenization"""
        char_counts = [len(text) for text in examples['text']]
        token_counts = []
        for text in examples['text']:
            # Tokenize WITHOUT padding or special tokens for accurate compression ratio
            tokens = tokenizer.encode(text, add_special_tokens=False, padding=False, truncation=False)
            token_counts.append(len(tokens))
        return {"char_count": char_counts, "token_count": token_counts}
    
    # Use batched map for speed
    result = dataset.map(tokenize_and_count, batched=True, remove_columns=["text"])
    
    total_chars = sum(result["char_count"])
    total_tokens = sum(result["token_count"])
    
    if total_tokens == 0:
        return 0.0
        
    return total_chars / total_tokens

def train_small_model_and_get_perplexity(tokenizer, tokenizer_name, train_dataset, eval_dataset, epochs=2):
    """
    Train a small Transformer model from scratch and calculate perplexity.
    This is a proxy for how 'learnable' the tokenization is.
    Uses pre-truncated text from dataset.
    """
    print(f"Training small model for {tokenizer_name} with epochs={epochs}")
    
    # Tokenize the preprocessed datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=True)
        
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Config Small Model (Tiny GPT-2 style)
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 2
    config.n_head = 4
    config.n_embd = 128
    config.vocab_size = len(tokenizer)

    model = AutoModelForCausalLM.from_config(config)

    # print number of total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}={total_params/1e6:.2f}M")


    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    

    
    # Training Arguments
    # Create a unique run name for this perplexity training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer_short_name = tokenizer_name.split('/')[-1]  # Remove organization prefix
    run_name = f"ppl-{tokenizer_short_name}-{len(train_dataset)//1000}k-{epochs}epochs"
    
    output_dir = os.path.join(os.getenv("PROJECT_ROOT", os.getcwd()), "Cache", "Training", "perplexity_eval", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=32,
        learning_rate=5e-4,
        num_train_epochs=epochs,
        logging_steps=100,
        save_steps=10**5, # Don't save checkpoints to save space
        report_to="wandb",
        use_cpu=not torch.cuda.is_available(),
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),  # Use fp16 if GPU available (more compatible than bf16)
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Evaluate Perplexity
    print(f"Evaluating perplexity for {tokenizer_name} with epochs={epochs}")
    eval_results = trainer.evaluate(tokenized_eval)
    perplexity = math.exp(eval_results['eval_loss'])
    print(f"Perplexity: {perplexity}")
    
    # Cleanup output directory to save space
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    return perplexity

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tokenizers: BLEU, Compression, Perplexity")
    parser.add_argument("--minipile_samples", type=int, default=10000, help="Number of samples for compression ratio")
    parser.add_argument("--skip_ppl", action="store_true", help="Skip perplexity evaluation (slow)")
    args = parser.parse_args()
    
    console = Console()
    console.print("[bold green]Starting Comprehensive Tokenizer Evaluation[/bold green]")
    
    # Load and preprocess dataset ONCE
    console.print("\n[bold]Loading and preprocessing dataset...[/bold]")
    dataset_train = None
    dataset_test = None
    try:
        # HF datasets will automatically use HF_DATASETS_CACHE env var if set
        dataset_train = load_dataset("JeanKaddour/minipile", split="train")
        dataset_test = load_dataset("JeanKaddour/minipile", split="test")

        # Select subsets first to save memory
        dataset_train = dataset_train.select(range(min(len(dataset_train), args.minipile_samples)))
        dataset_test = dataset_test.select(range(min(len(dataset_test), 5000)))
        
        # Truncate text to max 1000 characters for efficient processing
        def truncate_text(examples):
            return {"text": [text[:1000] for text in examples["text"]]}
        
        dataset_train = dataset_train.map(truncate_text, batched=True)
        dataset_test = dataset_test.map(truncate_text, batched=True)
        
        console.print(f"Loaded and truncated train dataset: {len(dataset_train)} samples")
        console.print(f"Loaded and truncated test dataset: {len(dataset_test)} samples")
        
    except Exception as e:
        console.print(f"[red]Could not load MiniPile: {e}[/red]")
        dataset_train = None
        dataset_test = None
    
    # Generate standardized W&B metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = socket.gethostname()
    
    wandb_tags = [
        "tokenizer-evaluation",
        "benchmark",
        hostname,
        f"samples-{args.minipile_samples}"
    ]
    
    if args.skip_ppl:
        wandb_tags.append("no-perplexity")
    
    # Initialize W&B with standard project settings
    wandb.init(
        project="MrCogito",
        job_type="tokenizer-evaluation",
        group=f"tokenizers-eval-{args.minipile_samples//1000}k",
        name=f"tokenizers-eval-{args.minipile_samples//1000}k",
        tags=wandb_tags,
        config={
            "minipile_samples": args.minipile_samples,
            "skip_ppl": args.skip_ppl,
            "evaluation_type": "comprehensive_tokenizer_benchmark",
            "timestamp": timestamp,
            "hostname": hostname,
            "platform": platform.platform()
        }
    )

    # 1. Load Tokenizers
    # You can expand this list
    tokenizer_names = [
        "bert-base-cased",
        "xlnet-base-cased",
        "gpt2",
        "meta-llama/Llama-3.2-1B-instruct",
        "answerdotai/ModernBERT-base",
        "ksopyla/minipile-english-unigram-32k",
        "ksopyla/minipile-english-unigram-64k",
        "ksopyla/minipile-unigram-32k-50k",
        "ksopyla/minipile-unigram-65k-50k",
        "ksopyla/minipile-unigram-32k-100k",
        "ksopyla/minipile-unigram-65k-100k"

    ]

    
    tokenizers = {}
    for name in tokenizer_names:
        try:
            console.print(f"Loading {name}...")
            # HF tokenizers will automatically use HF_HOME env var if set
            tokenizers[name] = AutoTokenizer.from_pretrained(name, token=hf_token)
                
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
    compression_ratios = {}
    if dataset_train is not None:
        for name, tok in tokenizers.items():
            console.print(f"Calculating compression for {name}...")
            ratio = calculate_compression_ratio(tok, dataset_train)
            compression_ratios[name] = ratio
    
    # 4. Perplexity (Optional)
    perplexities = {}
    if not args.skip_ppl and dataset_train is not None and dataset_test is not None:
        console.print(f"\n[bold]Evaluating Downstream Perplexity [/bold]")
        
        
        for name, tok in tokenizers.items():
            try:
                console.print(f"Training small model for {name} with vocab size {len(tok)}")
                ppl = train_small_model_and_get_perplexity(
                    tok,
                    tokenizer_name=name,
                    train_dataset=dataset_train,
                    eval_dataset=dataset_test,
                    epochs=2
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

    # Create a table for W&B
    wandb_table = wandb.Table(columns=["Tokenizer", "BLEU", "1-gram Precision", "Compression Ratio", "Perplexity"])

    for name in tokenizers.keys():
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

        # Log to W&B Table
        wandb_table.add_data(
            name,
            bleu['bleu'],
            bleu['precisions'][0],
            comp,
            ppl if ppl > 0 else None
        )
        
        # Log metrics directly for charts (using tokenizer name as prefix/group)
        wandb.log({
            f"{name}/bleu": bleu['bleu'],
            f"{name}/1gram_precision": bleu['precisions'][0],
            f"{name}/compression_ratio": comp,
            f"{name}/perplexity": ppl if ppl > 0 else None,
        })

        # Log summarized metrics for comparison (e.g. scatter plot ready)
        wandb.log({
            "tokenizer_name": name,
            "bleu": bleu['bleu'],
            "compression_ratio": comp,
            "perplexity": ppl if ppl > 0 else None,
        })
        
    console.print(table)
    
    # Log the main comparison table
    wandb.log({"tokenizer_evaluation_results": wandb_table})
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    main()

