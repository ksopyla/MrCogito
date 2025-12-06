
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
import logging
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)

# Add project root to path to import ground_truth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from playground.data.ground_truth import GROUND_TRUTH_MORPHEMS
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import GROUND_TRUTH_MORPHEMS. Please check the path.")
    GROUND_TRUTH_MORPHEMS = {}

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Setup module-level logger
logger = logging.getLogger(__name__)

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

def train_small_model_and_get_perplexity(tokenizer, tokenizer_name, train_dataset, eval_dataset, run_group_id, wandb_tags, epochs=2):
    """
    Train a small Transformer model from scratch and calculate perplexity.
    This is a proxy for how 'learnable' the tokenization is.
    Uses pre-truncated text from dataset.
    """
    logger.info(f"\nTraining small model for {tokenizer_name} with epochs={epochs}")
    

    
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Config Small Pythia Model (GPTNeoX architecture)
    # Using minimal configuration based on pythia-70m-deduped, scaled down for memory efficiency
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,            # Scaled down from 512 for memory
        num_hidden_layers=2,        # Minimal layers
        num_attention_heads=4,      # Must divide hidden_size evenly (256/4=64 per head)
        intermediate_size=1024,     # 4x hidden_size (standard FFN ratio)
        max_position_embeddings=512,
        rotary_pct=0.25,            # Pythia uses 25% rotary embeddings
        rotary_emb_base=10000,
        use_parallel_residual=True, # Pythia-specific optimization
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,  # Pythia doesn't tie embeddings
        _attn_implementation="sdpa"  # Use SDPA by default (Flash Attn 2 requires from_pretrained)
    )

    # Create model from config
    # Note: attn_implementation parameter only works with from_pretrained(), not __init__()
    # For models created from config, SDPA is used by default in PyTorch >= 2.1.1
    model = GPTNeoXForCausalLM(config)
    logger.info(f"Using SDPA attention (PyTorch native optimized attention)")

    # Print number of total parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params}={total_params/1e6:.2f}M")
    
    # Tokenize the preprocessed datasets
    # Using dynamic padding (pad to longest in batch, not max_length) for memory efficiency
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)
    
    logger.info(f"Tokenizing datasets for {tokenizer_name}")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Training Arguments
    # Create a unique run name for this perplexity training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer_short_name = tokenizer_name.split('/')[-1]  # Remove organization prefix
    run_name = f"ppl-{tokenizer_short_name}-{len(train_dataset)//1000}k-{epochs}epochs"
    
    output_dir = os.path.join(os.getenv("PROJECT_ROOT", os.getcwd()), "Cache", "Training", "perplexity_eval", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize nested W&B run for this tokenizer's training
    # This creates a separate run to track training curves
    tags = wandb_tags + ["perplexity", "tokenizer-training", tokenizer_short_name]
    ppl_run = wandb.init(
        project="MrCogito",
        job_type="perplexity-training",
        name=run_name,
        group=run_group_id,
        tags=tags,
        config={
            "tokenizer_name": tokenizer_name,
            "vocab_size": len(tokenizer),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "epochs": epochs,
            "model_type": "pythia-tiny",
            "hidden_size": 64,
            "num_layers": 2,
        },
        reinit=True  # Allow multiple init() calls
    )
    
    # Calculate eval_steps before creating TrainingArguments
    per_device_batch_size = 4
    gradient_accumulation_steps = 2
    # Calculate steps per epoch
    steps_per_epoch = len(tokenized_train) // (per_device_batch_size * gradient_accumulation_steps)
    eval_steps = max(100, int(0.25 * steps_per_epoch))  # Evaluate every 25% of epoch
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=6e-4,              # Pythia-recommended LR
        num_train_epochs=epochs,
        warmup_steps=100,                # Small warmup for stability
        weight_decay=0.1,                # Pythia default
        logging_steps=100,               # Log every 100 steps for detailed curves
        save_steps=10**8,                # Don't save checkpoints to save space
        report_to="wandb",
        use_cpu=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
        logging_first_step=True,         # Log first step for better visualization
        eval_strategy="steps",           # Changed to steps for more frequent eval
        eval_steps=eval_steps,           # Evaluate every 25% of epoch
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8  # Pad to multiple of 8 for tensor core efficiency
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,  # Add eval dataset for eval_strategy="epoch"
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Evaluate Perplexity
    logger.info(f"Evaluating perplexity for {tokenizer_name}")
    eval_results = trainer.evaluate(tokenized_eval)
    perplexity = math.exp(eval_results['eval_loss'])
    logger.info(f"Perplexity: {perplexity:.2f}")
    
    # Log final perplexity to this run's summary
    wandb.summary["final_perplexity"] = perplexity
    wandb.summary["final_eval_loss"] = eval_results['eval_loss']
    wandb.summary["total_params"] = total_params
    
    # Finish the nested W&B run
    wandb.finish()
    
    # Cleanup output directory to save space
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    del model, trainer
    torch.cuda.empty_cache()
        
    return perplexity, total_params

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tokenizers: BLEU, Compression, Perplexity")
    parser.add_argument("--minipile_samples", type=int, default=10000, help="Number of samples for compression ratio")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for perplexity training (default: 2)")
    parser.add_argument("--skip_ppl", action="store_true", help="Skip perplexity evaluation (slow)")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file (default: auto-generated)")
    args = parser.parse_args()
    
    # Setup logging to both console and file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_file is None:
        log_dir = os.path.join(os.getcwd(), "Cache", "Logs")
        os.makedirs(log_dir, exist_ok=True)
        args.log_file = os.path.join(log_dir, f"tokenizer_eval_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Important: This forces re-configuration of logging
    )
    
    # Also configure transformers logging to capture Trainer output
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)
    transformers_logger.addHandler(logging.FileHandler(args.log_file))
    transformers_logger.addHandler(logging.StreamHandler(sys.stdout))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {args.log_file}")
    
    logger.info("="*60)
    logger.info("Starting Comprehensive Tokenizer Evaluation")
    logger.info("="*60)
    
    # Load and preprocess dataset ONCE
    logger.info("\nLoading and preprocessing dataset...")
    dataset_train = None
    dataset_test = None
    try:
        # HF datasets will automatically use HF_DATASETS_CACHE env var if set
        dataset_train = load_dataset("JeanKaddour/minipile", split="train")
        dataset_test = load_dataset("JeanKaddour/minipile", split="test")

        # Select subsets first to save memory (using random sampling with fixed seed for reproducibility)
        rand_seed = 42
        dataset_train = dataset_train.shuffle(seed=rand_seed).select(range(min(len(dataset_train), args.minipile_samples)))
        dataset_test = dataset_test.shuffle(seed=rand_seed).select(range(min(len(dataset_test), 5000)))
        
        # Truncate text to max 1000 characters for efficient processing
        def truncate_text(examples):
            return {"text": [text[:1000] for text in examples["text"]]}
        
        dataset_train = dataset_train.map(truncate_text, batched=True)
        dataset_test = dataset_test.map(truncate_text, batched=True)
        
        logger.info(f"Loaded and truncated train dataset: {len(dataset_train)} samples")
        logger.info(f"Loaded and truncated test dataset: {len(dataset_test)} samples")
        
    except Exception as e:
        logger.error(f"Could not load MiniPile: {e}")
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
    
    # Common group ID for all runs in this execution
    # Format: tokenizer-eval-{sample_size}-epoch-{epoch}
    run_group_id = f"tokenizers-eval-{args.minipile_samples//1000}k-epoch-{args.epochs}"
    
    # Note: We delay W&B initialization until needed.
    # 1. Perplexity runs will be initialized individually inside the training function
    # 2. The final summary run will be initialized at the end

    # 1. Load Tokenizers
    logger.info("\nLoading tokenizers...")
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
            logger.info(f"Loading {name}...")
            # HF tokenizers will automatically use HF_HOME env var if set
            tokenizers[name] = AutoTokenizer.from_pretrained(name, token=hf_token)
                
            # Ensure pad token for training
            if tokenizers[name].pad_token is None:
                tokenizers[name].pad_token = tokenizers[name].eos_token
                
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    logger.info(f"Successfully loaded {len(tokenizers)} tokenizers")

    # 2. Morphology BLEU
    logger.info("\n" + "="*60)
    logger.info("Evaluating Morphology (BLEU)")
    logger.info("="*60)
    bleu_scores = evaluate_morphology_bleu(tokenizers, GROUND_TRUTH_MORPHEMS)
    
    # 3. Compression Ratio
    logger.info("\n" + "="*60)
    logger.info(f"Evaluating Compression Ratio (on {args.minipile_samples} samples)")
    logger.info("="*60)
    compression_ratios = {}
    if dataset_train is not None:
        for name, tok in tokenizers.items():
            logger.info(f"Calculating compression for {name}...")
            ratio = calculate_compression_ratio(tok, dataset_train)
            compression_ratios[name] = ratio
            logger.info(f"  Compression ratio: {ratio:.2f}")
    
    # 4. Perplexity (Optional)
    perplexities = {}
    model_params = {}  # Store actual model parameters
    if not args.skip_ppl and dataset_train is not None and dataset_test is not None:
        logger.info("\n" + "="*60)
        logger.info("Evaluating Downstream Perplexity")
        logger.info("="*60)
        
        for name, tok in tokenizers.items():
            try:
                logger.info(f"\nTraining small model for {name} (vocab size: {len(tok)})")
                ppl, params = train_small_model_and_get_perplexity(
                    tok,
                    tokenizer_name=name,
                    train_dataset=dataset_train,
                    eval_dataset=dataset_test,
                    run_group_id=run_group_id,
                    wandb_tags=wandb_tags,
                    epochs=args.epochs
                )
                perplexities[name] = ppl
                model_params[name] = params
                logger.info(f"  Final perplexity: {ppl:.2f}")
                logger.info(f"  Model parameters: {params/1e6:.2f}M")
            except Exception as e:
                logger.error(f"Failed PPL for {name}: {e}")
                perplexities[name] = float('nan')
                model_params[name] = 0
        
    else:
        for name in tokenizers:
            perplexities[name] = 0.0
            model_params[name] = 0

    # 5. Report and Log to W&B
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS - Creating Summary Run")
    logger.info("="*60)
    
    # Initialize the Summary Run
    summary_run = wandb.init(
        project="MrCogito",
        job_type="tokenizer-evaluation-summary",
        group=run_group_id,
        name=f"summary-{args.minipile_samples//1000}k",
        tags=wandb_tags + ["summary"],
        config={
            "minipile_samples": args.minipile_samples,
            "skip_ppl": args.skip_ppl,
            "evaluation_type": "comprehensive_tokenizer_benchmark",
            "timestamp": timestamp,
            "hostname": hostname,
            "platform": platform.platform()
        },
        reinit=True
    )
    
    # Create a table for W&B with model parameters
    wandb_table = wandb.Table(columns=["Tokenizer", "Vocab Size", "Model Params (M)", "BLEU", "1-gram Precision", "Compression Ratio", "Perplexity"])
    
    # Prepare data for bar charts
    tokenizer_names_list = []
    bleu_values = []
    compression_values = []
    perplexity_values = []
    vocab_sizes = []

    # Print header
    logger.info(f"\n{'Tokenizer':<45} {'BLEU':>8} {'1-gram':>8} {'Compression':>12} {'Perplexity':>12} {'Vocab':>10} {'Params(M)':>10}")
    logger.info("-" * 110)

    for name in tokenizers.keys():
        bleu = bleu_scores.get(name, {'bleu': 0, 'precisions': [0,0,0]})
        comp = compression_ratios.get(name, 0)
        ppl = perplexities.get(name, 0)
        vocab_size = len(tokenizers[name])
        params = model_params.get(name, 0)
        model_params_m = params / 1_000_000 if params > 0 else 0
        
        # Log to console/file
        ppl_str = f"{ppl:.2f}" if ppl > 0 and not math.isnan(ppl) else "N/A"
        params_str = f"{model_params_m:.2f}" if params > 0 else "N/A"
        logger.info(f"{name:<45} {bleu['bleu']:>8.4f} {bleu['precisions'][0]:>8.4f} {comp:>12.2f} {ppl_str:>12} {vocab_size:>10} {params_str:>10}")

        # Add to W&B table
        wandb_table.add_data(
            name,
            vocab_size,
            f"{model_params_m:.2f}" if params > 0 else "N/A",
            bleu['bleu'],
            bleu['precisions'][0],
            comp,
            ppl if ppl > 0 and not math.isnan(ppl) else None
        )
        
        # Collect data for bar charts
        tokenizer_names_list.append(name.split('/')[-1])  # Short names for X-axis
        bleu_values.append(bleu['bleu'])
        compression_values.append(comp)
        perplexity_values.append(ppl if ppl > 0 and not math.isnan(ppl) else None)
        vocab_sizes.append(vocab_size)
        
        # Log individual metrics to W&B summary (creates bar charts)
        wandb.summary[f"{name}/bleu"] = bleu['bleu']
        wandb.summary[f"{name}/1gram_precision"] = bleu['precisions'][0]
        wandb.summary[f"{name}/compression_ratio"] = comp
        wandb.summary[f"{name}/perplexity"] = ppl if ppl > 0 and not math.isnan(ppl) else None
        wandb.summary[f"{name}/vocab_size"] = vocab_size
    
    logger.info("-" * 110)
    
    # Log the main comparison table
    wandb.log({"tokenizer_evaluation_table": wandb_table})
    
    # Create bar chart data for easy comparison
    wandb.log({
        "bleu_comparison": wandb.plot.bar(
            wandb.Table(data=list(zip(tokenizer_names_list, bleu_values)), columns=["Tokenizer", "BLEU"]),
            "Tokenizer", "BLEU", title="BLEU Score Comparison"
        ),
        "compression_comparison": wandb.plot.bar(
            wandb.Table(data=list(zip(tokenizer_names_list, compression_values)), columns=["Tokenizer", "Compression"]),
            "Tokenizer", "Compression", title="Compression Ratio Comparison"
        ),
    })
    
    # Log perplexity comparison if available
    ppl_data = [(name, ppl) for name, ppl in zip(tokenizer_names_list, perplexity_values) if ppl is not None]
    if ppl_data:
        wandb.log({
            "perplexity_comparison": wandb.plot.bar(
                wandb.Table(data=ppl_data, columns=["Tokenizer", "Perplexity"]),
                "Tokenizer", "Perplexity", title="Perplexity Comparison (Lower is Better)"
            )
        })
    
    # Finish the run
    logger.info(f"\nEvaluation complete. Results logged to W&B and {args.log_file}")
    wandb.finish()

if __name__ == "__main__":
    main()

