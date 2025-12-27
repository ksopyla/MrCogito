
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
import json
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

def load_mwts_from_files(mwt_files, max_mwts=None):
    pass  # Removed MWT functionality

def tokenizer_vocab_has_token(tokenizer, token: str) -> bool:
    pass  # Removed MWT functionality

def get_tokens_for_text(tokenizer, text: str):
    """
    Return token strings (best-effort) for a given text, without adding special tokens.
    """
    try:
        if hasattr(tokenizer, "tokenize"):
            # Most HF tokenizers expose tokenize(text) -> List[str]
            return tokenizer.tokenize(text)
    except Exception:
        pass

    try:
        ids = tokenizer.encode(text, add_special_tokens=False, padding=False, truncation=False)
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            return tokenizer.convert_ids_to_tokens(ids)
        if hasattr(tokenizer, "decode"):
            return [tokenizer.decode([i]) for i in ids]
    except Exception:
        pass

    return []

def evaluate_mwt_utilization(tokenizer, dataset, mwt_set: set, top_k: int = 25):
    pass # Removed MWT functionality

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

def is_code_document(text: str) -> bool:
    """
    Heuristic to detect if a document is primarily code.
    Based on common code markers across multiple languages.
    """
    if not text or len(text) < 10:
        return False
    
    code_markers = [
        'def ', 'class ', 'import ', 'function ', '#include', 'SELECT ', 
        'INSERT ', 'fn ', 'const ', 'let ', 'var ', '{', '}', '();', 
        'std::', 'return ', 'void ', 'int ', 'async ', '__init__', 
        'from ', '::', 'VALUES', '->', '=>', 'public ', 'private ',
        'namespace ', 'template ', 'lambda ', 'try:', 'except ',
        'if __name__', 'package ', 'interface ', 'extends ', 'implements '
    ]
    
    marker_count = sum(1 for m in code_markers if m in text)
    # Require at least 2 markers to classify as code (reduces false positives)
    return marker_count >= 2


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


def calculate_code_compression_ratio(tokenizer, dataset, min_code_samples: int = 100):
    """
    Calculate compression ratio specifically on code documents.
    This metric is crucial for evaluating tokenizers on programming tasks.
    
    Args:
        tokenizer: Tokenizer to evaluate
        dataset: Dataset to evaluate on
        min_code_samples: Minimum number of code samples required to compute metric
        
    Returns:
        Compression ratio (chars/tokens) for code, or None if insufficient code samples
    """
    
    def filter_and_tokenize_code(examples):
        """Filter code documents and tokenize them"""
        code_texts = []
        code_char_counts = []
        code_token_counts = []
        
        for text in examples['text']:
            if is_code_document(text):
                code_texts.append(text)
                code_char_counts.append(len(text))
                # Tokenize WITHOUT padding or special tokens
                tokens = tokenizer.encode(text, add_special_tokens=False, padding=False, truncation=False)
                code_token_counts.append(len(tokens))
        
        return {
            "code_char_count": code_char_counts,
            "code_token_count": code_token_counts
        }
    
    # Process dataset to find code samples
    code_stats = dataset.map(
        filter_and_tokenize_code,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Filtering code samples"
    )
    
    total_code_chars = sum(code_stats["code_char_count"])
    total_code_tokens = sum(code_stats["code_token_count"])
    num_code_samples = len(code_stats["code_char_count"])
    
    if num_code_samples < min_code_samples:
        logger.warning(f"Only found {num_code_samples} code samples (minimum: {min_code_samples}). Skipping code compression metric.")
        return None
    
    if total_code_tokens == 0:
        return None
        
    compression_ratio = total_code_chars / total_code_tokens
    logger.info(f"  Code samples: {num_code_samples}, Code compression ratio: {compression_ratio:.2f}")
    
    return compression_ratio

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

    tokenizer_short_name = tokenizer_name.split('/')[-1]  # Remove organization prefix
    run_name = f"ppl-{tokenizer_short_name}-{len(train_dataset)//1000}k-{epochs}epochs"
    
    output_dir = os.path.join(os.getenv("PROJECT_ROOT", os.getcwd()), "Cache", "Training", "perplexity_eval", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize nested W&B run for this tokenizer's training
    # This creates a separate run to track training curves
    tags = wandb_tags + ["perplexity", "tokenizer-training", tokenizer_short_name, "GPTNeoX"]
    ppl_run = wandb.init(
        project="MrCogito",
        mode="online",
        sync_tensorboard=True,
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
        # Custom Unigram tokenizers (trained on MiniPile)
        "ksopyla/minipile-english-unigram-32k",
        "ksopyla/minipile-english-unigram-64k",
        "ksopyla/minipile-unigram-32k-50k",
        "ksopyla/minipile-unigram-65k-50k",
        "ksopyla/minipile-unigram-32k-100k",
        "ksopyla/minipile-unigram-65k-100k",
        # Custom BPE tokenizers (trained on MiniPile) - add these after training with --algorithm bpe
        # "ksopyla/minipile-bpe-32k-100k",
        # "ksopyla/minipile-bpe-50k-100k",
        # "ksopyla/minipile-bpe-65k-100k",
        "cl100k_base" # OpenAI GPT-4
    ]

    
    tokenizers = {}
    tiktoken_encoder = None  # Store tiktoken encoder separately for compression metrics
    
    for name in tokenizer_names:
        try:
            logger.info(f"Loading {name}...")
            
            if name == "cl100k_base":
                # Special handling for tiktoken (OpenAI GPT-4 tokenizer)
                # We'll use it for compression metrics but skip perplexity training
                # since it doesn't have a direct HF Trainer-compatible interface
                try:
                    import tiktoken
                    tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
                    logger.info("Loaded tiktoken cl100k_base encoder (will use for compression metrics only)")
                    # Create a minimal wrapper for compression metrics
                    # We'll handle this specially in compression functions
                    class TikTokenWrapper:
                        def __init__(self, encoder):
                            self.encoder = encoder
                            self.vocab_size = encoder.n_vocab
                        def encode(self, text, add_special_tokens=False, padding=False, truncation=False):
                            return self.encoder.encode(text)
                        def __len__(self):
                            return self.vocab_size
                    tokenizers[name] = TikTokenWrapper(tiktoken_encoder)
                except Exception as e:
                    logger.warning(f"Could not load tiktoken cl100k_base: {e}. Skipping.")
                    continue
            else:
                # HF tokenizers will automatically use HF_HOME env var if set
                tokenizers[name] = AutoTokenizer.from_pretrained(name, token=hf_token)
                
            # Ensure pad token for training (skip for tiktoken wrapper)
            if hasattr(tokenizers[name], 'pad_token') and tokenizers[name].pad_token is None:
                if hasattr(tokenizers[name], 'eos_token') and tokenizers[name].eos_token is not None:
                    tokenizers[name].pad_token = tokenizers[name].eos_token
                
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    logger.info(f"Successfully loaded {len(tokenizers)} tokenizers")

    # 2. Morphology BLEU
    logger.info("\n" + "="*60)
    logger.info("Evaluating Morphology (BLEU)")
    logger.info("="*60)
    bleu_scores = evaluate_morphology_bleu(tokenizers, GROUND_TRUTH_MORPHEMS)
    
    # 3. Compression Ratio (General)
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

    # 3b. Code-Specific Compression Ratio
    logger.info("\n" + "="*60)
    logger.info(f"Evaluating Code Compression Ratio")
    logger.info("="*60)
    code_compression_ratios = {}
    if dataset_train is not None:
        for name, tok in tokenizers.items():
            logger.info(f"Calculating code compression for {name}...")
            code_ratio = calculate_code_compression_ratio(tok, dataset_train, min_code_samples=50)
            code_compression_ratios[name] = code_ratio
            if code_ratio is not None:
                logger.info(f"  Code compression ratio: {code_ratio:.2f}")
            else:
                logger.info(f"  Code compression ratio: N/A (insufficient code samples)")

    # 4. Perplexity (Optional)
    # Note: Skip tiktoken for perplexity training as it doesn't have HF Trainer compatibility
    perplexities = {}
    model_params = {}  # Store actual model parameters
    if not args.skip_ppl and dataset_train is not None and dataset_test is not None:
        logger.info("\n" + "="*60)
        logger.info("Evaluating Downstream Perplexity")
        logger.info("="*60)
        
        for name, tok in tokenizers.items():
            # Skip tiktoken for perplexity (requires HF Trainer interface)
            if name == "cl100k_base":
                logger.info(f"Skipping perplexity for {name} (tiktoken not compatible with HF Trainer)")
                perplexities[name] = float('nan')
                model_params[name] = 0
                continue
                
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
    wandb_table = wandb.Table(columns=[
        "Tokenizer", "Vocab Size", "Model Params (M)",
        "BLEU", "1-gram Precision", "Compression Ratio", "Code Compression Ratio", "Perplexity"
    ])
    
    # Prepare data for bar charts
    tokenizer_names_list = []
    bleu_values = []
    compression_values = []
    perplexity_values = []
    vocab_sizes = []

    # Print header
    logger.info(f"\n{'Tokenizer':<45} {'BLEU':>8} {'1-gram':>8} {'Compression':>12} {'Code Comp.':>18} {'Perplexity':>12} {'Vocab':>10} {'Params(M)':>10}")
    logger.info("-" * 130)

    for name in tokenizers.keys():
        bleu = bleu_scores.get(name, {'bleu': 0, 'precisions': [0,0,0]})
        comp = compression_ratios.get(name, 0)
        code_comp = code_compression_ratios.get(name, None)
        ppl = perplexities.get(name, 0)
        vocab_size = len(tokenizers[name])
        params = model_params.get(name, 0)
        model_params_m = params / 1_000_000 if params > 0 else 0

        # Log to console/file
        ppl_str = f"{ppl:.2f}" if ppl > 0 and not math.isnan(ppl) else "N/A"
        code_comp_str = f"{code_comp:.2f}" if code_comp is not None else "N/A"
        params_str = f"{model_params_m:.2f}" if params > 0 else "N/A"
        logger.info(f"{name:<45} {bleu['bleu']:>8.4f} {bleu['precisions'][0]:>8.4f} {comp:>12.2f} {code_comp_str:>18} {ppl_str:>12} {vocab_size:>10} {params_str:>10}")

        # Add to W&B table
        wandb_table.add_data(
            name,
            vocab_size,
            f"{model_params_m:.2f}" if params > 0 else "N/A",
            bleu['bleu'],
            bleu['precisions'][0],
            comp,
            code_comp if code_comp is not None else None,
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
        if code_comp is not None:
            wandb.summary[f"{name}/code_compression_ratio"] = code_comp
        wandb.summary[f"{name}/perplexity"] = ppl if ppl > 0 and not math.isnan(ppl) else None
        wandb.summary[f"{name}/vocab_size"] = vocab_size
    
    logger.info("-" * 130)
    
    # Log the main comparison table
    wandb.log({"tokenizer_evaluation_table": wandb_table})
    
    # Create bar chart data for easy comparison
    code_comp_values = [code_compression_ratios.get(name, None) for name in tokenizers.keys()]
    code_comp_data = [(name, comp) for name, comp in zip(tokenizer_names_list, code_comp_values) if comp is not None]
    
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
    
    # Log code compression comparison if we have data
    if code_comp_data:
        wandb.log({
            "code_compression_comparison": wandb.plot.bar(
                wandb.Table(data=code_comp_data, columns=["Tokenizer", "Code Compression"]),
                "Tokenizer", "Code Compression", title="Code Compression Ratio Comparison"
            )
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

