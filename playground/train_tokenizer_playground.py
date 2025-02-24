"""
This script demonstrates how to train and evaluate different tokenizer configurations
using the 🤗 Tokenizers library. We'll focus on the SentencePiece Unigram algorithm
which is particularly good for morphologically rich languages.

Key features demonstrated:
- Loading and preparing training data
- Configuring and training SentencePiece Unigram tokenizer (LLAMA-style)
- Evaluating tokenization results
- Saving and loading tokenizers
- Testing different configurations
"""

import os
from datasets import load_dataset, Dataset
from tokenizers import (
    SentencePieceUnigramTokenizer,
    SentencePieceBPETokenizer,
    decoders,
    Tokenizer,
    models,
    trainers,
    normalizers,
    pre_tokenizers,
    AddedToken,
    Regex,
    processors
)
from transformers import AutoTokenizer
from dotenv import dotenv_values
from rich import print, box
from rich.console import Console
from rich.table import Table
import evaluate
import morfessor
from huggingface_hub import login

from data.ground_truth import (
    GROUND_TRUTH_MORPHEMS,
    HARD_WORDS_CORPUS,
    SIMPLE_WORDS_CORPUS,
    TEST_SENTENCES
)

# Constants
DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
MORFESSOR_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Morfessor"))
TOKENIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers"))

# Morfessor model files
MORFESSOR_NLTK_MODEL = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_model.bin")
MORFESSOR_WIKI_MODEL = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_model.bin")
MORFESSOR_WIKI_SENT_MODEL = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_model_sentences.bin")
MORFESSOR_WIKIPEDIA_300M_MODEL = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_300M.bin")

MORFESSOR_WIKIPEDIA_10M_MODEL = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_10M.bin")

MORFESSOR_WIKIPEDIA_1M_UNIQUE_MODEL = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_split_1M_art.bin")

MORFESSOR_WIKIPEDIA_1M_UNIQUE_TOK_MODEL = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_nltk_1M_art.bin")

SPECIAL_TOKENS = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]

def _load_morfessor_model(file_path):
    """Load a Morfessor model from a binary file"""
    io_local = morfessor.MorfessorIO()
    return io_local.read_binary_model_file(file_path)

def preprocess_with_morfessor(batch, model_file):
    """
    Morfessor splitting for a batch of text.
    A new model instance is created for each process to avoid conflicts.
    
    Args:
        batch: Dictionary containing text data to process
        model_file: Path to the Morfessor model file
        
    Returns:
        Dictionary containing processed text
    """
    # Cache the model in a static attribute so each worker loads it only once
    if not hasattr(preprocess_with_morfessor, "morfessor_model"):
        preprocess_with_morfessor.morfessor_model = _load_morfessor_model(model_file)
    model_local = preprocess_with_morfessor.morfessor_model
        
    processed_text = []
    
    #print(f"preprocessing {len(batch['text'])} lines with morfessor model {model_file} process {os.getpid()}")
    for text_line in batch["text"]:
        words = text_line.split()
        morf_segments = []
        for word in words:
            segments, _cost = model_local.viterbi_segment(word)
            morf_segments.append(" ".join(segments))
        processed_text.append(morf_segments)
    return {"morfessor_processed": processed_text}

def get_preprocessed_morfessor_dataset(train_dataset, 
                                     cached_file_name_suffix="_morphems",
                                     morfessor_model_file="",
                                     output_cache_directory=DATASET_CACHE_DIR,
                                     num_proc=4,
                                     batch_size=5000):
    """
    Process the dataset using Morfessor, caching results to disk.
    
    Args:
        train_dataset: Input dataset to process
        cached_file_name_suffix: Suffix for the cached file name
        morfessor_model_file: Path to the Morfessor model file
        output_cache_directory: Directory to store cached results
        num_proc: Number of processes to use for parallel processing
        
    Returns:
        Processed dataset with morphological segmentation
    """
    ds_info = train_dataset.info
    ds_name = ds_info.dataset_name
    ds_config = ds_info.config_name
    ds_len = len(train_dataset)
    processed_dataset_path = os.path.join(
        output_cache_directory,
        f"{ds_name}_{ds_config}_{ds_len}_{cached_file_name_suffix}"
    )
    
    if os.path.isdir(processed_dataset_path):
        print(f"[INFO] Loading existing preprocessed dataset from: {processed_dataset_path}")
        return Dataset.load_from_disk(processed_dataset_path)
    
    print(f"[INFO] No cached dataset found {processed_dataset_path}.")
    print(f"[INFO] Starting Morfessor processing with model: {morfessor_model_file} batch={batch_size} num_proc={num_proc}")
    print(f"[INFO] Dataset size: {ds_len} samples")
    ds_processed = train_dataset.map(
        lambda batch: preprocess_with_morfessor(batch, morfessor_model_file),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc
    )

    ds_processed.save_to_disk(processed_dataset_path)
    return ds_processed

def setup_environment():
    """Setup environment variables and directories"""
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    envs = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))
    login(token=envs["HF_TOKEN"])
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

def load_pretrained_tokenizers():
    """Load various pretrained tokenizers for comparison"""
    return {
        'bert': AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=TOKENIZER_DIR),
        'modernbert': AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", cache_dir=TOKENIZER_DIR),
        'llama32': AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-instruct", cache_dir=TOKENIZER_DIR),
        'gpt2': AutoTokenizer.from_pretrained("gpt2", cache_dir=TOKENIZER_DIR),
        'xlnet': AutoTokenizer.from_pretrained("xlnet-base-cased", cache_dir=TOKENIZER_DIR),
        'albert': AutoTokenizer.from_pretrained("albert-base-v2", cache_dir=TOKENIZER_DIR)
    }

def initialize_unigram_tokenizer():
    """Initialize and configure the Unigram tokenizer"""
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    return tokenizer

def configure_trainer():
    """Configure the Unigram trainer"""
    return trainers.UnigramTrainer(
        vocab_size=2**15,
        unk_token="<unk>",
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        max_piece_length=16,
        shrinking_factor=0.85,
        n_sub_iterations=5
    )

def batch_iterator(batch_size=1000, train_dataset=None):
    """Iterator for batching dataset"""
    tok_dataset = train_dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]

def batch_iterator_only_one_same_word(batch_size=1000, train_dataset=None):
    """Iterator that returns unique words from dataset"""
    words_frequency = {}
    tok_dataset = train_dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        words_to_return = []
        for sentence in batch["text"]:
            for word in sentence.split():
                if word not in words_frequency:
                    words_frequency[word] = 1
                    words_to_return.append(word)
                else:
                    words_frequency[word] += 1
        yield words_to_return
    return words_frequency

def batch_iterator_morfessor_processed(batch_size=1000, morfessor_train_dataset=None):
    """Iterator for Morfessor processed dataset"""
    preprocessed_dataset = morfessor_train_dataset.select_columns("morfessor_processed")
    for batch in preprocessed_dataset.iter(batch_size):
        joined_lines = [" ".join(tokens) for tokens in batch["morfessor_processed"]]
        yield joined_lines

def get_tokenizer_predictions(tokenizer, word):
    """Get tokenizer predictions for a word"""
    if isinstance(tokenizer, Tokenizer):
        tokens = tokenizer.encode(word).tokens
    elif isinstance(tokenizer, morfessor.baseline.BaselineModel):
        tokens = tokenizer.viterbi_segment(word)[0]
    else:
        tokens = tokenizer.tokenize(word)
    
    cleaned_tokens = []
    for token in tokens:
        token = token.replace('Ġ', '').replace('▁', '').replace('##', '')
        if token.strip():
            cleaned_tokens.append(token)
    
    return ' '.join(cleaned_tokens)

def evaluate_tokenizers(tokenizers_dict, ground_truth_morphems):
    """Evaluate tokenizers using BLEU score"""
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

def print_evaluation_results(bleu_scores):
    """Print evaluation results in a formatted table"""
    console = Console()
    bleu_table = Table(title="BLEU Scores for each tokenizer", box=box.MINIMAL_DOUBLE_HEAD)
    bleu_table.add_column("Tokenizer", style="bright_cyan")
    bleu_table.add_column("BLEU Score", justify="right", style="bright_green")
    bleu_table.add_column("1-gram", justify="right", style="bright_magenta")
    bleu_table.add_column("2-gram", justify="right", style="bright_magenta")
    bleu_table.add_column("3-gram", justify="right", style="bright_magenta")

    for tokenizer_name, score in bleu_scores.items():
        bleu_table.add_row(
            tokenizer_name,
            f"{score['bleu']:.4f}",
            f"{score['precisions'][0]:.4f}",
            f"{score['precisions'][1]:.4f}",
            f"{score['precisions'][2]:.4f}"
        )

    console.print(bleu_table)

def main():
    """Main function to run the tokenizer training and evaluation pipeline"""
    print("\n=== Starting Tokenizer Training and Evaluation ===\n")
    
    # Setup
    print(" Setting up environment...")
    setup_environment()
    
    # Load pretrained tokenizers
    print("Loading pretrained tokenizers...")
    tokenizers = load_pretrained_tokenizers()
    print(f"Loaded {len(tokenizers)} pretrained tokenizers: {list(tokenizers.keys())}")
    
    print("Loading Morfessor models...")
    # Load Morfessor models
    io = morfessor.MorfessorIO()
    model_nltk = io.read_binary_model_file(MORFESSOR_NLTK_MODEL)
    model_wiki = io.read_binary_model_file(MORFESSOR_WIKI_MODEL)
    model_sent = io.read_binary_model_file(MORFESSOR_WIKI_SENT_MODEL)
    model_wikipedia_300M = io.read_binary_model_file(MORFESSOR_WIKIPEDIA_300M_MODEL)
    model_wikipedia_10M = io.read_binary_model_file(MORFESSOR_WIKIPEDIA_10M_MODEL)
    model_wikipedia_1M_unique = io.read_binary_model_file(MORFESSOR_WIKIPEDIA_1M_UNIQUE_MODEL)
    model_wikipedia_1M_unique_tok = io.read_binary_model_file(MORFESSOR_WIKIPEDIA_1M_UNIQUE_TOK_MODEL)
    print("Successfully loaded all Morfessor models")
    
    print("Loading and preparing WikiText dataset...")
    wiki_text_dataset = load_dataset("Salesforce/wikitext", 
                          "wikitext-103-v1",
                          cache_dir=DATASET_CACHE_DIR,
                          split="train")
    wiki_text_train_dataset = wiki_text_dataset.select(range(500000))
    print(f"Selected {len(wiki_text_train_dataset)} samples from WikiText dataset")
    
    print("Training Unigram tokenizer on WikiText...")
    uni_wikitext_tokenizer = initialize_unigram_tokenizer()
    uni_wikitext_trainer = configure_trainer()
    
    print("Processing WikiText with Morfessor...")
    uni_wiki_words_train_dataset = get_preprocessed_morfessor_dataset(
        wiki_text_train_dataset,
        morfessor_model_file=MORFESSOR_WIKI_MODEL,
        output_cache_directory=DATASET_CACHE_DIR,
        cached_file_name_suffix="wiki_morphems",
        num_proc=4
    )
    
    print("Training Unigram tokenizer on WikiText morphemes...")
    uni_wikitext_tokenizer.train_from_iterator(
        batch_iterator_morfessor_processed(morfessor_train_dataset=uni_wiki_words_train_dataset),
        trainer=uni_wikitext_trainer
    )
    print("WikiText Unigram tokenizer training completed")
    
    print("Loading and preparing Wikipedia dataset...")
    
    wikipedia_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", cache_dir=DATASET_CACHE_DIR)
    wikipedia_train_dataset = wikipedia_dataset["train"].select(range(1_000_000))

    print(f"Selected {len(wikipedia_train_dataset)} samples from Wikipedia dataset")
    
    print("Training Unigram tokenizer on Wikipedia on Morfessor 300M words and morphemes...")
    uni_wikipedia_300M_tokenizer = initialize_unigram_tokenizer()
    uni_wikipedia_300M_trainer = configure_trainer()
    
    print("Processing Wikipedia with Morfessor 300M model...")
    uni_wikipedia_300M_words_train_dataset = get_preprocessed_morfessor_dataset(
        wikipedia_train_dataset,
        morfessor_model_file=MORFESSOR_WIKIPEDIA_300M_MODEL,
        output_cache_directory=DATASET_CACHE_DIR,
        cached_file_name_suffix="wikipedia300m_morphems",
        num_proc=48, batch_size=3000
    )
    
    print("Training Unigram tokenizer on Wikipedia 300M morphemes...")
    uni_wikipedia_300M_tokenizer.train_from_iterator(
        batch_iterator_morfessor_processed(morfessor_train_dataset=uni_wikipedia_300M_words_train_dataset),
        trainer=uni_wikipedia_300M_trainer
    )
    print("Wikipedia morfessor Unigram tokenizer training completed")



    print("Training Unigram tokenizer on Wikipedia on Morfessor 10M words and morphemes...")
    uni_wikipedia_10M_tokenizer = initialize_unigram_tokenizer()
    uni_wikipedia_10M_trainer = configure_trainer()
    
    print("Processing Wikipedia with Morfessor 10M model...")
    uni_wikipedia_10M_words_train_dataset = get_preprocessed_morfessor_dataset(
        wikipedia_train_dataset,
        morfessor_model_file=MORFESSOR_WIKIPEDIA_10M_MODEL,
        output_cache_directory=DATASET_CACHE_DIR,
        cached_file_name_suffix="wikipedia_10m_morphems",
        num_proc=48, batch_size=3000
    )
    
    print("Training Unigram tokenizer on Wikipedia 10M morphemes...")
    uni_wikipedia_10M_tokenizer.train_from_iterator(
        batch_iterator_morfessor_processed(morfessor_train_dataset=uni_wikipedia_10M_words_train_dataset),
        trainer=uni_wikipedia_10M_trainer
    )
    print("#####")

    ###
    print("Training Unigram tokenizer on Wikipedia on Morfessor 1M art unique split words and morphemes...")
    uni_wikipedia_1M_unique_tokenizer = initialize_unigram_tokenizer()
    uni_wikipedia_1M_unique_trainer = configure_trainer()
    
    print("Processing Wikipedia with Morfessor 10M unique model...")
    uni_wikipedia_1M_unique_words_train_dataset = get_preprocessed_morfessor_dataset(
        wikipedia_train_dataset,
        morfessor_model_file=MORFESSOR_WIKIPEDIA_1M_UNIQUE_MODEL,
        output_cache_directory=DATASET_CACHE_DIR,
        cached_file_name_suffix="wikipedia_1m_art_unique_morphems",
        num_proc=48, batch_size=3000
    )
    
    print("Training Unigram tokenizer on Wikipedia Morfessor 1M unique split words morphemes...")
    uni_wikipedia_1M_unique_tokenizer.train_from_iterator(
        batch_iterator_morfessor_processed(morfessor_train_dataset=uni_wikipedia_1M_unique_words_train_dataset),
        trainer=uni_wikipedia_1M_unique_trainer
    )
    print("####")

    

    ######
    print("Training Unigram tokenizer on Wikipedia on Morfessor 1M unique nltk tokenize words and morphemes...")
    uni_wikipedia_1M_unique_tok_tokenizer = initialize_unigram_tokenizer()
    uni_wikipedia_1M_unique_tok_trainer = configure_trainer()
    
    print("Processing Wikipedia with Morfessor 1M unique nltk tokenize model...")
    uni_wikipedia_1M_unique_words_tok_train_dataset = get_preprocessed_morfessor_dataset(
        wikipedia_train_dataset,
        morfessor_model_file=MORFESSOR_WIKIPEDIA_1M_UNIQUE_TOK_MODEL,
        output_cache_directory=DATASET_CACHE_DIR,
        cached_file_name_suffix="wikipedia_1m_unique_tok_morphems",
        num_proc=48, batch_size=3000
    )
    
    print("Training Unigram tokenizer on Wikipedia Morfessor 10M unique nltk tokenize morphemes...")
    uni_wikipedia_1M_unique_tok_tokenizer.train_from_iterator(
        batch_iterator_morfessor_processed(morfessor_train_dataset=uni_wikipedia_1M_unique_words_tok_train_dataset),
        trainer=uni_wikipedia_1M_unique_tok_trainer
    )
    print("####")


    # build unigram tokenizer just on dataset withou any preprecessing
    print("Training Unigram tokenizer on Wikipedia corpus (withou pre-processing)...")

    uni_normal_wikipedia_tokenizer = initialize_unigram_tokenizer()
    uni_normal_wikipedia_trainer = configure_trainer()
    print("Training Unigram tokenizer on Wikipedia words...")
    uni_normal_wikipedia_tokenizer.train_from_iterator(
        batch_iterator(train_dataset=wikipedia_train_dataset),
        trainer=uni_normal_wikipedia_trainer
    )
    print("Wikipedia Unigram tokenizer training completed")
    
    print("\nPreparing for evaluation...")
    tokenizers.update({
        'morfessor_nltk': model_nltk,
        'morfessor_wiki': model_wiki,
        'morfessor_sent': model_sent,   
        'morfessor_wikipedia_300M': model_wikipedia_300M,
        "morfessor_wikipedia_10M": model_wikipedia_10M,
        "morfessor_wikipedia_10M_unique": model_wikipedia_1M_unique,
        "morfessor_wikipedia_10M_unique_tok": model_wikipedia_1M_unique_tok,
        'uni_wiki': uni_wikitext_tokenizer,
        'uni_wikipedia_300M': uni_wikipedia_300M_tokenizer,
        'uni_wikipedia_10M': uni_wikipedia_10M_tokenizer,
        'uni_wikipedia_10M_unique': uni_wikipedia_1M_unique_tokenizer,
        'uni_wikipedia_10M_unique_tok': uni_wikipedia_1M_unique_tok_tokenizer,
        'uni_normal_wikipedia': uni_normal_wikipedia_tokenizer
    })
    
    # # Configure post-processor
    # uni_sp_tokenizer.post_processor = processors.TemplateProcessing(
    #     single="<s> $A:0 </s>:0 <cls>:2",
    #     pair="<s> $A:0 <sep>:0 $B:1 <sep>:1 </s> <cls>:2",
    #     special_tokens=[
    #         ("<sep>", uni_sp_tokenizer.token_to_id("<sep>")),
    #         ("<cls>", uni_sp_tokenizer.token_to_id("<cls>")),
    #         ("<s>", uni_sp_tokenizer.token_to_id("<s>")),
    #         ("</s>", uni_sp_tokenizer.token_to_id("</s>"))
    #     ]
    # )
    # uni_sp_tokenizer.decoder = decoders.Metaspace()
    
    # Evaluate tokenizers
    print("\n=== Final Evaluation Results ===")
    bleu_scores = evaluate_tokenizers(tokenizers, GROUND_TRUTH_MORPHEMS)
    print("\n=== Final Evaluation Results ===")
    print_evaluation_results(bleu_scores)
    
if __name__ == "__main__":
    main()
