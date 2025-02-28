"""
Script for training the morfessor model for English language based on the nltk word corpus.

"""
#%%
import math
import morfessor
import os
from nltk.corpus import words
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize


MORFESSOR_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Morfessor"))

DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))

os.makedirs(MORFESSOR_CACHE_DIR, exist_ok=True)

# using nltk word corpus as training data, get the words from nltk and save them to the file



# file with words is saved in the Cache

morfessor_nltk_en_model_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_model.bin")
morfessor_nltk_en_train_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_train.txt")

morfessor_wiki_en_train_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_train.txt")
morfessor_wiki_en_model_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_model.bin")

morfessor_wiki_en_train_file_sentences = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_train_sentences.txt")
morfessor_wiki_en_model_file_sentences = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_model_sentences.bin")


morfessor_wikipedia_en_train_300M_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_300M.txt")
morfessor_wikipedia_en_model_300M_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_300M.bin")

morfessor_wikipedia_en_train_10M_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_10M.txt")
morfessor_wikipedia_en_model_10M_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_10M.bin")

morfessor_wikipedia_en_train_1M_art_unique_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_split_1M_art.txt")
morfessor_wikipedia_en_model_1M_art_unique_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_split_1M_art.bin")

morfessor_wikipedia_en_train_1M_art_unique_3M_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_split_1M_art_3M_words.txt")
morfessor_wikipedia_en_model_1M_art_unique_3M_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_split_1M_art_3M_words.bin")

morfessor_wikipedia_en_train_1M_art_unique_nltk_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_nltk_1M_art.txt")
morfessor_wikipedia_en_model_1M_art_unique_nltk_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_nltk_1M_art.bin")

morfessor_wikipedia_en_train_1M_art_min_7_nltk_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_nltk_1M_art_min_7.txt")
morfessor_wikipedia_en_model_1M_art_min_7_nltk_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_nltk_1M_art_min_7.bin")
morfessor_wikipedia_en_model_1M_art_min_7_nltk_words_log = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_unique_nltk_1M_art_min_7_log.bin")

#%% Utility functions

# Function for adjusting the counts of each compound using log
def log_func(x):
    return int(round(math.log(x + 1, 2)))

# Helper function to save words to a file
def save_words_to_file(words, file_path):
    with open(file_path, "w", encoding="utf-8") as outfile:
        for w in words:
            if w.strip():
                outfile.write(f"{w.strip()}\n")

#%% Prepare NLTK training corpus
def prepare_nltk_corpus(output_file):
    """Fetch words from NLTK and save them to a text file."""
    nltk_corpus_words = words.words()
    save_words_to_file(nltk_corpus_words, output_file)

#%% prepare the wikipedia training data with using the Huggingface dataset
def prepare_wiki_words_corpus(output_file):
    """Process Wikitext dataset and save text to training file"""
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1",
                       cache_dir=DATASET_CACHE_DIR,
                       )
    dataset = wikitext["train"]
    
    # Process with built-in progress bar
    processed = dataset.map(
        lambda x: {'processed_text': x['text'].replace('\n', ' ').split()},
        desc="Processing articles (words)",
        batched=False,
        with_progress=True
    )
    
    all_words = []
    for article in processed:
        all_words.extend(article['processed_text'])

    save_words_to_file(all_words, output_file)
    
    
def prepare_wikipedia2023_words_corpus(output_file):
    """Process Wikipedia dataset and save text to training file.
    
    Args:
        output_file (str): Path to the output file where words will be saved
        
    Returns:
        None
    """
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        return
        
    # Load dataset
    wikitext = load_dataset("wikimedia/wikipedia", "20231101.en",
                       cache_dir=DATASET_CACHE_DIR,
                       )
    dataset = wikitext["train"]
    
    total_rows = dataset.num_rows
    pbar = tqdm(total=total_rows, desc="Processing Wikipedia articles")
    processed_rows = 0
    batch_size = 100
    
    # Process and write directly to file
    with open(output_file, "w", encoding="utf-8") as f:
        # Process with built-in progress bar using streaming
        for batch in dataset.iter(batch_size=batch_size):
            for text in batch["text"]:
                # Split text into words and write each word
                words = text.replace('\n', ' ').split()
                for word in words:
                    if word.strip():  # Only write non-empty words
                        f.write(f"{word.strip()}\n")
            # Update progress bar based on batch size
            processed_rows += batch_size
            pbar.update(batch_size)
        
        pbar.close()
                        
    print(f"Processed Wikipedia dataset and saved words to: {output_file}")

def prepare_wikipedia2023_unique_words_corpus(output_file, sub_set=10_000, spliting='split'):
    """Process Wikipedia dataset and save unique words to text to training file.
    
    Args:
        output_file (str): Path to the output file where unique words will be saved
        spliting (str): The method to split the text into words. Default is 'split' which uses the split method. 'nltk' uses nltk word_tokenize
        
    Returns:
        None
    """
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        return
        
    # Load dataset
    wikitext = load_dataset("wikimedia/wikipedia", "20231101.en",
                       cache_dir=DATASET_CACHE_DIR,
                       )
    
    
    dataset = wikitext["train"].select(range(sub_set))
    
    total_rows = dataset.num_rows
    pbar = tqdm(total=total_rows, desc="Processing Wikipedia articles")
    processed_rows = 0
    batch_size = 10_000
    unique_words = set()
    all_unique_words = set()  # Master set to store all unique words
    
    # Process and write in batches
    with open(output_file, "w", encoding="utf-8") as f:
        for batch in dataset.iter(batch_size=batch_size):
            for text in batch["text"]:
                # Split text into words and create a set of stripped words
                if spliting == 'nltk':
                    words = set(word_tokenize(text.replace('\n', ' '))) 
                else:
                    words = set(text.replace('\n', ' ').split())
                
                # Find new unique words using set difference
                new_words = words - all_unique_words
                
                # Update the sets
                unique_words.update(new_words)
                all_unique_words.update(new_words)
                
            # After each batch save unique words to file
            for word in unique_words:
                f.write(f"{word}\n")
            unique_words.clear()
            
            # Update progress bar based on batch size
            processed_rows += batch_size
            pbar.update(batch_size)
        
        # Write any remaining words
        if unique_words:
            for word in unique_words:
                f.write(f"{word}\n")
        
        pbar.close()
                        
    print(f"Processed Wikipedia dataset and saved unique words to: {output_file}")

def prepare_wikipedia2023_unique_words_corpus_v2(output_file, sub_set=10_000, spliting='split', batch_size=1000, min_occurrences=2, num_proc=4):
    """Process Wikipedia dataset and save unique words to text to training file using dataset.map functionality.
    
    This version uses Huggingface datasets' map functionality for better performance and memory efficiency.
    The process is done in two steps:
    1. Map over batches to collect words with their occurrences per batch with their counts (using multiprocessing)
    2. Reduce all batches to get final words with total counts and filter by minimum occurrences (single core)
    
    Args:
        output_file (str): Path to the output file where unique words will be saved
        sub_set (int): Number of articles to process. Default is 10,000
        spliting (str): The method to split text into words. Either 'split' or 'nltk'
        batch_size (int): Size of batches for processing. Default is 1000
        min_occurrences (int): Minimum number of occurrences required to keep a word. Default is 2
        num_proc (int): Number of processes to use for parallel processing. Default is 4
        
    Returns:
        None
    """
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        return
        
    # Load dataset
    wikitext = load_dataset("wikimedia/wikipedia", "20231101.en",
                       cache_dir=DATASET_CACHE_DIR,
                       )
    
    dataset = wikitext["train"].select(range(sub_set))
    
    # Step 1: Process each batch to get unique words and their counts
    def extract_words_from_batch(examples):
        """Extract words from batch of texts and count frequencies directly.
        
        This optimized version counts words directly without creating intermediate lists.
        Returns properly formatted output for Huggingface datasets.map().
        """
        # Count words directly as we process them
        word_freq = {}
        
        for text in examples["text"]:
            if spliting == 'nltk':
                words = word_tokenize(text.replace('\n', ' '))
            else:
                words = text.replace('\n', ' ').split()
                
            # Count each word directly as we see it
            for word in words:
                if word.strip():  # Only count non-empty words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Convert to lists for HF dataset storage - must be in this format!
        words = list(word_freq.keys())
        counts = list(word_freq.values())
        
        # This will be flattned to a list of dictionaries {word: word, counts: counts}
        return {"word": words, "counts": counts}
    
    print("Processing batches to collect words and their counts...")
    processed_dataset = dataset.map(
        extract_words_from_batch,
        batched=True,
        batch_size=batch_size,
        desc="Collecting word counts per batch",
        remove_columns=dataset.column_names,
        num_proc=num_proc
    )
    
    print(f"Processed dataset view, length: {len(processed_dataset)}")
    print(processed_dataset)
    print(processed_dataset[0])
       
    # Step 2: Reduce all batches to get final word counts (on a single core)
    print("Reducing batches to get final word counts...")
    word_counts = {}
    
    for batch in processed_dataset.iter(batch_size=1000):
        batch_words = batch["word"]
        batch_counts = batch["counts"]
        
        # Add counts from this batch to the master count
        for word, count in zip(batch_words, batch_counts):
            word_counts[word] = word_counts.get(word, 0) + count
    
    # Filter words by minimum occurrences
    filtered_words = {word: count for word, count in word_counts.items() 
                     if count >= min_occurrences}
    
    # Save to file
    print(f"Saving {len(filtered_words)} words (min occurrences: {min_occurrences}) to file...")
    with open(output_file, "w", encoding="utf-8") as f:
        # Sort words by occurrence count in descending order
        for word, count in sorted(filtered_words.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{count} {word}\n")
                
    print(f"Successfully saved words to: {output_file}")
    print(f"Total unique words: {len(word_counts)}")
    print(f"Words with >= {min_occurrences} occurrences: {len(filtered_words)}")


def prepare_wiki_corpus_sentences(output_file):
    """Process Wikitext dataset and save sentences to training file"""
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1",
                       cache_dir=DATASET_CACHE_DIR,
                       )
    dataset = wikitext["train"]
    
    # Process with built-in progress bar
    processed = dataset.map(
        lambda x: {'sentences': sent_tokenize(x['text'])},
        desc="Processing articles (sentences)",
        batched=False,
        with_progress=True,
        num_proc=4
    )
    
    all_sentences = []
    for article in processed:
        all_sentences.extend(article['sentences'])

    save_words_to_file(all_sentences, output_file)

#%% train function for the morfessor model
def train_morfessor_model(input_path, output_path, count_modifier=None):
    """Train and save Morfessor model from training file"""
    io = morfessor.MorfessorIO()
    train_data = list(io.read_corpus_file(input_path))
    model = morfessor.BaselineModel()
    model.load_data(train_data, count_modifier=count_modifier)
    model.train_batch()
    io.write_binary_model_file(output_path, model)
    print(f"Model trained and saved to {output_path}")

#%% main training execution
if __name__ == "__main__":
    print("\n=== Starting Morfessor Model Training Pipeline ===\n")
    
    # # Prepare NLTK corpus
    # print("Preparing NLTK corpus...")
    # prepare_nltk_corpus(morfessor_nltk_en_train_file)
    # print(f"   ✓ NLTK corpus saved to: {morfessor_nltk_en_train_file}")



    # # Train the NLTK-based model
    # print("Training NLTK-based Morfessor model...")
    # train_morfessor_model(
    #     morfessor_nltk_en_train_file,
    #     morfessor_nltk_en_model_file,
    #     count_modifier=log_func
    # )
    # print(f"   ✓ NLTK model saved to: {morfessor_nltk_en_model_file}")
    
    
    
    # # Prepare Wikitext corpus
    # print("Preparing Wikitext corpus (this may take a while)...")
    # prepare_wiki_words_corpus(morfessor_wiki_en_train_file)
    # print(f"   ✓ Wikitext corpus saved to: {morfessor_wiki_en_train_file}")
    # # Train the Wikipedia-based model
    # print("Training Wikitext Morfessor model...")
    # train_morfessor_model(
    #     morfessor_wiki_en_train_file,
    #     morfessor_wiki_en_model_file,
    #     count_modifier=log_func
    # )
    # print(f"   ✓ Wikitext model saved to: {morfessor_wiki_en_model_file}")
    
    
    # # Prepare Wikipedia corpus for sentences
    # print("Preparing Wikitext corpus for sentences...")
    # prepare_wiki_corpus_sentences(morfessor_wiki_en_train_file_sentences)
    # print(f"   ✓ Wikitext corpus for sentences saved to: {morfessor_wiki_en_train_file_sentences}")
    
    # # Train the Wikipedia-based model for sentences
    # print("Training Wikitext Morfessor model for sentences...")
    # train_morfessor_model(
    #     morfessor_wiki_en_train_file_sentences,
    #     morfessor_wiki_en_model_file_sentences,
    #     count_modifier=log_func
    # )
    
    # # Prepare Wikipedia corpus 
    # print("Preparing Wikipedia corpus with words...")
    # prepare_wikipedia2023_words_corpus(morfessor_wikipedia_en_train_10M_words)
    # print(f"   ✓ Wikipedia corpus with words saved to: {morfessor_wikipedia_en_train_10M_words}")
    
    # # Train the Wikipedia-based model for sentences
    # print("Training Wikipedia-based Morfessor model for words...")
    # train_morfessor_model(
    #     morfessor_wikipedia_en_train_10M_words,
    #     morfessor_wikipedia_en_model_10M_words,
    #     count_modifier=log_func
    # )


    # # # Prepare Wikipedia corpus 
    # print(f"Preparing Wikipedia corpus with words {morfessor_wikipedia_en_train_1M_art_unique_words}")
    # prepare_wikipedia2023_unique_words_corpus(morfessor_wikipedia_en_train_1M_art_unique_words, spliting='split', sub_set=1_000_000)
    # print(f"   ✓ Wikipedia corpus with words saved to: {morfessor_wikipedia_en_train_1M_art_unique_words}")
    
    # # Train the Wikipedia-based model for sentences
    # print(f"Training Wikipedia-based Morfessor model {morfessor_wikipedia_en_model_1M_art_unique_words} ")
    # train_morfessor_model(
    #     morfessor_wikipedia_en_train_1M_art_unique_words,
    #     morfessor_wikipedia_en_model_1M_art_unique_words,
    #     count_modifier=lambda x: 1
    # )
    
    # # Train the Wikipedia-based model for sentences
    # print(f"Training Wikipedia-based Morfessor model {morfessor_wikipedia_en_train_1M_art_unique_3M_words} ")
    # train_morfessor_model(
    #     morfessor_wikipedia_en_train_1M_art_unique_3M_words,
    #     morfessor_wikipedia_en_model_1M_art_unique_3M_words,
    #     count_modifier=lambda x: 1
    # )

   
    

    # # # Prepare Wikipedia corpus 
    # print(f"Preparing Wikipedia corpus {morfessor_wikipedia_en_train_1M_art_unique_nltk_words}")
    # prepare_wikipedia2023_unique_words_corpus(morfessor_wikipedia_en_train_1M_art_unique_nltk_words , spliting='nltk', sub_set=1_000_000)
    # print(f"   ✓ Wikipedia corpus with words saved to: {morfessor_wikipedia_en_train_1M_art_unique_nltk_words}")
    
    # # Train the Wikipedia-based model for sentences
    # print(f"Training Wikipedia-based Morfessor model {morfessor_wikipedia_en_model_1M_art_unique_nltk_words}")
    # train_morfessor_model(
    #     morfessor_wikipedia_en_train_1M_art_unique_nltk_words,
    #     morfessor_wikipedia_en_model_1M_art_unique_nltk_words,
    #     count_modifier=lambda x: 1
    # )


    
    
    
    # # Prepare Wikipedia corpus 
    # print(f"Preparing Wikipedia corpus {morfessor_wikipedia_en_train_1M_art_min_7_nltk_words}")
    # prepare_wikipedia2023_unique_words_corpus_v2(morfessor_wikipedia_en_train_1M_art_min_7_nltk_words , spliting='nltk', sub_set=1_000_000, batch_size=5000, min_occurrences=7, num_proc=60)
    # print(f"   ✓ Wikipedia corpus with words saved to: {morfessor_wikipedia_en_train_1M_art_min_7_nltk_words}")
    
    # # Train the Wikipedia-based model for sentences
    # print(f"Training Wikipedia-based Morfessor model {morfessor_wikipedia_en_model_1M_art_min_7_nltk_words}")
    # train_morfessor_model(
    #     morfessor_wikipedia_en_train_1M_art_min_7_nltk_words,
    #     morfessor_wikipedia_en_model_1M_art_min_7_nltk_words,
    #     count_modifier=lambda x: 1
    # )
    # Train the Wikipedia-based model for sentences
    print(f"Training Wikipedia-based Morfessor model {morfessor_wikipedia_en_model_1M_art_min_7_nltk_words}")
    train_morfessor_model(
        morfessor_wikipedia_en_train_1M_art_min_7_nltk_words,
        morfessor_wikipedia_en_model_1M_art_min_7_nltk_words_log,
        count_modifier=log_func
    )



    
    # test the model
    # io = morfessor.MorfessorIO()
    # model = io.read_binary_model_file(morfessor_nltk_en_model_file)
    # # for segmenting new words we use the viterbi_segment(compound) method
    # print(model.viterbi_segment("windsurfing")[0])

# %%

