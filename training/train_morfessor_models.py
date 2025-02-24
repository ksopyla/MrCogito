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


MORFESSOR_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Morfessor"))

DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))

os.makedirs(MORFESSOR_CACHE_DIR, exist_ok=True)

# using nltk word corpus as training data, get the words from nltk and save them to the file

nltk_corpus_words = words.words()

# file with words is saved in the Cache

morfessor_nltk_en_model_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_model.bin")
morfessor_nltk_en_train_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_train.txt")

morfessor_wiki_en_train_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_train.txt")
morfessor_wiki_en_model_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_model.bin")

morfessor_wiki_en_train_file_sentences = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_train_sentences.txt")
morfessor_wiki_en_model_file_sentences = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wiki_en_model_sentences.bin")


morfessor_wikipedia_en_train_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_300M.txt")
morfessor_wikipedia_en_model_words = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_words_300M.bin")

#%% save the words to the file
outfile = open(morfessor_nltk_en_train_file, "w")
for word in nltk_corpus_words:
    outfile.write(word+"\n")

outfile.close()

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
    """Process Wikipedia dataset and save text to training file"""
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

def prepare_wikipedia_corpus_sentences(output_file):
    """Process Wikipedia dataset and save sentences to training file"""
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
    # prepare_wikipedia_corpus_sentences(morfessor_wiki_en_train_file_sentences)
    # print(f"   ✓ Wikitext corpus for sentences saved to: {morfessor_wiki_en_train_file_sentences}")
    
    # # Train the Wikipedia-based model for sentences
    # print("Training Wikitext Morfessor model for sentences...")
    # train_morfessor_model(
    #     morfessor_wiki_en_train_file_sentences,
    #     morfessor_wiki_en_model_file_sentences,
    #     count_modifier=log_func
    # )
    
    # # Prepare Wikipedia corpus 
    print("Preparing Wikipedia corpus with words...")
    prepare_wikipedia2023_words_corpus(morfessor_wikipedia_en_train_words)
    print(f"   ✓ Wikipedia corpus with words saved to: {morfessor_wikipedia_en_train_words}")
    
    # Train the Wikipedia-based model for sentences
    print("Training Wikipedia-based Morfessor model for words...")
    train_morfessor_model(
        morfessor_wikipedia_en_train_words,
        morfessor_wikipedia_en_model_words,
        count_modifier=log_func
    )
    

    
    # test the model
    # io = morfessor.MorfessorIO()
    # model = io.read_binary_model_file(morfessor_nltk_en_model_file)
    # # for segmenting new words we use the viterbi_segment(compound) method
    # print(model.viterbi_segment("windsurfing")[0])

# %%
