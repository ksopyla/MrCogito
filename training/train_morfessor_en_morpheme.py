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


MORFESSOR_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Morfessor"))

DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))

os.makedirs(MORFESSOR_CACHE_DIR, exist_ok=True)

# using nltk word corpus as training data, get the words from nltk and save them to the file

nltk_corpus_words = words.words()

# file with words is saved in the Cache

morfessor_nltk_en_model_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_model.bin")
morfessor_nltk_en_train_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_train.txt")

morfessor_wikipedia_en_train_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train.txt")
morfessor_wikipedia_en_model_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_model.bin")

morfessor_wikipedia_en_train_file_sentences = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_sentences.txt")
morfessor_wikipedia_en_model_file_sentences = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_model_sentences.bin")

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
def prepare_wikipedia_corpus(output_file):
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
    # print("1. Preparing NLTK corpus...")
    # prepare_nltk_corpus(morfessor_nltk_en_train_file)
    # print(f"   ✓ NLTK corpus saved to: {morfessor_nltk_en_train_file}")

    # # Prepare Wikipedia corpus
    # print("\n2. Preparing Wikipedia corpus (this may take a while)...")
    # prepare_wikipedia_corpus(morfessor_wikipedia_en_train_file)
    # print(f"   ✓ Wikipedia corpus saved to: {morfessor_wikipedia_en_train_file}")

    # # Train the NLTK-based model
    # print("\n3. Training NLTK-based Morfessor model...")
    # train_morfessor_model(
    #     morfessor_nltk_en_train_file,
    #     morfessor_nltk_en_model_file,
    #     count_modifier=log_func
    # )
    # print(f"   ✓ NLTK model saved to: {morfessor_nltk_en_model_file}")
    
    # # Train the Wikipedia-based model
    # print("\n4. Training Wikipedia-based Morfessor model...")
    # train_morfessor_model(
    #     morfessor_wikipedia_en_train_file,
    #     morfessor_wikipedia_en_model_file,
    #     count_modifier=log_func
    # )
    # print(f"   ✓ Wikipedia model saved to: {morfessor_wikipedia_en_model_file}")
    
    
    # Prepare Wikipedia corpus for sentences
    print("\n5. Preparing Wikipedia corpus for sentences...")
    prepare_wikipedia_corpus_sentences(morfessor_wikipedia_en_train_file_sentences)
    print(f"   ✓ Wikipedia corpus for sentences saved to: {morfessor_wikipedia_en_train_file_sentences}")
    
    # Train the Wikipedia-based model for sentences
    print("\n6. Training Wikipedia-based Morfessor model for sentences...")
    train_morfessor_model(
        morfessor_wikipedia_en_train_file_sentences,
        morfessor_wikipedia_en_model_file_sentences,
        count_modifier=log_func
    )
    
    print("\n=== Training Pipeline Completed Successfully ===\n")
    
    # test the model
    # io = morfessor.MorfessorIO()
    # model = io.read_binary_model_file(morfessor_nltk_en_model_file)
    # # for segmenting new words we use the viterbi_segment(compound) method
    # print(model.viterbi_segment("windsurfing")[0])

# %%
