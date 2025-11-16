#%%

import os

from tokenizers import Tokenizer

from datasets import load_dataset, load_from_disk

from rich import print



# Constants
DATASET_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
MORFESSOR_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Morfessor"))
TOKENIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers"))

#%%



morfessor_wikipedia_en_model_1M_art_min_7_nltk_words_ds = os.path.join(DATASET_CACHE_DIR, "wikipedia_20231101.en_1000000_wikipedia_1m_min_7_nltk_morphems")


wikipedia_7nltk_ds = load_from_disk(morfessor_wikipedia_en_model_1M_art_min_7_nltk_words_ds)
# %%
print(wikipedia_7nltk_ds)

# print first 10 examples forom dataset, udr the column "morfessor_processed"
print(" ".join(wikipedia_7nltk_ds[0]["morfessor_processed"]))

# %%
wiki_