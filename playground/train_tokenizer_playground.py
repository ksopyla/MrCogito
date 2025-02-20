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

#%%
import os
from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer,SentencePieceBPETokenizer,  decoders, Tokenizer, models, trainers, normalizers, pre_tokenizers, AddedToken, Regex, processors
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dotenv import load_dotenv, find_dotenv, dotenv_values

from rich import  print


# import wordsegment  # For morphological analysis
# use the polyglot library to segment the text into words with the morfessor model, 
# https://polyglot.readthedocs.io/en/latest/MorphologicalAnalysis.html 
#from polyglot.text import Text, Word


# Login to Hugging Face, HF token is stored in the .env file 
from huggingface_hub import login

#load the .env file
envs =  dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))
login(token=envs["HF_TOKEN"])

# Constants
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Datasets"))
TOKENIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Tokenizers"))
os.makedirs(TOKENIZER_DIR, exist_ok=True)


#%%load tokenizers 
# BERT tokenizer - WordPiece tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=TOKENIZER_DIR)

# ModernBERT tokenizer - BPE byte-level tokenizer
mb_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", cache_dir=TOKENIZER_DIR)

# OLMo tokenizer - same as ModernBERT
#olmo_tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf", cache_dir=TOKENIZER_DIR)

# tiktoken based tokenizer
llama32_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-instruct", cache_dir=TOKENIZER_DIR)

# gpt2 tokenizer - BPE byte-level tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=TOKENIZER_DIR)

# XLNet tokenizer - 
xlnet_tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased", cache_dir=TOKENIZER_DIR)

# ALBERT tokenizer - 
albert_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", cache_dir=TOKENIZER_DIR)


#%% test corpuses

hard_words_corpus = [
    "This is a windsurfing club.",
    "windsurfing "*3,
    "Today is a beautiful but windy day.",
    "The windsurfer is trying to catch the wave.",
    "Unfortunately, the wind is too strong and the wave is too big.",
    "Unhappily the beautiful kitesurfer has to leave the beach early.",
    "Inproper weather conditions make it difficult to enjoy the beach.",
    # Derivational morphology
    "The recyclable materials underwent repurposing through decentralised reprocessing systems.",
    "Unpredictable weather patterns necessitate preemptive disaster preparedness measures.",
    
    # Inflectional + derivational combinations
    "Misinterpretations of antiestablishmentarianism could lead to counterproductive deregulations.",
    "Overcompensating hyperactive employees often demonstrate underappreciated self-sacrificial tendencies.",
    
    # Multiple affixation
    "Postmodern pseudointellectualism frequently mischaracterizes interdisciplinary socioeconomic transformations.",
    "Premarital counseling helps prevent irreconcilable differences in non-traditional cohabitational arrangements.",
    
    # Challenging compounds
    "The neuropsychologist studied psychophysiological interrelationships in decision-making processes.",
    "Semiconductor nanotechnology enables miniaturized transcontinental telecommunications.",
    
    # Negative prefixes
    "Disinformation campaigns increasingly utilize anti-scientific rhetoric to undermine evidence-based policymaking.",
    "Nonpartisan oversight committees investigate unconstitutional overreach in counterterrorism operations."
]

simple_words_corpus = [
    "windsurfing wind surfing wind kitesurfing windy",
    "surface windowing window windows windy wind",
    "work work worked working doing done cooking cooked cookbook cookbooks cookbook",
    "workplace workforce workday workweek workweekend",
    "unfortunately unhappily unavailable unpredictable unpleasantly unpleasant unpleasantness unpleasantnesses",
    "able unable comfortable uncomfortable",
    "workday workweek workweekend workweekend w",
    "unfortunately unhappily unavailable unpredictable unpleasantly unpleasant unpleasantness unpleasantnesses u",
    "running running runner runners runs run",
    "cooking cook cooked cooks.",
    "he she it they them their their's he's she's it's they're they've at in on with by for of to up",
    "PlayStation Microsoft Xbox Nintendo Google Apple NVIDIA",
    "Donald Trump Joe Biden Barack Obama Hillary Clinton George W. Bush Bill Clinton",
    "New York City Los Angeles Chicago Houston Miami Washington D.C. San Francisco",
    "Amazon Apple Google Facebook Microsoft Tesla",
    "Bitcoin Ethereum Ripple Litecoin Dogecoin",
    "Coca-Cola Pepsi Sprite Fanta Coke Diet Coke",
    "McDonald's Burger King Wendy's Taco Bell",
]

test_sentences = [
        "He worked at a windsurfing as a cook.",
        "The windsurfer is unhappily unable to cook a meal.",
        "Unfortunately, the unpleasant wind at workplace made him unable to cook.",
        "PlayStation is the best gaming console.", 
        "Donald Trump is the president of the United States.",
        "New York City is the largest city in the United States.",
        "Amazon is the online retailer.",
        "Google is the search engine.",
        "Facebook is the social media platform.",
        "Microsoft is the software company.",
        "Tesla is the electric car company.",
]




#%%
    
dataset = load_dataset("Salesforce/wikitext", 
                          "wikitext-103-v1",
                          cache_dir=CACHE_DIR,
                          split="train")
    
train_dataset = dataset.select(range(100000))


def batch_iterator(batch_size=1000, train_dataset=train_dataset):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = train_dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]
        

def batch_iterator_morfessor_processed(batch_size=1000, train_dataset=train_dataset):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = train_dataset.select_columns("text")
    
    def morfessor_process(text):
        # load the morfessor model
        
        model_file = "model.bin"
        io = morfessor.MorfessorIO()
        model = io.read_binary_model_file(model_file)
        # for segmenting new words we use the viterbi_segment(compound) method
        return model.viterbi_segment(text)[0]
        
    
    # Process the text column with Morfessor
    processed_dataset = tok_dataset.map(
        lambda x: {'processed_text': morfessor_process(x['text'])},
        desc="Processing articles (Morfessor)",
        batched=False,
        with_progress=True,
        num_proc=4  
    )
    
    for batch in processed_dataset.iter(batch_size):
        
        
        
        yield batch["text"]

def batch_iterator_only_one_same_word(batch_size=1000, train_dataset=train_dataset):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    
    #add words to the dictionary in to prevent returning the same word many times,
    # We maintain one main dictionary for all the words in the dataset with therir counts, and prepare a list of word to be returned
    
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



def simple_corpus_only_one_same_word(corpus=simple_words_corpus):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    
    #add words to the dictionary in to prevent returning the same word many times,
    # We maintain one main dictionary for all the words in the dataset with therir counts, and prepare a list of word to be returned
    
    words_frequency = {}
    
   
    for sentence in corpus:
        
        words_to_return = []
        for word in sentence.split():
            if word not in words_frequency:
                words_frequency[word] = 1
                words_to_return.append(word)
            else:
                words_frequency[word] += 1
        
        
        yield words_to_return
        
    return words_frequency


def simple_courpus_iterator(corpus=simple_words_corpus): 

    yield "aa bb cc dd ee ff f x z y x z y z x y xyz"
    
    # for sentence in corpus:
    #     yield sentence





#%% Initialize Unigram model 

# https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt

special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]


vocab = [
    ("xyz", 10.5),
]
uni_sp_tokenizer = Tokenizer(models.Unigram(vocab=vocab))



uni_sp_tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

uni_sp_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()



#%% 
# for sentence in test_sentences:
#     print(uni_sp_tokenizer.pre_tokenizer.pre_tokenize_str(sentence))



#%% Configure trainer with initial alphabet from preserved tokens
trainer = trainers.UnigramTrainer(
    vocab_size=2**13,
    unk_token="<unk>",
    special_tokens=special_tokens,
    show_progress=True,
    max_piece_length=16,
    shrinking_factor=0.75,
    nbest_size=10
)

# start training

#uni_sp_tokenizer.train_from_iterator(batch_simple_courpus_iterator(), trainer=trainer)
uni_sp_tokenizer.train_from_iterator(batch_iterator_only_one_same_word(), trainer=trainer)


#uni_sp_tokenizer.train_from_iterator(batch_iterator_only_one_same_word(), trainer=trainer)

#%%
uni_vocab = uni_sp_tokenizer.get_vocab()
# print the first 100 tokens
print(list(uni_vocab.keys())[:100])
print(uni_sp_tokenizer.get_vocab_size())
#%% encode the sentences

# print the encoded sentences side by side
for sentence in test_sentences:
    
    # trained tokenizer
    uni_sp_encoding = uni_sp_tokenizer.encode(sentence)
    uni_sp_tokens = "|".join(uni_sp_encoding.tokens)

    # pretrained tokenizers ModernBERT
    mb_encoding = mb_tokenizer(sentence, add_special_tokens=False, return_token_type_ids=True )
    mb_tokens = "|".join(mb_tokenizer.convert_ids_to_tokens(mb_encoding["input_ids"]))
    
    # pretrained tokenizers GPT2
    gpt2_encoding = gpt2_tokenizer(sentence, add_special_tokens=False, return_token_type_ids=True )
    gpt2_tokens = "|".join(gpt2_tokenizer.convert_ids_to_tokens(gpt2_encoding["input_ids"]))
    
    # pretrained tokenizers BERT
    bert_encoding = bert_tokenizer(sentence, add_special_tokens=False, return_token_type_ids=True )
    bert_tokens = "|".join(bert_tokenizer.convert_ids_to_tokens(bert_encoding["input_ids"]))
    
    # pretrained tokenizers LLAMA3.2
    llama32_encoding = llama32_tokenizer(sentence, add_special_tokens=False, return_token_type_ids=True )
    llama32_tokens = "|".join(llama32_tokenizer.convert_ids_to_tokens(llama32_encoding["input_ids"]))
    
    # pretrained tokenizers XLNet
    xlnet_encoding = xlnet_tokenizer(sentence, add_special_tokens=False, return_token_type_ids=True )
    xlnet_tokens = "|".join(xlnet_tokenizer.convert_ids_to_tokens(xlnet_encoding["input_ids"]))
    
    # pretrained tokenizers ALBERT
    albert_encoding = albert_tokenizer(sentence, add_special_tokens=False, return_token_type_ids=True )
    albert_tokens = "|".join(albert_tokenizer.convert_ids_to_tokens(albert_encoding["input_ids"]))
    
    print(f"uni_sp:{uni_sp_tokens}\nmodernbert: {mb_tokens}\nbert: {bert_tokens}\nllama32: {llama32_tokens}\ngpt2: {gpt2_tokens}\nxlnet: {xlnet_tokens}\nalbert: {albert_tokens}\n")

#%%
uni_sp_tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A:0 </s>:0 <cls>:2",
    #pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    pair="<s> $A:0 <sep>:0 $B:1 <sep>:1 </s> <cls>:2",
    special_tokens=[
        ("<sep>", uni_sp_tokenizer.token_to_id("<sep>")), 
        ("<cls>", uni_sp_tokenizer.token_to_id("<cls>")),
        ("<s>", uni_sp_tokenizer.token_to_id("<s>")),
        ("</s>", uni_sp_tokenizer.token_to_id("</s>")),
        ],
)

uni_sp_tokenizer.decoder = decoders.Metaspace()

#%% encode the sentences again with the post processor

encoded_sentences = uni_sp_tokenizer.encode_batch(test_sentences)

# print the encoded sentences
for encoded_sentence in encoded_sentences:
    print(encoded_sentence.tokens)
    print(encoded_sentence.ids)
    print(encoded_sentence.attention_mask)
    print(encoded_sentence.type_ids)
    
    
#%% decode the sentences

# print the decoded sentences
for encoding in encoded_sentences:
    decoded_sentence = uni_sp_tokenizer.decode(encoding.ids)
    print(encoding.tokens)
    print(decoded_sentence)




#%% morfessor tokenization

MORFESSOR_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Cache", "Morfessor"))
# already trained models

# trained on nltk word corpus
morfessor_nltk_en_model_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_model.bin")
morfessor_nltk_en_train_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_nltk_en_train.txt")

# trained on wikipedia corpus
morfessor_wikipedia_en_train_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train.txt")
morfessor_wikipedia_en_model_file = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_model.bin")

# trained on wikipedia corpus sentences
morfessor_wikipedia_en_train_file_sentences = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_train_sentences.txt")
morfessor_wikipedia_en_model_file_sentences = os.path.join(MORFESSOR_CACHE_DIR, "morfessor_wikipedia_en_model_sentences.bin")


#%% load the models
import morfessor

io = morfessor.MorfessorIO()
model_nltk = io.read_binary_model_file(morfessor_nltk_en_model_file)
model_wiki = io.read_binary_model_file(morfessor_wikipedia_en_model_file)
model_sent = io.read_binary_model_file(morfessor_wikipedia_en_model_file_sentences)


# %%

for line in simple_words_corpus[:1]:
    for word in line.split():
        print(f"nltk={model_nltk.viterbi_segment(word)}\nwiki={model_wiki.viterbi_segment(word)}")
        
#%%        
for line in test_sentences[:1]:
    print(f"nltk= {model_nltk.viterbi_segment(line)}")
    print(f"wiki= {model_wiki.viterbi_segment(line)}")
    print(f"sent= {model_sent.viterbi_segment(line)}")
    
    
    
#%% anlayze the tokenization quality by means of bleu score

# Derivational morphology
    "The recyclable materials underwent repurposing through decentralised reprocessing systems.",
    "Unpredictable weather patterns necessitate preemptive disaster preparedness measures.",
    
    # Inflectional + derivational combinations
    "Misinterpretations of antiestablishmentarianism could lead to counterproductive deregulations.",
    "Overcompensating hyperactive employees often demonstrate underappreciated self-sacrificial tendencies.",
    
    # Multiple affixation
    "Postmodern pseudointellectualism frequently mischaracterizes interdisciplinary socioeconomic transformations.",
    "Premarital counseling helps prevent irreconcilable differences in non-traditional cohabitational arrangements.",
    
    # Challenging compounds
    "The neuropsychologist studied psychophysiological interrelationships in decision-making processes.",
    "Semiconductor nanotechnology enables miniaturized transcontinental telecommunications.",
    
    # Negative prefixes
    "Disinformation campaigns increasingly utilize anti-scientific rhetoric to undermine evidence-based policymaking.",
    "Nonpartisan oversight committees investigate unconstitutional overreach in counterterrorism operations."


ground_truth_morphems = {
    
    "windsurfing" : ["wind surf ing", "wind surfing"],
    "kitesurfing" : ["kite surf ing", "kite surfing"],
    "unfortunately" : ["un fortunately", "un fortunate ly"],
    "postmodern" : ["post modern"],
    "premarital" : ["premar ital", "pre martial"],
    "neuropsychologist" : ["neuro psychologist", "neuro psycho logist"],
    "non-traditional" : ["non - traditional", "non- traditional"],
    "nationwide" : ["nation wide"],
    "disinformation" : ["dis information", "dis informa tion"],
    "overcompensating" : ["over compensate", "over compensate ing"],
    "overreach" : ["over reach"],
    "socioeconomic" : ["socio economic"],
    "counterproductive" : ["counter productive", "counter produc tive"],
    "reprocessing" : ["reprocess ing", "re process ing", "re processing"],
    "transcontinental" : ["trans continent al", "trans continental"],
    "interrelationship" : ["inter relation ship", "inter relationship"],
    "telecommunications" : ["tele communications", "tele communication s"],
    "overworked" : ["over work ed"],
    "overthinker" : ["over think er"],
    "madness" : ["mad ness"],
    "sadness" : ["sad ness"],
    "overwhelming" : ["over whelming", "over whelm ing"],
    "working" : ["work ing", "working"],
    "doing" : ["doing", "do ing"],
    "cookbook" : ["cook book", "cookbook"],
    "workplace" : ["work place", "workplace"],
    "workweek" : ["work week"],
    "unfortunately" : ["un fortunate ly", "unfortun ately"],
    "unpredictable" : ["un predict able", "unpredict able", "un predictable"],
    "runner's" : ["runner 's", "runner ' s"],
    "mispronounce" : ["mis pronounce"],
    ""
    "they" : ["they"],
    "he" : ["he"],
    "she" : ["she"],
    "it" : ["it"],
    "they're" : ["they 're", "they ' re"],
    "they've" : ["they 've", "they ' ve"],
    "they'll" : ["they 'll", "they ' ll"],
    "their's" : ["their 's", "their ' s"],
    "it's" : ["it 's", "it ' s"],
    "at" : ["at"],
    "in" : ["in"],
    "on" : ["on"],
    "with" : ["with"],
    "by" : ["by"],
    "for" : ["for"],
    "of" : ["of"],
    "PlayStation" : ["Play Station", "PlayStation"],
    "Xbox" : ["X box", "Xbox"],
    "Nintendo" : ["Nintendo"],
    "Google" : ["Google"],
    "Apple" : ["Apple"],
    "NVIDIA" : ["NVIDIA"],
    "Donald Trump" : ["Donald Trump"],
    "Joe Biden" : ["Joe Biden"],
    "New York" : ["New York"],
    "Bitcoin" : ["Bitcoin", "Bit coin"],
    "Coca-Cola" : ["Coca-Cola", "Coca - Cola"],
    "McDonald's" : ["McDonald's", "McDonald ' s"],
    "Burger King" : ["Burger King"],
    "Wendy's" : ["Wendy's", "Wendy ' s"],
    "Taco Bell" : ["Taco Bell"],
        
}


#%%
#evaluate all the toeknizers with Hugging Face evaluate library with use of BLEU score
# compute the BLEU score for each tokenizer based on the ground_truth_morphems dictionary, where as a key is the word or phrase to tokenize and as values are the lists with references, 
# create a loop over all the tokenizers, tokenize each word then concatenate the tokens with space as a separator, add to metric useing "add" method. At the end print the metrics for each tokenizer.

from rich import box, print
from rich.console import Console
from rich.table import Table

import evaluate

bleu = evaluate.load("bleu")


# Create predictions and references for each tokenizer
tokenizers = {
    'uni_sp': uni_sp_tokenizer,
    'modernbert': mb_tokenizer,
    'bert': bert_tokenizer,
    'llama32': llama32_tokenizer,
    'gpt2': gpt2_tokenizer,
    'xlnet': xlnet_tokenizer,
    'albert': albert_tokenizer,
    'morfessor_nltk': model_nltk,
    'morfessor_wiki': model_wiki,
    'morfessor_sent': model_sent
}

def get_tokenizer_predictions(tokenizer, word):
    if isinstance(tokenizer, Tokenizer):  # For our custom Unigram tokenizer
        tokens = tokenizer.encode(word).tokens
    elif isinstance(tokenizer, morfessor.baseline.BaselineModel):  # For Morfessor models
        tokens = tokenizer.viterbi_segment(word)[0]
    else:  # For HuggingFace tokenizers
        tokens = tokenizer.tokenize(word)
    
    # Clean up special characters and join tokens
    cleaned_tokens = []
    for token in tokens:
        # Remove special characters like Ġ, ▁, ##, etc.
        token = token.replace('Ġ', '').replace('▁', '').replace('##', '')
        if token.strip():  # Only add non-empty tokens
            cleaned_tokens.append(token)
    
    return ' '.join(cleaned_tokens)

# Compute BLEU scores for each tokenizer
bleu_scores = {}

for tokenizer_name, tokenizer in tokenizers.items():
    predictions = []
    references = []
    
    for word, refs in ground_truth_morphems.items():
        pred = get_tokenizer_predictions(tokenizer, word)
        predictions.append(pred)
        references.append(refs)
    
    # Compute BLEU score
    score = bleu.compute(predictions=predictions, references=references, max_order=3)
    bleu_scores[tokenizer_name] = score

# Print results
console = Console()

# Create and display BLEU scores table
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

#%% Print compact example tokenizations
example_words = list(ground_truth_morphems.keys())

# Create tokenization comparison table
token_table = Table(
    title="Tokenization Comparison",
    box=box.MINIMAL_DOUBLE_HEAD,
    show_lines=True
)
token_table.add_column("Word", style="bright_cyan", no_wrap=True)
token_table.add_column("Ground Truth", style="bright_yellow")
token_table.add_column("Tokenizer", style="bright_magenta")
token_table.add_column("Tokens", style="bright_green")

# Add rows to the table
for word in example_words:
    first_row = True
    for tokenizer_name, tokenizer in tokenizers.items():
        pred = get_tokenizer_predictions(tokenizer, word)
        # Add word and ground truth only for the first tokenizer
        token_table.add_row(
            word if first_row else "",
            " | ".join(ground_truth_morphems[word]) if first_row else "",
            tokenizer_name,
            pred
        )
        first_row = False

console.print(token_table)

# %%
