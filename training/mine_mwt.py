import os
import argparse
from collections import Counter
import math
import re
from typing import List, Dict, Set
import multiprocessing

from datasets import load_dataset
from tqdm import tqdm
import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def mine_batch(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Process a batch of texts to find candidate concepts across all domains.
    Returns a dict with lists of candidates for aggregation.
    """
    texts = batch['text']
    
    # 1. Text Collocations (Simplified for batch processing)
    # Full PMI requires global stats, so we collect raw n-grams here and filter globally later
    text_candidates = []
    for text in texts:
        # Keep only alphanumeric and basic punctuation
        clean_text = re.sub(r'[^\w\s-]', '', text.lower())
        words = clean_text.split()
        if len(words) < 2: continue
        
        # Collect raw bigrams/trigrams for global counting
        # (Naive sliding window is faster for map/reduce than NLTK per doc)
        for i in range(len(words)-1):
            text_candidates.append(f"{words[i]} {words[i+1]}")
        for i in range(len(words)-2):
            text_candidates.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    # 2. Code Idioms
    code_patterns = [
        r'import\s+\w+\s+as\s+\w+',      # import numpy as np
        r'from\s+\w+\s+import\s+\w+',    # from typing import List
        r'def\s+__init__\(self',         # def __init__(self
        r'if\s+__name__\s+==\s+[\'"]__main__[\'"]:', # if __name__ == "__main__":
        r'for\s+\w+\s+in\s+range\(',     # for i in range(
        r'return\s+True',
        r'return\s+False',
        r'raise\s+ValueError\(',
        r'class\s+\w+\(.*\):'            # class MyClass(object):
    ]
    
    code_candidates = []
    for text in texts:
        # Heuristic: Only scan "code-looking" texts
        if "def " in text or "import " in text or "{" in text:
            for pattern in code_patterns:
                matches = re.findall(pattern, text)
                code_candidates.extend(matches)
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if 4 < len(line) < 40 and line.endswith((':', ')', ';')):
                     code_candidates.append(line)

    # 3. Math/LaTeX
    latex_patterns = [
        r'\\[a-zA-Z]+', 
        r'\\[a-zA-Z]+\{[^}]+\}', 
        r'\\begin\{[^}]+\}', 
        r'\\end\{[^}]+\}',
    ]
    math_phrases = [
        "if and only if", "without loss of generality", "random variable", 
        "standard deviation", "neural network", "machine learning", 
        "artificial intelligence", "deep learning"
    ]
    
    math_candidates = []
    for text in texts:
        if "\\" in text:
            for pattern in latex_patterns:
                matches = re.findall(pattern, text)
                math_candidates.extend(matches)
        
        text_lower = text.lower()
        for p in math_phrases:
            if p in text_lower:
                math_candidates.append(p)

    return {
        "text_candidates": text_candidates, 
        "code_candidates": code_candidates, 
        "math_candidates": math_candidates
    }

def main():
    parser = argparse.ArgumentParser(description="Mine Multi-Word Tokens (MWT) from Dataset (Multiprocessing)")
    parser.add_argument("--dataset", type=str, default="JeanKaddour/minipile", help="HF Dataset")
    parser.add_argument("--samples", type=int, default=1_000_000, help="Number of samples to scan")
    parser.add_argument("--output", type=str, default="mwt", help="Output filename prefix (e.g. 'mwt' -> mwt_text.txt)")
    parser.add_argument("--min_freq", type=int, default=100, help="Minimum frequency to keep a MWT")
    args = parser.parse_args()
    
    print(f"Loading {args.samples} samples from {args.dataset}...")
    # Load fully into RAM since Polonez is powerful
    dataset = load_dataset(args.dataset, split="train", streaming=False)
    
    if len(dataset) > args.samples:
        dataset = dataset.select(range(args.samples))
    
    print(f"Processing {len(dataset)} documents with multiprocessing...")
    num_proc = max(1, multiprocessing.cpu_count() - 2)
    
    # 1. Map: Extract raw candidates in parallel
    results = dataset.map(
        mine_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names, # We only want the candidates back
        desc="Mining candidates"
    )
    
    # 2. Save MWTs by Domain (Skip the single monolithic counter)
    print(f"Filtering (min_freq={args.min_freq}) and Saving to MWT files...")
    
    def save_mwt(filename, domain_name):
        # Optimized aggregation by domain
        domain_counter = Counter()
        
        # Iterate over the results dataset again to sum up counts for this specific domain
        # (This is slightly inefficient to iterate 3 times, but safe for memory)
        print(f"  Aggregating {domain_name} MWTs...")
        col_name = f"{domain_name}_candidates"
        
        # Use a generator expression for memory efficiency if dataset is huge, 
        # but here we can iterate batches
        for batch in tqdm(results.iter(batch_size=10_000), total=len(results)//10_000):
             for sublist in batch[col_name]:
                 domain_counter.update(sublist)
                 
        # Filter
        valid_mwt = []
        for mwt, count in domain_counter.most_common(5000):
            if count >= args.min_freq:
                if len(mwt) < 3 or len(mwt) > 50: continue
                valid_mwt.append(mwt)
        
        valid_mwt = sorted(valid_mwt)
        print(f"  Saving {len(valid_mwt)} {domain_name} MWTs to {filename}...")
        
        with open(filename, "w", encoding="utf-8") as f:
            for m in valid_mwt:
                f.write(m + "\n")

    # Save separated files
    prefix = os.path.splitext(args.output)[0]
    if prefix.endswith("_mwt"): prefix = prefix[:-4] # clean up if user provided 'foo_mwt.txt'
    
    save_mwt(f"{prefix}_mwt_text.txt", "text")
    save_mwt(f"{prefix}_mwt_code.txt", "code")
    save_mwt(f"{prefix}_mwt_math.txt", "math")
            
    print("Done. MWT files generated.")

if __name__ == "__main__":
    main()