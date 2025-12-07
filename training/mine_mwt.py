"""
Mine Multi-Word Tokens (MWT) from Dataset using Domain-Appropriate Methods.

Methodology:
- Natural Language: Statistical (PMI / Likelihood Ratio) - words co-occur due to SEMANTIC relationships
- Code: Template/Frequency (idioms are SYNTACTIC, not probabilistic)
- Math/LaTeX: Grammar-Aware Extraction (LaTeX has a defined GRAMMAR)

This separation is methodologically necessary because statistical co-occurrence
methods assume a "chance baseline" that doesn't exist for deterministic syntax.
"""

import os
import argparse
from collections import Counter
import re
from typing import List, Dict, Set
import multiprocessing

from datasets import load_dataset
from tqdm import tqdm
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Get English stopwords
ENGLISH_STOPWORDS = set(stopwords.words('english'))

# Add common auxiliaries and pronouns that create grammatical (not conceptual) collocations
GRAMMATICAL_WORDS = ENGLISH_STOPWORDS | {
    'is', 'are', 'was', 'were', 'be', 'been', 'being',  # Auxiliaries
    'has', 'have', 'had', 'do', 'does', 'did',
    'will', 'would', 'should', 'could', 'may', 'might', 'must',
    'he', 'she', 'it', 'they', 'we', 'you', 'i',  # Pronouns
    'what', 'which', 'who', 'where', 'when', 'how', 'why',  # Question words
    'let', 'suppose', 'if', 'then', 'else',  # Pseudocode markers
    're', 'll', 've', 't',  # Contractions
}


# ============================================================================
# CODE IDIOM TEMPLATES (Multi-Language)
# These are SYNTACTIC patterns, not semantic collocations.
# We use frequency ranking, NOT statistical significance.
# ============================================================================
CODE_PATTERNS = {
    # Python
    'python': [
        r'import\s+\w+\s+as\s+\w+',           # import numpy as np
        r'from\s+\w+\s+import\s+\w+',         # from typing import List
        r'def\s+__init__\s*\(',               # def __init__(
        r'if\s+__name__\s*==\s*[\'"]__main__[\'"]:', 
        r'for\s+\w+\s+in\s+range\s*\(',
        r'with\s+open\s*\(',
        r'except\s+\w+\s+as\s+\w+',           # except Exception as e
        r'@\w+\.\w+',                          # @property.setter
        r'self\.\w+\s*=',                      # self.x =
        r'lambda\s+\w+\s*:',
    ],
    # C/C++
    'cpp': [
        r'#include\s*<\w+>',                   # #include <iostream>
        r'#include\s*"\w+\.h"',                # #include "mylib.h"
        r'std::\w+',                           # std::vector, std::string
        r'namespace\s+\w+',
        r'using\s+namespace\s+\w+',
        r'template\s*<',
        r'nullptr',
        r'auto\s+\w+\s*=',
    ],
    # JavaScript/TypeScript
    'javascript': [
        r'const\s+\w+\s*=',
        r'let\s+\w+\s*=',
        r'import\s+\{[^}]+\}\s+from',          # import { x } from
        r'export\s+default',
        r'async\s+function',
        r'await\s+\w+',
        r'=>\s*\{',                             # arrow function
        r'console\.log\s*\(',
    ],
    # SQL
    'sql': [
        r'SELECT\s+\*\s+FROM',
        r'INSERT\s+INTO',
        r'UPDATE\s+\w+\s+SET',
        r'DELETE\s+FROM',
        r'WHERE\s+\w+\s*=',
        r'ORDER\s+BY',
        r'GROUP\s+BY',
        r'JOIN\s+\w+\s+ON',
        r'CREATE\s+TABLE',
    ],
    # Rust
    'rust': [
        r'fn\s+\w+\s*\(',
        r'let\s+mut\s+\w+',
        r'impl\s+\w+',
        r'pub\s+fn',
        r'use\s+\w+::',
        r'match\s+\w+\s*\{',
        r'Option<\w+>',
        r'Result<\w+',
    ],
}

# Flatten all code patterns for easy iteration
ALL_CODE_PATTERNS = []
for lang_patterns in CODE_PATTERNS.values():
    ALL_CODE_PATTERNS.extend(lang_patterns)


def is_technical_artifact_document(text: str) -> bool:
    """
    Detect Java/XML config files, .NET assemblies, database dumps, and other
    technical artifacts that create noise in statistical extraction.
    """
    artifact_markers = [
        'servlet-mapping', 'servlet-class', 'servlet-name', 'url-pattern',  # Java web.xml
        'publickeytoken', 'culture neutral', 'version culture',  # .NET assembly
        'drop procedure', 'create table', 'alter table', 'db dba',  # SQL dumps
        'syncml_exec', 'controlpanel', 'mockey ui',  # Various build/config tools
        'xmlns:', '<servlet>', '<?xml',  # XML markers
        'uart', 'gpio', 'i2c', 'spi',  # Hardware/embedded specs (UART0, B5, etc.)
        'msgstr', 'msgid',  # Translation/localization files
        '.properties', 'web.config',  # Config file markers
    ]
    # Count markers for better detection
    marker_count = sum(1 for marker in artifact_markers if marker in text.lower())
    return marker_count >= 2  # Require 2+ markers to classify as artifact


def is_code_document(text: str) -> bool:
    """Heuristic to detect if a document is primarily code."""
    code_markers = ['def ', 'class ', 'import ', 'function ', '#include', 'SELECT ', 
                    'INSERT ', 'fn ', 'const ', 'let ', 'var ', '{', '}', '();', 
                    'std::', 'return ', 'void ', 'int ', 'async ', '__init__', 
                    'from ', '::', 'VALUES']
    marker_count = sum(1 for m in code_markers if m in text)
    # Threshold of 1 for minimal snippets, 2 for better confidence
    return marker_count >= 1


def is_math_document(text: str) -> bool:
    """Heuristic to detect if a document contains significant math/LaTeX."""
    # Check for LaTeX commands OR strong math keywords
    has_latex = '\\' in text and any(cmd in text for cmd in ['\\frac', '\\sum', '\\begin', '\\alpha', '\\beta', '\\gamma', '\\delta'])
    # Only strong algorithmic/proof keywords (not general math terms like "prime number")
    # We want "prime number" to appear in statistical MWTs, so we don't classify it as "math"
    math_keywords = ['theorem', 'proof', 'lemma', 'let be the', 'suppose suppose', 'algorithm:']
    has_strong_math = any(kw in text.lower() for kw in math_keywords)
    return has_latex or has_strong_math


def mine_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[str]]]:
    """
    Process a batch of texts to find candidate MWTs across all domains.
    
    CRITICAL: Returns Dict[str, List[List[str]]] where each inner list corresponds
    to one document in the batch. This preserves batch structure for HuggingFace datasets.
    
    Key Methodological Point:
    - Statistical n-grams are ONLY collected from natural language text
    - Code/Math patterns are extracted via templates (deterministic)
    """
    texts = batch['text']
    batch_size = len(texts)
    
    # Initialize lists of lists (one per document)
    statistical_candidates_batch = []
    code_candidates_batch = []
    math_candidates_batch = []
    entity_candidates_batch = []
    
    for text in texts:
        # Per-document candidate lists
        statistical_candidates = []
        code_candidates = []
        math_candidates = []
        entity_candidates = []
        
        # Classify the document type
        is_code = is_code_document(text)
        is_math = is_math_document(text)
        is_artifact = is_technical_artifact_document(text)
        
        # ============================================================
        # 1. Statistical N-grams (ONLY for Natural Language)
        # ============================================================
        # Methodological Note: PMI assumes "chance baseline" which doesn't
        # exist for code (syntax is deterministic). So we SKIP code docs.
        # Experiment 4: Also skip technical artifacts, but KEEP grammatical collocations
        if not is_code and not is_math and not is_artifact:
            # Pre-processing to reduce noise (Experiment 2 improvements)
            # 1. Filter out SVN/version control lines
            lines = text.split('\n')
            clean_lines = []
            for line in lines:
                # Skip lines with version control markers
                if any(marker in line for marker in ['node-path', 'content-length', 'props-end', 
                                                       'prop-content-length', 'text-content-length']):
                    continue
                # Skip lines that are mostly URLs or DOIs
                if line.count('http://') + line.count('https://') + line.count('doi:') > 0:
                    continue
                # Strip list item numbers (1., 2), 3-)
                line_stripped = line.strip()
                if line_stripped and len(line_stripped) > 2 and line_stripped[0].isdigit():
                    if line_stripped[1] in '.)-' or (len(line_stripped) > 2 and line_stripped[1].isdigit() and line_stripped[2] in '.)-'):
                        # Skip the number prefix
                        first_space = line_stripped.find(' ')
                        line_stripped = line_stripped[first_space+1:] if first_space > 0 else ""
                
                if line_stripped:
                    clean_lines.append(line_stripped)
            
            clean_text = ' '.join(clean_lines)
            
            # 2. Tokenize FIRST (before lowercasing) to preserve capitalization
            words_original = clean_text.split()
            
            # 3. Remove punctuation from each word but keep structure
            words_cleaned = []
            for word in words_original:
                # Remove leading/trailing punctuation but keep internal (for hyphenated words)
                word_clean = word.strip('.,!?;:()"\'')
                if word_clean and len(word_clean) >= 2:
                    words_cleaned.append(word_clean)
            
            # 4. Collect n-grams with ORIGINAL CASE preserved for proper nouns
            if len(words_cleaned) >= 2:
                for i in range(len(words_cleaned)-1):
                    # Preserve capitalization for proper nouns, lowercase for common words
                    w1, w2 = words_cleaned[i], words_cleaned[i+1]
                    
                    # Check if this looks like a proper noun phrase
                    if w1[0].isupper() or w2[0].isupper():
                        # Keep original case
                        ngram = f"{w1} {w2}"
                    else:
                        # Lowercase for common words
                        ngram = f"{w1.lower()} {w2.lower()}"
                    
                    statistical_candidates.append(ngram)
                
                # Trigrams
                for i in range(len(words_cleaned)-2):
                    w1, w2, w3 = words_cleaned[i], words_cleaned[i+1], words_cleaned[i+2]
                    
                    if w1[0].isupper() or w2[0].isupper() or w3[0].isupper():
                        ngram = f"{w1} {w2} {w3}"
                    else:
                        ngram = f"{w1.lower()} {w2.lower()} {w3.lower()}"
                    
                    statistical_candidates.append(ngram)
        
        # ============================================================
        # 2. Code Idioms (Template Matching - NOT Statistical)
        # ============================================================
        if is_code:
            # Filter out academic citations (DOI patterns) before code extraction
            # These should go to entities, not code
            is_citation = text.count('@') > 3 or 'doi:' in text.lower() or '@pone.' in text or '@plos.' in text
            
            if not is_citation:
                for pattern in ALL_CODE_PATTERNS:
                    try:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        code_candidates.extend(matches)
                    except re.error:
                        pass
        
        # ============================================================
        # 3. Math/LaTeX (Grammar-Aware - NOT Statistical)
        # ============================================================
        if is_math:
            # LaTeX commands (syntactic, not semantic)
            latex_patterns = [
                r'\\[a-zA-Z]+',              # \frac, \sum
                r'\\[a-zA-Z]+\{[^}]*\}',     # \mathbb{R}
                r'\\begin\{[^}]+\}',         # \begin{equation}
                r'\\end\{[^}]+\}',
            ]
            for pattern in latex_patterns:
                matches = re.findall(pattern, text)
                math_candidates.extend(matches)
        
        # Math PHRASES (these are semantic, so we check in all docs)
        text_lower = text.lower()
        math_phrases = [
            "if and only if", "without loss of generality", "random variable",
            "standard deviation", "neural network", "machine learning",
            "artificial intelligence", "deep learning", "gradient descent",
            "loss function", "activation function", "hidden layer",
        ]
        for phrase in math_phrases:
            if phrase in text_lower:
                math_candidates.append(phrase)
        
        # ============================================================
        # 4. Entities (Abbreviations, Ranks, Units - Template Based)
        # ============================================================
        # These are structural, not semantic
        has_upper = not text.islower()
        has_digit = any(c.isdigit() for c in text)
        
        if has_upper or has_digit:
            entity_patterns = []
            if has_upper:
                entity_patterns.extend([
                    r'\b[A-Z]\.[A-Z]\.[A-Z]?\.?\b',  # U.S.A.
                    r'\b(?:Dr|Mr|Mrs|Ms|Prof|Gen|Lt|Sgt|Col|Maj|Cpt|Adm|Det|Rev)\.',
                    r'\b(?:II|III|IV|VI|VII|VIII|IX|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI|XXII)\b',
                    r'\b(?:USD|EUR|GBP|JPY|CNY|BTC|ETH)\b',
                    r'\b(?:PhD|MBA|CEO|CFO|CTO|COO|VP|SVP|EVP)\b',
                ])
            if has_digit:
                # Skip measurement units with numbers - these are too specific
                # entity_patterns.extend([
                #     r'\b\d+\s?(?:mln|bln|mil|bil|tril|k|M|B|T)\b',
                #     r'\b\d+(?:\.\d+)?\s?(?:kg|km|cm|mm|ml|mg|GB|MB|KB|TB)\b',
                # ])
                # Only extract scale abbreviations WITHOUT numbers
                entity_patterns.extend([
                    r'\b(?:mln|bln|mil|bil|tril)\b',
                ])
            if "e.g." in text or "i.e." in text or "et al" in text or "vs." in text:
                entity_patterns.extend([r'\be\.g\.', r'\bi\.e\.', r'\bet\s+al\.', r'\bvs\.'])
            
            for pattern in entity_patterns:
                try:
                    matches = re.findall(pattern, text)
                    entity_candidates.extend(matches)
                except re.error:
                    pass
        
        # Append per-document results to batch lists
        statistical_candidates_batch.append(statistical_candidates)
        code_candidates_batch.append(code_candidates)
        math_candidates_batch.append(math_candidates)
        entity_candidates_batch.append(entity_candidates)

    return {
        "statistical_candidates": statistical_candidates_batch,
        "code_candidates": code_candidates_batch,
        "math_candidates": math_candidates_batch,
        "entity_candidates": entity_candidates_batch
    }


def save_mwt_statistical(results, filename: str, min_freq: int, min_score: float = 10.0):
    """
    Uses NLTK's Likelihood Ratio to find true collocations.
    
    Experiment 4 Philosophy:
    - KEEP grammatical collocations (they aid compression and have semantic value)
    - KEEP all high-scoring patterns (concepts + grammar)
    - ONLY filter pure noise (numbers, artifacts)
    - PRIORITIZE capitalized phrases (proper nouns = strong concepts)
    """
    print(f"  Aggregating Statistical Candidates...")
    
    # 1. Build frequency distribution
    ngram_fd = Counter()
    
    for batch in tqdm(results.iter(batch_size=10_000), total=len(results)//10_000 + 1):
        for doc_candidates in batch["statistical_candidates"]:
            ngram_fd.update(doc_candidates)
    
    print(f"    Unique N-grams: {len(ngram_fd)}")
    
    # 2. Separate bigrams and trigrams (preserve case)
    bigram_fd = nltk.FreqDist()
    trigram_list = []
    
    for ngram_str, freq in ngram_fd.items():
        # Normalize to lowercase for statistical analysis
        parts = ngram_str.lower().split()
        
        if len(parts) == 2:
            bigram_fd[tuple(parts)] = freq
        elif len(parts) == 3:
            # Store with original case for later filtering
            trigram_list.append((ngram_str, freq))
    
    # 3. Build word frequency distribution
    word_fd = nltk.FreqDist()
    for (w1, w2), freq in bigram_fd.items():
        word_fd[w1] += freq
        word_fd[w2] += freq
    
    print(f"    Scoring Bigrams with Likelihood Ratio...")
    
    # 4. Use NLTK's BigramCollocationFinder
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder(word_fd, bigram_fd)
    finder.apply_freq_filter(min_freq)
    
    # Score using Likelihood Ratio
    scored_bigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
    
    # 5. Filter ONLY for artifacts and pure noise (NOT grammatical patterns)
    print(f"    Filtering artifacts and noise...")
    valid_mwt = []
    
    for (w1, w2), score in scored_bigrams:
        if score < min_score:
            continue
        
        # Skip pure number patterns
        if (w1.replace('-', '').replace('.', '').isdigit() or 
            w2.replace('-', '').replace('.', '').isdigit()):
            continue
        
        # Skip technical artifact fragments
        artifact_terms = ['servlet', 'publickeytoken', 'controlpanel', 'mockey', 
                         'syncml', 'xmlns', 'dba', 'uart', 'msgstr', 'gpio']
        if any(term in w1 or term in w2 for term in artifact_terms):
            continue
        
        mwt = f"{w1} {w2}"
        
        # Basic length check
        if len(mwt) < 5 or len(mwt) > 50:
            continue
        
        # Find the original case version in ngram_fd
        # Try to find a capitalized version
        mwt_original = mwt
        for original_ngram in ngram_fd:
            if original_ngram.lower() == mwt:
                # Use the most common casing
                mwt_original = original_ngram
                break
        
        # Check if it's a proper noun (capitalized)
        is_proper_noun = mwt_original[0].isupper()
        
        valid_mwt.append((mwt_original, score, is_proper_noun))
    
    # 6. Skip trigrams - user requested bigrams only
    print(f"    Skipping trigrams (bigrams only per user request)...")
    
    # Sort by score/frequency, prioritizing proper nouns
    valid_mwt.sort(key=lambda x: (-x[2], -x[1]))  # Proper nouns first, then by score
    
    # Deduplicate
    seen = set()
    valid_mwt_final = []
    for mwt, score, is_proper in valid_mwt:
        mwt_normalized = mwt.lower()
        if mwt_normalized not in seen:
            seen.add(mwt_normalized)
            valid_mwt_final.append(mwt)
    
    print(f"  Saving {len(valid_mwt_final)} MWTs to {filename}...")
    proper_noun_count = sum(1 for _, _, is_proper in valid_mwt if is_proper)
    print(f"    (Proper nouns: {proper_noun_count}/{len(valid_mwt)})")
    
    with open(filename, "w", encoding="utf-8") as f:
        for m in valid_mwt_final:
            f.write(m + "\n")


def save_mwt_frequency(results, filename: str, col_name: str, domain_name: str, min_freq: int):
    """
    Simple frequency-based filtering for deterministic patterns (Code, Math, Entities).
    
    For these domains, PMI is meaningless because co-occurrence is governed by
    SYNTAX, not SEMANTICS. We just count and filter.
    """
    domain_counter = Counter()
    print(f"  Aggregating {domain_name} MWTs...")
    
    for batch in tqdm(results.iter(batch_size=10_000), total=len(results)//10_000 + 1):
        # Flatten the lists of lists
        for doc_candidates in batch[col_name]:
            domain_counter.update(doc_candidates)
    
    valid_mwt = []
    for mwt, count in domain_counter.most_common(5000):
        if count >= min_freq:
            if len(mwt) < 3 or len(mwt) > 60:
                continue
            valid_mwt.append(mwt)
    
    valid_mwt = sorted(set(valid_mwt))
    print(f"  Saving {len(valid_mwt)} {domain_name} MWTs to {filename}...")
    
    with open(filename, "w", encoding="utf-8") as f:
        for m in valid_mwt:
            f.write(m + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mine Multi-Word Tokens (MWT) using Domain-Appropriate Methods"
    )
    parser.add_argument("--dataset", type=str, default="JeanKaddour/minipile", help="HF Dataset")
    parser.add_argument("--samples", type=int, default=1_000_000, help="Number of samples to scan")
    parser.add_argument("--output", type=str, default="mwt", help="Output filename prefix")
    parser.add_argument("--min_freq", type=int, default=100, help="Minimum frequency for MWT")
    parser.add_argument("--min_score", type=float, default=10.0, help="Minimum Likelihood Ratio score for statistical MWTs")
    args = parser.parse_args()
    
    print(f"Loading {args.samples} samples from {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train", streaming=False)
    
    if len(dataset) > args.samples:
        dataset = dataset.select(range(args.samples))
    
    print(f"Processing {len(dataset)} documents with multiprocessing...")
    num_proc = max(1, multiprocessing.cpu_count() - 2)
    
    # 1. Map: Extract raw candidates in parallel
    # Define explicit features to avoid PyArrow type inference issues
    from datasets import Features, Sequence, Value
    
    output_features = Features({
        "statistical_candidates": Sequence(Value("string")),
        "code_candidates": Sequence(Value("string")),
        "math_candidates": Sequence(Value("string")),
        "entity_candidates": Sequence(Value("string"))
    })
    
    results = dataset.map(
        mine_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        features=output_features,
        desc="Mining candidates"
    )
    
    # 2. Save MWTs using domain-appropriate methods
    print(f"\nFiltering and Saving MWT files...")
    prefix = os.path.splitext(args.output)[0]
    if prefix.endswith("_mwt"):
        prefix = prefix[:-4]
    
    # Statistical (Natural Language) - Uses Likelihood Ratio
    save_mwt_statistical(
        results, 
        f"{prefix}_mwt_statistical.txt", 
        min_freq=args.min_freq,
        min_score=args.min_score
    )
    
    # Code Idioms (Frequency only - syntax is deterministic)
    save_mwt_frequency(
        results, 
        f"{prefix}_mwt_code.txt", 
        "code_candidates", 
        "Code",
        min_freq=max(10, args.min_freq // 10)  # Lower threshold for code (less frequent)
    )
    
    # Math/LaTeX (Frequency only - grammar is deterministic)
    save_mwt_frequency(
        results, 
        f"{prefix}_mwt_math.txt", 
        "math_candidates", 
        "Math",
        min_freq=args.min_freq
    )
    
    # Entities (Frequency only - structured patterns)
    save_mwt_frequency(
        results, 
        f"{prefix}_mwt_entities.txt", 
        "entity_candidates", 
        "Entity",
        min_freq=max(10, args.min_freq // 5)
    )
    
    print("\nDone. MWT files generated:")
    print(f"  - {prefix}_mwt_statistical.txt  (Natural Language - Likelihood Ratio)")
    print(f"  - {prefix}_mwt_code.txt         (Code Idioms - Frequency)")
    print(f"  - {prefix}_mwt_math.txt         (Math/LaTeX - Frequency)")
    print(f"  - {prefix}_mwt_entities.txt     (Entities - Frequency)")


if __name__ == "__main__":
    main()
