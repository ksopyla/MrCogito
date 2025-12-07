"""
Unit tests for Multi-Word Token (MWT) Mining.

These tests act as safety nets to ensure we don't lose good extraction patterns
while improving noise reduction.

Test Categories:
1. Code Idiom Extraction (multi-language)
2. Math/LaTeX Pattern Recognition
3. Entity Detection (abbreviations, titles)
4. Statistical Collocation Quality
"""

import pytest
import sys
import os
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.mine_mwt import (
    mine_batch,
    is_code_document,
    is_math_document,
    CODE_PATTERNS,
    ALL_CODE_PATTERNS
)


class TestDocumentClassification:
    """Test document type detection heuristics."""
    
    def test_code_detection_python(self):
        """Should detect Python code."""
        text = """
        def calculate_sum(a, b):
            return a + b
        
        class MyClass:
            def __init__(self):
                pass
        """
        assert is_code_document(text) is True
    
    def test_code_detection_cpp(self):
        """Should detect C++ code."""
        text = """
        #include <iostream>
        
        std::vector<int> numbers;
        std::string name = "test";
        """
        assert is_code_document(text) is True
    
    def test_code_detection_sql(self):
        """Should detect SQL."""
        text = """
        SELECT * FROM users WHERE age > 21;
        INSERT INTO table_name VALUES (1, 'test');
        """
        assert is_code_document(text) is True
    
    def test_math_detection_latex(self):
        """Should detect LaTeX math."""
        text = r"""
        The equation is:
        \begin{equation}
        E = mc^2
        \end{equation}
        where \alpha and \beta are constants.
        """
        assert is_math_document(text) is True
    
    def test_natural_language_not_code(self):
        """Natural language should not be classified as code."""
        text = """
        This is a regular English sentence. It talks about
        various topics like history and science. No code here.
        """
        assert is_code_document(text) is False
        assert is_math_document(text) is False


class TestCodeIdiomExtraction:
    """Test extraction of code idioms across languages."""
    
    def test_python_patterns(self):
        """Should extract Python idioms."""
        texts = [
            "def __init__(self, name):",
            "from typing import List",
            "import numpy as np",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        code_found = []
        for doc_codes in result["code_candidates"]:
            code_found.extend(doc_codes)
        
        assert any("__init__" in c for c in code_found), "Should find __init__"
        assert any("import" in c for c in code_found), "Should find import patterns"
    
    def test_cpp_patterns(self):
        """Should extract C++ idioms."""
        texts = [
            "std::vector<int> v;",
            "std::string name;",
            "#include <iostream>",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        code_found = []
        for doc_codes in result["code_candidates"]:
            code_found.extend(doc_codes)
        
        assert any("std::" in c for c in code_found), "Should find std:: patterns"
    
    def test_javascript_patterns(self):
        """Should extract JavaScript idioms."""
        texts = [
            "const myFunc = () => { console.log('test'); }",
            "let x = 10;",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        code_found = []
        for doc_codes in result["code_candidates"]:
            code_found.extend(doc_codes)
        
        assert any("=>" in c for c in code_found), "Should find arrow functions"
        assert any("console.log" in c for c in code_found), "Should find console.log"
    
    def test_sql_patterns(self):
        """Should extract SQL patterns."""
        texts = [
            "SELECT * FROM users;",
            "INSERT INTO table_name VALUES (1, 2);",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        code_found = []
        for doc_codes in result["code_candidates"]:
            code_found.extend(doc_codes)
        
        assert any("SELECT" in c.upper() for c in code_found), "Should find SELECT"


class TestMathLatexExtraction:
    """Test LaTeX and mathematical pattern extraction."""
    
    def test_greek_letters(self):
        """Should extract Greek letter commands."""
        texts = [
            r"The variables \alpha and \beta are important.",
            r"We use \gamma, \delta, and \epsilon in the proof.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        math_found = []
        for doc_math in result["math_candidates"]:
            math_found.extend(doc_math)
        
        assert any("alpha" in m for m in math_found), "Should find \\alpha"
        assert any("beta" in m for m in math_found), "Should find \\beta"
    
    def test_latex_environments(self):
        """Should extract LaTeX environments."""
        texts = [
            r"\begin{equation} x = y \end{equation}",
            r"\begin{aligned} a &= b \\ c &= d \end{aligned}",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        math_found = []
        for doc_math in result["math_candidates"]:
            math_found.extend(doc_math)
        
        assert any("begin{equation}" in m for m in math_found), "Should find equation env"
        assert any("begin{aligned}" in m for m in math_found), "Should find aligned env"
    
    def test_math_phrases(self):
        """Should extract common math phrases."""
        texts = [
            "This holds if and only if x > 0.",
            "Without loss of generality, assume n is even.",
            "Let X be a random variable.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        math_found = []
        for doc_math in result["math_candidates"]:
            math_found.extend(doc_math)
        
        assert "if and only if" in math_found, "Should find 'if and only if'"
        assert "without loss of generality" in math_found, "Should find 'without loss of generality'"
        assert "random variable" in math_found, "Should find 'random variable'"


class TestEntityExtraction:
    """Test extraction of entities (abbreviations, titles, units)."""
    
    def test_titles(self):
        """Should extract professional titles."""
        texts = [
            "Dr. Smith is a professor.",
            "Mr. Jones and Mrs. Williams attended.",
            "Gen. Patton led the troops.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        entity_found = []
        for doc_entities in result["entity_candidates"]:
            entity_found.extend(doc_entities)
        
        assert any("Dr." in e for e in entity_found), "Should find Dr."
        assert any("Mr." in e for e in entity_found), "Should find Mr."
        assert any("Gen." in e for e in entity_found), "Should find Gen."
    
    def test_abbreviations(self):
        """Should extract common abbreviations."""
        texts = [
            "The method works well, e.g., in simple cases.",
            "Many authors (Smith et al., 2020) studied this.",
            "The result, i.e., the final value, is important.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        entity_found = []
        for doc_entities in result["entity_candidates"]:
            entity_found.extend(doc_entities)
        
        assert any("e.g." in e for e in entity_found), "Should find e.g."
        assert any("et al." in e for e in entity_found), "Should find et al."
        assert any("i.e." in e for e in entity_found), "Should find i.e."
    
    def test_currencies(self):
        """Should extract currency codes."""
        texts = [
            "The price is 100 USD.",
            "Exchange rate: EUR to GBP.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        entity_found = []
        for doc_entities in result["entity_candidates"]:
            entity_found.extend(doc_entities)
        
        assert any("USD" in e for e in entity_found), "Should find USD"
        assert any("EUR" in e or "GBP" in e for e in entity_found), "Should find EUR or GBP"
    
    def test_roman_numerals(self):
        """Should extract Roman numerals."""
        texts = [
            "World War II was devastating.",
            "Chapter VII discusses the results.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        entity_found = []
        for doc_entities in result["entity_candidates"]:
            entity_found.extend(doc_entities)
        
        # Roman numerals should be detected
        assert any("II" in e or "VII" in e for e in entity_found), "Should find Roman numerals"


class TestStatisticalCollocation:
    """Test statistical n-gram collection (not scoring, just collection)."""
    
    def test_natural_language_collection(self):
        """Should collect n-grams from natural language."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        statistical_found = []
        for doc_stats in result["statistical_candidates"]:
            statistical_found.extend(doc_stats)
        
        # Should find some common bigrams
        assert len(statistical_found) > 0, "Should collect some n-grams"
        # Common phrases should be there
        assert any("machine learning" in s.lower() for s in statistical_found), "Should find 'machine learning'"
    
    def test_proper_noun_preservation(self):
        """Should preserve capitalization for proper nouns."""
        texts = [
            "New York is a major city.",
            "Abraham Lincoln was president.",
            "World War II ended in 1945.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        statistical_found = []
        for doc_stats in result["statistical_candidates"]:
            statistical_found.extend(doc_stats)
        
        # Should preserve capitalization
        assert any("New York" in s for s in statistical_found), "Should find 'New York' capitalized"
    
    def test_math_terms_in_text(self):
        """Should find math terms written in natural language."""
        texts = [
            "A prime number is only divisible by itself.",
            "Calculate the common denominator.",
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        statistical_found = []
        for doc_stats in result["statistical_candidates"]:
            statistical_found.extend(doc_stats)
        
        # Should find math terms
        assert any("prime number" in s.lower() for s in statistical_found), "Should find 'prime number'"
    
    def test_code_excluded_from_statistical(self):
        """Code documents should not contribute to statistical n-grams."""
        texts = [
            """
            def my_function():
                return True
            class MyClass:
                pass
            """,
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        # Statistical candidates should be empty or minimal for code docs
        statistical_found = []
        for doc_stats in result["statistical_candidates"]:
            statistical_found.extend(doc_stats)
        
        # Code doc should have been skipped for statistical analysis
        assert len(statistical_found) == 0, "Code should not generate statistical candidates"


class TestNoiseReduction:
    """Tests to ensure noise patterns are filtered out (regression tests)."""
    
    def test_no_svn_metadata_in_output(self):
        """SVN metadata should not appear in final output."""
        # This is a regression test - will fail initially, pass after fixes
        texts = [
            """
            node-path: trunk/src
            content-length: 1234
            props-end
            """,
        ]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        # Check that SVN terms don't leak through
        all_candidates = []
        for doc_stats in result["statistical_candidates"]:
            all_candidates.extend(doc_stats)
        
        # After improvements, these should be filtered
        # (This test will fail in Experiment 1, pass in Experiment 2)
        svn_terms = ["node-path", "content-length", "props-end"]
        for term in svn_terms:
            assert not any(term in c for c in all_candidates), f"Should filter out {term}"


class TestReturnStructure:
    """Test that return structure is correct for HuggingFace datasets."""
    
    def test_return_format(self):
        """mine_batch should return Dict[str, List[List[str]]]."""
        texts = ["Sample text one.", "Sample text two."]
        batch = {"text": texts}
        result = mine_batch(batch)
        
        # Check structure
        assert isinstance(result, dict), "Should return dict"
        assert "statistical_candidates" in result
        assert "code_candidates" in result
        assert "math_candidates" in result
        assert "entity_candidates" in result
        
        # Each value should be a list of lists
        for key in result:
            assert isinstance(result[key], list), f"{key} should be list"
            assert len(result[key]) == len(texts), f"{key} should have one list per input"
            for item in result[key]:
                assert isinstance(item, list), f"Items in {key} should be lists"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

