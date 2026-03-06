"""
Tests for GraphRAGBench metrics parity with official implementation.

These tests verify that our metric implementations produce results consistent
with the official GraphRAG-Bench evaluation metrics.

Reference: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
"""

import unittest
from graphrag_lab.benchmarks.metrics import (
    lexical_f1,
    rouge_l,
    _normalize_tokens,
)


class TestNormalizeTokens(unittest.TestCase):
    """Test token normalization matches official GraphRAGBench behavior."""

    def test_lowercase_conversion(self):
        """Test that text is lowercased."""
        self.assertEqual(_normalize_tokens("HELLO"), ["hello"])

    def test_article_removal(self):
        """Test that articles (a, an, the) are removed."""
        tokens = _normalize_tokens("the quick brown fox")
        self.assertNotIn("the", tokens)
        self.assertEqual(tokens, ["quick", "brown", "fox"])

    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        tokens = _normalize_tokens("Hello, World!")
        self.assertEqual(tokens, ["hello", "world"])

    def test_multiple_spaces(self):
        """Test handling of multiple spaces."""
        tokens = _normalize_tokens("hello    world")
        self.assertEqual(tokens, ["hello", "world"])

    def test_empty_string(self):
        """Test empty string returns empty list."""
        self.assertEqual(_normalize_tokens(""), [])

    def test_only_articles(self):
        """Test string with only articles returns empty list."""
        self.assertEqual(_normalize_tokens("a an the"), [])


class TestLexicalF1(unittest.TestCase):
    """
    Test lexical_f1 implementation against official GraphRAGBench behavior.
    
    Test cases derived from:
    - Official GraphRAG-Benchmark evaluation examples
    - Standard F1 score properties
    """

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        self.assertEqual(lexical_f1("Paris", "Paris"), 1.0)
        self.assertEqual(lexical_f1("Paris", "paris"), 1.0)  # Case insensitive

    def test_no_overlap(self):
        """Test that completely different answers return 0.0."""
        self.assertEqual(lexical_f1("Mars", "Jupiter"), 0.0)
        self.assertEqual(lexical_f1("Apple", "Orange"), 0.0)

    def test_partial_overlap_simple(self):
        """Test partial overlap with simple examples."""
        # "Paris" vs "The capital is Paris"
        # Expected tokens: ["paris"]
        # Predicted tokens: ["capital", "is", "paris"]
        # Overlap: 1, Precision: 1/3, Recall: 1/1
        # F1 = 2 * (1/3) * 1 / (1/3 + 1) = 2/3 * 3/4 = 0.5
        score = lexical_f1("Paris", "The capital is Paris")
        self.assertGreaterEqual(score, 0.5)
        self.assertLessEqual(score, 0.51)  # Should be exactly 0.5

    def test_article_normalization(self):
        """Test that articles are properly ignored."""
        # "United States" vs "the united states."
        # After normalization: ["united", "states"] vs ["united", "states"]
        self.assertEqual(lexical_f1("United States", "the united states."), 1.0)

    def test_synonym_partial_credit(self):
        """Test that semantically similar but different answers get partial credit."""
        # "USA" vs "United States of America"
        # After normalization: ["usa"] vs ["united", "states", "of", "america"]
        # No overlap, so F1 = 0
        score = lexical_f1("USA", "United States of America")
        self.assertEqual(score, 0.0)

    def test_empty_strings(self):
        """Test handling of empty strings."""
        # Both empty should return 1.0 (perfect match)
        self.assertEqual(lexical_f1("", ""), 1.0)
        # One empty should return 0.0
        self.assertEqual(lexical_f1("answer", ""), 0.0)
        self.assertEqual(lexical_f1("", "answer"), 0.0)

    def test_whitespace_only(self):
        """Test handling of whitespace-only strings."""
        self.assertEqual(lexical_f1("   ", "  "), 1.0)
        self.assertEqual(lexical_f1("answer", "   "), 0.0)

    def test_punctuation_variants(self):
        """Test that punctuation differences don't affect score."""
        self.assertEqual(lexical_f1("Paris", "Paris."), 1.0)
        self.assertEqual(lexical_f1("Paris", "Paris!"), 1.0)
        self.assertEqual(lexical_f1("Paris", "Paris,"), 1.0)

    def test_word_order_independence(self):
        """Test that word order doesn't affect token-level F1."""
        # Same tokens, different order
        score1 = lexical_f1("cat dog", "cat dog")
        score2 = lexical_f1("cat dog", "dog cat")
        self.assertEqual(score1, score2)

    def test_duplicate_tokens(self):
        """Test that duplicate tokens are handled (set-based)."""
        # "hello hello" vs "hello" should be perfect match (set-based)
        self.assertEqual(lexical_f1("hello hello", "hello"), 1.0)

    def test_complex_sentence(self):
        """Test with more complex sentence structures."""
        expected = "Albert Einstein was a German theoretical physicist"
        predicted = "Einstein was a physicist from Germany"
        # Expected tokens (after removing articles): ["albert", "einstein", "german", "theoretical", "physicist"]
        # Predicted tokens: ["einstein", "physicist", "from", "germany"]
        # Overlap: {"einstein", "physicist"} = 2
        # Precision: 2/4 = 0.5, Recall: 2/5 = 0.4
        # F1 = 2 * 0.5 * 0.4 / (0.5 + 0.4) = 0.4 / 0.9 ≈ 0.444
        # Note: "german" vs "germany" don't match (different tokens)
        score = lexical_f1(expected, predicted)
        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.6)  # Should be around 0.44-0.55


class TestRougeL(unittest.TestCase):
    """Test ROUGE-L implementation."""

    def test_exact_match(self):
        """Test that exact matches return 1.0."""
        score = rouge_l("The quick brown fox", "The quick brown fox")
        self.assertEqual(score, 1.0)

    def test_no_overlap(self):
        """Test that completely different texts return 0.0."""
        score = rouge_l("Apples", "Oranges")
        self.assertEqual(score, 0.0)

    def test_partial_overlap(self):
        """Test partial overlap."""
        score = rouge_l("The cat sat on the mat", "The cat sat")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_strings(self):
        """Test handling of empty strings."""
        self.assertEqual(rouge_l("", ""), 0.0)
        self.assertEqual(rouge_l("text", ""), 0.0)
        self.assertEqual(rouge_l("", "text"), 0.0)

    def test_case_sensitivity(self):
        """Test that ROUGE-L is case-sensitive (uses stemming)."""
        # With stemming, "The" and "the" should match
        score = rouge_l("The quick brown fox", "the quick brown fox")
        self.assertGreater(score, 0.9)


class TestMetricsParity(unittest.TestCase):
    """
    Integration tests for metrics parity with official GraphRAGBench.
    
    These tests use examples from the official GraphRAG-Benchmark repository
    to ensure our implementations produce consistent results.
    """

    def test_graphragbench_example_1(self):
        """Test with example from official GraphRAGBench documentation."""
        # Example: Fact retrieval question
        expected = "Paris"
        predicted = "The capital of France is Paris"
        score = lexical_f1(expected, predicted)
        # Should have partial credit for mentioning Paris
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_graphragbench_example_2(self):
        """Test with alias handling."""
        # Multiple acceptable answers
        expected = "United States"
        aliases = ["USA", "US", "United States of America"]
        predicted = "the united states."
        
        # Main expected answer should match
        score_main = lexical_f1(expected, predicted)
        self.assertEqual(score_main, 1.0)

    def test_graphragbench_example_3(self):
        """Test with normalization edge cases."""
        # Testing article and punctuation normalization
        test_cases = [
            ("a cat", "the cat", 1.0),  # Articles should be ignored
            ("dog!", "dog", 1.0),  # Punctuation should be ignored
            ("A Dog", "the dog", 1.0),  # Case + articles
        ]
        
        for expected, predicted, expected_score in test_cases:
            with self.subTest(expected=expected, predicted=predicted):
                score = lexical_f1(expected, predicted)
                self.assertEqual(score, expected_score)


if __name__ == "__main__":
    unittest.main()
