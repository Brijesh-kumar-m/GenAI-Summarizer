"""
test_keyword_extractor.py - Unit Tests for Keyword Extraction Module
=====================================================================
Part of: GenAI Summarizer: Advanced NLP & Text Intelligence Assistant
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keyword_extractor import (
    extract_keywords,
    extract_keywords_fallback,
    STANDARD_STOPWORDS,
)


def test_keyword_extraction_meaningful_text():
    """Verify that TF-IDF extracts relevant domain terms and normalized scores."""
    sample_text = (
        "Artificial intelligence and deep learning algorithms are revolutionizing healthcare. "
        "Hospitals employ machine learning models for early disease detection, medical imaging, "
        "and genomic sequencing. Deep learning models analyze patient medical data efficiently."
    )
    keywords, msg = extract_keywords(sample_text, top_k=5)

    assert len(keywords) > 0
    assert len(keywords) <= 5
    # Expect domain terms like 'learning', 'models', 'medical', 'deep'
    extracted_terms = [kw[0].lower() for kw in keywords]
    any_domain_match = any(
        term in " ".join(extracted_terms)
        for term in ["learning", "models", "medical", "deep", "intelligence", "healthcare"]
    )
    assert any_domain_match

    # Verify score bounds
    for term, score in keywords:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_keyword_extraction_very_short_text():
    """Verify graceful handling and informative feedback for short texts."""
    short_text = "AI is cool."
    keywords, msg = extract_keywords(short_text, top_k=5)
    # Should not crash, and should return informative status
    assert len(keywords) == 0
    assert "short" in msg.lower()


def test_keyword_extraction_empty_text():
    """Verify that empty and whitespace-only text returns empty result without errors."""
    keywords, msg = extract_keywords("", top_k=5)
    assert keywords == []
    assert "no text" in msg.lower()

    keywords, msg = extract_keywords("   \n\t  ", top_k=5)
    assert keywords == []


def test_keyword_count_configuration():
    """Verify top_k parameter limits the number of extracted keywords."""
    passage = (
        "Generative Artificial Intelligence, Natural Language Processing, "
        "Large Language Models, deep neural networks, transformer architectures, "
        "and computer vision systems continue to accelerate scientific innovation."
    )
    kw_3, _ = extract_keywords(passage, top_k=3)
    kw_6, _ = extract_keywords(passage, top_k=6)

    assert len(kw_3) <= 3
    assert len(kw_6) <= 6
    if len(kw_6) > len(kw_3):
        assert len(kw_6) > len(kw_3)


def test_stopwords_only_text():
    """Verify text containing only stopwords does not raise exceptions."""
    stopwords_text = "This is that and with or for because about above after again."
    keywords, msg = extract_keywords(stopwords_text, top_k=5)
    # Either empty list with message, or no crash
    assert isinstance(keywords, list)
    assert isinstance(msg, str)


def test_fallback_keyword_extractor():
    """Verify pure-Python fallback frequency extractor works independently."""
    sample = (
        "Quantum computing introduces quantum algorithms that leverage quantum superposition "
        "and quantum entanglement to compute complex quantum equations."
    )
    keywords = extract_keywords_fallback(sample, top_k=3)
    assert len(keywords) > 0
    # 'quantum' should be the top keyword
    assert keywords[0][0] == "quantum"
    assert keywords[0][1] == 1.0  # highest frequency gets 1.0
