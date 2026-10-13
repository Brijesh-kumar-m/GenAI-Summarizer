"""
keyword_extractor.py - TF-IDF and Frequency-Based Keyword Extraction
=====================================================================
Part of: GenAI Summarizer: Advanced NLP & Text Intelligence Assistant

Extracts top keywords and relevance scores from user text using scikit-learn's
TF-IDF Vectorizer with a robust pure-Python frequency fallback for edge cases.
"""

import re
from typing import List, Tuple, Dict, Any, Optional

# Basic list of English stopwords for fallback and preprocessing
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "also", "using", "however", "therefore", "thus", "overall", "within"
}


def extract_keywords_fallback(text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Pure Python frequency-based keyword extraction fallback when TF-IDF
    cannot form a vocabulary or scikit-learn is unavailable.
    
    Args:
        text (str): Source text.
        top_k (int): Number of keywords to return.
        
    Returns:
        List[Tuple[str, float]]: List of (keyword, normalized_frequency_score).
    """
    if not text or not text.strip():
        return []

    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [t for t in tokens if t not in STANDARD_STOPWORDS]

    if not filtered:
        return []

    freqs: Dict[str, int] = {}
    for t in filtered:
        freqs[t] = freqs.get(t, 0) + 1

    max_freq = max(freqs.values()) if freqs else 1
    sorted_words = sorted(freqs.items(), key=lambda item: item[1], reverse=True)

    results = []
    for word, count in sorted_words[:top_k]:
        score = round(count / max_freq, 4)
        results.append((word, score))

    return results


def extract_keywords(
    text: str,
    top_k: int = 7,
    min_word_length: int = 3
) -> Tuple[List[Tuple[str, float]], str]:
    """
    Extracts top keywords and their TF-IDF relevance scores.
    
    Divides the text into sentences to form an internal corpus for TF-IDF computation,
    allowing sentence-level Inverse Document Frequency (IDF) discrimination.
    
    Args:
        text (str): User input text.
        top_k (int): Max number of keywords to extract (default 7).
        min_word_length (int): Minimum character length for candidate words.
        
    Returns:
        Tuple[List[Tuple[str, float]], str]:
            - List of (keyword, relevance_score) sorted descending by score.
            - Status message (empty if successful, or explanation if none found).
    """
    if not text or not text.strip():
        return [], "No text provided. Please enter text to extract keywords."

    # Validate non-trivial content (require at least 5 words to extract meaningful keywords)
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) < 5:
        return [], "Input text is too short to extract meaningful keywords."

    # Split into sentences for pseudo-corpus
    from text_utils import split_sentences
    sentences = split_sentences(text)

    # If there's only 1 sentence or very few sentences, chunk text by clauses or use fallback
    if len(sentences) < 2:
        clauses = [c.strip() for c in re.split(r"[,;:\n]+", text) if len(c.strip()) > 10]
        corpus = clauses if len(clauses) >= 2 else [text]
    else:
        corpus = sentences

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Configure TF-IDF with English stop words and unigram/bigram features
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=rf"(?u)\b[a-zA-Z]{{{min_word_length},}}\b",
            ngram_range=(1, 2),  # captures single words and meaningful 2-word phrases
            max_features=100
        )

        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores across sentences
        import numpy as np
        scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()

        if len(feature_names) == 0 or scores.max() == 0:
            fallback = extract_keywords_fallback(text, top_k=top_k)
            if fallback:
                return fallback, "Extracted using frequency analysis (low sentence diversity)."
            return [], "No distinctive keywords could be extracted (text may contain only common stopwords)."

        # Normalize scores to 0.0 - 1.0 range
        max_score = scores.max()
        if max_score > 0:
            normalized_scores = scores / max_score
        else:
            normalized_scores = scores

        # Sort features by score descending
        sorted_indices = normalized_scores.argsort()[::-1]

        keywords: List[Tuple[str, float]] = []
        seen_words = set()

        for idx in sorted_indices:
            term = str(feature_names[idx])
            # Filter out single stopwords that might bypass sklearn
            if term in STANDARD_STOPWORDS:
                continue

            # Skip redundant unigram if already in a selected bigram
            parts = term.split()
            if len(parts) == 1 and term in seen_words:
                continue

            keywords.append((term, round(float(normalized_scores[idx]), 4)))
            for p in parts:
                seen_words.add(p)

            if len(keywords) >= top_k:
                break

        if not keywords:
            fallback = extract_keywords_fallback(text, top_k=top_k)
            if fallback:
                return fallback, "Extracted using frequency fallback."
            return [], "No distinctive keywords found after filtering stopwords."

        return keywords, "Successfully extracted keywords via TF-IDF."

    except Exception as exc:
        # Graceful fallback if sklearn fails or has incompatible input
        fallback = extract_keywords_fallback(text, top_k=top_k)
        if fallback:
            return fallback, f"Extracted via frequency fallback (TF-IDF notice: {str(exc)[:50]})."
        return [], f"Keyword extraction could not complete: {str(exc)[:100]}"
