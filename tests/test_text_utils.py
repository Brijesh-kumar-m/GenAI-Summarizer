"""
test_text_utils.py - Unit Tests for Text Utility Functions
==========================================================
Part of: GenAI Summarizer: Advanced NLP & Text Intelligence Assistant
"""

import sys
import os

# Add parent directory to path so tests can run from tests/ or project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from text_utils import (
    count_words,
    count_characters,
    split_sentences,
    count_sentences,
    calculate_reduction,
    validate_input_text,
    get_text_statistics,
    format_summary_txt,
    format_complete_report_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
    generate_audio_summary,
)


def test_empty_input_handling():
    """Verify that empty, None, and whitespace-only inputs are handled gracefully."""
    assert count_words("") == 0
    assert count_words("   \n\t  ") == 0
    assert count_characters("") == 0
    assert count_characters("   ", include_spaces=False) == 0
    assert count_sentences("") == 0
    assert split_sentences("") == []

    is_valid, msg = validate_input_text("")
    assert not is_valid
    assert "empty" in msg.lower()

    is_valid, msg = validate_input_text("    ")
    assert not is_valid


def test_word_count_calculation():
    """Verify accurate word counting with standard punctuation, hyphens, and whitespace."""
    text1 = "Natural Language Processing and Generative AI are transforming education."
    assert count_words(text1) == 9

    text2 = "State-of-the-art models like GPT-4, PaLM, and Flan-T5 offer high performance."
    # Hyphenated words are counted cleanly
    words = count_words(text2)
    assert words >= 9

    text3 = "Word... with, unusual! punctuation??? and 'quotes'."
    assert count_words(text3) == 6


def test_sentence_splitting_and_counting():
    """Verify sentence segmentation handles abbreviations (Dr., e.g., Inc.) and numbers."""
    text = (
        "Dr. Smith visited Global Innovations Pvt. Ltd. yesterday. "
        "The AI workshop was attended by 50 students, e.g., from computer science! "
        "Did everyone enjoy the practical hands-on session? Yes, they did."
    )
    sentences = split_sentences(text)
    # Should not split on Dr., Pvt., Ltd., or e.g.
    assert len(sentences) == 4
    assert count_sentences(text) == 4
    assert "Dr. Smith" in sentences[0]
    assert "e.g." in sentences[1]


def test_input_length_validation():
    """Test min_words and max_chars validation boundaries."""
    short_text = "This is too short."
    is_valid, msg = validate_input_text(short_text, min_words=15)
    assert not is_valid
    assert "word" in msg.lower()

    valid_text = (
        "Artificial Intelligence enables machines to emulate human cognitive capabilities "
        "such as learning, reasoning, and automated text synthesis across various domain areas."
    )
    is_valid, msg = validate_input_text(valid_text, min_words=10, max_chars=1000)
    assert is_valid

    long_text = "Artificial intelligence " * 1000
    is_valid, msg = validate_input_text(long_text, max_chars=500)
    assert not is_valid
    assert "too long" in msg.lower()


def test_reduction_percentage_calculation():
    """Test word reduction formula and division-by-zero prevention."""
    # 50% reduction
    assert calculate_reduction(100, 50) == 50.0

    # 75% reduction
    assert calculate_reduction(200, 50) == 75.0

    # Zero original words (safety check)
    assert calculate_reduction(0, 10) == 0.0
    assert calculate_reduction(-5, 10) == 0.0

    # Summary longer than original (e.g. edge case)
    assert calculate_reduction(50, 60) == 0.0

    # Identical lengths
    assert calculate_reduction(100, 100) == 0.0


def test_text_statistics_dictionary():
    """Test computation of comprehensive statistics dictionary."""
    orig = "Natural language processing is an essential branch of modern artificial intelligence."
    summ = "NLP is a branch of modern AI."
    stats = get_text_statistics(orig, summ)

    assert stats["original_words"] > stats["summary_words"]
    assert stats["original_characters"] > stats["summary_characters"]
    assert stats["original_sentences"] == 1
    assert stats["summary_sentences"] == 1
    assert stats["reduction_percentage"] > 0.0


def test_output_formatting():
    """Test generation of standalone and comprehensive report text exports."""
    summary = "Flan-T5 is a compact, open-source Seq2Seq model."
    txt = format_summary_txt(summary, model_name="google/flan-t5-small", is_fallback=False)
    assert "Flan-T5" in txt
    assert "GENAI SUMMARIZER" in txt
    assert "Generative AI" in txt

    report = format_complete_report_txt(
        original_text="Original text here.",
        summary_text=summary,
        stats={"original_words": 100, "summary_words": 20, "reduction_percentage": 80.0},
        keywords=[("nlp", 0.95), ("transformer", 0.88)],
        model_name="google/flan-t5-small",
        is_fallback=False
    )
    assert "COMPLETE ANALYSIS REPORT" in report
    assert "TF-IDF Score" in report
    assert "Original text here." in report


def test_extract_text_from_pdf_invalid_stream():
    """Test PDF extraction handles corrupted or non-PDF streams gracefully."""
    import io
    fake_stream = io.BytesIO(b"not a real pdf file content")
    success, text, pages = extract_text_from_pdf(fake_stream)
    assert success is False
    assert "Failed" in text or "empty" in text


def test_generate_audio_summary_empty_text():
    """Test audio synthesis handles empty text safely."""
    success, audio_bytes, msg = generate_audio_summary("")
    assert success is False
    assert audio_bytes is None


def test_extract_text_from_docx_invalid_stream():
    """Test DOCX extraction handles invalid stream gracefully."""
    import io
    fake_stream = io.BytesIO(b"corrupted word doc content")
    success, text = extract_text_from_docx(fake_stream)
    assert success is False
    assert "Failed" in text or "no readable text" in text

