"""
text_utils.py - Text Processing and Statistical Utilities
==========================================================
Part of: GenAI Summarizer: Advanced NLP & Text Intelligence Assistant

This module provides pure-Python text analysis, sentence splitting,
word/character counting, validation, and export formatting utilities.
No heavy dependencies or model downloads are required for this module.
"""

import re
from typing import List, Tuple, Dict, Any


def clean_text(text: str) -> str:
    """
    Normalizes whitespace and strips leading/trailing spaces.
    
    Args:
        text (str): Input raw text.
        
    Returns:
        str: Cleaned text with normalized spacing.
    """
    if not text:
        return ""
    # Replace multiple spaces/newlines with a single clean space where appropriate,
    # but preserve paragraph separation if double newlines exist.
    paragraphs = text.split("\n")
    cleaned_paragraphs = []
    for p in paragraphs:
        cleaned_p = re.sub(r"[ \t]+", " ", p).strip()
        if cleaned_p:
            cleaned_paragraphs.append(cleaned_p)
    return "\n\n".join(cleaned_paragraphs)


def count_words(text: str) -> int:
    """
    Counts words in a text string.
    Punctuation-only tokens are excluded from the word count.
    
    Args:
        text (str): Input text.
        
    Returns:
        int: Total valid word count.
    """
    if not text or not text.strip():
        return 0
    # Match alphanumeric sequences including hyphens and apostrophes inside words
    words = re.findall(r"\b[\w'-]+\b", text)
    return len(words)


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Counts characters in the text.
    
    Args:
        text (str): Input text.
        include_spaces (bool): Whether to include whitespace characters.
        
    Returns:
        int: Total character count.
    """
    if not text:
        return 0
    if include_spaces:
        return len(text)
    return len(re.sub(r"\s+", "", text))


def split_sentences(text: str) -> List[str]:
    """
    Splits text into a list of sentences, handling common abbreviations
    and punctuation robustly.
    
    Args:
        text (str): Input text.
        
    Returns:
        List[str]: List of individual sentences.
    """
    if not text or not text.strip():
        return []

    cleaned = text.strip()
    # Protect common abbreviations like Mr., Dr., Ms., Prof., Inc., Ltd., e.g., i.e., vs.
    abbreviations = [
        (r"\bMr\.", "Mr__DOT__"),
        (r"\bMrs\.", "Mrs__DOT__"),
        (r"\bMs\.", "Ms__DOT__"),
        (r"\bDr\.", "Dr__DOT__"),
        (r"\bProf\.", "Prof__DOT__"),
        (r"\be\.g\.", "e__DOT__g__DOT__"),
        (r"\bi\.e\.", "i__DOT__e__DOT__"),
        (r"\bvs\.", "vs__DOT__"),
        (r"\bPvt\.", "Pvt__DOT__"),
        (r"\bLtd\.", "Ltd__DOT__"),
        (r"\bInc\.", "Inc__DOT__"),
        (r"\bCorp\.", "Corp__DOT__"),
        (r"\bCo\.", "Co__DOT__"),
        (r"\bU\.S\.", "U__DOT__S__DOT__"),
        (r"\bU\.K\.", "U__DOT__K__DOT__"),
        (r"\bJan\.", "Jan__DOT__"),
        (r"\bFeb\.", "Feb__DOT__"),
        (r"\bMar\.", "Mar__DOT__"),
        (r"\bApr\.", "Apr__DOT__"),
        (r"\bAug\.", "Aug__DOT__"),
        (r"\bSept?\.", "Sep__DOT__"),
        (r"\bOct\.", "Oct__DOT__"),
        (r"\bNov\.", "Nov__DOT__"),
        (r"\bDec\.", "Dec__DOT__"),
    ]

    protected = cleaned
    for pattern, placeholder in abbreviations:
        protected = re.sub(pattern, placeholder, protected, flags=re.IGNORECASE)

    # Protect floating point numbers like 3.14
    protected = re.sub(r"(\d+)\.(\d+)", r"\1<NUMDOT>\2", protected)

    # Split on sentence boundaries: ., !, ?, or newlines followed by capital/space/end
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", protected)

    sentences: List[str] = []
    for s in raw_sentences:
        # Restore protected tokens
        restored = s.replace("__DOT__", ".").replace("<NUMDOT>", ".")
        restored = restored.strip()
        if restored:
            sentences.append(restored)

    return sentences


def count_sentences(text: str) -> int:
    """
    Returns the number of sentences in the text.
    
    Args:
        text (str): Input text.
        
    Returns:
        int: Number of sentences.
    """
    return len(split_sentences(text))


def calculate_reduction(original_count: int, summary_count: int) -> float:
    """
    Calculates the word reduction percentage.
    Formula: ((original_count - summary_count) / original_count) * 100
    
    Safe against zero division.
    Clamps result to [0.0, 100.0].
    
    Args:
        original_count (int): Word count of source text.
        summary_count (int): Word count of generated summary.
        
    Returns:
        float: Reduction percentage rounded to 1 decimal place.
    """
    if original_count <= 0:
        return 0.0
    if summary_count >= original_count:
        return 0.0

    reduction = ((original_count - summary_count) / original_count) * 100.0
    return round(max(0.0, min(100.0, reduction)), 1)


def validate_input_text(
    text: str,
    min_words: int = 15,
    max_chars: int = 8000
) -> Tuple[bool, str]:
    """
    Validates user input text for summarization suitability.
    
    Args:
        text (str): User input text.
        min_words (int): Minimum required word count (default 15).
        max_chars (int): Maximum allowed characters (default 8000).
        
    Returns:
        Tuple[bool, str]: (is_valid, validation_message)
    """
    if not text or not text.strip():
        return False, "Input text is empty. Please enter or paste some text to summarize."

    char_count = len(text)
    if char_count > max_chars:
        return (
            False,
            f"Input text is too long ({char_count:,} characters). "
            f"The maximum allowed length is {max_chars:,} characters."
        )

    words = count_words(text)
    if words < min_words:
        return (
            False,
            f"Input text has only {words} word(s). A meaningful summary requires "
            f"at least {min_words} words. Please provide a longer passage."
        )

    # Check for meaningful alphabetic content (prevent pure symbols or numbers)
    letters = re.findall(r"[a-zA-Z]", text)
    if len(letters) < 10:
        return (
            False,
            "Input text does not contain enough recognizable words or letters. "
            "Please provide coherent natural language text."
        )

    return True, "Valid text."


def count_syllables_word(word: str) -> int:
    """Estimates the syllable count of an individual English word."""
    word = word.lower().strip(".:;?!,'\"-")
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    # Count vowel groups
    vowels = "aeiouy"
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and len(word) > 2 and word[-2] not in vowels:
        count = max(1, count - 1)
    return max(1, count)


def calculate_reading_time_seconds(word_count: int, wpm: int = 200) -> int:
    """Calculates reading time in seconds based on average words per minute."""
    if word_count <= 0:
        return 0
    return max(1, round((word_count / wpm) * 60))


def format_reading_time(seconds: int) -> str:
    """Formats seconds into human-readable minutes and seconds string."""
    if seconds <= 0:
        return "0s"
    if seconds < 60:
        return f"{seconds}s"
    mins = seconds // 60
    rem = seconds % 60
    if rem == 0:
        return f"{mins}m"
    return f"{mins}m {rem}s"


def calculate_flesch_reading_ease(text: str) -> Tuple[float, str]:
    """
    Computes standard Flesch Reading Ease score and qualitative grade.
    Formula: 206.835 - (1.015 * ASL) - (84.6 * ASW)
    where ASL = Average Sentence Length, ASW = Average Syllables per Word.
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0, "N/A"

    total_words = len(words)
    total_sentences = max(len(sentences), 1)
    total_syllables = sum(count_syllables_word(w) for w in words)

    asl = total_words / total_sentences
    asw = total_syllables / total_words

    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    score = round(max(0.0, min(100.0, score)), 1)

    if score >= 80:
        grade = "Easy (Conversational)"
    elif score >= 60:
        grade = "Standard (Plain English)"
    elif score >= 40:
        grade = "Moderate (High School / News)"
    else:
        grade = "Complex (Academic / Technical)"

    return score, grade


def format_as_bullet_points(text: str) -> str:
    """Formats summary sentences into clean bullet points."""
    sentences = split_sentences(text)
    if not sentences:
        return text
    bullets = [f"• {s.strip()}" for s in sentences if s.strip()]
    return "\n\n".join(bullets)


def get_text_statistics(original_text: str, summary_text: str = "") -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of text statistics including
    reading time and readability metrics.
    
    Args:
        original_text (str): The original input text.
        summary_text (str): The summary text (optional).
        
    Returns:
        Dict[str, Any]: Computed metrics dictionary.
    """
    orig_words = count_words(original_text)
    orig_chars = count_characters(original_text, include_spaces=True)
    orig_chars_nospace = count_characters(original_text, include_spaces=False)
    orig_sentences = count_sentences(original_text)

    summary_words = count_words(summary_text) if summary_text else 0
    summary_chars = count_characters(summary_text, include_spaces=True) if summary_text else 0
    summary_sentences = count_sentences(summary_text) if summary_text else 0

    reduction_pct = calculate_reduction(orig_words, summary_words) if summary_text else 0.0

    # Reading time calculation
    orig_read_sec = calculate_reading_time_seconds(orig_words)
    summ_read_sec = calculate_reading_time_seconds(summary_words) if summary_words else 0
    time_saved_sec = max(0, orig_read_sec - summ_read_sec)

    # Readability scoring
    orig_flesch_score, orig_flesch_grade = calculate_flesch_reading_ease(original_text)
    summ_flesch_score, summ_flesch_grade = (
        calculate_flesch_reading_ease(summary_text) if summary_text else (0.0, "N/A")
    )

    return {
        "original_words": orig_words,
        "original_characters": orig_chars,
        "original_chars_nospace": orig_chars_nospace,
        "original_sentences": orig_sentences,
        "summary_words": summary_words,
        "summary_characters": summary_chars,
        "summary_sentences": summary_sentences,
        "reduction_percentage": reduction_pct,
        "orig_reading_time_str": format_reading_time(orig_read_sec),
        "summ_reading_time_str": format_reading_time(summ_read_sec),
        "time_saved_str": format_reading_time(time_saved_sec),
        "orig_flesch_score": orig_flesch_score,
        "orig_flesch_grade": orig_flesch_grade,
        "summ_flesch_score": summ_flesch_score,
        "summ_flesch_grade": summ_flesch_grade,
    }


def format_summary_txt(summary_text: str, model_name: str, is_fallback: bool = False) -> str:
    """
    Formats the summary output for a standalone TXT file download.
    
    Args:
        summary_text (str): The generated summary.
        model_name (str): Model or technique used.
        is_fallback (bool): Whether fallback was utilized.
        
    Returns:
        str: UTF-8 friendly formatted summary text.
    """
    mode = "Extractive NLP Fallback" if is_fallback else "Generative AI (Flan-T5)"
    header = (
        "==========================================================\n"
        "GENAI SUMMARIZER — GENERATED SUMMARY\n"
        "Advanced NLP & Text Intelligence Assistant\n"
        "==========================================================\n"
        f"Engine: {mode}\n"
        f"Model: {model_name}\n"
        "----------------------------------------------------------\n\n"
    )
    footer = (
        "\n\n----------------------------------------------------------\n"
        "Notice: AI-generated summaries may omit nuance or contain errors.\n"
        "Always review the generated text against the original source."
    )
    return header + summary_text.strip() + footer


def format_complete_report_txt(
    original_text: str,
    summary_text: str,
    stats: Dict[str, Any],
    keywords: List[Tuple[str, float]],
    model_name: str,
    is_fallback: bool = False,
) -> str:
    """
    Formats complete analysis (Original, Summary, Statistics, Keywords)
    into a structured text report.
    
    Args:
        original_text (str): Source text.
        summary_text (str): Summary text.
        stats (Dict[str, Any]): Text statistics dictionary.
        keywords (List[Tuple[str, float]]): Extracted keywords and scores.
        model_name (str): Model name or fallback label.
        is_fallback (bool): Whether extractive fallback was used.
        
    Returns:
        str: Comprehensive report text.
    """
    mode = "Extractive NLP (Sentence Scoring)" if is_fallback else "Generative AI (Flan-T5 Seq2Seq)"
    
    kw_lines = []
    if keywords:
        for idx, (kw, score) in enumerate(keywords, start=1):
            kw_lines.append(f"  {idx:2d}. {kw:<25} (TF-IDF Score: {score:.4f})")
    else:
        kw_lines.append("  No significant keywords extracted.")
    kw_block = "\n".join(kw_lines)

    report = f"""================================================================================
GENAI SUMMARIZER: ADVANCED NLP & TEXT INTELLIGENCE ASSISTANT — COMPLETE ANALYSIS REPORT
Generative AI with NLP Project
================================================================================

1. ENGINE & MODEL METADATA
--------------------------------------------------------------------------------
Architecture / Method : {mode}
Model Identifier      : {model_name}
Offline Capable       : Yes (once weights are cached)
Free / Open Source    : Yes (Apache-2.0 / MIT)

2. TEXT METRICS & STATISTICS
--------------------------------------------------------------------------------
Original Word Count      : {stats.get('original_words', 0):,} words
Original Character Count : {stats.get('original_characters', 0):,} chars
Original Sentences       : {stats.get('original_sentences', 0):,} sentences
Summary Word Count       : {stats.get('summary_words', 0):,} words
Summary Character Count  : {stats.get('summary_characters', 0):,} chars
Summary Sentences        : {stats.get('summary_sentences', 0):,} sentences
Word Reduction Rate      : {stats.get('reduction_percentage', 0.0):.1f}% reduction

3. EXTRACTED KEYWORDS (TF-IDF ANALYSIS)
--------------------------------------------------------------------------------
{kw_block}

4. GENERATED SUMMARY
--------------------------------------------------------------------------------
{summary_text.strip()}

5. ORIGINAL SOURCE TEXT
--------------------------------------------------------------------------------
{original_text.strip()}

================================================================================
Educational Training Project | Generative AI with NLP
Note: Always verify critical data against the original source document.
================================================================================
"""
    return report


def extract_text_from_pdf(file_stream) -> Tuple[bool, str, int]:
    """
    Extracts text content from an uploaded PDF file stream.
    
    Args:
        file_stream: File-like object (bytes or upload stream).
        
    Returns:
        Tuple[bool, str, int]: (success, extracted_text, page_count)
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(file_stream)
        total_pages = len(reader.pages)
        extracted = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                extracted.append(t.strip())
        full_text = "\n\n".join(extracted).strip()
        if not full_text:
            return False, "PDF appears to be empty or contains only scanned images.", total_pages
        return True, clean_text(full_text), total_pages
    except Exception as exc:
        return False, f"Failed to extract text from PDF: {str(exc)}", 0


def generate_audio_summary(text: str) -> Tuple[bool, Optional[bytes], str]:
    """
    Generates an MP3 audio bytes stream from text using Google Text-to-Speech (gTTS).
    
    Args:
        text (str): Summary text to synthesize.
        
    Returns:
        Tuple[bool, Optional[bytes], str]: (success, mp3_bytes, message)
    """
    if not text or not text.strip():
        return False, None, "Text is empty."
    try:
        from io import BytesIO
        from gtts import gTTS
        clean_speech_text = re.sub(r"[•\-\*]", "", text).strip()
        tts = gTTS(text=clean_speech_text, lang="en", slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return True, fp.read(), "Audio generated successfully."
    except Exception as exc:
        return False, None, f"Audio synthesis unavailable: {str(exc)}"


def extract_text_from_docx(file_stream) -> Tuple[bool, str]:
    """
    Extracts text content from an uploaded Word document (.docx).
    
    Args:
        file_stream: File-like object (bytes or upload stream).
        
    Returns:
        Tuple[bool, str]: (success, extracted_text)
    """
    # 1. Try python-docx
    try:
        import docx
        doc = docx.Document(file_stream)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs).strip()
        if full_text:
            return True, clean_text(full_text)
    except Exception:
        pass

    # 2. Pure-Python zipfile + XML fallback
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        file_stream.seek(0)
        with zipfile.ZipFile(file_stream) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            paragraphs = []
            for p in tree.iter(f"{{{ns}}}p"):
                texts = [node.text for node in p.iter(f"{{{ns}}}t") if node.text]
                if texts:
                    paragraphs.append("".join(texts).strip())
            full_text = "\n\n".join([p for p in paragraphs if p]).strip()
            if not full_text:
                return False, "DOCX file contains no readable text."
            return True, clean_text(full_text)
    except Exception as exc:
        return False, f"Failed to extract text from Word document: {str(exc)}"


