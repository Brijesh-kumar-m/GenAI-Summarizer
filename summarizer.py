"""
summarizer.py - Generative AI and Extractive NLP Summarization Engine
=====================================================================
Part of: GenAI Summarizer: Advanced NLP & Text Intelligence Assistant

Provides:
1. Generative AI summarization via Hugging Face Transformers (`google/flan-t5-small`).
2. CPU-compatible inference with caching and lazy loading.
3. Configurable summary lengths (Short, Medium, Long).
4. Documented, high-quality Extractive NLP Fallback (TF-IDF sentence ranking)
   if Transformers/PyTorch cannot be loaded, environment is offline, or memory is constrained.
5. Educational comparison capabilities between Generative AI and Extractive NLP.
"""

import os
import re
import logging
from typing import Dict, Any, Optional, Tuple, List

from text_utils import split_sentences, count_words, validate_input_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_summarizer")

# Default model configuration
DEFAULT_MODEL_NAME = "google/flan-t5-small"

# Summary length configurations
LENGTH_PRESETS = {
    "short": {
        "label": "Short (~25-50 words)",
        "min_length": 20,
        "max_length": 55,
        "length_penalty": 1.5,
        "extractive_sentences": 2,
    },
    "medium": {
        "label": "Medium (~60-110 words)",
        "min_length": 45,
        "max_length": 120,
        "length_penalty": 1.0,
        "extractive_sentences": 4,
    },
    "long": {
        "label": "Long (~120-180 words)",
        "min_length": 90,
        "max_length": 200,
        "length_penalty": 0.8,
        "extractive_sentences": 6,
    },
}


class ModelManager:
    """
    Singleton-style manager to load and cache the Hugging Face Seq2Seq model.
    Avoids reloading model weights on every inference call.
    """
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.model_name = DEFAULT_MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.torch_module = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_name: str = DEFAULT_MODEL_NAME) -> Tuple[bool, str]:
        """
        Loads the Hugging Face tokenizer and model weights on CPU.
        Caches in memory once loaded.
        
        Args:
            model_name (str): Hugging Face model identifier.
            
        Returns:
            Tuple[bool, str]: (is_success, status_or_error_message)
        """
        if self.is_loaded and self.tokenizer is not None and self.model is not None:
            return True, f"Model '{self.model_name}' is already loaded and cached in memory."

        self.model_name = model_name
        self.load_error = None

        try:
            # Check for PyTorch
            import torch
            self.torch_module = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.is_loaded = False
            self.load_error = (
                "PyTorch is not installed in the current Python environment. "
                "Install PyTorch with `pip install torch` to enable generative AI features."
            )
            logger.warning(self.load_error)
            return False, self.load_error

        try:
            # Check for Transformers
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            self.is_loaded = False
            self.load_error = (
                "Hugging Face Transformers is not installed. "
                "Install with `pip install transformers` to enable generative AI features."
            )
            logger.warning(self.load_error)
            return False, self.load_error

        try:
            logger.info(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            logger.info(f"Loading model weights for {model_name} on {self.device}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode

            self.is_loaded = True
            logger.info(f"Successfully loaded and cached {model_name} on {self.device}.")
            return True, f"Successfully loaded and cached {model_name} on {self.device}."

        except Exception as exc:
            self.is_loaded = False
            error_str = str(exc)
            logger.error(f"Failed to load model {model_name}: {error_str}")

            if "Connection" in error_str or "offline" in error_str.lower() or "timeout" in error_str.lower():
                self.load_error = (
                    f"First-time model download failed for '{model_name}'. "
                    "An active internet connection is required to download weights on first run (~300MB). "
                    "Once downloaded, offline inference will work seamlessly. "
                    f"Details: {error_str[:120]}"
                )
            elif "memory" in error_str.lower():
                self.load_error = (
                    f"Insufficient system memory to allocate weights for '{model_name}'. "
                    f"Details: {error_str[:120]}"
                )
            else:
                self.load_error = (
                    f"Could not initialize model '{model_name}'. "
                    f"Details: {error_str[:150]}"
                )

            return False, self.load_error

    def generate(self, text: str, length_mode: str = "medium") -> Dict[str, Any]:
        """
        Executes generative summarization using the cached Seq2Seq model.
        
        Args:
            text (str): Cleaned input text provided by the user.
            length_mode (str): 'short', 'medium', or 'long'.
            
        Returns:
            Dict[str, Any]: Generation result payload.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(self.load_error or "Model is not loaded.")

        user_text_clean = text.strip()
        selected_mode = length_mode.lower() if length_mode else "medium"
        preset = LENGTH_PRESETS.get(selected_mode, LENGTH_PRESETS["medium"])

        # Debugging log: Input length & selected length (no sensitive text logged)
        input_char_len = len(user_text_clean)
        input_word_len = count_words(user_text_clean)
        logger.info(
            "Summarization inference started | input_length=%d chars (%d words) | summary_length=%s",
            input_char_len,
            input_word_len,
            selected_mode,
        )

        # Explicit instruction templates based on length mode
        instruction_templates = {
            "short": "Provide a brief, concise summary of the following text:",
            "medium": "Summarize the following text, focusing on the key ideas and facts:",
            "long": "Provide a comprehensive, detailed summary covering all main points of the following text:",
        }
        instruction = instruction_templates.get(
            selected_mode,
            "Summarize the following text, focusing on the key ideas and facts:"
        )

        # Clearly separate summarization instruction and actual user text with distinct newlines
        prompt = f"{instruction}\n\n{user_text_clean}"

        # Tokenize with safety truncation (Flan-T5 supports up to 512 tokens input)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=False
        ).to(self.device)

        torch = self.torch_module
        with torch.no_grad():
            output_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=preset["max_length"],
                min_length=preset["min_length"],
                length_penalty=preset["length_penalty"],
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()

        # Clean any leftover whitespace artifacts
        summary = re.sub(r"\s+", " ", summary)

        # Debugging log: Output length & mode (no sensitive text logged)
        output_char_len = len(summary)
        output_word_len = count_words(summary)
        logger.info(
            "Summarization inference completed | output_length=%d chars (%d words) | summary_length=%s",
            output_char_len,
            output_word_len,
            selected_mode,
        )

        return {
            "summary": summary,
            "engine": "Generative AI",
            "model_name": self.model_name,
            "length_mode": selected_mode,
            "is_fallback": False,
            "device": self.device,
            "disclaimer": (
                "AI-generated text: Model output is generated probabilistically. "
                "Always verify factual claims and critical statistics against the original source."
            ),
        }


def extractive_summarize(
    text: str,
    length_mode: str = "medium",
    fallback_reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extractive NLP Summarization based on TF-IDF sentence salience scoring
    and position weighting.
    
    Acts as a documented, high-quality fallback when PyTorch or Hugging Face weights
    are unavailable, and as an educational comparison tool for students.
    
    Args:
        text (str): Input text.
        length_mode (str): 'short', 'medium', or 'long'.
        fallback_reason (str, optional): Explanation for why fallback was triggered.
        
    Returns:
        Dict[str, Any]: Extractive summary payload.
    """
    selected_mode = length_mode.lower() if length_mode else "medium"
    preset = LENGTH_PRESETS.get(selected_mode, LENGTH_PRESETS["medium"])
    target_sentences = preset["extractive_sentences"]

    # Debugging log: Input length (no sensitive text logged)
    input_char_len = len(text)
    input_word_len = count_words(text)
    logger.info(
        "Extractive summarization started | input_length=%d chars (%d words) | summary_length=%s",
        input_char_len,
        input_word_len,
        selected_mode,
    )

    sentences = split_sentences(text)

    # If the text has fewer or equal sentences than target, return as-is
    if len(sentences) <= target_sentences:
        summary = " ".join(sentences)
        output_char_len = len(summary)
        output_word_len = count_words(summary)
        logger.info(
            "Extractive summarization completed | output_length=%d chars (%d words) | summary_length=%s",
            output_char_len,
            output_word_len,
            selected_mode,
        )
        return {
            "summary": summary,
            "engine": "Extractive NLP Fallback",
            "model_name": "Sentence Salience (Short Passage)",
            "length_mode": selected_mode,
            "is_fallback": True,
            "fallback_reason": fallback_reason or "Input length is already compact.",
            "disclaimer": "Extractive Summary: Exact sentences selected from source based on statistical importance.",
        }

    # Score sentences using word frequency & TF-IDF style sentence relevance
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Sentence score = mean TF-IDF weight of words in sentence
        sentence_scores = np.asarray(tfidf_matrix.mean(axis=1)).ravel()

    except Exception:
        # Pure-Python frequency scoring if sklearn fails
        from keyword_extractor import STANDARD_STOPWORDS
        word_freqs: Dict[str, int] = {}
        for s in sentences:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", s.lower())
            for w in words:
                if w not in STANDARD_STOPWORDS:
                    word_freqs[w] = word_freqs.get(w, 0) + 1

        sentence_scores = []
        for s in sentences:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", s.lower())
            score = sum(word_freqs.get(w, 0) for w in words if w not in STANDARD_STOPWORDS)
            length_norm = max(len(words), 1)
            sentence_scores.append(score / length_norm)

    # Position weighting: The first sentence and early sentences often introduce core topics
    n_sents = len(sentences)
    final_scores: List[Tuple[int, float]] = []
    for idx, raw_score in enumerate(sentence_scores):
        # Position boost (0.0 to 0.3)
        position_boost = 0.3 * (1.0 - (idx / max(n_sents, 1)))
        total_score = float(raw_score) + position_boost
        final_scores.append((idx, total_score))

    # Rank sentences by total score descending
    sorted_by_score = sorted(final_scores, key=lambda item: item[1], reverse=True)

    # Pick top N sentences
    chosen_indices = sorted([item[0] for item in sorted_by_score[:target_sentences]])

    # Reassemble in original order to preserve chronological narrative flow
    selected_sentences = [sentences[i] for i in chosen_indices]
    summary = " ".join(selected_sentences)

    output_char_len = len(summary)
    output_word_len = count_words(summary)
    logger.info(
        "Extractive summarization completed | output_length=%d chars (%d words) | summary_length=%s",
        output_char_len,
        output_word_len,
        selected_mode,
    )

    return {
        "summary": summary,
        "engine": "Extractive NLP Fallback",
        "model_name": "TF-IDF Sentence Salience & Position Ranker",
        "length_mode": selected_mode,
        "is_fallback": True,
        "fallback_reason": fallback_reason or "Extractive mode selected or triggered as fallback.",
        "disclaimer": "Extractive Summary: Exact sentences extracted from original text based on TF-IDF relevance.",
    }


def summarize_text(
    text: str,
    length_mode: str = "medium",
    force_extractive: bool = False,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict[str, Any]:
    """
    High-level entry point for text summarization.
    
    1. Validates input text.
    2. If force_extractive is True: executes extractive sentence ranking.
    3. If force_extractive is False: tries generative model.
       If generative model fails or is uninitialized, automatically triggers
       the extractive fallback and transparently sets `is_fallback: True`
       with the exact documented reason.
    
    Args:
        text (str): Input text to summarize.
        length_mode (str): 'short', 'medium', or 'long'.
        force_extractive (bool): When True, bypasses neural generative model.
        model_name (str): Hugging Face model identifier.
        
    Returns:
        Dict[str, Any]: Complete summary payload with metadata.
    """
    is_valid, validation_msg = validate_input_text(text)
    if not is_valid:
        return {
            "summary": "",
            "engine": "Validation Failed",
            "model_name": "None",
            "length_mode": length_mode,
            "is_fallback": False,
            "error": validation_msg,
        }

    # If extractive mode explicitly requested
    if force_extractive:
        return extractive_summarize(
            text,
            length_mode=length_mode,
            fallback_reason="Explicitly selected Extractive NLP mode for comparison."
        )

    # Attempt Generative AI via Hugging Face
    manager = ModelManager.get_instance()
    if not manager.is_loaded:
        success, load_msg = manager.load_model(model_name=model_name)
        if not success:
            logger.info("Generative model unavailable. Invoking extractive NLP fallback.")
            return extractive_summarize(
                text,
                length_mode=length_mode,
                fallback_reason=load_msg
            )

    try:
        return manager.generate(text, length_mode=length_mode)
    except Exception as exc:
        logger.warning(f"Generative inference error: {exc}. Invoking extractive fallback.")
        return extractive_summarize(
            text,
            length_mode=length_mode,
            fallback_reason=f"Model inference failed: {str(exc)}"
        )


def get_model_status() -> Dict[str, Any]:
    """
    Returns current health and caching status of the Generative AI model.
    
    Returns:
        Dict[str, Any]: Status dictionary suitable for Streamlit sidebar display.
    """
    manager = ModelManager.get_instance()
    return {
        "model_name": manager.model_name,
        "is_loaded": manager.is_loaded,
        "device": manager.device,
        "error": manager.load_error,
        "presets": {k: v["label"] for k, v in LENGTH_PRESETS.items()},
    }
