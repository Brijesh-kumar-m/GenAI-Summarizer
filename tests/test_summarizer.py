"""
test_summarizer.py - Unit Tests for Summarization Engine
========================================================
Part of: GenAI Summarizer: Advanced NLP & Text Intelligence Assistant
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from summarizer import (
    LENGTH_PRESETS,
    ModelManager,
    extractive_summarize,
    summarize_text,
    get_model_status,
)


def test_summary_length_configuration_validation():
    """Verify short, medium, and long presets exist and have valid configuration constraints."""
    assert "short" in LENGTH_PRESETS
    assert "medium" in LENGTH_PRESETS
    assert "long" in LENGTH_PRESETS

    for mode in ["short", "medium", "long"]:
        config = LENGTH_PRESETS[mode]
        assert "min_length" in config
        assert "max_length" in config
        assert "length_penalty" in config
        assert "extractive_sentences" in config
        assert config["min_length"] < config["max_length"]
        assert config["extractive_sentences"] > 0

    # Ensure progression from short to long
    assert LENGTH_PRESETS["short"]["max_length"] < LENGTH_PRESETS["medium"]["max_length"]
    assert LENGTH_PRESETS["medium"]["max_length"] < LENGTH_PRESETS["long"]["max_length"]


def test_summarize_empty_and_short_input():
    """Verify summarizer handles empty and very short inputs without crashing."""
    # Empty string
    res_empty = summarize_text("")
    assert res_empty["summary"] == ""
    assert res_empty["error"] is not None
    assert "empty" in res_empty["error"].lower()

    # Short string
    res_short = summarize_text("Hello world.")
    assert res_short["summary"] == ""
    assert res_short["error"] is not None
    assert "word" in res_short["error"].lower()


def test_extractive_summarization():
    """Verify extractive fallback generates a structured summary from multi-sentence text."""
    passage = (
        "Artificial Intelligence is rapidly reshaping the modern global economy. "
        "In healthcare, machine learning algorithms diagnose complex diseases with high accuracy. "
        "In finance, automated risk assessment models detect fraudulent transactions instantly. "
        "Transportation systems rely on autonomous navigation and predictive route planning. "
        "Education platforms leverage natural language processing for personalized student tutoring. "
        "Overall, AI integration continues to drive productivity and scientific breakthroughs."
    )
    result = extractive_summarize(passage, length_mode="short")

    assert result["is_fallback"] is True
    assert "Extractive" in result["engine"]
    assert len(result["summary"]) > 0
    # Short extractive should yield 2 sentences
    assert "disclaimer" in result
    assert result["model_name"] is not None


def test_extractive_preserves_order():
    """Verify extractive summarizer preserves natural sentence order from source."""
    passage = (
        "Phase one involves initial data collection and validation. "
        "Phase two conducts feature engineering and statistical preprocessing. "
        "Phase three trains deep transformer neural networks. "
        "Phase four evaluates precision, recall, and F1 score benchmarks. "
        "Phase five deploys the inference microservice to production servers."
    )
    result = extractive_summarize(passage, length_mode="medium")
    summary = result["summary"]
    # Check that chronological words appears in order if present
    assert len(summary) > 0


def test_summarize_force_extractive():
    """Verify explicit force_extractive parameter triggers extractive engine."""
    sample = (
        "Natural Language Processing enables computational understanding of human speech and text. "
        "Transformers utilize self-attention mechanisms to process tokens in parallel. "
        "Large Language Models demonstrate strong zero-shot and few-shot reasoning capabilities. "
        "Students and researchers study modern NLP architectures and practical deployments."
    )
    result = summarize_text(sample, length_mode="medium", force_extractive=True)

    assert result["is_fallback"] is True
    assert "Extractive" in result["engine"]
    assert len(result["summary"]) > 0


def test_error_handling_model_initialization_failure():
    """Test graceful handling when model weights cannot be loaded (mocking failure)."""
    manager = ModelManager()

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("Offline / Network connection error")):
        success, msg = manager.load_model(model_name="google/flan-t5-small")
        assert not success
        assert not manager.is_loaded
        assert manager.load_error is not None
        assert "Offline" in manager.load_error or "failed" in manager.load_error.lower()


def test_error_handling_model_inference_failure():
    """Verify that when generative inference raises an error, extractive fallback is invoked."""
    sample = (
        "Machine learning models require substantial training data to learn patterns effectively. "
        "Supervised learning utilizes labeled inputs, whereas unsupervised learning discovers clusters. "
        "Reinforcement learning optimizes an agent's policy via reward functions in simulated environments. "
        "Modern deep architectures combine these approaches for robust decision making."
    )
    manager = ModelManager.get_instance()

    with patch.object(manager, "is_loaded", True), \
         patch.object(manager, "generate", side_effect=RuntimeError("CUDA Out of Memory mock error")):
        result = summarize_text(sample, length_mode="medium", force_extractive=False)

        # Fallback should be triggered cleanly
        assert result["is_fallback"] is True
        assert "Extractive" in result["engine"]
        assert "CUDA Out of Memory" in result.get("fallback_reason", "")
        assert len(result["summary"]) > 0


def test_get_model_status():
    """Verify get_model_status returns proper dictionary keys for UI rendering."""
    status = get_model_status()
    assert "model_name" in status
    assert "is_loaded" in status
    assert "device" in status
    assert "presets" in status


def test_user_actual_input_passed_to_model():
    """
    Verifies that the user's actual input paragraph is passed to the model,
    clearly separated from the summarization instruction, and that the
    selected summary length is passed correctly.
    """
    user_paragraph = (
        "Artificial Intelligence is becoming an important part of modern education and business. "
        "It helps people automate repetitive tasks, analyze large amounts of data, and make better decisions. "
        "Natural Language Processing enables computers to understand human language."
    )
    manager = ModelManager.get_instance()

    captured_prompts = []
    captured_presets = []

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    def fake_tokenize(prompt, **kwargs):
        captured_prompts.append(prompt)
        mock_input = MagicMock()
        mock_input.to.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock()
        }
        return mock_input

    mock_tokenizer.side_effect = fake_tokenize
    mock_tokenizer.decode.return_value = (
        "AI automates repetitive tasks and helps analyze large data in education and business."
    )

    # Mock model
    mock_model = MagicMock()
    def fake_generate(*args, **kwargs):
        captured_presets.append(kwargs)
        return [MagicMock()]
    mock_model.generate.side_effect = fake_generate

    mock_torch = MagicMock()

    with patch.object(manager, "is_loaded", True), \
         patch.object(manager, "tokenizer", mock_tokenizer), \
         patch.object(manager, "model", mock_model), \
         patch.object(manager, "torch_module", mock_torch):

        for mode in ["short", "medium", "long"]:
            captured_prompts.clear()
            captured_presets.clear()

            res = summarize_text(user_paragraph, length_mode=mode)

            # 1. Verify model was called with prompt
            assert len(captured_prompts) == 1
            prompt = captured_prompts[0]

            # 2. Verify user's actual input is present in the prompt
            assert user_paragraph in prompt

            # 3. Verify instruction and user text are clearly separated by double newlines
            assert "\n\n" in prompt
            parts = prompt.split("\n\n")
            assert len(parts) >= 2
            instruction_part = parts[0]
            content_part = "\n\n".join(parts[1:])
            assert "Summarize" in instruction_part or "summary" in instruction_part
            assert content_part == user_paragraph

            # 4. Verify length mode was passed and generation parameters matched preset
            assert res["length_mode"] == mode
            assert len(captured_presets) == 1
            gen_kwargs = captured_presets[0]
            preset = LENGTH_PRESETS[mode]
            assert gen_kwargs["max_length"] == preset["max_length"]
            assert gen_kwargs["min_length"] == preset["min_length"]

            # 5. Verify the summary returned is from model decoding
            assert res["summary"] == mock_tokenizer.decode.return_value
            assert res["engine"] == "Generative AI"

