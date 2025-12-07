"""Unit tests for MemoryCurator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from htma.core.exceptions import LLMResponseError
from htma.core.types import SalienceResult
from htma.curator.curator import MemoryCurator
from htma.llm.client import OllamaClient


class TestMemoryCuratorInit:
    """Tests for MemoryCurator initialization."""

    def test_default_initialization(self):
        """Test curator initializes with default values."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)
        assert curator.llm == llm
        assert curator.model == "mistral:7b"

    def test_custom_model(self):
        """Test curator initializes with custom model."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm, model="llama3:8b")
        assert curator.model == "llama3:8b"


class TestLoadPromptTemplate:
    """Tests for loading prompt templates."""

    def test_load_existing_template(self):
        """Test loading an existing prompt template."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # The salience.txt template should exist
        template = curator._load_prompt_template("salience.txt")
        assert isinstance(template, str)
        assert len(template) > 0
        assert "{content}" in template
        assert "{context}" in template

    def test_load_nonexistent_template(self):
        """Test loading a non-existent template raises error."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        with pytest.raises(FileNotFoundError) as exc_info:
            curator._load_prompt_template("nonexistent.txt")

        assert "Prompt template not found" in str(exc_info.value)


class TestEvaluateSalience:
    """Tests for salience evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_salience_high_score(self):
        """Test salience evaluation with high-importance content."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # Mock LLM response
        llm_response = json.dumps({
            "score": 0.85,
            "reasoning": "Contains important user preference about work schedule",
            "memory_type": "semantic",
            "key_elements": ["work schedule", "preference", "Monday-Friday"]
        })
        llm.generate = AsyncMock(return_value=llm_response)

        # Evaluate salience
        content = "I prefer to work Monday through Friday, 9am to 5pm."
        result = await curator.evaluate_salience(content=content, context="")

        # Verify result
        assert isinstance(result, SalienceResult)
        assert result.score == 0.85
        assert result.reasoning == "Contains important user preference about work schedule"
        assert result.memory_type == "semantic"
        assert "work schedule" in result.key_elements

        # Verify LLM was called correctly
        llm.generate.assert_called_once()
        call_args = llm.generate.call_args
        assert call_args.kwargs["model"] == "mistral:7b"
        assert call_args.kwargs["temperature"] == 0.3
        assert content in call_args.kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_evaluate_salience_low_score(self):
        """Test salience evaluation with low-importance content."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # Mock LLM response for trivial content
        llm_response = json.dumps({
            "score": 0.1,
            "reasoning": "Trivial greeting, no new information",
            "memory_type": "episodic",
            "key_elements": []
        })
        llm.generate = AsyncMock(return_value=llm_response)

        content = "Hello, how are you?"
        result = await curator.evaluate_salience(content=content)

        assert result.score == 0.1
        assert "Trivial" in result.reasoning
        assert result.memory_type == "episodic"

    @pytest.mark.asyncio
    async def test_evaluate_salience_both_memory_type(self):
        """Test salience evaluation with both semantic and episodic content."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        llm_response = json.dumps({
            "score": 0.75,
            "reasoning": "Significant life event with factual details",
            "memory_type": "both",
            "key_elements": ["graduation", "university", "computer science"]
        })
        llm.generate = AsyncMock(return_value=llm_response)

        content = "I just graduated from university with a degree in computer science!"
        result = await curator.evaluate_salience(content=content)

        assert result.score == 0.75
        assert result.memory_type == "both"
        assert len(result.key_elements) > 0

    @pytest.mark.asyncio
    async def test_evaluate_salience_with_context(self):
        """Test salience evaluation with additional context."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        llm_response = json.dumps({
            "score": 0.6,
            "reasoning": "Relevant given prior context",
            "memory_type": "semantic",
            "key_elements": ["Python", "programming"]
        })
        llm.generate = AsyncMock(return_value=llm_response)

        content = "I prefer Python for scripting tasks."
        context = "User is a software developer interested in automation."
        result = await curator.evaluate_salience(content=content, context=context)

        assert result.score == 0.6

        # Verify context was included in the prompt
        call_args = llm.generate.call_args
        assert context in call_args.kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_evaluate_salience_empty_content(self):
        """Test salience evaluation with empty content."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # LLM should not be called for empty content
        result = await curator.evaluate_salience(content="")

        assert result.score == 0.0
        assert result.reasoning == "Content is empty"
        assert result.memory_type == "episodic"
        assert len(result.key_elements) == 0

        # Verify LLM was NOT called
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_salience_whitespace_only(self):
        """Test salience evaluation with whitespace-only content."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        result = await curator.evaluate_salience(content="   \n\t  ")

        assert result.score == 0.0
        assert "empty" in result.reasoning.lower()
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_salience_very_long_content(self):
        """Test salience evaluation truncates very long content."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # Mock response with high score
        llm_response = json.dumps({
            "score": 0.9,
            "reasoning": "Important information",
            "memory_type": "semantic",
            "key_elements": ["key1", "key2"]
        })
        llm.generate = AsyncMock(return_value=llm_response)

        # Create very long content (over 4000 chars)
        long_content = "Important fact. " * 300

        result = await curator.evaluate_salience(content=long_content)

        # Score should be adjusted down for truncated content
        assert result.score < 0.9  # Adjusted from 0.9
        assert result.score == 0.8  # Should be reduced by 0.1
        assert "truncated" in result.reasoning.lower()

        # Verify the content passed to LLM was truncated
        call_args = llm.generate.call_args
        prompt = call_args.kwargs["prompt"]
        # Should contain truncation marker
        assert "..." in prompt

    @pytest.mark.asyncio
    async def test_evaluate_salience_json_with_extra_text(self):
        """Test parsing JSON when LLM adds extra text around it."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # LLM response with extra text before/after JSON
        llm_response = """Here is my evaluation:
        {
            "score": 0.7,
            "reasoning": "Useful information",
            "memory_type": "both",
            "key_elements": ["element1"]
        }
        This completes the evaluation."""
        llm.generate = AsyncMock(return_value=llm_response)

        result = await curator.evaluate_salience(content="Test content")

        assert result.score == 0.7
        assert result.reasoning == "Useful information"
        assert result.memory_type == "both"

    @pytest.mark.asyncio
    async def test_evaluate_salience_invalid_json(self):
        """Test error handling for invalid JSON response."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # LLM returns invalid JSON
        llm.generate = AsyncMock(return_value="This is not JSON at all")

        with pytest.raises(LLMResponseError) as exc_info:
            await curator.evaluate_salience(content="Test content")

        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg and "json" in error_msg

    @pytest.mark.asyncio
    async def test_evaluate_salience_missing_required_fields(self):
        """Test error handling when JSON is missing required fields."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # Missing 'reasoning' field
        llm_response = json.dumps({
            "score": 0.5,
            "memory_type": "semantic"
        })
        llm.generate = AsyncMock(return_value=llm_response)

        with pytest.raises(LLMResponseError) as exc_info:
            await curator.evaluate_salience(content="Test content")

        assert "Missing required fields" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_salience_llm_connection_error(self):
        """Test error handling when LLM connection fails."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # LLM raises connection error
        from htma.core.exceptions import LLMConnectionError
        llm.generate = AsyncMock(side_effect=LLMConnectionError("Connection failed"))

        with pytest.raises(LLMResponseError) as exc_info:
            await curator.evaluate_salience(content="Test content")

        assert "Failed to generate salience evaluation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_salience_key_elements_optional(self):
        """Test that key_elements is optional in response."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # Response without key_elements
        llm_response = json.dumps({
            "score": 0.5,
            "reasoning": "Moderate importance",
            "memory_type": "episodic"
        })
        llm.generate = AsyncMock(return_value=llm_response)

        result = await curator.evaluate_salience(content="Test content")

        assert result.score == 0.5
        assert result.key_elements == []  # Should default to empty list

    @pytest.mark.asyncio
    async def test_evaluate_salience_score_validation(self):
        """Test that invalid scores are caught by Pydantic validation."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # Score out of range
        llm_response = json.dumps({
            "score": 1.5,  # Invalid: > 1.0
            "reasoning": "Test",
            "memory_type": "semantic"
        })
        llm.generate = AsyncMock(return_value=llm_response)

        with pytest.raises(LLMResponseError):
            await curator.evaluate_salience(content="Test content")
