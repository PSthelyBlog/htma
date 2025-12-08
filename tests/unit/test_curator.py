"""Unit tests for MemoryCurator."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from htma.core.exceptions import ConflictResolutionError, LLMResponseError
from htma.core.types import (
    BiTemporalRecord,
    ConflictResolution,
    Fact,
    FactConflict,
    SalienceResult,
    TemporalRange,
)
from htma.core.utils import generate_entity_id, generate_fact_id
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


class TestDetectConflicts:
    """Tests for conflict detection."""

    @pytest.mark.asyncio
    async def test_detect_conflicts_no_conflicts(self):
        """Test conflict detection when there are no conflicts."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        # Create a new fact
        subject_id = generate_entity_id()
        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="New York",
            confidence=0.9,
        )

        # Mock semantic memory with no existing facts
        semantic_memory = MagicMock()
        semantic_memory.query_entity_facts = AsyncMock(return_value=[])

        conflicts = await curator.detect_conflicts(new_fact, semantic_memory)

        assert len(conflicts) == 0
        semantic_memory.query_entity_facts.assert_called_once_with(
            entity_id=subject_id, predicate="lives_in"
        )

    @pytest.mark.asyncio
    async def test_detect_conflicts_same_value_no_conflict(self):
        """Test that same fact value doesn't create conflict."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        subject_id = generate_entity_id()
        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="New York",
        )

        # Existing fact with same value
        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="New York",
        )

        semantic_memory = MagicMock()
        semantic_memory.query_entity_facts = AsyncMock(return_value=[existing_fact])

        conflicts = await curator.detect_conflicts(new_fact, semantic_memory)

        assert len(conflicts) == 0  # Same value, no conflict

    @pytest.mark.asyncio
    async def test_detect_conflicts_different_value_creates_conflict(self):
        """Test that different values create a conflict."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        subject_id = generate_entity_id()
        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="Boston",
        )

        # Existing fact with different value
        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="New York",
        )

        semantic_memory = MagicMock()
        semantic_memory.query_entity_facts = AsyncMock(return_value=[existing_fact])

        conflicts = await curator.detect_conflicts(new_fact, semantic_memory)

        assert len(conflicts) == 1
        assert isinstance(conflicts[0], FactConflict)
        assert conflicts[0].new_fact == new_fact
        assert existing_fact in conflicts[0].conflicting_facts

    @pytest.mark.asyncio
    async def test_detect_conflicts_ignores_invalidated_facts(self):
        """Test that invalidated facts don't create conflicts."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        subject_id = generate_entity_id()
        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="Boston",
        )

        # Existing fact that's been invalidated
        invalidated_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="New York",
            temporal=BiTemporalRecord(
                transaction_time=TemporalRange(
                    valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    valid_to=datetime(2024, 6, 1, tzinfo=timezone.utc),  # Invalidated
                )
            ),
        )

        semantic_memory = MagicMock()
        semantic_memory.query_entity_facts = AsyncMock(return_value=[invalidated_fact])

        conflicts = await curator.detect_conflicts(new_fact, semantic_memory)

        assert len(conflicts) == 0  # Invalidated fact shouldn't conflict

    @pytest.mark.asyncio
    async def test_detect_conflicts_object_id_vs_value(self):
        """Test conflict detection with object_id vs object_value."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        subject_id = generate_entity_id()
        city_id = generate_entity_id()

        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_id=city_id,  # Using entity ID
        )

        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="New York",  # Using string value
        )

        semantic_memory = MagicMock()
        semantic_memory.query_entity_facts = AsyncMock(return_value=[existing_fact])

        conflicts = await curator.detect_conflicts(new_fact, semantic_memory)

        assert len(conflicts) == 1  # Different representation types should conflict


class TestResolveConflict:
    """Tests for conflict resolution."""

    @pytest.mark.asyncio
    async def test_resolve_conflict_temporal_succession(self):
        """Test temporal succession resolution strategy."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        subject_id = generate_entity_id()
        old_fact_id = generate_fact_id()
        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="lives_in",
            object_value="Boston",
            confidence=0.9,
        )

        old_fact = Fact(
            id=old_fact_id,
            subject_id=subject_id,
            predicate="lives_in",
            object_value="New York",
            confidence=0.9,
        )

        # Mock LLM response for temporal succession
        llm_response = json.dumps({
            "strategy": "temporal_succession",
            "reasoning": "User has moved from New York to Boston",
            "invalidate_facts": [old_fact_id],
            "confidence_updates": {},
            "new_fact_accepted": True,
            "new_fact_modifications": {},
        })
        llm.generate = AsyncMock(return_value=llm_response)

        resolution = await curator.resolve_conflict(new_fact, [old_fact])

        assert isinstance(resolution, ConflictResolution)
        assert resolution.strategy == "temporal_succession"
        assert len(resolution.invalidations) == 1
        assert resolution.invalidations[0][0] == old_fact_id
        assert resolution.new_fact == new_fact
        assert "moved" in resolution.reasoning.lower()

    @pytest.mark.asyncio
    async def test_resolve_conflict_confidence_adjustment(self):
        """Test confidence adjustment resolution strategy."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        subject_id = generate_entity_id()
        existing_fact_id = generate_fact_id()
        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="favorite_color",
            object_value="blue",
            confidence=0.6,
        )

        existing_fact = Fact(
            id=existing_fact_id,
            subject_id=subject_id,
            predicate="favorite_color",
            object_value="red",
            confidence=0.7,
        )

        # Mock LLM response for confidence adjustment
        llm_response = json.dumps({
            "strategy": "confidence_adjustment",
            "reasoning": "Uncertain which color preference is current",
            "invalidate_facts": [],
            "confidence_updates": {existing_fact_id: 0.5},
            "new_fact_accepted": True,
            "new_fact_modifications": {"confidence": 0.5},
        })
        llm.generate = AsyncMock(return_value=llm_response)

        resolution = await curator.resolve_conflict(new_fact, [existing_fact])

        assert resolution.strategy == "confidence_adjustment"
        assert len(resolution.confidence_updates) == 1
        assert resolution.confidence_updates[0] == (existing_fact_id, 0.5)
        assert resolution.new_fact.confidence == 0.5  # Modified
        assert len(resolution.invalidations) == 0  # No invalidations

    @pytest.mark.asyncio
    async def test_resolve_conflict_coexistence(self):
        """Test coexistence resolution strategy."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        subject_id = generate_entity_id()
        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="name",
            object_value="Bob",
        )

        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="name",
            object_value="Robert",
        )

        # Mock LLM response for coexistence
        llm_response = json.dumps({
            "strategy": "coexistence",
            "reasoning": "Bob is likely a nickname for Robert, both can coexist",
            "invalidate_facts": [],
            "confidence_updates": {},
            "new_fact_accepted": True,
            "new_fact_modifications": {
                "metadata": {"context": "nickname"}
            },
        })
        llm.generate = AsyncMock(return_value=llm_response)

        resolution = await curator.resolve_conflict(new_fact, [existing_fact])

        assert resolution.strategy == "coexistence"
        assert len(resolution.invalidations) == 0
        assert len(resolution.confidence_updates) == 0
        assert resolution.new_fact is not None
        assert "nickname" in resolution.new_fact.metadata.get("context", "")

    @pytest.mark.asyncio
    async def test_resolve_conflict_rejection(self):
        """Test rejection resolution strategy."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        subject_id = generate_entity_id()
        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="birth_year",
            object_value="2025",  # Implausible
            confidence=0.5,
        )

        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=subject_id,
            predicate="birth_year",
            object_value="1990",
            confidence=0.95,
        )

        # Mock LLM response for rejection
        llm_response = json.dumps({
            "strategy": "rejection",
            "reasoning": "New birth year is implausible and conflicts with high-confidence fact",
            "invalidate_facts": [],
            "confidence_updates": {},
            "new_fact_accepted": False,
        })
        llm.generate = AsyncMock(return_value=llm_response)

        resolution = await curator.resolve_conflict(new_fact, [existing_fact])

        assert resolution.strategy == "rejection"
        assert resolution.new_fact is None  # Rejected
        assert len(resolution.invalidations) == 0
        assert len(resolution.confidence_updates) == 0

    @pytest.mark.asyncio
    async def test_resolve_conflict_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="value",
        )

        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=new_fact.subject_id,
            predicate="test",
            object_value="other",
        )

        # Invalid strategy in response
        llm_response = json.dumps({
            "strategy": "invalid_strategy",
            "reasoning": "Test",
            "new_fact_accepted": True,
        })
        llm.generate = AsyncMock(return_value=llm_response)

        with pytest.raises(LLMResponseError) as exc_info:
            await curator.resolve_conflict(new_fact, [existing_fact])

        assert "Invalid strategy" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_conflict_missing_required_fields(self):
        """Test error handling for missing required fields in response."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="value",
        )

        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=new_fact.subject_id,
            predicate="test",
            object_value="other",
        )

        # Missing required field 'reasoning'
        llm_response = json.dumps({
            "strategy": "rejection",
            "new_fact_accepted": False,
        })
        llm.generate = AsyncMock(return_value=llm_response)

        with pytest.raises(LLMResponseError) as exc_info:
            await curator.resolve_conflict(new_fact, [existing_fact])

        assert "Missing required fields" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_conflict_invalid_json(self):
        """Test error handling for invalid JSON response."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="value",
        )

        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=new_fact.subject_id,
            predicate="test",
            object_value="other",
        )

        llm.generate = AsyncMock(return_value="Not valid JSON")

        with pytest.raises(LLMResponseError) as exc_info:
            await curator.resolve_conflict(new_fact, [existing_fact])

        assert "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_resolve_conflict_llm_error(self):
        """Test error handling when LLM generation fails."""
        llm = MagicMock(spec=OllamaClient)
        curator = MemoryCurator(llm=llm)

        new_fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="value",
        )

        existing_fact = Fact(
            id=generate_fact_id(),
            subject_id=new_fact.subject_id,
            predicate="test",
            object_value="other",
        )

        llm.generate = AsyncMock(side_effect=Exception("LLM connection failed"))

        with pytest.raises(LLMResponseError) as exc_info:
            await curator.resolve_conflict(new_fact, [existing_fact])

        assert "Failed to generate conflict resolution" in str(exc_info.value)
