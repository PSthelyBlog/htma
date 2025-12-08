"""Unit tests for Entity and Fact Extractors."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from htma.core.exceptions import LLMResponseError
from htma.core.types import ExtractedEntity, ExtractedFact
from htma.curator.extractors import EntityExtractor, FactExtractor
from htma.llm.client import OllamaClient


class TestEntityExtractorInit:
    """Tests for EntityExtractor initialization."""

    def test_default_initialization(self):
        """Test extractor initializes with default values."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)
        assert extractor.llm == llm
        assert extractor.model == "mistral:7b"

    def test_custom_model(self):
        """Test extractor initializes with custom model."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm, model="llama3:8b")
        assert extractor.model == "llama3:8b"


class TestEntityExtractorLoadTemplate:
    """Tests for loading entity extraction template."""

    def test_load_existing_template(self):
        """Test loading the entity extraction template."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        template = extractor._load_prompt_template("entity_extraction.txt")
        assert isinstance(template, str)
        assert len(template) > 0
        assert "{text}" in template
        assert "{context}" in template

    def test_load_nonexistent_template(self):
        """Test loading a non-existent template raises error."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        with pytest.raises(FileNotFoundError) as exc_info:
            extractor._load_prompt_template("nonexistent.txt")

        assert "Prompt template not found" in str(exc_info.value)


class TestEntityExtraction:
    """Tests for entity extraction."""

    @pytest.mark.asyncio
    async def test_extract_multiple_entities(self):
        """Test extracting multiple entities from text."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        # Mock LLM response with multiple entities
        llm_response = json.dumps([
            {
                "name": "Alice",
                "type": "person",
                "mentions": ["Alice works"],
                "confidence": 0.95
            },
            {
                "name": "Google",
                "type": "organization",
                "mentions": ["works at Google"],
                "confidence": 0.9
            },
            {
                "name": "Mountain View",
                "type": "place",
                "mentions": ["in Mountain View"],
                "confidence": 0.85
            }
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Alice works at Google in Mountain View."
        entities = await extractor.extract(text=text)

        assert len(entities) == 3
        assert all(isinstance(e, ExtractedEntity) for e in entities)

        # Check first entity
        assert entities[0].name == "Alice"
        assert entities[0].entity_type == "person"
        assert entities[0].confidence == 0.95

        # Check second entity
        assert entities[1].name == "Google"
        assert entities[1].entity_type == "organization"

        # Check third entity
        assert entities[2].name == "Mountain View"
        assert entities[2].entity_type == "place"

    @pytest.mark.asyncio
    async def test_extract_single_entity(self):
        """Test extracting a single entity."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        # Mock LLM response with single entity
        llm_response = json.dumps([
            {
                "name": "Python",
                "type": "concept",
                "mentions": ["Python programming language"],
                "confidence": 1.0
            }
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "I love Python programming language."
        entities = await extractor.extract(text=text)

        assert len(entities) == 1
        assert entities[0].name == "Python"
        assert entities[0].entity_type == "concept"
        assert entities[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_extract_no_entities(self):
        """Test extraction when no entities are found."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        # Mock LLM response with empty array
        llm_response = json.dumps([])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "This is just some text."
        entities = await extractor.extract(text=text)

        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_empty_text(self):
        """Test extraction with empty text."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        entities = await extractor.extract(text="")

        assert len(entities) == 0
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_with_context(self):
        """Test extraction with additional context."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        llm_response = json.dumps([
            {
                "name": "Bob",
                "type": "person",
                "mentions": ["Bob"],
                "confidence": 0.9
            }
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Bob is here."
        context = "Talking about Bob, the software engineer."
        entities = await extractor.extract(text=text, context=context)

        assert len(entities) == 1

        # Verify context was included in the prompt
        call_args = llm.generate.call_args
        assert context in call_args.kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_extract_entity_types(self):
        """Test extraction of different entity types."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        llm_response = json.dumps([
            {"name": "Einstein", "type": "person", "confidence": 1.0},
            {"name": "Berlin", "type": "place", "confidence": 1.0},
            {"name": "IBM", "type": "organization", "confidence": 1.0},
            {"name": "laptop", "type": "object", "confidence": 1.0},
            {"name": "relativity", "type": "concept", "confidence": 1.0},
            {"name": "meeting", "type": "event", "confidence": 1.0},
            {"name": "2023", "type": "time", "confidence": 1.0}
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Test text"
        entities = await extractor.extract(text=text)

        assert len(entities) == 7
        types = [e.entity_type for e in entities]
        assert "person" in types
        assert "place" in types
        assert "organization" in types
        assert "object" in types
        assert "concept" in types
        assert "event" in types
        assert "time" in types

    @pytest.mark.asyncio
    async def test_extract_with_mentions(self):
        """Test extraction preserves mention locations."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        llm_response = json.dumps([
            {
                "name": "Alice",
                "type": "person",
                "mentions": ["Alice said", "She"],
                "confidence": 0.95
            }
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Alice said hello. She is nice."
        entities = await extractor.extract(text=text)

        assert len(entities) == 1
        assert len(entities[0].mentions) == 2
        assert "Alice said" in entities[0].mentions
        assert "She" in entities[0].mentions

    @pytest.mark.asyncio
    async def test_extract_json_wrapped_in_object(self):
        """Test parsing when LLM wraps entities in an object."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        # LLM response with entities wrapped in "entities" key
        llm_response = json.dumps({
            "entities": [
                {"name": "Alice", "type": "person", "confidence": 1.0}
            ]
        })
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Alice is here."
        entities = await extractor.extract(text=text)

        assert len(entities) == 1
        assert entities[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_extract_json_with_extra_text(self):
        """Test parsing JSON when LLM adds extra text."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        llm_response = """Here are the entities:
        [
            {"name": "Alice", "type": "person", "confidence": 1.0}
        ]
        End of extraction."""
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Alice is here."
        entities = await extractor.extract(text=text)

        assert len(entities) == 1
        assert entities[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_extract_invalid_json(self):
        """Test error handling for invalid JSON response."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        llm.generate = AsyncMock(return_value="This is not JSON")

        with pytest.raises(LLMResponseError) as exc_info:
            await extractor.extract(text="Test")

        assert "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_extract_skips_invalid_entities(self):
        """Test that invalid entity data is skipped."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        # Mix of valid and invalid entities
        llm_response = json.dumps([
            {"name": "Alice", "type": "person", "confidence": 1.0},
            {"name": "Bob"},  # Missing 'type' field
            {"name": "Charlie", "type": "person", "confidence": 1.0}
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Test text"
        entities = await extractor.extract(text=text)

        # Should only get valid entities
        assert len(entities) == 2
        assert entities[0].name == "Alice"
        assert entities[1].name == "Charlie"

    @pytest.mark.asyncio
    async def test_extract_llm_failure(self):
        """Test error handling when LLM fails."""
        llm = MagicMock(spec=OllamaClient)
        extractor = EntityExtractor(llm=llm)

        from htma.core.exceptions import LLMConnectionError
        llm.generate = AsyncMock(side_effect=LLMConnectionError("Connection failed"))

        with pytest.raises(LLMResponseError) as exc_info:
            await extractor.extract(text="Test")

        assert "Failed to generate entity extraction" in str(exc_info.value)


class TestFactExtractorInit:
    """Tests for FactExtractor initialization."""

    def test_default_initialization(self):
        """Test extractor initializes with default values."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)
        assert extractor.llm == llm
        assert extractor.model == "mistral:7b"

    def test_custom_model(self):
        """Test extractor initializes with custom model."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm, model="llama3:8b")
        assert extractor.model == "llama3:8b"


class TestFactExtractorLoadTemplate:
    """Tests for loading fact extraction template."""

    def test_load_existing_template(self):
        """Test loading the fact extraction template."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        template = extractor._load_prompt_template("fact_extraction.txt")
        assert isinstance(template, str)
        assert len(template) > 0
        assert "{text}" in template
        assert "{entities}" in template


class TestFactExtraction:
    """Tests for fact extraction."""

    @pytest.mark.asyncio
    async def test_extract_multiple_facts(self):
        """Test extracting multiple facts from text."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [
            ExtractedEntity(name="Alice", entity_type="person"),
            ExtractedEntity(name="Google", entity_type="organization")
        ]

        llm_response = json.dumps([
            {
                "subject": "Alice",
                "predicate": "works_at",
                "object_entity": "Google",
                "object_value": None,
                "confidence": 0.95,
                "source_text": "Alice works at Google"
            },
            {
                "subject": "Alice",
                "predicate": "is_a",
                "object_entity": None,
                "object_value": "engineer",
                "confidence": 0.9,
                "source_text": "Alice is an engineer"
            }
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Alice works at Google. Alice is an engineer."
        facts = await extractor.extract(text=text, entities=entities)

        assert len(facts) == 2
        assert all(isinstance(f, ExtractedFact) for f in facts)

        # Check first fact (entity-to-entity)
        assert facts[0].subject == "Alice"
        assert facts[0].predicate == "works_at"
        assert facts[0].object_entity == "Google"
        assert facts[0].object_value is None

        # Check second fact (entity-to-value)
        assert facts[1].subject == "Alice"
        assert facts[1].predicate == "is_a"
        assert facts[1].object_entity is None
        assert facts[1].object_value == "engineer"

    @pytest.mark.asyncio
    async def test_extract_with_temporal_markers(self):
        """Test extraction with temporal information."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [
            ExtractedEntity(name="Alice", entity_type="person"),
            ExtractedEntity(name="Google", entity_type="organization")
        ]

        llm_response = json.dumps([
            {
                "subject": "Alice",
                "predicate": "works_at",
                "object_entity": "Google",
                "temporal_marker": "since 2020",
                "confidence": 0.95,
                "source_text": "Alice has worked at Google since 2020"
            }
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Alice has worked at Google since 2020."
        facts = await extractor.extract(text=text, entities=entities)

        assert len(facts) == 1
        assert facts[0].temporal_marker == "since 2020"

    @pytest.mark.asyncio
    async def test_extract_empty_text(self):
        """Test extraction with empty text."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [ExtractedEntity(name="Alice", entity_type="person")]
        facts = await extractor.extract(text="", entities=entities)

        assert len(facts) == 0
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_no_entities(self):
        """Test extraction with no entities."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        facts = await extractor.extract(text="Some text", entities=[])

        assert len(facts) == 0
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_no_facts(self):
        """Test extraction when no facts are found."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [ExtractedEntity(name="Alice", entity_type="person")]
        llm_response = json.dumps([])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Alice."
        facts = await extractor.extract(text=text, entities=entities)

        assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_extract_with_context(self):
        """Test extraction with additional context."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [ExtractedEntity(name="Bob", entity_type="person")]
        llm_response = json.dumps([
            {
                "subject": "Bob",
                "predicate": "prefers",
                "object_value": "Python",
                "confidence": 0.9
            }
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Bob prefers Python."
        context = "Discussion about programming languages."
        facts = await extractor.extract(text=text, entities=entities, context=context)

        assert len(facts) == 1

        # Verify context was included
        call_args = llm.generate.call_args
        assert context in call_args.kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_extract_common_predicates(self):
        """Test extraction uses common predicates."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [
            ExtractedEntity(name="Alice", entity_type="person"),
            ExtractedEntity(name="Bob", entity_type="person"),
            ExtractedEntity(name="Google", entity_type="organization")
        ]

        llm_response = json.dumps([
            {"subject": "Alice", "predicate": "is_a", "object_value": "engineer", "confidence": 1.0},
            {"subject": "Alice", "predicate": "works_at", "object_entity": "Google", "confidence": 1.0},
            {"subject": "Alice", "predicate": "knows", "object_entity": "Bob", "confidence": 1.0},
            {"subject": "Alice", "predicate": "prefers", "object_value": "Python", "confidence": 1.0}
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Test text"
        facts = await extractor.extract(text=text, entities=entities)

        predicates = [f.predicate for f in facts]
        assert "is_a" in predicates
        assert "works_at" in predicates
        assert "knows" in predicates
        assert "prefers" in predicates

    @pytest.mark.asyncio
    async def test_extract_json_wrapped_in_object(self):
        """Test parsing when LLM wraps facts in an object."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [ExtractedEntity(name="Alice", entity_type="person")]
        llm_response = json.dumps({
            "facts": [
                {
                    "subject": "Alice",
                    "predicate": "is_a",
                    "object_value": "engineer",
                    "confidence": 1.0
                }
            ]
        })
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Alice is an engineer."
        facts = await extractor.extract(text=text, entities=entities)

        assert len(facts) == 1
        assert facts[0].subject == "Alice"

    @pytest.mark.asyncio
    async def test_extract_skips_invalid_facts(self):
        """Test that invalid fact data is skipped."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [ExtractedEntity(name="Alice", entity_type="person")]
        llm_response = json.dumps([
            {"subject": "Alice", "predicate": "is_a", "object_value": "engineer", "confidence": 1.0},
            {"subject": "Alice"},  # Missing 'predicate' field
            {"subject": "Alice", "predicate": "works_at", "object_entity": "Google", "confidence": 1.0}
        ])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Test text"
        facts = await extractor.extract(text=text, entities=entities)

        assert len(facts) == 2
        assert facts[0].predicate == "is_a"
        assert facts[1].predicate == "works_at"

    @pytest.mark.asyncio
    async def test_extract_entities_list_in_prompt(self):
        """Test that entities are formatted correctly in the prompt."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [
            ExtractedEntity(name="Alice", entity_type="person"),
            ExtractedEntity(name="Google", entity_type="organization")
        ]
        llm_response = json.dumps([])
        llm.generate = AsyncMock(return_value=llm_response)

        text = "Test text"
        await extractor.extract(text=text, entities=entities)

        # Check that entities were formatted in the prompt
        call_args = llm.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "Alice (person)" in prompt
        assert "Google (organization)" in prompt

    @pytest.mark.asyncio
    async def test_extract_invalid_json(self):
        """Test error handling for invalid JSON response."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [ExtractedEntity(name="Alice", entity_type="person")]
        llm.generate = AsyncMock(return_value="This is not JSON")

        with pytest.raises(LLMResponseError) as exc_info:
            await extractor.extract(text="Test", entities=entities)

        assert "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_extract_llm_failure(self):
        """Test error handling when LLM fails."""
        llm = MagicMock(spec=OllamaClient)
        extractor = FactExtractor(llm=llm)

        entities = [ExtractedEntity(name="Alice", entity_type="person")]
        from htma.core.exceptions import LLMConnectionError
        llm.generate = AsyncMock(side_effect=LLMConnectionError("Connection failed"))

        with pytest.raises(LLMResponseError) as exc_info:
            await extractor.extract(text="Test", entities=entities)

        assert "Failed to generate fact extraction" in str(exc_info.value)
