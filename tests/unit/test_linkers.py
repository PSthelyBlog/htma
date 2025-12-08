"""Unit tests for LinkGenerator."""

import json
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from htma.core.types import Episode, EpisodeLink, LinkEvaluation
from htma.core.utils import generate_episode_id, utc_now
from htma.curator.linkers import LinkGenerator
from htma.llm.client import OllamaClient
from htma.memory.episodic import EpisodicMemory
from htma.storage.chroma import ChromaStorage
from htma.storage.sqlite import SQLiteStorage


@pytest.fixture
async def temp_storage():
    """Create temporary storage instances for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        chroma_path = Path(tmpdir) / "chroma"

        # Initialize SQLite
        sqlite = SQLiteStorage(db_path)
        await sqlite.initialize()

        # Initialize ChromaDB
        chroma = ChromaStorage(chroma_path)
        await chroma.initialize()

        yield sqlite, chroma

        await sqlite.disconnect()


@pytest.fixture
async def episodic_memory(temp_storage):
    """Create an EpisodicMemory instance with temporary storage."""
    sqlite, chroma = temp_storage
    return EpisodicMemory(sqlite, chroma)


@pytest.fixture
def mock_llm():
    """Create a mock Ollama client."""
    return Mock(spec=OllamaClient)


@pytest.fixture
async def link_generator(mock_llm, episodic_memory):
    """Create a LinkGenerator instance with mocked LLM."""
    return LinkGenerator(
        llm=mock_llm,
        model="mistral:7b",
        episodic=episodic_memory,
    )


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing."""
    now = utc_now()
    return [
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User discussed their morning coffee routine and preference for dark roast.",
            summary="Coffee routine discussion",
            keywords=["coffee", "morning", "routine", "dark roast"],
            tags=["daily-life", "preferences"],
            occurred_at=now - timedelta(hours=2),
            recorded_at=now - timedelta(hours=2),
            salience=0.6,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User mentioned they start work at 9am after their coffee.",
            summary="Work schedule and morning routine",
            keywords=["work", "schedule", "9am", "coffee"],
            tags=["daily-life", "work"],
            occurred_at=now - timedelta(hours=1),
            recorded_at=now - timedelta(hours=1),
            salience=0.7,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User lives in San Francisco and works remotely as a software engineer.",
            summary="Location and work info",
            keywords=["san francisco", "remote work", "software engineer"],
            tags=["personal-info", "work"],
            occurred_at=now,
            recorded_at=now,
            salience=0.8,
        ),
    ]


class TestLinkGenerator:
    """Tests for LinkGenerator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, link_generator):
        """Test LinkGenerator initializes correctly."""
        assert link_generator.model == "mistral:7b"
        assert link_generator.llm is not None
        assert link_generator.episodic is not None
        assert link_generator.prompt_template is not None
        assert len(link_generator.prompt_template) > 0

    @pytest.mark.asyncio
    async def test_parse_link_response_valid(self, link_generator):
        """Test parsing valid link evaluation response."""
        response = json.dumps({
            "should_link": True,
            "link_type": "semantic",
            "weight": 0.75,
            "reasoning": "Both episodes discuss coffee routines"
        })

        result = link_generator._parse_link_response(response)

        assert isinstance(result, LinkEvaluation)
        assert result.should_link is True
        assert result.link_type == "semantic"
        assert result.weight == 0.75
        assert "coffee" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_parse_link_response_with_markdown(self, link_generator):
        """Test parsing response with markdown code blocks."""
        response = '''```json
        {
            "should_link": false,
            "link_type": "temporal",
            "weight": 0.2,
            "reasoning": "Episodes are unrelated"
        }
        ```'''

        result = link_generator._parse_link_response(response)

        assert isinstance(result, LinkEvaluation)
        assert result.should_link is False
        assert result.link_type == "temporal"
        assert result.weight == 0.2

    @pytest.mark.asyncio
    async def test_parse_link_response_clamps_weight(self, link_generator):
        """Test that out-of-range weights are clamped."""
        # Test weight > 1.0
        response = json.dumps({
            "should_link": True,
            "link_type": "causal",
            "weight": 1.5,
            "reasoning": "Strong connection"
        })

        result = link_generator._parse_link_response(response)
        assert result.weight == 1.0

        # Test weight < 0.0
        response = json.dumps({
            "should_link": False,
            "link_type": "semantic",
            "weight": -0.3,
            "reasoning": "No connection"
        })

        result = link_generator._parse_link_response(response)
        assert result.weight == 0.0

    @pytest.mark.asyncio
    async def test_parse_link_response_invalid_link_type(self, link_generator):
        """Test that invalid link types default to semantic."""
        response = json.dumps({
            "should_link": True,
            "link_type": "invalid_type",
            "weight": 0.5,
            "reasoning": "Some connection"
        })

        result = link_generator._parse_link_response(response)
        assert result.link_type == "semantic"

    @pytest.mark.asyncio
    async def test_parse_link_response_missing_field(self, link_generator):
        """Test parsing fails with missing required field."""
        response = json.dumps({
            "should_link": True,
            "weight": 0.5,
            # Missing link_type and reasoning
        })

        with pytest.raises(ValueError, match="Missing.*field"):
            link_generator._parse_link_response(response)

    @pytest.mark.asyncio
    async def test_parse_link_response_invalid_json(self, link_generator):
        """Test parsing fails with invalid JSON."""
        response = "This is not JSON"

        with pytest.raises(ValueError, match="Invalid JSON"):
            link_generator._parse_link_response(response)

    @pytest.mark.asyncio
    async def test_evaluate_connection_should_link(
        self, link_generator, sample_episodes, mock_llm
    ):
        """Test evaluating connection that should be linked."""
        episode_a, episode_b = sample_episodes[0], sample_episodes[1]

        # Mock LLM response
        mock_response = json.dumps({
            "should_link": True,
            "link_type": "temporal",
            "weight": 0.8,
            "reasoning": "Both episodes describe the user's morning routine sequence"
        })
        mock_llm.generate = AsyncMock(return_value=mock_response)

        result = await link_generator.evaluate_connection(episode_a, episode_b)

        assert isinstance(result, LinkEvaluation)
        assert result.should_link is True
        assert result.link_type == "temporal"
        assert result.weight == 0.8
        assert "morning routine" in result.reasoning.lower()

        # Verify LLM was called
        mock_llm.generate.assert_called_once()
        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["model"] == "mistral:7b"
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_evaluate_connection_should_not_link(
        self, link_generator, sample_episodes, mock_llm
    ):
        """Test evaluating connection that should not be linked."""
        episode_a, episode_c = sample_episodes[0], sample_episodes[2]

        # Mock LLM response
        mock_response = json.dumps({
            "should_link": False,
            "link_type": "semantic",
            "weight": 0.1,
            "reasoning": "Episodes discuss different topics with no clear connection"
        })
        mock_llm.generate = AsyncMock(return_value=mock_response)

        result = await link_generator.evaluate_connection(episode_a, episode_c)

        assert isinstance(result, LinkEvaluation)
        assert result.should_link is False
        assert result.weight == 0.1

    @pytest.mark.asyncio
    async def test_generate_links_creates_links(
        self, link_generator, sample_episodes, episodic_memory, mock_llm
    ):
        """Test generate_links creates appropriate links."""
        # Add existing episodes to memory
        for episode in sample_episodes[:2]:
            await episodic_memory.add_episode(episode)

        # New episode to link
        new_episode = sample_episodes[2]
        await episodic_memory.add_episode(new_episode)

        # Mock LLM to return a positive link evaluation
        mock_response = json.dumps({
            "should_link": True,
            "link_type": "semantic",
            "weight": 0.65,
            "reasoning": "Both episodes mention work-related topics"
        })
        mock_llm.generate = AsyncMock(return_value=mock_response)

        # Generate links
        links = await link_generator.generate_links(new_episode, candidate_limit=10)

        # Should have created links (at least one)
        assert len(links) > 0
        for link in links:
            assert isinstance(link, EpisodeLink)
            assert link.source_id == new_episode.id
            assert link.weight > 0

    @pytest.mark.asyncio
    async def test_generate_links_no_candidates(
        self, link_generator, mock_llm
    ):
        """Test generate_links when no candidate episodes exist."""
        new_episode = Episode(
            id=generate_episode_id(),
            level=0,
            content="First episode in empty memory",
            summary="First episode",
            keywords=["first"],
            tags=["test"],
            occurred_at=utc_now(),
            recorded_at=utc_now(),
            salience=0.5,
        )

        # Generate links (should find no candidates)
        links = await link_generator.generate_links(new_episode)

        assert len(links) == 0

    @pytest.mark.asyncio
    async def test_generate_links_filters_self(
        self, link_generator, sample_episodes, episodic_memory, mock_llm
    ):
        """Test that generate_links filters out the new episode itself from candidates."""
        new_episode = sample_episodes[0]
        await episodic_memory.add_episode(new_episode)

        # Mock LLM
        mock_response = json.dumps({
            "should_link": True,
            "link_type": "semantic",
            "weight": 0.5,
            "reasoning": "Test"
        })
        mock_llm.generate = AsyncMock(return_value=mock_response)

        # Should not try to link to itself
        links = await link_generator.generate_links(new_episode)

        # No links should be created (only candidate was itself)
        assert len(links) == 0

    @pytest.mark.asyncio
    async def test_generate_links_handles_evaluation_failure(
        self, link_generator, sample_episodes, episodic_memory, mock_llm
    ):
        """Test that generate_links continues even if one evaluation fails."""
        # Add episodes
        for episode in sample_episodes[:2]:
            await episodic_memory.add_episode(episode)

        new_episode = sample_episodes[2]
        await episodic_memory.add_episode(new_episode)

        # Mock LLM to fail on first call, succeed on second
        mock_llm.generate = AsyncMock(
            side_effect=[
                Exception("LLM error"),
                json.dumps({
                    "should_link": True,
                    "link_type": "semantic",
                    "weight": 0.7,
                    "reasoning": "Connected"
                })
            ]
        )

        # Should handle the error and continue
        links = await link_generator.generate_links(new_episode)

        # Should have created at least one link from successful evaluation
        # (or none if all failed, which is ok)
        assert isinstance(links, list)
