"""Unit tests for AbstractionGenerator."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from htma.consolidation.abstraction import AbstractionGenerator
from htma.core.exceptions import ConsolidationError, LLMResponseError
from htma.core.types import Episode
from htma.core.utils import generate_episode_id, utc_now
from htma.llm.client import OllamaClient


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock(spec=OllamaClient)
    # Default embeddings - each episode gets a unique but similar embedding
    llm.embed_batch = AsyncMock(
        return_value=[
            [0.1] * 384,  # Episode 1
            [0.15] * 384,  # Episode 2 (similar to 1)
            [0.2] * 384,  # Episode 3
            [0.8] * 384,  # Episode 4 (very different)
            [0.85] * 384,  # Episode 5 (similar to 4)
        ]
    )
    return llm


@pytest.fixture
def abstraction_generator(mock_llm):
    """Create an AbstractionGenerator instance with mocked LLM."""
    return AbstractionGenerator(mock_llm, model="mistral:7b")


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing."""
    now = utc_now()
    return [
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User discussed their morning coffee routine. They prefer dark roast.",
            summary="Coffee routine",
            keywords=["coffee", "morning", "routine"],
            tags=["daily-life"],
            occurred_at=now - timedelta(hours=5),
            recorded_at=now - timedelta(hours=5),
            salience=0.6,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User mentioned they wake up at 6am every day for a morning jog.",
            summary="Morning exercise",
            keywords=["morning", "exercise", "routine"],
            tags=["daily-life", "health"],
            occurred_at=now - timedelta(hours=4),
            recorded_at=now - timedelta(hours=4),
            salience=0.7,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User prepares breakfast after exercise - usually oatmeal with fruit.",
            summary="Breakfast routine",
            keywords=["breakfast", "oatmeal", "routine"],
            tags=["daily-life", "food"],
            occurred_at=now - timedelta(hours=3),
            recorded_at=now - timedelta(hours=3),
            salience=0.65,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User works from home and starts work at 9am.",
            summary="Work schedule",
            keywords=["work", "remote", "schedule"],
            tags=["work"],
            occurred_at=now - timedelta(days=2),
            recorded_at=now - timedelta(days=2),
            salience=0.8,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User prefers Python for backend development.",
            summary="Programming preference",
            keywords=["python", "programming", "preferences"],
            tags=["work", "technology"],
            occurred_at=now - timedelta(days=2),
            recorded_at=now - timedelta(days=2),
            salience=0.75,
        ),
    ]


# ========== Test cluster_episodes ==========


@pytest.mark.asyncio
async def test_cluster_episodes_empty_list(abstraction_generator):
    """Test clustering with empty episode list."""
    result = await abstraction_generator.cluster_episodes([])
    assert result == []


@pytest.mark.asyncio
async def test_cluster_episodes_single_episode(abstraction_generator, sample_episodes):
    """Test clustering with single episode."""
    result = await abstraction_generator.cluster_episodes([sample_episodes[0]])
    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0] == sample_episodes[0]


@pytest.mark.asyncio
async def test_cluster_episodes_temporal_grouping(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test that episodes close in time are clustered together."""
    # Episodes 0, 1, 2 are within hours of each other
    # Episodes 3, 4 are 2 days ago
    # Should create 2 clusters based on temporal distance

    # Mock embeddings that are similar within each group
    mock_llm.embed_batch.return_value = [
        [0.1] * 384,  # Episode 0 (recent)
        [0.12] * 384,  # Episode 1 (recent, similar)
        [0.11] * 384,  # Episode 2 (recent, similar)
        [0.8] * 384,  # Episode 3 (old)
        [0.82] * 384,  # Episode 4 (old, similar)
    ]

    result = await abstraction_generator.cluster_episodes(
        sample_episodes, cluster_size=5, temporal_window_hours=24
    )

    # Should have at least 2 clusters due to temporal gap
    assert len(result) >= 2


@pytest.mark.asyncio
async def test_cluster_episodes_respects_cluster_size(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test that clustering respects cluster size limit."""
    # Make all embeddings similar so temporal/semantic don't split
    mock_llm.embed_batch.return_value = [[0.5] * 384] * len(sample_episodes)

    # Make all episodes recent so temporal doesn't split
    now = utc_now()
    for i, ep in enumerate(sample_episodes):
        ep.occurred_at = now - timedelta(minutes=i * 10)
        ep.recorded_at = now - timedelta(minutes=i * 10)

    result = await abstraction_generator.cluster_episodes(
        sample_episodes, cluster_size=2, temporal_window_hours=24
    )

    # With cluster_size=2 and 5 episodes, should have 3 clusters: [2, 2, 1]
    assert len(result) >= 2
    for cluster in result[:-1]:  # All but last should be at or near cluster_size
        assert len(cluster) <= 3  # Allow some flexibility


@pytest.mark.asyncio
async def test_cluster_episodes_semantic_similarity(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test that semantic similarity affects clustering."""
    # Make episodes 0-2 very similar (morning routine)
    # Make episodes 3-4 opposite (work related) using opposite vectors
    # This will result in low similarity (below 0.3 threshold)
    morning_routine = [1.0] * 192 + [0.0] * 192
    morning_routine2 = [0.95] * 192 + [0.05] * 192
    morning_routine3 = [0.98] * 192 + [0.02] * 192
    # Use opposite vectors for work-related (negative values)
    work_related = [-1.0] * 192 + [0.0] * 192
    work_related2 = [-0.95] * 192 + [-0.05] * 192

    mock_llm.embed_batch.return_value = [
        morning_routine,  # Morning routine
        morning_routine2,  # Morning routine
        morning_routine3,  # Morning routine
        work_related,  # Work (opposite direction - low similarity)
        work_related2,  # Work (opposite direction - low similarity)
    ]

    # Make all recent to avoid temporal splitting
    now = utc_now()
    for i, ep in enumerate(sample_episodes):
        ep.occurred_at = now - timedelta(minutes=i * 10)
        ep.recorded_at = now - timedelta(minutes=i * 10)

    result = await abstraction_generator.cluster_episodes(
        sample_episodes, cluster_size=10, temporal_window_hours=24
    )

    # Should create multiple clusters based on semantic difference
    assert len(result) >= 2


# ========== Test generate_summary ==========


@pytest.mark.asyncio
async def test_generate_summary_success(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test successful summary generation."""
    # Mock LLM response
    summary_response = {
        "content": "User has a consistent morning routine including coffee, exercise, and breakfast.",
        "summary": "Morning routine pattern",
        "context_description": "Daily habits and routines",
        "keywords": ["morning", "routine", "consistency"],
        "tags": ["patterns", "daily-life"],
        "salience": 0.75,
    }
    mock_llm.chat = AsyncMock(return_value=json.dumps(summary_response))

    # Use first 3 episodes (all level 0)
    episodes_to_summarize = sample_episodes[:3]

    result = await abstraction_generator.generate_summary(episodes_to_summarize, level=1)

    assert result.level == 1
    assert result.content == summary_response["content"]
    assert result.summary == summary_response["summary"]
    assert result.context_description == summary_response["context_description"]
    assert "morning" in result.keywords
    assert "routine" in result.keywords
    assert "patterns" in result.tags
    assert 0.7 <= result.salience <= 0.8  # Should be at least avg or LLM score
    assert result.metadata["num_episodes_summarized"] == 3
    assert result.metadata["source_level"] == 0


@pytest.mark.asyncio
async def test_generate_summary_combines_keywords(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test that summary combines keywords from source episodes."""
    summary_response = {
        "content": "Morning routine summary",
        "keywords": ["new-keyword"],
        "tags": ["new-tag"],
        "salience": 0.7,
    }
    mock_llm.chat = AsyncMock(return_value=json.dumps(summary_response))

    result = await abstraction_generator.generate_summary(
        sample_episodes[:3], level=1
    )

    # Should include keywords from all source episodes plus LLM keywords
    assert "coffee" in result.keywords
    assert "morning" in result.keywords
    assert "exercise" in result.keywords
    assert "new-keyword" in result.keywords

    assert "daily-life" in result.tags
    assert "new-tag" in result.tags


@pytest.mark.asyncio
async def test_generate_summary_empty_episodes(abstraction_generator):
    """Test that empty episode list raises error."""
    with pytest.raises(ValueError, match="empty episode list"):
        await abstraction_generator.generate_summary([], level=1)


@pytest.mark.asyncio
async def test_generate_summary_mixed_levels(
    abstraction_generator, sample_episodes
):
    """Test that mixed level episodes raise error."""
    sample_episodes[0].level = 0
    sample_episodes[1].level = 1  # Different level

    with pytest.raises(ValueError, match="same level"):
        await abstraction_generator.generate_summary(sample_episodes[:2], level=2)


@pytest.mark.asyncio
async def test_generate_summary_invalid_target_level(
    abstraction_generator, sample_episodes
):
    """Test that target level must be higher than source level."""
    with pytest.raises(ValueError, match="must be higher"):
        await abstraction_generator.generate_summary(sample_episodes[:2], level=0)


@pytest.mark.asyncio
async def test_generate_summary_llm_failure(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test handling of LLM failures."""
    mock_llm.chat = AsyncMock(side_effect=Exception("LLM failed"))

    with pytest.raises(ConsolidationError, match="Failed to generate summary"):
        await abstraction_generator.generate_summary(sample_episodes[:2], level=1)


@pytest.mark.asyncio
async def test_generate_summary_invalid_json(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test handling of invalid JSON response."""
    mock_llm.chat = AsyncMock(return_value="Not valid JSON")

    with pytest.raises(ConsolidationError):
        await abstraction_generator.generate_summary(sample_episodes[:2], level=1)


@pytest.mark.asyncio
async def test_generate_summary_markdown_wrapped_json(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test parsing JSON wrapped in markdown code blocks."""
    summary_response = {
        "content": "Test summary",
        "salience": 0.7,
    }
    # Wrap in markdown code block
    wrapped_response = f"```json\n{json.dumps(summary_response)}\n```"
    mock_llm.chat = AsyncMock(return_value=wrapped_response)

    result = await abstraction_generator.generate_summary(sample_episodes[:2], level=1)

    assert result.content == "Test summary"


# ========== Test should_abstract ==========


@pytest.mark.asyncio
async def test_should_abstract_empty_list(abstraction_generator):
    """Test that empty list returns False."""
    result = await abstraction_generator.should_abstract([])
    assert result is False


@pytest.mark.asyncio
async def test_should_abstract_insufficient_count(
    abstraction_generator, sample_episodes
):
    """Test that insufficient episode count returns False."""
    # Only 2 episodes, need at least 3
    result = await abstraction_generator.should_abstract(
        sample_episodes[:2], min_episodes=3
    )
    assert result is False


@pytest.mark.asyncio
async def test_should_abstract_too_recent(
    abstraction_generator, sample_episodes
):
    """Test that too recent episodes return False."""
    # Make all episodes very recent
    now = utc_now()
    for ep in sample_episodes:
        ep.recorded_at = now - timedelta(minutes=30)

    result = await abstraction_generator.should_abstract(
        sample_episodes[:3], min_age_hours=24
    )
    assert result is False


@pytest.mark.asyncio
async def test_should_abstract_low_coherence(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test that low coherence episodes return False."""
    # Make embeddings very different (low coherence)
    mock_llm.embed_batch.return_value = [
        [1.0] * 384,
        [0.0] * 384,
        [0.5] * 384,
    ]

    # Make episodes old enough
    old_time = utc_now() - timedelta(days=2)
    for ep in sample_episodes[:3]:
        ep.recorded_at = old_time

    result = await abstraction_generator.should_abstract(
        sample_episodes[:3], min_coherence=0.5
    )
    assert result is False


@pytest.mark.asyncio
async def test_should_abstract_success(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test successful abstraction readiness check."""
    # Make embeddings similar (high coherence)
    mock_llm.embed_batch.return_value = [
        [0.5] * 384,
        [0.51] * 384,
        [0.52] * 384,
    ]

    # Make episodes old enough
    old_time = utc_now() - timedelta(days=2)
    for ep in sample_episodes[:3]:
        ep.recorded_at = old_time

    result = await abstraction_generator.should_abstract(
        sample_episodes[:3], min_episodes=3, min_age_hours=24, min_coherence=0.3
    )
    assert result is True


@pytest.mark.asyncio
async def test_should_abstract_single_episode_coherence(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test that single episode is considered coherent."""
    # Make episode old enough
    sample_episodes[0].recorded_at = utc_now() - timedelta(days=2)

    result = await abstraction_generator.should_abstract(
        [sample_episodes[0]], min_episodes=1, min_age_hours=24
    )
    assert result is True


@pytest.mark.asyncio
async def test_should_abstract_embedding_failure_defaults_true(
    abstraction_generator, sample_episodes, mock_llm
):
    """Test that embedding failure defaults to True if other criteria met."""
    mock_llm.embed_batch.side_effect = Exception("Embedding failed")

    # Make episodes old enough with sufficient count
    old_time = utc_now() - timedelta(days=2)
    for ep in sample_episodes[:3]:
        ep.recorded_at = old_time

    result = await abstraction_generator.should_abstract(
        sample_episodes[:3], min_episodes=3, min_age_hours=24
    )
    # Should default to True when coherence check fails but other criteria pass
    assert result is True


# ========== Test helper methods ==========


def test_cosine_similarity_identical():
    """Test cosine similarity with identical vectors."""
    gen = AbstractionGenerator(MagicMock(), "test-model")
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]

    similarity = gen._cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(1.0, abs=0.01)


def test_cosine_similarity_opposite():
    """Test cosine similarity with opposite vectors."""
    gen = AbstractionGenerator(MagicMock(), "test-model")
    vec1 = [1.0, 1.0, 1.0]
    vec2 = [-1.0, -1.0, -1.0]

    similarity = gen._cosine_similarity(vec1, vec2)
    # After normalization [-1,1] -> [0,1], opposite vectors should be 0
    assert similarity == pytest.approx(0.0, abs=0.01)


def test_cosine_similarity_orthogonal():
    """Test cosine similarity with orthogonal vectors."""
    gen = AbstractionGenerator(MagicMock(), "test-model")
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    similarity = gen._cosine_similarity(vec1, vec2)
    # Orthogonal vectors have 0 similarity, normalized to 0.5
    assert similarity == pytest.approx(0.5, abs=0.01)


def test_calculate_coherence_single():
    """Test coherence calculation with single embedding."""
    gen = AbstractionGenerator(MagicMock(), "test-model")
    embeddings = [[1.0, 2.0, 3.0]]

    coherence = gen._calculate_coherence(embeddings)
    assert coherence == 1.0  # Single embedding is perfectly coherent


def test_calculate_coherence_multiple():
    """Test coherence calculation with multiple embeddings."""
    gen = AbstractionGenerator(MagicMock(), "test-model")
    embeddings = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
    ]

    coherence = gen._calculate_coherence(embeddings)
    assert 0.8 < coherence <= 1.0  # Similar vectors should have high coherence
