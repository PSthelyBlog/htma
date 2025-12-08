"""Unit tests for PatternDetector."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from htma.consolidation.patterns import PatternDetector
from htma.core.exceptions import ConsolidationError
from htma.core.types import Episode, Pattern, PatternDetectionResult
from htma.core.utils import generate_episode_id, utc_now
from htma.llm.client import OllamaClient


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock(spec=OllamaClient)
    # Default embeddings
    llm.embed_batch = AsyncMock(
        return_value=[
            [0.5] * 384,  # Similar embeddings by default
            [0.51] * 384,
            [0.52] * 384,
        ]
    )
    return llm


@pytest.fixture
def pattern_detector(mock_llm):
    """Create a PatternDetector instance with mocked LLM."""
    return PatternDetector(mock_llm, model="mistral:7b")


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing."""
    now = utc_now()
    return [
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User woke up at 6am and went for a morning jog.",
            keywords=["morning", "exercise", "routine"],
            tags=["health"],
            occurred_at=now - timedelta(days=7),
            recorded_at=now - timedelta(days=7),
            salience=0.6,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User started the day with a 6am run, as usual.",
            keywords=["morning", "exercise", "routine"],
            tags=["health"],
            occurred_at=now - timedelta(days=5),
            recorded_at=now - timedelta(days=5),
            salience=0.6,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User mentioned they exercise every morning at 6am.",
            keywords=["morning", "exercise", "routine", "consistency"],
            tags=["health"],
            occurred_at=now - timedelta(days=3),
            recorded_at=now - timedelta(days=3),
            salience=0.7,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User prefers dark roast coffee over light roast.",
            keywords=["coffee", "preference", "dark-roast"],
            tags=["food"],
            occurred_at=now - timedelta(days=2),
            recorded_at=now - timedelta(days=2),
            salience=0.5,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User always chooses Python for backend development projects.",
            keywords=["python", "programming", "preference"],
            tags=["work", "technology"],
            occurred_at=now - timedelta(days=1),
            recorded_at=now - timedelta(days=1),
            salience=0.8,
        ),
    ]


@pytest.fixture
def sample_patterns():
    """Create sample existing patterns."""
    now = utc_now()
    return [
        Pattern(
            id="pat_001",
            description="User exercises every morning at 6am",
            pattern_type="behavioral",
            confidence=0.8,
            occurrences=["epi_001", "epi_002"],
            first_seen=now - timedelta(days=30),
            last_seen=now - timedelta(days=10),
            consolidation_strength=8.0,
        ),
        Pattern(
            id="pat_002",
            description="User prefers Python for development",
            pattern_type="preference",
            confidence=0.7,
            occurrences=["epi_003", "epi_004"],
            first_seen=now - timedelta(days=20),
            last_seen=now - timedelta(days=5),
            consolidation_strength=7.0,
        ),
    ]


# ========== Test detect_patterns ==========


@pytest.mark.asyncio
async def test_detect_patterns_empty_episodes(pattern_detector):
    """Test pattern detection with empty episode list."""
    result = await pattern_detector.detect_patterns([], [])
    assert isinstance(result, PatternDetectionResult)
    assert len(result.new_patterns) == 0
    assert len(result.strengthened) == 0
    assert len(result.weakened) == 0


@pytest.mark.asyncio
async def test_detect_patterns_insufficient_episodes(
    pattern_detector, sample_episodes
):
    """Test that insufficient episodes returns empty result."""
    result = await pattern_detector.detect_patterns(
        sample_episodes[:1], [], min_occurrences=2
    )
    assert len(result.new_patterns) == 0


@pytest.mark.asyncio
async def test_detect_patterns_new_pattern(
    pattern_detector, sample_episodes, mock_llm
):
    """Test detection of new pattern."""
    # Mock LLM to return a pattern
    pattern_response = {
        "has_pattern": True,
        "description": "User exercises every morning at 6am",
        "pattern_type": "behavioral",
        "confidence": 0.75,
        "keywords": ["morning", "exercise", "6am"],
        "evidence": ["Episode 1: morning jog", "Episode 2: 6am run"],
    }
    mock_llm.chat = AsyncMock(return_value=json.dumps(pattern_response))

    # Use first 3 episodes (exercise pattern)
    result = await pattern_detector.detect_patterns(
        sample_episodes[:3], [], min_occurrences=2
    )

    assert len(result.new_patterns) >= 1
    pattern = result.new_patterns[0]
    assert pattern.pattern_type == "behavioral"
    assert "exercise" in pattern.description.lower() or "morning" in pattern.description.lower()
    assert len(pattern.occurrences) == 3


@pytest.mark.asyncio
async def test_detect_patterns_strengthens_existing(
    pattern_detector, sample_episodes, sample_patterns, mock_llm
):
    """Test that matching patterns strengthen existing ones."""
    # Mock LLM to return a pattern similar to existing one
    pattern_response = {
        "has_pattern": True,
        "description": "User does morning workouts at 6am",
        "pattern_type": "behavioral",
        "confidence": 0.75,
        "keywords": ["morning", "exercise", "6am"],
        "evidence": ["Episode 1: morning jog"],
    }
    mock_llm.chat = AsyncMock(return_value=json.dumps(pattern_response))

    # Mock embeddings to show high similarity
    mock_llm.embed_batch = AsyncMock(
        return_value=[
            [0.5] * 384,  # Candidate pattern
            [0.51] * 384,  # Existing pattern (very similar)
            [0.1] * 384,  # Other existing pattern (different)
        ]
    )

    result = await pattern_detector.detect_patterns(
        sample_episodes[:3], sample_patterns, min_occurrences=2, min_similarity=0.7
    )

    # Should strengthen existing pattern
    assert len(result.strengthened) >= 1
    strengthened_id, new_confidence = result.strengthened[0]
    assert strengthened_id == sample_patterns[0].id
    assert new_confidence > sample_patterns[0].confidence


@pytest.mark.asyncio
async def test_detect_patterns_weakens_unseen(
    pattern_detector, sample_episodes, sample_patterns, mock_llm
):
    """Test that patterns not in recent episodes are weakened."""
    # Mock LLM to return no pattern or different pattern
    pattern_response = {
        "has_pattern": True,
        "description": "User prefers coffee in the afternoon",
        "pattern_type": "preference",
        "confidence": 0.6,
        "keywords": ["coffee", "afternoon"],
        "evidence": ["Episode 4: mentioned coffee"],
    }
    mock_llm.chat = AsyncMock(return_value=json.dumps(pattern_response))

    # Mock embeddings to show low similarity
    mock_llm.embed_batch = AsyncMock(
        return_value=[
            [0.5] * 384,  # Candidate pattern
            [0.1] * 384,  # Existing pattern 1 (very different)
            [0.15] * 384,  # Existing pattern 2 (very different)
        ]
    )

    # Use episodes that don't match existing patterns
    result = await pattern_detector.detect_patterns(
        sample_episodes[3:5], sample_patterns, min_occurrences=1, min_similarity=0.7
    )

    # Existing patterns should be weakened since they don't appear
    assert len(result.weakened) >= 1


@pytest.mark.asyncio
async def test_detect_patterns_no_pattern_found(
    pattern_detector, sample_episodes, mock_llm
):
    """Test when LLM finds no pattern."""
    # Mock LLM to return no pattern
    no_pattern_response = {"has_pattern": False}
    mock_llm.chat = AsyncMock(return_value=json.dumps(no_pattern_response))

    result = await pattern_detector.detect_patterns(
        sample_episodes[:2], [], min_occurrences=2
    )

    assert len(result.new_patterns) == 0


# ========== Test extract_pattern ==========


@pytest.mark.asyncio
async def test_extract_pattern_success(
    pattern_detector, sample_episodes, mock_llm
):
    """Test successful pattern extraction."""
    pattern_response = {
        "has_pattern": True,
        "description": "User exercises every morning at 6am",
        "pattern_type": "behavioral",
        "confidence": 0.8,
        "keywords": ["morning", "exercise", "routine"],
        "evidence": ["Episode 1: jogging", "Episode 2: running"],
    }
    mock_llm.chat = AsyncMock(return_value=json.dumps(pattern_response))

    pattern = await pattern_detector.extract_pattern(sample_episodes[:3])

    assert pattern is not None
    assert pattern.description == pattern_response["description"]
    assert pattern.pattern_type == "behavioral"
    assert pattern.confidence == 0.8
    assert len(pattern.occurrences) == 3
    assert "morning" in pattern.metadata.get("keywords", [])


@pytest.mark.asyncio
async def test_extract_pattern_empty_episodes(pattern_detector):
    """Test extraction with empty episode list."""
    pattern = await pattern_detector.extract_pattern([])
    assert pattern is None


@pytest.mark.asyncio
async def test_extract_pattern_no_pattern_found(
    pattern_detector, sample_episodes, mock_llm
):
    """Test when no pattern is found."""
    no_pattern_response = {"has_pattern": False}
    mock_llm.chat = AsyncMock(return_value=json.dumps(no_pattern_response))

    pattern = await pattern_detector.extract_pattern(sample_episodes[:2])
    assert pattern is None


@pytest.mark.asyncio
async def test_extract_pattern_with_type_hint(
    pattern_detector, sample_episodes, mock_llm
):
    """Test extraction with pattern type hint."""
    pattern_response = {
        "has_pattern": True,
        "description": "User prefers dark roast coffee",
        "pattern_type": "preference",
        "confidence": 0.7,
        "keywords": ["coffee", "dark-roast"],
        "evidence": ["Episode 4: mentioned preference"],
    }
    mock_llm.chat = AsyncMock(return_value=json.dumps(pattern_response))

    pattern = await pattern_detector.extract_pattern(
        [sample_episodes[3]], pattern_type="preference"
    )

    assert pattern is not None
    assert pattern.pattern_type == "preference"


@pytest.mark.asyncio
async def test_extract_pattern_llm_failure(
    pattern_detector, sample_episodes, mock_llm
):
    """Test handling of LLM failure."""
    mock_llm.chat = AsyncMock(side_effect=Exception("LLM failed"))

    with pytest.raises(ConsolidationError, match="Failed to extract pattern"):
        await pattern_detector.extract_pattern(sample_episodes[:2])


@pytest.mark.asyncio
async def test_extract_pattern_invalid_json(
    pattern_detector, sample_episodes, mock_llm
):
    """Test handling of invalid JSON response."""
    mock_llm.chat = AsyncMock(return_value="Not valid JSON")

    with pytest.raises(ConsolidationError):
        await pattern_detector.extract_pattern(sample_episodes[:2])


@pytest.mark.asyncio
async def test_extract_pattern_markdown_wrapped(
    pattern_detector, sample_episodes, mock_llm
):
    """Test parsing JSON wrapped in markdown."""
    pattern_response = {
        "has_pattern": True,
        "description": "Test pattern",
        "pattern_type": "behavioral",
        "confidence": 0.7,
    }
    wrapped = f"```json\n{json.dumps(pattern_response)}\n```"
    mock_llm.chat = AsyncMock(return_value=wrapped)

    pattern = await pattern_detector.extract_pattern(sample_episodes[:2])
    assert pattern is not None
    assert pattern.description == "Test pattern"


# ========== Test match_to_existing ==========


@pytest.mark.asyncio
async def test_match_to_existing_no_patterns(pattern_detector, mock_llm):
    """Test matching with no existing patterns."""
    candidate = Pattern(
        id="pat_new",
        description="User exercises daily",
        pattern_type="behavioral",
        confidence=0.7,
        occurrences=["epi_001"],
    )

    match = await pattern_detector.match_to_existing(candidate, [], min_similarity=0.7)
    assert match is None


@pytest.mark.asyncio
async def test_match_to_existing_different_type(
    pattern_detector, sample_patterns, mock_llm
):
    """Test that only same-type patterns are compared."""
    candidate = Pattern(
        id="pat_new",
        description="User prefers working in the morning",
        pattern_type="preference",
        confidence=0.7,
        occurrences=["epi_001"],
    )

    # Mock high similarity but different types should not match
    mock_llm.embed_batch = AsyncMock(
        return_value=[
            [0.9] * 384,  # Candidate
            [0.91] * 384,  # Existing pattern (different type, should be filtered)
        ]
    )

    # Should only compare with preference-type patterns
    match = await pattern_detector.match_to_existing(
        candidate, [sample_patterns[0]], min_similarity=0.7
    )
    # sample_patterns[0] is behavioral, so no match
    assert match is None


@pytest.mark.asyncio
async def test_match_to_existing_high_similarity(
    pattern_detector, sample_patterns, mock_llm
):
    """Test matching with high similarity."""
    candidate = Pattern(
        id="pat_new",
        description="User does morning workouts at 6am",
        pattern_type="behavioral",
        confidence=0.7,
        occurrences=["epi_005"],
    )

    # Mock high similarity
    mock_llm.embed_batch = AsyncMock(
        return_value=[
            [0.5] * 384,  # Candidate
            [0.51] * 384,  # Existing pattern (very similar)
        ]
    )

    match = await pattern_detector.match_to_existing(
        candidate, [sample_patterns[0]], min_similarity=0.7
    )

    assert match is not None
    assert match.id == sample_patterns[0].id


@pytest.mark.asyncio
async def test_match_to_existing_low_similarity(
    pattern_detector, sample_patterns, mock_llm
):
    """Test no match with low similarity."""
    candidate = Pattern(
        id="pat_new",
        description="User drinks coffee in the evening",
        pattern_type="behavioral",
        confidence=0.7,
        occurrences=["epi_005"],
    )

    # Mock low similarity
    mock_llm.embed_batch = AsyncMock(
        return_value=[
            [1.0] * 384,  # Candidate
            [0.0] * 384,  # Existing pattern (very different)
        ]
    )

    match = await pattern_detector.match_to_existing(
        candidate, [sample_patterns[0]], min_similarity=0.7
    )

    assert match is None


@pytest.mark.asyncio
async def test_match_to_existing_embedding_failure(
    pattern_detector, sample_patterns, mock_llm
):
    """Test handling of embedding failure."""
    candidate = Pattern(
        id="pat_new",
        description="Test pattern",
        pattern_type="behavioral",
        confidence=0.7,
        occurrences=["epi_005"],
    )

    mock_llm.embed_batch = AsyncMock(side_effect=Exception("Embedding failed"))

    # Should return None on failure
    match = await pattern_detector.match_to_existing(
        candidate, [sample_patterns[0]], min_similarity=0.7
    )
    assert match is None


# ========== Test helper methods ==========


def test_cosine_similarity_identical():
    """Test cosine similarity with identical vectors."""
    detector = PatternDetector(MagicMock(), "test-model")
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]

    similarity = detector._cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(1.0, abs=0.01)


def test_cosine_similarity_opposite():
    """Test cosine similarity with opposite vectors."""
    detector = PatternDetector(MagicMock(), "test-model")
    vec1 = [1.0, 1.0, 1.0]
    vec2 = [-1.0, -1.0, -1.0]

    similarity = detector._cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(0.0, abs=0.01)


def test_cosine_similarity_orthogonal():
    """Test cosine similarity with orthogonal vectors."""
    detector = PatternDetector(MagicMock(), "test-model")
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    similarity = detector._cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(0.5, abs=0.01)


def test_calculate_strength_low_occurrences():
    """Test strength calculation with few occurrences."""
    detector = PatternDetector(MagicMock(), "test-model")
    strength = detector._calculate_strength(2, 0.5)
    # base_strength = 5.0 + (2.0 * sqrt(2)) ≈ 7.83
    # adjusted = 7.83 * 0.5 ≈ 3.91
    assert 3.5 <= strength <= 4.5


def test_calculate_strength_high_occurrences():
    """Test strength calculation with many occurrences."""
    detector = PatternDetector(MagicMock(), "test-model")
    strength = detector._calculate_strength(20, 0.9)
    assert strength > 10.0
    assert strength <= 20.0  # Capped at 20


def test_calculate_strength_high_confidence():
    """Test that high confidence increases strength."""
    detector = PatternDetector(MagicMock(), "test-model")
    strength_low = detector._calculate_strength(5, 0.3)
    strength_high = detector._calculate_strength(5, 0.9)
    assert strength_high > strength_low


def test_descriptions_similar_exact():
    """Test description similarity with exact match."""
    detector = PatternDetector(MagicMock(), "test-model")
    assert detector._descriptions_similar(
        "User exercises daily", "User exercises daily"
    )


def test_descriptions_similar_case():
    """Test description similarity is case-insensitive."""
    detector = PatternDetector(MagicMock(), "test-model")
    assert detector._descriptions_similar(
        "User Exercises Daily", "user exercises daily"
    )


def test_descriptions_similar_substring():
    """Test description similarity with substring."""
    detector = PatternDetector(MagicMock(), "test-model")
    assert detector._descriptions_similar(
        "User exercises", "User exercises every morning"
    )


def test_descriptions_similar_word_overlap():
    """Test description similarity with significant word overlap."""
    detector = PatternDetector(MagicMock(), "test-model")
    assert detector._descriptions_similar(
        "User exercises every morning", "User workouts every morning"
    )


def test_descriptions_not_similar():
    """Test descriptions that are not similar."""
    detector = PatternDetector(MagicMock(), "test-model")
    assert not detector._descriptions_similar(
        "User exercises daily", "User prefers coffee"
    )


# ========== Test parse_extraction_response ==========


def test_parse_extraction_response_valid():
    """Test parsing valid response."""
    detector = PatternDetector(MagicMock(), "test-model")
    response = json.dumps(
        {
            "has_pattern": True,
            "description": "Test pattern",
            "pattern_type": "behavioral",
            "confidence": 0.8,
            "keywords": ["test"],
            "evidence": ["Episode 1: test"],
        }
    )

    data = detector._parse_extraction_response(response)
    assert data["has_pattern"] is True
    assert data["description"] == "Test pattern"
    assert data["pattern_type"] == "behavioral"
    assert data["confidence"] == 0.8


def test_parse_extraction_response_no_pattern():
    """Test parsing response with no pattern."""
    detector = PatternDetector(MagicMock(), "test-model")
    response = json.dumps({"has_pattern": False})

    data = detector._parse_extraction_response(response)
    assert data["has_pattern"] is False


def test_parse_extraction_response_missing_fields():
    """Test parsing response with missing required fields."""
    detector = PatternDetector(MagicMock(), "test-model")
    response = json.dumps({"has_pattern": True, "pattern_type": "behavioral"})

    with pytest.raises(Exception):  # Should raise LLMResponseError
        detector._parse_extraction_response(response)


def test_parse_extraction_response_invalid_pattern_type():
    """Test parsing response with invalid pattern type."""
    detector = PatternDetector(MagicMock(), "test-model")
    response = json.dumps(
        {
            "has_pattern": True,
            "description": "Test",
            "pattern_type": "invalid_type",
        }
    )

    with pytest.raises(Exception):  # Should raise LLMResponseError
        detector._parse_extraction_response(response)


def test_parse_extraction_response_clamps_confidence():
    """Test that confidence is clamped to valid range."""
    detector = PatternDetector(MagicMock(), "test-model")
    response = json.dumps(
        {
            "has_pattern": True,
            "description": "Test",
            "pattern_type": "behavioral",
            "confidence": 1.5,  # Invalid, should be clamped
        }
    )

    data = detector._parse_extraction_response(response)
    assert 0.0 <= data["confidence"] <= 1.0
