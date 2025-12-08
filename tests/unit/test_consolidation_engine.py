"""Unit tests for ConsolidationEngine link maintenance and full cycle."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from htma.consolidation.engine import ConsolidationConfig, ConsolidationEngine
from htma.core.exceptions import ConsolidationError, DatabaseError
from htma.core.types import (
    ConsolidationReport,
    Episode,
    EpisodeLink,
    LinkMaintenanceReport,
    PruneReport,
)
from htma.core.utils import generate_episode_id, generate_link_id, utc_now
from htma.memory.episodic import EpisodicMemory


@pytest.fixture
def mock_sqlite():
    """Create a mock SQLite storage."""
    sqlite = MagicMock()
    sqlite.fetch_all = AsyncMock(return_value=[])
    sqlite.fetch_one = AsyncMock(return_value=None)
    sqlite.execute = AsyncMock()
    return sqlite


@pytest.fixture
def mock_chroma():
    """Create a mock ChromaDB storage."""
    chroma = MagicMock()
    return chroma


@pytest.fixture
def episodic_memory(mock_sqlite, mock_chroma):
    """Create an EpisodicMemory instance with mocked storage."""
    return EpisodicMemory(sqlite=mock_sqlite, chroma=mock_chroma)


@pytest.fixture
def mock_curator():
    """Create a mock MemoryCurator."""
    curator = MagicMock()
    curator.resolve_conflict = AsyncMock()
    return curator


@pytest.fixture
def mock_semantic():
    """Create a mock SemanticMemory."""
    semantic = MagicMock()
    return semantic


@pytest.fixture
def mock_abstraction_generator():
    """Create a mock AbstractionGenerator."""
    generator = MagicMock()
    generator.cluster_episodes = AsyncMock(return_value=[])
    generator.generate_abstraction = AsyncMock()
    return generator


@pytest.fixture
def mock_pattern_detector():
    """Create a mock PatternDetector."""
    from htma.core.types import PatternDetectionResult

    detector = MagicMock()
    detector.detect_patterns = AsyncMock(return_value=PatternDetectionResult())
    return detector


@pytest.fixture
def consolidation_engine(
    mock_curator,
    mock_semantic,
    episodic_memory,
    mock_abstraction_generator,
    mock_pattern_detector,
):
    """Create a ConsolidationEngine instance."""
    return ConsolidationEngine(
        curator=mock_curator,
        semantic=mock_semantic,
        episodic=episodic_memory,
        abstraction_generator=mock_abstraction_generator,
        pattern_detector=mock_pattern_detector,
    )


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing."""
    now = utc_now()
    return [
        Episode(
            id=generate_episode_id(),
            level=0,
            content="First episode",
            occurred_at=now - timedelta(hours=2),
            recorded_at=now - timedelta(hours=2),
            last_accessed=now - timedelta(minutes=30),
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="Second episode",
            occurred_at=now - timedelta(hours=1),
            recorded_at=now - timedelta(hours=1),
            last_accessed=now - timedelta(minutes=25),
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="Third episode",
            occurred_at=now - timedelta(minutes=30),
            recorded_at=now - timedelta(minutes=30),
            last_accessed=now - timedelta(minutes=5),
        ),
    ]


@pytest.fixture
def sample_links(sample_episodes):
    """Create sample episode links for testing."""
    return [
        EpisodeLink(
            id=generate_link_id(),
            source_id=sample_episodes[0].id,
            target_id=sample_episodes[1].id,
            link_type="semantic",
            weight=0.8,
            created_at=utc_now() - timedelta(days=1),
        ),
        EpisodeLink(
            id=generate_link_id(),
            source_id=sample_episodes[1].id,
            target_id=sample_episodes[2].id,
            link_type="temporal",
            weight=0.5,
            created_at=utc_now() - timedelta(days=1),
        ),
        EpisodeLink(
            id=generate_link_id(),
            source_id=sample_episodes[0].id,
            target_id=sample_episodes[2].id,
            link_type="causal",
            weight=0.05,  # Very weak link
            created_at=utc_now() - timedelta(days=2),
        ),
    ]


class TestConsolidationEngineInit:
    """Test ConsolidationEngine initialization."""

    def test_init(
        self,
        mock_curator,
        mock_semantic,
        episodic_memory,
        mock_abstraction_generator,
        mock_pattern_detector,
    ):
        """Test basic initialization."""
        engine = ConsolidationEngine(
            curator=mock_curator,
            semantic=mock_semantic,
            episodic=episodic_memory,
            abstraction_generator=mock_abstraction_generator,
            pattern_detector=mock_pattern_detector,
        )
        assert engine.episodic == episodic_memory
        assert engine.curator == mock_curator
        assert engine.semantic == mock_semantic
        assert engine.abstraction_generator == mock_abstraction_generator
        assert engine.pattern_detector == mock_pattern_detector


class TestStrengthenCoaccessed:
    """Test strengthen_coaccessed method."""

    @pytest.mark.asyncio
    async def test_no_recent_accesses(self, consolidation_engine, mock_sqlite):
        """Test with no recently accessed episodes."""
        mock_sqlite.fetch_all.return_value = []

        count = await consolidation_engine.strengthen_coaccessed()

        assert count == 0
        # Should query for recently accessed episodes
        assert mock_sqlite.fetch_all.called

    @pytest.mark.asyncio
    async def test_single_access(self, consolidation_engine, mock_sqlite):
        """Test with only one recently accessed episode."""
        now = utc_now()
        mock_sqlite.fetch_all.return_value = [
            {"id": "epi_001", "last_accessed": now.isoformat()}
        ]

        count = await consolidation_engine.strengthen_coaccessed()

        assert count == 0  # Need at least 2 episodes

    @pytest.mark.asyncio
    async def test_no_coaccessed_pairs(self, consolidation_engine, mock_sqlite):
        """Test when episodes are accessed but not within window."""
        now = utc_now()
        # Episodes accessed 3 hours apart (outside default 1-hour window)
        mock_sqlite.fetch_all.return_value = [
            {
                "id": "epi_001",
                "last_accessed": (now - timedelta(hours=3)).isoformat(),
            },
            {"id": "epi_002", "last_accessed": now.isoformat()},
        ]

        count = await consolidation_engine.strengthen_coaccessed()

        assert count == 0

    @pytest.mark.asyncio
    async def test_strengthen_coaccessed_link(
        self, consolidation_engine, mock_sqlite, sample_episodes
    ):
        """Test strengthening a link between co-accessed episodes."""
        now = utc_now()

        # Two episodes accessed within 30 minutes
        accesses = [
            {
                "id": sample_episodes[0].id,
                "last_accessed": (now - timedelta(minutes=30)).isoformat(),
            },
            {
                "id": sample_episodes[1].id,
                "last_accessed": (now - timedelta(minutes=25)).isoformat(),
            },
        ]

        # Mock existing link between these episodes
        existing_link = {"id": "link_001", "weight": 0.8}

        # Setup mock responses
        async def mock_fetch_all(query, params):
            if "episodes" in query:
                return accesses
            return []

        async def mock_fetch_one(query, params):
            if "episode_links" in query:
                return existing_link
            return None

        mock_sqlite.fetch_all.side_effect = mock_fetch_all
        mock_sqlite.fetch_one.side_effect = mock_fetch_one

        count = await consolidation_engine.strengthen_coaccessed()

        assert count == 1
        # Should have updated the link weight
        assert mock_sqlite.execute.called
        # Verify the update query was called
        update_call = mock_sqlite.execute.call_args
        assert update_call is not None

    @pytest.mark.asyncio
    async def test_no_link_exists_between_coaccessed(
        self, consolidation_engine, mock_sqlite, sample_episodes
    ):
        """Test when episodes are co-accessed but no link exists."""
        now = utc_now()

        # Two episodes accessed within 30 minutes
        accesses = [
            {
                "id": sample_episodes[0].id,
                "last_accessed": (now - timedelta(minutes=30)).isoformat(),
            },
            {
                "id": sample_episodes[1].id,
                "last_accessed": (now - timedelta(minutes=25)).isoformat(),
            },
        ]

        # Mock no existing link
        async def mock_fetch_all(query, params):
            if "episodes" in query:
                return accesses
            return []

        async def mock_fetch_one(query, params):
            # No link exists
            return None

        mock_sqlite.fetch_all.side_effect = mock_fetch_all
        mock_sqlite.fetch_one.side_effect = mock_fetch_one

        count = await consolidation_engine.strengthen_coaccessed()

        assert count == 0  # No links strengthened (no link existed)

    @pytest.mark.asyncio
    async def test_multiple_coaccessed_pairs(
        self, consolidation_engine, mock_sqlite, sample_episodes
    ):
        """Test strengthening multiple co-accessed pairs."""
        now = utc_now()

        # Three episodes all accessed within a short window
        accesses = [
            {
                "id": sample_episodes[0].id,
                "last_accessed": (now - timedelta(minutes=30)).isoformat(),
            },
            {
                "id": sample_episodes[1].id,
                "last_accessed": (now - timedelta(minutes=25)).isoformat(),
            },
            {
                "id": sample_episodes[2].id,
                "last_accessed": (now - timedelta(minutes=20)).isoformat(),
            },
        ]

        # Mock links exist for all pairs
        link_counter = [0]

        async def mock_fetch_all(query, params):
            if "episodes" in query:
                return accesses
            return []

        async def mock_fetch_one(query, params):
            if "episode_links" in query:
                link_counter[0] += 1
                return {"id": f"link_{link_counter[0]}", "weight": 0.5}
            return None

        mock_sqlite.fetch_all.side_effect = mock_fetch_all
        mock_sqlite.fetch_one.side_effect = mock_fetch_one

        count = await consolidation_engine.strengthen_coaccessed()

        # Should find 3 pairs: (0,1), (0,2), (1,2)
        assert count == 3


class TestDecayUnused:
    """Test decay_unused method."""

    @pytest.mark.asyncio
    async def test_no_links(self, consolidation_engine, mock_sqlite):
        """Test with no links in the database."""
        mock_sqlite.fetch_all.return_value = []

        count = await consolidation_engine.decay_unused()

        assert count == 0

    @pytest.mark.asyncio
    async def test_decay_all_links(self, consolidation_engine, mock_sqlite):
        """Test decaying all links."""
        links = [
            {"id": "link_001", "weight": 1.0},
            {"id": "link_002", "weight": 0.5},
            {"id": "link_003", "weight": 0.2},
        ]

        mock_sqlite.fetch_all.return_value = links

        count = await consolidation_engine.decay_unused(decay_rate=0.1)

        assert count == 3  # All links decayed
        # Should have 3 update calls
        assert mock_sqlite.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_decay_respects_min_weight(self, consolidation_engine, mock_sqlite):
        """Test that decay doesn't go below min_weight."""
        links = [{"id": "link_001", "weight": 0.15}]

        mock_sqlite.fetch_all.return_value = links

        count = await consolidation_engine.decay_unused(
            decay_rate=0.5, min_weight=0.1
        )

        assert count == 1
        # Check that update was called with weight >= min_weight
        update_call = mock_sqlite.execute.call_args
        assert update_call is not None
        new_weight = update_call[0][1][0]
        assert new_weight >= 0.1

    @pytest.mark.asyncio
    async def test_invalid_decay_rate(self, consolidation_engine):
        """Test with invalid decay rate."""
        with pytest.raises(ValueError, match="Decay rate must be between"):
            await consolidation_engine.decay_unused(decay_rate=1.5)

        with pytest.raises(ValueError, match="Decay rate must be between"):
            await consolidation_engine.decay_unused(decay_rate=-0.1)

    @pytest.mark.asyncio
    async def test_invalid_min_weight(self, consolidation_engine):
        """Test with invalid min_weight."""
        with pytest.raises(ValueError, match="Min weight must be"):
            await consolidation_engine.decay_unused(min_weight=-0.1)


class TestPruneWeakLinks:
    """Test prune_weak_links method."""

    @pytest.mark.asyncio
    async def test_no_weak_links(self, consolidation_engine, mock_sqlite):
        """Test when no links are below threshold."""
        mock_sqlite.fetch_one.return_value = {"count": 0}

        count = await consolidation_engine.prune_weak_links(threshold=0.1)

        assert count == 0
        # Should not execute deletion if count is 0
        # Only the count query should be called
        assert mock_sqlite.fetch_one.called

    @pytest.mark.asyncio
    async def test_prune_weak_links(self, consolidation_engine, mock_sqlite):
        """Test pruning weak links."""
        mock_sqlite.fetch_one.return_value = {"count": 5}

        count = await consolidation_engine.prune_weak_links(threshold=0.1)

        assert count == 5
        # Should execute deletion
        assert mock_sqlite.execute.called

    @pytest.mark.asyncio
    async def test_prune_with_custom_threshold(self, consolidation_engine, mock_sqlite):
        """Test pruning with custom threshold."""
        mock_sqlite.fetch_one.return_value = {"count": 3}

        count = await consolidation_engine.prune_weak_links(threshold=0.5)

        assert count == 3
        # Verify threshold was used in query
        query_call = mock_sqlite.fetch_one.call_args
        assert query_call is not None
        threshold_param = query_call[0][1][0]
        assert threshold_param == 0.5

    @pytest.mark.asyncio
    async def test_invalid_threshold(self, consolidation_engine):
        """Test with invalid threshold."""
        with pytest.raises(ValueError, match="Threshold must be"):
            await consolidation_engine.prune_weak_links(threshold=-0.1)


class TestUpdateLinkWeights:
    """Test update_link_weights orchestration method."""

    @pytest.mark.asyncio
    async def test_full_cycle(self, consolidation_engine, mock_sqlite):
        """Test complete link maintenance cycle."""
        # Mock initial link count
        async def mock_fetch_one(query, params):
            if "COUNT" in query:
                return {"count": 10}
            return None

        mock_sqlite.fetch_one.side_effect = mock_fetch_one
        mock_sqlite.fetch_all.return_value = []

        report = await consolidation_engine.update_link_weights()

        assert isinstance(report, LinkMaintenanceReport)
        assert report.total_links_before == 10
        assert report.total_links_after == 10
        assert report.processing_time >= 0
        assert "access_window_hours" in report.metadata
        assert "decay_rate" in report.metadata

    @pytest.mark.asyncio
    async def test_cycle_with_operations(self, consolidation_engine, mock_sqlite):
        """Test cycle with actual link operations."""
        # Setup: 15 links initially, 2 weak ones pruned
        call_count = [0]

        async def mock_fetch_one(query, params):
            if "COUNT" in query:
                # First call: before maintenance
                # Second call: after maintenance
                call_count[0] += 1
                if call_count[0] == 1:
                    return {"count": 15}
                elif call_count[0] == 2:
                    return {"count": 13}  # After pruning 2
                return {"count": 0}
            return None

        async def mock_fetch_all(query, params):
            if "episodes" in query:
                # No co-accessed episodes
                return []
            elif "episode_links" in query:
                # Mock some links for decay
                return [
                    {"id": "link_001", "weight": 1.0},
                    {"id": "link_002", "weight": 0.5},
                ]
            return []

        mock_sqlite.fetch_one.side_effect = mock_fetch_one
        mock_sqlite.fetch_all.side_effect = mock_fetch_all

        report = await consolidation_engine.update_link_weights(
            decay_rate=0.1, prune_threshold=0.1
        )

        assert report.total_links_before == 15
        assert report.total_links_after == 13
        assert report.links_decayed == 2

    @pytest.mark.asyncio
    async def test_cycle_error_handling(self, consolidation_engine, mock_sqlite):
        """Test error handling in update cycle."""
        # Make a database operation fail
        mock_sqlite.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(ConsolidationError, match="Link maintenance failed"):
            await consolidation_engine.update_link_weights()


class TestHelperMethods:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_count_total_links(self, consolidation_engine, mock_sqlite):
        """Test counting total links."""
        mock_sqlite.fetch_one.return_value = {"count": 42}

        count = await consolidation_engine._count_total_links()

        assert count == 42

    @pytest.mark.asyncio
    async def test_count_total_links_empty(self, consolidation_engine, mock_sqlite):
        """Test counting when no links exist."""
        mock_sqlite.fetch_one.return_value = {"count": 0}

        count = await consolidation_engine._count_total_links()

        assert count == 0
