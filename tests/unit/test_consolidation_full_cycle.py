"""Unit tests for ConsolidationEngine full cycle operations."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from htma.consolidation.engine import ConsolidationConfig, ConsolidationEngine
from htma.core.exceptions import ConsolidationError, DatabaseError
from htma.core.types import (
    ConsolidationReport,
    Episode,
    PatternDetectionResult,
    PruneReport,
)
from htma.core.utils import generate_episode_id, utc_now


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
    chroma.delete_episode = AsyncMock()
    return chroma


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
def mock_episodic(mock_sqlite, mock_chroma):
    """Create a mock EpisodicMemory."""
    from htma.memory.episodic import EpisodicMemory

    return EpisodicMemory(sqlite=mock_sqlite, chroma=mock_chroma)


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
    detector = MagicMock()
    detector.detect_patterns = AsyncMock(return_value=PatternDetectionResult())
    return detector


@pytest.fixture
def consolidation_config():
    """Create a test configuration."""
    return ConsolidationConfig(
        min_episodes_before_cycle=5,
        max_time_between_cycles=timedelta(hours=12),
        abstraction_cluster_size=3,
        pattern_min_occurrences=2,
        prune_access_threshold=0,
        prune_age_threshold=timedelta(days=7),
        max_episodes_per_cycle=50,
    )


@pytest.fixture
def consolidation_engine(
    mock_curator,
    mock_semantic,
    mock_episodic,
    mock_abstraction_generator,
    mock_pattern_detector,
    consolidation_config,
):
    """Create a ConsolidationEngine instance."""
    return ConsolidationEngine(
        curator=mock_curator,
        semantic=mock_semantic,
        episodic=mock_episodic,
        abstraction_generator=mock_abstraction_generator,
        pattern_detector=mock_pattern_detector,
        config=consolidation_config,
    )


class TestConsolidationEngineInit:
    """Test ConsolidationEngine initialization with full dependencies."""

    def test_init_with_config(
        self,
        mock_curator,
        mock_semantic,
        mock_episodic,
        mock_abstraction_generator,
        mock_pattern_detector,
        consolidation_config,
    ):
        """Test initialization with configuration."""
        engine = ConsolidationEngine(
            curator=mock_curator,
            semantic=mock_semantic,
            episodic=mock_episodic,
            abstraction_generator=mock_abstraction_generator,
            pattern_detector=mock_pattern_detector,
            config=consolidation_config,
        )

        assert engine.curator == mock_curator
        assert engine.semantic == mock_semantic
        assert engine.episodic == mock_episodic
        assert engine.abstraction_generator == mock_abstraction_generator
        assert engine.pattern_detector == mock_pattern_detector
        assert engine.config == consolidation_config
        assert engine.last_cycle is None

    def test_init_with_default_config(
        self,
        mock_curator,
        mock_semantic,
        mock_episodic,
        mock_abstraction_generator,
        mock_pattern_detector,
    ):
        """Test initialization with default configuration."""
        engine = ConsolidationEngine(
            curator=mock_curator,
            semantic=mock_semantic,
            episodic=mock_episodic,
            abstraction_generator=mock_abstraction_generator,
            pattern_detector=mock_pattern_detector,
        )

        assert isinstance(engine.config, ConsolidationConfig)
        assert engine.config.min_episodes_before_cycle == 10
        assert engine.config.max_time_between_cycles == timedelta(hours=24)


class TestShouldRun:
    """Test should_run method."""

    @pytest.mark.asyncio
    async def test_never_run_with_enough_episodes(
        self, consolidation_engine, mock_sqlite
    ):
        """Test when never run and enough episodes exist."""
        mock_sqlite.fetch_one.return_value = {"count": 10}

        should_run = await consolidation_engine.should_run()

        assert should_run is True

    @pytest.mark.asyncio
    async def test_never_run_not_enough_episodes(
        self, consolidation_engine, mock_sqlite
    ):
        """Test when never run and not enough episodes."""
        mock_sqlite.fetch_one.return_value = {"count": 3}

        should_run = await consolidation_engine.should_run()

        assert should_run is False

    @pytest.mark.asyncio
    async def test_time_exceeded(self, consolidation_engine, mock_sqlite):
        """Test when max time between cycles exceeded."""
        consolidation_engine.last_cycle = utc_now() - timedelta(hours=24)

        should_run = await consolidation_engine.should_run()

        assert should_run is True

    @pytest.mark.asyncio
    async def test_enough_new_episodes(self, consolidation_engine, mock_sqlite):
        """Test when enough new episodes since last cycle."""
        consolidation_engine.last_cycle = utc_now() - timedelta(hours=2)
        mock_sqlite.fetch_one.return_value = {"count": 10}

        should_run = await consolidation_engine.should_run()

        assert should_run is True

    @pytest.mark.asyncio
    async def test_not_enough_new_episodes(self, consolidation_engine, mock_sqlite):
        """Test when not enough new episodes and time not exceeded."""
        consolidation_engine.last_cycle = utc_now() - timedelta(hours=2)
        mock_sqlite.fetch_one.return_value = {"count": 2}

        should_run = await consolidation_engine.should_run()

        assert should_run is False


class TestPruneStale:
    """Test prune_stale method."""

    @pytest.mark.asyncio
    async def test_no_episodes_to_prune(self, consolidation_engine, mock_sqlite):
        """Test when no episodes are eligible for pruning."""
        # Mock episode count
        async def mock_fetch_one(query, params):
            if "COUNT" in query:
                return {"count": 100}
            return None

        # Mock no episodes to prune
        async def mock_fetch_all(query, params):
            return []

        mock_sqlite.fetch_one.side_effect = mock_fetch_one
        mock_sqlite.fetch_all.side_effect = mock_fetch_all

        report = await consolidation_engine.prune_stale()

        assert isinstance(report, PruneReport)
        assert report.episodes_pruned == 0
        assert report.links_pruned == 0
        assert report.total_episodes_before == 100
        assert report.total_episodes_after == 100

    @pytest.mark.asyncio
    async def test_prune_stale_episodes(
        self, consolidation_engine, mock_sqlite, mock_chroma
    ):
        """Test pruning stale episodes."""
        episode_ids = [generate_episode_id() for _ in range(3)]

        # Mock episode counts
        count_calls = [0]

        async def mock_fetch_one(query, params):
            if "COUNT" in query:
                count_calls[0] += 1
                if "episode_links" in query:
                    return {"count": 2}  # 2 links will be pruned
                # Episode counts: before and after
                if count_calls[0] == 1:
                    return {"count": 50}
                return {"count": 47}
            return None

        # Mock episodes to prune
        async def mock_fetch_all(query, params):
            if "SELECT id FROM episodes" in query:
                return [{"id": eid} for eid in episode_ids]
            return []

        mock_sqlite.fetch_one.side_effect = mock_fetch_one
        mock_sqlite.fetch_all.side_effect = mock_fetch_all

        report = await consolidation_engine.prune_stale()

        assert report.episodes_pruned == 3
        assert report.links_pruned == 2
        assert report.total_episodes_before == 50
        assert report.total_episodes_after == 47
        # Should delete from vector store
        assert mock_chroma.delete_episode.call_count == 3

    @pytest.mark.asyncio
    async def test_prune_database_error(self, consolidation_engine, mock_sqlite):
        """Test error handling during pruning."""
        mock_sqlite.fetch_one.side_effect = Exception("Database error")

        with pytest.raises(DatabaseError, match="Failed to prune stale memories"):
            await consolidation_engine.prune_stale()


class TestRunCycle:
    """Test run_cycle method."""

    @pytest.mark.asyncio
    async def test_full_cycle_empty_memory(
        self,
        consolidation_engine,
        mock_sqlite,
        mock_abstraction_generator,
        mock_pattern_detector,
    ):
        """Test full cycle with empty memory."""
        # Mock empty responses
        async def mock_fetch_one(query, params):
            return {"count": 0}

        async def mock_fetch_all(query, params):
            return []

        mock_sqlite.fetch_one.side_effect = mock_fetch_one
        mock_sqlite.fetch_all.side_effect = mock_fetch_all

        report = await consolidation_engine.run_cycle()

        assert isinstance(report, ConsolidationReport)
        assert report.abstractions_created == 0
        assert report.patterns_detected == 0
        assert report.patterns_strengthened == 0
        assert report.conflicts_resolved == 0
        assert report.links_strengthened == 0
        assert report.links_pruned == 0
        assert report.episodes_pruned == 0
        assert report.duration >= 0
        assert "cycle_completed_at" in report.metadata
        assert consolidation_engine.last_cycle is not None

    @pytest.mark.asyncio
    async def test_full_cycle_with_operations(
        self,
        consolidation_engine,
        mock_sqlite,
        mock_abstraction_generator,
        mock_pattern_detector,
    ):
        """Test full cycle with actual operations."""
        # Mock episodes for abstraction
        now = utc_now()
        episodes = [
            {
                "id": generate_episode_id(),
                "level": 0,
                "parent_id": None,
                "content": f"Episode {i}",
                "summary": None,
                "context_description": None,
                "keywords": "[]",
                "tags": "[]",
                "occurred_at": (now - timedelta(hours=i)).isoformat(),
                "recorded_at": (now - timedelta(hours=i)).isoformat(),
                "salience": 0.5,
                "consolidation_strength": 5.0,
                "access_count": 0,
                "last_accessed": None,
                "metadata": "{}",
            }
            for i in range(6)
        ]

        # Mock abstraction creation
        from htma.core.types import Episode

        mock_cluster = [
            Episode(
                id=episodes[0]["id"],
                level=0,
                content=episodes[0]["content"],
                occurred_at=datetime.fromisoformat(episodes[0]["occurred_at"]),
            ),
            Episode(
                id=episodes[1]["id"],
                level=0,
                content=episodes[1]["content"],
                occurred_at=datetime.fromisoformat(episodes[1]["occurred_at"]),
            ),
        ]

        mock_abstraction = Episode(
            id=generate_episode_id(),
            level=1,
            content="Abstraction of cluster",
            occurred_at=now,
        )

        mock_abstraction_generator.cluster_episodes.return_value = [mock_cluster]
        mock_abstraction_generator.generate_abstraction.return_value = mock_abstraction

        # Mock pattern detection
        from htma.core.types import Pattern

        pattern_result = PatternDetectionResult(
            new_patterns=[
                Pattern(
                    id="pat_001",
                    description="Test pattern",
                    pattern_type="behavioral",
                    occurrences=[episodes[0]["id"]],
                )
            ],
            strengthened=[("pat_002", 0.8)],
        )
        mock_pattern_detector.detect_patterns.return_value = pattern_result

        # Mock database responses
        call_sequence = []

        async def mock_fetch_all(query, params):
            call_sequence.append("fetch_all")
            if "level = 0 AND parent_id IS NULL" in query:
                # Episodes for abstraction
                return episodes
            elif "level = 0" in query and "ORDER BY occurred_at DESC" in query:
                # Episodes for pattern detection
                return episodes
            elif "episode_links" in query:
                return []
            return []

        async def mock_fetch_one(query, params):
            call_sequence.append("fetch_one")
            if "COUNT" in query:
                if "episodes" in query:
                    # Episode counts
                    if "WHERE (level = 0" in query:
                        return {"count": 0}  # No episodes to prune
                    return {"count": 10}
                elif "episode_links" in query:
                    return {"count": 5}
            return None

        mock_sqlite.fetch_all.side_effect = mock_fetch_all
        mock_sqlite.fetch_one.side_effect = mock_fetch_one

        # Mock episodic.add_episode
        consolidation_engine.episodic.add_episode = AsyncMock()

        report = await consolidation_engine.run_cycle()

        assert isinstance(report, ConsolidationReport)
        assert report.abstractions_created > 0
        assert report.patterns_detected == 1
        assert report.patterns_strengthened == 1
        assert report.duration >= 0
        assert "config" in report.metadata

    @pytest.mark.asyncio
    async def test_cycle_error_handling(self, consolidation_engine, mock_sqlite):
        """Test error handling during consolidation cycle."""
        mock_sqlite.fetch_all.side_effect = Exception("Database error")

        with pytest.raises(ConsolidationError, match="Consolidation cycle failed"):
            await consolidation_engine.run_cycle()


class TestHelperMethods:
    """Test private helper methods."""

    @pytest.mark.asyncio
    async def test_count_total_episodes(self, consolidation_engine, mock_sqlite):
        """Test counting total episodes."""
        mock_sqlite.fetch_one.return_value = {"count": 123}

        count = await consolidation_engine._count_total_episodes()

        assert count == 123

    @pytest.mark.asyncio
    async def test_generate_abstractions(
        self, consolidation_engine, mock_sqlite, mock_abstraction_generator
    ):
        """Test abstraction generation helper."""
        now = utc_now()
        episodes = [
            {
                "id": generate_episode_id(),
                "level": 0,
                "parent_id": None,
                "content": f"Episode {i}",
                "summary": None,
                "context_description": None,
                "keywords": "[]",
                "tags": "[]",
                "occurred_at": (now - timedelta(hours=i)).isoformat(),
                "recorded_at": (now - timedelta(hours=i)).isoformat(),
                "salience": 0.5,
                "consolidation_strength": 5.0,
                "access_count": 0,
                "last_accessed": None,
                "metadata": "{}",
            }
            for i in range(4)
        ]

        mock_sqlite.fetch_all.return_value = episodes

        # Mock cluster and abstraction
        from htma.core.types import Episode

        mock_cluster = [
            Episode(
                id=episodes[0]["id"],
                level=0,
                content=episodes[0]["content"],
                occurred_at=datetime.fromisoformat(episodes[0]["occurred_at"]),
            ),
            Episode(
                id=episodes[1]["id"],
                level=0,
                content=episodes[1]["content"],
                occurred_at=datetime.fromisoformat(episodes[1]["occurred_at"]),
            ),
        ]

        mock_abstraction = Episode(
            id=generate_episode_id(),
            level=1,
            content="Abstraction",
            occurred_at=now,
        )

        mock_abstraction_generator.cluster_episodes.return_value = [mock_cluster]
        mock_abstraction_generator.generate_abstraction.return_value = mock_abstraction

        # Mock episodic.add_episode
        consolidation_engine.episodic.add_episode = AsyncMock()

        count = await consolidation_engine._generate_abstractions()

        assert count == 1
        assert consolidation_engine.episodic.add_episode.called

    @pytest.mark.asyncio
    async def test_detect_patterns(
        self, consolidation_engine, mock_sqlite, mock_pattern_detector
    ):
        """Test pattern detection helper."""
        now = utc_now()
        episodes = [
            {
                "id": generate_episode_id(),
                "level": 0,
                "parent_id": None,
                "content": f"Episode {i}",
                "summary": None,
                "context_description": None,
                "keywords": "[]",
                "tags": "[]",
                "occurred_at": (now - timedelta(hours=i)).isoformat(),
                "recorded_at": (now - timedelta(hours=i)).isoformat(),
                "salience": 0.5,
                "consolidation_strength": 5.0,
                "access_count": 0,
                "last_accessed": None,
                "metadata": "{}",
            }
            for i in range(3)
        ]

        mock_sqlite.fetch_all.return_value = episodes

        result = await consolidation_engine._detect_patterns()

        assert isinstance(result, PatternDetectionResult)
        assert mock_pattern_detector.detect_patterns.called
