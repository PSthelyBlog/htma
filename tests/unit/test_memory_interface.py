"""Unit tests for memory interface implementation."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from htma.core.types import (
    Entity,
    Episode,
    EpisodeLink,
    Fact,
    Interaction,
    RetrievalResult,
    StorageResult,
    TemporalFilter,
)
from htma.core.utils import generate_entity_id, generate_episode_id, generate_fact_id
from htma.curator.curator import MemoryCurator
from htma.memory.episodic import EpisodicMemory
from htma.memory.interface import MemoryInterface
from htma.memory.semantic import SemanticMemory
from htma.memory.working import MemoryItem, WorkingMemory


@pytest.fixture
def mock_working_memory():
    """Create a mock WorkingMemory."""
    working = MagicMock(spec=WorkingMemory)
    working.task_context = "Test task context"
    working.add_retrieved = MagicMock()
    working.handle_pressure = AsyncMock(return_value=[])
    return working


@pytest.fixture
def mock_semantic_memory():
    """Create a mock SemanticMemory."""
    semantic = MagicMock(spec=SemanticMemory)
    semantic.search_entities = AsyncMock(return_value=[])
    semantic.query_entity_facts = AsyncMock(return_value=[])
    semantic.record_access = AsyncMock()
    semantic.add_entity = AsyncMock()
    semantic.add_fact = AsyncMock()
    semantic.invalidate_fact = AsyncMock()
    return semantic


@pytest.fixture
def mock_episodic_memory():
    """Create a mock EpisodicMemory."""
    episodic = MagicMock(spec=EpisodicMemory)
    episodic.search = AsyncMock(return_value=[])
    episodic.record_access = AsyncMock()
    episodic.add_episode = AsyncMock()
    episodic.add_link = AsyncMock()
    episodic.get_linked_episodes = AsyncMock(return_value=[])
    return episodic


@pytest.fixture
def mock_curator():
    """Create a mock MemoryCurator."""
    from htma.core.types import MemoryNote

    curator = MagicMock(spec=MemoryCurator)

    # Default salience evaluation
    curator.evaluate_salience = AsyncMock(
        return_value=MemoryNote(
            content="Test content",
            context="Test context",
            keywords=["test", "memory"],
            tags=["interaction"],
            salience=0.8,
        )
    )

    curator.extract_entities = AsyncMock(return_value=[])
    curator.extract_facts = AsyncMock(return_value=[])
    curator.generate_links = AsyncMock(return_value=[])
    curator.resolve_conflicts = AsyncMock(
        return_value={
            "invalidations": [],
            "confidence_updates": [],
            "new_facts": [],
        }
    )
    curator.trigger_evolution = AsyncMock(return_value=[])

    return curator


@pytest.fixture
def memory_interface(
    mock_working_memory,
    mock_semantic_memory,
    mock_episodic_memory,
    mock_curator,
):
    """Create a MemoryInterface instance for testing."""
    return MemoryInterface(
        working=mock_working_memory,
        semantic=mock_semantic_memory,
        episodic=mock_episodic_memory,
        curator=mock_curator,
    )


class TestMemoryInterfaceInit:
    """Test MemoryInterface initialization."""

    def test_initialization(self, memory_interface):
        """Test that interface initializes correctly."""
        assert memory_interface.working is not None
        assert memory_interface.semantic is not None
        assert memory_interface.episodic is not None
        assert memory_interface.curator is not None


class TestQuery:
    """Test query operations."""

    @pytest.mark.asyncio
    async def test_query_both_stores(self, memory_interface, mock_semantic_memory, mock_episodic_memory):
        """Test querying both semantic and episodic memory."""
        # Setup mock entities and facts
        entity_id = generate_entity_id()
        entity = Entity(
            id=entity_id,
            name="Test Entity",
            entity_type="concept",
        )
        fact = Fact(
            id=generate_fact_id(),
            subject_id=entity_id,
            predicate="test_predicate",
            object_value="test_value",
        )

        mock_semantic_memory.search_entities.return_value = [entity]
        mock_semantic_memory.query_entity_facts.return_value = [fact]

        # Setup mock episodes
        episode = Episode(
            id=generate_episode_id(),
            content="Test episode content",
            salience=0.7,
        )
        mock_episodic_memory.search.return_value = [episode]

        # Execute query
        result = await memory_interface.query(
            "test query",
            include_semantic=True,
            include_episodic=True,
        )

        # Verify results
        assert len(result.facts) == 1
        assert len(result.episodes) == 1
        assert result.facts[0].id == fact.id
        assert result.episodes[0].id == episode.id

        # Verify access tracking
        mock_semantic_memory.record_access.assert_called_once_with(entity_id)
        mock_episodic_memory.record_access.assert_called_once_with(episode.id)

    @pytest.mark.asyncio
    async def test_query_semantic_only(self, memory_interface, mock_semantic_memory, mock_episodic_memory):
        """Test querying only semantic memory."""
        entity_id = generate_entity_id()
        entity = Entity(id=entity_id, name="Test", entity_type="person")
        mock_semantic_memory.search_entities.return_value = [entity]
        mock_semantic_memory.query_entity_facts.return_value = []

        result = await memory_interface.query(
            "test",
            include_semantic=True,
            include_episodic=False,
        )

        # Should only query semantic, not episodic
        mock_semantic_memory.search_entities.assert_called_once()
        mock_episodic_memory.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_episodic_only(self, memory_interface, mock_semantic_memory, mock_episodic_memory):
        """Test querying only episodic memory."""
        episode = Episode(id=generate_episode_id(), content="Test", salience=0.5)
        mock_episodic_memory.search.return_value = [episode]

        result = await memory_interface.query(
            "test",
            include_semantic=False,
            include_episodic=True,
        )

        # Should only query episodic, not semantic
        mock_episodic_memory.search.assert_called_once()
        mock_semantic_memory.search_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_with_temporal_filter(self, memory_interface, mock_episodic_memory):
        """Test query with temporal filter."""
        temporal_filter = TemporalFilter(
            valid_at=datetime.utcnow() - timedelta(days=7)
        )

        await memory_interface.query(
            "test",
            include_episodic=True,
            temporal=temporal_filter,
        )

        # Verify temporal filter was passed
        call_args = mock_episodic_memory.search.call_args
        assert call_args.kwargs["temporal"] == temporal_filter

    @pytest.mark.asyncio
    async def test_query_semantic_direct(self, memory_interface, mock_semantic_memory):
        """Test direct semantic memory query."""
        entity_id = generate_entity_id()
        entity = Entity(id=entity_id, name="Test", entity_type="person")
        fact = Fact(
            id=generate_fact_id(),
            subject_id=entity_id,
            predicate="test",
            object_value="value",
        )

        mock_semantic_memory.search_entities.return_value = [entity]
        mock_semantic_memory.query_entity_facts.return_value = [fact]

        facts = await memory_interface.query_semantic("test")

        assert len(facts) == 1
        assert facts[0].id == fact.id

    @pytest.mark.asyncio
    async def test_query_episodic_direct(self, memory_interface, mock_episodic_memory):
        """Test direct episodic memory query."""
        episode = Episode(id=generate_episode_id(), content="Test", salience=0.5)
        mock_episodic_memory.search.return_value = [episode]

        episodes = await memory_interface.query_episodic("test", level=0)

        assert len(episodes) == 1
        assert episodes[0].id == episode.id

        # Verify level parameter was passed
        call_args = mock_episodic_memory.search.call_args
        assert call_args.kwargs["level"] == 0


class TestStoreInteraction:
    """Test storing interactions."""

    @pytest.mark.asyncio
    async def test_store_high_salience_interaction(
        self,
        memory_interface,
        mock_curator,
        mock_episodic_memory,
    ):
        """Test storing an interaction with high salience."""
        from htma.core.types import MemoryNote

        # Setup high salience response
        mock_curator.evaluate_salience.return_value = MemoryNote(
            content="Important content",
            context="Important context",
            keywords=["important"],
            tags=["high-priority"],
            salience=0.9,
        )

        interaction = Interaction(
            user_message="Tell me something important",
            assistant_message="Here's important information",
        )

        with patch("htma.memory.interface.generate_episode_id") as mock_gen:
            episode_id = generate_episode_id()
            mock_gen.return_value = episode_id

            result = await memory_interface.store_interaction(interaction)

        # Verify episode was created
        assert result.episode_id == episode_id
        assert result.salience_score == 0.9
        mock_episodic_memory.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_low_salience_interaction(
        self,
        memory_interface,
        mock_curator,
        mock_episodic_memory,
    ):
        """Test that low salience interactions are not stored."""
        from htma.core.types import MemoryNote

        # Setup low salience response
        mock_curator.evaluate_salience.return_value = MemoryNote(
            content="Trivial content",
            context="",
            keywords=[],
            tags=[],
            salience=0.2,
        )

        interaction = Interaction(
            user_message="Hi",
            assistant_message="Hello",
        )

        result = await memory_interface.store_interaction(interaction)

        # Verify no episode was created
        assert result.episode_id is None
        assert result.salience_score == 0.2
        mock_episodic_memory.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_entity_extraction(
        self,
        memory_interface,
        mock_curator,
        mock_semantic_memory,
    ):
        """Test storing interaction with entity extraction."""
        from htma.core.types import MemoryNote

        # Setup entities to extract
        entity = Entity(
            id=generate_entity_id(),
            name="John Doe",
            entity_type="person",
        )
        mock_curator.extract_entities.return_value = [entity]

        # Ensure high salience
        mock_curator.evaluate_salience.return_value = MemoryNote(
            content="Test", context="", keywords=[], tags=[], salience=0.8
        )

        interaction = Interaction(
            user_message="I met John Doe",
            assistant_message="That's interesting",
        )

        result = await memory_interface.store_interaction(interaction)

        # Verify entity was added
        assert len(result.entities_created) == 1
        assert result.entities_created[0] == entity.id
        mock_semantic_memory.add_entity.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_fact_extraction(
        self,
        memory_interface,
        mock_curator,
        mock_semantic_memory,
    ):
        """Test storing interaction with fact extraction."""
        from htma.core.types import MemoryNote

        # Setup facts to extract
        fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="works_at",
            object_value="Acme Corp",
        )
        mock_curator.extract_facts.return_value = [fact]

        # Ensure high salience
        mock_curator.evaluate_salience.return_value = MemoryNote(
            content="Test", context="", keywords=[], tags=[], salience=0.8
        )

        interaction = Interaction(
            user_message="I work at Acme Corp",
            assistant_message="Great!",
        )

        result = await memory_interface.store_interaction(interaction)

        # Verify fact was added
        assert len(result.facts_created) == 1
        assert result.facts_created[0] == fact.id
        mock_semantic_memory.add_fact.assert_called_once()


class TestContextManagement:
    """Test context management operations."""

    @pytest.mark.asyncio
    async def test_inject_context(self, memory_interface, mock_working_memory):
        """Test injecting retrieval results into working memory."""
        # Create retrieval result with facts and episodes
        fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="value",
        )
        episode = Episode(
            id=generate_episode_id(),
            content="Test episode",
            summary="Summary",
            level=1,
            salience=0.8,
        )

        result = RetrievalResult(
            facts=[fact],
            episodes=[episode],
            relevance_scores={
                fact.id: 0.9,
                episode.id: 0.7,
            },
        )

        await memory_interface.inject_context(result)

        # Verify items were added to working memory
        mock_working_memory.add_retrieved.assert_called_once()
        items = mock_working_memory.add_retrieved.call_args[0][0]

        # Should have 2 items (1 fact + 1 episode)
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_handle_memory_pressure(
        self,
        memory_interface,
        mock_working_memory,
        mock_curator,
    ):
        """Test handling memory pressure."""
        from htma.core.types import MemoryNote

        # Setup items to persist
        item_to_persist = MemoryItem(
            content="Important info to persist",
            source="dialog_summary",
            relevance=0.8,
        )
        mock_working_memory.handle_pressure.return_value = [item_to_persist]

        # Setup curator to accept the interaction
        mock_curator.evaluate_salience.return_value = MemoryNote(
            content="Test", context="", keywords=[], tags=[], salience=0.8
        )

        await memory_interface.handle_memory_pressure()

        # Verify pressure was handled
        mock_working_memory.handle_pressure.assert_called_once()

        # Verify salience evaluation was called for persisted item
        mock_curator.evaluate_salience.assert_called()


class TestUtilityMethods:
    """Test utility methods."""

    @pytest.mark.asyncio
    async def test_get_related_context(self, memory_interface, mock_episodic_memory):
        """Test getting related context via episode links."""
        episode_id = generate_episode_id()

        # Setup linked episodes
        related_episode = Episode(
            id=generate_episode_id(),
            content="Related content",
            salience=0.6,
        )
        mock_episodic_memory.get_linked_episodes.return_value = [related_episode]

        # Get related context with depth 1
        related = await memory_interface.get_related_context(episode_id, depth=1)

        assert len(related) == 1
        assert related[0].id == related_episode.id
        mock_episodic_memory.get_linked_episodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_related_context_depth_2(
        self,
        memory_interface,
        mock_episodic_memory,
    ):
        """Test getting related context with depth 2."""
        episode_id = generate_episode_id()

        # Setup two layers of linked episodes
        related_1 = Episode(id=generate_episode_id(), content="Related 1", salience=0.6)
        related_2 = Episode(id=generate_episode_id(), content="Related 2", salience=0.5)

        # First call returns related_1, second call returns related_2
        mock_episodic_memory.get_linked_episodes.side_effect = [
            [related_1],
            [related_2],
        ]

        related = await memory_interface.get_related_context(episode_id, depth=2)

        # Should have 2 episodes from 2 levels
        assert len(related) == 2
        assert mock_episodic_memory.get_linked_episodes.call_count == 2

    def test_deduplicate_results(self, memory_interface):
        """Test deduplication of results."""
        fact_id = generate_fact_id()
        fact = Fact(
            id=fact_id,
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="value",
        )

        # Create result with duplicate facts
        result = RetrievalResult(
            facts=[fact, fact],  # Duplicate
            episodes=[],
        )

        deduplicated = memory_interface._deduplicate_results(result)

        # Should have only one fact
        assert len(deduplicated.facts) == 1

    def test_rank_results(self, memory_interface):
        """Test ranking of results by relevance."""
        fact1 = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test1",
            object_value="value1",
        )
        fact2 = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test2",
            object_value="value2",
        )

        result = RetrievalResult(
            facts=[fact1, fact2],
            relevance_scores={
                fact1.id: 0.3,
                fact2.id: 0.9,
            },
        )

        ranked = memory_interface._rank_results(result)

        # fact2 should be first (higher relevance)
        assert ranked.facts[0].id == fact2.id
        assert ranked.facts[1].id == fact1.id
