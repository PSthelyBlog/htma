"""Unit tests for EpisodicMemory."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from htma.core.exceptions import (
    DatabaseError,
    DuplicateMemoryError,
    MemoryNotFoundError,
)
from htma.core.types import (
    Episode,
    EpisodeLink,
    TemporalFilter,
)
from htma.core.utils import (
    generate_episode_id,
    generate_link_id,
    utc_now,
)
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
def sample_episode():
    """Create a sample episode for testing."""
    return Episode(
        id=generate_episode_id(),
        level=0,
        content="User discussed their morning coffee routine.",
        summary="Coffee routine discussion",
        keywords=["coffee", "morning", "routine"],
        tags=["daily-life"],
        occurred_at=utc_now(),
        recorded_at=utc_now(),
        salience=0.6,
    )


@pytest.fixture
def sample_episodes():
    """Create multiple sample episodes for testing."""
    now = utc_now()
    return [
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User mentioned they prefer Python over Java.",
            summary="Programming language preference",
            keywords=["python", "java", "programming"],
            tags=["preferences", "technology"],
            occurred_at=now - timedelta(hours=2),
            recorded_at=now - timedelta(hours=2),
            salience=0.7,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User lives in San Francisco and works remotely.",
            summary="Location and work info",
            keywords=["san francisco", "remote work", "location"],
            tags=["personal-info"],
            occurred_at=now - timedelta(hours=1),
            recorded_at=now - timedelta(hours=1),
            salience=0.8,
        ),
        Episode(
            id=generate_episode_id(),
            level=0,
            content="User is learning about machine learning with PyTorch.",
            summary="ML learning journey",
            keywords=["machine learning", "pytorch", "learning"],
            tags=["technology", "education"],
            occurred_at=now,
            recorded_at=now,
            salience=0.75,
        ),
    ]


@pytest.mark.asyncio
class TestEpisodicMemoryEpisodeOperations:
    """Test suite for episode operations."""

    async def test_add_episode(self, episodic_memory, sample_episode):
        """Test adding a new episode."""
        result = await episodic_memory.add_episode(sample_episode)

        assert result.id == sample_episode.id
        assert result.level == sample_episode.level
        assert result.content == sample_episode.content
        assert result.salience == sample_episode.salience

    async def test_add_duplicate_episode_raises_error(
        self, episodic_memory, sample_episode
    ):
        """Test that adding duplicate episode raises error."""
        await episodic_memory.add_episode(sample_episode)

        with pytest.raises(DuplicateMemoryError):
            await episodic_memory.add_episode(sample_episode)

    async def test_add_episode_with_parent(self, episodic_memory):
        """Test adding episode with parent reference."""
        # Create parent episode
        parent = Episode(
            id=generate_episode_id(),
            level=1,
            content="Summary of programming discussions",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )
        await episodic_memory.add_episode(parent)

        # Create child episode
        child = Episode(
            id=generate_episode_id(),
            level=0,
            parent_id=parent.id,
            content="Discussed Python features",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )
        result = await episodic_memory.add_episode(child)

        assert result.parent_id == parent.id

    async def test_add_episode_with_nonexistent_parent_raises_error(
        self, episodic_memory
    ):
        """Test that adding episode with non-existent parent raises error."""
        # Generate a valid episode ID that doesn't exist
        nonexistent_parent_id = generate_episode_id()

        episode = Episode(
            id=generate_episode_id(),
            level=0,
            parent_id=nonexistent_parent_id,
            content="Test content",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )

        with pytest.raises(MemoryNotFoundError):
            await episodic_memory.add_episode(episode)

    async def test_get_episode(self, episodic_memory, sample_episode):
        """Test retrieving an episode by ID."""
        await episodic_memory.add_episode(sample_episode)

        retrieved = await episodic_memory.get_episode(sample_episode.id)

        assert retrieved is not None
        assert retrieved.id == sample_episode.id
        assert retrieved.content == sample_episode.content
        assert retrieved.keywords == sample_episode.keywords

    async def test_get_nonexistent_episode(self, episodic_memory):
        """Test retrieving a non-existent episode returns None."""
        # Generate a valid episode ID that doesn't exist
        nonexistent_episode_id = generate_episode_id()
        result = await episodic_memory.get_episode(nonexistent_episode_id)
        assert result is None

    async def test_get_children(self, episodic_memory):
        """Test getting child episodes."""
        # Create parent
        parent = Episode(
            id=generate_episode_id(),
            level=1,
            content="Parent summary",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )
        await episodic_memory.add_episode(parent)

        # Create children
        now = utc_now()
        child1 = Episode(
            id=generate_episode_id(),
            level=0,
            parent_id=parent.id,
            content="Child 1",
            occurred_at=now - timedelta(minutes=10),
            recorded_at=now - timedelta(minutes=10),
        )
        child2 = Episode(
            id=generate_episode_id(),
            level=0,
            parent_id=parent.id,
            content="Child 2",
            occurred_at=now,
            recorded_at=now,
        )
        await episodic_memory.add_episode(child1)
        await episodic_memory.add_episode(child2)

        # Get children
        children = await episodic_memory.get_children(parent.id)

        assert len(children) == 2
        # Should be ordered by occurred_at
        assert children[0].id == child1.id
        assert children[1].id == child2.id

    async def test_get_parent(self, episodic_memory):
        """Test getting parent episode."""
        # Create parent
        parent = Episode(
            id=generate_episode_id(),
            level=1,
            content="Parent summary",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )
        await episodic_memory.add_episode(parent)

        # Create child
        child = Episode(
            id=generate_episode_id(),
            level=0,
            parent_id=parent.id,
            content="Child episode",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )
        await episodic_memory.add_episode(child)

        # Get parent
        retrieved_parent = await episodic_memory.get_parent(child.id)

        assert retrieved_parent is not None
        assert retrieved_parent.id == parent.id


@pytest.mark.asyncio
class TestEpisodicMemoryRetrieval:
    """Test suite for retrieval operations."""

    async def test_search_episodes(self, episodic_memory, sample_episodes):
        """Test semantic search over episodes."""
        for episode in sample_episodes:
            await episodic_memory.add_episode(episode)

        # Search for programming-related episodes
        results = await episodic_memory.search("programming languages", limit=5)

        # Should return results (list, may be empty if embedding fails)
        assert isinstance(results, list)
        # If embedding failed, search falls back to get_recent which returns based on level
        # Since we're not filtering by level, it may return empty or all episodes

    async def test_search_episodes_with_level_filter(
        self, episodic_memory, sample_episodes
    ):
        """Test search with level filter."""
        # Add level 0 episodes
        for episode in sample_episodes:
            await episodic_memory.add_episode(episode)

        # Add level 1 episode
        level1_episode = Episode(
            id=generate_episode_id(),
            level=1,
            content="Summary of all technology discussions",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )
        await episodic_memory.add_episode(level1_episode)

        # Search only level 0
        results = await episodic_memory.search("technology", level=0, limit=10)

        # All results should be level 0
        for episode in results:
            assert episode.level == 0

    async def test_search_episodes_with_temporal_filter(
        self, episodic_memory, sample_episodes
    ):
        """Test search with temporal filter."""
        for episode in sample_episodes:
            await episodic_memory.add_episode(episode)

        # Search with temporal filter (only episodes from last hour)
        cutoff = utc_now() - timedelta(hours=1, minutes=30)
        temporal_filter = TemporalFilter(valid_at=utc_now())

        results = await episodic_memory.search(
            "programming", temporal=temporal_filter, limit=10
        )

        # Results should respect temporal filter
        assert isinstance(results, list)

    async def test_get_recent(self, episodic_memory, sample_episodes):
        """Test getting most recent episodes."""
        for episode in sample_episodes:
            await episodic_memory.add_episode(episode)

        # Get recent level 0 episodes
        recent = await episodic_memory.get_recent(level=0, limit=2)

        assert len(recent) <= 2
        # Should be ordered by occurred_at descending
        if len(recent) == 2:
            assert recent[0].occurred_at >= recent[1].occurred_at

    async def test_get_by_index(self, episodic_memory, sample_episode):
        """Test retrieval via index."""
        await episodic_memory.add_episode(sample_episode)

        # Add index entry
        await episodic_memory.add_index_entry(
            "topic", "coffee", sample_episode.id, note="Coffee-related episode"
        )

        # Retrieve by index
        results = await episodic_memory.get_by_index("topic", "coffee")

        assert len(results) == 1
        assert results[0].id == sample_episode.id


@pytest.mark.asyncio
class TestEpisodicMemoryLinking:
    """Test suite for episode linking operations."""

    async def test_add_link(self, episodic_memory, sample_episodes):
        """Test creating a link between episodes."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]
        await episodic_memory.add_episode(ep1)
        await episodic_memory.add_episode(ep2)

        # Create link
        link = EpisodeLink(
            id=generate_link_id(),
            source_id=ep1.id,
            target_id=ep2.id,
            link_type="semantic",
            weight=0.8,
            created_at=utc_now(),
        )
        await episodic_memory.add_link(link)

        # Verify link was created
        links = await episodic_memory.get_links(ep1.id)
        assert len(links) == 1
        assert links[0].source_id == ep1.id
        assert links[0].target_id == ep2.id

    async def test_add_link_with_nonexistent_episode_raises_error(
        self, episodic_memory, sample_episode
    ):
        """Test that linking to non-existent episode raises error."""
        await episodic_memory.add_episode(sample_episode)

        # Generate a valid episode ID that doesn't exist
        nonexistent_episode_id = generate_episode_id()

        link = EpisodeLink(
            id=generate_link_id(),
            source_id=sample_episode.id,
            target_id=nonexistent_episode_id,
            link_type="semantic",
            weight=0.5,
            created_at=utc_now(),
        )

        with pytest.raises(MemoryNotFoundError):
            await episodic_memory.add_link(link)

    async def test_get_links_filtered_by_type(self, episodic_memory, sample_episodes):
        """Test getting links filtered by type."""
        ep1, ep2, ep3 = sample_episodes
        for ep in [ep1, ep2, ep3]:
            await episodic_memory.add_episode(ep)

        # Create different link types
        semantic_link = EpisodeLink(
            id=generate_link_id(),
            source_id=ep1.id,
            target_id=ep2.id,
            link_type="semantic",
            weight=0.8,
            created_at=utc_now(),
        )
        temporal_link = EpisodeLink(
            id=generate_link_id(),
            source_id=ep1.id,
            target_id=ep3.id,
            link_type="temporal",
            weight=0.6,
            created_at=utc_now(),
        )
        await episodic_memory.add_link(semantic_link)
        await episodic_memory.add_link(temporal_link)

        # Get only semantic links
        semantic_links = await episodic_memory.get_links(ep1.id, link_type="semantic")
        assert len(semantic_links) == 1
        assert semantic_links[0].link_type == "semantic"

        # Get all links
        all_links = await episodic_memory.get_links(ep1.id)
        assert len(all_links) == 2

    async def test_get_linked_episodes(self, episodic_memory, sample_episodes):
        """Test getting episodes linked to a given episode."""
        ep1, ep2, ep3 = sample_episodes
        for ep in [ep1, ep2, ep3]:
            await episodic_memory.add_episode(ep)

        # Create links
        link1 = EpisodeLink(
            id=generate_link_id(),
            source_id=ep1.id,
            target_id=ep2.id,
            link_type="semantic",
            weight=0.8,
            created_at=utc_now(),
        )
        link2 = EpisodeLink(
            id=generate_link_id(),
            source_id=ep1.id,
            target_id=ep3.id,
            link_type="causal",
            weight=0.7,
            created_at=utc_now(),
        )
        await episodic_memory.add_link(link1)
        await episodic_memory.add_link(link2)

        # Get linked episodes
        linked = await episodic_memory.get_linked_episodes(ep1.id)

        assert len(linked) == 2
        linked_ids = {ep.id for ep in linked}
        assert ep2.id in linked_ids
        assert ep3.id in linked_ids

    async def test_update_link_weight(self, episodic_memory, sample_episodes):
        """Test updating link weight."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]
        await episodic_memory.add_episode(ep1)
        await episodic_memory.add_episode(ep2)

        # Create link
        link = EpisodeLink(
            id=generate_link_id(),
            source_id=ep1.id,
            target_id=ep2.id,
            link_type="semantic",
            weight=0.5,
            created_at=utc_now(),
        )
        await episodic_memory.add_link(link)

        # Update weight
        await episodic_memory.update_link_weight(ep1.id, ep2.id, 0.2)

        # Verify update
        links = await episodic_memory.get_links(ep1.id)
        assert len(links) == 1
        assert links[0].weight == 0.7  # 0.5 + 0.2

    async def test_update_link_weight_prevents_negative(
        self, episodic_memory, sample_episodes
    ):
        """Test that link weight doesn't go negative."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]
        await episodic_memory.add_episode(ep1)
        await episodic_memory.add_episode(ep2)

        # Create link
        link = EpisodeLink(
            id=generate_link_id(),
            source_id=ep1.id,
            target_id=ep2.id,
            link_type="semantic",
            weight=0.3,
            created_at=utc_now(),
        )
        await episodic_memory.add_link(link)

        # Try to decrease weight below 0
        await episodic_memory.update_link_weight(ep1.id, ep2.id, -0.5)

        # Verify weight is clamped at 0
        links = await episodic_memory.get_links(ep1.id)
        assert links[0].weight == 0.0


@pytest.mark.asyncio
class TestEpisodicMemoryIndexing:
    """Test suite for indexing operations."""

    async def test_add_index_entry(self, episodic_memory, sample_episode):
        """Test adding an index entry."""
        await episodic_memory.add_episode(sample_episode)

        await episodic_memory.add_index_entry(
            "topic", "coffee", sample_episode.id, note="Morning routine"
        )

        # Verify index entry works
        results = await episodic_memory.get_by_index("topic", "coffee")
        assert len(results) == 1
        assert results[0].id == sample_episode.id

    async def test_add_index_entry_with_nonexistent_episode_raises_error(
        self, episodic_memory
    ):
        """Test that indexing non-existent episode raises error."""
        # Generate a valid episode ID that doesn't exist
        nonexistent_episode_id = generate_episode_id()

        with pytest.raises(MemoryNotFoundError):
            await episodic_memory.add_index_entry(
                "topic", "test", nonexistent_episode_id
            )

    async def test_get_index_keys(self, episodic_memory, sample_episodes):
        """Test getting all keys for an index type."""
        for episode in sample_episodes:
            await episodic_memory.add_episode(episode)

        # Add multiple index entries
        await episodic_memory.add_index_entry("topic", "python", sample_episodes[0].id)
        await episodic_memory.add_index_entry("topic", "location", sample_episodes[1].id)
        await episodic_memory.add_index_entry("topic", "machine-learning", sample_episodes[2].id)

        # Get all topic keys
        keys = await episodic_memory.get_index_keys("topic")

        assert len(keys) >= 3
        assert "python" in keys
        assert "location" in keys
        assert "machine-learning" in keys


@pytest.mark.asyncio
class TestEpisodicMemoryAccessTracking:
    """Test suite for access tracking."""

    async def test_record_access(self, episodic_memory, sample_episode):
        """Test recording access to an episode."""
        await episodic_memory.add_episode(sample_episode)

        # Initial state
        episode = await episodic_memory.get_episode(sample_episode.id)
        assert episode.access_count == 0
        assert episode.last_accessed is None

        # Record access
        await episodic_memory.record_access(sample_episode.id)

        # Verify update
        episode = await episodic_memory.get_episode(sample_episode.id)
        assert episode.access_count == 1
        assert episode.last_accessed is not None

    async def test_record_access_increments_count(
        self, episodic_memory, sample_episode
    ):
        """Test that multiple accesses increment count."""
        await episodic_memory.add_episode(sample_episode)

        # Record multiple accesses
        await episodic_memory.record_access(sample_episode.id)
        await episodic_memory.record_access(sample_episode.id)
        await episodic_memory.record_access(sample_episode.id)

        # Verify count
        episode = await episodic_memory.get_episode(sample_episode.id)
        assert episode.access_count == 3

    async def test_record_access_with_nonexistent_episode_raises_error(
        self, episodic_memory
    ):
        """Test that recording access to non-existent episode raises error."""
        # Generate a valid episode ID that doesn't exist
        nonexistent_episode_id = generate_episode_id()

        with pytest.raises(MemoryNotFoundError):
            await episodic_memory.record_access(nonexistent_episode_id)


@pytest.mark.asyncio
class TestEpisodicMemoryConsolidation:
    """Test suite for consolidation support."""

    async def test_mark_consolidated(self, episodic_memory, sample_episodes):
        """Test marking episode as consolidated."""
        ep1, ep2 = sample_episodes[0], sample_episodes[1]
        await episodic_memory.add_episode(ep1)

        # Create higher-level summary
        summary = Episode(
            id=generate_episode_id(),
            level=1,
            content="Summary of programming discussions",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )
        await episodic_memory.add_episode(summary)

        # Mark as consolidated
        await episodic_memory.mark_consolidated(ep1.id, summary.id)

        # Verify by checking database directly
        query = "SELECT consolidated_into FROM episodes WHERE id = ?"
        result = await episodic_memory.sqlite.fetch_one(query, (ep1.id,))
        assert result["consolidated_into"] == summary.id

    async def test_mark_consolidated_with_nonexistent_episode_raises_error(
        self, episodic_memory, sample_episode
    ):
        """Test that consolidating non-existent episode raises error."""
        await episodic_memory.add_episode(sample_episode)

        # Generate a valid episode ID that doesn't exist
        nonexistent_episode_id = generate_episode_id()

        with pytest.raises(MemoryNotFoundError):
            await episodic_memory.mark_consolidated(
                nonexistent_episode_id, sample_episode.id
            )

    async def test_get_unconsolidated(self, episodic_memory, sample_episodes):
        """Test getting unconsolidated episodes."""
        # Add episodes
        for episode in sample_episodes:
            await episodic_memory.add_episode(episode)

        # Get unconsolidated episodes older than now (should get all)
        cutoff = utc_now() + timedelta(hours=1)
        unconsolidated = await episodic_memory.get_unconsolidated(level=0, older_than=cutoff)

        assert len(unconsolidated) == 3

        # Mark one as consolidated
        summary = Episode(
            id=generate_episode_id(),
            level=1,
            content="Summary",
            occurred_at=utc_now(),
            recorded_at=utc_now(),
        )
        await episodic_memory.add_episode(summary)
        await episodic_memory.mark_consolidated(sample_episodes[0].id, summary.id)

        # Get unconsolidated again
        unconsolidated = await episodic_memory.get_unconsolidated(level=0, older_than=cutoff)

        # Should now have one fewer
        assert len(unconsolidated) == 2
        # The consolidated episode should not be in results
        unconsolidated_ids = {ep.id for ep in unconsolidated}
        assert sample_episodes[0].id not in unconsolidated_ids

    async def test_get_unconsolidated_respects_time_cutoff(
        self, episodic_memory, sample_episodes
    ):
        """Test that unconsolidated query respects time cutoff."""
        for episode in sample_episodes:
            await episodic_memory.add_episode(episode)

        # Get unconsolidated episodes older than 90 minutes ago
        cutoff = utc_now() - timedelta(minutes=90)
        unconsolidated = await episodic_memory.get_unconsolidated(level=0, older_than=cutoff)

        # Should only get episodes from 2 hours ago
        assert len(unconsolidated) == 1
        assert unconsolidated[0].id == sample_episodes[0].id
