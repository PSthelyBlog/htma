"""Unit tests for ChromaDB storage."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from htma.core.exceptions import VectorStoreError
from htma.core.types import Entity, Episode
from htma.core.utils import generate_entity_id, generate_episode_id
from htma.storage.chroma import ChromaStorage


def mock_embedding_function(text: str) -> list[float]:
    """Simple mock embedding function for testing.

    Returns a deterministic embedding based on text length and content.
    """
    # Create a simple 384-dimensional embedding (common size)
    # Based on text characteristics for deterministic results
    base_value = len(text) / 1000.0
    return [base_value + (i * 0.001) for i in range(384)]


@pytest.fixture
async def temp_chroma():
    """Create a temporary ChromaDB storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = Path(tmpdir) / "chroma"
        storage = ChromaStorage(
            persist_path=persist_path,
            embedding_function=mock_embedding_function,
        )
        await storage.initialize()
        yield storage


@pytest.fixture
async def temp_chroma_default():
    """Create a temporary ChromaDB storage with default embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = Path(tmpdir) / "chroma"
        storage = ChromaStorage(persist_path=persist_path)
        await storage.initialize()
        yield storage


@pytest.fixture
def sample_episode() -> Episode:
    """Create a sample episode for testing."""
    return Episode(
        id=generate_episode_id(),
        level=0,
        content="This is a test episode about machine learning.",
        summary="Test episode summary",
        keywords=["machine", "learning", "test"],
        tags=["tech", "AI"],
        occurred_at=datetime(2024, 1, 1, 12, 0, 0),
        recorded_at=datetime(2024, 1, 1, 12, 0, 0),
        salience=0.8,
    )


@pytest.fixture
def sample_entity() -> Entity:
    """Create a sample entity for testing."""
    return Entity(
        id=generate_entity_id(),
        name="Python",
        entity_type="concept",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.mark.asyncio
class TestChromaStorage:
    """Test suite for ChromaStorage."""

    async def test_initialization(self):
        """Test ChromaDB initialization creates directory and collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "chroma"
            storage = ChromaStorage(persist_path=persist_path)

            # Directory should not exist yet
            assert not persist_path.exists()

            # Initialize
            await storage.initialize()

            # Directory should now exist
            assert persist_path.exists()
            assert storage.client is not None
            assert storage.episodes_collection is not None
            assert storage.entities_collection is not None

    async def test_collections_created(self, temp_chroma):
        """Test that episodes and entities collections are created."""
        stats = await temp_chroma.get_collection_stats()

        assert "episodes_count" in stats
        assert "entities_count" in stats
        assert stats["episodes_count"] == 0
        assert stats["entities_count"] == 0

    async def test_add_episode(self, temp_chroma, sample_episode):
        """Test adding an episode to the vector store."""
        await temp_chroma.add_episode(sample_episode)

        # Verify episode was added
        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 1

    async def test_add_multiple_episodes(self, temp_chroma):
        """Test adding multiple episodes."""
        episodes = [
            Episode(
                id=generate_episode_id(),
                level=0,
                content=f"Episode {i} content",
                occurred_at=datetime(2024, 1, i + 1, 12, 0, 0),
                recorded_at=datetime(2024, 1, i + 1, 12, 0, 0),
            )
            for i in range(5)
        ]

        for episode in episodes:
            await temp_chroma.add_episode(episode)

        # Verify all episodes were added
        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 5

    async def test_add_entity(self, temp_chroma, sample_entity):
        """Test adding an entity to the vector store."""
        await temp_chroma.add_entity(sample_entity)

        # Verify entity was added
        stats = await temp_chroma.get_collection_stats()
        assert stats["entities_count"] == 1

    async def test_add_multiple_entities(self, temp_chroma):
        """Test adding multiple entities."""
        entities = [
            Entity(
                id=generate_entity_id(),
                name=f"Entity {i}",
                entity_type="concept",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
            )
            for i in range(5)
        ]

        for entity in entities:
            await temp_chroma.add_entity(entity)

        # Verify all entities were added
        stats = await temp_chroma.get_collection_stats()
        assert stats["entities_count"] == 5

    async def test_search_episodes(self, temp_chroma):
        """Test semantic search over episodes."""
        # Add test episodes
        episodes = [
            Episode(
                id=generate_episode_id(),
                level=0,
                content="Python is a programming language",
                occurred_at=datetime(2024, 1, 1, 12, 0, 0),
                recorded_at=datetime(2024, 1, 1, 12, 0, 0),
            ),
            Episode(
                id=generate_episode_id(),
                level=0,
                content="Machine learning with neural networks",
                occurred_at=datetime(2024, 1, 2, 12, 0, 0),
                recorded_at=datetime(2024, 1, 2, 12, 0, 0),
            ),
            Episode(
                id=generate_episode_id(),
                level=0,
                content="Cooking pasta for dinner",
                occurred_at=datetime(2024, 1, 3, 12, 0, 0),
                recorded_at=datetime(2024, 1, 3, 12, 0, 0),
            ),
        ]

        for episode in episodes:
            await temp_chroma.add_episode(episode)

        # Search for programming-related content
        results = await temp_chroma.search_episodes(
            query="programming languages",
            n_results=2,
        )

        # Should return results
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

        # Extract episode IDs
        episode_ids = [r[0] for r in results]

        # Python or ML episode should be in results (both are tech-related)
        assert episodes[0].id in episode_ids or episodes[1].id in episode_ids

    async def test_search_episodes_with_level_filter(self, temp_chroma):
        """Test searching episodes with level filter."""
        # Add episodes at different levels
        episodes = [
            Episode(
                id=generate_episode_id(),
                level=0,
                content="Level 0 raw episode",
                occurred_at=datetime(2024, 1, 1, 12, 0, 0),
                recorded_at=datetime(2024, 1, 1, 12, 0, 0),
            ),
            Episode(
                id=generate_episode_id(),
                level=1,
                content="Level 1 summary episode",
                occurred_at=datetime(2024, 1, 2, 12, 0, 0),
                recorded_at=datetime(2024, 1, 2, 12, 0, 0),
            ),
            Episode(
                id=generate_episode_id(),
                level=0,
                content="Another level 0 episode",
                occurred_at=datetime(2024, 1, 3, 12, 0, 0),
                recorded_at=datetime(2024, 1, 3, 12, 0, 0),
            ),
        ]

        for episode in episodes:
            await temp_chroma.add_episode(episode)

        # Search only level 0 episodes
        results = await temp_chroma.search_episodes(
            query="episode",
            n_results=10,
            level=0,
        )

        # Should only return level 0 episodes
        episode_ids = [r[0] for r in results]
        assert episodes[0].id in episode_ids  # Level 0
        assert episodes[2].id in episode_ids  # Level 0
        assert episodes[1].id not in episode_ids  # Level 1

    async def test_search_entities(self, temp_chroma):
        """Test semantic search over entities."""
        # Add test entities
        entities = [
            Entity(
                id=generate_entity_id(),
                name="Python",
                entity_type="concept",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
            ),
            Entity(
                id=generate_entity_id(),
                name="JavaScript",
                entity_type="concept",
                created_at=datetime(2024, 1, 2, 12, 0, 0),
            ),
            Entity(
                id=generate_entity_id(),
                name="Albert Einstein",
                entity_type="person",
                created_at=datetime(2024, 1, 3, 12, 0, 0),
            ),
        ]

        for entity in entities:
            await temp_chroma.add_entity(entity)

        # Search for programming languages
        results = await temp_chroma.search_entities(
            query="programming language",
            n_results=2,
        )

        # Should return results
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    async def test_search_entities_with_type_filter(self, temp_chroma):
        """Test searching entities with type filter."""
        # Add entities of different types
        entities = [
            Entity(
                id=generate_entity_id(),
                name="John Doe",
                entity_type="person",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
            ),
            Entity(
                id=generate_entity_id(),
                name="Paris",
                entity_type="place",
                created_at=datetime(2024, 1, 2, 12, 0, 0),
            ),
            Entity(
                id=generate_entity_id(),
                name="Jane Smith",
                entity_type="person",
                created_at=datetime(2024, 1, 3, 12, 0, 0),
            ),
        ]

        for entity in entities:
            await temp_chroma.add_entity(entity)

        # Search only for people
        results = await temp_chroma.search_entities(
            query="person",
            n_results=10,
            entity_type="person",
        )

        # Should only return person entities
        entity_ids = [r[0] for r in results]
        assert entities[0].id in entity_ids  # Person
        assert entities[2].id in entity_ids  # Person
        assert entities[1].id not in entity_ids  # Place

    async def test_delete_episode(self, temp_chroma, sample_episode):
        """Test deleting an episode from the vector store."""
        # Add episode
        await temp_chroma.add_episode(sample_episode)

        # Verify it was added
        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 1

        # Delete episode
        await temp_chroma.delete_episode(sample_episode.id)

        # Verify it was deleted
        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 0

    async def test_delete_entity(self, temp_chroma, sample_entity):
        """Test deleting an entity from the vector store."""
        # Add entity
        await temp_chroma.add_entity(sample_entity)

        # Verify it was added
        stats = await temp_chroma.get_collection_stats()
        assert stats["entities_count"] == 1

        # Delete entity
        await temp_chroma.delete_entity(sample_entity.id)

        # Verify it was deleted
        stats = await temp_chroma.get_collection_stats()
        assert stats["entities_count"] == 0

    async def test_update_episode(self, temp_chroma, sample_episode):
        """Test updating an episode in the vector store."""
        # Add episode
        await temp_chroma.add_episode(sample_episode)

        # Update episode content
        sample_episode.content = "Updated content about deep learning"
        sample_episode.keywords = ["deep", "learning", "neural"]

        # Update in vector store
        await temp_chroma.update_episode(sample_episode)

        # Verify still only one episode
        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 1

        # Search with new content
        results = await temp_chroma.search_episodes(
            query="deep learning",
            n_results=1,
        )

        # Should find the updated episode
        assert len(results) > 0
        assert results[0][0] == sample_episode.id

    async def test_update_entity(self, temp_chroma, sample_entity):
        """Test updating an entity in the vector store."""
        # Add entity
        await temp_chroma.add_entity(sample_entity)

        # Update entity
        sample_entity.name = "Python Programming Language"

        # Update in vector store
        await temp_chroma.update_entity(sample_entity)

        # Verify still only one entity
        stats = await temp_chroma.get_collection_stats()
        assert stats["entities_count"] == 1

    async def test_reset(self, temp_chroma, sample_episode, sample_entity):
        """Test resetting the vector store."""
        # Add some data
        await temp_chroma.add_episode(sample_episode)
        await temp_chroma.add_entity(sample_entity)

        # Verify data was added
        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 1
        assert stats["entities_count"] == 1

        # Reset
        await temp_chroma.reset()

        # Verify all data was cleared
        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 0
        assert stats["entities_count"] == 0

    async def test_episode_with_minimal_data(self, temp_chroma):
        """Test adding episode with minimal required fields."""
        minimal_episode = Episode(
            id=generate_episode_id(),
            level=0,
            content="Minimal episode",
            occurred_at=datetime(2024, 1, 1, 12, 0, 0),
            recorded_at=datetime(2024, 1, 1, 12, 0, 0),
        )

        await temp_chroma.add_episode(minimal_episode)

        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 1

    async def test_episode_with_all_fields(self, temp_chroma):
        """Test adding episode with all optional fields."""
        full_episode = Episode(
            id=generate_episode_id(),
            level=1,
            parent_id=generate_episode_id(),
            content="Full episode content",
            summary="Full episode summary",
            context_description="Context description",
            keywords=["key1", "key2", "key3"],
            tags=["tag1", "tag2"],
            occurred_at=datetime(2024, 1, 1, 12, 0, 0),
            recorded_at=datetime(2024, 1, 1, 12, 0, 0),
            salience=0.9,
            consolidation_strength=8.0,
            access_count=5,
            last_accessed=datetime(2024, 1, 2, 12, 0, 0),
            metadata={"custom": "data"},
        )

        await temp_chroma.add_episode(full_episode)

        stats = await temp_chroma.get_collection_stats()
        assert stats["episodes_count"] == 1

    async def test_search_empty_collection(self, temp_chroma):
        """Test searching empty collections returns empty results."""
        episode_results = await temp_chroma.search_episodes(
            query="test query",
            n_results=10,
        )

        entity_results = await temp_chroma.search_entities(
            query="test query",
            n_results=10,
        )

        assert episode_results == []
        assert entity_results == []

    async def test_operations_before_initialization(self):
        """Test that operations fail before initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ChromaStorage(persist_path=Path(tmpdir) / "chroma")

            episode = Episode(
                id=generate_episode_id(),
                level=0,
                content="Test",
                occurred_at=datetime(2024, 1, 1, 12, 0, 0),
                recorded_at=datetime(2024, 1, 1, 12, 0, 0),
            )

            # Should raise error when not initialized
            with pytest.raises(VectorStoreError):
                await storage.add_episode(episode)

    async def test_persistence_across_restarts(self, sample_episode):
        """Test that data persists across storage restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "chroma"

            # Create first instance and add data
            storage1 = ChromaStorage(
                persist_path=persist_path,
                embedding_function=mock_embedding_function,
            )
            await storage1.initialize()
            await storage1.add_episode(sample_episode)

            stats1 = await storage1.get_collection_stats()
            assert stats1["episodes_count"] == 1

            # Create second instance with same path
            storage2 = ChromaStorage(
                persist_path=persist_path,
                embedding_function=mock_embedding_function,
            )
            await storage2.initialize()

            # Data should still be there
            stats2 = await storage2.get_collection_stats()
            assert stats2["episodes_count"] == 1

    @pytest.mark.skip(reason="Requires network access to download default embedding model")
    async def test_with_default_embeddings(self, temp_chroma_default, sample_episode):
        """Test ChromaDB works with default embeddings."""
        await temp_chroma_default.add_episode(sample_episode)

        stats = await temp_chroma_default.get_collection_stats()
        assert stats["episodes_count"] == 1

        # Search should work
        results = await temp_chroma_default.search_episodes(
            query="machine learning",
            n_results=1,
        )

        assert len(results) > 0

    async def test_get_collection_stats(self, temp_chroma, sample_episode, sample_entity):
        """Test getting collection statistics."""
        stats_empty = await temp_chroma.get_collection_stats()
        assert stats_empty["episodes_count"] == 0
        assert stats_empty["entities_count"] == 0
        assert "persist_path" in stats_empty

        # Add data
        await temp_chroma.add_episode(sample_episode)
        await temp_chroma.add_entity(sample_entity)

        stats_filled = await temp_chroma.get_collection_stats()
        assert stats_filled["episodes_count"] == 1
        assert stats_filled["entities_count"] == 1
