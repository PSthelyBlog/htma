"""Unit tests for SemanticMemory."""

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
    BiTemporalRecord,
    Entity,
    Fact,
    TemporalFilter,
    TemporalRange,
)
from htma.core.utils import (
    generate_entity_id,
    generate_fact_id,
    utc_now,
)
from htma.memory.semantic import SemanticMemory
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
async def semantic_memory(temp_storage):
    """Create a SemanticMemory instance with temporary storage."""
    sqlite, chroma = temp_storage
    return SemanticMemory(sqlite, chroma)


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return Entity(
        id=generate_entity_id(),
        name="John Doe",
        entity_type="person",
        created_at=utc_now(),
    )


@pytest.fixture
def sample_entities():
    """Create multiple sample entities for testing."""
    now = utc_now()
    return [
        Entity(
            id=generate_entity_id(),
            name="Alice Smith",
            entity_type="person",
            created_at=now,
        ),
        Entity(
            id=generate_entity_id(),
            name="New York",
            entity_type="place",
            created_at=now,
        ),
        Entity(
            id=generate_entity_id(),
            name="Python",
            entity_type="concept",
            created_at=now,
        ),
    ]


@pytest.mark.asyncio
class TestSemanticMemoryEntityOperations:
    """Test suite for entity operations."""

    async def test_add_entity(self, semantic_memory, sample_entity):
        """Test adding a new entity."""
        result = await semantic_memory.add_entity(sample_entity)

        assert result.id == sample_entity.id
        assert result.name == sample_entity.name
        assert result.entity_type == sample_entity.entity_type

    async def test_add_duplicate_entity_raises_error(
        self, semantic_memory, sample_entity
    ):
        """Test that adding duplicate entity raises error."""
        await semantic_memory.add_entity(sample_entity)

        with pytest.raises(DuplicateMemoryError):
            await semantic_memory.add_entity(sample_entity)

    async def test_get_entity(self, semantic_memory, sample_entity):
        """Test retrieving an entity by ID."""
        await semantic_memory.add_entity(sample_entity)

        retrieved = await semantic_memory.get_entity(sample_entity.id)

        assert retrieved is not None
        assert retrieved.id == sample_entity.id
        assert retrieved.name == sample_entity.name
        assert retrieved.entity_type == sample_entity.entity_type

    async def test_get_nonexistent_entity(self, semantic_memory):
        """Test retrieving a non-existent entity returns None."""
        result = await semantic_memory.get_entity("ent_nonexistent")
        assert result is None

    async def test_find_entity_by_name(self, semantic_memory, sample_entities):
        """Test finding entities by name."""
        for entity in sample_entities:
            await semantic_memory.add_entity(entity)

        # Find by exact name
        results = await semantic_memory.find_entity("Alice Smith")
        assert len(results) == 1
        assert results[0].name == "Alice Smith"

        # Find by partial name
        results = await semantic_memory.find_entity("Smith")
        assert len(results) == 1
        assert results[0].name == "Alice Smith"

    async def test_find_entity_by_type(self, semantic_memory, sample_entities):
        """Test finding entities filtered by type."""
        for entity in sample_entities:
            await semantic_memory.add_entity(entity)

        # Find person entities
        results = await semantic_memory.find_entity("", entity_type="person")
        assert len(results) == 1
        assert results[0].entity_type == "person"

        # Find place entities
        results = await semantic_memory.find_entity("", entity_type="place")
        assert len(results) == 1
        assert results[0].entity_type == "place"

    async def test_find_entity_case_insensitive(self, semantic_memory, sample_entity):
        """Test that name search is case-insensitive."""
        await semantic_memory.add_entity(sample_entity)

        results = await semantic_memory.find_entity("john doe")
        assert len(results) == 1
        assert results[0].name == "John Doe"

        results = await semantic_memory.find_entity("JOHN DOE")
        assert len(results) == 1

    async def test_search_entities_semantic(self, semantic_memory, sample_entities):
        """Test semantic search over entities."""
        for entity in sample_entities:
            await semantic_memory.add_entity(entity)

        # Search for entities
        results = await semantic_memory.search_entities("programming language", limit=5)

        # Should return results (exact matches depend on embedding model)
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestSemanticMemoryFactOperations:
    """Test suite for fact operations."""

    async def test_add_fact(self, semantic_memory, sample_entities):
        """Test adding a new fact."""
        # Add entities first
        person = sample_entities[0]
        place = sample_entities[1]
        await semantic_memory.add_entity(person)
        await semantic_memory.add_entity(place)

        # Create fact
        fact = Fact(
            id=generate_fact_id(),
            subject_id=person.id,
            predicate="lives_in",
            object_id=place.id,
        )

        result = await semantic_memory.add_fact(fact)

        assert result.id == fact.id
        assert result.subject_id == person.id
        assert result.object_id == place.id

    async def test_add_fact_with_value(self, semantic_memory, sample_entity):
        """Test adding a fact with object_value instead of object_id."""
        await semantic_memory.add_entity(sample_entity)

        fact = Fact(
            id=generate_fact_id(),
            subject_id=sample_entity.id,
            predicate="age",
            object_value="30",
        )

        result = await semantic_memory.add_fact(fact)

        assert result.object_value == "30"
        assert result.object_id is None

    async def test_add_fact_nonexistent_subject_raises_error(
        self, semantic_memory
    ):
        """Test that adding fact with non-existent subject raises error."""
        # Use a valid UUID format that doesn't exist in database
        nonexistent_id = "ent_00000000-0000-0000-0000-000000000000"
        fact = Fact(
            id=generate_fact_id(),
            subject_id=nonexistent_id,
            predicate="test",
            object_value="value",
        )

        with pytest.raises(MemoryNotFoundError):
            await semantic_memory.add_fact(fact)

    async def test_add_fact_nonexistent_object_raises_error(
        self, semantic_memory, sample_entity
    ):
        """Test that adding fact with non-existent object raises error."""
        await semantic_memory.add_entity(sample_entity)

        # Use a valid UUID format that doesn't exist in database
        nonexistent_id = "ent_00000000-0000-0000-0000-000000000000"
        fact = Fact(
            id=generate_fact_id(),
            subject_id=sample_entity.id,
            predicate="related_to",
            object_id=nonexistent_id,
        )

        with pytest.raises(MemoryNotFoundError):
            await semantic_memory.add_fact(fact)

    async def test_invalidate_fact(self, semantic_memory, sample_entities):
        """Test invalidating a fact."""
        # Add entities and fact
        person = sample_entities[0]
        place = sample_entities[1]
        await semantic_memory.add_entity(person)
        await semantic_memory.add_entity(place)

        fact = Fact(
            id=generate_fact_id(),
            subject_id=person.id,
            predicate="lives_in",
            object_id=place.id,
        )
        await semantic_memory.add_fact(fact)

        # Invalidate the fact
        invalidation_time = utc_now()
        await semantic_memory.invalidate_fact(fact.id, invalidation_time)

        # Verify fact is invalidated (should not appear in default queries)
        facts = await semantic_memory.query_entity_facts(person.id)
        assert len(facts) == 0

    async def test_invalidate_nonexistent_fact_raises_error(self, semantic_memory):
        """Test that invalidating non-existent fact raises error."""
        # Use a valid UUID format that doesn't exist in database
        nonexistent_id = "fct_00000000-0000-0000-0000-000000000000"
        with pytest.raises(MemoryNotFoundError):
            await semantic_memory.invalidate_fact(nonexistent_id, utc_now())


@pytest.mark.asyncio
class TestSemanticMemoryTemporalQueries:
    """Test suite for bi-temporal queries."""

    async def test_query_entity_facts_no_filter(
        self, semantic_memory, sample_entities
    ):
        """Test querying facts without temporal filter."""
        person = sample_entities[0]
        await semantic_memory.add_entity(person)

        # Add multiple facts
        fact1 = Fact(
            id=generate_fact_id(),
            subject_id=person.id,
            predicate="age",
            object_value="25",
        )
        fact2 = Fact(
            id=generate_fact_id(),
            subject_id=person.id,
            predicate="occupation",
            object_value="engineer",
        )

        await semantic_memory.add_fact(fact1)
        await semantic_memory.add_fact(fact2)

        # Query all facts
        facts = await semantic_memory.query_entity_facts(person.id)

        assert len(facts) == 2
        predicates = {f.predicate for f in facts}
        assert "age" in predicates
        assert "occupation" in predicates

    async def test_query_entity_facts_by_predicate(
        self, semantic_memory, sample_entity
    ):
        """Test querying facts filtered by predicate."""
        await semantic_memory.add_entity(sample_entity)

        # Add facts with different predicates
        fact1 = Fact(
            id=generate_fact_id(),
            subject_id=sample_entity.id,
            predicate="age",
            object_value="25",
        )
        fact2 = Fact(
            id=generate_fact_id(),
            subject_id=sample_entity.id,
            predicate="occupation",
            object_value="engineer",
        )

        await semantic_memory.add_fact(fact1)
        await semantic_memory.add_fact(fact2)

        # Query by predicate
        facts = await semantic_memory.query_entity_facts(
            sample_entity.id, predicate="age"
        )

        assert len(facts) == 1
        assert facts[0].predicate == "age"
        assert facts[0].object_value == "25"

    async def test_query_at_time_transaction(self, semantic_memory, sample_entity):
        """Test querying what we knew at a specific transaction time."""
        await semantic_memory.add_entity(sample_entity)

        # Record time before adding facts
        time_before = utc_now()

        # Wait a moment to ensure time difference
        import asyncio
        await asyncio.sleep(0.1)

        # Add a fact
        fact = Fact(
            id=generate_fact_id(),
            subject_id=sample_entity.id,
            predicate="status",
            object_value="active",
        )
        await semantic_memory.add_fact(fact)

        time_after = utc_now()

        # Query before fact was added
        facts_before = await semantic_memory.query_at_time(
            sample_entity.id, time_before
        )
        assert len(facts_before) == 0

        # Query after fact was added
        facts_after = await semantic_memory.query_at_time(sample_entity.id, time_after)
        assert len(facts_after) == 1
        assert facts_after[0].predicate == "status"

    async def test_query_valid_at_event_time(self, semantic_memory, sample_entity):
        """Test querying what was true at a specific event time."""
        await semantic_memory.add_entity(sample_entity)

        # Create fact valid for a specific time range
        start_time = utc_now() - timedelta(days=30)
        end_time = utc_now() - timedelta(days=10)

        fact = Fact(
            id=generate_fact_id(),
            subject_id=sample_entity.id,
            predicate="location",
            object_value="New York",
            temporal=BiTemporalRecord(
                event_time=TemporalRange(valid_from=start_time, valid_to=end_time)
            ),
        )
        await semantic_memory.add_fact(fact)

        # Query during valid period
        query_time = utc_now() - timedelta(days=20)
        facts_valid = await semantic_memory.query_valid_at(sample_entity.id, query_time)
        assert len(facts_valid) == 1

        # Query before valid period
        query_time_before = utc_now() - timedelta(days=40)
        facts_before = await semantic_memory.query_valid_at(
            sample_entity.id, query_time_before
        )
        assert len(facts_before) == 0

        # Query after valid period
        query_time_after = utc_now() - timedelta(days=5)
        facts_after = await semantic_memory.query_valid_at(
            sample_entity.id, query_time_after
        )
        assert len(facts_after) == 0

    async def test_get_fact_history(self, semantic_memory, sample_entity):
        """Test retrieving complete fact history including invalidated facts."""
        await semantic_memory.add_entity(sample_entity)

        # Add initial fact
        fact1 = Fact(
            id=generate_fact_id(),
            subject_id=sample_entity.id,
            predicate="age",
            object_value="25",
        )
        await semantic_memory.add_fact(fact1)

        # Wait a bit
        import asyncio
        await asyncio.sleep(0.1)

        # Invalidate first fact
        await semantic_memory.invalidate_fact(fact1.id, utc_now())

        # Add new fact
        fact2 = Fact(
            id=generate_fact_id(),
            subject_id=sample_entity.id,
            predicate="age",
            object_value="26",
        )
        await semantic_memory.add_fact(fact2)

        # Get history
        history = await semantic_memory.get_fact_history(sample_entity.id, "age")

        assert len(history) == 2
        # Should be ordered by recorded_at
        assert history[0].object_value == "25"
        assert history[1].object_value == "26"


@pytest.mark.asyncio
class TestSemanticMemoryAccessTracking:
    """Test suite for access tracking."""

    async def test_record_access(self, semantic_memory, sample_entity):
        """Test recording entity access."""
        await semantic_memory.add_entity(sample_entity)

        # Initial access count should be 0
        entity = await semantic_memory.get_entity(sample_entity.id)
        assert entity.access_count == 0
        assert entity.last_accessed is None

        # Record access
        await semantic_memory.record_access(sample_entity.id)

        # Check updated values
        entity = await semantic_memory.get_entity(sample_entity.id)
        assert entity.access_count == 1
        assert entity.last_accessed is not None

        # Record another access
        await semantic_memory.record_access(sample_entity.id)

        entity = await semantic_memory.get_entity(sample_entity.id)
        assert entity.access_count == 2

    async def test_record_access_nonexistent_entity_raises_error(
        self, semantic_memory
    ):
        """Test that recording access for non-existent entity raises error."""
        with pytest.raises(MemoryNotFoundError):
            await semantic_memory.record_access("ent_nonexistent")


@pytest.mark.asyncio
class TestSemanticMemoryCommunityOperations:
    """Test suite for community operations."""

    async def test_add_to_community(self, semantic_memory, sample_entity, temp_storage):
        """Test adding entity to a community."""
        sqlite, _ = temp_storage
        await semantic_memory.add_entity(sample_entity)

        # Create a community first
        community_id = "com_test123"
        await sqlite.execute(
            """
            INSERT INTO communities (id, name, description, entity_ids)
            VALUES (?, ?, ?, ?)
            """,
            (community_id, "Test Community", "A test community", "[]"),
        )

        # Add entity to community
        await semantic_memory.add_to_community(sample_entity.id, community_id)

        # Verify entity is in community
        entities = await semantic_memory.get_community_entities(community_id)
        assert len(entities) == 1
        assert entities[0].id == sample_entity.id

    async def test_add_to_nonexistent_community_raises_error(
        self, semantic_memory, sample_entity
    ):
        """Test that adding to non-existent community raises error."""
        await semantic_memory.add_entity(sample_entity)

        with pytest.raises(MemoryNotFoundError):
            await semantic_memory.add_to_community(sample_entity.id, "com_nonexistent")

    async def test_get_community_entities(
        self, semantic_memory, sample_entities, temp_storage
    ):
        """Test retrieving all entities in a community."""
        sqlite, _ = temp_storage

        # Add entities
        for entity in sample_entities:
            await semantic_memory.add_entity(entity)

        # Create community with entities
        community_id = "com_test456"
        entity_ids = [e.id for e in sample_entities[:2]]  # Add first two entities

        await sqlite.execute(
            """
            INSERT INTO communities (id, name, entity_ids)
            VALUES (?, ?, ?)
            """,
            (community_id, "Test Community", str(entity_ids).replace("'", '"')),
        )

        # Get community entities
        entities = await semantic_memory.get_community_entities(community_id)
        assert len(entities) == 2

        entity_names = {e.name for e in entities}
        assert "Alice Smith" in entity_names
        assert "New York" in entity_names

    async def test_get_nonexistent_community_raises_error(self, semantic_memory):
        """Test that getting non-existent community raises error."""
        with pytest.raises(MemoryNotFoundError):
            await semantic_memory.get_community_entities("com_nonexistent")
