"""Unit tests for core data types."""

from datetime import datetime, timezone

import pytest

from htma.core.types import (
    BiTemporalRecord,
    Entity,
    Episode,
    EpisodeLink,
    Fact,
    MemoryNote,
    RetrievalResult,
    TemporalFilter,
    TemporalRange,
)
from htma.core.utils import generate_entity_id, generate_episode_id, generate_fact_id


class TestTemporalRange:
    """Tests for TemporalRange model."""

    def test_create_empty(self):
        """Test creating an empty temporal range."""
        tr = TemporalRange()
        assert tr.valid_from is None
        assert tr.valid_to is None

    def test_create_with_bounds(self):
        """Test creating a temporal range with bounds."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        tr = TemporalRange(valid_from=start, valid_to=end)
        assert tr.valid_from == start
        assert tr.valid_to == end

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)
        tr = TemporalRange(valid_from=start, valid_to=end)

        # Serialize to dict
        data = tr.model_dump()
        assert "valid_from" in data
        assert "valid_to" in data

        # Deserialize from dict
        tr2 = TemporalRange(**data)
        assert tr2.valid_from == tr.valid_from
        assert tr2.valid_to == tr.valid_to


class TestBiTemporalRecord:
    """Tests for BiTemporalRecord model."""

    def test_create_default(self):
        """Test creating with default empty ranges."""
        bt = BiTemporalRecord()
        assert isinstance(bt.event_time, TemporalRange)
        assert isinstance(bt.transaction_time, TemporalRange)
        assert bt.event_time.valid_from is None
        assert bt.transaction_time.valid_from is None

    def test_create_with_ranges(self):
        """Test creating with explicit ranges."""
        event_range = TemporalRange(
            valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        transaction_range = TemporalRange(
            valid_from=datetime(2024, 6, 1, tzinfo=timezone.utc)
        )

        bt = BiTemporalRecord(event_time=event_range, transaction_time=transaction_range)
        assert bt.event_time == event_range
        assert bt.transaction_time == transaction_range

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        bt = BiTemporalRecord(
            event_time=TemporalRange(valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc)),
            transaction_time=TemporalRange(
                valid_from=datetime(2024, 6, 1, tzinfo=timezone.utc)
            ),
        )

        data = bt.model_dump()
        bt2 = BiTemporalRecord(**data)
        assert bt2.event_time.valid_from == bt.event_time.valid_from
        assert bt2.transaction_time.valid_from == bt.transaction_time.valid_from


class TestEntity:
    """Tests for Entity model."""

    def test_create_minimal(self):
        """Test creating entity with minimal required fields."""
        entity_id = generate_entity_id()
        entity = Entity(id=entity_id, name="John Doe", entity_type="person")

        assert entity.id == entity_id
        assert entity.name == "John Doe"
        assert entity.entity_type == "person"
        assert isinstance(entity.created_at, datetime)
        assert entity.last_accessed is None
        assert entity.access_count == 0
        assert entity.metadata == {}

    def test_create_full(self):
        """Test creating entity with all fields."""
        entity_id = generate_entity_id()
        created = datetime(2024, 1, 1, tzinfo=timezone.utc)
        accessed = datetime(2024, 6, 1, tzinfo=timezone.utc)

        entity = Entity(
            id=entity_id,
            name="John Doe",
            entity_type="person",
            created_at=created,
            last_accessed=accessed,
            access_count=5,
            metadata={"occupation": "engineer"},
        )

        assert entity.created_at == created
        assert entity.last_accessed == accessed
        assert entity.access_count == 5
        assert entity.metadata["occupation"] == "engineer"

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        entity = Entity(
            id=generate_entity_id(),
            name="Test Entity",
            entity_type="concept",
            metadata={"key": "value"},
        )

        data = entity.model_dump()
        entity2 = Entity(**data)
        assert entity2.id == entity.id
        assert entity2.name == entity.name
        assert entity2.metadata == entity.metadata


class TestFact:
    """Tests for Fact model."""

    def test_create_entity_relation(self):
        """Test creating fact relating two entities."""
        fact_id = generate_fact_id()
        subject_id = generate_entity_id()
        object_id = generate_entity_id()

        fact = Fact(
            id=fact_id, subject_id=subject_id, predicate="knows", object_id=object_id
        )

        assert fact.id == fact_id
        assert fact.subject_id == subject_id
        assert fact.predicate == "knows"
        assert fact.object_id == object_id
        assert fact.object_value is None
        assert fact.confidence == 1.0

    def test_create_value_property(self):
        """Test creating fact with literal value."""
        fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="age",
            object_value="30",
        )

        assert fact.object_id is None
        assert fact.object_value == "30"

    def test_confidence_bounds(self):
        """Test confidence field validation."""
        # Valid confidence
        fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="test",
            confidence=0.5,
        )
        assert fact.confidence == 0.5

        # Test boundaries
        fact_low = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="test",
            confidence=0.0,
        )
        assert fact_low.confidence == 0.0

        fact_high = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="test",
            confidence=1.0,
        )
        assert fact_high.confidence == 1.0

    def test_bi_temporal(self):
        """Test bi-temporal information."""
        fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="test",
            temporal=BiTemporalRecord(
                event_time=TemporalRange(valid_from=datetime(2024, 1, 1, tzinfo=timezone.utc))
            ),
        )

        assert isinstance(fact.temporal, BiTemporalRecord)
        assert fact.temporal.event_time.valid_from.year == 2024

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="test",
            source_episode_id=generate_episode_id(),
            metadata={"source": "user"},
        )

        data = fact.model_dump()
        fact2 = Fact(**data)
        assert fact2.id == fact.id
        assert fact2.predicate == fact.predicate
        assert fact2.metadata == fact.metadata


class TestEpisode:
    """Tests for Episode model."""

    def test_create_minimal(self):
        """Test creating episode with minimal fields."""
        episode_id = generate_episode_id()
        episode = Episode(id=episode_id, content="User asked about Python")

        assert episode.id == episode_id
        assert episode.content == "User asked about Python"
        assert episode.level == 0
        assert episode.parent_id is None
        assert episode.summary is None
        assert episode.keywords == []
        assert episode.tags == []
        assert episode.salience == 0.5
        assert episode.consolidation_strength == 5.0
        assert episode.access_count == 0

    def test_create_hierarchical(self):
        """Test creating episode in hierarchy."""
        parent_id = generate_episode_id()
        child_id = generate_episode_id()

        child = Episode(
            id=child_id,
            level=1,
            parent_id=parent_id,
            content="Summary of multiple interactions",
            summary="User learned Python basics",
        )

        assert child.level == 1
        assert child.parent_id == parent_id
        assert child.summary is not None

    def test_salience_bounds(self):
        """Test salience field validation."""
        episode = Episode(
            id=generate_episode_id(), content="test", salience=0.8
        )
        assert episode.salience == 0.8

        # Test boundaries
        episode_low = Episode(
            id=generate_episode_id(), content="test", salience=0.0
        )
        assert episode_low.salience == 0.0

        episode_high = Episode(
            id=generate_episode_id(), content="test", salience=1.0
        )
        assert episode_high.salience == 1.0

    def test_with_metadata(self):
        """Test episode with keywords, tags, and metadata."""
        episode = Episode(
            id=generate_episode_id(),
            content="test",
            keywords=["python", "programming"],
            tags=["learning", "tutorial"],
            context_description="Learning Python basics",
            metadata={"session": "123"},
        )

        assert "python" in episode.keywords
        assert "learning" in episode.tags
        assert episode.context_description is not None
        assert episode.metadata["session"] == "123"

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        episode = Episode(
            id=generate_episode_id(),
            content="test content",
            level=1,
            keywords=["test"],
            tags=["tag1"],
            salience=0.7,
        )

        data = episode.model_dump()
        episode2 = Episode(**data)
        assert episode2.id == episode.id
        assert episode2.level == episode.level
        assert episode2.salience == episode.salience


class TestEpisodeLink:
    """Tests for EpisodeLink model."""

    def test_create_link(self):
        """Test creating episode link."""
        source_id = generate_episode_id()
        target_id = generate_episode_id()

        link = EpisodeLink(
            id="lnk_123",
            source_id=source_id,
            target_id=target_id,
            link_type="semantic",
        )

        assert link.source_id == source_id
        assert link.target_id == target_id
        assert link.link_type == "semantic"
        assert link.weight == 1.0
        assert isinstance(link.created_at, datetime)

    def test_link_types(self):
        """Test different link types."""
        for link_type in ["semantic", "temporal", "causal", "analogical"]:
            link = EpisodeLink(
                id=f"lnk_{link_type}",
                source_id=generate_episode_id(),
                target_id=generate_episode_id(),
                link_type=link_type,
            )
            assert link.link_type == link_type

    def test_custom_weight(self):
        """Test link with custom weight."""
        link = EpisodeLink(
            id="lnk_123",
            source_id=generate_episode_id(),
            target_id=generate_episode_id(),
            link_type="semantic",
            weight=0.5,
        )
        assert link.weight == 0.5

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        link = EpisodeLink(
            id="lnk_123",
            source_id=generate_episode_id(),
            target_id=generate_episode_id(),
            link_type="causal",
            weight=0.8,
        )

        data = link.model_dump()
        link2 = EpisodeLink(**data)
        assert link2.id == link.id
        assert link2.link_type == link.link_type
        assert link2.weight == link.weight


class TestMemoryNote:
    """Tests for MemoryNote model."""

    def test_create_note(self):
        """Test creating memory note."""
        note = MemoryNote(
            content="User prefers Python over Java",
            context="Programming language discussion",
            keywords=["python", "preference"],
            tags=["learning"],
            salience=0.7,
        )

        assert note.content == "User prefers Python over Java"
        assert note.context == "Programming language discussion"
        assert "python" in note.keywords
        assert note.salience == 0.7

    def test_default_fields(self):
        """Test default field values."""
        note = MemoryNote(
            content="test", context="test context"
        )

        assert note.keywords == []
        assert note.tags == []
        assert note.salience == 0.5

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        note = MemoryNote(
            content="test",
            context="context",
            keywords=["key1"],
            tags=["tag1"],
            salience=0.6,
        )

        data = note.model_dump()
        note2 = MemoryNote(**data)
        assert note2.content == note.content
        assert note2.salience == note.salience


class TestTemporalFilter:
    """Tests for TemporalFilter model."""

    def test_create_empty(self):
        """Test creating empty filter."""
        tf = TemporalFilter()
        assert tf.as_of is None
        assert tf.valid_at is None

    def test_transaction_time_filter(self):
        """Test transaction time filter."""
        as_of = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tf = TemporalFilter(as_of=as_of)
        assert tf.as_of == as_of
        assert tf.valid_at is None

    def test_event_time_filter(self):
        """Test event time filter."""
        valid_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tf = TemporalFilter(valid_at=valid_at)
        assert tf.as_of is None
        assert tf.valid_at == valid_at

    def test_both_filters(self):
        """Test both filters together."""
        as_of = datetime(2024, 1, 1, tzinfo=timezone.utc)
        valid_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
        tf = TemporalFilter(as_of=as_of, valid_at=valid_at)
        assert tf.as_of == as_of
        assert tf.valid_at == valid_at


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_create_empty(self):
        """Test creating empty result."""
        result = RetrievalResult()
        assert result.facts == []
        assert result.episodes == []
        assert result.relevance_scores == {}

    def test_create_with_data(self):
        """Test creating result with data."""
        fact = Fact(
            id=generate_fact_id(),
            subject_id=generate_entity_id(),
            predicate="test",
            object_value="test",
        )
        episode = Episode(id=generate_episode_id(), content="test")

        result = RetrievalResult(
            facts=[fact],
            episodes=[episode],
            relevance_scores={fact.id: 0.9, episode.id: 0.8},
        )

        assert len(result.facts) == 1
        assert len(result.episodes) == 1
        assert result.relevance_scores[fact.id] == 0.9
        assert result.relevance_scores[episode.id] == 0.8

    def test_serialization(self):
        """Test JSON serialization and deserialization."""
        result = RetrievalResult(
            facts=[],
            episodes=[],
            relevance_scores={"id1": 0.5},
        )

        data = result.model_dump()
        result2 = RetrievalResult(**data)
        assert result2.relevance_scores == result.relevance_scores
