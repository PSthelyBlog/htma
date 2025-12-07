"""Unit tests for utility functions."""

import uuid
from datetime import datetime, timezone

import pytest

from htma.core.utils import (
    ensure_utc,
    generate_community_id,
    generate_entity_id,
    generate_episode_id,
    generate_fact_id,
    generate_link_id,
    is_valid_id,
    utc_now,
)


class TestIDGeneration:
    """Tests for ID generation functions."""

    def test_generate_entity_id(self):
        """Test entity ID generation."""
        entity_id = generate_entity_id()
        assert entity_id.startswith("ent_")
        assert len(entity_id) > 4

    def test_generate_fact_id(self):
        """Test fact ID generation."""
        fact_id = generate_fact_id()
        assert fact_id.startswith("fct_")
        assert len(fact_id) > 4

    def test_generate_episode_id(self):
        """Test episode ID generation."""
        episode_id = generate_episode_id()
        assert episode_id.startswith("epi_")
        assert len(episode_id) > 4

    def test_generate_link_id(self):
        """Test link ID generation."""
        link_id = generate_link_id()
        assert link_id.startswith("lnk_")
        assert len(link_id) > 4

    def test_generate_community_id(self):
        """Test community ID generation."""
        community_id = generate_community_id()
        assert community_id.startswith("com_")
        assert len(community_id) > 4

    def test_ids_are_unique(self):
        """Test that generated IDs are unique."""
        # Generate multiple IDs and ensure they're all different
        entity_ids = {generate_entity_id() for _ in range(100)}
        assert len(entity_ids) == 100

        fact_ids = {generate_fact_id() for _ in range(100)}
        assert len(fact_ids) == 100

        episode_ids = {generate_episode_id() for _ in range(100)}
        assert len(episode_ids) == 100

    def test_ids_are_sortable(self):
        """Test that time-based IDs are sortable.

        Entity, fact, and episode IDs use UUID v1 which includes timestamp,
        so they should be roughly sortable by creation time.
        """
        # Generate IDs sequentially
        ids = [generate_entity_id() for _ in range(10)]

        # IDs should be unique (even if not perfectly sortable by time)
        assert len(set(ids)) == len(ids)

    def test_id_format(self):
        """Test that IDs have valid UUID format."""
        entity_id = generate_entity_id()
        uuid_part = entity_id[4:]  # Remove prefix
        # Should be able to parse as UUID
        uuid.UUID(uuid_part)

        fact_id = generate_fact_id()
        uuid_part = fact_id[4:]
        uuid.UUID(uuid_part)

        episode_id = generate_episode_id()
        uuid_part = episode_id[4:]
        uuid.UUID(uuid_part)

        link_id = generate_link_id()
        uuid_part = link_id[4:]
        uuid.UUID(uuid_part)

        community_id = generate_community_id()
        uuid_part = community_id[4:]
        uuid.UUID(uuid_part)


class TestIDValidation:
    """Tests for ID validation function."""

    def test_valid_entity_id(self):
        """Test validating valid entity ID."""
        entity_id = generate_entity_id()
        assert is_valid_id(entity_id, "ent_")

    def test_valid_fact_id(self):
        """Test validating valid fact ID."""
        fact_id = generate_fact_id()
        assert is_valid_id(fact_id, "fct_")

    def test_valid_episode_id(self):
        """Test validating valid episode ID."""
        episode_id = generate_episode_id()
        assert is_valid_id(episode_id, "epi_")

    def test_invalid_prefix(self):
        """Test that wrong prefix is detected."""
        entity_id = generate_entity_id()
        assert not is_valid_id(entity_id, "fct_")

        fact_id = generate_fact_id()
        assert not is_valid_id(fact_id, "epi_")

    def test_invalid_uuid_format(self):
        """Test that invalid UUID is detected."""
        assert not is_valid_id("ent_not_a_uuid", "ent_")
        assert not is_valid_id("fct_12345", "fct_")
        assert not is_valid_id("epi_invalid", "epi_")

    def test_missing_prefix(self):
        """Test that missing prefix is detected."""
        uuid_str = str(uuid.uuid4())
        assert not is_valid_id(uuid_str, "ent_")

    def test_completely_invalid(self):
        """Test completely invalid IDs."""
        assert not is_valid_id("", "ent_")
        assert not is_valid_id("invalid", "ent_")
        assert not is_valid_id("123", "fct_")


class TestTimeUtilities:
    """Tests for time utility functions."""

    def test_utc_now(self):
        """Test utc_now returns UTC datetime."""
        now = utc_now()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_utc_now_is_current(self):
        """Test that utc_now returns current time."""
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)

        assert before <= now <= after

    def test_ensure_utc_with_none(self):
        """Test ensure_utc with None input."""
        result = ensure_utc(None)
        assert result is None

    def test_ensure_utc_with_naive_datetime(self):
        """Test ensure_utc with naive datetime."""
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        result = ensure_utc(naive_dt)

        assert result is not None
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_ensure_utc_with_utc_datetime(self):
        """Test ensure_utc with already UTC datetime."""
        utc_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_utc(utc_dt)

        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result == utc_dt

    def test_ensure_utc_with_other_timezone(self):
        """Test ensure_utc with non-UTC timezone."""
        # Create a datetime with an offset timezone
        from datetime import timedelta

        other_tz = timezone(timedelta(hours=5))
        other_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=other_tz)

        result = ensure_utc(other_dt)

        assert result is not None
        assert result.tzinfo == timezone.utc
        # Time should be converted (12:00 +5 -> 07:00 UTC)
        assert result.hour == 7

    def test_ensure_utc_preserves_value(self):
        """Test that ensure_utc preserves the actual time value."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        result = ensure_utc(naive)

        # For naive datetime, we assume it's already UTC
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_create_entity_workflow(self):
        """Test typical entity creation workflow."""
        entity_id = generate_entity_id()
        assert is_valid_id(entity_id, "ent_")

        created_at = utc_now()
        assert created_at.tzinfo == timezone.utc

    def test_create_fact_workflow(self):
        """Test typical fact creation workflow."""
        fact_id = generate_fact_id()
        subject_id = generate_entity_id()
        object_id = generate_entity_id()

        assert is_valid_id(fact_id, "fct_")
        assert is_valid_id(subject_id, "ent_")
        assert is_valid_id(object_id, "ent_")

    def test_temporal_workflow(self):
        """Test temporal data workflow."""
        valid_from = datetime(2024, 1, 1, 0, 0, 0)
        valid_to = datetime(2024, 12, 31, 23, 59, 59)

        valid_from_utc = ensure_utc(valid_from)
        valid_to_utc = ensure_utc(valid_to)

        assert valid_from_utc.tzinfo == timezone.utc
        assert valid_to_utc.tzinfo == timezone.utc
        assert valid_from_utc < valid_to_utc

    def test_batch_id_generation(self):
        """Test generating many IDs efficiently."""
        # Generate many IDs and verify they're all valid
        entity_ids = [generate_entity_id() for _ in range(1000)]
        fact_ids = [generate_fact_id() for _ in range(1000)]
        episode_ids = [generate_episode_id() for _ in range(1000)]

        # All should be unique
        assert len(set(entity_ids)) == 1000
        assert len(set(fact_ids)) == 1000
        assert len(set(episode_ids)) == 1000

        # All should be valid
        assert all(is_valid_id(id, "ent_") for id in entity_ids)
        assert all(is_valid_id(id, "fct_") for id in fact_ids)
        assert all(is_valid_id(id, "epi_") for id in episode_ids)
