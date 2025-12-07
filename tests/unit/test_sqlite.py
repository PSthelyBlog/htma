"""Unit tests for SQLite storage."""

import json
import tempfile
from pathlib import Path

import pytest

from htma.core.exceptions import DatabaseError, MigrationError
from htma.storage.sqlite import SQLiteStorage


@pytest.fixture
async def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteStorage(db_path)
        await storage.initialize()
        yield storage
        await storage.disconnect()


@pytest.mark.asyncio
class TestSQLiteStorage:
    """Test suite for SQLiteStorage."""

    async def test_initialization(self):
        """Test database initialization creates file and runs migrations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = SQLiteStorage(db_path)

            # Database file should not exist yet
            assert not db_path.exists()

            # Initialize
            await storage.initialize()

            # Database file should now exist
            assert db_path.exists()

    async def test_migrations_create_tables(self, temp_db):
        """Test that migrations create all required tables."""
        tables = [
            "entities",
            "facts",
            "episodes",
            "episode_links",
            "retrieval_indices",
            "communities",
        ]

        for table in tables:
            assert await temp_db.table_exists(table), f"Table {table} was not created"

    async def test_execute_insert(self, temp_db):
        """Test executing an INSERT query."""
        await temp_db.execute(
            """
            INSERT INTO entities (id, name, entity_type, metadata)
            VALUES (?, ?, ?, ?)
            """,
            ("ent_test123", "Test Entity", "concept", json.dumps({"key": "value"})),
        )

        # Verify insertion
        result = await temp_db.fetch_one(
            "SELECT * FROM entities WHERE id = ?", ("ent_test123",)
        )

        assert result is not None
        assert result["id"] == "ent_test123"
        assert result["name"] == "Test Entity"
        assert result["entity_type"] == "concept"

    async def test_execute_many(self, temp_db):
        """Test executing multiple inserts at once."""
        entities = [
            ("ent_001", "Entity 1", "person", None),
            ("ent_002", "Entity 2", "place", None),
            ("ent_003", "Entity 3", "concept", None),
        ]

        await temp_db.execute_many(
            "INSERT INTO entities (id, name, entity_type, metadata) VALUES (?, ?, ?, ?)",
            entities,
        )

        # Verify all insertions
        count = await temp_db.count("entities")
        assert count == 3

    async def test_fetch_one(self, temp_db):
        """Test fetching a single row."""
        # Insert test data
        await temp_db.execute(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
            ("ent_test", "Test", "person"),
        )

        # Fetch one
        result = await temp_db.fetch_one(
            "SELECT * FROM entities WHERE id = ?", ("ent_test",)
        )

        assert result is not None
        assert isinstance(result, dict)
        assert result["id"] == "ent_test"
        assert result["name"] == "Test"

    async def test_fetch_one_not_found(self, temp_db):
        """Test fetch_one returns None when no results."""
        result = await temp_db.fetch_one(
            "SELECT * FROM entities WHERE id = ?", ("nonexistent",)
        )

        assert result is None

    async def test_fetch_all(self, temp_db):
        """Test fetching all rows."""
        # Insert multiple entities
        entities = [
            ("ent_001", "Entity 1", "person"),
            ("ent_002", "Entity 2", "place"),
            ("ent_003", "Entity 3", "concept"),
        ]

        await temp_db.execute_many(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
            entities,
        )

        # Fetch all
        results = await temp_db.fetch_all("SELECT * FROM entities ORDER BY id")

        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert results[0]["id"] == "ent_001"
        assert results[1]["id"] == "ent_002"
        assert results[2]["id"] == "ent_003"

    async def test_fetch_all_empty(self, temp_db):
        """Test fetch_all returns empty list when no results."""
        results = await temp_db.fetch_all("SELECT * FROM entities")

        assert results == []

    async def test_count(self, temp_db):
        """Test counting rows in a table."""
        # Initially empty
        count = await temp_db.count("entities")
        assert count == 0

        # Insert entities
        await temp_db.execute_many(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
            [
                ("ent_001", "Entity 1", "person"),
                ("ent_002", "Entity 2", "place"),
            ],
        )

        # Count all
        count = await temp_db.count("entities")
        assert count == 2

    async def test_count_with_where(self, temp_db):
        """Test counting rows with WHERE clause."""
        # Insert entities of different types
        await temp_db.execute_many(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
            [
                ("ent_001", "Entity 1", "person"),
                ("ent_002", "Entity 2", "person"),
                ("ent_003", "Entity 3", "place"),
            ],
        )

        # Count only persons
        count = await temp_db.count("entities", "entity_type = ?", ("person",))
        assert count == 2

        # Count only places
        count = await temp_db.count("entities", "entity_type = ?", ("place",))
        assert count == 1

    async def test_transaction_commit(self, temp_db):
        """Test transaction commits on success."""
        async with temp_db.transaction() as conn:
            await conn.execute(
                "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
                ("ent_001", "Entity 1", "person"),
            )
            await conn.execute(
                "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
                ("ent_002", "Entity 2", "place"),
            )

        # Verify both inserts committed
        count = await temp_db.count("entities")
        assert count == 2

    async def test_transaction_rollback(self, temp_db):
        """Test transaction rolls back on error."""
        # Insert one entity first
        await temp_db.execute(
            "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
            ("ent_001", "Entity 1", "person"),
        )

        # Try to insert with duplicate ID in transaction (should fail)
        with pytest.raises(Exception):
            async with temp_db.transaction() as conn:
                await conn.execute(
                    "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
                    ("ent_002", "Entity 2", "place"),
                )
                # This should fail due to duplicate primary key
                await conn.execute(
                    "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
                    ("ent_001", "Duplicate", "concept"),
                )

        # Only the first entity should exist (transaction rolled back)
        count = await temp_db.count("entities")
        assert count == 1

    async def test_foreign_key_constraints(self, temp_db):
        """Test that foreign key constraints are enforced."""
        # Try to insert a fact referencing non-existent entity
        # This should fail because foreign keys are enabled
        with pytest.raises(DatabaseError):
            await temp_db.execute(
                """
                INSERT INTO facts (id, subject_id, predicate)
                VALUES (?, ?, ?)
                """,
                ("fct_001", "ent_nonexistent", "is_a"),
            )

    async def test_indexes_created(self, temp_db):
        """Test that indexes were created."""
        # Query sqlite_master for indexes
        indexes = await temp_db.fetch_all(
            """
            SELECT name FROM sqlite_master
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
            """
        )

        index_names = {idx["name"] for idx in indexes}

        expected_indexes = {
            "idx_facts_subject",
            "idx_facts_valid_period",
            "idx_facts_recorded",
            "idx_entities_type",
            "idx_episodes_level",
            "idx_episodes_occurred",
            "idx_retrieval_type_key",
        }

        assert expected_indexes.issubset(
            index_names
        ), f"Missing indexes: {expected_indexes - index_names}"

    async def test_json_serialization(self, temp_db):
        """Test JSON field serialization and deserialization."""
        metadata = {"key": "value", "number": 42, "nested": {"inner": "data"}}

        # Insert with JSON metadata
        await temp_db.execute(
            "INSERT INTO entities (id, name, entity_type, metadata) VALUES (?, ?, ?, ?)",
            ("ent_001", "Entity", "concept", temp_db._serialize_json_field(metadata)),
        )

        # Fetch and deserialize
        result = await temp_db.fetch_one(
            "SELECT * FROM entities WHERE id = ?", ("ent_001",)
        )

        assert result is not None
        deserialized = temp_db._deserialize_json_field(result["metadata"])
        assert deserialized == metadata

    async def test_json_serialization_none(self, temp_db):
        """Test JSON field serialization handles None."""
        serialized = temp_db._serialize_json_field(None)
        assert serialized is None

        deserialized = temp_db._deserialize_json_field(None)
        assert deserialized is None

    async def test_connection_context_manager(self, temp_db):
        """Test using connection as context manager."""
        async with temp_db.connection() as conn:
            await conn.execute(
                "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
                ("ent_001", "Entity", "person"),
            )
            await conn.commit()

        # Verify insertion
        count = await temp_db.count("entities")
        assert count == 1

    async def test_table_exists(self, temp_db):
        """Test checking if table exists."""
        assert await temp_db.table_exists("entities")
        assert await temp_db.table_exists("facts")
        assert not await temp_db.table_exists("nonexistent_table")

    async def test_connect_disconnect(self):
        """Test explicit connect and disconnect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = SQLiteStorage(db_path)
            await storage.initialize()

            # Connect
            await storage.connect()
            assert storage._conn is not None

            # Can execute queries
            await storage.execute(
                "INSERT INTO entities (id, name, entity_type) VALUES (?, ?, ?)",
                ("ent_001", "Entity", "person"),
            )

            # Disconnect
            await storage.disconnect()
            assert storage._conn is None

    async def test_episode_hierarchy(self, temp_db):
        """Test that episode parent-child relationships work."""
        # Insert parent episode
        await temp_db.execute(
            """
            INSERT INTO episodes (id, level, content)
            VALUES (?, ?, ?)
            """,
            ("epi_parent", 1, "Parent episode"),
        )

        # Insert child episode
        await temp_db.execute(
            """
            INSERT INTO episodes (id, level, parent_id, content)
            VALUES (?, ?, ?, ?)
            """,
            ("epi_child", 0, "epi_parent", "Child episode"),
        )

        # Verify relationship
        result = await temp_db.fetch_one(
            "SELECT * FROM episodes WHERE id = ?", ("epi_child",)
        )

        assert result is not None
        assert result["parent_id"] == "epi_parent"

    async def test_episode_links_unique_constraint(self, temp_db):
        """Test that episode links enforce uniqueness constraint."""
        # Insert episodes
        await temp_db.execute_many(
            "INSERT INTO episodes (id, level, content) VALUES (?, ?, ?)",
            [
                ("epi_001", 0, "Episode 1"),
                ("epi_002", 0, "Episode 2"),
            ],
        )

        # Insert link
        await temp_db.execute(
            """
            INSERT INTO episode_links (id, source_id, target_id, link_type)
            VALUES (?, ?, ?, ?)
            """,
            ("link_001", "epi_001", "epi_002", "semantic"),
        )

        # Try to insert duplicate link (should fail)
        with pytest.raises(DatabaseError):
            await temp_db.execute(
                """
                INSERT INTO episode_links (id, source_id, target_id, link_type)
                VALUES (?, ?, ?, ?)
                """,
                ("link_002", "epi_001", "epi_002", "semantic"),
            )

    async def test_retrieval_indices(self, temp_db):
        """Test retrieval indices table."""
        # Insert episode
        await temp_db.execute(
            "INSERT INTO episodes (id, level, content) VALUES (?, ?, ?)",
            ("epi_001", 0, "Test episode"),
        )

        # Insert retrieval index entries
        await temp_db.execute_many(
            """
            INSERT INTO retrieval_indices (id, index_type, key, episode_id, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("idx_001", "keyword", "python", "epi_001", "Programming topic"),
                ("idx_002", "tag", "tutorial", "epi_001", "Educational content"),
            ],
        )

        # Verify indices
        results = await temp_db.fetch_all(
            "SELECT * FROM retrieval_indices WHERE episode_id = ?", ("epi_001",)
        )

        assert len(results) == 2
        assert results[0]["index_type"] in ["keyword", "tag"]
