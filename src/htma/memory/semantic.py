"""Semantic memory implementation with bi-temporal knowledge graph.

This module implements the semantic memory component, which stores entities and facts
with full bi-temporal support, enabling temporal reasoning and contradiction handling.
"""

import json
import logging
from datetime import datetime
from typing import Any

from htma.core.exceptions import (
    DatabaseError,
    DuplicateMemoryError,
    MemoryNotFoundError,
    TemporalValidationError,
)
from htma.core.types import Entity, EntityID, Fact, FactID, TemporalFilter
from htma.core.utils import ensure_utc, generate_entity_id, generate_fact_id, utc_now
from htma.storage.chroma import ChromaStorage
from htma.storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Semantic memory with bi-temporal knowledge graph.

    Manages entities and facts with full bi-temporal support, enabling queries like:
    - "What did we know about X as of date Y?" (transaction time)
    - "What was true about X at date Y?" (event time)

    Attributes:
        sqlite: SQLite storage for structured data.
        chroma: ChromaDB storage for semantic search.
    """

    def __init__(self, sqlite: SQLiteStorage, chroma: ChromaStorage):
        """Initialize semantic memory.

        Args:
            sqlite: SQLite storage instance.
            chroma: ChromaDB storage instance.
        """
        self.sqlite = sqlite
        self.chroma = chroma

    # ========== Entity Operations ==========

    async def add_entity(self, entity: Entity) -> Entity:
        """Add new entity to semantic memory.

        Args:
            entity: Entity to add.

        Returns:
            The added entity.

        Raises:
            DuplicateMemoryError: If entity ID already exists.
            DatabaseError: If database operation fails.
        """
        # Check if entity already exists
        existing = await self.get_entity(entity.id)
        if existing is not None:
            raise DuplicateMemoryError(entity.id, "entity")

        # Insert into SQLite
        query = """
            INSERT INTO entities (
                id, name, entity_type, created_at, last_accessed,
                access_count, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entity.id,
            entity.name,
            entity.entity_type,
            entity.created_at.isoformat(),
            entity.last_accessed.isoformat() if entity.last_accessed else None,
            entity.access_count,
            json.dumps(entity.metadata),
        )

        try:
            await self.sqlite.execute(query, params)
            logger.info(f"Added entity {entity.id} ({entity.name})")
        except Exception as e:
            raise DatabaseError(f"Failed to add entity {entity.id}: {e}") from e

        # Add to ChromaDB for semantic search
        try:
            await self.chroma.add_entity(entity)
        except Exception as e:
            logger.warning(f"Failed to add entity {entity.id} to vector store: {e}")

        return entity

    async def get_entity(self, entity_id: EntityID) -> Entity | None:
        """Get entity by ID.

        Args:
            entity_id: ID of entity to retrieve.

        Returns:
            Entity if found, None otherwise.

        Raises:
            DatabaseError: If database operation fails.
        """
        query = "SELECT * FROM entities WHERE id = ?"
        row = await self.sqlite.fetch_one(query, (entity_id,))

        if row is None:
            return None

        return self._row_to_entity(row)

    async def find_entity(
        self, name: str, entity_type: str | None = None
    ) -> list[Entity]:
        """Find entities by name (fuzzy match).

        Args:
            name: Entity name to search for (case-insensitive, partial match).
            entity_type: Optional filter by entity type.

        Returns:
            List of matching entities.

        Raises:
            DatabaseError: If database operation fails.
        """
        if entity_type is not None:
            query = """
                SELECT * FROM entities
                WHERE LOWER(name) LIKE LOWER(?) AND entity_type = ?
                ORDER BY created_at DESC
            """
            params = (f"%{name}%", entity_type)
        else:
            query = """
                SELECT * FROM entities
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY created_at DESC
            """
            params = (f"%{name}%",)

        rows = await self.sqlite.fetch_all(query, params)
        return [self._row_to_entity(row) for row in rows]

    async def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        """Semantic search over entities.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.

        Returns:
            List of matching entities, sorted by relevance.

        Raises:
            DatabaseError: If search fails.
        """
        # Use ChromaDB for semantic search
        try:
            results = await self.chroma.search_entities(query, n_results=limit)
        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to name search: {e}")
            return await self.find_entity(query)

        # Fetch full entities from SQLite
        entities = []
        for entity_id, _ in results:
            entity = await self.get_entity(entity_id)
            if entity is not None:
                entities.append(entity)

        return entities

    # ========== Fact Operations ==========

    async def add_fact(self, fact: Fact) -> Fact:
        """Add new fact to semantic memory.

        Args:
            fact: Fact to add.

        Returns:
            The added fact.

        Raises:
            DuplicateMemoryError: If fact ID already exists.
            MemoryNotFoundError: If subject or object entity doesn't exist.
            DatabaseError: If database operation fails.
        """
        # Verify subject entity exists
        subject = await self.get_entity(fact.subject_id)
        if subject is None:
            raise MemoryNotFoundError(fact.subject_id, "entity")

        # Verify object entity exists if specified
        if fact.object_id is not None:
            obj = await self.get_entity(fact.object_id)
            if obj is None:
                raise MemoryNotFoundError(fact.object_id, "entity")

        # Check if fact already exists
        existing_query = "SELECT id FROM facts WHERE id = ?"
        existing = await self.sqlite.fetch_one(existing_query, (fact.id,))
        if existing is not None:
            raise DuplicateMemoryError(fact.id, "fact")

        # Ensure timestamps are UTC
        valid_from = ensure_utc(fact.temporal.event_time.valid_from)
        valid_to = ensure_utc(fact.temporal.event_time.valid_to)
        recorded_at = ensure_utc(fact.temporal.transaction_time.valid_from) or utc_now()
        invalidated_at = ensure_utc(fact.temporal.transaction_time.valid_to)

        # Insert into SQLite
        query = """
            INSERT INTO facts (
                id, subject_id, predicate, object_id, object_value,
                valid_from, valid_to, recorded_at, invalidated_at,
                confidence, source_episode_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            fact.id,
            fact.subject_id,
            fact.predicate,
            fact.object_id,
            fact.object_value,
            valid_from.isoformat() if valid_from else None,
            valid_to.isoformat() if valid_to else None,
            recorded_at.isoformat(),
            invalidated_at.isoformat() if invalidated_at else None,
            fact.confidence,
            fact.source_episode_id,
            json.dumps(fact.metadata),
        )

        try:
            await self.sqlite.execute(query, params)
            logger.info(
                f"Added fact {fact.id}: {fact.subject_id} {fact.predicate} "
                f"{fact.object_id or fact.object_value}"
            )
        except Exception as e:
            raise DatabaseError(f"Failed to add fact {fact.id}: {e}") from e

        return fact

    async def invalidate_fact(self, fact_id: FactID, as_of: datetime) -> None:
        """Mark fact as invalid from given time (transaction time).

        This sets the transaction time's valid_to (invalidated_at) to indicate
        that this fact is no longer considered valid in our knowledge as of this time.

        Args:
            fact_id: ID of fact to invalidate.
            as_of: Transaction time when fact becomes invalid.

        Raises:
            MemoryNotFoundError: If fact doesn't exist.
            DatabaseError: If database operation fails.
        """
        # Verify fact exists
        fact = await self._get_fact_by_id(fact_id)
        if fact is None:
            raise MemoryNotFoundError(fact_id, "fact")

        # Ensure timestamp is UTC
        invalidation_time = ensure_utc(as_of) or utc_now()

        # Update invalidated_at
        query = "UPDATE facts SET invalidated_at = ? WHERE id = ?"
        params = (invalidation_time.isoformat(), fact_id)

        try:
            await self.sqlite.execute(query, params)
            logger.info(f"Invalidated fact {fact_id} as of {invalidation_time}")
        except Exception as e:
            raise DatabaseError(f"Failed to invalidate fact {fact_id}: {e}") from e

    # ========== Temporal Queries ==========

    async def query_entity_facts(
        self,
        entity_id: EntityID,
        predicate: str | None = None,
        temporal: TemporalFilter | None = None,
    ) -> list[Fact]:
        """Get facts about entity with optional predicate and temporal filtering.

        Args:
            entity_id: ID of entity to query.
            predicate: Optional filter by predicate.
            temporal: Optional temporal filter.

        Returns:
            List of matching facts.

        Raises:
            DatabaseError: If query fails.
        """
        # Build query
        conditions = ["subject_id = ?"]
        params: list[Any] = [entity_id]

        if predicate is not None:
            conditions.append("predicate = ?")
            params.append(predicate)

        # Apply temporal filters
        if temporal is not None:
            if temporal.as_of is not None:
                # Transaction time filter
                as_of_time = ensure_utc(temporal.as_of)
                conditions.append("recorded_at <= ?")
                conditions.append("(invalidated_at IS NULL OR invalidated_at > ?)")
                params.extend([as_of_time.isoformat(), as_of_time.isoformat()])

            if temporal.valid_at is not None:
                # Event time filter
                valid_time = ensure_utc(temporal.valid_at)
                conditions.append("(valid_from IS NULL OR valid_from <= ?)")
                conditions.append("(valid_to IS NULL OR valid_to > ?)")
                params.extend([valid_time.isoformat(), valid_time.isoformat()])
        else:
            # Default: only currently valid facts (not invalidated)
            conditions.append("invalidated_at IS NULL")

        query = f"SELECT * FROM facts WHERE {' AND '.join(conditions)} ORDER BY recorded_at DESC"

        rows = await self.sqlite.fetch_all(query, tuple(params))
        return [self._row_to_fact(row) for row in rows]

    async def query_at_time(self, entity_id: EntityID, as_of: datetime) -> list[Fact]:
        """What did we know about entity as of transaction time?

        This queries based on when facts were recorded in the system.

        Args:
            entity_id: ID of entity to query.
            as_of: Transaction time to query.

        Returns:
            List of facts known as of the given time.

        Raises:
            DatabaseError: If query fails.
        """
        temporal_filter = TemporalFilter(as_of=as_of)
        return await self.query_entity_facts(entity_id, temporal=temporal_filter)

    async def query_valid_at(self, entity_id: EntityID, when: datetime) -> list[Fact]:
        """What facts were true about entity at event time?

        This queries based on when facts were actually true in the world.

        Args:
            entity_id: ID of entity to query.
            when: Event time to query.

        Returns:
            List of facts that were true at the given time.

        Raises:
            DatabaseError: If query fails.
        """
        temporal_filter = TemporalFilter(valid_at=when)
        return await self.query_entity_facts(entity_id, temporal=temporal_filter)

    async def get_fact_history(
        self, subject_id: EntityID, predicate: str
    ) -> list[Fact]:
        """Get all versions of a fact over time.

        Returns all facts (including invalidated ones) for a subject-predicate pair,
        ordered chronologically to show how the fact evolved.

        Args:
            subject_id: ID of subject entity.
            predicate: Predicate to query.

        Returns:
            List of all fact versions, ordered by recorded_at.

        Raises:
            DatabaseError: If query fails.
        """
        query = """
            SELECT * FROM facts
            WHERE subject_id = ? AND predicate = ?
            ORDER BY recorded_at ASC
        """
        params = (subject_id, predicate)

        rows = await self.sqlite.fetch_all(query, params)
        return [self._row_to_fact(row) for row in rows]

    # ========== Access Tracking ==========

    async def record_access(self, entity_id: EntityID) -> None:
        """Update access count and timestamp for an entity.

        Args:
            entity_id: ID of entity that was accessed.

        Raises:
            MemoryNotFoundError: If entity doesn't exist.
            DatabaseError: If update fails.
        """
        # Verify entity exists
        entity = await self.get_entity(entity_id)
        if entity is None:
            raise MemoryNotFoundError(entity_id, "entity")

        # Update access tracking
        now = utc_now()
        query = """
            UPDATE entities
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """
        params = (now.isoformat(), entity_id)

        try:
            await self.sqlite.execute(query, params)
            logger.debug(f"Recorded access to entity {entity_id}")
        except Exception as e:
            raise DatabaseError(f"Failed to record access for {entity_id}: {e}") from e

    # ========== Community Operations ==========

    async def add_to_community(self, entity_id: EntityID, community_id: str) -> None:
        """Add entity to community cluster.

        Args:
            entity_id: ID of entity to add.
            community_id: ID of community to add to.

        Raises:
            MemoryNotFoundError: If entity or community doesn't exist.
            DatabaseError: If update fails.
        """
        # Verify entity exists
        entity = await self.get_entity(entity_id)
        if entity is None:
            raise MemoryNotFoundError(entity_id, "entity")

        # Get or create community
        community_query = "SELECT * FROM communities WHERE id = ?"
        community = await self.sqlite.fetch_one(community_query, (community_id,))

        if community is None:
            raise MemoryNotFoundError(community_id, "community")

        # Get current entity_ids
        entity_ids = json.loads(community["entity_ids"]) if community["entity_ids"] else []

        # Add entity if not already in community
        if entity_id not in entity_ids:
            entity_ids.append(entity_id)

            # Update community
            update_query = "UPDATE communities SET entity_ids = ? WHERE id = ?"
            params = (json.dumps(entity_ids), community_id)

            try:
                await self.sqlite.execute(update_query, params)
                logger.info(f"Added entity {entity_id} to community {community_id}")
            except Exception as e:
                raise DatabaseError(
                    f"Failed to add entity to community: {e}"
                ) from e

    async def get_community_entities(self, community_id: str) -> list[Entity]:
        """Get all entities in a community.

        Args:
            community_id: ID of community to query.

        Returns:
            List of entities in the community.

        Raises:
            MemoryNotFoundError: If community doesn't exist.
            DatabaseError: If query fails.
        """
        # Get community
        query = "SELECT * FROM communities WHERE id = ?"
        community = await self.sqlite.fetch_one(query, (community_id,))

        if community is None:
            raise MemoryNotFoundError(community_id, "community")

        # Get entity IDs
        entity_ids = json.loads(community["entity_ids"]) if community["entity_ids"] else []

        # Fetch entities
        entities = []
        for entity_id in entity_ids:
            entity = await self.get_entity(entity_id)
            if entity is not None:
                entities.append(entity)

        return entities

    # ========== Helper Methods ==========

    def _row_to_entity(self, row: dict[str, Any]) -> Entity:
        """Convert database row to Entity model.

        Args:
            row: Database row as dictionary.

        Returns:
            Entity instance.
        """
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=(
                datetime.fromisoformat(row["last_accessed"])
                if row["last_accessed"]
                else None
            ),
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_fact(self, row: dict[str, Any]) -> Fact:
        """Convert database row to Fact model.

        Args:
            row: Database row as dictionary.

        Returns:
            Fact instance.
        """
        from htma.core.types import BiTemporalRecord, TemporalRange

        # Build bi-temporal record
        event_time = TemporalRange(
            valid_from=(
                datetime.fromisoformat(row["valid_from"]) if row["valid_from"] else None
            ),
            valid_to=(
                datetime.fromisoformat(row["valid_to"]) if row["valid_to"] else None
            ),
        )

        transaction_time = TemporalRange(
            valid_from=(
                datetime.fromisoformat(row["recorded_at"])
                if row["recorded_at"]
                else None
            ),
            valid_to=(
                datetime.fromisoformat(row["invalidated_at"])
                if row["invalidated_at"]
                else None
            ),
        )

        temporal = BiTemporalRecord(
            event_time=event_time, transaction_time=transaction_time
        )

        return Fact(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            object_value=row["object_value"],
            temporal=temporal,
            confidence=row["confidence"],
            source_episode_id=row["source_episode_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def _get_fact_by_id(self, fact_id: FactID) -> Fact | None:
        """Get fact by ID (internal helper).

        Args:
            fact_id: ID of fact to retrieve.

        Returns:
            Fact if found, None otherwise.
        """
        query = "SELECT * FROM facts WHERE id = ?"
        row = await self.sqlite.fetch_one(query, (fact_id,))

        if row is None:
            return None

        return self._row_to_fact(row)
