"""Episodic memory implementation with hierarchical storage.

This module implements the episodic memory component with RAPTOR-style hierarchical
storage, enabling multi-level retrieval from raw episodes to progressive abstractions.
"""

import json
import logging
from datetime import datetime
from typing import Any

from htma.core.exceptions import (
    DatabaseError,
    DuplicateMemoryError,
    MemoryNotFoundError,
)
from htma.core.types import Episode, EpisodeID, EpisodeLink, TemporalFilter
from htma.core.utils import ensure_utc, generate_episode_id, generate_link_id, utc_now
from htma.storage.chroma import ChromaStorage
from htma.storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Hierarchical episodic memory with RAPTOR-style abstraction levels.

    Manages episodes in a hierarchy:
    - Level 0: Raw episodes (full interactions)
    - Level 1: Summaries (clustered episodes)
    - Level 2+: Progressive abstractions

    Supports multi-path retrieval via:
    - Semantic search
    - Temporal queries
    - Hierarchical navigation
    - Retrieval indices
    - Episode links

    Attributes:
        sqlite: SQLite storage for structured data.
        chroma: ChromaDB storage for semantic search.
    """

    def __init__(self, sqlite: SQLiteStorage, chroma: ChromaStorage):
        """Initialize episodic memory.

        Args:
            sqlite: SQLite storage instance.
            chroma: ChromaDB storage instance.
        """
        self.sqlite = sqlite
        self.chroma = chroma

    # ========== Episode Operations ==========

    async def add_episode(self, episode: Episode) -> Episode:
        """Add new episode at specified level.

        Args:
            episode: Episode to add.

        Returns:
            The added episode.

        Raises:
            DuplicateMemoryError: If episode ID already exists.
            MemoryNotFoundError: If parent_id is specified but doesn't exist.
            DatabaseError: If database operation fails.
        """
        # Check if episode already exists
        existing = await self.get_episode(episode.id)
        if existing is not None:
            raise DuplicateMemoryError(episode.id, "episode")

        # Verify parent exists if specified
        if episode.parent_id is not None:
            parent = await self.get_episode(episode.parent_id)
            if parent is None:
                raise MemoryNotFoundError(episode.parent_id, "episode")

        # Insert into SQLite
        query = """
            INSERT INTO episodes (
                id, level, parent_id, content, summary, context_description,
                keywords, tags, occurred_at, recorded_at, salience,
                access_count, last_accessed, consolidation_strength,
                consolidated_into, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            episode.id,
            episode.level,
            episode.parent_id,
            episode.content,
            episode.summary,
            episode.context_description,
            json.dumps(episode.keywords),
            json.dumps(episode.tags),
            episode.occurred_at.isoformat(),
            episode.recorded_at.isoformat(),
            episode.salience,
            episode.access_count,
            episode.last_accessed.isoformat() if episode.last_accessed else None,
            episode.consolidation_strength,
            None,  # consolidated_into - initially None
            json.dumps(episode.metadata),
        )

        try:
            await self.sqlite.execute(query, params)
            logger.info(
                f"Added episode {episode.id} (level {episode.level}, "
                f"salience {episode.salience:.2f})"
            )
        except Exception as e:
            raise DatabaseError(f"Failed to add episode {episode.id}: {e}") from e

        # Add to ChromaDB for semantic search
        try:
            await self.chroma.add_episode(episode)
        except Exception as e:
            logger.warning(f"Failed to add episode {episode.id} to vector store: {e}")

        return episode

    async def get_episode(self, episode_id: EpisodeID) -> Episode | None:
        """Get episode by ID.

        Args:
            episode_id: ID of episode to retrieve.

        Returns:
            Episode if found, None otherwise.

        Raises:
            DatabaseError: If database operation fails.
        """
        query = "SELECT * FROM episodes WHERE id = ?"
        row = await self.sqlite.fetch_one(query, (episode_id,))

        if row is None:
            return None

        return self._row_to_episode(row)

    async def get_children(self, episode_id: EpisodeID) -> list[Episode]:
        """Get child episodes (lower level).

        Args:
            episode_id: ID of parent episode.

        Returns:
            List of child episodes, ordered by occurred_at.

        Raises:
            DatabaseError: If query fails.
        """
        query = """
            SELECT * FROM episodes
            WHERE parent_id = ?
            ORDER BY occurred_at ASC
        """
        rows = await self.sqlite.fetch_all(query, (episode_id,))
        return [self._row_to_episode(row) for row in rows]

    async def get_parent(self, episode_id: EpisodeID) -> Episode | None:
        """Get parent episode (higher level).

        Args:
            episode_id: ID of child episode.

        Returns:
            Parent episode if it exists, None otherwise.

        Raises:
            DatabaseError: If query fails.
        """
        # First get the episode to find its parent_id
        episode = await self.get_episode(episode_id)
        if episode is None or episode.parent_id is None:
            return None

        return await self.get_episode(episode.parent_id)

    # ========== Retrieval Methods ==========

    async def search(
        self,
        query: str,
        level: int | None = None,
        temporal: TemporalFilter | None = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Semantic search with optional filters.

        Args:
            query: Search query text.
            level: Optional filter by specific abstraction level.
            temporal: Optional temporal filter.
            limit: Maximum number of results to return.

        Returns:
            List of matching episodes, sorted by relevance.

        Raises:
            DatabaseError: If search fails.
        """
        # Build metadata filter for ChromaDB
        where: dict[str, Any] = {}
        if level is not None:
            where["level"] = level

        # Use ChromaDB for semantic search
        try:
            results = await self.chroma.search_episodes(
                query, n_results=limit, where=where if where else None
            )
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            # Fall back to recency-based retrieval
            return await self.get_recent(level=level, limit=limit)

        # Fetch full episodes from SQLite
        episodes = []
        for episode_id, _ in results:
            episode = await self.get_episode(episode_id)
            if episode is not None:
                # Apply temporal filter if specified
                if temporal is not None:
                    if not self._matches_temporal_filter(episode, temporal):
                        continue
                episodes.append(episode)

        return episodes

    async def get_recent(self, level: int = 0, limit: int = 10) -> list[Episode]:
        """Get most recent episodes at specified level.

        Args:
            level: Abstraction level to query.
            limit: Maximum number of results to return.

        Returns:
            List of recent episodes, sorted by occurred_at descending.

        Raises:
            DatabaseError: If query fails.
        """
        query = """
            SELECT * FROM episodes
            WHERE level = ?
            ORDER BY occurred_at DESC
            LIMIT ?
        """
        rows = await self.sqlite.fetch_all(query, (level, limit))
        return [self._row_to_episode(row) for row in rows]

    async def get_by_index(self, index_type: str, key: str) -> list[Episode]:
        """Retrieve via retrieval index.

        Args:
            index_type: Type of index (e.g., "topic", "entity", "event").
            key: Index key to look up.

        Returns:
            List of episodes matching the index.

        Raises:
            DatabaseError: If query fails.
        """
        # Query retrieval indices
        query = """
            SELECT episode_id FROM retrieval_indices
            WHERE index_type = ? AND key = ?
        """
        rows = await self.sqlite.fetch_all(query, (index_type, key))

        # Fetch episodes
        episodes = []
        for row in rows:
            episode = await self.get_episode(row["episode_id"])
            if episode is not None:
                episodes.append(episode)

        return episodes

    # ========== Linking Operations ==========

    async def add_link(self, link: EpisodeLink) -> None:
        """Create bidirectional link between episodes.

        Args:
            link: Episode link to create.

        Raises:
            MemoryNotFoundError: If source or target episode doesn't exist.
            DatabaseError: If database operation fails.
        """
        # Verify both episodes exist
        source = await self.get_episode(link.source_id)
        if source is None:
            raise MemoryNotFoundError(link.source_id, "episode")

        target = await self.get_episode(link.target_id)
        if target is None:
            raise MemoryNotFoundError(link.target_id, "episode")

        # Insert link
        query = """
            INSERT OR REPLACE INTO episode_links (
                id, source_id, target_id, link_type, weight, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            link.id,
            link.source_id,
            link.target_id,
            link.link_type,
            link.weight,
            link.created_at.isoformat(),
        )

        try:
            await self.sqlite.execute(query, params)
            logger.info(
                f"Added {link.link_type} link {link.id}: "
                f"{link.source_id} -> {link.target_id} (weight: {link.weight:.2f})"
            )
        except Exception as e:
            raise DatabaseError(f"Failed to add link {link.id}: {e}") from e

    async def get_links(
        self, episode_id: EpisodeID, link_type: str | None = None
    ) -> list[EpisodeLink]:
        """Get links for episode.

        Args:
            episode_id: ID of episode to get links for.
            link_type: Optional filter by link type.

        Returns:
            List of episode links.

        Raises:
            DatabaseError: If query fails.
        """
        if link_type is not None:
            query = """
                SELECT * FROM episode_links
                WHERE (source_id = ? OR target_id = ?) AND link_type = ?
                ORDER BY weight DESC
            """
            params = (episode_id, episode_id, link_type)
        else:
            query = """
                SELECT * FROM episode_links
                WHERE source_id = ? OR target_id = ?
                ORDER BY weight DESC
            """
            params = (episode_id, episode_id)

        rows = await self.sqlite.fetch_all(query, params)
        return [self._row_to_link(row) for row in rows]

    async def get_linked_episodes(
        self, episode_id: EpisodeID, link_type: str | None = None
    ) -> list[Episode]:
        """Get episodes linked to given episode.

        Args:
            episode_id: ID of episode to find links from.
            link_type: Optional filter by link type.

        Returns:
            List of linked episodes.

        Raises:
            DatabaseError: If query fails.
        """
        # Get links
        links = await self.get_links(episode_id, link_type)

        # Collect linked episode IDs
        linked_ids = set()
        for link in links:
            if link.source_id == episode_id:
                linked_ids.add(link.target_id)
            else:
                linked_ids.add(link.source_id)

        # Fetch episodes
        episodes = []
        for linked_id in linked_ids:
            episode = await self.get_episode(linked_id)
            if episode is not None:
                episodes.append(episode)

        return episodes

    async def update_link_weight(
        self, source_id: EpisodeID, target_id: EpisodeID, delta: float
    ) -> None:
        """Adjust link weight.

        Args:
            source_id: Source episode ID.
            target_id: Target episode ID.
            delta: Amount to add to weight (can be negative).

        Raises:
            MemoryNotFoundError: If link doesn't exist.
            DatabaseError: If update fails.
        """
        # Find the link (could be in either direction)
        query = """
            SELECT * FROM episode_links
            WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)
        """
        params = (source_id, target_id, target_id, source_id)
        row = await self.sqlite.fetch_one(query, params)

        if row is None:
            raise MemoryNotFoundError(
                f"{source_id}->{target_id}", "episode link"
            )

        # Update weight
        new_weight = max(0.0, row["weight"] + delta)  # Ensure non-negative
        update_query = "UPDATE episode_links SET weight = ? WHERE id = ?"
        update_params = (new_weight, row["id"])

        try:
            await self.sqlite.execute(update_query, update_params)
            logger.debug(
                f"Updated link weight {row['id']}: {row['weight']:.2f} -> {new_weight:.2f}"
            )
        except Exception as e:
            raise DatabaseError(f"Failed to update link weight: {e}") from e

    # ========== Indexing Operations ==========

    async def add_index_entry(
        self,
        index_type: str,
        key: str,
        episode_id: EpisodeID,
        note: str | None = None,
    ) -> None:
        """Add retrieval index entry.

        Args:
            index_type: Type of index (e.g., "topic", "entity", "event").
            key: Index key.
            episode_id: Episode to index.
            note: Optional note about why this index entry exists.

        Raises:
            MemoryNotFoundError: If episode doesn't exist.
            DatabaseError: If operation fails.
        """
        # Verify episode exists
        episode = await self.get_episode(episode_id)
        if episode is None:
            raise MemoryNotFoundError(episode_id, "episode")

        # Generate index entry ID
        index_id = f"idx_{index_type}_{key}_{episode_id}"

        # Insert index entry
        query = """
            INSERT OR REPLACE INTO retrieval_indices (
                id, index_type, key, episode_id, note
            ) VALUES (?, ?, ?, ?, ?)
        """
        params = (index_id, index_type, key, episode_id, note)

        try:
            await self.sqlite.execute(query, params)
            logger.debug(f"Added index entry: {index_type}:{key} -> {episode_id}")
        except Exception as e:
            raise DatabaseError(f"Failed to add index entry: {e}") from e

    async def get_index_keys(self, index_type: str) -> list[str]:
        """Get all keys for an index type.

        Args:
            index_type: Type of index to query.

        Returns:
            List of unique keys for the index type.

        Raises:
            DatabaseError: If query fails.
        """
        query = """
            SELECT DISTINCT key FROM retrieval_indices
            WHERE index_type = ?
            ORDER BY key
        """
        rows = await self.sqlite.fetch_all(query, (index_type,))
        return [row["key"] for row in rows]

    # ========== Access Tracking ==========

    async def record_access(self, episode_id: EpisodeID) -> None:
        """Update access count and timestamp.

        Args:
            episode_id: ID of episode that was accessed.

        Raises:
            MemoryNotFoundError: If episode doesn't exist.
            DatabaseError: If update fails.
        """
        # Verify episode exists
        episode = await self.get_episode(episode_id)
        if episode is None:
            raise MemoryNotFoundError(episode_id, "episode")

        # Update access tracking
        now = utc_now()
        query = """
            UPDATE episodes
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """
        params = (now.isoformat(), episode_id)

        try:
            await self.sqlite.execute(query, params)
            logger.debug(f"Recorded access to episode {episode_id}")
        except Exception as e:
            raise DatabaseError(f"Failed to record access for {episode_id}: {e}") from e

    # ========== Consolidation Support ==========

    async def mark_consolidated(
        self, episode_id: EpisodeID, consolidated_into: EpisodeID
    ) -> None:
        """Mark episode as consolidated into higher-level.

        Args:
            episode_id: ID of episode that was consolidated.
            consolidated_into: ID of higher-level episode it was consolidated into.

        Raises:
            MemoryNotFoundError: If either episode doesn't exist.
            DatabaseError: If update fails.
        """
        # Verify both episodes exist
        episode = await self.get_episode(episode_id)
        if episode is None:
            raise MemoryNotFoundError(episode_id, "episode")

        parent = await self.get_episode(consolidated_into)
        if parent is None:
            raise MemoryNotFoundError(consolidated_into, "episode")

        # Update consolidated_into field
        query = "UPDATE episodes SET consolidated_into = ? WHERE id = ?"
        params = (consolidated_into, episode_id)

        try:
            await self.sqlite.execute(query, params)
            logger.info(f"Marked episode {episode_id} as consolidated into {consolidated_into}")
        except Exception as e:
            raise DatabaseError(f"Failed to mark episode as consolidated: {e}") from e

    async def get_unconsolidated(
        self, level: int, older_than: datetime
    ) -> list[Episode]:
        """Get episodes ready for consolidation.

        Args:
            level: Level of episodes to query.
            older_than: Only return episodes recorded before this time.

        Returns:
            List of unconsolidated episodes that are ready for consolidation.

        Raises:
            DatabaseError: If query fails.
        """
        # Ensure timestamp is UTC
        cutoff_time = ensure_utc(older_than) or utc_now()

        query = """
            SELECT * FROM episodes
            WHERE level = ?
              AND recorded_at < ?
              AND consolidated_into IS NULL
            ORDER BY recorded_at ASC
        """
        params = (level, cutoff_time.isoformat())

        rows = await self.sqlite.fetch_all(query, params)
        return [self._row_to_episode(row) for row in rows]

    # ========== Helper Methods ==========

    def _row_to_episode(self, row: dict[str, Any]) -> Episode:
        """Convert database row to Episode model.

        Args:
            row: Database row as dictionary.

        Returns:
            Episode instance.
        """
        return Episode(
            id=row["id"],
            level=row["level"],
            parent_id=row["parent_id"],
            content=row["content"],
            summary=row["summary"],
            context_description=row["context_description"],
            keywords=json.loads(row["keywords"]) if row["keywords"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            salience=row["salience"],
            consolidation_strength=row["consolidation_strength"],
            access_count=row["access_count"],
            last_accessed=(
                datetime.fromisoformat(row["last_accessed"])
                if row["last_accessed"]
                else None
            ),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_link(self, row: dict[str, Any]) -> EpisodeLink:
        """Convert database row to EpisodeLink model.

        Args:
            row: Database row as dictionary.

        Returns:
            EpisodeLink instance.
        """
        return EpisodeLink(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            link_type=row["link_type"],
            weight=row["weight"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _matches_temporal_filter(
        self, episode: Episode, temporal: TemporalFilter
    ) -> bool:
        """Check if episode matches temporal filter.

        Args:
            episode: Episode to check.
            temporal: Temporal filter to apply.

        Returns:
            True if episode matches filter.
        """
        # For episodes, we use occurred_at for temporal filtering
        if temporal.valid_at is not None:
            valid_time = ensure_utc(temporal.valid_at)
            episode_time = ensure_utc(episode.occurred_at)
            if episode_time is None or valid_time is None:
                return False
            # Episode must have occurred before or at the valid_at time
            if episode_time > valid_time:
                return False

        if temporal.as_of is not None:
            # Transaction time: when was it recorded
            as_of_time = ensure_utc(temporal.as_of)
            recorded_time = ensure_utc(episode.recorded_at)
            if recorded_time is None or as_of_time is None:
                return False
            # Episode must have been recorded before or at the as_of time
            if recorded_time > as_of_time:
                return False

        return True
