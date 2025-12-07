"""ChromaDB vector storage implementation for HTMA.

Provides vector embeddings and semantic search for episodes and entities.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Collection

from htma.core.exceptions import VectorStoreError
from htma.core.types import Entity, EntityID, Episode, EpisodeID

logger = logging.getLogger(__name__)


class ChromaStorage:
    """ChromaDB vector storage for semantic search.

    This class manages vector embeddings for episodes and entities,
    enabling semantic search across the memory system.

    Attributes:
        persist_path: Path to ChromaDB persistence directory.
        embedding_function: Function to generate embeddings from text.
        client: ChromaDB client instance.
        episodes_collection: Collection for episode embeddings.
        entities_collection: Collection for entity embeddings.
    """

    def __init__(
        self,
        persist_path: str | Path,
        embedding_function: Callable[[str], list[float]] | None = None,
    ):
        """Initialize ChromaDB storage.

        Args:
            persist_path: Path to ChromaDB persistence directory.
            embedding_function: Optional custom embedding function.
                               If None, uses ChromaDB's default embeddings.
        """
        self.persist_path = Path(persist_path)
        self.embedding_function = embedding_function
        self.client: chromadb.Client | None = None
        self.episodes_collection: Collection | None = None
        self.entities_collection: Collection | None = None

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collections.

        Creates the persistence directory if it doesn't exist and
        sets up collections for episodes and entities.

        Raises:
            VectorStoreError: If initialization fails.
        """
        try:
            # Ensure persistence directory exists
            self.persist_path.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client with persistence
            # Use PersistentClient to avoid singleton conflicts in tests
            self.client = chromadb.PersistentClient(path=str(self.persist_path))

            # Create or get episodes collection
            self.episodes_collection = self.client.get_or_create_collection(
                name="episodes",
                metadata={
                    "description": "Episode embeddings for semantic search",
                    "hnsw:space": "cosine",
                },
            )

            # Create or get entities collection
            self.entities_collection = self.client.get_or_create_collection(
                name="entities",
                metadata={
                    "description": "Entity embeddings for semantic search",
                    "hnsw:space": "cosine",
                },
            )

            logger.info(f"ChromaDB initialized at {self.persist_path}")
            logger.info(
                f"Episodes collection: {self.episodes_collection.count()} documents"
            )
            logger.info(
                f"Entities collection: {self.entities_collection.count()} documents"
            )

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {e}") from e

    def _get_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using the configured function.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector, or None if using ChromaDB's default embeddings.
        """
        if self.embedding_function is None:
            return None
        return self.embedding_function(text)

    async def add_episode(self, episode: Episode) -> None:
        """Add episode with embedding to the vector store.

        Args:
            episode: Episode to add.

        Raises:
            VectorStoreError: If adding the episode fails.
        """
        if self.episodes_collection is None:
            raise VectorStoreError("ChromaDB not initialized. Call initialize() first.")

        try:
            # Prepare embedding text (combine content, summary, and keywords)
            embedding_text_parts = [episode.content]
            if episode.summary:
                embedding_text_parts.append(episode.summary)
            if episode.keywords:
                embedding_text_parts.append(" ".join(episode.keywords))

            embedding_text = " ".join(embedding_text_parts)

            # Prepare metadata
            metadata: dict[str, Any] = {
                "level": episode.level,
                "salience": episode.salience,
                "occurred_at": episode.occurred_at.isoformat(),
                "recorded_at": episode.recorded_at.isoformat(),
            }

            # Add tags and parent_id if present
            if episode.tags:
                metadata["tags"] = ",".join(episode.tags)
            if episode.parent_id:
                metadata["parent_id"] = episode.parent_id

            # Generate embedding if custom function provided
            embedding = self._get_embedding(embedding_text)

            # Add to collection
            if embedding is not None:
                self.episodes_collection.add(
                    ids=[episode.id],
                    embeddings=[embedding],
                    documents=[embedding_text],
                    metadatas=[metadata],
                )
            else:
                # Let ChromaDB generate embedding
                self.episodes_collection.add(
                    ids=[episode.id],
                    documents=[embedding_text],
                    metadatas=[metadata],
                )

            logger.debug(f"Added episode {episode.id} to vector store")

        except Exception as e:
            raise VectorStoreError(f"Failed to add episode {episode.id}: {e}") from e

    async def add_entity(self, entity: Entity) -> None:
        """Add entity with embedding to the vector store.

        Args:
            entity: Entity to add.

        Raises:
            VectorStoreError: If adding the entity fails.
        """
        if self.entities_collection is None:
            raise VectorStoreError("ChromaDB not initialized. Call initialize() first.")

        try:
            # Prepare embedding text
            embedding_text = f"{entity.name} ({entity.entity_type})"

            # Prepare metadata
            metadata: dict[str, Any] = {
                "entity_type": entity.entity_type,
                "created_at": entity.created_at.isoformat(),
            }

            # Generate embedding if custom function provided
            embedding = self._get_embedding(embedding_text)

            # Add to collection
            if embedding is not None:
                self.entities_collection.add(
                    ids=[entity.id],
                    embeddings=[embedding],
                    documents=[embedding_text],
                    metadatas=[metadata],
                )
            else:
                # Let ChromaDB generate embedding
                self.entities_collection.add(
                    ids=[entity.id],
                    documents=[embedding_text],
                    metadatas=[metadata],
                )

            logger.debug(f"Added entity {entity.id} to vector store")

        except Exception as e:
            raise VectorStoreError(f"Failed to add entity {entity.id}: {e}") from e

    async def search_episodes(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        level: int | None = None,
    ) -> list[tuple[EpisodeID, float]]:
        """Semantic search over episodes.

        Args:
            query: Search query text.
            n_results: Maximum number of results to return.
            where: Optional metadata filter conditions.
            level: Optional filter by episode level.

        Returns:
            List of (episode_id, distance) tuples, sorted by relevance.
            Lower distance means higher similarity.

        Raises:
            VectorStoreError: If search fails.
        """
        if self.episodes_collection is None:
            raise VectorStoreError("ChromaDB not initialized. Call initialize() first.")

        try:
            # Build where clause
            where_clause = where or {}
            if level is not None:
                where_clause["level"] = level

            # Generate query embedding if custom function provided
            query_embedding = self._get_embedding(query)

            # Perform search
            if query_embedding is not None:
                results = self.episodes_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause if where_clause else None,
                )
            else:
                results = self.episodes_collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause if where_clause else None,
                )

            # Extract IDs and distances
            if not results["ids"] or not results["distances"]:
                return []

            episode_results: list[tuple[EpisodeID, float]] = []
            for episode_id, distance in zip(
                results["ids"][0], results["distances"][0], strict=False
            ):
                episode_results.append((episode_id, distance))

            return episode_results

        except Exception as e:
            raise VectorStoreError(f"Failed to search episodes: {e}") from e

    async def search_entities(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        entity_type: str | None = None,
    ) -> list[tuple[EntityID, float]]:
        """Semantic search over entities.

        Args:
            query: Search query text.
            n_results: Maximum number of results to return.
            where: Optional metadata filter conditions.
            entity_type: Optional filter by entity type.

        Returns:
            List of (entity_id, distance) tuples, sorted by relevance.
            Lower distance means higher similarity.

        Raises:
            VectorStoreError: If search fails.
        """
        if self.entities_collection is None:
            raise VectorStoreError("ChromaDB not initialized. Call initialize() first.")

        try:
            # Build where clause
            where_clause = where or {}
            if entity_type is not None:
                where_clause["entity_type"] = entity_type

            # Generate query embedding if custom function provided
            query_embedding = self._get_embedding(query)

            # Perform search
            if query_embedding is not None:
                results = self.entities_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause if where_clause else None,
                )
            else:
                results = self.entities_collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause if where_clause else None,
                )

            # Extract IDs and distances
            if not results["ids"] or not results["distances"]:
                return []

            entity_results: list[tuple[EntityID, float]] = []
            for entity_id, distance in zip(
                results["ids"][0], results["distances"][0], strict=False
            ):
                entity_results.append((entity_id, distance))

            return entity_results

        except Exception as e:
            raise VectorStoreError(f"Failed to search entities: {e}") from e

    async def delete_episode(self, episode_id: EpisodeID) -> None:
        """Remove episode from the vector store.

        Args:
            episode_id: ID of episode to delete.

        Raises:
            VectorStoreError: If deletion fails.
        """
        if self.episodes_collection is None:
            raise VectorStoreError("ChromaDB not initialized. Call initialize() first.")

        try:
            self.episodes_collection.delete(ids=[episode_id])
            logger.debug(f"Deleted episode {episode_id} from vector store")

        except Exception as e:
            raise VectorStoreError(f"Failed to delete episode {episode_id}: {e}") from e

    async def delete_entity(self, entity_id: EntityID) -> None:
        """Remove entity from the vector store.

        Args:
            entity_id: ID of entity to delete.

        Raises:
            VectorStoreError: If deletion fails.
        """
        if self.entities_collection is None:
            raise VectorStoreError("ChromaDB not initialized. Call initialize() first.")

        try:
            self.entities_collection.delete(ids=[entity_id])
            logger.debug(f"Deleted entity {entity_id} from vector store")

        except Exception as e:
            raise VectorStoreError(f"Failed to delete entity {entity_id}: {e}") from e

    async def update_episode(self, episode: Episode) -> None:
        """Update episode embedding in the vector store.

        This is implemented as a delete and re-add operation.

        Args:
            episode: Episode with updated data.

        Raises:
            VectorStoreError: If update fails.
        """
        try:
            # Delete existing
            await self.delete_episode(episode.id)
            # Add updated version
            await self.add_episode(episode)
            logger.debug(f"Updated episode {episode.id} in vector store")

        except Exception as e:
            raise VectorStoreError(f"Failed to update episode {episode.id}: {e}") from e

    async def update_entity(self, entity: Entity) -> None:
        """Update entity embedding in the vector store.

        This is implemented as a delete and re-add operation.

        Args:
            entity: Entity with updated data.

        Raises:
            VectorStoreError: If update fails.
        """
        try:
            # Delete existing
            await self.delete_entity(entity.id)
            # Add updated version
            await self.add_entity(entity)
            logger.debug(f"Updated entity {entity.id} in vector store")

        except Exception as e:
            raise VectorStoreError(f"Failed to update entity {entity.id}: {e}") from e

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store collections.

        Returns:
            Dictionary with collection statistics.
        """
        if self.episodes_collection is None or self.entities_collection is None:
            raise VectorStoreError("ChromaDB not initialized. Call initialize() first.")

        return {
            "episodes_count": self.episodes_collection.count(),
            "entities_count": self.entities_collection.count(),
            "persist_path": str(self.persist_path),
        }

    async def reset(self) -> None:
        """Clear all data from the vector store.

        Warning: This is destructive and cannot be undone.

        Raises:
            VectorStoreError: If reset fails.
        """
        if self.client is None:
            raise VectorStoreError("ChromaDB not initialized. Call initialize() first.")

        try:
            # Delete collections
            if self.episodes_collection is not None:
                self.client.delete_collection("episodes")
            if self.entities_collection is not None:
                self.client.delete_collection("entities")

            # Recreate collections
            await self.initialize()

            logger.warning("Vector store has been reset")

        except Exception as e:
            raise VectorStoreError(f"Failed to reset vector store: {e}") from e
