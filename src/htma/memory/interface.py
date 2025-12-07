"""Memory interface for query routing and synthesis.

This module implements the coordination layer between LLM1 (reasoner) and memory stores.
It handles query routing, result synthesis, and orchestrates memory storage operations.
"""

import logging
import time
from datetime import datetime
from typing import Any

from htma.core.exceptions import MemoryError
from htma.core.types import (
    Episode,
    EpisodeID,
    EpisodeLink,
    Fact,
    Interaction,
    RetrievalResult,
    StorageResult,
    TemporalFilter,
)
from htma.core.utils import generate_episode_id, generate_link_id, utc_now
from htma.curator.curator import MemoryCurator
from htma.memory.episodic import EpisodicMemory
from htma.memory.semantic import SemanticMemory
from htma.memory.working import MemoryItem, WorkingMemory

logger = logging.getLogger(__name__)


class MemoryInterface:
    """Coordination layer between LLM1 and memory stores.

    The MemoryInterface routes queries to appropriate memory stores, synthesizes
    results, and orchestrates memory storage operations through the curator.

    Query routing logic:
    - Entity names detected → include semantic memory
    - Temporal markers → add temporal filter
    - Experience references → include episodic memory
    - Default: search both stores

    Result synthesis:
    - Deduplicate overlapping results
    - Rank by relevance + recency
    - Format for context injection

    Attributes:
        working: Working memory for in-context information.
        semantic: Semantic memory (temporal knowledge graph).
        episodic: Episodic memory (hierarchical experiences).
        curator: Memory curator for salience evaluation and processing.
    """

    def __init__(
        self,
        working: WorkingMemory,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        curator: MemoryCurator,
    ):
        """Initialize memory interface.

        Args:
            working: Working memory instance.
            semantic: Semantic memory instance.
            episodic: Episodic memory instance.
            curator: Memory curator instance.
        """
        self.working = working
        self.semantic = semantic
        self.episodic = episodic
        self.curator = curator
        logger.info("Initialized MemoryInterface")

    # ========== Query Operations ==========

    async def query(
        self,
        query: str,
        include_semantic: bool = True,
        include_episodic: bool = True,
        temporal: TemporalFilter | None = None,
        limit: int = 10,
    ) -> RetrievalResult:
        """Route query to appropriate stores and synthesize results.

        Process:
        1. Analyze query for routing hints
        2. Dispatch parallel queries to selected stores
        3. Rank and deduplicate results
        4. Return synthesized result

        Args:
            query: Query text.
            include_semantic: Whether to query semantic memory.
            include_episodic: Whether to query episodic memory.
            temporal: Optional temporal filter.
            limit: Maximum number of results per store.

        Returns:
            RetrievalResult with facts and episodes.

        Raises:
            MemoryError: If query fails.
        """
        logger.debug(
            f"Querying memory: semantic={include_semantic}, "
            f"episodic={include_episodic}, query='{query[:50]}...'"
        )

        result = RetrievalResult()

        try:
            # Query semantic memory if requested
            if include_semantic:
                # Search for entities semantically
                entities = await self.semantic.search_entities(query, limit=limit)

                # Get facts for found entities
                for entity in entities:
                    facts = await self.semantic.query_entity_facts(
                        entity.id, temporal=temporal
                    )
                    result.facts.extend(facts)

                    # Track entity access
                    await self.semantic.record_access(entity.id)

                    # Add relevance score for entity (based on position in results)
                    result.relevance_scores[entity.id] = 1.0 - (
                        entities.index(entity) / len(entities)
                    )

            # Query episodic memory if requested
            if include_episodic:
                episodes = await self.episodic.search(
                    query, temporal=temporal, limit=limit
                )
                result.episodes.extend(episodes)

                # Track episode access
                for episode in episodes:
                    await self.episodic.record_access(episode.id)

                    # Add relevance score (based on position in results)
                    result.relevance_scores[episode.id] = 1.0 - (
                        episodes.index(episode) / len(episodes)
                    )

            # Deduplicate and rank
            result = self._deduplicate_results(result)
            result = self._rank_results(result)

            logger.info(
                f"Query returned {len(result.facts)} facts and "
                f"{len(result.episodes)} episodes"
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise MemoryError(f"Failed to query memory: {e}") from e

        return result

    async def query_semantic(
        self, query: str, temporal: TemporalFilter | None = None
    ) -> list[Fact]:
        """Direct semantic memory query.

        Args:
            query: Query text.
            temporal: Optional temporal filter.

        Returns:
            List of matching facts.

        Raises:
            MemoryError: If query fails.
        """
        result = await self.query(
            query,
            include_semantic=True,
            include_episodic=False,
            temporal=temporal,
        )
        return result.facts

    async def query_episodic(
        self,
        query: str,
        level: int | None = None,
        temporal: TemporalFilter | None = None,
    ) -> list[Episode]:
        """Direct episodic memory query.

        Args:
            query: Query text.
            level: Optional filter by abstraction level.
            temporal: Optional temporal filter.

        Returns:
            List of matching episodes.

        Raises:
            MemoryError: If query fails.
        """
        try:
            episodes = await self.episodic.search(
                query, level=level, temporal=temporal
            )

            # Track access
            for episode in episodes:
                await self.episodic.record_access(episode.id)

            return episodes

        except Exception as e:
            logger.error(f"Episodic query failed: {e}")
            raise MemoryError(f"Failed to query episodic memory: {e}") from e

    # ========== Write Operations ==========

    async def store_interaction(self, interaction: Interaction) -> StorageResult:
        """Process and store new interaction.

        Process:
        1. Curator evaluates salience
        2. Create episode if salient enough
        3. Extract entities and facts (via curator)
        4. Generate links to related episodes
        5. Trigger evolution of related memories

        Args:
            interaction: The interaction to store.

        Returns:
            StorageResult with IDs of created memories.

        Raises:
            MemoryError: If storage fails.
        """
        start_time = time.time()
        result = StorageResult()

        try:
            logger.debug("Storing interaction in memory")

            # Step 1: Evaluate salience
            memory_note = await self.curator.evaluate_salience(
                interaction, context=self.working.task_context
            )
            result.salience_score = memory_note.salience
            result.metadata["memory_note"] = memory_note.model_dump()

            # Only store if salience is above threshold
            if memory_note.salience < 0.3:
                logger.info(
                    f"Interaction salience too low ({memory_note.salience:.2f}), "
                    "not storing"
                )
                result.processing_time = time.time() - start_time
                return result

            # Step 2: Create episode
            episode_id = generate_episode_id()
            episode = Episode(
                id=episode_id,
                level=0,  # Raw episode
                content=memory_note.content,
                summary=None,
                context_description=memory_note.context,
                keywords=memory_note.keywords,
                tags=memory_note.tags,
                occurred_at=interaction.occurred_at,
                recorded_at=utc_now(),
                salience=memory_note.salience,
            )

            await self.episodic.add_episode(episode)
            result.episode_id = episode_id
            logger.info(f"Created episode {episode_id} with salience {memory_note.salience:.2f}")

            # Step 3: Extract entities and facts
            entities = await self.curator.extract_entities(interaction)
            for entity in entities:
                await self.semantic.add_entity(entity)
                result.entities_created.append(entity.id)

            facts = await self.curator.extract_facts(interaction, entities)
            for fact in facts:
                await self.semantic.add_fact(fact)
                result.facts_created.append(fact.id)

            logger.info(
                f"Extracted {len(entities)} entities and {len(facts)} facts"
            )

            # Step 4: Generate links to related episodes
            # Find candidate episodes via semantic search
            candidates = await self.episodic.search(
                memory_note.content, limit=20
            )
            # Filter out the just-created episode
            candidates = [e for e in candidates if e.id != episode_id]

            # Generate links
            link_specs = await self.curator.generate_links(episode, candidates)
            for target_id, link_type, weight in link_specs:
                link = EpisodeLink(
                    id=generate_link_id(),
                    source_id=episode_id,
                    target_id=target_id,
                    link_type=link_type,
                    weight=weight,
                    created_at=utc_now(),
                )
                await self.episodic.add_link(link)
                result.links_created.append(link.id)

            logger.info(f"Created {len(link_specs)} links")

            # Step 5: Trigger evolution
            if candidates:
                updates = await self.curator.trigger_evolution(
                    episode, candidates[:5]
                )
                result.metadata["evolution_updates"] = len(updates)
                logger.debug(f"Triggered {len(updates)} evolution updates")

            # Step 6: Resolve conflicts (for facts)
            if facts:
                # Get existing facts for the same entities
                existing_facts = []
                for fact in facts:
                    entity_facts = await self.semantic.query_entity_facts(
                        fact.subject_id
                    )
                    existing_facts.extend(entity_facts)

                # Resolve conflicts
                resolution = await self.curator.resolve_conflicts(
                    facts, existing_facts
                )

                # Apply invalidations
                for fact_id in resolution.get("invalidations", []):
                    await self.semantic.invalidate_fact(fact_id, utc_now())

                result.metadata["conflicts_resolved"] = len(
                    resolution.get("invalidations", [])
                )

        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            raise MemoryError(f"Failed to store interaction: {e}") from e

        finally:
            result.processing_time = time.time() - start_time

        logger.info(
            f"Stored interaction in {result.processing_time:.2f}s: "
            f"episode={result.episode_id}, entities={len(result.entities_created)}, "
            f"facts={len(result.facts_created)}, links={len(result.links_created)}"
        )

        return result

    # ========== Context Management ==========

    async def inject_context(self, result: RetrievalResult) -> None:
        """Add retrieved memories to working memory.

        Args:
            result: Retrieval result to inject into working memory.
        """
        items = []

        # Add facts
        for fact in result.facts:
            relevance = result.relevance_scores.get(fact.id, 0.5)
            content = f"{fact.subject_id} {fact.predicate} {fact.object_id or fact.object_value}"
            items.append(
                MemoryItem(
                    content=content,
                    source="semantic",
                    relevance=relevance,
                    metadata={"fact_id": fact.id},
                )
            )

        # Add episodes
        for episode in result.episodes:
            relevance = result.relevance_scores.get(episode.id, 0.5)
            content = episode.summary or episode.content
            items.append(
                MemoryItem(
                    content=content,
                    source=f"episodic_L{episode.level}",
                    relevance=relevance,
                    metadata={
                        "episode_id": episode.id,
                        "salience": episode.salience,
                    },
                )
            )

        self.working.add_retrieved(items)
        logger.debug(f"Injected {len(items)} items into working memory")

    async def handle_memory_pressure(self) -> None:
        """Coordinate pressure handling with curator.

        This method handles memory pressure by:
        1. Evicting items from working memory
        2. Storing important evicted items
        3. Coordinating with curator for what to persist
        """
        logger.info("Handling memory pressure")

        # Get items to persist before eviction
        items_to_persist = await self.working.handle_pressure()

        # Store important items
        for item in items_to_persist:
            # Only store items with sufficient relevance
            if item.relevance < 0.5:
                continue

            # Create an interaction from the memory item
            interaction = Interaction(
                user_message=item.content,
                assistant_message="",  # No response for evicted context
                context=f"Evicted from working memory: {item.source}",
                metadata=item.metadata,
            )

            # Store with curator evaluation
            await self.store_interaction(interaction)

        logger.info(
            f"Handled memory pressure: persisted {len(items_to_persist)} items"
        )

    # ========== Utility Methods ==========

    async def get_related_context(
        self, episode_id: EpisodeID, depth: int = 1
    ) -> list[Episode]:
        """Get episodes related via links.

        Args:
            episode_id: ID of episode to find related context for.
            depth: How many link hops to traverse (default: 1).

        Returns:
            List of related episodes.

        Raises:
            MemoryError: If retrieval fails.
        """
        try:
            visited = {episode_id}
            related = []

            current_layer = [episode_id]

            for _ in range(depth):
                next_layer = []

                for current_id in current_layer:
                    # Get linked episodes
                    linked = await self.episodic.get_linked_episodes(current_id)

                    for episode in linked:
                        if episode.id not in visited:
                            visited.add(episode.id)
                            related.append(episode)
                            next_layer.append(episode.id)

                current_layer = next_layer

            logger.debug(
                f"Found {len(related)} related episodes at depth {depth}"
            )
            return related

        except Exception as e:
            logger.error(f"Failed to get related context: {e}")
            raise MemoryError(f"Failed to get related context: {e}") from e

    # ========== Helper Methods ==========

    def _deduplicate_results(self, result: RetrievalResult) -> RetrievalResult:
        """Deduplicate overlapping results.

        Args:
            result: Result to deduplicate.

        Returns:
            Deduplicated result.
        """
        # Deduplicate facts by ID
        seen_facts = set()
        unique_facts = []
        for fact in result.facts:
            if fact.id not in seen_facts:
                seen_facts.add(fact.id)
                unique_facts.append(fact)

        # Deduplicate episodes by ID
        seen_episodes = set()
        unique_episodes = []
        for episode in result.episodes:
            if episode.id not in seen_episodes:
                seen_episodes.add(episode.id)
                unique_episodes.append(episode)

        result.facts = unique_facts
        result.episodes = unique_episodes

        return result

    def _rank_results(self, result: RetrievalResult) -> RetrievalResult:
        """Rank results by relevance and recency.

        Args:
            result: Result to rank.

        Returns:
            Ranked result.
        """
        # Rank facts by relevance score
        result.facts.sort(
            key=lambda f: result.relevance_scores.get(f.id, 0.0), reverse=True
        )

        # Rank episodes by combined relevance and salience
        result.episodes.sort(
            key=lambda e: (
                result.relevance_scores.get(e.id, 0.0) * 0.7 + e.salience * 0.3
            ),
            reverse=True,
        )

        return result
