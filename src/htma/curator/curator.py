"""Memory curator for salience evaluation and memory formation.

This module implements the memory curator (LLM2), which handles:
- Salience evaluation (what's worth remembering)
- Entity and fact extraction
- Conflict resolution
- Memory evolution

Note: This is a basic stub for Phase 2. Full implementation in Phase 3 (Issues #10-14).
"""

import logging
from typing import Any

from htma.core.types import (
    Entity,
    Episode,
    Fact,
    Interaction,
    MemoryNote,
)
from htma.llm.client import OllamaClient

logger = logging.getLogger(__name__)


class MemoryCurator:
    """Memory curator for evaluating and processing memories.

    The curator acts as LLM2 in the HTMA architecture, responsible for:
    - Evaluating what information is worth remembering (salience)
    - Extracting entities and facts from interactions
    - Resolving conflicts between new and existing memories
    - Triggering memory evolution when new information arrives

    This is a basic stub implementation. Full functionality will be added in Phase 3.

    Attributes:
        llm: LLM client for curator operations.
        model: Model name to use for curator operations.
    """

    def __init__(self, llm: OllamaClient, model: str = "mistral:7b"):
        """Initialize memory curator.

        Args:
            llm: LLM client for curator operations.
            model: Model name to use (default: mistral:7b).
        """
        self.llm = llm
        self.model = model
        logger.info(f"Initialized MemoryCurator with model {model}")

    async def evaluate_salience(
        self, interaction: Interaction, context: str = ""
    ) -> MemoryNote:
        """Evaluate if interaction is worth remembering and create memory note.

        This method determines the importance of an interaction and generates
        a memory note if it's worth storing.

        Args:
            interaction: The interaction to evaluate.
            context: Additional context for evaluation.

        Returns:
            MemoryNote with salience score and extracted information.

        Note:
            This is a stub implementation. Full implementation in Issue #10.
            Currently returns a basic memory note with moderate salience.
        """
        logger.debug("Evaluating salience for interaction (stub implementation)")

        # Stub: Create a basic memory note
        # Full implementation will use LLM to evaluate salience
        content = f"User: {interaction.user_message}\nAssistant: {interaction.assistant_message}"

        # Simple heuristic: longer interactions are more salient
        salience = min(0.5 + len(content) / 2000, 1.0)

        # Extract simple keywords (stub)
        words = content.lower().split()
        keywords = list(set(w for w in words if len(w) > 5))[:5]

        return MemoryNote(
            content=content,
            context=context,
            keywords=keywords,
            tags=["interaction"],
            salience=salience,
        )

    async def extract_entities(
        self, interaction: Interaction
    ) -> list[Entity]:
        """Extract entities from interaction.

        Args:
            interaction: The interaction to extract entities from.

        Returns:
            List of extracted entities.

        Note:
            This is a stub implementation. Full implementation in Issue #11.
            Currently returns an empty list.
        """
        logger.debug("Extracting entities (stub implementation)")
        # Stub: Return empty list
        # Full implementation will use LLM to extract entities
        return []

    async def extract_facts(
        self, interaction: Interaction, entities: list[Entity]
    ) -> list[Fact]:
        """Extract facts from interaction given entities.

        Args:
            interaction: The interaction to extract facts from.
            entities: Entities to use for fact extraction.

        Returns:
            List of extracted facts.

        Note:
            This is a stub implementation. Full implementation in Issue #11.
            Currently returns an empty list.
        """
        logger.debug("Extracting facts (stub implementation)")
        # Stub: Return empty list
        # Full implementation will use LLM to extract facts
        return []

    async def generate_links(
        self, episode: Episode, candidate_episodes: list[Episode]
    ) -> list[tuple[str, str, float]]:
        """Generate links between new episode and existing episodes.

        Args:
            episode: The new episode to link.
            candidate_episodes: Candidate episodes to link to.

        Returns:
            List of tuples (target_episode_id, link_type, weight).

        Note:
            This is a stub implementation. Full implementation in Issue #12.
            Currently returns an empty list.
        """
        logger.debug("Generating links (stub implementation)")
        # Stub: Return empty list
        # Full implementation will use LLM to evaluate connections
        return []

    async def resolve_conflicts(
        self, new_facts: list[Fact], existing_facts: list[Fact]
    ) -> dict[str, Any]:
        """Resolve conflicts between new and existing facts.

        Args:
            new_facts: New facts to check for conflicts.
            existing_facts: Existing facts to check against.

        Returns:
            Dictionary with resolution actions (invalidations, updates, etc.).

        Note:
            This is a stub implementation. Full implementation in Issue #13.
            Currently returns an empty resolution.
        """
        logger.debug("Resolving conflicts (stub implementation)")
        # Stub: Return empty resolution
        # Full implementation will use LLM to resolve conflicts
        return {
            "invalidations": [],
            "confidence_updates": [],
            "new_facts": new_facts,
        }

    async def trigger_evolution(
        self, new_episode: Episode, related_episodes: list[Episode]
    ) -> list[dict[str, Any]]:
        """Trigger evolution of existing memories based on new episode.

        Args:
            new_episode: The new episode that may trigger updates.
            related_episodes: Related episodes that may be updated.

        Returns:
            List of update dictionaries for existing episodes.

        Note:
            This is a stub implementation. Full implementation in Issue #14.
            Currently returns an empty list.
        """
        logger.debug("Triggering evolution (stub implementation)")
        # Stub: Return empty list
        # Full implementation will use LLM to evaluate evolution opportunities
        return []
