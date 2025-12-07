"""Memory curator for salience evaluation and memory formation.

This module implements the memory curator (LLM2), which handles:
- Salience evaluation (what's worth remembering)
- Entity and fact extraction
- Conflict resolution
- Memory evolution

Note: This is a basic stub for Phase 2. Full implementation in Phase 3 (Issues #10-14).
"""

import json
import logging
from pathlib import Path
from typing import Any

from htma.core.exceptions import LLMResponseError
from htma.core.types import (
    Entity,
    Episode,
    Fact,
    Interaction,
    SalienceResult,
)
from htma.llm.client import OllamaClient

logger = logging.getLogger(__name__)

# Path to prompt templates
PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts" / "curator"


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

    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from file.

        Args:
            template_name: Name of the template file (e.g., "salience.txt")

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        template_path = PROMPTS_DIR / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        return template_path.read_text()

    async def evaluate_salience(
        self, content: str, context: str = ""
    ) -> SalienceResult:
        """Evaluate if content is worth remembering.

        This method uses the LLM to determine the importance of content
        and classify what type of memory (semantic, episodic, or both) it should be.

        Args:
            content: The content to evaluate (e.g., interaction text).
            context: Additional context for evaluation (e.g., conversation history).

        Returns:
            SalienceResult with score, reasoning, memory type, and key elements.

        Raises:
            LLMResponseError: If LLM fails to return valid JSON or returns invalid data.

        Note:
            Salience thresholds:
            - 0.0-0.3: Don't store (trivial, ephemeral)
            - 0.3-0.6: Store minimal (somewhat useful)
            - 0.6-0.8: Store standard (important facts)
            - 0.8-1.0: Store rich (critical information)
        """
        logger.debug("Evaluating salience for content")

        # Handle edge cases
        if not content or not content.strip():
            logger.warning("Empty content provided for salience evaluation")
            return SalienceResult(
                score=0.0,
                reasoning="Content is empty",
                memory_type="episodic",
                key_elements=[],
            )

        # Very long content - truncate for evaluation but note it
        max_length = 4000
        truncated = False
        if len(content) > max_length:
            content = content[:max_length] + "..."
            truncated = True
            logger.debug(f"Content truncated to {max_length} characters for evaluation")

        # Load and format prompt
        try:
            prompt_template = self._load_prompt_template("salience.txt")
            prompt = prompt_template.format(context=context, content=content)
        except Exception as e:
            logger.error(f"Failed to load/format prompt template: {e}")
            raise LLMResponseError(f"Failed to load prompt template: {e}") from e

        # Get LLM evaluation
        try:
            response = await self.llm.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=500,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMResponseError(f"Failed to generate salience evaluation: {e}") from e

        # Parse JSON response
        try:
            # Try to extract JSON from response (handle cases where LLM adds extra text)
            response = response.strip()

            # Find JSON object in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Validate required fields
            if not all(key in data for key in ["score", "reasoning", "memory_type"]):
                raise ValueError(
                    "Missing required fields (score, reasoning, memory_type) in response"
                )

            # Create result
            result = SalienceResult(
                score=float(data["score"]),
                reasoning=data["reasoning"],
                memory_type=data["memory_type"],
                key_elements=data.get("key_elements", []),
            )

            # If content was truncated, adjust score slightly downward
            if truncated and result.score > 0.3:
                result.score = max(0.3, result.score - 0.1)
                result.reasoning += " (Note: Content was truncated for evaluation)"

            logger.info(
                f"Salience evaluation complete: score={result.score:.2f}, "
                f"type={result.memory_type}"
            )
            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}\nResponse: {response}")
            raise LLMResponseError(
                f"LLM returned invalid JSON response: {e}\nResponse: {response[:200]}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error parsing salience result: {e}")
            raise LLMResponseError(f"Failed to parse salience result: {e}") from e

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
