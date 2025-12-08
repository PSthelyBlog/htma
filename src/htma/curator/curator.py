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

from htma.core.exceptions import ConflictResolutionError, LLMResponseError
from htma.core.types import (
    ConflictResolution,
    Entity,
    Episode,
    Fact,
    FactConflict,
    Interaction,
    SalienceResult,
)
from htma.core.utils import utc_now
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

    async def detect_conflicts(
        self, new_fact: Fact, semantic_memory: Any
    ) -> list[FactConflict]:
        """Find existing facts that conflict with a new fact.

        A conflict occurs when:
        1. Same subject and predicate but different object/value
        2. Both facts are currently valid (not invalidated)
        3. Facts have overlapping event time validity

        Args:
            new_fact: New fact to check for conflicts.
            semantic_memory: SemanticMemory instance to query existing facts.

        Returns:
            List of FactConflict objects representing detected conflicts.

        Raises:
            DatabaseError: If querying semantic memory fails.
        """
        logger.debug(
            f"Detecting conflicts for fact: {new_fact.subject_id} "
            f"{new_fact.predicate} {new_fact.object_id or new_fact.object_value}"
        )

        conflicts = []

        try:
            # Query existing facts with same subject and predicate
            existing_facts = await semantic_memory.query_entity_facts(
                entity_id=new_fact.subject_id, predicate=new_fact.predicate
            )

            # Check each existing fact for conflicts
            for existing_fact in existing_facts:
                # Skip if already invalidated
                if existing_fact.temporal.transaction_time.valid_to is not None:
                    continue

                # Check if objects/values differ (potential conflict)
                objects_differ = False
                if new_fact.object_id and existing_fact.object_id:
                    objects_differ = new_fact.object_id != existing_fact.object_id
                elif new_fact.object_value and existing_fact.object_value:
                    objects_differ = new_fact.object_value != existing_fact.object_value
                elif (new_fact.object_id and existing_fact.object_value) or (
                    new_fact.object_value and existing_fact.object_id
                ):
                    objects_differ = True

                if objects_differ:
                    # This is a potential conflict
                    conflict = FactConflict(
                        new_fact=new_fact,
                        conflicting_facts=[existing_fact],
                        conflict_type="contradiction",
                    )
                    conflicts.append(conflict)
                    logger.info(
                        f"Detected conflict: {new_fact.id} conflicts with {existing_fact.id}"
                    )

            if not conflicts:
                logger.debug("No conflicts detected")

        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            raise

        return conflicts

    async def resolve_conflict(
        self, new_fact: Fact, existing_facts: list[Fact]
    ) -> ConflictResolution:
        """Resolve contradiction between new and existing facts.

        Uses the LLM to evaluate the conflict and determine the best resolution
        strategy based on confidence, temporal context, and the nature of the facts.

        Strategies:
        1. temporal_succession: Old fact was true, now new fact is true
        2. confidence_adjustment: Uncertain which is correct, lower confidence
        3. coexistence: Both can be true in different contexts
        4. rejection: New fact is likely wrong

        Args:
            new_fact: The new fact being added.
            existing_facts: Existing facts that conflict with the new fact.

        Returns:
            ConflictResolution with strategy and actions to take.

        Raises:
            LLMResponseError: If LLM fails to return valid response.
            ConflictResolutionError: If resolution fails.
        """
        logger.debug(
            f"Resolving conflict for new fact {new_fact.id} "
            f"against {len(existing_facts)} existing facts"
        )

        # Load and format prompt
        try:
            prompt_template = self._load_prompt_template("conflict_resolution.txt")

            # Format existing facts for prompt
            existing_facts_str = ""
            for i, fact in enumerate(existing_facts, 1):
                existing_facts_str += f"\nFact {i} (ID: {fact.id}):\n"
                existing_facts_str += f"  Subject: {fact.subject_id}\n"
                existing_facts_str += f"  Predicate: {fact.predicate}\n"
                existing_facts_str += (
                    f"  Object: {fact.object_id or fact.object_value}\n"
                )
                existing_facts_str += f"  Confidence: {fact.confidence}\n"
                existing_facts_str += f"  Recorded: {fact.temporal.transaction_time.valid_from}\n"
                if fact.source_episode_id:
                    existing_facts_str += f"  Source: {fact.source_episode_id}\n"

            prompt = prompt_template.format(
                new_subject=new_fact.subject_id,
                new_predicate=new_fact.predicate,
                new_object=new_fact.object_id or new_fact.object_value,
                new_confidence=new_fact.confidence,
                new_source=new_fact.source_episode_id or "unknown",
                existing_facts=existing_facts_str,
            )
        except Exception as e:
            logger.error(f"Failed to load/format conflict resolution prompt: {e}")
            raise LLMResponseError(
                f"Failed to load conflict resolution prompt: {e}"
            ) from e

        # Get LLM evaluation
        try:
            response = await self.llm.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.2,  # Lower temperature for consistent decision-making
                max_tokens=1000,
            )
        except Exception as e:
            logger.error(f"LLM generation failed during conflict resolution: {e}")
            raise LLMResponseError(
                f"Failed to generate conflict resolution: {e}"
            ) from e

        # Parse JSON response
        try:
            # Extract JSON from response
            response = response.strip()
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Validate required fields
            required_fields = ["strategy", "reasoning", "new_fact_accepted"]
            if not all(key in data for key in required_fields):
                raise ValueError(f"Missing required fields in response: {required_fields}")

            # Validate strategy
            valid_strategies = [
                "temporal_succession",
                "confidence_adjustment",
                "coexistence",
                "rejection",
            ]
            strategy = data["strategy"]
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}"
                )

            # Build resolution result
            invalidations = []
            confidence_updates = []
            result_fact = new_fact if data["new_fact_accepted"] else None

            # Process invalidations
            if "invalidate_facts" in data and data["invalidate_facts"]:
                now = utc_now()
                for fact_id in data["invalidate_facts"]:
                    invalidations.append((fact_id, now))

            # Process confidence updates
            if "confidence_updates" in data and data["confidence_updates"]:
                for fact_id, new_confidence in data["confidence_updates"].items():
                    confidence_updates.append((fact_id, float(new_confidence)))

            # Apply modifications to new fact if specified
            if result_fact and "new_fact_modifications" in data:
                mods = data["new_fact_modifications"]
                if "confidence" in mods:
                    result_fact.confidence = float(mods["confidence"])
                if "metadata" in mods and isinstance(mods["metadata"], dict):
                    result_fact.metadata.update(mods["metadata"])

            resolution = ConflictResolution(
                strategy=strategy,
                invalidations=invalidations,
                confidence_updates=confidence_updates,
                new_fact=result_fact,
                reasoning=data["reasoning"],
                metadata={"llm_response": data},
            )

            logger.info(
                f"Conflict resolved with strategy '{strategy}': {data['reasoning'][:100]}"
            )
            return resolution

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(
                f"Failed to parse conflict resolution response: {e}\nResponse: {response}"
            )
            raise LLMResponseError(
                f"LLM returned invalid conflict resolution response: {e}\n"
                f"Response: {response[:200]}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during conflict resolution: {e}")
            raise ConflictResolutionError(
                new_fact.id, [f.id for f in existing_facts]
            ) from e

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
