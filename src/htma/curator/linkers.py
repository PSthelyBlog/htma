"""Memory link generation for connecting related episodes.

This module implements the linking component that identifies and creates
connections between new memories and existing ones, enabling associative
memory retrieval.
"""

import json
import logging
import os
from typing import Any

from htma.core.exceptions import LLMResponseError
from htma.core.types import Episode, EpisodeID, EpisodeLink, LinkEvaluation
from htma.core.utils import generate_link_id
from htma.llm.client import OllamaClient
from htma.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class LinkGenerator:
    """Generates links between memory episodes.

    The LinkGenerator identifies connections between new episodes and existing
    memories by:
    1. Finding candidate episodes via semantic search
    2. Using LLM to evaluate connection strength and type
    3. Creating weighted bidirectional links

    Link types:
    - semantic: Similar topics/concepts
    - temporal: Close in time or part of sequence
    - causal: One led to or caused another
    - analogical: Similar patterns/structures

    Attributes:
        llm: Ollama client for LLM operations.
        model: Model name to use for link evaluation.
        episodic: Episodic memory store for retrieval.
        prompt_template: Template for link evaluation prompt.
    """

    def __init__(
        self,
        llm: OllamaClient,
        model: str,
        episodic: EpisodicMemory,
    ):
        """Initialize link generator.

        Args:
            llm: Ollama client instance.
            model: Model name for link evaluation (e.g., "mistral:7b").
            episodic: Episodic memory instance.
        """
        self.llm = llm
        self.model = model
        self.episodic = episodic
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load link evaluation prompt template.

        Returns:
            Prompt template string.

        Raises:
            FileNotFoundError: If template file doesn't exist.
        """
        # Get the path to the prompt template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(
            current_dir, "..", "llm", "prompts", "curator", "link_evaluation.txt"
        )
        prompt_path = os.path.normpath(prompt_path)

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError as e:
            logger.error(f"Link evaluation prompt template not found at {prompt_path}")
            raise FileNotFoundError(
                f"Link evaluation prompt template not found: {prompt_path}"
            ) from e

    async def generate_links(
        self,
        new_episode: Episode,
        candidate_limit: int = 20,
    ) -> list[EpisodeLink]:
        """Find and create links to existing memories.

        Process:
        1. Semantic search for candidate episodes
        2. LLM evaluates each candidate for connection strength
        3. Creates links for candidates that should be linked

        Args:
            new_episode: The new episode to link.
            candidate_limit: Maximum number of candidates to evaluate.

        Returns:
            List of created episode links.

        Raises:
            Exception: If semantic search or evaluation fails.
        """
        logger.info(
            f"Generating links for episode {new_episode.id} (level {new_episode.level})"
        )

        # Step 1: Find candidate episodes via semantic search
        # Use the episode's summary if available, otherwise content
        query_text = new_episode.summary or new_episode.content

        try:
            candidates = await self.episodic.search(
                query=query_text,
                level=new_episode.level,  # Search within same level
                limit=candidate_limit,
            )
        except Exception as e:
            logger.warning(f"Semantic search failed for link generation: {e}")
            # Fall back to recent episodes if search fails
            candidates = await self.episodic.get_recent(
                level=new_episode.level,
                limit=candidate_limit
            )

        # Filter out the new episode itself if it appears in results
        candidates = [c for c in candidates if c.id != new_episode.id]

        if not candidates:
            logger.info(f"No candidate episodes found for linking to {new_episode.id}")
            return []

        logger.info(
            f"Evaluating {len(candidates)} candidate episodes for linking"
        )

        # Step 2: Evaluate each candidate
        links: list[EpisodeLink] = []
        for candidate in candidates:
            try:
                evaluation = await self.evaluate_connection(new_episode, candidate)

                if evaluation.should_link:
                    # Create the link
                    link = EpisodeLink(
                        id=generate_link_id(),
                        source_id=new_episode.id,
                        target_id=candidate.id,
                        link_type=evaluation.link_type,
                        weight=evaluation.weight,
                    )

                    # Add link to episodic memory
                    await self.episodic.add_link(link)
                    links.append(link)

                    logger.info(
                        f"Created {evaluation.link_type} link between "
                        f"{new_episode.id} and {candidate.id} "
                        f"(weight: {evaluation.weight:.2f}): {evaluation.reasoning}"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to evaluate link between {new_episode.id} "
                    f"and {candidate.id}: {e}"
                )
                continue

        logger.info(
            f"Created {len(links)} links for episode {new_episode.id}"
        )
        return links

    async def evaluate_connection(
        self,
        episode_a: Episode,
        episode_b: Episode,
    ) -> LinkEvaluation:
        """Evaluate if two episodes should be linked.

        Uses LLM to determine connection type, strength, and whether
        the episodes should be linked at all.

        Args:
            episode_a: First episode.
            episode_b: Second episode.

        Returns:
            LinkEvaluation with assessment details.

        Raises:
            LLMResponseError: If LLM response is invalid or can't be parsed.
        """
        # Format prompt with episode details
        prompt = self.prompt_template.format(
            episode_a_id=episode_a.id,
            episode_a_occurred=episode_a.occurred_at.isoformat(),
            episode_a_content=episode_a.content[:500],  # Truncate for context window
            episode_a_summary=episode_a.summary or "N/A",
            episode_a_keywords=", ".join(episode_a.keywords) if episode_a.keywords else "N/A",
            episode_a_tags=", ".join(episode_a.tags) if episode_a.tags else "N/A",
            episode_b_id=episode_b.id,
            episode_b_occurred=episode_b.occurred_at.isoformat(),
            episode_b_content=episode_b.content[:500],  # Truncate for context window
            episode_b_summary=episode_b.summary or "N/A",
            episode_b_keywords=", ".join(episode_b.keywords) if episode_b.keywords else "N/A",
            episode_b_tags=", ".join(episode_b.tags) if episode_b.tags else "N/A",
        )

        # Get LLM evaluation
        response = await self.llm.generate(
            model=self.model,
            prompt=prompt,
            temperature=0.3,  # Lower temperature for more consistent evaluation
            max_tokens=500,
        )

        # Parse JSON response
        try:
            result = self._parse_link_response(response)
        except Exception as e:
            logger.error(f"Failed to parse link evaluation response: {e}")
            logger.debug(f"Raw response: {response}")
            raise LLMResponseError(
                f"Failed to parse link evaluation response: {e}"
            ) from e

        return result

    def _parse_link_response(self, response: str) -> LinkEvaluation:
        """Parse LLM response into LinkEvaluation.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed LinkEvaluation.

        Raises:
            ValueError: If response format is invalid.
            json.JSONDecodeError: If JSON is malformed.
        """
        # Clean response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        # Validate required fields
        if "should_link" not in data:
            raise ValueError("Missing 'should_link' field in response")
        if "link_type" not in data:
            raise ValueError("Missing 'link_type' field in response")
        if "weight" not in data:
            raise ValueError("Missing 'weight' field in response")
        if "reasoning" not in data:
            raise ValueError("Missing 'reasoning' field in response")

        # Validate link type
        valid_types = {"semantic", "temporal", "causal", "analogical"}
        if data["link_type"] not in valid_types:
            logger.warning(
                f"Invalid link type '{data['link_type']}', defaulting to 'semantic'"
            )
            data["link_type"] = "semantic"

        # Validate weight range
        weight = float(data["weight"])
        if weight < 0.0 or weight > 1.0:
            logger.warning(
                f"Weight {weight} out of range [0.0, 1.0], clamping"
            )
            weight = max(0.0, min(1.0, weight))
            data["weight"] = weight

        # Create LinkEvaluation
        return LinkEvaluation(
            should_link=bool(data["should_link"]),
            link_type=data["link_type"],
            weight=weight,
            reasoning=data["reasoning"],
            metadata={},
        )
