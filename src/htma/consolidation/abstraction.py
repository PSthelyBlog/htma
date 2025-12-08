"""Abstraction generation for episodic memory consolidation.

This module implements the AbstractionGenerator that creates higher-level summaries
from clusters of lower-level episodes, following the RAPTOR hierarchical approach.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from htma.core.exceptions import ConsolidationError, LLMResponseError
from htma.core.types import Episode, EpisodeID
from htma.core.utils import generate_episode_id, utc_now
from htma.llm.client import OllamaClient

logger = logging.getLogger(__name__)


class AbstractionGenerator:
    """Generates hierarchical abstractions from episodes.

    Creates higher-level summaries by clustering related episodes and
    generating consolidated narratives using an LLM.

    Attributes:
        llm: Ollama client for LLM operations.
        model: Model name to use for summarization.
    """

    def __init__(self, llm: OllamaClient, model: str = "mistral:7b"):
        """Initialize abstraction generator.

        Args:
            llm: Ollama client instance.
            model: Model name for summarization (default: mistral:7b).
        """
        self.llm = llm
        self.model = model

    async def cluster_episodes(
        self,
        episodes: list[Episode],
        cluster_size: int = 5,
        temporal_window_hours: int = 168,  # 1 week
    ) -> list[list[Episode]]:
        """Group related episodes for summarization.

        Uses embedding similarity and temporal proximity to create clusters.
        Episodes that are too far apart in time won't be clustered together.

        Args:
            episodes: List of episodes to cluster.
            cluster_size: Target number of episodes per cluster.
            temporal_window_hours: Maximum time gap between episodes in a cluster.

        Returns:
            List of episode clusters, where each cluster is a list of related episodes.

        Raises:
            ConsolidationError: If clustering fails.
        """
        if not episodes:
            return []

        if len(episodes) == 1:
            return [episodes]

        # Sort episodes by occurred_at (temporal ordering)
        sorted_episodes = sorted(episodes, key=lambda e: e.occurred_at)

        try:
            # Get embeddings for all episodes
            texts = [self._get_episode_text(ep) for ep in sorted_episodes]
            embeddings = await self.llm.embed_batch(self.model, texts)

            # Create clusters using temporal windowing + similarity
            clusters = []
            current_cluster: list[Episode] = []
            current_embeddings: list[list[float]] = []

            for i, episode in enumerate(sorted_episodes):
                embedding = embeddings[i]

                # Check if we should start a new cluster
                should_start_new = False

                if not current_cluster:
                    # First episode - start new cluster
                    current_cluster.append(episode)
                    current_embeddings.append(embedding)
                else:
                    # Check temporal proximity
                    time_gap = (
                        episode.occurred_at - current_cluster[-1].occurred_at
                    )
                    if time_gap > timedelta(hours=temporal_window_hours):
                        should_start_new = True

                    # Check cluster size
                    elif len(current_cluster) >= cluster_size:
                        should_start_new = True

                    # Check semantic similarity (average with cluster)
                    elif len(current_cluster) > 0:
                        avg_similarity = self._average_similarity(
                            embedding, current_embeddings
                        )
                        # If similarity is too low, start new cluster
                        if avg_similarity < 0.3:
                            should_start_new = True

                    if should_start_new:
                        # Save current cluster and start new one
                        if current_cluster:
                            clusters.append(current_cluster)
                        current_cluster = [episode]
                        current_embeddings = [embedding]
                    else:
                        # Add to current cluster
                        current_cluster.append(episode)
                        current_embeddings.append(embedding)

            # Add final cluster
            if current_cluster:
                clusters.append(current_cluster)

            logger.info(
                f"Clustered {len(episodes)} episodes into {len(clusters)} clusters"
            )
            return clusters

        except Exception as e:
            raise ConsolidationError(f"Failed to cluster episodes: {e}") from e

    async def generate_summary(
        self, episodes: list[Episode], level: int
    ) -> Episode:
        """Create Level N+1 episode from Level N episodes.

        Generates a consolidated summary that captures key themes and
        maintains temporal ordering while preserving important information.

        Args:
            episodes: List of episodes to summarize (should be from same level).
            level: Target abstraction level for the new summary episode.

        Returns:
            New episode at level N+1 summarizing the input episodes.

        Raises:
            ConsolidationError: If summary generation fails.
            ValueError: If episodes list is empty or has mixed levels.
        """
        if not episodes:
            raise ValueError("Cannot generate summary from empty episode list")

        # Validate all episodes are from the same level
        source_level = episodes[0].level
        if not all(ep.level == source_level for ep in episodes):
            raise ValueError(
                f"All episodes must be from same level, got mixed levels"
            )

        if level <= source_level:
            raise ValueError(
                f"Target level {level} must be higher than source level {source_level}"
            )

        try:
            # Sort episodes by time
            sorted_episodes = sorted(episodes, key=lambda e: e.occurred_at)

            # Build prompt for summarization
            prompt = self._build_summary_prompt(sorted_episodes, level)

            # Generate summary using LLM
            response = await self.llm.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory consolidation assistant that creates concise, "
                        "meaningful summaries of episodic memories. Preserve key information, "
                        "identify themes, maintain temporal ordering, and note significance.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for consistent summaries
                max_tokens=2000,
            )

            # Parse response
            summary_data = self._parse_summary_response(response)

            # Combine keywords and tags from all source episodes
            all_keywords = set()
            all_tags = set()
            for ep in sorted_episodes:
                all_keywords.update(ep.keywords)
                all_tags.update(ep.tags)

            # Add new keywords/tags from LLM response
            all_keywords.update(summary_data.get("keywords", []))
            all_tags.update(summary_data.get("tags", []))

            # Calculate average salience
            avg_salience = sum(ep.salience for ep in sorted_episodes) / len(
                sorted_episodes
            )

            # Create new summary episode
            summary_episode = Episode(
                id=generate_episode_id(),
                level=level,
                parent_id=None,  # Summary episodes don't have a single parent
                content=summary_data["content"],
                summary=summary_data.get("summary"),
                context_description=summary_data.get("context_description"),
                keywords=sorted(list(all_keywords)),
                tags=sorted(list(all_tags)),
                occurred_at=sorted_episodes[0].occurred_at,  # Start time
                recorded_at=utc_now(),
                salience=max(avg_salience, summary_data.get("salience", avg_salience)),
                consolidation_strength=10.0,  # Higher strength for summaries
                metadata={
                    "source_episode_ids": [ep.id for ep in sorted_episodes],
                    "source_level": source_level,
                    "num_episodes_summarized": len(sorted_episodes),
                    "time_span_hours": (
                        sorted_episodes[-1].occurred_at - sorted_episodes[0].occurred_at
                    ).total_seconds()
                    / 3600,
                },
            )

            logger.info(
                f"Generated Level {level} summary from {len(episodes)} Level {source_level} episodes"
            )
            return summary_episode

        except LLMResponseError as e:
            raise ConsolidationError(f"LLM failed to generate summary: {e}") from e
        except Exception as e:
            raise ConsolidationError(f"Failed to generate summary: {e}") from e

    async def should_abstract(
        self,
        episodes: list[Episode],
        min_episodes: int = 3,
        min_age_hours: int = 24,
        min_coherence: float = 0.3,
    ) -> bool:
        """Evaluate if episodes are ready for abstraction.

        Checks criteria including sufficient count, age, and semantic coherence.

        Args:
            episodes: List of episodes to evaluate.
            min_episodes: Minimum number of episodes required.
            min_age_hours: Minimum age in hours for oldest episode.
            min_coherence: Minimum semantic coherence threshold (0.0-1.0).

        Returns:
            True if episodes should be abstracted, False otherwise.
        """
        if not episodes:
            return False

        # Check count criterion
        if len(episodes) < min_episodes:
            logger.debug(
                f"Not enough episodes for abstraction: {len(episodes)} < {min_episodes}"
            )
            return False

        # Check age criterion
        now = utc_now()
        oldest = min(ep.recorded_at for ep in episodes)
        age = (now - oldest).total_seconds() / 3600  # Convert to hours
        if age < min_age_hours:
            logger.debug(
                f"Episodes not old enough for abstraction: {age:.1f}h < {min_age_hours}h"
            )
            return False

        # Check semantic coherence if we have more than one episode
        if len(episodes) > 1:
            try:
                # Get embeddings for all episodes
                texts = [self._get_episode_text(ep) for ep in episodes]
                embeddings = await self.llm.embed_batch(self.model, texts)

                # Calculate pairwise similarities
                coherence = self._calculate_coherence(embeddings)

                if coherence < min_coherence:
                    logger.debug(
                        f"Episodes lack semantic coherence: {coherence:.2f} < {min_coherence}"
                    )
                    return False

                logger.debug(
                    f"Episodes ready for abstraction: count={len(episodes)}, "
                    f"age={age:.1f}h, coherence={coherence:.2f}"
                )
            except Exception as e:
                logger.warning(f"Failed to check coherence, defaulting to True: {e}")
                # If coherence check fails, use other criteria
                return True

        return True

    # ========== Helper Methods ==========

    def _get_episode_text(self, episode: Episode) -> str:
        """Extract text representation of episode for embedding.

        Args:
            episode: Episode to convert to text.

        Returns:
            Text representation combining content and metadata.
        """
        parts = [episode.content]

        if episode.summary:
            parts.append(f"Summary: {episode.summary}")

        if episode.context_description:
            parts.append(f"Context: {episode.context_description}")

        if episode.keywords:
            parts.append(f"Keywords: {', '.join(episode.keywords)}")

        return " | ".join(parts)

    def _average_similarity(
        self, embedding: list[float], cluster_embeddings: list[list[float]]
    ) -> float:
        """Calculate average cosine similarity between embedding and cluster.

        Args:
            embedding: Target embedding vector.
            cluster_embeddings: List of embeddings in the cluster.

        Returns:
            Average similarity score (0.0-1.0).
        """
        if not cluster_embeddings:
            return 0.0

        similarities = [
            self._cosine_similarity(embedding, cluster_emb)
            for cluster_emb in cluster_embeddings
        ]
        return sum(similarities) / len(similarities)

    def _cosine_similarity(
        self, vec1: list[float], vec2: list[float]
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score (-1.0 to 1.0, normalized to 0.0-1.0).
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)
        # Normalize from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def _calculate_coherence(self, embeddings: list[list[float]]) -> float:
        """Calculate semantic coherence across a set of embeddings.

        Coherence is measured as the average pairwise similarity.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Coherence score (0.0-1.0).
        """
        if len(embeddings) < 2:
            return 1.0  # Single episode is perfectly coherent

        # Calculate all pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _build_summary_prompt(
        self, episodes: list[Episode], target_level: int
    ) -> str:
        """Build prompt for LLM summarization.

        Args:
            episodes: Episodes to summarize.
            target_level: Target abstraction level.

        Returns:
            Formatted prompt string.
        """
        episodes_text = []
        for i, ep in enumerate(episodes, 1):
            occurred = ep.occurred_at.strftime("%Y-%m-%d %H:%M")
            episodes_text.append(
                f"Episode {i} (Level {ep.level}, occurred: {occurred}):\n{ep.content}"
            )
            if ep.summary:
                episodes_text.append(f"  Summary: {ep.summary}")
            if ep.keywords:
                episodes_text.append(f"  Keywords: {', '.join(ep.keywords[:10])}")

        episodes_str = "\n\n".join(episodes_text)

        prompt = f"""You are consolidating {len(episodes)} Level {episodes[0].level} episodes into a Level {target_level} summary.

EPISODES TO SUMMARIZE:
{episodes_str}

Create a comprehensive summary that:
1. Captures the key themes and patterns across these episodes
2. Preserves important factual information
3. Maintains temporal context (what happened when)
4. Identifies significance and relationships
5. Is concise but informative

Respond with JSON in this format:
{{
  "content": "The consolidated narrative summarizing all episodes...",
  "summary": "A one-sentence overview of this abstraction",
  "context_description": "The broader context or theme connecting these episodes",
  "keywords": ["key", "terms", "from", "episodes"],
  "tags": ["theme1", "theme2"],
  "salience": 0.0-1.0 (how important is this summary overall)
}}

Ensure your response is valid JSON."""

        return prompt

    def _parse_summary_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response for summary generation.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed summary data as dictionary.

        Raises:
            LLMResponseError: If response cannot be parsed.
        """
        try:
            # Try to find JSON in the response
            # LLM might wrap it in markdown code blocks
            response = response.strip()

            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.replace("```json", "").replace("```", "").strip()

            data = json.loads(response)

            # Validate required fields
            if "content" not in data:
                raise ValueError("Missing required field: content")

            # Ensure salience is valid
            if "salience" in data:
                data["salience"] = max(0.0, min(1.0, float(data["salience"])))

            # Ensure lists are present
            data.setdefault("keywords", [])
            data.setdefault("tags", [])

            return data

        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse JSON response: {e}") from e
        except (ValueError, KeyError) as e:
            raise LLMResponseError(f"Invalid summary response format: {e}") from e
