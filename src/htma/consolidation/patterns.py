"""Pattern detection for episodic memory consolidation.

This module implements the PatternDetector that identifies recurring themes,
behaviors, and preferences across episodes, tracking their evolution over time.
"""

import json
import logging
from typing import Any

from htma.core.exceptions import ConsolidationError, LLMResponseError
from htma.core.types import Episode, EpisodeID, Pattern, PatternDetectionResult
from htma.core.utils import utc_now
from htma.llm.client import OllamaClient

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detects and tracks recurring patterns across episodes.

    Identifies behavioral patterns, preferences, procedures, and common errors
    by analyzing episodes using LLM-based pattern recognition. Tracks pattern
    lifecycle from emerging to established to consolidated.

    Attributes:
        llm: Ollama client for LLM operations.
        model: Model name to use for pattern detection.
    """

    def __init__(self, llm: OllamaClient, model: str = "mistral:7b"):
        """Initialize pattern detector.

        Args:
            llm: Ollama client instance.
            model: Model name for pattern detection (default: mistral:7b).
        """
        self.llm = llm
        self.model = model

    async def detect_patterns(
        self,
        episodes: list[Episode],
        existing_patterns: list[Pattern],
        min_occurrences: int = 2,
        min_similarity: float = 0.7,
    ) -> PatternDetectionResult:
        """Identify patterns in episode collection.

        Analyzes episodes to find recurring themes and patterns, checking against
        existing patterns to strengthen or weaken them.

        Args:
            episodes: List of episodes to analyze for patterns.
            existing_patterns: Previously detected patterns to check against.
            min_occurrences: Minimum occurrences required to detect a pattern.
            min_similarity: Minimum similarity threshold for pattern matching.

        Returns:
            PatternDetectionResult with new, strengthened, and weakened patterns.

        Raises:
            ConsolidationError: If pattern detection fails.
        """
        if not episodes:
            return PatternDetectionResult()

        if len(episodes) < min_occurrences:
            logger.debug(
                f"Not enough episodes for pattern detection: {len(episodes)} < {min_occurrences}"
            )
            return PatternDetectionResult()

        try:
            # Extract candidate patterns from episodes
            candidate_patterns = await self._extract_candidates(episodes)

            if not candidate_patterns:
                logger.info("No candidate patterns found in episodes")
                return PatternDetectionResult()

            # Check candidates against existing patterns
            new_patterns: list[Pattern] = []
            strengthened: list[tuple[str, float]] = []
            pattern_episode_map: dict[str, set[EpisodeID]] = {
                p.id: set(p.occurrences) for p in existing_patterns
            }

            for candidate in candidate_patterns:
                # Try to match with existing patterns
                matched_pattern = await self.match_to_existing(
                    candidate, existing_patterns, min_similarity
                )

                if matched_pattern:
                    # Strengthen existing pattern
                    new_episodes = set(candidate.occurrences) - pattern_episode_map[
                        matched_pattern.id
                    ]
                    if new_episodes:
                        # Calculate new confidence (increase by 0.1 per new occurrence)
                        new_confidence = min(
                            1.0,
                            matched_pattern.confidence
                            + (len(new_episodes) * 0.1),
                        )

                        strengthened.append((matched_pattern.id, new_confidence))
                        pattern_episode_map[matched_pattern.id].update(new_episodes)

                        logger.info(
                            f"Strengthened pattern '{matched_pattern.id}' with {len(new_episodes)} new episodes"
                        )
                else:
                    # Check if candidate has enough occurrences to be a new pattern
                    if len(candidate.occurrences) >= min_occurrences:
                        new_patterns.append(candidate)
                        pattern_episode_map[candidate.id] = set(candidate.occurrences)
                        logger.info(
                            f"Detected new pattern '{candidate.id}' with {len(candidate.occurrences)} occurrences"
                        )

            # Weaken patterns not seen in recent episodes
            weakened = await self._weaken_unseen_patterns(
                existing_patterns,
                episodes,
                pattern_episode_map,
            )

            result = PatternDetectionResult(
                new_patterns=new_patterns,
                strengthened=strengthened,
                weakened=weakened,
                metadata={
                    "episodes_analyzed": len(episodes),
                    "candidates_found": len(candidate_patterns),
                    "new_patterns_detected": len(new_patterns),
                    "patterns_strengthened": len(strengthened),
                    "patterns_weakened": len(weakened),
                },
            )

            logger.info(
                f"Pattern detection complete: {len(new_patterns)} new, "
                f"{len(strengthened)} strengthened, {len(weakened)} weakened"
            )
            return result

        except Exception as e:
            raise ConsolidationError(f"Failed to detect patterns: {e}") from e

    async def extract_pattern(
        self, episodes: list[Episode], pattern_type: str | None = None
    ) -> Pattern | None:
        """Extract a single pattern from related episodes.

        Uses LLM to analyze episodes and extract a meaningful pattern description.

        Args:
            episodes: Related episodes to analyze.
            pattern_type: Optional hint for pattern type (behavioral, preference, procedural, error).

        Returns:
            Extracted Pattern or None if no pattern found.

        Raises:
            ConsolidationError: If pattern extraction fails.
        """
        if not episodes:
            return None

        try:
            # Build prompt for pattern extraction
            prompt = self._build_extraction_prompt(episodes, pattern_type)

            # Generate pattern using LLM
            response = await self.llm.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory analyst that identifies recurring patterns "
                        "in user behavior, preferences, and procedures. Extract meaningful patterns "
                        "that would be useful for understanding the user's habits and tendencies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for consistent analysis
                max_tokens=1000,
            )

            # Parse response
            pattern_data = self._parse_extraction_response(response)

            if not pattern_data or not pattern_data.get("has_pattern"):
                return None

            # Create pattern object
            now = utc_now()
            pattern = Pattern(
                id=f"pat_{int(now.timestamp() * 1000)}_{hash(pattern_data['description']) % 10000:04d}",
                description=pattern_data["description"],
                pattern_type=pattern_data["pattern_type"],
                confidence=pattern_data.get("confidence", 0.5),
                occurrences=[ep.id for ep in episodes],
                first_seen=min(ep.occurred_at for ep in episodes),
                last_seen=max(ep.occurred_at for ep in episodes),
                consolidation_strength=self._calculate_strength(
                    len(episodes), pattern_data.get("confidence", 0.5)
                ),
                metadata={
                    "keywords": pattern_data.get("keywords", []),
                    "evidence": pattern_data.get("evidence", []),
                    "num_episodes": len(episodes),
                },
            )

            logger.info(
                f"Extracted {pattern.pattern_type} pattern: '{pattern.description}'"
            )
            return pattern

        except LLMResponseError as e:
            raise ConsolidationError(f"LLM failed to extract pattern: {e}") from e
        except Exception as e:
            raise ConsolidationError(f"Failed to extract pattern: {e}") from e

    async def match_to_existing(
        self,
        candidate: Pattern,
        existing: list[Pattern],
        min_similarity: float = 0.7,
    ) -> Pattern | None:
        """Check if candidate matches an existing pattern.

        Uses semantic similarity to determine if a candidate pattern is similar
        enough to an existing pattern to be considered the same.

        Args:
            candidate: Candidate pattern to match.
            existing: List of existing patterns.
            min_similarity: Minimum similarity threshold (0.0-1.0).

        Returns:
            Matching existing pattern or None if no match found.
        """
        if not existing:
            return None

        try:
            # Filter existing patterns by type (only compare same types)
            same_type_patterns = [
                p for p in existing if p.pattern_type == candidate.pattern_type
            ]

            if not same_type_patterns:
                return None

            # Get embeddings for candidate and existing patterns
            candidate_text = f"{candidate.description}"
            existing_texts = [p.description for p in same_type_patterns]
            all_texts = [candidate_text] + existing_texts

            embeddings = await self.llm.embed_batch(self.model, all_texts)
            candidate_embedding = embeddings[0]
            existing_embeddings = embeddings[1:]

            # Find most similar existing pattern
            best_similarity = 0.0
            best_match = None

            for i, existing_pattern in enumerate(same_type_patterns):
                similarity = self._cosine_similarity(
                    candidate_embedding, existing_embeddings[i]
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = existing_pattern

            # Return match if similarity is above threshold
            if best_similarity >= min_similarity and best_match:
                logger.debug(
                    f"Matched candidate to existing pattern '{best_match.id}' "
                    f"(similarity: {best_similarity:.2f})"
                )
                return best_match

            logger.debug(
                f"No match found for candidate (best similarity: {best_similarity:.2f})"
            )
            return None

        except Exception as e:
            logger.warning(f"Failed to match pattern, treating as new: {e}")
            return None

    # ========== Helper Methods ==========

    async def _extract_candidates(
        self, episodes: list[Episode]
    ) -> list[Pattern]:
        """Extract candidate patterns from episodes.

        Groups episodes by similarity and extracts patterns from each group.

        Args:
            episodes: Episodes to analyze.

        Returns:
            List of candidate patterns.
        """
        candidates: list[Pattern] = []

        # Sort episodes by time
        sorted_episodes = sorted(episodes, key=lambda e: e.occurred_at)

        # Extract pattern from all episodes together
        pattern = await self.extract_pattern(sorted_episodes)
        if pattern:
            candidates.append(pattern)

        # Also try to find specific pattern types
        for pattern_type in ["behavioral", "preference", "procedural", "error"]:
            pattern = await self.extract_pattern(sorted_episodes, pattern_type)
            if pattern:
                # Check if this is distinct from already found patterns
                is_distinct = True
                for existing in candidates:
                    if self._descriptions_similar(
                        pattern.description, existing.description
                    ):
                        is_distinct = False
                        break

                if is_distinct:
                    candidates.append(pattern)

        return candidates

    async def _weaken_unseen_patterns(
        self,
        existing_patterns: list[Pattern],
        recent_episodes: list[Episode],
        pattern_episode_map: dict[str, set[EpisodeID]],
    ) -> list[tuple[str, float]]:
        """Weaken patterns not observed in recent episodes.

        Args:
            existing_patterns: All existing patterns.
            recent_episodes: Recently analyzed episodes.
            pattern_episode_map: Mapping of pattern IDs to episode IDs.

        Returns:
            List of (pattern_id, new_confidence) for weakened patterns.
        """
        weakened: list[tuple[str, float]] = []
        recent_episode_ids = {ep.id for ep in recent_episodes}

        for pattern in existing_patterns:
            # Check if pattern appeared in recent episodes
            pattern_episodes = pattern_episode_map.get(pattern.id, set())
            appears_in_recent = bool(pattern_episodes & recent_episode_ids)

            if not appears_in_recent:
                # Pattern not seen, reduce confidence slightly
                new_confidence = max(0.1, pattern.confidence - 0.05)
                if new_confidence < pattern.confidence:
                    weakened.append((pattern.id, new_confidence))
                    logger.debug(
                        f"Weakened pattern '{pattern.id}' (not seen in recent episodes)"
                    )

        return weakened

    def _calculate_strength(self, num_occurrences: int, confidence: float) -> float:
        """Calculate consolidation strength based on occurrences and confidence.

        Args:
            num_occurrences: Number of times pattern observed.
            confidence: Pattern confidence score.

        Returns:
            Consolidation strength value.
        """
        # Base strength increases with occurrences (logarithmic)
        base_strength = 5.0 + (2.0 * (num_occurrences ** 0.5))

        # Adjust by confidence
        adjusted_strength = base_strength * confidence

        return min(20.0, adjusted_strength)  # Cap at 20.0

    def _build_extraction_prompt(
        self, episodes: list[Episode], pattern_type: str | None = None
    ) -> str:
        """Build prompt for pattern extraction.

        Args:
            episodes: Episodes to analyze.
            pattern_type: Optional pattern type hint.

        Returns:
            Formatted prompt string.
        """
        episodes_text = []
        for i, ep in enumerate(episodes, 1):
            occurred = ep.occurred_at.strftime("%Y-%m-%d %H:%M")
            episodes_text.append(
                f"Episode {i} ({occurred}):\n{ep.content}"
            )
            if ep.keywords:
                episodes_text.append(f"  Keywords: {', '.join(ep.keywords[:10])}")

        episodes_str = "\n\n".join(episodes_text)

        type_hint = ""
        if pattern_type:
            type_hint = f"\nFocus on identifying {pattern_type} patterns."

        prompt = f"""Analyze these {len(episodes)} episodes to identify recurring patterns.

EPISODES:
{episodes_str}

Pattern Types:
- behavioral: User tends to do X (e.g., "User exercises every morning")
- preference: User prefers X over Y (e.g., "User prefers dark roast coffee")
- procedural: Steps for accomplishing X (e.g., "User's morning routine: wake up, exercise, breakfast")
- error: Common mistake pattern (e.g., "User often forgets to save files before closing")
{type_hint}

Look for:
1. Recurring behaviors or actions
2. Consistent preferences or choices
3. Repeated procedures or workflows
4. Common mistakes or patterns

Respond with JSON:
{{
  "has_pattern": true/false,
  "description": "Clear, concise description of the pattern",
  "pattern_type": "behavioral|preference|procedural|error",
  "confidence": 0.0-1.0 (how confident are you in this pattern),
  "keywords": ["relevant", "keywords"],
  "evidence": ["Episode 1: shows X", "Episode 3: shows X again"]
}}

If no clear pattern exists, respond with {{"has_pattern": false}}.
Ensure your response is valid JSON."""

        return prompt

    def _parse_extraction_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response for pattern extraction.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed pattern data as dictionary.

        Raises:
            LLMResponseError: If response cannot be parsed.
        """
        try:
            # Try to find JSON in the response
            response = response.strip()

            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.replace("```json", "").replace("```", "").strip()

            data = json.loads(response)

            # Validate has_pattern field
            if not isinstance(data.get("has_pattern"), bool):
                raise ValueError("Missing or invalid 'has_pattern' field")

            # If no pattern, return early
            if not data["has_pattern"]:
                return data

            # Validate required fields for pattern
            required = ["description", "pattern_type"]
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Validate pattern_type
            valid_types = ["behavioral", "preference", "procedural", "error"]
            if data["pattern_type"] not in valid_types:
                raise ValueError(
                    f"Invalid pattern_type: {data['pattern_type']} (must be one of {valid_types})"
                )

            # Ensure confidence is valid
            if "confidence" in data:
                data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
            else:
                data["confidence"] = 0.5

            # Ensure lists are present
            data.setdefault("keywords", [])
            data.setdefault("evidence", [])

            return data

        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Failed to parse JSON response: {e}") from e
        except (ValueError, KeyError) as e:
            raise LLMResponseError(f"Invalid pattern response format: {e}") from e

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score (0.0-1.0).
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)
        # Normalize from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def _descriptions_similar(self, desc1: str, desc2: str) -> bool:
        """Check if two pattern descriptions are similar.

        Simple string-based similarity check for deduplication.

        Args:
            desc1: First description.
            desc2: Second description.

        Returns:
            True if descriptions are similar enough to be considered duplicates.
        """
        # Normalize
        d1 = desc1.lower().strip()
        d2 = desc2.lower().strip()

        # Exact match
        if d1 == d2:
            return True

        # Check if one is substring of other
        if d1 in d2 or d2 in d1:
            return True

        # Check significant word overlap
        words1 = set(d1.split())
        words2 = set(d2.split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        min_words = min(len(words1), len(words2))

        # Consider similar if >70% of smaller set overlaps
        return overlap / min_words > 0.7
