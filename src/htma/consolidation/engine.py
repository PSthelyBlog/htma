"""Consolidation engine for memory evolution and maintenance.

This module implements the ConsolidationEngine that orchestrates memory evolution
processes including link maintenance, abstraction generation, and pattern detection.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from htma.consolidation.abstraction import AbstractionGenerator
from htma.consolidation.patterns import PatternDetector
from htma.core.exceptions import ConsolidationError, DatabaseError
from htma.core.types import (
    ConsolidationReport,
    Episode,
    EpisodeID,
    LinkMaintenanceReport,
    Pattern,
    PruneReport,
)
from htma.core.utils import utc_now
from htma.curator.curator import MemoryCurator
from htma.memory.episodic import EpisodicMemory
from htma.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation cycle behavior.

    Attributes:
        min_episodes_before_cycle: Minimum number of new episodes before triggering cycle.
        max_time_between_cycles: Maximum time between cycles (forces cycle if exceeded).
        abstraction_cluster_size: Target number of episodes to cluster for abstraction.
        pattern_min_occurrences: Minimum occurrences required to detect a pattern.
        prune_access_threshold: Episodes with access_count below this are candidates for pruning.
        prune_age_threshold: Episodes older than this are candidates for pruning.
        max_episodes_per_cycle: Maximum episodes to process in a single cycle.
    """

    min_episodes_before_cycle: int = 10
    max_time_between_cycles: timedelta = timedelta(hours=24)
    abstraction_cluster_size: int = 5
    pattern_min_occurrences: int = 3
    prune_access_threshold: int = 0
    prune_age_threshold: timedelta = timedelta(days=30)
    max_episodes_per_cycle: int = 100


class ConsolidationEngine:
    """Orchestrates memory consolidation and evolution processes.

    The consolidation engine manages:
    - Link strengthening and decay based on access patterns
    - Pruning of weak links
    - Abstraction generation from episode clusters
    - Pattern detection across experiences
    - Conflict resolution in semantic memory
    - Stale memory pruning

    Attributes:
        curator: Memory curator for conflict resolution and extraction.
        semantic: Semantic memory instance for fact operations.
        episodic: Episodic memory instance for episode operations.
        abstraction_generator: Generator for creating hierarchical abstractions.
        pattern_detector: Detector for identifying recurring patterns.
        config: Configuration for consolidation behavior.
        last_cycle: Timestamp of the last consolidation cycle.
    """

    def __init__(
        self,
        curator: MemoryCurator,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        abstraction_generator: AbstractionGenerator,
        pattern_detector: PatternDetector,
        config: ConsolidationConfig | None = None,
    ):
        """Initialize consolidation engine.

        Args:
            curator: Memory curator instance.
            semantic: Semantic memory instance.
            episodic: Episodic memory instance.
            abstraction_generator: Abstraction generator instance.
            pattern_detector: Pattern detector instance.
            config: Configuration for consolidation (uses defaults if None).
        """
        self.curator = curator
        self.semantic = semantic
        self.episodic = episodic
        self.abstraction_generator = abstraction_generator
        self.pattern_detector = pattern_detector
        self.config = config or ConsolidationConfig()
        self.last_cycle: datetime | None = None

    async def update_link_weights(
        self,
        access_window: timedelta = timedelta(hours=1),
        decay_rate: float = 0.1,
        min_weight: float = 0.1,
        prune_threshold: float = 0.1,
    ) -> LinkMaintenanceReport:
        """Update link weights based on access patterns.

        Performs a complete link maintenance cycle:
        1. Strengthen co-accessed links
        2. Decay unused links
        3. Prune links below threshold

        Args:
            access_window: Time window for co-access detection (default: 1 hour).
            decay_rate: Rate of decay for unused links (default: 0.1).
            min_weight: Minimum weight after decay (default: 0.1).
            prune_threshold: Weight threshold for pruning (default: 0.1).

        Returns:
            LinkMaintenanceReport with statistics about the maintenance cycle.

        Raises:
            ConsolidationError: If link maintenance fails.
        """
        start_time = utc_now()

        try:
            # Count total links before maintenance
            total_before = await self._count_total_links()

            logger.info(
                f"Starting link maintenance cycle (total links: {total_before})"
            )

            # Step 1: Strengthen co-accessed links
            strengthened = await self.strengthen_coaccessed(access_window)

            # Step 2: Decay unused links
            decayed = await self.decay_unused(decay_rate, min_weight)

            # Step 3: Prune weak links
            pruned = await self.prune_weak_links(prune_threshold)

            # Count total links after maintenance
            total_after = await self._count_total_links()

            # Calculate processing time
            processing_time = (utc_now() - start_time).total_seconds()

            report = LinkMaintenanceReport(
                links_strengthened=strengthened,
                links_decayed=decayed,
                links_pruned=pruned,
                total_links_before=total_before,
                total_links_after=total_after,
                processing_time=processing_time,
                metadata={
                    "access_window_hours": access_window.total_seconds() / 3600,
                    "decay_rate": decay_rate,
                    "min_weight": min_weight,
                    "prune_threshold": prune_threshold,
                },
            )

            logger.info(
                f"Link maintenance complete: {strengthened} strengthened, "
                f"{decayed} decayed, {pruned} pruned "
                f"(total: {total_before} -> {total_after}) "
                f"in {processing_time:.2f}s"
            )

            return report

        except Exception as e:
            raise ConsolidationError(f"Link maintenance failed: {e}") from e

    async def strengthen_coaccessed(
        self, access_window: timedelta = timedelta(hours=1)
    ) -> int:
        """Links between episodes accessed together get stronger.

        When Episode A and Episode B are accessed within a time window,
        any existing link between them is strengthened. Strengthening
        uses logarithmic scaling to provide diminishing returns.

        Args:
            access_window: Time window for considering accesses as co-occurring.

        Returns:
            Number of links strengthened.

        Raises:
            DatabaseError: If database operations fail.
        """
        try:
            # Get all episodes accessed in the recent period
            # We look back 2x the window to catch co-access patterns
            lookback_time = utc_now() - (access_window * 2)

            query = """
                SELECT id, last_accessed
                FROM episodes
                WHERE last_accessed IS NOT NULL
                  AND last_accessed >= ?
                ORDER BY last_accessed ASC
            """
            params = (lookback_time.isoformat(),)

            rows = await self.episodic.sqlite.fetch_all(query, params)

            if len(rows) < 2:
                logger.debug(
                    f"Not enough recently accessed episodes for co-access detection: {len(rows)}"
                )
                return 0

            # Build list of (episode_id, access_time) tuples
            accesses = [
                (row["id"], datetime.fromisoformat(row["last_accessed"]))
                for row in rows
            ]

            # Find co-accessed pairs (accessed within window of each other)
            coaccessed_pairs: set[tuple[str, str]] = set()

            for i in range(len(accesses)):
                episode_a, time_a = accesses[i]
                for j in range(i + 1, len(accesses)):
                    episode_b, time_b = accesses[j]

                    # Check if within window
                    time_diff = abs((time_b - time_a).total_seconds())
                    if time_diff <= access_window.total_seconds():
                        # Create normalized pair (always smaller ID first)
                        pair = tuple(sorted([episode_a, episode_b]))
                        coaccessed_pairs.add(pair)

            if not coaccessed_pairs:
                logger.debug("No co-accessed episode pairs found")
                return 0

            # For each co-accessed pair, strengthen any existing link
            strengthened_count = 0

            for episode_a, episode_b in coaccessed_pairs:
                # Check if link exists between these episodes
                link_query = """
                    SELECT id, weight
                    FROM episode_links
                    WHERE (source_id = ? AND target_id = ?)
                       OR (source_id = ? AND target_id = ?)
                """
                link_params = (episode_a, episode_b, episode_b, episode_a)
                link_row = await self.episodic.sqlite.fetch_one(
                    link_query, link_params
                )

                if link_row:
                    # Link exists - strengthen it
                    current_weight = link_row["weight"]

                    # Logarithmic strengthening: diminishing returns as weight increases
                    # Formula: strength_increase = 0.1 * log(2 + current_weight)
                    # This gives ~0.1 for weight=1, ~0.08 for weight=5, ~0.06 for weight=10
                    strength_increase = 0.1 * math.log(2 + current_weight)
                    new_weight = min(
                        10.0, current_weight + strength_increase
                    )  # Cap at 10.0

                    # Update the link weight
                    update_query = """
                        UPDATE episode_links
                        SET weight = ?
                        WHERE id = ?
                    """
                    update_params = (new_weight, link_row["id"])
                    await self.episodic.sqlite.execute(update_query, update_params)

                    strengthened_count += 1

                    logger.debug(
                        f"Strengthened link between {episode_a} and {episode_b}: "
                        f"{current_weight:.2f} -> {new_weight:.2f}"
                    )

            logger.info(
                f"Strengthened {strengthened_count} links from {len(coaccessed_pairs)} co-accessed pairs"
            )
            return strengthened_count

        except Exception as e:
            raise DatabaseError(f"Failed to strengthen co-accessed links: {e}") from e

    async def decay_unused(
        self, decay_rate: float = 0.1, min_weight: float = 0.1
    ) -> int:
        """Links not used decay over time.

        Each consolidation cycle, all links are decayed by the decay rate.
        Frequently used links are strengthened faster than decay, so they
        grow stronger. Unused links eventually fall below the pruning threshold.

        Args:
            decay_rate: Fraction to reduce weight by (0.0-1.0). Default: 0.1 (10% reduction).
            min_weight: Minimum weight after decay (prevents going to zero).

        Returns:
            Number of links that were decayed.

        Raises:
            DatabaseError: If database operations fail.
        """
        if not (0.0 <= decay_rate <= 1.0):
            raise ValueError(f"Decay rate must be between 0.0 and 1.0, got {decay_rate}")

        if min_weight < 0.0:
            raise ValueError(f"Min weight must be >= 0.0, got {min_weight}")

        try:
            # Get all links
            query = "SELECT id, weight FROM episode_links"
            rows = await self.episodic.sqlite.fetch_all(query, ())

            if not rows:
                logger.debug("No links to decay")
                return 0

            decayed_count = 0

            # Decay each link
            for row in rows:
                current_weight = row["weight"]

                # Apply decay: new_weight = current_weight * (1 - decay_rate)
                # But ensure it doesn't go below min_weight
                decayed_weight = current_weight * (1 - decay_rate)
                new_weight = max(min_weight, decayed_weight)

                # Only update if weight actually changed
                if new_weight != current_weight:
                    update_query = "UPDATE episode_links SET weight = ? WHERE id = ?"
                    update_params = (new_weight, row["id"])
                    await self.episodic.sqlite.execute(update_query, update_params)

                    decayed_count += 1

                    logger.debug(
                        f"Decayed link {row['id']}: {current_weight:.2f} -> {new_weight:.2f}"
                    )

            logger.info(f"Decayed {decayed_count} links (rate: {decay_rate})")
            return decayed_count

        except Exception as e:
            raise DatabaseError(f"Failed to decay links: {e}") from e

    async def prune_weak_links(self, threshold: float = 0.1) -> int:
        """Remove links below weight threshold.

        Weak links that have decayed below the threshold are removed from
        the graph to reduce clutter and improve retrieval performance.

        Args:
            threshold: Weight threshold below which links are pruned.

        Returns:
            Number of links pruned.

        Raises:
            DatabaseError: If database operations fail.
        """
        if threshold < 0.0:
            raise ValueError(f"Threshold must be >= 0.0, got {threshold}")

        try:
            # Delete links below threshold
            query = "DELETE FROM episode_links WHERE weight < ?"
            params = (threshold,)

            # Get count before deletion
            count_query = "SELECT COUNT(*) as count FROM episode_links WHERE weight < ?"
            count_row = await self.episodic.sqlite.fetch_one(count_query, params)
            pruned_count = count_row["count"] if count_row else 0

            if pruned_count > 0:
                # Execute deletion
                await self.episodic.sqlite.execute(query, params)
                logger.info(
                    f"Pruned {pruned_count} weak links (threshold: {threshold})"
                )
            else:
                logger.debug(f"No links below threshold {threshold} to prune")

            return pruned_count

        except Exception as e:
            raise DatabaseError(f"Failed to prune weak links: {e}") from e

    # ========== Full Consolidation Cycle ==========

    async def should_run(self) -> bool:
        """Check if consolidation should run.

        Consolidation triggers when:
        1. Sufficient new episodes have accumulated (min_episodes_before_cycle)
        2. Time since last cycle exceeded (max_time_between_cycles)
        3. Never run before

        Returns:
            True if consolidation should run, False otherwise.
        """
        # If never run before, check if we have enough episodes
        if self.last_cycle is None:
            query = "SELECT COUNT(*) as count FROM episodes WHERE level = 0"
            row = await self.episodic.sqlite.fetch_one(query, ())
            count = row["count"] if row else 0
            if count >= self.config.min_episodes_before_cycle:
                logger.info(
                    f"Consolidation should run: {count} episodes (no previous cycle)"
                )
                return True
            return False

        # Check if max time exceeded
        time_since_last = utc_now() - self.last_cycle
        if time_since_last >= self.config.max_time_between_cycles:
            logger.info(
                f"Consolidation should run: {time_since_last.total_seconds() / 3600:.1f}h since last cycle"
            )
            return True

        # Check if enough new episodes since last cycle
        query = """
            SELECT COUNT(*) as count FROM episodes
            WHERE level = 0 AND recorded_at > ?
        """
        params = (self.last_cycle.isoformat(),)
        row = await self.episodic.sqlite.fetch_one(query, params)
        new_episodes = row["count"] if row else 0

        if new_episodes >= self.config.min_episodes_before_cycle:
            logger.info(
                f"Consolidation should run: {new_episodes} new episodes since last cycle"
            )
            return True

        logger.debug(
            f"Consolidation not needed: {new_episodes} new episodes, "
            f"{time_since_last.total_seconds() / 3600:.1f}h since last cycle"
        )
        return False

    async def run_cycle(self) -> ConsolidationReport:
        """Run full consolidation cycle.

        Full consolidation cycle:

        1. Generate abstractions
           - Get unconsolidated Level 0 episodes
           - Cluster and summarize into Level 1
           - Recursively up the hierarchy

        2. Detect patterns
           - Analyze recent episodes for patterns
           - Strengthen existing patterns with new evidence
           - Create new patterns if threshold met

        3. Resolve contradictions
           - Scan semantic memory for conflicts (future)
           - Apply resolution strategies

        4. Maintain links
           - Strengthen co-accessed links
           - Decay unused links
           - Prune weak links

        5. Prune stale content
           - Archive/delete old, unaccessed memories
           - Respect consolidation_strength

        6. Update metadata
           - Record cycle completion
           - Update statistics

        Returns:
            ConsolidationReport with statistics about all operations.

        Raises:
            ConsolidationError: If consolidation cycle fails.
        """
        start_time = utc_now()
        logger.info("Starting consolidation cycle...")

        try:
            # Initialize counters
            abstractions_created = 0
            patterns_detected = 0
            patterns_strengthened = 0
            conflicts_resolved = 0
            links_strengthened = 0
            links_pruned = 0
            episodes_pruned = 0

            # Step 1: Generate abstractions
            logger.info("Step 1/5: Generating abstractions...")
            abstractions_created = await self._generate_abstractions()

            # Step 2: Detect patterns
            logger.info("Step 2/5: Detecting patterns...")
            pattern_result = await self._detect_patterns()
            patterns_detected = len(pattern_result.new_patterns)
            patterns_strengthened = len(pattern_result.strengthened)

            # Step 3: Resolve contradictions
            logger.info("Step 3/5: Resolving contradictions...")
            # For now, conflicts are resolved during fact insertion
            # Future: scan semantic memory for latent conflicts
            conflicts_resolved = 0

            # Step 4: Maintain links
            logger.info("Step 4/5: Maintaining links...")
            link_report = await self.update_link_weights()
            links_strengthened = link_report.links_strengthened
            links_pruned = link_report.links_pruned

            # Step 5: Prune stale content
            logger.info("Step 5/5: Pruning stale content...")
            prune_report = await self.prune_stale()
            episodes_pruned = prune_report.episodes_pruned

            # Step 6: Update metadata
            self.last_cycle = utc_now()
            duration = (self.last_cycle - start_time).total_seconds()

            report = ConsolidationReport(
                abstractions_created=abstractions_created,
                patterns_detected=patterns_detected,
                patterns_strengthened=patterns_strengthened,
                conflicts_resolved=conflicts_resolved,
                links_strengthened=links_strengthened,
                links_pruned=links_pruned,
                episodes_pruned=episodes_pruned,
                duration=duration,
                metadata={
                    "cycle_completed_at": self.last_cycle.isoformat(),
                    "config": {
                        "min_episodes_before_cycle": self.config.min_episodes_before_cycle,
                        "max_time_between_cycles": self.config.max_time_between_cycles.total_seconds(),
                        "abstraction_cluster_size": self.config.abstraction_cluster_size,
                        "pattern_min_occurrences": self.config.pattern_min_occurrences,
                    },
                },
            )

            logger.info(
                f"Consolidation cycle complete in {duration:.2f}s: "
                f"{abstractions_created} abstractions, "
                f"{patterns_detected} patterns detected, "
                f"{patterns_strengthened} patterns strengthened, "
                f"{links_strengthened} links strengthened, "
                f"{links_pruned} links pruned, "
                f"{episodes_pruned} episodes pruned"
            )

            return report

        except Exception as e:
            raise ConsolidationError(f"Consolidation cycle failed: {e}") from e

    async def prune_stale(self) -> PruneReport:
        """Remove or archive stale memories.

        Criteria for pruning:
        - access_count below threshold
        - age above threshold
        - consolidation_strength below threshold
        - Already consolidated into higher level

        Level 0 episodes that have been consolidated into higher levels and
        are rarely accessed can be pruned. Higher level episodes are only
        pruned if they have very low access counts and are very old.

        Returns:
            PruneReport with statistics about pruned memories.

        Raises:
            DatabaseError: If pruning operations fail.
        """
        start_time = utc_now()

        try:
            # Count total episodes before pruning
            total_before = await self._count_total_episodes()

            # Calculate age threshold
            age_cutoff = utc_now() - self.config.prune_age_threshold

            # Find episodes eligible for pruning
            # Criteria:
            # 1. Level 0 episodes that are old AND have low access count AND have parent (consolidated)
            # 2. Any episode with consolidation_strength = 0 (explicitly marked for removal)
            query = """
                SELECT id FROM episodes
                WHERE (
                    (level = 0
                     AND parent_id IS NOT NULL
                     AND access_count <= ?
                     AND occurred_at < ?)
                    OR consolidation_strength <= 0
                )
                ORDER BY consolidation_strength ASC, access_count ASC
                LIMIT ?
            """
            params = (
                self.config.prune_access_threshold,
                age_cutoff.isoformat(),
                self.config.max_episodes_per_cycle,
            )

            rows = await self.episodic.sqlite.fetch_all(query, params)
            episode_ids = [row["id"] for row in rows]

            if not episode_ids:
                logger.info("No episodes eligible for pruning")
                return PruneReport(
                    episodes_pruned=0,
                    links_pruned=0,
                    total_episodes_before=total_before,
                    total_episodes_after=total_before,
                    processing_time=0.0,
                )

            # Count links that will be pruned
            placeholders = ",".join(["?"] * len(episode_ids))
            link_count_query = f"""
                SELECT COUNT(*) as count FROM episode_links
                WHERE source_id IN ({placeholders})
                   OR target_id IN ({placeholders})
            """
            link_count_params = episode_ids + episode_ids
            link_row = await self.episodic.sqlite.fetch_one(
                link_count_query, tuple(link_count_params)
            )
            links_to_prune = link_row["count"] if link_row else 0

            # Delete links first (foreign key constraints)
            if links_to_prune > 0:
                delete_links_query = f"""
                    DELETE FROM episode_links
                    WHERE source_id IN ({placeholders})
                       OR target_id IN ({placeholders})
                """
                await self.episodic.sqlite.execute(
                    delete_links_query, tuple(link_count_params)
                )

            # Delete episodes
            delete_episodes_query = f"DELETE FROM episodes WHERE id IN ({placeholders})"
            await self.episodic.sqlite.execute(
                delete_episodes_query, tuple(episode_ids)
            )

            # Delete from vector store
            for episode_id in episode_ids:
                try:
                    await self.episodic.chroma.delete_episode(episode_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to delete episode {episode_id} from vector store: {e}"
                    )

            # Count total episodes after pruning
            total_after = await self._count_total_episodes()

            # Calculate processing time
            processing_time = (utc_now() - start_time).total_seconds()

            report = PruneReport(
                episodes_pruned=len(episode_ids),
                links_pruned=links_to_prune,
                total_episodes_before=total_before,
                total_episodes_after=total_after,
                processing_time=processing_time,
                metadata={
                    "age_threshold_days": self.config.prune_age_threshold.days,
                    "access_threshold": self.config.prune_access_threshold,
                },
            )

            logger.info(
                f"Pruned {len(episode_ids)} episodes and {links_to_prune} links "
                f"(total: {total_before} -> {total_after}) "
                f"in {processing_time:.2f}s"
            )

            return report

        except Exception as e:
            raise DatabaseError(f"Failed to prune stale memories: {e}") from e

    # ========== Helper Methods ==========

    async def _count_total_links(self) -> int:
        """Count total number of links in the graph.

        Returns:
            Total number of episode links.
        """
        query = "SELECT COUNT(*) as count FROM episode_links"
        row = await self.episodic.sqlite.fetch_one(query, ())
        return row["count"] if row else 0

    async def _count_total_episodes(self) -> int:
        """Count total number of episodes.

        Returns:
            Total number of episodes.
        """
        query = "SELECT COUNT(*) as count FROM episodes"
        row = await self.episodic.sqlite.fetch_one(query, ())
        return row["count"] if row else 0

    async def _generate_abstractions(self) -> int:
        """Generate abstractions from unconsolidated episodes.

        Returns:
            Number of abstractions created.
        """
        # Get unconsolidated Level 0 episodes
        query = """
            SELECT * FROM episodes
            WHERE level = 0 AND parent_id IS NULL
            ORDER BY occurred_at ASC
            LIMIT ?
        """
        params = (self.config.max_episodes_per_cycle,)
        rows = await self.episodic.sqlite.fetch_all(query, params)

        if not rows:
            logger.debug("No unconsolidated episodes for abstraction")
            return 0

        # Convert to Episode objects
        episodes = [self._row_to_episode(row) for row in rows]

        # Cluster episodes
        clusters = await self.abstraction_generator.cluster_episodes(
            episodes, cluster_size=self.config.abstraction_cluster_size
        )

        if not clusters:
            logger.debug("No clusters formed for abstraction")
            return 0

        # Generate abstraction for each cluster
        abstractions_created = 0
        for cluster in clusters:
            if len(cluster) < 2:
                # Skip clusters with single episode
                continue

            # Generate abstraction
            abstraction = await self.abstraction_generator.generate_abstraction(cluster)

            # Store abstraction
            await self.episodic.add_episode(abstraction)

            # Update cluster episodes to point to parent
            for episode in cluster:
                update_query = "UPDATE episodes SET parent_id = ? WHERE id = ?"
                update_params = (abstraction.id, episode.id)
                await self.episodic.sqlite.execute(update_query, update_params)

            abstractions_created += 1

        logger.info(
            f"Generated {abstractions_created} abstractions from {len(episodes)} episodes"
        )
        return abstractions_created

    async def _detect_patterns(self):
        """Detect patterns in recent episodes.

        Returns:
            PatternDetectionResult with pattern statistics.
        """
        from htma.core.types import PatternDetectionResult

        # Get recent episodes for pattern detection
        query = """
            SELECT * FROM episodes
            WHERE level = 0
            ORDER BY occurred_at DESC
            LIMIT ?
        """
        params = (self.config.max_episodes_per_cycle,)
        rows = await self.episodic.sqlite.fetch_all(query, params)

        if not rows:
            logger.debug("No episodes for pattern detection")
            return PatternDetectionResult()

        # Convert to Episode objects
        episodes = [self._row_to_episode(row) for row in rows]

        # Get existing patterns from semantic memory
        # For now, use empty list (pattern storage not yet implemented in semantic memory)
        existing_patterns: list[Pattern] = []

        # Detect patterns
        result = await self.pattern_detector.detect_patterns(
            episodes=episodes,
            existing_patterns=existing_patterns,
            min_occurrences=self.config.pattern_min_occurrences,
        )

        # TODO: Store patterns in semantic memory (future implementation)
        # For now, just return the result
        logger.info(
            f"Pattern detection: {len(result.new_patterns)} new, "
            f"{len(result.strengthened)} strengthened"
        )

        return result

    def _row_to_episode(self, row: dict[str, Any]) -> Episode:
        """Convert database row to Episode object.

        Args:
            row: Database row dictionary.

        Returns:
            Episode object.
        """
        import json

        return Episode(
            id=row["id"],
            level=row["level"],
            parent_id=row.get("parent_id"),
            content=row["content"],
            summary=row.get("summary"),
            context_description=row.get("context_description"),
            keywords=json.loads(row["keywords"]) if row.get("keywords") else [],
            tags=json.loads(row["tags"]) if row.get("tags") else [],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            salience=row["salience"],
            consolidation_strength=row["consolidation_strength"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"])
            if row.get("last_accessed")
            else None,
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
        )
