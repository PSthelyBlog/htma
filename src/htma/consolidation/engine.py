"""Consolidation engine for memory evolution and maintenance.

This module implements the ConsolidationEngine that orchestrates memory evolution
processes including link maintenance, abstraction generation, and pattern detection.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any

from htma.core.exceptions import ConsolidationError, DatabaseError
from htma.core.types import EpisodeID, LinkMaintenanceReport
from htma.core.utils import utc_now
from htma.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Orchestrates memory consolidation and evolution processes.

    The consolidation engine manages:
    - Link strengthening and decay based on access patterns
    - Pruning of weak links
    - Abstraction generation (future)
    - Pattern detection (future)
    - Stale memory pruning (future)

    Attributes:
        episodic: Episodic memory instance for link operations.
    """

    def __init__(self, episodic: EpisodicMemory):
        """Initialize consolidation engine.

        Args:
            episodic: Episodic memory instance.
        """
        self.episodic = episodic

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

    # ========== Helper Methods ==========

    async def _count_total_links(self) -> int:
        """Count total number of links in the graph.

        Returns:
            Total number of episode links.
        """
        query = "SELECT COUNT(*) as count FROM episode_links"
        row = await self.episodic.sqlite.fetch_one(query, ())
        return row["count"] if row else 0
