"""Consolidation module for memory abstraction and pattern detection.

This module provides background processes that evolve memory through:
- Abstraction generation (summaries from episodes)
- Pattern detection (recurring themes)
- Link maintenance (strengthening and pruning)
- Memory pruning (removing stale content)
"""

from htma.consolidation.abstraction import AbstractionGenerator
from htma.consolidation.engine import ConsolidationEngine

__all__ = ["AbstractionGenerator", "ConsolidationEngine"]
