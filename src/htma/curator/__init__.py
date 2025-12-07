"""Memory curator for salience evaluation and memory formation.

The curator acts as LLM2 in the HTMA architecture, responsible for evaluating
what information is worth remembering and how to structure it.
"""

from htma.curator.curator import MemoryCurator

__all__ = [
    "MemoryCurator",
]
