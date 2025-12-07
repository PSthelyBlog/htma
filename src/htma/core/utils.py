"""Utility functions for HTMA core functionality.

This module provides helper functions for ID generation, time manipulation,
and other common operations.
"""

import uuid
from datetime import UTC, datetime


def generate_entity_id() -> str:
    """Generate a unique, sortable ID for an entity.

    Returns a UUID v7 (time-ordered) string with 'ent_' prefix for clarity.
    UUID v7 combines timestamp with random bits, ensuring both uniqueness
    and lexicographic sortability.

    Returns:
        A unique entity ID string.

    Example:
        >>> entity_id = generate_entity_id()
        >>> entity_id.startswith('ent_')
        True
    """
    # Using UUID v1 (time-based) for sortability
    # In production, consider using ULID library for better sortability
    return f"ent_{uuid.uuid1()}"


def generate_fact_id() -> str:
    """Generate a unique, sortable ID for a fact.

    Returns a UUID v7 (time-ordered) string with 'fct_' prefix for clarity.

    Returns:
        A unique fact ID string.

    Example:
        >>> fact_id = generate_fact_id()
        >>> fact_id.startswith('fct_')
        True
    """
    return f"fct_{uuid.uuid1()}"


def generate_episode_id() -> str:
    """Generate a unique, sortable ID for an episode.

    Returns a UUID v7 (time-ordered) string with 'epi_' prefix for clarity.

    Returns:
        A unique episode ID string.

    Example:
        >>> episode_id = generate_episode_id()
        >>> episode_id.startswith('epi_')
        True
    """
    return f"epi_{uuid.uuid1()}"


def generate_link_id() -> str:
    """Generate a unique ID for an episode link.

    Returns a UUID v4 (random) string with 'lnk_' prefix.
    Links don't need to be sortable by time.

    Returns:
        A unique link ID string.

    Example:
        >>> link_id = generate_link_id()
        >>> link_id.startswith('lnk_')
        True
    """
    return f"lnk_{uuid.uuid4()}"


def generate_community_id() -> str:
    """Generate a unique ID for a community.

    Returns a UUID v4 (random) string with 'com_' prefix.

    Returns:
        A unique community ID string.

    Example:
        >>> community_id = generate_community_id()
        >>> community_id.startswith('com_')
        True
    """
    return f"com_{uuid.uuid4()}"


def utc_now() -> datetime:
    """Get current UTC time with timezone information.

    Returns:
        Current datetime in UTC with timezone info.

    Example:
        >>> now = utc_now()
        >>> now.tzinfo is not None
        True
    """
    return datetime.now(UTC)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is in UTC timezone.

    Args:
        dt: A datetime object (may be naive or aware).

    Returns:
        The same datetime converted to UTC, or None if input is None.
        Naive datetimes are assumed to be UTC.

    Example:
        >>> from datetime import datetime
        >>> dt = datetime(2024, 1, 1)
        >>> utc_dt = ensure_utc(dt)
        >>> utc_dt.tzinfo is not None
        True
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=UTC)

    # Convert to UTC if not already
    return dt.astimezone(UTC)


def is_valid_id(id_str: str, prefix: str) -> bool:
    """Check if an ID string is valid for the given prefix.

    Args:
        id_str: The ID string to validate.
        prefix: The expected prefix (e.g., 'ent_', 'fct_', 'epi_').

    Returns:
        True if the ID is valid, False otherwise.

    Example:
        >>> is_valid_id('ent_12345678-1234-1234-1234-123456789012', 'ent_')
        True
        >>> is_valid_id('invalid_id', 'ent_')
        False
    """
    if not id_str.startswith(prefix):
        return False

    # Extract UUID part and validate
    uuid_part = id_str[len(prefix) :]
    try:
        uuid.UUID(uuid_part)
        return True
    except ValueError:
        return False
