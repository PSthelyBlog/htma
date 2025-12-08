"""Custom exceptions for HTMA.

This module defines the exception hierarchy used throughout the system
to handle various error conditions with appropriate context.
"""


class HTMAError(Exception):
    """Base exception for all HTMA errors.

    All custom exceptions in the HTMA system should inherit from this class.
    """

    pass


# Memory-related exceptions
class MemoryError(HTMAError):
    """Base exception for memory-related errors."""

    pass


class MemoryNotFoundError(MemoryError):
    """Raised when a requested memory does not exist.

    Attributes:
        memory_id: The ID of the memory that was not found.
        memory_type: The type of memory (entity, fact, episode).
    """

    def __init__(self, memory_id: str, memory_type: str = "memory"):
        self.memory_id = memory_id
        self.memory_type = memory_type
        super().__init__(f"{memory_type.capitalize()} not found: {memory_id}")


class DuplicateMemoryError(MemoryError):
    """Raised when attempting to create a duplicate memory."""

    def __init__(self, memory_id: str, memory_type: str = "memory"):
        self.memory_id = memory_id
        self.memory_type = memory_type
        super().__init__(f"{memory_type.capitalize()} already exists: {memory_id}")


class MemoryPressureError(MemoryError):
    """Raised when working memory is under excessive pressure."""

    def __init__(self, current_usage: int, max_capacity: int):
        self.current_usage = current_usage
        self.max_capacity = max_capacity
        super().__init__(
            f"Memory pressure critical: {current_usage}/{max_capacity} tokens used"
        )


# Storage-related exceptions
class StorageError(HTMAError):
    """Base exception for storage-related errors."""

    pass


class DatabaseError(StorageError):
    """Raised when a database operation fails."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when unable to connect to the database."""

    pass


class MigrationError(DatabaseError):
    """Raised when a database migration fails."""

    pass


class VectorStoreError(StorageError):
    """Raised when a vector store operation fails."""

    pass


# Curator-related exceptions
class CuratorError(HTMAError):
    """Base exception for memory curator errors."""

    pass


class SalienceEvaluationError(CuratorError):
    """Raised when salience evaluation fails."""

    pass


class ExtractionError(CuratorError):
    """Raised when entity or fact extraction fails."""

    pass


class ConflictResolutionError(CuratorError):
    """Raised when unable to resolve a memory conflict.

    Attributes:
        fact_id: The ID of the conflicting fact.
        conflicts: List of conflicting fact IDs.
    """

    def __init__(self, fact_id: str, conflicts: list[str]):
        self.fact_id = fact_id
        self.conflicts = conflicts
        super().__init__(
            f"Failed to resolve conflict for fact {fact_id} with {len(conflicts)} conflicting facts"
        )


# LLM-related exceptions
class LLMError(HTMAError):
    """Base exception for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to the LLM service."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""

    pass


class LLMResponseError(LLMError):
    """Raised when the LLM returns an invalid or unexpected response."""

    pass


class ModelNotFoundError(LLMError):
    """Raised when a requested model is not available."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model not found: {model_name}")


# Consolidation-related exceptions
class ConsolidationError(HTMAError):
    """Base exception for consolidation-related errors."""

    pass


class AbstractionError(ConsolidationError):
    """Raised when abstraction generation fails."""

    pass


class PatternDetectionError(ConsolidationError):
    """Raised when pattern detection fails."""

    pass


class LinkMaintenanceError(ConsolidationError):
    """Raised when link maintenance fails."""

    pass


# Validation exceptions
class ValidationError(HTMAError):
    """Base exception for validation errors."""

    pass


class TemporalValidationError(ValidationError):
    """Raised when temporal constraints are violated."""

    def __init__(self, message: str):
        super().__init__(f"Temporal validation error: {message}")


class ConfigurationError(HTMAError):
    """Raised when there is a configuration error."""

    pass


# Agent-related exceptions
class AgentError(HTMAError):
    """Base exception for agent-related errors."""

    pass
