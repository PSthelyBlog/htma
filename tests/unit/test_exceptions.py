"""Unit tests for custom exceptions."""

import pytest

from htma.core.exceptions import (
    AbstractionError,
    ConfigurationError,
    ConflictResolutionError,
    ConsolidationError,
    CuratorError,
    DatabaseConnectionError,
    DatabaseError,
    DuplicateMemoryError,
    ExtractionError,
    HTMAError,
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    LinkMaintenanceError,
    MemoryError,
    MemoryNotFoundError,
    MemoryPressureError,
    MigrationError,
    ModelNotFoundError,
    PatternDetectionError,
    SalienceEvaluationError,
    StorageError,
    TemporalValidationError,
    ValidationError,
    VectorStoreError,
)


class TestHTMAError:
    """Tests for base HTMAError exception."""

    def test_raise_base_error(self):
        """Test raising base HTMA error."""
        with pytest.raises(HTMAError) as exc_info:
            raise HTMAError("Test error")
        assert str(exc_info.value) == "Test error"

    def test_all_inherit_from_base(self):
        """Test that all custom exceptions inherit from HTMAError."""
        exceptions = [
            MemoryError,
            StorageError,
            CuratorError,
            LLMError,
            ConsolidationError,
            ValidationError,
            ConfigurationError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, HTMAError)


class TestMemoryExceptions:
    """Tests for memory-related exceptions."""

    def test_memory_not_found_error(self):
        """Test MemoryNotFoundError with details."""
        error = MemoryNotFoundError("ent_123", "entity")
        assert error.memory_id == "ent_123"
        assert error.memory_type == "entity"
        assert "Entity not found: ent_123" in str(error)

    def test_memory_not_found_default_type(self):
        """Test MemoryNotFoundError with default type."""
        error = MemoryNotFoundError("id_123")
        assert error.memory_type == "memory"
        assert "Memory not found: id_123" in str(error)

    def test_duplicate_memory_error(self):
        """Test DuplicateMemoryError."""
        error = DuplicateMemoryError("fct_456", "fact")
        assert error.memory_id == "fct_456"
        assert error.memory_type == "fact"
        assert "Fact already exists: fct_456" in str(error)

    def test_memory_pressure_error(self):
        """Test MemoryPressureError with usage info."""
        error = MemoryPressureError(current_usage=9000, max_capacity=10000)
        assert error.current_usage == 9000
        assert error.max_capacity == 10000
        assert "9000/10000" in str(error)

    def test_memory_error_inheritance(self):
        """Test memory exception inheritance."""
        exceptions = [MemoryNotFoundError, DuplicateMemoryError, MemoryPressureError]
        for exc_class in exceptions:
            assert issubclass(exc_class, MemoryError)
            assert issubclass(exc_class, HTMAError)


class TestStorageExceptions:
    """Tests for storage-related exceptions."""

    def test_database_error(self):
        """Test DatabaseError."""
        with pytest.raises(DatabaseError) as exc_info:
            raise DatabaseError("Connection failed")
        assert "Connection failed" in str(exc_info.value)

    def test_database_connection_error(self):
        """Test DatabaseConnectionError."""
        with pytest.raises(DatabaseConnectionError):
            raise DatabaseConnectionError("Cannot connect")

    def test_migration_error(self):
        """Test MigrationError."""
        with pytest.raises(MigrationError):
            raise MigrationError("Migration failed")

    def test_vector_store_error(self):
        """Test VectorStoreError."""
        with pytest.raises(VectorStoreError):
            raise VectorStoreError("ChromaDB error")

    def test_storage_error_inheritance(self):
        """Test storage exception inheritance."""
        exceptions = [
            DatabaseError,
            DatabaseConnectionError,
            MigrationError,
            VectorStoreError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, StorageError)
            assert issubclass(exc_class, HTMAError)


class TestCuratorExceptions:
    """Tests for curator-related exceptions."""

    def test_salience_evaluation_error(self):
        """Test SalienceEvaluationError."""
        with pytest.raises(SalienceEvaluationError):
            raise SalienceEvaluationError("Evaluation failed")

    def test_extraction_error(self):
        """Test ExtractionError."""
        with pytest.raises(ExtractionError):
            raise ExtractionError("Entity extraction failed")

    def test_conflict_resolution_error(self):
        """Test ConflictResolutionError with details."""
        error = ConflictResolutionError("fct_123", ["fct_456", "fct_789"])
        assert error.fact_id == "fct_123"
        assert error.conflicts == ["fct_456", "fct_789"]
        assert "fct_123" in str(error)
        assert "2 conflicting facts" in str(error)

    def test_curator_error_inheritance(self):
        """Test curator exception inheritance."""
        exceptions = [
            SalienceEvaluationError,
            ExtractionError,
            ConflictResolutionError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, CuratorError)
            assert issubclass(exc_class, HTMAError)


class TestLLMExceptions:
    """Tests for LLM-related exceptions."""

    def test_llm_connection_error(self):
        """Test LLMConnectionError."""
        with pytest.raises(LLMConnectionError):
            raise LLMConnectionError("Ollama not running")

    def test_llm_timeout_error(self):
        """Test LLMTimeoutError."""
        with pytest.raises(LLMTimeoutError):
            raise LLMTimeoutError("Request timed out")

    def test_llm_response_error(self):
        """Test LLMResponseError."""
        with pytest.raises(LLMResponseError):
            raise LLMResponseError("Invalid JSON response")

    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("llama3:8b")
        assert error.model_name == "llama3:8b"
        assert "llama3:8b" in str(error)

    def test_llm_error_inheritance(self):
        """Test LLM exception inheritance."""
        exceptions = [
            LLMConnectionError,
            LLMTimeoutError,
            LLMResponseError,
            ModelNotFoundError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, LLMError)
            assert issubclass(exc_class, HTMAError)


class TestConsolidationExceptions:
    """Tests for consolidation-related exceptions."""

    def test_abstraction_error(self):
        """Test AbstractionError."""
        with pytest.raises(AbstractionError):
            raise AbstractionError("Failed to generate summary")

    def test_pattern_detection_error(self):
        """Test PatternDetectionError."""
        with pytest.raises(PatternDetectionError):
            raise PatternDetectionError("Pattern detection failed")

    def test_link_maintenance_error(self):
        """Test LinkMaintenanceError."""
        with pytest.raises(LinkMaintenanceError):
            raise LinkMaintenanceError("Link update failed")

    def test_consolidation_error_inheritance(self):
        """Test consolidation exception inheritance."""
        exceptions = [
            AbstractionError,
            PatternDetectionError,
            LinkMaintenanceError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, ConsolidationError)
            assert issubclass(exc_class, HTMAError)


class TestValidationExceptions:
    """Tests for validation-related exceptions."""

    def test_temporal_validation_error(self):
        """Test TemporalValidationError."""
        error = TemporalValidationError("valid_to must be after valid_from")
        assert "valid_to must be after valid_from" in str(error)

    def test_validation_error_inheritance(self):
        """Test validation exception inheritance."""
        assert issubclass(TemporalValidationError, ValidationError)
        assert issubclass(ValidationError, HTMAError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid config")

    def test_configuration_error_inheritance(self):
        """Test configuration error inheritance."""
        assert issubclass(ConfigurationError, HTMAError)


class TestExceptionCatching:
    """Tests for exception catching and hierarchy."""

    def test_catch_with_base_class(self):
        """Test catching specific exceptions with base class."""
        with pytest.raises(HTMAError):
            raise MemoryNotFoundError("test_id")

        with pytest.raises(HTMAError):
            raise LLMConnectionError("test")

    def test_catch_with_category(self):
        """Test catching with category base class."""
        with pytest.raises(MemoryError):
            raise MemoryNotFoundError("test_id")

        with pytest.raises(StorageError):
            raise DatabaseError("test")

        with pytest.raises(LLMError):
            raise ModelNotFoundError("test_model")

    def test_exception_message_preservation(self):
        """Test that exception messages are preserved."""
        msg = "Custom error message"
        with pytest.raises(HTMAError) as exc_info:
            raise HTMAError(msg)
        assert str(exc_info.value) == msg
