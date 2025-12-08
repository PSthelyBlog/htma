# Integration Tests

This directory contains end-to-end integration tests for the HTMA system.

## Prerequisites

Before running integration tests, ensure you have:

1. **Ollama installed and running**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh

   # Start Ollama service
   ollama serve
   ```

2. **Required models downloaded**:
   ```bash
   ollama pull llama3:8b
   ollama pull mistral:7b
   ollama pull nomic-embed-text
   ```

3. **Dependencies installed**:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Integration Tests

### Run all integration tests:
```bash
pytest tests/integration/ -v
```

### Run specific test classes:
```bash
# Basic memory formation
pytest tests/integration/test_full_flow.py::TestBasicMemoryFormation -v

# Episodic storage and retrieval
pytest tests/integration/test_full_flow.py::TestEpisodicStorageAndRetrieval -v

# Temporal reasoning
pytest tests/integration/test_full_flow.py::TestTemporalReasoning -v

# Memory linking
pytest tests/integration/test_full_flow.py::TestMemoryLinking -v

# Consolidation
pytest tests/integration/test_full_flow.py::TestConsolidation -v

# Memory pressure
pytest tests/integration/test_full_flow.py::TestMemoryPressure -v

# Conflict resolution
pytest tests/integration/test_full_flow.py::TestConflictResolution -v

# End-to-end flow
pytest tests/integration/test_full_flow.py::TestEndToEndFlow -v
```

### Run with coverage:
```bash
pytest tests/integration/ --cov=htma --cov-report=html
```

### Skip integration tests (when Ollama is not available):
```bash
pytest tests/ -m "not integration"
```

## Test Scenarios

### 1. Basic Memory Formation
Tests that facts mentioned in conversations are stored in semantic memory and can be retrieved later.

### 2. Episodic Storage and Retrieval
Tests multi-turn conversations create episodes and that semantic search finds relevant episodes.

### 3. Temporal Reasoning
Tests bi-temporal model with fact invalidation and historical queries.

### 4. Memory Linking
Tests that related conversations create links between episodes.

### 5. Consolidation
Tests abstraction generation, pattern detection, and link maintenance during consolidation cycles.

### 6. Memory Pressure
Tests working memory pressure handling and persistence of important information.

### 7. Conflict Resolution
Tests resolution of contradictory facts with temporal invalidation.

### 8. End-to-End Flow
Tests the complete pipeline from user input through storage, retrieval, and memory evolution.

## Test Data

Integration tests use temporary directories and databases that are cleaned up after each test. No persistent data is created.

## Troubleshooting

### Ollama Connection Errors
If tests fail with connection errors:
1. Ensure Ollama is running: `curl http://localhost:11434/api/tags`
2. Check Ollama logs for errors
3. Restart Ollama service if needed

### Model Not Found Errors
If tests fail because models are not found:
```bash
ollama list  # Check installed models
ollama pull <model-name>  # Download missing models
```

### Slow Test Execution
Integration tests involve real LLM calls and are slower than unit tests. Expect:
- Individual tests: 5-30 seconds
- Full suite: 5-15 minutes

You can run tests in parallel with pytest-xdist:
```bash
pip install pytest-xdist
pytest tests/integration/ -n auto
```

## CI/CD Considerations

For CI/CD pipelines:
1. Set up Ollama in the CI environment
2. Use smaller/faster models for testing
3. Consider mocking LLM calls for faster CI runs
4. Use the `@pytest.mark.integration` marker to separate fast unit tests from slow integration tests

Example CI command:
```bash
# Run only unit tests in CI (fast)
pytest tests/unit/ -v

# Run integration tests only on main branch or manually
pytest tests/integration/ -v -m integration
```
