#!/bin/bash
# Script to run HTMA integration tests
#
# This script checks prerequisites and runs integration tests.
# Usage: ./scripts/run_integration_tests.sh [pytest-args]

set -e

echo "HTMA Integration Test Runner"
echo "============================"
echo ""

# Check if Ollama is running
echo "Checking Ollama availability..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama is running"
else
    echo "✗ Ollama is not running or not accessible"
    echo ""
    echo "Please start Ollama:"
    echo "  ollama serve"
    echo ""
    exit 1
fi

# Check for required models
echo ""
echo "Checking required models..."
REQUIRED_MODELS=("llama3:8b" "mistral:7b" "nomic-embed-text")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo "✓ $model is installed"
    else
        echo "✗ $model is not installed"
        MISSING_MODELS+=("$model")
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "Missing models. Install them with:"
    for model in "${MISSING_MODELS[@]}"; do
        echo "  ollama pull $model"
    done
    echo ""
    exit 1
fi

# Run tests
echo ""
echo "Running integration tests..."
echo ""

# Pass any additional arguments to pytest
pytest tests/integration/ -v -m integration "$@"

echo ""
echo "Integration tests complete!"
