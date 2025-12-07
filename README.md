# HTMA - Hierarchical-Temporal Memory Architecture

A Python implementation of a hierarchical-temporal memory system for LLM-based personal agents. HTMA enables autonomous memory curation, temporal reasoning, and evolving knowledge across extended interactions.

## Features

- **Tri-Memory Architecture**: Working memory (in-context), Semantic memory (temporal knowledge graph), and Episodic memory (hierarchical experiences)
- **Bi-Temporal Model**: Track both when facts were true in the world and when they were recorded
- **Autonomous Curation**: Dedicated LLM evaluates salience, extracts entities/facts, and manages conflicts
- **Memory Evolution**: New memories update and enrich existing context (A-MEM style)
- **Consolidation**: Background process generates abstractions, detects patterns, and maintains link health
- **Multi-Path Retrieval**: Semantic search, temporal queries, and associative linking

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM₁ (Reasoner) ←──→ Memory Interface ←──→ LLM₂ (Curator)     │
│                              │                                  │
│              ┌───────────────┼───────────────┐                 │
│              │               │               │                 │
│         Working          Semantic        Episodic              │
│          Memory           Memory          Memory               │
│        (in-context)   (temporal KG)   (hierarchical)          │
│                              │                                  │
│              └───────────────┼───────────────┘                 │
│                        Storage Layer                            │
│                   (SQLite + ChromaDB)                          │
│                              │                                  │
│                   Consolidation Engine                          │
│                   (background evolution)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) (for local LLMs)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/htma.git
cd htma
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install HTMA:
```bash
pip install -e ".[dev]"
```

4. Pull required Ollama models:
```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull nomic-embed-text
```

## Quick Start

```python
from htma import HTMAAgent

# Initialize agent
agent = HTMAAgent()

# Have a conversation
response = await agent.process_message("Tell me about the project I'm working on")
print(response.content)

# Query memory
results = await agent.query_memory("What do I know about Python?")
```

## Project Structure

```
htma/
├── src/htma/
│   ├── core/          # Types, exceptions, utilities
│   ├── memory/        # Working, semantic, episodic stores + interface
│   ├── curator/       # Memory curator (LLM₂ operations)
│   ├── consolidation/ # Background memory evolution
│   ├── storage/       # SQLite + ChromaDB operations
│   ├── llm/           # Ollama client + prompt templates
│   └── agent/         # Main agent integration
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/           # Utilities and demos
└── data/              # Local storage (gitignored)
```

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=htma --cov-report=html

# Specific test file
pytest tests/unit/test_memory.py
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format code
ruff format src/
```

## Configuration

HTMA can be configured via environment variables:

```bash
# Database
export HTMA_SQLITE_PATH="data/htma.db"

# Ollama
export HTMA_OLLAMA_URL="http://localhost:11434"
export HTMA_REASONER_MODEL="llama3:8b"
export HTMA_CURATOR_MODEL="mistral:7b"
export HTMA_EMBEDDING_MODEL="nomic-embed-text"

# Memory
export HTMA_WORKING_MEMORY_MAX_TOKENS="8000"
export HTMA_CONSOLIDATION_INTERVAL="24h"
```

## Documentation

- [Architecture Document](htma-mvp-architecture.md) - Detailed system design
- [Development Guide](CLAUDE.md) - Code patterns and conventions
- API Documentation - Coming soon

## Status

**Current Phase**: Foundation (Phase 1)

- [x] Project setup
- [ ] Core types and data models
- [ ] Storage layer (SQLite + ChromaDB)
- [ ] LLM client wrapper
- [ ] Memory stores implementation
- [ ] Curator operations
- [ ] Consolidation engine
- [ ] Agent integration

See [GitHub Issues](https://github.com/yourusername/htma/issues) for detailed progress.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure `ruff check` and `mypy` pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## References

- [MemGPT Paper](https://arxiv.org/abs/2310.08560) - Virtual context management
- [A-MEM Paper](https://arxiv.org/abs/2501.xxxxx) - Evolving associative memory
- [Graphiti Paper](https://arxiv.org/abs/2501.13956) - Temporal knowledge graphs
- [RAPTOR Paper](https://arxiv.org/abs/2401.xxxxx) - Recursive abstractive processing

## Acknowledgments

Built on research from MemGPT, A-MEM, Zep/Graphiti, and RAPTOR projects.
