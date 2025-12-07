# HTMA - Hierarchical-Temporal Memory Architecture

## Project Overview

HTMA is a Python implementation of a hierarchical-temporal memory system for LLM-based personal agents. It enables autonomous memory curation, temporal reasoning, and evolving knowledge across extended interactions.

**Status**: MVP Development
**Stack**: Python 3.11+, SQLite, ChromaDB, Ollama (local LLMs)

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

## Core Concepts

### Tri-Memory Structure
- **Working Memory**: In-context information with pressure-based management
- **Semantic Memory**: Temporal knowledge graph (entities + facts with bi-temporal validity)
- **Episodic Memory**: Hierarchical experiences (raw → summaries → abstractions)

### Bi-Temporal Model
Every fact tracks:
- **Event time (T)**: When the fact was true in the world
- **Transaction time (T')**: When the fact was recorded in memory

This enables queries like "What did I know about X as of date Y?"

### Memory Curator (LLM₂)
Specialized model handling:
- Salience evaluation (what's worth remembering)
- Entity/fact extraction
- Link generation between memories
- Conflict resolution via temporal invalidation
- Memory evolution (new memories update existing context)

### Consolidation
Background process that:
- Generates abstractions from episodes
- Detects patterns across experiences
- Strengthens/weakens links based on access
- Prunes stale memories

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
├── scripts/
└── data/              # Local storage (gitignored)
```

## Code Patterns

### Async-First
All I/O operations are async:
```python
async def query_semantic(self, query: str) -> list[Fact]:
    async with self.db.connection() as conn:
        # ...
```

### Pydantic Models
Use Pydantic for all data structures:
```python
from pydantic import BaseModel
from datetime import datetime

class Fact(BaseModel):
    id: str
    subject_id: str
    predicate: str
    object_id: str | None = None
    object_value: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 1.0
```

### Configuration
Use pydantic-settings with environment variables:
```python
from pydantic_settings import BaseSettings

class HTMAConfig(BaseSettings):
    sqlite_path: str = "data/htma.db"
    ollama_url: str = "http://localhost:11434"
    
    class Config:
        env_prefix = "HTMA_"
```

### Error Handling
Custom exceptions with context:
```python
class HTMAError(Exception):
    """Base exception for HTMA."""
    pass

class MemoryNotFoundError(HTMAError):
    """Requested memory does not exist."""
    pass

class ConflictResolutionError(HTMAError):
    """Failed to resolve memory conflict."""
    pass
```

## Database Schema

### SQLite Tables
- `entities`: Named concepts with metadata
- `facts`: Relationships with bi-temporal validity
- `communities`: Topic clusters
- `episodes`: Hierarchical experiences
- `episode_links`: Bidirectional connections
- `retrieval_indices`: Multi-path retrieval cues

### ChromaDB Collections
- `episodes`: Episode embeddings for semantic search
- `entities`: Entity embeddings

## Testing Strategy

- Unit tests for each component
- Integration tests for memory flow
- Use `pytest-asyncio` for async tests
- Mock LLM calls with predictable responses

```python
@pytest.fixture
def mock_curator(mocker):
    return mocker.patch('htma.curator.MemoryCurator')
```

## LLM Integration

### Models (via Ollama)
- **Reasoner (LLM₁)**: `llama3:8b` or similar
- **Curator (LLM₂)**: `mistral:7b` or similar (can be smaller)
- **Embeddings**: `nomic-embed-text`

### Prompt Templates
Located in `src/htma/llm/prompts/`:
- Structured for consistent output parsing
- Include examples for few-shot learning
- Request JSON output where structured data needed

## Known Constraints

- **Context window**: Working memory limited to ~8K tokens
- **Local models**: Smaller than cloud models, may need prompt tuning
- **Single user**: MVP designed for personal agent, not multi-tenant

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/

# Initialize database
python scripts/init_db.py

# Run demo
python scripts/demo.py
```

## Current State

**Completed**:
- Architecture design
- Schema design
- Project structure

**In Progress**:
- Phase 1: Foundation

**Next**:
- Core types and data models
- SQLite schema implementation
- ChromaDB setup

## Failed Approaches

(Document what doesn't work to prevent repeating)

*None yet - project starting*

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite + ChromaDB hybrid | Best of both: structured queries + semantic search |
| Separate curator model | Enables specialization, reduces load on reasoner |
| Bi-temporal model | Essential for temporal reasoning and contradiction handling |
| RAPTOR-style hierarchy | Retrieval at multiple abstraction levels |
| A-MEM linking | Memories evolve together, not just accumulate |

## References

- [HTMA Foundation Document](docs/HTMA-Foundation-Document.md)
- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
- [A-MEM Paper](https://arxiv.org/abs/2501.xxxxx)
- [Zep/Graphiti Paper](https://arxiv.org/abs/2501.13956)
- [RAPTOR Paper](https://arxiv.org/abs/2401.xxxxx)
