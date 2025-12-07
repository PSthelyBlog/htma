# HTMA MVP Technical Architecture

## Overview

**Goal**: Implement a Hierarchical-Temporal Memory Architecture for a personal agent assistant using Python with lightweight local storage.

**Stack**:
- **Language**: Python 3.11+
- **Storage**: SQLite (structured data) + ChromaDB (vector embeddings)
- **LLM Integration**: Ollama for local models (separate reasoning + curator models)
- **Framework**: Async-first with `asyncio`

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HTMA System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │   LLM₁ (Reasoner)│    │  LLM₂ (Curator)  │                  │
│  │   (e.g. llama3)  │    │ (e.g. mistral)   │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           │    ┌──────────────────┴──────────────┐             │
│           │    │      Memory Interface Layer      │             │
│           │    │  ┌─────────────────────────────┐│             │
│           └────┼─►│  Query Router & Synthesizer ││             │
│                │  └─────────────────────────────┘│             │
│                └──────────────┬──────────────────┘             │
│                               │                                 │
│  ┌────────────────────────────┼────────────────────────────┐   │
│  │              Tri-Memory Store                            │   │
│  │                            │                             │   │
│  │  ┌─────────────┐  ┌───────┴───────┐  ┌─────────────┐   │   │
│  │  │   Working   │  │   Semantic    │  │  Episodic   │   │   │
│  │  │   Memory    │  │    Memory     │  │   Memory    │   │   │
│  │  │ (In-Context)│  │(Temporal KG)  │  │(Hierarchical│   │   │
│  │  │             │  │               │  │  Episodes)  │   │   │
│  │  └─────────────┘  └───────────────┘  └─────────────┘   │   │
│  │                            │                             │   │
│  │  ┌─────────────────────────┴─────────────────────────┐  │   │
│  │  │              Storage Layer                         │  │   │
│  │  │  ┌─────────────┐          ┌─────────────────────┐ │  │   │
│  │  │  │   SQLite    │          │      ChromaDB       │ │  │   │
│  │  │  │ (Entities,  │          │ (Vector Embeddings, │ │  │   │
│  │  │  │  Facts,     │          │  Semantic Search)   │ │  │   │
│  │  │  │  Episodes)  │          │                     │ │  │   │
│  │  │  └─────────────┘          └─────────────────────┘ │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Consolidation Engine                         │  │
│  │  (Background async process for memory evolution)          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Working Memory

**Purpose**: Manage the in-context information available to LLM₁.

**Implementation**:
```python
@dataclass
class WorkingMemory:
    system_context: str           # Persona, capabilities
    task_context: str             # Current goals, constraints
    dialog_history: list[Message] # Recent conversation (FIFO)
    retrieved_context: list[MemoryItem]  # From semantic/episodic
    
    max_tokens: int = 8000        # Configurable limit
    pressure_threshold: float = 0.8  # Trigger offload at 80%
```

**Key Features**:
- Token counting with tiktoken
- Automatic summarization when approaching limit
- Memory pressure warnings trigger proactive offload
- Priority-based eviction (retrieved < old dialog < task < system)

---

### 2. Semantic Memory (Temporal Knowledge Graph)

**Purpose**: Store structured knowledge with temporal validity.

**Schema** (SQLite):
```sql
-- Entities
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- person, place, concept, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    metadata JSON
);

-- Facts (edges with temporal validity)
CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    subject_id TEXT REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id TEXT REFERENCES entities(id),
    object_value TEXT,  -- For literal values
    
    -- Bi-temporal model
    valid_from TIMESTAMP,        -- When fact became true (event time)
    valid_to TIMESTAMP,          -- When fact stopped being true
    recorded_at TIMESTAMP,       -- When we learned this (transaction time)
    invalidated_at TIMESTAMP,    -- When we learned it was no longer true
    
    confidence REAL DEFAULT 1.0,
    source_episode_id TEXT,      -- Provenance
    metadata JSON
);

-- Community/topic clusters
CREATE TABLE communities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    entity_ids JSON,  -- List of member entity IDs
    created_at TIMESTAMP,
    metadata JSON
);

-- Indexes for temporal queries
CREATE INDEX idx_facts_valid_period ON facts(valid_from, valid_to);
CREATE INDEX idx_facts_recorded ON facts(recorded_at);
CREATE INDEX idx_entities_type ON entities(entity_type);
```

**Temporal Query Support**:
```python
class SemanticMemory:
    async def query_at_time(self, entity: str, as_of: datetime) -> list[Fact]:
        """What did we know about entity as of a specific time?"""
        
    async def query_valid_at(self, entity: str, when: datetime) -> list[Fact]:
        """What facts were true about entity at a specific time?"""
        
    async def get_fact_history(self, subject: str, predicate: str) -> list[Fact]:
        """How has a specific fact changed over time?"""
```

---

### 3. Episodic Memory (Hierarchical Experience Store)

**Purpose**: Store experiences at multiple abstraction levels.

**Schema** (SQLite + ChromaDB):
```sql
-- Episodes at all levels
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    level INTEGER NOT NULL,  -- 0=raw, 1=summary, 2=abstraction, etc.
    parent_id TEXT REFERENCES episodes(id),  -- For hierarchy
    
    content TEXT NOT NULL,
    summary TEXT,  -- LLM-generated summary
    
    -- A-MEM style metadata
    context_description TEXT,  -- LLM-generated significance
    keywords JSON,
    tags JSON,
    
    -- Temporal
    occurred_at TIMESTAMP,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Retrieval stats
    salience REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    
    -- Consolidation
    consolidation_strength REAL DEFAULT 5.0,
    consolidated_into TEXT REFERENCES episodes(id),
    
    metadata JSON
);

-- Episode links (bidirectional)
CREATE TABLE episode_links (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES episodes(id),
    target_id TEXT REFERENCES episodes(id),
    link_type TEXT,  -- semantic, temporal, causal, etc.
    weight REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_id, target_id, link_type)
);

-- Retrieval indices
CREATE TABLE retrieval_indices (
    id TEXT PRIMARY KEY,
    index_type TEXT NOT NULL,  -- by_emotion, by_concept, by_error, etc.
    key TEXT NOT NULL,
    episode_id TEXT REFERENCES episodes(id),
    note TEXT,
    
    UNIQUE(index_type, key, episode_id)
);
```

**ChromaDB Collections**:
```python
# Episode embeddings for semantic search
episodes_collection = chroma_client.create_collection(
    name="episodes",
    metadata={"hnsw:space": "cosine"}
)

# Entity embeddings
entities_collection = chroma_client.create_collection(
    name="entities", 
    metadata={"hnsw:space": "cosine"}
)
```

---

### 4. Memory Curator (LLM₂)

**Purpose**: Specialized agent for all memory operations.

**Responsibilities**:
```python
class MemoryCurator:
    def __init__(self, llm: OllamaClient, model: str = "mistral"):
        self.llm = llm
        self.model = model
    
    # Formation
    async def evaluate_salience(self, content: str) -> float:
        """Is this worth remembering? Returns 0-1 score."""
        
    async def create_memory_note(self, content: str) -> MemoryNote:
        """Generate structured note with context, keywords, tags."""
        
    # Extraction
    async def extract_entities(self, text: str) -> list[Entity]:
        """Identify entities from episode content."""
        
    async def extract_facts(self, text: str, entities: list[Entity]) -> list[Fact]:
        """Derive relationships and assertions."""
        
    # Resolution
    async def resolve_entity(self, new: Entity, candidates: list[Entity]) -> Entity | None:
        """Match new entity to existing or return None for new."""
        
    async def resolve_fact_conflict(self, new: Fact, existing: Fact) -> FactResolution:
        """Handle contradictions - temporal invalidation or confidence adjustment."""
        
    # Linking
    async def generate_links(self, episode: Episode, existing: list[Episode]) -> list[EpisodeLink]:
        """Find semantic connections to existing memories."""
        
    async def update_existing_context(self, new: Episode, related: list[Episode]) -> list[ContextUpdate]:
        """New memory can update context of existing memories."""
        
    # Consolidation
    async def generate_summary(self, episodes: list[Episode]) -> Episode:
        """Create Level N+1 summary from Level N episodes."""
        
    async def extract_pattern(self, episodes: list[Episode]) -> Pattern | None:
        """Identify recurring pattern across episodes."""
        
    async def distill_principle(self, patterns: list[Pattern]) -> Principle | None:
        """High-level principle from established patterns."""
```

**Prompt Templates** (stored in `prompts/curator/`):
- `salience_evaluation.txt`
- `entity_extraction.txt`
- `fact_extraction.txt`
- `link_generation.txt`
- `summary_generation.txt`
- `pattern_extraction.txt`
- `conflict_resolution.txt`

---

### 5. Memory Interface Layer

**Purpose**: Coordinate queries and writes between LLM₁ and memory stores.

```python
class MemoryInterface:
    def __init__(
        self,
        working: WorkingMemory,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        curator: MemoryCurator
    ):
        self.working = working
        self.semantic = semantic
        self.episodic = episodic
        self.curator = curator
    
    # Query operations
    async def query(self, query: str, context: QueryContext = None) -> RetrievalResult:
        """
        Route query to appropriate stores, synthesize results.
        
        1. Analyze query to determine relevant stores
        2. Dispatch parallel queries
        3. Rank and deduplicate results
        4. Format for context injection
        """
        
    async def query_semantic(self, query: str, temporal: TemporalFilter = None) -> list[Fact]:
        """Query knowledge graph with optional temporal constraints."""
        
    async def query_episodic(
        self, 
        query: str, 
        level: int = None,
        temporal: TemporalFilter = None
    ) -> list[Episode]:
        """Query experiences at specified abstraction level."""
    
    # Write operations
    async def store_interaction(self, interaction: Interaction) -> StorageResult:
        """
        Process new interaction through curator:
        
        1. Curator evaluates salience
        2. If salient: create memory note
        3. Extract entities/facts for semantic memory
        4. Generate links to existing memories
        5. Trigger evolution of related memories
        """
        
    # Working memory management
    async def inject_context(self, items: list[MemoryItem]) -> None:
        """Add retrieved memories to working memory."""
        
    async def handle_memory_pressure(self) -> None:
        """Offload important info before eviction."""
```

---

### 6. Consolidation Engine

**Purpose**: Background process for memory evolution.

```python
class ConsolidationEngine:
    def __init__(
        self,
        curator: MemoryCurator,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        config: ConsolidationConfig
    ):
        self.curator = curator
        self.semantic = semantic
        self.episodic = episodic
        self.config = config
    
    async def run_cycle(self) -> ConsolidationReport:
        """
        Full consolidation cycle (the 'sleep' process):
        
        1. Generate abstractions from recent episodes
        2. Detect patterns across episodes
        3. Resolve accumulated contradictions
        4. Strengthen frequently-accessed links
        5. Prune/archive stale entries
        6. Update community clusters
        """
        
    async def generate_abstractions(self) -> list[Episode]:
        """Cluster recent Level N episodes, create Level N+1 summaries."""
        
    async def detect_patterns(self) -> list[Pattern]:
        """Find recurring themes across episodes."""
        
    async def resolve_contradictions(self) -> list[FactResolution]:
        """Scan for and resolve temporal inconsistencies."""
        
    async def update_link_weights(self) -> int:
        """Strengthen co-accessed links, weaken unused."""
        
    async def prune_stale(self) -> PruneReport:
        """Archive/delete below-threshold memories."""

@dataclass
class ConsolidationConfig:
    # Scheduling
    min_episodes_before_cycle: int = 10
    max_time_between_cycles: timedelta = timedelta(hours=24)
    
    # Thresholds
    abstraction_cluster_size: int = 5
    pattern_min_occurrences: int = 3
    prune_access_threshold: int = 0
    prune_age_threshold: timedelta = timedelta(days=30)
    
    # Limits
    max_episodes_per_cycle: int = 100
```

---

## Project Structure

```
htma/
├── pyproject.toml
├── README.md
├── CLAUDE.md                    # For Claude Code Web context
│
├── src/
│   └── htma/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── types.py         # Core data types
│       │   ├── exceptions.py    # Custom exceptions
│       │   └── utils.py         # Shared utilities
│       │
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── working.py       # Working memory
│       │   ├── semantic.py      # Semantic memory (temporal KG)
│       │   ├── episodic.py      # Episodic memory (hierarchical)
│       │   └── interface.py     # Memory interface layer
│       │
│       ├── curator/
│       │   ├── __init__.py
│       │   ├── curator.py       # Memory curator (LLM₂)
│       │   ├── extractors.py    # Entity/fact extraction
│       │   └── linkers.py       # Link generation
│       │
│       ├── consolidation/
│       │   ├── __init__.py
│       │   ├── engine.py        # Consolidation engine
│       │   ├── abstraction.py   # Summary generation
│       │   └── patterns.py      # Pattern detection
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── sqlite.py        # SQLite operations
│       │   ├── chroma.py        # ChromaDB operations
│       │   └── migrations/      # Schema migrations
│       │       └── 001_initial.sql
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py        # Ollama client wrapper
│       │   └── prompts/         # Prompt templates
│       │       ├── curator/
│       │       │   ├── salience.txt
│       │       │   ├── entity_extraction.txt
│       │       │   ├── fact_extraction.txt
│       │       │   └── ...
│       │       └── reasoner/
│       │           └── ...
│       │
│       └── agent/
│           ├── __init__.py
│           ├── agent.py         # Main agent (LLM₁) 
│           └── tools.py         # Agent tools/functions
│
├── tests/
│   ├── conftest.py
│   ├── test_working_memory.py
│   ├── test_semantic_memory.py
│   ├── test_episodic_memory.py
│   ├── test_curator.py
│   ├── test_consolidation.py
│   └── integration/
│       └── test_full_flow.py
│
├── scripts/
│   ├── init_db.py               # Database initialization
│   ├── run_consolidation.py     # Manual consolidation trigger
│   └── demo.py                  # Interactive demo
│
└── data/                        # Local data (gitignored)
    ├── htma.db                  # SQLite database
    └── chroma/                  # ChromaDB persistence
```

---

## Implementation Phases

### Phase 1: Foundation (Issues #1-5)
- [ ] Project setup (pyproject.toml, structure, deps)
- [ ] Core types and data models
- [ ] SQLite schema and migrations
- [ ] ChromaDB setup
- [ ] Ollama client wrapper

### Phase 2: Memory Stores (Issues #6-9)
- [ ] Working memory with pressure management
- [ ] Semantic memory with bi-temporal queries
- [ ] Episodic memory with hierarchy
- [ ] Memory interface layer

### Phase 3: Curator (Issues #10-14)
- [ ] Salience evaluation
- [ ] Entity/fact extraction
- [ ] Link generation
- [ ] Conflict resolution
- [ ] Memory evolution triggers

### Phase 4: Consolidation (Issues #15-18)
- [ ] Abstraction generation
- [ ] Pattern detection
- [ ] Link strengthening/pruning
- [ ] Full consolidation cycle

### Phase 5: Integration (Issues #19-21)
- [ ] Agent integration (LLM₁)
- [ ] End-to-end flow
- [ ] Interactive demo

---

## Key Design Decisions

### 1. Bi-Temporal Model
Every fact tracks both when it was true (event time) and when we learned it (transaction time). This enables:
- "What did I know as of last week?"
- "When did I learn X?"
- Clean contradiction handling via temporal invalidation

### 2. Separate Curator Model
Using a dedicated (potentially smaller) model for memory operations:
- Reduces load on reasoning model
- Enables fine-tuning for memory tasks
- Clear separation of concerns
- Can run async without blocking interaction

### 3. Hierarchical Episodic Storage
RAPTOR-style recursive abstraction:
- Level 0: Raw episodes
- Level 1: Clustered summaries
- Level 2+: Progressive abstraction
- Enables retrieval at appropriate granularity

### 4. A-MEM Style Linking
Memories actively link and update each other:
- New memories generate links to existing
- Adding memory can update existing memory context
- Network evolves, not just grows

### 5. SQLite + ChromaDB Hybrid
- SQLite: Structured queries, temporal logic, relationships
- ChromaDB: Semantic similarity, embedding search
- Best of both worlds for retrieval

---

## Configuration

```python
# config.py
from pydantic import BaseSettings

class HTMAConfig(BaseSettings):
    # Storage
    sqlite_path: str = "data/htma.db"
    chroma_path: str = "data/chroma"
    
    # LLM
    ollama_url: str = "http://localhost:11434"
    reasoner_model: str = "llama3:8b"
    curator_model: str = "mistral:7b"
    embedding_model: str = "nomic-embed-text"
    
    # Working Memory
    working_memory_max_tokens: int = 8000
    working_memory_pressure_threshold: float = 0.8
    
    # Consolidation
    consolidation_min_episodes: int = 10
    consolidation_cluster_size: int = 5
    consolidation_prune_days: int = 30
    
    # Retrieval
    semantic_retrieval_limit: int = 10
    episodic_retrieval_limit: int = 10
    
    class Config:
        env_prefix = "HTMA_"
```

---

## Dependencies

```toml
[project]
name = "htma"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.4.0",
    "ollama>=0.1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "aiosqlite>=0.19.0",
    "tiktoken>=0.5.0",
    "rich>=13.0",      # For CLI output
    "typer>=0.9.0",    # For CLI
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
```

---

## Next Steps

1. **Create GitHub Issues** from the implementation phases
2. **Generate CLAUDE.md** for Claude Code Web context
3. **Start with Phase 1** foundation work

Ready to proceed?
