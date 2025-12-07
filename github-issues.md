# HTMA MVP Implementation Issues

Use these to create GitHub issues for the HTMA repository.

---

## Phase 1: Foundation

### Issue #1: Project Setup

**Title**: Setup project structure, dependencies, and tooling

**Labels**: `phase-1`, `setup`, `priority-high`

**Description**:

Initialize the HTMA project with proper Python packaging and development tooling.

**Tasks**:
- [ ] Create `pyproject.toml` with dependencies:
  - chromadb>=0.4.0
  - ollama>=0.1.0  
  - pydantic>=2.0
  - pydantic-settings>=2.0
  - aiosqlite>=0.19.0
  - tiktoken>=0.5.0
  - rich>=13.0
  - typer>=0.9.0
  - Dev: pytest, pytest-asyncio, pytest-cov, ruff, mypy
- [ ] Create directory structure per architecture doc
- [ ] Setup `.gitignore` (include `data/`, `__pycache__/`, `.venv/`)
- [ ] Add `README.md` with project overview
- [ ] Configure ruff and mypy in `pyproject.toml`
- [ ] Create empty `__init__.py` files for package structure

**Acceptance Criteria**:
- `pip install -e ".[dev]"` works
- `ruff check src/` runs without config errors
- `mypy src/` runs without config errors
- `pytest` discovers test directory

---

### Issue #2: Core Types and Data Models

**Title**: Define core data types and Pydantic models

**Labels**: `phase-1`, `core`, `priority-high`

**Description**:

Create the foundational data types used throughout the system.

**Location**: `src/htma/core/types.py`

**Models to implement**:

```python
# Identifiers
class EntityID(str): ...
class FactID(str): ...
class EpisodeID(str): ...

# Temporal
class TemporalRange(BaseModel):
    valid_from: datetime | None
    valid_to: datetime | None

class BiTemporalRecord(BaseModel):
    event_time: TemporalRange      # When true in world
    transaction_time: TemporalRange # When recorded

# Entities
class Entity(BaseModel):
    id: EntityID
    name: str
    entity_type: str  # person, place, concept, object, event
    created_at: datetime
    last_accessed: datetime | None
    access_count: int = 0
    metadata: dict = {}

# Facts
class Fact(BaseModel):
    id: FactID
    subject_id: EntityID
    predicate: str
    object_id: EntityID | None = None
    object_value: str | None = None
    temporal: BiTemporalRecord
    confidence: float = 1.0
    source_episode_id: EpisodeID | None = None
    metadata: dict = {}

# Episodes
class Episode(BaseModel):
    id: EpisodeID
    level: int  # 0=raw, 1=summary, 2+=abstraction
    parent_id: EpisodeID | None = None
    content: str
    summary: str | None = None
    context_description: str | None = None
    keywords: list[str] = []
    tags: list[str] = []
    occurred_at: datetime
    recorded_at: datetime
    salience: float = 0.5
    consolidation_strength: float = 5.0
    access_count: int = 0
    last_accessed: datetime | None = None
    metadata: dict = {}

# Episode Links
class EpisodeLink(BaseModel):
    id: str
    source_id: EpisodeID
    target_id: EpisodeID
    link_type: str  # semantic, temporal, causal, analogical
    weight: float = 1.0
    created_at: datetime

# Memory Notes (A-MEM style)
class MemoryNote(BaseModel):
    content: str
    context: str
    keywords: list[str]
    tags: list[str]
    salience: float

# Query/Retrieval
class TemporalFilter(BaseModel):
    as_of: datetime | None = None      # Transaction time filter
    valid_at: datetime | None = None   # Event time filter

class RetrievalResult(BaseModel):
    facts: list[Fact] = []
    episodes: list[Episode] = []
    relevance_scores: dict[str, float] = {}
```

**Also create**:
- `src/htma/core/exceptions.py` with custom exceptions
- `src/htma/core/utils.py` with ID generation utilities

**Acceptance Criteria**:
- All models validate correctly
- Models serialize to/from JSON
- ID generators produce unique, sortable IDs
- Unit tests pass

---

### Issue #3: SQLite Schema and Migrations

**Title**: Implement SQLite schema with migration support

**Labels**: `phase-1`, `storage`, `priority-high`

**Description**:

Create the SQLite database schema and migration system.

**Location**: `src/htma/storage/sqlite.py`, `src/htma/storage/migrations/`

**Schema** (in `migrations/001_initial.sql`):

```sql
-- Entities
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    metadata JSON
);

-- Facts with bi-temporal model
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id TEXT REFERENCES entities(id),
    object_value TEXT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    invalidated_at TIMESTAMP,
    confidence REAL DEFAULT 1.0,
    source_episode_id TEXT,
    metadata JSON
);

-- Episodes
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    level INTEGER NOT NULL DEFAULT 0,
    parent_id TEXT REFERENCES episodes(id),
    content TEXT NOT NULL,
    summary TEXT,
    context_description TEXT,
    keywords JSON,
    tags JSON,
    occurred_at TIMESTAMP,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    salience REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    consolidation_strength REAL DEFAULT 5.0,
    consolidated_into TEXT REFERENCES episodes(id),
    metadata JSON
);

-- Episode links
CREATE TABLE IF NOT EXISTS episode_links (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES episodes(id),
    target_id TEXT NOT NULL REFERENCES episodes(id),
    link_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, target_id, link_type)
);

-- Retrieval indices
CREATE TABLE IF NOT EXISTS retrieval_indices (
    id TEXT PRIMARY KEY,
    index_type TEXT NOT NULL,
    key TEXT NOT NULL,
    episode_id TEXT NOT NULL REFERENCES episodes(id),
    note TEXT,
    UNIQUE(index_type, key, episode_id)
);

-- Communities
CREATE TABLE IF NOT EXISTS communities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    entity_ids JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id);
CREATE INDEX IF NOT EXISTS idx_facts_valid_period ON facts(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_recorded ON facts(recorded_at);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_episodes_level ON episodes(level);
CREATE INDEX IF NOT EXISTS idx_episodes_occurred ON episodes(occurred_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_type_key ON retrieval_indices(index_type, key);
```

**SQLite wrapper class**:

```python
class SQLiteStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def initialize(self) -> None:
        """Run migrations and setup database."""
    
    async def execute(self, query: str, params: tuple = ()) -> None:
        """Execute a write query."""
    
    async def fetch_one(self, query: str, params: tuple = ()) -> dict | None:
        """Fetch single row."""
    
    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Fetch all rows."""
    
    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager."""
```

**Acceptance Criteria**:
- Database initializes with all tables
- Migration runs idempotently
- CRUD operations work for all tables
- Temporal queries work correctly
- Unit tests pass

---

### Issue #4: ChromaDB Setup

**Title**: Implement ChromaDB vector storage

**Labels**: `phase-1`, `storage`, `priority-high`

**Description**:

Setup ChromaDB for vector embeddings and semantic search.

**Location**: `src/htma/storage/chroma.py`

**Implementation**:

```python
class ChromaStorage:
    def __init__(self, persist_path: str, embedding_model: str = "nomic-embed-text"):
        self.persist_path = persist_path
        self.embedding_model = embedding_model
        self.client: chromadb.Client | None = None
        self.episodes_collection: Collection | None = None
        self.entities_collection: Collection | None = None
    
    async def initialize(self) -> None:
        """Setup ChromaDB client and collections."""
    
    async def add_episode(self, episode: Episode) -> None:
        """Add episode with embedding."""
    
    async def add_entity(self, entity: Entity) -> None:
        """Add entity with embedding."""
    
    async def search_episodes(
        self, 
        query: str, 
        n_results: int = 10,
        where: dict | None = None
    ) -> list[tuple[EpisodeID, float]]:
        """Semantic search over episodes."""
    
    async def search_entities(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None
    ) -> list[tuple[EntityID, float]]:
        """Semantic search over entities."""
    
    async def delete_episode(self, episode_id: EpisodeID) -> None:
        """Remove episode from collection."""
    
    async def update_episode(self, episode: Episode) -> None:
        """Update episode embedding."""
```

**Collections**:
- `episodes`: Episode embeddings with metadata (level, salience, occurred_at)
- `entities`: Entity embeddings with metadata (entity_type)

**Embedding generation**:
- Use Ollama's embedding endpoint
- Model: `nomic-embed-text` (or configurable)

**Acceptance Criteria**:
- Collections persist across restarts
- Semantic search returns relevant results
- Metadata filtering works
- Unit tests pass

---

### Issue #5: Ollama Client Wrapper

**Title**: Implement Ollama client for LLM operations

**Labels**: `phase-1`, `llm`, `priority-high`

**Description**:

Create async wrapper for Ollama API supporting both chat and embeddings.

**Location**: `src/htma/llm/client.py`

**Implementation**:

```python
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate completion."""
    
    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Chat completion."""
    
    async def embed(
        self,
        model: str,
        text: str
    ) -> list[float]:
        """Generate embedding vector."""
    
    async def embed_batch(
        self,
        model: str,
        texts: list[str]
    ) -> list[list[float]]:
        """Batch embedding generation."""
    
    async def health_check(self) -> bool:
        """Check if Ollama is available."""
    
    async def list_models(self) -> list[str]:
        """List available models."""
```

**Error handling**:
- Connection errors with retry
- Model not found errors
- Timeout handling
- Rate limiting (if needed)

**Acceptance Criteria**:
- Chat completions work
- Embeddings generate correctly
- Error handling is robust
- Health check detects Ollama availability
- Integration tests pass (require running Ollama)

---

## Phase 2: Memory Stores

### Issue #6: Working Memory Implementation

**Title**: Implement working memory with pressure management

**Labels**: `phase-2`, `memory`, `priority-high`

**Description**:

Create the working memory component that manages in-context information.

**Location**: `src/htma/memory/working.py`

**Implementation**:

```python
@dataclass
class WorkingMemoryConfig:
    max_tokens: int = 8000
    pressure_threshold: float = 0.8
    summarization_model: str = "mistral:7b"

class WorkingMemory:
    def __init__(self, config: WorkingMemoryConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm
        self.system_context: str = ""
        self.task_context: str = ""
        self.dialog_history: list[Message] = []
        self.retrieved_context: list[MemoryItem] = []
    
    @property
    def current_tokens(self) -> int:
        """Calculate current token usage."""
    
    @property
    def utilization(self) -> float:
        """Current usage as fraction of max."""
    
    @property
    def under_pressure(self) -> bool:
        """True if above pressure threshold."""
    
    def set_system_context(self, context: str) -> None:
        """Set static system instructions."""
    
    def set_task_context(self, context: str) -> None:
        """Set current task information."""
    
    def add_message(self, message: Message) -> None:
        """Add dialog turn, evict old if needed."""
    
    def add_retrieved(self, items: list[MemoryItem]) -> None:
        """Add retrieved memories to context."""
    
    def clear_retrieved(self) -> None:
        """Clear retrieved context."""
    
    async def handle_pressure(self) -> list[MemoryItem]:
        """
        Handle memory pressure:
        1. Identify important items to offload
        2. Summarize dialog history
        3. Return items that should be persisted
        """
    
    def render_context(self) -> str:
        """Render full context for LLM."""
    
    def get_offload_candidates(self) -> list[MemoryItem]:
        """Get items that can be safely evicted."""
```

**Token counting**: Use `tiktoken` with appropriate encoding.

**Eviction priority** (lowest first):
1. Retrieved context (can be re-retrieved)
2. Old dialog turns
3. Task context
4. System context (never evict)

**Acceptance Criteria**:
- Token counting is accurate
- Pressure detection works
- Eviction follows priority order
- Summarization produces coherent summaries
- Unit tests pass

---

### Issue #7: Semantic Memory with Bi-Temporal Queries

**Title**: Implement semantic memory (temporal knowledge graph)

**Labels**: `phase-2`, `memory`, `priority-high`

**Description**:

Create the semantic memory component with full bi-temporal support.

**Location**: `src/htma/memory/semantic.py`

**Implementation**:

```python
class SemanticMemory:
    def __init__(self, sqlite: SQLiteStorage, chroma: ChromaStorage):
        self.sqlite = sqlite
        self.chroma = chroma
    
    # Entity operations
    async def add_entity(self, entity: Entity) -> Entity:
        """Add new entity."""
    
    async def get_entity(self, entity_id: EntityID) -> Entity | None:
        """Get entity by ID."""
    
    async def find_entity(self, name: str, entity_type: str | None = None) -> list[Entity]:
        """Find entities by name (fuzzy match)."""
    
    async def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        """Semantic search over entities."""
    
    # Fact operations
    async def add_fact(self, fact: Fact) -> Fact:
        """Add new fact."""
    
    async def invalidate_fact(self, fact_id: FactID, as_of: datetime) -> None:
        """Mark fact as invalid from given time."""
    
    # Temporal queries
    async def query_entity_facts(
        self,
        entity_id: EntityID,
        predicate: str | None = None,
        temporal: TemporalFilter | None = None
    ) -> list[Fact]:
        """Get facts about entity with temporal filtering."""
    
    async def query_at_time(
        self,
        entity_id: EntityID,
        as_of: datetime
    ) -> list[Fact]:
        """What did we know about entity as of transaction time?"""
    
    async def query_valid_at(
        self,
        entity_id: EntityID,
        when: datetime
    ) -> list[Fact]:
        """What facts were true about entity at event time?"""
    
    async def get_fact_history(
        self,
        subject_id: EntityID,
        predicate: str
    ) -> list[Fact]:
        """Get all versions of a fact over time."""
    
    # Access tracking
    async def record_access(self, entity_id: EntityID) -> None:
        """Update access count and timestamp."""
    
    # Community operations
    async def add_to_community(self, entity_id: EntityID, community_id: str) -> None:
        """Add entity to community cluster."""
    
    async def get_community_entities(self, community_id: str) -> list[Entity]:
        """Get all entities in a community."""
```

**Bi-temporal query logic**:

```sql
-- Query as of transaction time (what we knew when)
SELECT * FROM facts 
WHERE subject_id = ? 
  AND recorded_at <= ?
  AND (invalidated_at IS NULL OR invalidated_at > ?)

-- Query valid at event time (what was true when)
SELECT * FROM facts
WHERE subject_id = ?
  AND (valid_from IS NULL OR valid_from <= ?)
  AND (valid_to IS NULL OR valid_to > ?)
  AND invalidated_at IS NULL
```

**Acceptance Criteria**:
- Entity CRUD works
- Fact CRUD works
- Bi-temporal queries return correct results
- Semantic search finds relevant entities
- Access tracking updates correctly
- Unit tests pass

---

### Issue #8: Episodic Memory with Hierarchy

**Title**: Implement hierarchical episodic memory

**Labels**: `phase-2`, `memory`, `priority-high`

**Description**:

Create the episodic memory component with RAPTOR-style hierarchical storage.

**Location**: `src/htma/memory/episodic.py`

**Implementation**:

```python
class EpisodicMemory:
    def __init__(self, sqlite: SQLiteStorage, chroma: ChromaStorage):
        self.sqlite = sqlite
        self.chroma = chroma
    
    # Episode operations
    async def add_episode(self, episode: Episode) -> Episode:
        """Add new episode at specified level."""
    
    async def get_episode(self, episode_id: EpisodeID) -> Episode | None:
        """Get episode by ID."""
    
    async def get_children(self, episode_id: EpisodeID) -> list[Episode]:
        """Get child episodes (lower level)."""
    
    async def get_parent(self, episode_id: EpisodeID) -> Episode | None:
        """Get parent episode (higher level)."""
    
    # Retrieval
    async def search(
        self,
        query: str,
        level: int | None = None,
        temporal: TemporalFilter | None = None,
        limit: int = 10
    ) -> list[Episode]:
        """
        Semantic search with optional filters:
        - level: Specific abstraction level
        - temporal: Time range filter
        """
    
    async def get_recent(
        self,
        level: int = 0,
        limit: int = 10
    ) -> list[Episode]:
        """Get most recent episodes at level."""
    
    async def get_by_index(
        self,
        index_type: str,
        key: str
    ) -> list[Episode]:
        """Retrieve via retrieval index."""
    
    # Linking
    async def add_link(self, link: EpisodeLink) -> None:
        """Create bidirectional link between episodes."""
    
    async def get_links(
        self,
        episode_id: EpisodeID,
        link_type: str | None = None
    ) -> list[EpisodeLink]:
        """Get links for episode."""
    
    async def get_linked_episodes(
        self,
        episode_id: EpisodeID,
        link_type: str | None = None
    ) -> list[Episode]:
        """Get episodes linked to given episode."""
    
    async def update_link_weight(
        self,
        source_id: EpisodeID,
        target_id: EpisodeID,
        delta: float
    ) -> None:
        """Adjust link weight."""
    
    # Indexing
    async def add_index_entry(
        self,
        index_type: str,
        key: str,
        episode_id: EpisodeID,
        note: str | None = None
    ) -> None:
        """Add retrieval index entry."""
    
    async def get_index_keys(self, index_type: str) -> list[str]:
        """Get all keys for an index type."""
    
    # Access tracking
    async def record_access(self, episode_id: EpisodeID) -> None:
        """Update access count and timestamp."""
    
    # Consolidation support
    async def mark_consolidated(
        self,
        episode_id: EpisodeID,
        consolidated_into: EpisodeID
    ) -> None:
        """Mark episode as consolidated into higher-level."""
    
    async def get_unconsolidated(
        self,
        level: int,
        older_than: datetime
    ) -> list[Episode]:
        """Get episodes ready for consolidation."""
```

**Hierarchy levels**:
- Level 0: Raw episodes (full interactions)
- Level 1: Summaries (clustered episodes)
- Level 2+: Progressive abstractions

**Acceptance Criteria**:
- Episode CRUD works at all levels
- Hierarchy navigation works
- Semantic search finds relevant episodes
- Linking works bidirectionally
- Retrieval indices work
- Access tracking updates correctly
- Unit tests pass

---

### Issue #9: Memory Interface Layer

**Title**: Implement memory interface for query routing and synthesis

**Labels**: `phase-2`, `memory`, `priority-high`

**Description**:

Create the coordination layer between LLM₁ and memory stores.

**Location**: `src/htma/memory/interface.py`

**Implementation**:

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
    async def query(
        self,
        query: str,
        include_semantic: bool = True,
        include_episodic: bool = True,
        temporal: TemporalFilter | None = None,
        limit: int = 10
    ) -> RetrievalResult:
        """
        Route query to appropriate stores:
        1. Analyze query for routing
        2. Dispatch parallel queries
        3. Rank and deduplicate
        4. Return synthesized result
        """
    
    async def query_semantic(
        self,
        query: str,
        temporal: TemporalFilter | None = None
    ) -> list[Fact]:
        """Direct semantic memory query."""
    
    async def query_episodic(
        self,
        query: str,
        level: int | None = None,
        temporal: TemporalFilter | None = None
    ) -> list[Episode]:
        """Direct episodic memory query."""
    
    # Write operations
    async def store_interaction(
        self,
        interaction: Interaction
    ) -> StorageResult:
        """
        Process new interaction:
        1. Curator evaluates salience
        2. Create memory note if salient
        3. Extract entities/facts
        4. Generate links
        5. Trigger evolution
        """
    
    # Context management
    async def inject_context(
        self,
        result: RetrievalResult
    ) -> None:
        """Add retrieved memories to working memory."""
    
    async def handle_memory_pressure(self) -> None:
        """Coordinate pressure handling with curator."""
    
    # Utility
    async def get_related_context(
        self,
        episode_id: EpisodeID,
        depth: int = 1
    ) -> list[Episode]:
        """Get episodes related via links."""
```

**Query routing logic**:
- Entity names detected → include semantic
- Temporal markers → add temporal filter
- Experience references → include episodic
- Default: search both

**Result synthesis**:
- Deduplicate overlapping results
- Rank by relevance + recency
- Format for context injection

**Acceptance Criteria**:
- Query routing works correctly
- Parallel queries improve latency
- Results are properly synthesized
- Storage flow triggers curator
- Context injection works
- Integration tests pass

---

## Phase 3: Curator

### Issue #10: Salience Evaluation

**Title**: Implement salience evaluation for memory formation

**Labels**: `phase-3`, `curator`, `priority-high`

**Description**:

Create the salience evaluation component that determines what's worth remembering.

**Location**: `src/htma/curator/curator.py`

**Implementation**:

```python
class MemoryCurator:
    def __init__(self, llm: OllamaClient, model: str = "mistral:7b"):
        self.llm = llm
        self.model = model
    
    async def evaluate_salience(self, content: str, context: str = "") -> SalienceResult:
        """
        Evaluate if content is worth remembering.
        
        Returns:
            SalienceResult with:
            - score: 0.0-1.0
            - reasoning: Why this score
            - memory_type: semantic, episodic, or both
        """
```

**Prompt template** (`prompts/curator/salience.txt`):

```
You are evaluating whether content should be stored in long-term memory.

Context of conversation:
{context}

Content to evaluate:
{content}

Consider:
1. Does this contain new information not already known?
2. Is this likely to be useful in future conversations?
3. Does this reveal user preferences, facts, or important events?
4. Would forgetting this degrade future interactions?

Respond with JSON:
{
  "score": 0.0-1.0,
  "reasoning": "brief explanation",
  "memory_type": "semantic" | "episodic" | "both",
  "key_elements": ["list", "of", "important", "items"]
}
```

**Salience thresholds**:
- 0.0-0.3: Don't store
- 0.3-0.6: Store minimal
- 0.6-0.8: Store standard
- 0.8-1.0: Store rich

**Acceptance Criteria**:
- Returns valid scores
- Reasoning is coherent
- Memory type classification is accurate
- Handles edge cases (empty, very long)
- Unit tests pass

---

### Issue #11: Entity and Fact Extraction

**Title**: Implement entity and fact extraction

**Labels**: `phase-3`, `curator`, `priority-high`

**Description**:

Create extractors for identifying entities and facts from text.

**Location**: `src/htma/curator/extractors.py`

**Implementation**:

```python
class EntityExtractor:
    def __init__(self, llm: OllamaClient, model: str):
        self.llm = llm
        self.model = model
    
    async def extract(self, text: str) -> list[ExtractedEntity]:
        """
        Extract entities from text.
        
        Returns list of:
        - name: Entity name
        - type: person, place, concept, object, event
        - mentions: Where in text
        - confidence: Extraction confidence
        """

class FactExtractor:
    def __init__(self, llm: OllamaClient, model: str):
        self.llm = llm
        self.model = model
    
    async def extract(
        self, 
        text: str, 
        entities: list[ExtractedEntity]
    ) -> list[ExtractedFact]:
        """
        Extract facts/relationships from text.
        
        Returns list of:
        - subject: Entity name
        - predicate: Relationship type
        - object: Entity name or literal value
        - temporal: Any temporal markers
        - confidence: Extraction confidence
        """
```

**Prompt templates**:
- `prompts/curator/entity_extraction.txt`
- `prompts/curator/fact_extraction.txt`

**Entity types**: person, place, organization, concept, object, event, time

**Common predicates**: is_a, has_property, located_in, works_at, owns, prefers, said, believes, happened_at

**Acceptance Criteria**:
- Extracts entities accurately
- Identifies relationships correctly
- Handles temporal markers
- Confidence scores are reasonable
- Unit tests pass

---

### Issue #12: Link Generation

**Title**: Implement memory link generation

**Labels**: `phase-3`, `curator`, `priority-medium`

**Description**:

Create the linking component that connects new memories to existing ones.

**Location**: `src/htma/curator/linkers.py`

**Implementation**:

```python
class LinkGenerator:
    def __init__(
        self, 
        llm: OllamaClient, 
        model: str,
        episodic: EpisodicMemory
    ):
        self.llm = llm
        self.model = model
        self.episodic = episodic
    
    async def generate_links(
        self,
        new_episode: Episode,
        candidate_limit: int = 20
    ) -> list[EpisodeLink]:
        """
        Find and create links to existing memories.
        
        Process:
        1. Semantic search for candidates
        2. LLM evaluates connection strength
        3. Return weighted links
        """
    
    async def evaluate_connection(
        self,
        episode_a: Episode,
        episode_b: Episode
    ) -> LinkEvaluation:
        """
        Evaluate if two episodes should be linked.
        
        Returns:
        - should_link: bool
        - link_type: semantic, temporal, causal, analogical
        - weight: 0.0-1.0
        - reasoning: explanation
        """
```

**Link types**:
- **semantic**: Similar topics/concepts
- **temporal**: Close in time or part of sequence
- **causal**: One led to or caused another
- **analogical**: Similar patterns/structures

**Acceptance Criteria**:
- Finds relevant candidates via search
- Link evaluation is accurate
- Creates bidirectional links
- Weight assignment is meaningful
- Unit tests pass

---

### Issue #13: Conflict Resolution

**Title**: Implement fact conflict resolution

**Labels**: `phase-3`, `curator`, `priority-medium`

**Description**:

Handle contradictions between new and existing facts.

**Location**: `src/htma/curator/curator.py` (extend MemoryCurator)

**Implementation**:

```python
class MemoryCurator:
    # ... existing methods ...
    
    async def resolve_conflict(
        self,
        new_fact: Fact,
        existing_facts: list[Fact]
    ) -> ConflictResolution:
        """
        Resolve contradiction between new and existing facts.
        
        Strategies:
        1. Temporal succession: Old becomes invalid, new is current
        2. Confidence adjustment: Lower confidence on uncertain
        3. Coexistence: Both can be true in different contexts
        4. Rejection: New fact is likely wrong
        
        Returns:
        - strategy: Which resolution to apply
        - invalidations: Facts to invalidate with timestamps
        - confidence_updates: Facts with new confidence scores
        - reasoning: Explanation
        """
    
    async def detect_conflicts(
        self,
        new_facts: list[Fact],
        semantic: SemanticMemory
    ) -> list[FactConflict]:
        """
        Find existing facts that conflict with new ones.
        """
```

**Resolution strategies**:

1. **Temporal succession**: Most common. Old fact was true, now new fact is true.
   - Set `valid_to` on old fact
   - Set `valid_from` on new fact

2. **Confidence adjustment**: When uncertain which is correct.
   - Lower confidence on less certain fact
   - May need user clarification

3. **Coexistence**: Facts apply in different contexts.
   - Add context metadata to distinguish
   - Both remain valid

4. **Rejection**: New information appears wrong.
   - Don't add new fact
   - Log for potential review

**Acceptance Criteria**:
- Detects conflicts correctly
- Applies appropriate strategy
- Temporal invalidation works
- Confidence updates work
- Reasoning is coherent
- Unit tests pass

---

### Issue #14: Memory Evolution Triggers

**Title**: Implement memory evolution when new memories arrive

**Labels**: `phase-3`, `curator`, `priority-medium`

**Description**:

New memories can trigger updates to existing memory context (A-MEM style).

**Location**: `src/htma/curator/curator.py` (extend MemoryCurator)

**Implementation**:

```python
class MemoryCurator:
    # ... existing methods ...
    
    async def trigger_evolution(
        self,
        new_episode: Episode,
        related_episodes: list[Episode]
    ) -> list[EpisodeUpdate]:
        """
        Check if new episode should update existing memories.
        
        Updates can include:
        - Context description refinement
        - New keywords added
        - Tag updates
        - Salience adjustment
        
        Returns list of updates to apply.
        """
    
    async def evaluate_evolution(
        self,
        new_episode: Episode,
        existing_episode: Episode
    ) -> EpisodeUpdate | None:
        """
        Evaluate if existing episode should be updated.
        
        Consider:
        - Does new episode provide context for old?
        - Does new episode change significance of old?
        - Should old episode link to new?
        """
```

**Evolution types**:
- **Context enrichment**: New info explains or contextualizes old
- **Significance change**: New events change importance of old
- **Pattern recognition**: New episode confirms pattern from old
- **Contradiction**: New episode contradicts old (trigger resolution)

**Acceptance Criteria**:
- Identifies evolution opportunities
- Updates are meaningful
- Doesn't create circular updates
- Respects consolidation_strength
- Unit tests pass

---

## Phase 4: Consolidation

### Issue #15: Abstraction Generation

**Title**: Implement summary/abstraction generation

**Labels**: `phase-4`, `consolidation`, `priority-high`

**Description**:

Create higher-level summaries from clusters of lower-level episodes.

**Location**: `src/htma/consolidation/abstraction.py`

**Implementation**:

```python
class AbstractionGenerator:
    def __init__(self, llm: OllamaClient, model: str):
        self.llm = llm
        self.model = model
    
    async def cluster_episodes(
        self,
        episodes: list[Episode],
        cluster_size: int = 5
    ) -> list[list[Episode]]:
        """
        Group related episodes for summarization.
        Uses embedding similarity + temporal proximity.
        """
    
    async def generate_summary(
        self,
        episodes: list[Episode],
        level: int
    ) -> Episode:
        """
        Create Level N+1 episode from Level N episodes.
        
        Summary includes:
        - Consolidated content
        - Combined keywords
        - Unified tags
        - Average salience
        - Links to source episodes
        """
    
    async def should_abstract(
        self,
        episodes: list[Episode]
    ) -> bool:
        """
        Evaluate if episodes are ready for abstraction.
        
        Criteria:
        - Sufficient count
        - Sufficient age
        - Semantic coherence
        """
```

**Clustering approach**:
1. Embed all episodes
2. Hierarchical clustering by similarity
3. Respect temporal boundaries (don't cluster distant episodes)
4. Target cluster_size episodes per summary

**Summary prompt** should:
- Preserve key information
- Identify themes
- Maintain temporal ordering
- Note significance

**Acceptance Criteria**:
- Clustering groups related episodes
- Summaries capture key content
- Hierarchy is correctly maintained
- Parent-child links work
- Unit tests pass

---

### Issue #16: Pattern Detection

**Title**: Implement pattern detection across episodes

**Labels**: `phase-4`, `consolidation`, `priority-medium`

**Description**:

Identify recurring patterns and themes across episodic memory.

**Location**: `src/htma/consolidation/patterns.py`

**Implementation**:

```python
@dataclass
class Pattern:
    id: str
    description: str
    pattern_type: str  # behavioral, preference, procedural, error
    confidence: float
    occurrences: list[EpisodeID]
    first_seen: datetime
    last_seen: datetime
    consolidation_strength: float

class PatternDetector:
    def __init__(self, llm: OllamaClient, model: str):
        self.llm = llm
        self.model = model
    
    async def detect_patterns(
        self,
        episodes: list[Episode],
        existing_patterns: list[Pattern]
    ) -> PatternDetectionResult:
        """
        Identify patterns in episode collection.
        
        Returns:
        - new_patterns: Newly discovered patterns
        - strengthened: Existing patterns with new evidence
        - weakened: Patterns not seen recently
        """
    
    async def extract_pattern(
        self,
        episodes: list[Episode]
    ) -> Pattern | None:
        """
        Extract pattern from related episodes.
        """
    
    async def match_to_existing(
        self,
        candidate: Pattern,
        existing: list[Pattern]
    ) -> Pattern | None:
        """
        Check if candidate matches existing pattern.
        """
```

**Pattern types**:
- **behavioral**: User tends to do X
- **preference**: User prefers X over Y
- **procedural**: Steps for accomplishing X
- **error**: Common mistake pattern

**Pattern lifecycle**:
- Emerging (1-2 occurrences, low confidence)
- Established (3+ occurrences, medium confidence)
- Consolidated (10+ occurrences, high confidence, becomes principle)

**Acceptance Criteria**:
- Identifies recurring themes
- Tracks pattern evidence
- Strengthens confirmed patterns
- Handles pattern evolution
- Unit tests pass

---

### Issue #17: Link Maintenance

**Title**: Implement link strengthening and pruning

**Labels**: `phase-4`, `consolidation`, `priority-medium`

**Description**:

Maintain link health through access-based strengthening and decay.

**Location**: `src/htma/consolidation/engine.py`

**Implementation**:

```python
class ConsolidationEngine:
    # ... other methods ...
    
    async def update_link_weights(self) -> LinkMaintenanceReport:
        """
        Update link weights based on access patterns.
        
        Process:
        1. Strengthen co-accessed links
        2. Decay unused links
        3. Prune links below threshold
        """
    
    async def strengthen_coaccessed(
        self,
        access_window: timedelta = timedelta(hours=1)
    ) -> int:
        """
        Links between episodes accessed together get stronger.
        """
    
    async def decay_unused(
        self,
        decay_rate: float = 0.1,
        min_weight: float = 0.1
    ) -> int:
        """
        Links not used decay over time.
        """
    
    async def prune_weak_links(
        self,
        threshold: float = 0.1
    ) -> int:
        """
        Remove links below weight threshold.
        """
```

**Strengthening logic**:
- When Episode A accessed, check if Episode B accessed within window
- If yes, strengthen A↔B link
- Strengthening is logarithmic (diminishing returns)

**Decay logic**:
- Each consolidation cycle, decay all links by rate
- Frequently used links get strengthened faster than decay
- Unused links eventually prune

**Acceptance Criteria**:
- Co-access strengthens links
- Unused links decay
- Pruning removes weak links
- Doesn't prune important links
- Unit tests pass

---

### Issue #18: Full Consolidation Cycle

**Title**: Implement complete consolidation cycle orchestration

**Labels**: `phase-4`, `consolidation`, `priority-high`

**Description**:

Orchestrate full "sleep" cycle that evolves memory.

**Location**: `src/htma/consolidation/engine.py`

**Implementation**:

```python
@dataclass
class ConsolidationConfig:
    min_episodes_before_cycle: int = 10
    max_time_between_cycles: timedelta = timedelta(hours=24)
    abstraction_cluster_size: int = 5
    pattern_min_occurrences: int = 3
    prune_access_threshold: int = 0
    prune_age_threshold: timedelta = timedelta(days=30)
    max_episodes_per_cycle: int = 100

@dataclass  
class ConsolidationReport:
    abstractions_created: int
    patterns_detected: int
    patterns_strengthened: int
    conflicts_resolved: int
    links_strengthened: int
    links_pruned: int
    episodes_pruned: int
    duration: timedelta

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
        self.last_cycle: datetime | None = None
    
    async def should_run(self) -> bool:
        """Check if consolidation should run."""
    
    async def run_cycle(self) -> ConsolidationReport:
        """
        Full consolidation cycle:
        
        1. Generate abstractions
           - Get unconsolidated Level 0 episodes
           - Cluster and summarize into Level 1
           - Recursively up the hierarchy
        
        2. Detect patterns
           - Analyze recent episodes for patterns
           - Strengthen existing patterns with new evidence
           - Create new patterns if threshold met
        
        3. Resolve contradictions
           - Scan semantic memory for conflicts
           - Apply resolution strategies
        
        4. Maintain links
           - Strengthen co-accessed links
           - Decay unused links
           - Prune weak links
        
        5. Prune stale content
           - Archive/delete old, unaccessed memories
           - Respect consolidation_strength
        
        6. Update metadata
           - Record cycle completion
           - Update statistics
        """
    
    async def prune_stale(self) -> PruneReport:
        """
        Remove or archive stale memories.
        
        Criteria:
        - access_count below threshold
        - age above threshold
        - consolidation_strength below threshold
        - Already consolidated into higher level
        """
```

**Cycle triggers**:
- Sufficient new episodes accumulated
- Time since last cycle exceeded
- Manual trigger
- Memory pressure

**Acceptance Criteria**:
- Full cycle completes successfully
- All sub-processes run
- Report is accurate
- Memory evolves appropriately
- No data corruption
- Integration tests pass

---

## Phase 5: Integration

### Issue #19: Agent Integration

**Title**: Integrate memory system with reasoning agent (LLM₁)

**Labels**: `phase-5`, `agent`, `priority-high`

**Description**:

Create the main agent that uses memory for enhanced conversations.

**Location**: `src/htma/agent/agent.py`

**Implementation**:

```python
class HTMAAgent:
    def __init__(
        self,
        llm: OllamaClient,
        reasoner_model: str,
        memory: MemoryInterface,
        config: AgentConfig
    ):
        self.llm = llm
        self.reasoner_model = reasoner_model
        self.memory = memory
        self.config = config
    
    async def process_message(
        self,
        message: str,
        conversation_id: str | None = None
    ) -> AgentResponse:
        """
        Process user message with memory augmentation.
        
        Flow:
        1. Query relevant memories
        2. Inject context into working memory
        3. Generate response
        4. Store interaction (async)
        5. Return response
        """
    
    async def query_memory(self, query: str) -> RetrievalResult:
        """Explicit memory query."""
    
    async def start_conversation(self) -> str:
        """Initialize new conversation, return ID."""
    
    async def end_conversation(self, conversation_id: str) -> None:
        """Finalize conversation, trigger storage."""
```

**Memory-augmented prompting**:
```
System: You are a helpful assistant with access to long-term memory.

Relevant memories:
{retrieved_context}

Current conversation:
{dialog_history}

User: {message}
```

**Acceptance Criteria**:
- Conversations work end-to-end
- Memory retrieval augments responses
- Interactions are stored
- Handles multi-turn conversations
- Integration tests pass

---

### Issue #20: End-to-End Flow Testing

**Title**: Create comprehensive end-to-end tests

**Labels**: `phase-5`, `testing`, `priority-high`

**Description**:

Test the complete system flow from user input to memory evolution.

**Location**: `tests/integration/test_full_flow.py`

**Test scenarios**:

1. **Basic memory formation**
   - User mentions fact → stored in semantic memory
   - Verify retrieval in subsequent query

2. **Episodic storage and retrieval**
   - Multi-turn conversation
   - Verify episodes created
   - Verify semantic search finds relevant episodes

3. **Temporal reasoning**
   - Fact changes over time
   - Old fact invalidated, new fact current
   - Historical query returns old fact

4. **Memory linking**
   - Related conversations occur
   - Verify links created
   - Verify linked retrieval works

5. **Consolidation**
   - Accumulate episodes
   - Run consolidation
   - Verify abstractions created
   - Verify patterns detected

6. **Memory pressure**
   - Fill working memory
   - Verify offload happens
   - Verify important info persisted

7. **Conflict resolution**
   - Contradictory facts introduced
   - Verify resolution applied
   - Verify temporal validity correct

**Acceptance Criteria**:
- All scenarios pass
- No race conditions
- Proper cleanup
- Reasonable performance

---

### Issue #21: Interactive Demo

**Title**: Create interactive demo CLI

**Labels**: `phase-5`, `demo`, `priority-medium`

**Description**:

Build a CLI demo showcasing HTMA capabilities.

**Location**: `scripts/demo.py`

**Features**:

```python
# CLI commands
htma chat           # Interactive conversation
htma query <text>   # Query memory directly
htma consolidate    # Trigger consolidation
htma status         # Show memory statistics
htma export         # Export memory state
htma reset          # Clear all memory
```

**Demo scenarios** (guided):
1. Introduction and basic conversation
2. Teaching facts and seeing retrieval
3. Time-based changes
4. Pattern emergence over multiple conversations
5. Consolidation and abstraction

**Output formatting**:
- Use Rich for colorful output
- Show memory operations in sidebar
- Highlight retrieved context
- Display consolidation progress

**Acceptance Criteria**:
- CLI works reliably
- Demo scenarios are compelling
- Error handling is graceful
- Help text is clear

---

## Summary

| Phase | Issues | Focus |
|-------|--------|-------|
| 1 | #1-5 | Project setup, types, storage |
| 2 | #6-9 | Memory stores and interface |
| 3 | #10-14 | Memory curator operations |
| 4 | #15-18 | Consolidation engine |
| 5 | #19-21 | Integration and demo |

**Total: 21 issues**

Estimated timeline: 4-6 weeks with focused effort.
