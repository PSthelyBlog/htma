"""Core data types and Pydantic models for HTMA.

This module defines the foundational data structures used throughout the system,
including entities, facts, episodes, and temporal models.
"""

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

# Type aliases for identifiers
# Using Annotated with StringConstraints for validation
EntityID = Annotated[str, StringConstraints(pattern=r"^ent_[a-f0-9\-]+$")]
FactID = Annotated[str, StringConstraints(pattern=r"^fct_[a-f0-9\-]+$")]
EpisodeID = Annotated[str, StringConstraints(pattern=r"^epi_[a-f0-9\-]+$")]


# Temporal models
class TemporalRange(BaseModel):
    """Represents a time range with optional start and end.

    Attributes:
        valid_from: Start of the time range (inclusive). None means unbounded start.
        valid_to: End of the time range (exclusive). None means unbounded end.
    """

    valid_from: datetime | None = None
    valid_to: datetime | None = None


class BiTemporalRecord(BaseModel):
    """Bi-temporal model tracking both event time and transaction time.

    Event time (T): When the fact was true in the world.
    Transaction time (T'): When the fact was recorded in memory.

    This enables queries like "What did I know about X as of date Y?"
    """

    event_time: TemporalRange = Field(
        default_factory=TemporalRange, description="When the fact was true in the world"
    )
    transaction_time: TemporalRange = Field(
        default_factory=TemporalRange, description="When the fact was recorded in memory"
    )


# Entities
class Entity(BaseModel):
    """Represents a named entity in the knowledge graph.

    Entities can be people, places, concepts, objects, or events.
    """

    id: EntityID
    name: str
    entity_type: str  # person, place, concept, object, event
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime | None = None
    access_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# Facts
class Fact(BaseModel):
    """Represents a relationship or property in the knowledge graph.

    Facts connect entities via predicates and support bi-temporal validity.
    A fact can relate two entities (subject-predicate-object) or
    assign a value to an entity (subject-predicate-value).
    """

    id: FactID
    subject_id: EntityID
    predicate: str
    object_id: EntityID | None = None
    object_value: str | None = None
    temporal: BiTemporalRecord = Field(default_factory=BiTemporalRecord)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_episode_id: EpisodeID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Episodes
class Episode(BaseModel):
    """Represents a memory episode in the hierarchical episodic memory.

    Episodes form a hierarchy:
    - Level 0: Raw episodes (full interactions)
    - Level 1: Summaries (clustered episodes)
    - Level 2+: Progressive abstractions
    """

    id: EpisodeID
    level: int = Field(default=0, ge=0, description="Abstraction level in hierarchy")
    parent_id: EpisodeID | None = None
    content: str
    summary: str | None = None
    context_description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    salience: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    consolidation_strength: float = Field(
        default=5.0, ge=0.0, description="Resistance to pruning"
    )
    access_count: int = 0
    last_accessed: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Episode Links
class EpisodeLink(BaseModel):
    """Represents a bidirectional link between two episodes.

    Links can be semantic, temporal, causal, or analogical in nature.
    Weights are adjusted based on access patterns.
    """

    id: str
    source_id: EpisodeID
    target_id: EpisodeID
    link_type: str  # semantic, temporal, causal, analogical
    weight: float = Field(default=1.0, ge=0.0, description="Link strength")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Memory Notes (A-MEM style)
class MemoryNote(BaseModel):
    """A curated note for memory storage (A-MEM style).

    Memory notes are the result of salience evaluation and represent
    what the curator deems worth remembering.
    """

    content: str
    context: str
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)


# Query/Retrieval
class TemporalFilter(BaseModel):
    """Filters for temporal queries.

    Attributes:
        as_of: Transaction time filter - "What did we know as of this date?"
        valid_at: Event time filter - "What was true at this date?"
    """

    as_of: datetime | None = None  # Transaction time filter
    valid_at: datetime | None = None  # Event time filter


class RetrievalResult(BaseModel):
    """Result of a memory retrieval query.

    Contains facts and episodes matching the query, along with relevance scores.
    """

    facts: list[Fact] = Field(default_factory=list)
    episodes: list[Episode] = Field(default_factory=list)
    relevance_scores: dict[str, float] = Field(
        default_factory=dict, description="ID -> relevance score mapping"
    )


# Interaction and Storage
class Interaction(BaseModel):
    """Represents a user interaction/conversation turn to be stored.

    Attributes:
        user_message: The user's message.
        assistant_message: The assistant's response.
        occurred_at: When the interaction occurred.
        context: Additional context about the interaction.
        metadata: Additional metadata.
    """

    user_message: str
    assistant_message: str
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SalienceResult(BaseModel):
    """Result of salience evaluation for an interaction.

    Attributes:
        score: Salience score from 0.0 (not worth remembering) to 1.0 (very important).
        reasoning: Explanation of why this score was assigned.
        memory_type: Type of memory to store: "semantic", "episodic", or "both".
        key_elements: List of important elements extracted from the content.
    """

    score: float = Field(ge=0.0, le=1.0, description="Importance score 0.0-1.0")
    reasoning: str = Field(description="Explanation for the score")
    memory_type: str = Field(description="semantic, episodic, or both")
    key_elements: list[str] = Field(
        default_factory=list, description="Important items from content"
    )


class StorageResult(BaseModel):
    """Result of storing an interaction in memory.

    Attributes:
        episode_id: ID of the created episode (if any).
        entities_created: List of entity IDs created.
        facts_created: List of fact IDs created.
        links_created: List of link IDs created.
        salience_score: Salience score assigned to the interaction.
        processing_time: Time taken to process and store (in seconds).
        metadata: Additional metadata about the storage operation.
    """

    episode_id: EpisodeID | None = None
    entities_created: list[EntityID] = Field(default_factory=list)
    facts_created: list[FactID] = Field(default_factory=list)
    links_created: list[str] = Field(default_factory=list)
    salience_score: float = Field(default=0.5, ge=0.0, le=1.0)
    processing_time: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# Extraction Results (for curator operations)
class ExtractedEntity(BaseModel):
    """Result of entity extraction from text.

    Represents an entity identified in text before it's persisted to memory.

    Attributes:
        name: The entity's name or identifier.
        entity_type: Type of entity (person, place, organization, concept, object, event, time).
        mentions: List of text snippets where the entity was mentioned.
        confidence: Extraction confidence score (0.0-1.0).
        metadata: Additional metadata about the extraction.
    """

    name: str = Field(description="Entity name or identifier")
    entity_type: str = Field(
        description="Entity type: person, place, organization, concept, object, event, time"
    )
    mentions: list[str] = Field(
        default_factory=list, description="Text snippets where entity appears"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedFact(BaseModel):
    """Result of fact extraction from text.

    Represents a relationship or property identified in text before it's persisted.

    Attributes:
        subject: Subject entity name.
        predicate: Relationship type or property name.
        object_entity: Object entity name (for entity-to-entity relationships).
        object_value: Literal value (for entity-to-value properties).
        temporal_marker: Any temporal information extracted (e.g., "in 2020", "yesterday").
        confidence: Extraction confidence score (0.0-1.0).
        source_text: The text snippet this fact was extracted from.
        metadata: Additional metadata about the extraction.
    """

    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Relationship type or property name")
    object_entity: str | None = Field(
        default=None, description="Object entity name (for relationships)"
    )
    object_value: str | None = Field(
        default=None, description="Literal value (for properties)"
    )
    temporal_marker: str | None = Field(
        default=None, description="Temporal information if present"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    source_text: str = Field(default="", description="Source text snippet")
    metadata: dict[str, Any] = Field(default_factory=dict)


class LinkEvaluation(BaseModel):
    """Result of evaluating a potential link between two episodes.

    Represents the LLM's assessment of whether two episodes should be linked
    and the nature of their connection.

    Attributes:
        should_link: Whether the two episodes should be linked.
        link_type: Type of link (semantic, temporal, causal, analogical).
        weight: Strength of the connection (0.0-1.0).
        reasoning: Explanation of why the episodes should or shouldn't be linked.
        metadata: Additional metadata about the evaluation.
    """

    should_link: bool = Field(description="Whether episodes should be linked")
    link_type: str = Field(
        description="Link type: semantic, temporal, causal, or analogical"
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Connection strength"
    )
    reasoning: str = Field(description="Explanation of the link assessment")
    metadata: dict[str, Any] = Field(default_factory=dict)
