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


# Conflict Resolution Types
class FactConflict(BaseModel):
    """Represents a conflict between a new fact and existing facts.

    Attributes:
        new_fact: The new fact being added.
        conflicting_facts: List of existing facts that conflict with the new fact.
        conflict_type: Type of conflict (e.g., "contradiction", "update", "refinement").
        detected_at: When the conflict was detected.
    """

    new_fact: Fact = Field(description="The new fact being added")
    conflicting_facts: list[Fact] = Field(
        description="Existing facts that conflict with new fact"
    )
    conflict_type: str = Field(
        default="contradiction", description="Type of conflict detected"
    )
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class ConflictResolution(BaseModel):
    """Result of resolving a conflict between facts.

    Represents the strategy chosen to resolve a conflict and the actions to take.

    Resolution strategies:
    1. temporal_succession: Old fact was true, now new fact is true
    2. confidence_adjustment: Uncertain which is correct, lower confidence
    3. coexistence: Both can be true in different contexts
    4. rejection: New fact appears to be wrong

    Attributes:
        strategy: Which resolution strategy to apply.
        invalidations: List of (fact_id, invalidation_timestamp) to invalidate.
        confidence_updates: List of (fact_id, new_confidence) to update.
        new_fact: The new fact to add (may be modified from original).
        reasoning: Explanation of why this resolution was chosen.
        metadata: Additional metadata about the resolution.
    """

    strategy: str = Field(
        description="Resolution strategy: temporal_succession, confidence_adjustment, "
        "coexistence, or rejection"
    )
    invalidations: list[tuple[FactID, datetime]] = Field(
        default_factory=list,
        description="Facts to invalidate with their invalidation timestamps",
    )
    confidence_updates: list[tuple[FactID, float]] = Field(
        default_factory=list, description="Facts to update with new confidence scores"
    )
    new_fact: Fact | None = Field(
        default=None, description="New fact to add (None if rejected)"
    )
    reasoning: str = Field(description="Explanation of the resolution decision")
    metadata: dict[str, Any] = Field(default_factory=dict)


# Memory Evolution Types
class EpisodeUpdate(BaseModel):
    """Represents an update to an existing episode triggered by new information.

    When a new episode is added, it may provide context for or change the significance
    of existing episodes. This model captures those updates (A-MEM style evolution).

    Evolution types:
    - context_enrichment: New info explains or contextualizes old
    - significance_change: New events change importance of old
    - pattern_recognition: New episode confirms pattern from old
    - contradiction: New episode contradicts old (may trigger resolution)

    Attributes:
        episode_id: ID of the episode to update.
        evolution_type: Type of evolution (context_enrichment, significance_change, etc.).
        updates: Dictionary of fields to update with their new values.
        reasoning: Explanation of why this update is needed.
        triggered_by: ID of the new episode that triggered this update.
        metadata: Additional metadata about the evolution.
    """

    episode_id: EpisodeID = Field(description="ID of episode to update")
    evolution_type: str = Field(
        description="Evolution type: context_enrichment, significance_change, "
        "pattern_recognition, or contradiction"
    )
    updates: dict[str, Any] = Field(
        default_factory=dict,
        description="Fields to update (e.g., {'salience': 0.8, 'keywords': ['new', 'words']})",
    )
    reasoning: str = Field(description="Explanation for the update")
    triggered_by: EpisodeID = Field(description="ID of episode that triggered update")
    metadata: dict[str, Any] = Field(default_factory=dict)


# Pattern Detection Types
class Pattern(BaseModel):
    """Represents a recurring pattern detected across episodes.

    Patterns represent recurring themes, behaviors, or preferences observed
    across multiple episodes. They strengthen with evidence and evolve over time.

    Pattern types:
    - behavioral: User tends to do X
    - preference: User prefers X over Y
    - procedural: Steps for accomplishing X
    - error: Common mistake pattern

    Pattern lifecycle:
    - Emerging: 1-2 occurrences, low confidence
    - Established: 3+ occurrences, medium confidence
    - Consolidated: 10+ occurrences, high confidence, becomes principle

    Attributes:
        id: Unique pattern identifier.
        description: Human-readable description of the pattern.
        pattern_type: Type of pattern (behavioral, preference, procedural, error).
        confidence: Confidence score (0.0-1.0) based on evidence strength.
        occurrences: List of episode IDs where this pattern was observed.
        first_seen: When the pattern was first detected.
        last_seen: When the pattern was most recently observed.
        consolidation_strength: Resistance to weakening/pruning.
        metadata: Additional metadata about the pattern.
    """

    id: str = Field(description="Unique pattern identifier")
    description: str = Field(description="Human-readable pattern description")
    pattern_type: str = Field(
        description="Pattern type: behavioral, preference, procedural, or error"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence based on evidence"
    )
    occurrences: list[EpisodeID] = Field(
        default_factory=list, description="Episode IDs where pattern observed"
    )
    first_seen: datetime = Field(
        default_factory=datetime.utcnow, description="When pattern first detected"
    )
    last_seen: datetime = Field(
        default_factory=datetime.utcnow, description="Most recent observation"
    )
    consolidation_strength: float = Field(
        default=5.0, ge=0.0, description="Resistance to weakening"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatternDetectionResult(BaseModel):
    """Result of pattern detection across episodes.

    Contains newly discovered patterns, existing patterns with new evidence,
    and patterns that have weakened due to lack of recent observation.

    Attributes:
        new_patterns: Patterns discovered in this detection cycle.
        strengthened: Existing patterns with new supporting evidence.
        weakened: Patterns not observed recently, with reduced confidence.
        metadata: Additional metadata about the detection process.
    """

    new_patterns: list[Pattern] = Field(
        default_factory=list, description="Newly discovered patterns"
    )
    strengthened: list[tuple[str, float]] = Field(
        default_factory=list,
        description="(pattern_id, new_confidence) for strengthened patterns",
    )
    weakened: list[tuple[str, float]] = Field(
        default_factory=list,
        description="(pattern_id, new_confidence) for weakened patterns",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# Consolidation Types
class LinkMaintenanceReport(BaseModel):
    """Result of link maintenance operations during consolidation.

    Contains statistics about link strengthening, decay, and pruning operations.

    Attributes:
        links_strengthened: Number of links strengthened due to co-access.
        links_decayed: Number of links that decayed due to lack of use.
        links_pruned: Number of weak links removed from the graph.
        total_links_before: Total number of links before maintenance.
        total_links_after: Total number of links after maintenance.
        processing_time: Time taken for maintenance operations (in seconds).
        metadata: Additional metadata about the maintenance cycle.
    """

    links_strengthened: int = Field(
        default=0, ge=0, description="Links strengthened via co-access"
    )
    links_decayed: int = Field(default=0, ge=0, description="Links that decayed")
    links_pruned: int = Field(default=0, ge=0, description="Weak links removed")
    total_links_before: int = Field(
        default=0, ge=0, description="Total links before maintenance"
    )
    total_links_after: int = Field(
        default=0, ge=0, description="Total links after maintenance"
    )
    processing_time: float = Field(
        default=0.0, ge=0.0, description="Processing time in seconds"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PruneReport(BaseModel):
    """Result of pruning stale memories during consolidation.

    Contains statistics about episodes and links removed from memory.

    Attributes:
        episodes_pruned: Number of episodes removed.
        links_pruned: Number of links removed due to episode pruning.
        total_episodes_before: Total episodes before pruning.
        total_episodes_after: Total episodes after pruning.
        processing_time: Time taken for pruning operations (in seconds).
        metadata: Additional metadata about the pruning cycle.
    """

    episodes_pruned: int = Field(default=0, ge=0, description="Episodes removed")
    links_pruned: int = Field(
        default=0, ge=0, description="Links removed due to episode pruning"
    )
    total_episodes_before: int = Field(
        default=0, ge=0, description="Total episodes before pruning"
    )
    total_episodes_after: int = Field(
        default=0, ge=0, description="Total episodes after pruning"
    )
    processing_time: float = Field(
        default=0.0, ge=0.0, description="Processing time in seconds"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidationReport(BaseModel):
    """Result of a full consolidation cycle.

    Contains comprehensive statistics about all consolidation operations
    including abstraction generation, pattern detection, conflict resolution,
    link maintenance, and memory pruning.

    Attributes:
        abstractions_created: Number of higher-level abstractions generated.
        patterns_detected: Number of new patterns discovered.
        patterns_strengthened: Number of existing patterns with new evidence.
        conflicts_resolved: Number of fact conflicts resolved.
        links_strengthened: Number of links strengthened via co-access.
        links_pruned: Number of weak links removed.
        episodes_pruned: Number of stale episodes removed.
        duration: Time taken for the full consolidation cycle.
        metadata: Additional metadata about the consolidation process.
    """

    abstractions_created: int = Field(
        default=0, ge=0, description="Higher-level abstractions generated"
    )
    patterns_detected: int = Field(
        default=0, ge=0, description="New patterns discovered"
    )
    patterns_strengthened: int = Field(
        default=0, ge=0, description="Existing patterns with new evidence"
    )
    conflicts_resolved: int = Field(
        default=0, ge=0, description="Fact conflicts resolved"
    )
    links_strengthened: int = Field(
        default=0, ge=0, description="Links strengthened via co-access"
    )
    links_pruned: int = Field(default=0, ge=0, description="Weak links removed")
    episodes_pruned: int = Field(default=0, ge=0, description="Stale episodes removed")
    duration: float = Field(
        default=0.0, ge=0.0, description="Total consolidation time in seconds"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
