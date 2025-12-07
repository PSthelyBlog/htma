-- HTMA Initial Schema Migration
-- Creates all core tables for the tri-memory architecture

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
