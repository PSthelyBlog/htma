"""Pytest fixtures for integration tests.

This module provides shared fixtures for integration testing of the HTMA system.
These fixtures set up real instances of components (not mocks) to test
the complete system flow.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from htma.agent.agent import HTMAAgent
from htma.consolidation.abstraction import AbstractionGenerator
from htma.consolidation.engine import ConsolidationConfig, ConsolidationEngine
from htma.consolidation.patterns import PatternDetector
from htma.core.types import AgentConfig, WorkingMemoryConfig
from htma.curator.curator import MemoryCurator
from htma.curator.extractors import EntityExtractor, FactExtractor
from htma.curator.linkers import LinkGenerator
from htma.llm.client import OllamaClient
from htma.memory.episodic import EpisodicMemory
from htma.memory.interface import MemoryInterface
from htma.memory.semantic import SemanticMemory
from htma.memory.working import WorkingMemory
from htma.storage.chroma import ChromaStorage
from htma.storage.sqlite import SQLiteStorage


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test data.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
async def sqlite_storage(temp_dir):
    """Create and initialize SQLite storage for testing.

    Args:
        temp_dir: Temporary directory fixture.

    Yields:
        Initialized SQLiteStorage instance.
    """
    db_path = temp_dir / "test_htma.db"
    storage = SQLiteStorage(str(db_path))
    await storage.initialize()
    yield storage


@pytest.fixture(scope="function")
async def chroma_storage(temp_dir):
    """Create and initialize ChromaDB storage for testing.

    Args:
        temp_dir: Temporary directory fixture.

    Yields:
        Initialized ChromaStorage instance.
    """
    persist_path = str(temp_dir / "chroma")
    storage = ChromaStorage(
        persist_path=persist_path,
        embedding_model="nomic-embed-text"
    )
    await storage.initialize()
    yield storage


@pytest.fixture(scope="function")
def ollama_client():
    """Create Ollama client for testing.

    Note: Requires running Ollama instance.

    Yields:
        OllamaClient instance.
    """
    client = OllamaClient(base_url="http://localhost:11434")
    yield client


@pytest.fixture(scope="function")
def working_memory(ollama_client):
    """Create working memory for testing.

    Args:
        ollama_client: Ollama client fixture.

    Yields:
        WorkingMemory instance.
    """
    config = WorkingMemoryConfig(
        max_tokens=4000,  # Smaller for testing
        pressure_threshold=0.8,
        summarization_model="mistral:7b"
    )
    memory = WorkingMemory(config=config, llm=ollama_client)
    yield memory


@pytest.fixture(scope="function")
async def semantic_memory(sqlite_storage, chroma_storage):
    """Create semantic memory for testing.

    Args:
        sqlite_storage: SQLite storage fixture.
        chroma_storage: ChromaDB storage fixture.

    Yields:
        SemanticMemory instance.
    """
    memory = SemanticMemory(sqlite=sqlite_storage, chroma=chroma_storage)
    yield memory


@pytest.fixture(scope="function")
async def episodic_memory(sqlite_storage, chroma_storage):
    """Create episodic memory for testing.

    Args:
        sqlite_storage: SQLite storage fixture.
        chroma_storage: ChromaDB storage fixture.

    Yields:
        EpisodicMemory instance.
    """
    memory = EpisodicMemory(sqlite=sqlite_storage, chroma=chroma_storage)
    yield memory


@pytest.fixture(scope="function")
def curator(ollama_client, episodic_memory):
    """Create memory curator for testing.

    Args:
        ollama_client: Ollama client fixture.
        episodic_memory: Episodic memory fixture.

    Yields:
        MemoryCurator instance.
    """
    entity_extractor = EntityExtractor(
        llm=ollama_client,
        model="mistral:7b"
    )
    fact_extractor = FactExtractor(
        llm=ollama_client,
        model="mistral:7b"
    )
    link_generator = LinkGenerator(
        llm=ollama_client,
        model="mistral:7b",
        episodic=episodic_memory
    )

    curator = MemoryCurator(
        llm=ollama_client,
        model="mistral:7b",
        entity_extractor=entity_extractor,
        fact_extractor=fact_extractor,
        link_generator=link_generator
    )
    yield curator


@pytest.fixture(scope="function")
async def memory_interface(
    working_memory,
    semantic_memory,
    episodic_memory,
    curator
):
    """Create memory interface for testing.

    Args:
        working_memory: Working memory fixture.
        semantic_memory: Semantic memory fixture.
        episodic_memory: Episodic memory fixture.
        curator: Memory curator fixture.

    Yields:
        MemoryInterface instance.
    """
    interface = MemoryInterface(
        working=working_memory,
        semantic=semantic_memory,
        episodic=episodic_memory,
        curator=curator
    )
    yield interface


@pytest.fixture(scope="function")
def abstraction_generator(ollama_client):
    """Create abstraction generator for testing.

    Args:
        ollama_client: Ollama client fixture.

    Yields:
        AbstractionGenerator instance.
    """
    generator = AbstractionGenerator(
        llm=ollama_client,
        model="mistral:7b"
    )
    yield generator


@pytest.fixture(scope="function")
def pattern_detector(ollama_client):
    """Create pattern detector for testing.

    Args:
        ollama_client: Ollama client fixture.

    Yields:
        PatternDetector instance.
    """
    detector = PatternDetector(
        llm=ollama_client,
        model="mistral:7b"
    )
    yield detector


@pytest.fixture(scope="function")
def consolidation_engine(
    curator,
    semantic_memory,
    episodic_memory,
    abstraction_generator,
    pattern_detector
):
    """Create consolidation engine for testing.

    Args:
        curator: Memory curator fixture.
        semantic_memory: Semantic memory fixture.
        episodic_memory: Episodic memory fixture.
        abstraction_generator: Abstraction generator fixture.
        pattern_detector: Pattern detector fixture.

    Yields:
        ConsolidationEngine instance.
    """
    config = ConsolidationConfig(
        min_episodes_before_cycle=5,  # Lower threshold for testing
        abstraction_cluster_size=3,  # Smaller clusters for testing
        pattern_min_occurrences=2,  # Lower threshold for testing
    )

    engine = ConsolidationEngine(
        curator=curator,
        semantic=semantic_memory,
        episodic=episodic_memory,
        abstraction_generator=abstraction_generator,
        pattern_detector=pattern_detector,
        config=config
    )
    yield engine


@pytest.fixture(scope="function")
async def htma_agent(ollama_client, memory_interface):
    """Create HTMA agent for testing.

    Args:
        ollama_client: Ollama client fixture.
        memory_interface: Memory interface fixture.

    Yields:
        HTMAAgent instance.
    """
    config = AgentConfig(
        reasoner_model="llama3:8b",
        system_context="You are a helpful AI assistant with long-term memory.",
        auto_store_interactions=True,
        max_retrieved_episodes=5,
        max_retrieved_facts=5
    )

    agent = HTMAAgent(
        llm=ollama_client,
        memory=memory_interface,
        config=config
    )
    yield agent


@pytest.fixture(scope="function")
def check_ollama():
    """Check if Ollama is available for testing.

    Raises:
        pytest.skip: If Ollama is not available.
    """
    import httpx

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code != 200:
            pytest.skip("Ollama is not responding")
    except Exception:
        pytest.skip("Ollama is not available")
