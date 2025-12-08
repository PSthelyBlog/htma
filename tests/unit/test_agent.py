"""Unit tests for HTMAAgent."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from htma.agent.agent import HTMAAgent
from htma.core.exceptions import AgentError
from htma.core.types import (
    AgentConfig,
    AgentResponse,
    Episode,
    Fact,
    Interaction,
    RetrievalResult,
    StorageResult,
)
from htma.llm.client import OllamaClient
from htma.memory.interface import MemoryInterface
from htma.memory.working import Message, WorkingMemory


class TestHTMAAgentInit:
    """Tests for HTMAAgent initialization."""

    def test_default_initialization(self):
        """Test agent initializes with default config."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        assert agent.llm == llm
        assert agent.memory == memory
        assert isinstance(agent.config, AgentConfig)
        assert agent.reasoner_model == "llama3:8b"  # default
        assert agent.conversations == {}
        memory.working.set_system_context.assert_called_once()

    def test_custom_config_initialization(self):
        """Test agent initializes with custom config."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        custom_config = AgentConfig(
            reasoner_model="custom-model:7b",
            system_context="Custom system prompt",
            auto_store_interactions=False,
        )

        agent = HTMAAgent(llm=llm, memory=memory, config=custom_config)

        assert agent.config == custom_config
        assert agent.reasoner_model == "custom-model:7b"
        assert agent.config.auto_store_interactions is False


class TestHTMAAgentProcessMessage:
    """Tests for process_message method."""

    @pytest.mark.asyncio
    async def test_process_message_basic_flow(self):
        """Test basic message processing flow."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        # Setup mocks
        retrieval_result = RetrievalResult(episodes=[], facts=[])
        memory.query = AsyncMock(return_value=retrieval_result)
        memory.inject_context = AsyncMock()
        memory.store_interaction = AsyncMock(
            return_value=StorageResult(salience_score=0.5)
        )
        memory.working.render_context = MagicMock(
            return_value="System: You are a helpful assistant."
        )
        memory.working.clear_retrieved = MagicMock()

        llm.chat = AsyncMock(return_value="Hello! How can I help you?")

        agent = HTMAAgent(llm=llm, memory=memory)

        # Process message
        response = await agent.process_message("Hi there!")

        # Verify response
        assert isinstance(response, AgentResponse)
        assert response.message == "Hello! How can I help you?"
        assert response.conversation_id is not None
        assert response.processing_time > 0

        # Verify calls
        memory.query.assert_called_once()
        memory.inject_context.assert_called_once()
        llm.chat.assert_called_once()
        memory.store_interaction.assert_called_once()
        memory.working.clear_retrieved.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_with_conversation_id(self):
        """Test processing message with explicit conversation ID."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        retrieval_result = RetrievalResult()
        memory.query = AsyncMock(return_value=retrieval_result)
        memory.inject_context = AsyncMock()
        memory.store_interaction = AsyncMock(return_value=StorageResult())
        memory.working.render_context = MagicMock(return_value="System prompt")
        memory.working.clear_retrieved = MagicMock()

        llm.chat = AsyncMock(return_value="Response")

        agent = HTMAAgent(llm=llm, memory=memory)

        # First message
        response1 = await agent.process_message("Message 1", conversation_id="conv_123")
        assert response1.conversation_id == "conv_123"
        assert len(agent.conversations["conv_123"]) == 2  # user + assistant

        # Second message in same conversation
        response2 = await agent.process_message("Message 2", conversation_id="conv_123")
        assert response2.conversation_id == "conv_123"
        assert len(agent.conversations["conv_123"]) == 4  # 2 previous + 2 new

    @pytest.mark.asyncio
    async def test_process_message_without_auto_store(self):
        """Test processing message with auto-store disabled."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        config = AgentConfig(auto_store_interactions=False)

        retrieval_result = RetrievalResult()
        memory.query = AsyncMock(return_value=retrieval_result)
        memory.inject_context = AsyncMock()
        memory.store_interaction = AsyncMock()
        memory.working.render_context = MagicMock(return_value="System")
        memory.working.clear_retrieved = MagicMock()

        llm.chat = AsyncMock(return_value="Response")

        agent = HTMAAgent(llm=llm, memory=memory, config=config)

        response = await agent.process_message("Test")

        # Verify store_interaction was NOT called
        memory.store_interaction.assert_not_called()
        assert response.storage_result is None

    @pytest.mark.asyncio
    async def test_process_message_with_retrieved_context(self):
        """Test message processing with retrieved context."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        # Create episodes and facts
        episode = Episode(
            id="epi_abc123def456",
            content="Previous conversation about Python",
            summary="User likes Python",
            level=0,
        )
        fact = Fact(
            id="fct_abc123def456",
            subject_id="ent_abc123def456",
            predicate="prefers",
            object_value="Python",
        )

        retrieval_result = RetrievalResult(
            episodes=[episode],
            facts=[fact],
            relevance_scores={"epi_abc123def456": 0.9, "fct_abc123def456": 0.8},
        )

        memory.query = AsyncMock(return_value=retrieval_result)
        memory.inject_context = AsyncMock()
        memory.store_interaction = AsyncMock(return_value=StorageResult())
        memory.working.render_context = MagicMock(return_value="System + context")
        memory.working.clear_retrieved = MagicMock()

        llm.chat = AsyncMock(return_value="I remember you like Python!")

        agent = HTMAAgent(llm=llm, memory=memory)

        response = await agent.process_message("What programming language do I like?")

        # Verify context was retrieved and injected
        assert response.retrieved_context["episodes_count"] == 1
        assert response.retrieved_context["facts_count"] == 1
        memory.inject_context.assert_called_once_with(retrieval_result)

    @pytest.mark.asyncio
    async def test_process_message_error_handling(self):
        """Test error handling in message processing."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        # Make query raise an exception
        memory.query = AsyncMock(side_effect=Exception("Memory error"))

        agent = HTMAAgent(llm=llm, memory=memory)

        with pytest.raises(AgentError) as exc_info:
            await agent.process_message("Test")

        assert "Message processing failed" in str(exc_info.value)


class TestHTMAAgentConversationManagement:
    """Tests for conversation management methods."""

    def test_start_conversation(self):
        """Test starting a new conversation."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        conv_id = agent.start_conversation()

        assert conv_id.startswith("conv_")
        assert conv_id in agent.conversations
        assert agent.conversations[conv_id] == []

    @pytest.mark.asyncio
    async def test_end_conversation(self):
        """Test ending a conversation."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        # Start conversation
        conv_id = agent.start_conversation()
        agent.conversations[conv_id].append(
            Message(role="user", content="Test message")
        )

        # End conversation
        await agent.end_conversation(conv_id)

        assert conv_id not in agent.conversations
        memory.working.clear_retrieved.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_unknown_conversation(self):
        """Test ending unknown conversation doesn't raise error."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        # Should not raise
        await agent.end_conversation("unknown_conv_id")

    def test_get_conversation_history(self):
        """Test retrieving conversation history."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        conv_id = agent.start_conversation()
        msg1 = Message(role="user", content="Message 1")
        msg2 = Message(role="assistant", content="Response 1")
        agent.conversations[conv_id].extend([msg1, msg2])

        history = agent.get_conversation_history(conv_id)

        assert len(history) == 2
        assert history[0] == msg1
        assert history[1] == msg2

    def test_get_unknown_conversation_history(self):
        """Test retrieving history of unknown conversation."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        history = agent.get_conversation_history("unknown_conv")

        assert history == []


class TestHTMAAgentQueryMemory:
    """Tests for explicit memory querying."""

    @pytest.mark.asyncio
    async def test_query_memory(self):
        """Test explicit memory query."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        episode = Episode(id="epi_abc123def456", content="Test episode", level=0)
        retrieval_result = RetrievalResult(episodes=[episode])
        memory.query = AsyncMock(return_value=retrieval_result)

        agent = HTMAAgent(llm=llm, memory=memory)

        result = await agent.query_memory("test query")

        assert len(result.episodes) == 1
        assert result.episodes[0].id == "epi_abc123def456"
        memory.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_memory_with_filters(self):
        """Test memory query with semantic/episodic filters."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        memory.query = AsyncMock(return_value=RetrievalResult())

        agent = HTMAAgent(llm=llm, memory=memory)

        # Query only semantic
        await agent.query_memory("test", include_semantic=True, include_episodic=False)

        call_args = memory.query.call_args
        assert call_args[1]["include_semantic"] is True
        assert call_args[1]["include_episodic"] is False


class TestHTMAAgentHelperMethods:
    """Tests for private helper methods."""

    def test_generate_conversation_id(self):
        """Test conversation ID generation."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        conv_id1 = agent._generate_conversation_id()
        conv_id2 = agent._generate_conversation_id()

        assert conv_id1.startswith("conv_")
        assert conv_id2.startswith("conv_")
        assert conv_id1 != conv_id2  # Should be unique

    def test_summarize_retrieval(self):
        """Test retrieval result summarization."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        episode1 = Episode(id="epi_abc123def456", content="Test 1", level=0)
        episode2 = Episode(id="epi_def456abc789", content="Test 2", level=0)
        fact = Fact(
            id="fct_abc123def456",
            subject_id="ent_abc123def456",
            predicate="test",
            object_value="value",
        )

        result = RetrievalResult(
            episodes=[episode1, episode2],
            facts=[fact],
            relevance_scores={
                "epi_abc123def456": 0.9,
                "epi_def456abc789": 0.8,
                "fct_abc123def456": 0.7,
            },
        )

        summary = agent._summarize_retrieval(result)

        assert summary["episodes_count"] == 2
        assert summary["facts_count"] == 1
        assert len(summary["episode_ids"]) == 2
        assert "epi_abc123def456" in summary["episode_ids"]

    def test_format_retrieval_context(self):
        """Test formatting retrieval context as string."""
        llm = MagicMock(spec=OllamaClient)
        memory = MagicMock(spec=MemoryInterface)
        memory.working = MagicMock(spec=WorkingMemory)

        agent = HTMAAgent(llm=llm, memory=memory)

        episode = Episode(
            id="epi_abc123def456", content="Test episode", summary="Summary", level=0
        )
        fact = Fact(
            id="fct_abc123def456",
            subject_id="ent_abc123def456",
            predicate="likes",
            object_value="Python",
        )

        result = RetrievalResult(episodes=[episode], facts=[fact])

        formatted = agent._format_retrieval_context(result)

        assert "Retrieved 1 episodes" in formatted
        assert "Retrieved 1 facts" in formatted
        assert "epi_abc123def456" in formatted
        assert "ent_abc123def456 likes Python" in formatted
