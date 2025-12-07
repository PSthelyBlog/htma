"""Unit tests for working memory implementation."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from htma.llm.client import OllamaClient
from htma.memory.working import (
    MemoryItem,
    Message,
    WorkingMemory,
    WorkingMemoryConfig,
)


@pytest.fixture
def mock_tiktoken_encoder():
    """Create a mock tiktoken encoder."""
    encoder = MagicMock()
    # Mock encode method to return a list with length based on word count
    # Roughly 1 token per 4 characters as a simple approximation
    encoder.encode = lambda text: [0] * max(1, len(text) // 4)
    return encoder


@pytest.fixture
def mock_llm():
    """Create a mock OllamaClient."""
    llm = MagicMock(spec=OllamaClient)
    llm.generate = AsyncMock(return_value="This is a summary of the conversation.")
    return llm


@pytest.fixture
def config():
    """Create a test configuration."""
    return WorkingMemoryConfig(
        max_tokens=1000,
        pressure_threshold=0.8,
        summarization_model="mistral:7b",
    )


@pytest.fixture
def working_memory(config, mock_llm, mock_tiktoken_encoder):
    """Create a WorkingMemory instance for testing."""
    with patch("tiktoken.get_encoding", return_value=mock_tiktoken_encoder):
        return WorkingMemory(config=config, llm=mock_llm)


class TestWorkingMemoryConfig:
    """Test WorkingMemoryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WorkingMemoryConfig()
        assert config.max_tokens == 8000
        assert config.pressure_threshold == 0.8
        assert config.summarization_model == "mistral:7b"
        assert config.encoding_name == "cl100k_base"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WorkingMemoryConfig(
            max_tokens=4000,
            pressure_threshold=0.7,
            summarization_model="llama3:8b",
        )
        assert config.max_tokens == 4000
        assert config.pressure_threshold == 0.7
        assert config.summarization_model == "llama3:8b"


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self):
        """Test message creation with defaults."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_message_str(self):
        """Test message string representation."""
        msg = Message(role="user", content="Hello world")
        assert str(msg) == "user: Hello world"


class TestMemoryItem:
    """Test MemoryItem dataclass."""

    def test_memory_item_creation(self):
        """Test memory item creation with defaults."""
        item = MemoryItem(content="Test content", source="semantic")
        assert item.content == "Test content"
        assert item.source == "semantic"
        assert item.relevance == 1.0
        assert item.metadata == {}

    def test_memory_item_str(self):
        """Test memory item string representation."""
        item = MemoryItem(content="User likes Python", source="semantic")
        assert str(item) == "[semantic] User likes Python"


class TestWorkingMemory:
    """Test WorkingMemory class."""

    def test_initialization(self, working_memory, config, mock_llm):
        """Test working memory initialization."""
        assert working_memory.config == config
        assert working_memory.llm == mock_llm
        assert working_memory.system_context == ""
        assert working_memory.task_context == ""
        assert working_memory.dialog_history == []
        assert working_memory.retrieved_context == []

    def test_set_system_context(self, working_memory):
        """Test setting system context."""
        context = "You are a helpful assistant."
        working_memory.set_system_context(context)
        assert working_memory.system_context == context

    def test_set_task_context(self, working_memory):
        """Test setting task context."""
        context = "Current task: Implement feature X"
        working_memory.set_task_context(context)
        assert working_memory.task_context == context

    def test_add_message(self, working_memory):
        """Test adding messages to dialog history."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="assistant", content="Hi there!")

        working_memory.add_message(msg1)
        assert len(working_memory.dialog_history) == 1
        assert working_memory.dialog_history[0] == msg1

        working_memory.add_message(msg2)
        assert len(working_memory.dialog_history) == 2
        assert working_memory.dialog_history[1] == msg2

    def test_add_retrieved(self, working_memory):
        """Test adding retrieved context."""
        items = [
            MemoryItem(content="User prefers Python", source="semantic"),
            MemoryItem(content="Previous conversation about coding", source="episodic"),
        ]

        working_memory.add_retrieved(items)
        assert len(working_memory.retrieved_context) == 2
        assert working_memory.retrieved_context == items

    def test_clear_retrieved(self, working_memory):
        """Test clearing retrieved context."""
        items = [MemoryItem(content="Test", source="semantic")]
        working_memory.add_retrieved(items)
        assert len(working_memory.retrieved_context) == 1

        working_memory.clear_retrieved()
        assert len(working_memory.retrieved_context) == 0

    def test_token_counting(self, working_memory):
        """Test token counting."""
        # Empty context should have 0 tokens
        assert working_memory.current_tokens == 0

        # Add some content
        working_memory.set_system_context("You are a helpful assistant.")
        tokens_after_system = working_memory.current_tokens
        assert tokens_after_system > 0

        # Add more content
        working_memory.set_task_context("Implement feature X")
        tokens_after_task = working_memory.current_tokens
        assert tokens_after_task > tokens_after_system

    def test_utilization(self, working_memory):
        """Test utilization calculation."""
        # Empty memory should have 0 utilization
        assert working_memory.utilization == 0.0

        # Add content to reach ~50% utilization
        # Approximate: 1000 max tokens, so ~500 tokens of content
        content = "word " * 100  # Roughly 100-200 tokens
        working_memory.set_system_context(content)

        utilization = working_memory.utilization
        assert 0.0 < utilization < 1.0

    def test_under_pressure(self, working_memory):
        """Test pressure detection."""
        # Initially not under pressure
        assert not working_memory.under_pressure

        # Fill up to exceed threshold (80% of 1000 = 800 tokens)
        # Create large content to exceed threshold
        # With our mock (1 token per 4 chars), we need 4 * 800 = 3200 chars
        large_content = "x" * 1100  # 275 tokens per item
        working_memory.set_system_context(large_content)  # 275 tokens
        working_memory.set_task_context(large_content)  # 275 tokens
        working_memory.add_message(Message(role="user", content=large_content))  # ~280 tokens (includes "user: " prefix)

        # Should be under pressure now (total > 800)
        assert working_memory.under_pressure

    def test_automatic_eviction_on_add_message(self, working_memory):
        """Test that adding messages triggers automatic eviction under pressure."""
        # Fill memory to create pressure
        # With our mock (1 token per 4 chars), we need enough content to exceed threshold
        large_content = "x" * 400  # 100 tokens per message

        # Add many messages to exceed capacity
        for _ in range(10):
            working_memory.add_message(Message(role="user", content=large_content))

        # Should have evicted old messages
        # Exact count depends on token counting, but should be less than 10
        assert len(working_memory.dialog_history) < 10

    @pytest.mark.asyncio
    async def test_handle_pressure_clears_retrieved(self, working_memory):
        """Test that handle_pressure clears retrieved context first."""
        # Add retrieved context
        items = [MemoryItem(content="Test memory", source="semantic")]
        working_memory.add_retrieved(items)
        assert len(working_memory.retrieved_context) == 1

        # Handle pressure
        await working_memory.handle_pressure()

        # Retrieved context should be cleared
        assert len(working_memory.retrieved_context) == 0

    @pytest.mark.asyncio
    async def test_handle_pressure_summarizes_dialog(self, working_memory, mock_llm):
        """Test that handle_pressure summarizes dialog history."""
        # Add dialog history and create pressure
        large_content = "x" * 400  # 100 tokens per message
        for i in range(5):
            working_memory.add_message(
                Message(role="user" if i % 2 == 0 else "assistant", content=large_content)
            )

        # Force pressure
        working_memory.config.pressure_threshold = 0.01  # Very low threshold

        # Handle pressure
        items = await working_memory.handle_pressure()

        # Should have called summarization
        mock_llm.generate.assert_called_once()

        # Should return summary as item to persist
        assert len(items) > 0
        assert any(item.source == "dialog_summary" for item in items)

        # Should have reduced dialog history
        assert len(working_memory.dialog_history) <= 2

    @pytest.mark.asyncio
    async def test_handle_pressure_evicts_task_context(self, working_memory):
        """Test that handle_pressure evicts task context if needed."""
        # Set task context
        working_memory.set_task_context("Important task")

        # Force extreme pressure
        working_memory.config.pressure_threshold = 0.0

        # Handle pressure
        items = await working_memory.handle_pressure()

        # Task context should be cleared
        assert working_memory.task_context == ""

        # Should return task context as item to persist
        assert any(item.source == "task_context" for item in items)

    def test_render_context(self, working_memory):
        """Test rendering full context."""
        # Empty context
        rendered = working_memory.render_context()
        assert rendered == ""

        # Add various components
        working_memory.set_system_context("System instructions")
        working_memory.set_task_context("Current task")
        working_memory.add_message(Message(role="user", content="Hello"))
        working_memory.add_message(Message(role="assistant", content="Hi"))
        working_memory.add_retrieved([MemoryItem(content="Memory", source="semantic")])

        # Render context
        rendered = working_memory.render_context()

        # Should contain all components
        assert "System instructions" in rendered
        assert "Current task" in rendered
        assert "Hello" in rendered
        assert "Hi" in rendered
        assert "Memory" in rendered

    def test_render_context_sections(self, working_memory):
        """Test that render_context properly formats sections."""
        working_memory.set_system_context("System")
        working_memory.set_task_context("Task")

        rendered = working_memory.render_context()

        # Should have section headers
        assert "# System Context" in rendered
        assert "# Current Task" in rendered

    def test_get_offload_candidates(self, working_memory):
        """Test getting offload candidates."""
        # Initially no candidates
        candidates = working_memory.get_offload_candidates()
        assert len(candidates) == 0

        # Add various items
        working_memory.add_retrieved([MemoryItem(content="Memory", source="semantic")])
        working_memory.add_message(Message(role="user", content="Msg 1"))
        working_memory.add_message(Message(role="user", content="Msg 2"))
        working_memory.add_message(Message(role="user", content="Msg 3"))
        working_memory.set_task_context("Task")

        candidates = working_memory.get_offload_candidates()

        # Should have candidates:
        # - 1 retrieved item
        # - 1 old dialog message (keeps last 2)
        # - 1 task context
        assert len(candidates) >= 3

        # Retrieved context should be first (lowest priority)
        assert candidates[0].source == "semantic"

        # System context should never be in candidates
        working_memory.set_system_context("System")
        candidates = working_memory.get_offload_candidates()
        assert not any(item.content == "System" for item in candidates)

    def test_get_offload_candidates_priority(self, working_memory):
        """Test that offload candidates are in priority order."""
        # Add items in reverse priority order
        working_memory.set_task_context("Task context")
        working_memory.add_message(Message(role="user", content="Old message"))
        working_memory.add_message(Message(role="user", content="Recent message 1"))
        working_memory.add_message(Message(role="user", content="Recent message 2"))
        working_memory.add_retrieved([MemoryItem(content="Retrieved", source="semantic")])

        candidates = working_memory.get_offload_candidates()

        # First item should be retrieved context (lowest priority)
        assert candidates[0].source == "semantic"

        # Last item should be task context (higher priority than retrieved/old dialog)
        # But system context should never be included
        sources = [c.source for c in candidates]
        assert "task_context" in sources
        assert not any("system" in s for s in sources)

    @pytest.mark.asyncio
    async def test_summarize_dialog_calls_llm(self, working_memory, mock_llm):
        """Test that dialog summarization calls LLM correctly."""
        # Add dialog
        working_memory.add_message(Message(role="user", content="What's Python?"))
        working_memory.add_message(
            Message(role="assistant", content="Python is a programming language.")
        )

        # Summarize
        summary = await working_memory._summarize_dialog()

        # Should have called LLM
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args

        # Check parameters
        assert call_args.kwargs["model"] == "mistral:7b"
        assert "Python" in call_args.kwargs["prompt"]
        assert call_args.kwargs["temperature"] == 0.3

        # Should return summary
        assert summary == "This is a summary of the conversation."

    @pytest.mark.asyncio
    async def test_summarize_dialog_empty(self, working_memory):
        """Test summarizing empty dialog."""
        summary = await working_memory._summarize_dialog()
        assert summary == ""

    @pytest.mark.asyncio
    async def test_summarize_dialog_error_handling(self, working_memory, mock_llm):
        """Test that summarization handles LLM errors gracefully."""
        # Make LLM raise an error
        mock_llm.generate.side_effect = Exception("LLM error")

        # Add dialog
        working_memory.add_message(Message(role="user", content="Hello"))

        # Should not raise, but return fallback summary
        summary = await working_memory._summarize_dialog()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_render_context_with_sorted_retrieved(self, working_memory):
        """Test that rendered context sorts retrieved items by relevance."""
        # Add items with different relevance scores
        items = [
            MemoryItem(content="Low relevance", source="semantic", relevance=0.3),
            MemoryItem(content="High relevance", source="semantic", relevance=0.9),
            MemoryItem(content="Medium relevance", source="semantic", relevance=0.6),
        ]
        working_memory.add_retrieved(items)

        rendered = working_memory.render_context()

        # High relevance should appear first
        high_pos = rendered.index("High relevance")
        medium_pos = rendered.index("Medium relevance")
        low_pos = rendered.index("Low relevance")

        assert high_pos < medium_pos < low_pos
