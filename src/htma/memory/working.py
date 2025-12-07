"""Working memory implementation with pressure management.

This module implements the working memory component that manages in-context information
with automatic token counting and pressure-based eviction.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import tiktoken

from htma.llm.client import OllamaClient


@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory.

    Attributes:
        max_tokens: Maximum tokens allowed in working memory
        pressure_threshold: Fraction (0.0-1.0) at which pressure handling triggers
        summarization_model: Model to use for summarizing dialog history
        encoding_name: Tiktoken encoding to use for token counting
    """

    max_tokens: int = 8000
    pressure_threshold: float = 0.8
    summarization_model: str = "mistral:7b"
    encoding_name: str = "cl100k_base"  # GPT-4 encoding, good default


@dataclass
class Message:
    """A single message in the dialog history.

    Attributes:
        role: Message role (system, user, assistant)
        content: Message content
        timestamp: When the message was created
        metadata: Additional metadata
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation for token counting."""
        return f"{self.role}: {self.content}"


@dataclass
class MemoryItem:
    """An item in working memory (retrieved context).

    Attributes:
        content: The memory content
        source: Where this memory came from (e.g., 'semantic', 'episodic')
        relevance: Relevance score (0.0-1.0)
        metadata: Additional metadata
    """

    content: str
    source: str
    relevance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation for token counting."""
        return f"[{self.source}] {self.content}"


class WorkingMemory:
    """Manages in-context information with automatic pressure handling.

    Working memory consists of:
    - System context: Static system instructions (never evicted)
    - Task context: Current task information
    - Dialog history: Conversation turns
    - Retrieved context: Memories pulled from long-term storage

    When memory pressure is detected (usage above threshold), the system
    will evict items in priority order and summarize dialog history.

    Attributes:
        config: Working memory configuration
        llm: LLM client for summarization
        system_context: Static system instructions
        task_context: Current task information
        dialog_history: List of conversation messages
        retrieved_context: List of retrieved memory items
    """

    def __init__(self, config: WorkingMemoryConfig, llm: OllamaClient):
        """Initialize working memory.

        Args:
            config: Working memory configuration
            llm: LLM client for summarization
        """
        self.config = config
        self.llm = llm
        self.system_context: str = ""
        self.task_context: str = ""
        self.dialog_history: list[Message] = []
        self.retrieved_context: list[MemoryItem] = []

        # Initialize tiktoken encoder
        try:
            self._encoder = tiktoken.get_encoding(config.encoding_name)
        except Exception:
            # Fallback to a basic encoding if the specified one isn't available
            self._encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self._encoder.encode(text))

    @property
    def current_tokens(self) -> int:
        """Calculate current token usage across all memory components.

        Returns:
            Total number of tokens currently in working memory
        """
        total = 0

        # System context
        total += self._count_tokens(self.system_context)

        # Task context
        total += self._count_tokens(self.task_context)

        # Dialog history
        for message in self.dialog_history:
            total += self._count_tokens(str(message))

        # Retrieved context
        for item in self.retrieved_context:
            total += self._count_tokens(str(item))

        return total

    @property
    def utilization(self) -> float:
        """Calculate current memory utilization as fraction of max.

        Returns:
            Utilization fraction (0.0-1.0+)
        """
        return self.current_tokens / self.config.max_tokens

    @property
    def under_pressure(self) -> bool:
        """Check if memory is under pressure.

        Returns:
            True if utilization is above pressure threshold
        """
        return self.utilization >= self.config.pressure_threshold

    def set_system_context(self, context: str) -> None:
        """Set static system instructions.

        System context is never evicted and should be kept minimal.

        Args:
            context: System instructions/context
        """
        self.system_context = context

    def set_task_context(self, context: str) -> None:
        """Set current task information.

        Task context may be evicted under memory pressure.

        Args:
            context: Current task description/context
        """
        self.task_context = context

    def add_message(self, message: Message) -> None:
        """Add a dialog turn to history.

        If adding the message would exceed capacity, oldest messages
        may be evicted automatically.

        Args:
            message: Message to add
        """
        self.dialog_history.append(message)

        # Check if we need to handle pressure
        if self.under_pressure:
            # Evict oldest messages until we're below threshold
            target_tokens = int(self.config.max_tokens * self.config.pressure_threshold * 0.9)

            while self.current_tokens > target_tokens and len(self.dialog_history) > 1:
                # Keep at least the most recent message
                self.dialog_history.pop(0)

    def add_retrieved(self, items: list[MemoryItem]) -> None:
        """Add retrieved memories to context.

        Args:
            items: Memory items to add to context
        """
        self.retrieved_context.extend(items)

    def clear_retrieved(self) -> None:
        """Clear all retrieved context.

        This is useful when starting a new query or when context becomes stale.
        """
        self.retrieved_context.clear()

    async def handle_pressure(self) -> list[MemoryItem]:
        """Handle memory pressure by evicting and summarizing.

        This method:
        1. Identifies items that should be persisted to long-term memory
        2. Summarizes dialog history to reduce token count
        3. Evicts retrieved context
        4. Returns items that should be stored

        Eviction priority (lowest to highest):
        1. Retrieved context (can be re-retrieved)
        2. Old dialog turns
        3. Task context
        4. System context (never evicted)

        Returns:
            List of memory items that should be persisted
        """
        items_to_persist: list[MemoryItem] = []

        # Step 1: Clear retrieved context (lowest priority)
        if self.retrieved_context:
            # These can be re-retrieved if needed, so we don't persist them
            self.clear_retrieved()

        # Step 2: Summarize dialog history if still under pressure
        if self.under_pressure and len(self.dialog_history) > 2:
            summary = await self._summarize_dialog()

            # Extract important information for persistence
            if summary:
                items_to_persist.append(
                    MemoryItem(
                        content=summary,
                        source="dialog_summary",
                        relevance=0.8,
                        metadata={
                            "original_messages": len(self.dialog_history),
                            "summarized_at": datetime.utcnow().isoformat(),
                        },
                    )
                )

            # Keep only the most recent messages
            messages_to_keep = min(2, len(self.dialog_history))
            self.dialog_history = self.dialog_history[-messages_to_keep:]

        # Step 3: Clear task context if still under pressure
        if self.under_pressure and self.task_context:
            # Save task context for potential persistence
            items_to_persist.append(
                MemoryItem(
                    content=self.task_context,
                    source="task_context",
                    relevance=0.6,
                    metadata={"evicted_at": datetime.utcnow().isoformat()},
                )
            )
            self.task_context = ""

        return items_to_persist

    async def _summarize_dialog(self) -> str:
        """Summarize dialog history using LLM.

        Returns:
            Summary of the dialog history
        """
        if not self.dialog_history:
            return ""

        # Construct dialog text
        dialog_text = "\n".join(str(msg) for msg in self.dialog_history)

        # Create summarization prompt
        prompt = f"""Summarize the following conversation, preserving key information, facts, and context:

{dialog_text}

Provide a concise summary that captures:
1. Main topics discussed
2. Important facts or decisions
3. Current state of the conversation

Summary:"""

        try:
            summary = await self.llm.generate(
                model=self.config.summarization_model,
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more focused summaries
                max_tokens=500,
            )
            return summary.strip()
        except Exception:
            # If summarization fails, return a simple concatenation
            return f"Conversation covered: {', '.join(msg.content[:50] for msg in self.dialog_history)}"

    def render_context(self) -> str:
        """Render full working memory context for LLM consumption.

        This assembles all memory components into a single string
        suitable for inclusion in an LLM prompt.

        Returns:
            Formatted context string
        """
        sections = []

        # System context
        if self.system_context:
            sections.append(f"# System Context\n{self.system_context}")

        # Task context
        if self.task_context:
            sections.append(f"# Current Task\n{self.task_context}")

        # Retrieved memories
        if self.retrieved_context:
            retrieved_text = "\n".join(
                f"- {item.content} (from {item.source})"
                for item in sorted(self.retrieved_context, key=lambda x: x.relevance, reverse=True)
            )
            sections.append(f"# Relevant Memories\n{retrieved_text}")

        # Dialog history
        if self.dialog_history:
            dialog_text = "\n".join(
                f"{msg.role.capitalize()}: {msg.content}" for msg in self.dialog_history
            )
            sections.append(f"# Conversation History\n{dialog_text}")

        return "\n\n".join(sections)

    def get_offload_candidates(self) -> list[MemoryItem]:
        """Get items that can be safely evicted.

        Returns items in order of eviction priority (lowest priority first).

        Returns:
            List of memory items that can be evicted
        """
        candidates: list[MemoryItem] = []

        # Retrieved context (lowest priority - can always be re-retrieved)
        candidates.extend(self.retrieved_context)

        # Old dialog messages (convert to MemoryItem for uniformity)
        if len(self.dialog_history) > 2:
            for msg in self.dialog_history[:-2]:  # Keep last 2 messages
                candidates.append(
                    MemoryItem(
                        content=msg.content,
                        source=f"dialog_{msg.role}",
                        relevance=0.5,
                        metadata={"timestamp": msg.timestamp.isoformat()},
                    )
                )

        # Task context
        if self.task_context:
            candidates.append(
                MemoryItem(
                    content=self.task_context,
                    source="task_context",
                    relevance=0.6,
                )
            )

        # System context is NEVER evicted

        return candidates
