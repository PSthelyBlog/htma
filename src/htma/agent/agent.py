"""HTMA Agent implementation.

This module implements the main agent that integrates memory systems with
the reasoning LLM (LLM₁) for enhanced, memory-augmented conversations.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from htma.core.exceptions import AgentError
from htma.core.types import (
    AgentConfig,
    AgentResponse,
    Interaction,
    RetrievalResult,
    StorageResult,
)
from htma.core.utils import utc_now
from htma.llm.client import OllamaClient
from htma.memory.interface import MemoryInterface
from htma.memory.working import Message

logger = logging.getLogger(__name__)


class HTMAAgent:
    """Main agent that integrates memory with reasoning LLM.

    The HTMAAgent coordinates between the reasoning model (LLM₁) and the memory
    system to provide memory-augmented conversations. It handles:
    - Memory retrieval before generating responses
    - Context injection into working memory
    - Interaction storage for long-term memory
    - Multi-turn conversation management

    The agent implements the memory-augmented prompting pattern:
    1. Query relevant memories based on user input
    2. Inject retrieved context into working memory
    3. Generate response with full context
    4. Store interaction for future retrieval

    Attributes:
        llm: Ollama client for LLM operations.
        reasoner_model: Model name for reasoning (LLM₁).
        memory: Memory interface for all memory operations.
        config: Agent configuration.
        conversations: Active conversation tracking.
    """

    def __init__(
        self,
        llm: OllamaClient,
        memory: MemoryInterface,
        config: AgentConfig | None = None,
    ):
        """Initialize HTMA agent.

        Args:
            llm: Ollama client for LLM operations.
            memory: Memory interface instance.
            config: Optional agent configuration. Defaults to AgentConfig().
        """
        self.llm = llm
        self.memory = memory
        self.config = config or AgentConfig()
        self.reasoner_model = self.config.reasoner_model

        # Track active conversations: conversation_id -> list of messages
        self.conversations: dict[str, list[Message]] = {}

        # Set system context in working memory
        self.memory.working.set_system_context(self.config.system_context)

        logger.info(
            f"Initialized HTMAAgent with model={self.reasoner_model}, "
            f"auto_store={self.config.auto_store_interactions}"
        )

    async def process_message(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        """Process user message with memory augmentation.

        This is the main entry point for message processing. Flow:
        1. Query relevant memories based on message
        2. Inject context into working memory
        3. Generate response using reasoning model
        4. Store interaction (if auto-store enabled)
        5. Return response with metadata

        Args:
            message: User's message text.
            conversation_id: Optional conversation ID for multi-turn tracking.
                If None, treats as single-turn interaction.

        Returns:
            AgentResponse with the assistant's message and metadata.

        Raises:
            AgentError: If message processing fails.
        """
        start_time = time.time()

        try:
            # Create or get conversation history
            if conversation_id is None:
                conversation_id = self._generate_conversation_id()
                self.conversations[conversation_id] = []

            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []

            # Add user message to conversation
            user_msg = Message(role="user", content=message, timestamp=utc_now())
            self.conversations[conversation_id].append(user_msg)

            logger.debug(f"Processing message in conversation {conversation_id}")

            # Step 1: Query relevant memories
            retrieval_result = await self._retrieve_context(message)

            # Step 2: Inject context into working memory
            await self.memory.inject_context(retrieval_result)

            # Step 3: Build conversation history for prompt
            conversation_history = self._build_conversation_history(conversation_id)

            # Step 4: Generate response
            assistant_message = await self._generate_response(
                conversation_history, message
            )

            # Add assistant message to conversation
            assistant_msg = Message(
                role="assistant", content=assistant_message, timestamp=utc_now()
            )
            self.conversations[conversation_id].append(assistant_msg)

            # Step 5: Store interaction (async, don't block response)
            storage_result = None
            if self.config.auto_store_interactions:
                storage_result = await self._store_interaction(
                    user_message=message,
                    assistant_message=assistant_message,
                    retrieval_context=retrieval_result,
                )

            # Clear retrieved context from working memory (prepare for next message)
            self.memory.working.clear_retrieved()

            processing_time = time.time() - start_time

            # Build response
            response = AgentResponse(
                message=assistant_message,
                conversation_id=conversation_id,
                retrieved_context=self._summarize_retrieval(retrieval_result),
                storage_result=storage_result,
                processing_time=processing_time,
            )

            logger.info(
                f"Processed message in {processing_time:.2f}s, "
                f"retrieved {len(retrieval_result.episodes)} episodes, "
                f"{len(retrieval_result.facts)} facts"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to process message: {e}", exc_info=True)
            raise AgentError(f"Message processing failed: {e}") from e

    async def query_memory(
        self,
        query: str,
        include_semantic: bool = True,
        include_episodic: bool = True,
    ) -> RetrievalResult:
        """Explicitly query memory without generating a response.

        Useful for memory exploration and debugging.

        Args:
            query: Query text.
            include_semantic: Whether to query semantic memory.
            include_episodic: Whether to query episodic memory.

        Returns:
            RetrievalResult with matching facts and episodes.
        """
        logger.debug(f"Explicit memory query: {query}")
        return await self.memory.query(
            query=query,
            include_semantic=include_semantic,
            include_episodic=include_episodic,
            limit=max(self.config.max_retrieved_episodes, self.config.max_retrieved_facts),
        )

    def start_conversation(self) -> str:
        """Initialize a new conversation.

        Returns:
            Conversation ID for the new conversation.
        """
        conversation_id = self._generate_conversation_id()
        self.conversations[conversation_id] = []
        logger.info(f"Started conversation {conversation_id}")
        return conversation_id

    async def end_conversation(self, conversation_id: str) -> None:
        """Finalize conversation and clean up.

        This method:
        - Ensures all interactions are stored
        - Clears conversation history
        - Prepares working memory for next conversation

        Args:
            conversation_id: ID of conversation to end.
        """
        if conversation_id not in self.conversations:
            logger.warning(f"Attempted to end unknown conversation {conversation_id}")
            return

        # Clear from tracking
        del self.conversations[conversation_id]

        # Clear working memory for fresh start
        self.memory.working.clear_retrieved()

        logger.info(f"Ended conversation {conversation_id}")

    def get_conversation_history(self, conversation_id: str) -> list[Message]:
        """Get the full history of a conversation.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            List of messages in chronological order.
        """
        return self.conversations.get(conversation_id, [])

    # ========== Private Helper Methods ==========

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        return f"conv_{uuid.uuid4().hex[:16]}"

    async def _retrieve_context(self, query: str) -> RetrievalResult:
        """Retrieve relevant context from memory.

        Args:
            query: Query text (typically the user's message).

        Returns:
            RetrievalResult with relevant facts and episodes.
        """
        return await self.memory.query(
            query=query,
            include_semantic=True,
            include_episodic=True,
            limit=max(self.config.max_retrieved_episodes, self.config.max_retrieved_facts),
        )

    def _build_conversation_history(self, conversation_id: str) -> list[Message]:
        """Build conversation history for prompt.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            List of messages in the conversation.
        """
        return self.conversations.get(conversation_id, [])

    async def _generate_response(
        self, conversation_history: list[Message], user_message: str
    ) -> str:
        """Generate response using reasoning model.

        Args:
            conversation_history: Previous messages in conversation.
            user_message: Current user message.

        Returns:
            Generated assistant response.
        """
        # Render full context from working memory
        context = self.memory.working.render_context()

        # Build messages for chat completion
        messages = []

        # System message with context
        system_message = {
            "role": "system",
            "content": context,
        }
        messages.append(system_message)

        # Add conversation history (excluding current message which is already in context)
        for msg in conversation_history[:-1]:  # Exclude last message (current user msg)
            messages.append({"role": msg.role, "content": msg.content})

        # Current user message
        messages.append({"role": "user", "content": user_message})

        # Generate response
        response = await self.llm.chat(
            model=self.reasoner_model,
            messages=messages,
            temperature=0.7,
        )

        return response

    async def _store_interaction(
        self,
        user_message: str,
        assistant_message: str,
        retrieval_context: RetrievalResult,
    ) -> StorageResult:
        """Store interaction in memory.

        Args:
            user_message: User's message.
            assistant_message: Assistant's response.
            retrieval_context: Retrieved context used for the response.

        Returns:
            StorageResult with storage metadata.
        """
        interaction = Interaction(
            user_message=user_message,
            assistant_message=assistant_message,
            occurred_at=utc_now(),
            context=self._format_retrieval_context(retrieval_context),
        )

        return await self.memory.store_interaction(interaction)

    def _summarize_retrieval(self, result: RetrievalResult) -> dict[str, Any]:
        """Create summary of retrieved context for response metadata.

        Args:
            result: Retrieval result to summarize.

        Returns:
            Dictionary with retrieval summary.
        """
        return {
            "episodes_count": len(result.episodes),
            "facts_count": len(result.facts),
            "episode_ids": [ep.id for ep in result.episodes[:3]],  # First 3
            "top_relevance_scores": dict(
                list(result.relevance_scores.items())[:5]  # Top 5
            ),
        }

    def _format_retrieval_context(self, result: RetrievalResult) -> str:
        """Format retrieval result as string for storage context.

        Args:
            result: Retrieval result to format.

        Returns:
            Formatted string summary.
        """
        parts = []

        if result.episodes:
            parts.append(f"Retrieved {len(result.episodes)} episodes")
            for ep in result.episodes[:2]:  # Top 2
                parts.append(f"  - [{ep.id}] {ep.summary or ep.content[:100]}")

        if result.facts:
            parts.append(f"Retrieved {len(result.facts)} facts")
            for fact in result.facts[:3]:  # Top 3
                parts.append(f"  - {fact.subject_id} {fact.predicate} {fact.object_value}")

        return "\n".join(parts) if parts else "No context retrieved"
