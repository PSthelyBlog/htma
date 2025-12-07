"""Ollama client wrapper for LLM operations.

This module provides an async wrapper around the Ollama API for
chat completions and embeddings with robust error handling and retry logic.
"""

import asyncio
from collections.abc import Callable
from typing import Any

import ollama

from htma.core.exceptions import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
    ModelNotFoundError,
)


class OllamaClient:
    """Async client for Ollama LLM operations.

    Provides methods for chat completions, text generation, embeddings,
    and model management with automatic retry and error handling.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434)
        default_timeout: Default timeout in seconds for requests (default: 60)
        max_retries: Maximum number of retry attempts for failed requests (default: 3)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_timeout: int = 60,
        max_retries: int = 3,
    ):
        self.base_url = base_url
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self._client = ollama.AsyncClient(host=base_url)

    async def _retry_with_backoff(
        self,
        func: Callable[..., Any],
        model: str = "unknown",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with exponential backoff retry.

        Args:
            func: Async function to execute
            model: Model name for error messages
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            LLMConnectionError: If all retries fail due to connection issues
            LLMTimeoutError: If request times out
            ModelNotFoundError: If model is not found
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except TimeoutError as e:
                # Handle timeout immediately, no retry
                raise LLMTimeoutError(f"Request timed out after {self.default_timeout}s") from e
            except ollama.ResponseError as e:
                # Check if it's a model not found error
                if "not found" in str(e).lower():
                    raise ModelNotFoundError(model) from e
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    await asyncio.sleep(2**attempt)
            except (ConnectionError, OSError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
            except Exception as e:
                # Catch-all for unexpected errors
                raise LLMResponseError(f"Unexpected error: {str(e)}") from e

        # All retries failed
        raise LLMConnectionError(
            f"Failed to connect to Ollama at {self.base_url} after {self.max_retries} attempts"
        ) from last_exception

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate completion for a prompt.

        Args:
            model: Model name (e.g., "llama3:8b", "mistral:7b")
            prompt: The prompt text
            system: Optional system message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text completion

        Raises:
            ModelNotFoundError: If the model is not available
            LLMConnectionError: If connection to Ollama fails
            LLMTimeoutError: If request times out
        """
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        async def _generate() -> str:
            response = await self._client.generate(
                model=model,
                prompt=prompt,
                system=system,
                options=options,
            )
            return response.get("response", "")  # type: ignore[no-any-return]

        return await self._retry_with_backoff(_generate, model=model)  # type: ignore[no-any-return]

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Chat completion with message history.

        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Assistant's response text

        Raises:
            ModelNotFoundError: If the model is not available
            LLMConnectionError: If connection to Ollama fails
            LLMTimeoutError: If request times out
        """
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        async def _chat() -> str:
            response = await self._client.chat(
                model=model,
                messages=messages,
                options=options,
            )
            message = response.get("message", {})
            return message.get("content", "")  # type: ignore[no-any-return]

        return await self._retry_with_backoff(_chat, model=model)  # type: ignore[no-any-return]

    async def embed(
        self,
        model: str,
        text: str,
    ) -> list[float]:
        """Generate embedding vector for text.

        Args:
            model: Embedding model name (e.g., "nomic-embed-text")
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            ModelNotFoundError: If the model is not available
            LLMConnectionError: If connection to Ollama fails
            LLMTimeoutError: If request times out
        """

        async def _embed() -> list[float]:
            response = await self._client.embeddings(
                model=model,
                prompt=text,
            )
            return response.get("embedding", [])  # type: ignore[no-any-return]

        return await self._retry_with_backoff(_embed, model=model)  # type: ignore[no-any-return]

    async def embed_batch(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            model: Embedding model name
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ModelNotFoundError: If the model is not available
            LLMConnectionError: If connection to Ollama fails
            LLMTimeoutError: If request times out
        """
        # Ollama doesn't have a native batch endpoint, so we do them individually
        # but concurrently for better performance
        tasks = [self.embed(model, text) for text in texts]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """Check if Ollama service is available.

        Returns:
            True if Ollama is reachable and responding, False otherwise
        """
        try:
            await self.list_models()
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names

        Raises:
            LLMConnectionError: If connection to Ollama fails
        """

        async def _list_models() -> list[str]:
            response = await self._client.list()
            models = response.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]

        return await self._retry_with_backoff(_list_models, model="list")  # type: ignore[no-any-return]
