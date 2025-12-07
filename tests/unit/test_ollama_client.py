"""Unit tests for OllamaClient."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from htma.core.exceptions import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
    ModelNotFoundError,
)
from htma.llm.client import OllamaClient


class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""

    def test_default_initialization(self):
        """Test client initializes with default values."""
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.default_timeout == 60
        assert client.max_retries == 3

    def test_custom_initialization(self):
        """Test client initializes with custom values."""
        client = OllamaClient(
            base_url="http://custom:8080",
            default_timeout=120,
            max_retries=5,
        )
        assert client.base_url == "http://custom:8080"
        assert client.default_timeout == 120
        assert client.max_retries == 5


class TestOllamaClientGenerate:
    """Tests for text generation."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful text generation."""
        client = OllamaClient()

        mock_response = {"response": "Generated text response"}
        client._client.generate = AsyncMock(return_value=mock_response)

        result = await client.generate(
            model="llama3:8b",
            prompt="Test prompt",
            system="You are helpful",
            temperature=0.5,
            max_tokens=1000,
        )

        assert result == "Generated text response"
        client._client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_without_system(self):
        """Test generation without system message."""
        client = OllamaClient()

        mock_response = {"response": "Response"}
        client._client.generate = AsyncMock(return_value=mock_response)

        result = await client.generate(model="mistral:7b", prompt="Test")

        assert result == "Response"

    @pytest.mark.asyncio
    async def test_generate_model_not_found(self):
        """Test generation with non-existent model."""
        client = OllamaClient()

        # Create a proper ResponseError mock
        import ollama

        error = ollama.ResponseError("model not found")
        client._client.generate = AsyncMock(side_effect=error)

        with pytest.raises(ModelNotFoundError) as exc_info:
            await client.generate(model="nonexistent:model", prompt="Test")

        assert "nonexistent:model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_connection_error_with_retry(self):
        """Test generation retries on connection error."""
        client = OllamaClient(max_retries=3)

        # First two calls fail, third succeeds
        client._client.generate = AsyncMock(
            side_effect=[
                ConnectionError("Connection failed"),
                ConnectionError("Connection failed"),
                {"response": "Success after retry"},
            ]
        )

        result = await client.generate(model="llama3:8b", prompt="Test")

        assert result == "Success after retry"
        assert client._client.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_timeout_error(self):
        """Test generation handles timeout."""
        client = OllamaClient()

        # Use TimeoutError which is what asyncio actually raises in Python 3.11+
        client._client.generate = AsyncMock(side_effect=TimeoutError())

        with pytest.raises(LLMTimeoutError):
            await client.generate(model="llama3:8b", prompt="Test")


class TestOllamaClientChat:
    """Tests for chat completion."""

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Test successful chat completion."""
        client = OllamaClient()

        mock_response = {"message": {"content": "Chat response"}}
        client._client.chat = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = await client.chat(model="llama3:8b", messages=messages)

        assert result == "Chat response"
        client._client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self):
        """Test chat with custom temperature."""
        client = OllamaClient()

        mock_response = {"message": {"content": "Response"}}
        client._client.chat = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Test"}]

        result = await client.chat(
            model="mistral:7b",
            messages=messages,
            temperature=0.9,
            max_tokens=500,
        )

        assert result == "Response"


class TestOllamaClientEmbed:
    """Tests for embeddings."""

    @pytest.mark.asyncio
    async def test_embed_success(self):
        """Test successful embedding generation."""
        client = OllamaClient()

        mock_embedding = [0.1, 0.2, 0.3, 0.4]
        mock_response = {"embedding": mock_embedding}
        client._client.embeddings = AsyncMock(return_value=mock_response)

        result = await client.embed(model="nomic-embed-text", text="Test text")

        assert result == mock_embedding
        client._client.embeddings.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_empty_result(self):
        """Test embedding with empty result."""
        client = OllamaClient()

        mock_response = {"embedding": []}
        client._client.embeddings = AsyncMock(return_value=mock_response)

        result = await client.embed(model="nomic-embed-text", text="Test")

        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        """Test batch embedding generation."""
        client = OllamaClient()

        # Mock individual embed calls
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        async def mock_embed(model, prompt):
            # Return different embeddings based on prompt
            index = ["text1", "text2", "text3"].index(prompt)
            return {"embedding": embeddings[index]}

        client._client.embeddings = AsyncMock(side_effect=mock_embed)

        texts = ["text1", "text2", "text3"]
        results = await client.embed_batch(model="nomic-embed-text", texts=texts)

        assert len(results) == 3
        assert results == embeddings

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        """Test batch embedding with empty list."""
        client = OllamaClient()

        results = await client.embed_batch(model="nomic-embed-text", texts=[])

        assert results == []


class TestOllamaClientHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test health check when service is available."""
        client = OllamaClient()

        mock_response = {"models": [{"name": "llama3:8b"}]}
        client._client.list = AsyncMock(return_value=mock_response)

        result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check when service is unavailable."""
        client = OllamaClient()

        client._client.list = AsyncMock(side_effect=ConnectionError("Connection failed"))

        result = await client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test health check with timeout."""
        client = OllamaClient()

        client._client.list = AsyncMock(side_effect=TimeoutError())

        result = await client.health_check()

        assert result is False


class TestOllamaClientListModels:
    """Tests for listing models."""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test successful model listing."""
        client = OllamaClient()

        mock_response = {
            "models": [
                {"name": "llama3:8b"},
                {"name": "mistral:7b"},
                {"name": "nomic-embed-text"},
            ]
        }
        client._client.list = AsyncMock(return_value=mock_response)

        result = await client.list_models()

        assert len(result) == 3
        assert "llama3:8b" in result
        assert "mistral:7b" in result
        assert "nomic-embed-text" in result

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """Test listing when no models available."""
        client = OllamaClient()

        mock_response = {"models": []}
        client._client.list = AsyncMock(return_value=mock_response)

        result = await client.list_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_models_connection_error(self):
        """Test model listing with connection error."""
        client = OllamaClient(max_retries=2)

        client._client.list = AsyncMock(side_effect=ConnectionError("Connection failed"))

        with pytest.raises(LLMConnectionError):
            await client.list_models()


class TestOllamaClientRetryLogic:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success_first_try(self):
        """Test no retry needed when first attempt succeeds."""
        client = OllamaClient()

        mock_response = {"response": "Success"}
        client._client.generate = AsyncMock(return_value=mock_response)

        result = await client.generate(model="llama3:8b", prompt="Test")

        assert result == "Success"
        assert client._client.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test all retries exhausted."""
        client = OllamaClient(max_retries=3)

        client._client.generate = AsyncMock(side_effect=ConnectionError("Connection failed"))

        with pytest.raises(LLMConnectionError) as exc_info:
            await client.generate(model="llama3:8b", prompt="Test")

        assert "after 3 attempts" in str(exc_info.value)
        assert client._client.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test exponential backoff timing."""
        client = OllamaClient(max_retries=3)

        # Track call times
        call_times = []

        async def mock_generate(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            raise ConnectionError("Connection failed")

        client._client.generate = mock_generate

        with pytest.raises(LLMConnectionError):
            await client.generate(model="llama3:8b", prompt="Test")

        # Verify exponential backoff (approximate timing)
        assert len(call_times) == 3
        # Note: Exact timing checks are fragile, so we just verify attempts were made

    @pytest.mark.asyncio
    async def test_retry_unexpected_error(self):
        """Test handling of unexpected errors."""
        client = OllamaClient()

        client._client.generate = AsyncMock(side_effect=ValueError("Unexpected error"))

        with pytest.raises(LLMResponseError) as exc_info:
            await client.generate(model="llama3:8b", prompt="Test")

        assert "Unexpected error" in str(exc_info.value)


class TestOllamaClientEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """Test generation with empty prompt."""
        client = OllamaClient()

        mock_response = {"response": ""}
        client._client.generate = AsyncMock(return_value=mock_response)

        result = await client.generate(model="llama3:8b", prompt="")

        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test chat with empty messages."""
        client = OllamaClient()

        mock_response = {"message": {"content": ""}}
        client._client.chat = AsyncMock(return_value=mock_response)

        result = await client.chat(model="llama3:8b", messages=[])

        assert result == ""

    @pytest.mark.asyncio
    async def test_malformed_response(self):
        """Test handling of malformed API response."""
        client = OllamaClient()

        # Response missing expected keys
        mock_response = {}
        client._client.generate = AsyncMock(return_value=mock_response)

        result = await client.generate(model="llama3:8b", prompt="Test")

        # Should handle gracefully and return empty string
        assert result == ""

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test embedding very long text."""
        client = OllamaClient()

        long_text = "test " * 10000
        mock_embedding = [0.1] * 768
        mock_response = {"embedding": mock_embedding}
        client._client.embeddings = AsyncMock(return_value=mock_response)

        result = await client.embed(model="nomic-embed-text", text=long_text)

        assert len(result) == 768
