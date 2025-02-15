"""Embedding providers and utilities."""

from abc import ABC, abstractmethod
from typing import List, Protocol, Optional
import os
import logging
import aiohttp

from utils.config import get_settings
from utils.errors import ConfigurationError

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def name(self) -> str:
        """Get provider name."""
        ...

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        ...


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get provider name."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        pass


class OpenAIEmbedding(BaseEmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self) -> None:
        """Initialize OpenAI embedding provider."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY environment variable not set")

        # Lazy import to avoid dependency if not used
        import openai

        openai.api_key = api_key
        self._client = openai.AsyncClient()

    @property
    def name(self) -> str:
        """Get provider name."""
        return "OpenAI"

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return settings.OPENAI_EMBEDDING_DIMENSION

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI's API."""
        try:
            response = await self._client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise


class LMStudioEmbedding(BaseEmbeddingProvider):
    """LM Studio embedding provider."""

    def __init__(self) -> None:
        """Initialize LM Studio embedding provider."""
        self.base_url = str(settings.LMSTUDIO_URL) if settings.LMSTUDIO_URL else None
        if not self.base_url:
            raise ConfigurationError("LMStudio URL not configured")

        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        """Get provider name."""
        return "LM Studio"

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return settings.LMSTUDIO_EMBEDDING_DIMENSION

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using LM Studio's API."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            async with self._session.post(
                f"{self.base_url}/embeddings",
                json={"input": texts},
            ) as response:
                if response.status != 200:
                    raise ValueError(f"LM Studio API error: {await response.text()}")
                data = await response.json()
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"LM Studio embedding failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None


class EmbeddingManager:
    """Manages embedding providers with fallback support."""

    def __init__(
        self, provider: str = settings.DEFAULT_EMBEDDING_PROVIDER.value
    ) -> None:
        """Initialize embedding manager.

        Args:
            provider: Name of primary embedding provider ('openai' or 'lmstudio')

        Raises:
            ConfigurationError: If provider configuration is invalid
        """
        self.primary_provider = provider
        self.embedding: BaseEmbeddingProvider
        self.fallback_embedding: Optional[BaseEmbeddingProvider] = None

        # Set up primary provider
        if provider == "openai":
            self.embedding = OpenAIEmbedding()
        elif provider == "lmstudio":
            self.embedding = LMStudioEmbedding()
            # Create fallback OpenAI embedder
            try:
                self.fallback_embedding = OpenAIEmbedding()
            except ConfigurationError:
                self.fallback_embedding = None
                logger.warning("No OpenAI fallback available - missing API key")
        else:
            raise ConfigurationError(f"Unknown embedding provider: {provider}")

        logger.info(f"Using {self.embedding.name} embedding provider")

    async def get_embeddings(
        self, texts: List[str], batch_size: int = settings.EMBEDDING_BATCH_SIZE
    ) -> List[List[float]]:
        """Get embeddings for texts in batches with fallback support."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"Processing embedding batch {batch_num}/{total_batches}")

            try:
                # Clean and validate input texts
                cleaned_batch = [
                    text.strip() for text in batch if text and text.strip()
                ]
                if not cleaned_batch:
                    logger.warning(f"Skipping empty batch {batch_num}")
                    continue

                try:
                    # Try primary embedding provider
                    response = await self.embedding.get_embeddings(cleaned_batch)
                    batch_embeddings = [item for item in response]
                except Exception as e:
                    logger.error(f"Primary embedding provider failed: {e}")
                    if self.fallback_embedding:
                        logger.info("Falling back to OpenAI embeddings")
                        response = await self.fallback_embedding.get_embeddings(
                            cleaned_batch
                        )
                        batch_embeddings = [item for item in response]
                    else:
                        raise

                all_embeddings.extend(batch_embeddings)
                logger.info(
                    f"Successfully processed embedding batch {batch_num}/{total_batches}"
                )
            except Exception as e:
                logger.error(f"Failed to process embedding batch {batch_num}: {e}")
                raise

        return all_embeddings
