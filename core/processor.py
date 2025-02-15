"""Core processing functionality for code indexing."""

import logging
from typing import Optional

from rich.console import Console

from plugins import CodeSourcePlugin
from utils.config import get_settings
from utils.errors import ConfigurationError
from core.collection import CollectionManager
from core.embeddings import EmbeddingManager
from core.search import SearchManager
from core.source_processor import SourceProcessor

logger = logging.getLogger(__name__)
settings = get_settings()


class ProcessingError(Exception):
    """Base class for processing errors."""

    pass


class CodeProcessor:
    """Process and index code from various sources."""

    def __init__(
        self, embedding_provider: str = settings.DEFAULT_EMBEDDING_PROVIDER.value
    ):
        """Initialize processor.

        Args:
            embedding_provider: Name of embedding provider to use ('openai' or 'lmstudio')

        Raises:
            ProcessingError: If initialization fails
            ConfigurationError: If provider configuration is invalid
        """
        try:
            # Initialize managers
            self.collection = CollectionManager()
            self.embeddings = EmbeddingManager(embedding_provider)
            self.source = SourceProcessor(self.collection, self.embeddings)
            self.search = SearchManager(self.collection.client)

        except ConfigurationError as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e
        except Exception as e:
            raise ProcessingError(f"Failed to initialize processor: {e}") from e

    async def setup_collection(self, force: bool = False) -> None:
        """Setup or reset collection.

        Args:
            force: If True, recreate collection even if it exists

        Raises:
            ProcessingError: If setup fails
        """
        try:
            await self.collection.setup_collection(
                vector_size=self.embeddings.embedding.dimension,
                force=force,
            )
            await self.search.setup()
        except Exception as e:
            raise ProcessingError(f"Failed to setup collection: {e}") from e

    async def process_source(self, source: CodeSourcePlugin, console: Console) -> None:
        """Process code from a source plugin.

        Args:
            source: Source plugin instance
            console: Console for output

        Raises:
            ProcessingError: If processing fails
        """
        await self.source.process_source(source, console)

    async def search_code(
        self,
        query: str,
        *,
        filter_paths: Optional[list[str]] = None,
        min_score: float = settings.DEFAULT_MIN_SCORE,
        limit: int = settings.DEFAULT_SEARCH_LIMIT,
        context_lines: int = settings.DEFAULT_CONTEXT_LINES,
        console: Console,
    ) -> None:
        """Search for code.

        Args:
            query: Search query
            filter_paths: Optional paths to filter results
            min_score: Minimum similarity score
            limit: Maximum number of results
            context_lines: Number of context lines to show
            console: Console for output

        Raises:
            ProcessingError: If search fails
        """
        await self.search.search(
            query,
            filter_paths=filter_paths,
            min_score=min_score,
            limit=limit,
            context_lines=context_lines,
            console=console,
        )
