#!/usr/bin/env python3
import os
from typing import Any, Dict, List, Optional
from typing_extensions import TypeAlias

from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import FieldCondition

from .base import CodeSearchPlugin
from .logger import IndexerLogger

# Load environment variables
load_dotenv()

# Configure logging
logger = IndexerLogger(__name__).get_logger()

# Type alias for conditions
Condition: TypeAlias = (
    FieldCondition
    | models.IsEmptyCondition
    | models.IsNullCondition
    | models.HasIdCondition
    | models.HasVectorCondition
    | models.NestedCondition
    | models.Filter
)


class QdrantSearchPlugin(CodeSearchPlugin):
    """Plugin for searching code using Qdrant vector store."""

    def __init__(
        self,
        collection_name: str = "code_reference",
        qdrant_url: str = "http://localhost:6550",
        qdrant_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        qdrant_client: Optional[QdrantClient] = None,
    ):
        """Initialize the plugin.

        Args:
            collection_name: Name of the Qdrant collection to search
            qdrant_url: URL of the Qdrant server
            qdrant_api_key: Optional API key for Qdrant
            openai_api_key: Optional API key for OpenAI
            qdrant_client: Optional existing Qdrant client
        """
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.qdrant_client = qdrant_client
        self.openai_client: Optional[AsyncOpenAI] = None

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "qdrant"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Search code using Qdrant vector store"

    async def setup(self) -> None:
        """Set up Qdrant and OpenAI clients."""
        if not self.qdrant_api_key:
            raise ValueError("Qdrant API key not provided")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")

        # Initialize clients if not provided
        if not self.qdrant_client:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=30,
                verify=False,  # Suppress SSL warnings for local connections
            )
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)

        # Verify collection exists
        collections = self.qdrant_client.get_collections()
        logger.info(
            f"Found collections: {[col.name for col in collections.collections]}"
        )
        if not any(
            collection.name == self.collection_name
            for collection in collections.collections
        ):
            raise ValueError(
                f"Collection {self.collection_name} does not exist. Please index some code first."
            )
        logger.info(f"Using collection: {self.collection_name}")

        # Get collection info
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        logger.info(f"Collection size: {collection_info.points_count} points")

    async def search(
        self,
        query: str,
        *,
        filter_paths: Optional[List[str]] = None,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for code snippets using semantic search."""
        if not self.qdrant_client or not self.openai_client:
            raise RuntimeError("Plugin not set up. Call setup() first.")

        # Get query embedding
        response = await self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query.lower(),  # Convert query to lowercase for consistency
        )
        query_vector = response.data[0].embedding

        # Build filter if paths provided
        search_filter = None
        if filter_paths:
            path_conditions: List[Condition] = []
            for path in filter_paths:
                # Convert path to lowercase for case-insensitive matching
                path_lower = path.lower()
                # Search in filepath and repository name if available
                conditions: List[Condition] = [
                    models.FieldCondition(
                        key="metadata.filepath",
                        match=models.MatchText(text=path_lower),
                    )
                ]
                if "repo" in path_lower:
                    conditions.append(
                        models.FieldCondition(
                            key="metadata.repo.name",
                            match=models.MatchText(text=path_lower),
                        )
                    )
                path_conditions.extend(conditions)
            if path_conditions:
                search_filter = models.Filter(should=path_conditions)

        # For TODO searches, prioritize chunks with has_todo flag
        if "todo" in query.lower():
            todo_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.has_todo", match=models.MatchValue(value=True)
                    ),
                    models.FieldCondition(
                        key="metadata.is_comment", match=models.MatchValue(value=True)
                    ),
                ]
            )
            if search_filter:
                search_filter = models.Filter(must=[search_filter, todo_filter])
            else:
                search_filter = todo_filter

        # Search in Qdrant
        logger.info(f"Searching with min_score={min_score}")
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit * 2,  # Get more results for post-filtering
            score_threshold=min_score,
            with_payload=True,
            with_vectors=False,
        )
        logger.info(f"Found {len(search_results)} initial matches")

        # Format results
        results = []
        for point in search_results:
            # Skip if payload is missing
            if not point.payload:
                logger.debug("Skipping result with no payload")
                continue

            # Skip if code is too short or empty
            code = point.payload.get("text", "").strip()
            if len(code) < 10:
                logger.debug(f"Skipping short code: {len(code)} chars")
                continue

            # Get metadata safely
            metadata = point.payload.get("metadata", {})
            if not isinstance(metadata, dict):
                logger.debug("Skipping invalid metadata")
                continue

            # Get the full context
            start_line = metadata.get("start_line", 0)
            end_line = metadata.get("end_line", 0)

            # Split code into lines and add context
            lines = code.split("\n")

            # Remove empty lines at the start and end
            original_line_count = len(lines)
            while lines and not lines[0].strip():
                lines.pop(0)
                start_line += 1
            while lines and not lines[-1].strip():
                lines.pop()
                end_line -= 1

            logger.debug(
                f"Lines after cleanup: {len(lines)} (was {original_line_count})"
            )

            # Ensure we have at least 3 lines of context
            if len(lines) < 3:
                logger.debug("Skipping result with less than 3 lines")
                continue

            # Build result with context
            result = {
                "score": point.score,
                "filepath": metadata.get("filepath", "unknown"),
                "code": "\n".join(lines),
                "start_line": start_line,
                "end_line": start_line + len(lines) - 1,  # Fix end line calculation
                "source": metadata.get("source", "unknown"),
                "file_type": metadata.get("file_type", "unknown"),
                "metadata": metadata,  # Include full metadata for display
            }

            logger.info(
                f"Found match in {result['filepath']} (score: {point.score:.2f}, lines: {len(lines)})"
            )

            # Add repository info if available
            repo = metadata.get("repo")
            if repo and isinstance(repo, dict):
                result.update(
                    {
                        "repository": {
                            "name": repo.get("name", "unknown"),
                            "owner": repo.get("owner", "unknown"),
                            "url": repo.get("url", "unknown"),
                            "branch": repo.get("branch", "main"),
                            "commit": repo.get("commit", "unknown"),
                        }
                    }
                )

            results.append(result)

            # Stop after getting enough valid results
            if len(results) >= limit:
                break

        logger.info(f"Returning {len(results)} final matches")
        return results

    async def cleanup(self) -> None:
        """Clean up clients."""
        # Nothing to clean up for these clients
        self.qdrant_client = None
        self.openai_client = None
