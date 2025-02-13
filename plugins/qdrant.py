#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .base import CodeSearchPlugin
from .logger import IndexerLogger

# Load environment variables
load_dotenv()

# Configure logging
logger = IndexerLogger(__name__).get_logger()


class QdrantSearchPlugin(CodeSearchPlugin):
    """Plugin for searching code using Qdrant vector store."""

    def __init__(
        self,
        collection_name: str = "code_reference",
        qdrant_url: str = "http://localhost:6335",
        qdrant_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """Initialize the plugin.

        Args:
            collection_name: Name of the Qdrant collection to search
            qdrant_url: URL of the Qdrant server
            qdrant_api_key: Optional API key for Qdrant
            openai_api_key: Optional API key for OpenAI
        """
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.qdrant_client = None
        self.openai_client = None

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

        # Initialize clients
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=30,
        )
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)

        # Verify collection exists
        collections = self.qdrant_client.get_collections()
        if not any(
            collection.name == self.collection_name
            for collection in collections.collections
        ):
            raise ValueError(
                f"Collection {self.collection_name} does not exist. Please index some code first."
            )

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
            input=query,
        )
        query_vector = response.data[0].embedding

        # Build filter if paths provided
        search_filter = None
        if filter_paths:
            path_conditions = []
            for path in filter_paths:
                # Search in filepath and repository name if available
                path_conditions.extend(
                    [
                        models.FieldCondition(
                            key="metadata.filepath", match=models.MatchText(text=path)
                        ),
                        models.FieldCondition(
                            key="metadata.repo.name", match=models.MatchText(text=path)
                        )
                        if "repo" in path
                        else None,
                    ]
                )
            path_conditions = [c for c in path_conditions if c is not None]
            if path_conditions:
                search_filter = models.Filter(should=path_conditions)

        # Search in Qdrant
        response = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit * 2,  # Get more results for post-filtering
            score_threshold=min_score,
            with_payload=True,
            with_vectors=False,
        )

        # Format results
        results = []
        for point in response:
            # Skip if payload is missing
            if not point.payload:
                continue

            # Skip if code is too short or empty
            code = point.payload.get("text", "").strip()
            if len(code) < 10:
                continue

            # Get metadata safely
            metadata = point.payload.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            # Build result with context
            result = {
                "score": point.score,
                "filepath": metadata.get("filepath", "unknown"),
                "code": code,
                "start_line": metadata.get("start_line", 0),
                "end_line": metadata.get("end_line", 0),
                "source": metadata.get("source", "unknown"),
                "file_type": metadata.get("file_type", "unknown"),
            }

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

        return results

    async def cleanup(self) -> None:
        """Clean up clients."""
        # Nothing to clean up for these clients
        self.qdrant_client = None
        self.openai_client = None
