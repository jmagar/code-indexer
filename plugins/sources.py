"""Source management using Qdrant collection."""

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, cast
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.http import models
from rich.console import Console
from rich.table import Table

from .logger import IndexerLogger

logger = IndexerLogger(__name__).get_logger()

SOURCES_COLLECTION = "sources"


class SourceMetadata(TypedDict):
    """Type definition for source metadata."""

    source_id: str
    type: str
    added_at: str
    settings: Dict[str, List[str]]
    url: Optional[str]
    branch: Optional[str]
    path: Optional[str]
    last_hash: Optional[str]
    last_commit: Optional[str]
    last_ingested: Optional[str]
    last_updated: Optional[str]


PayloadDict = Dict[str, Any]


class SourceManager:
    """Manage code sources in Qdrant."""

    def __init__(self) -> None:
        """Initialize source manager."""
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6335"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30,
        )

    async def setup_collection(self) -> None:
        """Create sources collection if it doesn't exist."""
        try:
            collections = self.qdrant_client.get_collections()
            exists = any(
                col.name == SOURCES_COLLECTION for col in collections.collections
            )

            if not exists:
                # Create collection with minimal vector size (we don't need vectors)
                self.qdrant_client.create_collection(
                    collection_name=SOURCES_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=1,  # Minimal size since we don't use vectors
                        distance=models.Distance.COSINE,
                    ),
                )

                # Create payload indexes
                self.qdrant_client.create_payload_index(
                    collection_name=SOURCES_COLLECTION,
                    field_name="source_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.qdrant_client.create_payload_index(
                    collection_name=SOURCES_COLLECTION,
                    field_name="type",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.qdrant_client.create_payload_index(
                    collection_name=SOURCES_COLLECTION,
                    field_name="url",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.qdrant_client.create_payload_index(
                    collection_name=SOURCES_COLLECTION,
                    field_name="path",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

                logger.info(f"Created {SOURCES_COLLECTION} collection")

        except Exception as e:
            logger.error(f"Error setting up sources collection: {e}")
            raise

    def _generate_source_id(self, source_type: str, path: str) -> str:
        """Generate a unique source ID."""
        if source_type == "github":
            # Extract user/repo from GitHub URL
            parsed = urlparse(path)
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2:
                return f"github:{parts[0]}/{parts[1]}"
            raise ValueError(f"Invalid GitHub URL: {path}")
        else:
            # Use absolute path for local sources
            abs_path = os.path.abspath(path)
            return f"local:{abs_path}"

    async def add_source(
        self,
        source_type: str,
        path: str,
        branch: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
    ) -> str:
        """Add a new source.

        Args:
            source_type: Type of source ("github" or "local")
            path: Repository URL or local path
            branch: Git branch (for GitHub sources)
            exclude_patterns: Patterns to exclude
            include_patterns: Patterns to include

        Returns:
            Source ID
        """
        try:
            source_id = self._generate_source_id(source_type, path)

            # Prepare source data
            source: PayloadDict = {
                "source_id": source_id,  # Store source ID in payload
                "type": source_type,
                "added_at": datetime.now(timezone.utc).isoformat(),
                "settings": {
                    "exclude_patterns": exclude_patterns or [],
                    "include_patterns": include_patterns or [],
                },
                "url": None,
                "branch": None,
                "path": None,
                "last_hash": None,
                "last_commit": None,
                "last_ingested": None,
                "last_updated": None,
            }

            if source_type == "github":
                source["url"] = path
                source["branch"] = branch
                source["last_commit"] = None
            else:
                source["path"] = os.path.abspath(path)
                source["last_hash"] = None

            # Store in Qdrant with UUID as point ID
            point_id = str(uuid.uuid4())
            self.qdrant_client.upsert(
                collection_name=SOURCES_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,  # Use UUID as point ID
                        vector=[0.0],  # Dummy vector
                        payload=source,  # Dict[str, Any] is compatible with Payload
                    )
                ],
            )

            logger.info(f"Added source: {source_id}")
            return source_id

        except Exception as e:
            logger.error(f"Error adding source: {e}")
            raise

    async def remove_source(self, source_id: str) -> bool:
        """Remove a source.

        Args:
            source_id: Source ID to remove

        Returns:
            True if source was removed
        """
        try:
            # Find point by source_id in payload
            response = self.qdrant_client.scroll(
                collection_name=SOURCES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        )
                    ]
                ),
                limit=1,
            )

            points, _ = response
            if not points:
                logger.warning(f"Source not found: {source_id}")
                return False

            # Remove source using point ID
            self.qdrant_client.delete(
                collection_name=SOURCES_COLLECTION,
                points_selector=models.PointIdsList(points=[points[0].id]),
            )

            logger.info(f"Removed source: {source_id}")
            return True

        except Exception as e:
            logger.error(f"Error removing source: {e}")
            raise

    async def list_sources(
        self, console: Optional[Console] = None
    ) -> List[SourceMetadata]:
        """List all sources.

        Args:
            console: Optional console for pretty printing

        Returns:
            List of sources
        """
        try:
            # Get all sources
            response = self.qdrant_client.scroll(
                collection_name=SOURCES_COLLECTION,
                limit=100,  # Adjust if needed
                with_payload=True,
            )

            points, _ = response

            if console:
                # Create pretty table
                table = Table(
                    show_header=True,
                    header_style="bold #81A1C1",
                    show_lines=True,
                    title="[bold #88C0D0]Indexed Sources[/]",
                )

                table.add_column("ID", style="#88C0D0")
                table.add_column("Type", style="#8FBCBB")
                table.add_column("Location", style="#A3BE8C")
                table.add_column("Last Ingested", style="#616E88")
                table.add_column("Status", style="#81A1C1")

                for point in points:
                    if point.payload is None:
                        continue

                    source = point.payload
                    source_id = source.get("source_id", "unknown")
                    source_type = source.get("type", "unknown")

                    # Format location
                    location = (
                        source.get("url")
                        if source_type == "github"
                        else source.get("path", "")
                    )

                    # Format last ingestion
                    last_ingested = source.get("last_ingested", "Never")
                    if last_ingested and last_ingested != "Never":
                        last_dt = datetime.fromisoformat(
                            last_ingested.replace("Z", "+00:00")
                        )
                        last_ingested = last_dt.strftime("%Y-%m-%d %H:%M:%S")

                    # Determine status
                    if source_type == "github":
                        status = f"Branch: {source.get('branch', 'default')}"
                        if source.get("last_commit"):
                            status += f"\nCommit: {source['last_commit'][:8]}"
                    else:
                        status = "Ready"
                        if source.get("last_hash"):
                            status = "Indexed"

                    table.add_row(
                        str(source_id),
                        source_type,
                        str(location),
                        last_ingested,
                        status,
                    )

                console.print(table)
                console.print()

            # Return sources for programmatic use
            result: List[SourceMetadata] = []
            for point in points:
                if point.payload is None:
                    continue
                payload = cast(PayloadDict, point.payload)
                source_id = payload.get("source_id")
                if not isinstance(source_id, str):
                    continue
                result.append(cast(SourceMetadata, {"id": source_id, **payload}))
            return result

        except Exception as e:
            logger.error(f"Error listing sources: {e}")
            raise

    async def get_source(self, source_id: str) -> Optional[SourceMetadata]:
        """Get a specific source.

        Args:
            source_id: Source ID to retrieve

        Returns:
            Source data or None if not found
        """
        try:
            # Find point by source_id in payload
            response = self.qdrant_client.scroll(
                collection_name=SOURCES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )

            points, _ = response
            if not points or points[0].payload is None:
                return None

            point = points[0]
            payload = cast(PayloadDict, point.payload)
            source_id = payload.get("source_id")
            if not isinstance(source_id, str):
                return None
            return cast(SourceMetadata, {"id": source_id, **payload})

        except Exception as e:
            logger.error(f"Error getting source: {e}")
            raise

    async def update_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """Update source metadata.

        Args:
            source_id: Source ID to update
            updates: Dictionary of updates to apply

        Returns:
            True if source was updated
        """
        try:
            # Get current source
            source = await self.get_source(source_id)
            if not source:
                return False

            # Find point by source_id
            response = self.qdrant_client.scroll(
                collection_name=SOURCES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        )
                    ]
                ),
                limit=1,
            )

            points, _ = response
            if not points:
                return False

            point_id = points[0].id

            # Update fields
            source_dict = dict(source)
            source_dict.update(updates)
            source_dict["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Store updated source using original point ID
            self.qdrant_client.upsert(
                collection_name=SOURCES_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,  # Use original point ID
                        vector=[0.0],  # Dummy vector
                        payload=source_dict,
                    )
                ],
            )

            return True

        except Exception as e:
            logger.error(f"Error updating source: {e}")
            raise
