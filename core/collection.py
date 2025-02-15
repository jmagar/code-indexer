"""Qdrant collection management functionality."""

import logging
import os
from typing import Dict, List, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance, PointStruct

from utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CollectionError(Exception):
    """Error related to Qdrant collection operations."""

    pass


class CollectionManager:
    """Manages Qdrant collection operations."""

    def __init__(self) -> None:
        """Initialize collection manager."""
        try:
            self.client = QdrantClient(
                url=str(settings.QDRANT_URL) if settings.QDRANT_URL else None,
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=settings.QDRANT_TIMEOUT,
                verify=settings.QDRANT_VERIFY_SSL,
            )
        except Exception as e:
            raise CollectionError(f"Failed to initialize Qdrant client: {e}") from e

    @property
    def collection_name(self) -> str:
        """Get collection name."""
        return settings.COLLECTION_NAME

    async def setup_collection(self, vector_size: int, force: bool = False) -> None:
        """Setup or reset collection.

        Args:
            vector_size: Size of vectors to store
            force: If True, recreate collection even if it exists

        Raises:
            CollectionError: If collection setup fails
        """
        try:
            logger.info("Setting up Qdrant collection")
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(
                collection.name == settings.COLLECTION_NAME
                for collection in collections.collections
            )

            if exists and force:
                logger.info(f"Recreating collection {settings.COLLECTION_NAME}")
                self.client.delete_collection(collection_name=settings.COLLECTION_NAME)
                exists = False

            if not exists:
                # Create collection with correct vector size
                logger.info(f"Creating collection {settings.COLLECTION_NAME}")
                self.client.create_collection(
                    collection_name=settings.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )

                # Create payload indexes
                logger.info("Creating payload indexes")
                self.client.create_payload_index(
                    collection_name=settings.COLLECTION_NAME,
                    field_name="metadata.filepath",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=settings.COLLECTION_NAME,
                    field_name="metadata.source",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=settings.COLLECTION_NAME,
                    field_name="metadata.file_type",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=settings.COLLECTION_NAME,
                    field_name="metadata.has_todo",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                self.client.create_payload_index(
                    collection_name=settings.COLLECTION_NAME,
                    field_name="metadata.is_comment",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                logger.info("Successfully set up Qdrant collection")
            else:
                logger.info(f"Collection {settings.COLLECTION_NAME} already exists")

        except Exception as e:
            raise CollectionError(f"Failed to setup collection: {e}") from e

    def get_file_hashes(self, source_name: str) -> Dict[str, str]:
        """Get existing file hashes for a source.

        Args:
            source_name: Name of source to get hashes for

        Returns:
            Dictionary mapping filepaths to their hashes

        Raises:
            CollectionError: If query fails
        """
        try:
            file_hashes: Dict[str, str] = {}
            offset = 0

            while True:
                points = self.client.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=source_name),
                            )
                        ]
                    ),
                    offset=offset,
                    limit=settings.QDRANT_BATCH_SIZE,
                    with_payload=True,
                )[0]

                if not points:
                    break

                # Process points in this batch
                for point in points:
                    if not point.payload:
                        continue
                    metadata = point.payload.get("metadata", {})
                    filepath = metadata.get("filepath")
                    file_hash = metadata.get("file_hash")
                    if filepath and file_hash:
                        file_hashes[filepath] = file_hash

                offset += len(points)

            return file_hashes

        except Exception as e:
            raise CollectionError(f"Failed to get file hashes: {e}") from e

    def delete_files(self, source_name: str, filepaths: List[str]) -> None:
        """Delete files from collection.

        Args:
            source_name: Name of source the files belong to
            filepaths: List of filepaths to delete

        Raises:
            CollectionError: If deletion fails
        """
        try:
            for filepath in filepaths:
                self.client.delete(
                    collection_name=settings.COLLECTION_NAME,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.filepath",
                                match=models.MatchValue(value=filepath),
                            ),
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=source_name),
                            ),
                        ]
                    ),
                )
        except Exception as e:
            raise CollectionError(f"Failed to delete files: {e}") from e

    def store_vectors(
        self,
        vectors: List[List[float]],
        chunks: List[Dict[str, Any]],
        batch_size: int = settings.QDRANT_BATCH_SIZE,
    ) -> None:
        """Store vectors and chunks in collection.

        Args:
            vectors: List of vectors to store
            chunks: List of chunks with metadata
            batch_size: Size of batches to store

        Raises:
            CollectionError: If storing fails
        """
        try:
            points = [
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "token_count": len(chunk["text"].split()),
                    },
                )
                for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]

            # Store in batches
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=settings.COLLECTION_NAME,
                    points=batch,
                )

        except Exception as e:
            raise CollectionError(f"Failed to store vectors: {e}") from e
