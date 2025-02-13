#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add the current directory to Python path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from dotenv import load_dotenv
from plugins import (
    CodeSourcePlugin,
    GitHubPlugin,
    IndexerLogger,
    LocalCodePlugin,
    QdrantSearchPlugin,
)
from plugins.embeddings import LMStudioEmbedding, OpenAIEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Configure logging
logger = IndexerLogger(__name__).get_logger()

# Configuration
COLLECTION_NAME = "code_reference"
CHUNK_SIZE = 750  # Reduced from 1500 to stay within LM Studio's context length
CHUNK_OVERLAP = 100  # Reduced from 200 to maintain ratio


class CodeProcessor:
    """Process and index code from various sources."""

    def __init__(self, embedding_provider: str = "openai"):
        """Initialize processor.

        Args:
            embedding_provider: Name of embedding provider to use ('openai' or 'lmstudio')
        """
        # Initialize clients
        self.qdrant_client = QdrantClient(
            url="http://localhost:6335",
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30,
        )

        # Set up embedding provider with fallback
        self.primary_provider = embedding_provider
        if embedding_provider == "openai":
            self.embedding = OpenAIEmbedding()
            self.fallback_embedding = None  # No fallback needed for OpenAI
        elif embedding_provider == "lmstudio":
            self.embedding = LMStudioEmbedding()
            # Create fallback OpenAI embedder
            try:
                self.fallback_embedding = OpenAIEmbedding()
            except ValueError:
                self.fallback_embedding = None
                logger.warning("No OpenAI fallback available - missing API key")
        else:
            raise ValueError(f"Unknown embedding provider: {embedding_provider}")

        logger.info(f"Using {self.embedding.name} embedding provider")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_existing_file_hashes(self, source_name: str) -> Dict[str, str]:
        """Get existing file hashes from Qdrant for the given source."""
        try:
            # Search for all points from this source
            file_hashes = {}
            offset = 0
            batch_size = 100

            while True:
                response = self.qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=source_name),
                            )
                        ]
                    ),
                    offset=offset,
                    limit=batch_size,
                    with_payload=True,
                )

                points, next_offset = response
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

                if next_offset is None:
                    break
                offset = next_offset

            return file_hashes
        except Exception as e:
            logger.error(f"Error getting existing file hashes: {e}")
            return {}

    async def setup_collection(self):
        """Setup or reset Qdrant collection."""
        try:
            logger.info("Setting up Qdrant collection")
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            exists = any(
                collection.name == COLLECTION_NAME
                for collection in collections.collections
            )

            if not exists:
                # Create collection with correct vector size for embeddings
                logger.info(f"Creating collection {COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.embedding.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )

                # Create payload indexes
                logger.info("Creating payload indexes")
                self.qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="metadata.filepath",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="metadata.source",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="metadata.file_type",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info("Successfully set up Qdrant collection")
            else:
                logger.info(f"Collection {COLLECTION_NAME} already exists")

        except Exception as e:
            logger.error(f"Error setting up Qdrant collection: {e}")
            raise

    def create_chunks(self, content: str, filepath: str, source: str) -> List[dict]:
        """Split content into chunks with metadata."""
        if not content.strip():
            return []

        lines = content.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0
        start_line = 0

        # Estimate tokens (rough approximation: 4 chars per token)
        MAX_TOKENS = 2048  # 25% of OpenAI's 8192 limit for safety
        CHARS_PER_TOKEN = 4
        MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN

        for i, line in enumerate(lines):
            line = line.rstrip()  # Remove trailing whitespace
            if not line:  # Skip empty lines
                continue

            # Check if adding this line would exceed the token limit
            potential_chunk = "\n".join(current_chunk + [line])
            if len(potential_chunk) > MAX_CHARS:
                if current_chunk:  # Only create chunk if we have content
                    chunk_text = "\n".join(current_chunk)
                    chunk = {
                        "text": chunk_text,
                        "metadata": {
                            "filepath": filepath,
                            "start_line": start_line,
                            "end_line": i,  # Current line number
                            "source": source,
                            "file_type": Path(filepath).suffix[1:],  # Remove the dot
                        },
                    }
                    chunks.append(chunk)

                    # Start new chunk with current line
                    current_chunk = [line]
                    current_size = len(line.split())
                    start_line = i
                else:
                    # Line itself is too long, need to split it
                    while line:
                        chunk_text = line[:MAX_CHARS]
                        chunk = {
                            "text": chunk_text,
                            "metadata": {
                                "filepath": filepath,
                                "start_line": i,
                                "end_line": i + 1,
                                "source": source,
                                "file_type": Path(filepath).suffix[1:],
                            },
                        }
                        chunks.append(chunk)
                        line = line[MAX_CHARS:]
                    start_line = i + 1
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk.append(line)
                current_size += len(line.split())

        # Add the remaining lines as the last chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "filepath": filepath,
                    "start_line": start_line,
                    "end_line": len(lines),
                    "source": source,
                    "file_type": Path(filepath).suffix[1:],
                },
            }
            chunks.append(chunk)

        return chunks

    async def get_embeddings(
        self, texts: List[str], batch_size: int = 50
    ) -> List[List[float]]:
        """Get embeddings for texts in batches."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"Processing embedding batch {batch_num}/{total_batches}")

            try:
                # Clean and validate input texts
                cleaned_batch = [
                    text.strip()
                    for text in batch
                    if text and isinstance(text, str) and text.strip()
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
                    if hasattr(self, "fallback_embedding") and self.fallback_embedding:
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

    async def process_source(self, source: CodeSourcePlugin) -> None:
        """Process code from a source plugin."""
        try:
            logger.info(f"Processing source: {source.name}")
            start_time = asyncio.get_event_loop().time()

            # Prepare the source
            logger.info("Preparing source...")
            await source.prepare()

            # Get all files
            logger.info("Getting files...")
            current_files = await source.get_files()
            current_file_paths = {str(f) for f in current_files}
            logger.info(f"Found {len(current_files)} files to process")

            # Get existing file hashes
            existing_hashes = self._get_existing_file_hashes(source.name)
            logger.info(f"Found {len(existing_hashes)} existing files in index")

            # Delete removed files from Qdrant
            removed_files = set(existing_hashes.keys()) - current_file_paths
            if removed_files:
                logger.info(f"Removing {len(removed_files)} deleted files from index")
                for filepath in removed_files:
                    self.qdrant_client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.filepath",
                                    match=models.MatchValue(value=filepath),
                                ),
                                models.FieldCondition(
                                    key="metadata.source",
                                    match=models.MatchValue(value=source.name),
                                ),
                            ]
                        ),
                    )

            # Track which files to process
            files_to_process = []
            files_unchanged = 0

            # Check which files need processing
            for file in current_files:
                try:
                    file_path_str = str(file)
                    current_hash = self._compute_file_hash(file)

                    if file_path_str in existing_hashes:
                        if existing_hashes[file_path_str] == current_hash:
                            files_unchanged += 1
                            continue
                        else:
                            # Delete old version of changed file
                            logger.info(f"File changed: {file_path_str}")
                            self.qdrant_client.delete(
                                collection_name=COLLECTION_NAME,
                                points_selector=models.Filter(
                                    must=[
                                        models.FieldCondition(
                                            key="metadata.filepath",
                                            match=models.MatchValue(
                                                value=file_path_str
                                            ),
                                        ),
                                        models.FieldCondition(
                                            key="metadata.source",
                                            match=models.MatchValue(value=source.name),
                                        ),
                                    ]
                                ),
                            )

                    # Add file to processing list
                    with open(file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create chunks with file hash
                    chunks = self.create_chunks(content, file_path_str, source.name)
                    for chunk in chunks:
                        chunk["metadata"]["file_hash"] = current_hash

                    files_to_process.extend(chunks)
                    logger.info(f"New file: {file_path_str}")

                except Exception as e:
                    logger.error(f"Error processing file {file}: {e}")
                    continue

            logger.info(f"Files unchanged: {files_unchanged}")
            logger.info(f"Files to process: {len(files_to_process)}")

            if not files_to_process:
                logger.info("No files need processing")
                return

            # Get embeddings in batches
            logger.info("Generating embeddings...")
            texts = [chunk["text"] for chunk in files_to_process]
            embeddings = await self.get_embeddings(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Store in Qdrant
            logger.info("Storing vectors in Qdrant...")
            points = [
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "token_count": len(chunk["text"].split()),
                    },
                )
                for idx, (chunk, embedding) in enumerate(
                    zip(files_to_process, embeddings)
                )
            ]

            # Store in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(points) + batch_size - 1) // batch_size
                logger.info(
                    f"Storing batch {batch_num}/{total_batches} in Qdrant ({len(batch)} points)"
                )
                self.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)

            # Clean up
            try:
                await source.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

            # Log statistics
            total_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Completed processing {source.name}")
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Total vectors: {len(points)}")
            logger.info(f"Vectors per second: {len(points) / total_time:.1f}")

        except Exception as e:
            logger.error(f"Error processing source {source.name}: {e}")
            # Try to clean up even if processing failed
            try:
                await source.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup after failure: {e}")
            raise

    async def search(
        self,
        query: str,
        *,
        filter_paths: Optional[List[str]] = None,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> None:
        """Search for code using the query."""
        try:
            # Use QdrantSearchPlugin by default
            plugin = QdrantSearchPlugin()
            await plugin.setup()

            logger.info(f"Searching with query: {query}")
            if filter_paths:
                logger.info(f"Filtering by paths: {filter_paths}")

            results = await plugin.search(
                query,
                filter_paths=filter_paths,
                min_score=min_score,
                limit=limit,
            )

            # Display results
            if not results:
                print(f"\nNo results found for query: '{query}'")
                return

            print(f"\nFound {len(results)} results for query: '{query}'\n")
            for i, result in enumerate(results, 1):
                print(f"[{i}] {result['filepath']} (score: {result['score']:.3f})")
                print(f"Lines {result['start_line']}-{result['end_line']}")
                print(result["code"].rstrip())
                print("\n" + "=" * 80 + "\n")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

        finally:
            await plugin.cleanup()

    async def ingest(self, source: CodeSourcePlugin) -> None:
        """Ingest code from a source."""
        try:
            await self.process_source(source)
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process, index, and search code using vector embeddings"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test embedding provider")
    test_parser.add_argument(
        "--embedding",
        choices=["openai", "lmstudio"],
        default="lmstudio",
        help="Embedding provider to test",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed code")
    search_parser.add_argument("query", help="The search query")
    search_parser.add_argument(
        "--paths",
        nargs="*",
        help="Optional paths to filter results (e.g., 'executor' 'workflow')",
    )
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=0.7,
        help="Minimum similarity score (0-1), default: 0.7",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results, default: 5",
    )

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest code into vector store"
    )
    ingest_subparsers = ingest_parser.add_subparsers(dest="source", required=True)

    # GitHub source
    github_parser = ingest_subparsers.add_parser(
        "github", help="Ingest from GitHub repository"
    )
    github_parser.add_argument(
        "url",
        help="URL of the GitHub repository (e.g., https://github.com/user/repo)",
    )
    github_parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch repository for changes after ingesting",
    )

    # Local source
    local_parser = ingest_subparsers.add_parser("local", help="Ingest from local paths")
    local_parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Local paths to index (can specify multiple)",
    )

    # Watch commands
    watch_parser = subparsers.add_parser("watch", help="Manage repository watching")
    watch_subparsers = watch_parser.add_subparsers(dest="watch_command", required=True)

    # List watched repos
    watch_subparsers.add_parser("list", help="List currently watched repositories")

    # Add repo to watch
    watch_add = watch_subparsers.add_parser("add", help="Start watching a repository")
    watch_add.add_argument("url", help="URL of the GitHub repository to watch")

    # Remove repo from watch
    watch_remove = watch_subparsers.add_parser(
        "remove", help="Stop watching a repository"
    )
    watch_remove.add_argument(
        "repo_name", help="Name of the repository to stop watching (e.g., 'owner/repo')"
    )

    # Start watching server
    watch_serve = watch_subparsers.add_parser(
        "serve", help="Start the repository watcher server"
    )
    watch_serve.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    watch_serve.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )

    # Add embedding provider option
    parser.add_argument(
        "--embedding",
        choices=["openai", "lmstudio"],
        default="openai",
        help="Embedding provider to use (default: openai)",
    )

    return parser


async def test_embeddings(processor: CodeProcessor) -> None:
    """Test the embedding provider."""
    test_texts = [
        "This is a test sentence.",
        "Another test sentence to try.",
        "A third test sentence to verify batching.",
    ]
    try:
        print(f"Testing {processor.embedding.name} embeddings...")
        embeddings = await processor.get_embeddings(test_texts)
        print(f"Successfully generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
    except Exception as e:
        print(f"Error: {e}")
        raise


async def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    processor = CodeProcessor(embedding_provider=args.embedding)
    await processor.setup_collection()

    try:
        if args.command == "test":
            await test_embeddings(processor)
        elif args.command == "search":
            await processor.search(
                args.query,
                filter_paths=args.paths,
                min_score=args.min_score,
                limit=args.limit,
            )
        elif args.command == "ingest":
            if args.source == "github":
                plugin = GitHubPlugin(args.url)
                await processor.ingest(plugin)

                # Set up watching if requested
                if args.watch:
                    from plugins.github_watcher import GitHubWatcher

                    watcher = GitHubWatcher(processor)
                    await watcher.watch_repository(args.url)
                    logger.info(f"Now watching repository: {args.url}")
            else:  # local
                plugin = LocalCodePlugin(args.paths)
                await processor.ingest(plugin)

        elif args.command == "watch":
            from plugins.github_watcher import GitHubWatcher

            watcher = GitHubWatcher(processor)

            if args.watch_command == "list":
                repos = watcher.get_watched_repositories()
                if not repos:
                    print("No repositories are currently being watched.")
                else:
                    print("\nCurrently watched repositories:")
                    for repo in repos:
                        print(f"\nRepository: {repo['name']}")
                        print(f"URL: {repo['url']}")
                        print(f"Branch: {repo['default_branch']}")
                        print(f"Last Commit: {repo['last_commit'][:8]}")
                        print(f"Last Checked: {repo['last_checked']}")
                        if "last_error" in repo:
                            print(f"Last Error: {repo['last_error']}")
                            print(f"Error Time: {repo['last_error_time']}")

            elif args.watch_command == "add":
                success = await watcher.watch_repository(args.url)
                if success:
                    print(f"Successfully started watching: {args.url}")
                else:
                    print(f"Failed to start watching: {args.url}")

            elif args.watch_command == "remove":
                await watcher._remove_repository(args.repo_name)
                print(f"Stopped watching: {args.repo_name}")

            elif args.watch_command == "serve":
                print(f"Starting watcher server on {args.host}:{args.port}")
                await watcher.start(host=args.host, port=args.port)

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
