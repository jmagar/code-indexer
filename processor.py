#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich import print as rprint
from datetime import datetime, timezone
import re

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
from plugins.search.search import TreeSitterSearch, CombinedSearchPlugin
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import get_lexer_for_filename, TextLexer
from plugins.colors import NordStyle
from plugins.sources import SourceManager
from plugins.analysis.code_explainer import OpenAICodeExplainer

# Load environment variables
load_dotenv()

# Configure logging
logger = IndexerLogger(__name__).get_logger()
logger.setLevel("WARNING")  # Only show warning and above by default

# Suppress urllib3 warnings about insecure requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
            url="http://localhost:6550",
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30,
            verify=False  # Suppress SSL warnings for local connections
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

    async def setup_collection(self, force: bool = False):
        """Setup or reset Qdrant collection.
        
        Args:
            force: If True, recreate the collection even if it exists
        """
        try:
            logger.info("Setting up Qdrant collection")
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            exists = any(
                collection.name == COLLECTION_NAME
                for collection in collections.collections
            )

            if exists and force:
                logger.info(f"Recreating collection {COLLECTION_NAME}")
                self.qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
                exists = False

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
                self.qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="metadata.has_todo",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                self.qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="metadata.is_comment",
                    field_schema=models.PayloadSchemaType.BOOL,
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

        # Determine origin based on source type
        origin = ""
        if source == "local":
            # Use the root directory as origin for local files
            origin = str(Path(filepath).resolve().parent)

        # Estimate tokens (rough approximation: 4 chars per token)
        MAX_TOKENS = 2048  # 25% of OpenAI's 8192 limit for safety
        CHARS_PER_TOKEN = 4
        MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN

        # Track if we're in a comment block
        in_comment_block = False
        comment_chunk = []
        comment_start = 0

        for i, line in enumerate(lines):
            line = line.rstrip()  # Remove trailing whitespace
            if not line:  # Skip empty lines
                continue

            # Check for comment blocks and TODOs
            is_comment = bool(re.match(r'^\s*#|^\s*//|^\s*/\*|\*/', line))
            has_todo = 'todo' in line.lower()

            # If it's a comment or TODO, add to comment chunk
            if is_comment or has_todo:
                if not comment_chunk:
                    comment_start = i
                comment_chunk.append(line)
                if has_todo:  # Force create a chunk for TODOs
                    chunk_text = "\n".join(comment_chunk)
                    chunk = {
                        "text": chunk_text,
                        "metadata": {
                            "filepath": filepath,
                            "start_line": comment_start,
                            "end_line": i + 1,
                            "source": source,
                            "origin": origin,
                            "file_type": Path(filepath).suffix[1:],
                            "is_comment": True,
                            "has_todo": has_todo
                        },
                    }
                    chunks.append(chunk)
                    comment_chunk = []
                continue

            # If we were collecting comments and now hit code, create a comment chunk
            if comment_chunk:
                chunk_text = "\n".join(comment_chunk)
                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        "filepath": filepath,
                        "start_line": comment_start,
                        "end_line": i,
                        "source": source,
                        "origin": origin,
                        "file_type": Path(filepath).suffix[1:],
                        "is_comment": True
                    },
                }
                chunks.append(chunk)
                comment_chunk = []

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
                            "end_line": i,
                            "source": source,
                            "origin": origin,
                            "file_type": Path(filepath).suffix[1:],
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
                                "origin": origin,
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

        # Add any remaining comments
        if comment_chunk:
            chunk_text = "\n".join(comment_chunk)
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "filepath": filepath,
                    "start_line": comment_start,
                    "end_line": len(lines),
                    "source": source,
                    "origin": origin,
                    "file_type": Path(filepath).suffix[1:],
                    "is_comment": True
                },
            }
            chunks.append(chunk)

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
                    "origin": origin,
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
            console = Console()
            console.print(f"[bold #81A1C1]Processing source:[/] [#88C0D0]{source.name}[/]")
            start_time = asyncio.get_event_loop().time()

            # Prepare the source
            console.print("[bold #8FBCBB]Preparing source...[/]")
            await source.prepare()

            # Get all files
            console.print("[bold #8FBCBB]Getting files...[/]")
            current_files = await source.get_files()
            current_file_paths = {str(f) for f in current_files}
            console.print(f"[#88C0D0]Found[/] [bold #A3BE8C]{len(current_files)}[/] [#88C0D0]files to process[/]")

            # Get existing file hashes
            existing_hashes = self._get_existing_file_hashes(source.name)
            console.print(f"[#88C0D0]Found[/] [bold #A3BE8C]{len(existing_hashes)}[/] [#88C0D0]existing files in index[/]")

            # Delete removed files from Qdrant
            removed_files = set(existing_hashes.keys()) - current_file_paths
            if removed_files:
                console.print(f"[#88C0D0]Removing[/] [bold #BF616A]{len(removed_files)}[/] [#88C0D0]deleted files from index[/]")
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

            # Progress bar for file processing
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="#88C0D0", finished_style="#A3BE8C"),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                process_task = progress.add_task("[bold #81A1C1]Processing files...", total=len(current_files))
                
                # Check which files need processing
                for file in current_files:
                    try:
                        file_path_str = str(file)
                        current_hash = self._compute_file_hash(file)

                        if file_path_str in existing_hashes:
                            if existing_hashes[file_path_str] == current_hash:
                                files_unchanged += 1
                                progress.advance(process_task)
                                continue
                            else:
                                # Delete old version of changed file
                                console.print(f"[#88C0D0]File changed:[/] [italic #81A1C1]{file_path_str}[/]")
                                self.qdrant_client.delete(
                                    collection_name=COLLECTION_NAME,
                                    points_selector=models.Filter(
                                        must=[
                                            models.FieldCondition(
                                                key="metadata.filepath",
                                                match=models.MatchValue(value=file_path_str),
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
                        console.print(f"[#88C0D0]New file:[/] [italic #81A1C1]{file_path_str}[/]")
                        progress.advance(process_task)

                    except Exception as e:
                        console.print(f"[bold #BF616A]Error processing file {file}:[/] [italic #BF616A]{e}[/]")
                        progress.advance(process_task)
                        continue

            console.print(f"[#88C0D0]Files unchanged:[/] [bold #A3BE8C]{files_unchanged}[/]")
            console.print(f"[#88C0D0]Files to process:[/] [bold #A3BE8C]{len(files_to_process)}[/]")

            if not files_to_process:
                console.print("[bold #A3BE8C]No files need processing[/]")
                return

            # Get embeddings in batches
            console.print("[bold #8FBCBB]Generating embeddings...[/]")
            texts = [chunk["text"] for chunk in files_to_process]
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="#88C0D0", finished_style="#A3BE8C"),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                embedding_task = progress.add_task("[bold #81A1C1]Generating embeddings...", total=len(texts))
                embeddings = []
                batch_size = 50
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = await self.get_embeddings(batch)
                    embeddings.extend(batch_embeddings)
                    progress.advance(embedding_task, len(batch))

            console.print(f"[#88C0D0]Generated[/] [bold #A3BE8C]{len(embeddings)}[/] [#88C0D0]embeddings[/]")

            # Store in Qdrant with progress bar
            console.print("[bold #8FBCBB]Storing vectors in Qdrant...[/]")
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
                for idx, (chunk, embedding) in enumerate(zip(files_to_process, embeddings))
            ]

            batch_size = 100
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="#88C0D0", finished_style="#A3BE8C"),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                store_task = progress.add_task("[bold #81A1C1]Storing vectors...", total=len(points))
                
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)
                    progress.advance(store_task, len(batch))

            # Clean up
            try:
                await source.cleanup()
            except Exception as e:
                console.print(f"[bold #BF616A]Error during cleanup:[/] [italic #BF616A]{e}[/]")

            # Log statistics
            total_time = asyncio.get_event_loop().time() - start_time
            console.print(f"\n[bold #A3BE8C]Processing completed![/]")
            console.print(f"[#88C0D0]Total time:[/] [bold #81A1C1]{total_time:.2f}[/] [#88C0D0]seconds[/]")
            console.print(f"[#88C0D0]Total vectors:[/] [bold #81A1C1]{len(points)}[/]")
            console.print(f"[#88C0D0]Vectors per second:[/] [bold #81A1C1]{len(points) / total_time:.1f}[/]")

        except Exception as e:
            console.print(f"[bold #BF616A]Error processing source {source.name}:[/] [italic #BF616A]{e}[/]")
            # Try to clean up even if processing failed
            try:
                await source.cleanup()
            except Exception as e:
                console.print(f"[bold #BF616A]Error during cleanup after failure:[/] [italic #BF616A]{e}[/]")
            raise

    async def search(
        self,
        query: str,
        *,
        filter_paths: Optional[List[str]] = None,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> None:
        """Search for code using combined semantic and syntax-aware search."""
        try:
            console = Console()
            console.print(f"\n[bold #81A1C1]Searching for:[/] [italic #88C0D0]{query}[/]")
            if filter_paths:
                console.print(f"[#88C0D0]Filtering by paths:[/] [italic #A3BE8C]{', '.join(filter_paths)}[/]")
            
            # Initialize search plugin (reuse existing Qdrant client)
            search_plugin = CombinedSearchPlugin(self.qdrant_client)
            await search_plugin.setup()
            
            # Get search results
            results = await search_plugin.search(
                query,
                filter_paths=filter_paths,
                min_score=min_score,
                limit=limit
            )
            
            # Print results
            if not results:
                console.print("\n[bold #BF616A]No matching code found.[/]")
                return
            
            console.print(f"\n[bold #A3BE8C]Found {len(results)} matches:[/]")
                
            # Initialize code explainer only if needed (not for TODO searches)
            code_explainer = None
            if 'todo' not in query.lower():
                code_explainer = OpenAICodeExplainer()
                
            for i, result in enumerate(results, 1):
                # Print separator
                console.print(f"\n[#616E88]{'─' * 80}[/]")
                
                # Print match header with score color based on value
                score = result['score']
                if score >= 0.9:
                    score_color = "#A3BE8C"  # nord14 - green for high scores
                elif score >= 0.8:
                    score_color = "#EBCB8B"  # nord13 - yellow for medium scores
                else:
                    score_color = "#D08770"  # nord12 - orange for lower scores
                    
                console.print(f"[bold #88C0D0]Match {i}[/] (score: [bold {score_color}]{score:.2f}[/])")
                
                # Print file info
                filepath = result['filepath']
                filename = Path(filepath).name
                directory = str(Path(filepath).parent)
                
                # Get and display origin source
                metadata = result.get('metadata', {})
                if metadata.get('source') == 'github':
                    repo = metadata.get('repo', {})
                    if repo:
                        console.print(f"[#88C0D0]Source:[/] [bold #81A1C1]GitHub[/] ([italic #616E88]https://{repo['url']}[/])")
                        console.print(f"[#88C0D0]Branch:[/] [italic #616E88]{repo['branch']}[/] [#88C0D0]Commit:[/] [italic #616E88]{repo['commit'][:8]}[/]")
                elif metadata.get('origin'):
                    console.print(f"[#88C0D0]Source:[/] [bold #81A1C1]Local[/] ([italic #616E88]{metadata['origin']}[/])")
                
                console.print(f"[#88C0D0]File:[/] [bold #81A1C1]{filename}[/] [italic #616E88]in {directory}[/]")
                
                # Print separator before code
                console.print(f"[#616E88]{'─' * 80}[/]")
                
                # Add syntax highlighting
                try:
                    lexer = get_lexer_for_filename(result['filepath'])
                except Exception:
                    lexer = TextLexer()
                    
                highlighted_code = highlight(
                    result['code'],
                    lexer,
                    Terminal256Formatter(style=NordStyle)
                )
                print(highlighted_code, end='')
                
                # Only generate code explanation for non-TODO searches
                if code_explainer:
                    # Generate and print code explanation
                    console.print(f"\n[#616E88]{'─' * 80}[/]")
                    console.print("[bold #88C0D0]Code Explanation:[/]")
                    
                    # Try to explain as a function first, fall back to snippet explanation
                    code = result['code']
                    function_match = re.search(r'\b(?:def|function|fn|func)\s+(\w+)', code)
                    
                    if function_match:
                        # It's a function, use function explanation
                        function_name = function_match.group(1)
                        explanation = await code_explainer.explain_function(code, function_name)
                    else:
                        # Use general snippet explanation
                        explanation = await code_explainer.explain_snippet(
                            code,
                            context=f"This code was found while searching for: {query}"
                        )
                    
                    # Print explanation with proper formatting
                    for line in explanation.split('\n'):
                        if line.strip():
                            console.print(f"[#D8DEE9]{line}[/]")
            
            # Clean up
            await search_plugin.cleanup()
                
        except Exception as e:
            console.print(f"[bold #BF616A]Search failed:[/] [italic #BF616A]{e}[/]")
            logger.error(f"Search failed: {e}")
            print("Please try again.")

    async def ingest(self, source: CodeSourcePlugin) -> None:
        """Ingest code from a source."""
        try:
            await self.process_source(source)
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise


def print_help(console: Console):
    """Print formatted help message with available commands."""
    console.print("\n[bold #88C0D0]Code Indexer[/] - Semantic code search and analysis\n")
    
    console.print("[bold #81A1C1]Available Commands:[/]")
    
    # Search commands
    console.print("\n[bold #8FBCBB]Search[/]")
    console.print("  [#88C0D0]index search[/] [italic #616E88]<query>[/]     Search code semantically")
    console.print("  [#D8DEE9]Options:[/]")
    console.print("    [#616E88]--paths[/] [italic]<paths>[/]     Filter by file paths")
    console.print("    [#616E88]--min-score[/] [italic]<float>[/] Minimum similarity (default: 0.7)")
    console.print("    [#616E88]--limit[/] [italic]<int>[/]       Maximum results (default: 5)")
    
    # Source management
    console.print("\n[bold #8FBCBB]Source Management[/]")
    console.print("  [#88C0D0]index sources list[/]           List all indexed sources")
    console.print("  [#88C0D0]index sources add[/] [italic #616E88]<type> <path>[/]    Add a source")
    console.print("    [#D8DEE9]Types:[/]")
    console.print("      [#616E88]github[/] [italic]<url>[/]        GitHub repository")
    console.print("      [#616E88]local[/] [italic]<path>[/]        Local directory")
    console.print("  [#88C0D0]index sources remove[/] [italic #616E88]<id>[/]     Remove a source")
    console.print("  [#88C0D0]index sources reingest[/] [italic #616E88][id][/]   Reingest source(s)")
    
    # Ingestion commands
    console.print("\n[bold #8FBCBB]Direct Ingestion[/]")
    console.print("  [#88C0D0]index ingest github[/] [italic #616E88]<url>[/]    Ingest GitHub repository")
    console.print("  [#88C0D0]index ingest local[/] [italic #616E88]<path>[/]    Ingest local directory")
    
    # Watch commands
    console.print("\n[bold #8FBCBB]Repository Watching[/]")
    console.print("  [#88C0D0]index watch list[/]              List watched repositories")
    console.print("  [#88C0D0]index watch add[/] [italic #616E88]<url>[/]        Start watching repository")
    console.print("  [#88C0D0]index watch remove[/] [italic #616E88]<name>[/]    Stop watching repository")
    console.print("  [#88C0D0]index watch serve[/]             Start watch server")
    
    # Other commands
    console.print("\n[bold #8FBCBB]Other[/]")
    console.print("  [#88C0D0]index test[/] [italic #616E88]--embedding <provider>[/]  Test embedding provider")
    
    console.print("\n[bold #81A1C1]Examples:[/]")
    console.print("[#616E88]  # Search for code[/]")
    console.print("  [#88C0D0]index search[/] [italic #A3BE8C]\"function to handle database connections\"[/]")
    console.print("")
    console.print("[#616E88]  # Add and ingest a GitHub repository[/]")
    console.print("  [#88C0D0]index sources add github[/] [italic #A3BE8C]https://github.com/user/repo[/]")
    console.print("  [#88C0D0]index sources reingest[/] [italic #A3BE8C]github:user/repo[/]")
    console.print("")
    console.print("[#616E88]  # Watch a repository for changes[/]")
    console.print("  [#88C0D0]index watch add[/] [italic #A3BE8C]https://github.com/user/repo[/]")
    console.print("  [#88C0D0]index watch serve[/]")
    console.print("")

def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process, index, and search code using vector embeddings",
        usage="%(prog)s [-h] [--embedding {openai,lmstudio}] <command> [options]"
    )
    
    # Global options
    parser.add_argument(
        "--embedding",
        choices=["openai", "lmstudio"],
        default="openai",
        help="Embedding provider to use (default: openai)",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Collection management
    collection_parser = subparsers.add_parser("collection", help="Manage collections")
    collection_subparsers = collection_parser.add_subparsers(dest="collection_command", required=True)
    
    # Recreate collection
    collection_subparsers.add_parser(
        "recreate",
        help="Recreate the collection with updated schema (WARNING: this will delete all indexed data)"
    )

    # Source management commands
    sources_parser = subparsers.add_parser("sources", help="Manage code sources")
    sources_subparsers = sources_parser.add_subparsers(dest="source_command", required=True)
    
    # List sources
    sources_subparsers.add_parser("list", help="List all indexed sources")
    
    # Add source
    add_source = sources_subparsers.add_parser("add", help="Add a new source")
    add_source.add_argument("type", choices=["github", "local"], help="Source type")
    add_source.add_argument("path", help="Repository URL or local path")
    add_source.add_argument("--branch", help="Git branch (for GitHub sources)")
    add_source.add_argument("--exclude", help="Patterns to exclude (comma-separated)")
    add_source.add_argument("--include", help="Patterns to include (comma-separated)")
    
    # Remove source
    remove_source = sources_subparsers.add_parser("remove", help="Remove a source")
    remove_source.add_argument("id", help="Source ID (e.g., github:user/repo)")
    
    # Reingest source(s)
    reingest = sources_subparsers.add_parser("reingest", help="Reingest source(s)")
    reingest.add_argument("id", nargs="?", help="Source ID (optional, reingest all if omitted)")

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

    # Direct ingestion command (legacy support)
    ingest_parser = subparsers.add_parser("ingest", help="Directly ingest code (legacy)")
    ingest_subparsers = ingest_parser.add_subparsers(dest="source", required=True)

    # GitHub source
    github_parser = ingest_subparsers.add_parser("github", help="Ingest from GitHub repository")
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
    watch_remove = watch_subparsers.add_parser("remove", help="Stop watching a repository")
    watch_remove.add_argument(
        "repo_name", help="Name of the repository to stop watching (e.g., 'owner/repo')"
    )

    # Start watching server
    watch_serve = watch_subparsers.add_parser("serve", help="Start the repository watcher server")
    watch_serve.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    watch_serve.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test embedding provider")
    test_parser.add_argument(
        "--embedding",
        choices=["openai", "lmstudio"],
        default="lmstudio",
        help="Embedding provider to test",
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
    
    # If no arguments, print help
    if len(sys.argv) == 1:
        console = Console()
        print_help(console)
        return
        
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
        elif args.command == "sources":
            source_manager = SourceManager()
            await source_manager.setup_collection()
            
            if args.source_command == "list":
                console = Console()
                await source_manager.list_sources(console)
                
            elif args.source_command == "add":
                # Split patterns into lists if provided
                exclude_patterns = args.exclude.split(",") if args.exclude else None
                include_patterns = args.include.split(",") if args.include else None
                
                source_id = await source_manager.add_source(
                    args.type,
                    args.path,
                    branch=args.branch,
                    exclude_patterns=exclude_patterns,
                    include_patterns=include_patterns,
                )
                console = Console()
                console.print(f"[bold #A3BE8C]Added source:[/] [#88C0D0]{source_id}[/]")
                
            elif args.source_command == "remove":
                success = await source_manager.remove_source(args.id)
                console = Console()
                if success:
                    console.print(f"[bold #A3BE8C]Removed source:[/] [#88C0D0]{args.id}[/]")
                else:
                    console.print(f"[bold #BF616A]Source not found:[/] [#88C0D0]{args.id}[/]")
                    
            elif args.source_command == "reingest":
                console = Console()
                if args.id:
                    # Reingest specific source
                    source = await source_manager.get_source(args.id)
                    if not source:
                        console.print(f"[bold #BF616A]Source not found:[/] [#88C0D0]{args.id}[/]")
                        return
                        
                    console.print(f"[bold #8FBCBB]Reingesting source:[/] [#88C0D0]{args.id}[/]")
                    if source["type"] == "github":
                        plugin = GitHubPlugin(source["url"], branch=source.get("branch"))
                    else:
                        plugin = LocalCodePlugin([Path(source["path"])])
                        
                    await processor.ingest(plugin)
                    
                    # Update last ingestion time
                    await source_manager.update_source(args.id, {
                        "last_ingested": datetime.now(timezone.utc).isoformat()
                    })
                    
                else:
                    # Reingest all sources
                    sources = await source_manager.list_sources()
                    if not sources:
                        console.print("[bold #BF616A]No sources found to reingest[/]")
                        return
                        
                    console.print(f"[bold #8FBCBB]Reingesting all sources[/] ([#88C0D0]{len(sources)}[/] total)")
                    for source in sources:
                        source_id = source["id"]
                        console.print(f"\n[bold #81A1C1]Processing:[/] [#88C0D0]{source_id}[/]")
                        
                        if source["type"] == "github":
                            plugin = GitHubPlugin(source["url"], branch=source.get("branch"))
                        else:
                            plugin = LocalCodePlugin([Path(source["path"])])
                            
                        try:
                            await processor.ingest(plugin)
                            # Update last ingestion time
                            await source_manager.update_source(source_id, {
                                "last_ingested": datetime.now(timezone.utc).isoformat()
                            })
                            console.print(f"[bold #A3BE8C]Successfully reingested:[/] [#88C0D0]{source_id}[/]")
                        except Exception as e:
                            console.print(f"[bold #BF616A]Failed to reingest[/] [#88C0D0]{source_id}[/]:")
                            console.print(f"[italic #BF616A]{str(e)}[/]")
                            continue
                    
                    console.print("\n[bold #A3BE8C]Reingestion complete![/]")
                
        elif args.command == "ingest":
            source_manager = SourceManager()
            await source_manager.setup_collection()
            
            if args.source == "github":
                # Add GitHub source before ingesting
                source_id = await source_manager.add_source("github", args.url)
                console = Console()
                console.print(f"[bold #A3BE8C]Added source:[/] [#88C0D0]{source_id}[/]")
                
                plugin = GitHubPlugin(args.url)
                await processor.ingest(plugin)
                
                # Update last ingestion time
                await source_manager.update_source(source_id, {
                    "last_ingested": datetime.now(timezone.utc).isoformat()
                })

                # Set up watching if requested
                if args.watch:
                    from plugins.github_watcher import GitHubWatcher

                    watcher = GitHubWatcher(processor)
                    await watcher.watch_repository(args.url)
                    logger.info(f"Now watching repository: {args.url}")
            else:  # local
                # Add local source before ingesting
                for path in args.paths:
                    source_id = await source_manager.add_source("local", str(path))
                    console = Console()
                    console.print(f"[bold #A3BE8C]Added source:[/] [#88C0D0]{source_id}[/]")
                
                plugin = LocalCodePlugin(args.paths)
                await processor.ingest(plugin)
                
                # Update last ingestion time for all paths
                for path in args.paths:
                    source_id = f"local:{os.path.abspath(str(path))}"
                    await source_manager.update_source(source_id, {
                        "last_ingested": datetime.now(timezone.utc).isoformat()
                    })

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
