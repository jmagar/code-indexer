"""Source processing functionality."""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import List

from rich.console import Console

from plugins import CodeSourcePlugin
from utils.console import (
    create_progress,
    print_header,
    print_info,
    print_success,
)
from core.chunking import create_chunks, Chunk
from core.collection import CollectionManager
from core.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base class for processing errors."""

    pass


class SourceProcessor:
    """Processes code from various sources."""

    def __init__(
        self,
        collection_manager: CollectionManager,
        embedding_manager: EmbeddingManager,
    ) -> None:
        """Initialize source processor.

        Args:
            collection_manager: Collection manager instance
            embedding_manager: Embedding manager instance
        """
        self.collection = collection_manager
        self.embeddings = embedding_manager

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents.

        Args:
            file_path: Path to file to hash

        Returns:
            SHA-256 hash of file contents

        Raises:
            ProcessingError: If file cannot be read
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            raise ProcessingError(f"Failed to compute hash for {file_path}: {e}") from e

    async def process_source(self, source: CodeSourcePlugin, console: Console) -> None:
        """Process code from a source plugin.

        Args:
            source: Source plugin instance
            console: Console for output

        Raises:
            ProcessingError: If processing fails
        """
        try:
            print_header(console, f"Processing source: {source.name}")
            start_time = asyncio.get_event_loop().time()

            # Prepare the source
            print_info(console, "Preparing source...")
            await source.prepare()

            # Get all files
            print_info(console, "Getting files...")
            current_files = await source.get_files()
            current_file_paths = {str(f) for f in current_files}
            print_info(
                console,
                f"Found {len(current_files)} files to process",
            )

            # Get existing file hashes
            existing_hashes = self.collection.get_file_hashes(source.name)
            print_info(
                console,
                f"Found {len(existing_hashes)} existing files in index",
            )

            # Delete removed files
            removed_files = set(existing_hashes.keys()) - current_file_paths
            if removed_files:
                print_info(
                    console,
                    f"Removing {len(removed_files)} deleted files from index",
                )
                self.collection.delete_files(source.name, list(removed_files))

            # Track which files to process
            files_to_process: List[Chunk] = []
            files_unchanged = 0
            total_lines = 0
            processed_lines = 0

            # Progress bar for file processing
            with create_progress() as progress:
                process_task = progress.add_task(
                    "Processing files...", total=len(current_files)
                )

                # First pass - count total lines
                for file in current_files:
                    try:
                        with open(file, "r", encoding="utf-8") as f:
                            total_lines += sum(1 for _ in f)
                    except Exception as e:
                        logger.error(f"Error counting lines in {file}: {e}")
                        continue

                print_info(console, f"Total lines of code: {total_lines:,}")

                # Second pass - process files
                for file in current_files:
                    try:
                        file_path_str = str(file)
                        current_hash = self._compute_file_hash(file)

                        if file_path_str in existing_hashes:
                            if existing_hashes[file_path_str] == current_hash:
                                # Count lines in unchanged files
                                try:
                                    with open(file, "r", encoding="utf-8") as f:
                                        processed_lines += sum(1 for _ in f)
                                except Exception:
                                    pass
                                files_unchanged += 1
                                progress.advance(process_task)
                                continue

                        # Add file to processing list
                        with open(file, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Count lines in new/changed files
                            processed_lines += len(content.splitlines())

                        # Create chunks with file hash
                        chunks = create_chunks(content, file_path_str, source.name)
                        for chunk in chunks:
                            # Update metadata with file hash
                            chunk["metadata"]["file_hash"] = current_hash

                        files_to_process.extend(chunks)
                        print_info(
                            console,
                            f"New file: {file_path_str}",
                        )
                        progress.advance(process_task)

                    except Exception as e:
                        logger.error(f"Error processing file {file}: {e}")
                        progress.advance(process_task)
                        continue

            print_info(console, f"Files unchanged: {files_unchanged}")
            print_info(console, f"Files to process: {len(files_to_process)}")
            print_info(
                console, f"Lines processed: {processed_lines:,} of {total_lines:,}"
            )

            if not files_to_process:
                print_success(console, "No files need processing")
                return

            # Get embeddings in batches
            print_info(console, "Generating embeddings...")
            texts = [chunk["text"] for chunk in files_to_process]

            with create_progress() as progress:
                embedding_task = progress.add_task(
                    "Generating embeddings...", total=len(texts)
                )
                embeddings = await self.embeddings.get_embeddings(texts)
                progress.advance(embedding_task, len(texts))

            print_info(console, f"Generated {len(embeddings)} embeddings")

            # Store vectors
            print_info(console, "Storing vectors in Qdrant...")
            # Convert chunks to Dict[str, Any] for storage
            chunks_for_storage = [
                {"text": chunk["text"], "metadata": chunk["metadata"]}
                for chunk in files_to_process
            ]
            self.collection.store_vectors(embeddings, chunks_for_storage)

            # Clean up
            try:
                await source.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

            # Log statistics
            total_time = asyncio.get_event_loop().time() - start_time
            print_info(console, f"Total time: {total_time:.2f} seconds")
            print_info(console, f"Total vectors: {len(embeddings)}")
            print_info(
                console, f"Vectors per second: {len(embeddings) / total_time:.1f}"
            )

        except Exception as e:
            # Try to clean up even if processing failed
            try:
                await source.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup after failure: {cleanup_error}")
            raise ProcessingError(f"Failed to process source {source.name}: {e}") from e
