"""Code chunking functionality for processing source files."""

from pathlib import Path
from typing import List, TypedDict
import re

from utils.config import get_settings

settings = get_settings()


class ChunkMetadata(TypedDict):
    """Type definition for chunk metadata."""

    filepath: str
    start_line: int
    end_line: int
    source: str
    origin: str
    file_type: str
    is_comment: bool
    has_todo: bool
    file_hash: str  # Added to support file hash tracking


class Chunk(TypedDict):
    """Type definition for a code chunk."""

    text: str
    metadata: ChunkMetadata


def create_chunks(content: str, filepath: str, source: str) -> List[Chunk]:
    """Split content into chunks with metadata.

    Args:
        content: The file content to chunk
        filepath: Path to the source file
        source: Source identifier (e.g., 'github', 'local')

    Returns:
        List of chunks with metadata
    """
    if not content.strip():
        return []

    lines = content.split("\n")
    chunks: List[Chunk] = []
    current_chunk: List[str] = []
    current_size = 0
    start_line = 0

    # Determine origin based on source type
    origin = ""
    if source == "local":
        # Use the root directory as origin for local files
        origin = str(Path(filepath).resolve().parent)

    # Track comment blocks
    comment_chunk: List[str] = []
    comment_start = 0

    for i, line in enumerate(lines):
        line = line.rstrip()  # Remove trailing whitespace
        if not line:  # Skip empty lines
            continue

        # Check for comment blocks and TODOs
        is_comment = bool(re.match(r"^\s*#|^\s*//|^\s*/\*|\*/", line))
        has_todo = "todo" in line.lower()

        # If it's a comment or TODO, add to comment chunk
        if is_comment or has_todo:
            if not comment_chunk:
                comment_start = i
            comment_chunk.append(line)
            if has_todo:  # Force create a chunk for TODOs
                chunk_text = "\n".join(comment_chunk)
                chunk: Chunk = {
                    "text": chunk_text,
                    "metadata": {
                        "filepath": filepath,
                        "start_line": comment_start,
                        "end_line": i + 1,
                        "source": source,
                        "origin": origin,
                        "file_type": Path(filepath).suffix[1:],
                        "is_comment": True,
                        "has_todo": has_todo,
                        "file_hash": "",  # Will be set by the caller
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
                    "is_comment": True,
                    "has_todo": False,
                    "file_hash": "",  # Will be set by the caller
                },
            }
            chunks.append(chunk)
            comment_chunk = []

        # Check if adding this line would exceed the token limit
        potential_chunk = "\n".join(current_chunk + [line])
        max_chars = settings.MAX_TOKENS * settings.CHARS_PER_TOKEN
        if len(potential_chunk) > max_chars:
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
                        "is_comment": False,
                        "has_todo": False,
                        "file_hash": "",  # Will be set by the caller
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
                    chunk_text = line[:max_chars]
                    chunk = {
                        "text": chunk_text,
                        "metadata": {
                            "filepath": filepath,
                            "start_line": i,
                            "end_line": i + 1,
                            "source": source,
                            "origin": origin,
                            "file_type": Path(filepath).suffix[1:],
                            "is_comment": False,
                            "has_todo": False,
                            "file_hash": "",  # Will be set by the caller
                        },
                    }
                    chunks.append(chunk)
                    line = line[max_chars:]
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
                "is_comment": True,
                "has_todo": False,
                "file_hash": "",  # Will be set by the caller
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
                "is_comment": False,
                "has_todo": False,
                "file_hash": "",  # Will be set by the caller
            },
        }
        chunks.append(chunk)

    return chunks
