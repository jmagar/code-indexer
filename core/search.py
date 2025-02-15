"""Search functionality for code indexing."""

import logging
from typing import Dict, List, Optional, TypedDict, Any

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import get_lexer_for_filename, TextLexer
from rich.console import Console
from qdrant_client import QdrantClient

from plugins.colors import NordStyle
from plugins.search.search import CombinedSearchPlugin
from utils.config import get_settings
from utils.console import (
    print_header,
    print_info,
    print_error,
    print_separator,
    format_score,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class SearchError(Exception):
    """Error related to search operations."""

    pass


class SearchResult(TypedDict):
    """Type definition for search results."""

    score: float
    filepath: str
    code: str
    start_line: int
    end_line: int
    source: str
    metadata: Dict[str, Any]


class SearchManager:
    """Manages code search operations."""

    def __init__(self, qdrant_client: QdrantClient) -> None:
        """Initialize search manager.

        Args:
            qdrant_client: Qdrant client instance
        """
        self.search_plugin: CombinedSearchPlugin = CombinedSearchPlugin(qdrant_client)

    async def setup(self) -> None:
        """Set up search plugin.

        Raises:
            SearchError: If setup fails
        """
        try:
            await self.search_plugin.setup()
        except Exception as e:
            raise SearchError(f"Failed to setup search: {e}") from e

    async def search(
        self,
        query: str,
        *,
        filter_paths: Optional[List[str]] = None,
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
            SearchError: If search fails
        """
        try:
            print_header(console, f"Searching for: {query}")
            if filter_paths:
                print_info(console, f"Filtering by paths: {', '.join(filter_paths)}")

            # Get search results
            results = await self.search_plugin.search(
                query,
                filter_paths=filter_paths,
                min_score=min_score,
                limit=limit,
            )

            if not results:
                print_error(console, "No matching code found.")
                return

            print_info(console, f"Found {len(results)} matches:")

            for i, result in enumerate(results, 1):
                print_separator(console)

                # Print match header with score
                score = result["score"]
                print_info(
                    console,
                    f"Match {i} (score: {format_score(score)})",
                )

                # Print file info
                filepath = result["filepath"]

                # Display source info based on type
                metadata = result.get("metadata", {})
                if metadata.get("source") == "crawl4ai":
                    url = metadata.get("url", "unknown")
                    print_info(console, f"Source: Web ({url})")
                    print_info(console, f"File: {url}")
                elif metadata.get("source") == "github":
                    repo = metadata.get("repo", {})
                    if repo:
                        print_info(console, f"Source: GitHub (https://{repo['url']})")
                        print_info(
                            console,
                            f"Branch: {repo['branch']} Commit: {repo['commit'][:8]}",
                        )
                else:
                    # Local source
                    print_info(
                        console, f"Source: Local ({metadata.get('origin', 'unknown')})"
                    )
                    print_info(console, f"File: {result['filepath']}")

                print_separator(console)

                # Display the content from Qdrant with syntax highlighting
                try:
                    code = result["code"]
                    start_line = result.get("start_line", 0)
                    end_line = result.get("end_line", 0)

                    # Add line numbers and context
                    lines = code.split("\n")
                    numbered_lines = []

                    # Add a header line to show context
                    if start_line > 1:
                        numbered_lines.append("     ┄ (previous lines)")

                    # Add the actual code lines with numbers
                    for i, line in enumerate(lines, start=start_line):
                        # Add line numbers with proper padding
                        line_num = f"{i:4d}"
                        if i == start_line and i == end_line:
                            numbered_lines.append(f"{line_num} ─ {line}")
                        elif i == start_line:
                            numbered_lines.append(f"{line_num} ┌ {line}")
                        elif i == end_line:
                            numbered_lines.append(f"{line_num} └ {line}")
                        else:
                            numbered_lines.append(f"{line_num} │ {line}")

                    # Add a footer line to show context
                    if end_line < 1000000:  # Arbitrary large number
                        numbered_lines.append("     ┄ (more lines)")

                    # Join with newlines and ensure it ends with one
                    code_with_numbers = "\n".join(numbered_lines) + "\n"

                    # Get lexer for syntax highlighting
                    lexer = (
                        TextLexer()
                        if metadata.get("source") == "crawl4ai"
                        else get_lexer_for_filename(filepath)
                    )

                    # Highlight and display
                    highlighted_code = highlight(
                        code_with_numbers,
                        lexer,
                        Terminal256Formatter(style=NordStyle),
                    )
                    print(highlighted_code, end="")

                    # Show line range
                    print_info(
                        console,
                        f"Lines {start_line}-{end_line}",
                    )
                except Exception as e:
                    logger.error(f"Error displaying result: {e}")
                    print(result["code"])

            # Clean up
            await self.search_plugin.cleanup()

        except Exception as e:
            # Try to clean up even if search failed
            try:
                await self.search_plugin.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup after failure: {cleanup_error}")
            raise SearchError(f"Search failed: {e}") from e
