"""Command line argument parser setup."""

import argparse
from pathlib import Path

from utils.config import get_settings

settings = get_settings()


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process, index, and search code using vector embeddings",
        usage="%(prog)s [-h] [--embedding {openai,lmstudio}] <command> [options]",
    )

    # Global options
    parser.add_argument(
        "--embedding",
        choices=["openai", "lmstudio"],
        default=settings.DEFAULT_EMBEDDING_PROVIDER.value,
        help="Embedding provider to use (default: openai)",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Collection management
    collection_parser = subparsers.add_parser("collection", help="Manage collections")
    collection_subparsers = collection_parser.add_subparsers(
        dest="collection_command", required=True
    )

    # Recreate collection
    collection_subparsers.add_parser(
        "recreate",
        help="Recreate the collection with updated schema (WARNING: this will delete all indexed data)",
    )

    # Collection status
    collection_subparsers.add_parser(
        "status",
        help="Show collection status and statistics",
    )

    # Source management commands
    sources_parser = subparsers.add_parser("sources", help="Manage code sources")
    sources_subparsers = sources_parser.add_subparsers(
        dest="source_command", required=True
    )

    # List sources
    sources_subparsers.add_parser("list", help="List all indexed sources")

    # Add source
    add_source = sources_subparsers.add_parser("add", help="Add a new source")
    add_source.add_argument(
        "type", choices=["github", "local", "scrape"], help="Source type"
    )
    add_source.add_argument("path", help="Repository URL, local path, or web URL")
    add_source.add_argument("--branch", help="Git branch (for GitHub sources)")
    add_source.add_argument("--exclude", help="Patterns to exclude (comma-separated)")
    add_source.add_argument("--include", help="Patterns to include (comma-separated)")

    # Remove source
    remove_source = sources_subparsers.add_parser("remove", help="Remove a source")
    remove_source.add_argument("id", help="Source ID (e.g., github:user/repo)")

    # Reingest source(s)
    reingest = sources_subparsers.add_parser("reingest", help="Reingest source(s)")
    reingest.add_argument(
        "id", nargs="?", help="Source ID (optional, reingest all if omitted)"
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
        default=settings.DEFAULT_MIN_SCORE,
        help=f"Minimum similarity score (0-1), default: {settings.DEFAULT_MIN_SCORE}",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=settings.DEFAULT_SEARCH_LIMIT,
        help=f"Maximum number of results, default: {settings.DEFAULT_SEARCH_LIMIT}",
    )

    # Direct ingestion command (legacy support)
    ingest_parser = subparsers.add_parser(
        "ingest", help="Directly ingest code (legacy)"
    )
    ingest_subparsers = ingest_parser.add_subparsers(dest="source", required=True)

    # GitHub source
    github_parser = ingest_subparsers.add_parser(
        "github", help="Ingest from GitHub repository"
    )
    github_parser.add_argument(
        "urls",
        nargs="+",
        help="URLs of GitHub repositories (e.g., https://github.com/user/repo)",
    )
    github_parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch repositories for changes after ingesting",
    )

    # Local source
    local_parser = ingest_subparsers.add_parser("local", help="Ingest from local paths")
    local_parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Local paths to index (can specify multiple)",
    )

    # Web scraping source
    scrape_parser = ingest_subparsers.add_parser("scrape", help="Ingest from web URLs")
    scrape_parser.add_argument(
        "urls",
        nargs="+",
        help="URLs to scrape and index (e.g., https://docs.example.com)",
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
        "--host",
        default=settings.DEFAULT_HOST,
        help=f"Host to bind to (default: {settings.DEFAULT_HOST})",
    )
    watch_serve.add_argument(
        "--port",
        type=int,
        default=settings.DEFAULT_PORT,
        help=f"Port to listen on (default: {settings.DEFAULT_PORT})",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test embedding provider")
    test_parser.add_argument(
        "--embedding",
        choices=["openai", "lmstudio"],
        default=settings.DEFAULT_EMBEDDING_PROVIDER.value,
        help="Embedding provider to test",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show codebase statistics")
    stats_subparsers = stats_parser.add_subparsers(dest="stats_command", required=True)

    # Files by size
    files_parser = stats_subparsers.add_parser("files", help="Analyze files by size")
    files_parser.add_argument(
        "--min-lines", type=int, help="Show only files with at least this many lines"
    )
    files_parser.add_argument(
        "--max-lines", type=int, help="Show only files with at most this many lines"
    )
    files_parser.add_argument(
        "--sort",
        choices=["lines", "name"],
        default="lines",
        help="Sort files by line count (lines) or name (name)",
    )
    files_parser.add_argument("--paths", nargs="*", help="Filter by path patterns")

    return parser
