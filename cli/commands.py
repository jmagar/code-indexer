"""Command handlers for the CLI."""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, TypedDict, Union

from rich.console import Console
from rich.table import Table

from core.processor import CodeProcessor
from plugins import GitHubPlugin, LocalCodePlugin
from plugins.crawl4ai import Crawl4AIPlugin
from plugins.github_watcher import GitHubWatcher
from plugins.sources import SourceManager
from utils.config import get_settings
from utils.console import (
    print_error,
    print_success,
    print_info,
    print_header,
)
from utils.errors import (
    NetworkError,
    ProcessingError,
    SourceError,
    WatcherError,
)
from cli.help import print_help

logger = logging.getLogger(__name__)

# Type alias for source plugins
SourcePlugin = Union[GitHubPlugin, LocalCodePlugin, Crawl4AIPlugin]


class FileInfo(TypedDict):
    """Type definition for file information."""

    path: str
    lines: int
    source: str


async def handle_test(processor: CodeProcessor, args: argparse.Namespace) -> None:
    """Handle test command."""
    test_texts = [
        "This is a test sentence.",
        "Another test sentence to try.",
        "A third test sentence to verify batching.",
    ]
    try:
        print(f"Testing {processor.embeddings.embedding.name} embeddings...")
        embeddings = await processor.embeddings.get_embeddings(test_texts)
        print(f"Successfully generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
    except Exception as e:
        print(f"Error: {e}")
        raise


async def handle_search(processor: CodeProcessor, args: argparse.Namespace) -> None:
    """Handle search command."""
    console = Console()
    await processor.search_code(
        args.query,
        filter_paths=args.paths,
        min_score=args.min_score,
        limit=args.limit,
        console=console,
    )


async def handle_sources(processor: CodeProcessor, args: argparse.Namespace) -> None:
    """Handle sources command."""
    console = Console()
    source_manager = SourceManager()

    try:
        await source_manager.setup_collection()

        if args.source_command == "list":
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
            print_success(console, f"Added source: {source_id}")

            try:
                # Create appropriate source plugin
                plugin: SourcePlugin
                if args.type == "github":
                    plugin = GitHubPlugin(args.path)
                elif args.type == "local":
                    plugin = LocalCodePlugin([Path(args.path)])
                elif args.type == "crawl4ai":
                    plugin = Crawl4AIPlugin([args.path])
                else:
                    raise ValueError(f"Unknown source type: {args.type}")

                await processor.process_source(plugin, console)

                # Update last ingestion time
                await source_manager.update_source(
                    source_id,
                    {"last_ingested": datetime.now(timezone.utc).isoformat()},
                )
            except NetworkError as e:
                logger.error(f"Network error processing source: {e}")
                print_error(console, f"Network error: {e}")
            except ProcessingError as e:
                logger.error(f"Processing error: {e}")
                print_error(console, f"Processing error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print_error(console, f"Error processing source: {e}")

        elif args.source_command == "remove":
            try:
                success = await source_manager.remove_source(args.id)
                if success:
                    print_success(console, f"Removed source: {args.id}")
                else:
                    print_error(console, f"Source not found: {args.id}")
            except Exception as e:
                logger.error(f"Failed to remove source: {e}")
                raise SourceError(f"Failed to remove source: {e}") from e

        elif args.source_command == "reingest":
            await handle_reingest(processor, source_manager, args, console)

    except Exception as e:
        logger.error(f"Source operation failed: {e}")
        raise SourceError(f"Source operation failed: {e}") from e


async def handle_reingest(
    processor: CodeProcessor,
    source_manager: SourceManager,
    args: argparse.Namespace,
    console: Console,
) -> None:
    """Handle source reingestion."""
    try:
        if args.id:
            # Reingest specific source
            source = await source_manager.get_source(args.id)
            if not source:
                print_error(console, f"Source not found: {args.id}")
                return

            print_info(console, f"Reingesting source: {args.id}")

            plugin: SourcePlugin
            if source["type"] == "github":
                plugin = GitHubPlugin(source.get("url", ""))
            elif source["type"] == "local":
                plugin = LocalCodePlugin([Path(source.get("path", ""))])
            elif source["type"] == "crawl4ai":
                plugin = Crawl4AIPlugin([source.get("path", "")])
            else:
                raise ValueError(f"Unknown source type: {source['type']}")

            await processor.process_source(plugin, console)

            # Update last ingestion time
            await source_manager.update_source(
                args.id,
                {"last_ingested": datetime.now(timezone.utc).isoformat()},
            )

        else:
            # Reingest all sources
            sources = await source_manager.list_sources()
            if not sources:
                print_error(console, "No sources found to reingest")
                return

            print_info(console, f"Reingesting all sources ({len(sources)} total)")
            for source in sources:
                source_id = source.get("source_id", "")
                print_info(console, f"\nProcessing: {source_id}")

                try:
                    current_plugin: SourcePlugin
                    if source["type"] == "github":
                        current_plugin = GitHubPlugin(source.get("url", ""))
                    elif source["type"] == "local":
                        current_plugin = LocalCodePlugin([Path(source.get("path", ""))])
                    elif source["type"] == "crawl4ai":
                        current_plugin = Crawl4AIPlugin([source.get("path", "")])
                    else:
                        logger.warning(
                            f"Skipping unknown source type: {source['type']}"
                        )
                        continue

                    await processor.process_source(current_plugin, console)
                    # Update last ingestion time
                    await source_manager.update_source(
                        source_id,
                        {"last_ingested": datetime.now(timezone.utc).isoformat()},
                    )
                    print_success(console, f"Successfully reingested: {source_id}")
                except Exception as e:
                    logger.error(f"Failed to reingest {source_id}: {e}")
                    print_error(console, f"Failed to reingest {source_id}: {e}")
                    continue

            print_success(console, "\nReingestion complete!")

    except Exception as e:
        logger.error(f"Reingestion failed: {e}")
        raise ProcessingError(f"Reingestion failed: {e}") from e


async def handle_ingest(processor: CodeProcessor, args: argparse.Namespace) -> None:
    """Handle ingest command."""
    console = Console()

    try:
        if args.source == "github":
            # Handle multiple GitHub URLs
            for url in args.urls:
                try:
                    github_plugin = GitHubPlugin(url)
                    await processor.process_source(github_plugin, console)

                    # Set up watching if requested
                    if args.watch:
                        watcher = GitHubWatcher(processor)
                        await watcher.watch_repository(url)
                        print_success(console, f"Started watching: {url}")
                except NetworkError as e:
                    logger.error(f"Network error processing GitHub URL {url}: {e}")
                    print_error(console, f"Network error for {url}: {e}")
                except Exception as e:
                    logger.error(f"Failed to process GitHub URL {url}: {e}")
                    print_error(console, f"Failed to process {url}: {e}")

        elif args.source == "local":
            # Handle local paths
            try:
                local_plugin = LocalCodePlugin(args.paths)
                await processor.process_source(local_plugin, console)
            except ProcessingError as e:
                logger.error(f"Processing error for local paths: {e}")
                print_error(console, f"Processing error: {e}")
            except Exception as e:
                logger.error(f"Failed to process local paths: {e}")
                print_error(console, f"Failed to process local paths: {e}")

        elif args.source == "scrape":
            # Handle web scraping
            try:
                crawl_plugin = Crawl4AIPlugin(args.urls)
                await processor.process_source(crawl_plugin, console)
            except NetworkError as e:
                logger.error(f"Network error during web scraping: {e}")
                print_error(console, f"Network error: {e}")
            except Exception as e:
                logger.error(f"Failed to scrape URLs: {e}")
                print_error(console, f"Failed to scrape URLs: {e}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise ProcessingError(f"Ingestion failed: {e}") from e


async def handle_watch(processor: CodeProcessor, args: argparse.Namespace) -> None:
    """Handle watch command."""
    watcher = GitHubWatcher(processor)

    try:
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
            try:
                success = await watcher.watch_repository(args.url)
                if success:
                    print(f"Successfully started watching: {args.url}")
                else:
                    print(f"Failed to start watching: {args.url}")
            except NetworkError as e:
                logger.error(f"Network error adding watch: {e}")
                raise WatcherError(
                    f"Failed to add watch due to network error: {e}"
                ) from e
            except Exception as e:
                logger.error(f"Failed to add watch: {e}")
                raise WatcherError(f"Failed to add watch: {e}") from e

        elif args.watch_command == "remove":
            try:
                await watcher.unwatch_repository(args.repo_name)
                print(f"Stopped watching: {args.repo_name}")
            except Exception as e:
                logger.error(f"Failed to remove watch: {e}")
                raise WatcherError(f"Failed to remove watch: {e}") from e

        elif args.watch_command == "serve":
            try:
                print(f"Starting watcher server on {args.host}:{args.port}")
                await watcher.start(host=args.host, port=args.port)
            except Exception as e:
                logger.error(f"Watcher server failed: {e}")
                raise WatcherError(f"Watcher server failed: {e}") from e

    except Exception as e:
        logger.error(f"Watch operation failed: {e}")
        raise WatcherError(f"Watch operation failed: {e}") from e


async def handle_collection(processor: CodeProcessor, args: argparse.Namespace) -> None:
    """Handle collection commands."""
    console = Console()

    try:
        if args.collection_command == "recreate":
            print_info(console, "Recreating collection...")
            await processor.setup_collection(force=True)
            print_success(console, "Collection recreated successfully!")

        elif args.collection_command == "status":
            try:
                client = processor.collection.client
                collections = client.get_collections()

                # Code Collection Status
                print_header(console, "Code Collection")
                code_name = processor.collection.collection_name
                code_exists = any(
                    col.name == code_name for col in collections.collections
                )
                print_info(console, f"Name: {code_name}")
                print_info(
                    console, f"Status: {'Active' if code_exists else 'Not Found'}"
                )

                if code_exists:
                    # Collection Statistics
                    print_header(console, "Statistics")
                    points = client.count(code_name)
                    print_info(console, f"Total points: {points.count:,}")

                    # Vector Configuration
                    print_header(console, "Vector Configuration")
                    print_info(
                        console,
                        f"Size: {processor.embeddings.embedding.dimension} dimensions",
                    )
                    print_info(console, "Distance: Cosine")

                # Sources Collection Status
                print_header(console, "\nSources Collection")
                sources_name = "sources"
                sources_exists = any(
                    col.name == sources_name for col in collections.collections
                )
                print_info(console, f"Name: {sources_name}")
                print_info(
                    console, f"Status: {'Active' if sources_exists else 'Not Found'}"
                )

                if sources_exists:
                    # Collection Statistics
                    print_header(console, "Statistics")
                    points = client.count(sources_name)
                    print_info(console, f"Total sources: {points.count:,}")

                    # Get source types
                    source_types = set()
                    scroll_result = client.scroll(
                        collection_name=sources_name,
                        limit=100,
                        with_payload=True,
                    )[0]
                    for point in scroll_result:
                        if point.payload and "type" in point.payload:
                            source_types.add(point.payload["type"])

                    if source_types:
                        print_header(console, "Source Types")
                        for source_type in sorted(source_types):
                            print_info(console, f"  - {source_type}")

            except Exception as e:
                logger.error(f"Failed to get collection status: {e}")
                raise ProcessingError(f"Failed to get collection status: {e}") from e

    except Exception as e:
        logger.error(f"Collection operation failed: {e}")
        raise ProcessingError(f"Collection operation failed: {e}") from e


async def handle_stats(processor: CodeProcessor, args: argparse.Namespace) -> None:
    """Handle stats command."""
    console = Console()
    settings = get_settings()  # Keep this one as it's used for QDRANT_BATCH_SIZE

    if args.stats_command == "files":
        # Get all files from collection
        files_info: List[FileInfo] = []
        total_lines = 0
        total_files = 0

        try:
            # Get all points from collection
            points = processor.collection.client.scroll(
                collection_name=processor.collection.collection_name,
                with_payload=True,
                limit=settings.QDRANT_BATCH_SIZE,
            )

            seen_files = set()  # Track unique files to avoid duplicates from chunks

            while True:
                batch, next_page_offset = points
                if not batch:
                    break

                for point in batch:
                    if not point.payload or "metadata" not in point.payload:
                        continue

                    metadata = point.payload["metadata"]
                    filepath = metadata.get("filepath", "")

                    # Skip if we've already processed this file
                    if filepath in seen_files:
                        continue
                    seen_files.add(filepath)

                    # Apply path filters if specified
                    if args.paths:
                        matches = False
                        for pattern in args.paths:
                            if Path(filepath).match(pattern):
                                matches = True
                                break
                        if not matches:
                            continue

                    # Get line count from metadata
                    line_count = (
                        metadata.get("end_line", 0) - metadata.get("start_line", 0) + 1
                    )

                    # Apply line count filters
                    if args.min_lines and line_count < args.min_lines:
                        continue
                    if args.max_lines and line_count > args.max_lines:
                        continue

                    # Track file info
                    files_info.append(
                        FileInfo(
                            path=filepath,
                            lines=line_count,
                            source=metadata.get("source", "unknown"),
                        )
                    )
                    total_lines += line_count
                    total_files += 1

                if not next_page_offset:
                    break

                # Get next batch
                points = processor.collection.client.scroll(
                    collection_name=processor.collection.collection_name,
                    with_payload=True,
                    limit=settings.QDRANT_BATCH_SIZE,
                    offset=next_page_offset,
                )

            # Sort results
            if args.sort == "lines":
                files_info.sort(key=lambda x: x["lines"], reverse=True)
            else:  # sort by name
                files_info.sort(key=lambda x: x["path"].lower())

            # Print results
            print_header(console, "\nFiles by Size")

            if not files_info:
                print_info(console, "No matching files found")
                return

            # Create table
            table = Table(show_header=True)
            table.add_column("Lines", justify="right", style="cyan")
            table.add_column("Path", style="green")
            table.add_column("Source", style="blue")

            for file_info in files_info:
                table.add_row(
                    f"{file_info['lines']:,}",
                    file_info["path"],
                    file_info["source"],
                )

            console.print(table)

            # Print summary
            print_info(console, f"\nTotal files: {total_files:,}")
            print_info(console, f"Total lines: {total_lines:,}")
            if total_files > 0:
                avg_lines = total_lines / total_files
                print_info(console, f"Average lines per file: {avg_lines:.1f}")

        except Exception as e:
            logger.error(f"Failed to get file stats: {e}")
            raise ProcessingError(f"Failed to get file stats: {e}") from e


async def main() -> None:
    """Main entry point."""
    from cli.parser import setup_parser

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
            await handle_test(processor, args)
        elif args.command == "search":
            await handle_search(processor, args)
        elif args.command == "sources":
            await handle_sources(processor, args)
        elif args.command == "ingest":
            await handle_ingest(processor, args)
        elif args.command == "watch":
            await handle_watch(processor, args)
        elif args.command == "collection":
            await handle_collection(processor, args)
        elif args.command == "stats":
            await handle_stats(processor, args)

    except (NetworkError, ProcessingError, SourceError, WatcherError) as e:
        logger.error(f"Operation failed: {e}")
        print_error(Console(), str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print_error(Console(), f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
