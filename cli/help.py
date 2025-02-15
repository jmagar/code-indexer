"""Help text for the CLI."""

from rich.console import Console
from utils.console import print_header, print_info


def print_help(console: Console) -> None:
    """Print formatted help message with available commands."""
    print_header(console, "Code Indexer - Semantic code search and analysis")

    print_header(console, "Available Commands:")

    # Collection management
    print_header(console, "Collection Management")
    print_info(
        console,
        "  index collection recreate    Recreate the collection (WARNING: deletes all data)",
    )
    print_info(
        console, "  index collection status      Show collection status and statistics"
    )

    # Search commands
    print_header(console, "Search")
    print_info(console, "  index search <query>     Search code semantically")
    print_info(console, "  Options:")
    print_info(console, "    --paths <paths>     Filter by file paths")
    print_info(console, "    --min-score <float> Minimum similarity (default: 0.7)")
    print_info(console, "    --limit <int>       Maximum results (default: 5)")

    # Source management
    print_header(console, "Source Management")
    print_info(console, "  index sources list           List all indexed sources")
    print_info(console, "  index sources add <type> <path>    Add a source")
    print_info(console, "  Types:")
    print_info(console, "    github <url>        GitHub repository")
    print_info(console, "    local <path>        Local directory")
    print_info(console, "  index sources remove <id>     Remove a source")
    print_info(console, "  index sources reingest [id]   Reingest source(s)")

    # Ingestion commands
    print_header(console, "Direct Ingestion")
    print_info(console, "  index ingest github <url>    Ingest GitHub repository")
    print_info(console, "  index ingest local <path>    Ingest local directory")

    # Watch commands
    print_header(console, "Repository Watching")
    print_info(console, "  index watch list              List watched repositories")
    print_info(console, "  index watch add <url>        Start watching repository")
    print_info(console, "  index watch remove <name>    Stop watching repository")
    print_info(console, "  index watch serve             Start watch server")

    # Other commands
    print_header(console, "Other")
    print_info(console, "  index test --embedding <provider>  Test embedding provider")

    print_header(console, "Examples:")
    print_info(console, "  # Search for code")
    print_info(console, '  index search "function to handle database connections"')
    print_info(console, "")
    print_info(console, "  # Add and ingest a GitHub repository")
    print_info(console, "  index sources add github https://github.com/user/repo")
    print_info(console, "  index sources reingest github:user/repo")
    print_info(console, "")
    print_info(console, "  # Watch a repository for changes")
    print_info(console, "  index watch add https://github.com/user/repo")
    print_info(console, "  index watch serve")
    print_info(console, "")
    print_info(console, "  # Manage collection")
    print_info(console, "  index collection status")
    print_info(console, "  index collection recreate  # WARNING: Deletes all data")
