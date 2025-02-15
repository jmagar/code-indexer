"""CLI package for code indexer."""

import asyncio
import sys

from .commands import main


def run_cli() -> None:
    """Run the CLI application."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
