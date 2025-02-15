"""Main entry point for the indexer."""

import asyncio
import sys

from cli.commands import main
from plugins import IndexerLogger

# Configure logging
logger = IndexerLogger(__name__).get_logger()
logger.setLevel("WARNING")  # Only show warning and above by default

# Suppress urllib3 warnings about insecure requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
