#!/usr/bin/env python3
"""Wrapper script for the code indexer."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to Python path
script_dir = Path(__file__).parent.parent
sys.path.append(str(script_dir))

from cli.commands import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
