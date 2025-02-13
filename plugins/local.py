#!/usr/bin/env python3
import fnmatch
from pathlib import Path
from typing import List

from .base import CodeSourcePlugin


class LocalCodePlugin(CodeSourcePlugin):
    """Plugin for processing local code directories."""

    def __init__(self, paths: List[Path]):
        """Initialize with list of paths to process."""
        self.paths = paths
        self._supported_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".json",
            ".yaml",
            ".yml",
            ".md",
        }
        self._ignore_patterns = {
            # Development environments and dependencies
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.env/**",
            "**/env/**",
            "**/.virtualenv/**",
            "**/.virtualenvs/**",
            "**/virtualenv/**",
            "**/dist/**",
            "**/build/**",
            "**/.next/**",
            "**/.nuxt/**",
            # Version control
            "**/.git/**",
            "**/repos/**",
            # Test files
            "**/test/**",
            "**/tests/**",
            "**/cypress/**",
            "**/e2e/**",
            "**/__snapshots__/**",
            "**/*.test.*",
            "**/*.spec.*",
            # Cache and coverage
            "**/.pytest_cache/**",
            "**/coverage/**",
            "**/.coverage/**",
            "**/htmlcov/**",
            # Compiled files
            "**/*.pyc",
            "**/*.pyo",
            "**/*.pyd",
            "**/*.so",
            "**/*.egg",
            "**/*.egg-info/**",
            "**/*.min.js",
            "**/*.min.css",
            "**/*.map",
            # Data and logs
            "**/logs/**",
            "**/data/**",
            "**/tmp/**",
            "**/temp/**",
            "**/storage/**",
            "**/wal/**",
            "**/snapshots/**",
            # System files
            "**/.DS_Store",
            "**/Thumbs.db",
        }

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "local"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Process code from local directories"

    @property
    def supported_extensions(self) -> List[str]:
        """List of file extensions this plugin can handle."""
        return list(self._supported_extensions)

    @property
    def ignore_patterns(self) -> List[str]:
        """Patterns to ignore when processing files."""
        return list(self._ignore_patterns)

    async def prepare(self) -> None:
        """No preparation needed for local files."""
        pass

    async def get_files(self) -> List[Path]:
        """Get all code files from specified paths."""
        all_files = []
        for path in self.paths:
            if not path.exists():
                continue

            if path.is_file():
                if self._should_process_file(path):
                    all_files.append(path)
            else:
                for file in path.rglob("*"):
                    if self._should_process_file(file):
                        all_files.append(file)

        return all_files

    async def cleanup(self) -> None:
        """No cleanup needed for local files."""
        pass

    def _should_process_file(self, file: Path) -> bool:
        """Check if file should be processed based on extension and ignore patterns."""
        if not file.is_file():
            return False

        # Check file extension
        if file.suffix not in self.supported_extensions:
            return False

        # Convert path to string for pattern matching
        file_str = str(file)

        # Check if any part of the path matches ignore patterns
        path_parts = Path(file_str).parts
        for part in path_parts:
            if any(
                fnmatch.fnmatch(part, pattern.strip("*/"))
                for pattern in self.ignore_patterns
            ):
                return False

        # Also check the full path against patterns
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                return False

        return True
