#!/usr/bin/env python3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class BasePlugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self):
        """Initialize base plugin."""
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Analyze input and return results.
        
        This is the main entry point for all plugins.
        Each plugin should implement this method according to its specific needs.
        """
        pass


class CodeSourcePlugin(ABC):
    """Base class for code source plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the plugin."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the plugin does."""
        pass

    @abstractmethod
    async def get_files(self) -> List[Path]:
        """Get list of files to process."""
        pass

    @abstractmethod
    async def prepare(self) -> None:
        """Prepare the source (clone repo, setup paths, etc)."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any temporary resources."""
        pass

    @property
    def supported_extensions(self) -> List[str]:
        """List of file extensions this plugin can handle."""
        return [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"]

    @property
    def ignore_patterns(self) -> List[str]:
        """Patterns to ignore when processing files."""
        return [
            "**/venv/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/node_modules/**",
            "**/build/**",
            "**/dist/**",
        ]


class CodeSearchPlugin(ABC):
    """Base class for code search plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the plugin."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the plugin does."""
        pass

    @abstractmethod
    async def setup(self) -> None:
        """Set up any necessary resources (e.g., vector store connections)."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        filter_paths: Optional[List[str]] = None,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for code snippets using the query.

        Args:
            query: The search query
            filter_paths: Optional list of path patterns to filter results
            min_score: Minimum similarity score (0-1) for results
            limit: Maximum number of results to return

        Returns:
            List of results, each containing:
            - score: float, similarity score
            - filepath: str, path to the file
            - code: str, the code snippet
            - start_line: int, starting line number
            - end_line: int, ending line number
            - source: str, source identifier
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any resources."""
        pass
