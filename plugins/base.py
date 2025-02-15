#!/usr/bin/env python3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BasePlugin(ABC):
    """Base class for all plugins."""

    def __init__(self) -> None:
        """Initialize base plugin."""
        self.name: str = self.__class__.__name__
        self.config: Dict[str, Any] = {}

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin with the provided settings.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def validate_config(self) -> bool:
        """Validate plugin configuration.

        Returns:
            True if configuration is valid
        """
        return True

    @abstractmethod
    async def process(self, data: Any, **kwargs: Any) -> Any:
        """Process data using the plugin.

        Args:
            data: Input data to process
            **kwargs: Additional keyword arguments

        Returns:
            Processed data
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
        return [
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".go",
            ".rs",
            ".java",
            ".md",
            ".mdx",
            ".markdown",
        ]

    @property
    def ignore_patterns(self) -> List[str]:
        """Patterns to ignore when processing files."""
        return [
            # Development environments and dependencies
            "**/node_modules/**",
            "**/bower_components/**",
            "**/vendor/**",
            "**/target/**",  # Rust/Cargo
            "**/bin/**",
            "**/obj/**",
            "**/venv/**",
            "**/.venv/**",
            "**/env/**",
            "**/.env/**",
            "**/.virtualenv/**",
            "**/.virtualenvs/**",
            "**/virtualenv/**",
            # Version control
            "**/.git/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/.bzr/**",
            # Cache directories
            "**/__pycache__/**",
            "**/.mypy_cache/**",
            "**/.pytest_cache/**",
            "**/.ruff_cache/**",
            "**/.uv/**",
            "**/.cache/**",
            "**/*.pyc",
            "**/*.pyo",
            "**/*.pyd",
            # Build and distribution
            "**/build/**",
            "**/dist/**",
            "**/*.egg-info/**",
            "**/*.min.js",
            "**/*.min.css",
            "**/*.map",
            # IDE
            "**/.idea/**",
            "**/.vscode/**",
            "**/.vs/**",
            "**/*.sublime-*",
            # Logs and databases
            "**/logs/**",
            "**/*.log",
            "**/*.sqlite",
            "**/*.db",
            # Project specific
            "**/temp/**",
            "**/tmp/**",
            "**/embeddings/**",
            "**/vectors/**",
            "**/.qdrant/**",
            "**/qdrant_storage/**",
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
