#!/usr/bin/env python3
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from git import Repo

from .base import CodeSourcePlugin
from .logger import IndexerLogger

# Configure logging
logger = IndexerLogger(__name__).get_logger()


@dataclass
class RepoInfo:
    """Repository information."""

    url: str
    owner: str
    name: str
    branch: str
    commit_hash: str


class GitHubPlugin(CodeSourcePlugin):
    """Plugin for processing GitHub repositories."""

    def __init__(self, repo_url: str):
        """Initialize with GitHub repository URL."""
        self.repo_url = repo_url
        self.work_dir = None
        self.repo_info = None
        self._supported_extensions = {".py", ".js", ".ts", ".jsx", ".tsx"}
        self._ignore_patterns = {
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.git/**",
            "**/dist/**",
            "**/build/**",
            "**/*.min.js",
            "**/*.test.*",
            "**/*.spec.*",
        }

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "github"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Process code from GitHub repositories"

    @property
    def supported_extensions(self) -> List[str]:
        """List of file extensions this plugin can handle."""
        return list(self._supported_extensions)

    @property
    def ignore_patterns(self) -> List[str]:
        """Patterns to ignore when processing files."""
        return list(self._ignore_patterns)

    def _parse_repo_url(self) -> tuple[str, str]:
        """Parse repository URL to extract owner and name."""
        parsed = urlparse(self.repo_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) != 2:
            raise ValueError(f"Invalid GitHub repository URL: {self.repo_url}")
        owner, name = path_parts
        return owner, name

    async def prepare(self) -> None:
        """Clone repository to temporary directory."""
        # Parse repo URL to get repo name
        owner, name = self._parse_repo_url()

        # Create temporary directory
        self.work_dir = Path(tempfile.mkdtemp(prefix=f"github-{name}-"))
        logger.info(f"Created temporary directory: {self.work_dir}")

        # Clone repository
        logger.info(f"Cloning repository: {self.repo_url}")
        try:
            repo = Repo.clone_from(self.repo_url, self.work_dir)
            # Get repository information
            self.repo_info = RepoInfo(
                url=self.repo_url,
                owner=owner,
                name=name,
                branch=repo.active_branch.name,
                commit_hash=repo.head.commit.hexsha,
            )
            logger.info(f"Successfully cloned repository: {owner}/{name}")
            logger.info(
                f"Branch: {self.repo_info.branch}, Commit: {self.repo_info.commit_hash}"
            )
        except Exception as e:
            # Clean up on failure
            if self.work_dir and self.work_dir.exists():
                shutil.rmtree(self.work_dir)
            raise ValueError(f"Failed to clone repository: {e}")

    def get_chunk_metadata(self, filepath: str, file_hash: str) -> Dict:
        """Get metadata for a chunk including repository information."""
        if not self.repo_info:
            raise ValueError(
                "Repository information not available. Call prepare() first."
            )

        return {
            "filepath": filepath,
            "file_type": Path(filepath).suffix[1:],
            "file_hash": file_hash,
            "source": self.name,
            "repo": {
                "url": self.repo_info.url,
                "owner": self.repo_info.owner,
                "name": self.repo_info.name,
                "branch": self.repo_info.branch,
                "commit": self.repo_info.commit_hash,
            },
        }

    async def get_files(self) -> List[Path]:
        """Get all code files from cloned repository."""
        if not self.work_dir or not self.work_dir.exists():
            raise ValueError("Repository not cloned. Call prepare() first.")

        all_files = []
        for file in self.work_dir.rglob("*"):
            if self._should_process_file(file):
                all_files.append(file)

        return all_files

    async def cleanup(self) -> None:
        """Remove temporary directory."""
        if self.work_dir and self.work_dir.exists():
            logger.info(f"Cleaning up temporary directory: {self.work_dir}")
            shutil.rmtree(self.work_dir)
            self.work_dir = None

    def _should_process_file(self, file: Path) -> bool:
        """Check if file should be processed based on extension and ignore patterns."""
        if not file.is_file():
            return False

        # Check file extension
        if file.suffix not in self.supported_extensions:
            return False

        # Check ignore patterns
        file_str = str(file)
        for pattern in self.ignore_patterns:
            if Path(file_str).match(pattern):
                return False

        return True
