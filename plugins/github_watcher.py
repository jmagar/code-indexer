"""GitHub repository watcher plugin."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TypedDict, cast

from aiohttp import web
from rich.console import Console
from core.processor import CodeProcessor
from plugins import GitHubPlugin

logger = logging.getLogger(__name__)


class Repository(TypedDict):
    """Type definition for repository data in webhook payload."""

    full_name: str


class WebhookPayload(TypedDict):
    """Type definition for webhook payload."""

    repository: Repository


class RepoData(TypedDict):
    """Type definition for repository data."""

    name: str
    url: str
    default_branch: str
    last_commit: str
    last_checked: str
    last_error: Optional[str]
    last_error_time: Optional[str]


class GitHubWatcher:
    """Watches GitHub repositories for changes."""

    def __init__(self, processor: CodeProcessor):
        """Initialize GitHub watcher.

        Args:
            processor: Code processor instance
        """
        self.processor = processor
        self.console = Console()
        self.data_dir = Path.home() / ".indexer" / "watched_repos"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.data_dir / "repos.json"

    def _load_data(self) -> List[RepoData]:
        """Load watched repositories data."""
        if not self.data_file.exists():
            return []
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
                return cast(List[RepoData], data)
        except Exception as e:
            logger.error(f"Error loading watched repos data: {e}")
            return []

    def _save_data(self, data: List[RepoData]) -> None:
        """Save watched repositories data."""
        try:
            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving watched repos data: {e}")

    def get_watched_repositories(self) -> List[RepoData]:
        """Get list of watched repositories."""
        return self._load_data()

    async def watch_repository(self, repo_url: str) -> bool:
        """Start watching a repository.

        Args:
            repo_url: GitHub repository URL

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract owner/repo from URL
            repo_name = repo_url.split("github.com/")[1].rstrip("/")

            # Check if already watching
            repos = self._load_data()
            for repo in repos:
                if repo["name"] == repo_name:
                    logger.info(f"Already watching {repo_name}")
                    return True

            # Get repository info
            plugin = GitHubPlugin(repo_url)
            await plugin.prepare()

            # Add to watched repos
            new_repo: RepoData = {
                "name": repo_name,
                "url": repo_url,
                "default_branch": "main",  # Default to main
                "last_commit": "",  # Will be updated on first check
                "last_checked": datetime.now(timezone.utc).isoformat(),
                "last_error": None,
                "last_error_time": None,
            }
            repos.append(new_repo)
            self._save_data(repos)

            logger.info(f"Started watching {repo_name}")
            return True

        except Exception as e:
            logger.error(f"Error watching repository {repo_url}: {e}")
            return False

    async def unwatch_repository(self, repo_name: str) -> None:
        """Stop watching a repository.

        Args:
            repo_name: Repository name (owner/repo)
        """
        repos = self._load_data()
        repos = [r for r in repos if r["name"] != repo_name]
        self._save_data(repos)
        logger.info(f"Stopped watching {repo_name}")

    async def check_repository(self, repo_data: RepoData) -> None:
        """Check a repository for changes.

        Args:
            repo_data: Repository data
        """
        try:
            plugin = GitHubPlugin(repo_data["url"])
            await plugin.prepare()

            # TODO: Implement commit hash checking in GitHubPlugin
            # For now, we'll just process the repository
            logger.info(f"Processing repository {repo_data['name']}")
            await self.processor.process_source(plugin, self.console)

            # Update last checked time
            repos = self._load_data()
            for repo in repos:
                if repo["name"] == repo_data["name"]:
                    repo["last_checked"] = datetime.now(timezone.utc).isoformat()
                    break
            self._save_data(repos)

        except Exception as e:
            logger.error(f"Error checking repository {repo_data['name']}: {e}")
            # Update error info
            repos = self._load_data()
            for repo in repos:
                if repo["name"] == repo_data["name"]:
                    repo["last_error"] = str(e)
                    repo["last_error_time"] = datetime.now(timezone.utc).isoformat()
                    break
            self._save_data(repos)

    async def check_all_repositories(self) -> None:
        """Check all watched repositories for changes."""
        repos = self._load_data()
        for repo in repos:
            await self.check_repository(repo)

    async def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the watcher server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        app = web.Application()
        app.router.add_post("/webhook", self._handle_webhook)

        # Start periodic checks
        asyncio.create_task(self._periodic_check())

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info(f"Watcher server running on http://{host}:{port}")

        # Keep the server running
        while True:
            await asyncio.sleep(3600)

    async def _periodic_check(self, interval: int = 300) -> None:
        """Periodically check repositories for changes.

        Args:
            interval: Check interval in seconds
        """
        while True:
            await self.check_all_repositories()
            await asyncio.sleep(interval)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle GitHub webhook requests.

        Args:
            request: Web request

        Returns:
            Web response
        """
        try:
            # Verify GitHub signature
            signature = request.headers.get("X-Hub-Signature-256")
            if not signature:
                return web.Response(status=400, text="No signature")

            # Get payload
            data = await request.json()
            payload = cast(WebhookPayload, data)
            repo_name = payload["repository"]["full_name"]

            # Check if we're watching this repo
            repos = self._load_data()
            repo_data = None
            for repo in repos:
                if repo["name"] == repo_name:
                    repo_data = repo
                    break

            if repo_data:
                await self.check_repository(repo_data)
                return web.Response(text="OK")
            else:
                return web.Response(status=404, text="Repository not found")

        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return web.Response(status=500, text=str(e))
