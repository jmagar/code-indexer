#!/usr/bin/env python3
import asyncio
import hmac
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request

try:
    from github import Github  # type: ignore
    from github.repository import Repository as GithubRepo  # type: ignore
except ImportError:
    raise ImportError(
        "PyGithub package is required. Please install it with: pip install PyGithub"
    )

from .github import GitHubPlugin
from .logger import IndexerLogger

# Configure logging
logger = IndexerLogger(__name__).get_logger()


class GitHubWatcher:
    """Watch GitHub repositories for changes and manage webhooks."""

    def __init__(
        self,
        processor: Any,  # CodeProcessor instance
        github_token: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ):
        """Initialize GitHub watcher.

        Args:
            processor: CodeProcessor instance for reindexing
            github_token: GitHub API token (or from GITHUB_TOKEN env var)
            webhook_secret: Secret for webhook validation (or from GITHUB_WEBHOOK_SECRET env var)
        """
        self.processor = processor
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.webhook_secret = webhook_secret or os.getenv("GITHUB_WEBHOOK_SECRET")
        if not self.github_token:
            raise ValueError("GitHub token not provided")

        self.github = Github(self.github_token)
        self.app = FastAPI()
        self.watched_repos: Dict[str, Dict[str, Any]] = {}
        self._setup_routes()

    def _setup_routes(self):
        """Set up webhook routes."""

        @self.app.post("/webhook")
        async def handle_webhook(request: Request):
            # Verify webhook signature
            if self.webhook_secret:
                signature = request.headers.get("X-Hub-Signature-256")
                if not signature:
                    raise HTTPException(status_code=400, detail="No signature provided")

                body = await request.body()
                expected_signature = f"sha256={hmac.new(self.webhook_secret.encode(), body, 'sha256').hexdigest()}"
                if not hmac.compare_digest(signature, expected_signature):
                    raise HTTPException(status_code=400, detail="Invalid signature")

            # Process webhook payload
            payload = await request.json()
            event_type = request.headers.get("X-GitHub-Event")

            if event_type == "push":
                await self._handle_push_event(payload)
            elif event_type == "repository":
                await self._handle_repository_event(payload)

            return {"status": "ok"}

    async def _handle_push_event(self, payload: Dict) -> None:
        """Handle repository push events."""
        repo_name = payload["repository"]["full_name"]
        if repo_name in self.watched_repos:
            logger.info(f"Push detected in repository: {repo_name}")
            # Trigger reindex for this repository
            await self._trigger_reindex(repo_name, payload["after"])

    async def _handle_repository_event(self, payload: Dict) -> None:
        """Handle repository events (created, deleted, etc)."""
        action = payload.get("action")
        repo_name = payload["repository"]["full_name"]

        if action == "deleted" and repo_name in self.watched_repos:
            logger.info(f"Repository deleted: {repo_name}")
            await self._remove_repository(repo_name)

    async def watch_repository(self, repo_url: str) -> bool:
        """Start watching a repository for changes."""
        try:
            # Extract repo name from URL
            repo_name = repo_url.split("github.com/")[-1].strip("/")
            repo = self.github.get_repo(repo_name)

            # Store repo info
            self.watched_repos[repo_name] = {
                "url": repo_url,
                "last_checked": datetime.utcnow(),
                "default_branch": repo.default_branch,
                "last_commit": repo.get_branch(repo.default_branch).commit.sha,
            }

            # Set up webhook if secret is configured
            if self.webhook_secret:
                await self._setup_webhook(repo)

            logger.info(f"Now watching repository: {repo_name}")
            return True

        except Exception as e:
            logger.error(f"Error setting up repository watch: {e}")
            return False

    async def _setup_webhook(self, repo: GithubRepo) -> None:
        """Set up webhook for a repository."""
        config = {
            "url": "YOUR_WEBHOOK_URL",  # TODO: Configure webhook URL
            "content_type": "json",
            "secret": self.webhook_secret,
        }

        try:
            # Create hook synchronously since PyGithub doesn't support async
            repo.create_hook("web", config, events=["push", "repository"], active=True)
            logger.info(f"Webhook created for {repo.full_name}")
        except Exception as e:
            logger.error(f"Error creating webhook: {e}")
            raise

    async def _trigger_reindex(self, repo_name: str, commit_sha: str) -> None:
        """Trigger reindexing of a repository."""
        try:
            logger.info(f"Reindexing {repo_name} at commit {commit_sha}")

            # Get repository URL from stored info
            repo_info = self.watched_repos.get(repo_name)
            if not repo_info:
                logger.error(f"Repository {repo_name} not found in watched repos")
                return

            # Create GitHub plugin instance
            plugin = GitHubPlugin(repo_info["url"])

            # Process the repository
            await self.processor.process_source(plugin)

            # Update stored commit hash
            self.watched_repos[repo_name]["last_commit"] = commit_sha
            self.watched_repos[repo_name]["last_indexed"] = datetime.utcnow()

            logger.info(f"Successfully reindexed {repo_name} at {commit_sha}")

        except Exception as e:
            logger.error(f"Error reindexing {repo_name}: {e}")
            # Keep the old commit hash since reindex failed
            self.watched_repos[repo_name]["last_error"] = str(e)
            self.watched_repos[repo_name]["last_error_time"] = datetime.utcnow()

    async def _remove_repository(self, repo_name: str) -> None:
        """Remove a repository from watching."""
        if repo_name in self.watched_repos:
            del self.watched_repos[repo_name]
            # TODO: Clean up associated data (webhooks, indexes, etc)
            logger.info(f"Removed repository: {repo_name}")

    async def check_for_updates(self) -> None:
        """Periodically check watched repositories for updates."""
        while True:
            for repo_name, info in self.watched_repos.items():
                try:
                    repo = self.github.get_repo(repo_name)
                    latest_commit = repo.get_branch(info["default_branch"]).commit.sha

                    if latest_commit != info["last_commit"]:
                        logger.info(f"Update detected in {repo_name}")
                        await self._trigger_reindex(repo_name, latest_commit)
                        self.watched_repos[repo_name]["last_commit"] = latest_commit

                except Exception as e:
                    logger.error(f"Error checking {repo_name} for updates: {e}")

                self.watched_repos[repo_name]["last_checked"] = datetime.utcnow()

            await asyncio.sleep(300)  # Check every 5 minutes

    def get_watched_repositories(self) -> List[Dict]:
        """Get list of currently watched repositories."""
        return [
            {"name": repo_name, **repo_info}
            for repo_name, repo_info in self.watched_repos.items()
        ]

    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the webhook server and update checker."""
        # Start update checker in the background
        asyncio.create_task(self.check_for_updates())

        # Start webhook server
        config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()
