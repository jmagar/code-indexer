from typing import Any, Optional, Protocol
from enum import Enum

class CacheMode(str, Enum):
    """Cache mode for crawler."""

    ENABLED = "enabled"
    DISABLED = "disabled"

class BrowserConfig(Protocol):
    """Browser configuration."""

    headless: bool
    verbose: bool
    java_script_enabled: bool

class CrawlerRunConfig(Protocol):
    """Crawler run configuration."""

    cache_mode: CacheMode

class MarkdownGenerationResult(Protocol):
    """Result of markdown generation."""

    markdown: str

class CrawlerResult(Protocol):
    """Result of crawler run."""

    markdown: Optional[MarkdownGenerationResult]

class AsyncWebCrawler:
    """Async web crawler."""

    def __init__(self, config: BrowserConfig) -> None: ...
    async def __aenter__(self) -> "AsyncWebCrawler": ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    async def arun(self, url: str, config: CrawlerRunConfig) -> CrawlerResult: ...
    async def close(self) -> None: ...
