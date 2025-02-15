#!/usr/bin/env python3
"""Web content crawling and extraction using Crawl4AI."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from .base import CodeSourcePlugin
from .logger import IndexerLogger

logger = IndexerLogger(__name__).get_logger()


class Crawl4AIPlugin(CodeSourcePlugin):
    """Plugin for crawling and extracting code from web pages."""

    def __init__(
        self,
        urls: List[str],
        host: str = "localhost",
        port: int = 11235,
    ):
        """Initialize Crawl4AI plugin.

        Args:
            urls: List of URLs to crawl
            host: Crawl4AI server host (unused, kept for compatibility)
            port: Crawl4AI server port (unused, kept for compatibility)
        """
        self.urls = urls
        self.extracted_content: Dict[str, str] = {}
        self.temp_dir = tempfile.mkdtemp(prefix="crawl4ai-")
        self._crawler: Optional[AsyncWebCrawler] = None

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "scrape"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Extract code from web pages using Crawl4AI"

    async def prepare(self) -> None:
        """Initialize crawler and extract content from URLs."""
        browser_config = BrowserConfig(
            headless=True,
            verbose=True,
            java_script_enabled=True,
        )
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
        )

        # Use context manager to handle browser lifecycle
        async with AsyncWebCrawler(config=browser_config) as crawler:
            self._crawler = crawler
            # Crawl URLs
            for url in self.urls:
                try:
                    logger.info(f"Crawling: {url}")
                    result = await crawler.arun(url=url, config=run_config)
                    if result.markdown:
                        # Convert markdown to string if it's a MarkdownGenerationResult
                        markdown_content = str(result.markdown)
                        self.extracted_content[url] = markdown_content
                        logger.info(f"Successfully extracted content from {url}")
                    else:
                        logger.warning(f"No content found in {url}")

                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    continue

    async def get_files(self) -> List[Path]:
        """Get list of files to process.

        For web content, we create temporary files from the extracted content.
        """
        temp_files = []
        self.url_to_file_map: Dict[Path, str] = (
            {}
        )  # Track which URL each file came from

        for url, content in self.extracted_content.items():
            if not content.strip():
                continue

            # Create a file path based on the URL
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            if not path_parts:
                path_parts = ["index"]

            # Use the last part as filename, add .txt if no extension
            filename = path_parts[-1]
            if not os.path.splitext(filename)[1]:
                filename += ".txt"

            # Create directory structure
            dir_path = Path(self.temp_dir) / parsed.netloc / "/".join(path_parts[:-1])
            dir_path.mkdir(parents=True, exist_ok=True)

            # Write content to file
            file_path = dir_path / filename
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            temp_files.append(file_path)
            self.url_to_file_map[file_path] = url  # Store the mapping
            logger.info(f"Created temporary file for: {url}")

        return temp_files

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._crawler:
            await self._crawler.close()
            self._crawler = None

        # Clean up temporary files
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

    def get_chunk_metadata(self, filepath: str, file_hash: str) -> Dict[str, Any]:
        """Get metadata for a chunk including source URL."""
        # Get the original URL for this file
        path_obj = Path(filepath)
        url = self.url_to_file_map.get(
            path_obj, next(iter(self.extracted_content.keys()), "unknown")
        )

        # Create a clean identifier that doesn't expose local paths
        source_id = f"scrape:{url}"
        display_path = f"{source_id}/{''.join(Path(filepath).suffixes)}"

        return {
            "filepath": display_path,  # Show as scrape:url/extension
            "actual_filepath": str(
                filepath
            ),  # Keep the actual file path for reading content
            "file_type": Path(filepath).suffix[1:],
            "file_hash": file_hash,
            "source": "crawl4ai",  # Keep internal name as crawl4ai
            "url": url,
            "origin": source_id,  # Use source ID as origin
        }
