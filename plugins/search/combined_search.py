"""Combined search plugin using both semantic and syntax-aware search."""

from typing import Any, Dict, List, Optional, Set

from ..base import CodeSearchPlugin
from ..logger import IndexerLogger
from ..analysis.syntax_search import TreeSitterSearch
from ..qdrant import QdrantSearchPlugin

logger = IndexerLogger(__name__).get_logger()

class CombinedSearchPlugin(CodeSearchPlugin):
    """Plugin that combines semantic and syntax-aware search."""
    
    @property
    def name(self) -> str:
        return "combined-search"
        
    @property
    def description(self) -> str:
        return "Combined semantic and syntax-aware code search"
    
    def __init__(self, qdrant_client):
        """Initialize search plugins.
        
        Args:
            qdrant_client: Qdrant client for semantic search
        """
        # QdrantSearchPlugin will use environment variables for configuration
        self.semantic_search = QdrantSearchPlugin()
        self.syntax_search = TreeSitterSearch()
        
    async def setup(self) -> None:
        """Set up both search plugins."""
        await self.syntax_search.setup()
        await self.semantic_search.setup()
            
    async def search(
        self,
        query: str,
        *,
        filter_paths: Optional[List[str]] = None,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search using both semantic and syntax-aware methods.
        
        Args:
            query: Search query
            filter_paths: Optional list of path patterns to filter results
            min_score: Minimum similarity score (0-1)
            limit: Maximum number of results to return
            
        Returns:
            Combined and deduplicated search results
        """
        try:
            # Get semantic search results
            semantic_results = await self.semantic_search.search(
                query,
                filter_paths=filter_paths,
                min_score=min_score,
                limit=limit
            )
            
            # Only use syntax search for specific query types
            parsed_query = await self.syntax_search._parse_query(query)
            if parsed_query['type'] in ['function', 'class']:
                # Get syntax-aware results
                syntax_results = await self.syntax_search.search(
                    query,
                    filter_paths=filter_paths,
                    min_score=min_score,
                    limit=limit
                )
                
                # Combine and deduplicate results
                all_results = semantic_results + syntax_results
                seen_snippets = set()
                unique_results = []
                
                for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
                    snippet = result['code'].strip()
                    if snippet not in seen_snippets:
                        seen_snippets.add(snippet)
                        unique_results.append(result)
                        
                return unique_results[:limit]
            
            # For general queries, return semantic results only
            return semantic_results
            
        except Exception as e:
            logger.error(f"Combined search failed: {e}")
            return []
            
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.semantic_search.cleanup()
        await self.syntax_search.cleanup() 