"""Syntax-aware code search using tree-sitter for AST parsing."""

import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    from tree_sitter import Language, Parser
except ImportError:
    raise ImportError(
        "tree-sitter package is required. Please install it with: pip install tree-sitter"
    )

from ..base import CodeSearchPlugin
from ..logger import IndexerLogger

logger = IndexerLogger(__name__).get_logger()

class TreeSitterSearch(CodeSearchPlugin):
    """Syntax-aware code search using tree-sitter."""
    
    @property
    def name(self) -> str:
        return "tree-sitter-search"
        
    @property
    def description(self) -> str:
        return "Syntax-aware code search using tree-sitter AST parsing"
    
    def __init__(self):
        self.supported_languages = {'python', 'javascript', 'typescript', 'go', 'rust'}
        self.parsers: Dict[str, Parser] = {}
        
    async def setup(self) -> None:
        """Set up tree-sitter parsers."""
        try:
            # For now, we'll use regex-based search until we set up tree-sitter properly
            logger.info("Initializing syntax-aware search")
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter parsers: {e}")
            raise
            
    async def search(
        self,
        query: str,
        *,
        filter_paths: Optional[List[str]] = None,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for code using syntax-aware matching."""
        try:
            # Parse query into components
            parsed = await self._parse_query(query)
            results = []
            
            # For now, implement a simplified regex-based search
            type_patterns = {
                'function': r'(async\s+)?def\s+\w+\s*\([^)]*\)',
                'class': r'class\s+\w+[^:]*:',
                'variable': r'(\w+\s*=|const\s+\w+|let\s+\w+|var\s+\w+)'
            }
            
            # TODO: Implement actual file reading and searching
            # This is just a placeholder for now
            sample_results = [
                {
                    'score': 0.85,
                    'filepath': 'example/path.py',
                    'code': 'async def process_data():\n    pass',
                    'start_line': 1,
                    'end_line': 2,
                    'source': self.name
                }
            ]
            
            return sample_results[:limit]
            
        except Exception as e:
            logger.error(f"Error during syntax-aware search: {e}")
            return []
            
    async def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse search query into syntax-aware components."""
        components = {
            'type': None,  # function, class, method, etc
            'modifiers': [],  # async, static, public, etc
            'context': [],  # keywords about functionality
            'original': query
        }
        
        # Extract type
        type_patterns = {
            'function': r'\b(function|func|def|method)\b',
            'class': r'\b(class|interface|struct)\b',
            'variable': r'\b(var|let|const|variable)\b'
        }
        
        for type_name, pattern in type_patterns.items():
            if re.search(pattern, query, re.I):
                components['type'] = type_name
                break
                
        # Extract modifiers
        modifier_patterns = {
            'async': r'\b(async|asynchronous)\b',
            'static': r'\b(static|class method)\b',
            'public': r'\b(public|exported)\b',
            'private': r'\b(private|internal)\b'
        }
        
        for modifier, pattern in modifier_patterns.items():
            if re.search(pattern, query, re.I):
                components['modifiers'].append(modifier)
                
        # Extract context keywords
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        context_words = query_words - set(sum([p.split('|') for p in type_patterns.values()], []))
        context_words = context_words - set(sum([p.split('|') for p in modifier_patterns.values()], []))
        components['context'] = list(context_words)
        
        return components
            
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.parsers.clear() 