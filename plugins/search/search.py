"""Combined search plugin using both semantic and syntax-aware search."""

import re
from typing import Any, Dict, List, Optional, Set

try:
    from tree_sitter import Language, Parser
except ImportError:
    raise ImportError(
        "tree-sitter package is required. Please install it with: pip install tree-sitter"
    )

from ..base import CodeSearchPlugin
from ..logger import IndexerLogger
from ..qdrant import QdrantSearchPlugin

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
            # Parse query into components (case insensitive)
            parsed = await self._parse_query(query.lower())
            results = []
            
            # For now, implement a simplified regex-based search with case insensitivity
            type_patterns = {
                'function': r'(?i)(async\s+)?def\s+\w+\s*\([^)]*\)',
                'class': r'(?i)class\s+\w+[^:]*:',
                'variable': r'(?i)(\w+\s*=|const\s+\w+|let\s+\w+|var\s+\w+)'
            }
            
            # Get all files from the index
            files = await self.get_indexed_files()
            
            # Filter files if paths provided
            if filter_paths:
                filtered_files = set()
                for pattern in filter_paths:
                    pattern = pattern.lower()  # Case insensitive matching
                    filtered_files.update(
                        f for f in files 
                        if pattern in f.lower()
                    )
                files = filtered_files
            
            # Search through files
            for filepath in files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # If query specifies a type, use regex pattern
                    if parsed['type'] in type_patterns:
                        pattern = type_patterns[parsed['type']]
                        matches = re.finditer(pattern, content)
                        
                        for match in matches:
                            # Get the matched line and surrounding context
                            start_pos = match.start()
                            end_pos = match.end()
                            
                            # Get line numbers
                            start_line = content.count('\n', 0, start_pos) + 1
                            end_line = content.count('\n', 0, end_pos) + 1
                            
                            # Get the full line(s) containing the match
                            lines = content.split('\n')
                            matched_lines = '\n'.join(
                                lines[max(0, start_line-1):end_line]
                            )
                            
                            # Calculate a simple score based on exact matches
                            score = 0.0
                            if parsed['context']:
                                # Count how many context words match
                                context_matches = sum(
                                    1 for word in parsed['context']
                                    if word in matched_lines.lower()
                                )
                                score = context_matches / len(parsed['context'])
                            else:
                                score = 0.8  # Default score for type matches
                                
                            if score >= min_score:
                                results.append({
                                    'score': score,
                                    'filepath': filepath,
                                    'code': matched_lines,
                                    'start_line': start_line,
                                    'end_line': end_line,
                                    'source': self.name
                                })
                            
                except Exception as e:
                    logger.error(f"Error processing file {filepath}: {e}")
                    continue
            
            # Sort by score and limit results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]
            
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
        
        # Extract type with case insensitivity
        type_patterns = {
            'function': r'(?i)\b(function|func|def|method)\b',
            'class': r'(?i)\b(class|interface|struct)\b',
            'variable': r'(?i)\b(var|let|const|variable)\b'
        }
        
        for type_name, pattern in type_patterns.items():
            if re.search(pattern, query):  # No need for re.I flag as it's in the pattern
                components['type'] = type_name
                break
                
        # Extract modifiers with case insensitivity
        modifier_patterns = {
            'async': r'(?i)\b(async|asynchronous)\b',
            'static': r'(?i)\b(static|class method)\b',
            'public': r'(?i)\b(public|exported)\b',
            'private': r'(?i)\b(private|internal)\b'
        }
        
        for modifier, pattern in modifier_patterns.items():
            if re.search(pattern, query):  # No need for re.I flag as it's in the pattern
                components['modifiers'].append(modifier)
                
        # Extract context keywords (already case insensitive due to lowered query)
        query_words = set(re.findall(r'\b\w+\b', query))
        context_words = query_words - set(sum([p.split('|') for p in type_patterns.values()], []))
        context_words = context_words - set(sum([p.split('|') for p in modifier_patterns.values()], []))
        components['context'] = list(context_words)
        
        return components
            
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.parsers.clear()

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
        # Pass the existing Qdrant client to avoid creating new connections
        self.semantic_search = QdrantSearchPlugin(qdrant_client=qdrant_client)
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
        """Search using both semantic and syntax-aware methods."""
        try:
            # Convert query to lowercase for case-insensitive search
            query = query.lower()
            
            # For TODO queries, only use semantic search
            if 'todo' in query:
                return await self.semantic_search.search(
                    query,
                    filter_paths=filter_paths,
                    min_score=min_score,
                    limit=limit
                )
            
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