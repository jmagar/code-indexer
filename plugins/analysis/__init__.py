"""Code Analysis Plugins for advanced code understanding and metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from ..base import BasePlugin
from ..logger import IndexerLogger

logger = IndexerLogger(__name__).get_logger()

class CodeAnalysisPlugin(BasePlugin):
    """Base class for all code analysis plugins."""
    
    def __init__(self):
        super().__init__()
        self.supported_languages: Set[str] = set()
        self.requires_ast: bool = False
        
    @abstractmethod
    async def analyze(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code and return results.
        
        Args:
            code: The code to analyze
            language: Programming language of the code
            
        Returns:
            Dict containing analysis results
        """
        pass
    
    def supports_language(self, language: str) -> bool:
        """Check if plugin supports a language."""
        return language in self.supported_languages

class SyntaxAwareSearch(CodeAnalysisPlugin):
    """Base class for syntax-aware code search plugins."""
    
    @abstractmethod
    async def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse search query into syntax-aware components."""
        pass
    
    @abstractmethod
    async def search(self, parsed_query: Dict[str, Any], code: str) -> List[Tuple[str, float]]:
        """Search code using syntax-aware matching."""
        pass

class NLQueryOptimizer(CodeAnalysisPlugin):
    """Base class for natural language query optimization."""
    
    @abstractmethod
    async def optimize_query(self, query: str) -> str:
        """Optimize natural language query for code search."""
        pass
    
    @abstractmethod
    async def explain_optimization(self, original: str, optimized: str) -> str:
        """Explain how the query was optimized."""
        pass

class CodeExplainer(CodeAnalysisPlugin):
    """Base class for code explanation generation."""
    
    @abstractmethod
    async def explain_snippet(self, code: str, context: Optional[str] = None) -> str:
        """Generate natural language explanation of code snippet."""
        pass
    
    @abstractmethod
    async def explain_function(self, code: str, function_name: str) -> str:
        """Generate explanation of specific function."""
        pass

class FunctionAnalyzer(CodeAnalysisPlugin):
    """Base class for function analysis (similarity, dependencies, etc)."""
    
    @abstractmethod
    async def find_similar_functions(self, function: str, codebase: List[str]) -> List[Tuple[str, float]]:
        """Find similar functions in codebase."""
        pass
    
    @abstractmethod
    async def generate_dependency_graph(self, code: str) -> Dict[str, List[str]]:
        """Generate function/class dependency graph."""
        pass

class CodeQualityAnalyzer(CodeAnalysisPlugin):
    """Base class for code quality analysis."""
    
    @abstractmethod
    async def calculate_complexity(self, code: str) -> Dict[str, float]:
        """Calculate code complexity metrics."""
        pass
    
    @abstractmethod
    async def check_security(self, code: str) -> List[Dict[str, Any]]:
        """Scan for security vulnerabilities."""
        pass
    
    @abstractmethod
    async def check_style(self, code: str) -> List[Dict[str, Any]]:
        """Check code style and quality."""
        pass
    
    @abstractmethod
    async def analyze_types(self, code: str) -> Dict[str, Any]:
        """Analyze type hints and usage."""
        pass
    
    @abstractmethod
    async def detect_dead_code(self, code: str) -> List[Dict[str, Any]]:
        """Detect unused/dead code."""
        pass

# Plugin registration
available_plugins = {
    'syntax_search': SyntaxAwareSearch,
    'query_optimizer': NLQueryOptimizer,
    'code_explainer': CodeExplainer,
    'function_analyzer': FunctionAnalyzer,
    'quality_analyzer': CodeQualityAnalyzer,
} 