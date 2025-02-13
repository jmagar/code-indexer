"""Natural language query optimization for better code search results."""

import json
from typing import Dict, Optional, Tuple

from openai import AsyncOpenAI

from . import NLQueryOptimizer
from ..logger import IndexerLogger

logger = IndexerLogger(__name__).get_logger()

QUERY_OPTIMIZATION_PROMPT = """
You are a code search query optimizer. Your task is to enhance natural language queries to better find relevant code.
Given a search query, optimize it by:
1. Identifying key programming concepts
2. Adding relevant technical terms
3. Including common variations/synonyms
4. Removing irrelevant words
5. Structuring for better semantic matching

Original query: {query}

Respond in JSON format:
{
    "optimized_query": "enhanced search query",
    "added_terms": ["list", "of", "added", "terms"],
    "removed_terms": ["list", "of", "removed", "terms"],
    "explanation": "explanation of changes"
}
"""

class OpenAIQueryOptimizer(NLQueryOptimizer):
    """Query optimization using OpenAI's language models."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4-0125-preview"  # Using latest GPT-4 for best results
        
    async def optimize_query(self, query: str) -> str:
        """Optimize natural language query for code search.
        
        Args:
            query: Original search query
            
        Returns:
            Optimized query string
        """
        try:
            # Get optimization suggestions
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code search query optimizer."},
                    {"role": "user", "content": QUERY_OPTIMIZATION_PROMPT.format(query=query)}
                ],
                temperature=0.3,  # Lower temperature for more focused results
                max_tokens=200
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Log optimization details
            logger.info(
                "Query optimized",
                original=query,
                optimized=result["optimized_query"],
                added_terms=result["added_terms"],
                removed_terms=result["removed_terms"]
            )
            
            return result["optimized_query"]
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return query  # Return original query if optimization fails
            
    async def explain_optimization(self, original: str, optimized: str) -> str:
        """Explain how the query was optimized.
        
        Args:
            original: Original search query
            optimized: Optimized search query
            
        Returns:
            Natural language explanation of changes
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code search query optimizer."},
                    {
                        "role": "user",
                        "content": f"""
                        Explain how this code search query was optimized:
                        Original: "{original}"
                        Optimized: "{optimized}"
                        
                        Explain:
                        1. What technical terms were added and why
                        2. What terms were removed and why
                        3. How the structure was improved
                        4. How this will help find better code matches
                        """
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Optimization explanation failed: {e}")
            return "Could not generate optimization explanation."
            
    async def analyze(self, code: str, language: str) -> Dict[str, str]:
        """Implement required analyze method from base class."""
        return {
            "supported_models": [self.model],
            "optimization_enabled": True,
            "explanation_enabled": True
        } 