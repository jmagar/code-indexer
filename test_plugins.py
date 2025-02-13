#!/usr/bin/env python3
"""Interactive test script for analysis plugins."""

import asyncio
from plugins.analysis.syntax_search import TreeSitterSearch
from plugins.analysis.query_optimizer import OpenAIQueryOptimizer
from plugins.analysis.code_explainer import OpenAICodeExplainer

async def main():
    # Initialize plugins
    search = TreeSitterSearch()
    optimizer = OpenAIQueryOptimizer()
    explainer = OpenAICodeExplainer()
    
    # Test syntax-aware search
    print("\nTesting Syntax-Aware Search:")
    query = "async function that handles database operations"
    print(f"Query: {query}")
    
    parsed = await search.parse_query(query)
    print("\nParsed Query:")
    print(f"Type: {parsed['type']}")
    print(f"Modifiers: {parsed['modifiers']}")
    print(f"Context: {parsed['context']}")
    
    # Test query optimization
    print("\nTesting Query Optimization:")
    original_query = "find code that saves data to database"
    print(f"Original: {original_query}")
    
    optimized = await optimizer.optimize_query(original_query)
    print(f"Optimized: {optimized}")
    
    explanation = await optimizer.explain_optimization(original_query, optimized)
    print("\nOptimization Explanation:")
    print(explanation)
    
    # Test code explanation
    print("\nTesting Code Explanation:")
    code_snippet = """
    async def save_to_db(data: Dict[str, Any]) -> bool:
        try:
            await db.collection.insert_one(data)
            return True
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False
    """
    
    explanation = await explainer.explain_snippet(code_snippet)
    print("\nCode Explanation:")
    print(explanation)

if __name__ == "__main__":
    asyncio.run(main()) 