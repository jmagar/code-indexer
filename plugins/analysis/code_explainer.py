"""Code explanation generation using language models."""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from . import CodeExplainer
from ..logger import IndexerLogger

logger = IndexerLogger(__name__).get_logger()

SNIPPET_EXPLANATION_PROMPT = """
You are a code explanation expert. Explain this code snippet clearly and concisely:

```{language}
{code}
```

Context: {context}

Explain:
1. What the code does
2. Key components and their purpose
3. Important patterns or techniques used
4. Any notable dependencies or assumptions
5. Potential edge cases or limitations

Keep the explanation technical but clear.
"""

FUNCTION_EXPLANATION_PROMPT = """
Explain this function in detail:

```{language}
{code}
```

Function name: {function_name}

Provide:
1. Purpose and functionality
2. Parameters and return values
3. Implementation details
4. Usage examples
5. Error handling
6. Performance considerations

Focus on practical understanding and usage.
"""

class OpenAICodeExplainer(CodeExplainer):
    """Code explanation using OpenAI's language models."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4-0125-preview"  # Using latest GPT-4 for best results
        self.supported_languages = {'python', 'javascript', 'typescript', 'go', 'rust'}
        
    async def explain_snippet(self, code: str, context: Optional[str] = None) -> str:
        """Generate natural language explanation of code snippet.
        
        Args:
            code: Code snippet to explain
            context: Optional context about the code
            
        Returns:
            Natural language explanation
        """
        try:
            # Detect language (simplified)
            language = self._detect_language(code)
            
            # Generate explanation
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code explanation expert."},
                    {
                        "role": "user",
                        "content": SNIPPET_EXPLANATION_PROMPT.format(
                            language=language,
                            code=code,
                            context=context or "No additional context provided."
                        )
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Code explanation failed: {e}")
            return "Could not generate code explanation."
            
    async def explain_function(self, code: str, function_name: str) -> str:
        """Generate explanation of specific function.
        
        Args:
            code: Full code containing the function
            function_name: Name of function to explain
            
        Returns:
            Natural language explanation
        """
        try:
            # Extract function code
            function_code = self._extract_function(code, function_name)
            if not function_code:
                return f"Could not find function '{function_name}' in the code."
                
            # Detect language
            language = self._detect_language(code)
            
            # Generate explanation
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code explanation expert."},
                    {
                        "role": "user",
                        "content": FUNCTION_EXPLANATION_PROMPT.format(
                            language=language,
                            code=function_code,
                            function_name=function_name
                        )
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Function explanation failed: {e}")
            return f"Could not generate explanation for function '{function_name}'."
            
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code snippet."""
        # Simple heuristics - would need more robust detection
        if re.search(r'\bdef\s+\w+\s*\(', code):
            return 'python'
        elif re.search(r'\bfunction\s+\w+\s*\(', code):
            return 'javascript'
        elif re.search(r'\bfn\s+\w+\s*\(', code):
            return 'rust'
        elif re.search(r'\bfunc\s+\w+\s*\(', code):
            return 'go'
        return 'unknown'
        
    def _extract_function(self, code: str, function_name: str) -> Optional[str]:
        """Extract function definition from code."""
        try:
            # Parse Python code
            tree = ast.parse(code)
            
            # Find function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get function source
                    lines = code.split('\n')
                    return '\n'.join(lines[node.lineno-1:node.end_lineno])
                    
            return None
            
        except Exception as e:
            logger.error(f"Function extraction failed: {e}")
            return None
            
    async def analyze(self, code: str, language: str) -> Dict[str, Any]:
        """Implement required analyze method from base class."""
        return {
            "supported_languages": list(self.supported_languages),
            "model": self.model,
            "can_explain_snippets": True,
            "can_explain_functions": True
        } 