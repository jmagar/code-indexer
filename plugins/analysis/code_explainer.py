"""Code explanation generation using language models."""

import ast
import re
import time
from typing import Any, Dict, Optional, Tuple

from openai import AsyncOpenAI

from . import CodeExplainer
from ..logger import IndexerLogger

logger = IndexerLogger(__name__).get_logger()

# Cache configuration
CACHE_TTL = 3600  # 1 hour in seconds
CACHE_SIZE = 1000  # Maximum number of cached items


class ExplanationCache:
    """Simple cache with TTL for code explanations."""

    def __init__(self, ttl: int = CACHE_TTL, max_size: int = CACHE_SIZE):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, Tuple[str, float]] = {}

    def get(self, key: str) -> Optional[str]:
        """Get cached explanation if not expired."""
        if key in self.cache:
            explanation, timestamp = self.cache[key]
            if time.time() - timestamp <= self.ttl:
                return explanation
            # Remove expired entry
            del self.cache[key]
        return None

    def set(self, key: str, value: str) -> None:
        """Cache explanation with timestamp."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            del self.cache[oldest[0]]
        self.cache[key] = (value, time.time())


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
        self.supported_languages = {"python", "javascript", "typescript", "go", "rust"}
        self.cache = ExplanationCache()

    def _get_cache_key(self, code: str, context: Optional[str] = None) -> str:
        """Generate cache key from code and context."""
        # Normalize code by removing whitespace
        normalized_code = re.sub(r"\s+", " ", code.strip())
        if context:
            normalized_context = re.sub(r"\s+", " ", context.strip())
            return f"{normalized_code}::{normalized_context}"
        return normalized_code

    async def explain_snippet(self, code: str, context: Optional[str] = None) -> str:
        """Generate natural language explanation of code snippet with caching."""
        try:
            # Check cache first
            cache_key = self._get_cache_key(code, context)
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("Using cached explanation")
                return cached

            # Generate new explanation
            language = self._detect_language(code)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code explanation expert."},
                    {
                        "role": "user",
                        "content": SNIPPET_EXPLANATION_PROMPT.format(
                            language=language,
                            code=code,
                            context=context or "No additional context provided.",
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=500,
            )

            explanation = response.choices[0].message.content
            if not explanation:
                raise ValueError("Empty response from OpenAI")

            # Cache the result
            self.cache.set(cache_key, explanation)
            return explanation

        except Exception as e:
            logger.error(f"Code explanation failed: {e}")
            return "Could not generate code explanation."

    async def explain_function(self, code: str, function_name: str) -> str:
        """Generate explanation of specific function with caching."""
        try:
            # Extract function code
            function_code = self._extract_function(code, function_name)
            if not function_code:
                return f"Could not find function '{function_name}' in the code."

            # Check cache first
            cache_key = self._get_cache_key(function_code, function_name)
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("Using cached function explanation")
                return cached

            # Generate new explanation
            language = self._detect_language(code)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code explanation expert."},
                    {
                        "role": "user",
                        "content": FUNCTION_EXPLANATION_PROMPT.format(
                            language=language,
                            code=function_code,
                            function_name=function_name,
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=500,
            )

            explanation = response.choices[0].message.content
            if not explanation:
                raise ValueError("Empty response from OpenAI")

            # Cache the result
            self.cache.set(cache_key, explanation)
            return explanation

        except Exception as e:
            logger.error(f"Function explanation failed: {e}")
            return f"Could not generate explanation for function '{function_name}'."

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code snippet."""
        # Simple heuristics - would need more robust detection
        if re.search(r"\bdef\s+\w+\s*\(", code):
            return "python"
        elif re.search(r"\bfunction\s+\w+\s*\(", code):
            return "javascript"
        elif re.search(r"\bfn\s+\w+\s*\(", code):
            return "rust"
        elif re.search(r"\bfunc\s+\w+\s*\(", code):
            return "go"
        return "unknown"

    def _extract_function(self, code: str, function_name: str) -> Optional[str]:
        """Extract function definition from code."""
        try:
            # Parse Python code
            tree = ast.parse(code)

            # Find function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get function source
                    lines = code.split("\n")
                    return "\n".join(lines[node.lineno - 1 : node.end_lineno])

            return None

        except Exception as e:
            logger.error(f"Function extraction failed: {e}")
            return None

    async def analyze(self, code: str, language: str, **kwargs: Any) -> Dict[str, Any]:
        """Analyze code and return results.

        Args:
            code: The code to analyze
            language: Programming language of the code
            **kwargs: Additional keyword arguments

        Returns:
            Dict containing analysis results
        """
        return {
            "supported_languages": list(self.supported_languages),
            "model": self.model,
            "can_explain_snippets": True,
            "can_explain_functions": True,
        }

    async def process(self, data: Any, **kwargs: Any) -> Any:
        """Process data using the plugin.

        Args:
            data: Input data to process
            **kwargs: Additional keyword arguments

        Returns:
            Processed data
        """
        if isinstance(data, str):
            return await self.explain_snippet(data, context=kwargs.get("context"))
        return None
