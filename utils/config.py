"""Configuration management using Pydantic."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.networks import AnyHttpUrl
from pydantic_core.core_schema import ValidationInfo
from functools import lru_cache


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    LMSTUDIO = "lmstudio"


class Settings(BaseModel):
    """Application settings with validation."""

    # Environment
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )

    # Collection settings
    COLLECTION_NAME: str = Field(
        default="code_reference",
        description="Name of the vector collection",
    )
    CHUNK_SIZE: int = Field(
        default=2000,
        description="Size of code chunks for processing",
        gt=0,
    )
    CHUNK_OVERLAP: int = Field(
        default=500,
        description="Overlap between chunks",
        ge=0,
    )

    # Token settings
    CHARS_PER_TOKEN: int = Field(
        default=4,
        description="Rough approximation for token estimation",
        gt=0,
    )
    MAX_TOKENS: int = Field(
        default=2048,
        description="Maximum tokens (25% of OpenAI's 8192 limit for safety)",
        gt=0,
    )
    MAX_CHARS: int = Field(
        default=8192,
        description="Maximum characters per chunk (computed from tokens)",
        gt=0,
    )

    # Batch processing
    EMBEDDING_BATCH_SIZE: int = Field(
        default=50,
        description="Batch size for embedding generation",
        gt=0,
    )
    QDRANT_BATCH_SIZE: int = Field(
        default=100,
        description="Batch size for Qdrant operations",
        gt=0,
    )

    # Search settings
    DEFAULT_MIN_SCORE: float = Field(
        default=0.7,
        description="Default minimum similarity score",
        ge=0.0,
        le=1.0,
    )
    DEFAULT_SEARCH_LIMIT: int = Field(
        default=5,
        description="Default number of search results",
        gt=0,
    )
    DEFAULT_CONTEXT_LINES: int = Field(
        default=5,
        description="Default number of context lines",
        gt=0,
    )

    # Server settings
    DEFAULT_HOST: str = Field(
        default="0.0.0.0",
        description="Default server host",
    )
    DEFAULT_PORT: int = Field(
        default=8000,
        description="Default server port",
        gt=0,
    )

    # Qdrant settings
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Qdrant server host",
    )
    QDRANT_PORT: int = Field(
        default=6550,
        description="Qdrant HTTP port",
        gt=0,
    )
    QDRANT_GRPC_PORT: int = Field(
        default=6551,
        description="Qdrant gRPC port",
        gt=0,
    )
    QDRANT_URL: Optional[AnyHttpUrl] = None
    QDRANT_TIMEOUT: int = Field(
        default=30,
        description="Qdrant timeout in seconds",
        gt=0,
    )
    QDRANT_VERIFY_SSL: bool = Field(
        default=False,
        description="Verify SSL for Qdrant connections",
    )

    # Embedding settings
    DEFAULT_EMBEDDING_PROVIDER: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="Default embedding provider",
    )

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key",
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-ada-002",
        description="OpenAI embedding model",
    )
    OPENAI_EMBEDDING_DIMENSION: int = Field(
        default=1536,
        description="OpenAI embedding dimension",
        gt=0,
    )

    # LMStudio settings
    LMSTUDIO_HOST: str = Field(
        default="localhost",
        description="LMStudio server host",
    )
    LMSTUDIO_PORT: int = Field(
        default=1234,
        description="LMStudio server port",
        gt=0,
    )
    LMSTUDIO_URL: Optional[AnyHttpUrl] = None
    LMSTUDIO_EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="LMStudio embedding dimension (all-MiniLM-L6-v2)",
        gt=0,
    )

    @field_validator("MAX_CHARS", mode="before")
    @classmethod
    def compute_max_chars(cls, value: Optional[int], info: ValidationInfo) -> int:
        """Compute maximum characters from tokens."""
        data = info.data
        chars_per_token = data.get("CHARS_PER_TOKEN", 4)
        max_tokens = data.get("MAX_TOKENS", 2048)
        if not isinstance(chars_per_token, int) or not isinstance(max_tokens, int):
            return 8192  # Default fallback
        return max_tokens * chars_per_token

    @field_validator("QDRANT_URL", mode="before")
    @classmethod
    def compute_qdrant_url(
        cls, value: Optional[AnyHttpUrl], info: ValidationInfo
    ) -> AnyHttpUrl:
        """Compute Qdrant URL from host and port."""
        data = info.data
        host = data.get("QDRANT_HOST", "localhost")
        port = data.get("QDRANT_PORT", 6550)
        if not isinstance(host, str) or not isinstance(port, int):
            return AnyHttpUrl("http://localhost:6550")
        return AnyHttpUrl(f"http://{host}:{port}")

    @field_validator("LMSTUDIO_URL", mode="before")
    @classmethod
    def compute_lmstudio_url(
        cls, value: Optional[AnyHttpUrl], info: ValidationInfo
    ) -> AnyHttpUrl:
        """Compute LMStudio URL from host and port."""
        data = info.data
        host = data.get("LMSTUDIO_HOST", "localhost")
        port = data.get("LMSTUDIO_PORT", 1234)
        if not isinstance(host, str) or not isinstance(port, int):
            return AnyHttpUrl("http://localhost:1234/v1")
        return AnyHttpUrl(f"http://{host}:{port}/v1")

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
        extra="allow",
        json_schema_extra={
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "case_sensitive": True,
            "use_enum_values": True,
        },
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
