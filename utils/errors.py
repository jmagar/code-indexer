"""Custom error types for the application."""


class NetworkError(Exception):
    """Raised when network operations fail."""

    pass


class ProcessingError(Exception):
    """Raised when code processing operations fail."""

    pass


class SourceError(Exception):
    """Raised when source management operations fail."""

    pass


class WatcherError(Exception):
    """Raised when repository watching operations fail."""

    pass


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass
