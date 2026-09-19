"""Public, credential-safe exceptions."""

from typing import Any


class JevError(Exception):
    """Base error; ``detail`` holds a partial trace when requested."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.detail: dict[str, Any] | None = None


class ConfigurationError(JevError, ValueError):
    """Invalid configuration or missing credentials."""


class APIError(JevError):
    """HTTP or transport failure, without the provider's potentially private body."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ContextLimitError(JevError):
    """The query and smallest candidate group exceed a context budget."""


class ResponseValidationError(JevError, ValueError):
    """A successful HTTP response violated the scoring contract."""
