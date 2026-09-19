"""Tokenizer-only Gemma loading: no model weights or tensor framework."""

from pathlib import Path
from typing import Protocol

from .errors import ConfigurationError

DEFAULT_TOKENIZER = "google/embeddinggemma-300m"
DEFAULT_TOKENIZER_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"


class Tokenizer(Protocol):
    """An untruncated encoder without special tokens, and its matching decoder."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class HuggingFaceTokenizer:
    """Load tokenizer.json from a Hub repository or a local file.

    ``model_max_length`` is a counting configuration, not an enforced limit.
    Encoding deliberately includes all tokens, even beyond this length.
    """

    def __init__(
        self,
        name: str = DEFAULT_TOKENIZER,
        *,
        revision: str | None = None,
        model_max_length: int = 65536,
    ) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer as BackendTokenizer
        except ImportError:
            raise ConfigurationError(
                "Install jev-reranker[tokenizer] to use HuggingFaceTokenizer."
            ) from None
        if revision is None and name == DEFAULT_TOKENIZER:
            revision = DEFAULT_TOKENIZER_REVISION
        self.name = name
        self.revision = revision
        self.model_max_length = model_max_length
        path = Path(name)
        if not path.is_file():
            path = Path(hf_hub_download(name, "tokenizer.json", revision=revision))
        self.resolved_revision = (
            path.parent.name if path.parent.parent.name == "snapshots" else None
        )
        self._backend = BackendTokenizer.from_file(str(path))
        self._backend.no_truncation()
        self._backend.no_padding()

    def encode(self, text: str) -> list[int]:
        return self._backend.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list[int]) -> str:
        return self._backend.decode(tokens, skip_special_tokens=True)
