"""Synchronous multilingual reranking through TypeSafe Jev."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import platform
import string
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

from . import _client
from ._ranking import SHUFFLE_SEED, score_listwise
from .errors import ConfigurationError, ContextLimitError, JevError
from .tokenization import (
    DEFAULT_TOKENIZER,
    DEFAULT_TOKENIZER_REVISION,
    HuggingFaceTokenizer,
    Tokenizer,
)

INSTRUCTIONS = "Does {document} help answer `query`? Prefer passages with the specific facts needed."
PAIRWISE_INSTRUCTIONS = (
    "Does {left} help answer `query` better than {right}? "
    "Prefer passages containing the specific facts needed to answer the query."
)
CRITERIA = {
    "true": "Contains specific information that answers or is necessary for answering the query",
    "false": "Unrelated, only tangentially related, or lacks the needed facts",
}
PAIRWISE_CRITERIA = {
    "true": "The first passage provides more of the specific information needed to answer the query",
    "false": "The second passage provides more of the specific information needed to answer the query",
}


def positive_int(name: str, value: Any, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}.")


def nonempty(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a nonempty string.")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class _Run:
    detailed: bool
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "requests": 0,
            "attempts": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "max_tokens_errors": 0,
        }
    )
    requests: list[dict[str, Any]] = field(default_factory=list)
    models: set[str] = field(default_factory=set)
    splits: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    lock: Any = field(default_factory=Lock)


class JevReranker:
    """Rank document strings with listwise, pointwise, or pairwise Jev judgments.

    Explicit arguments override environment/.env configuration. Unknown keyword
    arguments fail rather than silently changing scoring semantics. A supplied
    httpx client is borrowed: closing this ranker does not close that client.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        mode: str = "listwise",
        api_key: str | None = None,
        api_key_env: str = "TYPESAFE_API_KEY",
        dotenv_path: str | Path | None = ".env",
        endpoint: str | None = None,
        max_concurrency: int | None = None,
        timeout: float = 180.0,
        max_retries: int = 8,
        document_max_tokens: int | None = 4000,
        split_state_token_budget: int = 26000,
        split_request_token_budget: int = 48000,
        split_tokenizer_name: str = DEFAULT_TOKENIZER,
        split_tokenizer_revision: str | None = None,
        tokenizer_max_length: int = 65536,
        tokenizer: Tokenizer | None = None,
        instructions: str | None = None,
        criteria: dict[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if mode not in ("listwise", "pointwise", "pairwise"):
            raise ConfigurationError("mode must be listwise, pointwise, or pairwise.")
        max_concurrency = (
            (4 if mode == "listwise" else 20)
            if max_concurrency is None
            else max_concurrency
        )
        for name, value in (
            ("max_concurrency", max_concurrency),
            ("split_state_token_budget", split_state_token_budget),
            ("split_request_token_budget", split_request_token_budget),
            ("tokenizer_max_length", tokenizer_max_length),
        ):
            positive_int(name, value)
        positive_int("max_retries", max_retries, minimum=0)
        if document_max_tokens is not None:
            positive_int("document_max_tokens", document_max_tokens)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ConfigurationError("timeout must be positive and finite.")
        nonempty("api_key_env", api_key_env)
        nonempty("split_tokenizer_name", split_tokenizer_name)
        if split_tokenizer_revision is not None:
            nonempty("split_tokenizer_revision", split_tokenizer_revision)
        if tokenizer is not None and not all(
            callable(getattr(tokenizer, m, None)) for m in ("encode", "decode")
        ):
            raise ConfigurationError(
                "tokenizer must implement encode(text) and decode(tokens)."
            )
        file_env = (
            dotenv_values(dotenv_path, interpolate=False)
            if dotenv_path is not None
            else {}
        )

        def env(name: str) -> str | None:
            return os.environ.get(name) or file_env.get(name)

        api_key = api_key if api_key is not None else env(api_key_env)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError(f"Set {api_key_env} or pass api_key to use Jev.")
        model = model if model is not None else env("JEV_MODEL") or "jev-latest"
        nonempty("model", model)
        endpoint = (
            endpoint
            if endpoint is not None
            else env("TYPESAFE_ENDPOINT") or "https://api.typesafe.ai/v1/systemone"
        )
        nonempty("endpoint", endpoint)
        try:
            url = urlsplit(endpoint)
            allowed = url.scheme == "https" or (
                url.scheme == "http"
                and url.hostname in ("localhost", "127.0.0.1", "::1")
            )
            if (
                not allowed
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError
            _ = url.port
        except ValueError:
            raise ConfigurationError(
                "endpoint must be an HTTPS URL without credentials, query, or fragment."
            ) from None
        instructions = (
            instructions
            if instructions is not None
            else (PAIRWISE_INSTRUCTIONS if mode == "pairwise" else INSTRUCTIONS)
        )
        nonempty("instructions", instructions)
        fields = {"left", "right"} if mode == "pairwise" else {"document"}
        try:
            parsed = list(string.Formatter().parse(instructions))
            actual = {f for _, f, _, _ in parsed if f is not None}
            if actual != fields or any(
                spec or conversion for _, _, spec, conversion in parsed
            ):
                raise ValueError
            instructions.format(**dict.fromkeys(fields, "reference"))
        except (ValueError, KeyError, IndexError, AttributeError):
            raise ConfigurationError(
                f"instructions must use only these placeholders: {sorted(fields)}."
            ) from None
        criteria = copy.deepcopy(
            criteria
            if criteria is not None
            else (PAIRWISE_CRITERIA if mode == "pairwise" else CRITERIA)
        )
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise ConfigurationError(
                "criteria must contain true and false descriptions."
            )
        for value in criteria.values():
            nonempty("criteria description", value)
        self.model, self.mode, self.endpoint = model, mode, endpoint
        self.max_concurrency, self.timeout, self.max_retries = (
            max_concurrency,
            timeout,
            max_retries,
        )
        self.document_max_tokens = document_max_tokens
        self.split_state_token_budget, self.split_request_token_budget = (
            split_state_token_budget,
            split_request_token_budget,
        )
        if (
            split_tokenizer_revision is None
            and split_tokenizer_name == DEFAULT_TOKENIZER
            and tokenizer is None
        ):
            split_tokenizer_revision = DEFAULT_TOKENIZER_REVISION
        self.split_tokenizer_name, self.split_tokenizer_revision = (
            split_tokenizer_name,
            split_tokenizer_revision,
        )
        self.tokenizer_max_length = tokenizer_max_length
        self.instructions, self.criteria = instructions, criteria
        self._api_key = api_key
        self._tokenizer = tokenizer
        self._custom_tokenizer = tokenizer is not None
        self._tokenizer_lock = Lock()
        self._semaphore = BoundedSemaphore(max_concurrency)
        self._client = (
            client
            if client is not None
            else httpx.Client(
                limits=httpx.Limits(
                    max_connections=max_concurrency,
                    max_keepalive_connections=max_concurrency,
                )
            )
        )
        self._owns_client = client is None
        self._closed = False

    def close(self) -> None:
        """Close owned HTTP resources. Borrowed clients remain open."""
        if not self._closed:
            self._closed = True
            if self._owns_client:
                self._client.close()

    def __enter__(self) -> Self:
        if self._closed:
            raise ConfigurationError("JevReranker is closed.")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> list[dict[str, Any]]:
        """Return sorted results from :meth:`raw_rerank`."""
        return self.raw_rerank(
            query,
            documents,
            top_k=top_k,
            return_documents=return_documents,
            detail=detail,
        )["results"]

    rank = rerank

    def raw_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Return ``results`` and, optionally, a JSON-serializable execution ``detail``.

        detail includes the text sent to the API. It contains no authentication
        headers or API key. No files are written automatically.
        """
        if self._closed:
            raise ConfigurationError("JevReranker is closed.")
        nonempty("query", query)
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of strings.")
        docs = list(documents)
        if not all(isinstance(d, str) for d in docs):
            raise TypeError("Every document must be a string.")
        if top_k is not None:
            positive_int("top_k", top_k, minimum=0)
        if not isinstance(detail, bool) or not isinstance(return_documents, bool):
            raise TypeError("detail and return_documents must be bool.")
        started, start_utc = time.monotonic(), datetime.now(UTC).isoformat()
        run = _Run(detail)
        try:
            if not docs or top_k == 0:
                scores: list[float] = []
            else:
                prepared = self._prepare(docs, run)
                scores = self._score(query, prepared, run)
            results = []
            for index in sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]:
                result: dict[str, Any] = {"corpus_id": index, "score": scores[index]}
                if return_documents:
                    result["text"] = docs[index]
                if detail:
                    result["detail"] = copy.deepcopy(run.documents[index])
                results.append(result)
            raw: dict[str, Any] = {"results": results}
            if detail:
                raw["detail"] = self._detail(
                    query, docs, run, start_utc, started, "success"
                )
            return raw
        except JevError as exc:
            if detail:
                exc.detail = self._detail(
                    query, docs, run, start_utc, started, "failed"
                )
            raise

    def _get_tokenizer(self) -> Tokenizer:
        # Caller owns _tokenizer_lock; tokenizers can have mutable configuration.
        if self._tokenizer is None:
            try:
                self._tokenizer = HuggingFaceTokenizer(
                    self.split_tokenizer_name,
                    revision=self.split_tokenizer_revision,
                    model_max_length=self.tokenizer_max_length,
                )
            except Exception:  # noqa: BLE001 - tokenizers raises a generic Exception for invalid files
                raise ConfigurationError(
                    "Could not load tokenizer. Check the tokenizer name/revision, HF access and cache, "
                    "accept the Gemma license if required, or supply a local/custom tokenizer."
                ) from None
        return self._tokenizer

    def _count(self, text: str) -> int:
        with self._tokenizer_lock:
            return len(self._get_tokenizer().encode(text))

    def _prepare(self, docs: list[str], run: _Run) -> list[str]:
        prepared = []
        with self._tokenizer_lock:
            tokenizer = self._get_tokenizer()
            for index, original in enumerate(docs):
                tokens = tokenizer.encode(original)
                original_count = len(tokens)
                text = original
                limit = self.document_max_tokens
                if limit is not None and original_count > limit:
                    tokens = tokens[:limit]
                    while True:
                        text = tokenizer.decode(tokens)
                        sent_count = len(tokenizer.encode(text))
                        if sent_count <= limit:
                            break
                        if not tokens:
                            raise ConfigurationError(
                                "Tokenizer cannot produce text within document_max_tokens."
                            )
                        tokens = tokens[
                            : max(0, len(tokens) - max(1, sent_count - limit))
                        ]
                sent_count = len(tokenizer.encode(text))
                prepared.append(text)
                if run.detailed:
                    run.documents.append(
                        {
                            "document_index": index,
                            "sha256": sha(original),
                            "original_tokens": original_count,
                            "sent_tokens": sent_count,
                            "truncated": text != original,
                            "request_ids": [],
                            "comparisons": [],
                        }
                    )
        return prepared

    def _question(self, **references: str) -> dict[str, Any]:
        return {
            "type": "noul",
            "instructions": self.instructions.format(**references),
            "criteria": self.criteria,
        }

    def _payload(
        self, query: str, docs: list[str], indices: list[int]
    ) -> dict[str, Any]:
        if self.mode == "pairwise":
            state = {
                "query": query,
                "left": docs[indices[0]],
                "right": docs[indices[1]],
            }
            questions = {
                "left_wins": self._question(left="`left`", right="`right`"),
                "right_wins": self._question(left="`right`", right="`left`"),
            }
        elif self.mode == "pointwise":
            state = {"query": query, "document": docs[indices[0]]}
            questions = {"relevant": self._question(document="`document`")}
        else:
            state = {
                "query": query,
                "documents": {f"doc_{i}": docs[i] for i in indices},
            }
            questions = {
                f"doc_{i}": self._question(document=f"`documents.doc_{i}`")
                for i in indices
            }
        return {"model": self.model, "state": state, "questions": questions}

    def _estimate(self, payload: dict[str, Any]) -> dict[str, int]:
        state = self._count(_client.json_text(payload["state"]))
        questions = [
            self._count(_client.json_text(q)) for q in payload["questions"].values()
        ]
        return {
            "state_plus_longest_question": state + max(questions),
            "request": self._count(_client.json_text(payload)),
        }

    def _request(
        self, payload: dict[str, Any], indices: list[int], run: _Run
    ) -> list[float]:
        trace: dict[str, Any] = {}
        if run.detailed:
            with run.lock:
                trace.update(
                    id=len(run.requests),
                    document_indices=indices,
                    payload=copy.deepcopy(payload),
                    estimated_tokens=self._estimate(payload),
                )
                run.requests.append(trace)
                for i in indices:
                    run.documents[i]["request_ids"].append(trace["id"])
        try:
            with self._semaphore:
                scores, data = _client.request(
                    client=self._client,
                    endpoint=self.endpoint,
                    api_key=self._api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    payload=payload,
                    trace=trace,
                )
            with run.lock:
                run.usage["requests"] += 1
                run.usage["input_tokens"] += data["usage"]["input_tokens"]
                run.usage["output_tokens"] += data["usage"]["output_tokens"]
                run.models.add(data["model"])
                if run.detailed:
                    trace["response"] = data
            return scores
        finally:
            with run.lock:
                run.usage["attempts"] += trace.get("attempts", 0)
                run.usage["retries"] += trace.get("retries", 0)
                run.usage["max_tokens_errors"] += (
                    trace.get("status") == "max_tokens_exceeded"
                )

    def _score(self, query: str, docs: list[str], run: _Run) -> list[float]:
        def payload(indices: list[int]) -> dict[str, Any]:
            return self._payload(query, docs, indices)

        def score_one(indices: list[int]) -> list[float]:
            body = payload(indices)
            estimate = self._estimate(body)
            if (
                estimate["state_plus_longest_question"] > self.split_state_token_budget
                or estimate["request"] > self.split_request_token_budget
            ):
                raise ContextLimitError(
                    f"Query and document group {indices} exceed the estimated context budget."
                )
            return self._request(body, indices, run)

        if self.mode == "listwise":
            return score_listwise(
                query=query,
                lengths=[self._count(d) for d in docs],
                state_budget=self.split_state_token_budget,
                request_budget=self.split_request_token_budget,
                estimate=lambda ids: self._estimate(payload(ids)),
                request=lambda ids: self._request(payload(ids), ids, run),
                splits=run.splits,
            )
        if self.mode == "pointwise":
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                return [
                    values[0]
                    for values in pool.map(score_one, ([i] for i in range(len(docs))))
                ]
        if len(docs) == 1:
            return [0.5]
        scores = [0.0] * len(docs)

        def compare(pair: tuple[int, int]) -> tuple[tuple[int, int], list[float]]:
            return pair, score_one(list(pair))

        # Bound queued futures too: O(n²) comparisons must not create O(n²) futures.
        pairs = iter(combinations(range(len(docs)), 2))
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            from itertools import islice

            while batch := list(islice(pairs, self.max_concurrency)):
                for (left, right), (forward, reverse) in pool.map(compare, batch):
                    win = (forward + 1 - reverse) / 2
                    scores[left] += win
                    scores[right] += 1 - win
                    if run.detailed:
                        run.documents[left]["comparisons"].append(
                            {
                                "opponent": right,
                                "win_probability": win,
                                "forward_noul": forward,
                                "reverse_noul": reverse,
                            }
                        )
                        run.documents[right]["comparisons"].append(
                            {
                                "opponent": left,
                                "win_probability": 1 - win,
                                "forward_noul": reverse,
                                "reverse_noul": forward,
                            }
                        )
        return [score / (len(docs) - 1) for score in scores]

    def _detail(
        self,
        query: str,
        docs: list[str],
        run: _Run,
        start: str,
        started: float,
        status: str,
    ) -> dict[str, Any]:
        tokenizer = self._tokenizer
        data = {
            "schema_version": 1,
            "status": status,
            "started_at": start,
            "elapsed_seconds": time.monotonic() - started,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.system(),
                "package_version": version("jev-reranker"),
                "dependencies": {
                    name: version(name)
                    for name in (
                        "httpx",
                        "tokenizers",
                        "huggingface-hub",
                        "python-dotenv",
                    )
                },
            },
            "configuration": {
                "model": self.model,
                "mode": self.mode,
                "endpoint": self.endpoint,
                "instructions": self.instructions,
                "criteria": self.criteria,
                "max_concurrency": self.max_concurrency,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "document_max_tokens": self.document_max_tokens,
                "split_state_token_budget": self.split_state_token_budget,
                "split_request_token_budget": self.split_request_token_budget,
                "tokenizer": (
                    f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
                    if self._custom_tokenizer
                    else self.split_tokenizer_name
                ),
                "tokenizer_revision": self.split_tokenizer_revision,
                "tokenizer_resolved_revision": getattr(
                    tokenizer, "resolved_revision", None
                ),
                "tokenizer_max_length": getattr(
                    tokenizer, "model_max_length", self.tokenizer_max_length
                ),
                "estimates_are_provider_tokens": False,
                "shuffle_seed": SHUFFLE_SEED,
                "tie_break": "stable_input_order",
                "score_semantics": "mean_pairwise_win_probability"
                if self.mode == "pairwise"
                else "noul",
            },
            "query_sha256": sha(query),
            "document_count": len(docs),
            "documents": run.documents,
            "document_sha256": [sha(d) for d in docs],
            "resolved_models": sorted(run.models),
            "usage": run.usage,
            "requests": run.requests,
            "splits": run.splits,
        }
        # Defense in depth against an upstream response echoing the credential.
        return self._redact(copy.deepcopy(data))

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        if isinstance(value, dict):
            return {self._redact(k): self._redact(v) for k, v in value.items()}
        return value
