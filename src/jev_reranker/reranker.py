"""Async-first multilingual reranking through TypeSafe Jev."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
import os
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations, islice
from pathlib import Path
from threading import Lock
from typing import Any, Self
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

from . import _client
from ._ranking import SHUFFLE_SEED, score_listwise
from ._runtime import Runtime, require_sync_context
from .errors import ConfigurationError, ContextLimitError, JevError
from .instructions import (
    _LISTWISE_RELEVANCE_INSTRUCTION,
    _LISTWISE_RELEVANCE_REFERENCE,
    CRITERIA,
    INSTRUCTIONS,
    PAIRWISE_CRITERIA,
    PAIRWISE_INSTRUCTIONS,
    POINTWISE_RELEVANCE_INSTRUCTION,
    RELEVANCE_INSTRUCTION,
    validate_instruction,
)
from .tokenization import (
    DEFAULT_TOKENIZER,
    DEFAULT_TOKENIZER_REVISION,
    HuggingFaceTokenizer,
    Tokenizer,
)


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
    instruction: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.0
    selection: dict[str, Any] = field(default_factory=dict)
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


class _Unset(Enum):
    VALUE = "unset"


def _legacy_limit(name: str, value: Any, legacy: Any, default: Any) -> Any:
    if legacy is _Unset.VALUE:
        return value
    if value != default and value != legacy:
        raise ConfigurationError(f"Conflicting {name} and legacy token limit.")
    return legacy


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("httpx", "python-dotenv", "tokenizers", "huggingface-hub"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


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
        document_max_length: int | None = 4000,
        split_state_budget: int = 26000,
        split_request_budget: int = 48000,
        split_tokenizer_name: str | None = None,
        split_tokenizer_revision: str | None = None,
        tokenizer_max_length: int = 65536,
        tokenizer: Tokenizer | str | None = None,
        length_fn: Callable[[str], int] | None = None,
        document_max_tokens: int | None | _Unset = _Unset.VALUE,
        split_state_token_budget: int | _Unset = _Unset.VALUE,
        split_request_token_budget: int | _Unset = _Unset.VALUE,
        instruction: dict[str, Any] | None = None,
        instructions: str | None = None,
        criteria: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        document_max_length = _legacy_limit(
            "document_max_length", document_max_length, document_max_tokens, 4000
        )
        split_state_budget = _legacy_limit(
            "split_state_budget", split_state_budget, split_state_token_budget, 26000
        )
        split_request_budget = _legacy_limit(
            "split_request_budget",
            split_request_budget,
            split_request_token_budget,
            48000,
        )
        if isinstance(tokenizer, str):
            if split_tokenizer_name is not None:
                raise ConfigurationError(
                    "Pass tokenizer or split_tokenizer_name, not both."
                )
            split_tokenizer_name, tokenizer = tokenizer, None
        elif tokenizer is not None and split_tokenizer_name is not None:
            raise ConfigurationError(
                "Pass a tokenizer object or a tokenizer name, not both."
            )
        if length_fn is not None and not callable(length_fn):
            raise ConfigurationError("length_fn must be callable: (str) -> int.")
        if length_fn is not None and (
            tokenizer is not None or split_tokenizer_name is not None
        ):
            raise ConfigurationError("Pass length_fn or tokenizer, not both.")
        if split_tokenizer_revision is not None and split_tokenizer_name is None:
            raise ConfigurationError(
                "split_tokenizer_revision requires a tokenizer name."
            )
        if client is not None and not isinstance(client, httpx.AsyncClient):
            raise ConfigurationError(
                "client must be an httpx.AsyncClient; use transport= for custom transports."
            )
        if client is not None and transport is not None:
            raise ConfigurationError("Pass either client or transport, not both.")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise ConfigurationError(
                "transport must support asynchronous HTTP requests."
            )
        if mode not in ("listwise", "pointwise", "pairwise"):
            raise ConfigurationError("mode must be listwise, pointwise, or pairwise.")
        max_concurrency = (
            (4 if mode == "listwise" else 20)
            if max_concurrency is None
            else max_concurrency
        )
        for name, value in (
            ("max_concurrency", max_concurrency),
            ("split_state_budget", split_state_budget),
            ("split_request_budget", split_request_budget),
            ("tokenizer_max_length", tokenizer_max_length),
        ):
            positive_int(name, value)
        positive_int("max_retries", max_retries, minimum=0)
        if document_max_length is not None:
            positive_int("document_max_length", document_max_length)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ConfigurationError("timeout must be positive and finite.")
        nonempty("api_key_env", api_key_env)
        if split_tokenizer_name is not None:
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
        if instruction is not None and (
            instructions is not None or criteria is not None
        ):
            raise ConfigurationError(
                "Pass instruction or legacy instructions/criteria, not both."
            )
        prompt = validate_instruction(
            instruction
            if instruction is not None
            else {
                "instructions": instructions
                if instructions is not None
                else (PAIRWISE_INSTRUCTIONS if mode == "pairwise" else INSTRUCTIONS),
                "criteria": criteria
                if criteria is not None
                else (PAIRWISE_CRITERIA if mode == "pairwise" else CRITERIA),
            },
            mode,
        )
        instructions, criteria = prompt["instructions"], prompt["criteria"]
        self.model, self.mode, self.endpoint = model, mode, endpoint
        self.max_concurrency, self.timeout, self.max_retries = (
            max_concurrency,
            timeout,
            max_retries,
        )
        self.document_max_length = document_max_length
        self.split_state_budget, self.split_request_budget = (
            split_state_budget,
            split_request_budget,
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
        uses_tokenizer = tokenizer is not None or split_tokenizer_name is not None
        self._length_fn = (
            None if uses_tokenizer else (len if length_fn is None else length_fn)
        )
        self.length_unit = (
            "tokens"
            if uses_tokenizer
            else ("characters" if self._length_fn is len else "custom")
        )
        self._tokenizer_lock = Lock()
        self._runtime = Runtime(max_concurrency, client, transport)

    def close(self) -> None:
        """Optionally disable the instance and wait for active calls to finish."""
        self._runtime.close()

    async def aclose(self) -> None:
        """Optionally disable the instance and await active calls; no idle pool is held."""
        await self._runtime.aclose()

    def __enter__(self) -> Self:
        require_sync_context()
        if self._runtime.closed or self._runtime.closing:
            raise ConfigurationError("JevReranker is closed or closing.")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        self._runtime.bind()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def relevance_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.2,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Rank evidence usefulness with the relevance prompt; default cutoff 0.2."""
        self._check_relevance_mode()
        if instruction is None:
            instruction = (
                POINTWISE_RELEVANCE_INSTRUCTION
                if self.mode == "pointwise"
                else RELEVANCE_INSTRUCTION
            )
        return self.rerank(
            query,
            documents,
            instruction=instruction,
            threshold=threshold,
            top_k=top_k,
            return_documents=return_documents,
            detail=detail,
        )

    def _check_relevance_mode(self) -> None:
        if self.mode == "pairwise":
            raise ConfigurationError(
                "relevance_rerank requires listwise or pointwise, not pairwise win probabilities."
            )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.0,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Blocking common scoring pipeline with optional complete execution detail."""
        return self._runtime.run(
            lambda: self.a_rerank(
                query,
                documents,
                instruction=instruction,
                threshold=threshold,
                top_k=top_k,
                return_documents=return_documents,
                detail=detail,
            )
        )

    async def a_relevance_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.2,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Async relevance scoring and filtering through the shared rerank pipeline."""
        self._check_relevance_mode()
        if instruction is None:
            instruction = (
                POINTWISE_RELEVANCE_INSTRUCTION
                if self.mode == "pointwise"
                else RELEVANCE_INSTRUCTION
            )
        return await self.a_rerank(
            query,
            documents,
            instruction=instruction,
            threshold=threshold,
            top_k=top_k,
            return_documents=return_documents,
            detail=detail,
        )

    async def a_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.0,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Score documents and return results plus optional complete execution logs.

        Cancellation drains pending HTTP tasks. Detail includes rejected document
        scores and sent text, but never credentials. No logs are written to disk.
        """
        async with self._runtime.operation():
            return await self._rerank(
                query,
                documents,
                instruction=instruction,
                threshold=threshold,
                top_k=top_k,
                return_documents=return_documents,
                detail=detail,
            )

    async def _rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: dict[str, Any] | None = None,
        threshold: float = 0.0,
        top_k: int | None = None,
        return_documents: bool = True,
        detail: bool = False,
    ) -> dict[str, Any]:
        """Return ``results`` and, optionally, a JSON-serializable execution ``detail``.

        detail includes the text sent to the API. It contains no authentication
        headers or API key. No files are written automatically.
        """
        prompt = validate_instruction(
            {"instructions": self.instructions, "criteria": self.criteria}
            if instruction is None
            else instruction,
            self.mode,
        )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1
        ):
            raise ConfigurationError(
                "threshold must be a finite number between 0 and 1."
            )
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
        run = _Run(detail, instruction=prompt, threshold=float(threshold))
        try:
            if not docs or top_k == 0:
                scores: list[float] = []
            else:
                prepared = await asyncio.to_thread(self._prepare, docs, run)
                scores = await self._score(query, prepared, run)
            eligible = [
                i
                for i in sorted(range(len(scores)), key=lambda i: -scores[i])
                if scores[i] >= threshold
            ]
            selected = eligible[:top_k]
            run.selection = {
                "top_k": top_k,
                "scored_count": len(scores),
                "above_threshold_count": len(eligible),
                "returned_count": len(selected),
            }
            if detail:
                for index, score in enumerate(scores):
                    run.documents[index].update(
                        score=score, passes_threshold=score >= threshold
                    )
            results = []
            for index in selected:
                result: dict[str, Any] = {
                    "document_index": index,
                    "score": scores[index],
                }
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
            if self.split_tokenizer_name is None:
                raise ConfigurationError("No tokenizer selected.")
            try:
                self._tokenizer = HuggingFaceTokenizer(
                    self.split_tokenizer_name,
                    revision=self.split_tokenizer_revision,
                    model_max_length=self.tokenizer_max_length,
                )
            except ConfigurationError:
                raise
            except Exception:  # noqa: BLE001 - tokenizers raises a generic Exception for invalid files
                raise ConfigurationError(
                    "Could not load tokenizer. Check the tokenizer name/revision, HF access and cache, "
                    "accept the Gemma license if required, or supply a local/custom tokenizer."
                ) from None
        current_limit = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(current_limit, int) and current_limit < self.tokenizer_max_length:
            try:
                # Optional attribute, deliberately outside the Tokenizer protocol.
                setattr(self._tokenizer, "model_max_length", self.tokenizer_max_length)  # noqa: B010
            except (AttributeError, TypeError, ValueError):
                raise ConfigurationError(
                    "Tokenizer model_max_length cannot be increased. "
                    "Supply a tokenizer with a writable limit or an untruncated wrapper."
                ) from None
        return self._tokenizer

    def _count_locked(self, text: str) -> int:
        if self._length_fn is None:
            return len(self._get_tokenizer().encode(text))
        try:
            value = self._length_fn(text)
        except Exception:  # noqa: BLE001 - user callback failures need a stable public exception
            raise ConfigurationError("length_fn failed while counting text.") from None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError("length_fn must return a nonnegative integer.")
        return value

    def _count(self, text: str) -> int:
        with self._tokenizer_lock:
            return self._count_locked(text)

    def _truncate(self, text: str, limit: int) -> str:
        # Caller holds the counter/tokenizer lock.
        if self._length_fn is len:
            return text[:limit]
        if self._length_fn is not None:
            if self._count_locked("") > limit:
                raise ConfigurationError(
                    "length_fn cannot fit even an empty prefix within the limit."
                )
            low, high = 0, len(text)
            while high - low > 1:
                middle = (low + high) // 2
                if self._count_locked(text[:middle]) <= limit:
                    low = middle
                else:
                    high = middle
            # Safe prefix; not necessarily the longest for nonmonotonic counters.
            return text[:low]
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.encode(text)[:limit]
        while True:
            decoded = tokenizer.decode(tokens)
            length = self._count_locked(decoded)
            if length <= limit:
                return decoded
            if not tokens:
                raise ConfigurationError(
                    "Tokenizer cannot produce text within document_max_length."
                )
            tokens = tokens[: max(0, len(tokens) - max(1, length - limit))]

    def _prepare(self, docs: list[str], run: _Run) -> list[str]:
        prepared = []
        with self._tokenizer_lock:
            for index, original in enumerate(docs):
                original_count = self._count_locked(original)
                text = original
                limit = self.document_max_length
                if limit is not None and original_count > limit:
                    text = self._truncate(original, limit)
                sent_count = self._count_locked(text)
                if limit is not None and sent_count > limit:
                    raise ConfigurationError(
                        "length_fn must be deterministic; truncated text exceeds the limit."
                    )
                prepared.append(text)
                if run.detailed:
                    run.documents.append(
                        {
                            "document_index": index,
                            "sha256": sha(original),
                            "original_length": original_count,
                            "sent_length": sent_count,
                            "length_unit": self.length_unit,
                            "truncated": text != original,
                            "request_ids": [],
                            "comparisons": [],
                        }
                    )
        return prepared

    def _question(
        self, instruction: dict[str, Any], **references: str
    ) -> dict[str, Any]:
        return {
            "type": "noul",
            "instructions": instruction["instructions"].format(**references),
            "criteria": instruction["criteria"],
        }

    def _payload(
        self,
        query: str,
        docs: list[str],
        indices: list[int],
        instruction: dict[str, Any],
    ) -> dict[str, Any]:
        if self.mode == "pairwise":
            state = {
                "query": query,
                "left": docs[indices[0]],
                "right": docs[indices[1]],
            }
            questions = {
                "left_wins": self._question(
                    instruction, left="`left`", right="`right`"
                ),
                "right_wins": self._question(
                    instruction, left="`right`", right="`left`"
                ),
            }
        elif self.mode == "pointwise":
            state = {"query": query, "document": docs[indices[0]]}
            questions = {"relevant": self._question(instruction, document="`document`")}
        else:
            state = {
                "query": query,
                "documents": {f"doc_{i}": docs[i] for i in indices},
            }
            question_instruction = instruction
            if instruction == _LISTWISE_RELEVANCE_INSTRUCTION:
                rubric = copy.deepcopy(instruction)
                rubric["instructions"] = rubric["instructions"].format(
                    document="the candidate"
                )
                state = {"rubric": rubric, **state}
                question_instruction = {
                    "instructions": _LISTWISE_RELEVANCE_REFERENCE,
                    "criteria": instruction["criteria"],
                }
            questions = {
                f"doc_{i}": self._question(
                    question_instruction, document=f"`documents.doc_{i}`"
                )
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

    async def _estimate_async(self, payload: dict[str, Any]) -> dict[str, int]:
        return await asyncio.to_thread(self._estimate, payload)

    async def _request(
        self,
        payload: dict[str, Any],
        indices: list[int],
        run: _Run,
    ) -> list[float]:
        trace: dict[str, Any] = {}
        if run.detailed:
            estimated = await self._estimate_async(payload)
            trace.update(
                id=len(run.requests),
                document_indices=indices,
                payload=copy.deepcopy(payload),
                estimated_length=estimated,
            )
            run.requests.append(trace)
            for i in indices:
                run.documents[i]["request_ids"].append(trace["id"])
        try:
            async with self._runtime.semaphore:
                scores, data = await _client.request(
                    client=self._runtime.get_client(),
                    endpoint=self.endpoint,
                    api_key=self._api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    payload=payload,
                    trace=trace,
                )
            run.usage["requests"] += 1
            run.usage["input_tokens"] += data["usage"]["input_tokens"]
            run.usage["output_tokens"] += data["usage"]["output_tokens"]
            run.models.add(data["model"])
            if run.detailed:
                trace["response"] = data
            return scores
        except asyncio.CancelledError:
            trace["status"] = "cancelled"
            raise
        finally:
            run.usage["attempts"] += trace.get("attempts", 0)
            run.usage["retries"] += trace.get("retries", 0)
            run.usage["max_tokens_errors"] += (
                trace.get("status") == "max_tokens_exceeded"
            )

    async def _score(self, query: str, docs: list[str], run: _Run) -> list[float]:
        def payload(indices: list[int]) -> dict[str, Any]:
            return self._payload(query, docs, indices, run.instruction)

        async def score_one(indices: list[int]) -> list[float]:
            body = payload(indices)
            estimate = await self._estimate_async(body)
            if (
                estimate["state_plus_longest_question"] > self.split_state_budget
                or estimate["request"] > self.split_request_budget
            ):
                raise ContextLimitError(
                    f"Query and document group {indices} exceed the estimated context budget."
                )
            return await self._request(body, indices, run)

        if self.mode == "listwise":
            lengths = await asyncio.to_thread(lambda: [self._count(d) for d in docs])
            return await score_listwise(
                query=query,
                lengths=lengths,
                state_budget=self.split_state_budget,
                request_budget=self.split_request_budget,
                estimate=lambda ids: self._estimate_async(payload(ids)),
                request=lambda ids: self._request(payload(ids), ids, run),
                splits=run.splits,
            )
        scores = [0.0] * len(docs)
        if self.mode == "pointwise":
            indices = iter(range(len(docs)))

            async def worker() -> None:
                # Iterator advances synchronously on this loop, before awaiting.
                for index in indices:
                    scores[index] = (await score_one([index]))[0]

            await _gather_cancel_on_error(
                *(worker() for _ in range(min(len(docs), self.max_concurrency)))
            )
            return scores
        if len(docs) == 1:
            return [0.5]
        pairs = iter(combinations(range(len(docs)), 2))
        # Keep queued tasks and temporary results bounded for O(n²) comparisons.
        while batch := list(islice(pairs, self.max_concurrency)):
            values = await _gather_cancel_on_error(
                *(score_one(list(pair)) for pair in batch)
            )
            for (left, right), (forward, reverse) in zip(batch, values, strict=True):
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
            "schema_version": 2,
            "status": status,
            "started_at": start,
            "elapsed_seconds": time.monotonic() - started,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.system(),
                "package_version": version("jev-reranker"),
                "dependencies": _dependency_versions(),
            },
            "configuration": {
                "length_unit": self.length_unit,
                "length_function": (
                    f"{getattr(self._length_fn, '__module__', type(self._length_fn).__module__)}."
                    f"{getattr(self._length_fn, '__qualname__', type(self._length_fn).__qualname__)}"
                    if self._length_fn is not None
                    else "tokenizer.encode"
                ),
                "execution_backend": "httpx.AsyncClient/asyncio",
                "model": self.model,
                "mode": self.mode,
                "endpoint": self.endpoint,
                "instructions": run.instruction["instructions"],
                "criteria": run.instruction["criteria"],
                "threshold": run.threshold,
                "max_concurrency": self.max_concurrency,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "document_max_length": self.document_max_length,
                "split_state_budget": self.split_state_budget,
                "split_request_budget": self.split_request_budget,
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
                )
                if self.length_unit == "tokens"
                else None,
                "estimates_are_provider_tokens": False,
                "shuffle_seed": SHUFFLE_SEED,
                "tie_break": "stable_input_order",
                "score_semantics": "mean_pairwise_win_probability"
                if self.mode == "pairwise"
                else "noul",
            },
            "query_sha256": sha(query),
            "document_count": len(docs),
            "selection": run.selection,
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


async def _gather_cancel_on_error(*coroutines: Any) -> list[Any]:
    """Drain siblings on failure/cancellation, preserving public exception types."""
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
