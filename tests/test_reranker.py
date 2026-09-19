"""Offline contract tests: no credentials, downloads, or network required."""

import asyncio
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest

from jev_reranker import (
    APIError,
    ConfigurationError,
    ContextLimitError,
    JevReranker,
    ResponseValidationError,
)

_TEST_RANKERS = []


async def no_sleep(_):
    pass


class CharacterTokenizer:
    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


def response(payload, scores=None):
    keys = list(payload["questions"])
    return {
        "model": "jev-test-1",
        "answers": {
            k: {"type": "noul", "noul": (scores or {}).get(k, 0.5)} for k in keys
        },
        "usage": {"input_tokens": 100, "output_tokens": 5},
    }


def make(handler=None, **kwargs):
    calls = []

    async def serve(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if handler:
            result = handler(payload, len(calls), request)
            return await result if inspect.isawaitable(result) else result
        return httpx.Response(200, json=response(payload))

    ranker = JevReranker(
        api_key="secret-test-key",
        dotenv_path=None,
        tokenizer=CharacterTokenizer(),
        transport=httpx.MockTransport(serve),
        **kwargs,
    )
    _TEST_RANKERS.append(ranker)
    return ranker, calls


@pytest.mark.parametrize("mode", ["listwise", "pointwise"])
def test_scores_sorted_and_duplicate_identity(mode):
    def handler(p, count, request):
        if mode == "listwise":
            scores = {"doc_0": 0.1, "doc_1": 0.9, "doc_2": 0.9}
        else:
            scores = {"relevant": 0.1 if p["state"]["document"] == "bad" else 0.9}
        return httpx.Response(200, json=response(p, scores))

    ranker, calls = make(handler, mode=mode)
    results = ranker.rerank("q", ["bad", "good", "good"])["results"]
    assert [r["document_index"] for r in results] == [1, 2, 0]
    assert [r["score"] for r in results] == [0.9, 0.9, 0.1]
    assert results[0]["text"] == "good"
    assert len(calls) == (1 if mode == "listwise" else 3)


def test_top_k():
    ranker, calls = make()
    assert ranker.rerank("q", [], detail=True)["results"] == []
    assert ranker.rerank("q", ["a"], top_k=0) == {"results": []}
    assert not calls
    assert ranker.rerank("q", ["a", "b"], top_k=1, return_documents=False)["results"] == [
        {"document_index": 0, "score": 0.5}
    ]
    assert len(calls[0]["questions"]) == 2



@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bad"},
        {"max_concurrency": True},
        {"max_concurrency": 0},
        {"timeout": float("nan")},
        {"timeout": True},
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"document_max_tokens": 0},
        {"document_max_tokens": True},
        {"split_state_token_budget": 0},
        {"split_request_token_budget": False},
        {"instructions": "missing placeholder"},
        {"instructions": "{document} {oops}"},
        {"criteria": {"true": "yes"}},
        {"model": ""},
        {"api_key": ""},
        {"endpoint": "https://user:password@example.com/"},
        {"endpoint": "https://example.com/?token=secret"},
        {"endpoint": "http://example.com/"},
    ],
)
def test_invalid_configuration(kwargs):
    options: dict[str, Any] = {
        "api_key": "test",
        "dotenv_path": None,
        "tokenizer": CharacterTokenizer(),
    }
    options.update(kwargs)
    with pytest.raises(ConfigurationError):
        JevReranker(**options)


@pytest.mark.parametrize(
    "query,docs,kwargs",
    [
        ("", ["a"], {}),
        (None, ["a"], {}),
        ("q", "document", {}),
        ("q", [1], {}),
        ("q", ["a"], {"top_k": -1}),
        ("q", ["a"], {"top_k": True}),
        ("q", ["a"], {"detail": "yes"}),
    ],
)
def test_bad_input_never_calls_api(query, docs, kwargs):
    ranker, calls = make()
    with pytest.raises((ValueError, TypeError)):
        ranker.rerank(query, docs, **kwargs)
    assert not calls


def test_unknown_kwargs():
    ranker, calls = make()
    with pytest.raises(TypeError):
        ranker.rerank("q", ["a"], typo=True)
    with pytest.raises(TypeError):
        JevReranker(api_key="test", batch_size=3)  # ty: ignore[unknown-argument]
    assert not calls


def test_env_precedence_and_no_global_mutation(monkeypatch, tmp_path):
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    monkeypatch.delenv("JEV_MODEL", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("CUSTOM_KEY=file-key\nJEV_MODEL=jev-file\n")
    with JevReranker(api_key_env="CUSTOM_KEY", dotenv_path=dotenv) as ranker:
        assert ranker.model == "jev-file"
        assert ranker._api_key == "file-key"
    import os

    assert "CUSTOM_KEY" not in os.environ
    monkeypatch.setenv("CUSTOM_KEY", "env-key")
    monkeypatch.setenv("JEV_MODEL", "jev-env")
    with JevReranker(api_key_env="CUSTOM_KEY", dotenv_path=dotenv) as ranker:
        assert ranker._api_key == "env-key"
        assert ranker.model == "jev-env"
    with JevReranker(
        api_key="explicit", model="jev-pinned", dotenv_path=dotenv
    ) as ranker:
        assert ranker._api_key == "explicit"
        assert ranker.model == "jev-pinned"
    monkeypatch.delenv("CUSTOM_KEY")
    with pytest.raises(ConfigurationError, match="CUSTOM_KEY"):
        JevReranker(api_key_env="CUSTOM_KEY", dotenv_path=None)


def test_truncation_and_detail():
    ranker, calls = make(document_max_tokens=3)
    raw = ranker.rerank("質問", ["abcdef", "猫"], detail=True)
    assert calls[0]["state"]["documents"] == {"doc_0": "abc", "doc_1": "猫"}
    assert raw["results"][0]["text"] == "abcdef"
    assert raw["results"][0]["detail"]["sent_length"] == 3
    assert raw["results"][0]["detail"]["original_length"] == 6
    detail = raw["detail"]
    assert detail["usage"]["input_tokens"] == 100
    assert detail["resolved_models"] == ["jev-test-1"]
    assert detail["requests"][0]["response"] == response(calls[0])
    assert detail["configuration"]["model"] == "jev-latest"
    assert detail["environment"]["python"]
    assert "secret-test-key" not in json.dumps(raw)
    assert "detail" not in ranker.rerank("q", ["a"])


def test_budget_split_every_candidate_once_stable_ties():
    ranker, calls = make(split_state_token_budget=1100, split_request_token_budget=1800)
    docs = [str(i) * 100 for i in range(11)]
    raw = ranker.rerank("q", docs, detail=True)
    keys = [k for p in calls for k in p["questions"]]
    assert len(calls) > 1
    assert sorted(keys) == sorted(f"doc_{i}" for i in range(11))
    assert [r["document_index"] for r in raw["results"]] == list(range(11))
    assert raw["detail"]["splits"]
    for p in calls:
        state = len(json.dumps(p["state"], ensure_ascii=False, separators=(",", ":")))
        lengths = [
            len(json.dumps(q, ensure_ascii=False, separators=(",", ":")))
            for q in p["questions"].values()
        ]
        assert state + max(lengths) <= 1100


def test_server_overflow_splits_without_resending_successes():
    def handler(p, count, req):
        if len(p["questions"]) > 2:
            return httpx.Response(
                422, json={"detail": {"error_type": "max_tokens_exceeded"}}
            )
        return httpx.Response(200, json=response(p))

    ranker, calls = make(handler)
    raw = ranker.rerank("q", ["a"] * 5, detail=True)
    successful = [k for p in calls if len(p["questions"]) <= 2 for k in p["questions"]]
    assert sorted(successful) == [f"doc_{i}" for i in range(5)]
    assert raw["detail"]["usage"]["max_tokens_errors"] >= 1


@pytest.mark.parametrize("mode", ["listwise", "pointwise", "pairwise"])
def test_unsplittable_raises_with_detail(mode):
    ranker, calls = make(mode=mode, split_state_token_budget=1)
    with pytest.raises(ContextLimitError) as exc:
        ranker.rerank("q", ["a", "b"] if mode == "pairwise" else ["a"], detail=True)
    assert exc.value.detail is not None
    assert exc.value.detail["status"] == "failed"
    assert not calls


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_retries_transient_status(monkeypatch, status):
    waits = []

    async def record_sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("jev_reranker._client.asyncio.sleep", record_sleep)

    def handler(p, count, request):
        if count == 1:
            return httpx.Response(status, headers={"Retry-After": "2"})
        return httpx.Response(200, json=response(p))

    ranker, calls = make(handler, max_retries=1)
    raw = ranker.rerank("q", ["a"], detail=True)
    assert len(calls) == 2
    assert waits == [2.0]
    assert raw["detail"]["usage"]["retries"] == 1


@pytest.mark.parametrize("status", [401, 403, 400, 422, 302])
def test_no_retry_permanent_errors(status):
    ranker, calls = make(lambda p, n, r: httpx.Response(status, text="secret-test-key"))
    with pytest.raises(APIError) as exc:
        ranker.rerank("q", ["a"])
    assert exc.value.status_code == status
    assert "secret-test-key" not in str(exc.value)
    assert len(calls) == 1


def test_transport_retry_exhaustion(monkeypatch):
    monkeypatch.setattr("jev_reranker._client.asyncio.sleep", no_sleep)

    def handler(p, n, request):
        raise httpx.ReadTimeout("secret-test-key", request=request)

    ranker, calls = make(handler, max_retries=2)
    with pytest.raises(APIError) as exc:
        ranker.rerank("q", ["a"], detail=True)
    assert len(calls) == 3
    assert "secret-test-key" not in str(exc.value)
    assert exc.value.detail is not None
    assert exc.value.detail["usage"]["retries"] == 2


@pytest.mark.parametrize(
    "value", [True, None, "0.5", -1, 1.1, float("nan"), float("inf")]
)
def test_invalid_scores(value):
    def handler(p, n, r):
        data = response(p)
        data["answers"]["doc_0"]["noul"] = value
        return httpx.Response(200, content=json.dumps(data))

    ranker, calls = make(handler)
    with pytest.raises(ResponseValidationError):
        ranker.rerank("q", ["a"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("answers", {}),
        ("model", ""),
        ("usage", {}),
        ("usage", {"input_tokens": True, "output_tokens": 0}),
    ],
)
def test_bad_response_metadata(field, value):
    def handler(p, n, r):
        data = response(p)
        data[field] = value
        return httpx.Response(200, json=data)

    ranker, _ = make(handler)
    with pytest.raises(ResponseValidationError):
        ranker.rerank("q", ["a"])


def test_pairwise_average_symmetric_wins():
    def handler(p, n, r):
        state = p["state"]
        better = int(state["left"]) > int(state["right"])
        return httpx.Response(
            200,
            json=response(
                p,
                {
                    "left_wins": 0.9 if better else 0.1,
                    "right_wins": 0.1 if better else 0.9,
                },
            ),
        )

    ranker, calls = make(handler, mode="pairwise")
    raw = ranker.rerank("q", ["2", "0", "3", "1"], detail=True)
    assert [r["document_index"] for r in raw["results"]] == [2, 0, 3, 1]
    assert [r["score"] for r in raw["results"]] == pytest.approx(
        [0.9, 1.9 / 3, 1.1 / 3, 0.1]
    )
    assert len(calls) == 6
    assert len(raw["results"][0]["detail"]["comparisons"]) == 3
    assert ranker.rerank("q", ["a"])["results"][0]["score"] == 0.5
    assert len(calls) == 6


def test_parallel_calls_keep_statistics_separate():
    ranker, _ = make(mode="pointwise")
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(
            pool.map(lambda n: ranker.rerank("q", ["a"] * n, detail=True), [2, 3])
        )
    assert [out["detail"]["usage"]["requests"] for out in outputs] == [2, 3]
    assert [len(out["detail"]["requests"]) for out in outputs] == [2, 3]


def test_closed_client_fails_before_request():
    ranker, calls = make()
    ranker.close()
    with pytest.raises(ConfigurationError, match="closed"):
        ranker.rerank("q", ["a"])
    assert not calls


def test_env_is_ignored_and_sample_has_no_openai():
    assert ".env" in Path(".gitignore").read_text().splitlines()
    assert "OPENAI_" not in Path(".env.sample").read_text()


def test_custom_instructions_and_criteria_are_sent():
    criteria = {"true": "Answers the question", "false": "Does not answer"}
    ranker, calls = make(
        instructions="Does {document} answer `query`?", criteria=criteria
    )
    criteria["true"] = "mutated"
    ranker.rerank("q", ["a"])
    question = calls[0]["questions"]["doc_0"]
    assert question["instructions"] == "Does `documents.doc_0` answer `query`?"
    assert question["criteria"]["true"] == "Answers the question"


def test_successful_chunk_not_replayed_when_later_chunk_overflows():
    success_keys = []
    overflowed = False

    def handler(p, n, r):
        nonlocal overflowed
        keys = list(p["questions"])
        if success_keys and len(keys) > 1 and not overflowed:
            overflowed = True
            return httpx.Response(
                400, json={"detail": {"error_type": "max_tokens_exceeded"}}
            )
        success_keys.extend(keys)
        return httpx.Response(200, json=response(p))

    ranker, _ = make(
        handler, split_state_token_budget=1500, split_request_token_budget=3000
    )
    raw = ranker.rerank("q", ["a" * 100] * 20, detail=True)
    assert overflowed
    assert len(success_keys) == len(set(success_keys)) == 20
    assert raw["detail"]["usage"]["max_tokens_errors"] == 1


def test_transport_success_after_timeout(monkeypatch):
    monkeypatch.setattr("jev_reranker._client.asyncio.sleep", no_sleep)

    def handler(p, n, r):
        if n == 1:
            raise httpx.ConnectError("connection lost", request=r)
        return httpx.Response(200, json=response(p))

    ranker, calls = make(handler, max_retries=1)
    assert ranker.rerank("q", ["a"])["results"][0]["score"] == 0.5
    assert len(calls) == 2


def test_exhausted_http_retry_and_zero_retry(monkeypatch):
    monkeypatch.setattr("jev_reranker._client.asyncio.sleep", no_sleep)
    ranker, calls = make(lambda p, n, r: httpx.Response(503), max_retries=0)
    with pytest.raises(APIError) as exc:
        ranker.rerank("q", ["a"], detail=True)
    assert len(calls) == 1
    assert exc.value.detail is not None
    assert exc.value.detail["usage"]["attempts"] == 1
    assert exc.value.detail["usage"]["requests"] == 0


@pytest.mark.parametrize("body", ["invalid JSON", "[]", "null"])
def test_malformed_success_response(body):
    ranker, calls = make(lambda p, n, r: httpx.Response(200, content=body))
    with pytest.raises(ResponseValidationError):
        ranker.rerank("q", ["a"])
    assert len(calls) == 1


def test_api_overflow_single_document_is_not_retried():
    ranker, calls = make(
        lambda p, n, r: httpx.Response(
            422, json={"detail": {"error_type": "max_tokens_exceeded"}}
        )
    )
    with pytest.raises(ContextLimitError):
        ranker.rerank("q", ["a"])
    assert len(calls) == 1


def test_detail_redacts_echoed_api_key():
    def handler(p, n, r):
        data = response(p)
        data["diagnostic"] = "Bearer secret-test-key"
        return httpx.Response(200, json=data)

    ranker, _ = make(handler)
    assert "secret-test-key" not in json.dumps(
        ranker.rerank("q", ["a"], detail=True)
    )


def test_sync_client_rejected_explicitly():
    with (
        httpx.Client() as client,
        pytest.raises(ConfigurationError, match="AsyncClient"),
    ):
        JevReranker(api_key="test", dotenv_path=None, client=client)  # ty: ignore[invalid-argument-type]


def test_retry_after_date_and_caps():
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    from jev_reranker._client import retry_delay

    future = format_datetime(datetime.now(UTC) + timedelta(seconds=30))
    assert 28 <= retry_delay(0, future) <= 30
    assert retry_delay(0, "9999999") == 60
    assert retry_delay(0, "-1") == 0
    assert 0.5 <= retry_delay(0, "bad header") <= 0.7
    assert 0.5 <= retry_delay(0, "NaN") <= 0.7


def test_global_concurrency_bound_across_calls():
    import threading

    lock = threading.Lock()
    active = maximum = 0

    async def handler(p, n, r):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        with lock:
            active -= 1
        return httpx.Response(200, json=response(p))

    ranker, _ = make(handler, mode="pointwise", max_concurrency=2)
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda _: ranker.rerank("q", ["a"] * 4), range(3)))
    assert maximum == 2
    assert all(len(result["results"]) == 4 for result in outputs)


def test_huge_integer_score_is_validation_error():
    def handler(p, n, r):
        return httpx.Response(200, content=json.dumps(response(p, {"doc_0": 10**400})))

    ranker, _ = make(handler)
    with pytest.raises(ResponseValidationError):
        ranker.rerank("q", ["a"])
