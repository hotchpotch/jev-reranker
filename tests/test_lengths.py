"""Character defaults, pluggable counters, and optional tokenizer contracts."""

import json

import httpx
import pytest

from jev_reranker import ConfigurationError, JevReranker


def make(**kwargs):
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "test",
                "answers": {
                    k: {"type": "noul", "noul": 0.5} for k in payload["questions"]
                },
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    return JevReranker(
        api_key="test",
        dotenv_path=None,
        transport=httpx.MockTransport(handle),
        **kwargs,
    ), calls


def test_default_counts_characters_without_tokenizer():
    ranker, calls = make()
    with ranker:
        raw = ranker.raw_rerank("q", ["猫🙂abc"], detail=True)
        assert raw["results"][0]["document_index"] == 0
        assert "corpus_id" not in raw["results"][0]
        assert raw["results"][0]["detail"]["original_length"] == 5
        assert raw["results"][0]["detail"]["length_unit"] == "characters"
        assert raw["detail"]["configuration"]["tokenizer"] is None
        assert raw["detail"]["configuration"]["document_max_length"] == 4000
        assert ranker._tokenizer is None
    assert calls[0]["state"]["documents"]["doc_0"] == "猫🙂abc"


def test_character_truncation_and_limit_override():
    ranker, calls = make(document_max_length=3)
    with ranker:
        raw = ranker.raw_rerank("q", ["猫🙂abcd"], detail=True)
    assert calls[0]["state"]["documents"]["doc_0"] == "猫🙂a"
    assert raw["results"][0]["text"] == "猫🙂abcd"
    assert raw["results"][0]["detail"]["sent_length"] == 3


def test_custom_counter_used_for_documents_and_request_budgets():
    seen = []

    def utf8_length(text):
        seen.append(text)
        return len(text.encode("utf-8"))

    ranker, calls = make(length_fn=utf8_length, document_max_length=7)
    with ranker:
        raw = ranker.raw_rerank("q", ["猫🙂abc"], detail=True)
    assert calls[0]["state"]["documents"]["doc_0"] == "猫🙂"
    assert raw["results"][0]["detail"]["original_length"] == 10
    assert raw["results"][0]["detail"]["sent_length"] == 7
    assert raw["detail"]["configuration"]["length_unit"] == "custom"
    assert any('"questions"' in text for text in seen)


@pytest.mark.parametrize("bad", [True, -1, 1.5, None])
def test_invalid_counter_result_is_configuration_error(bad):
    ranker, calls = make(length_fn=lambda _: bad)
    with ranker, pytest.raises(ConfigurationError, match="length_fn"):
        ranker.rank("q", ["a"])
    assert not calls


def test_custom_counter_cannot_fit_empty_prefix():
    ranker, calls = make(length_fn=lambda _: 100, document_max_length=3)
    with ranker, pytest.raises(ConfigurationError, match="empty"):
        ranker.rank("q", ["a"])
    assert not calls


def test_custom_nonmonotonic_counter_still_returns_a_fitting_prefix():
    def counter(text):
        return {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}.get(len(text), len(text))

    ranker, calls = make(length_fn=counter, document_max_length=2)
    with ranker:
        raw = ranker.raw_rerank("q", ["abcde"], detail=True)
    assert counter(calls[0]["state"]["documents"]["doc_0"]) <= 2
    assert raw["results"][0]["detail"]["sent_length"] <= 2


def test_counter_tokenizer_conflict_and_noncallable():
    for kwargs in (
        {"length_fn": 123},
        {"length_fn": len, "tokenizer": "google/embeddinggemma-300m"},
    ):
        with pytest.raises(ConfigurationError):
            make(**kwargs)


def test_legacy_length_keyword_aliases():
    ranker, calls = make(
        document_max_tokens=3,
        split_state_token_budget=2000,
        split_request_token_budget=3000,
    )
    with ranker:
        raw = ranker.raw_rerank("q", ["abcde"], detail=True)
    assert calls[0]["state"]["documents"]["doc_0"] == "abc"
    assert raw["detail"]["configuration"]["document_max_length"] == 3
    assert raw["detail"]["configuration"]["split_state_budget"] == 2000


def test_named_tokenizer_is_opt_in_and_lazy():
    ranker, calls = make(tokenizer="google/embeddinggemma-300m")
    with ranker:
        raw = ranker.raw_rerank("q", [], detail=True)
        assert raw["detail"]["configuration"]["length_unit"] == "tokens"
        assert ranker._tokenizer is None
    assert not calls


def test_truncation_can_be_disabled():
    ranker, calls = make(document_max_length=None)
    with ranker:
        ranker.rank("q", ["x" * 5000])
    assert len(calls[0]["state"]["documents"]["doc_0"]) == 5000
