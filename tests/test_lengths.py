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
        raw = ranker.rerank("q", ["猫🙂abc"], detail=True)
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
        raw = ranker.rerank("q", ["猫🙂abcd"], detail=True)
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
        raw = ranker.rerank("q", ["猫🙂abc"], detail=True)
    assert calls[0]["state"]["documents"]["doc_0"] == "猫🙂"
    assert raw["results"][0]["detail"]["original_length"] == 10
    assert raw["results"][0]["detail"]["sent_length"] == 7
    assert raw["detail"]["configuration"]["length_unit"] == "custom"
    assert any('"questions"' in text for text in seen)


@pytest.mark.parametrize("bad", [True, -1, 1.5, None])
def test_invalid_counter_result_is_configuration_error(bad):
    ranker, calls = make(length_fn=lambda _: bad)
    with ranker, pytest.raises(ConfigurationError, match="length_fn"):
        ranker.rerank("q", ["a"])
    assert not calls


def test_custom_counter_cannot_fit_empty_prefix():
    ranker, calls = make(length_fn=lambda _: 100, document_max_length=3)
    with ranker, pytest.raises(ConfigurationError, match="empty"):
        ranker.rerank("q", ["a"])
    assert not calls


def test_custom_nonmonotonic_counter_still_returns_a_fitting_prefix():
    def counter(text):
        return {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}.get(len(text), len(text))

    ranker, calls = make(length_fn=counter, document_max_length=2)
    with ranker:
        raw = ranker.rerank("q", ["abcde"], detail=True)
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
        raw = ranker.rerank("q", ["abcde"], detail=True)
    assert calls[0]["state"]["documents"]["doc_0"] == "abc"
    assert raw["detail"]["configuration"]["document_max_length"] == 3
    assert raw["detail"]["configuration"]["split_state_budget"] == 2000


def test_named_tokenizer_is_opt_in_and_lazy():
    ranker, calls = make(tokenizer="google/embeddinggemma-300m")
    with ranker:
        raw = ranker.rerank("q", [], detail=True)
        assert raw["detail"]["configuration"]["length_unit"] == "tokens"
        assert ranker._tokenizer is None
    assert not calls


def test_truncation_can_be_disabled():
    ranker, calls = make(document_max_length=None)
    with ranker:
        ranker.rerank("q", ["x" * 5000])
    assert len(calls[0]["state"]["documents"]["doc_0"]) == 5000


class CountingTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


class SubwordTokenizer:
    """Greedy merges make counts depend on neighboring characters."""

    pieces = ("abc", "ab", "猫犬", "犬鳥")

    def encode(self, text):
        tokens = []
        offset = 0
        while offset < len(text):
            for index, piece in enumerate(self.pieces):
                if text.startswith(piece, offset):
                    tokens.append(-index - 1)
                    offset += len(piece)
                    break
            else:
                tokens.append(ord(text[offset]))
                offset += 1
        return tokens

    def decode(self, tokens):
        return "".join(
            self.pieces[-token - 1] if token < 0 else chr(token) for token in tokens
        )


class ByteTokenizer:
    """Partial UTF-8 tokens decode to a replacement character."""

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")


def test_subword_counts_depend_on_context():
    tokenizer = SubwordTokenizer()
    ranker, _ = make(tokenizer=tokenizer)
    for _ in range(2):
        assert ranker._count("abc") == 1
        assert ranker._count("acb") == 3
        assert ranker._count("猫犬") == 1
        assert ranker._count("犬猫") == 2


@pytest.mark.parametrize("cache_limit", [0, 4_000_000])
def test_cached_truncation_rechecks_partial_byte_tokens(cache_limit):
    tokenizer = ByteTokenizer()
    ranker, calls = make(tokenizer=tokenizer, document_max_length=3)
    ranker._count_cache_limit = cache_limit
    assert len(tokenizer.encode(tokenizer.decode(tokenizer.encode("a猫z")[:3]))) > 3
    for _ in range(2):
        raw = ranker.rerank("q", ["a猫z"], detail=True)
        assert calls[-1]["state"]["documents"]["doc_0"] == "a"
        assert raw["results"][0]["detail"]["original_length"] == 5
        assert raw["results"][0]["detail"]["sent_length"] == 1


def test_token_counts_reused_for_identical_text():
    tokenizer = CountingTokenizer()
    ranker, _ = make(tokenizer=tokenizer)
    assert ranker._count("same text") == 9
    assert ranker._count("same text") == 9
    assert tokenizer.calls == ["same text"]


def test_long_document_encoded_once_before_truncation():
    tokenizer = CountingTokenizer()
    ranker, calls = make(tokenizer=tokenizer, document_max_length=3)
    result = ranker.rerank("q", ["abcdef"], detail=True)
    assert calls[0]["state"]["documents"]["doc_0"] == "abc"
    assert result["results"][0]["detail"]["original_length"] == 6
    assert tokenizer.calls.count("abcdef") == 1


def test_count_cache_is_bounded_and_evicts():
    tokenizer = CountingTokenizer()
    ranker, _ = make(tokenizer=tokenizer)
    ranker._count_cache_limit = 10
    for text in ["aaaaaa", "bbbbbb", "aaaaaa"]:
        assert ranker._count(text) == 6
    assert tokenizer.calls == ["aaaaaa", "bbbbbb", "aaaaaa"]
    assert ranker._count_cache_size <= 10
    ranker._count("x" * 11)
    assert "x" * 11 not in ranker._count_cache


@pytest.mark.parametrize("mode", ["listwise", "pointwise", "pairwise"])
def test_detailed_requests_reuse_budget_estimate(mode, monkeypatch):
    ranker, calls = make(
        mode=mode,
        document_max_length=None,
        split_state_budget=2000,
        split_request_budget=4000,
    )
    original = ranker._estimate
    seen = []

    def estimate(payload):
        seen.append(json.dumps(payload))
        return original(payload)

    monkeypatch.setattr(ranker, "_estimate", estimate)
    result = ranker.rerank("q", ["a" * 700, "b" * 700, "c" * 700], detail=True)
    assert calls
    assert len(seen) == len(set(seen))
    for request, payload in zip(result["detail"]["requests"], calls, strict=True):
        assert request["estimated_length"] == original(payload)


@pytest.mark.parametrize("tokenizer_type", [CountingTokenizer, SubwordTokenizer])
def test_cache_preserves_split_payloads_and_truncation(tokenizer_type):
    outputs = []
    for limit in (0, 4_000_000):
        ranker, calls = make(
            tokenizer=tokenizer_type(),
            document_max_length=700,
            split_state_budget=1600,
            split_request_budget=3000,
        )
        ranker._count_cache_limit = limit
        raw = ranker.rerank(
            "日本語 query", ["猫犬" * 900, "abc" * 900, "犬鳥" * 600], detail=True
        )
        assert raw["detail"]["splits"]
        outputs.append(
            (
                calls,
                raw["results"],
                raw["detail"]["splits"],
                [r["estimated_length"] for r in raw["detail"]["requests"]],
            )
        )
    assert outputs[0] == outputs[1]


def test_token_count_cache_is_safe_for_concurrent_queries():
    from concurrent.futures import ThreadPoolExecutor

    tokenizer = CountingTokenizer()
    ranker, _ = make(tokenizer=tokenizer)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(ranker._count, ["shared"] * 20)) == [6] * 20
    assert tokenizer.calls == ["shared"]


def test_custom_counter_is_not_memoized():
    seen = []

    def counter(text):
        seen.append(text)
        return len(text)

    ranker, _ = make(length_fn=counter)
    assert ranker._count("text") == ranker._count("text") == 4
    assert seen == ["text", "text"]
