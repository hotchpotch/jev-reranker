"""Real local tokenizer backend tests, without a Hub download."""

from test_reranker import make
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from jev_reranker import HuggingFaceTokenizer


def test_native_backend_counts_beyond_model_length_without_truncation(tmp_path):
    backend = BackendTokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.enable_truncation(max_length=2)
    backend.enable_padding(length=5)
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    tokenizer = HuggingFaceTokenizer(str(path), model_max_length=65536)
    assert len(tokenizer.encode("hello " * 65537)) == 65537
    assert tokenizer.encode("hello") == [1]
    assert tokenizer.decode([1]) == "hello"


def test_truncation_rechecks_decoded_length():
    class ExpandingTokenizer:
        def encode(self, text):
            return list(map(ord, text))

        def decode(self, tokens):
            return "".join(map(chr, tokens)) + "!"

    ranker, calls = make(document_max_tokens=3)
    ranker._tokenizer = ExpandingTokenizer()
    raw = ranker.raw_rerank("q", ["abcdef"], detail=True)
    assert len(calls[0]["state"]["documents"]["doc_0"]) == 3
    assert raw["results"][0]["detail"]["sent_length"] == 3


def test_truncation_disabled_and_short_text_verbatim():
    ranker, calls = make(document_max_tokens=None)
    text = "  日本語\n中文   español  "
    ranker.rank("q", [text])
    assert calls[0]["state"]["documents"]["doc_0"] == text


def test_supplied_tokenizer_maximum_is_raised_without_lowering_larger_limit():
    from test_lengths import make as make_lengths

    class LimitedTokenizer:
        model_max_length = 2

        def encode(self, text):
            return list(map(ord, text[: self.model_max_length]))

        def decode(self, tokens):
            return "".join(map(chr, tokens))

    tokenizer = LimitedTokenizer()
    ranker, _ = make_lengths(tokenizer=tokenizer, tokenizer_max_length=100000)
    with ranker:
        raw = ranker.raw_rerank("q", ["abcdef"], detail=True)
    assert tokenizer.model_max_length == 100000
    assert raw["results"][0]["detail"]["original_length"] == 6

    tokenizer.model_max_length = 200000
    ranker, _ = make_lengths(tokenizer=tokenizer)
    with ranker:
        ranker.rank("q", ["abcdef"])
    assert tokenizer.model_max_length == 200000


def test_readonly_small_tokenizer_limit_fails_before_http():
    import pytest
    from test_lengths import make as make_lengths

    from jev_reranker import ConfigurationError

    class ReadonlyTokenizer:
        @property
        def model_max_length(self):
            return 2

        def encode(self, text):
            return list(map(ord, text[:2]))

        def decode(self, tokens):
            return "".join(map(chr, tokens))

    ranker, calls = make_lengths(tokenizer=ReadonlyTokenizer())
    with ranker, pytest.raises(ConfigurationError, match="model_max_length"):
        ranker.rank("q", ["abcdef"])
    assert not calls
