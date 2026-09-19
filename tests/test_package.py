"""Smoke test for the installed distribution."""

from importlib.metadata import distribution

import jev_reranker


def test_installed_package() -> None:
    assert jev_reranker.__name__ == "jev_reranker"
    assert distribution("jev-reranker").metadata["Name"] == "jev-reranker"
