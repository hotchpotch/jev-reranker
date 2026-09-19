"""Live requests require explicit opt-in, including when credentials exist."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        help="Run paid live TypeSafe/Gemma E2E tests using .env",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: real network, credentials and tokenizer access required"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        for item in items:
            if "live" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="requires explicit --live"))


@pytest.fixture(autouse=True)
def offline_environment(request, monkeypatch):
    if "live" in request.node.keywords:
        return
    import httpx

    for name in ("TYPESAFE_API_KEY", "TYPESAFE_ENDPOINT", "JEV_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def forbid_network(*args, **kwargs):
        pytest.fail(
            "Real HTTP is forbidden in offline tests; use MockTransport or --live tests."
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)
