# Running live tests

Live tests exercise the real Jev API through the library's sync and async methods.
They are separate from offline tests and are skipped unless explicitly enabled.

## Setup

From the repository root:

```sh
uv sync --locked --dev
# If you do not already have a .env file:
cp .env.sample .env
```

Set `TYPESAFE_API_KEY` in `.env` or the environment. Existing environment variables
take precedence. Do not commit credentials. The tests use the Gemma tokenizer;
Hugging Face authentication and model license acceptance may be needed for the
first download. Only tokenizer files are needed, not embedding model weights.

## Run

```sh
uv run --locked pytest tests/test_live.py --live -q
```

This makes billable requests. The model is selected through the library's usual
configuration, including `JEV_MODEL`, falling back to `jev-latest`. Pin a model
when comparing behavior across changes; use latest when checking compatibility
with the service's current default.

To focus on evidence filtering during prompt development:

```sh
uv run --locked pytest tests/test_live.py --live -k relevance -q
```

## What the tests check

- Four-document ranking in English, Japanese, Chinese, Spanish, and mixed languages.
- Expected document indices and strict score differences, rather than only checking
  that returned scores are sorted.
- Listwise, pointwise, and pairwise behavior through synchronous and asynchronous APIs.
- Relevance ranking in listwise and pointwise, including retention of direct and
  partial answers and removal of clearly unrelated content at the default threshold.
- Reusing a reranker across independently cleaned-up calls, including default character-based length counting.

These are smoke tests, not a multilingual quality benchmark. Relevance prompts
can retain incomplete supporting information; a related overview is not necessarily
expected to fall below the threshold. Model updates and input context can change
scores. Inspect a failure before changing its expected behavior.

## Diagnose failures

Execution details are saved under Git-ignored `.live-results/`. They contain query
and document text, effective configuration, model information, and request traces;
treat them as application data when saving or sharing them. Credentials are not
included in the library's details.

Check the resolved model, prompt, scores, truncation, splits, and retries to
understand an unexpected ranking. Do not silently relax assertions to accommodate
a service change. Offline tests cover deterministic error paths, malformed
responses, retry behavior, partitioning, and lifecycle rules without real API calls.

Run those checks with:

```sh
uv run --locked tox
```
