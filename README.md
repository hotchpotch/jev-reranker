# jev-reranker: Reranking and Relevance Filtering for RAG

[![CI](https://github.com/hotchpotch/jev-reranker/actions/workflows/ci.yml/badge.svg)](https://github.com/hotchpotch/jev-reranker/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/jev-reranker.svg)](https://pypi.org/project/jev-reranker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

jev-reranker is a Python library for reranking search results and filtering
retrieved documents with TypeSafe.AI's Jev. It provides prompts for both tasks
and handles concurrent requests, splitting long candidate lists, and retries.

Search results can match a question without helping answer it. Passing every
match to an LLM adds input tokens and potentially distracting context.
`relevance_rerank()` scores documents for their usefulness as evidence, sorts
them, and removes those below a configurable threshold. If nothing passes,
your application can try another search or stop before generation.

Use it after retrieval and before assembling context for RAG. Use `rerank()`
when you want to reorder candidates without filtering by default. Both accept
a query string and a list of document strings; scoring runs through the Jev API.
The base package needs no local model or GPU.

## Highlights

- Relevance filtering scores documents for their contribution to an answer,
  including partial answers and facts needed for multi-hop reasoning.
- A configurable threshold determines which documents to keep. Results are sorted
  by score; an empty list means no candidate passed the threshold.
- Prompts are Python dictionaries. You can change the instructions and criteria
  to describe what counts as useful evidence in your application.
- Long candidate lists are split automatically. Document limits and split budgets
  can use character counts, a tokenizer, or a custom length function.
- Sync and async methods share request handling, concurrency limits, retries,
  and automatic HTTP cleanup.
- Optional details record all scores, including excluded documents, along with
  the prompts, model, input lengths, API usage, and request history.

## Getting started

Requires Python 3.11+ and a TypeSafe API key. Jev API usage is billed by the service.

```sh
uv add jev-reranker
# Or: pip install jev-reranker
```

Replace `YOUR-TYPESAFE-API-KEY...` in the examples with your TypeSafe API key.
You can also omit `api_key` and set `TYPESAFE_API_KEY` in the environment or `.env`.
Keep real API keys out of version control.
Save either example below as `example.py` and run `uv run python example.py`.

### Filter for useful evidence

```python
from jev_reranker import JevReranker

query = "How long do I have to return an online order to ACME Shop?"
documents = [
    "ACME Shop accepts online returns within 30 days of delivery.",
    "For ACME Shop online orders, submit your return request within 30 days of receiving the item.",
    "ACME Shop in-store purchases can be returned within 14 days of purchase.",
    "ACME Shop products come with a one-year repair warranty.",
    "FooBar Shop accepts online returns within 60 days of delivery.",
]

reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...")
evidence = reranker.relevance_rerank(query, documents, threshold=0.2)

for item in evidence["results"]:
    print(f"{item['score']:.2f}  {item['text']}")
```

Example output (scores can vary by model and input context):

```text
0.98  ACME Shop accepts online returns within 30 days of delivery.
0.97  For ACME Shop online orders, submit your return request within 30 days of receiving the item.
```

Only the two passages about ACME Shop online returns remain. The in-store return,
repair warranty, and FooBar Shop passages fall below the 0.2 threshold and are omitted.
This example uses the default listwise mode; it does not limit the output with
`top_k`.

Results are sorted by descending score. Each result also includes
`document_index`, its position in the original input, so you can retrieve URLs or
other metadata from your search results. If nothing passes the threshold,
`evidence["results"]` is an empty list: your application can try another search or abstain.

Lower thresholds retain more documents; higher thresholds are more selective.
The default keeps scores of 0.2 or higher. Test the threshold on your own queries
to see whether it drops useful passages or keeps unrelated ones. Filtering happens
after scoring, so it reduces the context sent to the next model, not the work
done by Jev.

### Rerank without filtering

```python
from jev_reranker import JevReranker

reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...")
results = reranker.rerank(
    "Which planet is called the Red Planet?",
    ["Venus has a thick atmosphere.", "Mars is known as the Red Planet."],
    top_k=1,
)

for item in results["results"]:
    print(item["document_index"], item["score"], item["text"])
```

`rerank()` defaults to threshold 0.0, keeping all scored candidates unless you
set `top_k` or raise the threshold. Each call closes its HTTP client automatically,
including on failure; no context manager or explicit cleanup is needed.

## Choosing between reranking and relevance filtering

| Method | Intended use | Default threshold |
| --- | --- | ---: |
| `rerank()` | Order retrieved documents by relevance | 0.0 |
| `relevance_rerank()` | Select evidence that can help answer the query | 0.2 |

Both methods use the same Jev model and request handling. Each supplies a
different prompt and default threshold. The relevance prompt accepts partial
answers and linking facts: a passage can be useful without answering the whole
question.

You can read or replace the [built-in prompts](src/jev_reranker/instructions.py).
A suitable threshold depends on your queries, prompt, model, and scoring mode.

## API reference

Expand the sections below for signatures, options, and examples. Code snippets
that only show configuration assume `from jev_reranker import JevReranker`.

<details>
<summary>Requests, responses, and selection</summary>

```text
rerank(query, documents, *, instruction=None, threshold=0.0,
       top_k=None, return_documents=True, detail=False)
relevance_rerank(query, documents, *, instruction=None, threshold=0.2,
                 top_k=None, return_documents=True, detail=False)
```

`query` is a string and `documents` is a sequence of strings. The library accepts
text documents, not image payloads. All ranking methods return a dictionary
with a `results` list. The following is an illustrative shape, not a measured score:

```python
{"results": [{"document_index": 1, "score": 0.9, "text": "Original document text"}]}
```

With `detail=True`, every method adds a top-level `detail` with the complete
execution record, and each returned result also contains document-level details.
If every candidate is filtered out, the response still contains `"results": []`
and the requested top-level details, including excluded documents.

| Argument | Behavior |
| --- | --- |
| `instruction` | Override the prompt with an instructions/criteria dictionary |
| `threshold` | Keep `score >= threshold`; finite number in [0, 1] |
| `top_k` | None for all eligible results, or a nonnegative output limit |
| `return_documents` | Include original text by default; False omits it |
| `detail` | Include diagnostics; False by default |

Sort by descending score, preserving input order for ties. Duplicate texts remain
separate candidates. Text in results is untruncated. Apply the threshold before
`top_k`; top-k does not reduce scoring work. Empty input and `top_k=0` make no
requests. Unknown kwargs and invalid inputs raise errors rather than being ignored.

`relevance_rerank()` supports listwise and pointwise, not pairwise.

For full relevance diagnostics, request details:

```python
from jev_reranker import JevReranker

reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...")
response = reranker.relevance_rerank(
    "Which planet is called the Red Planet?",
    ["Mars is called the Red Planet.", "Bread is made from flour."],
    threshold=0.2,
    detail=True,
)
```

`response["detail"]["documents"]` includes excluded scores and `passes_threshold`.
`response["detail"]["selection"]` records counts before and after selection.

</details>

<details>
<summary>Authentication and optional dependencies</summary>

Install from PyPI:

```sh
uv add jev-reranker
# Or: pip install jev-reranker
```

Set `TYPESAFE_API_KEY` in `.env` or the environment. An explicit `api_key=` takes precedence over the environment, which takes precedence over `.env`. Keep `.env` out of version control. Set `dotenv_path=None` to disable file loading.

By default, lengths use Python's `len(text)`; no tokenizer dependencies or downloads are required. Select extras as needed:

```sh
uv add 'jev-reranker[all]'
# Tokenizer only: uv add 'jev-reranker[tokenizer]'
# Sentence Transformers only: uv add 'jev-reranker[sentence-transformers]'
```

The `all` extra installs tokenizer dependencies, Sentence Transformers (including PyTorch), and evaluation-only pyarrow. The base installation does not include PyTorch. Model weights are downloaded when used, not when installing the extra.

Use `api_key_env=` to read a different environment variable name.

</details>

<details>
<summary>Custom instructions</summary>

`relevance_rerank()` selects its preset by mode: the default `listwise` uses
`RELEVANCE_INSTRUCTION`; explicit `pointwise` uses `POINTWISE_RELEVANCE_INSTRUCTION`.
The latter judges each document without assuming access to other candidates.
Both default to threshold 0.2, and an explicit per-call `instruction=` takes precedence.
`rerank()` uses its own ranking prompt.

```python
from jev_reranker import JevReranker

reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...", mode="pointwise")
results = reranker.relevance_rerank(
    "Which planet is called the Red Planet?",
    ["Mars is called the Red Planet.", "Bread is made from flour."],
)
```

The pointwise prompt judges each document using only that document and the query.
Use `detail=True` to inspect the prompt and scores, and check the threshold when
switching between listwise and pointwise.

`RERANK_INSTRUCTION`, `PAIRWISE_INSTRUCTION`, `RELEVANCE_INSTRUCTION`, and `POINTWISE_RELEVANCE_INSTRUCTION` in [`instructions.py`](src/jev_reranker/instructions.py) are ordinary dictionaries. You can provide the same structure:

```python
custom = {
    "instructions": "Does {document} contain concrete evidence for `query`?",
    "criteria": {
        "true": "Contains concrete supporting facts",
        "false": "Contains no supporting facts",
    },
}
reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...")
results = reranker.rerank("query", ["document"], instruction=custom, threshold=0.3)
# JevReranker(api_key="YOUR-TYPESAFE-API-KEY...", instruction=custom) sets the instance default.
```

Only the keys `instructions` and `criteria` are accepted. Criteria must contain nonempty `true` and `false` strings. Templates use `{document}`, or `{left}` and `{right}` for pairwise. Unknown keys and invalid placeholders fail before any request. Each call copies its instructions without mutating instance configuration, keeping concurrent calls separate. Legacy constructor arguments `instructions=` and `criteria=` remain supported but cannot be combined with constructor `instruction=`. `relevance_rerank()` selects the relevance preset independently of the constructor's ordinary ranking prompt; its per-call `instruction=` can override that preset.

</details>

<details>
<summary>Async calls, threads, and client lifecycle</summary>

HTTP uses `httpx.AsyncClient` and `asyncio` internally:

```python
import asyncio
from jev_reranker import JevReranker

async def main():
    reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...", mode="pointwise", max_concurrency=8)
    results = await reranker.a_rerank("The Red Planet?", ["Mars.", "Venus."])
    response = await reranker.a_rerank(
        "The Red Planet?", ["Mars.", "Venus."], detail=True,
    )
    # Concurrent queries can share the same instance.
    batches = await asyncio.gather(
        reranker.a_rerank("The Red Planet?", ["Mars.", "Venus."]),
        reranker.a_rerank("Earth's satellite?", ["The Moon.", "The Sun."]),
    )
    print(results, response["detail"]["usage"], batches)

asyncio.run(main())
```

| Sync API | Async API | Return value |
| --- | --- | --- |
| `rerank()` | `a_rerank()` | Ranked results and optional execution details |
| `relevance_rerank()` | `a_relevance_rerank()` | Results passing the relevance threshold and optional execution details |
| `close()` / `with` (optional) | `await aclose()` / `async with` (optional) | Disable the instance and drain active calls |

Sync and async methods share arguments, result structure, and scoring logic, including `top_k`, `return_documents`, and `detail`.

Ordinary instances can be reused across sync calls, caller threads, and async
event loops. Each synchronous call uses a temporary `asyncio.run()` loop; no
persistent loop thread is kept. Calling a sync method inside a running loop still
raises `ConfigurationError`; use the async method instead.

Each call owns a separate, lazily created HTTP client. Success, failure, and
cancellation all close it before the call finishes. A context-local call state
keeps concurrent requests separate. No explicit shutdown is needed for normal use.

HTTP concurrency is bounded across the entire instance. Pointwise uses worker tasks, pairwise processes bounded groups of tasks, and listwise chunks run sequentially. Retries use `asyncio.sleep`. Synchronous tokenizer loading and counting use `asyncio.to_thread`.

Cancellation drains child tasks before propagating `asyncio.CancelledError`. HTTP failures also drain children and preserve public `JevError` exceptions rather than wrapping them in `ExceptionGroup`. `aclose()` and `close()` remain optional compatibility methods: they reject new work and wait for active calls, whose resources are cleaned up automatically. Cancel caller tasks first if you need to stop active work early.

</details>

<details>
<summary>Length limits and optional tokenizers</summary>

`document_max_length` defaults to 4000. The default `len(text)` counts Unicode code points, not bytes or display width. Documents exceeding the limit are sent as prefixes; returned text remains unchanged.

```python
reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...", document_max_length=8000)
results = reranker.rerank("query", ["document"])
# document_max_length=None disables document truncation.
```

For production use, consider token counting. English text in particular often has
many more characters than tokens, so character limits can truncate documents or
split requests earlier than needed. Install the optional tokenizer dependencies:

```sh
uv add 'jev-reranker[tokenizer]'
```

```python
reranker = JevReranker(
    api_key="YOUR-TYPESAFE-API-KEY...",
    tokenizer="google/embeddinggemma-300m",
    document_max_length=8000,  # 8000 tokens here; the default is 4000 tokens.
    split_state_budget=26000,
    split_request_budget=48000,
)
results = reranker.rerank("query", ["document"])
```

Only the tokenizer is fetched on first scoring. Gemma license acceptance and Hugging Face authentication such as `HF_TOKEN` may be required. No model weights, PyTorch, or GPU are needed. The default Gemma revision is pinned; override it with `split_tokenizer_revision="main"`, for example. Other Hub repositories, local `tokenizer.json` files, and objects implementing `encode(text)` / `decode(ids)` are supported. Supplying your own object does not require this library's tokenizer extra.

You can also provide a synchronous `length_fn: Callable[[str], int]`, such as a UTF-8 byte counter:

```python
def utf8_length(text: str) -> int:
    return len(text.encode("utf-8"))

reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...", length_fn=utf8_length, document_max_length=8000)
results = reranker.rerank("query", ["document"])  # Prefix of at most 8000 bytes.
```

The function must return a deterministic nonnegative integer and cannot be combined with `tokenizer`. It may run in a worker thread and receives both documents and serialized JSON state/questions/requests. Custom counters use binary search over text prefixes, followed by a bound check; the longest possible prefix is not guaranteed for nonmonotonic functions.

`split_state_budget` (26000) and `split_request_budget` (48000) use the same selected units. Queries are not truncated. Character counts, custom counts, and Gemma tokens are not guaranteed to match Jev's internal tokens; API context errors are handled separately. Legacy names `document_max_tokens`, `split_state_token_budget`, and `split_request_token_budget` remain aliases and do not force token-based measurement.

</details>

<details>
<summary>Scoring modes and constructor options</summary>

| Mode | Operation | Score |
| --- | --- | --- |
| `listwise` (default) | Ask about candidates in a shared state; split when limits are exceeded | Per-document score from 0 to 1 |
| `pointwise` | Ask independently for each query/document pair | Per-document score from 0 to 1 |
| `pairwise` | Compare every pair in both directions | Mean win probability, not absolute relevance |

```python
reranker = JevReranker(
    api_key="YOUR-TYPESAFE-API-KEY...",
    model="jev-latest",  # Override to select another Jev model.
    mode="listwise",
    document_max_length=4000,  # None disables prefix truncation.
    split_state_budget=26000,
    split_request_budget=48000,
)
results = reranker.rerank("query", ["document"])
```

| Option | Meaning / default |
| --- | --- |
| `model` | Explicit value → `JEV_MODEL` (environment/`.env`) → `jev-latest` |
| `api_key`, `api_key_env` | Explicit key and environment variable name (default `TYPESAFE_API_KEY`) |
| `dotenv_path` | `.env`; does not mutate the process environment |
| `endpoint` | Explicit value → `TYPESAFE_ENDPOINT` → `https://api.typesafe.ai/v1/systemone`; a complete endpoint URL |
| `max_concurrency` | 4 for listwise, 20 for pointwise/pairwise; shared instance-wide HTTP limit |
| `timeout`, `max_retries` | 180 seconds per HTTP I/O; up to 8 retries after the initial attempt |
| `split_tokenizer_name` | None; alias for a string-valued `tokenizer` |
| `split_tokenizer_revision` | Pinned commit for default Gemma; Hub default for other tokenizers unless specified |
| `tokenizer_max_length` | 65536; automatically raises a smaller supplied `model_max_length` and preserves larger values |
| `tokenizer` | None; Hub repository, local tokenizer.json, or encode/decode object |
| `length_fn` | None (uses `len` without a tokenizer); custom synchronous counter |
| `document_max_length` | 4000; None disables document truncation |
| `split_state_budget`, `split_request_budget` | 26000 / 48000 in the selected measurement units |
| `instruction` | Dictionary containing instructions/criteria; also accepted per call |
| `instructions`, `criteria` | Legacy constructor template and criteria |
| `client` | Advanced: borrowed `httpx.AsyncClient`; async methods only, on one loop; caller closes it |
| `transport` | Advanced: borrowed `httpx.AsyncBaseTransport`, such as `httpx.MockTransport`; caller closes it; cannot be combined with `client` |

Explicitly supplied clients and transports are caller-owned and are never closed
by the reranker. A borrowed client intentionally overrides per-call connection
ownership, remains bound to its first event loop, and requires async methods.
Custom transports must support the caller's concurrency and loop usage. These
advanced options are unnecessary for ordinary API calls.

Gemma length settings estimate Jev input size; they do not extend the EmbeddingGemma model's context. Queries and questions also consume the budgets. Partitioned listwise scoring uses different shared contexts and is not guaranteed to produce the same scores as an unsplit request. Pairwise uses n(n−1)/2 requests and is most suitable for small candidate sets.

Named tokenizers disable truncation and padding and count the full input even beyond 65536 tokens. Supplied objects with writable `model_max_length` are adjusted to at least `tokenizer_max_length`, mutating that object. Objects without this attribute are left unchanged. This does not change `document_max_length`.

Custom encoders should not add special tokens or truncate input. Wrap Transformers tokenizers to pass `add_special_tokens=False, truncation=False`. Custom question templates must contain `{document}` for listwise/pointwise or `{left}` and `{right}` for pairwise. Other fields and unknown kwargs are rejected.

```python
reranker = JevReranker(
    api_key="YOUR-TYPESAFE-API-KEY...",
    instructions="Does {document} directly answer `query`?",
    criteria={"true": "Provides the requested facts", "false": "Does not provide the requested facts"},
)
```

</details>

<details>
<summary>Execution details, retries, and errors</summary>

```python
import json
from pathlib import Path
from jev_reranker import JevError, JevReranker

reranker = JevReranker(api_key="YOUR-TYPESAFE-API-KEY...")
try:
    response = reranker.rerank("The Red Planet?", ["Mars.", "Venus."], detail=True)
except JevError as exc:
    # Inspect APIError.status_code or partial execution details in exc.detail.
    raise

Path("rerank-log.json").write_text(
    json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8",
)
```

`response["detail"]` records effective settings, Python/dependency versions, requested/resolved models, tokenizer revision, usage, retries, splits, submitted state/questions, and responses. Per-result `detail` contains `original_length`, `sent_length`, `length_unit`, related request IDs, and pairwise comparisons. `rerank(..., detail=True)` and `relevance_rerank(..., detail=True)` return the same full execution record, including excluded documents.

Detail `schema_version` is 2. `usage.input_tokens` and `output_tokens` are API-reported token counts, distinct from local length estimates. Logs include document text but exclude API keys, authentication headers, and arbitrary environment variables.

HTTP 429/500/502/503/504/529 and transport errors receive exponential backoff with jitter, respecting `Retry-After` up to 60 seconds. Authentication failures and malformed successful responses are not retried. Listwise context errors repartition only the failed candidates. A document or pair that cannot fit raises `ContextLimitError`. Failures are never converted to zero scores. Other public exceptions are `ConfigurationError`, `APIError`, and `ResponseValidationError`, all derived from `JevError`.

Do not mutate constructor settings during use; create another instance for different settings.

</details>

## Evaluation

Evaluate relevance filtering or reranking on a Nano-set benchmark. Use the same
candidate selection and metrics to compare Jev with a Sentence
Transformers CrossEncoder, or inspect how a relevance threshold changes the
retained documents and positives.

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --task relevance --target en
```

Run this command from a source checkout; `examples/eval.py` is a repository script.
This example scores the full hybrid candidate pool. Add `--top-k 10` to evaluate
a controlled ten-candidate pool containing every labeled positive. See the [evaluation guide](docs/eval.md) for dataset selection, dependencies,
metrics, and output files. Evaluation calls the real Jev API and incurs usage charges.

## Contributing

Bug reports and contributions are welcome. For a ranking issue, include a minimal
reproduction, the model and mode, and the behavior you expected. Remove credentials
and private document text before sharing execution details.

Contributor guides:

- [Design and behavior](docs/spec.md)
- [Live test setup](docs/live-validation.md)
- [Repository guidelines](AGENTS.md)
- [Release guide](docs/release.md) and [changelog](CHANGELOG.md)

```sh
git clone https://github.com/hotchpotch/jev-reranker.git
cd jev-reranker
uv sync --locked --dev
uv run --locked tox
```

Normal tests do not call Jev. Live tests require explicit opt-in and credentials;
see the live test guide before running them.

## References

### HAKARI-Bench

[HAKARI-Bench](https://github.com/hakari-bench/hakari-bench) is a related project
by the same author. Its TypeSafe reranker implementation informed this library's
Jev scoring and listwise partitioning design. Its Nano-set benchmark tooling also
informed the evaluation example's hybrid candidate handling and nDCG calculation.

HAKARI-Bench also publishes the
[Nano-set benchmarks](https://huggingface.co/hakari-bench/datasets?search=nano)
used by the evaluation script. See the [evaluation guide](docs/eval.md) for
choosing a dataset and benchmark, the supported data format, and how to run
comparisons.

## License

MIT. See [LICENSE](LICENSE).

## Author

Yuichi Tateno ([@hotchpotch](https://github.com/hotchpotch)).
