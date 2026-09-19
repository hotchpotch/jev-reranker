# Design and behavior

## Purpose and public API

This document describes the library's scoring contracts, resource ownership, and
failure behavior. Start with the [README](../README.md) for installation and usage.

The library separates two retrieval decisions: ordering candidates and selecting
evidence for an answer. Both use Jev with different instructions and share the
same request pipeline. The design draws on HAKARI-Bench's TypeSafe reranker while
remaining independent of its evaluation framework and local model dependencies.

`JevReranker(...).rerank(query, documents, **kwargs)` returns a dictionary: `{"results": [...]}` with an optional top-level `detail`. This structure also applies to `relevance_rerank()` and all async equivalents. Inputs are a query string and a sequence of document strings. Results contain `document_index` (zero-based input position), `score`, and, by default, the original `text`. Sort by descending score and preserve input order for ties. Keep duplicate documents as independent candidates. `top_k=None` returns all eligible results, zero returns none, and a positive integer limits the output without reducing the scored candidates. `return_documents=False` omits text.

`rerank(..., detail=True)` returns JSON-serializable results and execution details, including per-document details on each result. It combines multiple requests rather than returning an unmodified HTTP response. With `detail=False`, details are not retained. Invalid inputs and unknown kwargs fail before API calls. Provide synchronous wrappers and both context-manager forms around `a_rerank()` / `a_relevance_rerank()`. The public ranking API consists of these two async methods and their synchronous counterparts. Isolate statistics for concurrent calls.

## Instructions and relevance filtering

`instructions.py` exposes ordinary dictionary presets `RERANK_INSTRUCTION`, `PAIRWISE_INSTRUCTION`, `RELEVANCE_INSTRUCTION`, and `POINTWISE_RELEVANCE_INSTRUCTION`. Dictionaries contain only `instructions: str` and `criteria: {true: str, false: str}`. Reject metadata and unknown keys. Templates require `{document}` for ordinary scoring, or `{left}` and `{right}` for pairwise. Reject other placeholders, format specifiers, and conversions.

Accept `instruction=` on the constructor and each call. Preserve legacy constructor `instructions` / `criteria`, but reject combining them with constructor `instruction`. Deep-copy validated instructions into each `_Run` and pass them through every chunk, retry, and pointwise task. Do not mutate shared instance settings. Record the effective prompt in details.

`relevance_rerank` selects `RELEVANCE_INSTRUCTION` for listwise and `POINTWISE_RELEVANCE_INSTRUCTION` for pointwise, then passes that preset and default `threshold=0.2` through `rerank`. The async path is `a_relevance_rerank` → `a_rerank`. Reuse scoring, partitioning, and error handling; do not provide a separate relevance HTTP implementation or `relevance_filter` method. A per-call instruction overrides the relevance preset. Relevance evaluates absolute usefulness as evidence and supports listwise/pointwise; pairwise's relative win probability raises `ConfigurationError` here.

Ordinary reranking methods default to `threshold=0.0`. Accept only finite numeric thresholds in [0, 1], excluding booleans. Select scored candidates satisfying `score >= threshold`, sort them stably, then apply `top_k`. Do not transform, round, or recalibrate scores. Thresholds change downstream output, not request contents, scored candidates, or billable work. Zero scores survive threshold zero. If all candidates are rejected, return `{"results": []}` and retain top-level execution details when `detail=True`. Empty input and `top_k=0` still make no requests.

The relevance preset credits direct answers, partial answers, concrete linking facts,
and entity identification as useful evidence. The prompt lives in
[`instructions.py`](../src/jev_reranker/instructions.py), which is the source of truth.
The pointwise preset judges specific facts, entity disambiguation, and partial
evidence using only the query and one document. It does not assume access to a
partner document. Listwise remains the default; neither preset guarantees
calibration or quality on a new task. Sync and async routing use the same presets.

Do not send evaluation labels or selection thresholds in the scoring payload.
The default threshold is 0.2; validate it on representative queries. Complete
positive retention and calibrated probabilities are not guaranteed. Partitioning
and pointwise scoring change the shared context and can change usefulness scores.

Detail schema version 2 records `configuration.threshold`, every document's `score` and `passes_threshold` (independent of top-k selection), and `selection.top_k/scored_count/above_threshold_count/returned_count`. Use the top-level details from any ranking method to diagnose excluded documents, even when `results` is empty. Attach effective prompts and thresholds to `JevError` details for failures after input validation.

## Async execution and lifecycle

Use `httpx.AsyncClient` and `asyncio` for HTTP and retries. Each reranking call
owns a context-local operation state and lazily creates one HTTP client on its
first request. Reuse that client for the call's chunks, workers, and retries.
Close it before returning or raising, including cancellation; shield cleanup and
wait for it even if the caller is cancelled again. Empty input and top_k=0 do not
create a client or load a tokenizer. No connection pool survives an ordinary call.

Synchronous methods use `asyncio.run()` per call with no persistent helper thread.
The same instance can be used by multiple synchronous threads, by native async
callers, or sequentially across different event loops. Reject sync methods inside
a running event loop and direct callers to async methods. Async methods preserve
the same arguments, results, tie ordering, and exception types as sync methods.

An instance-wide concurrency limiter coordinates requests across threads and loops
using a lock and awaitable futures. Cancellation removes waiters or releases their
assigned permits without leaking capacity. Retry waits retain their request slot.
Pointwise uses bounded worker tasks; pairwise processes bounded groups of pairs.
Listwise chunks run sequentially. Move tokenizer loading and counting to
`asyncio.to_thread`, retaining the instance tokenizer lock.

Injected `client` and `transport` are advanced, caller-owned resources and are
never closed by the library. Reject specifying both. A borrowed AsyncClient
requires native async methods and binds to the first event loop; callers must
close it on that loop. A transport is wrapped so closing an operation's client
does not close the shared transport. The caller is responsible for that transport's
lifecycle and support for concurrent or cross-loop use.

Each `_Run` keeps independent statistics. Cancellation and failure drain child
tasks before closing the operation's client. Preserve `CancelledError` and public
`JevError` exceptions rather than wrapping them in ExceptionGroup. Cancelled HTTP
traces retain status `cancelled` for diagnostics.

No `with`, `close`, or `aclose` is needed for normal use. These methods remain for
compatibility: explicit closing rejects new work and waits for active calls to
finish their own cleanup. Closing does not close injected resources. Do not mutate
configuration during use.

## Scoring modes and rationale

Listwise and pointwise use Jev's binary decision output (Noul) as a score from
0 to 1 for the configured true/false criteria. This is not a guarantee of
empirically calibrated relevance probability.

- **Listwise (default):** Put the query and candidate set in one state, asking for a separate usefulness score for each document. This shares comparison context and reduces repeated query transmission; the API does not directly return a permutation. Split candidates when they exceed the context budget.
- **Pointwise:** Score each query/document state independently. Adding or removing other candidates does not change that question's context. This supports large candidate sets but requires one request per candidate and repeats the query.
- **Pairwise:** Ask both directions of “Is A more useful than B?” for every unordered pair in a shared state. A's win probability is `(p(A>B) + 1-p(B>A))/2`; B receives its complement. Rank by mean win probability. This supports relative-comparison experiments and mitigates positional bias, at O(n²) cost. It is not the default. A singleton receives 0.5. Scores are relative to the candidate set, not absolute relevance probabilities.

Ordinary reranking uses an English relevance question and true/false criteria. Send multilingual input unchanged as JSON query/documents. Prompts are configurable. The model defaults to `jev-latest` and can be pinned to a specific version.

## Length measurement, tokenizers, and partitioning

Long documents and candidate lists can exceed a request context budget. Prefix
limits bound each submitted document; partitioning bounds the shared listwise
context. These are separate controls: increasing a document limit can require
more listwise chunks, while increasing a split budget cannot restore truncated
text. Both limits use the configured length counter.

Default to Python `len(text)`, counting Unicode code points. The tokenizer is optional; base dependencies are only httpx and python-dotenv. Accept a deterministic synchronous `length_fn: Callable[[str], int]` returning nonnegative integers. Invalid values, including booleans and negatives, or counter failures raise `ConfigurationError`. Run counters in worker threads under an instance counting lock. Use the same measurement for documents and serialized state/questions/requests. Reject combining `length_fn` and `tokenizer`.

`tokenizer="google/embeddinggemma-300m"` lazily fetches tokenizer.json from the Hub and counts tokens. Named tokenizers require the tokenizer extra (`.[tokenizer]` in a checkout). Accept arbitrary encode/decode objects without requiring that extra. No model weights or PyTorch are needed. Use caller authentication/cache when Gemma license acceptance or Hugging Face authentication is required. Pin Gemma to revision `57c266a740f537b4dc058e1b0cda161fd15afa75`; allow overrides such as `split_tokenizer_revision="main"`. Other Hub repositories and local tokenizer.json files are supported. `split_tokenizer_name` aliases named selection.

Recommend tokenizers in production because character counts, especially in English, can cause premature truncation and partitioning. Gemma counts are still not guaranteed to equal Jev's internal tokens. For supplied objects, automatically raise `model_max_length` to `tokenizer_max_length` (default 65536) when smaller, mutating the object; preserve larger values. Leave objects without that attribute unchanged. A read-only attribute that needs raising causes `ConfigurationError`. Named tokenizers use the configured maximum-length metadata and disable truncation/padding, counting all tokens even beyond that value. This does not extend the EmbeddingGemma embedding model's context.

`document_max_length=4000` defaults to character prefixes with len, or decoded token prefixes with tokenizer measurement, re-encoding to verify the limit. Custom counters use binary search over text prefixes and validate the resulting bound; fail if even the empty prefix cannot fit. Nonmonotonic counters need not yield the longest possible prefix. Preserve short texts unchanged. None disables document truncation. Never truncate the query. Keep original text and input indices distinct from submitted prefixes.

Default listwise budgets are `split_state_budget=26000` for state plus the largest question and `split_request_budget=48000` for the full request, in the selected units. Include query, JSON, and questions in estimates. Balance measured load starting with long documents, with chunk document counts differing by at most one. Deterministically shuffle using the query hash and input index. Successfully score each candidate exactly once and combine the returned scores. Tie order uses original input, not partition order. Calibration across different chunk contexts is not guaranteed. Legacy token-named settings remain aliases without forcing token units.

On HTTP 400/422 with `detail.error_type=max_tokens_exceeded`, halve the failed chunk's budget and repartition it. Do not resend successful chunks. If a singleton, pointwise document, or pairwise pair cannot fit, raise `ContextLimitError` without silently truncating further or switching modes.

## Authentication and configuration

Key precedence: explicit `api_key` → actual environment variable named by `api_key_env` → the same key in the selected `.env`. Default `api_key_env='TYPESAFE_API_KEY'` preserves TypeSafe compatibility. Read dotenv without changing process-wide environment variables. `dotenv_path=None` disables it. Treat empty environment values as missing; a missing key raises `ConfigurationError`. OpenAI variables are unnecessary.

Model precedence: explicit model → `JEV_MODEL` in environment/`.env` → `jev-latest`. Endpoint precedence: explicit endpoint → `TYPESAFE_ENDPOINT` → `https://api.typesafe.ai/v1/systemone`. Require an absolute HTTPS URL, allowing local HTTP for tests. Reject URL credentials, query strings, and fragments. Do not follow redirects.

Default `max_concurrency` is 4 for listwise and 20 for pointwise/pairwise, shared across the instance. Listwise chunks remain sequential. The 180-second timeout applies to each HTTP I/O, not the entire ranking call.

## Errors and retries

Retry only HTTP 429/500/502/503/504/529 and httpx transport errors. `max_retries=8` allows eight retries after the initial attempt. Use exponential backoff with jitter and respect Retry-After seconds or HTTP dates, capped at 60 seconds per wait. Fail immediately on 401/403, other validation failures, malformed JSON, missing/extra answers, nonfinite or out-of-range scores, and missing/invalid usage or model data. Never substitute zero scores for failures. Requests with lost responses may still be billed; usage logs are not a billing ledger.

Public exceptions derive from `JevError`: `ConfigurationError`, `APIError` (with `status_code`), `ResponseValidationError`, and `ContextLimitError`. When requested, attach partial execution details to failures. Do not expose response error bodies or Authorization values in exceptions or logs.

## Details and reproducibility

Detail schema version 2 records UTC start time, elapsed seconds, package/Python/platform/dependency versions, endpoint, requested/resolved model, mode, effective instructions/criteria, tokenizer/revision/budgets, timeout/retry/concurrency settings, query/document hashes, original/sent lengths and units, split traces, per-request questions/state/responses/attempts/waits/usage, and aggregate usage. Record `configuration.length_function`, `length_unit`, and request `estimated_length`. API token usage is measured by the service and is distinct from local estimates. Pairwise details include opponent indices and directional judgments.

Do not collect API keys, authentication headers, arbitrary environment variables, or hostnames. Details contain query and document text, so callers control storage and sharing. The library does not write files automatically; callers can use `json.dump(response, ...)`.

## Testing and distribution

Use TDD with async HTTP mocks and small tokenizers for offline tests. Cover rankings, ties, duplicates, kwargs, environment precedence, partition/recovery, retries, validation, details, concurrent calls, sync/async equivalence, borrowed-client loop ownership, cancellation, and per-call client cleanup.

Live E2E runs only with explicit pytest `--live`; normal CI skips it. Use the real `.env` and Gemma tokenizer to verify four-document relevance ordering in English, Japanese, Chinese, Spanish, and mixed languages through both sync and async APIs. Assert expected input indices and strict score differences, not merely sorted output. Diagnose failures through details instead of automatically relaxing expectations when the service changes.

Preserve the uv cooldown and lock. Validate tox, clean builds, strict twine checks, isolated wheel installation, and sdist/wheel contents. Follow the [release guide](release.md) for publishing; never reuse a published version or tag.

## References

- [HAKARI-Bench](https://github.com/hakari-bench/hakari-bench): TypeSafe reranker and partitioning design (MIT).
- https://docs.typesafe.ai/api
- https://docs.typesafe.ai/models
- https://docs.typesafe.ai/cookbooks/rerank_typesafe
- https://huggingface.co/google/embeddinggemma-300m
