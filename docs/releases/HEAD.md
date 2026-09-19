# HEAD

## Relevance filtering and reranking

- Add `JevReranker.rerank()` and `relevance_rerank()`, with async equivalents
  `a_rerank()` and `a_relevance_rerank()`. All return a `results` envelope with
  original document indices, scores, and optional text.
- Provide listwise and pointwise evidence-selection prompts. Relevance filtering
  defaults to threshold 0.2; ordinary reranking defaults to 0.0. Threshold equality
  is included, ties preserve input order, and output `top_k` applies after filtering.
- Support ordinary pairwise ranking for relative-comparison experiments. Relevance
  filtering supports listwise and pointwise only.
- Accept custom instruction dictionaries per instance or call. Per-call overrides
  are isolated from concurrent calls.

## Input handling and execution

- Default to character counting with a 4000-character document prefix limit.
  Support optional tokenizers, custom length counters, and configurable budgets.
  Automatically raise a supplied tokenizer's maximum-length metadata when needed;
  this does not extend an embedding model's context.
- Split long listwise candidate lists and repartition failed chunks after API
  context-limit errors. Preserve original document text in results.
- Use `httpx` and `asyncio` for bounded concurrency, transient-error retries, and
  cancellation cleanup. Each call closes its HTTP client automatically; ordinary
  usage needs no context manager or explicit shutdown.
- Support caller-owned clients and transports for advanced integrations. Borrowed
  clients require async methods on one event loop.
- Read authentication and model settings from explicit arguments, environment
  variables, or `.env`. Validate requests and responses; failures never become
  fabricated zero scores.
- Include optional execution details with every candidate's score, selection
  counts, prompts, model information, lengths, usage, retries, and splits—even
  when no document passes the threshold.
- Keep optional shutdown safe with active calls and multiple shutdown waiters,
  without occupying executor threads while waiting.

## Evaluation and distribution

- Add `examples/eval.py` for compatible 50-query Nano-set benchmarks, defaulting
  to NanoBEIR-en / NanoHotpotQA. Support
  full hybrid pools by default or fixed-size pools containing every labeled
  positive, with deterministic shuffling and nDCG@10.
- Evaluate relevance thresholds with document removal and positive retention
  statistics, distinguishing filtering errors from missing retrieval candidates.
- Compare Jev with local Sentence Transformers CrossEncoders using the same
  candidate selection and metrics. Provide `tokenizer`, `sentence-transformers`,
  and `all` extras while keeping base runtime dependencies minimal.
- Add offline coverage and opt-in multilingual live tests for sync and async APIs.
  Validate wheel installation without tokenizer dependencies.
- Document public usage, evaluation, and release procedures. Generate GitHub
  Release notes from versioned Markdown and attach the distributions published
  to PyPI.
