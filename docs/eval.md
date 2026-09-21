# Evaluating rerankers with `examples/eval.py`

[The evaluation script](../examples/eval.py) evaluates rerankers on compatible
[HAKARI-Bench Nano-set benchmarks](https://huggingface.co/hakari-bench/datasets?search=nano)
and reports mean **nDCG@10**. **NanoBEIR-en / NanoHotpotQA is the default**,
not the only supported benchmark. Choose another benchmark split with `--split`,
or another dataset repository with `--dataset` and `--revision`. It supports two backends: Jev through its API, and Sentence Transformers through a local `CrossEncoder`, such as `BAAI/bge-reranker-v2-m3`.

Here, a **candidate** is an input document to score; a **positive** is a document
labeled relevant in the dataset's query/document relevance judgments (**qrels**).

Both backends share candidate selection, shuffling, document-ID mapping, and metric calculation. This lets you compare their rankings on the same inputs. For relevance filtering, the script also reports how many candidates and labeled positives the threshold removes. This guide covers script usage and metric interpretation; it does not report benchmark results.

## Quick start

Run all commands below from the **root of this repository**. The script explicitly imports `jev_reranker` from this checkout's `src/` directory and verifies that location.

### Jev: full candidate pool

Set `TYPESAFE_API_KEY` in the environment or the root `.env` file, then run:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py
```

This evaluates English HotPotQA with every original hybrid candidate (100 or 101 per query), without limiting the candidate pool or injecting missing positives. It uses listwise scoring, the Gemma tokenizer, seed 42, and four concurrent queries. The model is taken from `JEV_MODEL` in the environment or `.env`, falling back to `jev-latest`. Existing environment variables take precedence over `.env`. This command makes real, billable Jev API calls.

### Relevance scoring and filtering

Use `--task relevance` to select Jev's evidence-usefulness prompt:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --backend jev --task relevance --target en
```

Both commands above score the full original candidate pool. Add `--top-k 10`
to use the controlled ten-candidate policy described below.

`--task rerank` is the default and uses **threshold 0.0**. `--task relevance` uses **threshold 0.2**. Override either with `--threshold`, for example `--threshold 0.33`. A document is kept when `score >= threshold`. These settings are recorded in both the manifest and summary.

The relevance task selects `RELEVANCE_INSTRUCTION` for listwise and
`POINTWISE_RELEVANCE_INSTRUCTION` for pointwise through `a_relevance_rerank()`. It retains the full scoring detail so discarded positives can be audited. It supports listwise and pointwise, not pairwise. The task and threshold options are Jev-only: CrossEncoder scores are raw logits, so the Sentence Transformers backend remains unfiltered and rejects relevance/threshold options.

To score each document independently with the dedicated pointwise relevance prompt:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --task relevance --mode pointwise --target en --top-k 10
```

This still defaults to threshold 0.2. The default mode remains listwise; pointwise
uses one query/document request per candidate and does not share other candidates
as context.

Filtering occurs **after** all candidates have been scored. It reduces the documents sent to a downstream application, not the number of Jev scoring calls. `--top-k 10` still controls the number of input candidates; it does not force ten outputs after filtering.

### Sentence Transformers

Use the `all` extra to install the dependencies for both backends and dataset loading:

```sh
uv run --locked --extra all python examples/eval.py \
  --backend sentence-transformers \
  --model BAAI/bge-reranker-v2-m3 \
  --device cuda:0 --dtype float16 \
  --target en --top-k 10
```

Choose an available GPU with `--device cuda:0`, `cuda:1`, etc. For CPU inference, use `--device cpu --dtype float32`. If omitted, the device is selected by Sentence Transformers; the default dtype is float32.

This backend runs locally. It does not call Jev, require a Jev API key, or require a separate inference server. Model weights are downloaded on first use if not already cached. For reproducible BGE weights, add:

```sh
--model-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
```

### Installation extras

| Extra | Includes |
| --- | --- |
| `tokenizer` | Hugging Face Hub and tokenizers, for the optional Jev token counter |
| `sentence-transformers` | Sentence Transformers and its dependencies, including PyTorch |
| `all` | Both extras above, plus pyarrow for evaluation data loading |

The base package keeps only its HTTP and `.env` runtime dependencies. The checkout also provides pyarrow in the `examples` dependency group, which is why the Jev-only command uses `--group examples`.

For application use, install all optional dependencies with:

```sh
uv add 'jev-reranker[all]'
```

The evaluation script lives in the repository; installing the package alone is not a substitute for checking out the example. Installing dependencies does not itself download model weights.

## Choose the dataset and candidate count

| Option | Selects | Default |
| --- | --- | --- |
| `--target` | Built-in dataset preset (`en` or `ja`) | `en` |
| `--dataset` + `--revision` | Another compatible dataset repository and commit | From the preset |
| `--split` | Benchmark within the dataset | `NanoHotpotQA` |
| `--task` | Jev scoring prompt: `rerank` or `relevance` | `rerank` |
| `--query-limit` | Maximum queries, selected in sorted query-ID order | All dataset queries |
| `--top-k` | Number of **input candidates** per query | No limit |

Unlike the library's `top_k` argument, the script's `--top-k` controls input
selection, not the number of returned results. The metric cutoff stays at 10.


### Built-in language presets

`--target en` is the default. To evaluate Japanese with an explicit 10-candidate policy:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --target ja --top-k 10
```

Both presets default to the `NanoHotpotQA` split and pin the dataset revision.
They are convenience presets, not a restriction to these two repositories:

| Target | Dataset | Revision |
| --- | --- | --- |
| `en` | `hakari-bench/NanoBEIR-en` | `d3962aa8efe48ed79044c5e155b848982667b4ba` |
| `ja` | `hakari-bench/NanoBEIR-ja` | `5c1d5564643f9ca7a8c275688acf09fd940aa5f2` |

### Other Nano-set benchmarks

Browse the [Nano-set dataset list](https://huggingface.co/hakari-bench/datasets?search=nano)
and choose a repository and benchmark split. `--split` selects the benchmark;
`--task` selects the scoring behavior (`rerank` or `relevance`). These are
independent options.

For another split in a built-in dataset, keep `--target` and specify `--split`.
For another repository, specify all three values below. Replace the placeholder
values with the dataset ID, commit SHA, and split name from that repository:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --task relevance \
  --dataset 'hakari-bench/DATASET_NAME' \
  --revision 'DATASET_COMMIT_SHA' \
  --split 'BENCHMARK_SPLIT'
```

`--dataset` requires an explicit `--revision`; `--revision` can also override a
built-in preset's pinned revision. These options work with either backend.

The loader currently requires the following Nano-set layout and constraints:

- A nonempty query set with matching hybrid retrieval rows.
- 100 or 101 hybrid candidates per query.
- Positive-only qrels: every listed query/document pair is treated as relevant;
  graded relevance and explicit negative qrels are not interpreted.
- One file at `<config>/<split>-00000-of-00001.parquet` for each of `queries`,
  `corpus`, `qrels`, and `reranking_hybrid`, with the same columns as the presets.

By default, every query in the selected split is evaluated: a 200-query dataset
runs 200 queries. Add `--query-limit 50` to evaluate only the first 50 query IDs
in sorted order. The limit must be positive; values above the dataset size use
all queries. This is a deterministic subset, not a random sample. The manifest
and summary record the dataset size, requested limit, and evaluated count.

Other Nano-set benchmarks can run when they satisfy this format. The script does
not automatically discover or run every split, and arbitrary dataset layouts or
candidate counts require loader changes. Each invocation evaluates one split.


### Fixed count: include every positive

`--top-k 10` means **10 candidates to score**, not 10 results taken after scoring. You can use another positive count, such as `--top-k 20`.

For each query, the script:

1. Reserves a place for **every qrels-positive document**.
2. Fills the remaining places with the highest-ranked nonpositive documents from the hybrid list.
3. Keeps the selected documents in hybrid order. Positives absent from the hybrid pool are loaded from the corpus and appended in sorted document-ID order.
4. Shuffles the list before sending it to the reranker.

The default NanoHotpotQA split has two positives per query. With 10 candidates, each query therefore has two positives and eight negatives. Other benchmarks can have different numbers of positives. If the requested count cannot accommodate all positives, the script fails instead of dropping a positive.

The hybrid lists already encode the BM25/dense RRF ranking; the dataset does not store numeric hybrid scores. Selection uses that saved order. Because this fixed-count policy uses qrels to guarantee positives, it measures reranking under controlled candidate selection, rather than ordinary end-to-end retrieval performance.

### No candidate limit (default): use the original hybrid pool

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --target ja --top-k none
```

`--top-k none`, `--top-k all`, and bare `--top-k` are equivalent. **Omitting the option entirely also means no candidate limit; this is the default.**

In this mode, the original 100/101 candidates are used without removing or adding documents. In particular, missing positives are **not** injected. Shuffling still happens. The logs identify any positives missing from the pool.

These target and candidate-count options work with both backends. The scoring cutoff remains **nDCG@10**, even when all 100/101 candidates are scored.

## Backend-specific settings

### Jev: full candidate pool

| Option | Default | Purpose |
| --- | --- | --- |
| `--mode` | `listwise` | Also accepts `pointwise` or `pairwise` |
| `--model` | Environment/`.env`, then `jev-latest` | Jev model name; use a fixed version for comparisons |
| `--tokenizer` | `google/embeddinggemma-300m` | Length counter; `none` selects character counting |
| `--document-max-length` | `4000` | Per-document limit in the counter's units |
| `--split-state-budget` | `26000` | Estimated state plus longest question budget, in counter units |
| `--split-request-budget` | `48000` | Estimated whole-request budget, in counter units |
| `--concurrency` | `4` | Concurrent evaluation queries and instance HTTP limit |

The example opts into Gemma token counting, although the library itself defaults to character counting. Gemma may require Hugging Face authentication and acceptance of its license. Query text is not truncated by the Jev document-length setting. Long listwise requests may be split by the library; splitting and API retries are recorded in detail logs.

Both split budgets must be positive integers. They apply to the Jev backend
and use tokenizer tokens, or characters with `--tokenizer none`. These are local
estimates, not the provider's token counts. For relevance scoring with Gemma
and more conservative listwise grouping, use:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --task relevance --tokenizer google/embeddinggemma-300m \
  --document-max-length 4000 \
  --split-state-budget 16000 --split-request-budget 30000
```

This example uses threshold 0.2 and the full original candidate pool. The split
budget defaults remain 26000 and 48000. Smaller budgets may reduce input-limit errors and
subsequent retries, but can increase the number of requests. They change which
documents are scored together, so listwise scores may also change. Compare
results with the same candidate selection and document-length limit. Effective
budgets are recorded in the summary's `model_configuration` and query details.
Pointwise and pairwise use these budgets as preflight checks rather than
splitting their individual requests.

Pairwise compares every document pair, so a full 100-candidate run requires many more requests than a 10-candidate listwise run.

### Sentence Transformers

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | `BAAI/bge-reranker-v2-m3` | CrossEncoder model ID or local path |
| `--model-revision` | Model's default revision | Pin model weights independently of the dataset |
| `--device` | Automatic | For example `cuda:0`, `cuda:1`, or `cpu` |
| `--dtype` | `float32` | Also accepts `float16` or `bfloat16` |
| `--batch-size` | `16` | Query-document pairs per inference batch |
| `--max-length` | Model limit | Token limit for the combined query-document pair |

The adapter uses [Sentence Transformers' CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html) to score query-document pairs. It requires one scalar relevance score per pair, uses identity activation to retain raw logits, and sorts scores descending. Ties preserve input order. Queries are processed serially through a shared model, with batching inside each query.

The model's own tokenizer and context limit apply. Pairs longer than `--max-length` are truncated by the model's tokenizer. Logs include the original pair lengths and the number of truncated pairs. Jev's `--mode`, `--tokenizer`, `--document-max-length`, and concurrency settings do not configure CrossEncoder inference. Jev scores and CrossEncoder logits should not be compared numerically; compare ranking metrics instead.

## Read the metrics

For binary relevance, a relevant document at rank `r` contributes `1 / log2(r + 1)` to DCG. The script sums these contributions through rank 10, divides by the ideal DCG, and reports the arithmetic mean across all 50 queries.

| JSON field | Interpretation |
| --- | --- |
| `ndcg_at_10` | Main metric after filtering: ideal ranking uses **all** positives in the query's qrels |
| `unfiltered_ndcg_at_10` | The same scores before threshold filtering; no additional model call |
| `candidate_ndcg_at_10` | Ideal ranking uses only positives present in the selected pool |
| `hybrid_ndcg_at_10` | Selected candidates in their pre-shuffle hybrid order, using all qrels |
| `shuffled_ndcg_at_10` | Shuffled input order, using all qrels |
| `oracle_ndcg_at_10` | Best attainable ranking within this pool, using all qrels |

For the default NanoHotpotQA split, when both positives are present, putting them at ranks 1 and 2 gives nDCG@10 = **1.0**. The two nDCG variants agree in fixed-count mode because all positives are included.

If an original hybrid pool contains only one of two positives, ranking that one first gives approximately **0.613** on the main metric and **1.0** on the candidate-normalized metric. Its oracle is also approximately 0.613. Check `queries_missing_positives` before interpreting a gap between the two metrics.

The hybrid baseline in fixed-count mode uses the **selected, positive-completed pool**, not the untouched original top 10. Do not compare fixed-count and full-pool runs as if they had identical candidates.

### How many documents and positives were removed?

`summary.json` contains a `filtering` object with:

| Field | Meaning |
| --- | --- |
| `documents_scored`, `documents_retained`, `documents_removed` | Counts of query-document pairs before and after filtering |
| `document_removal_rate` | Removed / scored |
| `positives_in_candidates` | Qrels-positive pairs present in the input pools |
| `positives_retained`, `positives_removed` | How many of those positives survived or were discarded |
| `positive_retention_rate` | Retained positives / positives in candidates; null if none exist |
| `queries_with_removed_positives` | Queries where the threshold discarded at least one positive |
| `empty_result_queries` | Queries with no retained documents |

Each query log also records `removed_document_ids`, `removed_positive_ids`, and the unfiltered ranking. Its execution detail retains every candidate's score, even when the document is absent from the returned results. Qrels are used only to evaluate selection/filtering, never sent as labels in the API request.

A positive absent from the original candidate pool is a **retrieval miss**, not a filtering error. `queries_missing_positives` / `missing_positive_ids` track those separately. Counts are query-document pairs, not globally unique corpus documents. A nonpositive qrels label does not prove the passage is useless, so the document removal count is not a claim about how much true noise was removed.

nDCG's ideal denominator is not recomputed using only retained positives. Dropping a positive can therefore lower nDCG, and an empty ranking scores zero.

## Reproducibility and saved results

Query IDs are sorted. A single `random.Random(seed)` instance shuffles each query's candidates in that order. The default seed is 42. With the same dataset revision, candidate policy, and seed, both backends receive the same documents in the same order.

Each successful run produces:

```text
.live-results/hotpotqa-<UTC>/
  manifest.json       # dataset revision, hashes, selected and shuffled IDs, qrels
  <query-id>.json      # ranking, per-query metrics, raw scores and backend details
  summary.json        # averages and effective configuration
```

The `hotpotqa` directory prefix is historical; the manifest identifies the actual
dataset and split, including when you evaluate another benchmark.

Use `--output <new-directory>` to change the destination. Existing directories are rejected. A failed run can leave completed query logs but does not write a success summary. Data is cached separately by dataset, split, and revision under `.cache/hotpotqa/`.

Result logs contain query/document text and are excluded from Git. Jev detail records requests, usage, retries, and resolved models. CrossEncoder detail records the model revision, device, dtype, package versions, pair lengths, and truncation counts.

For comparisons, check the manifests rather than relying only on command names. `elapsed_seconds` excludes dataset loading and candidate preparation, but includes backend setup/model loading, scoring, and per-query log writes. Compare timings only when setup, warmup, and measurement boundaries match.

## Offline tests

The script includes tests for candidate selection, deterministic shuffling, nDCG, document-index mapping, variable pool sizes, and the CrossEncoder adapter with a fake model. They do not require an API key, GPU, or model download:

```sh
uv run --locked pytest examples/eval.py -q
```

They also run as part of `uv run --locked tox`. Use `python examples/eval.py --help` inside the configured environment to read the in-file usage guide and option list.
