# Evaluating rerankers with `examples/eval.py`

[The evaluation script](../examples/eval.py) runs **50 HotPotQA queries** from NanoBEIR-en or NanoBEIR-ja and reports mean **nDCG@10**. It supports two backends: Jev through its API, and Sentence Transformers through a local `CrossEncoder`, such as `BAAI/bge-reranker-v2-m3`.

Both backends share candidate selection, shuffling, document-ID mapping, and metric calculation. This lets you compare their rankings on the same inputs.

## Quick start

Run all commands below from the **root of this repository**. The script explicitly imports `jev_reranker` from this checkout's `src/` directory and verifies that location.

### Jev

Set `TYPESAFE_API_KEY` in the environment or the root `.env` file, then run:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py
```

This evaluates English HotPotQA with every original hybrid candidate (100 or 101 per query), without filtering or injecting missing positives. It uses listwise scoring, the Gemma tokenizer, seed 42, and four concurrent queries. The model is taken from `JEV_MODEL` in the environment or `.env`, falling back to `jev-latest`. Existing environment variables take precedence over `.env`. This command makes real, billable Jev API calls.

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

This backend runs locally. It does not call Jev, require a Jev API key, or use the TEI server. Model weights are downloaded on first use if not already cached. For reproducible BGE weights, add:

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

After the next package release, an application can install all optional dependencies with:

```sh
uv add 'jev-reranker[all]'
```

**The published 0.0.1 release contains scaffolding only.** Until a new version is published, use this checkout and `uv run --extra all` or `uv sync --locked --extra all`. The evaluation script lives in the repository; installing the package alone is not a substitute for checking out the example. Installing dependencies does not itself download model weights.

## Choose the dataset and candidate count

### English or Japanese

`--target en` is the default. To evaluate Japanese with an explicit 10-candidate policy:

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --target ja --top-k 10
```

Both presets use the `NanoHotpotQA` split and pinned dataset revisions:

| Target | Dataset | Revision |
| --- | --- | --- |
| `en` | `hakari-bench/NanoBEIR-en` | `d3962aa8efe48ed79044c5e155b848982667b4ba` |
| `ja` | `hakari-bench/NanoBEIR-ja` | `5c1d5564643f9ca7a8c275688acf09fd940aa5f2` |

`--revision` overrides the dataset revision. `--dataset` accepts a custom Hub dataset ID but requires an explicit `--revision`. `--split` changes the split. These overrides must follow the supported format: 50 queries, binary qrels, and 100 or 101 hybrid candidates per query, stored in the corresponding `queries`, `corpus`, `qrels`, and `reranking_hybrid` parquet files. This is not a loader for arbitrary benchmark schemas.

### Fixed count: include every positive

`--top-k 10` means **10 candidates to score**, not 10 results taken after scoring. You can use another positive count, such as `--top-k 20`.

For each query, the script:

1. Reserves a place for **every qrels-positive document**.
2. Fills the remaining places with the highest-ranked nonpositive documents from the hybrid list.
3. Keeps the selected documents in hybrid order. Positives absent from the hybrid pool are loaded from the corpus and appended in sorted document-ID order.
4. Shuffles the list before sending it to the reranker.

The HotPotQA presets have two positives per query. With 10 candidates, each query therefore has two positives and eight negatives. If the requested count cannot accommodate all positives, the script fails instead of dropping a positive.

The hybrid lists already encode the BM25/dense RRF ranking; the dataset does not store numeric hybrid scores. Selection uses that saved order. Because this fixed-count policy uses qrels to guarantee positives, it measures reranking under controlled candidate selection, rather than ordinary end-to-end retrieval performance.

### No filtering (default): use the original hybrid pool

```sh
uv run --locked --group examples --extra tokenizer python examples/eval.py \
  --target ja --top-k none
```

`--top-k none`, `--top-k all`, and bare `--top-k` are equivalent. **Omitting the option entirely also means no filtering; this is the default.**

In this mode, the original 100/101 candidates are used without removing or adding documents. In particular, missing positives are **not** injected. Shuffling still happens. The logs identify any positives missing from the pool.

These target and candidate-count options work with both backends. The scoring cutoff remains **nDCG@10**, even when all 100/101 candidates are scored.

## Backend-specific settings

### Jev

| Option | Default | Purpose |
| --- | --- | --- |
| `--mode` | `listwise` | Also accepts `pointwise` or `pairwise` |
| `--model` | Environment/`.env`, then `jev-latest` | Jev model name; use a fixed version for comparisons |
| `--tokenizer` | `google/embeddinggemma-300m` | Length counter; `none` selects character counting |
| `--document-max-length` | `4000` | Per-document limit in the counter's units |
| `--concurrency` | `4` | Concurrent evaluation queries and instance HTTP limit |

The example opts into Gemma token counting, although the library itself defaults to character counting. Gemma may require Hugging Face authentication and acceptance of its license. Query text is not truncated by the Jev document-length setting. Long listwise requests may be split by the library; splitting and API retries are recorded in detail logs.

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
| `ndcg_at_10` | Main metric: ideal ranking uses **all** positives in the query's qrels |
| `candidate_ndcg_at_10` | Ideal ranking uses only positives present in the selected pool |
| `hybrid_ndcg_at_10` | Selected candidates in their pre-shuffle hybrid order, using all qrels |
| `shuffled_ndcg_at_10` | Shuffled input order, using all qrels |
| `oracle_ndcg_at_10` | Best attainable ranking within this pool, using all qrels |

When both positives are present, putting them at ranks 1 and 2 gives nDCG@10 = **1.0**. The two nDCG variants agree in fixed-count mode because all positives are included.

If an unfiltered pool contains only one of two positives, ranking that one first gives approximately **0.613** on the main metric and **1.0** on the candidate-normalized metric. Its oracle is also approximately 0.613. Check `queries_missing_positives` before interpreting a gap between the two metrics.

The hybrid baseline in fixed-count mode uses the **selected, positive-completed pool**, not the untouched original top 10. Do not compare fixed-count and unfiltered runs as if they had identical candidates.

## Reproducibility and saved results

Query IDs are sorted. A single `random.Random(seed)` instance shuffles each query's candidates in that order. The default seed is 42. With the same dataset revision, candidate policy, and seed, both backends receive the same documents in the same order.

Each successful run produces:

```text
.live-results/hotpotqa-<UTC>/
  manifest.json       # dataset revision, hashes, selected and shuffled IDs, qrels
  <query-id>.json      # ranking, per-query metrics, raw scores and backend details
  summary.json        # averages and effective configuration
```

Use `--output <new-directory>` to change the destination. Existing directories are rejected. A failed run can leave completed query logs but does not write a success summary. Data is cached separately by dataset, split, and revision under `.cache/hotpotqa/`.

Result logs contain query/document text and are excluded from Git. Jev detail records requests, usage, retries, and resolved models. CrossEncoder detail records the model revision, device, dtype, package versions, pair lengths, and truncation counts.

For comparisons, check the manifests rather than relying only on command names. `elapsed_seconds` excludes dataset loading and candidate preparation, but includes backend setup/model loading, scoring, and per-query log writes. It is not directly comparable to a warmed, already-running TEI server's request timing.

## Recorded English 10-candidate result

On 2026-09-19, all 50 queries were evaluated with both positives included and seed 42:

| Backend/model | Mean nDCG@10 | Queries with both positives in the top two |
| --- | ---: | ---: |
| Jev listwise, resolved `jev-1.13.0` | 0.995183 | 47/50 |
| Sentence Transformers, `BAAI/bge-reranker-v2-m3` | 0.978154 | 42/50 |

The BGE run used revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, GPU 1, float16, batch size 16, and an 8192-token pair limit. No pairs were truncated. Its manifests matched the Jev run's candidate IDs and input order. The previous TEI/BGE run also scored 0.978154 on these inputs.

These are results for this selected 50-query task, not a general quality claim. See [the example run records](../examples/README.md) for Japanese results and local log paths.

## Offline tests

The script includes tests for candidate selection, deterministic shuffling, nDCG, document-index mapping, variable pool sizes, and the CrossEncoder adapter with a fake model. They do not require an API key, GPU, or model download:

```sh
uv run --locked pytest examples/eval.py -q
```

They also run as part of `uv run --locked tox`. Use `python examples/eval.py --help` inside the configured environment to read the in-file usage guide and option list.

## Recorded English unfiltered result

On 2026-09-19, both backends evaluated all 50 English HotPotQA queries with the
original hybrid candidates, seed 42, and no positive injection. This pinned split
has exactly 100 candidates for every query (5,000 query-document pairs).
Dataset hashes, candidate IDs, and shuffled input order matched across both runs.

| Ranking | Mean nDCG@10 (all qrels) |
| --- | ---: |
| Jev listwise, resolved `jev-1.13.0` | **0.967446** |
| Sentence Transformers / `BAAI/bge-reranker-v2-m3` | **0.948155** |
| Original hybrid order | 0.832519 |
| Oracle within the available candidates | 0.976789 |

Three queries lack one of their positives in the original candidate pool. These
positives were not added; the main metric retains them in its ideal denominator.
Jev achieved nDCG=1.0 on 44 queries and BGE on 39.
BGE used the same pinned revision, GPU 1, float16, batch size 16, and 8192-token
limit as the 10-candidate run above, with no truncated pairs.

Local records:

- Jev: `.live-results/hotpotqa-20260919T100832591545Z/`
- BGE: `.live-results/hotpotqa-20260919T100832594886Z/`

Commands (the explicit `--top-k none` is also the default):

```sh
uv run --locked --extra all python examples/eval.py \
  --backend jev --target en --top-k none
uv run --locked --extra all python examples/eval.py \
  --backend sentence-transformers --model BAAI/bge-reranker-v2-m3 \
  --model-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --device cuda:1 --dtype float16 --target en --top-k none
```
