r"""Evaluate rerankers on compatible Nano-set benchmarks (default: NanoBEIR-en / NanoHotpotQA).

Run from this repository's root:
    # Jev: English, all original hybrid candidates (100/101 per query).
    # Set TYPESAFE_API_KEY in the environment or root .env; API calls incur usage.
    uv run --locked --group examples --extra tokenizer python examples/eval.py

    # Relevance prompt, threshold 0.2, all positives included in 10 candidates.
    uv run --locked --group examples --extra tokenizer python examples/eval.py --task relevance --top-k 10
    # Standard reranking: --task rerank (default), threshold 0.0.
    # Override either cutoff with --threshold 0.33. Kept scores satisfy >= cutoff.

    # Relevance with Gemma token counting and explicit, smaller split budgets.
    uv run --locked --group examples --extra tokenizer python examples/eval.py \
        --task relevance --tokenizer google/embeddinggemma-300m \
        --document-max-length 4000 --split-state-budget 16000 --split-request-budget 30000

    # Japanese, using every original hybrid candidate (100 or 101 per query).
    uv run --locked --group examples --extra tokenizer python examples/eval.py --target ja --top-k none

    # Sentence Transformers: local CrossEncoder inference, no Jev API key required.
    uv run --locked --extra all python examples/eval.py --backend sentence-transformers \
        --model BAAI/bge-reranker-v2-m3 --device cuda:0 --dtype float16 --target en --top-k 10
    # Use --device cpu --dtype float32 for CPU inference.
    # Pin model weights with --model-revision <commit SHA>.
    # Control CrossEncoder batching and pair length with --batch-size / --max-length.

    # Offline tests embedded in this file; no model downloads or live API calls.
    uv run --locked pytest examples/eval.py -q

Installation:
    The checkout's all extra includes tokenizer dependencies, Sentence Transformers
    (including PyTorch), and pyarrow. The commands above use this checkout's src/.
    For application use: uv add 'jev-reranker[all]'. The evaluation script itself
    runs from a checkout. Model weights are downloaded on first use, not installation.

Candidates and targets:
    --target en (default) or ja selects a pinned dataset revision; --split defaults
    to NanoHotpotQA. Use --split for another benchmark. --dataset requires --revision
    for another compatible dataset repository. All dataset queries are evaluated
    unless --query-limit N is specified. The loader requires 100/101 hybrid
    candidates each and positive-only qrels in the preset parquet layout.
    Browse Nano-set datasets: https://huggingface.co/hakari-bench/datasets?search=nano
    --top-k defaults to none (no candidate limit). A positive count includes ALL
    qrels positives and fills
    remaining slots with the highest hybrid-ranked negatives. Positives outside the
    hybrid pool are added from the corpus. Too many positives for the count is an error.
    --top-k none, --top-k all, or bare --top-k uses the original hybrid pool unchanged:
    no documents removed or added. Some positives may consequently be missing.
    Query IDs are sorted; one random.Random(42) shuffles each selected candidate list.
    --seed changes this shuffle. The evaluation cutoff remains nDCG@10 at any pool size.

Backend settings:
    Jev defaults: rerank task (threshold 0.0), listwise, Gemma tokenizer,
    4000 tokens per document, concurrency 4. --task relevance selects the evidence
    prompt and defaults to threshold 0.2; pairwise is not supported for this task.
    --task relevance and --threshold are Jev-only. CrossEncoder logits are not
    probability scores and remain unfiltered by this example.
    Model: --model, then JEV_MODEL from environment/.env, then jev-latest.
    Environment credentials take precedence over root .env.
    Sentence Transformers defaults: BAAI/bge-reranker-v2-m3, automatic device,
    float32, batch size 16, and the model's pair-token limit. Scores are raw logits;
    ties retain input order. Queries run serially, with batched pairs within each query.
    --split-state-budget (26000) and --split-request-budget (48000) control
    local length estimates for grouping requests, in the selected counter units.
    --mode / --tokenizer / --document-max-length / --concurrency are Jev settings;
    CrossEncoder uses its own tokenizer and --max-length for query+document pairs.

Metrics and output:
    ndcg_at_10 uses ALL qrels positives for its ideal ranking, averaged over
    the evaluated queries.
    candidate_ndcg_at_10 uses positives within the selected pool. They agree when all
    positives are included; unfiltered pools may have a lower attainable oracle score.
    nDCG is computed AFTER filtering; unfiltered_ndcg_at_10 records the same run
    before filtering. The ideal denominator remains unchanged after filtering.
    Logs record documents retained/removed, positives retained/removed, removal
    rates, removed IDs, and empty results. Positive retention uses positives in
    the input pool; missing retrieval positives are counted separately.
    Filtering happens after all candidates are scored, so it does not save Jev calls.
    Data cache: .cache/hotpotqa/<dataset>/<split>/<revision>/.
    Results: .live-results/hotpotqa-<UTC>/{manifest.json, <query-id>.json, summary.json}.
    Logs contain document text and are gitignored. --output requires a new directory.

See docs/eval.md for the full guide and examples/README.md for a short introduction.
"""

from __future__ import annotations

import math
import random
from typing import Any


def select_documents(
    ranking: list[str], relevant: set[str], count: int | None = None
) -> list[str]:
    if not ranking or len(set(ranking)) != len(ranking):
        raise ValueError("Candidates must be nonempty and unique")
    if count is None:
        return ranking.copy()
    if count < 1 or len(ranking) < count:
        raise ValueError(
            "Candidates must be unique and contain at least count documents"
        )
    if not relevant or len(relevant) > count:
        raise ValueError("All positives must fit within the document count")
    negatives = [doc for doc in ranking if doc not in relevant][: count - len(relevant)]
    wanted = relevant | set(negatives)
    # Preserve hybrid order; positives outside the pool follow in stable ID order.
    selected = [doc for doc in ranking if doc in wanted]
    selected.extend(sorted(relevant.difference(ranking)))
    if len(selected) != count or not relevant.issubset(selected):
        raise ValueError("Could not select count documents including all positives")
    return selected


def ndcg(ranking: list[str], relevant: set[str], k: int = 10) -> float:
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    seen = set()
    dcg = 0.0
    for i, doc in enumerate(ranking[:k]):
        if doc in relevant and doc not in seen:
            dcg += 1 / math.log2(i + 2)
        seen.add(doc)
    return dcg / ideal if ideal else 0.0


def test_selection_keeps_top_ten_when_positive_present():
    ranking = [str(i) for i in range(100)]
    assert select_documents(ranking, {"3", "8"}, 10) == ranking[:10]
    assert select_documents(ranking, {"3", "20"}, 10) == ranking[:9] + ["20"]


def test_selection_promotes_highest_positive_including_rank_101():
    ranking = [str(i) for i in range(101)]
    assert select_documents(ranking, {"100"}, 10) == ranking[:9] + ["100"]
    assert select_documents(ranking, {"40", "30"}, 10) == ranking[:8] + ["30", "40"]
    assert select_documents(ranking, {"100", "outside"}, 10) == ranking[:8] + [
        "100",
        "outside",
    ]


def test_invalid_candidates_fail_instead_of_dropping_queries():
    import pytest

    for ranking, positives in [
        (["a"] * 100, {"a"}),
        (["a"], {"a"}),
        ([str(i) for i in range(100)], set()),
        ([str(i) for i in range(100)], {str(i) for i in range(11)}),
    ]:
        with pytest.raises(ValueError):
            select_documents(ranking, positives, 10)


def test_ndcg_uses_all_qrels_and_log_discount():
    import pytest

    assert ndcg(["a", "b", "c"], {"a", "b"}) == 1.0
    assert ndcg(["a", "x"], {"a", "b"}) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ndcg(["x", "a"], {"a"}) == pytest.approx(1 / math.log2(3))
    assert ndcg(["x"] * 10 + ["a"], {"a"}) == 0


def test_shuffle_is_reproducible_and_preserves_candidates():
    rankings = {"q": [str(i) for i in range(101)]}
    a = prepare_jobs({"q": "query"}, rankings, {"q": {"100"}}, count=10)
    b = prepare_jobs({"q": "query"}, rankings, {"q": {"100"}}, count=10)
    assert a == b
    _, selected, shuffled = a[0]
    assert shuffled == ["7", "3", "2", "8", "5", "6", "100", "4", "0", "1"]
    assert set(selected) == set(shuffled)


def test_unfiltered_preserves_100_or_101_candidates_without_positive_injection():
    for count in (100, 101):
        ranking = [str(i) for i in range(count)]
        selected = select_documents(ranking, {"0", "outside"})
        assert selected == ranking
        assert selected is not ranking
        jobs = prepare_jobs({"q": "query"}, {"q": ranking}, {"q": {"outside"}})
        assert set(jobs[0][2]) == set(ranking)
        assert len(jobs[0][2]) == count


def test_candidate_count_cli_and_targets():
    import argparse

    import pytest

    assert candidate_count("none") is None
    assert candidate_count("all") is None
    assert candidate_count("20") == 20
    for value in ("0", "-1", "oops"):
        with pytest.raises(argparse.ArgumentTypeError):
            candidate_count(value)
    assert TARGETS["ja"][0] == "hakari-bench/NanoBEIR-ja"
    assert TARGETS["ja"][1] != TARGETS["en"][1]


DATASET = "hakari-bench/NanoBEIR-en"
REVISION = "d3962aa8efe48ed79044c5e155b848982667b4ba"
SPLIT = "NanoHotpotQA"
TARGETS = {
    "en": (DATASET, REVISION),
    "ja": ("hakari-bench/NanoBEIR-ja", "5c1d5564643f9ca7a8c275688acf09fd940aa5f2"),
}


def candidate_count(value):
    import argparse

    if value.lower() in ("none", "all"):
        return None
    try:
        count = int(value)
        if count > 0:
            return count
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use a positive integer or 'none'/'all'")


def prepare_jobs(queries, hybrid, qrels, seed=42, count=None):
    rng = random.Random(seed)
    jobs = []
    for qid in sorted(queries):
        selected = select_documents(hybrid[qid], qrels[qid], count)
        shuffled = selected.copy()
        rng.shuffle(shuffled)
        jobs.append((qid, selected, shuffled))
    return jobs


def load_data(cache, revision, dataset=DATASET, split=SPLIT):
    """Read immutable parquet files; pyarrow is an examples-only dependency."""
    import hashlib
    import importlib

    import httpx

    parquet = importlib.import_module("pyarrow.parquet")
    tables, hashes = {}, {}
    for config in ("queries", "corpus", "qrels", "reranking_hybrid"):
        path = (
            cache / dataset.replace("/", "--") / split / revision / f"{config}.parquet"
        )
        if not path.exists():
            url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{config}/{split}-00000-of-00001.parquet"
            response = httpx.get(url, follow_redirects=True, timeout=120)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_bytes(response.content)
            temp.replace(path)
        hashes[config] = hashlib.sha256(path.read_bytes()).hexdigest()
        tables[config] = parquet.read_table(path).to_pylist()
    queries = {r["_id"]: r["text"] for r in tables["queries"]}
    corpus = {r["_id"]: r["text"] for r in tables["corpus"]}
    qrels = {qid: set() for qid in queries}
    for row in tables["qrels"]:
        qrels[row["query-id"]].add(row["corpus-id"])
    hybrid = {r["query-id"]: r["corpus-ids"] for r in tables["reranking_hybrid"]}
    if not queries or set(queries) != set(hybrid):
        raise ValueError("Expected nonempty matching query and hybrid rows")
    for qid in queries:
        if len(hybrid[qid]) not in (100, 101):
            raise ValueError(f"Expected 100/101 candidates for {qid}")
        select_documents(hybrid[qid], qrels[qid], None)
        if not (set(hybrid[qid]) | qrels[qid]).issubset(corpus):
            raise ValueError(f"Missing corpus documents for {qid}")
    return queries, corpus, qrels, hybrid, hashes


def test_loader_accepts_variable_query_counts_and_checks_alignment(
    tmp_path, monkeypatch
):
    import importlib
    from types import SimpleNamespace

    import pytest

    tables = {}
    original_import = importlib.import_module

    def read_table(path):
        return SimpleNamespace(to_pylist=lambda: tables[path.stem])

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(read_table=read_table)
            if name == "pyarrow.parquet"
            else original_import(name)
        ),
    )
    cache = tmp_path / DATASET.replace("/", "--") / SPLIT / REVISION
    cache.mkdir(parents=True)
    for config in ("queries", "corpus", "qrels", "reranking_hybrid"):
        (cache / f"{config}.parquet").write_bytes(b"offline fixture")
    tables["corpus"] = [{"_id": str(i), "text": "document"} for i in range(100)]
    for count in (1, 50, 200):
        tables["queries"] = [{"_id": str(i), "text": "query"} for i in range(count)]
        tables["qrels"] = [{"query-id": str(i), "corpus-id": "0"} for i in range(count)]
        tables["reranking_hybrid"] = [
            {"query-id": str(i), "corpus-ids": [str(j) for j in range(100)]}
            for i in range(count)
        ]
        queries, _, _, hybrid, hashes = load_data(tmp_path, REVISION)
        assert len(queries) == count
        assert set(queries) == set(hybrid)
        assert len(hashes) == 4
    tables["reranking_hybrid"].pop()
    with pytest.raises(ValueError, match="matching query and hybrid"):
        load_data(tmp_path, REVISION)
    for config in ("queries", "qrels", "reranking_hybrid"):
        tables[config] = []
    with pytest.raises(ValueError, match="nonempty"):
        load_data(tmp_path, REVISION)


def cross_encoder_results(scores, count):
    """Validate one scalar per document; preserve input order for tied scores."""
    if len(scores) != count:
        raise ValueError("CrossEncoder must return one score per document")
    try:
        values = [float(score) for score in scores]
    except (TypeError, ValueError):
        raise ValueError("CrossEncoder must return scalar relevance scores") from None
    if not all(math.isfinite(score) for score in values):
        raise ValueError("CrossEncoder returned non-finite scores")
    return [
        {"document_index": i, "score": values[i]}
        for i in sorted(range(count), key=lambda i: (-values[i], i))
    ]


class SentenceTransformersRanker:
    """Example-only adapter; no Jev credentials or tokenizer are used."""

    def __init__(self, args):
        import asyncio

        self.args = args
        self.encoder: Any = None
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        import asyncio
        import importlib
        import time
        from importlib.metadata import version

        try:
            st: Any = importlib.import_module("sentence_transformers")
            torch = importlib.import_module("torch")
        except ImportError:
            raise RuntimeError(
                "Install jev-reranker[all] (checkout: uv sync --extra all)"
            ) from None
        started = time.perf_counter()
        self.encoder = await asyncio.to_thread(
            st.CrossEncoder,
            self.args.model or "BAAI/bge-reranker-v2-m3",
            revision=self.args.model_revision,
            device=self.args.device,
            max_length=self.args.max_length,
            model_kwargs={"torch_dtype": getattr(torch, self.args.dtype)},
            activation_fn=torch.nn.Identity(),
            trust_remote_code=False,
        )
        if self.encoder.config.num_labels != 1:
            raise ValueError("Use a CrossEncoder with one relevance score per pair")
        limit = getattr(self.encoder, "max_seq_length", None)
        if limit is None:
            limit = self.encoder.max_length
        self.max_length = int(limit)
        self.configuration = {
            "backend": "sentence-transformers",
            "model": self.args.model or "BAAI/bge-reranker-v2-m3",
            "requested_revision": self.args.model_revision,
            "resolved_revision": getattr(self.encoder.config, "_commit_hash", None),
            "device": str(self.encoder.device),
            "dtype": self.args.dtype,
            "max_length": self.max_length,
            "batch_size": self.args.batch_size,
            "activation": "identity (raw logits)",
            "load_seconds": time.perf_counter() - started,
            "sentence_transformers_version": version("sentence-transformers"),
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
        }
        return self

    async def __aexit__(self, *_):
        del self.encoder

    def _rank(self, query, docs):
        import time

        started = time.perf_counter()
        pairs = [(query, doc) for doc in docs]
        encoded = self.encoder.tokenizer(pairs, truncation=False, padding=False)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        scores = self.encoder.predict(
            pairs,
            batch_size=self.args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return {
            "results": cross_encoder_results(scores, len(docs)),
            "detail": {
                "configuration": self.configuration,
                "elapsed_seconds": time.perf_counter() - started,
                "pair_token_lengths": lengths,
                "truncated_pairs": sum(n > self.max_length for n in lengths),
            },
        }

    async def a_rerank(self, query, docs, *, detail=True):
        import asyncio

        # One model instance, serialized inference; batching happens in predict().
        async with self.lock:
            task = asyncio.create_task(asyncio.to_thread(self._rank, query, docs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Keep the model alive until its worker has actually finished.
                await task
                raise


def filter_statistics(input_ids, retained_ids, relevant):
    retained = set(retained_ids)
    removed = [doc for doc in input_ids if doc not in retained]
    removed_positive = [doc for doc in removed if doc in relevant]
    return {
        "documents_scored": len(input_ids),
        "documents_retained": len(retained_ids),
        "documents_removed": len(removed),
        "positives_in_candidates": len(set(input_ids) & relevant),
        "positives_retained": len(retained & relevant),
        "positives_removed": len(removed_positive),
        "removed_document_ids": removed,
        "removed_positive_ids": removed_positive,
    }


async def evaluate(args, root):
    import asyncio
    import json
    import sys
    import time
    from datetime import UTC, datetime
    from pathlib import Path
    from statistics import mean

    # Always prefer this checkout, even when an older PyPI package is installed.
    sys.path.insert(0, str(root / "src"))
    import jev_reranker
    from jev_reranker import JevReranker

    if Path(jev_reranker.__file__).resolve().parent != root / "src/jev_reranker":
        raise RuntimeError("jev_reranker was imported from outside this checkout")
    queries, corpus, qrels, hybrid, hashes = load_data(
        args.cache, args.revision, args.dataset, args.split
    )
    dataset_query_count = len(queries)
    queries = {qid: queries[qid] for qid in sorted(queries)[: args.query_limit]}
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = prepare_jobs(queries, hybrid, qrels, args.seed, args.top_k)
    manifest = {
        "backend": args.backend,
        "task": args.task,
        "threshold": args.threshold,
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "file_sha256": hashes,
        "seed": args.seed,
        "query_count": len(jobs),
        "dataset_query_count": dataset_query_count,
        "query_limit": args.query_limit,
        "query_selection": "first query IDs in sorted order, up to query_limit",
        "documents_per_query": args.top_k,
        "candidate_count_min": min(len(d) for _, _, d in jobs),
        "candidate_count_max": max(len(d) for _, _, d in jobs),
        "library_path": jev_reranker.__file__,
        "started_at": datetime.now(UTC).isoformat(),
        "selection": (
            "unfiltered hybrid candidates; no positive injection"
            if args.top_k is None
            else f"all qrels positives plus highest hybrid negatives to total {args.top_k}; missing positives from corpus"
        ),
        "shuffle": "one Random(seed), queries sorted by query ID",
        "metric": "binary nDCG@10; IDCG uses all query qrels",
        "queries": [
            {
                "query_id": q,
                "hybrid_selected": s,
                "input_ids": d,
                "relevant_ids": sorted(qrels[q]),
            }
            for q, s, d in jobs
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    rows = []
    if args.backend == "sentence-transformers":
        ranker = SentenceTransformersRanker(args)
    else:
        ranker = JevReranker(
            dotenv_path=root / ".env",
            mode=args.mode,
            model=args.model,
            tokenizer=None if args.tokenizer == "none" else args.tokenizer,
            document_max_length=args.document_max_length,
            split_state_budget=args.split_state_budget,
            split_request_budget=args.split_request_budget,
            max_concurrency=args.concurrency,
        )
    async with ranker as reranker:

        async def run(job):
            qid, selected, shuffled = job
            async with semaphore:
                if args.backend == "jev":
                    method = (
                        reranker.a_relevance_rerank
                        if args.task == "relevance"
                        else reranker.a_rerank
                    )
                    raw = await method(
                        queries[qid],
                        [corpus[d] for d in shuffled],
                        threshold=args.threshold,
                        detail=True,
                    )
                    all_results = sorted(
                        raw["detail"]["documents"],
                        key=lambda r: (-r["score"], r["document_index"]),
                    )
                else:
                    raw = await reranker.a_rerank(
                        queries[qid],
                        [corpus[d] for d in shuffled],
                        detail=True,
                    )
                    all_results = raw["results"]
            all_indices = [r["document_index"] for r in all_results]
            if sorted(all_indices) != list(range(len(shuffled))):
                raise ValueError(f"Incomplete or duplicate scores for {qid}")
            indices = [r["document_index"] for r in raw["results"]]
            expected = [
                r["document_index"]
                for r in all_results
                if args.threshold is None or r["score"] >= args.threshold
            ]
            if indices != expected:
                raise ValueError(f"Filtered results disagree with threshold for {qid}")
            ranking = [shuffled[i] for i in indices]
            full_ranking = [shuffled[i] for i in all_indices]
            filtering = filter_statistics(shuffled, ranking, qrels[qid])
            selected_relevant = qrels[qid].intersection(selected)
            if args.top_k is not None and selected_relevant != qrels[qid]:
                raise ValueError(f"Selected documents omit positives for {qid}")
            row = {
                "query_id": qid,
                "ranking": ranking,
                "unfiltered_ranking": full_ranking,
                "filtering": filtering,
                "unfiltered_ndcg_at_10": ndcg(full_ranking, qrels[qid]),
                "ndcg_at_10": ndcg(ranking, qrels[qid]),
                "candidate_ndcg_at_10": ndcg(ranking, selected_relevant),
                "hybrid_ndcg_at_10": ndcg(selected, qrels[qid]),
                "shuffled_ndcg_at_10": ndcg(shuffled, qrels[qid]),
                "oracle_ndcg_at_10": ndcg(
                    sorted(selected, key=lambda d: d not in qrels[qid]), qrels[qid]
                ),
                "positive_promoted": selected != hybrid[qid][: args.top_k],
                "candidate_count": len(shuffled),
                "missing_positive_ids": sorted(qrels[qid].difference(selected)),
                "positives_outside_hybrid": sorted(qrels[qid].difference(hybrid[qid])),
                "raw": raw,
            }
            (args.output / f"{qid}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2)
            )
            rows.append(row)
            print(
                f"{len(rows):2}/{len(jobs)} {qid} nDCG@10={row['ndcg_at_10']:.6f} "
                f"kept={len(ranking)}/{len(shuffled)} positives_removed={filtering['positives_removed']}",
                flush=True,
            )

        tasks = [asyncio.create_task(run(job)) for job in jobs]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    totals = {
        key: sum(row["filtering"][key] for row in rows)
        for key in (
            "documents_scored",
            "documents_retained",
            "documents_removed",
            "positives_in_candidates",
            "positives_retained",
            "positives_removed",
        )
    }
    totals["document_removal_rate"] = (
        totals["documents_removed"] / totals["documents_scored"]
    )
    totals["positive_retention_rate"] = (
        totals["positives_retained"] / totals["positives_in_candidates"]
        if totals["positives_in_candidates"]
        else None
    )
    totals["queries_with_removed_positives"] = sum(
        bool(row["filtering"]["positives_removed"]) for row in rows
    )
    totals["empty_result_queries"] = sum(not row["ranking"] for row in rows)
    summary = {
        "query_count": len(rows),
        "dataset_query_count": dataset_query_count,
        "query_limit": args.query_limit,
        "filtering": totals,
        "backend": args.backend,
        "task": args.task,
        "threshold": args.threshold,
        "dataset": args.dataset,
        "split": args.split,
        "queries_missing_positives": sum(bool(r["missing_positive_ids"]) for r in rows),
        "documents_per_query": args.top_k,
        "candidate_count_min": min(len(d) for _, _, d in jobs),
        "candidate_count_max": max(len(d) for _, _, d in jobs),
        "elapsed_seconds": time.perf_counter() - start,
        "mode": args.mode if args.backend == "jev" else "cross-encoder",
        "tokenizer": args.tokenizer if args.backend == "jev" else "model tokenizer",
        "model_configuration": rows[0]["raw"]["detail"]["configuration"],
        "positive_promoted_queries": sum(row["positive_promoted"] for row in rows),
        **{
            key: mean(row[key] for row in rows)
            for key in (
                "ndcg_at_10",
                "unfiltered_ndcg_at_10",
                "candidate_ndcg_at_10",
                "hybrid_ndcg_at_10",
                "shuffled_ndcg_at_10",
                "oracle_ndcg_at_10",
            )
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    import argparse
    import asyncio
    from datetime import UTC, datetime
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        choices=("rerank", "relevance"),
        default="rerank",
        help="Jev prompt preset; relevance retains supporting evidence",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Jev score cutoff: rerank default 0.0, relevance default 0.2",
    )
    parser.add_argument(
        "--backend", choices=("jev", "sentence-transformers"), default="jev"
    )
    parser.add_argument(
        "--device", default=None, help="CrossEncoder device, e.g. cuda:1 or cpu"
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="CrossEncoder batch size"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="CrossEncoder query+document token limit; default model limit",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="CrossEncoder model revision (not dataset revision)",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Evaluate at most N queries in sorted ID order; default all queries",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target", choices=tuple(TARGETS), default="en", help="NanoBEIR language"
    )
    parser.add_argument(
        "--dataset", help="Override the target Hub dataset ID; requires --revision"
    )
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument(
        "--revision", default=None, help="Default: pinned commit for --target"
    )
    parser.add_argument(
        "--top-k",
        type=candidate_count,
        nargs="?",
        const=None,
        default=None,
        help="Candidate count; default none (unfiltered). Set 10 for ten candidates",
    )
    parser.add_argument(
        "--mode", choices=("listwise", "pointwise", "pairwise"), default="listwise"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Jev: env/.env then jev-latest; CrossEncoder: BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument(
        "--tokenizer",
        default="google/embeddinggemma-300m",
        help="Hub name or 'none' for len",
    )
    parser.add_argument("--document-max-length", type=int, default=4000)
    parser.add_argument(
        "--split-state-budget",
        type=int,
        default=26000,
        help="Jev estimated state plus longest question budget, in counter units",
    )
    parser.add_argument(
        "--split-request-budget",
        type=int,
        default=48000,
        help="Jev estimated whole-request budget, in counter units",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=root / ".cache/hotpotqa")
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / ".live-results"
        / ("hotpotqa-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    args = parser.parse_args()
    if args.backend == "sentence-transformers":
        if args.task != "rerank" or args.threshold is not None:
            parser.error(
                "--task relevance and --threshold are Jev-only; CrossEncoder uses raw logits"
            )
    else:
        if args.task == "relevance" and args.mode == "pairwise":
            parser.error("Relevance scoring requires listwise or pointwise")
        if args.threshold is None:
            args.threshold = 0.2 if args.task == "relevance" else 0.0
        if not 0 <= args.threshold <= 1:
            parser.error("--threshold must be finite and between 0 and 1")
    if args.dataset is not None and args.revision is None:
        parser.error("--dataset requires --revision")
    target_dataset, target_revision = TARGETS[args.target]
    args.dataset = args.dataset or target_dataset
    args.revision = args.revision or target_revision
    if args.batch_size < 1 or (args.max_length is not None and args.max_length < 1):
        parser.error("--batch-size and --max-length must be positive")
    if args.query_limit is not None and args.query_limit < 1:
        parser.error("--query-limit must be positive")
    if args.split_state_budget < 1 or args.split_request_budget < 1:
        parser.error("--split-state-budget and --split-request-budget must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    asyncio.run(evaluate(args, root))


def test_evaluation_maps_shuffled_indices_and_writes_complete_run(
    tmp_path, monkeypatch
):
    import argparse
    import asyncio
    import json
    import sys
    from pathlib import Path

    import httpx

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import jev_reranker

    queries = {str(i): "query" for i in range(200)}
    corpus = {str(i): f"document {i}" for i in range(101)}
    qrels = {q: {"99"} for q in queries}
    hybrid = {q: list(corpus)[: 100 + int(q) % 2] for q in queries}
    monkeypatch.setitem(
        evaluate.__globals__,
        "load_data",
        lambda *_: (queries, corpus, qrels, hybrid, {}),
    )
    original = jev_reranker.JevReranker

    def respond(request):
        body = json.loads(request.content)
        is_relevance = next(iter(body["questions"].values()))[
            "instructions"
        ].startswith(("Score how useful", "How useful is"))
        if "document" in body["state"] and is_relevance:
            assert (
                body["questions"]["relevant"]["criteria"]
                == (jev_reranker.POINTWISE_RELEVANCE_INSTRUCTION["criteria"])
            )
        documents = body["state"].get("documents")
        if documents is None:
            documents = {"relevant": body["state"]["document"]}
        positive_score = 0.1 if is_relevance else 1.0
        return httpx.Response(
            200,
            json={
                "model": "offline",
                "answers": {
                    key: {
                        "type": "noul",
                        "noul": positive_score if text == "document 99" else 0.0,
                    }
                    for key, text in documents.items()
                },
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    def offline_ranker(**kwargs):
        kwargs["api_key"] = "test"
        kwargs["dotenv_path"] = None
        kwargs["transport"] = httpx.MockTransport(respond)
        return original(**kwargs)

    monkeypatch.setattr(jev_reranker, "JevReranker", offline_ranker)
    # tox may already have imported its installed package; this in-process test
    # exercises the runner with that same implementation and a fake transport.
    monkeypatch.setattr(
        jev_reranker, "__file__", str(root / "src/jev_reranker/__init__.py")
    )
    args = argparse.Namespace(
        backend="jev",
        task="rerank",
        threshold=0.0,
        cache=tmp_path,
        revision=REVISION,
        dataset=DATASET,
        split=SPLIT,
        top_k=10,
        query_limit=None,
        output=tmp_path / "run",
        seed=42,
        concurrency=4,
        mode="listwise",
        model=None,
        tokenizer="none",
        document_max_length=4000,
        split_state_budget=16000,
        split_request_budget=30000,
    )
    for count, task, threshold, mode, limit in (
        (10, "rerank", 0.0, "listwise", None),
        (None, "rerank", 0.0, "listwise", 50),
        (10, "relevance", 0.2, "listwise", 300),
        (10, "relevance", 0.1, "listwise", 1),
        (10, "relevance", 0.2, "pointwise", 10),
        (10, "relevance", 0.1, "pointwise", 10),
    ):
        args.query_limit = limit
        expected_count = min(limit or 200, 200)
        args.top_k, args.task, args.threshold, args.mode = count, task, threshold, mode
        args.output = tmp_path / f"run-{count}-{task}-{threshold}-{mode}"
        asyncio.run(evaluate(args, root))
        summary = json.loads((args.output / "summary.json").read_text())
        loses_positive = task == "relevance" and threshold == 0.2
        assert summary["query_count"] == expected_count
        manifest = json.loads((args.output / "manifest.json").read_text())
        for record in (manifest, summary):
            assert record["query_count"] == expected_count
            assert record["dataset_query_count"] == 200
            assert record["query_limit"] == limit
        assert [q["query_id"] for q in manifest["queries"]] == sorted(queries)[:limit]
        assert summary["model_configuration"]["split_state_budget"] == 16000
        assert summary["model_configuration"]["split_request_budget"] == 30000
        assert summary["ndcg_at_10"] == (0 if loses_positive else 1)
        assert summary["unfiltered_ndcg_at_10"] == 1
        assert summary["positive_promoted_queries"] == (expected_count if count else 0)
        assert summary["candidate_count_min"] == (10 if count else 100)
        assert summary["candidate_count_max"] == (10 if count else 101)
        assert len(list(args.output.glob("*.json"))) == expected_count + 2
        assert summary["filtering"]["positives_removed"] == (
            expected_count if loses_positive else 0
        )
        assert summary["filtering"]["positive_retention_rate"] == (
            0 if loses_positive else 1
        )
        assert summary["filtering"]["empty_result_queries"] == (
            expected_count if loses_positive else 0
        )
        if task == "rerank":
            assert summary["filtering"]["documents_removed"] == 0
        else:
            assert summary["filtering"]["documents_retained"] == (
                0 if loses_positive else expected_count
            )
        row = json.loads((args.output / "0.json").read_text())
        if loses_positive:
            assert row["ranking"] == []
            assert row["filtering"]["removed_positive_ids"] == ["99"]
        else:
            assert row["ranking"][0] == "99"


def test_cross_encoder_scores_keep_input_mapping_and_validate():
    import pytest

    assert cross_encoder_results([0.1, 0.9, 0.9], 3) == [
        {"document_index": 1, "score": 0.9},
        {"document_index": 2, "score": 0.9},
        {"document_index": 0, "score": 0.1},
    ]
    for scores in ([1], [float("nan"), 1], [[0.1, 0.9], [0.2, 0.8]]):
        with pytest.raises(ValueError):
            cross_encoder_results(scores, 2)


def test_sentence_transformers_adapter_without_gpu_or_api(monkeypatch):
    import argparse
    import asyncio
    import importlib.metadata
    import sys
    from types import SimpleNamespace

    calls = {}

    class FakeEncoder:
        config = SimpleNamespace(num_labels=1, _commit_hash="resolved-sha")
        device = "cpu"
        max_length = 16

        def __init__(self, model, **kwargs):
            calls["model"] = model
            calls["init"] = kwargs

        def tokenizer(self, pairs, **kwargs):
            assert kwargs == {"truncation": False, "padding": False}
            return {"input_ids": [[0] * 20, [0] * 5]}

        def predict(self, pairs, **kwargs):
            calls["pairs"] = pairs
            calls["predict"] = kwargs
            return [-1.0, 2.0]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeEncoder)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32", nn=SimpleNamespace(Identity=lambda: "identity")
        ),
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-version")
    args = argparse.Namespace(
        model="test/model",
        model_revision="sha",
        device="cpu",
        max_length=16,
        dtype="float32",
        batch_size=2,
    )

    async def run():
        async with SentenceTransformersRanker(args) as ranker:
            raw = await ranker.a_rerank("query", ["negative", "positive"])
        assert raw["results"][0] == {"document_index": 1, "score": 2.0}
        assert raw["detail"]["truncated_pairs"] == 1
        assert raw["detail"]["configuration"]["resolved_revision"] == "resolved-sha"

    asyncio.run(run())
    assert calls["pairs"] == [("query", "negative"), ("query", "positive")]
    assert calls["init"]["revision"] == "sha"
    assert calls["init"]["activation_fn"] == "identity"
    assert calls["predict"]["batch_size"] == 2


def test_cli_default_has_no_candidate_limit(monkeypatch):
    import sys

    seen = []

    async def capture(args, root):
        seen.append(args.top_k)

    monkeypatch.setitem(main.__globals__, "evaluate", capture)
    for options in ([], ["--top-k", "10"], ["--top-k", "none"]):
        monkeypatch.setattr(sys, "argv", ["eval.py", *options])
        main()
    assert seen == [None, 10, None]


def test_cli_split_budgets_defaults_overrides_and_validation(monkeypatch):
    import sys

    import pytest

    seen = []

    async def capture(args, root):
        seen.append((args.split_state_budget, args.split_request_budget))

    monkeypatch.setitem(main.__globals__, "evaluate", capture)
    for options in (
        [],
        ["--split-state-budget", "16000"],
        ["--split-request-budget", "30000"],
        ["--split-state-budget", "16000", "--split-request-budget", "30000"],
    ):
        monkeypatch.setattr(sys, "argv", ["eval.py", *options])
        main()
    assert seen == [(26000, 48000), (16000, 48000), (26000, 30000), (16000, 30000)]
    for option in ("--split-state-budget", "--split-request-budget"):
        for value in ("0", "-1", "1.5"):
            monkeypatch.setattr(sys, "argv", ["eval.py", option, value])
            with pytest.raises(SystemExit) as error:
                main()
            assert error.value.code == 2


def test_cli_query_limit_defaults_and_validation(monkeypatch):
    import sys

    import pytest

    seen = []

    async def capture(args, root):
        seen.append(args.query_limit)

    monkeypatch.setitem(main.__globals__, "evaluate", capture)
    for options in ([], ["--query-limit", "50"]):
        monkeypatch.setattr(sys, "argv", ["eval.py", *options])
        main()
    assert seen == [None, 50]
    for value in ("0", "-1", "none", "1.5"):
        monkeypatch.setattr(sys, "argv", ["eval.py", "--query-limit", value])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2


def test_filter_statistics_separates_retrieval_misses_from_removed_positives():
    stats = filter_statistics(["a", "b", "c"], ["b"], {"a", "b", "outside"})
    assert stats["documents_removed"] == 2
    assert stats["positives_in_candidates"] == 2
    assert stats["positives_retained"] == 1
    assert stats["positives_removed"] == 1
    assert stats["removed_positive_ids"] == ["a"]
    assert stats["removed_document_ids"] == ["a", "c"]
    assert filter_statistics(["a"], [], {"a"})["positives_removed"] == 1
    assert filter_statistics(["a"], ["a"], {"a"})["documents_removed"] == 0


def test_cli_task_threshold_defaults_and_overrides(monkeypatch):
    import sys

    seen = []

    async def capture(args, root):
        seen.append((args.task, args.threshold))

    monkeypatch.setitem(main.__globals__, "evaluate", capture)
    for options in (
        [],
        ["--task", "relevance"],
        ["--task", "relevance", "--threshold", "0.33"],
    ):
        monkeypatch.setattr(sys, "argv", ["eval.py", *options])
        main()
    assert seen == [("rerank", 0.0), ("relevance", 0.2), ("relevance", 0.33)]


if __name__ == "__main__":
    main()
