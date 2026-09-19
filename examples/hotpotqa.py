"""NanoBEIR-en HotPotQA: 50 query × 全正解を含む10 docs の評価。

実行方法（jev-reranker のリポジトリ root から）:
    # .env の TYPESAFE_API_KEY を設定して実行。実 API の利用料金が発生します。
    uv run --locked --group examples --extra tokenizer python examples/hotpotqa.py

    # モデル・採点方式・文書上限を変更する例
    uv run --locked --group examples --extra tokenizer python examples/hotpotqa.py \
        --model jev-1.13.0 --mode pointwise --document-max-length 8000

    # tokenizer を使わず文字数で計測する場合
    uv run --locked --group examples python examples/hotpotqa.py --tokenizer none

    # ファイル内の offline テスト（データ取得・実 API 通信なし）
    uv run --locked pytest examples/hotpotqa.py -q

既定設定:
    listwise、Gemma tokenizer、文書上限4000 token、同時実行数4、seed 42。
    モデルは JEV_MODEL（環境変数/.env）、未設定なら jev-latest。
    認証は環境変数を .env より優先。この checkout の src/ を import します。
    pyarrow は examples group、tokenizer 関連依存は tokenizer extra で導入します。

候補選択:
    固定 revision の NanoBEIR-en / NanoHotpotQA を使用します。
    hybrid の保存順位は BM25 と dense の RRF 順（数値スコアは未収録）。
    qrels の全正解を確保し、残りを hybrid 上位の非正解で埋めて計10件にします。
    hybrid 外の正解は corpus から補完。正解が10件を超える場合はエラーです。
    query ID 順に1つの random.Random(42) でシャッフル。
    不正データはエラーとし、query を黙って除外しません。

評価・出力:
    document_index を文書IDへ戻し、二値 nDCG@10 の50 query 平均を計測します。
    主指標 ndcg_at_10 は全 qrels、candidate_ndcg_at_10 は選択10件内の正解で
    IDCG を計算します。全正解を含むため両指標は一致し、oracle は1.0です。
    hybrid 順・シャッフル順も全 qrels 基準です。
    正解保証の候補選択なので、元の100件 rerank の評価とは直接比較しません。
    データは .cache/hotpotqa/、選択ID・データハッシュ・採点詳細・集計は
    .live-results/hotpotqa-<UTC>/ に保存（Git 対象外）。ログには本文を含みます。
    --output で保存先を変更できますが、既存ディレクトリは上書きしません。
    --revision には再現性のため dataset の commit SHA を指定してください。

詳細と実測結果は examples/README.md を参照してください。
"""

from __future__ import annotations

import math
import random


def select_documents(
    ranking: list[str], relevant: set[str], count: int = 10
) -> list[str]:
    if count < 1 or len(ranking) < count or len(set(ranking)) != len(ranking):
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
    assert select_documents(ranking, {"3", "8"}) == ranking[:10]
    assert select_documents(ranking, {"3", "20"}) == ranking[:9] + ["20"]


def test_selection_promotes_highest_positive_including_rank_101():
    ranking = [str(i) for i in range(101)]
    assert select_documents(ranking, {"100"}) == ranking[:9] + ["100"]
    assert select_documents(ranking, {"40", "30"}) == ranking[:8] + ["30", "40"]
    assert select_documents(ranking, {"100", "outside"}) == ranking[:8] + [
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
            select_documents(ranking, positives)


def test_ndcg_uses_all_qrels_and_log_discount():
    import pytest

    assert ndcg(["a", "b", "c"], {"a", "b"}) == 1.0
    assert ndcg(["a", "x"], {"a", "b"}) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ndcg(["x", "a"], {"a"}) == pytest.approx(1 / math.log2(3))
    assert ndcg(["x"] * 10 + ["a"], {"a"}) == 0


def test_shuffle_is_reproducible_and_preserves_candidates():
    rankings = {"q": [str(i) for i in range(101)]}
    a = prepare_jobs({"q": "query"}, rankings, {"q": {"100"}})
    b = prepare_jobs({"q": "query"}, rankings, {"q": {"100"}})
    assert a == b
    _, selected, shuffled = a[0]
    assert shuffled == ["7", "3", "2", "8", "5", "6", "100", "4", "0", "1"]
    assert set(selected) == set(shuffled)


DATASET = "hakari-bench/NanoBEIR-en"
REVISION = "d3962aa8efe48ed79044c5e155b848982667b4ba"
SPLIT = "NanoHotpotQA"


def prepare_jobs(queries, hybrid, qrels, seed=42):
    rng = random.Random(seed)
    jobs = []
    for qid in sorted(queries):
        selected = select_documents(hybrid[qid], qrels[qid])
        shuffled = selected.copy()
        rng.shuffle(shuffled)
        jobs.append((qid, selected, shuffled))
    return jobs


def load_data(cache, revision):
    """Read immutable parquet files; pyarrow is an examples-only dependency."""
    import hashlib
    import importlib

    import httpx

    parquet = importlib.import_module("pyarrow.parquet")
    tables, hashes = {}, {}
    for config in ("queries", "corpus", "qrels", "reranking_hybrid"):
        path = cache / revision / f"{config}.parquet"
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{config}/{SPLIT}-00000-of-00001.parquet"
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
    if len(queries) != 50 or set(queries) != set(hybrid):
        raise ValueError("Expected exactly 50 matching query and hybrid rows")
    for qid in queries:
        if len(hybrid[qid]) not in (100, 101):
            raise ValueError(f"Expected 100/101 candidates for {qid}")
        select_documents(hybrid[qid], qrels[qid])
        if not (set(hybrid[qid]) | qrels[qid]).issubset(corpus):
            raise ValueError(f"Missing corpus documents for {qid}")
    return queries, corpus, qrels, hybrid, hashes


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
    queries, corpus, qrels, hybrid, hashes = load_data(args.cache, args.revision)
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = prepare_jobs(queries, hybrid, qrels, args.seed)
    manifest = {
        "dataset": DATASET,
        "revision": args.revision,
        "split": SPLIT,
        "file_sha256": hashes,
        "seed": args.seed,
        "query_count": 50,
        "documents_per_query": 10,
        "library_path": jev_reranker.__file__,
        "started_at": datetime.now(UTC).isoformat(),
        "selection": "all qrels positives plus highest hybrid negatives to total 10; missing positives from corpus",
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
    async with JevReranker(
        dotenv_path=root / ".env",
        mode=args.mode,
        model=args.model,
        tokenizer=None if args.tokenizer == "none" else args.tokenizer,
        document_max_length=args.document_max_length,
        max_concurrency=args.concurrency,
    ) as reranker:

        async def run(job):
            qid, selected, shuffled = job
            async with semaphore:
                raw = await reranker.a_raw_rank(
                    queries[qid],
                    [corpus[d] for d in shuffled],
                    detail=True,
                )
            indices = [r["document_index"] for r in raw["results"]]
            if sorted(indices) != list(range(10)):
                raise ValueError(f"Incomplete or duplicate results for {qid}")
            ranking = [shuffled[i] for i in indices]
            selected_relevant = qrels[qid].intersection(selected)
            if selected_relevant != qrels[qid]:
                raise ValueError(f"Selected documents omit positives for {qid}")
            row = {
                "query_id": qid,
                "ranking": ranking,
                "ndcg_at_10": ndcg(ranking, qrels[qid]),
                "candidate_ndcg_at_10": ndcg(ranking, selected_relevant),
                "hybrid_ndcg_at_10": ndcg(selected, qrels[qid]),
                "shuffled_ndcg_at_10": ndcg(shuffled, qrels[qid]),
                "oracle_ndcg_at_10": ndcg(
                    sorted(selected, key=lambda d: d not in qrels[qid]), qrels[qid]
                ),
                "positive_promoted": selected != hybrid[qid][:10],
                "positives_outside_hybrid": sorted(qrels[qid].difference(hybrid[qid])),
                "raw": raw,
            }
            (args.output / f"{qid}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2)
            )
            rows.append(row)
            print(f"{len(rows):2}/50 {qid} nDCG@10={row['ndcg_at_10']:.6f}", flush=True)

        tasks = [asyncio.create_task(run(job)) for job in jobs]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    summary = {
        "query_count": len(rows),
        "documents_per_query": 10,
        "elapsed_seconds": time.perf_counter() - start,
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "positive_promoted_queries": sum(row["positive_promoted"] for row in rows),
        **{
            key: mean(row[key] for row in rows)
            for key in (
                "ndcg_at_10",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument(
        "--mode", choices=("listwise", "pointwise", "pairwise"), default="listwise"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Default: JEV_MODEL from env/.env, then jev-latest",
    )
    parser.add_argument(
        "--tokenizer",
        default="google/embeddinggemma-300m",
        help="Hub name or 'none' for len",
    )
    parser.add_argument("--document-max-length", type=int, default=4000)
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

    queries = {str(i): "query" for i in range(50)}
    corpus = {str(i): f"document {i}" for i in range(100)}
    qrels = {q: {"99"} for q in queries}
    hybrid = {q: list(corpus) for q in queries}
    monkeypatch.setitem(
        evaluate.__globals__,
        "load_data",
        lambda *_: (queries, corpus, qrels, hybrid, {}),
    )
    original = jev_reranker.JevReranker

    def respond(request):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "offline",
                "answers": {
                    key: {"type": "noul", "noul": float(text == "document 99")}
                    for key, text in body["state"]["documents"].items()
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
        cache=tmp_path,
        revision=REVISION,
        output=tmp_path / "run",
        seed=42,
        concurrency=4,
        mode="listwise",
        model=None,
        tokenizer="none",
        document_max_length=4000,
    )
    asyncio.run(evaluate(args, root))
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["query_count"] == 50
    assert summary["ndcg_at_10"] == 1
    assert summary["positive_promoted_queries"] == 50
    assert len(list(args.output.glob("*.json"))) == 52
    assert json.loads((args.output / "0.json").read_text())["ranking"][0] == "99"


if __name__ == "__main__":
    main()
