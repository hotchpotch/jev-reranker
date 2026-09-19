"""Per-call prompts and inclusive relevance thresholds share the ranking pipeline."""

import asyncio
import copy

import httpx
import pytest
from test_reranker import make, response

from jev_reranker import ConfigurationError


def scored():
    return make(
        lambda p, *_: httpx.Response(
            200,
            json=response(
                p,
                {
                    "doc_0": 0.0,
                    "doc_1": 0.19,
                    "doc_2": 0.2,
                    "doc_3": 0.9,
                },
            ),
        )
    )


@pytest.mark.parametrize(
    "method_name",
    ["rerank", "relevance_rerank", "a_rerank", "a_relevance_rerank"],
)
@pytest.mark.parametrize("detail", [False, True])
@pytest.mark.parametrize("case", ["retained", "filtered", "empty", "zero_top_k"])
def test_ranking_response_envelope(method_name, detail, case):
    ranker, calls = scored()
    docs = [] if case == "empty" else ["zero", "noise", "partial", "answer"]
    method = getattr(ranker, method_name)
    kwargs = {
        "threshold": 1.0 if case == "filtered" else 0.2,
        "top_k": 0 if case == "zero_top_k" else None,
        "detail": detail,
        "return_documents": False,
    }
    if method_name.startswith("a_"):
        result = asyncio.run(method("q", docs, **kwargs))
    else:
        result = method("q", docs, **kwargs)
    assert set(result) == ({"results", "detail"} if detail else {"results"})
    if case == "retained":
        assert [r["document_index"] for r in result["results"]] == [3, 2]
        for row in result["results"]:
            assert "text" not in row
            assert ("detail" in row) is detail
    else:
        assert result["results"] == []
    scored_count = 4 if case in ("retained", "filtered") else 0
    assert len(calls) == (1 if scored_count else 0)
    if detail:
        logs = result["detail"]
        assert len(logs["documents"]) == scored_count
        assert logs["selection"] == {
            "top_k": kwargs["top_k"],
            "scored_count": scored_count,
            "above_threshold_count": 2 if case == "retained" else 0,
            "returned_count": len(result["results"]),
        }
        if scored_count:
            assert [d["document_index"] for d in logs["documents"]] == [0, 1, 2, 3]
            assert [d["score"] for d in logs["documents"]] == [0.0, 0.19, 0.2, 0.9]
        if case == "filtered":
            assert all(not d["passes_threshold"] for d in logs["documents"])


def test_rank_default_zero_and_relevance_default_point_two():
    ranker, calls = scored()
    docs = ["zero", "noise", "partial", "answer"]
    assert [r["document_index"] for r in ranker.rerank("q", docs)["results"]] == [3, 2, 1, 0]
    result = ranker.relevance_rerank("q", docs, detail=True)
    assert [r["document_index"] for r in result["results"]] == [3, 2]
    assert result["results"][1]["score"] == 0.2
    assert (
        calls[0]["questions"]["doc_0"]["instructions"]
        != calls[1]["questions"]["doc_0"]["instructions"]
    )
    assert len(calls[1]["questions"]) == 4


def test_rerank_logs_discarded_scores_and_applies_threshold_before_top_k():
    ranker, _ = scored()
    raw = ranker.rerank(
        "q", ["a", "b", "c", "d"], threshold=0.2, top_k=1, detail=True
    )
    assert [r["document_index"] for r in raw["results"]] == [3]
    assert raw["detail"]["configuration"]["threshold"] == 0.2
    assert [d["score"] for d in raw["detail"]["documents"]] == [0.0, 0.19, 0.2, 0.9]
    assert [d["passes_threshold"] for d in raw["detail"]["documents"]] == [
        False,
        False,
        True,
        True,
    ]
    assert raw["detail"]["selection"] == {
        "top_k": 1,
        "scored_count": 4,
        "above_threshold_count": 2,
        "returned_count": 1,
    }


def test_custom_instruction_and_constructor_compatibility():
    prompt = {
        "instructions": "Custom check {document} for `query`.",
        "criteria": {"true": "yes", "false": "no"},
    }
    snapshot = copy.deepcopy(prompt)
    ranker, calls = make(instruction=prompt)
    prompt["criteria"]["true"] = "mutated"
    raw = ranker.rerank("q", ["a"], detail=True)
    assert raw["detail"]["configuration"]["criteria"] == snapshot["criteria"]
    assert calls[0]["questions"]["doc_0"]["instructions"].startswith("Custom check")
    other = {
        "instructions": "Other {document}.",
        "criteria": {"true": "ok", "false": "bad"},
    }
    ranker.rerank("q", ["a"], instruction=other)
    ranker.rerank("q", ["a"])
    assert calls[1]["questions"]["doc_0"]["instructions"].startswith("Other")
    assert calls[2] == calls[0]
    assert other["instructions"] == "Other {document}."


@pytest.mark.parametrize(
    "threshold", [-0.1, 1.1, True, None, "0.2", float("nan"), float("inf")]
)
def test_bad_threshold_fails_before_http(threshold):
    ranker, calls = make()
    with pytest.raises(ConfigurationError, match="threshold"):
        ranker.rerank("q", ["a"], threshold=threshold)
    assert calls == []


@pytest.mark.parametrize(
    "prompt",
    [
        {},
        {"instructions": "no placeholder", "criteria": {"true": "yes", "false": "no"}},
        {"instructions": "{document}", "criteria": {"true": "yes"}},
        {
            "instructions": "{document}",
            "criteria": {"true": "yes", "false": "no"},
            "unknown": True,
        },
    ],
)
def test_bad_instruction_fails_before_http(prompt):
    ranker, calls = make()
    with pytest.raises(ConfigurationError):
        ranker.rerank("q", ["a"], instruction=prompt)
    assert not calls


def test_relevance_rerank_delegates_with_preset(monkeypatch):
    ranker, _ = make()
    seen = []
    monkeypatch.setattr(
        ranker, "rerank", lambda *a, **kw: seen.append(kw) or {"results": ["ok"]}
    )
    assert ranker.relevance_rerank("q", ["a"]) == {"results": ["ok"]}
    assert seen[0]["threshold"] == 0.2
    assert "evidence" in seen[0]["instruction"]["instructions"]


def test_async_prompt_isolation():
    async def handler(p, *_):
        await asyncio.sleep(0.001)
        return httpx.Response(200, json=response(p))

    async def run():
        ranker, calls = make(handler)
        async with ranker:
            normal, relevant = await asyncio.gather(
                ranker.a_rerank("q", ["a"], detail=True),
                ranker.a_relevance_rerank("q", ["a"], detail=True),
            )
            assert normal["detail"]["configuration"]["threshold"] == 0.0
            assert len(relevant["results"]) == 1
        prompts = [p["questions"]["doc_0"]["instructions"] for p in calls]
        assert len(set(prompts)) == 2

    asyncio.run(run())


def test_relevance_rejects_pairwise_and_threshold_can_return_empty():
    ranker, calls = make(mode="pairwise")
    with pytest.raises(ConfigurationError):
        ranker.relevance_rerank("q", ["a", "b"])
    assert not calls
    ranker, _ = scored()
    assert ranker.relevance_rerank("q", ["a", "b"], threshold=1.0) == {"results": []}
    assert len(ranker.relevance_rerank("q", ["a", "b"], threshold=0.0)["results"]) == 2


def test_threshold_one_is_inclusive():
    ranker, _ = make(
        lambda p, *_: httpx.Response(
            200, json=response(p, {"doc_0": 1.0, "doc_1": 0.99})
        )
    )
    assert ranker.rerank("q", ["a", "b"], threshold=1.0)["results"] == [
        {"document_index": 0, "score": 1.0, "text": "a"}
    ]


def test_constructor_prompt_conflicts_and_presets_are_not_mutated():
    from jev_reranker import RELEVANCE_INSTRUCTION

    before = copy.deepcopy(RELEVANCE_INSTRUCTION)
    with pytest.raises(ConfigurationError):
        make(instruction=RELEVANCE_INSTRUCTION, instructions="{document}")
    ranker, _ = make()
    raw = ranker.rerank(
        "q", ["a"], instruction=RELEVANCE_INSTRUCTION, threshold=0.2, detail=True
    )
    raw["detail"]["configuration"]["criteria"]["true"] = "changed"
    assert RELEVANCE_INSTRUCTION == before


def test_prompt_snapshot_survives_caller_mutation_during_request():
    from jev_reranker import RELEVANCE_INSTRUCTION

    async def run():
        started, finish = asyncio.Event(), asyncio.Event()
        prompt = copy.deepcopy(RELEVANCE_INSTRUCTION)

        async def handle(p, *_):
            started.set()
            await finish.wait()
            return httpx.Response(200, json=response(p))

        ranker, calls = make(handle)
        async with ranker:
            task = asyncio.create_task(
                ranker.a_rerank("q", ["a"], instruction=prompt, detail=True)
            )
            await started.wait()
            prompt["instructions"] = "mutated {document}"
            prompt["criteria"]["true"] = "mutated"
            finish.set()
            raw = await task
        assert (
            raw["detail"]["configuration"]["instructions"]
            == RELEVANCE_INSTRUCTION["instructions"]
        )
        assert (
            calls[0]["questions"]["doc_0"]["criteria"]
            == RELEVANCE_INSTRUCTION["criteria"]
        )

    asyncio.run(run())


@pytest.mark.parametrize("interface", ["sync", "async"])
@pytest.mark.parametrize("mode", ["listwise", "pointwise"])
def test_relevance_selects_mode_specific_prompt_and_preserves_overrides(interface, mode):
    from jev_reranker import POINTWISE_RELEVANCE_INSTRUCTION, RELEVANCE_INSTRUCTION

    expected = (
        POINTWISE_RELEVANCE_INSTRUCTION if mode == "pointwise" else RELEVANCE_INSTRUCTION
    )
    custom = {
        "instructions": "Custom evidence in {document} for `query`.",
        "criteria": {"true": "Useful", "false": "Not useful"},
    }
    ranker, calls = make(mode=mode, instruction=custom)

    async def run_async():
        await ranker.a_relevance_rerank("q", ["a"])
        await ranker.a_relevance_rerank("q", ["a"], instruction=custom)
        await ranker.a_rerank("q", ["a"])

    if interface == "async":
        asyncio.run(run_async())
    else:
        ranker.relevance_rerank("q", ["a"])
        ranker.relevance_rerank("q", ["a"], instruction=custom)
        ranker.rerank("q", ["a"])
    questions = [next(iter(call["questions"].values())) for call in calls]
    assert questions[0]["criteria"] == expected["criteria"]
    assert questions[1]["criteria"] == questions[2]["criteria"] == custom["criteria"]
    assert questions[0]["instructions"] == expected["instructions"].format(
        document="`document`" if mode == "pointwise" else "`documents.doc_0`"
    )


def test_pointwise_relevance_prompt_only_uses_single_document_context():
    from jev_reranker import POINTWISE_RELEVANCE_INSTRUCTION
    from jev_reranker.instructions import validate_instruction

    prompt = validate_instruction(POINTWISE_RELEVANCE_INSTRUCTION, "pointwise")
    assert "{document}" in prompt["instructions"]
    for absent in ("{left}", "{right}", "other passage", "Use the pair"):
        assert absent not in prompt["instructions"]
    assert "different referent" in prompt["criteria"]["false"]
