"""Async transport, lifecycle, cancellation, and sync bridge contracts."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from test_reranker import CharacterTokenizer, response

from jev_reranker import APIError, ConfigurationError, JevReranker


def make_async(handler=None, **kwargs):
    calls = []

    async def serve(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if handler:
            return await handler(payload, len(calls), request)
        return httpx.Response(200, json=response(payload))

    return JevReranker(
        api_key="async-test",
        dotenv_path=None,
        tokenizer=CharacterTokenizer(),
        transport=httpx.MockTransport(serve),
        **kwargs,
    ), calls


@pytest.mark.parametrize("mode", ["listwise", "pointwise", "pairwise"])
def test_async_api_and_aliases(mode):
    async def scenario():
        ranker, calls = make_async(mode=mode)
        async with ranker:
            raw = await ranker.a_raw_rank("q", ["a", "b"], detail=True)
            assert [r["document_index"] for r in raw["results"]] == [0, 1]
            assert raw["detail"]["usage"]["requests"] == (
                2 if mode == "pointwise" else 1
            )
            assert await ranker.a_rank("q", ["a"], top_k=1, return_documents=False) == [
                {"document_index": 0, "score": 0.5}
            ]
            assert await ranker.a_rerank("q", []) == []
            assert (await ranker.a_raw_rerank("q", [])) == {"results": []}
        with pytest.raises(ConfigurationError, match="closed"):
            await ranker.a_rank("q", ["a"])
        await ranker.aclose()
        assert calls

    asyncio.run(scenario())


def test_sync_bridge_uses_async_transport_and_reuses_loop():
    loops = []

    async def handler(p, n, r):
        loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return httpx.Response(200, json=response(p))

    ranker, _ = make_async(handler)
    with ranker:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: ranker.rank("q", ["a"]), range(6)))
        assert len(results) == 6
    assert len(set(loops)) == 1
    assert loops[0].is_closed()


def test_sync_api_inside_running_loop_has_clear_error():
    async def scenario():
        ranker, _ = make_async()
        with pytest.raises(ConfigurationError, match="a_rank|async"):
            ranker.rank("q", ["a"])
        async with ranker:
            assert await ranker.a_rank("q", ["a"])

    asyncio.run(scenario())


def test_cross_loop_reuse_rejected_before_network():
    ranker, calls = make_async()
    with asyncio.Runner() as owner:
        owner.run(ranker.a_rank("q", ["a"]))
        with asyncio.Runner() as other, pytest.raises(ConfigurationError, match="loop"):
            other.run(ranker.a_rank("q", ["a"]))
        owner.run(ranker.aclose())
    assert len(calls) == 1


def test_async_concurrency_bound_and_input_ties():
    async def scenario():
        active = maximum = 0

        async def handler(p, n, r):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.005)
            active -= 1
            return httpx.Response(200, json=response(p))

        ranker, _ = make_async(handler, mode="pointwise", max_concurrency=2)
        async with ranker:
            raws = await asyncio.gather(
                *(ranker.a_raw_rank("q", ["a"] * n, detail=True) for n in [3, 4, 5])
            )
        assert maximum == 2
        assert [r["detail"]["usage"]["requests"] for r in raws] == [3, 4, 5]
        assert all(
            [r["document_index"] for r in raw["results"]] == list(range(n))
            for raw, n in zip(raws, [3, 4, 5], strict=True)
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["listwise", "pointwise", "pairwise"])
def test_cancellation_cleans_children_and_releases_permits(mode):
    async def scenario():
        entered = asyncio.Event()
        block = True
        active = 0

        async def handler(p, n, r):
            nonlocal active
            active += 1
            entered.set()
            try:
                if block:
                    await asyncio.Event().wait()
                return httpx.Response(200, json=response(p))
            finally:
                active -= 1

        ranker, _ = make_async(handler, mode=mode, max_concurrency=2)
        async with ranker:
            task = asyncio.create_task(ranker.a_rank("q", ["a"] * 20))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert active == 0
            block = False
            assert len(await asyncio.wait_for(ranker.a_rank("q", ["a"] * 3), 2)) == 3

    asyncio.run(scenario())


def test_failure_cancels_siblings_without_exception_group():
    async def scenario():
        slow_started = asyncio.Event()
        slow_stopped = asyncio.Event()

        async def handler(p, n, r):
            if p["state"]["document"] == "bad":
                await slow_started.wait()
                return httpx.Response(401)
            slow_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                slow_stopped.set()

        ranker, _ = make_async(handler, mode="pointwise", max_concurrency=2)
        async with ranker:
            with pytest.raises(APIError) as exc:
                await asyncio.wait_for(
                    ranker.a_raw_rank("q", ["bad", "slow"], detail=True), 2
                )
            assert exc.value.status_code == 401
            assert exc.value.detail is not None
            assert slow_stopped.is_set()

    asyncio.run(scenario())


def test_retry_sleep_does_not_block_loop_and_can_cancel():
    async def scenario():
        sent = asyncio.Event()

        async def handler(p, n, r):
            sent.set()
            return httpx.Response(429, headers={"Retry-After": "60"})

        ranker, calls = make_async(handler)
        async with ranker:
            task = asyncio.create_task(ranker.a_rank("q", ["a"]))
            await asyncio.wait_for(sent.wait(), 2)
            await asyncio.sleep(0.01)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        assert len(calls) == 1

    asyncio.run(scenario())


def test_aclose_waits_for_active_calls_and_rejects_new_ones():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(p, n, r):
            entered.set()
            await release.wait()
            return httpx.Response(200, json=response(p))

        ranker, _ = make_async(handler)
        task = asyncio.create_task(ranker.a_rank("q", ["a"]))
        await asyncio.wait_for(entered.wait(), 2)
        closer = asyncio.create_task(ranker.aclose())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closer.done()
        with pytest.raises(ConfigurationError, match="clos"):
            await ranker.a_rank("q", ["a"])
        release.set()
        await task
        await asyncio.wait_for(closer, 2)

    asyncio.run(scenario())


def test_borrowed_async_client_not_closed():
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=response(json.loads(r.content)))
            )
        ) as client:
            async with JevReranker(
                api_key="test",
                dotenv_path=None,
                tokenizer=CharacterTokenizer(),
                client=client,
            ) as ranker:
                assert await ranker.a_rank("q", ["a"])
            assert not client.is_closed

    asyncio.run(scenario())


def test_async_preprocessing_does_not_block_loop():
    import threading

    async def scenario():
        entered, release = threading.Event(), threading.Event()

        class BlockingTokenizer(CharacterTokenizer):
            def encode(self, text):
                entered.set()
                assert release.wait(2)
                return super().encode(text)

        ranker = JevReranker(
            api_key="test",
            dotenv_path=None,
            tokenizer=BlockingTokenizer(),
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=response(json.loads(r.content)))
            ),
        )
        async with ranker:
            task = asyncio.create_task(ranker.a_rank("q", ["a"]))
            assert await asyncio.to_thread(entered.wait, 1)
            release.set()
            assert await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["listwise", "pointwise", "pairwise"])
def test_sync_async_ranking_parity(mode):
    async def handler(p, n, r):
        state = p["state"]
        if mode == "listwise":
            scores = {k: int(v) / 10 for k, v in state["documents"].items()}
        elif mode == "pointwise":
            scores = {"relevant": int(state["document"]) / 10}
        else:
            better = int(state["left"]) > int(state["right"])
            scores = {
                "left_wins": 0.9 if better else 0.1,
                "right_wins": 0.1 if better else 0.9,
            }
        return httpx.Response(200, json=response(p, scores))

    sync, _ = make_async(handler, mode=mode)
    with sync:
        expected = sync.rank("q", ["2", "0", "3", "1"])

    async def scenario():
        ranker, _ = make_async(handler, mode=mode)
        async with ranker:
            actual = await ranker.a_rank("q", ["2", "0", "3", "1"])
            assert actual == expected
            assert [r["document_index"] for r in actual] == [2, 0, 3, 1]

    asyncio.run(scenario())


def test_sync_and_async_instances_have_explicit_loop_ownership():
    ranker, _ = make_async()
    with ranker:
        ranker.rank("q", ["a"])

        async def other_loop():
            with pytest.raises(ConfigurationError, match="loop"):
                await ranker.a_rank("q", ["a"])

        asyncio.run(other_loop())


def test_cancelled_aclose_waiter_does_not_cancel_cleanup():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(p, n, r):
            entered.set()
            await release.wait()
            return httpx.Response(200, json=response(p))

        ranker, _ = make_async(handler)
        task = asyncio.create_task(ranker.a_rank("q", ["a"]))
        await asyncio.wait_for(entered.wait(), 2)
        closer = asyncio.create_task(ranker.aclose())
        await asyncio.sleep(0)
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer
        release.set()
        await task
        await asyncio.wait_for(ranker.aclose(), 2)
        assert ranker._runtime.client.is_closed

    asyncio.run(scenario())


def test_async_empty_input_and_zero_top_k_do_not_load_tokenizer_or_call_api():
    async def scenario():
        ranker, calls = make_async()
        ranker._tokenizer = None
        async with ranker:
            assert await ranker.a_rank("q", []) == []
            assert await ranker.a_raw_rank("q", ["a"], top_k=0) == {"results": []}
            assert ranker._tokenizer is None
            assert ranker._runtime.client is None
        assert not calls

    asyncio.run(scenario())


def test_sync_close_waits_for_inflight_work():
    import threading

    entered, release = threading.Event(), threading.Event()
    loops = []

    async def handler(p, n, r):
        loops.append(asyncio.get_running_loop())
        entered.set()
        await asyncio.to_thread(release.wait)
        return httpx.Response(200, json=response(p))

    ranker, _ = make_async(handler)
    with ThreadPoolExecutor(max_workers=2) as pool:
        call = pool.submit(ranker.rank, "q", ["a"])
        assert entered.wait(2)
        closing = pool.submit(ranker.close)
        try:
            assert not closing.done()
        finally:
            release.set()
        assert call.result(timeout=2)
        closing.result(timeout=2)
    assert loops[0].is_closed()


@pytest.mark.parametrize("interface", ["sync", "async"])
def test_owned_transport_closed_even_without_requests(interface):
    class Transport(httpx.AsyncBaseTransport):
        closed = False

        async def handle_async_request(self, request):
            raise AssertionError("No HTTP expected")

        async def aclose(self):
            self.closed = True

    transport = Transport()
    ranker = JevReranker(api_key="test", dotenv_path=None, transport=transport)
    if interface == "sync":
        ranker.close()
    else:
        asyncio.run(ranker.aclose())
    assert transport.closed
