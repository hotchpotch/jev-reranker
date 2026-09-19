"""Async transport, lifecycle, cancellation, and sync bridge contracts."""

import asyncio
import json
import threading
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
def test_async_api(mode):
    async def scenario():
        ranker, calls = make_async(mode=mode)
        async with ranker:
            raw = await ranker.a_rerank("q", ["a", "b"], detail=True)
            assert [r["document_index"] for r in raw["results"]] == [0, 1]
            assert raw["detail"]["usage"]["requests"] == (
                2 if mode == "pointwise" else 1
            )
            assert (await ranker.a_rerank("q", ["a"], top_k=1, return_documents=False))["results"] == [
                {"document_index": 0, "score": 0.5}
            ]
            assert await ranker.a_rerank("q", []) == {"results": []}
        with pytest.raises(ConfigurationError, match="closed"):
            await ranker.a_rerank("q", ["a"])
        await ranker.aclose()
        assert calls

    asyncio.run(scenario())


def test_sync_calls_use_temporary_loops():
    loops = []

    async def handler(p, n, r):
        loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return httpx.Response(200, json=response(p))

    ranker, _ = make_async(handler)
    with ranker:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: ranker.rerank("q", ["a"]), range(6)))
        assert len(results) == 6
    assert len(set(loops)) == 6
    assert all(loop.is_closed() for loop in loops)


def test_sync_api_inside_running_loop_has_clear_error():
    async def scenario():
        ranker, _ = make_async()
        with pytest.raises(ConfigurationError, match="a_rerank|async"):
            ranker.rerank("q", ["a"])
        async with ranker:
            assert await ranker.a_rerank("q", ["a"])

    asyncio.run(scenario())


def test_borrowed_client_cross_loop_reuse_rejected_before_network():
    async def handler(request):
        return httpx.Response(200, json=response(json.loads(request.content)))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ranker = JevReranker(api_key="test", dotenv_path=None, client=client)
    with asyncio.Runner() as owner:
        owner.run(ranker.a_rerank("q", ["a"]))
        with asyncio.Runner() as other, pytest.raises(ConfigurationError, match="loop"):
            other.run(ranker.a_rerank("q", ["a"]))
        owner.run(client.aclose())


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
                *(ranker.a_rerank("q", ["a"] * n, detail=True) for n in [3, 4, 5])
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
            task = asyncio.create_task(ranker.a_rerank("q", ["a"] * 20))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert active == 0
            block = False
            assert len((await asyncio.wait_for(ranker.a_rerank("q", ["a"] * 3), 2))["results"]) == 3

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
                    ranker.a_rerank("q", ["bad", "slow"], detail=True), 2
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
            task = asyncio.create_task(ranker.a_rerank("q", ["a"]))
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
        task = asyncio.create_task(ranker.a_rerank("q", ["a"]))
        await asyncio.wait_for(entered.wait(), 2)
        closer = asyncio.create_task(ranker.aclose())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closer.done()
        with pytest.raises(ConfigurationError, match="clos"):
            await ranker.a_rerank("q", ["a"])
        release.set()
        await task
        await asyncio.wait_for(closer, 2)

    asyncio.run(scenario())


@pytest.mark.parametrize("workers, waiters", [(1, 1), (2, 4)])
def test_aclose_during_preparation_does_not_exhaust_executor(monkeypatch, workers, waiters):
    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=workers))
        entered = asyncio.Event()
        release = threading.Event()
        ranker, calls = make_async()
        prepare = ranker._prepare

        def blocked_prepare(docs, run):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            return prepare(docs, run)

        monkeypatch.setattr(ranker, "_prepare", blocked_prepare)
        task = asyncio.create_task(ranker.a_rerank("q", ["a", "b"]))
        closers = []
        try:
            await asyncio.wait_for(entered.wait(), 2)
            closers = [asyncio.create_task(ranker.aclose()) for _ in range(waiters)]
            await asyncio.sleep(0)
            assert ranker._runtime.closing
            assert all(not closer.done() for closer in closers)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(task, *closers), 2)
            assert len(results[0]["results"]) == 2
            assert calls
            assert ranker._runtime.closed
        finally:
            release.set()
            for pending in [task, *closers]:
                pending.cancel()
            await asyncio.gather(task, *closers, return_exceptions=True)

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
                assert await ranker.a_rerank("q", ["a"])
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
            task = asyncio.create_task(ranker.a_rerank("q", ["a"]))
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
        expected = sync.rerank("q", ["2", "0", "3", "1"])

    async def scenario():
        ranker, _ = make_async(handler, mode=mode)
        async with ranker:
            actual = await ranker.a_rerank("q", ["2", "0", "3", "1"])
            assert actual == expected
            assert [r["document_index"] for r in actual["results"]] == [2, 0, 3, 1]

    asyncio.run(scenario())


def test_cancelled_aclose_waiter_does_not_cancel_cleanup():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(p, n, r):
            entered.set()
            await release.wait()
            return httpx.Response(200, json=response(p))

        ranker, _ = make_async(handler)
        task = asyncio.create_task(ranker.a_rerank("q", ["a"]))
        await asyncio.wait_for(entered.wait(), 2)
        closer = asyncio.create_task(ranker.aclose())
        await asyncio.sleep(0)
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer
        release.set()
        await task
        assert ranker._runtime.closed
        await asyncio.wait_for(ranker.aclose(), 2)
        assert ranker._runtime.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("include_sync_close", [False, True])
def test_shutdown_across_loops_after_cancelled_waiter(include_sync_close):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(p, n, r):
            entered.set()
            await release.wait()
            return httpx.Response(200, json=response(p))

        ranker, _ = make_async(handler)

        async def close_on_another_loop(ready, cancel=False):
            closer = asyncio.create_task(ranker.aclose())
            await asyncio.sleep(0)
            assert not closer.done()
            ready.set()
            if cancel:
                closer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await closer
            else:
                await closer

        def close_synchronously(ready):
            ready.set()
            ranker.close()

        call = asyncio.create_task(ranker.a_rerank("q", ["a"]))
        with ThreadPoolExecutor(max_workers=3) as pool:
            try:
                await asyncio.wait_for(entered.wait(), 2)
                # asyncio.run closes this waiter's loop before shutdown completes.
                cancelled = pool.submit(
                    asyncio.run, close_on_another_loop(threading.Event(), cancel=True)
                )
                await asyncio.wait_for(asyncio.wrap_future(cancelled), 2)
                assert ranker._runtime.closing
                assert not ranker._runtime.closed

                ready = [threading.Event(), threading.Event()]
                closers = [
                    pool.submit(asyncio.run, close_on_another_loop(event))
                    for event in ready
                ]
                if include_sync_close:
                    ready.append(threading.Event())
                    closers.append(pool.submit(close_synchronously, ready[-1]))
                for event in ready:
                    assert await asyncio.to_thread(event.wait, 2)
                assert all(not closer.done() for closer in closers)

                release.set()
                result = await asyncio.wait_for(call, 2)
                assert result["results"][0]["document_index"] == 0
                for closer in closers:
                    await asyncio.wait_for(asyncio.wrap_future(closer), 2)
                assert ranker._runtime.closed
            finally:
                release.set()
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)

    asyncio.run(scenario())


def test_async_empty_input_and_zero_top_k_do_not_load_tokenizer_or_call_api():
    async def scenario():
        ranker, calls = make_async()
        ranker._tokenizer = None
        async with ranker:
            assert await ranker.a_rerank("q", []) == {"results": []}
            assert await ranker.a_rerank("q", ["a"], top_k=0) == {"results": []}
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
        call = pool.submit(ranker.rerank, "q", ["a"])
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
def test_borrowed_transport_not_closed_by_ranker(interface):
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
    assert not transport.closed


def test_each_call_closes_its_client_without_explicit_close(monkeypatch):
    created = []
    original = httpx.AsyncClient

    class TrackingClient(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", TrackingClient)
    ranker, calls = make_async(mode="pointwise")
    for _ in range(2):
        assert len(ranker.rerank("q", ["a", "b"])["results"]) == 2
        assert all(client.is_closed for client in created)
    assert len(created) == 2
    assert len(calls) == 4


def test_default_instance_can_move_between_sync_and_async_loops():
    ranker, calls = make_async()
    assert ranker.rerank("q", ["a"])
    assert asyncio.run(ranker.a_rerank("q", ["a"]))
    assert asyncio.run(ranker.a_rerank("q", ["a"]))
    assert ranker.rerank("q", ["a"])
    assert len(calls) == 4


@pytest.mark.parametrize("cancel", [False, True])
def test_per_call_client_closed_on_failure_or_cancellation(monkeypatch, cancel):
    clients = []
    original = httpx.AsyncClient

    class TrackingClient(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", TrackingClient)

    async def scenario():
        entered = asyncio.Event()

        async def handler(p, n, r):
            entered.set()
            if cancel:
                await asyncio.Event().wait()
            return httpx.Response(401)

        ranker, _ = make_async(handler)
        task = asyncio.create_task(ranker.a_rerank("q", ["a"]))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else APIError):
            await task
        assert clients and all(client.is_closed for client in clients)

    asyncio.run(scenario())


def test_cancelled_waiters_do_not_leak_shared_capacity():
    from jev_reranker._runtime import _Limiter

    async def scenario():
        limiter = _Limiter(1)
        await limiter.__aenter__()
        tasks = [asyncio.create_task(limiter.__aenter__()) for _ in range(3)]
        await asyncio.sleep(0)
        tasks[0].cancel()
        await limiter.__aexit__()
        tasks[1].cancel()
        for task in tasks[:2]:
            with pytest.raises(asyncio.CancelledError):
                await task
        await asyncio.wait_for(tasks[2], 1)
        await limiter.__aexit__()
        assert limiter.available == 1
        assert not limiter.waiters

    asyncio.run(scenario())


def test_concurrent_sync_calls_share_instance_http_limit():
    import threading
    import time

    guard = threading.Lock()
    active = maximum = 0

    async def handler(p, n, r):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        await asyncio.to_thread(time.sleep, 0.01)
        with guard:
            active -= 1
        return httpx.Response(200, json=response(p))

    ranker, _ = make_async(handler, mode="pointwise", max_concurrency=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(lambda _: ranker.rerank("q", ["a", "b"]), range(8)))
    assert maximum == 2
    assert active == 0


def test_cancelling_again_during_client_cleanup_still_closes(monkeypatch):
    original = httpx.AsyncClient

    async def scenario():
        closing, release = asyncio.Event(), asyncio.Event()
        clients = []

        class SlowClose(original):
            async def aclose(self):
                clients.append(self)
                closing.set()
                await release.wait()
                await super().aclose()

        monkeypatch.setattr(httpx, "AsyncClient", SlowClose)
        ranker, _ = make_async()
        task = asyncio.create_task(ranker.a_rerank("q", ["a"]))
        await closing.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert clients and clients[0].is_closed
        assert ranker._runtime._active == 0

    asyncio.run(scenario())
