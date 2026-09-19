"""Loop ownership and a persistent bridge for the synchronous API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from threading import Lock, Thread
from typing import Any, TypeVar

import httpx

from .errors import ConfigurationError

T = TypeVar("T")


def require_sync_context() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise ConfigurationError(
        "Use await a_rank()/a_raw_rank() and aclose() inside an async event loop."
    )


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self._run, name="jev-reranker-asyncio", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.run_until_complete(self.loop.shutdown_default_executor())
            self.loop.close()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


class Runtime:
    """One instance owns one loop and one HTTP pool; no cross-loop resources.

    Native async calls bind to their first caller's loop. Sync calls instead
    start one loop thread, shared by all calling threads until close().
    """

    def __init__(
        self,
        concurrency: int,
        client: httpx.AsyncClient | None,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        self.concurrency = concurrency
        self.client = client
        self.transport = transport
        self.owns_client = client is None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runner: _LoopThread | None = None
        self.closed = False
        self.closing = False
        self._guard = Lock()
        self._sync_close_guard = Lock()
        self._semaphore: asyncio.Semaphore | None = None
        self._idle: asyncio.Event | None = None
        self._active = 0
        self._close_task: asyncio.Task[None] | None = None

    def bind(self, *, allow_closing: bool = False) -> None:
        current = asyncio.get_running_loop()
        with self._guard:
            if self.closed or (self.closing and not allow_closing):
                raise ConfigurationError("JevReranker is closed or closing.")
            if self.loop is not None and self.loop is not current:
                raise ConfigurationError(
                    "JevReranker belongs to a different event loop; use a separate instance."
                )
            self.loop = current
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self.concurrency)
                self._idle = asyncio.Event()
                self._idle.set()

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        self.bind()
        assert self._idle is not None
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._active -= 1
            if not self._active:
                self._idle.set()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        assert self._semaphore is not None
        return self._semaphore

    def get_client(self) -> httpx.AsyncClient:
        # Called only within an operation on the owning loop.
        if self.client is None:
            self.client = httpx.AsyncClient(
                transport=self.transport,
                limits=httpx.Limits(
                    max_connections=self.concurrency,
                    max_keepalive_connections=self.concurrency,
                ),
            )
        if self.client.is_closed:
            raise ConfigurationError("HTTP client is closed.")
        return self.client

    def run(self, factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        require_sync_context()
        with self._guard:
            if self.closed or self.closing:
                raise ConfigurationError("JevReranker is closed or closing.")
            if self.runner is None:
                if self.loop is not None:
                    raise ConfigurationError(
                        "This instance uses an async event loop; use its async methods."
                    )
                self.runner = _LoopThread()
                self.loop = self.runner.loop
            future = asyncio.run_coroutine_threadsafe(factory(), self.runner.loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    async def aclose(self) -> None:
        if self.closed:
            return
        self.bind(allow_closing=True)
        if self._close_task is None:
            self.closing = True
            self._close_task = asyncio.create_task(self._finish_close())
        # Cancelling a waiter must not interrupt HTTP pool cleanup.
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        assert self._idle is not None
        await self._idle.wait()
        try:
            if self.owns_client and self.client is not None:
                await self.client.aclose()
            elif self.owns_client and self.transport is not None:
                await self.transport.aclose()
        finally:
            self.closed = True

    def close(self) -> None:
        require_sync_context()
        with self._sync_close_guard:
            with self._guard:
                if self.closed:
                    return
                if self.runner is None:
                    if self.loop is not None:
                        raise ConfigurationError(
                            "Use await aclose() on the owning async event loop."
                        )
                    if self.transport is None:
                        self.closed = True
                        return
                    self.runner = _LoopThread()
                    self.loop = self.runner.loop
                self.closing = True
                runner = self.runner
                future = asyncio.run_coroutine_threadsafe(self.aclose(), runner.loop)
            try:
                future.result()
            finally:
                runner.stop()
