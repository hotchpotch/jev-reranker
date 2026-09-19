"""Per-call HTTP ownership with no persistent synchronous event loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from concurrent.futures import Future
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Any, TypeVar

import httpx

from .errors import ConfigurationError

T = TypeVar("T")


def require_sync_context() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise ConfigurationError("Use await a_rerank()/a_relevance_rerank() inside an async event loop.")


class _BorrowedTransport(httpx.AsyncBaseTransport):
    """Closing a per-call client must not close a caller-supplied transport."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self.transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.transport.handle_async_request(request)


class _Limiter:
    """A cancellation-safe limit shared across caller threads and event loops."""

    def __init__(self, count: int) -> None:
        self.available = count
        self.lock = Lock()
        self.waiters: list[Future[None]] = []

    async def __aenter__(self) -> None:
        waiter: Future[None] = Future()
        with self.lock:
            if self.available:
                self.available -= 1
                return
            self.waiters.append(waiter)
        try:
            await asyncio.wrap_future(waiter)
        except BaseException:
            with self.lock:
                if waiter in self.waiters:
                    self.waiters.remove(waiter)
                else:
                    self._release()
            raise

    def _release(self) -> None:
        if self.waiters:
            waiter = self.waiters.pop(0)
            # Cancellation is reconciled by the acquire handler under this lock.
            if waiter.set_running_or_notify_cancel():
                waiter.set_result(None)
        else:
            self.available += 1

    async def __aexit__(self, *exc: object) -> None:
        with self.lock:
            self._release()


@dataclass
class _Operation:
    client: httpx.AsyncClient | None = None


class Runtime:
    """Create and close owned HTTP clients within each ranking operation.

    Explicit clients and transports remain caller-owned. Only borrowed clients
    bind to a loop; ordinary instances can be reused across sync and async calls.
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
        self.loop: asyncio.AbstractEventLoop | None = None
        self.closed = False
        self.closing = False
        self._condition = Condition()
        self._active = 0
        self._close_complete: Future[None] = Future()
        self.semaphore = _Limiter(concurrency)
        self._operation: ContextVar[_Operation] = ContextVar("jev_operation")

    def bind(self) -> None:
        with self._condition:
            if self.closed or self.closing:
                raise ConfigurationError("JevReranker is closed or closing.")
            if self.client is not None:
                current = asyncio.get_running_loop()
                if self.loop is not None and self.loop is not current:
                    raise ConfigurationError("Borrowed HTTP client belongs to a different event loop.")
                self.loop = current

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        with self._condition:
            self.bind()
            self._active += 1
        state = _Operation()
        token = self._operation.set(state)
        try:
            yield
        finally:
            try:
                if state.client is not None and self.client is None:
                    cleanup = asyncio.create_task(state.client.aclose())
                    cancelled = False
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            cancelled = True
                    cleanup.result()
                    if cancelled:
                        raise asyncio.CancelledError
            finally:
                self._operation.reset(token)
                with self._condition:
                    self._active -= 1
                    if not self._active:
                        if self.closing:
                            self._mark_closed()
                        self._condition.notify_all()

    def get_client(self) -> httpx.AsyncClient:
        state = self._operation.get()
        if state.client is None:
            state.client = self.client or httpx.AsyncClient(
                transport=_BorrowedTransport(self.transport) if self.transport else None,
                limits=httpx.Limits(
                    max_connections=self.concurrency,
                    max_keepalive_connections=self.concurrency,
                ),
            )
        if state.client.is_closed:
            raise ConfigurationError("HTTP client is closed.")
        return state.client

    def run(self, factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        require_sync_context()
        if self.client is not None:
            raise ConfigurationError("Use async methods with a borrowed AsyncClient.")
        return asyncio.run(factory())

    def _mark_closed(self) -> None:
        # Called under the condition lock once all operations have drained.
        if not self.closed:
            self.closed = True
            self._close_complete.set_result(None)

    def _finish_close(self) -> None:
        with self._condition:
            self.closing = True
            self._condition.wait_for(lambda: self._active == 0)
            self._mark_closed()

    async def aclose(self) -> None:
        # Compatibility API: mark closed and drain calls, with no owned idle pool.
        with self._condition:
            self.closing = True
            if not self._active:
                self._mark_closed()
        # Share completion across loops without occupying an executor worker.
        # Cancelling one waiter must not cancel the shared shutdown signal.
        await asyncio.shield(asyncio.wrap_future(self._close_complete))

    def close(self) -> None:
        require_sync_context()
        self._finish_close()
