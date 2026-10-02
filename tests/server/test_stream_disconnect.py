"""Streaming cancellation must fire on a real ASGI http.disconnect.

starlette 1.6 Request.is_disconnected awaits receive() inside a PRE-cancelled
anyio scope, so the disconnect message is never observed - the per-chunk
is_disconnected() polling in the old stream_with_cancellation was dead code.
The fix is a watcher task that awaits request.receive() and aborts on the
first http.disconnect: it aborts from outside the generator frame (a driving
task parked in send() backpressure / flow.drain included), keeps working on
spec 2.4 servers that have no listen_for_disconnect, and logs the disconnect
explicitly. Direct client cuts already aborted on the current stacks
(uvicorn 0.52.4 dev / 0.53.0 prod over h11, scope spec 2.3 -> starlette
listen_for_disconnect); the watcher makes the behavior independent of those
starlette internals.
"""

import asyncio
import time

import pytest

from freetoken.server.api_server import FrontendManager


class _StubState:
    """Methods under test, bound onto a minimal harness."""

    stream_with_cancellation = FrontendManager.stream_with_cancellation

    def __init__(self):
        self.aborted = []

    async def _watch_disconnect(self, request, on_disconnect):
        # Lazy delegation: FrontendManager._watch_disconnect only exists post-fix,
        # so the pre-fix code fails on behavior, not on import.
        await FrontendManager._watch_disconnect(self, request, on_disconnect)

    async def abort_user(self, uid):
        self.aborted.append(uid)


class _FakeRequest:
    """receive() replays queued ASGI messages, then blocks like a live client."""

    def __init__(self, messages=()):
        self._messages = list(messages)

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        await asyncio.Event().wait()  # never set: connection stays alive
        return {"type": "http.disconnect"}

    async def is_disconnected(self):
        # starlette 1.6's polling in practice never observes the message.
        return False


class _DeferredDisconnectRequest:
    """receive() blocks until released, then delivers exactly one http.disconnect.

    is_disconnected mirrors starlette 1.6's broken polling (always False) so the
    test also exercises the pre-fix wrapper unchanged.
    """

    def __init__(self):
        self._release = asyncio.Event()
        self._delivered = False

    def release(self):
        self._release.set()

    async def receive(self):
        if not self._delivered:
            await self._release.wait()
            self._delivered = True
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()  # connection stays alive afterwards
        return {"type": "http.disconnect"}

    async def is_disconnected(self):
        return False


async def _slow_stream(chunks=3, delay=0.05):
    for i in range(chunks):
        await asyncio.sleep(delay)
        yield f"chunk-{i}".encode()


async def _stream_with_done_tail():
    """Three content chunks plus the terminal [DONE] frame."""
    for i in range(3):
        await asyncio.sleep(0.01)
        yield f"chunk-{i}".encode()
    await asyncio.sleep(0.01)
    yield b"data: [DONE]\n\n"


def _pending_tasks():
    return [
        t for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]


def test_disconnect_mid_stream_aborts_within_1s():
    state = _StubState()
    request = _FakeRequest(messages=[{"type": "http.disconnect"}])

    async def run():
        out = []
        t0 = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            async for chunk in state.stream_with_cancellation(_slow_stream(), request, uid=7):
                out.append(chunk)
                await asyncio.sleep(0)  # let the watcher task run
        return out, time.monotonic() - t0

    chunks, elapsed = asyncio.run(run())

    assert state.aborted == [7], f"orphan decode: abort never fired, aborted={state.aborted}"
    assert elapsed < 1.0, f"abort took {elapsed:.2f}s (>1s)"
    assert len(chunks) < 3, "stream kept flowing after disconnect"


def test_no_disconnect_streams_everything():
    state = _StubState()
    request = _FakeRequest()  # receive blocks: live client

    async def run():
        chunks = []
        async for chunk in state.stream_with_cancellation(_slow_stream(), request, 9):
            chunks.append(chunk)
        # no leaked watcher task (the wrapper's cleanup must have reaped it)
        await asyncio.sleep(0.01)
        return chunks, _pending_tasks()

    chunks, leaked = asyncio.run(run())

    assert chunks == [b"chunk-0", b"chunk-1", b"chunk-2"]
    assert state.aborted == []
    assert leaked == []


def test_disconnect_after_last_chunk_completes_stream():
    """Deferred delivery: the disconnect is only queued once the last content
    chunk is already out. The stream must finish untouched (every chunk plus
    the [DONE] tail) with at most one abort, and leave no pending tasks."""
    state = _StubState()
    request = _DeferredDisconnectRequest()

    async def run():
        chunks = []

        async def release_late():
            await asyncio.sleep(0.05)  # the 4x10ms stream is fully consumed by then
            request.release()

        kicker = asyncio.create_task(release_late())
        async for chunk in state.stream_with_cancellation(_stream_with_done_tail(), request, 13):
            chunks.append(chunk)
        await kicker
        await asyncio.sleep(0.05)  # a late watcher wakeup must not break anything
        return chunks, _pending_tasks()

    chunks, leaked = asyncio.run(run())

    assert chunks == [b"chunk-0", b"chunk-1", b"chunk-2", b"data: [DONE]\n\n"]
    assert len(state.aborted) <= 1
    assert leaked == []
