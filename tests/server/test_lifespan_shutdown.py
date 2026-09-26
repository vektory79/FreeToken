"""Graceful-stop tests for the lifespan shutdown path: worker reaping (SIGINT relay ->
join -> terminate -> kill) and ack-queue release. No existing test exercises lifespan /
FrontendManager.shutdown (grep "lifespan" in tests/ = 0 hits), hence this file.

The subprocess test reproduces the SIGTERM topology (mp.Queue in parent + one spawn child,
uvicorn's capture_signals re-raise replicated) and asserts the resource_tracker no longer
warns about leaked /mp-* semaphores: 3 of the observed 4 were the ack_queue's locks.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace


def _wait_for_ready(ready_file: str, attempts: int = 1200) -> None:
    """Block until the spawn child says it is inside its sleep loop. Without this the
    SIGINT relay can land during the child's bootstrap (exit -SIGINT, no KeyboardInterrupt),
    which is a race, not the behavior under test."""
    for _ in range(attempts):
        if os.path.exists(ready_file):
            return
        time.sleep(0.025)
    raise AssertionError("spawn worker never became ready")


def _sleepy_worker(ready_file: str, ignore_sigint: bool = False, arm_sigint: bool = False) -> None:
    # Module-level: spawn children pickle the target by reference and re-import this
    # module, which must stay import-light (freetoken imports happen inside the tests).
    import pathlib

    if arm_sigint:
        # Mirror the production worker entry; the getattr fallback stands in for the
        # pre-fix worker, which had no re-arm and dropped a relayed SIGINT entirely.
        from freetoken.server import launch

        arm = getattr(launch, "_arm_sigint_handler", None)
        if arm is not None:
            arm()
    if ignore_sigint:
        import signal as _signal

        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    pathlib.Path(ready_file).touch()
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        pass  # the same graceful KeyboardInterrupt exit the scheduler worker takes


class _StubStop:
    """Stand-in for the ZmqAsync* queue attributes FrontendManager.shutdown stops."""

    def stop(self) -> None:
        pass


def _make_topology(tmp_path, ignore_sigint: bool = False, arm_sigint: bool = False):
    """One real spawn-context worker + one real mp.Queue wired through a FrontendManager
    the way run_api_server does (full BackendHandle retained on the state object)."""
    from freetoken.server import api_server

    # The one-shot reap guard is process-global; reset it so each test gets its own pass.
    api_server._BACKEND_REAP_STARTED.clear()

    ctx = mp.get_context("spawn")
    ready = str(tmp_path / "worker_ready")
    ack_queue = ctx.Queue()
    worker = ctx.Process(
        target=_sleepy_worker,
        args=(ready, ignore_sigint, arm_sigint),
        name="freetoken-TP0-scheduler",
    )
    worker.start()
    _wait_for_ready(ready)

    fm = api_server.FrontendManager(
        config=None, send_tokenizer=_StubStop(), recv_tokenizer=_StubStop()
    )
    fm.backend_processes = [worker]
    fm.backend_handle = SimpleNamespace(ack_queue=ack_queue, processes=[worker])
    return api_server, fm, ack_queue, worker


def test_lifespan_shutdown_reaps_workers_and_closes_ack_queue(monkeypatch, tmp_path):
    import pytest

    from freetoken.server import api_server

    _, fm, ack_queue, worker = _make_topology(tmp_path)

    # The supervisor attributes worker deaths to an orderly stop only when the flag is set
    # BEFORE any terminate reaches them (lifespan sets it, then calls shutdown()).
    flag_at_terminate = []
    real_terminate = api_server._terminate_backend_workers

    def terminate_spy(processes):
        flag_at_terminate.append(api_server._SHUTTING_DOWN.is_set())
        return real_terminate(processes)

    monkeypatch.setattr(api_server, "_terminate_backend_workers", terminate_spy)
    api_server._SHUTTING_DOWN.set()  # what lifespan does before _GLOBAL_STATE.shutdown()
    try:
        fm.shutdown()
    finally:
        api_server._SHUTTING_DOWN.clear()

    worker.join(timeout=5)
    assert flag_at_terminate == [True]  # (a) flag set before terminate
    # (b) reaped: the worker took the graceful SIGINT path (exit 0), not SIGTERM (-15).
    assert worker.exitcode == 0
    # (c) the ack queue is closed inside the graceful window: late puts are rejected.
    with pytest.raises(ValueError):
        ack_queue.put("late")


def test_lifespan_shutdown_escalates_to_sigterm_for_a_sigint_ignoring_worker(
    monkeypatch, tmp_path
):
    """A worker that ignores the SIGINT relay (e.g. wedged in teardown) must still be
    reaped inside the graceful window: join -> terminate -> join -> kill."""
    import pytest

    from freetoken.server import api_server

    # Shrink the escalation budget so the test stays fast; the helper reads the module
    # globals at call time.
    monkeypatch.setattr(api_server, "WORKER_SIGINT_GRACE_S", 0.5)

    _, fm, ack_queue, worker = _make_topology(tmp_path, ignore_sigint=True)
    api_server._SHUTTING_DOWN.set()
    try:
        fm.shutdown()
    finally:
        api_server._SHUTTING_DOWN.clear()

    worker.join(timeout=5)
    assert worker.exitcode == -signal.SIGTERM  # relay ignored -> terminate reaped it
    with pytest.raises(ValueError):
        ack_queue.put("late")


def test_lifespan_shutdown_reaps_worker_born_with_sigint_ignored(monkeypatch, tmp_path):
    """A background-job launch leaves the main process SIGINT=SIG_IGN at birth and spawn
    children inherit it across exec, so the kernel silently drops the shutdown relay and
    the worker escalates to SIGTERM. The worker entry re-arms default_int_handler before
    anything else, so the graceful KeyboardInterrupt path survives that birth state."""
    import pytest

    from freetoken.server import api_server

    terminate_calls = []
    real_terminate = api_server._terminate_backend_workers

    def terminate_spy(processes):
        terminate_calls.append([p.is_alive() for p in processes])
        return real_terminate(processes)

    monkeypatch.setattr(api_server, "_terminate_backend_workers", terminate_spy)

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # what a background-job birth looks like
    try:
        _, fm, ack_queue, worker = _make_topology(tmp_path, arm_sigint=True)
        api_server._SHUTTING_DOWN.set()
        fm.shutdown()
    finally:
        signal.signal(signal.SIGINT, previous)
        api_server._SHUTTING_DOWN.clear()

    worker.join(timeout=5)
    assert worker.exitcode == 0  # graceful KeyboardInterrupt path, not SIGTERM escalation
    assert terminate_calls == [[False]]  # terminate reached, found nothing left to signal
    with pytest.raises(ValueError):
        ack_queue.put("late")


def test_arm_sigterm_handler_maps_sigterm_to_graceful_keyboardinterrupt():
    """The SIGTERM arming helper routes a group SIGTERM into the same graceful
    KeyboardInterrupt teardown as Ctrl+C, and leaves BOTH stop signals ignored afterwards
    so the parent's relayed SIGINT cannot re-raise inside the teardown."""
    import pytest

    from freetoken.server import launch

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        launch._arm_sigterm_handler()
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
        # One-shot defense: after the raise, both companion signals are dead ends.
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        # The relayed SIGINT landing mid-teardown must be a no-op, not a second raise.
        signal.raise_signal(signal.SIGINT)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


_SIGTERM_TOPOLOGY_SCRIPT = textwrap.dedent(
    """\
    import multiprocessing as mp
    import os
    import signal
    import sys
    import time
    from types import SimpleNamespace


    class _Stop:
        def stop(self):
            pass


    def _wait_for_ready(ready_file, attempts=400):
        for _ in range(attempts):
            if os.path.exists(ready_file):
                return
            time.sleep(0.025)
        raise SystemExit("worker never became ready")


    def _sleepy_worker(ready_file):
        import pathlib
        pathlib.Path(ready_file).touch()
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            pass


    def main():
        mp.set_start_method("spawn", force=True)
        from freetoken.server import api_server

        ack_queue = mp.Queue()
        ready = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_ready")
        worker = mp.Process(target=_sleepy_worker, args=(ready,), name="freetoken-TP0-scheduler")
        worker.start()
        _wait_for_ready(ready)

        fm = api_server.FrontendManager(
            config=None, send_tokenizer=_Stop(), recv_tokenizer=_Stop()
        )
        fm.backend_processes = [worker]
        fm.backend_handle = SimpleNamespace(ack_queue=ack_queue, processes=[worker])

        def on_sigterm(signum, frame):
            # uvicorn: the lifespan shutdown sets the flag, then FrontendManager.shutdown().
            api_server._SHUTTING_DOWN.set()
            fm.shutdown()
            # capture_signals' finally: restore SIG_DFL and re-raise -> death by signal.
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.raise_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, on_sigterm)
        os.kill(os.getpid(), signal.SIGTERM)
        sys.stderr.write("probe survived the re-raised SIGTERM\\n")
        sys.exit(97)


    if __name__ == "__main__":
        main()
    """
)


def test_sigterm_parent_leaves_no_leaked_tracker_semaphores(tmp_path):
    """Fails-before/passes-after for the /mp-* semaphore leak: a minimal parent carrying
    the ft serve topology is SIGTERMed with uvicorn's re-raise replicated; the shared
    resource_tracker must report nothing leaked once the tree is dead."""
    script = tmp_path / "_sigterm_topology.py"
    script.write_text(_SIGTERM_TOPOLOGY_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # The re-raise must have killed the probe by SIGTERM: the wrapper contract is
    # "process exited => tree dead", not exit code 0 (SIGTERM -> 143 stays).
    assert proc.returncode == -signal.SIGTERM, (proc.returncode, proc.stderr[-2000:])
    assert "leaked semaphore" not in proc.stderr, proc.stderr[-4000:]


def test_shutdown_backend_workers_is_a_one_shot(monkeypatch, tmp_path):
    """The uvicorn lifespan and the shell teardown can both reach the stop helper (the
    shell's uvicorn-thread join is shorter than the worst-case escalation), so the second
    caller must no-op instead of racing a second teardown pass over a dead tree."""
    import pytest

    from freetoken.server import api_server

    terminate_calls = []
    real_terminate = api_server._terminate_backend_workers

    def counting_terminate(processes):
        terminate_calls.append(1)
        return real_terminate(processes)

    monkeypatch.setattr(api_server, "_terminate_backend_workers", counting_terminate)
    _, fm, ack_queue, worker = _make_topology(tmp_path)
    api_server._SHUTTING_DOWN.set()
    try:
        fm.shutdown()
        fm.shutdown()  # second call hits the one-shot guard
    finally:
        api_server._SHUTTING_DOWN.clear()

    worker.join(timeout=5)
    assert terminate_calls == [1]  # exactly one teardown pass ran
    assert worker.exitcode == 0  # and it was the full graceful one
    with pytest.raises(ValueError):
        ack_queue.put("late")  # the queue stayed closed; no second release pass


_SIGTERM_WORKER_SCRIPT = textwrap.dedent(
    """\
    import multiprocessing as mp
    import os
    import signal
    import sys
    import time


    def _wait_for_ready(ready_file, attempts=1200):
        for _ in range(attempts):
            if os.path.exists(ready_file):
                return
            time.sleep(0.025)
        raise SystemExit("worker never became ready")


    def _worker(ready_file, marker_file):
        # Production-shaped worker entry: re-arm the graceful handlers, idle until a stop
        # signal, then run the teardown stand-in (one marker line = one flush pass).
        import pathlib
        from freetoken.server import launch

        arm = getattr(launch, "_arm_sigint_handler", None)
        if arm is not None:
            arm()
        arm = getattr(launch, "_arm_sigterm_handler", None)
        if arm is not None:
            arm()
        pathlib.Path(ready_file).touch()
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            # Mirror the production one-shot defense first: whichever signal raised, the
            # other may still be pending-stale and must not re-raise inside the flush.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            with open(marker_file, "a") as fh:
                fh.write("flush\\n")
                fh.flush()
                os.fsync(fh.fileno())


    def main():
        mp.set_start_method("spawn", force=True)
        mode, workdir = sys.argv[1], sys.argv[2]
        ready = os.path.join(workdir, "worker_ready")
        marker = os.path.join(workdir, "flush_marker")
        worker = mp.Process(
            target=_worker, args=(ready, marker), name="freetoken-TP0-scheduler"
        )
        worker.start()
        _wait_for_ready(ready)
        # Orchestrator topology: the group SIGTERM lands on the worker itself; the
        # term_then_int mode adds the parent's SIGINT relay once the teardown has begun
        # (marker present), mirroring the milliseconds-late uvicorn relay.
        os.kill(worker.pid, signal.SIGTERM)
        if mode == "term_then_int":
            for _ in range(2000):
                if os.path.exists(marker):
                    break
                time.sleep(0.001)
            try:
                os.kill(worker.pid, signal.SIGINT)
            except ProcessLookupError:
                pass  # worker already exited gracefully
        worker.join(timeout=15)
        if worker.exitcode != 0:
            sys.stderr.write(f"worker exitcode={worker.exitcode}\\n")
            sys.exit(1)
        sys.exit(0)


    if __name__ == "__main__":
        main()
    """
)


def _run_sigterm_worker_mode(tmp_path, mode: str) -> None:
    script = tmp_path / "_sigterm_worker.py"
    script.write_text(_SIGTERM_WORKER_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script), mode, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (mode, proc.returncode, proc.stderr[-2000:])
    marker = tmp_path / "flush_marker"
    # Exactly one teardown pass: a second raise would have killed the worker mid-flush.
    assert marker.exists() and marker.read_text().splitlines() == ["flush"], (
        marker.read_text() if marker.exists() else "marker missing"
    )


def test_group_sigterm_runs_the_worker_graceful_teardown(tmp_path):
    """Fails-before for the SIGTERM gap: llama-swap/systemd/docker SIGTERM the whole process
    group, and a pre-fix worker died on the default SIGTERM action (-15) so the session-tier
    flush never ran. The armed worker must take the KeyboardInterrupt path: exit 0, one flush."""
    _run_sigterm_worker_mode(tmp_path, "term")


def test_sigterm_then_relayed_sigint_runs_a_single_teardown(tmp_path):
    """The parent's SIGINT relay lands while/after the teardown runs; the one-shot defense
    (SIG_IGN both signals inside the handler and again on the except path) must keep it
    from re-raising inside the teardown: still exit 0 with a single flush, never a double."""
    _run_sigterm_worker_mode(tmp_path, "term_then_int")
