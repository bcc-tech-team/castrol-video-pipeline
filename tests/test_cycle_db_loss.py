"""A cycle whose database goes away, or that is stopped from outside.

The incident behind these tests (2026-11-18): an unattended upgrade restarted
docker under a running 100-video cycle. The container lost its network, the
next query timed out, and the `finally` in `advisory_lock` then raised a second
`OperationalError` from `pg_advisory_unlock` on the same dead socket. That
replaced the real error, and no end-of-cycle event was ever written. Nothing
in the database said the batch had stopped.

None of this needs a database. The lock connection and the orchestrator are
faked, because what is under test is how the cycle behaves when they fail.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from castrol_pipeline import cli, cycle, orchestrator
from castrol_pipeline.common import db

# --------------------------------------------------------------------------
# advisory_lock
# --------------------------------------------------------------------------


class _FakeLockConn:
    """Answers the try-lock, then behaves however the test says on unlock."""

    def __init__(self, *, got: bool = True, unlock_raises: BaseException | None = None) -> None:
        self.got = got
        self.unlock_raises = unlock_raises
        self.statements: list[str] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def execute(self, sql: str, params: Any = None) -> SimpleNamespace:
        self.statements.append(sql)
        if "pg_try_advisory_lock" in sql:
            return SimpleNamespace(fetchone=lambda: {"got": self.got})
        if "pg_advisory_unlock" in sql and self.unlock_raises is not None:
            self._closed = True  # psycopg marks a connection BAD after this
            raise self.unlock_raises
        return SimpleNamespace(fetchone=lambda: None)

    def close(self) -> None:
        self._closed = True


@pytest.fixture
def lock_conn(monkeypatch: pytest.MonkeyPatch):
    """Install a fake lock connection. Returns a setter for its behaviour."""
    holder: dict[str, _FakeLockConn] = {}

    def install(**kwargs: Any) -> _FakeLockConn:
        conn = _FakeLockConn(**kwargs)
        holder["conn"] = conn

        def fake_connect(dsn: str, **kw: Any) -> _FakeLockConn:
            holder["kwargs"] = kw  # type: ignore[assignment]
            return conn

        monkeypatch.setattr(db.psycopg, "connect", fake_connect)
        monkeypatch.setattr(
            db, "get_settings",
            lambda: SimpleNamespace(require=lambda name: "postgresql://fake"),
        )
        return conn

    install.holder = holder  # type: ignore[attr-defined]
    return install


DEAD_SOCKET = psycopg.OperationalError(
    "consuming input failed: SSL SYSCALL error: Connection timed out"
)


class TestAdvisoryLock:
    def test_a_dead_connection_at_unlock_does_not_raise(self, lock_conn) -> None:
        conn = lock_conn(unlock_raises=DEAD_SOCKET)
        with db.advisory_lock(1) as got:
            assert got is True
        assert any("pg_advisory_unlock" in s for s in conn.statements)

    def test_the_bodys_own_error_is_the_one_that_propagates(self, lock_conn) -> None:
        """The regression. The unlock's error used to replace the real one."""
        lock_conn(unlock_raises=DEAD_SOCKET)

        class TheRealCause(Exception):
            pass

        with pytest.raises(TheRealCause), db.advisory_lock(1):
            raise TheRealCause("pool lost the database")

    def test_a_closed_connection_is_not_asked_to_unlock(self, lock_conn) -> None:
        conn = lock_conn()
        with db.advisory_lock(1):
            conn._closed = True
        assert not any("pg_advisory_unlock" in s for s in conn.statements)

    def test_an_untaken_lock_is_not_released(self, lock_conn) -> None:
        conn = lock_conn(got=False)
        with db.advisory_lock(1) as got:
            assert got is False
        assert not any("pg_advisory_unlock" in s for s in conn.statements)

    def test_the_connection_is_closed_on_every_path(self, lock_conn) -> None:
        conn = lock_conn()
        with db.advisory_lock(1):
            pass
        assert conn.closed

        conn = lock_conn(unlock_raises=DEAD_SOCKET)
        with pytest.raises(RuntimeError), db.advisory_lock(1):
            raise RuntimeError
        assert conn.closed

    def test_a_healthy_unlock_still_happens(self, lock_conn) -> None:
        conn = lock_conn()
        with db.advisory_lock(1):
            pass
        assert any("pg_advisory_unlock" in s for s in conn.statements)

    def test_the_idle_lock_connection_has_a_bounded_write_timeout(self, lock_conn) -> None:
        """Without one, the unlock on a dead peer waited ~15 minutes to fail."""
        lock_conn()
        with db.advisory_lock(1):
            pass
        kw = lock_conn.holder["kwargs"]
        assert kw["keepalives"] == 1
        assert 0 < kw["tcp_user_timeout"] <= 120_000


# --------------------------------------------------------------------------
# run_cycle
# --------------------------------------------------------------------------


@pytest.fixture
def quiet_cycle(monkeypatch: pytest.MonkeyPatch):
    """A cycle with every database touch faked. Returns the recorded events.

    One pass, nothing outstanding afterwards, unless a test overrides a piece.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    @contextmanager
    def fake_lock(key: int):
        yield True

    monkeypatch.setattr(cycle.db, "advisory_lock", fake_lock)
    monkeypatch.setattr(cycle.db, "fetch_all", lambda *a, **k: [{"status": "completed", "n": 3}])
    monkeypatch.setattr(cycle, "record_event", lambda name, **kw: events.append((name, kw)))
    monkeypatch.setattr(cycle, "_repair_orphans", lambda: {})
    monkeypatch.setattr(cycle, "_reopen_suppressed_deliveries", lambda: [])
    monkeypatch.setattr(cycle, "_schedule_all", lambda: 0)
    monkeypatch.setattr(
        cycle, "outstanding_work",
        lambda: {"queued": 0, "in_flight": 0, "next_due": None, "total": 0},
    )
    monkeypatch.setattr(orchestrator, "_checked_mode", lambda: "real")
    monkeypatch.setattr(orchestrator, "reap", lambda: 0)
    monkeypatch.setattr(orchestrator, "drain_stage", lambda stage: 0)
    # The upstream slice and the composite worker thread both claim through
    # execute_one. False is an empty queue.
    monkeypatch.setattr(orchestrator, "execute_one", lambda stage: False)
    monkeypatch.setattr(orchestrator, "poll_once", lambda **k: 0)
    return events


def _names(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [name for name, _ in events]


class TestDatabaseLostMidRun:
    def test_a_clean_run_starts_and_ends_with_one_event_each(self, quiet_cycle) -> None:
        out = cycle.run_cycle(fetch=False)
        assert _names(quiet_cycle) == ["cycle.start", "cycle.done"]
        assert cli.cycle_exit_code(out) == 0

    def test_a_lost_connection_is_reported_as_one(self, quiet_cycle, monkeypatch) -> None:
        def boom(stage: object) -> int:
            raise DEAD_SOCKET

        monkeypatch.setattr(orchestrator, "drain_stage", boom)
        out = cycle.run_cycle(fetch=False)

        assert "SSL SYSCALL error" in out["db_lost"]
        assert _names(quiet_cycle) == ["cycle.start", "cycle.db_lost"]
        name, fields = quiet_cycle[-1]
        assert fields["level"] == "error"
        assert fields["passes"] == 1
        assert cli.cycle_exit_code(out) == cli.EXIT_DB_LOST

    def test_a_pool_that_gave_up_counts_as_a_lost_database(self, quiet_cycle, monkeypatch) -> None:
        def no_connection() -> dict:
            raise PoolTimeout("couldn't get a connection after 30.00 sec")

        monkeypatch.setattr(cycle, "outstanding_work", no_connection)
        out = cycle.run_cycle(fetch=False)
        assert out["db_lost"].startswith("PoolTimeout")
        assert "cycle.done" not in _names(quiet_cycle)

    def test_a_lost_database_during_intake_is_not_blamed_on_the_export(
        self, quiet_cycle, monkeypatch
    ) -> None:
        """Intake's broad except exists for a dead export API, not a dead database."""
        import castrol_pipeline.intake.runner as runner

        def intake(**kw: object) -> object:
            raise DEAD_SOCKET

        monkeypatch.setattr(runner, "run_intake", intake)
        out = cycle.run_cycle(fetch=True)
        assert "intake_error" not in out
        assert out["db_lost"]
        assert "cycle.intake_failed" not in _names(quiet_cycle)

    def test_a_non_database_error_is_not_swallowed(self, quiet_cycle, monkeypatch) -> None:
        """Only a lost database is caught. A bug must still crash loudly."""
        def bug(stage: object) -> int:
            raise KeyError("stage")

        monkeypatch.setattr(orchestrator, "drain_stage", bug)
        with pytest.raises(KeyError):
            cycle.run_cycle(fetch=False)

    def test_a_lost_database_still_releases_the_lock(self, quiet_cycle, monkeypatch) -> None:
        released: list[bool] = []

        @contextmanager
        def tracking_lock(key: int):
            try:
                yield True
            finally:
                released.append(True)

        def boom(stage: object) -> int:
            raise DEAD_SOCKET

        monkeypatch.setattr(cycle.db, "advisory_lock", tracking_lock)
        monkeypatch.setattr(orchestrator, "drain_stage", boom)
        cycle.run_cycle(fetch=False)
        assert released == [True]


class TestStoppedFromOutside:
    def _one_pass_then(self, monkeypatch, on_drain) -> None:
        monkeypatch.setattr(
            cycle, "outstanding_work",
            lambda: {"queued": 5, "in_flight": 5, "next_due": None, "total": 10},
        )
        monkeypatch.setattr(orchestrator, "drain_stage", on_drain)

    @pytest.mark.skipif(
        threading.current_thread() is not threading.main_thread(),
        reason="signal handlers can only be installed from the main thread",
    )
    def test_sigterm_ends_the_cycle_with_an_event(self, quiet_cycle, monkeypatch) -> None:
        sent: list[bool] = []

        def drain(stage: object) -> int:
            if not sent:
                sent.append(True)
                signal.raise_signal(signal.SIGTERM)
            return 0

        self._one_pass_then(monkeypatch, drain)
        out = cycle.run_cycle(fetch=False, interval_s=3600)

        assert out["interrupted"] == "SIGTERM"
        assert _names(quiet_cycle) == ["cycle.start", "cycle.interrupted"]
        assert cli.cycle_exit_code(out) == cli.EXIT_INTERRUPTED

    def test_the_previous_sigterm_handler_is_restored(self, quiet_cycle) -> None:
        if threading.current_thread() is not threading.main_thread():
            pytest.skip("signal handlers can only be installed from the main thread")
        before = signal.getsignal(signal.SIGTERM)
        cycle.run_cycle(fetch=False)
        assert signal.getsignal(signal.SIGTERM) is before

    def test_a_stop_cuts_the_wait_short(self, quiet_cycle, monkeypatch) -> None:
        """The wait is an Event, so a stop does not sit out a 5-minute sleep."""
        stop = threading.Event()

        @contextmanager
        def fake_stop():
            yield stop

        waits: list[float] = []

        def drain(stage: object) -> int:
            return 0

        self._one_pass_then(monkeypatch, drain)
        monkeypatch.setattr(cycle, "_stop_on_sigterm", fake_stop)
        real_wait = stop.wait

        def wait(timeout: float | None = None) -> bool:
            waits.append(timeout or 0)
            stop.set()
            return real_wait(0)

        monkeypatch.setattr(stop, "wait", wait)
        out = cycle.run_cycle(fetch=False, interval_s=300)
        assert waits == [300]
        assert out["interrupted"] == "SIGTERM"
        assert out["passes"] == 1


class TestExitCodes:
    @pytest.mark.parametrize(
        ("summary", "code"),
        [
            ({}, 0),
            ({"skipped": True}, 0),
            ({"deadline_hit": True}, 1),
            ({"intake_error": "503"}, 1),
            ({"interrupted": "SIGTERM"}, cli.EXIT_INTERRUPTED),
            ({"db_lost": "x"}, cli.EXIT_DB_LOST),
            # A lost database explains a deadline, never the other way round.
            ({"db_lost": "x", "deadline_hit": True, "intake_error": "e"}, cli.EXIT_DB_LOST),
        ],
    )
    def test_codes(self, summary: dict, code: int) -> None:
        assert cli.cycle_exit_code(summary) == code
