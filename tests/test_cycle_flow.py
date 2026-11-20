"""The cycle's tick loop: finished renders are delivered while upstream is busy.

These drive `run_cycle` over a small in-memory pipeline instead of a database.
`execute_one`, `drain_stage`, `poll_once` and `outstanding_work` are replaced
by a fake that moves job ids from one stage queue to the next, so what is under
test is the loop's ORDER and its exit condition - which is exactly what Batch 2
got wrong: the old loop drained each stage to empty in turn, so a render that
came back early waited behind the whole card-burn backlog.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any

import psycopg
import pytest

from castrol_pipeline import cycle, orchestrator
from castrol_pipeline.stages.base import DEFAULT_PLAN, PipelineStage

S = PipelineStage

NEXT: dict[PipelineStage, PipelineStage | None] = {
    S.PREP: S.AUDIO,
    S.AUDIO: S.IMAGE,  # serialised here for simplicity; the real DAG forks
    S.IMAGE: S.VIDEO,
    S.VIDEO: None,  # goes in flight; the poll brings it back as COMPOSITE
    S.COMPOSITE: S.CHECKS,
    S.CHECKS: S.PUBLISH,
    S.PUBLISH: S.DELIVER,
    S.DELIVER: None,
}


class FakePipeline:
    """Queues per stage. One lock, because the composite worker is a thread."""

    def __init__(self, jobs: int, *, composite_s: float = 0.0, step_s: float = 0.0) -> None:
        self.lock = threading.Lock()
        self.queues: dict[PipelineStage, list[int]] = {s: [] for s in NEXT}
        self.queues[S.PREP] = list(range(jobs))
        self.in_flight: list[int] = []
        self.claimed = 0
        self.composite_s = composite_s
        self.step_s = step_s
        self.delivered: list[int] = []
        self.log: list[tuple[str, int]] = []  # (stage, job) in execution order

    def execute_one(self, stage: PipelineStage, *a: Any, **k: Any) -> bool:
        with self.lock:
            if not self.queues[stage]:
                return False
            job = self.queues[stage].pop(0)
            self.claimed += 1
        if stage == S.COMPOSITE and self.composite_s:
            time.sleep(self.composite_s)
        elif self.step_s:
            time.sleep(self.step_s)
        with self.lock:
            self.claimed -= 1
            self.log.append((str(stage), job))
            if stage == S.VIDEO:
                self.in_flight.append(job)
            elif stage == S.DELIVER:
                self.delivered.append(job)
            else:
                nxt = NEXT[stage]
                assert nxt is not None
                self.queues[nxt].append(job)
        return True

    def drain_stage(self, stage: PipelineStage, **k: Any) -> int:
        n = 0
        while self.execute_one(stage):
            n += 1
        return n

    def poll_once(self, **k: Any) -> int:
        with self.lock:
            done, self.in_flight = self.in_flight, []
            self.queues[S.COMPOSITE].extend(done)
        return len(done)

    def outstanding(self) -> dict[str, Any]:
        with self.lock:
            queued = sum(len(q) for q in self.queues.values()) + self.claimed
            return {"queued": queued, "in_flight": len(self.in_flight),
                    "next_due": None, "total": queued + len(self.in_flight)}


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch):
    def install(fake: FakePipeline) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []

        @contextmanager
        def lock(key: int):
            yield True

        monkeypatch.setattr(cycle.db, "advisory_lock", lock)
        monkeypatch.setattr(cycle.db, "fetch_all", lambda *a, **k: [])
        monkeypatch.setattr(cycle, "record_event", lambda n, **kw: events.append((n, kw)))
        monkeypatch.setattr(cycle, "_repair_orphans", lambda: {})
        monkeypatch.setattr(cycle, "_reopen_suppressed_deliveries", lambda: [])
        monkeypatch.setattr(cycle, "_schedule_all", lambda: 0)
        monkeypatch.setattr(cycle, "outstanding_work", fake.outstanding)
        monkeypatch.setattr(cycle, "UPSTREAM_SLICE_S", 0.002)
        monkeypatch.setattr(cycle, "COMPOSITE_IDLE_S", 0.001)
        monkeypatch.setattr(cycle, "next_sleep", lambda *a, **k: 0.01)
        monkeypatch.setattr(orchestrator, "_checked_mode", lambda: "mock")
        monkeypatch.setattr(orchestrator, "reap", lambda: 0)
        monkeypatch.setattr(orchestrator, "execute_one", fake.execute_one)
        monkeypatch.setattr(orchestrator, "drain_stage", fake.drain_stage)
        monkeypatch.setattr(orchestrator, "poll_once", fake.poll_once)
        return events

    return install


def test_the_lanes_are_exactly_the_plan() -> None:
    lanes = (cycle.DOWNSTREAM_STAGES + cycle.RENDER_STAGES
             + cycle.UPSTREAM_STAGES + cycle.WORKER_STAGES)
    assert sorted(lanes) == sorted(DEFAULT_PLAN)
    assert len(set(lanes)) == len(lanes)


def test_every_job_is_delivered_and_the_cycle_ends(world) -> None:
    fake = FakePipeline(40)
    events = world(fake)
    out = cycle.run_cycle(fetch=False, interval_s=1)
    assert sorted(fake.delivered) == list(range(40))
    assert [n for n, _ in events] == ["cycle.start", "cycle.done"]
    assert out["composites"] == 40


def test_a_finished_render_is_delivered_before_upstream_is_drained(world) -> None:
    """The Batch 2 failure. With a big intake, the old loop prepped, voiced and
    submitted everything before it looked at a returned render again."""
    fake = FakePipeline(150, step_s=0.001)
    world(fake)
    cycle.run_cycle(fetch=False, interval_s=1)

    order = [stage for stage, _ in fake.log]
    first_delivery = order.index("deliver")
    last_prep = len(order) - 1 - order[::-1].index("prep")
    assert first_delivery < last_prep, (
        "the first video was delivered only after every job had been prepped"
    )


def test_the_card_burn_does_not_hold_up_the_tick(world, monkeypatch) -> None:
    """Composite is on its own thread: polling carries on while it burns."""
    fake = FakePipeline(4, composite_s=0.2)
    world(fake)
    polls_during_burn: list[int] = []
    real_poll = fake.poll_once

    def poll(**k: Any) -> int:
        polls_during_burn.append(fake.claimed)
        return real_poll()

    monkeypatch.setattr(orchestrator, "poll_once", poll)
    cycle.run_cycle(fetch=False, interval_s=1)
    assert sorted(fake.delivered) == list(range(4))
    # Four 200ms burns run back to back on the worker while the tick, waking
    # every 10ms, keeps polling. In the old loop no poll could see one.
    assert sum(1 for c in polls_during_burn if c) > 10


def test_a_queue_that_reads_empty_mid_burn_does_not_end_the_cycle(world) -> None:
    """Composite enqueues `checks` in its own `finally`. Between the claim and
    that enqueue the queue reads empty; ending there would strand the video."""
    fake = FakePipeline(1, composite_s=0.3)
    world(fake)
    cycle.run_cycle(fetch=False, interval_s=1)
    assert fake.delivered == [0]


def test_a_lost_database_on_the_worker_ends_the_cycle_as_db_lost(world, monkeypatch) -> None:
    fake = FakePipeline(3)
    events = world(fake)
    real = fake.execute_one

    def execute(stage: PipelineStage, *a: Any, **k: Any) -> bool:
        if stage == S.COMPOSITE and fake.queues[S.COMPOSITE]:
            raise psycopg.OperationalError("SSL SYSCALL error: EOF detected")
        return real(stage)

    monkeypatch.setattr(orchestrator, "execute_one", execute)
    out = cycle.run_cycle(fetch=False, interval_s=1)
    assert "SSL SYSCALL" in out["db_lost"]
    assert [n for n, _ in events][-1] == "cycle.db_lost"


def test_a_bug_on_the_worker_still_crashes(world, monkeypatch) -> None:
    fake = FakePipeline(2)
    world(fake)
    real = fake.execute_one

    def execute(stage: PipelineStage, *a: Any, **k: Any) -> bool:
        if stage == S.COMPOSITE and fake.queues[S.COMPOSITE]:
            raise KeyError("bug")
        return real(stage)

    monkeypatch.setattr(orchestrator, "execute_one", execute)
    with pytest.raises(KeyError):
        cycle.run_cycle(fetch=False, interval_s=1)


def test_the_worker_stops_claiming_once_halted(monkeypatch) -> None:
    calls: list[PipelineStage] = []
    monkeypatch.setattr(cycle, "COMPOSITE_IDLE_S", 0.001)
    monkeypatch.setattr(orchestrator, "execute_one",
                        lambda stage, *a, **k: calls.append(stage) or False)
    w = cycle.CompositeWorker(threading.Event())
    w.start()
    time.sleep(0.05)
    w.halt()
    n = len(calls)
    time.sleep(0.05)
    assert not w.is_alive()
    assert len(calls) == n
    assert set(calls) == {S.COMPOSITE}
