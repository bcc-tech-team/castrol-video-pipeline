"""A 500-video batch on a bad day, simulated: does it finish with nobody watching?

NOT the rehearsal. The rehearsal (`STAGE_MODE=mock`, deploy/README.md) runs the
real cycle against a real database and was not run for these changes - there
was no throwaway database to run it on, and the shared one must not hold
rehearsal rows. This is the next best thing: a discrete-event model of the
cycle's tick loop, on a simulated clock, driving the REAL policy objects -

  * `pacing.RateLimiter`, the per-provider request window
  * `pacing.in_flight_cap`, the render slots
  * `health.ProviderHealth`, the outage pause and the credits pause
  * `orchestrator.retry_delay_for`, the VENDOR_UNAVAILABLE schedule
  * `orchestrator.failure_decision`, the self-releasing hold and its cap
  * `orchestrator.poll_order`, which in-flight tasks get polled

- against a modelled video provider with Batch 1 and 2's failure profile: 50%
of render attempts come back "500 internal error" within seconds, the account
answers more than 20 requests in 10s with a 429, there is a stretch where
EVERY request is a 429, and a stretch with no credits (402 on submit, a zero
balance). What it does not model: SQL, the real mock stages, ffmpeg. Those are
covered by their own tests; this pins how the policies add up.

Acceptance, from the plan: every job delivered, no manual command, first
deliveries within ~15 minutes of the first renders, and never more than 20
requests in any 10 seconds. Run with `-s` to print the batch report.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from castrol_pipeline import orchestrator
from castrol_pipeline.common import alerts, health, pacing
from castrol_pipeline.common.errors import StageErrorCode
from castrol_pipeline.config import get_settings
from castrol_pipeline.stages import mocks
from castrol_pipeline.stages.base import PipelineStage

EPOCH = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)

#: Seconds of the cycle's own (main-thread) time per operation. Measured on the
#: worker where we have it (cycle.py docstring); the rest are generous.
COST = {"prep": 0.5, "audio": 1.5, "image_submit": 0.4, "video_submit": 0.5,
        "poll": 0.3, "checks": 1.0, "publish": 1.0, "deliver": 1.0}
COMPOSITE_S = 35.0          # PR #11, on the worker
IMAGE_RENDER_S = (60, 120)  # avg ~83s observed
VIDEO_RENDER_S = (480, 720)  # 8-12 minutes
FAIL_FAST_S = (3, 30)       # Batch 1/2's 500s were "0s renders"


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(s, 1e-3)


@dataclass
class Scenario:
    jobs: int = 500
    render_500_share: float = 0.5
    throttle_windows: list[tuple[float, float]] = field(default_factory=list)
    no_credits_windows: list[tuple[float, float]] = field(default_factory=list)
    outage_windows: list[tuple[float, float]] = field(default_factory=list)
    provider_limit: int = 20
    seed: int = 7


class Provider:
    """The modelled video provider. Logs every request, refused or not."""

    def __init__(self, sc: Scenario, clock: Clock, rng: random.Random) -> None:
        self.sc, self.clock, self.rng = sc, clock, rng
        self.log: list[tuple[float, str, str]] = []  # (t, kind, answer)
        self.times: list[float] = []  # the same, for the window count
        self.tasks: dict[str, tuple[float, bool]] = {}  # id -> (done_at, failed)
        self.n = 0

    @staticmethod
    def _in(windows: list[tuple[float, float]], t: float) -> bool:
        return any(a <= t < b for a, b in windows)

    def request(self, kind: str) -> str:
        t = self.clock.t
        recent = len(self.times) - bisect_right(self.times, t - 10.0) + 1
        if self._in(self.sc.throttle_windows, t) or recent > self.sc.provider_limit:
            answer = "429"
        elif kind == "submit" and self._in(self.sc.no_credits_windows, t):
            answer = "402"
        else:
            answer = "200"
        self.log.append((t, kind, answer))
        self.times.append(t)
        return answer

    def submit(self) -> str:
        self.n += 1
        tid = f"t{self.n}"
        t = self.clock.t
        fails = self._in(self.sc.outage_windows, t) or self.rng.random() < self.sc.render_500_share
        lo, hi = FAIL_FAST_S if fails else VIDEO_RENDER_S
        self.tasks[tid] = (t + self.rng.uniform(lo, hi), fails)
        return tid

    def balance(self) -> float:
        return 0.0 if self._in(self.sc.no_credits_windows, self.clock.t) else 10_000.0


@dataclass
class Job:
    id: int
    stage: str = "prep"                # next thing to do, or where it waits
    image_done_at: float | None = None
    video_task: str | None = None
    video_attempts: int = 0            # on the current run
    video_lifetime: int = 0            # across every run
    video_next_at: float = 0.0
    video_failed_at: float | None = None
    render_done_at: float | None = None
    composite_done_at: float | None = None
    delivered_at: float | None = None
    auto_released: int = 0


def simulate(sc: Scenario, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    s = get_settings()
    clock, rng = Clock(), random.Random(sc.seed)
    provider = Provider(sc, clock, rng)
    events: list[tuple[float, str]] = []

    def record(name: str, **kw: Any) -> None:
        events.append((clock.t, name))

    for mod in (health, alerts, orchestrator):
        monkeypatch.setattr(mod, "record_event", record)
    monkeypatch.setattr(alerts, "_recently_raised",
                        lambda a: any(n == a.event for _, n in events))

    video_gate = pacing.RateLimiter(s.video_requests_per_window, s.provider_window_s,
                                    clock=clock, sleep=clock.sleep)
    image_gate = pacing.RateLimiter(s.image_requests_per_window, s.provider_window_s,
                                    clock=clock, sleep=clock.sleep)
    vh = health.ProviderHealth("video", health.OutagePolicy.from_settings(), clock=clock)
    monkeypatch.setattr(health, "_HEALTH", {"video": vh})
    cap = pacing.in_flight_cap(PipelineStage.VIDEO) or 60
    image_cap = pacing.in_flight_cap(PipelineStage.IMAGE) or 40

    jobs = [Job(i) for i in range(sc.jobs)]
    last_polled: dict[str, float] = {}
    composite_q: deque[Job] = deque()
    worker_free_at = 0.0
    composite_running: Job | None = None
    deadline = 8 * 3600.0
    ticks = 0

    def spend(op: str) -> None:
        clock.t += COST[op]

    def at(job_t: float) -> datetime:
        return EPOCH + timedelta(seconds=job_t)

    def run_composite_worker() -> None:
        nonlocal worker_free_at, composite_running
        while True:
            if composite_running is not None:
                if composite_running.composite_done_at <= clock.t:
                    composite_running.stage = "checks"
                    composite_running = None
                else:
                    return
            if not composite_q:
                return
            job = composite_q.popleft()
            start = max(worker_free_at, job.render_done_at or 0.0)
            job.composite_done_at = start + COMPOSITE_S
            worker_free_at = job.composite_done_at
            composite_running = job

    def video_blocked() -> str | None:
        if vh.balance_check_due():
            if video_gate.acquire(s.provider_slot_wait_s):
                if provider.request("balance") == "200":
                    owed = sum(1 for j in jobs if j.stage in ("video", "wait_video"))
                    needed = min(owed, cap) * 1.08
                    vh.balance_checked(covered=provider.balance() >= needed,
                                       balance_usd=provider.balance(), needed_usd=needed)
        reason = vh.submit_block_reason()
        if reason:
            return reason
        if sum(1 for j in jobs if j.stage == "rendering") >= cap:
            return "cap"
        return None

    while clock.t < deadline:
        ticks += 1
        run_composite_worker()

        # Self-release sweep: the real decision function, on the sim clock.
        for j in jobs:
            if j.stage == "video_failed":
                row = {"id": j.id, "status": "failed", "input_hash": "h",
                       "error_code": str(StageErrorCode.VENDOR_UNAVAILABLE),
                       "finished_at": at(j.video_failed_at or 0.0),
                       "lifetime_attempts": j.video_lifetime}
                if orchestrator.failure_decision(row, "h", now=at(clock.t)) == orchestrator.RELEASE:
                    j.stage, j.video_attempts, j.video_next_at = "video", 0, clock.t
                    j.auto_released += 1

        # Poll a slice, least recently polled first.
        rows = [{"id": j.video_task, "stage": "video", "started_at": at(0), "job": j}
                for j in jobs if j.stage == "rendering"]
        polls, exhausted = 0, False
        for row in orchestrator.poll_order(rows, last_polled, clock.t,
                                           lambda st: s.video_poll_interval_s):
            if polls >= 20 or exhausted:
                break
            if not video_gate.acquire(s.provider_slot_wait_s):
                break
            polls += 1
            first_look = row["id"] not in last_polled
            last_polled[row["id"]] = clock.t
            answer = provider.request("poll")
            spend("poll")
            if answer == "429":
                video_gate.back_off(s.throttle_backoff_s)
                exhausted = True
                continue
            j = row["job"]
            done_at, failed = provider.tasks[j.video_task]
            if done_at > clock.t:
                if first_look:
                    vh.record_verdict(failed=False)  # as orchestrator.poll_once
                continue
            if failed:
                vh.record_verdict(failed=True)
                delay = orchestrator.retry_delay_for(j.video_attempts,
                                                     StageErrorCode.VENDOR_UNAVAILABLE)
                if delay is None:
                    j.stage, j.video_failed_at = "video_failed", clock.t
                else:
                    j.stage, j.video_next_at = "video", clock.t + delay
            else:
                if first_look:
                    vh.record_verdict(failed=False)
                j.stage, j.render_done_at = "composite", clock.t
                composite_q.append(j)
        # Images: a separate provider and limiter; never refused in this model.
        for j in jobs:
            if j.stage == "imaging" and j.image_done_at <= clock.t:
                if image_gate.acquire(s.provider_slot_wait_s):
                    spend("poll")
                    j.stage = "video"
                    j.video_next_at = clock.t
        run_composite_worker()

        # Downstream, to empty.
        for step, nxt in (("checks", "publish"), ("publish", "deliver"), ("deliver", "done")):
            for j in jobs:
                if j.stage == step:
                    spend(step)
                    j.stage = nxt
                    if nxt == "done":
                        j.delivered_at = clock.t

        # Fill render slots.
        for j in sorted((j for j in jobs if j.stage == "video" and j.video_next_at <= clock.t),
                        key=lambda j: j.video_next_at):
            if video_blocked() or not video_gate.acquire(s.provider_slot_wait_s):
                break
            answer = provider.request("submit")
            spend("video_submit")
            if answer == "429":
                video_gate.back_off(s.throttle_backoff_s)
                j.video_next_at = clock.t + s.throttle_backoff_s  # attempt NOT spent
                break
            if answer == "402":
                if vh.pause_for_credits("402"):
                    alerts.raise_alert(alerts.CREDITS_LOW, "out of credits")
                break  # attempt NOT spent
            j.video_attempts += 1
            j.video_lifetime += 1
            j.video_task = provider.submit()
            j.stage = "rendering"
            vh.note_submit()

        # Upstream slice, round-robin, most-downstream first.
        end = clock.t + 15.0
        while clock.t < end:
            did = False
            imaging = sum(1 for j in jobs if j.stage == "imaging")
            for step in ("image", "audio", "prep"):
                j = next((j for j in jobs if j.stage == step), None)
                if j is None or (step == "image" and imaging >= image_cap):
                    continue
                if step == "image":
                    if not image_gate.acquire(s.provider_slot_wait_s):
                        continue
                    spend("image_submit")
                    j.stage = "imaging"
                    j.image_done_at = clock.t + rng.uniform(*IMAGE_RENDER_S)
                    imaging += 1
                else:
                    spend(step)
                    j.stage = "audio" if step == "prep" else "image"
                did = True
            if not did:
                break

        if all(j.stage == "done" for j in jobs):
            break
        if vh.outage_minutes() > s.alert_outage_minutes:
            alerts.raise_alert(alerts.PROVIDER_OUTAGE, "long outage", dedupe_key="o")
        clock.t += 5.0  # the idle wait; the real loop waits up to 20s

    sent = [t for t, _, _ in provider.log]
    renders_done = sorted(j.render_done_at for j in jobs if j.render_done_at is not None)
    delivered = sorted(j.delivered_at for j in jobs if j.delivered_at is not None)
    attempts = [j.video_lifetime for j in jobs]
    return {
        "jobs": sc.jobs,
        "delivered": len(delivered),
        "needs_a_person": sum(1 for j in jobs if j.stage == "video_failed"
                              and j.video_lifetime >= s.unavailable_lifetime_attempts),
        "hours": round(clock.t / 3600, 2),
        "ticks": ticks,
        "first_render_min": round(renders_done[0] / 60, 1) if renders_done else None,
        "first_delivery_min": round(delivered[0] / 60, 1) if delivered else None,
        "max_requests_in_10s": mocks.max_in_any_window(sent, 10.0),
        "requests": len(sent),
        "refused_429": sum(1 for _, _, a in provider.log if a == "429"),
        "refused_402": sum(1 for _, _, a in provider.log if a == "402"),
        "render_attempts": sum(attempts),
        "max_attempts_one_job": max(attempts),
        "auto_released": sum(j.auto_released for j in jobs),
        "events": sorted({n for _, n in events}),
        "credits_alerts": sum(1 for _, n in events if n == "alert.credits_low"),
    }


def _print(name: str, out: dict[str, Any]) -> None:
    print(f"\n--- {name} ---")
    for k, v in out.items():
        print(f"  {k:<22} {v}")


def test_a_bad_day_at_500_videos_finishes_by_itself(monkeypatch) -> None:
    """The acceptance run: 50% render 500s, a 5-minute stretch of nothing but
    429s, and 30 minutes out of credits in the middle of the batch."""
    sc = Scenario(
        throttle_windows=[(1800, 2100)],
        no_credits_windows=[(5400, 7200)],
    )
    out = simulate(sc, monkeypatch)
    _print("500 jobs, 50% 500s, 429 burst, 30 min with no credits", out)

    assert out["delivered"] == 500
    assert out["needs_a_person"] == 0
    assert out["hours"] < 8, "did not finish inside one cycle's deadline"
    assert out["max_requests_in_10s"] <= 20
    assert out["first_delivery_min"] - out["first_render_min"] <= 15
    assert out["refused_429"] > 0 and out["refused_402"] > 0, "the bad day did not happen"
    assert "provider.credits_exhausted" in out["events"]
    assert "provider.credits_restored" in out["events"]
    assert out["credits_alerts"] == 1
    # 50% is the ordinary bad rate, not an outage: it must not pause.
    assert "provider.outage_paused" not in out["events"]


def test_an_outage_pauses_probes_and_resumes(monkeypatch) -> None:
    """A 40-minute window where every render fails, on top of the 50%."""
    sc = Scenario(jobs=300, outage_windows=[(2400, 4800)])
    out = simulate(sc, monkeypatch)
    _print("300 jobs, 50% 500s, 40-minute total outage", out)

    assert out["delivered"] == 300
    assert out["needs_a_person"] == 0
    assert out["max_requests_in_10s"] <= 20
    assert "provider.outage_paused" in out["events"]
    assert "provider.outage_probing" in out["events"]
    assert "provider.outage_ended" in out["events"]
