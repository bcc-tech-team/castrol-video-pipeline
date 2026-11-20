"""One unattended run, start to finish: pull the window, then drive every job
to a terminal state.

This is what the twice-daily timer invokes, and the only entry point expected
to run with nobody watching. It differs from `drain` in three ways that only
matter when there is no human in the room:

  * **It takes a lock.** The noon run must not start on top of a midnight run
    that is still rendering. Two cycles at once would not corrupt anything —
    claiming is `SKIP LOCKED` and readiness is computed — but they would double
    the concurrent load on a paid vendor for no gain, and make the logs
    unreadable at exactly the moment someone is trying to work out what broke.

  * **It waits.** `drain` stops the moment a sweep moves nothing, which for an
    async stage means "the video is still rendering". Under a timer that would
    submit every paid render and exit before collecting any of them, leaving
    the money spent and the results uncollected until the next cycle.

  * **It stops at a deadline** instead of running until the next timer fires.

Throughput is not what the deadline limits, and the arithmetic is worth writing
down because it looks otherwise. A render takes ~8.6 minutes, but renders do not
queue behind each other - image and video submit and release, so a hundred of
them are in flight at the vendor at once and the wall clock is the LONGEST one,
not the sum. What is serial is only our own work, measured end to end on a real
job at **~32 seconds per video**, nearly all of it the ffmpeg composite:

    prep 1s | audio <1s | image submit <1s | video submit <1s
    | composite 28s | checks 1s | publish 1s | deliver 1s

At 8 hours that is roughly 900 videos per cycle, so the deadline is never what
stops a realistic batch. The binding limit is `vendor_limits`, specifically
`daily_cost_cap_usd`: $5000 on video after migrations 0011 and 0012, about 4732
videos a day at the measured ~$1.06 each. The call caps were raised alongside
it so the cost cap is what trips. That is a runaway guard rather than a budget
- observed volume is ~40 videos a day - and it fails closed, as does the call
cap behind it. Neither is an accident to route around.

Note the cap day is IST, and the 00:00 and 12:00 IST cycles fall inside the
same one - a batch that consumes the day's allowance at midnight leaves the
noon cycle nothing.

Nothing here is a new source of truth. Readiness is still computed from
`stage_runs`, so a cycle killed at any point — deadline, deploy, instance
reboot — resumes from the next one with no bookkeeping to repair. That is the
property that makes it safe to run this unattended at all.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import psycopg

from .common import db
from .common.events import record_event
from .common.logging import get_logger
from .config import get_settings
from .stages.base import PipelineStage

log = get_logger(__name__)

#: Lock key within `db.ADVISORY_LOCK_NAMESPACE`. One cycle at a time, across
#: every host pointed at this database — not just this machine, which is what a
#: pidfile or a flock would have given us.
CYCLE_LOCK_KEY = 1

#: Longest we will sit idle waiting for a retry that is not due yet. A backed-off
#: run has a known due time and could be slept to exactly, but waking up
#: periodically keeps the log showing a live process.
MAX_SLEEP_S = 300

#: The three lanes of a tick, plus the card burn on its own thread. Together
#: they are exactly `DEFAULT_PLAN`, which `tests/test_cycle_flow.py` pins, so a
#: stage added to the plan cannot silently be left out of the loop.
#:
#: DOWNSTREAM is drained to empty on every tick. It is bounded by the card
#: burn's own throughput (one video every ~35s on the worker), each run takes
#: about a second, and these are the stages that put a finished video in front
#: of the mechanic.
DOWNSTREAM_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage.CHECKS,
    PipelineStage.PUBLISH,
    PipelineStage.DELIVER,
)
#: The paid render. Submit and release, so a drain is a few HTTP calls.
RENDER_STAGES: tuple[PipelineStage, ...] = (PipelineStage.VIDEO,)
#: Upstream gets a time slice per tick, round-robin, most-downstream first, so
#: a fresh 500-job intake feeds renders as it goes instead of prepping all 500
#: before the first voiceover is made.
UPSTREAM_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage.IMAGE,
    PipelineStage.AUDIO,
    PipelineStage.PREP,
)
#: Run by `CompositeWorker`, never by the tick.
WORKER_STAGES: tuple[PipelineStage, ...] = (PipelineStage.COMPOSITE,)

#: Seconds of upstream work per tick before the loop goes back to polling.
UPSTREAM_SLICE_S = 15.0

#: How long the composite worker waits before looking again when its queue is
#: empty. Short, because a render that just came back is waiting on it.
COMPOSITE_IDLE_S = 2.0

#: How long `halt()` waits for an in-progress card burn to finish. A burn that
#: is cut off leaves its run `claimed`, and the reaper returns it to the queue,
#: so nothing is lost past this - the stage is local ffmpeg and free.
COMPOSITE_JOIN_S = 120.0
#: The same, after a SIGTERM, which leaves ten seconds before the SIGKILL.
COMPOSITE_STOP_JOIN_S = 3.0

#: What "the database went away" looks like from here. `PoolTimeout`, raised
#: when the pool cannot hand out a connection, subclasses `OperationalError`,
#: so a pool that has given up and a socket that timed out are both covered.
#: `InterfaceError` covers using a connection psycopg has already closed.
DB_LOST_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
)


@contextmanager
def _stop_on_sigterm() -> Iterator[threading.Event]:
    """Turn SIGTERM into a flag that the wait loop checks between steps.

    In the container the cycle is PID 1, and the kernel does not apply the
    default action to a signal sent to PID 1. With no handler, `docker stop`
    and a daemon restart deliver a SIGTERM that does nothing, and the process
    is SIGKILLed when the grace period runs out. That leaves no end event and
    no unlock. A flag, and not an exception, because an exception raised
    between a paid submit and the write that records its task id would lose
    the task (invariant 13). The flag is only checked where nothing is in
    flight.

    Only installed from the main thread, which is the only place Python allows
    it. Anything else, a test runner thread for example, gets a flag that
    never fires.
    """
    stop = threading.Event()
    if threading.current_thread() is not threading.main_thread():
        yield stop
        return

    def handler(signum: int, _frame: object) -> None:
        stop.set()

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield stop
    finally:
        signal.signal(signal.SIGTERM, previous)


class CompositeWorker(threading.Thread):
    """Burns cards one after another, off the tick loop's thread.

    The card burn is the only slow thing the cycle does itself: ~35s per video
    on the worker at `-preset medium` (PR #11), against about a second for
    every other stage. In the main loop it blocked polling and delivery for as
    long as its queue lasted. On a thread it runs back to back while the tick
    keeps polling and delivering.

    One thread, not a pool. The t3.medium has two vCPUs and ffmpeg already uses
    both (`-threads 2`), so a second burn would halve the speed of each and
    deliver nothing sooner.

    It never decides anything. `execute_one` claims, runs, records and
    reschedules exactly as it does in the main thread. The pool is thread-safe
    and claiming is `SKIP LOCKED`.

    An exception that escapes `execute_one` is a database or code failure, not
    a stage failure (those are caught and recorded inside it). It stops the
    worker and is re-raised on the main thread by `raise_if_failed`, so a lost
    database still ends the cycle as `cycle.db_lost` and a bug still crashes.
    """

    def __init__(self, stop: threading.Event) -> None:
        super().__init__(name="composite-worker", daemon=True)
        self._stop_signal = stop
        self._halt = threading.Event()
        self.error: BaseException | None = None
        #: True from before a claim until after the reschedule that follows.
        self.busy = False
        #: Calls to `execute_one` that returned, found work or not.
        self.calls = 0
        #: Runs actually executed.
        self.done = 0

    def _should_stop(self) -> bool:
        return self._halt.is_set() or self._stop_signal.is_set()

    def run(self) -> None:
        from .orchestrator import execute_one

        while not self._should_stop():
            self.busy = True
            try:
                did = execute_one(PipelineStage.COMPOSITE)
            except BaseException as exc:  # noqa: BLE001 - handed to the main thread
                self.error = exc
                log.error("cycle.composite_worker_failed", error=str(exc)[:400])
                return
            finally:
                self.calls += 1
                self.busy = False
            if did:
                self.done += 1
            else:
                self._halt.wait(COMPOSITE_IDLE_S)

    def halt(self) -> None:
        """Stop claiming, and give a burn in progress a chance to finish.

        After a SIGTERM the wait is short: `docker stop` sends SIGKILL ten
        seconds later, and the cycle still has to write `cycle.interrupted`
        before then. The thread is a daemon, so an unfinished burn dies with
        the process and its `claimed` run is reaped back to the queue.
        """
        self._halt.set()
        if self.is_alive():
            self.join(COMPOSITE_STOP_JOIN_S if self._stop_signal.is_set() else COMPOSITE_JOIN_S)

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise self.error


def _drain_all(
    stages: tuple[PipelineStage, ...], totals: dict[str, int], stop: threading.Event
) -> int:
    """Drain each stage to empty, in order. Returns runs executed."""
    from .orchestrator import drain_stage

    ran = 0
    for stage in stages:
        if stop.is_set():
            break
        n = drain_stage(stage)
        if n:
            totals[str(stage)] = totals.get(str(stage), 0) + n
            ran += n
    return ran


def _drain_for(
    stages: tuple[PipelineStage, ...],
    seconds: float,
    totals: dict[str, int],
    stop: threading.Event,
) -> bool:
    """Round-robin one run per stage until the slice is spent or all are empty.

    Returns True if it stopped on time with work still queued, which tells the
    loop to come straight back rather than wait out a tick.
    """
    from .orchestrator import execute_one

    end = time.monotonic() + seconds
    active = list(stages)
    while active and not stop.is_set():
        for stage in list(active):
            if stop.is_set():
                return False
            if execute_one(stage):
                totals[str(stage)] = totals.get(str(stage), 0) + 1
            else:
                active.remove(stage)
            if time.monotonic() >= end:
                return bool(active)
    return False


def intake_window(lookback_days: int, *, today: date | None = None) -> tuple[date, date]:
    """The window to pull, on the client's calendar.

    Anchored in the client's timezone, not the server's: EC2 runs UTC, and at
    00:00 IST it is still yesterday there — a window built from the server's own
    date would miss the day that has just started, every single midnight run.

    Both ends are deliberately loose. `from` reaches back because a submission
    can land after its window was already pulled, and `to` reaches forward one
    day because we have not confirmed which timezone the API filters on; a
    future `to` is accepted and simply returns nothing extra. Overlap is free —
    intake dedupes on the client's own row id — and a missed row is not
    recoverable by anything downstream, so the trade only runs one way.
    """
    tz = ZoneInfo(get_settings().export_timezone)
    anchor = today or datetime.now(tz).date()
    return anchor - timedelta(days=lookback_days), anchor + timedelta(days=1)


def outstanding_work() -> dict[str, Any]:
    """What is still owed: queued runs, in-flight vendor tasks, next retry due.

    Deliberately not filtered by job status. A job can be marked `failed` on one
    stage while another is still running at a vendor, and that task still has to
    be collected — the submit was already billed (invariant 24), so walking away
    from it loses the money AND the evidence of what it cost.
    """
    row = db.fetch_one(
        """
        SELECT count(*) FILTER (WHERE status = 'running')              AS in_flight,
               count(*) FILTER (WHERE status IN ('pending', 'claimed')) AS queued,
               min(next_attempt_at) FILTER (WHERE status = 'pending')   AS next_due
          FROM stage_runs
         WHERE status IN ('pending', 'claimed', 'running');
        """
    )
    queued = int(row["queued"]) if row else 0
    in_flight = int(row["in_flight"]) if row else 0
    return {
        "queued": queued,
        "in_flight": in_flight,
        "next_due": row["next_due"] if row else None,
        "total": queued + in_flight,
    }


def next_sleep(left: dict[str, Any], interval_s: int, remaining_s: float) -> float:
    """How long to wait before the next pass.

    Anything in flight at a vendor is polled at the interval. If the only thing
    left is a run behind a retry backoff, its due time is known, so waiting for
    it beats waking every interval to find it still not due.
    """
    wait = float(interval_s)
    due = left.get("next_due")
    if left["in_flight"] == 0 and due is not None:
        delta = (due - datetime.now(UTC)).total_seconds()
        wait = max(wait, min(delta, MAX_SLEEP_S))
    return max(1.0, min(wait, remaining_s))


def _schedule_all() -> int:
    """Enqueue every ready stage across every unfinished job.

    Intake creates jobs but does not schedule them, and a previous cycle may
    have stopped at its deadline with work left. This is the one place that
    catches up, and it is why the cycle needs no memory of what the last one did.
    """
    from .orchestrator import advance_job, schedule_ready

    enqueued = 0
    for row in db.fetch_all(
        "SELECT id FROM jobs WHERE status NOT IN ('completed', 'cancelled');"
    ):
        enqueued += len(schedule_ready(str(row["id"])))
        advance_job(str(row["id"]))
    return enqueued


def _repair_orphans() -> dict[str, int]:
    """Finish the intake rows a crash left half-created.

    Intake writes one mechanic across THREE separate transactions - the
    submission, then the job, then the photo asset - because each db helper
    opens its own. A process killed between them (deadline, deploy, instance
    reboot) leaves either a valid submission with no job, or a job with no
    source photo.

    Neither state is an error anywhere. No stage fails, no row is marked, the
    batch counters look right, and the mechanic simply never gets a video. The
    only way it surfaces today is somebody asking why.

    Both halves are free and idempotent, so this runs every cycle rather than
    waiting to be noticed. A photo whose SAS url has since expired cannot be
    recovered - that is logged and skipped, because the alternative is failing
    the whole cycle over one unreachable blob.
    """
    from .common.s3 import get_storage, job_key
    from .intake.media import MediaError, fetch_photo
    from .intake.runner import _resolve_plate
    from .prep.script import current_version

    settings = get_settings()
    repaired = {"jobs_created": 0, "photos_restored": 0, "photos_unrecoverable": 0}

    for row in db.fetch_all(
        """
        SELECT s.id, s.outfit_choice, s.background_choice, s.batch_id,
               COALESCE(s.language, 'hi') AS language
          FROM submissions s
          LEFT JOIN jobs j ON j.submission_id = s.id
         WHERE s.validation_status = 'valid'
           AND j.id IS NULL;
        """
    ):
        created = db.fetch_one(
            """
            INSERT INTO jobs (submission_id, script_version, voice_id, plate_id,
                              batch_id, language)
            VALUES (%(sub)s, %(script)s, %(voice)s, %(plate)s, %(batch)s,
                    %(language)s)
            ON CONFLICT (submission_id) DO NOTHING
            RETURNING id;
            """,
            {
                "sub": str(row["id"]),
                # The language intake resolved, so a repaired job speaks what
                # the mechanic asked for rather than defaulting to Hindi.
                "language": row["language"],
                "script": current_version(row["language"]),
                "voice": settings.tts_voice_id or "stub-voice",
                "plate": _resolve_plate(
                    row["outfit_choice"] or "", row["background_choice"] or ""
                ),
                "batch": row["batch_id"],
            },
        )
        if created:
            repaired["jobs_created"] += 1
            record_event(
                "cycle.orphan_job_created",
                job_id=str(created["id"]),
                level="warning",
                submission_id=str(row["id"]),
            )

    for row in db.fetch_all(
        """
        SELECT j.id, s.image_url_raw
          FROM jobs j
          JOIN submissions s ON s.id = j.submission_id
          LEFT JOIN assets a ON a.job_id = j.id AND a.kind = 'source_photo'
         WHERE a.id IS NULL
           AND j.status <> 'cancelled'
           AND coalesce(s.image_url_raw, '') <> '';
        """
    ):
        job_id = str(row["id"])
        try:
            # Byte-exact, never rebuilt from parts (invariant 1).
            photo = fetch_photo(row["image_url_raw"])
        except MediaError as exc:
            repaired["photos_unrecoverable"] += 1
            log.warning("cycle.orphan_photo_failed", job_id=job_id, error=str(exc)[:200])
            record_event(
                "cycle.orphan_photo_failed",
                job_id=job_id,
                level="error",
                reason=exc.reason,
            )
            continue

        obj = get_storage().put(
            job_key(job_id, f"source.{photo.extension}"),
            photo.data,
            content_type=photo.mime_type,
        )
        db.execute(
            """
            INSERT INTO assets (job_id, kind, s3_key, sha256, bytes,
                                mime_type, width, height)
            VALUES (%(job)s, 'source_photo', %(key)s, %(sha)s,
                    %(bytes)s, %(mime)s, %(w)s, %(h)s)
            ON CONFLICT (s3_key) DO NOTHING;
            """,
            {
                "job": job_id,
                "key": obj.key,
                "sha": obj.sha256,
                "bytes": obj.bytes,
                "mime": photo.mime_type,
                "w": photo.width,
                "h": photo.height,
            },
        )
        repaired["photos_restored"] += 1
        record_event("cycle.orphan_photo_restored", job_id=job_id, level="warning")

    if any(repaired.values()):
        log.warning("cycle.repaired_orphans", **repaired)
    return repaired


def _reopen_suppressed_deliveries() -> list[str]:
    """Re-deliver jobs that finished while DELIVERY_ENABLED was false.

    A suppressed delivery SUCCEEDS the stage. It has to - the video is made and
    published, and failing the stage would park a finished job as `failed`
    forever. It records `dry_run` and posts nothing.

    Which means the day delivery is switched on, every video made before that
    day is already marked delivered and will never be sent. The job is
    `completed`, so nothing schedules it; the input hash covers the CDN url and
    the phone, and neither changed. Without this the backlog is silently
    stranded, and the failure looks like nothing at all.

    Only runs when delivery is enabled. The stage it re-runs is free, and the
    client stores {phone, videoLink} idempotently (invariant 14) - so a repeat
    post is harmless where a missed one is a video nobody ever gets.
    """
    if not get_settings().delivery_enabled:
        return []

    from .common.errors import PipelineError
    from .orchestrator import redo_stage
    from .stages.base import PipelineStage

    rows = db.fetch_all(
        """
        SELECT DISTINCT ON (job_id) job_id
          FROM stage_runs
         WHERE stage = 'deliver' AND status = 'succeeded'
           AND params -> 'result' -> 'meta' ->> 'dry_run' = 'true'
         ORDER BY job_id, finished_at DESC;
        """
    )
    reopened: list[str] = []
    for row in rows:
        job_id = str(row["job_id"])
        try:
            redo_stage(job_id, PipelineStage.DELIVER)
        except PipelineError as exc:
            # A deliver run is already in flight for this job. Nothing to fix -
            # the next cycle sees it either delivered or failed.
            log.warning("cycle.redeliver_skipped", job_id=job_id, reason=str(exc)[:200])
            continue
        reopened.append(job_id)

    if reopened:
        log.warning("cycle.redelivering", jobs=len(reopened))
        record_event("cycle.redelivering", level="warning", jobs=len(reopened))
    return reopened


def run_cycle(
    *,
    lookback_days: int = 1,
    deadline_minutes: int = 480,
    interval_s: int = 20,
    fetch: bool = True,
) -> dict[str, Any]:
    """Pull, then work until everything is terminal or the deadline is reached.

    Every cycle that takes the lock ends in exactly one durable event:
    `cycle.done`, `cycle.interrupted` or `cycle.db_lost`. `cycle.deadline`
    comes before a `cycle.done`. So a `cycle.start` with nothing after it
    means the process was killed outright.
    """
    started_at = datetime.now(UTC)
    clock_start = time.monotonic()

    summary: dict[str, Any] = {
        "started_at": started_at,
        "skipped": False,
        "deadline_hit": False,
        "passes": 0,
        "stage_runs": {},
    }

    with _stop_on_sigterm() as stop, db.advisory_lock(CYCLE_LOCK_KEY) as acquired:
        if not acquired:
            # Not an error. The previous cycle is still rendering, which on a
            # large batch is the expected state, and the work it is doing is
            # the same work this one would have picked up.
            log.warning("cycle.already_running")
            record_event("cycle.skipped", level="warning", reason="lock_held")
            return {**summary, "skipped": True, "reason": "another cycle holds the lock"}

        try:
            _work(summary, stop=stop, lookback_days=lookback_days,
                  deadline_minutes=deadline_minutes, interval_s=interval_s,
                  fetch=fetch, clock_start=clock_start)
        except DB_LOST_ERRORS as exc:
            # The cycle cannot do anything useful without its database, and
            # nothing is lost by stopping here. Readiness is recomputed from
            # `stage_runs`, a render already submitted keeps going at the
            # vendor, and the reaper returns a stranded claim to the queue.
            # What matters is that the failure is named, here and in the exit
            # code, and not left to the unlock to report as something else.
            summary["db_lost"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            summary["elapsed_s"] = round(time.monotonic() - clock_start, 1)
            # The journal is the record that survives here. The event insert
            # is a best effort, and it lands only if the pool has a working
            # connection again by now. record_event swallows its own failure.
            record_event(
                "cycle.db_lost",
                level="error",
                error=summary["db_lost"],
                passes=summary["passes"],
                elapsed_s=summary["elapsed_s"],
            )

    return summary


def _work(
    summary: dict[str, Any],
    *,
    stop: threading.Event,
    lookback_days: int,
    deadline_minutes: int,
    interval_s: int,
    fetch: bool,
    clock_start: float,
) -> None:
    """Everything the cycle does while it holds the lock. Fills in `summary`.

    A tick loop, not a sequence of drain-everything passes. Each tick: reap,
    poll what is in flight, then advance every quick stage, downstream first.
    The card burn runs on its own thread (`CompositeWorker`), so a 35-second
    ffmpeg never holds up a poll or a delivery, and upstream work gets a
    bounded slice of each tick so a 500-job intake cannot either.

    It replaced passes that drained each stage of `DEFAULT_PLAN` to empty in
    turn and polled once at the end. On Batch 2 that meant a render finishing
    a minute into a pass waited behind the whole card-burn backlog before
    anything looked at it again, and a finished video waited up to an hour
    for its delivery.
    """
    from .intake.runner import run_intake
    from .orchestrator import _checked_mode, poll_once, reap

    deadline = clock_start + deadline_minutes * 60

    # Before intake, not at the first schedule: a real cycle over rehearsal
    # data would otherwise pull and dedupe first, and only then refuse.
    summary["stage_mode"] = _checked_mode()
    record_event("cycle.start", deadline_minutes=deadline_minutes, fetch=fetch,
                 stage_mode=summary["stage_mode"])

    if fetch:
        from_date, to_date = intake_window(lookback_days)
        summary["window"] = [str(from_date), str(to_date)]
        try:
            counters = run_intake(from_date=from_date, to_date=to_date)
            summary["intake"] = counters.as_dict()
        except DB_LOST_ERRORS:
            # Not the export API's fault. Let the cycle boundary report it
            # as what it is.
            raise
        except Exception as exc:  # noqa: BLE001 - the cycle is the boundary
            # A dead export API is not a reason to abandon the jobs already
            # in the database. Yesterday's renders still need finishing;
            # the non-zero exit is what makes the failed pull visible.
            summary["intake_error"] = str(exc)[:500]
            log.error("cycle.intake_failed", error=str(exc))
            record_event("cycle.intake_failed", level="error", error=str(exc)[:500])

    summary["repaired"] = _repair_orphans()
    summary["redelivering"] = len(_reopen_suppressed_deliveries())
    summary["scheduled"] = _schedule_all()

    totals: dict[str, int] = summary["stage_runs"]
    worker = CompositeWorker(stop)
    worker.start()
    try:
        while not stop.is_set():
            summary["passes"] += 1
            worker.raise_if_failed()
            reap()

            completed = poll_once()
            if completed:
                totals["poll"] = totals.get("poll", 0) + completed
            if stop.is_set():
                break

            # Finished renders first: a video that came back this tick is
            # carded by the worker and then checked, published and delivered
            # here, without waiting behind anybody else's upstream work.
            _drain_all(DOWNSTREAM_STAGES, totals, stop)
            _drain_all(RENDER_STAGES, totals, stop)
            more_upstream = _drain_for(UPSTREAM_STAGES, UPSTREAM_SLICE_S, totals, stop)
            if stop.is_set():
                break

            # Read the worker's state BEFORE the queue and again after. The
            # composite stage enqueues `checks` in its own `finally`, so a
            # queue that reads empty while the worker is mid-run is not empty.
            calls_before, busy_before = worker.calls, worker.busy
            left = outstanding_work()
            log.info("cycle.pass", pass_no=summary["passes"],
                     composites=worker.done, **{
                         k: v for k, v in left.items() if k != "next_due"
                     })
            if (
                left["total"] == 0
                and not busy_before
                and not worker.busy
                and worker.calls == calls_before
            ):
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Not a failure of any job — everything is still queued and the
                # next cycle resumes it. It IS a signal that a batch is taking
                # longer than the gap between runs, which is worth waking up for.
                summary["deadline_hit"] = True
                summary["outstanding"] = {k: left[k] for k in ("queued", "in_flight")}
                log.warning("cycle.deadline", **summary["outstanding"])
                record_event("cycle.deadline", level="warning", **summary["outstanding"])
                break

            if more_upstream:
                # The upstream slice ran out of time, not out of work. Go round
                # again at once: the poll and the downstream drain come first
                # on every tick, so nothing finished waits behind this.
                continue

            wait = next_sleep(left, interval_s, remaining)
            if worker.busy or left["queued"]:
                # A composite finishing enqueues `checks`; do not sit out a
                # long retry backoff while a finished video waits for it.
                wait = min(wait, float(interval_s))
            # An Event wait and not time.sleep, so that a SIGTERM ends the wait
            # at once and not up to MAX_SLEEP_S later.
            stop.wait(wait)
    finally:
        worker.halt()
    worker.raise_if_failed()
    summary["composites"] = worker.done

    summary["elapsed_s"] = round(time.monotonic() - clock_start, 1)
    if stop.is_set():
        # Something stopped the cycle from outside: `systemctl stop`, `docker
        # stop`, or a docker restart during an upgrade. Nothing is lost, for
        # the same reasons a deadline loses nothing. The event is what tells a
        # stopped batch apart from a finished one.
        summary["interrupted"] = "SIGTERM"
        record_event(
            "cycle.interrupted",
            level="warning",
            signal="SIGTERM",
            passes=summary["passes"],
            elapsed_s=summary["elapsed_s"],
        )
        return

    summary["jobs"] = {
        r["status"]: r["n"]
        for r in db.fetch_all("SELECT status, count(*) AS n FROM jobs GROUP BY status;")
    }
    log.info("cycle.done", **{k: summary[k] for k in ("passes", "elapsed_s", "jobs")})
    record_event(
        "cycle.done",
        passes=summary["passes"],
        elapsed_s=summary["elapsed_s"],
        deadline_hit=summary["deadline_hit"],
    )
