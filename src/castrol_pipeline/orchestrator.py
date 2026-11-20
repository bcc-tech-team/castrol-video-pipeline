"""The orchestrator: readiness, claiming, retries, idempotency, job state.

This module owns the two things stages are forbidden to touch:
  * writes to `jobs`
  * the retry decision

Stage readiness is COMPUTED from `stage_runs`, never read from
`jobs.current_stage` — that column is a display convenience for the panel and
is never used for routing. Recomputing it is what makes a killed batch resumable
without bookkeeping.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from psycopg.types.json import Jsonb

from .common import budget, db, health, pacing
from .common.errors import (
    TERMINAL_STAGE_ERRORS,
    PipelineError,
    StageErrorCode,
    VendorOutOfCredits,
    VendorThrottled,
)
from .common.events import record_event
from .common.logging import get_logger, job_context
from .config import get_settings
from .stages.base import (
    DEFAULT_PLAN,
    STAGE_DEPENDENCIES,
    AsyncSubmission,
    JobContext,
    PipelineStage,
    StageResult,
)

log = get_logger(__name__)


def worker_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def get_stage_registry() -> dict[PipelineStage, Any]:
    """Which implementations are in play.

    Stub mode is what makes the orchestrator provable without spending money,
    and it stays: the DAG, retries and idempotency are still exercised without
    a provider account. USE_STUB_STAGES is the only difference between a dry
    run and a real one.
    """
    mode = _checked_mode()
    if mode == "stub":
        from .stages.stubs import STUB_STAGES

        return STUB_STAGES
    if mode == "mock":
        from .stages.mocks import MOCK_STAGES

        return MOCK_STAGES

    from .stages.real import REAL_STAGES

    return REAL_STAGES


_MODE_CHECKED: str | None = None


def _checked_mode() -> str:
    """`dryrun.check_mode_is_safe`, once per process.

    Every path that executes a stage comes through the registry, so this is the
    one place the rehearsal/real guard cannot be walked around. Once, because
    the registry is consulted per job and the answer cannot change mid-process.
    """
    global _MODE_CHECKED
    if _MODE_CHECKED is None:
        from .dryrun import check_mode_is_safe

        _MODE_CHECKED = check_mode_is_safe()
    return _MODE_CHECKED


# --------------------------------------------------------------- retry policy --

#: Exponential backoff between attempts, in seconds, capped so a stuck vendor
#: does not push a retry past the nightly window.
RETRY_BASE_S = 30
RETRY_CAP_S = 900


def backoff_seconds(attempts: int) -> int:
    return min(RETRY_BASE_S * (2 ** max(attempts - 1, 0)), RETRY_CAP_S)


def retry_delay_for(attempts: int, code: StageErrorCode) -> int | None:
    """None means terminal: do not schedule another attempt.

    Which codes are terminal is decided in ONE place, `TERMINAL_STAGE_ERRORS`,
    and that set carries the reasoning for each. It used to be restated here as
    a bare `code == BUDGET_EXHAUSTED`, which is how VENDOR_REJECTED came to be
    retried three times while both the invariant and the raising code said it
    must not be - the set existed and nothing read it.
    """
    if code in TERMINAL_STAGE_ERRORS:
        return None
    if code == StageErrorCode.VENDOR_UNAVAILABLE:
        return unavailable_delay(attempts)
    if attempts >= get_settings().stage_max_attempts:
        return None
    return backoff_seconds(attempts)


def unavailable_backoff() -> list[int]:
    """The VENDOR_UNAVAILABLE schedule, in seconds per attempt. The last repeats."""
    raw = get_settings().unavailable_backoff_s
    steps = [int(float(x)) for x in raw.split(",") if x.strip()]
    if not steps or any(s <= 0 for s in steps):
        raise ValueError(f"UNAVAILABLE_BACKOFF_S must be positive seconds: {raw!r}")
    return steps


def unavailable_delay(attempts: int) -> int | None:
    """Backoff for a provider-side 500, on its own long budget.

    Batches 1 and 2: 46% of renders came back "500 internal error, please try
    again later", in windows, and every affected job rendered in the end - but
    only after four rounds of `castrol retry-failed` by hand, because three
    attempts at 30s and 60s are over inside two minutes and these windows last
    hours. 1, 2, 4, 8, 15 then 30 minutes across 12 attempts rides out 3.5
    hours of backoff, plus the renders themselves. A failed render refunds,
    so the money is not at stake; the reservations against `vendor_usage`
    are, which is why it is still bounded.
    """
    if attempts >= get_settings().unavailable_max_attempts:
        return None
    steps = unavailable_backoff()
    return steps[min(max(attempts - 1, 0), len(steps) - 1)]


# ------------------------------------------------------------------ context --


def _result_of(run: dict[str, Any]) -> dict[str, Any]:
    params = run.get("params") or {}
    return params.get("result", {}) if isinstance(params, dict) else {}


def succeeded_runs(job_id: str) -> dict[str, dict[str, Any]]:
    """The latest succeeded run per stage, as {stage: result-payload}."""
    rows = db.fetch_all(
        """
        SELECT DISTINCT ON (stage) stage, output_key, params, finished_at
          FROM stage_runs
         WHERE job_id = %(job_id)s AND status = 'succeeded'
         ORDER BY stage, finished_at DESC;
        """,
        {"job_id": job_id},
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(_result_of(row))
        payload.setdefault("output_key", row["output_key"])
        out[str(row["stage"])] = payload
    return out


def load_context(job_id: str) -> JobContext:
    row = db.fetch_one(
        """
        SELECT j.id AS job_id, j.submission_id, j.script_version, j.voice_id,
               j.plate_id, j.language,
               s.raw, s.phone_e164, s.user_name, s.workshop_name,
               s.address_normalized, s.image_url_raw,
               s.background_choice, s.outfit_choice
          FROM jobs j
          JOIN submissions s ON s.id = j.submission_id
         WHERE j.id = %(job_id)s;
        """,
        {"job_id": job_id},
    )
    if row is None:
        raise LookupError(f"No job {job_id}")

    return JobContext(
        job_id=str(row["job_id"]),
        submission_id=str(row["submission_id"]),
        script_version=row["script_version"],
        voice_id=row["voice_id"],
        plate_id=str(row["plate_id"]) if row["plate_id"] else None,
        language=row["language"],
        user_name=row["user_name"] or "",
        workshop_name=row["workshop_name"] or "",
        # address_normalized IS the spoken form. It used to be split on ", "
        # here and rejoined in PrepStage, a round-trip that existed only to
        # satisfy a two-part address rule that no longer applies.
        spoken_place=row["address_normalized"] or "",
        phone_e164=row["phone_e164"] or "",
        uniform_id=row["outfit_choice"] or "",
        background_id=row["background_choice"] or "",
        image_url_raw=row["image_url_raw"] or "",
        raw=row["raw"] or {},
        upstream=succeeded_runs(str(row["job_id"])),
    )


# ---------------------------------------------------------------- readiness --


def ready_stages(job_id: str, done: dict[str, dict[str, Any]] | None = None) -> list[PipelineStage]:
    """Stages whose dependencies have all succeeded and which are not done."""
    done = succeeded_runs(job_id) if done is None else done
    ready: list[PipelineStage] = []
    for stage in DEFAULT_PLAN:
        if str(stage) in done:
            continue
        if all(str(dep) in done for dep in STAGE_DEPENDENCIES[stage]):
            ready.append(stage)
    return ready


def latest_runs(job_id: str) -> dict[str, dict[str, Any]]:
    """The newest run per stage, whatever its status. `skipped` is history, not state.

    `lifetime_attempts` is the attempts summed over every run of that stage at
    the newest run's inputs - what the auto-release cap counts. Per input hash,
    so a changed prompt starts a fresh budget, exactly as it releases the hold.
    """
    rows = db.fetch_all(
        """
        SELECT DISTINCT ON (stage) id, stage, status, input_hash, error_code,
               finished_at, lifetime_attempts
          FROM (SELECT *, sum(attempts) OVER (PARTITION BY stage, input_hash)
                          AS lifetime_attempts
                  FROM stage_runs
                 WHERE job_id = %(job_id)s AND status <> 'skipped') runs
         ORDER BY stage, created_at DESC;
        """,
        {"job_id": job_id},
    )
    return {str(r["stage"]): r for r in rows}


#: What `failure_decision` can say about a stage's latest run.
NO_FAILURE = "none"      # nothing failed, or the failure is about other inputs
HOLD = "hold"            # a terminal failure stands
RELEASE = "release"      # a provider-outage failure has cooled off: try again
CAPPED = "capped"        # a provider-outage failure past its lifetime cap


def failure_decision(
    latest: dict[str, Any] | None, input_hash: str, *, now: datetime | None = None
) -> str:
    """`failure_holds`, with the reason. Pure.

    VENDOR_UNAVAILABLE is the one failure that releases ITSELF. It is the
    provider failing on its own side, not a verdict on our inputs, and every
    job Batches 1 and 2 lost to it rendered in the end - after four rounds of
    `castrol retry-failed` by hand. Once its own 12 attempts are spent the
    failure holds for `UNAVAILABLE_RELEASE_AFTER_S`, then releases with a fresh
    run, until the attempts summed over every run at these inputs reach
    `UNAVAILABLE_LIFETIME_ATTEMPTS` (24). Past that it is CAPPED: held for a
    person, and `release_due_failures` raises it.
    """
    if latest is None or str(latest["status"]) != "failed":
        return NO_FAILURE
    if latest["input_hash"] != input_hash:
        return NO_FAILURE
    now = now or datetime.now(UTC)
    s = get_settings()
    code = latest["error_code"]
    finished = latest["finished_at"]
    if code == str(StageErrorCode.BUDGET_EXHAUSTED):
        if finished is None:
            return HOLD
        tz = ZoneInfo(s.export_timezone)
        today = now.astimezone(tz).date()
        return HOLD if finished.astimezone(tz).date() >= today else NO_FAILURE
    if code == str(StageErrorCode.VENDOR_UNAVAILABLE):
        lifetime = int(latest.get("lifetime_attempts") or 0)
        if lifetime >= s.unavailable_lifetime_attempts:
            return CAPPED
        if finished is None:
            return HOLD
        if (now - finished).total_seconds() < s.unavailable_release_after_s:
            return HOLD
        return RELEASE
    return HOLD


def failure_holds(
    latest: dict[str, Any] | None, input_hash: str, *, now: datetime | None = None
) -> bool:
    """Does a terminal failure still stand, so the stage must NOT be re-enqueued?

    The in-flight unique index only covers pending/claimed/running, so without
    this a terminal `failed` row does not stop `schedule_ready` from inserting a
    fresh `pending` one with attempts back at zero — and `execute_one` calls
    `schedule_ready` in its `finally`, straight after marking the failure. A
    content-safety rejection would resubmit the same inputs forever inside one
    `drain_stage`; a BUDGET_EXHAUSTED stop would hot-loop against the cap. That
    is the opposite of what `retry_delay_for` returning None is meant to say.

    A failure stops holding when:
      * the inputs changed — a new prompt or plate is a different question
      * it was BUDGET_EXHAUSTED on an earlier IST day — the cap is per day
        (`vendor_usage.usage_date`), so the stop is for the day, not forever
      * it was VENDOR_UNAVAILABLE, has cooled off, and is under its lifetime
        attempt cap (`failure_decision`)

    Everything else waits for a person: `castrol run <ref> --retry`.
    """
    return failure_decision(latest, input_hash, now=now) in (HOLD, CAPPED)


def schedule_ready(job_id: str, *, retry_failed: bool = False) -> list[PipelineStage]:
    """Enqueue a pending run for every ready stage. Idempotent.

    Two guards make double-work impossible rather than merely unlikely:
      * a stage whose exact input_hash already succeeded is skipped outright
      * the partial unique index rejects a second in-flight run per (job, stage)

    And one makes a terminal failure terminal (`failure_holds`). `retry_failed`
    overrides it — that is a person deciding to pay for another attempt.
    """
    registry = get_stage_registry()
    ctx = load_context(job_id)
    latest = latest_runs(job_id)
    enqueued: list[PipelineStage] = []

    for stage in ready_stages(job_id, ctx.upstream):
        impl = registry.get(stage)
        if impl is None:
            continue
        input_hash = impl.input_hash(ctx)

        if db.find_succeeded_run(job_id, str(stage), input_hash):
            # Already done at this exact input. Nothing to do — this is what
            # makes a re-run after a code change redo only what actually moved.
            continue

        prior = latest.get(str(stage))
        decision = failure_decision(prior, input_hash)
        if not retry_failed and decision in (HOLD, CAPPED):
            continue

        if db.enqueue_stage_run(job_id, str(stage), input_hash):
            enqueued.append(stage)
            if decision == RELEASE and not retry_failed and prior is not None:
                # Durable, because a self-healing system that heals silently
                # is one nobody can tell from a lucky one.
                record_event(
                    "job.auto_retry",
                    job_id=job_id,
                    stage=str(stage),
                    stage_run_id=str(prior["id"]),
                    level="warning",
                    error_code=prior["error_code"],
                    lifetime_attempts=int(prior.get("lifetime_attempts") or 0),
                    lifetime_cap=get_settings().unavailable_lifetime_attempts,
                )

    if enqueued:
        log.info("orchestrator.scheduled", job_id=job_id, stages=[str(s) for s in enqueued])
    return enqueued


# ------------------------------------------------------------- persistence --


def _persist_result(ctx: JobContext, run: dict[str, Any], result: StageResult) -> None:
    payload = {k: v for k, v in asdict(result).items() if v is not None}
    payload.pop("asset_kind", None)
    # Decimals do not survive JSON. Cost lives in its own typed columns; this
    # copy is only for the params blob, so stringifying it loses nothing.
    for money in ("cost_usd", "billed_seconds", "billed_units"):
        if money in payload:
            payload[money] = str(payload[money])

    if result.asset_kind and result.output_key and result.sha256:
        db.execute(
            """
            INSERT INTO assets (job_id, kind, s3_key, sha256, bytes,
                                duration_ms, width, height, cdn_url, meta)
            VALUES (%(job_id)s, %(kind)s, %(key)s, %(sha)s, %(bytes)s,
                    %(dur)s, %(w)s, %(h)s, %(cdn)s, %(meta)s)
            ON CONFLICT (s3_key) DO NOTHING;
            """,
            {
                "job_id": ctx.job_id,
                "kind": str(result.asset_kind),
                "key": result.output_key,
                "sha": result.sha256,
                "bytes": result.bytes,
                "dur": result.duration_ms,
                "w": result.width,
                "h": result.height,
                "cdn": result.cdn_url,
                "meta": Jsonb(result.meta or {}),
            },
        )

    # Checks are written regardless of outcome, and are non-blocking this release.
    for check in (result.meta or {}).get("checks", []):
        db.execute(
            """
            INSERT INTO checks (job_id, check_name, passed, score, details)
            VALUES (%(job_id)s, %(name)s, %(passed)s, %(score)s, %(details)s);
            """,
            {
                "job_id": ctx.job_id,
                "name": check.get("check_name", "unnamed"),
                "passed": bool(check.get("passed")),
                "score": check.get("score"),
                "details": Jsonb(check.get("details", {})),
            },
        )

    db.mark_succeeded(
        str(run["id"]),
        output_key=result.output_key,
        params={"result": payload, "vendor_params": result.params or {}},
        cost_usd=result.cost_usd,
        billed_seconds=result.billed_seconds,
        billed_units=result.billed_units,
    )


def _record_delivery(ctx: JobContext, result: StageResult) -> None:
    cdn = (
        result.cdn_url
        or (result.meta or {}).get("stub_cdn_url")
        or (result.meta or {}).get("cdn_url")
    )
    if not cdn:
        return
    db.execute(
        """
        INSERT INTO deliveries (job_id, phone_e164, cdn_url, attempts, posted_at, response_code)
        VALUES (%(job_id)s, %(phone)s, %(cdn)s, %(attempts)s, %(posted)s, %(code)s)
        ON CONFLICT (job_id) DO UPDATE
           SET cdn_url = EXCLUDED.cdn_url,
               attempts = deliveries.attempts + 1,
               posted_at = EXCLUDED.posted_at,
               response_code = EXCLUDED.response_code;
        """,
        {
            "job_id": ctx.job_id,
            "phone": ctx.phone_e164,
            "cdn": cdn,
            "attempts": 1,
            "posted": (result.meta or {}).get("posted_at"),
            "code": (result.meta or {}).get("response_code"),
        },
    )


def advance_job(job_id: str) -> None:
    """Recompute job status from stage_runs. The only writer of `jobs`."""
    done = succeeded_runs(job_id)

    if str(PipelineStage.DELIVER) in done:
        db.execute(
            """
            UPDATE jobs SET status = 'completed', current_stage = 'deliver',
                            completed_at = coalesce(completed_at, now())
             WHERE id = %(id)s;
            """,
            {"id": job_id},
        )
        return

    # The LATEST run per stage decides, not any failed row ever. Once a person
    # retries a failed stage, the old row is history; judging by it would show
    # the job `failed` for the whole of the retry that is fixing it.
    dead = db.fetch_one(
        """
        SELECT stage, error_code, error_message
          FROM (SELECT DISTINCT ON (stage) stage, status, error_code,
                       error_message, finished_at
                  FROM stage_runs
                 WHERE job_id = %(id)s AND status <> 'skipped'
                 ORDER BY stage, created_at DESC) latest
         WHERE status = 'failed'
         ORDER BY finished_at DESC LIMIT 1;
        """,
        {"id": job_id},
    )
    if dead:
        # No partial delivery, ever: a job that did not complete every stage is
        # never POSTed. It sits here and in the panel, and nowhere else.
        db.execute(
            """
            UPDATE jobs SET status = 'failed',
                            failure_reason = %(reason)s
             WHERE id = %(id)s;
            """,
            {"id": job_id, "reason": f"{dead['stage']}: {dead['error_code']}"},
        )
        return

    nxt = ready_stages(job_id, done)
    db.execute(
        """
        UPDATE jobs SET status = 'running', current_stage = %(stage)s
         WHERE id = %(id)s AND status <> 'completed';
        """,
        {"id": job_id, "stage": str(nxt[0]) if nxt else str(PipelineStage.DELIVER)},
    )


# ------------------------------------------------------------------- redo --


def downstream_of(stage: PipelineStage) -> list[PipelineStage]:
    """Every stage that depends on `stage`, transitively, in plan order.

    Computed from STAGE_DEPENDENCIES rather than listed, so adding a stage to
    the DAG cannot leave a stale hand-written list behind.
    """
    affected = {stage}
    changed = True
    while changed:
        changed = False
        for candidate, deps in STAGE_DEPENDENCIES.items():
            if candidate not in affected and any(d in affected for d in deps):
                affected.add(candidate)
                changed = True
    return [s for s in DEFAULT_PLAN if s in affected and s != stage]


def failed_at_stage(
    stage: PipelineStage, codes: list[str] | None = None
) -> list[dict[str, Any]]:
    """Jobs whose LATEST run of `stage` is a terminal failure, oldest job first.

    Latest, not any: a job that failed once and then succeeded on a retry is not
    failed, and judging by any failed row would re-run a finished video.
    """
    return db.fetch_all(
        """
        SELECT job_id, error_code, error_message
          FROM (SELECT DISTINCT ON (sr.job_id) sr.job_id, sr.status, sr.error_code,
                       sr.error_message, j.created_at
                  FROM stage_runs sr JOIN jobs j ON j.id = sr.job_id
                 WHERE sr.stage = %(stage)s AND sr.status <> 'skipped'
                 ORDER BY sr.job_id, sr.created_at DESC) latest
         WHERE status = 'failed'
           AND (%(codes)s::text[] IS NULL OR error_code = ANY(%(codes)s::text[]))
         ORDER BY created_at, job_id;
        """,
        {"stage": str(stage), "codes": codes},
    )


def retry_failed_jobs(job_ids: list[str]) -> dict[str, Any]:
    """Put terminally failed jobs back in the queue, in bulk. Runs nothing.

    The same decision `castrol run <ref> --retry` makes for one job (invariant
    34), without driving each one to a finish: it re-enqueues and returns, and
    the next `cycle` renders and delivers them all at once. One-at-a-time
    `run --retry` was the only way back before this, and at ~4 runs per
    15-connection pool it put Batch 1's 193 provider failures ~15 hours away.

    `retry_failed=True` lets EVERY ready failed stage of these jobs re-enqueue,
    not only the one they were selected by - the same as `run --retry`.
    """
    requeued: list[str] = []
    for job_id in job_ids:
        if schedule_ready(job_id, retry_failed=True):
            advance_job(job_id)
            record_event("job.retry_failed", job_id=job_id, level="warning")
            requeued.append(job_id)
    return {"requested": len(job_ids), "requeued": len(requeued),
            "not_requeued": [j for j in job_ids if j not in set(requeued)]}


def redo_stage(job_id: str, stage: PipelineStage) -> dict[str, Any]:
    """Make `stage` runnable again for a finished job, and everything after it.

    Used when an input the hash cannot see has changed — a reworded prompt, a
    corrected model id, a plate reissued under the same key. It does NOT run
    anything; it clears the way and re-schedules.

    Succeeded runs are marked `skipped`, never deleted. Two reasons: the
    partial unique index only constrains `succeeded`, so demoting is enough to
    free the slot; and the row carries what that attempt COST, which is the one
    number a deleted row would take with it (invariant 24).

    Downstream stages are demoted too. They would re-run on their own once the
    new video lands — their input hashes cover the new output — but leaving
    them `succeeded` in the meantime makes the job look finished while it is
    not, and `advance_job` would mark it completed on the next sweep.
    """
    stages = [stage, *downstream_of(stage)]
    names = [str(s) for s in stages]

    inflight = db.fetch_one(
        """
        SELECT stage FROM stage_runs
         WHERE job_id = %(job)s AND stage::text = ANY(%(stages)s)
           AND status IN ('pending', 'claimed', 'running')
         LIMIT 1;
        """,
        {"job": job_id, "stages": names},
    )
    if inflight:
        raise PipelineError(
            f"A {inflight['stage']} run is already in flight for this job. "
            "Redoing now would submit a second paid call alongside it — wait "
            "for `castrol poll` to reconcile, or let it fail first."
        )

    demoted = db.execute(
        """
        UPDATE stage_runs
           SET status = 'skipped',
               error_code = 'SUPERSEDED_BY_REDO'
         WHERE job_id = %(job)s AND stage::text = ANY(%(stages)s) AND status = 'succeeded';
        """,
        {"job": job_id, "stages": names},
    )

    # The job is very likely `completed`, and `schedule` only looks at jobs that
    # are not. Reopen it or nothing will ever pick this up.
    db.execute(
        """
        UPDATE jobs SET status = 'running', current_stage = %(stage)s,
                        completed_at = NULL, failure_reason = NULL
         WHERE id = %(job)s;
        """,
        {"job": job_id, "stage": str(stage)},
    )

    enqueued = schedule_ready(job_id)
    record_event(
        "job.redo",
        job_id=job_id,
        stage=str(stage),
        level="warning",
        demoted_runs=demoted,
        cleared=names,
        enqueued=[str(s) for s in enqueued],
    )
    return {
        "job_id": job_id,
        "stage": str(stage),
        "cleared": names,
        "demoted_runs": demoted,
        "enqueued": [str(s) for s in enqueued],
    }


# ------------------------------------------------------------------ pacing --


def submit_blocked(stage: PipelineStage) -> str | None:
    """Why no new run of `stage` may be sent right now, or None if one may.

    Asked BEFORE claiming, so a blocked stage spends no attempt and reserves
    no budget. Three reasons, in order: the provider is out of credits, the
    provider is in an outage pause (`common/health.py`), or the in-flight cap
    (`pacing.in_flight_cap`) is full. While paused for credits this is also
    where the balance is re-checked, so every caller - the cycle, `castrol
    run`, `castrol work` - resumes by itself.
    """
    provider = pacing.STAGE_PROVIDER.get(stage)
    if provider is not None:
        h = health.health(provider)
        if h.balance_check_due():
            _recheck_balance(stage, h)
        reason = h.submit_block_reason()
        if reason:
            return reason
    cap = pacing.in_flight_cap(stage)
    if cap is not None and db.count_in_flight(str(stage)) >= cap:
        return f"{cap} {stage} runs already in flight"
    return None


# ----------------------------------------------------------------- credits --


def expected_render_spend() -> dict[str, Any]:
    """What the renders still owed would cost, at today's rate.

    Owed means: a job that is not finished, failed or cancelled and has no
    video run succeeded or in flight - a render in flight was paid for at
    submit (invariant 24). Priced from the job's probed audio where there is
    one, and from `RENDER_SECONDS_ESTIMATE` where the voice is not made yet.
    """
    s = get_settings()
    rows = db.fetch_all(
        """
        SELECT (SELECT a.duration_ms FROM assets a
                 WHERE a.job_id = j.id AND a.kind = 'audio'
                 ORDER BY a.created_at DESC LIMIT 1) AS audio_ms
          FROM jobs j
         WHERE j.status NOT IN ('completed', 'cancelled', 'failed')
           AND NOT EXISTS (SELECT 1 FROM stage_runs r
                            WHERE r.job_id = j.id AND r.stage = 'video'
                              AND r.status IN ('succeeded', 'running'));
        """
    )
    per_render = budget.video_cost_usd(s.render_seconds_estimate, pro=s.video_is_pro)
    total = sum(
        (budget.video_cost_usd(r["audio_ms"] / 1000, pro=s.video_is_pro)
         if r.get("audio_ms") else per_render)
        for r in rows
    )
    return {"renders": len(rows), "usd": float(total), "per_render_usd": float(per_render)}


def balance_usd(stage: PipelineStage = PipelineStage.VIDEO) -> float | None:
    """The provider's balance in USD, paced like any other request.

    None when the stage has no balance to read (stub mode, the image and
    voice providers) or when no request slot came free.
    """
    fn = getattr(get_stage_registry().get(stage), "balance_usd", None)
    if fn is None:
        return None
    gate = pacing.limiter_for(stage)
    if gate is not None and not gate.acquire(get_settings().provider_slot_wait_s):
        return None
    return float(fn())


def _recheck_balance(stage: PipelineStage, h: health.ProviderHealth) -> None:
    """Paused for credits: has a top-up landed that covers the work?

    "The work" is the renders owed, but no more than one full set of render
    slots: a top-up big enough to fill the in-flight cap resumes the batch
    rather than waiting for the whole of a 500-video queue to be covered. A
    queue the balance does not cover is the `credits_low` alert's business.
    A provider with no balance endpoint resumes on the recheck timer, and the
    next submit is the test.
    """
    try:
        balance = balance_usd(stage)
    except Exception as exc:  # noqa: BLE001 - an unreadable balance is "not yet"
        log.warning("provider.balance_unreadable", stage=str(stage), error=str(exc)[:300])
        h.balance_checked(covered=False, balance_usd=None, needed_usd=0.0)
        return
    if balance is None and getattr(get_stage_registry().get(stage), "balance_usd", None):
        return  # no request slot this time; still due, so the next ask retries
    if balance is None:
        h.balance_checked(covered=True, balance_usd=None, needed_usd=0.0)
        return
    spend = expected_render_spend()
    cap = pacing.in_flight_cap(stage) or spend["renders"] or 1
    needed = min(spend["usd"], cap * spend["per_render_usd"])
    needed = max(needed, spend["per_render_usd"])
    h.balance_checked(covered=balance >= needed, balance_usd=round(balance, 2),
                      needed_usd=round(needed, 2))


def credits_preflight() -> dict[str, Any] | None:
    """At cycle start: does the balance cover the renders queued?

    Recorded as a warning when it does not, so a batch that will run dry in
    the middle of the night says so before it starts rather than at the 402.
    None when there is no balance to read.
    """
    try:
        balance = balance_usd(PipelineStage.VIDEO)
    except Exception as exc:  # noqa: BLE001 - a preflight never stops a cycle
        log.warning("cycle.balance_unreadable", error=str(exc)[:300])
        return {"error": str(exc)[:300]}
    if balance is None:
        return None
    spend = expected_render_spend()
    out = {"balance_usd": round(balance, 2), "needed_usd": round(spend["usd"], 2),
           "renders": spend["renders"], "short": balance < spend["usd"]}
    if out["short"]:
        log.warning("cycle.credits_short", **out)
        record_event("cycle.credits_short", level="warning", **out)
    return out


# ----------------------------------------------------- self-releasing holds --


_RELEASABLE_SQL = """
SELECT job_id, stage, id, finished_at, lifetime_attempts
  FROM (SELECT DISTINCT ON (job_id, stage) job_id, stage, id, status, error_code,
               finished_at, lifetime_attempts
          FROM (SELECT *, sum(attempts) OVER (PARTITION BY job_id, stage, input_hash)
                          AS lifetime_attempts
                  FROM stage_runs
                 WHERE status <> 'skipped') runs
         ORDER BY job_id, stage, created_at DESC) latest
 WHERE status = 'failed' AND error_code = %(code)s;
"""


def unavailable_failures() -> list[dict[str, Any]]:
    """Every (job, stage) whose latest run is a terminal provider-500 failure."""
    return db.fetch_all(_RELEASABLE_SQL, {"code": str(StageErrorCode.VENDOR_UNAVAILABLE)})


def pending_releases(
    rows: list[dict[str, Any]] | None = None, *, now: datetime | None = None
) -> dict[str, Any]:
    """How many failures will release themselves later, and when the next one is.

    The cycle counts these as outstanding work, so it waits for them instead
    of ending and leaving them for the next timer twelve hours away.
    """
    s = get_settings()
    rows = unavailable_failures() if rows is None else rows
    due = [r["finished_at"] + timedelta(seconds=s.unavailable_release_after_s)
           for r in rows
           if int(r["lifetime_attempts"] or 0) < s.unavailable_lifetime_attempts
           and r["finished_at"] is not None]
    return {"count": len(due), "next_due": min(due) if due else None}


def release_due_failures(*, now: datetime | None = None) -> dict[str, int]:
    """Re-enqueue provider-500 failures that have cooled off; flag the capped.

    `schedule_ready` does the releasing (and records `job.auto_retry`), so a
    job is released here by exactly the rule that would release it at the
    next cycle start. A capped failure is recorded ONCE, as
    `job.attempt_cap_reached` against its run, for a person to look at.
    """
    now = now or datetime.now(UTC)
    s = get_settings()
    out = {"released": 0, "capped": 0}
    for row in unavailable_failures():
        job_id = str(row["job_id"])
        if int(row["lifetime_attempts"] or 0) >= s.unavailable_lifetime_attempts:
            seen = db.fetch_one(
                """
                SELECT 1 AS seen FROM job_events
                 WHERE stage_run_id = %(run)s AND event = 'job.attempt_cap_reached'
                 LIMIT 1;
                """,
                {"run": str(row["id"])},
            )
            if not seen:
                out["capped"] += 1
                record_event(
                    "job.attempt_cap_reached",
                    job_id=job_id,
                    stage=str(row["stage"]),
                    stage_run_id=str(row["id"]),
                    level="error",
                    lifetime_attempts=int(row["lifetime_attempts"]),
                    lifetime_cap=s.unavailable_lifetime_attempts,
                )
            continue
        finished = row["finished_at"]
        if finished is None or (now - finished).total_seconds() < s.unavailable_release_after_s:
            continue
        if schedule_ready(job_id):
            advance_job(job_id)
            out["released"] += 1
    return out


def _note_submit(stage: PipelineStage) -> None:
    provider = pacing.STAGE_PROVIDER.get(stage)
    if provider is not None:
        health.health(provider).note_submit()


def _record_outcome(stage: PipelineStage, *, failed: bool) -> None:
    """Feed a finished task to its provider's outage breaker."""
    provider = pacing.STAGE_PROVIDER.get(stage)
    if provider is not None:
        health.health(provider).record_render(failed=failed)


def _throttled(
    stage: PipelineStage, exc: BaseException, *, job_id: str, run_id: str, on: str
) -> None:
    """A provider said 429. Close its limiter and say so, once per refusal."""
    backoff = get_settings().throttle_backoff_s
    gate = pacing.limiter_for(stage)
    if gate is not None:
        gate.back_off(backoff)
    log.warning("vendor.throttled", stage=str(stage), on=on, backoff_s=backoff)
    record_event(
        "vendor.throttled",
        job_id=job_id,
        stage=str(stage),
        stage_run_id=run_id,
        level="warning",
        on=on,
        backoff_s=backoff,
        error=str(exc)[:300],
    )


# ------------------------------------------------------------- worker loop --


def execute_one(
    stage: PipelineStage, worker: str | None = None, *, job_id: str | None = None
) -> bool:
    """Claim and execute a single run of `stage`. False if the queue was empty.

    `job_id` restricts the claim to one job; see `jobrun.run_job`.
    """
    worker = worker or worker_identity()
    if submit_blocked(stage):
        # Not "empty", but nothing to do for now: the caller moves on and the
        # next tick looks again. Checked BEFORE the claim, so no attempt is
        # spent on a run that was never going to be sent.
        return False
    run = db.claim_stage_run(str(stage), worker, job_id)
    if run is None:
        return False

    job_id = str(run["job_id"])
    run_id = str(run["id"])
    impl = get_stage_registry()[stage]
    gate = pacing.limiter_for(stage)

    with job_context(job_id, stage=str(stage), run_id=run_id):
        log.info("stage.claimed", attempts=run["attempts"])
        try:
            if gate is not None and not gate.acquire(get_settings().provider_slot_wait_s):
                db.release_claim(run_id, delay_seconds=0,
                                 reason=f"{stage}: no request slot this tick")
                return False
            ctx = load_context(job_id)
            outcome = impl.run(ctx)

            if isinstance(outcome, AsyncSubmission):
                # Submit and release. The poller reconciles. A worker blocked on
                # a 4-minute vendor poll is a worker not doing the other 200 jobs.
                db.mark_running(
                    run_id,
                    vendor=outcome.vendor,
                    vendor_task_id=outcome.vendor_task_id,
                    model_id=outcome.model_id,
                    cost_usd=outcome.cost_usd,
                    billed_seconds=outcome.billed_seconds,
                    billed_units=outcome.billed_units,
                )
                log.info("stage.submitted", vendor_task_id=outcome.vendor_task_id)
                _note_submit(stage)
                return True

            _persist_result(ctx, run, outcome)
            if stage == PipelineStage.DELIVER:
                _record_delivery(ctx, outcome)
            log.info("stage.succeeded", output_key=outcome.output_key)

        except VendorThrottled as exc:
            # The provider refused the request before looking at it. Nothing
            # landed and nothing was billed, so the run goes straight back with
            # its attempt returned, and the provider gets a breather.
            _throttled(stage, exc, job_id=job_id, run_id=run_id, on="submit")
            db.release_claim(run_id, delay_seconds=get_settings().throttle_backoff_s,
                             reason=str(exc)[:500])
            return False

        except VendorOutOfCredits as exc:
            # Not this job's fault, and not something a retry fixes. Put the
            # run back untouched and stop sending this provider anything until
            # the balance covers the work again (`submit_blocked`).
            db.release_claim(run_id, delay_seconds=0, reason=str(exc)[:500])
            provider = pacing.STAGE_PROVIDER.get(stage)
            if provider is not None:
                health.health(provider).pause_for_credits(str(exc))
            return False

        except Exception as exc:  # noqa: BLE001 - the orchestrator is the boundary
            code = exc.code if isinstance(exc, PipelineError) else StageErrorCode.INTERNAL
            delay = retry_delay_for(int(run["attempts"]), code)
            db.mark_failed(
                run_id,
                error_code=str(code),
                error_message=str(exc)[:2000],
                retry_in_seconds=delay,
            )
            record_event(
                "stage.failed",
                job_id=job_id,
                stage=str(stage),
                stage_run_id=run_id,
                level="error",
                error_code=str(code),
                attempts=run["attempts"],
                retry_in_seconds=delay,
                error=str(exc)[:500],
            )
        finally:
            schedule_ready(job_id)
            advance_job(job_id)

    return True


def drain_stage(stage: PipelineStage, *, limit: int = 1000, job_id: str | None = None) -> int:
    processed = 0
    while processed < limit and execute_one(stage, job_id=job_id):
        processed += 1
    return processed


# ------------------------------------------------------------------ poller --


def _fail_if_overdue(run: dict[str, Any], stage: PipelineStage) -> None:
    """Fail an in-flight run that has outlived the vendor timeout.

    Terminal, not retried. The submit already spent the money; retrying an
    overdue task pays for it a second time on the chance the first one was
    merely slow, which is the wrong trade on a step that costs a dollar.
    """
    started = run.get("started_at")
    if started is None:
        return
    age = (datetime.now(UTC) - started).total_seconds()
    limit = get_settings().vendor_task_timeout_s
    if age <= limit:
        return

    db.mark_failed(
        str(run["id"]),
        error_code=str(StageErrorCode.VENDOR_TIMEOUT),
        error_message=(
            f"{run.get('vendor')} task {run.get('vendor_task_id')} was still "
            f"in flight after {age / 60:.0f} min (limit {limit / 60:.0f} min). "
            "The submit was already billed."
        ),
        retry_in_seconds=None,
    )
    record_event(
        "vendor.task_timeout",
        job_id=str(run["job_id"]),
        stage=str(stage),
        stage_run_id=str(run["id"]),
        level="error",
        vendor_task_id=str(run.get("vendor_task_id")),
        minutes=round(age / 60, 1),
    )


#: Failures of OUR connection while polling or fetching a result, as opposed to
#: a verdict from the vendor. `httpx.TransportError` covers connect / read /
#: write / protocol errors and timeouts; ConnectionError and TimeoutError catch
#: the same thing raised raw from a socket (WinError 10054 is a
#: ConnectionResetError). A vendor rejection is a PipelineError and is NOT here.
_TRANSIENT_POLL_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
)


#: When each in-flight run was last polled, by run id, in this process.
#: In memory on purpose: it only spaces out polls, and a restart that forgets
#: it costs one extra poll per task.
_LAST_POLLED: dict[str, float] = {}
_LAST_POLLED_LOCK = threading.Lock()


def _poll_interval(stage: PipelineStage) -> float:
    s = get_settings()
    return {
        PipelineStage.VIDEO: s.video_poll_interval_s,
        PipelineStage.IMAGE: s.image_poll_interval_s,
    }.get(stage, 0.0)


def poll_order(
    rows: list[dict[str, Any]],
    last_polled: dict[str, float],
    now: float,
    interval_for: Any = _poll_interval,
) -> list[dict[str, Any]]:
    """Which in-flight runs to poll this pass, least recently polled first. Pure.

    A run polled within its stage's interval is left out. Never-polled runs
    come first, oldest submit first. The old order was `started_at` with a
    LIMIT, which at more than `limit` in flight never polled the newest ones.
    """
    due = []
    for run in rows:
        rid = str(run["id"])
        seen = last_polled.get(rid)
        stage = PipelineStage(str(run["stage"]))
        if seen is not None and now - seen < interval_for(stage):
            continue
        due.append((seen if seen is not None else float("-inf"), run))
    never = datetime.min.replace(tzinfo=UTC)
    due.sort(key=lambda pair: (pair[0], pair[1].get("started_at") or never))
    return [run for _, run in due]


def poll_once(*, limit: int = 100, job_id: str | None = None) -> int:
    """Reconcile in-flight async vendor tasks. Returns how many completed.

    Paced: each poll takes a request slot from its provider's limiter, and a
    provider whose slots run out, or that answers 429, is left alone for the
    rest of the pass. `limit` bounds the polls made, not the rows read.
    """
    registry = get_stage_registry()
    rows = db.fetch_all(
        """
        SELECT * FROM stage_runs
         WHERE status = 'running' AND vendor_task_id IS NOT NULL
           AND (%(job_id)s::uuid IS NULL OR job_id = %(job_id)s::uuid)
         ORDER BY started_at;
        """,
        {"job_id": job_id},
    )
    with _LAST_POLLED_LOCK:
        if job_id is None:
            # Forget runs no longer in flight. Only on an unfiltered read: one
            # job's rows say nothing about whether anyone else's are live.
            live = {str(r["id"]) for r in rows}
            for rid in [k for k in _LAST_POLLED if k not in live]:
                del _LAST_POLLED[rid]
        ordered = poll_order(rows, dict(_LAST_POLLED), time.monotonic())

    completed = 0
    polled = 0
    exhausted: set[PipelineStage] = set()
    slot_wait = get_settings().provider_slot_wait_s
    for run in ordered:
        if polled >= limit:
            break
        stage = PipelineStage(str(run["stage"]))
        impl = registry.get(stage)
        if impl is None or not getattr(impl, "is_async", False) or stage in exhausted:
            continue
        gate = pacing.limiter_for(stage)
        if gate is not None and not gate.acquire(slot_wait):
            exhausted.add(stage)
            continue
        polled += 1
        with _LAST_POLLED_LOCK:
            _LAST_POLLED[str(run["id"])] = time.monotonic()

        job_id = str(run["job_id"])
        with job_context(job_id, stage=str(stage), run_id=str(run["id"])):
            try:
                ctx = load_context(job_id)
                result = impl.poll(str(run["vendor_task_id"]), ctx)
                if result is None:
                    # Still in flight — unless it has been in flight too long.
                    # The poller is stateless per pass, so without this a task
                    # the vendor silently dropped stays `running` forever: the
                    # job never fails, never completes, and never reports.
                    _fail_if_overdue(run, stage)
                    continue
                _persist_result(ctx, run, result)
                _record_outcome(stage, failed=False)
                completed += 1
                log.info("stage.polled_complete", output_key=result.output_key)
            except VendorThrottled as exc:
                # Not an answer about the task, which is untouched. Leave it
                # `running` and stop asking this provider for the rest of the
                # pass; the overdue check still bounds a task that never
                # comes back.
                exhausted.add(stage)
                _throttled(stage, exc, job_id=job_id, run_id=str(run["id"]), on="poll")
                _fail_if_overdue(run, stage)
            except _TRANSIENT_POLL_ERRORS as exc:
                # The vendor task is untouched by OUR connection dropping, so
                # the run stays `running` and the next pass polls the same
                # task id again. Failing it here is what cost real money: the
                # retry path re-SUBMITS, so a render the vendor had already
                # finished and billed was thrown away and paid for again -
                # twice on one job on 2026-09-22, both times a WinError 10054
                # while fetching a completed result. Invariant 13: reconcile,
                # never resubmit something that may have landed. The overdue
                # check still bounds a task that never comes back.
                record_event(
                    "stage.poll_transient",
                    job_id=job_id,
                    stage=str(stage),
                    stage_run_id=str(run["id"]),
                    level="warning",
                    error=str(exc)[:500],
                )
                _fail_if_overdue(run, stage)
            except Exception as exc:  # noqa: BLE001
                code = exc.code if isinstance(exc, PipelineError) else StageErrorCode.INTERNAL
                if code == StageErrorCode.VENDOR_UNAVAILABLE:
                    _record_outcome(stage, failed=True)
                delay = retry_delay_for(int(run["attempts"]), code)
                db.mark_failed(
                    str(run["id"]),
                    error_code=str(code),
                    error_message=str(exc)[:2000],
                    retry_in_seconds=delay,
                )
                record_event(
                    "stage.poll_failed",
                    job_id=job_id,
                    stage=str(stage),
                    stage_run_id=str(run["id"]),
                    level="error",
                    error_code=str(code),
                    error=str(exc)[:500],
                )
            finally:
                schedule_ready(job_id)
                advance_job(job_id)

    return completed


def reap() -> int:
    """Return stuck claims to the queue. Without this a worker OOM parks a job."""
    n = db.reap_stuck_claims(get_settings().stage_claim_timeout_s)
    if n:
        log.warning("orchestrator.reaped", rows=n)
    return n
