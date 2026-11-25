"""The vendor timeout times the CURRENT submit, never the first one.

`stage_runs.started_at` is the clock `_fail_if_overdue` reads, and a retry
re-submits on the same row. It used to be set with
`coalesce(started_at, now())`, so it kept the first attempt's time: once a
run had been retrying provider 500s for 2 hours, its next render was failed
as `VENDOR_TIMEOUT` - terminal - on the first poll, 25 seconds after it was
submitted. Two regenerations were lost that way on 2026-11-24.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from castrol_pipeline import orchestrator
from castrol_pipeline.common import db
from castrol_pipeline.stages.base import PipelineStage


def _mark_running_sql(monkeypatch) -> str:
    seen: list[str] = []
    monkeypatch.setattr(db, "execute", lambda sql, params=None: seen.append(sql))
    db.mark_running("run-1", vendor="v", vendor_task_id="t", model_id="m")
    assert len(seen) == 1
    return " ".join(seen[0].split())


def test_every_submit_restarts_the_clock(monkeypatch):
    sql = _mark_running_sql(monkeypatch)
    assert re.search(r"started_at = now\(\)", sql), sql


def test_the_first_attempts_time_is_not_kept(monkeypatch):
    sql = _mark_running_sql(monkeypatch)
    assert "coalesce(started_at" not in sql.lower(), sql


def _overdue(monkeypatch, started_at: datetime) -> list[str]:
    failed: list[str] = []
    monkeypatch.setattr(orchestrator.db, "mark_failed",
                        lambda run_id, **kw: failed.append(kw["error_code"]))
    monkeypatch.setattr(orchestrator, "record_event", lambda *a, **k: None)
    run = {"id": "run-1", "job_id": "job-1", "vendor": "v",
           "vendor_task_id": "t", "started_at": started_at}
    orchestrator._fail_if_overdue(run, PipelineStage.VIDEO)
    return failed


def test_a_fresh_submit_is_not_overdue(monkeypatch):
    # What the 9th attempt looks like once the clock restarts at its submit.
    assert _overdue(monkeypatch, datetime.now(UTC) - timedelta(seconds=25)) == []


def test_a_render_past_the_limit_still_times_out(monkeypatch):
    limit = orchestrator.get_settings().vendor_task_timeout_s
    old = datetime.now(UTC) - timedelta(seconds=limit + 60)
    assert _overdue(monkeypatch, old) == ["VENDOR_TIMEOUT"]
