"""Alerts: the few things a person has to act on, said once, durably.

Everything else the pipeline now heals by itself (cycle tick loop, provider
pacing, self-retry). What it cannot fix are these, and each is an alert:

  * `credits_low`       - the video balance does not cover the renders queued,
                          or a submit came back "Credits insufficient". Only a
                          person can top up.
  * `provider_outage`   - the outage pause has gone on for over an hour
                          (`ALERT_OUTAGE_MINUTES`). Nothing to fix on our side,
                          but a batch running hours late is worth knowing.
  * `job_attempt_cap`   - a job spent its whole lifetime attempt budget on
                          provider 500s. It holds for a person now.

**The record is the event**, `alert.<kind>` at level `error` in `job_events`,
written whatever channel is configured - so `castrol alerts` and the problems
index on `job_events` see every alert even when delivery to a person fails.
A channel (`Notifier`) is a second copy, best effort, never allowed to fail
the stage or the cycle that raised it.

**No channel is wired yet.** Email, Slack and WhatsApp are all plausible and
the choice is the owner's. `ALERT_CHANNEL=events` (the only one so far) means
the event and the log line and nothing else. Adding a channel is one class
with a `send(alert)` and one entry in `NOTIFIERS`.

**Each alert is said once per `ALERT_REPEAT_S` per dedupe key**, checked
against `job_events` rather than memory, so the noon cycle does not repeat the
midnight cycle's `credits_low` and a restart does not re-announce everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import get_settings
from .events import record_event
from .logging import get_logger

log = get_logger(__name__)

CREDITS_LOW = "credits_low"
PROVIDER_OUTAGE = "provider_outage"
JOB_ATTEMPT_CAP = "job_attempt_cap"
KINDS = (CREDITS_LOW, PROVIDER_OUTAGE, JOB_ATTEMPT_CAP)


@dataclass(frozen=True)
class Alert:
    kind: str
    #: One sentence a person can act on. Ours, never a vendor's text.
    summary: str
    dedupe_key: str
    job_id: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def event(self) -> str:
        return f"alert.{self.kind}"


class Notifier(Protocol):
    """Somewhere an alert is sent, beyond the event that records it."""

    name: str

    def send(self, alert: Alert) -> None: ...


class EventOnlyNotifier:
    """The event IS the alert. Sends nothing anywhere else."""

    name = "events"

    def send(self, alert: Alert) -> None:
        return None


#: Channel name -> factory. The owner has not chosen a channel yet.
NOTIFIERS: dict[str, type] = {"events": EventOnlyNotifier}


def notifier() -> Notifier:
    name = get_settings().alert_channel
    factory = NOTIFIERS.get(name)
    if factory is None:
        raise ValueError(f"ALERT_CHANNEL={name!r} is not one of {sorted(NOTIFIERS)}")
    return factory()


def _recently_raised(alert: Alert) -> bool:
    """Has this exact alert gone out inside the repeat window? Fails open."""
    from . import db

    try:
        row = db.fetch_one(
            """
            SELECT 1 AS seen FROM job_events
             WHERE event = %(event)s
               AND fields ->> 'dedupe_key' = %(key)s
               AND created_at > now() - make_interval(secs => %(window)s)
             LIMIT 1;
            """,
            {"event": alert.event, "key": alert.dedupe_key,
             "window": get_settings().alert_repeat_s},
        )
    except Exception as exc:  # noqa: BLE001 - better a repeat than a silence
        log.warning("alert.dedupe_failed", error=str(exc)[:200])
        return False
    return row is not None


def raise_alert(
    kind: str,
    summary: str,
    *,
    dedupe_key: str | None = None,
    job_id: str | None = None,
    stage: str | None = None,
    stage_run_id: str | None = None,
    **fields: Any,
) -> bool:
    """Record an alert and hand it to the channel. True if it went out.

    Never raises: an alert is raised from inside a cycle that is still doing
    useful work, and a broken channel must not stop the renders.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown alert kind {kind!r}")
    alert = Alert(kind=kind, summary=summary, dedupe_key=dedupe_key or kind,
                  job_id=job_id, fields=fields)
    if _recently_raised(alert):
        return False
    log.error(alert.event, summary=summary, **fields)
    record_event(alert.event, job_id=job_id, stage=stage, stage_run_id=stage_run_id,
                 level="error", summary=summary, dedupe_key=alert.dedupe_key, **fields)
    try:
        notifier().send(alert)
    except Exception as exc:  # noqa: BLE001 - the event above is the record
        log.error("alert.send_failed", kind=kind, channel=get_settings().alert_channel,
                  error=str(exc)[:300])
    return True


def recent(hours: float = 24) -> list[dict[str, Any]]:
    """Alerts raised in the last `hours`, newest first."""
    from . import db

    return db.fetch_all(
        """
        SELECT created_at, event, job_id, stage, fields ->> 'summary' AS summary, fields
          FROM job_events
         WHERE event LIKE 'alert.%%'
           AND created_at > now() - make_interval(secs => %(secs)s)
         ORDER BY created_at DESC;
        """,
        {"secs": hours * 3600},
    )
