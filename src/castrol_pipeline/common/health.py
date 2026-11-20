"""Per-provider health: the outage pause and the out-of-credits pause.

Two reasons to stop sending a provider new work, both of which used to need a
person:

**An outage.** Batches 1 and 2 saw 503 of 1,094 video submits (46%) come back
as "500 internal error, please try again later" - 0% between 12:00 and 14:00
IST on 2026-10-23, 84% at 23:00. Every affected job eventually rendered. When
nearly every recent render is failing, sending more only spends attempts (and
`vendor_usage` reservations) against a provider that is down. So the provider
is PAUSED for a few minutes, then PROBED with one or two submits before the
floodgates open again. A pause that goes on, pause after pause, is an alert.

**No credits.** A 402 / "Credits insufficient" answer used to be a terminal
VendorRejected on whichever job was unlucky enough to be next, and on every job
after it. Now it pauses all submits to that provider WITHOUT spending anyone's
attempt, and the balance endpoint is asked every few minutes until it covers
the work again.

Both are a circuit breaker on SUBMITS only. Polling carries on: a render that
was already accepted is collected whatever state the provider is in.

State is in memory, per process, like the limiters in `pacing.py`. The cycle
is the only long-running caller, and a restart that forgets a pause costs one
probe's worth of failures before it learns again.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import get_settings
from .events import record_event
from .logging import get_logger

log = get_logger(__name__)

CLOSED = "closed"
PAUSED = "paused"
PROBING = "probing"


@dataclass(frozen=True)
class OutagePolicy:
    """When to pause, for how long, and how to come back. Defaults in config."""

    window: int = 30          # how many recent render outcomes are judged
    min_samples: int = 20     # fewer than this in the window decides nothing
    max_age_s: float = 1200   # outcomes older than this are forgotten
    fail_share: float = 0.9   # pause when this share of the window is a 500
    pause_s: float = 300      # first pause
    max_pause_s: float = 1800  # pauses double, up to this
    probes: int = 2           # submits let through to test the water
    probe_timeout_s: float = 900  # no verdict on the probes by then: send more

    @classmethod
    def from_settings(cls) -> OutagePolicy:
        s = get_settings()
        return cls(
            window=s.outage_window,
            min_samples=s.outage_min_samples,
            max_age_s=s.outage_max_age_s,
            fail_share=s.outage_fail_share,
            pause_s=s.outage_pause_s,
            max_pause_s=s.outage_max_pause_s,
            probes=s.outage_probes,
        )


class ProviderHealth:
    """One provider's breaker. Thread-safe; `clock` is injectable for tests."""

    def __init__(
        self,
        provider: str,
        policy: OutagePolicy | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.policy = policy or OutagePolicy()
        self._clock = clock
        self._lock = threading.Lock()
        self._outcomes: deque[tuple[float, bool]] = deque()  # (when, failed)
        self.state = CLOSED
        self._paused_until = 0.0
        self._pause_s = self.policy.pause_s
        self._probes_sent = 0
        self._probing_since = 0.0
        #: When the current outage began, across every pause/probe round of
        #: it. None when the provider is healthy. What the >1h alert reads.
        self.outage_since: float | None = None
        #: When submits were paused for credits. None when they are not.
        self.credits_since: float | None = None
        self.next_balance_check = 0.0

    # ----------------------------------------------------------- outcomes --

    def record_verdict(self, *, failed: bool) -> None:
        """One task's verdict: the provider took it, or it came back a 500.

        "Took it" is the task still rendering (or done) at its first poll, NOT
        its completion: a 500 is a "0s render" and a good one takes 8-20
        minutes, so a breaker fed completions sees nothing but failures for the
        first ten minutes of every batch. Only those two count. A rejection is
        a verdict on our inputs and says nothing about whether the provider is
        up. See `orchestrator._record_outcome`.
        """
        p = self.policy
        with self._lock:
            now = self._clock()
            if self.state == PROBING:
                if failed:
                    self._pause(now, again=True)
                else:
                    self._close(now)
                return
            if self.state == PAUSED:
                return  # stragglers from before the pause decide nothing
            self._outcomes.append((now, failed))
            while len(self._outcomes) > p.window:
                self._outcomes.popleft()
            while self._outcomes and self._outcomes[0][0] < now - p.max_age_s:
                self._outcomes.popleft()
            n = len(self._outcomes)
            bad = sum(1 for _, f in self._outcomes if f)
            if n >= p.min_samples and bad >= p.fail_share * n:
                self._pause(now, again=False, bad=bad, n=n)

    def _pause(self, now: float, *, again: bool, bad: int = 0, n: int = 0) -> None:
        if again:
            self._pause_s = min(self._pause_s * 2, self.policy.max_pause_s)
        else:
            self._pause_s = self.policy.pause_s
        if self.outage_since is None:
            self.outage_since = now
        self.state = PAUSED
        self._paused_until = now + self._pause_s
        self._outcomes.clear()
        self._probes_sent = 0
        fields: dict[str, Any] = {"provider": self.provider, "pause_s": self._pause_s,
                                  "outage_minutes": round((now - self.outage_since) / 60, 1)}
        if not again:
            fields.update(failed=bad, of=n)
        log.warning("provider.outage_paused", **fields)
        record_event("provider.outage_paused", level="warning",
                     probe_failed=again, **fields)

    def _close(self, now: float) -> None:
        minutes = round((now - (self.outage_since or now)) / 60, 1)
        self.state = CLOSED
        self.outage_since = None
        self._pause_s = self.policy.pause_s
        self._outcomes.clear()
        log.info("provider.outage_ended", provider=self.provider, minutes=minutes)
        record_event("provider.outage_ended", level="warning",
                     provider=self.provider, minutes=minutes)

    # ------------------------------------------------------------ credits --

    def pause_for_credits(self, detail: str) -> None:
        with self._lock:
            if self.credits_since is not None:
                return
            now = self._clock()
            self.credits_since = now
            self.next_balance_check = now + get_settings().credits_recheck_s
        log.error("provider.credits_exhausted", provider=self.provider)
        record_event("provider.credits_exhausted", level="error",
                     provider=self.provider, detail=detail[:300])

    def balance_check_due(self) -> bool:
        with self._lock:
            return self.credits_since is not None and self._clock() >= self.next_balance_check

    def balance_checked(self, *, covered: bool, balance_usd: float | None,
                        needed_usd: float) -> None:
        """The outcome of a balance check while paused for credits."""
        with self._lock:
            now = self._clock()
            if not covered:
                self.next_balance_check = now + get_settings().credits_recheck_s
                return
            minutes = round((now - (self.credits_since or now)) / 60, 1)
            self.credits_since = None
        log.info("provider.credits_restored", provider=self.provider)
        record_event("provider.credits_restored", level="warning", provider=self.provider,
                     balance_usd=balance_usd, needed_usd=needed_usd, paused_minutes=minutes)

    # ------------------------------------------------------------ the gate --

    def submit_block_reason(self) -> str | None:
        """Why a new submit may not go out now, or None.

        Moving PAUSED -> PROBING happens here, on the first ask after the pause
        ends, so a pause needs no timer of its own.
        """
        with self._lock:
            if self.credits_since is not None:
                return f"{self.provider} provider is out of credits"
            now = self._clock()
            if self.state == PAUSED:
                if now < self._paused_until:
                    return f"{self.provider} provider outage pause"
                self.state = PROBING
                self._probes_sent = 0
                self._probing_since = now
                log.info("provider.outage_probing", provider=self.provider)
                record_event("provider.outage_probing", level="warning",
                             provider=self.provider, probes=self.policy.probes)
            if self.state == PROBING and self._probes_sent >= self.policy.probes:
                if now - self._probing_since < self.policy.probe_timeout_s:
                    return f"{self.provider} provider: waiting on probe renders"
                # No verdict on either probe in all that time - their polls
                # kept erroring, most likely. Never block on that forever:
                # send the next pair.
                self._probes_sent = 0
                self._probing_since = now
                log.warning("provider.probe_timeout", provider=self.provider)
            return None

    def note_submit(self) -> None:
        """A submit went out. While probing, it is one of the probes."""
        with self._lock:
            if self.state == PROBING:
                self._probes_sent += 1

    def outage_minutes(self) -> float:
        with self._lock:
            if self.outage_since is None:
                return 0.0
            return (self._clock() - self.outage_since) / 60

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            return {
                "state": self.state,
                "outage_minutes": round((now - self.outage_since) / 60, 1)
                if self.outage_since is not None else 0.0,
                "credits_paused": self.credits_since is not None,
                "recent_outcomes": len(self._outcomes),
            }


_HEALTH: dict[str, ProviderHealth] = {}
_HEALTH_LOCK = threading.Lock()


def health(provider: str) -> ProviderHealth:
    with _HEALTH_LOCK:
        found = _HEALTH.get(provider)
        if found is None:
            found = _HEALTH[provider] = ProviderHealth(provider, OutagePolicy.from_settings())
        return found


def all_health() -> dict[str, ProviderHealth]:
    with _HEALTH_LOCK:
        return dict(_HEALTH)


def reset() -> None:
    with _HEALTH_LOCK:
        _HEALTH.clear()
