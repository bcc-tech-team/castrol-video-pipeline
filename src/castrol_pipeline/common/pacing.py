"""Request pacing for the paid providers: one limiter per provider, per process.

Measured on Batches 1 and 2 and documented by the video provider: more than 20
requests in any 10 seconds is answered with HTTP 429, with no queueing. Our
SUBMITS never came close - about one every 3.4s - but `poll_once` checked every
running task back to back, so at 60+ renders in flight a single poll pass was a
burst well past 20 in 10s. That, not the submits, is what made the 429s.

So every request to a provider goes through ONE limiter for that provider:
submit, poll, and the balance check alike. It is a SLIDING WINDOW - at most
`max_requests` in any `window_s` - and not a token bucket, on purpose. A bucket
with capacity N refilling at N per window allows 2N inside one window (a full
bucket spent at the end of one window, refilled and spent again at the start of
the next), which is the very limit we are trying to stay under. The default is
15 per 10s, which leaves a margin of 5 below the provider's 20 for clock skew
and for anything else on the account.

A 429 that arrives anyway closes the limiter for `throttle_backoff_s`, so the
requests behind it wait instead of making it worse.

Per PROCESS, not per account. The cycle is the only long-running caller; a
`castrol run` started by hand alongside it has its own limiter, which is why
the default sits well below the provider's figure rather than at it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from ..config import get_settings
from ..stages.base import PipelineStage

#: Provider roles. Named by role, never by vendor (see CLAUDE.md conventions).
VIDEO = "video"
IMAGE = "image"
VOICE = "voice"

#: Which provider each paid stage talks to.
STAGE_PROVIDER: dict[PipelineStage, str] = {
    PipelineStage.VIDEO: VIDEO,
    PipelineStage.IMAGE: IMAGE,
    PipelineStage.AUDIO: VOICE,
}


class RateLimiter:
    """At most `max_requests` in any rolling `window_s`. Thread-safe.

    `clock` and `sleep` are injectable so tests can run a simulated hour in
    milliseconds and assert the window holds exactly.
    """

    def __init__(
        self,
        max_requests: int,
        window_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_requests < 1 or window_s <= 0:
            raise ValueError("a limiter needs at least one request per positive window")
        self.max_requests = max_requests
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._sent: deque[float] = deque()
        self._closed_until = 0.0

    def _wait_for_slot(self, now: float) -> float:
        """Seconds until a request may go out. Caller holds the lock."""
        while self._sent and self._sent[0] <= now - self.window_s:
            self._sent.popleft()
        wait = max(0.0, self._closed_until - now)
        if len(self._sent) >= self.max_requests:
            wait = max(wait, self._sent[0] + self.window_s - now)
        return wait

    def try_acquire(self) -> bool:
        """Take a slot if one is free right now."""
        with self._lock:
            now = self._clock()
            if self._wait_for_slot(now) > 0:
                return False
            self._sent.append(now)
            return True

    def acquire(self, timeout: float | None = None) -> bool:
        """Wait for a slot, up to `timeout` seconds. True if one was taken."""
        deadline = None if timeout is None else self._clock() + timeout
        while True:
            with self._lock:
                now = self._clock()
                wait = self._wait_for_slot(now)
                if wait <= 0:
                    self._sent.append(now)
                    return True
            if deadline is not None and now + wait > deadline:
                return False
            # Short naps, so a back_off() from another thread is seen promptly.
            self._sleep(min(wait, 0.5))

    def back_off(self, seconds: float) -> None:
        """Nothing goes out for `seconds`. Called on a 429."""
        with self._lock:
            self._closed_until = max(self._closed_until, self._clock() + seconds)

    def in_window(self) -> int:
        """Requests sent in the current window."""
        with self._lock:
            self._wait_for_slot(self._clock())
            return len(self._sent)


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def limiter(provider: str) -> RateLimiter:
    """The process-wide limiter for `provider`, built from settings once."""
    with _LIMITERS_LOCK:
        found = _LIMITERS.get(provider)
        if found is None:
            s = get_settings()
            per_window = {
                VIDEO: s.video_requests_per_window,
                IMAGE: s.image_requests_per_window,
                VOICE: s.voice_requests_per_window,
            }[provider]
            found = _LIMITERS[provider] = RateLimiter(per_window, s.provider_window_s)
        return found


def limiter_for(stage: PipelineStage) -> RateLimiter | None:
    provider = STAGE_PROVIDER.get(stage)
    return limiter(provider) if provider else None


def in_flight_cap(stage: PipelineStage) -> int | None:
    """How many runs of `stage` may be claimed or running at once, or None.

    The video provider runs ~100+ concurrent tasks per account, and its 500
    rate climbed with load on Batches 1 and 2: ~30% under 20 in flight, ~90%
    above 120. 60 is inside the account limit and well short of where it
    falls over. The image cap is there so a 500-job intake is not 500 edits
    in flight at once at a provider whose own limit we have never measured;
    at ~83s an edit, 40 in flight is ~29 a minute, far ahead of the renders
    it feeds.
    """
    s = get_settings()
    cap = {
        PipelineStage.VIDEO: s.max_concurrency_video,
        PipelineStage.IMAGE: s.max_concurrency_image,
    }.get(stage)
    return cap if cap and cap > 0 else None


def reset() -> None:
    """Forget every limiter. Tests, and a settings change in one process."""
    with _LIMITERS_LOCK:
        _LIMITERS.clear()
