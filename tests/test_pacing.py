"""Provider pacing: the request limit holds, and a 429 never fails a run.

Batches 1 and 2: the video provider refuses more than 20 requests in any 10
seconds with a 429. The submits were slow enough; `poll_once` checking every
running task back to back was not. And a 429 on submit read as `code != 200`,
which raised VendorRejected - terminal - so one burst killed jobs outright.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from castrol_pipeline import orchestrator
from castrol_pipeline.common import pacing
from castrol_pipeline.common.errors import (
    StageErrorCode,
    StageFailure,
    VendorRejected,
    VendorThrottled,
)
from castrol_pipeline.stages import mocks, vendors
from castrol_pipeline.stages.base import AsyncSubmission, PipelineStage

S = PipelineStage


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 1e-6)


def _limiter(n: int = 15, window: float = 10.0) -> tuple[pacing.RateLimiter, FakeClock]:
    clock = FakeClock()
    return pacing.RateLimiter(n, window, clock=clock.now, sleep=clock.sleep), clock


# ------------------------------------------------------------- the limiter --


class TestRateLimiter:
    def test_a_greedy_caller_never_exceeds_the_window(self) -> None:
        lim, clock = _limiter(15, 10.0)
        sent: list[float] = []
        for _ in range(2000):  # about 22 simulated minutes of flat-out requests
            assert lim.acquire()
            sent.append(clock.t)
        assert mocks.max_in_any_window(sent, 10.0) == 15

    def test_the_window_edge_does_not_allow_a_double_burst(self) -> None:
        """A token bucket refilled at N per window lets 2N through across a
        window edge. A sliding window does not."""
        lim, clock = _limiter(15, 10.0)
        sent = []
        clock.t = 9.99
        for _ in range(15):
            assert lim.try_acquire()
            sent.append(clock.t)
        clock.t = 10.01
        assert not lim.try_acquire()
        assert mocks.max_in_any_window(sent, 10.0) <= 15

    def test_acquire_gives_up_at_its_timeout(self) -> None:
        lim, clock = _limiter(2, 10.0)
        assert lim.acquire(0) and lim.acquire(0)
        start = clock.t
        assert lim.acquire(1.0) is False
        assert clock.t - start <= 1.0

    def test_a_429_closes_the_limiter(self) -> None:
        lim, clock = _limiter(15, 10.0)
        lim.back_off(15)
        assert lim.try_acquire() is False
        clock.t += 15.01
        assert lim.try_acquire() is True

    def test_rejects_a_nonsense_limit(self) -> None:
        with pytest.raises(ValueError):
            pacing.RateLimiter(0, 10)

    def test_each_paid_stage_has_its_own_limiter_and_free_ones_have_none(self) -> None:
        assert pacing.limiter_for(S.VIDEO) is pacing.limiter_for(S.VIDEO)
        assert pacing.limiter_for(S.VIDEO) is not pacing.limiter_for(S.IMAGE)
        assert pacing.limiter_for(S.AUDIO) is not None
        for free in (S.PREP, S.COMPOSITE, S.CHECKS, S.PUBLISH, S.DELIVER):
            assert pacing.limiter_for(free) is None

    def test_the_defaults_are_the_agreed_ones(self) -> None:
        from castrol_pipeline.config import Settings

        s = Settings()
        assert (s.video_requests_per_window, s.provider_window_s) == (15, 10.0)
        assert s.max_concurrency_video == 60
        assert pacing.in_flight_cap(S.VIDEO) == 60
        assert pacing.in_flight_cap(S.PREP) is None


# ------------------------------------------------------------ the vendors --


class _Resp:
    def __init__(self, status: int, body: Any = None, text: str = "") -> None:
        self.status_code = status
        self._body = body
        self.text = text or str(body)
        self.content = b""

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("not json")
        return self._body


def _client(monkeypatch, resp: _Resp) -> None:
    class _Client:
        def __init__(self, *a, **kw): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **kw): return resp
        def get(self, *a, **kw): return resp

    monkeypatch.setattr(vendors.httpx, "Client", _Client)
    monkeypatch.setattr(vendors, "_video_headers", lambda: {})
    monkeypatch.setattr(vendors, "_video_base", lambda: "https://x")
    monkeypatch.setattr(vendors, "_image_headers", lambda: {})
    monkeypatch.setattr(vendors, "_image_base", lambda: "https://x")


class TestVendorThrottle:
    @pytest.mark.parametrize("resp", [
        _Resp(429, None, "Too Many Requests"),
        _Resp(200, {"code": 429, "msg": "rate limit exceeded"}),
        _Resp(200, {"code": "429", "msg": "rate limit exceeded"}),
    ], ids=["http-429", "body-429", "body-429-string"])
    def test_a_429_on_video_submit_is_a_throttle_not_a_rejection(self, monkeypatch, resp):
        _client(monkeypatch, resp)
        with pytest.raises(VendorThrottled):
            vendors.video_submit("img", "aud", model_id="m")

    def test_a_429_on_video_poll_is_a_throttle_not_in_flight(self, monkeypatch):
        _client(monkeypatch, _Resp(200, {"code": 429, "msg": "rate limit"}))
        with pytest.raises(VendorThrottled):
            vendors.video_poll("task")

    def test_a_429_on_the_image_provider_too(self, monkeypatch):
        _client(monkeypatch, _Resp(429))
        with pytest.raises(VendorThrottled):
            vendors.image_submit("p", "q", model_id="m")
        with pytest.raises(VendorThrottled):
            vendors.image_poll("task")

    def test_a_real_refusal_is_still_a_rejection(self, monkeypatch):
        _client(monkeypatch, _Resp(200, {"code": 422, "msg": "bad image_url"}))
        with pytest.raises(VendorRejected):
            vendors.video_submit("img", "aud", model_id="m")

    def test_a_body_that_is_not_json_stays_retryable(self, monkeypatch):
        _client(monkeypatch, _Resp(502, None, "<html>Bad Gateway</html>"))
        with pytest.raises(StageFailure) as info:
            vendors.video_submit("img", "aud", model_id="m")
        assert info.value.code == StageErrorCode.INTERNAL

    def test_the_voice_provider_429_is_a_throttle(self, monkeypatch, tmp_path):
        from castrol_pipeline.config import Settings

        _client(monkeypatch, _Resp(429, None, "slow down"))
        monkeypatch.setattr(vendors, "get_settings", lambda: Settings(CARTESIA_API_KEY="k"))
        with pytest.raises(VendorThrottled):
            vendors.voice_tts("t", voice_id="v", model_id="m", dst=tmp_path / "a.wav")

    def test_a_throttle_is_not_terminal(self):
        assert not VendorThrottled("x").is_terminal


# -------------------------------------------------------- the orchestrator --


@pytest.fixture
def harness(monkeypatch):
    """execute_one / poll_once over a fake database. Returns what was written."""
    calls: dict[str, list] = {"released": [], "failed": [], "running": [],
                              "claimed": [], "events": []}

    monkeypatch.setattr(orchestrator, "record_event",
                        lambda name, **kw: calls["events"].append((name, kw)))
    monkeypatch.setattr(orchestrator, "load_context", lambda job_id: object())
    monkeypatch.setattr(orchestrator, "schedule_ready", lambda *a, **k: [])
    monkeypatch.setattr(orchestrator, "advance_job", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.db, "release_claim",
                        lambda run_id, **kw: calls["released"].append((run_id, kw)))
    monkeypatch.setattr(orchestrator.db, "mark_failed",
                        lambda run_id, **kw: calls["failed"].append((run_id, kw)))
    monkeypatch.setattr(orchestrator.db, "mark_running",
                        lambda run_id, **kw: calls["running"].append((run_id, kw)))
    monkeypatch.setattr(orchestrator.db, "count_in_flight", lambda stage: 0)

    def claim(stage, worker, job_id=None):
        calls["claimed"].append(stage)
        return {"id": "run-1", "job_id": "job-1", "attempts": 1}

    monkeypatch.setattr(orchestrator.db, "claim_stage_run", claim)
    return calls


def _registry(monkeypatch, impl: Any) -> None:
    monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: impl})


class TestThrottleNeverFailsARun:
    def test_a_429_on_submit_puts_the_run_back_without_spending_an_attempt(
        self, harness, monkeypatch
    ):
        class Video:
            def run(self, ctx):
                raise VendorThrottled("429")

        _registry(monkeypatch, Video())
        assert orchestrator.execute_one(S.VIDEO) is False
        assert harness["failed"] == []
        assert [r for r, _ in harness["released"]] == ["run-1"]
        # The limiter is closed: the next request waits instead of piling on.
        assert pacing.limiter(pacing.VIDEO).try_acquire() is False
        assert [n for n, _ in harness["events"]] == ["vendor.throttled"]

    def test_a_429_on_poll_leaves_the_task_running_and_stops_the_pass(
        self, harness, monkeypatch
    ):
        polled: list[str] = []

        class Video:
            is_async = True

            def poll(self, task_id, ctx):
                polled.append(task_id)
                raise VendorThrottled("429")

        _registry(monkeypatch, Video())
        now = datetime.now(UTC)
        rows = [{"id": f"r{i}", "job_id": "00000000-0000-0000-0000-000000000001",
                 "stage": "video", "vendor_task_id": f"t{i}", "attempts": 1,
                 "started_at": now - timedelta(seconds=i)} for i in range(5)]
        monkeypatch.setattr(orchestrator.db, "fetch_all", lambda *a, **k: rows)
        orchestrator.poll_once()
        assert harness["failed"] == []
        assert len(polled) == 1, "kept polling a provider that had just said 429"

    def test_the_in_flight_cap_stops_a_claim_before_it_happens(self, harness, monkeypatch):
        class Video:
            def run(self, ctx):
                raise AssertionError("submitted past the cap")

        _registry(monkeypatch, Video())
        monkeypatch.setattr(orchestrator.db, "count_in_flight", lambda stage: 60)
        assert orchestrator.execute_one(S.VIDEO) is False
        assert harness["claimed"] == []
        assert orchestrator.submit_blocked(S.VIDEO)

    def test_under_the_cap_a_submit_goes_out(self, harness, monkeypatch):
        class Video:
            def run(self, ctx):
                return AsyncSubmission(vendor="v", vendor_task_id="t", model_id="m")

        _registry(monkeypatch, Video())
        monkeypatch.setattr(orchestrator.db, "count_in_flight", lambda stage: 59)
        assert orchestrator.execute_one(S.VIDEO) is True
        assert [r for r, _ in harness["running"]] == ["run-1"]

    def test_no_request_slot_puts_the_run_back(self, harness, monkeypatch):
        class Video:
            def run(self, ctx):
                raise AssertionError("sent without a slot")

        _registry(monkeypatch, Video())
        pacing.limiter(pacing.VIDEO).back_off(60)
        monkeypatch.setattr(orchestrator, "get_settings", _settings(provider_slot_wait_s=0))
        assert orchestrator.execute_one(S.VIDEO) is False
        assert [r for r, _ in harness["released"]] == ["run-1"]


def _settings(**over: Any):
    from castrol_pipeline.config import Settings

    s = Settings(**over)
    return lambda: s


class TestPollPacing:
    def test_never_polled_first_then_least_recent(self):
        now = datetime.now(UTC)
        rows = [{"id": n, "stage": "video", "started_at": now - timedelta(minutes=m)}
                for n, m in (("a", 1), ("b", 5), ("c", 3))]
        order = orchestrator.poll_order(rows, {"a": 100.0}, 200.0, lambda s: 60.0)
        assert [r["id"] for r in order] == ["b", "c", "a"]

    def test_a_task_polled_within_its_interval_is_skipped(self):
        rows = [{"id": "a", "stage": "video", "started_at": datetime.now(UTC)}]
        assert orchestrator.poll_order(rows, {"a": 190.0}, 200.0, lambda s: 60.0) == []

    def test_a_poll_pass_at_scale_stays_inside_the_limit(self, harness, monkeypatch):
        """120 renders in flight - the load at which Batch 2 saw ~90% 500s and
        where the old back-to-back pass burst far past 20 in 10s."""
        clock = FakeClock()
        lim = pacing.RateLimiter(15, 10.0, clock=clock.now, sleep=clock.sleep)
        pacing._LIMITERS[pacing.VIDEO] = lim
        sent: list[float] = []

        class Video:
            is_async = True

            def poll(self, task_id, ctx):
                sent.append(clock.t)
                return None

        _registry(monkeypatch, Video())
        now = datetime.now(UTC)
        rows = [{"id": f"r{i}", "job_id": "00000000-0000-0000-0000-000000000001",
                 "stage": "video", "vendor_task_id": f"t{i}", "attempts": 1,
                 "started_at": now} for i in range(120)]
        monkeypatch.setattr(orchestrator.db, "fetch_all", lambda *a, **k: rows)
        monkeypatch.setattr(orchestrator, "get_settings",
                            _settings(provider_slot_wait_s=1000, video_poll_interval_s=0))
        for _ in range(5):
            orchestrator.poll_once(limit=1000)
        assert len(sent) == 600
        assert mocks.max_in_any_window(sent, 10.0) <= 15


# -------------------------------------------------- the mock video provider --


class TestMockProvider:
    @pytest.fixture
    def provider(self, monkeypatch):
        p = mocks.MockVideoProvider()
        monkeypatch.setattr(mocks, "get_settings", _settings())
        return p

    def test_the_twenty_first_request_in_ten_seconds_is_refused(self, provider, monkeypatch):
        t = [5000.0]
        monkeypatch.setattr(mocks.time, "time", lambda: t[0])
        provider.reset()
        for _ in range(20):
            provider.request("poll")
            t[0] += 0.1
        with pytest.raises(VendorThrottled):
            provider.request("poll")
        stats = provider.stats()
        assert stats["refused"] == 1
        assert stats["max_in_any_window"] == 21  # a refused request is still a request

    def test_a_throttle_window_refuses_everything(self, provider, monkeypatch):
        monkeypatch.setattr(mocks, "get_settings", _settings(mock_throttle_windows="0-60"))
        with pytest.raises(VendorThrottled):
            provider.request("submit")

    @pytest.mark.parametrize("spec", ["10", "20-10", "a-b", "-5-10"])
    def test_a_window_it_cannot_read_is_an_error(self, spec):
        with pytest.raises(ValueError):
            mocks.parse_windows(spec)

    def test_windows_parse(self):
        assert mocks.parse_windows(" 300-420, 900-960 ") == [(300, 420), (900, 960)]
        assert mocks.parse_windows("") == []
