"""Self-retry: provider 500s, outages and an empty balance heal without a person.

Batch 2 finished only after four rounds of `castrol retry-failed` by hand and a
credit run-out that killed every job it touched. These pin the three things
that replace the person: a long retry budget that releases itself, an outage
pause that probes before resuming, and a credits pause that spends nobody's
attempt and resumes when the balance does.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from castrol_pipeline import orchestrator
from castrol_pipeline.common import health, pacing
from castrol_pipeline.common.errors import (
    StageErrorCode,
    StageFailure,
    VendorOutOfCredits,
)
from castrol_pipeline.config import Settings
from castrol_pipeline.stages import mocks, vendors
from castrol_pipeline.stages.base import PipelineStage

S = PipelineStage


def _settings(**over: Any):
    s = Settings(**over)
    return lambda: s


@pytest.fixture
def events(monkeypatch) -> list[tuple[str, dict]]:
    got: list[tuple[str, dict]] = []
    rec = lambda name, **kw: got.append((name, kw))  # noqa: E731
    monkeypatch.setattr(health, "record_event", rec)
    monkeypatch.setattr(orchestrator, "record_event", rec)
    return got


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _breaker(**policy: Any) -> tuple[health.ProviderHealth, Clock]:
    clock = Clock()
    return health.ProviderHealth("video", health.OutagePolicy(**policy), clock=clock), clock


# ------------------------------------------------------------ the breaker --


class TestOutagePause:
    def test_pauses_when_nearly_every_recent_render_fails(self, events) -> None:
        h, clock = _breaker()
        for _ in range(19):
            h.record_verdict(failed=True)
        assert h.submit_block_reason() is None, "paused on too few samples"
        h.record_verdict(failed=True)
        assert "outage" in (h.submit_block_reason() or "")
        assert [n for n, _ in events] == ["provider.outage_paused"]

    def test_the_ordinary_failure_rate_never_pauses(self, events) -> None:
        """Batch 2 ran at ~30-60% 500s under load for hours; that is the retry
        budget's job, not an outage. A 1-in-2 pattern over 2,000 renders must
        never trip a 90% threshold."""
        h, clock = _breaker()
        for i in range(2000):
            clock.t += 1
            h.record_verdict(failed=i % 2 == 0)
        assert h.state == health.CLOSED
        assert events == []

    def test_after_the_pause_two_probes_go_out_then_it_waits(self, events) -> None:
        h, clock = _breaker(pause_s=300, probes=2)
        for _ in range(20):
            h.record_verdict(failed=True)
        clock.t += 299
        assert h.submit_block_reason()
        clock.t += 2
        assert h.submit_block_reason() is None
        h.note_submit()
        assert h.submit_block_reason() is None
        h.note_submit()
        assert "probe" in (h.submit_block_reason() or "")
        assert h.state == health.PROBING

    def test_a_probe_that_renders_ends_the_outage(self, events) -> None:
        h, clock = _breaker(pause_s=300)
        for _ in range(20):
            h.record_verdict(failed=True)
        clock.t += 301
        h.submit_block_reason()
        h.note_submit()
        h.record_verdict(failed=False)
        assert h.state == health.CLOSED
        assert h.outage_since is None
        assert h.submit_block_reason() is None
        assert [n for n, _ in events][-1] == "provider.outage_ended"

    def test_a_probe_that_fails_pauses_again_for_longer(self, events) -> None:
        h, clock = _breaker(pause_s=300, max_pause_s=1800)
        for _ in range(20):
            h.record_verdict(failed=True)
        started = h.outage_since
        clock.t += 301
        h.submit_block_reason()
        h.record_verdict(failed=True)
        assert h.state == health.PAUSED
        clock.t += 301
        assert h.submit_block_reason(), "the second pause should be 600s"
        clock.t += 300
        assert h.submit_block_reason() is None
        # One outage across both pauses: the >1h alert reads this.
        assert h.outage_since == started
        assert h.outage_minutes() == pytest.approx((clock.t - started) / 60)

    def test_renders_still_landing_during_a_pause_decide_nothing(self, events) -> None:
        h, clock = _breaker()
        for _ in range(20):
            h.record_verdict(failed=True)
        h.record_verdict(failed=False)  # a straggler from before the pause
        assert h.state == health.PAUSED


class TestCreditsPause:
    def test_blocks_until_the_balance_check_says_covered(self, events, monkeypatch) -> None:
        monkeypatch.setattr(health, "get_settings", _settings(credits_recheck_s=300))
        h, clock = _breaker()
        h.pause_for_credits("402")
        h.pause_for_credits("402 again")  # one event per episode
        assert "credits" in (h.submit_block_reason() or "")
        assert not h.balance_check_due()
        clock.t += 300
        assert h.balance_check_due()
        h.balance_checked(covered=False, balance_usd=3.0, needed_usd=60.0)
        assert h.submit_block_reason() and not h.balance_check_due()
        clock.t += 300
        h.balance_checked(covered=True, balance_usd=100.0, needed_usd=60.0)
        assert h.submit_block_reason() is None
        assert [n for n, _ in events] == ["provider.credits_exhausted",
                                          "provider.credits_restored"]


# ------------------------------------------------------- the orchestrator --


@pytest.fixture
def fake_db(monkeypatch, events):
    calls: dict[str, list] = {"released": [], "failed": [], "claimed": []}
    monkeypatch.setattr(orchestrator, "load_context", lambda job_id: object())
    monkeypatch.setattr(orchestrator, "schedule_ready", lambda *a, **k: [])
    monkeypatch.setattr(orchestrator, "advance_job", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.db, "count_in_flight", lambda stage: 0)
    monkeypatch.setattr(orchestrator.db, "release_claim",
                        lambda run_id, **kw: calls["released"].append((run_id, kw)))
    monkeypatch.setattr(orchestrator.db, "mark_failed",
                        lambda run_id, **kw: calls["failed"].append((run_id, kw)))

    def claim(stage, worker, job_id=None):
        calls["claimed"].append(stage)
        return {"id": "run-1", "job_id": "job-1", "attempts": 1}

    monkeypatch.setattr(orchestrator.db, "claim_stage_run", claim)
    return calls


class TestOutOfCredits:
    def test_a_402_pauses_every_submit_and_spends_no_attempt(self, fake_db, monkeypatch):
        class Video:
            def run(self, ctx):
                raise VendorOutOfCredits("402 Credits insufficient")

        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: Video()})
        assert orchestrator.execute_one(S.VIDEO) is False
        assert fake_db["failed"] == [], "a 402 failed the job it landed on"
        assert [r for r, _ in fake_db["released"]] == ["run-1"]

        # The next submit is not even claimed.
        assert orchestrator.execute_one(S.VIDEO) is False
        assert fake_db["claimed"] == [S.VIDEO]
        assert "credits" in (orchestrator.submit_blocked(S.VIDEO) or "")

    def test_it_resumes_by_itself_once_the_balance_covers_a_slot_fill(
        self, fake_db, monkeypatch
    ):
        balance = {"usd": 5.0}

        class Video:
            def balance_usd(self):
                return balance["usd"]

        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: Video()})
        monkeypatch.setattr(orchestrator, "expected_render_spend",
                            lambda: {"renders": 500, "usd": 540.0, "per_render_usd": 1.08})
        monkeypatch.setattr(health, "get_settings", _settings(credits_recheck_s=1))
        h = health.health(pacing.VIDEO)
        clock = Clock()
        h._clock = clock
        h.pause_for_credits("402")

        clock.t += 2
        assert orchestrator.submit_blocked(S.VIDEO), "resumed on $5 against 60 slots"
        balance["usd"] = 70.0   # 60 slots x $1.08 = $64.80: one full fill
        clock.t += 2
        assert orchestrator.submit_blocked(S.VIDEO) is None

    def test_the_preflight_warns_when_the_queue_is_not_covered(self, events, monkeypatch):
        class Video:
            def balance_usd(self):
                return 100.0

        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: Video()})
        monkeypatch.setattr(orchestrator, "expected_render_spend",
                            lambda: {"renders": 500, "usd": 540.0, "per_render_usd": 1.08})
        out = orchestrator.credits_preflight()
        assert out is not None and out["short"] is True
        assert [n for n, _ in events] == ["cycle.credits_short"]

    def test_no_balance_endpoint_means_no_preflight(self, monkeypatch):
        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: object()})
        assert orchestrator.credits_preflight() is None


def _failed(code: str = "VENDOR_UNAVAILABLE", *, minutes_ago: float = 60,
            lifetime: int = 12, input_hash: str = "h1") -> dict[str, Any]:
    return {"id": "run-9", "status": "failed", "error_code": code, "input_hash": input_hash,
            "finished_at": datetime.now(UTC) - timedelta(minutes=minutes_ago),
            "lifetime_attempts": lifetime}


class TestSelfReleasingHold:
    def test_a_provider_500_holds_while_it_cools_off(self):
        assert orchestrator.failure_decision(_failed(minutes_ago=5), "h1") == orchestrator.HOLD

    def test_then_releases_itself(self):
        assert orchestrator.failure_decision(_failed(minutes_ago=31), "h1") == orchestrator.RELEASE
        assert orchestrator.failure_holds(_failed(minutes_ago=31), "h1") is False

    def test_until_the_lifetime_cap(self):
        row = _failed(minutes_ago=600, lifetime=24)
        assert orchestrator.failure_decision(row, "h1") == orchestrator.CAPPED
        assert orchestrator.failure_holds(row, "h1") is True

    def test_a_rejection_still_waits_for_a_person(self):
        row = _failed("VENDOR_REJECTED", minutes_ago=6000, lifetime=1)
        assert orchestrator.failure_decision(row, "h1") == orchestrator.HOLD

    def test_new_inputs_are_a_new_question(self):
        assert orchestrator.failure_decision(_failed(lifetime=99), "h2") == orchestrator.NO_FAILURE

    def test_a_release_is_recorded(self, monkeypatch, events):
        ctx = type("Ctx", (), {"upstream": {}})()
        impl = type("Impl", (), {"input_hash": lambda self, c: "h1"})()
        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: impl})
        monkeypatch.setattr(orchestrator, "load_context", lambda j: ctx)
        monkeypatch.setattr(orchestrator, "ready_stages", lambda j, u: [S.VIDEO])
        monkeypatch.setattr(orchestrator, "latest_runs",
                            lambda j: {"video": _failed(minutes_ago=31, lifetime=12)})
        monkeypatch.setattr(orchestrator.db, "find_succeeded_run", lambda *a: None)
        monkeypatch.setattr(orchestrator.db, "enqueue_stage_run", lambda *a: {"id": "new"})
        assert orchestrator.schedule_ready("job-1") == [S.VIDEO]
        name, fields = events[-1]
        assert name == "job.auto_retry"
        assert fields["lifetime_attempts"] == 12 and fields["lifetime_cap"] == 24

    def test_the_sweep_flags_a_capped_job_once(self, monkeypatch, events):
        capped = {"job_id": "job-1", "stage": "video", "id": "run-9",
                  "finished_at": datetime.now(UTC) - timedelta(hours=2),
                  "lifetime_attempts": 24}
        due = {**capped, "job_id": "job-2", "id": "run-8", "lifetime_attempts": 12}
        released: list[str] = []
        seen: set[str] = set()
        monkeypatch.setattr(orchestrator, "unavailable_failures", lambda: [capped, due])
        monkeypatch.setattr(orchestrator, "schedule_ready",
                            lambda j, **k: released.append(j) or [S.VIDEO])
        monkeypatch.setattr(orchestrator, "advance_job", lambda j: None)

        def fetch_one(sql, params):
            if params["run"] in seen:
                return {"seen": 1}
            seen.add(params["run"])
            return None

        monkeypatch.setattr(orchestrator.db, "fetch_one", fetch_one)
        assert orchestrator.release_due_failures() == {"released": 1, "capped": 1}
        assert orchestrator.release_due_failures() == {"released": 1, "capped": 0}
        assert released == ["job-2", "job-2"]
        assert [n for n, _ in events].count("job.attempt_cap_reached") == 1

    def test_pending_releases_count_toward_the_cycle(self):
        now = datetime.now(UTC)
        rows = [{"finished_at": now, "lifetime_attempts": 12},
                {"finished_at": now - timedelta(minutes=10), "lifetime_attempts": 3},
                {"finished_at": now, "lifetime_attempts": 24}]  # capped: not coming back
        out = orchestrator.pending_releases(rows)
        assert out["count"] == 2
        assert out["next_due"] == rows[1]["finished_at"] + timedelta(minutes=30)


# --------------------------------------------------------------- vendors --


class _Resp:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code, self._body, self.text, self.content = status, body, str(body), b""

    def json(self) -> Any:
        if self._body is None:
            raise ValueError
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


class TestVendorCredits:
    @pytest.mark.parametrize("resp", [
        _Resp(402, {"msg": "Payment Required"}),
        _Resp(200, {"code": 402, "msg": "Credits insufficient"}),
        _Resp(200, {"code": 400, "msg": "Insufficient credits, please top up"}),
    ], ids=["http-402", "body-402", "message"])
    def test_out_of_credits_on_submit(self, monkeypatch, resp):
        _client(monkeypatch, resp)
        with pytest.raises(VendorOutOfCredits):
            vendors.video_submit("img", "aud", model_id="m")

    def test_out_of_credits_is_not_terminal(self):
        assert not VendorOutOfCredits("x").is_terminal

    @pytest.mark.parametrize(("data", "usd"), [
        (2070, 10.0),
        ({"credit": 207}, 1.0),
        ({"balance": "414"}, 2.0),
    ])
    def test_the_balance_is_read_in_usd(self, monkeypatch, data, usd):
        _client(monkeypatch, _Resp(200, {"code": 200, "msg": "success", "data": data}))
        assert vendors.video_balance_usd() == pytest.approx(usd)

    def test_an_unreadable_balance_raises_rather_than_guessing(self, monkeypatch):
        _client(monkeypatch, _Resp(200, {"code": 200, "data": {"what": "?"}}))
        with pytest.raises(StageFailure) as info:
            vendors.video_balance_usd()
        assert info.value.code == StageErrorCode.INTERNAL


# ------------------------------------------------------------------ mocks --


class TestMockRenderFailures:
    def test_per_attempt_not_per_job(self, monkeypatch):
        monkeypatch.setattr(mocks, "get_settings", _settings(mock_render_500_percent=50))
        job = "00000000-0000-0000-0000-00000000000a"
        outcomes = {mocks.render_500(f"mock-video-{1_700_000_000 + i}-{job}")
                    for i in range(0, 4000, 60)}
        assert outcomes == {True, False}, "one job failed every attempt, or none"

    def test_roughly_the_share_asked_for(self, monkeypatch):
        monkeypatch.setattr(mocks, "get_settings", _settings(mock_render_500_percent=50))
        tries = [mocks.render_500(f"mock-video-{1_700_000_000 + i}-job") for i in range(2000)]
        assert 0.45 < sum(tries) / len(tries) < 0.55

    def test_an_outage_window_fails_every_render_submitted_in_it(self, monkeypatch):
        monkeypatch.setattr(mocks, "get_settings", _settings(mock_outage_windows="0-600"))
        p = mocks.MockVideoProvider()
        monkeypatch.setattr(mocks, "VIDEO_PROVIDER", p)
        t0 = int(p.epoch)
        assert all(mocks.render_500(f"mock-video-{t0 + i}-j") for i in range(1, 500, 7))
        assert not mocks.render_500(f"mock-video-{t0 + 700}-j")


class TestBreakerIsFedAtFirstPoll:
    """A 500 is a "0s render"; a good render takes 8-20 minutes. Judged at
    completion, the first twenty outcomes of every batch are failures and an
    ordinary 50% day pauses - which the batch simulation caught."""

    def _poll(self, monkeypatch, answer):
        verdicts: list[bool] = []
        monkeypatch.setattr(orchestrator, "_record_outcome",
                            lambda stage, *, failed: verdicts.append(failed))

        class Video:
            is_async = True

            def poll(self, task_id, ctx):
                if isinstance(answer, Exception):
                    raise answer
                return answer

        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: Video()})
        row = {"id": "r1", "job_id": "00000000-0000-0000-0000-000000000001",
               "stage": "video", "vendor_task_id": "t1", "attempts": 1,
               "started_at": datetime.now(UTC)}
        monkeypatch.setattr(orchestrator.db, "fetch_all", lambda *a, **k: [row])
        monkeypatch.setattr(orchestrator.db, "mark_failed", lambda *a, **k: None)
        monkeypatch.setattr(orchestrator, "load_context", lambda j: object())
        monkeypatch.setattr(orchestrator, "record_event", lambda *a, **k: None)
        monkeypatch.setattr(orchestrator, "schedule_ready", lambda *a, **k: [])
        monkeypatch.setattr(orchestrator, "advance_job", lambda *a, **k: None)
        monkeypatch.setattr(orchestrator, "_poll_interval", lambda stage: 0.0)
        orchestrator.poll_once()
        orchestrator.poll_once()
        return verdicts

    def test_still_rendering_at_the_first_look_counts_as_taken_once(self, monkeypatch):
        assert self._poll(monkeypatch, None) == [False]

    def test_a_500_counts_as_failed(self, monkeypatch):
        from castrol_pipeline.common.errors import VendorUnavailable

        verdicts = self._poll(monkeypatch, VendorUnavailable("500"))
        assert verdicts and all(verdicts)
