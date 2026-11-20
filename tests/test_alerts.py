"""Alerts: what still needs a person is said once, durably, whatever the channel."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from typer.testing import CliRunner

from castrol_pipeline import cli, orchestrator
from castrol_pipeline.common import alerts, health
from castrol_pipeline.common.errors import VendorOutOfCredits
from castrol_pipeline.config import Settings
from castrol_pipeline.stages.base import PipelineStage

S = PipelineStage


@pytest.fixture
def raised(monkeypatch) -> list[tuple[str, dict]]:
    got: list[tuple[str, dict]] = []
    rec = lambda name, **kw: got.append((name, kw))  # noqa: E731
    for mod in (alerts, health, orchestrator):
        monkeypatch.setattr(mod, "record_event", rec)
    monkeypatch.setattr(alerts, "_recently_raised", lambda alert: False)
    return got


def _names(got: list[tuple[str, dict]]) -> list[str]:
    return [n for n, _ in got]


class TestRaise:
    def test_the_record_is_an_error_level_event(self, raised) -> None:
        assert alerts.raise_alert(alerts.CREDITS_LOW, "Top up.", balance_usd=3.0) is True
        name, fields = raised[0]
        assert name == "alert.credits_low"
        assert fields["level"] == "error"
        assert fields["summary"] == "Top up."
        assert fields["dedupe_key"] == "credits_low"
        assert fields["balance_usd"] == 3.0

    def test_said_once_per_window(self, raised, monkeypatch) -> None:
        monkeypatch.setattr(alerts, "_recently_raised", lambda alert: True)
        assert alerts.raise_alert(alerts.CREDITS_LOW, "Top up.") is False
        assert raised == []

    def test_a_broken_channel_never_stops_the_cycle(self, raised, monkeypatch) -> None:
        class Broken:
            name = "broken"

            def send(self, alert):
                raise ConnectionError("smtp down")

        monkeypatch.setattr(alerts, "notifier", lambda: Broken())
        assert alerts.raise_alert(alerts.JOB_ATTEMPT_CAP, "Look at job 1.") is True
        assert _names(raised) == ["alert.job_attempt_cap"]

    def test_the_channel_gets_the_alert(self, raised, monkeypatch) -> None:
        sent: list[alerts.Alert] = []

        class Capture:
            name = "capture"

            def send(self, alert):
                sent.append(alert)

        monkeypatch.setattr(alerts, "notifier", lambda: Capture())
        alerts.raise_alert(alerts.PROVIDER_OUTAGE, "Down 70 min.", dedupe_key="k", minutes=70)
        assert sent[0].kind == "provider_outage" and sent[0].fields == {"minutes": 70}

    def test_an_unknown_kind_is_a_bug(self, raised) -> None:
        with pytest.raises(ValueError):
            alerts.raise_alert("disk_full", "?")

    def test_events_is_the_default_and_only_channel_so_far(self, monkeypatch) -> None:
        """No channel is wired until the owner chooses one."""
        assert Settings().alert_channel == "events"
        assert isinstance(alerts.notifier(), alerts.EventOnlyNotifier)
        monkeypatch.setattr(alerts, "get_settings", lambda: Settings(alert_channel="slack"))
        with pytest.raises(ValueError):
            alerts.notifier()

    def test_a_dedupe_lookup_that_fails_does_not_swallow_the_alert(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise OSError("db down")

        monkeypatch.setattr(alerts.get_settings(), "alert_repeat_s", 60, raising=False)
        from castrol_pipeline.common import db

        monkeypatch.setattr(db, "fetch_one", boom)
        alert = alerts.Alert(kind="credits_low", summary="x", dedupe_key="k")
        assert alerts._recently_raised(alert) is False


class TestWhatRaisesThem:
    def test_an_outage_over_an_hour(self, raised, monkeypatch) -> None:
        t = [0.0]
        h = health.ProviderHealth("video", health.OutagePolicy(pause_s=300), clock=lambda: t[0])
        monkeypatch.setattr(health, "all_health", lambda: {"video": h})
        for _ in range(20):
            h.record_render(failed=True)
        t[0] = 59 * 60
        assert orchestrator.check_alerts() == []
        t[0] = 61 * 60
        assert orchestrator.check_alerts() == [alerts.PROVIDER_OUTAGE]
        fields = dict(raised)["alert.provider_outage"]
        assert fields["dedupe_key"] == "provider_outage:video"
        assert fields["minutes"] == pytest.approx(61, abs=0.1)

    def test_running_out_of_credits_once_per_pause(self, raised, monkeypatch) -> None:
        class Video:
            def run(self, ctx):
                raise VendorOutOfCredits("402")

        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: Video()})
        monkeypatch.setattr(orchestrator, "load_context", lambda j: object())
        monkeypatch.setattr(orchestrator, "schedule_ready", lambda *a, **k: [])
        monkeypatch.setattr(orchestrator, "advance_job", lambda *a, **k: None)
        monkeypatch.setattr(orchestrator.db, "count_in_flight", lambda s: 0)
        monkeypatch.setattr(orchestrator.db, "release_claim", lambda *a, **k: None)
        monkeypatch.setattr(orchestrator.db, "claim_stage_run",
                            lambda *a, **k: {"id": "r", "job_id": "j", "attempts": 1})
        orchestrator.execute_one(S.VIDEO)
        # A second 402 in the same pause (from a `castrol run` racing the
        # cycle, say) adds nothing.
        health.health("video").credits_since = None
        health.health("video").pause_for_credits("again")
        assert _names(raised).count("alert.credits_low") == 1

    def test_a_short_balance_at_cycle_start(self, raised, monkeypatch) -> None:
        class Video:
            def balance_usd(self):
                return 108.0

        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: Video()})
        monkeypatch.setattr(orchestrator, "expected_render_spend",
                            lambda: {"renders": 500, "usd": 540.0, "per_render_usd": 1.08})
        orchestrator.credits_preflight()
        fields = dict(raised)["alert.credits_low"]
        assert "about 100 of the 500 renders" in fields["summary"]
        assert "$432.00" in fields["summary"]

    def test_a_covered_balance_raises_nothing(self, raised, monkeypatch) -> None:
        class Video:
            def balance_usd(self):
                return 1000.0

        monkeypatch.setattr(orchestrator, "get_stage_registry", lambda: {S.VIDEO: Video()})
        monkeypatch.setattr(orchestrator, "expected_render_spend",
                            lambda: {"renders": 500, "usd": 540.0, "per_render_usd": 1.08})
        assert orchestrator.credits_preflight()["short"] is False
        assert raised == []


class TestCli:
    def _run(self, monkeypatch, rows: list[dict[str, Any]]):
        monkeypatch.setattr(cli, "_boot", lambda: None)
        monkeypatch.setattr(alerts, "recent", lambda hours: rows)
        return CliRunner().invoke(cli.app, ["alerts", "--hours", "12"])

    def test_nothing_to_do_exits_zero(self, monkeypatch) -> None:
        out = self._run(monkeypatch, [])
        assert out.exit_code == 0
        assert "No alerts in the last 12h" in out.output

    def test_something_to_do_exits_one(self, monkeypatch) -> None:
        out = self._run(monkeypatch, [{"created_at": datetime(2026, 11, 20, 3, 0, tzinfo=UTC),
                                       "event": "alert.credits_low", "summary": "Top up $432."}])
        assert out.exit_code == 1
        assert "alert.credits_low" in out.output and "Top up $432." in out.output
