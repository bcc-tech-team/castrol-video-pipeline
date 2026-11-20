"""Process-wide state that must not leak from one test into the next."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_provider_state():
    """Limiters and poll timestamps live for a process; tests get their own."""
    from castrol_pipeline import orchestrator
    from castrol_pipeline.common import pacing

    pacing.reset()
    orchestrator._LAST_POLLED.clear()
    yield
    pacing.reset()
    orchestrator._LAST_POLLED.clear()
