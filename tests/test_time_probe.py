from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "time_probe", Path(__file__).resolve().parents[1] / "scripts" / "time_probe.py"
)
time_probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(time_probe)


def test_go_within_cap() -> None:
    estimate = time_probe.estimate_seconds(steps_per_epoch=100, epochs=1, step_seconds=1.0)
    assert "GO" in time_probe.verdict(estimate, phase=1, remaining_s=None)


def test_scope_down_over_cap() -> None:
    estimate = time_probe.estimate_seconds(steps_per_epoch=1000, epochs=10, step_seconds=2.0)
    assert "SCOPE DOWN" in time_probe.verdict(estimate, phase=1, remaining_s=None)


def test_no_go_when_phase_closed() -> None:
    estimate = time_probe.estimate_seconds(steps_per_epoch=10, epochs=1, step_seconds=1.0)
    assert "NO-GO" in time_probe.verdict(estimate, phase=3, remaining_s=None)


def test_no_go_when_remaining_time_exceeded() -> None:
    estimate = time_probe.estimate_seconds(steps_per_epoch=100, epochs=1, step_seconds=1.0)
    assert "NO-GO" in time_probe.verdict(estimate, phase=2, remaining_s=10.0)


def test_remaining_seconds_counts_down() -> None:
    now = datetime(2026, 7, 19, 12, 0, 0)
    started = now - timedelta(hours=1)
    assert time_probe.remaining_seconds(started, now) == (24 - 1) * 3600
