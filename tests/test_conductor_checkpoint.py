"""Tests for BaseConductor's checkpoint integration (Phase 5)."""

from __future__ import annotations

import io
import wave

import pytest

from vox_romana.core.budget_tracker import BudgetTracker
from vox_romana.core.conductor import BaseConductor, ConductorResult
from vox_romana.core.config import ContentConfig
from vox_romana.tts.checkpoint import CheckpointRenderer
from vox_romana.tts.engine import BaseTTSEngine, SynthesizedAudio


def _make_wav_bytes(seconds: float, rate: int = 22050) -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


class _FixedDurationEngine(BaseTTSEngine):
    """Returns a WAV of exactly `seconds` regardless of text."""

    def __init__(self, seconds: float, *, raise_on_call: bool = False) -> None:
        self.seconds = seconds
        self.raise_on_call = raise_on_call
        self.calls = 0

    def synthesize(self, text, *, voice="default", emotion="",
                   speed=1.0, **kwargs) -> SynthesizedAudio:
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("engine boom")
        return SynthesizedAudio(
            audio=_make_wav_bytes(self.seconds),
            sample_rate=22050,
            format="wav",
        )

    def available_voices(self):
        return ["default"]


class _StubConfig(ContentConfig):
    content_type: str = "podcast"


class _StubConductor(BaseConductor):
    def _build_agents(self):
        pass

    def _run_turn_loop(self):
        return []

    def _package_result(self, raw_events):
        return ConductorResult(config=self.config, events=raw_events)


def _make_conductor(*, checkpoint=None, tracker=None):
    return _StubConductor(
        _StubConfig(content_type="podcast"),
        checkpoint_renderer=checkpoint,
        budget_tracker=tracker,
    )


# ── No checkpoint configured ─────────────────────────────────────


class TestNoCheckpoint:
    def test_returns_none_when_no_renderer(self):
        cond = _make_conductor()
        assert cond._checkpoint_segment(0, "hello world") is None


# ── With checkpoint, no tracker ──────────────────────────────────


class TestWithCheckpoint:
    def test_measures_text_and_returns_seconds(self):
        engine = _FixedDurationEngine(seconds=4.0)
        renderer = CheckpointRenderer(engine, default_voice="default")
        cond = _make_conductor(checkpoint=renderer)
        result = cond._checkpoint_segment(0, "some text here")
        assert result == pytest.approx(4.0, abs=0.05)
        assert engine.calls == 1

    def test_empty_text_returns_zero_no_engine_call(self):
        engine = _FixedDurationEngine(seconds=4.0)
        renderer = CheckpointRenderer(engine, default_voice="default")
        cond = _make_conductor(checkpoint=renderer)
        assert cond._checkpoint_segment(0, "") == 0.0
        assert cond._checkpoint_segment(1, "   \n  ") == 0.0
        assert engine.calls == 0

    def test_engine_failure_recorded_as_warning(self):
        engine = _FixedDurationEngine(seconds=4.0, raise_on_call=True)
        renderer = CheckpointRenderer(engine, default_voice="default")
        cond = _make_conductor(checkpoint=renderer)
        result = cond._checkpoint_segment(0, "hello")
        assert result is None
        assert any("measurement failed" in w for w in cond._length_warnings)


# ── With both checkpoint + tracker ───────────────────────────────


class TestWithBudgetTracker:
    def test_records_measurement_into_tracker(self):
        engine = _FixedDurationEngine(seconds=5.0)
        renderer = CheckpointRenderer(engine, default_voice="default")
        tracker = BudgetTracker(total_target_seconds=60.0, num_segments=10)
        cond = _make_conductor(checkpoint=renderer, tracker=tracker)
        cond._checkpoint_segment(0, "some text")
        assert len(tracker.records) == 1
        assert tracker.records[0].actual_seconds == pytest.approx(5.0, abs=0.05)

    def test_subsequent_targets_rebalance(self):
        # Total 60s / 10 segments = 6s nominal. First runs short at 3s.
        # Remaining 9 segments should share 57s = 6.33s each.
        engine = _FixedDurationEngine(seconds=3.0)
        renderer = CheckpointRenderer(engine, default_voice="default")
        tracker = BudgetTracker(total_target_seconds=60.0, num_segments=10)
        cond = _make_conductor(checkpoint=renderer, tracker=tracker)
        cond._checkpoint_segment(0, "one segment of text")
        next_target = tracker.segment_target_seconds(1)
        assert next_target > 6.0
        assert next_target == pytest.approx(57.0 / 9, abs=0.05)

    def test_out_of_range_index_recorded_as_warning(self):
        engine = _FixedDurationEngine(seconds=3.0)
        renderer = CheckpointRenderer(engine, default_voice="default")
        tracker = BudgetTracker(total_target_seconds=60.0, num_segments=2)
        cond = _make_conductor(checkpoint=renderer, tracker=tracker)
        cond._checkpoint_segment(99, "out of bounds")
        assert any("tracker rejected" in w for w in cond._length_warnings)
