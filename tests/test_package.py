"""Tests for the vox_romana.package YouTube packaging module.

Covers:
  - captions.py: _srt_ts, make_srt (spoken events, non-spoken, fallback, wrap)
  - notes_agent.py: _fmt_youtube_ts, extract_chapters, generate_show_notes
  - packager.py: package_episode (SRT + notes written, missing key, errors)
  - run.py: --mode package CLI wiring

Run with:
    PYTHONPATH=. .venv/bin/pytest tests/test_package.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.package.captions import _srt_ts, make_srt
from vox_romana.package.notes_agent import (
    ChapterMark,
    _fmt_youtube_ts,
    extract_chapters,
    generate_show_notes,
)
from vox_romana.package.packager import PackageResult, package_episode


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures / helpers
# ═══════════════════════════════════════════════════════════════════════════


def _event(
    text: str,
    speaker: str = "Alice",
    kind: str = "utterance",
    segment_id: str | None = None,
) -> dict:
    return {
        "text": text,
        "speaker_name": speaker,
        "event_kind": kind,
        "segment_id": segment_id,
    }


def _news_data(segments: list[dict] | None = None) -> dict:
    """Build a minimal news-show JSON dict."""
    segments = segments or [
        {
            "segment_id": "seg-1",
            "segment_order": 0,
            "headline": "AI Revolution",
            "events": [_event("AI is here.", segment_id="seg-1")],
        },
        {
            "segment_id": "seg-2",
            "segment_order": 1,
            "headline": "Economy Update",
            "events": [_event("Markets rise.", segment_id="seg-2")],
        },
    ]
    all_events = []
    for seg in segments:
        for ev in seg.get("events", []):
            all_events.append(dict(ev))  # ensure segment_id propagated

    return {
        "title": "Evening News",
        "show_type": "news",
        "topic": "Current events",
        "tone": "neutral",
        "anchors": ["Bob Anchor"],
        "segments": segments,
        "events": all_events,
    }


def _debate_data() -> dict:
    """Build a minimal debate show JSON dict (flat events, no segments)."""
    return {
        "title": "UBI Debate",
        "show_type": "debate",
        "topic": "Universal Basic Income",
        "events": [
            _event("We need UBI.", speaker="Alice"),
            _event("No we don't.", speaker="Bob"),
        ],
    }


def _show_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "show.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════════
# 1. captions._srt_ts
# ═══════════════════════════════════════════════════════════════════════════


class TestSrtTimestamp:
    def test_zero(self):
        assert _srt_ts(0.0) == "00:00:00,000"

    def test_one_second(self):
        assert _srt_ts(1.0) == "00:00:01,000"

    def test_fractional(self):
        assert _srt_ts(1.5) == "00:00:01,500"

    def test_minutes(self):
        assert _srt_ts(90.0) == "00:01:30,000"

    def test_hours(self):
        assert _srt_ts(3661.0) == "01:01:01,000"

    def test_rounding(self):
        # 1.001 s → 1001 ms
        assert _srt_ts(1.001) == "00:00:01,001"

    def test_negative_clamped(self):
        # Should not produce negative timestamps.
        ts = _srt_ts(-0.5)
        assert ts.startswith("00:00:00")


# ═══════════════════════════════════════════════════════════════════════════
# 2. captions.make_srt
# ═══════════════════════════════════════════════════════════════════════════


class TestMakeSrt:
    def test_empty_events(self):
        assert make_srt([], {}) == ""

    def test_single_utterance_correct_seq_and_ts(self):
        events = [_event("Hello world")]
        durations = {0: 2.5}
        srt = make_srt(events, durations)
        assert srt.startswith("1\n")
        assert "00:00:00,000 --> 00:00:02,500" in srt
        assert "Alice: Hello world" in srt

    def test_multiple_events_sequential(self):
        events = [_event("First."), _event("Second.")]
        durations = {0: 1.0, 1: 2.0}
        srt = make_srt(events, durations)
        assert "1\n" in srt
        assert "2\n" in srt
        # Second event starts at 1.0 s
        assert "00:00:01,000 --> 00:00:03,000" in srt

    def test_non_spoken_event_advances_clock_no_caption(self):
        events = [
            _event("", kind="break"),
            _event("After break.", speaker="Carol"),
        ]
        durations = {0: 0.5, 1: 3.0}
        srt = make_srt(events, durations)
        # Only one caption (the break has no text / is non-spoken)
        assert "1\n" in srt
        assert "2\n" not in srt
        # After-break caption should start at 0.5 s
        assert "00:00:00,500 --> 00:00:03,500" in srt

    def test_fallback_timing_used_when_duration_missing(self):
        events = [_event("No duration.", speaker="Dave")]
        # No entry in durations → fallback used (3.0 s by default)
        srt = make_srt(events, {})
        assert "00:00:00,000 --> 00:00:03,000" in srt

    def test_custom_fallback_secs(self):
        events = [_event("Fallback.", speaker="Eve")]
        srt = make_srt(events, {}, fallback_secs=5.0)
        assert "00:00:00,000 --> 00:00:05,000" in srt

    def test_long_line_wrapped(self):
        long_text = "A " * 50  # 100 chars of "A "
        events = [_event(long_text.strip(), speaker="Frank")]
        srt = make_srt(events, {0: 4.0}, max_caption_chars=30)
        # Should contain a newline inside the caption block
        block = srt.split("\n\n")[0]  # first (and only) SRT block
        lines = block.split("\n")
        # lines: seq, timecode, caption line 1, caption line 2
        assert len(lines) >= 4

    def test_no_speaker_name(self):
        events = [{"text": "Narrator voice.", "event_kind": "narration"}]
        srt = make_srt(events, {0: 2.0})
        # Should NOT prepend ": " when no speaker
        assert "Narrator voice." in srt
        assert ": " not in srt

    def test_bumper_event_not_captioned(self):
        events = [_event("", kind="bumper"), _event("After bumper.")]
        srt = make_srt(events, {0: 0.5, 1: 2.0})
        assert srt.count("\n00:00:") == 1  # only one timecode line

    def test_srt_entries_separated_by_blank_line(self):
        events = [_event("One."), _event("Two.")]
        durations = {0: 1.0, 1: 1.0}
        srt = make_srt(events, durations)
        assert "\n\n" in srt


# ═══════════════════════════════════════════════════════════════════════════
# 3. notes_agent._fmt_youtube_ts
# ═══════════════════════════════════════════════════════════════════════════


class TestFmtYoutubeTs:
    def test_zero(self):
        assert _fmt_youtube_ts(0) == "0:00"

    def test_under_a_minute(self):
        assert _fmt_youtube_ts(45) == "0:45"

    def test_minutes(self):
        assert _fmt_youtube_ts(90) == "1:30"

    def test_hour(self):
        assert _fmt_youtube_ts(3723) == "1:02:03"

    def test_negative_clamped(self):
        assert _fmt_youtube_ts(-5) == "0:00"


# ═══════════════════════════════════════════════════════════════════════════
# 4. notes_agent.extract_chapters
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractChapters:
    def test_no_segments_returns_single_chapter(self):
        chapters = extract_chapters(_debate_data(), {})
        assert len(chapters) == 1
        assert chapters[0].timestamp == "0:00"
        assert "UBI Debate" in chapters[0].title

    def test_news_segments_produce_chapters_with_headlines(self):
        data = _news_data()
        # 2 events, event 0 = seg-1, event 1 = seg-2
        durations = {0: 4.0, 1: 3.0}
        chapters = extract_chapters(data, durations)
        assert chapters[0].timestamp == "0:00"
        assert "AI Revolution" in chapters[0].title
        # Second segment starts at 4.0 s
        assert chapters[1].timestamp == "0:04"
        assert "Economy Update" in chapters[1].title

    def test_first_chapter_always_at_zero(self):
        """Even if the first segment starts after some non-spoken events."""
        data = _news_data()
        # Inject a leading break event (advances clock but no segment_id)
        data["events"].insert(
            0, {"text": "", "event_kind": "break", "segment_id": None}
        )
        # durations: break=0.5, seg-1 event=4.0, seg-2 event=3.0
        durations = {0: 0.5, 1: 4.0, 2: 3.0}
        chapters = extract_chapters(data, durations)
        # YouTube requires 0:00 as first chapter
        assert chapters[0].timestamp == "0:00"

    def test_fallback_timing_for_missing_durations(self):
        data = _news_data()
        # No durations at all → fallback 3.0 s per spoken event
        chapters = extract_chapters(data, {})
        assert len(chapters) == 2
        assert chapters[0].timestamp == "0:00"
        # Second segment should start at ~3.0 s (fallback for first event)
        assert chapters[1].timestamp == "0:03"

    def test_debate_chunks_produce_chapters(self):
        data = {
            "title": "Chunk Debate",
            "show_type": "debate",
            "chunks": [
                {"chunk_id": "c1", "title": "Opening", "events": []},
                {"chunk_id": "c2", "title": "Rebuttal", "events": []},
            ],
            "events": [
                {"text": "Hello.", "speaker_name": "A", "event_kind": "utterance", "segment_id": "c1"},
                {"text": "Nope.", "speaker_name": "B", "event_kind": "utterance", "segment_id": "c2"},
            ],
        }
        durations = {0: 2.0, 1: 2.0}
        chapters = extract_chapters(data, durations)
        titles = [c.title for c in chapters]
        assert "Opening" in titles
        assert "Rebuttal" in titles


# ═══════════════════════════════════════════════════════════════════════════
# 5. notes_agent.generate_show_notes
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateShowNotes:
    def test_returns_client_response(self):
        client = MagicMock()
        client.chat.return_value = "Great show today about AI."
        chapters = [ChapterMark("0:00", "Introduction"), ChapterMark("1:30", "Main Topic")]
        result = generate_show_notes(_debate_data(), chapters, client=client)
        assert result == "Great show today about AI."

    def test_sends_system_and_user_messages(self):
        client = MagicMock()
        client.chat.return_value = "description"
        chapters = [ChapterMark("0:00", "Intro")]
        generate_show_notes(_news_data(), chapters, client=client)
        messages = client.chat.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_title_and_topic(self):
        client = MagicMock()
        client.chat.return_value = "x"
        chapters = [ChapterMark("0:00", "Intro")]
        generate_show_notes(_news_data(), chapters, client=client)
        user_msg = client.chat.call_args.args[0][1]["content"]
        assert "Evening News" in user_msg
        assert "Current events" in user_msg

    def test_user_message_includes_chapter_timestamps(self):
        client = MagicMock()
        client.chat.return_value = "x"
        chapters = [
            ChapterMark("0:00", "Intro"),
            ChapterMark("2:30", "Deep Dive"),
        ]
        generate_show_notes(_debate_data(), chapters, client=client)
        user_msg = client.chat.call_args.args[0][1]["content"]
        assert "0:00 Intro" in user_msg
        assert "2:30 Deep Dive" in user_msg

    def test_passes_temperature(self):
        client = MagicMock()
        client.chat.return_value = "x"
        generate_show_notes(
            _debate_data(), [], client=client, temperature=0.3
        )
        assert client.chat.call_args.kwargs.get("temperature") == 0.3


# ═══════════════════════════════════════════════════════════════════════════
# 6. packager.package_episode
# ═══════════════════════════════════════════════════════════════════════════


class TestPackageEpisode:
    def test_srt_file_created(self, tmp_path):
        json_path = _show_json(tmp_path, _debate_data())
        result = package_episode(json_path, {}, generate_notes=False)
        assert result.srt_path is not None
        assert result.srt_path.exists()
        assert result.srt_path.suffix == ".srt"

    def test_srt_content_contains_dialogue(self, tmp_path):
        json_path = _show_json(tmp_path, _debate_data())
        result = package_episode(json_path, {0: 2.0, 1: 2.0}, generate_notes=False)
        content = result.srt_path.read_text()
        assert "We need UBI." in content
        assert "No we don't." in content

    def test_notes_file_created_when_client_supplied(self, tmp_path):
        client = MagicMock()
        client.chat.return_value = "Amazing debate about UBI!"
        json_path = _show_json(tmp_path, _debate_data())
        result = package_episode(json_path, {}, generate_notes=True, client=client)
        assert result.notes_path is not None
        assert result.notes_path.exists()
        assert result.notes_path.name.endswith("_show_notes.txt")
        assert "Amazing debate" in result.notes_path.read_text()

    def test_no_notes_when_generate_notes_false(self, tmp_path):
        json_path = _show_json(tmp_path, _debate_data())
        result = package_episode(json_path, {}, generate_notes=False)
        assert result.notes_path is None

    def test_no_notes_when_client_is_none(self, tmp_path):
        json_path = _show_json(tmp_path, _debate_data())
        result = package_episode(json_path, {}, generate_notes=True, client=None)
        assert result.notes_path is None

    def test_custom_dest_dir(self, tmp_path):
        json_path = _show_json(tmp_path, _debate_data())
        dest = tmp_path / "pkg_out"
        result = package_episode(json_path, {}, dest_dir=dest, generate_notes=False)
        assert result.srt_path.parent == dest

    def test_missing_json_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            package_episode(tmp_path / "nonexistent.json", {}, generate_notes=False)

    def test_none_event_durations_uses_fallback(self, tmp_path):
        """Passing None for event_durations must not crash."""
        json_path = _show_json(tmp_path, _debate_data())
        result = package_episode(json_path, None, generate_notes=False)
        assert result.srt_path is not None
        assert result.srt_path.exists()

    def test_notes_generation_error_does_not_crash(self, tmp_path):
        """If the LLM call raises, package_episode still returns a result."""
        client = MagicMock()
        client.chat.side_effect = RuntimeError("API down")
        json_path = _show_json(tmp_path, _debate_data())
        result = package_episode(json_path, {}, generate_notes=True, client=client)
        assert result.srt_path is not None  # SRT should still be written
        assert result.notes_path is None    # notes silently skipped


# ═══════════════════════════════════════════════════════════════════════════
# 7. run.py -- mode package CLI wiring
# ═══════════════════════════════════════════════════════════════════════════


class TestRunPyPackageMode:
    def test_package_mode_requires_script_input(self, capsys, tmp_path):
        """--mode package without --script-input should exit with code 1."""
        import run as run_mod
        with pytest.raises(SystemExit) as exc:
            run_mod.main(["--mode", "package"])
        assert exc.value.code == 1
        assert "--script-input" in capsys.readouterr().err

    def test_package_mode_nonexistent_file_exits_one(self, capsys, tmp_path):
        import run as run_mod
        with pytest.raises(SystemExit) as exc:
            run_mod.main(["--mode", "package", "--script-input", str(tmp_path / "ghost.json")])
        assert exc.value.code == 1

    def test_package_mode_calls_package_episode(self, tmp_path, monkeypatch):
        """--mode package with a valid JSON should call package_episode."""
        json_path = _show_json(tmp_path, _debate_data())

        called_with: list[dict] = []

        def fake_package(json_path, event_durations, dest_dir=None, *, generate_notes, client):
            called_with.append({"json_path": json_path, "generate_notes": generate_notes})
            return PackageResult(
                srt_path=json_path.with_suffix(".srt"),
                notes_path=None,
            )

        monkeypatch.setattr("vox_romana.package.packager.package_episode", fake_package)

        import run as run_mod
        # Patch _mode_package to use our monkeypatched packager
        with patch("vox_romana.package.package_episode", fake_package):
            run_mod.main(["--mode", "package", "--script-input", str(json_path), "--no-package-notes", "--quiet"])

        assert len(called_with) == 1
        assert called_with[0]["json_path"] == json_path

    def test_package_flag_parsed_correctly(self, tmp_path):
        """--package flag should be True in the parsed namespace."""
        import run as run_mod
        # We just need to confirm parse_args sets package=True.
        # Monkeypatch main dispatch to not actually run anything.
        # Use the parse_args directly:
        from run import parse_args
        args = parse_args([str(tmp_path / "x.txt"), "--package", "--quiet"])
        assert args.package is True
        assert args.quiet is True

    def test_no_package_notes_flag_parsed(self, tmp_path):
        from run import parse_args
        args = parse_args(["--mode", "package", "--no-package-notes"])
        assert args.no_package_notes is True

    def test_package_output_flag_parsed(self, tmp_path):
        from run import parse_args
        args = parse_args(["--mode", "package", "--package-output", str(tmp_path)])
        assert args.package_output == tmp_path
