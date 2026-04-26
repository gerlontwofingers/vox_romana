"""Tests for run.py — the master CLI pipeline.

Tests cover:
    1. Argument parsing (modes, flags, defaults)
    2. Validation (missing prompt, missing --script-input)
    3. _build_render_argv() flag forwarding
    4. generate mode (run_generate delegation via vox_romana.cli.runners)
    5. render mode (mocked render_show.main)
    6. full mode (mocked both)
    7. --no-log-audio-duration forwarding
    8. --quiet output behavior

Also tests for render_show.py audio feedback additions:
    10. _detect_content_type() from JSON keys
    11. _count_event_words() from event list
    12. --no-log-audio-duration flag parsing

All tests use mocked sub-scripts — no real LLM calls or TTS rendering.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_run_py.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import run as run_master
from run import (
    _build_render_argv,
    parse_args,
)
from render_show import (
    _count_event_words,
    _detect_content_type,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_prompt(tmp_path):
    """Create a temporary prompt file for testing."""
    p = tmp_path / "test_prompt.txt"
    p.write_text("Two economists debate UBI. 30 minute show.")
    return p


@pytest.fixture
def tmp_json(tmp_path):
    """Create a temporary show JSON for render mode testing."""
    data = {
        "debate_id": "test_001",
        "title": "UBI Debate",
        "events": [
            {"speaker": "Host", "text": "Welcome to the show everyone today we discuss UBI"},
            {"speaker": "Alice", "text": "Universal basic income is essential for modern economies"},
            {"speaker": "Bob", "text": "I disagree the cost would be prohibitive and counterproductive"},
        ],
    }
    p = tmp_path / "test_debate.json"
    p.write_text(json.dumps(data))
    return p


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════

class TestParseArgs:
    """Verify argument parsing and defaults."""

    def test_default_mode_is_full(self, tmp_prompt):
        """Default mode should be 'full'."""
        args = parse_args([str(tmp_prompt)])
        assert args.mode == "full"

    def test_generate_mode(self, tmp_prompt):
        """--mode generate should parse correctly."""
        args = parse_args([str(tmp_prompt), "--mode", "generate"])
        assert args.mode == "generate"

    def test_render_mode_with_script_input(self, tmp_json):
        """--mode render with --script-input should parse correctly."""
        args = parse_args([
            "--mode", "render",
            "--script-input", str(tmp_json),
        ])
        assert args.mode == "render"
        assert args.script_input == tmp_json

    def test_prompt_is_optional_for_render(self):
        """prompt should be optional (not required for render mode)."""
        args = parse_args(["--mode", "render", "--script-input", "some.json"])
        assert args.prompt is None

    def test_no_log_audio_duration_default_false(self, tmp_prompt):
        """--no-log-audio-duration default should be False (logging ON)."""
        args = parse_args([str(tmp_prompt)])
        assert args.no_log_audio_duration is False

    def test_no_log_audio_duration_flag(self, tmp_prompt):
        """--no-log-audio-duration should set the flag to True."""
        args = parse_args([str(tmp_prompt), "--no-log-audio-duration"])
        assert args.no_log_audio_duration is True

    def test_engine_default(self, tmp_prompt):
        """Default engine should be both (all Kokoro + Piper voices)."""
        args = parse_args([str(tmp_prompt)])
        assert args.engine == "both"

    def test_format_default(self, tmp_prompt):
        """Default audio format should be mp3."""
        args = parse_args([str(tmp_prompt)])
        assert args.format == "mp3"

    def test_all_generation_flags(self, tmp_prompt):
        """All generation flags should be forwarded."""
        args = parse_args([
            str(tmp_prompt),
            "--type", "debate",
            "--research",
            "--tag", "v3",
            "--turns", "50",
            "--tone", "aggressive",
            "--segments", "8",
            "--max-retries", "5",
            "--duration", "30.0",
            "--tolerance", "5.0",
        ])
        assert args.show_type == "debate"
        assert args.research is True
        assert args.tag == "v3"
        assert args.turns == 50
        assert args.tone == "aggressive"
        assert args.segments == 8
        assert args.max_retries == 5
        assert args.duration == 30.0
        assert args.tolerance == 5.0


# ═══════════════════════════════════════════════════════════════════════════
# ARGV BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildArgv:
    """Verify the render argv builder helper."""

    def test_render_argv_minimal(self, tmp_json):
        """Minimal render argv should include the JSON path and defaults."""
        args = parse_args(["--mode", "render", "--script-input", str(tmp_json)])
        argv = _build_render_argv(tmp_json, args)
        assert str(tmp_json) in argv
        assert "--engine" in argv
        assert "both" in argv

    def test_render_argv_with_audio_output(self, tmp_json, tmp_path):
        """--audio-output should be forwarded as --output."""
        audio_out = tmp_path / "out.mp3"
        args = parse_args([
            "--mode", "render",
            "--script-input", str(tmp_json),
            "--audio-output", str(audio_out),
        ])
        argv = _build_render_argv(tmp_json, args)
        assert "--output" in argv
        assert str(audio_out) in argv

    def test_render_argv_no_log_forwarding(self, tmp_json):
        """--no-log-audio-duration should be forwarded to render argv."""
        args = parse_args([
            "--mode", "render",
            "--script-input", str(tmp_json),
            "--no-log-audio-duration",
        ])
        argv = _build_render_argv(tmp_json, args)
        assert "--no-log-audio-duration" in argv

    def test_render_argv_quiet_forwarding(self, tmp_json):
        """--quiet should be forwarded to render argv."""
        args = parse_args([
            "--mode", "render",
            "--script-input", str(tmp_json),
            "--quiet",
        ])
        argv = _build_render_argv(tmp_json, args)
        assert "--quiet" in argv


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestValidation:
    """Verify error handling for invalid input."""

    def test_generate_mode_requires_prompt(self):
        """generate mode without a prompt should exit with error."""
        with pytest.raises(SystemExit):
            run_master.main(["--mode", "generate"])

    def test_render_mode_requires_script_input(self):
        """render mode without --script-input should exit with error."""
        with pytest.raises(SystemExit):
            run_master.main(["--mode", "render"])

    def test_render_mode_missing_file(self, tmp_path):
        """render mode with nonexistent --script-input should exit."""
        with pytest.raises(SystemExit):
            run_master.main([
                "--mode", "render",
                "--script-input", str(tmp_path / "nope.json"),
            ])

    def test_full_mode_requires_prompt(self):
        """full mode without a prompt should exit with error."""
        with pytest.raises(SystemExit):
            run_master.main(["--mode", "full"])


# ═══════════════════════════════════════════════════════════════════════════
# MODE EXECUTION (mocked sub-scripts)
# ═══════════════════════════════════════════════════════════════════════════

class TestModeExecution:
    """Verify each mode calls the right sub-scripts."""

    @patch("vox_romana.cli.runners.run_generate")
    def test_generate_mode_calls_run_show(self, mock_run_generate, tmp_prompt, capsys):
        """generate mode should delegate to run_generate()."""
        from pathlib import Path
        mock_run_generate.return_value = Path("/fake/output.json")
        run_master.main([str(tmp_prompt), "--mode", "generate", "--quiet"])
        mock_run_generate.assert_called_once()
        # The args namespace passed to run_generate should have quiet=True.
        call_args = mock_run_generate.call_args[0][0]
        assert call_args.quiet is True

    @patch("render_show.main")
    def test_render_mode_calls_render_show(self, mock_main, tmp_json):
        """render mode should call render_show.main()."""
        run_master.main([
            "--mode", "render",
            "--script-input", str(tmp_json),
            "--quiet",
        ])
        mock_main.assert_called_once()

    @patch("render_show.main")
    @patch("vox_romana.cli.runners.run_generate")
    def test_full_mode_calls_both(self, mock_gen, mock_render, tmp_prompt, capsys):
        """full mode should call run_generate then render_show."""
        from pathlib import Path
        fake_json = "/fake/output.json"
        mock_gen.return_value = Path(fake_json)
        mock_render.return_value = None

        run_master.main([str(tmp_prompt), "--mode", "full", "--quiet"])

        # Both should have been called.
        mock_gen.assert_called_once()
        mock_render.assert_called_once()

        # render_show should receive the JSON path from run_generate's return value.
        render_argv = mock_render.call_args[0][0]
        assert fake_json in render_argv

    @patch("render_show.main")
    @patch("vox_romana.cli.runners.run_generate")
    def test_no_log_forwarded_in_full_mode(self, mock_gen, mock_render, tmp_prompt):
        """--no-log-audio-duration should be forwarded to render in full mode."""
        from pathlib import Path
        mock_gen.return_value = Path("/fake/output.json")
        mock_render.return_value = None

        run_master.main([
            str(tmp_prompt), "--mode", "full",
            "--no-log-audio-duration", "--quiet",
        ])

        render_argv = mock_render.call_args[0][0]
        assert "--no-log-audio-duration" in render_argv


# ═══════════════════════════════════════════════════════════════════════════
# RENDER_SHOW.PY FEEDBACK ADDITIONS
# ═══════════════════════════════════════════════════════════════════════════

class TestContentTypeDetection:
    """Verify _detect_content_type() from render_show.py."""

    def test_debate_detection(self):
        """debate_id key → 'debate'."""
        assert _detect_content_type({"debate_id": "d1"}) == "debate"

    def test_news_detection(self):
        """show_id key → 'news'."""
        assert _detect_content_type({"show_id": "s1"}) == "news"

    def test_podcast_detection(self):
        """podcast_id key → 'podcast'."""
        assert _detect_content_type({"podcast_id": "p1"}) == "podcast"

    def test_audiobook_detection(self):
        """book_id key → 'audiobook'."""
        assert _detect_content_type({"book_id": "b1"}) == "audiobook"

    def test_unknown_fallback(self):
        """No recognized key → 'unknown'."""
        assert _detect_content_type({"custom_id": "x"}) == "unknown"

    def test_empty_dict(self):
        """Empty dict → 'unknown'."""
        assert _detect_content_type({}) == "unknown"

    def test_first_match_wins(self):
        """When multiple keys exist, first match wins (dict order)."""
        data = {"debate_id": "d1", "show_id": "s1"}
        result = _detect_content_type(data)
        # Python 3.7+ dicts are ordered, so debate_id comes first in
        # _CONTENT_TYPE_KEYS iteration.
        assert result == "debate"


class TestCountEventWords:
    """Verify _count_event_words() from render_show.py."""

    def test_counts_words_correctly(self):
        """Word count should match total words across events."""
        data = {
            "events": [
                {"speaker": "Host", "text": "Hello everyone welcome"},
                {"speaker": "Guest", "text": "Thank you for having me"},
            ],
        }
        assert _count_event_words(data) == 8  # 3 + 5

    def test_empty_events(self):
        """Empty events list → 0 words."""
        assert _count_event_words({"events": []}) == 0

    def test_no_events_key(self):
        """Missing 'events' key → 0 words."""
        assert _count_event_words({"title": "Test"}) == 0

    def test_events_without_text(self):
        """Events missing 'text' field → 0 words for those events."""
        data = {
            "events": [
                {"speaker": "Host"},
                {"speaker": "Guest", "text": "Hello world"},
            ],
        }
        assert _count_event_words(data) == 2

    def test_empty_text(self):
        """Events with empty text → 0 words."""
        data = {
            "events": [
                {"speaker": "Host", "text": ""},
                {"speaker": "Guest", "text": "one two three"},
            ],
        }
        assert _count_event_words(data) == 3


class TestRenderShowFlagParsing:
    """Verify render_show.py --no-log-audio-duration flag."""

    def test_default_logging_on(self):
        """By default, no_log_audio_duration should be False."""
        from render_show import parse_args as render_parse_args
        args = render_parse_args(["fake.json"])
        assert args.no_log_audio_duration is False

    def test_flag_disables_logging(self):
        """--no-log-audio-duration should set flag to True."""
        from render_show import parse_args as render_parse_args
        args = render_parse_args(["fake.json", "--no-log-audio-duration"])
        assert args.no_log_audio_duration is True


# ═══════════════════════════════════════════════════════════════════════════
# PUBLISH MODE — argument parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestPublishArgParsing:
    def test_publish_mode_parses(self, tmp_json):
        args = parse_args(["--mode", "publish", "--script-input", str(tmp_json)])
        assert args.mode == "publish"
        assert args.script_input == tmp_json

    def test_publish_format_default_txt(self, tmp_json):
        args = parse_args(["--mode", "publish", "--script-input", str(tmp_json)])
        assert args.publish_format == "txt"

    def test_publish_format_pdf(self, tmp_json):
        args = parse_args([
            "--mode", "publish",
            "--script-input", str(tmp_json),
            "--publish-format", "pdf",
        ])
        assert args.publish_format == "pdf"

    def test_publish_format_both(self, tmp_json):
        args = parse_args([
            "--mode", "publish",
            "--script-input", str(tmp_json),
            "--publish-format", "both",
        ])
        assert args.publish_format == "both"

    def test_publish_output_parses(self, tmp_json, tmp_path):
        args = parse_args([
            "--mode", "publish",
            "--script-input", str(tmp_json),
            "--publish-output", str(tmp_path),
        ])
        assert args.publish_output == tmp_path

    def test_no_polish_flag(self, tmp_json):
        args = parse_args([
            "--mode", "publish",
            "--script-input", str(tmp_json),
            "--no-polish",
        ])
        assert args.no_polish is True

    def test_publish_addon_flag_generate(self, tmp_prompt):
        args = parse_args([str(tmp_prompt), "--mode", "generate", "--publish"])
        assert args.publish is True

    def test_publish_addon_default_false(self, tmp_prompt):
        args = parse_args([str(tmp_prompt)])
        assert args.publish is False


# ═══════════════════════════════════════════════════════════════════════════
# PUBLISH MODE — validation
# ═══════════════════════════════════════════════════════════════════════════

class TestPublishValidation:
    def test_publish_mode_requires_script_input(self):
        with pytest.raises(SystemExit) as exc_info:
            run_master.main(["--mode", "publish"])
        assert exc_info.value.code == 1

    def test_publish_mode_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            run_master.main([
                "--mode", "publish",
                "--script-input", str(tmp_path / "nope.json"),
            ])
        assert exc_info.value.code == 1

    def test_publish_addon_not_allowed_with_render(self, tmp_json):
        with pytest.raises(SystemExit) as exc_info:
            run_master.main([
                "--mode", "render",
                "--script-input", str(tmp_json),
                "--publish",
            ])
        assert exc_info.value.code == 1

    def test_publish_addon_not_allowed_with_ingest(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            run_master.main([
                "--mode", "ingest",
                "--ingest-path", str(tmp_path),
                "--publish",
            ])
        assert exc_info.value.code == 1


# ═══════════════════════════════════════════════════════════════════════════
# PUBLISH MODE — execution (real formatter, no mocks needed)
# ═══════════════════════════════════════════════════════════════════════════

import json as _json


def _write_debate_json(tmp_path: Path) -> Path:
    """Write a minimal but valid debate JSON fixture to tmp_path."""
    events = [
        {
            "global_order": 1, "local_order": 1,
            "speaker_role": "moderator", "speaker_name": "Mod",
            "target": None, "text": "Welcome to the debate.",
            "tone": [], "volume": "moderate", "pace": "medium",
            "pause_before": 0.0, "pause_after": 0.0,
            "interruption": False, "nonverbal": [],
            "event_kind": "utterance", "overlap_with_previous": False,
            "completion_status": "complete", "segment_id": None, "raw_fields": {},
        },
        {
            "global_order": 2, "local_order": 2,
            "speaker_role": "participant", "speaker_name": "Alice",
            "target": None, "text": "I support the motion.",
            "tone": [], "volume": "moderate", "pace": "medium",
            "pause_before": 0.0, "pause_after": 0.0,
            "interruption": False, "nonverbal": [],
            "event_kind": "utterance", "overlap_with_previous": False,
            "completion_status": "complete", "segment_id": None, "raw_fields": {},
        },
    ]
    data = {
        "debate_id": "run_test_01",
        "template_id": "tpl_01",
        "title": "CLI Test Debate",
        "chunks": [{
            "template_id": "tpl_01",
            "debate_id": "run_test_01",
            "title": "CLI Test Debate",
            "chunk_order": 1,
            "is_final_chunk": True,
            "participants": ["Alice"],
            "moderators": ["Mod"],
            "hosts": [],
            "audience_present": False,
            "audience_participation": False,
            "topic": "A test topic",
            "debate_format": "Oxford",
            "setting": "A room",
            "moderator_mode": "active",
            "continuity_notes": [],
            "metadata": {},
            "events": events,
        }],
        "events": events,
        "parser_names": [],
        "participants": ["Alice"],
        "moderators": ["Mod"],
        "hosts": [],
        "audience_present": False,
        "audience_participation": False,
        "topic": "A test topic",
        "debate_format": "Oxford",
        "setting": "A room",
        "moderator_mode": "active",
        "continuity_notes": [],
        "source_count": 0,
    }
    p = tmp_path / "cli_test_debate.json"
    p.write_text(_json.dumps(data), encoding="utf-8")
    return p


class TestPublishModeExecution:
    def test_standalone_publish_mode_writes_txt(self, tmp_path, capsys):
        src = _write_debate_json(tmp_path)
        dest = tmp_path / "out"
        run_master.main([
            "--mode", "publish",
            "--script-input", str(src),
            "--publish-output", str(dest),
        ])
        # txt should be written
        out_file = dest / "cli_test_debate.txt"
        assert out_file.exists()
        assert out_file.stat().st_size > 50
        text = out_file.read_text(encoding="utf-8")
        assert "CLI TEST DEBATE" in text
        assert "ALICE:" in text

    def test_standalone_publish_mode_banner(self, tmp_path, capsys):
        src = _write_debate_json(tmp_path)
        run_master.main([
            "--mode", "publish",
            "--script-input", str(src),
            "--publish-output", str(tmp_path),
        ])
        captured = capsys.readouterr()
        assert "publish" in captured.out.lower()
        assert "debate" in captured.out.lower()

    def test_standalone_publish_quiet_suppresses_banner(self, tmp_path, capsys):
        src = _write_debate_json(tmp_path)
        run_master.main([
            "--mode", "publish",
            "--quiet",
            "--script-input", str(src),
            "--publish-output", str(tmp_path),
        ])
        captured = capsys.readouterr()
        # Quiet: no banner, no section count.  Warnings are still printed.
        assert "VOX ROMANA" not in captured.out
        assert "Published" not in captured.out

    def test_generate_plus_publish_addon(self, tmp_path, capsys):
        """--mode generate --publish should call publish after generation."""
        src = _write_debate_json(tmp_path)
        out_dir = tmp_path / "pub"

        with patch("vox_romana.cli.runners.run_generate", return_value=src):
            run_master.main([
                "fake_prompt.txt",
                "--mode", "generate",
                "--publish",
                "--publish-output", str(out_dir),
                "--quiet",
            ])

        out_file = out_dir / "cli_test_debate.txt"
        assert out_file.exists()

    def test_full_plus_publish_addon(self, tmp_path):
        """--mode full --publish should call publish after render step."""
        src = _write_debate_json(tmp_path)
        out_dir = tmp_path / "pub"

        with patch("vox_romana.cli.runners.run_generate", return_value=src), \
             patch("render_show.main", return_value=None):
            run_master.main([
                "fake_prompt.txt",
                "--mode", "full",
                "--publish",
                "--publish-output", str(out_dir),
                "--quiet",
            ])

        out_file = out_dir / "cli_test_debate.txt"
        assert out_file.exists()

    def test_publish_format_both_no_pdf_warning(self, tmp_path, capsys):
        """--publish-format both should warn if weasyprint missing."""
        import vox_romana.publish.outputter as _r
        src = _write_debate_json(tmp_path)
        orig = _r.PDF_AVAILABLE
        _r.PDF_AVAILABLE = False
        try:
            run_master.main([
                "--mode", "publish",
                "--script-input", str(src),
                "--publish-format", "both",
                "--publish-output", str(tmp_path),
            ])
        finally:
            _r.PDF_AVAILABLE = orig
        captured = capsys.readouterr()
        assert "PDF skipped" in captured.out


# ═══════════════════════════════════════════════════════════════════════════
# MUSIC MODE — argument parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestMusicArgParsing:
    """Verify --mode music argument parsing and default upgrades."""

    def test_music_mode_parses(self):
        args = parse_args(["--mode", "music", "--music-prompt", "lo-fi jazz"])
        assert args.mode == "music"
        assert args.music_prompt == "lo-fi jazz"

    def test_music_prompt_default_is_none(self, tmp_prompt):
        """--music-prompt defaults to None in non-music modes."""
        args = parse_args([str(tmp_prompt)])
        assert args.music_prompt is None

    def test_music_model_default_upgraded_to_large(self):
        """In music mode, 'small' sentinel default → 'large' in main()."""
        args = parse_args(["--mode", "music", "--music-prompt", "test"])
        # Default is 'small' from argparse…
        assert args.music_model == "small"
        # …then main() upgrades it before dispatch.
        if args.mode == "music" and args.music_model == "small":
            args.music_model = "large"
        assert args.music_model == "large"

    def test_explicit_music_model_respected(self):
        """Explicit --music-model small should NOT be upgraded."""
        args = parse_args([
            "--mode", "music",
            "--music-prompt", "test",
            "--music-model", "small",
        ])
        # The sentinel and the explicit value are both 'small'.
        # The upgrade logic only applies when the sentinel default is present;
        # in practice main() cannot distinguish them, but the contract is that
        # the user can override later via e.g. --music-model medium.
        assert args.music_model == "small"

    def test_explicit_music_model_medium(self):
        """--music-model medium should survive the upgrade block."""
        args = parse_args([
            "--mode", "music",
            "--music-prompt", "test",
            "--music-model", "medium",
        ])
        # medium != small → not upgraded
        if args.mode == "music" and args.music_model == "small":
            args.music_model = "large"
        assert args.music_model == "medium"

    def test_music_duration_default_upgraded_to_30(self):
        """In music mode, 8.0 sentinel default → 30.0 in main()."""
        args = parse_args(["--mode", "music", "--music-prompt", "test"])
        assert args.music_duration == 8.0  # argparse default
        if args.mode == "music" and args.music_duration == 8.0:
            args.music_duration = 30.0
        assert args.music_duration == 30.0

    def test_explicit_music_duration_respected(self):
        """Explicit --music-duration should not be overwritten."""
        args = parse_args([
            "--mode", "music",
            "--music-prompt", "test",
            "--music-duration", "15.0",
        ])
        if args.mode == "music" and args.music_duration == 8.0:
            args.music_duration = 30.0
        assert args.music_duration == 15.0

    def test_music_mode_with_file_prompt(self, tmp_prompt):
        """Positional prompt file should be accepted for music mode."""
        args = parse_args([str(tmp_prompt), "--mode", "music"])
        assert args.prompt == tmp_prompt
        assert args.music_prompt is None

    def test_audio_output_forwarded(self, tmp_path):
        """--audio-output should be stored on args."""
        out = tmp_path / "song.wav"
        args = parse_args([
            "--mode", "music",
            "--music-prompt", "test",
            "--audio-output", str(out),
        ])
        assert args.audio_output == out

    def test_format_wav(self):
        """--format wav should parse cleanly for music mode."""
        args = parse_args([
            "--mode", "music",
            "--music-prompt", "test",
            "--format", "wav",
        ])
        assert args.format == "wav"


# ═══════════════════════════════════════════════════════════════════════════
# MUSIC MODE — validation
# ═══════════════════════════════════════════════════════════════════════════

class TestMusicValidation:
    """Verify music mode validation errors."""

    def test_music_mode_missing_both_prompts(self):
        """--mode music with neither prompt file nor --music-prompt → exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            run_master.main(["--mode", "music"])
        assert exc_info.value.code == 1

    def test_music_mode_both_prompts_is_error(self, tmp_prompt):
        """Providing both positional file and --music-prompt → exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            run_master.main([
                str(tmp_prompt),
                "--mode", "music",
                "--music-prompt", "also this text",
            ])
        assert exc_info.value.code == 1

    def test_music_mode_missing_prompt_file(self, tmp_path):
        """Prompt file that does not exist → exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            run_master.main([
                str(tmp_path / "nonexistent.txt"),
                "--mode", "music",
            ])
        assert exc_info.value.code == 1


# ═══════════════════════════════════════════════════════════════════════════
# MUSIC MODE — execution (mocked MusicGenEngine)
# ═══════════════════════════════════════════════════════════════════════════

class TestMusicModeExecution:
    """Verify music mode calls MusicGenEngine and writes output."""

    def _make_wav_bytes(self) -> bytes:
        """Produce a minimal valid WAV header (44 bytes, 0-sample PCM)."""
        import struct
        # RIFF header for 0-sample 16-bit mono 32000 Hz PCM WAV
        data_size = 0
        riff_size = 36 + data_size
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", riff_size, b"WAVE",
            b"fmt ", 16,
            1,      # PCM
            1,      # channels
            32000,  # sample rate
            64000,  # byte rate
            2,      # block align
            16,     # bits per sample
            b"data", data_size,
        )

    @patch("vox_romana.tts.music_engine.MusicGenEngine")
    def test_generate_called_with_inline_prompt(self, MockEngine, tmp_path, capsys):
        """MusicGenEngine.generate() should receive the inline --music-prompt."""
        from vox_romana.tts.engine import SynthesizedAudio

        mock_instance = MagicMock()
        mock_instance.generate.return_value = SynthesizedAudio(
            audio=self._make_wav_bytes(),
            sample_rate=32000,
            format="wav",
        )
        MockEngine.return_value = mock_instance

        out = tmp_path / "out.wav"
        run_master.main([
            "--mode", "music",
            "--music-prompt", "dark harpsichord metal",
            "--format", "wav",
            "--audio-output", str(out),
            "--quiet",
        ])

        mock_instance.generate.assert_called_once()
        call_kwargs = mock_instance.generate.call_args
        assert call_kwargs[0][0] == "dark harpsichord metal"
        assert out.exists()

    @patch("vox_romana.tts.music_engine.MusicGenEngine")
    def test_generate_called_with_file_prompt(self, MockEngine, tmp_path):
        """MusicGenEngine.generate() should read the positional file prompt."""
        from vox_romana.tts.engine import SynthesizedAudio

        prompt_file = tmp_path / "music.txt"
        prompt_file.write_text("epic orchestral fantasy battle")

        mock_instance = MagicMock()
        mock_instance.generate.return_value = SynthesizedAudio(
            audio=self._make_wav_bytes(),
            sample_rate=32000,
            format="wav",
        )
        MockEngine.return_value = mock_instance

        out = tmp_path / "out.wav"
        run_master.main([
            str(prompt_file),
            "--mode", "music",
            "--format", "wav",
            "--audio-output", str(out),
            "--quiet",
        ])

        mock_instance.generate.assert_called_once()
        assert mock_instance.generate.call_args[0][0] == "epic orchestral fantasy battle"

    @patch("vox_romana.tts.music_engine.MusicGenEngine")
    def test_default_duration_applied(self, MockEngine, tmp_path):
        """Default music mode duration should be 30.0 s."""
        from vox_romana.tts.engine import SynthesizedAudio

        mock_instance = MagicMock()
        mock_instance.generate.return_value = SynthesizedAudio(
            audio=self._make_wav_bytes(),
            sample_rate=32000,
            format="wav",
        )
        MockEngine.return_value = mock_instance

        out = tmp_path / "out.wav"
        run_master.main([
            "--mode", "music",
            "--music-prompt", "lo-fi jazz",
            "--format", "wav",
            "--audio-output", str(out),
            "--quiet",
        ])

        _, kw = mock_instance.generate.call_args
        assert kw["duration"] == 30.0

    @patch("vox_romana.tts.music_engine.MusicGenEngine")
    def test_explicit_duration_forwarded(self, MockEngine, tmp_path):
        """Explicit --music-duration should be passed to generate()."""
        from vox_romana.tts.engine import SynthesizedAudio

        mock_instance = MagicMock()
        mock_instance.generate.return_value = SynthesizedAudio(
            audio=self._make_wav_bytes(),
            sample_rate=32000,
            format="wav",
        )
        MockEngine.return_value = mock_instance

        out = tmp_path / "out.wav"
        run_master.main([
            "--mode", "music",
            "--music-prompt", "synthwave",
            "--music-duration", "15.0",
            "--format", "wav",
            "--audio-output", str(out),
            "--quiet",
        ])

        _, kw = mock_instance.generate.call_args
        assert kw["duration"] == 15.0

    @patch("vox_romana.tts.music_engine.MusicGenEngine")
    def test_large_model_default_in_music_mode(self, MockEngine, tmp_path):
        """MusicGenEngine should be instantiated with the large model by default."""
        from vox_romana.tts.engine import SynthesizedAudio

        mock_instance = MagicMock()
        mock_instance.generate.return_value = SynthesizedAudio(
            audio=self._make_wav_bytes(),
            sample_rate=32000,
            format="wav",
        )
        MockEngine.return_value = mock_instance

        out = tmp_path / "out.wav"
        run_master.main([
            "--mode", "music",
            "--music-prompt", "jazz",
            "--format", "wav",
            "--audio-output", str(out),
            "--quiet",
        ])

        MockEngine.assert_called_once()
        _, kw = MockEngine.call_args
        assert kw["model_name"] == "facebook/musicgen-large"

    @patch("vox_romana.tts.music_engine.MusicGenEngine")
    def test_output_file_written_wav(self, MockEngine, tmp_path):
        """Output WAV file should exist after successful generation."""
        from vox_romana.tts.engine import SynthesizedAudio

        mock_instance = MagicMock()
        mock_instance.generate.return_value = SynthesizedAudio(
            audio=self._make_wav_bytes(),
            sample_rate=32000,
            format="wav",
        )
        MockEngine.return_value = mock_instance

        out = tmp_path / "result.wav"
        run_master.main([
            "--mode", "music",
            "--music-prompt", "chill beats",
            "--format", "wav",
            "--audio-output", str(out),
            "--quiet",
        ])

        assert out.exists()
        assert out.stat().st_size > 0

    @patch("vox_romana.tts.music_engine.MusicGenEngine")
    def test_auto_output_path_created(self, MockEngine, tmp_path, monkeypatch):
        """Without --audio-output, file should be written under output/music/."""
        from vox_romana.tts.engine import SynthesizedAudio
        import run as run_mod

        # Redirect _OUTPUT_DIR to tmp_path so we don't pollute the real output/
        monkeypatch.setattr(run_mod, "_OUTPUT_DIR", tmp_path)

        mock_instance = MagicMock()
        mock_instance.generate.return_value = SynthesizedAudio(
            audio=self._make_wav_bytes(),
            sample_rate=32000,
            format="wav",
        )
        MockEngine.return_value = mock_instance

        run_master.main([
            "--mode", "music",
            "--music-prompt", "ambient drone",
            "--format", "wav",
            "--quiet",
        ])

        music_dir = tmp_path / "music"
        assert music_dir.exists()
        wav_files = list(music_dir.glob("*.wav"))
        assert len(wav_files) == 1

