"""Tests for run.py — the master CLI pipeline.

Tests cover:
    1. Argument parsing (modes, flags, defaults)
    2. Validation (missing prompt, missing --script-input)
    3. _build_generate_argv() flag forwarding
    4. _build_render_argv() flag forwarding
    5. generate mode (mocked run_show.main)
    6. render mode (mocked render_show.main)
    7. full mode (mocked both)
    8. --no-log-audio-duration forwarding
    9. --quiet output behavior

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
    _build_generate_argv,
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
        """Default engine should be kokoro."""
        args = parse_args([str(tmp_prompt)])
        assert args.engine == "kokoro"

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
    """Verify the argv builder helpers."""

    def test_generate_argv_minimal(self, tmp_prompt):
        """Minimal generate argv should contain the prompt but NOT --quiet.

        --quiet is only forwarded when the user explicitly passes --quiet;
        run_show.py should stream its own progress output by default.
        """
        args = parse_args([str(tmp_prompt)])
        argv = _build_generate_argv(args)
        assert str(tmp_prompt) in argv
        assert "--quiet" not in argv

    def test_generate_argv_with_overrides(self, tmp_prompt):
        """All overrides should appear in the generated argv."""
        args = parse_args([
            str(tmp_prompt),
            "--type", "debate",
            "--tag", "test",
            "--turns", "40",
            "--duration", "20",
        ])
        argv = _build_generate_argv(args)
        assert "--type" in argv
        assert "debate" in argv
        assert "--tag" in argv
        assert "test" in argv
        assert "--turns" in argv
        assert "40" in argv
        assert "--duration" in argv
        assert "20.0" in argv  # float is stringified with decimal

    def test_render_argv_minimal(self, tmp_json):
        """Minimal render argv should include the JSON path and defaults."""
        args = parse_args(["--mode", "render", "--script-input", str(tmp_json)])
        argv = _build_render_argv(tmp_json, args)
        assert str(tmp_json) in argv
        assert "--engine" in argv
        assert "kokoro" in argv

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

    @patch("run_show.main")
    def test_generate_mode_calls_run_show(self, mock_main, tmp_prompt, capsys):
        """generate mode should call run_show.main()."""
        # Make run_show.main() print a fake path (simulating --quiet).
        mock_main.side_effect = lambda argv: print("/fake/output.json")
        run_master.main([str(tmp_prompt), "--mode", "generate", "--quiet"])
        mock_main.assert_called_once()
        # The argv passed to run_show should contain --quiet.
        call_argv = mock_main.call_args[0][0]
        assert "--quiet" in call_argv

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
    @patch("run_show.main")
    def test_full_mode_calls_both(self, mock_gen, mock_render, tmp_prompt, capsys):
        """full mode should call run_show then render_show."""
        from pathlib import Path
        fake_json = "/fake/output.json"
        # run_show.main() now returns a Path rather than printing to stdout.
        mock_gen.return_value = Path(fake_json)
        mock_render.return_value = None

        run_master.main([str(tmp_prompt), "--mode", "full", "--quiet"])

        # Both should have been called.
        mock_gen.assert_called_once()
        mock_render.assert_called_once()

        # render_show should receive the JSON path from run_show's return value.
        render_argv = mock_render.call_args[0][0]
        assert fake_json in render_argv

    @patch("render_show.main")
    @patch("run_show.main")
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
