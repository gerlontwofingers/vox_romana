"""Live integration tests for the generate pipeline.

These tests call ``run.main()`` with a **real** LLM (DeepSeek).  They are
skipped by default and run only when ``--run-integration`` is passed to pytest:

    pytest --run-integration tests/test_integration_live.py -v

Requirements:
  - ``DEEPSEEK_API_KEY`` set in ``.env`` or the environment.

These tests are NOT run in CI.  Use them for local smoke-testing after
major changes to the generate pipeline.

Design goals:
  - Each test is minimal: 1 segment / 2 turns, no research, no fact-check.
  - Assertions are structural only (keys exist, events non-empty) — no
    LLM-output quality checks.
  - Each test is independent: separate tmp_path, no shared state.
  - Total elapsed time should be well under 60 s per test on a fast API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run

pytestmark = pytest.mark.integration  # skipped unless --run-integration

# ─────────────────────────────────────────────────────────────────────────────
# Fixture paths (tiny prompts — keep LLM output small)
# ─────────────────────────────────────────────────────────────────────────────

_FIXTURES = Path(__file__).parent / "fixtures"
_NEWS_BRIEF      = _FIXTURES / "live_news_smoke.txt"
_DEBATE_BRIEF    = _FIXTURES / "live_debate_smoke.txt"
_PODCAST_BRIEF   = _FIXTURES / "live_podcast_smoke.txt"
_AUDIOBOOK_BRIEF = _FIXTURES / "live_audiobook_smoke.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate(argv: list[str]) -> dict:
    """Run ``run.main(argv)`` and return the parsed output JSON.

    Expects ``--output`` and ``--name`` to be in ``argv`` so the output
    path is predictable.  Fails the test if the JSON does not exist after
    the call.
    """
    run.main(argv)

    # Extract --output and --name from argv to reconstruct the path.
    out_dir: Path | None = None
    name: str | None = None
    for i, arg in enumerate(argv):
        if arg == "--output" and i + 1 < len(argv):
            out_dir = Path(argv[i + 1])
        if arg == "--name" and i + 1 < len(argv):
            name = argv[i + 1]

    assert out_dir is not None, "--output must be in argv"
    assert name is not None, "--name must be in argv"

    json_path = out_dir / f"{name}.json"
    assert json_path.exists(), f"Expected output JSON at {json_path}"
    return json.loads(json_path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_live_generate_news(tmp_path):
    """News pipeline: prompt → show JSON with show_id and events."""
    data = _generate([
        str(_NEWS_BRIEF),
        "--mode", "generate",
        "--type", "news",
        "--segments", "1",
        "--fact-check", "off",
        "--no-research",
        "--output", str(tmp_path),
        "--name", "live_news",
        "--quiet",
    ])

    assert "show_id" in data, f"Missing show_id. Keys: {list(data)}"
    assert data.get("events"), "News show JSON should contain at least one event"
    # Structural sanity: every event should have a text field.
    for evt in data["events"]:
        assert "text" in evt, f"Event missing 'text': {evt}"


def test_live_generate_debate(tmp_path):
    """Debate pipeline: prompt → debate JSON with debate_id and events."""
    data = _generate([
        str(_DEBATE_BRIEF),
        "--mode", "generate",
        "--type", "debate",
        "--turns", "2",
        "--fact-check", "off",
        "--no-research",
        "--output", str(tmp_path),
        "--name", "live_debate",
        "--quiet",
    ])

    assert "debate_id" in data, f"Missing debate_id. Keys: {list(data)}"
    assert data.get("events"), "Debate JSON should contain at least one event"
    for evt in data["events"]:
        assert "text" in evt, f"Event missing 'text': {evt}"


def test_live_generate_podcast(tmp_path):
    """Podcast pipeline: prompt → podcast JSON with podcast_id and events."""
    data = _generate([
        str(_PODCAST_BRIEF),
        "--mode", "generate",
        "--type", "podcast",
        "--segments", "1",
        "--fact-check", "off",
        "--no-research",
        "--output", str(tmp_path),
        "--name", "live_podcast",
        "--quiet",
    ])

    assert "podcast_id" in data, f"Missing podcast_id. Keys: {list(data)}"
    assert data.get("events"), "Podcast JSON should contain at least one event"
    for evt in data["events"]:
        assert "text" in evt, f"Event missing 'text': {evt}"


def test_live_generate_audiobook(tmp_path):
    """Audiobook pipeline: prompt → audiobook JSON with audiobook_id and events."""
    data = _generate([
        str(_AUDIOBOOK_BRIEF),
        "--mode", "generate",
        "--type", "audiobook",
        "--fact-check", "off",
        "--no-research",
        "--output", str(tmp_path),
        "--name", "live_audiobook",
        "--quiet",
    ])

    assert "audiobook_id" in data, f"Missing audiobook_id. Keys: {list(data)}"
    # Audiobook stores events nested inside chapters; check top-level chapters.
    assert data.get("chapters"), "Audiobook JSON should contain at least one chapter"
    chapter_events = [
        evt
        for ch in data["chapters"]
        for evt in ch.get("events", [])
    ]
    assert chapter_events, "Audiobook chapters should contain at least one event"
    for evt in chapter_events:
        assert "text" in evt, f"Event missing 'text': {evt}"
