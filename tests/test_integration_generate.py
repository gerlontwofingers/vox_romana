"""Integration tests for the generate pipeline using a SmartMockClient.

These exercise ``run.main(["--mode","generate", ...])`` end-to-end with
no network calls and no real LLM. ``SmartMockClient`` stands in for
``DeepSeekClient`` and inspects each request's system prompt to return a
plausible canned JSON response for the right pipeline stage.

The point isn't to validate LLM quality — it's to make sure the generator
plumbing (router → bootstrap → conductor → serializer) wires up cleanly,
parses every response, and writes a valid show JSON to disk.

Tested combinations (kept tiny — ``--segments 1 --fact-check off``):
- News show, type forced via ``--type news`` (skips router)
- News show with router classification (no ``--type``)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run


# ─────────────────────────────────────────────────────────────────────────────
# SmartMockClient — routes ``client.chat`` by system-prompt substring.
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (substring_in_system_prompt, response_json_or_str).
# First match wins. The substrings come from the actual prompt constants in
# vox_romana/core/router.py, vox_romana/core/fact_checker.py,
# vox_romana/shows/news/{bootstrap,prompts}.py, and conductor inline prompts.

_ROUTER_RESPONSE = {
    "show_type": "news",
    "confidence": 0.95,
    "needs_research": False,
    "reasoning": "Brief mentions headlines and an anchor — clearly a news broadcast.",
}

_FACT_EXTRACT_RESPONSE: list = []  # No claims to verify.
_FACT_VERIFY_RESPONSE: list = []

_ARCHITECT_RESPONSE = {
    "title": "The Smoke Test Brief",
    "topic": "Integration testing for vox_romana",
    "tone": "authoritative",
    "setting": "modern news studio",
    "target_segments": 1,
    "anchors": [
        {"name": "Sarah Chen", "style": "authoritative, precise"},
    ],
    "segments": [
        {
            "headline": "Mock Pipeline Reaches Green",
            "angle": "Engineering team validates end-to-end generation flow.",
            "key_facts": [
                "SmartMockClient routes calls by system-prompt substring.",
                "Pipeline produces valid show JSON without hitting any network.",
            ],
            "tone": "upbeat",
            "duration_hint": "short",
            "segment_type": "anchor_read",
            "target_beats": 1,
        },
    ],
    "show_host": {
        "name": "James Mitchell",
        "role": "narrator",
        "style": "warm, engaging",
    },
}

_PROFILER_RESPONSE = {
    "voice_gender": "female",
    "voice_hint": "clear alto, 30s, steady and precise, network news cadence",
    "voice_fingerprint": (
        "Speaks with measured authority. Pauses between major points, uses crisp "
        "consonants, projects quiet confidence — NPR cadence, never rushed."
    ),
}

_DIRECTOR_RESPONSE = {
    "show_arc": "Open with the lead, deliver the report, wrap warmly.",
    "mood_progression": ["urgent", "analytical", "warm"],
    "segment_order_notes": "Single segment — straightforward read.",
    "closing_strategy": "Anchor delivers the takeaway, host signs off warmly.",
}

_ANCHOR_DELIVERY_RESPONSE = {
    "text": (
        "In integration news tonight: the smart mock pipeline has reached green. "
        "Engineers report that every stage of the generator wires up cleanly, "
        "parses every response, and writes a valid show file to disk."
    ),
    "tone": "authoritative",
    "volume": "moderate",
    "pace": "medium",
    "pause_before": "short",
    "pause_after": "medium",
    "word_count": 38,
}

_HOST_INTRO_RESPONSE = {
    "text": (
        "Welcome to the Smoke Test Brief. Tonight: a quick check-in on the "
        "vox_romana generator pipeline. Let's get straight to the report."
    ),
    "tone": "warm",
    "volume": "moderate",
    "pace": "medium",
    "pause_before": "none",
    "pause_after": "long",
}

_DIRECTOR_CHECK_RESPONSE = {
    "mood_note": "Hold the analytical tone — we have one segment to land.",
    "nudge": "Keep it tight; one beat is enough.",
    "wrap_up": True,
}

_CONTEXT_SUMMARY_RESPONSE = (
    "Anchor delivered the lead segment. Tone steady and measured throughout."
)

# ─────────────────────────────────────────────────────────────────────────────
# Debate canned responses
# ─────────────────────────────────────────────────────────────────────────────

_DEBATE_ARCHITECT_RESPONSE = {
    "title": "Mock Debate: Test Pipeline Green",
    "topic": "Does the integration test pipeline produce valid debate JSON?",
    "debate_format": "crossfire",
    "setting": "university lecture hall",
    "tone": "academic but spirited",
    "target_turns": 2,
    "participants": [
        {"name": "Dr. Alpha", "side": "pro-testing", "role": "participant"},
        {"name": "Dr. Beta", "side": "anti-testing", "role": "participant"},
    ],
    "moderator": None,
    "audience": {"present": False, "participation": False, "size_description": "empty room"},
    "show_host": {"name": "Morgan Webb", "role": "narrator", "style": "warm and concise"},
}

_DEBATE_PROFILER_RESPONSE = {
    "voice_gender": "female",
    "voice_hint": "clear alto, 40s, academic authority",
    "tts_style": "Speak with measured precision.",
    "voice_fingerprint": "Uses complete sentences and ends arguments with rhetorical questions.",
    "behavioral_triggers": "Becomes more clipped when interrupted.",
    "emotional_arc": "Starts calm. Grows more passionate mid-debate. Ends with a clear thesis.",
    "key_positions": ["Integration tests reduce regression risk.", "Mocking external APIs is best practice."],
    "factual_claims": [],
    "tangent_topics": ["Relates everything to the scientific method."],
}

_DEBATE_DIRECTOR_RESPONSE = {
    "dramatic_arc": "Act 1 (turns 1-2): brief opening statements. Quick conclusions.",
    "mood_progression": ["engaged", "analytical"],
    "key_moments": ["Turn 2: participants reach unexpected common ground."],
    "closing_strategy": "Each participant gives a one-turn closing statement.",
}

_DEBATE_PARTICIPANT_TURN_RESPONSE = {
    "text": (
        "The integration test pipeline has reached green. "
        "This validates our end-to-end architecture."
    ),
    "tone": "analytical",
    "volume": "moderate",
    "pace": "medium",
    "target": "all",
    "pause_before": "short",
    "pause_after": "medium",
    "interruption": False,
    "nonverbal": [],
}

_DEBATE_HOST_RESPONSE = {
    "text": (
        "Welcome to this crossfire debate. Tonight, Dr. Alpha and Dr. Beta "
        "will explore the integration test pipeline."
    ),
    "tone": "warm",
    "volume": "moderate",
    "pace": "medium",
    "pause_before": "none",
    "pause_after": "long",
}

_DEBATE_DIRECTOR_CHECK_RESPONSE = {
    "mood_note": "Hold the analytical tone.",
    "nudge": "Move to closing statements.",
    "new_topic_suggestion": None,
    "repetition_warning": None,
    "wrap_up": True,
}

# ─────────────────────────────────────────────────────────────────────────────
# Podcast canned responses
# ─────────────────────────────────────────────────────────────────────────────

_PODCAST_ARCHITECT_RESPONSE = {
    "title": "Smoke Test Episode",
    "show_name": "The Pipeline Podcast",
    "topic": "Integration testing in software development",
    "tone": "casual and curious",
    "hosts": [
        {
            "name": "Alex",
            "gender": "female",
            "role": "lead host",
            "speech_style": "enthusiastic and curious",
            "expertise_domains": ["software testing"],
            "forbidden_knowledge": ["enterprise sales"],
            "perspective": "pragmatist",
        }
    ],
    "segments": [
        {
            "title": "The green run",
            "angle": "What does a passing test suite feel like?",
            "key_facts": ["The mock pipeline produced valid JSON on first run."],
            "tone": "celebratory",
            "duration_hint": "short",
            "target_exchanges": 1,
        }
    ],
}

_PODCAST_PROFILER_RESPONSE = {
    "voice_hint": "warm alto, late 20s, California laid-back energy",
    "tts_style": "Speak with infectious enthusiasm and quick pacing.",
    "voice_fingerprint": "Starts sentences with 'So...' Asks lots of follow-up questions.",
    "behavioral_triggers": "Gets excited when someone makes a clever point.",
    "emotional_arc": "Starts excited, becomes focused during analysis, ends warmly.",
    "key_positions": ["Testing is a form of documentation.", "Mocks should be realistic but fast."],
    "tangent_topics": ["Always connects software to cooking metaphors."],
}

_PODCAST_DIRECTOR_RESPONSE = {
    "arc": "One short segment, celebratory energy throughout.",
    "mood_progression": ["excited", "reflective"],
    "segment_notes": {},
    "closing_strategy": "Alex gives a one-sentence takeaway and signs off.",
}

# Podcast delivery expects a JSON *array* of utterance objects.
_PODCAST_HOST_DELIVERY_RESPONSE = [
    {
        "speaker_name": "Alex",
        "text": (
            "So, the integration test pipeline just hit green for the first time. "
            "That means SmartMockClient is routing every stage correctly."
        ),
        "tone": "excited",
        "volume": "moderate",
        "pace": "fast",
    }
]

_PODCAST_DIRECTOR_CHECK_RESPONSE = {
    "mood_note": "Keep the celebratory energy.",
    "nudge": "Wrap up warmly.",
    "wrap_up": True,
}

_PODCAST_CONTEXT_SUMMARY = (
    "Alex celebrated the green pipeline run. Energy was high throughout."
)

# ─────────────────────────────────────────────────────────────────────────────
# Audiobook canned responses
# ─────────────────────────────────────────────────────────────────────────────

_AUDIOBOOK_ARCHITECT_RESPONSE = {
    "title": "The Smoke Test Chronicles",
    "genre": "science fiction",
    "topic": "A heroic integration test pipeline achieving green status",
    "tone": "contemplative",
    "setting": "a vast digital landscape",
    "narrator": {"name": "The Archivist", "style": "calm, measured, documentary"},
    "characters": [{"name": "SmartMock", "role": "protagonist", "voice_gender": "female"}],
    "chapters": [
        {
            "title": "Chapter One: Green",
            "synopsis": "The integration tests pass. The pipeline is validated.",
            "key_points": [
                "SmartMock routes the first call correctly.",
                "The JSON is written to disk.",
                "The assertion passes.",
            ],
            "tone": "triumphant",
            "duration_hint": "short",
            "target_beats": 1,
            "characters": ["SmartMock"],
        }
    ],
}

_NARRATOR_PROFILER_RESPONSE = {
    "voice_fingerprint": (
        "Delivers with quiet authority. Uses long, flowing sentences. "
        "Pauses meaningfully between paragraphs."
    ),
}

_AUDIOBOOK_DIRECTOR_RESPONSE = {
    "narrative_arc": "Single-chapter triumph. Open with context, deliver the victory, close reflectively.",
    "mood_progression": ["anticipatory", "triumphant"],
    "chapter_notes": "One chapter — maintain steady contemplative pace.",
    "closing_strategy": "The narrator closes with a reflective sentence about the green run.",
}

_NARRATOR_DELIVERY_RESPONSE = {
    "text": (
        "In the beginning, there was the mock. And the mock was good. "
        "It routed every call, parsed every response, and wrote the show JSON to disk "
        "without a single network call."
    ),
    "tone": "contemplative",
    "volume": "moderate",
    "pace": "slow",
    "pause_before": "long",
    "pause_after": "long",
}

_AUDIOBOOK_DIRECTOR_CHECK_RESPONSE = {
    "mood_note": "Stay contemplative and measured.",
    "nudge": "Build toward the quiet triumph at the end.",
}


# Routing table — first matching substring wins. The needle is matched
# against the FULL concatenated text of every message in the chat call,
# so it can hit either the system prompt or the user content.
# ORDER MATTERS: more-specific substrings must appear before any
# less-specific substring that could match the same text.
_ROUTES: list[tuple[str, object]] = [
    # Router (vox_romana/core/router.py).
    ("show type classifier", _ROUTER_RESPONSE),
    # FactChecker stages (skipped when --fact-check off, but routed for safety).
    ("fact-extraction specialist", _FACT_EXTRACT_RESPONSE),
    ("fact-verification specialist", _FACT_VERIFY_RESPONSE),

    # ── DEBATE ───────────────────────────────────────────────────────────────
    # Bootstrap: Architect + Director + ProfileBuilder.
    ("Architect for a debate generation system", _DEBATE_ARCHITECT_RESPONSE),
    ("Character Psychologist for a debate generation system", _DEBATE_PROFILER_RESPONSE),
    ("Director for a debate generation system", _DEBATE_DIRECTOR_RESPONSE),
    # Conductor: participant turns (PARTICIPANT_SYSTEM_PROMPT contains this).
    ("YOUR KEY POSITIONS", _DEBATE_PARTICIPANT_TURN_RESPONSE),
    # Conductor: show-host events (SHOW_HOST_SYSTEM_PROMPT after format).
    ("produced audio show called", _DEBATE_HOST_RESPONSE),
    # Conductor: director check (inline system prompt).
    ("debate director. Respond with JSON only.", _DEBATE_DIRECTOR_CHECK_RESPONSE),

    # ── PODCAST ──────────────────────────────────────────────────────────────
    # Bootstrap: Architect + Director + HostProfiler.
    ("Architect for a podcast generation system", _PODCAST_ARCHITECT_RESPONSE),
    ("Character Designer for a podcast generation system", _PODCAST_PROFILER_RESPONSE),
    ("Director for a podcast episode", _PODCAST_DIRECTOR_RESPONSE),
    # Conductor: host delivery (PODCAST_HOST_SYSTEM_PROMPT after format).
    ("podcast called", _PODCAST_HOST_DELIVERY_RESPONSE),
    # Conductor: director check + context summary (both inline).
    ("podcast Director. Respond with JSON only.", _PODCAST_DIRECTOR_CHECK_RESPONSE),
    ("summarize podcast conversations concisely", _PODCAST_CONTEXT_SUMMARY),

    # ── AUDIOBOOK ────────────────────────────────────────────────────────────
    # Bootstrap: Architect + Director + NarratorProfiler.
    ("Architect for an audiobook generation system", _AUDIOBOOK_ARCHITECT_RESPONSE),
    # More specific profiler first, before the generic "creating a voice profile".
    ("creating a voice profile for an audiobook narrator", _NARRATOR_PROFILER_RESPONSE),
    ("Director for an audiobook production", _AUDIOBOOK_DIRECTOR_RESPONSE),
    # Conductor: narrator delivery (NARRATOR_SYSTEM_PROMPT after format).
    ("narrating an audiobook", _NARRATOR_DELIVERY_RESPONSE),
    # Conductor: director check (inline system prompt).
    ("narrative director. Respond with JSON only.", _AUDIOBOOK_DIRECTOR_CHECK_RESPONSE),

    # ── NEWS ─────────────────────────────────────────────────────────────────
    # Bootstrap: Architect.
    ("Architect for a news show generation system", _ARCHITECT_RESPONSE),
    # Bootstrap: Director (must come before any plain "broadcast director" match).
    ("Director for a news broadcast", _DIRECTOR_RESPONSE),
    # Bootstrap: anchor + host profilers — system prompt is generic, but the
    # user content always contains the ANCHOR_PROFILER_PROMPT body.
    # Generic "creating a voice profile" must come AFTER the audiobook-specific one.
    ("creating a voice profile", _PROFILER_RESPONSE),
    ("produce JSON voice profiles", _PROFILER_RESPONSE),
    # Conductor: anchor delivery + show-host events.
    ("news anchor delivering a broadcast", _ANCHOR_DELIVERY_RESPONSE),
    ("show host of a produced audio", _HOST_INTRO_RESPONSE),
    # Conductor: periodic director-check (inline system prompt).
    ("broadcast director", _DIRECTOR_CHECK_RESPONSE),
    # Conductor: context-window compression summary (shared by news + debate).
    ("concise summarizer", _CONTEXT_SUMMARY_RESPONSE),
]


def _serialize(payload: object) -> str:
    """Return ``payload`` as a string, JSON-encoding dicts/lists."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)


class SmartMockClient:
    """Stub for :class:`vox_romana.core.client.DeepSeekClient`.

    Inspects ``messages[0]["content"]`` (the system prompt) on each
    ``.chat`` call and returns a canned response for the matching
    pipeline stage. Unmatched prompts return an empty JSON object so the
    pipeline's lenient parsers fall back to defaults instead of crashing.

    Every call is recorded on ``self.calls`` for post-test assertions.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], **kwargs) -> str:
        system_prompt = messages[0]["content"] if messages else ""
        full_text = "\n".join(
            str(m.get("content", "")) for m in (messages or [])
        )
        matched: str | None = None
        response: object = {}
        for needle, payload in _ROUTES:
            if needle in full_text:
                matched = needle
                response = payload
                break

        self.calls.append(
            {
                "matched": matched,
                "system_head": system_prompt[:120],
                "kwargs": kwargs,
            }
        )

        return _serialize(response)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_news_brief(tmp_path: Path) -> Path:
    """Write a minimal prompt file and return its path."""
    brief = tmp_path / "smoke_news_brief.txt"
    brief.write_text(
        "A 1-segment evening news bulletin reporting that the integration "
        "test pipeline reached green. Authoritative tone, single anchor.\n",
        encoding="utf-8",
    )
    return brief


def _patch_clients(monkeypatch: pytest.MonkeyPatch) -> SmartMockClient:
    """Replace every ``DeepSeekClient`` patch site used by the generate path.

    ``run_generate`` constructs the client at ``vox_romana.cli.runners`` and
    sub-modules import their own reference (bootstrap, fact_checker). We
    patch the symbol *only* where the generate path actually instantiates
    it — that's the runners module and the news bootstrap module.

    Returns the shared ``SmartMockClient`` instance so tests can assert on
    ``.calls``.
    """
    shared = SmartMockClient()

    def _factory(*args, **kwargs):
        return shared

    # run_generate constructs the client here.
    monkeypatch.setattr("vox_romana.cli.runners.DeepSeekClient", _factory)
    return shared


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_news_show_type_forced(tmp_path, monkeypatch):
    """Forced ``--type news`` skips the router; bootstrap+conductor still run."""
    shared = _patch_clients(monkeypatch)
    brief = _write_news_brief(tmp_path)
    out_dir = tmp_path / "out"

    run.main(
        [
            str(brief),
            "--mode", "generate",
            "--type", "news",
            "--segments", "1",
            "--fact-check", "off",
            "--no-research",
            "--output", str(out_dir),
            "--name", "smoke_news",
            "--quiet",
        ]
    )

    json_path = out_dir / "smoke_news.json"
    assert json_path.exists(), (
        f"Expected {json_path} to exist. SmartMockClient calls: "
        f"{[c['matched'] for c in shared.calls]}"
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "show_id" in data
    assert data.get("title") == "The Smoke Test Brief"
    assert data.get("events"), "show JSON should contain at least one event"

    # Router must NOT have been called when --type was forced.
    matched_stages = [c["matched"] for c in shared.calls]
    assert "show type classifier" not in matched_stages

    # Architect must have run.
    assert "Architect for a news show generation system" in matched_stages


def test_generate_news_router_classifies(tmp_path, monkeypatch):
    """Without ``--type``, the router runs and classifies the brief as news."""
    shared = _patch_clients(monkeypatch)
    brief = _write_news_brief(tmp_path)
    out_dir = tmp_path / "out"

    run.main(
        [
            str(brief),
            "--mode", "generate",
            "--segments", "1",
            "--fact-check", "off",
            "--no-research",
            "--output", str(out_dir),
            "--name", "smoke_news_routed",
            "--quiet",
        ]
    )

    json_path = out_dir / "smoke_news_routed.json"
    assert json_path.exists(), (
        f"Expected {json_path} to exist. SmartMockClient calls: "
        f"{[c['matched'] for c in shared.calls]}"
    )

    matched_stages = [c["matched"] for c in shared.calls]
    # Router ran, then bootstrap.
    assert "show type classifier" in matched_stages
    assert "Architect for a news show generation system" in matched_stages


# ─────────────────────────────────────────────────────────────────────────────
# Debate generate tests
# ─────────────────────────────────────────────────────────────────────────────

def _write_debate_brief(tmp_path: Path) -> Path:
    """Write a minimal debate prompt file and return its path."""
    brief = tmp_path / "smoke_debate_brief.txt"
    brief.write_text(
        "A 2-turn crossfire debate between two researchers: one pro-testing, "
        "one anti-testing. Academic tone. No moderator. University setting.\n",
        encoding="utf-8",
    )
    return brief


def test_generate_debate_show_type_forced(tmp_path, monkeypatch):
    """Forced ``--type debate`` runs bootstrap+conductor and writes debate JSON."""
    shared = _patch_clients(monkeypatch)
    brief = _write_debate_brief(tmp_path)
    out_dir = tmp_path / "out"

    run.main(
        [
            str(brief),
            "--mode", "generate",
            "--type", "debate",
            "--turns", "2",
            "--fact-check", "off",
            "--no-research",
            "--output", str(out_dir),
            "--name", "smoke_debate",
            "--quiet",
        ]
    )

    json_path = out_dir / "smoke_debate.json"
    assert json_path.exists(), (
        f"Expected {json_path} to exist. SmartMockClient calls:\n"
        + "\n".join(
            f"  matched={c['matched']!r}  head={c['system_head']!r}"
            for c in shared.calls
        )
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "debate_id" in data
    assert data.get("title") == "Mock Debate: Test Pipeline Green"
    assert data.get("events"), "debate JSON should contain at least one event"

    matched_stages = [c["matched"] for c in shared.calls]
    assert "show type classifier" not in matched_stages
    assert "Architect for a debate generation system" in matched_stages


# ─────────────────────────────────────────────────────────────────────────────
# Podcast generate tests
# ─────────────────────────────────────────────────────────────────────────────

def _write_podcast_brief(tmp_path: Path) -> Path:
    """Write a minimal podcast prompt file and return its path."""
    brief = tmp_path / "smoke_podcast_brief.txt"
    brief.write_text(
        "A 1-segment casual podcast about integration testing. "
        "Single enthusiastic host named Alex. Short celebratory episode.\n",
        encoding="utf-8",
    )
    return brief


def test_generate_podcast_show_type_forced(tmp_path, monkeypatch):
    """Forced ``--type podcast`` runs bootstrap+conductor and writes podcast JSON."""
    shared = _patch_clients(monkeypatch)
    brief = _write_podcast_brief(tmp_path)
    out_dir = tmp_path / "out"

    run.main(
        [
            str(brief),
            "--mode", "generate",
            "--type", "podcast",
            "--segments", "1",
            "--fact-check", "off",
            "--no-research",
            "--output", str(out_dir),
            "--name", "smoke_podcast",
            "--quiet",
        ]
    )

    json_path = out_dir / "smoke_podcast.json"
    assert json_path.exists(), (
        f"Expected {json_path} to exist. SmartMockClient calls:\n"
        + "\n".join(
            f"  matched={c['matched']!r}  head={c['system_head']!r}"
            for c in shared.calls
        )
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "podcast_id" in data
    assert data.get("title") == "Smoke Test Episode"
    assert data.get("events"), "podcast JSON should contain at least one event"

    matched_stages = [c["matched"] for c in shared.calls]
    assert "show type classifier" not in matched_stages
    assert "Architect for a podcast generation system" in matched_stages


# ─────────────────────────────────────────────────────────────────────────────
# Audiobook generate tests
# ─────────────────────────────────────────────────────────────────────────────

def _write_audiobook_brief(tmp_path: Path) -> Path:
    """Write a minimal audiobook prompt file and return its path."""
    brief = tmp_path / "smoke_audiobook_brief.txt"
    brief.write_text(
        "A short single-chapter science fiction audiobook about a mock client "
        "achieving its first successful integration test run. Contemplative tone. "
        "Single narrator called The Archivist.\n",
        encoding="utf-8",
    )
    return brief


def test_generate_audiobook_show_type_forced(tmp_path, monkeypatch):
    """Forced ``--type audiobook`` runs bootstrap+conductor and writes book JSON."""
    shared = _patch_clients(monkeypatch)
    brief = _write_audiobook_brief(tmp_path)
    out_dir = tmp_path / "out"

    run.main(
        [
            str(brief),
            "--mode", "generate",
            "--type", "audiobook",
            "--fact-check", "off",
            "--no-research",
            "--output", str(out_dir),
            "--name", "smoke_audiobook",
            "--quiet",
        ]
    )

    json_path = out_dir / "smoke_audiobook.json"
    assert json_path.exists(), (
        f"Expected {json_path} to exist. SmartMockClient calls:\n"
        + "\n".join(
            f"  matched={c['matched']!r}  head={c['system_head']!r}"
            for c in shared.calls
        )
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "audiobook_id" in data
    assert data.get("title") == "The Smoke Test Chronicles"
    assert data.get("events"), "audiobook JSON should contain at least one event"

    matched_stages = [c["matched"] for c in shared.calls]
    assert "show type classifier" not in matched_stages
    assert "Architect for an audiobook generation system" in matched_stages
