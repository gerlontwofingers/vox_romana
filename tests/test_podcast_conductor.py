"""Tests for the podcast show infrastructure.

Tests:
    1. Config validation (host gender enforcement, segment validation)
    2. Bootstrap architect parsing (canned JSON → PodcastConfig)
    3. Bootstrap host profiler parsing (canned JSON → PodcastHostProfile)
    4. Bootstrap director parsing (canned JSON → PodcastDirectorBrief)
    5. Conductor response parsing (JSON array → event dicts, fallback)
    6. Context management (summarize, compress, build)
    7. Full conductor run with mocked LLM (2 hosts, 3 segments)
    8. Result packaging (PodcastEvent creation, MergedPodcast conversion)
    9. Serializer round-trip (save → load)
    10. Router integration (podcast briefs → "podcast" classification)
    11. Host description building for system prompts

All tests use a mocked DeepSeekClient — no real LLM calls.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_podcast_conductor.py -v

Run just config tests:
    python -m pytest tests/test_podcast_conductor.py -v -k "TestPodcastConfig"

Run just conductor tests:
    python -m pytest tests/test_podcast_conductor.py -v -k "TestConductorRun"
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from vox_romana.shows.podcast.bootstrap import PodcastBootstrap
from vox_romana.shows.podcast.conductor import (
    PodcastConductor,
    PodcastConductorResult,
)
from vox_romana.shows.podcast.config import (
    PodcastConfig,
    PodcastDirectorBrief,
    PodcastHostProfile,
    PodcastSegmentConfig,
)
from vox_romana.shows.podcast.models import MergedPodcast, PodcastEvent, PodcastSegment
from vox_romana.shows.podcast.serializer import PodcastSerializer


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES — reusable test data
# ═══════════════════════════════════════════════════════════════════════════

def make_test_config(
    *,
    num_hosts=2,
    num_segments=3,
    with_director=True,
):
    """Build a minimal but complete PodcastConfig for testing.

    Default: 2 hosts, 3 segments (intro + deep_dive + outro), with director.
    """
    hosts = []
    host_data = [
        ("The Maven", "female", "lead host", "rapid-fire with stats"),
        ("Chill Carlos", "male", "co-host", "laid-back storytelling"),
        ("The Rookie", "female", "co-host", "enthusiastic questions"),
    ]
    for i in range(min(num_hosts, len(host_data))):
        name, gender, role, style = host_data[i]
        hosts.append(PodcastHostProfile(
            name=name,
            gender=gender,
            role=role,
            speech_style=style,
            knowledge_domains=[f"domain_{i}"],
            forbidden_knowledge=[f"forbidden_{i}"],
            interruption_tendency="medium",
            voice_fingerprint=f"Voice fingerprint for {name}",
            behavioral_triggers=f"Triggers for {name}",
            emotional_arc=f"Arc for {name}",
            key_positions=[f"Position {i}_1", f"Position {i}_2"],
            tangent_topics=[f"Tangent {i}_1"],
            voice_hint=f"Voice hint for {name}",
            tts_style=f"TTS style for {name}",
        ))

    segment_configs = [
        ("intro", "Welcome to the show", "short"),
        ("deep_dive", "Main topic analysis", "long"),
        ("banter", "Light back-and-forth", "medium"),
        ("drama", "Heated discussion", "medium"),
        ("outro", "Wrap-up and sign-off", "short"),
    ]
    segments = []
    for i in range(min(num_segments, len(segment_configs))):
        stype, topic, duration = segment_configs[i]
        segments.append(PodcastSegmentConfig(
            segment_type=stype,
            topic=topic,
            lead_host=hosts[i % len(hosts)].name,
            duration_hint=duration,
            target_words=300 if duration == "medium" else 200,
        ))

    director_brief = None
    if with_director:
        director_brief = PodcastDirectorBrief(
            arc="Open high, settle into analysis, wrap warmly.",
            mood_progression=["excited", "focused", "warm"],
            segment_notes={0: "Make the intro punchy"},
            closing_strategy="Each host gives a takeaway, sign off with show name.",
            compression_interval=5,
        )

    return PodcastConfig(
        brief="Test podcast episode",
        title="Test Episode",
        show_name="Test Podcast",
        topic="Testing things",
        tone="playful",
        hosts=hosts,
        segments=segments,
        director_brief=director_brief,
    )


def make_segment_response(speakers: list[str], num_turns: int = 4) -> str:
    """Build a canned JSON array response for a segment.

    Returns a JSON string like:
    [
      {"speaker_name": "Host A", "text": "Line 1", "tone": "warm", ...},
      {"speaker_name": "Host B", "text": "Line 2", "tone": "playful", ...},
      ...
    ]
    """
    turns = []
    for i in range(num_turns):
        speaker = speakers[i % len(speakers)]
        turns.append({
            "speaker_name": speaker,
            "text": f"This is line {i + 1} from {speaker}.",
            "tone": "warm" if i % 2 == 0 else "playful",
            "volume": "moderate",
            "pace": "medium",
        })
    return json.dumps(turns, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TestPodcastConfig — validation and gender enforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestPodcastConfig:
    """Tests for PodcastConfig validation, including the gender bug fix."""

    def test_valid_config_no_problems(self):
        config = make_test_config()
        problems = config.validate()
        assert problems == []

    def test_empty_brief_flagged(self):
        config = make_test_config()
        config.brief = ""
        problems = config.validate()
        assert any("brief" in p for p in problems)

    def test_no_hosts_flagged(self):
        config = make_test_config()
        config.hosts = []
        problems = config.validate()
        assert any("host" in p.lower() for p in problems)

    def test_no_segments_flagged(self):
        config = make_test_config()
        config.segments = []
        problems = config.validate()
        assert any("segment" in p.lower() for p in problems)

    def test_missing_gender_flagged(self):
        """The Trash Pandas bug fix: gender must be set for all hosts."""
        config = make_test_config()
        config.hosts[0].gender = None  # Remove gender
        problems = config.validate()
        assert any("gender" in p.lower() for p in problems)
        assert any(config.hosts[0].name in p for p in problems)

    def test_missing_host_name_flagged(self):
        config = make_test_config()
        config.hosts[0].name = ""
        problems = config.validate()
        assert any("name" in p.lower() for p in problems)

    def test_invalid_lead_host_flagged(self):
        """Segment references a host that doesn't exist in the cast."""
        config = make_test_config()
        config.segments[0].lead_host = "NonExistent Host"
        problems = config.validate()
        assert any("lead_host" in p for p in problems)

    def test_valid_lead_host_ok(self):
        config = make_test_config()
        # Lead host matches a real host name — no problems.
        config.segments[0].lead_host = config.hosts[0].name
        problems = config.validate()
        assert not any("lead_host" in p for p in problems)

    def test_all_genders_set_no_warning(self):
        """When all hosts have gender set, no gender warnings."""
        config = make_test_config(num_hosts=3)
        problems = config.validate()
        assert not any("gender" in p.lower() for p in problems)


# ═══════════════════════════════════════════════════════════════════════════
# TestHostProfile — PodcastHostProfile defaults and fields
# ═══════════════════════════════════════════════════════════════════════════

class TestHostProfile:
    def test_defaults(self):
        profile = PodcastHostProfile(name="Test", gender="female")
        assert profile.role == "host"
        assert profile.interruption_tendency == "medium"
        assert profile.knowledge_domains == []
        assert profile.forbidden_knowledge == []

    def test_all_fields(self):
        profile = PodcastHostProfile(
            name="The Maven",
            gender="female",
            role="lead host",
            speech_style="rapid-fire",
            knowledge_domains=["soccer", "history"],
            forbidden_knowledge=["cricket"],
            interruption_tendency="high",
            voice_fingerprint="Quick, punchy sentences",
            voice_hint="warm alto",
            tts_style="Speak with energy",
        )
        assert profile.name == "The Maven"
        assert profile.gender == "female"
        assert profile.knowledge_domains == ["soccer", "history"]
        assert profile.forbidden_knowledge == ["cricket"]


# ═══════════════════════════════════════════════════════════════════════════
# TestSegmentConfig — PodcastSegmentConfig defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestSegmentConfig:
    def test_defaults(self):
        seg = PodcastSegmentConfig()
        assert seg.segment_type == "banter"
        assert seg.duration_hint == "medium"
        assert seg.target_words is None
        assert seg.lead_host is None

    def test_all_fields(self):
        seg = PodcastSegmentConfig(
            segment_type="deep_dive",
            topic="The 1999 WWC Final",
            lead_host="The Maven",
            target_words=800,
            duration_hint="long",
        )
        assert seg.segment_type == "deep_dive"
        assert seg.target_words == 800


# ═══════════════════════════════════════════════════════════════════════════
# TestBootstrapArchitect — parsing the Architect's JSON
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapArchitect:
    """Tests for PodcastBootstrap._parse_architect_response()."""

    def _make_bootstrap(self):
        mock_client = Mock()
        return PodcastBootstrap(client=mock_client)

    def test_parses_clean_json(self):
        bootstrap = self._make_bootstrap()
        raw = json.dumps({
            "title": "Soccer Rivalries",
            "show_name": "Trash Pandas",
            "topic": "Greatest rivalries in women's soccer",
            "tone": "playful",
            "hosts": [
                {
                    "name": "The Maven",
                    "gender": "female",
                    "role": "lead host",
                    "speech_style": "rapid-fire",
                    "knowledge_domains": ["soccer"],
                    "forbidden_knowledge": ["cricket"],
                    "interruption_tendency": "high",
                },
                {
                    "name": "The Rookie",
                    "gender": "female",
                    "role": "co-host",
                    "speech_style": "curious",
                    "knowledge_domains": ["pop culture"],
                    "forbidden_knowledge": [],
                    "interruption_tendency": "low",
                },
            ],
            "segments": [
                {
                    "segment_type": "intro",
                    "topic": "Welcome",
                    "lead_host": "The Maven",
                    "duration_hint": "short",
                    "target_words": 200,
                },
                {
                    "segment_type": "deep_dive",
                    "topic": "1999 WWC Final",
                    "lead_host": "The Maven",
                    "duration_hint": "long",
                    "target_words": 800,
                },
            ],
        })
        config = bootstrap._parse_architect_response(
            raw, brief="test brief", fact_check="off",
        )
        assert isinstance(config, PodcastConfig)
        assert config.title == "Soccer Rivalries"
        assert config.show_name == "Trash Pandas"
        assert len(config.hosts) == 2
        assert config.hosts[0].name == "The Maven"
        assert config.hosts[0].gender == "female"
        assert config.hosts[1].gender == "female"
        assert len(config.segments) == 2
        assert config.segments[0].segment_type == "intro"

    def test_parses_markdown_fenced_json(self):
        bootstrap = self._make_bootstrap()
        raw = '```json\n' + json.dumps({
            "title": "Test",
            "show_name": "Pod",
            "topic": "topic",
            "tone": "chill",
            "hosts": [{"name": "A", "gender": "male", "role": "host"}],
            "segments": [{"segment_type": "banter", "topic": "stuff"}],
        }) + '\n```'
        config = bootstrap._parse_architect_response(raw, brief="test")
        assert config.title == "Test"
        assert len(config.hosts) == 1

    def test_preserves_brief_and_fact_check(self):
        bootstrap = self._make_bootstrap()
        raw = json.dumps({
            "title": "T",
            "hosts": [{"name": "A", "gender": "female"}],
            "segments": [{"segment_type": "intro", "topic": "x"}],
        })
        config = bootstrap._parse_architect_response(
            raw, brief="original brief", reference_material="ref", fact_check="quick",
        )
        assert config.brief == "original brief"
        assert config.reference_material == "ref"
        assert config.fact_check == "quick"

    def test_invalid_json_raises(self):
        bootstrap = self._make_bootstrap()
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_architect_response("not json", brief="test")


# ═══════════════════════════════════════════════════════════════════════════
# TestBootstrapHostProfiler — parsing the Host Profiler's JSON
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapHostProfiler:
    """Tests for PodcastBootstrap._parse_host_profile()."""

    def _make_bootstrap(self):
        mock_client = Mock()
        return PodcastBootstrap(client=mock_client)

    def test_fills_personality_fields(self):
        bootstrap = self._make_bootstrap()
        host = PodcastHostProfile(
            name="The Maven",
            gender="female",
            role="lead host",
            speech_style="rapid-fire",
        )
        raw = json.dumps({
            "voice_hint": "warm alto, late 20s",
            "tts_style": "Speak with infectious enthusiasm",
            "voice_fingerprint": "Quick punchy sentences with stats",
            "behavioral_triggers": "Gets excited about data",
            "emotional_arc": "Starts focused, gets animated",
            "key_positions": ["Position A", "Position B"],
            "tangent_topics": ["Random history fact"],
        })
        result = bootstrap._parse_host_profile(raw, host=host)
        assert result.name == "The Maven"  # Identity preserved
        assert result.gender == "female"  # Identity preserved
        assert result.voice_hint == "warm alto, late 20s"
        assert result.voice_fingerprint == "Quick punchy sentences with stats"
        assert result.key_positions == ["Position A", "Position B"]
        assert result.tangent_topics == ["Random history fact"]

    def test_preserves_identity_on_bad_json(self):
        """Fallback: bad JSON should not crash, should keep host identity."""
        bootstrap = self._make_bootstrap()
        host = PodcastHostProfile(name="Test", gender="male")
        result = bootstrap._parse_host_profile("not json", host=host)
        assert result.name == "Test"
        assert result.gender == "male"
        # Should have a fallback fingerprint.
        assert result.voice_fingerprint is not None


# ═══════════════════════════════════════════════════════════════════════════
# TestBootstrapDirector — parsing the Director's JSON
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapDirector:
    """Tests for PodcastBootstrap._parse_director_response()."""

    def _make_bootstrap(self):
        mock_client = Mock()
        return PodcastBootstrap(client=mock_client)

    def test_parses_clean_json(self):
        bootstrap = self._make_bootstrap()
        raw = json.dumps({
            "arc": "Open high, settle, wrap warmly",
            "mood_progression": ["excited", "focused", "warm"],
            "segment_notes": {"0": "Punchy intro", "2": "Wind down"},
            "closing_strategy": "Each host gives a takeaway",
        })
        brief = bootstrap._parse_director_response(raw)
        assert isinstance(brief, PodcastDirectorBrief)
        assert brief.arc == "Open high, settle, wrap warmly"
        assert len(brief.mood_progression) == 3
        assert brief.segment_notes[0] == "Punchy intro"
        assert brief.segment_notes[2] == "Wind down"

    def test_segment_notes_int_keys(self):
        """Director returns string keys ('0', '2') → converted to int."""
        bootstrap = self._make_bootstrap()
        raw = json.dumps({
            "arc": "test",
            "segment_notes": {"0": "note A", "3": "note B"},
        })
        brief = bootstrap._parse_director_response(raw)
        assert 0 in brief.segment_notes
        assert 3 in brief.segment_notes
        # String keys like "abc" should be skipped gracefully.

    def test_invalid_json_raises(self):
        bootstrap = self._make_bootstrap()
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_director_response("not json")


# ═══════════════════════════════════════════════════════════════════════════
# TestResponseParsing — conductor segment response parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseParsing:
    """Tests for PodcastConductor._parse_segment_response()."""

    def _make_conductor(self):
        config = make_test_config()
        mock_client = Mock()
        return PodcastConductor(config, client=mock_client)

    def test_parses_json_array(self):
        conductor = self._make_conductor()
        raw = json.dumps([
            {"speaker_name": "The Maven", "text": "Hello!", "tone": "warm"},
            {"speaker_name": "Chill Carlos", "text": "Hey!", "tone": "playful"},
        ])
        events = conductor._parse_segment_response(
            raw, segment_id="seg_000", segment_type="intro",
            lead_host="The Maven", global_order_start=0,
        )
        assert len(events) == 2
        assert events[0]["speaker_name"] == "The Maven"
        assert events[1]["speaker_name"] == "Chill Carlos"
        assert events[0]["segment_id"] == "seg_000"
        assert events[0]["global_order"] == 0
        assert events[1]["global_order"] == 1

    def test_parses_markdown_fenced_array(self):
        conductor = self._make_conductor()
        raw = '```json\n' + json.dumps([
            {"speaker_name": "A", "text": "Line 1"},
        ]) + '\n```'
        events = conductor._parse_segment_response(
            raw, segment_id="seg_000", segment_type="banter",
            lead_host="A", global_order_start=0,
        )
        assert len(events) == 1
        assert events[0]["speaker_name"] == "A"

    def test_fallback_on_bad_json(self):
        """If JSON parsing fails, return a single event with raw text."""
        conductor = self._make_conductor()
        raw = "This is just plain text, not JSON at all."
        events = conductor._parse_segment_response(
            raw, segment_id="seg_000", segment_type="banter",
            lead_host="The Maven", global_order_start=0,
        )
        assert len(events) == 1
        assert events[0]["speaker_name"] == "The Maven"
        assert "plain text" in events[0]["text"]

    def test_fallback_on_empty_array(self):
        """An empty JSON array should produce a fallback event."""
        conductor = self._make_conductor()
        raw = "[]"
        events = conductor._parse_segment_response(
            raw, segment_id="seg_000", segment_type="intro",
            lead_host="The Maven", global_order_start=0,
        )
        assert len(events) == 1
        assert events[0]["speaker_name"] == "The Maven"

    def test_defaults_for_missing_fields(self):
        """Missing tone/volume/pace should get defaults."""
        conductor = self._make_conductor()
        raw = json.dumps([{"speaker_name": "A", "text": "Hello"}])
        events = conductor._parse_segment_response(
            raw, segment_id="seg_000", segment_type="banter",
            lead_host="A", global_order_start=0,
        )
        assert events[0]["tone"] == "neutral"
        assert events[0]["volume"] == "moderate"
        assert events[0]["pace"] == "medium"

    def test_non_dict_items_skipped(self):
        """Non-dict items in the array should be skipped."""
        conductor = self._make_conductor()
        raw = json.dumps([
            {"speaker_name": "A", "text": "Hello"},
            "not a dict",
            42,
            {"speaker_name": "B", "text": "World"},
        ])
        events = conductor._parse_segment_response(
            raw, segment_id="seg_000", segment_type="banter",
            lead_host="A", global_order_start=0,
        )
        assert len(events) == 2


# ═══════════════════════════════════════════════════════════════════════════
# TestContextManagement — segment summarization and compression
# ═══════════════════════════════════════════════════════════════════════════

class TestContextManagement:
    """Tests for context summary building, segment summarization, etc."""

    def test_summarize_segment(self):
        events = [
            {"speaker_name": "A", "text": "Hello, let's talk about soccer."},
            {"speaker_name": "B", "text": "Great topic! I love soccer."},
        ]
        seg_config = PodcastSegmentConfig(
            segment_type="intro", topic="Soccer intro",
        )
        summary = PodcastConductor._summarize_segment(events, seg_config)
        assert "intro" in summary.lower()
        assert "Soccer" in summary
        assert "A" in summary
        assert "B" in summary

    def test_build_context_summary_empty(self):
        config = make_test_config()
        conductor = PodcastConductor(config, client=Mock())
        # No context yet — should return empty.
        result = conductor._build_context_summary()
        assert result == ""

    def test_build_context_summary_with_entries(self):
        config = make_test_config()
        conductor = PodcastConductor(config, client=Mock())
        conductor._context_log = [
            "[intro] Welcome — speakers: A. A: Hello",
            "[deep_dive] Main topic — speakers: A, B. A: Analysis",
        ]
        result = conductor._build_context_summary()
        assert "intro" in result.lower()
        assert "deep_dive" in result.lower()

    def test_build_context_with_compressed(self):
        config = make_test_config()
        conductor = PodcastConductor(config, client=Mock())
        conductor._compressed_ctx = "Earlier topics included soccer history."
        conductor._context_log = ["Recent: discussed 1999 WWC"]
        result = conductor._build_context_summary()
        assert "soccer history" in result
        assert "1999" in result


# ═══════════════════════════════════════════════════════════════════════════
# TestHostDescriptions — building the system prompt host block
# ═══════════════════════════════════════════════════════════════════════════

class TestHostDescriptions:
    def test_builds_for_each_host(self):
        hosts = [
            PodcastHostProfile(
                name="The Maven", gender="female", role="lead host",
                speech_style="rapid-fire", knowledge_domains=["soccer"],
                forbidden_knowledge=["cricket"],
            ),
            PodcastHostProfile(
                name="Chill Carlos", gender="male", role="co-host",
                speech_style="laid-back",
            ),
        ]
        result = PodcastConductor._build_host_descriptions(hosts)
        assert "The Maven" in result
        assert "Chill Carlos" in result
        assert "rapid-fire" in result
        assert "soccer" in result
        assert "cricket" in result


# ═══════════════════════════════════════════════════════════════════════════
# TestConductorRun — full mocked conductor run
# ═══════════════════════════════════════════════════════════════════════════

class TestConductorRun:
    """Full end-to-end conductor run with mocked LLM responses."""

    def test_full_run_2_hosts_3_segments(self):
        """Run the conductor with 2 hosts and 3 segments, all mocked.

        Verifies:
        - All segments produce events
        - Speaker names are from the host cast
        - Event counts are reasonable
        - Segment IDs are correct
        """
        config = make_test_config(num_hosts=2, num_segments=3)
        mock_client = Mock()
        host_names = [h.name for h in config.hosts]

        # The conductor will call session.send() for each segment.
        # Mock the send() to return canned JSON arrays.
        mock_client.chat.return_value = json.dumps({
            "mood_note": "Keep the energy up",
            "wrap_up": False,
        })

        conductor = PodcastConductor(config, client=mock_client)

        # We need to mock the AgentSession that _build_agents creates.
        # The conductor calls self._create_session(), which uses self._client.
        # We'll patch the session's send() method after _build_agents runs.
        conductor._build_agents()

        # Now mock the session's send to return segment responses.
        assert conductor._session is not None
        responses = [
            make_segment_response(host_names, num_turns=4),
            make_segment_response(host_names, num_turns=6),
            make_segment_response(host_names, num_turns=3),
        ]
        call_count = [0]

        original_send = conductor._session.send

        def mock_send(message):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(responses):
                return responses[idx]
            return make_segment_response(host_names, num_turns=3)

        conductor._session.send = mock_send

        # Run the turn loop.
        raw_events = conductor._run_turn_loop()

        # Verify event count: 4 + 6 + 3 = 13 events
        assert len(raw_events) == 13

        # Verify all speaker names are from the cast.
        for evt in raw_events:
            assert evt["speaker_name"] in host_names

        # Verify segment IDs are assigned.
        segment_ids = set(evt["segment_id"] for evt in raw_events)
        assert "seg_000" in segment_ids
        assert "seg_001" in segment_ids
        assert "seg_002" in segment_ids

    def test_run_produces_conductor_result(self):
        """Full run() call produces a PodcastConductorResult."""
        config = make_test_config(num_hosts=2, num_segments=2, with_director=False)
        mock_client = Mock()
        host_names = [h.name for h in config.hosts]

        conductor = PodcastConductor(config, client=mock_client)
        conductor._build_agents()

        # Mock session.
        responses = [
            make_segment_response(host_names, num_turns=3),
            make_segment_response(host_names, num_turns=3),
        ]
        call_idx = [0]

        def mock_send(msg):
            idx = call_idx[0]
            call_idx[0] += 1
            return responses[idx] if idx < len(responses) else responses[-1]

        conductor._session.send = mock_send

        raw_events = conductor._run_turn_loop()
        result = conductor._package_result(raw_events)

        assert isinstance(result, PodcastConductorResult)
        assert len(result.podcast_events) == 6
        assert len(result.podcast_segments) == 2
        assert result.podcast_segments[0].segment_type == "intro"
        assert result.podcast_segments[1].segment_type == "deep_dive"


# ═══════════════════════════════════════════════════════════════════════════
# TestResultPackaging — MergedPodcast conversion and voice maps
# ═══════════════════════════════════════════════════════════════════════════

class TestResultPackaging:
    """Tests for PodcastConductorResult.to_merged_podcast()."""

    def test_to_merged_podcast_voice_maps(self):
        """Verify voice_casting, voice_hints, tts_styles from host profiles."""
        config = make_test_config(num_hosts=2)
        result = PodcastConductorResult(
            config=config,
            events=[],
            transcript=[],
            total_llm_calls=5,
            podcast_events=[],
            podcast_segments=[],
            podcast_id="test_123",
        )
        merged = result.to_merged_podcast()
        assert isinstance(merged, MergedPodcast)
        assert merged.podcast_id == "test_123"
        assert merged.title == "Test Episode"
        assert merged.show_name == "Test Podcast"

        # Voice casting should have the gender for each host.
        assert merged.voice_casting["The Maven"] == "female"
        assert merged.voice_casting["Chill Carlos"] == "male"

        # Voice hints and TTS styles should be populated.
        assert "The Maven" in merged.voice_hints
        assert "Chill Carlos" in merged.tts_styles

    def test_to_merged_podcast_hosts_tuple(self):
        config = make_test_config(num_hosts=3)
        result = PodcastConductorResult(
            config=config, events=[], podcast_events=[], podcast_segments=[],
            podcast_id="x",
        )
        merged = result.to_merged_podcast()
        assert len(merged.hosts) == 3
        assert merged.hosts[0] == "The Maven"

    def test_merged_podcast_to_dict(self):
        """MergedPodcast.to_dict() produces a JSON-serializable dict."""
        segment = PodcastSegment(
            segment_id="seg_000", segment_order=1,
            segment_type="intro", topic="Welcome",
            events=(
                PodcastEvent(
                    global_order=0, local_order=0,
                    speaker_role="host", speaker_name="A",
                    target=None, text="Hello!",
                    tone=("warm",), event_kind="utterance",
                ),
            ),
        )
        merged = MergedPodcast(
            podcast_id="test",
            title="Test",
            hosts=("A",),
            segments=(segment,),
            events=segment.events,
            voice_casting={"A": "female"},
        )
        d = merged.to_dict()
        # Should be JSON-serializable.
        json_str = json.dumps(d)
        assert "Hello!" in json_str
        assert d["voice_casting"]["A"] == "female"


# ═══════════════════════════════════════════════════════════════════════════
# TestPodcastSerializer — save/load round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestPodcastSerializer:
    def test_save_and_load_roundtrip(self):
        segment = PodcastSegment(
            segment_id="seg_000", segment_order=1,
            segment_type="intro", topic="Welcome",
            events=(
                PodcastEvent(
                    global_order=0, local_order=0,
                    speaker_role="host", speaker_name="The Maven",
                    target=None, text="Welcome to the show!",
                    tone=("warm",), event_kind="utterance",
                ),
            ),
        )
        podcast = MergedPodcast(
            podcast_id="round_trip_test",
            title="Round Trip",
            show_name="Test Pod",
            hosts=("The Maven",),
            segments=(segment,),
            events=segment.events,
            voice_casting={"The Maven": "female"},
            voice_hints={"The Maven": "warm alto"},
            tts_styles={"The Maven": "Speak warmly"},
        )

        serializer = PodcastSerializer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_podcast.json"
            saved_path = serializer.save(podcast, path)
            assert saved_path.exists()

            loaded = serializer.load(saved_path)
            assert loaded["podcast_id"] == "round_trip_test"
            assert loaded["title"] == "Round Trip"
            assert loaded["show_name"] == "Test Pod"
            assert len(loaded["segments"]) == 1
            assert loaded["segments"][0]["segment_type"] == "intro"
            assert len(loaded["events"]) == 1
            assert loaded["voice_casting"]["The Maven"] == "female"

    def test_to_json_string(self):
        podcast = MergedPodcast(
            podcast_id="str_test",
            title="String Test",
            hosts=("A",),
        )
        serializer = PodcastSerializer()
        result = serializer.to_json_string(podcast)
        assert '"podcast_id": "str_test"' in result
        parsed = json.loads(result)
        assert parsed["title"] == "String Test"


# ═══════════════════════════════════════════════════════════════════════════
# TestPodcastModels — PodcastSegment and MergedPodcast
# ═══════════════════════════════════════════════════════════════════════════

class TestPodcastModels:
    def test_podcast_segment_to_dict(self):
        seg = PodcastSegment(
            segment_id="seg_001", segment_order=2,
            segment_type="deep_dive", topic="Analysis",
        )
        d = seg.to_dict()
        assert d["segment_id"] == "seg_001"
        assert d["segment_type"] == "deep_dive"
        assert d["events"] == []

    def test_podcast_event_alias(self):
        """PodcastEvent is just an alias for AudioEvent."""
        from vox_romana.core.models import AudioEvent
        assert PodcastEvent is AudioEvent


# ═══════════════════════════════════════════════════════════════════════════
# TestBootstrapFullChain — full bootstrap chain with mocked LLM
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapFullChain:
    """Tests the full from_brief() chain with all 3 stages mocked."""

    def test_from_brief_produces_config(self):
        mock_client = Mock()

        # Stage 1: Architect returns a skeleton config.
        architect_response = json.dumps({
            "title": "Soccer Rivalries",
            "show_name": "Trash Pandas",
            "topic": "Women's soccer rivalries",
            "tone": "playful",
            "hosts": [
                {"name": "The Maven", "gender": "female", "role": "lead host",
                 "speech_style": "rapid-fire", "knowledge_domains": ["soccer"]},
                {"name": "The Rookie", "gender": "female", "role": "co-host",
                 "speech_style": "curious", "knowledge_domains": ["pop culture"]},
            ],
            "segments": [
                {"segment_type": "intro", "topic": "Welcome", "lead_host": "The Maven"},
                {"segment_type": "deep_dive", "topic": "1999 WWC", "lead_host": "The Maven"},
                {"segment_type": "outro", "topic": "Wrap-up", "lead_host": "The Rookie"},
            ],
        })

        # Stage 2: Host profiler returns voice profiles (called once per host).
        profiler_response = json.dumps({
            "voice_hint": "warm alto",
            "tts_style": "Speak with enthusiasm",
            "voice_fingerprint": "Quick punchy sentences",
            "behavioral_triggers": "Gets excited about stats",
            "emotional_arc": "Starts focused, gets animated",
            "key_positions": ["Position A"],
            "tangent_topics": ["Random tangent"],
        })

        # Stage 3: Director returns an arc.
        director_response = json.dumps({
            "arc": "Open high, settle, wrap warmly",
            "mood_progression": ["excited", "focused", "warm"],
            "segment_notes": {"0": "Punchy intro"},
            "closing_strategy": "Takeaways then sign-off",
        })

        # The client.chat() is called: Architect(1) + Profiler(2) + Director(1) = 4 times.
        mock_client.chat.side_effect = [
            architect_response,
            profiler_response,  # Host 1
            profiler_response,  # Host 2
            director_response,
        ]

        bootstrap = PodcastBootstrap(client=mock_client)
        config = bootstrap.from_brief(
            "Three women discuss soccer rivalries",
            target_words=5000,
        )

        assert isinstance(config, PodcastConfig)
        assert config.title == "Soccer Rivalries"
        assert config.show_name == "Trash Pandas"
        assert len(config.hosts) == 2
        assert config.hosts[0].voice_fingerprint == "Quick punchy sentences"
        assert config.director_brief is not None
        assert config.director_brief.arc == "Open high, settle, wrap warmly"
        assert mock_client.chat.call_count == 4

    def test_overrides_applied(self):
        mock_client = Mock()
        architect_response = json.dumps({
            "title": "Original",
            "show_name": "Original Pod",
            "tone": "serious",
            "hosts": [{"name": "A", "gender": "male"}],
            "segments": [{"segment_type": "banter", "topic": "stuff"}],
        })
        profiler_response = json.dumps({"voice_fingerprint": "test"})
        director_response = json.dumps({"arc": "test arc"})

        mock_client.chat.side_effect = [
            architect_response, profiler_response, director_response,
        ]

        bootstrap = PodcastBootstrap(client=mock_client)
        config = bootstrap.from_brief(
            "test",
            overrides={"tone": "chaotic energy", "show_name": "Overridden Pod"},
        )
        assert config.tone == "chaotic energy"
        assert config.show_name == "Overridden Pod"


# ═══════════════════════════════════════════════════════════════════════════
# TestRouterIntegration — the router recognizes podcast briefs
# ═══════════════════════════════════════════════════════════════════════════

class TestRouterIntegration:
    """Tests that the router correctly handles 'podcast' as a show type."""

    def test_podcast_is_valid_show_type(self):
        from vox_romana.core.router import SHOW_TYPES
        assert "podcast" in SHOW_TYPES

    def test_router_result_with_podcast(self):
        from vox_romana.core.router import RouterResult
        r = RouterResult("podcast", 0.9, "multi-host conversation")
        assert r.show_type == "podcast"
        assert r.confidence == 0.9

    def test_parse_podcast_response(self):
        from vox_romana.core.router import parse_router_response
        raw = json.dumps({
            "show_type": "podcast",
            "confidence": 0.85,
            "reasoning": "Multi-host casual conversation format",
            "needs_research": False,
        })
        result = parse_router_response(raw)
        assert result.show_type == "podcast"
        assert result.confidence == 0.85


# ═══════════════════════════════════════════════════════════════════════════
# TestDirectorBrief — PodcastDirectorBrief defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestDirectorBrief:
    def test_defaults(self):
        brief = PodcastDirectorBrief()
        assert brief.arc == ""
        assert brief.mood_progression == []
        assert brief.segment_notes == {}
        assert brief.compression_interval == 5

    def test_all_fields(self):
        brief = PodcastDirectorBrief(
            arc="Build and release",
            mood_progression=["excited", "tense", "warm"],
            segment_notes={0: "Hook them", 2: "Wind down"},
            closing_strategy="Thank and sign off",
            compression_interval=3,
        )
        assert brief.compression_interval == 3
        assert len(brief.segment_notes) == 2


# ═══════════════════════════════════════════════════════════════════════════
# TestImports — verify all public symbols are importable
# ═══════════════════════════════════════════════════════════════════════════

class TestImports:
    def test_import_from_package(self):
        from vox_romana.shows.podcast import (
            MergedPodcast,
            PodcastConfig,
            PodcastDirectorBrief,
            PodcastEvent,
            PodcastHostProfile,
            PodcastSegment,
            PodcastSegmentConfig,
        )
        # Just verify they're importable and not None.
        assert PodcastConfig is not None
        assert PodcastHostProfile is not None
        assert MergedPodcast is not None

    def test_import_from_top_level(self):
        from vox_romana import (
            MergedPodcast,
            PodcastBootstrap,
            PodcastConductor,
            PodcastConductorResult,
            PodcastConfig,
            PodcastDirectorBrief,
            PodcastEvent,
            PodcastHostProfile,
            PodcastSegment,
            PodcastSegmentConfig,
            PodcastSerializer,
        )
        assert PodcastConductor is not None
        assert PodcastSerializer is not None
