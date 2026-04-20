"""Tests for the Show Host feature — narrative voice wrapping debates.

This file tests:
    1. ShowHostConfig construction and defaults
    2. Architect parsing with show_host field
    3. Host prompt building (narrator vs emcee)
    4. Host event generation via _generate_host_event()
    5. Full turn loop with host (intro + transitions + outro)
    6. Result packaging (host events get event_kind="narration")
    7. MergedDebate hosts tuple population
    8. Bootstrap host profiling (Stage 2.5)
    9. Renderer handles narration events

All tests use mocked DeepSeekClient — no real LLM calls.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_show_host.py -v
"""

import json
from unittest.mock import Mock

import pytest

from vox_romana import (
    AudienceConfig,
    DebateArchitect,
    DebateBootstrap,
    DebateConductor,
    DebateConductorResult,
    DebateConfig,
    DebateEvent,
    DirectorBrief,
    MergedDebate,
    ModeratorConfig,
    ParticipantProfile,
    ShowHostConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES — reusable configs and canned LLM responses
# ═══════════════════════════════════════════════════════════════════════════

def make_host_config(*, role="narrator"):
    """Build a ShowHostConfig for testing."""
    return ShowHostConfig(
        name="Marcus Webb",
        role=role,
        style="Warm but authoritative. Uses vivid scene-setting language.",
        voice_fingerprint="Measured baritone cadence. Pauses before proper nouns.",
    )


def make_test_config(*, target_turns=10, with_host=True, with_moderator=True,
                     with_director=False, host_role="narrator"):
    """Build a DebateConfig with an optional show host."""
    participants = [
        ParticipantProfile(
            name="Alice",
            side="pro",
            voice_fingerprint="Short punchy sentences.",
            key_positions=["Point A is correct"],
        ),
        ParticipantProfile(
            name="Bob",
            side="con",
            voice_fingerprint="Long compound sentences.",
            key_positions=["Point A is flawed"],
        ),
    ]

    moderator = None
    if with_moderator:
        moderator = ModeratorConfig(
            name="Charlie",
            mode="active",
            style="Socratic",
        )

    show_host = make_host_config(role=host_role) if with_host else None

    director_brief = None
    if with_director:
        director_brief = DirectorBrief(
            dramatic_arc="Two acts.",
            mood_progression=["calm", "heated"],
            key_moments=["Sharp exchange"],
            closing_strategy="Each gets a closing line.",
            check_interval=4,
        )

    return DebateConfig(
        brief="Test debate",
        title="The X Debate",
        topic="Is X better than Y?",
        debate_format="public_forum",
        setting="Small seminar room",
        tone="academic",
        participants=participants,
        moderator=moderator,
        show_host=show_host,
        director_brief=director_brief,
        target_turns=target_turns,
    )


def make_turn_json(text="Test response.", **overrides):
    """Build a canned JSON response like an LLM agent would produce."""
    data = {
        "text": text,
        "tone": "neutral",
        "volume": "moderate",
        "pace": "medium",
        "target": "all",
        "pause_before": "none",
        "pause_after": "none",
        "interruption": False,
        "nonverbal": [],
    }
    data.update(overrides)
    return json.dumps(data)


def make_host_json(text="Welcome to Vox Romana.", **overrides):
    """Build a canned host narration JSON response."""
    data = {
        "text": text,
        "tone": "warm",
        "volume": "moderate",
        "pace": "medium",
        "pause_before": "none",
        "pause_after": "long",
    }
    data.update(overrides)
    return json.dumps(data)


def make_moderator_json(text="Interesting point.", next_speaker="Alice"):
    """Build a canned moderator JSON response."""
    return json.dumps({
        "text": text,
        "tone": "neutral",
        "volume": "moderate",
        "pace": "medium",
        "target": "all",
        "pause_before": "none",
        "pause_after": "short",
        "interruption": False,
        "nonverbal": [],
        "next_speaker": next_speaker,
    })


def make_mock_client():
    """Create a mock client that returns appropriate JSON for each call type."""
    mock = Mock()

    def chat_side_effect(messages, **kwargs):
        system = messages[0]["content"]
        if "concise summarizer" in system.lower():
            return "Summary of debate so far."
        elif "debate director" in system.lower():
            return json.dumps({"mood_note": "Keep it up.", "nudge": None, "wrap_up": False})
        elif "moderating" in system:
            return make_moderator_json()
        elif "Vox Romana" in system:
            # This is the show host prompt — it contains "Vox Romana"
            return make_host_json()
        else:
            return make_turn_json("I have a point to make.")

    mock.chat.side_effect = chat_side_effect
    return mock


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: ShowHostConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestShowHostConfig:
    """Tests for the ShowHostConfig dataclass."""

    def test_defaults(self):
        """ShowHostConfig should have sensible defaults."""
        host = ShowHostConfig()
        assert host.name == ""
        assert host.role == "narrator"
        assert host.style is None
        assert host.voice_fingerprint is None

    def test_narrator_role(self):
        """Narrator role should be accepted."""
        host = ShowHostConfig(name="The Narrator", role="narrator")
        assert host.role == "narrator"

    def test_emcee_role(self):
        """Emcee role should be accepted."""
        host = ShowHostConfig(name="Marcus Webb", role="emcee")
        assert host.role == "emcee"

    def test_full_construction(self):
        """Full config should store all fields."""
        host = make_host_config()
        assert host.name == "Marcus Webb"
        assert host.style is not None
        assert host.voice_fingerprint is not None


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: Architect Parsing with show_host
# ═══════════════════════════════════════════════════════════════════════════

class TestArchitectShowHost:
    """Tests for the Architect parsing show_host from JSON."""

    def _make_architect_json(self, *, with_host=True):
        """Build a canned Architect response with optional show_host."""
        data = {
            "title": "The UBI Question",
            "topic": "Should the US adopt UBI?",
            "debate_format": "public_forum",
            "setting": "University lecture hall",
            "tone": "academic",
            "target_turns": 30,
            "participants": [
                {"name": "Dr. Chen", "side": "pro-UBI", "role": "participant"},
                {"name": "James Hartley", "side": "anti-UBI", "role": "participant"},
            ],
            "moderator": {
                "name": "Jordan Lee",
                "mode": "active",
                "style": "Socratic",
            },
            "audience": {"present": True, "participation": False, "size_description": "200 people"},
        }
        if with_host:
            data["show_host"] = {
                "name": "Elena Voss",
                "role": "narrator",
                "style": "Documentary-style, authoritative yet warm",
            }
        return json.dumps(data)

    def test_parse_with_show_host(self):
        """Architect should parse show_host into ShowHostConfig."""
        architect = DebateArchitect(Mock())
        config = architect.parse_response(self._make_architect_json(with_host=True))

        assert config.show_host is not None
        assert config.show_host.name == "Elena Voss"
        assert config.show_host.role == "narrator"
        assert config.show_host.style == "Documentary-style, authoritative yet warm"

    def test_parse_without_show_host(self):
        """Config should have show_host=None when not in JSON."""
        architect = DebateArchitect(Mock())
        config = architect.parse_response(self._make_architect_json(with_host=False))

        assert config.show_host is None

    def test_parse_emcee_host(self):
        """Emcee role should be preserved through parsing."""
        architect = DebateArchitect(Mock())
        data = json.loads(self._make_architect_json(with_host=True))
        data["show_host"]["role"] = "emcee"
        data["show_host"]["name"] = "Marcus Webb"
        config = architect.parse_response(json.dumps(data))

        assert config.show_host.role == "emcee"
        assert config.show_host.name == "Marcus Webb"


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: Host Prompt Building
# ═══════════════════════════════════════════════════════════════════════════

class TestHostPromptBuilding:
    """Tests for _build_host_prompt in the conductor."""

    def test_narrator_prompt_includes_third_person(self):
        """Narrator prompt should contain third-person instructions."""
        config = make_test_config(host_role="narrator")
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        assert conductor._host_prompt is not None
        assert "NARRATOR" in conductor._host_prompt
        assert "third person" in conductor._host_prompt.lower()

    def test_emcee_prompt_includes_first_person(self):
        """Emcee prompt should contain first-person instructions."""
        config = make_test_config(host_role="emcee")
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        assert conductor._host_prompt is not None
        assert "EMCEE" in conductor._host_prompt
        assert "first person" in conductor._host_prompt.lower()

    def test_host_prompt_includes_participant_list(self):
        """Host prompt should list the debate participants."""
        config = make_test_config()
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        assert "Alice" in conductor._host_prompt
        assert "Bob" in conductor._host_prompt

    def test_host_prompt_includes_voice_fingerprint(self):
        """Host prompt should include the voice fingerprint."""
        config = make_test_config()
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        assert "Measured baritone" in conductor._host_prompt

    def test_no_host_means_no_prompt(self):
        """Config without show_host should result in None prompt."""
        config = make_test_config(with_host=False)
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        assert conductor._host_prompt is None


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: Host Event Generation
# ═══════════════════════════════════════════════════════════════════════════

class TestHostEventGeneration:
    """Tests for _generate_host_event."""

    def test_intro_event_structure(self):
        """Intro host event should have correct metadata."""
        config = make_test_config()
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)
        conductor._build_agents()

        event = conductor._generate_host_event(
            beat="intro",
            context_hint="Introduce the show and participants."
        )

        assert event["speaker"] == "Marcus Webb"
        assert event["role"] == "host"
        assert event["event_kind"] == "narration"
        assert event["narration_beat"] == "intro"
        assert event["target"] == "audience"

    def test_transition_event_structure(self):
        """Transition host event should have correct beat marker."""
        config = make_test_config()
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)
        conductor._build_agents()

        event = conductor._generate_host_event(
            beat="transition",
            context_hint="We're at the midpoint."
        )

        assert event["narration_beat"] == "transition"
        assert event["event_kind"] == "narration"

    def test_outro_event_structure(self):
        """Outro host event should have correct beat marker."""
        config = make_test_config()
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)
        conductor._build_agents()

        event = conductor._generate_host_event(
            beat="outro",
            context_hint="Sign off the show."
        )

        assert event["narration_beat"] == "outro"
        assert event["role"] == "host"

    def test_host_event_has_text(self):
        """Host event should have the LLM-generated text."""
        config = make_test_config()
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)
        conductor._build_agents()

        event = conductor._generate_host_event(
            beat="intro",
            context_hint="Test."
        )

        assert event["text"] == "Welcome to Vox Romana."


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: Full Turn Loop with Host
# ═══════════════════════════════════════════════════════════════════════════

class TestTurnLoopWithHost:
    """Tests the complete run() with host events integrated."""

    def test_run_includes_intro_and_outro(self):
        """Run with host should include intro and outro narration events."""
        config = make_test_config(target_turns=6, with_director=False)
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        # First event should be host intro
        assert result.debate_events[0].speaker_role == "host"
        assert result.debate_events[0].event_kind == "narration"

        # Last event should be host outro
        assert result.debate_events[-1].speaker_role == "host"
        assert result.debate_events[-1].event_kind == "narration"

    def test_host_events_have_narration_kind(self):
        """All host events should have event_kind='narration'."""
        config = make_test_config(target_turns=6, with_director=False)
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        host_events = [e for e in result.debate_events if e.speaker_role == "host"]
        assert len(host_events) >= 2  # At least intro + outro
        for event in host_events:
            assert event.event_kind == "narration"

    def test_debate_events_still_utterances(self):
        """Non-host events should still be event_kind='utterance'."""
        config = make_test_config(target_turns=6, with_director=False)
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        non_host = [e for e in result.debate_events if e.speaker_role != "host"]
        assert len(non_host) > 0
        for event in non_host:
            assert event.event_kind == "utterance"

    def test_no_host_means_no_narration_events(self):
        """Without a host, there should be no narration events."""
        config = make_test_config(target_turns=6, with_host=False, with_director=False)
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        host_events = [e for e in result.debate_events if e.speaker_role == "host"]
        assert len(host_events) == 0

    def test_longer_debate_gets_transitions(self):
        """A longer debate should include mid-debate transition narrations."""
        # With target_turns=20, we should hit the 40% and 70% checkpoints
        config = make_test_config(target_turns=20, with_director=False)
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        host_events = [e for e in result.debate_events if e.speaker_role == "host"]
        # Should have at least intro + some transitions + outro
        assert len(host_events) >= 3

    def test_host_speaker_name_is_correct(self):
        """Host events should use the show_host name."""
        config = make_test_config(target_turns=6, with_director=False)
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        host_events = [e for e in result.debate_events if e.speaker_role == "host"]
        for event in host_events:
            assert event.speaker_name == "Marcus Webb"


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 6: MergedDebate with Hosts
# ═══════════════════════════════════════════════════════════════════════════

class TestMergedDebateHosts:
    """Tests for host data in MergedDebate output."""

    def _run_debate_with_host(self):
        """Helper: run a minimal debate with host and return result."""
        config = make_test_config(target_turns=4, with_director=False)
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)
        return conductor.run()

    def test_merged_has_hosts_tuple(self):
        """MergedDebate should include host names in hosts tuple."""
        result = self._run_debate_with_host()
        merged = result.to_merged_debate()

        assert "Marcus Webb" in merged.hosts

    def test_merged_chunk_has_hosts(self):
        """DebateChunk should include host names."""
        result = self._run_debate_with_host()
        merged = result.to_merged_debate()

        assert "Marcus Webb" in merged.chunks[0].hosts

    def test_merged_to_dict_has_hosts(self):
        """Serialized dict should include hosts key."""
        result = self._run_debate_with_host()
        merged = result.to_merged_debate()
        data = merged.to_dict()

        assert "hosts" in data
        assert "Marcus Webb" in data["hosts"]

    def test_merged_no_host_empty_tuple(self):
        """Without host, merged.hosts should be empty tuple."""
        config = make_test_config(target_turns=4, with_host=False, with_director=False)
        mock_client = make_mock_client()
        conductor = DebateConductor(config, client=mock_client)
        result = conductor.run()
        merged = result.to_merged_debate()

        assert merged.hosts == ()


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 7: Bootstrap Host Profiling
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapHostProfiling:
    """Tests for Stage 2.5 — host voice_fingerprint generation."""

    def _make_bootstrap_mock(self):
        """Build a mock client that handles all bootstrap stages."""
        mock = Mock()

        # Track call number to return different responses for each stage
        call_count = [0]

        # Architect response includes a show_host
        architect_json = json.dumps({
            "title": "Test Debate",
            "topic": "Is X good?",
            "debate_format": "public_forum",
            "setting": "Studio",
            "tone": "academic",
            "target_turns": 10,
            "participants": [
                {"name": "Alice", "side": "pro", "role": "participant"},
                {"name": "Bob", "side": "con", "role": "participant"},
            ],
            "moderator": {"name": "Charlie", "mode": "active", "style": "Neutral"},
            "audience": {"present": False, "participation": False, "size_description": ""},
            "show_host": {
                "name": "Elena Voss",
                "role": "narrator",
                "style": "Documentary-style, authoritative",
            },
        })

        # Profile builder response (used for participants AND host)
        profile_json = json.dumps({
            "voice_fingerprint": "Test voice fingerprint for {name}",
            "behavioral_triggers": "Test triggers",
            "emotional_arc": "Test arc",
            "key_positions": ["Position 1"],
            "factual_claims": [],
            "reference_corpus": ["Example quote"],
        })

        # Director response
        director_json = json.dumps({
            "dramatic_arc": "Two acts.",
            "mood_progression": ["calm", "heated"],
            "key_moments": ["Moment 1"],
            "closing_strategy": "Wrap up.",
        })

        def chat_side_effect(messages, **kwargs):
            call_count[0] += 1
            system = messages[0]["content"]
            if "Architect" in system:
                return architect_json
            elif "Character Psychologist" in system:
                return profile_json
            elif "Director" in system:
                return director_json
            return "{}"

        mock.chat.side_effect = chat_side_effect
        return mock

    def test_bootstrap_populates_host_fingerprint(self):
        """Bootstrap should call Profile Builder for the host."""
        mock_client = self._make_bootstrap_mock()
        bootstrap = DebateBootstrap(client=mock_client)

        config = bootstrap.from_brief("Test debate about X")

        # The host should exist and have a voice fingerprint
        assert config.show_host is not None
        assert config.show_host.name == "Elena Voss"
        assert config.show_host.voice_fingerprint is not None
        assert config.show_host.voice_fingerprint != ""

    def test_bootstrap_makes_extra_llm_call_for_host(self):
        """Bootstrap with host should make one extra LLM call (for host profiling)."""
        mock_client = self._make_bootstrap_mock()
        bootstrap = DebateBootstrap(client=mock_client)

        config = bootstrap.from_brief("Test debate about X")

        # Expected calls: 1 architect + 2 participant profiles + 1 host profile + 1 director = 5
        assert mock_client.chat.call_count == 5

    def test_bootstrap_without_host_no_extra_call(self):
        """Bootstrap without host should not make the extra profiling call."""
        mock_client = Mock()

        # Architect response WITHOUT show_host
        architect_json = json.dumps({
            "title": "Test", "topic": "X?", "debate_format": "public_forum",
            "setting": "Studio", "tone": "academic", "target_turns": 10,
            "participants": [
                {"name": "Alice", "side": "pro", "role": "participant"},
                {"name": "Bob", "side": "con", "role": "participant"},
            ],
            "moderator": {"name": "Charlie", "mode": "active", "style": "Neutral"},
            "audience": {"present": False, "participation": False, "size_description": ""},
        })

        profile_json = json.dumps({
            "voice_fingerprint": "Test fingerprint",
            "behavioral_triggers": "Test",
            "emotional_arc": "Test",
            "key_positions": [], "factual_claims": [], "reference_corpus": [],
        })

        director_json = json.dumps({
            "dramatic_arc": "Two acts.",
            "mood_progression": ["calm"],
            "key_moments": [], "closing_strategy": "End.",
        })

        def chat_side_effect(messages, **kwargs):
            system = messages[0]["content"]
            if "Architect" in system:
                return architect_json
            elif "Character Psychologist" in system:
                return profile_json
            elif "Director" in system:
                return director_json
            return "{}"

        mock_client.chat.side_effect = chat_side_effect
        bootstrap = DebateBootstrap(client=mock_client)
        config = bootstrap.from_brief("Test debate")

        # Without host: 1 architect + 2 profiles + 1 director = 4
        assert mock_client.chat.call_count == 4
        assert config.show_host is None
