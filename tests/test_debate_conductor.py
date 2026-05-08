"""Tests for the DebateConductor — the runtime debate generation engine.

This file tests:
    1. Prompt building (system prompts for participants, moderator)
    2. Turn response parsing (JSON extraction, fallback for bad JSON)
    3. Speaker selection logic (round-robin, moderator interjections)
    4. Full turn loop with mocked LLM
    5. Result packaging (DebateEvent creation, MergedDebate conversion)
    6. Context compression triggering
    7. Director check-in triggering
    8. from_brief() convenience constructor

All tests use a mocked DeepSeekClient — no real LLM calls.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_debate_conductor.py -v
"""

import json
from unittest.mock import Mock, call, patch

import pytest

from vox_romana import (
    AudienceConfig,
    DebateConductor,
    DebateConductorResult,
    DebateConfig,
    DebateEvent,
    DirectorBrief,
    MergedDebate,
    ModeratorConfig,
    ParticipantProfile,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES — reusable config and canned LLM responses
# ═══════════════════════════════════════════════════════════════════════════

def make_test_config(*, target_turns=10, with_moderator=True, with_director=True):
    """Build a minimal but complete DebateConfig for testing."""
    participants = [
        ParticipantProfile(
            name="Alice",
            side="pro",
            voice_fingerprint="Short punchy sentences. Uses 'Look,' as opener.",
            behavioral_triggers="When interrupted, gets louder.",
            emotional_arc="Starts calm, gets passionate.",
            key_positions=["Point A is correct", "Point B supports A"],
            factual_claims=["Study X showed 40% improvement"],
            reference_corpus=["Look, the data is clear."],
        ),
        ParticipantProfile(
            name="Bob",
            side="con",
            voice_fingerprint="Long compound sentences. Favors analogies.",
            behavioral_triggers="When challenged, pivots to anecdotes.",
            emotional_arc="Starts confident, becomes reflective.",
            key_positions=["Point A is flawed", "Point C is better"],
            factual_claims=["Report Y found no significant effect"],
            reference_corpus=["Well, if you think about it carefully..."],
        ),
    ]

    moderator = None
    if with_moderator:
        moderator = ModeratorConfig(
            name="Charlie",
            mode="active",
            style="Socratic — probing questions",
        )

    director_brief = None
    if with_director:
        director_brief = DirectorBrief(
            dramatic_arc="Two acts. Act 1: establish. Act 2: escalate.",
            mood_progression=["calm", "engaged", "heated", "reflective"],
            key_moments=["Turn 5: sharp exchange", "Turn 8: concession"],
            closing_strategy="Moderator summarizes, each gets a closing line.",
            check_interval=4,
        )

    return DebateConfig(
        brief="Test debate about topic X",
        title="The X Debate",
        topic="Is X better than Y?",
        debate_format="public_forum",
        setting="Small seminar room",
        tone="academic",
        participants=participants,
        moderator=moderator,
        audience=AudienceConfig(present=True, participation=False),
        director_brief=director_brief,
        target_turns=target_turns,
    )


def make_turn_json(text="Test response.", **overrides):
    """Build a canned JSON response string like an agent would produce."""
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


def make_director_json(mood_note="Keep energy up.", nudge=None, wrap_up=False):
    """Build a canned Director check-in response."""
    return json.dumps({
        "mood_note": mood_note,
        "nudge": nudge,
        "wrap_up": wrap_up,
    })


SUMMARY_RESPONSE = "Alice argued pro with data. Bob countered with methodology critique."


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: Prompt Building
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptBuilding:
    """Tests that system prompts are correctly assembled from config."""

    def test_participant_prompt_includes_personality(self):
        """Participant system prompts should contain voice fingerprint and claims."""
        config = make_test_config()
        mock_client = Mock()
        conductor = DebateConductor(config, client=mock_client)
        conductor._build_agents()

        prompt = conductor._participant_prompts["Alice"]
        assert "Alice" in prompt
        assert "pro" in prompt
        assert "Short punchy sentences" in prompt
        assert "When interrupted" in prompt
        assert "Study X showed 40%" in prompt
        assert "Look, the data is clear." in prompt

    def test_participant_prompt_includes_topic(self):
        """System prompt should reference the debate topic."""
        config = make_test_config()
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        prompt = conductor._participant_prompts["Bob"]
        assert "Is X better than Y?" in prompt

    def test_moderator_prompt_includes_style(self):
        """Moderator prompt should contain style and participant list."""
        config = make_test_config()
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        prompt = conductor._moderator_prompt
        assert "Charlie" in prompt
        assert "Socratic" in prompt
        assert "Alice" in prompt
        assert "Bob" in prompt

    def test_no_moderator_means_no_prompt(self):
        """Config without moderator should result in None prompt."""
        config = make_test_config(with_moderator=False)
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        assert conductor._moderator_prompt is None


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: Turn Response Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestTurnParsing:
    """Tests for _parse_turn_response — extracting JSON from LLM output."""

    def setup_method(self):
        self.conductor = DebateConductor(make_test_config(), client=Mock())

    def test_parse_clean_json(self):
        """Clean JSON should parse correctly."""
        result = self.conductor._parse_turn_response(make_turn_json("Hello world."))
        assert result["text"] == "Hello world."
        assert result["tone"] == "neutral"

    def test_parse_markdown_wrapped(self):
        """JSON wrapped in ```json fences should parse."""
        wrapped = f"```json\n{make_turn_json('Wrapped.')}\n```"
        result = self.conductor._parse_turn_response(wrapped)
        assert result["text"] == "Wrapped."

    def test_parse_with_preamble(self):
        """JSON with preamble text should extract the JSON part."""
        preambled = f"Here's my response:\n{make_turn_json('After preamble.')}"
        result = self.conductor._parse_turn_response(preambled)
        assert result["text"] == "After preamble."

    def test_parse_garbage_falls_back(self):
        """Totally unparseable text should fall back to raw text."""
        result = self.conductor._parse_turn_response("I refuse to use JSON!")
        assert result["text"] == "I refuse to use JSON!"
        assert result["tone"] == "neutral"

    def test_parse_nonverbal_string_to_list(self):
        """A nonverbal string value should be converted to a list."""
        data = json.dumps({"text": "Ha!", "nonverbal": "laughter"})
        result = self.conductor._parse_turn_response(data)
        assert result["nonverbal"] == ["laughter"]

    def test_parse_empty_nonverbal_string(self):
        """Empty nonverbal string should become empty list."""
        data = json.dumps({"text": "Ok.", "nonverbal": ""})
        result = self.conductor._parse_turn_response(data)
        assert result["nonverbal"] == []


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: Speaker Selection
# ═══════════════════════════════════════════════════════════════════════════

class TestSpeakerSelection:
    """Tests for _pick_next_speaker logic."""

    def setup_method(self):
        self.conductor = DebateConductor(make_test_config(), client=Mock())
        self.conductor._build_agents()

    def test_moderator_opens(self):
        """Turn 1 with moderator should select the moderator."""
        name, role = self.conductor._pick_next_speaker(
            participant_names=["Alice", "Bob"],
            participant_index=0,
            turns_since_moderator=0,
            wrapping_up=False,
            turn_number=1,
            target_turns=10,
        )
        assert name == "Charlie"
        assert role == "moderator"

    def test_participant_round_robin(self):
        """After moderator opens, participants should alternate."""
        name, role = self.conductor._pick_next_speaker(
            participant_names=["Alice", "Bob"],
            participant_index=0,
            turns_since_moderator=1,
            wrapping_up=False,
            turn_number=2,
            target_turns=10,
        )
        assert name == "Alice"
        assert role == "participant"

    def test_moderator_interjects(self):
        """Moderator should interject after DEFAULT_MODERATOR_INTERVAL turns."""
        from vox_romana.shows.debates.conductor import DEFAULT_MODERATOR_INTERVAL
        name, role = self.conductor._pick_next_speaker(
            participant_names=["Alice", "Bob"],
            participant_index=0,
            turns_since_moderator=DEFAULT_MODERATOR_INTERVAL,
            wrapping_up=False,
            turn_number=5,
            target_turns=10,
        )
        assert name == "Charlie"
        assert role == "moderator"

    def test_no_moderator_goes_straight_to_participant(self):
        """Without a moderator, turn 1 should be a participant."""
        config = make_test_config(with_moderator=False)
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()

        name, role = conductor._pick_next_speaker(
            participant_names=["Alice", "Bob"],
            participant_index=0,
            turns_since_moderator=0,
            wrapping_up=False,
            turn_number=1,
            target_turns=10,
        )
        assert role == "participant"


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: Full Turn Loop (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestTurnLoop:
    """Tests the full run() with a mocked DeepSeekClient."""

    def _make_mock_client(self):
        """Create a mock client that returns appropriate JSON for each call."""
        mock = Mock()
        # The mock needs to return different things for different calls.
        # We'll use side_effect with a function that inspects the system prompt.
        def chat_side_effect(messages, **kwargs):
            system = messages[0]["content"]
            if "concise summarizer" in system.lower():
                return SUMMARY_RESPONSE
            elif "debate director" in system.lower():
                return make_director_json()
            elif "moderating" in system:
                return make_moderator_json()
            else:
                return make_turn_json("I have a point to make.")

        mock.chat.side_effect = chat_side_effect
        return mock

    def test_run_produces_result(self):
        """run() should produce a DebateConductorResult with events."""
        config = make_test_config(target_turns=6, with_director=False)
        mock_client = self._make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        assert isinstance(result, DebateConductorResult)
        assert len(result.debate_events) > 0
        assert result.total_llm_calls > 0
        assert result.debate_id != ""

    def test_events_are_typed(self):
        """Result events should be proper DebateEvent objects."""
        config = make_test_config(target_turns=4, with_director=False)
        mock_client = self._make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        for event in result.debate_events:
            assert isinstance(event, DebateEvent)
            assert event.global_order > 0
            assert event.speaker_name != ""
            assert event.text != ""

    def test_moderator_speaks_first(self):
        """With a moderator, the first event should be from the moderator."""
        config = make_test_config(target_turns=4, with_director=False)
        mock_client = self._make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        assert result.debate_events[0].speaker_name == "Charlie"
        assert result.debate_events[0].speaker_role == "moderator"

    def test_transcript_populated(self):
        """Result should include a human-readable transcript."""
        config = make_test_config(target_turns=4, with_director=False)
        mock_client = self._make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        assert len(result.transcript) > 0
        assert all(isinstance(line, str) for line in result.transcript)
        assert any("[Charlie]" in line for line in result.transcript)

    def test_no_moderator_still_works(self):
        """Debate without moderator should still produce events."""
        config = make_test_config(target_turns=4, with_moderator=False, with_director=False)
        mock_client = self._make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        assert len(result.debate_events) > 0
        roles = {e.speaker_role for e in result.debate_events}
        assert "moderator" not in roles

    def test_director_check_fires(self):
        """Director check should fire at the configured interval."""
        config = make_test_config(target_turns=8, with_director=True)
        config.director_brief.check_interval = 3
        mock_client = self._make_mock_client()
        conductor = DebateConductor(config, client=mock_client)

        result = conductor.run()

        # Director checks happen at turns divisible by check_interval (3, 6, 9...).
        # The director_note should be populated from at least one check.
        # We verify by checking that we made more LLM calls than just
        # participant/moderator turns.
        turn_calls = len(result.debate_events)
        total = result.total_llm_calls
        assert total > turn_calls  # Extra calls = director + possibly compression


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: MergedDebate Output
# ═══════════════════════════════════════════════════════════════════════════

class TestMergedDebateOutput:
    """Tests for to_merged_debate() — bridging conductor output to pipeline."""

    def _run_short_debate(self):
        """Helper: run a minimal debate and return the result."""
        config = make_test_config(target_turns=4, with_director=False)
        mock_client = Mock()
        mock_client.chat.return_value = make_turn_json("Short response.")
        conductor = DebateConductor(config, client=mock_client)
        return conductor.run()

    def test_to_merged_debate_type(self):
        """to_merged_debate() should return a MergedDebate."""
        result = self._run_short_debate()
        merged = result.to_merged_debate()
        assert isinstance(merged, MergedDebate)

    def test_merged_has_events(self):
        """MergedDebate should contain the same events."""
        result = self._run_short_debate()
        merged = result.to_merged_debate()
        assert len(merged.events) == len(result.debate_events)

    def test_merged_metadata(self):
        """MergedDebate should have correct metadata from config."""
        result = self._run_short_debate()
        merged = result.to_merged_debate()
        assert merged.title == "The X Debate"
        assert merged.topic == "Is X better than Y?"
        assert merged.debate_format == "public_forum"
        assert "Alice" in merged.participants
        assert "Bob" in merged.participants

    def test_merged_serializable(self):
        """MergedDebate should be serializable via to_dict()."""
        result = self._run_short_debate()
        merged = result.to_merged_debate()
        data = merged.to_dict()
        assert isinstance(data, dict)
        assert "events" in data
        assert "title" in data

    def test_merged_single_chunk(self):
        """Conductor output should be wrapped in a single chunk."""
        result = self._run_short_debate()
        merged = result.to_merged_debate()
        assert len(merged.chunks) == 1
        assert merged.chunks[0].is_final_chunk is True


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 6: Context Compression
# ═══════════════════════════════════════════════════════════════════════════

class TestContextCompression:
    """Tests for context compression behavior."""

    def test_compression_produces_summary(self):
        """After enough turns, _compress_context should call the LLM for a summary."""
        config = make_test_config(target_turns=4, with_director=False)
        mock_client = Mock()
        mock_client.chat.return_value = "Summary of the debate so far."
        conductor = DebateConductor(config, client=mock_client)
        conductor._build_agents()

        # Simulate a transcript with enough turns to trigger compression.
        from vox_romana.shows.debates.conductor import DEFAULT_RECENT_WINDOW
        for i in range(DEFAULT_RECENT_WINDOW + 5):
            conductor._transcript.append({
                "speaker": "Alice",
                "role": "participant",
                "text": f"Turn {i} statement.",
                "turn": i + 1,
            })

        conductor._compress_context()

        assert conductor._summary == "Summary of the debate so far."
        assert mock_client.chat.call_count == 1

    def test_no_compression_when_few_old_turns(self):
        """Compression should skip if not enough old turns to summarize."""
        config = make_test_config()
        mock_client = Mock()
        conductor = DebateConductor(config, client=mock_client)
        conductor._build_agents()

        # Only 2 turns — not enough to compress.
        conductor._transcript = [
            {"speaker": "Alice", "text": "Hi.", "role": "participant", "turn": 1},
            {"speaker": "Bob", "text": "Hello.", "role": "participant", "turn": 2},
        ]

        conductor._compress_context()

        # Should not have called the LLM.
        mock_client.chat.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION — DebateDirectorBrief.dramatic_arc attribute access
# ═══════════════════════════════════════════════════════════════════════════

class TestPreflightCheckBootstrap:
    """Regression tests for _preflight_check_bootstrap.

    DebateDirectorBrief uses `dramatic_arc` (not `arc`).  Accessing `.arc`
    previously raised AttributeError and aborted the conductor mid-run.
    """

    def _make_conductor_with_fact_checker(self, dramatic_arc: str = "") -> tuple:
        """Return (conductor, mock_fact_checker) with a minimal config."""
        from vox_romana.shows.debates.config import DebateDirectorBrief

        config = make_test_config()
        config.director_brief = DebateDirectorBrief(dramatic_arc=dramatic_arc)
        config.topic = "Garlic trade policy"
        mock_client = Mock()
        conductor = DebateConductor(config, client=mock_client)
        conductor._build_agents()

        mock_fc = Mock()
        mock_report = Mock()
        mock_report.has_fabrications = False
        mock_fc.check_content.return_value = mock_report
        conductor._fact_checker = mock_fc
        return conductor, mock_fc

    def test_no_attribute_error_with_dramatic_arc_set(self):
        """No AttributeError when dramatic_arc is populated."""
        conductor, mock_fc = self._make_conductor_with_fact_checker(
            dramatic_arc="Act 1: posturing. Act 2: clash. Act 3: détente."
        )
        # Must not raise AttributeError
        conductor._preflight_check_bootstrap()
        mock_fc.check_content.assert_called_once()

    def test_no_attribute_error_with_empty_dramatic_arc(self):
        """No AttributeError when dramatic_arc is empty string."""
        conductor, mock_fc = self._make_conductor_with_fact_checker(dramatic_arc="")
        conductor._preflight_check_bootstrap()
        # topic-only surface — still calls fact-checker (topic is non-empty)
        mock_fc.check_content.assert_called_once()

    def test_dramatic_arc_included_in_surfaces_when_non_empty(self):
        """dramatic_arc text is passed to fact-checker when present."""
        arc_text = "Three-act structure with escalating tension."
        conductor, mock_fc = self._make_conductor_with_fact_checker(dramatic_arc=arc_text)
        conductor._preflight_check_bootstrap()
        checked_text = mock_fc.check_content.call_args[0][0]
        assert arc_text in checked_text

    def test_no_fact_checker_skips_silently(self):
        """If _fact_checker is None, _preflight_check_bootstrap returns immediately."""
        from vox_romana.shows.debates.config import DebateDirectorBrief

        config = make_test_config()
        config.director_brief = DebateDirectorBrief(dramatic_arc="some arc")
        conductor = DebateConductor(config, client=Mock())
        conductor._build_agents()
        conductor._fact_checker = None
        # Must not raise
        conductor._preflight_check_bootstrap()
