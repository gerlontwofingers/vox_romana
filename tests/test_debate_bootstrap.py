"""Tests for the debate config models and bootstrap chain.

This file tests:
    1. Config construction and defaults
    2. Config validation (catches problems before they reach the conductor)
    3. Architect JSON parsing (canned LLM output → DebateConfig)
    4. Profile Builder JSON parsing (canned LLM output → ParticipantProfile)
    5. Director JSON parsing (canned LLM output → DirectorBrief)
    6. Full bootstrap chain with a mocked DeepSeekClient

All tests use canned JSON responses — no real LLM calls. The parsing logic
is deterministic and fast to test.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_debate_bootstrap.py -v
"""

import json
from unittest.mock import Mock

import pytest

from vox_romana import (
    AudienceConfig,
    ContentConfig,
    DebateArchitect,
    DebateBootstrap,
    DebateConfig,
    DebateDirector,
    DebateProfileBuilder,
    DirectorBrief,
    ModeratorConfig,
    ParticipantProfile,
)


# ═══════════════════════════════════════════════════════════════════════════
# CANNED LLM RESPONSES
# ═══════════════════════════════════════════════════════════════════════════
# These are the JSON blobs our mock LLM will return. They're realistic
# examples of what DeepSeek actually produces (minus minor formatting).

ARCHITECT_RESPONSE = json.dumps({
    "title": "The UBI Question: Promise or Pipe Dream?",
    "topic": "Should the US adopt universal basic income?",
    "debate_format": "public_forum",
    "setting": "University lecture hall, 200-seat capacity",
    "tone": "academic but accessible",
    "target_turns": 35,
    "participants": [
        {"name": "Dr. Sarah Chen", "side": "pro-UBI", "role": "participant"},
        {"name": "James Hartley", "side": "anti-UBI", "role": "participant"},
    ],
    "moderator": {
        "name": "Jordan Lee",
        "mode": "active",
        "style": "Socratic — asks probing questions, slightly favors the underdog",
    },
    "audience": {
        "present": True,
        "participation": False,
        "size_description": "200 university students and faculty",
    },
})

PROFILE_CHEN_RESPONSE = json.dumps({
    "voice_fingerprint": (
        "Opens with hedging phrases ('The data suggests...', 'It's worth noting...'). "
        "Builds to emphatic conclusions. Favors passive voice when citing studies. "
        "Uses rhetorical questions to disarm opponents. Short sentences when angry."
    ),
    "behavioral_triggers": (
        "When interrupted: finishes her sentence more firmly, volume rises slightly. "
        "When challenged with data: lights up, pulls out more statistics. "
        "When losing: pivots to moral arguments about human dignity."
    ),
    "emotional_arc": (
        "Starts measured and academic (turns 1-12). Becomes passionate after "
        "Hartley dismisses the Finland data (turns 13-20). Shows vulnerability "
        "when discussing personal motivation (turns 25-30). Ends gracious."
    ),
    "key_positions": [
        "UBI eliminates poverty traps inherent in means-tested welfare",
        "Finland's 2017 pilot showed improved wellbeing without reducing employment",
        "Automation will displace 30% of jobs by 2040 — we need a safety net now",
        "UBI is simpler and cheaper to administer than current welfare bureaucracy",
    ],
    "factual_claims": [
        "Finland's 2017-2018 UBI pilot: 2,000 participants received €560/month",
        "Stockton SEED program: $500/month to 125 residents, full-time employment rose 12%",
    ],
    "reference_corpus": [
        "The data suggests — and I want to be precise here — that basic income doesn't reduce work.",
        "Look, when we talk about 'welfare dependency,' what we really mean is 'poverty dependency.'",
        "I'd like to draw your attention to the Finnish results, because they're often misrepresented.",
    ],
})

PROFILE_HARTLEY_RESPONSE = json.dumps({
    "voice_fingerprint": (
        "Punchy declarative sentences. Favors economic metaphors ('you can't print "
        "prosperity', 'the bill always comes due'). Uses 'frankly' and 'let's be "
        "honest' as verbal tics. Slightly folksy cadence despite Ivy League credentials."
    ),
    "behavioral_triggers": (
        "When interrupted: raises hand, says 'let me finish.' "
        "When challenged with data: questions methodology rather than the numbers. "
        "When audience laughs: becomes more theatrical, plays to the crowd."
    ),
    "emotional_arc": (
        "Starts confident and slightly dismissive (turns 1-10). Sharpens when "
        "Chen lands a strong point (turns 11-18). Gets genuinely thoughtful "
        "about automation concerns (turns 25-30). Ends respectful but unswayed."
    ),
    "key_positions": [
        "UBI is unaffordable at scale — the math doesn't work",
        "Targeted programs are more efficient than universal payments",
        "UBI removes the dignity and purpose that comes from working",
        "Inflation would erode UBI's purchasing power within years",
    ],
    "factual_claims": [
        "A $1,000/month UBI for all US adults would cost ~$3.1 trillion annually",
        "Current total US welfare spending is approximately $700 billion",
    ],
    "reference_corpus": [
        "Frankly, the Finnish study is a red herring. Two thousand people is not a country.",
        "Let's be honest — nobody's against helping the poor. The question is HOW.",
        "You can't print prosperity. Someone has to pay for this, and it won't be the billionaires.",
    ],
})

DIRECTOR_RESPONSE = json.dumps({
    "dramatic_arc": (
        "Three-act structure. Act 1 (turns 1-10): polite opening statements, "
        "establishing credentials, first gentle disagreements. Act 2 (turns 11-25): "
        "escalation as Chen presents the Finland data and Hartley challenges the "
        "methodology — the debate gets personal around turn 18 when Hartley "
        "questions academic ivory towers. Act 3 (turns 26-35): peak tension "
        "followed by gradual de-escalation, closing statements with an unexpected "
        "moment of common ground on automation fears."
    ),
    "mood_progression": [
        "academic",
        "engaged",
        "pointed",
        "heated",
        "personal",
        "reflective",
        "conciliatory",
    ],
    "key_moments": [
        "Turn 8: Moderator asks both to respond to a specific Stockton data point",
        "Turn 18: Hartley makes a comment about 'ivory tower economics' — Chen takes it personally",
        "Turn 25: Chen shares why she got into economics (personal story)",
        "Turn 32: Hartley admits the automation argument gives him pause",
    ],
    "closing_strategy": (
        "Moderator announces final statements at turn 33. Each participant gets "
        "a 1-turn closing. Moderator summarizes the three key disagreements "
        "(cost, dignity, automation) and thanks both participants."
    ),
})


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: ContentConfig (base class)
# ═══════════════════════════════════════════════════════════════════════════

class TestContentConfig:
    """Tests for the base ContentConfig model."""

    def test_defaults(self):
        """ContentConfig with no args should have sane defaults."""
        c = ContentConfig()
        assert c.content_type == "generic"
        assert c.brief == ""
        assert c.reference_material is None
        assert c.fact_check == "off"
        assert c.title is None
        assert c.estimated_duration_minutes is None

    def test_validate_empty_brief(self):
        """Missing brief should be flagged."""
        c = ContentConfig(brief="")
        problems = c.validate()
        assert any("brief" in p.lower() for p in problems)

    def test_validate_fact_check_without_reference(self):
        """Enabling fact_check without reference_material should warn."""
        c = ContentConfig(brief="something", fact_check="quick")
        problems = c.validate()
        assert any("reference_material" in p for p in problems)

    def test_validate_ok(self):
        """Valid config should have no problems."""
        c = ContentConfig(
            brief="test brief",
            reference_material="some facts",
            fact_check="thorough",
        )
        assert c.validate() == []


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: DebateConfig construction and validation
# ═══════════════════════════════════════════════════════════════════════════

class TestDebateConfig:
    """Tests for DebateConfig construction and validation."""

    def test_defaults(self):
        """DebateConfig defaults should be debate-specific."""
        c = DebateConfig()
        assert c.content_type == "debate"
        assert c.participants == []
        assert c.moderator is None
        assert c.audience is None
        assert c.director_brief is None
        assert c.target_turns is None

    def test_inherits_content_config(self):
        """DebateConfig should inherit ContentConfig fields."""
        c = DebateConfig(brief="test", fact_check="quick")
        assert c.brief == "test"
        assert c.fact_check == "quick"

    def test_validate_too_few_participants(self):
        """Fewer than 2 participants should be flagged."""
        c = DebateConfig(
            brief="test",
            participants=[ParticipantProfile(name="Solo")],
        )
        problems = c.validate()
        assert any("at least 2" in p for p in problems)

    def test_validate_duplicate_names(self):
        """Duplicate participant names should be flagged."""
        c = DebateConfig(
            brief="test",
            participants=[
                ParticipantProfile(name="Alice"),
                ParticipantProfile(name="Alice"),
            ],
        )
        problems = c.validate()
        assert any("Duplicate" in p for p in problems)

    def test_validate_empty_name(self):
        """Participant with no name should be flagged."""
        c = DebateConfig(
            brief="test",
            participants=[
                ParticipantProfile(name=""),
                ParticipantProfile(name="Bob"),
            ],
        )
        problems = c.validate()
        assert any("no name" in p for p in problems)

    def test_validate_low_target_turns(self):
        """target_turns < 4 should be flagged."""
        c = DebateConfig(
            brief="test",
            target_turns=2,
            participants=[
                ParticipantProfile(name="A"),
                ParticipantProfile(name="B"),
            ],
        )
        problems = c.validate()
        assert any("too few" in p.lower() for p in problems)

    def test_validate_moderator_no_name(self):
        """Moderator with no name should be flagged."""
        c = DebateConfig(
            brief="test",
            moderator=ModeratorConfig(name=""),
            participants=[
                ParticipantProfile(name="A"),
                ParticipantProfile(name="B"),
            ],
        )
        problems = c.validate()
        assert any("Moderator" in p for p in problems)

    def test_validate_ok(self):
        """Fully valid config should have no problems."""
        c = DebateConfig(
            brief="test",
            participants=[
                ParticipantProfile(name="Alice", side="pro"),
                ParticipantProfile(name="Bob", side="con"),
            ],
            moderator=ModeratorConfig(name="Charlie"),
            target_turns=30,
        )
        assert c.validate() == []

    def test_mutability(self):
        """Config should be mutable (not frozen)."""
        c = DebateConfig(brief="old")
        c.brief = "new"
        assert c.brief == "new"
        c.participants.append(ParticipantProfile(name="Added"))
        assert len(c.participants) == 1


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: Architect JSON parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestDebateArchitect:
    """Tests for DebateArchitect.parse_response()."""

    def setup_method(self):
        """Create an Architect with a dummy client (won't be called)."""
        self.architect = DebateArchitect(client=Mock())

    def test_parse_valid_response(self):
        """Standard JSON response should parse into a complete config."""
        config = self.architect.parse_response(
            ARCHITECT_RESPONSE,
            brief="Two economists debate UBI",
            reference_material="Finland pilot data...",
            fact_check="quick",
        )
        assert config.title == "The UBI Question: Promise or Pipe Dream?"
        assert config.topic == "Should the US adopt universal basic income?"
        assert config.debate_format == "public_forum"
        assert config.target_turns == 35
        assert config.brief == "Two economists debate UBI"
        assert config.fact_check == "quick"
        assert len(config.participants) == 2
        assert config.participants[0].name == "Dr. Sarah Chen"
        assert config.participants[0].side == "pro-UBI"
        assert config.participants[1].name == "James Hartley"
        assert config.moderator is not None
        assert config.moderator.name == "Jordan Lee"
        assert config.audience is not None
        assert config.audience.present is True

    def test_parse_markdown_wrapped(self):
        """JSON wrapped in ```json ... ``` fences should still parse."""
        wrapped = f"```json\n{ARCHITECT_RESPONSE}\n```"
        config = self.architect.parse_response(wrapped, brief="test")
        assert config.title == "The UBI Question: Promise or Pipe Dream?"

    def test_parse_invalid_json(self):
        """Garbage input should raise ValueError."""
        with pytest.raises(ValueError, match="invalid JSON"):
            self.architect.parse_response("this is not json", brief="test")

    def test_parse_no_moderator(self):
        """Response without moderator should result in None."""
        data = {"title": "Test", "participants": []}
        config = self.architect.parse_response(json.dumps(data), brief="test")
        assert config.moderator is None

    def test_parse_no_audience(self):
        """Response without audience should result in None."""
        data = {"title": "Test", "participants": []}
        config = self.architect.parse_response(json.dumps(data), brief="test")
        assert config.audience is None


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: Profile Builder JSON parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestDebateProfileBuilder:
    """Tests for DebateProfileBuilder.parse_response()."""

    def setup_method(self):
        self.builder = DebateProfileBuilder(client=Mock())
        self.stub = ParticipantProfile(name="Dr. Sarah Chen", side="pro-UBI")

    def test_parse_valid_response(self):
        """Standard JSON response should produce a full profile."""
        profile = self.builder.parse_response(
            PROFILE_CHEN_RESPONSE, stub=self.stub
        )
        # Identity fields preserved from stub
        assert profile.name == "Dr. Sarah Chen"
        assert profile.side == "pro-UBI"
        assert profile.role == "participant"
        # Personality fields populated
        assert "hedging" in profile.voice_fingerprint
        assert "interrupted" in profile.behavioral_triggers
        assert len(profile.key_positions) == 4
        assert len(profile.factual_claims) == 2
        assert len(profile.reference_corpus) == 3

    def test_parse_preserves_stub_identity(self):
        """Profile builder must preserve name/side/role from the stub."""
        stub = ParticipantProfile(name="Custom Name", side="custom-side", role="moderator")
        profile = self.builder.parse_response(PROFILE_CHEN_RESPONSE, stub=stub)
        assert profile.name == "Custom Name"
        assert profile.side == "custom-side"
        assert profile.role == "moderator"

    def test_parse_invalid_json(self):
        """Garbage input should raise ValueError."""
        with pytest.raises(ValueError, match="invalid JSON"):
            self.builder.parse_response("not json", stub=self.stub)

    def test_parse_missing_fields(self):
        """Missing fields should default gracefully, not crash."""
        data = {"voice_fingerprint": "test"}
        profile = self.builder.parse_response(json.dumps(data), stub=self.stub)
        assert profile.voice_fingerprint == "test"
        assert profile.key_positions == []
        assert profile.factual_claims == []

    def test_parse_voice_hint_and_tts_style(self):
        """voice_hint and tts_style should be extracted when present."""
        data = {
            "voice_gender": "female",
            "voice_hint": "warm British alto, late 30s, academic",
            "tts_style": "Speak with measured precision and quiet confidence.",
            "voice_fingerprint": "hedging phrases",
        }
        profile = self.builder.parse_response(json.dumps(data), stub=self.stub)
        assert profile.voice_hint == "warm British alto, late 30s, academic"
        assert profile.tts_style == "Speak with measured precision and quiet confidence."
        assert profile.voice_gender == "female"

    def test_parse_missing_voice_hint_defaults_none(self):
        """voice_hint and tts_style should be None when LLM omits them."""
        data = {"voice_gender": "male", "voice_fingerprint": "punchy"}
        profile = self.builder.parse_response(json.dumps(data), stub=self.stub)
        assert profile.voice_hint is None
        assert profile.tts_style is None


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: Director JSON parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestDebateDirector:
    """Tests for DebateDirector.parse_response()."""

    def setup_method(self):
        self.director = DebateDirector(client=Mock())

    def test_parse_valid_response(self):
        """Standard JSON response should produce a complete DirectorBrief."""
        brief = self.director.parse_response(DIRECTOR_RESPONSE)
        assert "Three-act" in brief.dramatic_arc
        assert len(brief.mood_progression) == 7
        assert brief.mood_progression[0] == "academic"
        assert len(brief.key_moments) == 4
        assert "final statements" in brief.closing_strategy

    def test_parse_markdown_wrapped(self):
        """JSON wrapped in markdown fences should still parse."""
        wrapped = f"```json\n{DIRECTOR_RESPONSE}\n```"
        brief = self.director.parse_response(wrapped)
        assert "Three-act" in brief.dramatic_arc

    def test_parse_invalid_json(self):
        """Garbage input should raise ValueError."""
        with pytest.raises(ValueError, match="invalid JSON"):
            self.director.parse_response("not json")

    def test_default_check_interval(self):
        """Parsed DirectorBrief should have default check_interval=7."""
        brief = self.director.parse_response(DIRECTOR_RESPONSE)
        assert brief.check_interval == 7


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 6: Bootstrap orchestrator (full chain with mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestDebateBootstrap:
    """Tests the full bootstrap chain with a mocked DeepSeekClient.

    The mock returns different JSON depending on which agent is calling.
    We inspect the system prompt to figure out which agent it is.
    """

    def _mock_chat(self, messages, **kwargs):
        """Route LLM calls to the right canned response based on system prompt."""
        system = messages[0]["content"]
        if "Architect" in system:
            return ARCHITECT_RESPONSE
        elif "Character Psychologist" in system:
            # Return different profiles based on participant name in user message
            user = messages[1]["content"]
            if "Sarah Chen" in user:
                return PROFILE_CHEN_RESPONSE
            else:
                return PROFILE_HARTLEY_RESPONSE
        elif "Director" in system:
            return DIRECTOR_RESPONSE
        else:
            return "{}"

    def setup_method(self):
        """Create a bootstrap with a mocked client."""
        self.mock_client = Mock()
        self.mock_client.chat.side_effect = self._mock_chat
        self.bootstrap = DebateBootstrap(client=self.mock_client)

    def test_full_chain(self):
        """The complete bootstrap should produce a fully populated config."""
        config = self.bootstrap.from_brief(
            "Two economists debate universal basic income",
            reference_material="Finland's 2017 UBI pilot...",
            fact_check="quick",
        )

        # ── Architect output ──────────────────────────────────────────
        assert config.title == "The UBI Question: Promise or Pipe Dream?"
        assert config.debate_format == "public_forum"
        assert config.fact_check == "quick"
        assert config.brief == "Two economists debate universal basic income"

        # ── Profile Builder output ────────────────────────────────────
        assert len(config.participants) == 2
        chen = config.participants[0]
        assert chen.name == "Dr. Sarah Chen"
        assert chen.voice_fingerprint is not None  # fleshed out
        assert len(chen.key_positions) > 0

        hartley = config.participants[1]
        assert hartley.name == "James Hartley"
        assert hartley.voice_fingerprint is not None
        assert "frankly" in hartley.voice_fingerprint.lower()

        # ── Director output ───────────────────────────────────────────
        assert config.director_brief is not None
        assert "Three-act" in config.director_brief.dramatic_arc
        assert len(config.director_brief.mood_progression) > 0

    def test_call_count(self):
        """Bootstrap should make exactly 4 LLM calls for 2 participants."""
        # 1 architect + 2 profile builders + 1 director = 4
        self.bootstrap.from_brief("test debate")
        assert self.mock_client.chat.call_count == 4

    def test_overrides_applied(self):
        """User overrides should replace agent-generated values."""
        config = self.bootstrap.from_brief(
            "test debate",
            overrides={
                "tone": "unhinged cable news",
                "target_turns": 99,
                "debate_format": "crossfire",
            },
        )
        assert config.tone == "unhinged cable news"
        assert config.target_turns == 99
        assert config.debate_format == "crossfire"

    def test_unknown_overrides_ignored(self):
        """Unknown override keys should NOT crash."""
        config = self.bootstrap.from_brief(
            "test debate",
            overrides={"nonexistent_field_xyz": "value"},
        )
        # Should complete without error
        assert config.title is not None

    def test_reference_material_passed_through(self):
        """Reference material should end up on the final config."""
        config = self.bootstrap.from_brief(
            "test debate",
            reference_material="Important facts here",
        )
        assert config.reference_material == "Important facts here"

    def test_fact_check_default_off(self):
        """Default fact_check should be 'off'."""
        config = self.bootstrap.from_brief("test debate")
        assert config.fact_check == "off"
