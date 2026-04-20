"""Tests for the news show infrastructure.

Tests:
    1. Config validation
    2. Bootstrap architect parsing (canned JSON → NewsConfig)
    3. Bootstrap director parsing (canned JSON → DirectorBrief)
    4. Conductor prompt building
    5. Conductor response parsing (JSON extraction, fallback)
    6. Full conductor run with mocked LLM
    7. Result packaging (NewsEvent creation, MergedNewsShow conversion)
    8. Serializer round-trip
    9. Host event generation

All tests use a mocked DeepSeekClient — no real LLM calls.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_news_conductor.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from vox_romana.core.config import ShowHostConfig
from vox_romana.shows.news.bootstrap import NewsBootstrap
from vox_romana.shows.news.conductor import (
    NewsConductor,
    NewsConductorResult,
)
from vox_romana.shows.news.config import (
    AnchorConfig,
    DirectorBrief,
    NewsConfig,
    SegmentConfig,
)
from vox_romana.shows.news.models import MergedNewsShow, NewsEvent, NewsSegment
from vox_romana.shows.news.serializer import NewsSerializer


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def make_test_config(
    *,
    num_segments=3,
    with_host=True,
    with_director=True,
):
    """Build a minimal but complete NewsConfig for testing."""
    segments = []
    for i in range(num_segments):
        segments.append(SegmentConfig(
            headline=f"Story {i + 1} Headline",
            angle=f"Focus on aspect {i + 1}",
            key_facts=[f"Fact {i + 1}A: 42% increase", f"Fact {i + 1}B: launched in 2025"],
            tone="authoritative" if i == 0 else "conversational",
            duration_hint="medium" if i < num_segments - 1 else "short",
            segment_type="anchor_read",
        ))

    anchors = [
        AnchorConfig(
            name="Jordan Lee",
            style="Warm, authoritative NPR delivery",
            voice_fingerprint="Measured cadence, emphasis on key numbers, slight warmth.",
        ),
    ]

    show_host = None
    if with_host:
        show_host = ShowHostConfig(
            name="Marcus Webb",
            role="emcee",
            style="Polished, engaging, podcast-host energy",
            voice_fingerprint="Smooth transitions, vivid scene-setting.",
        )

    director_brief = None
    if with_director:
        director_brief = DirectorBrief(
            show_arc="Lead with the heaviest story, middle segments lighter, close with kicker.",
            mood_progression=["urgent", "analytical", "conversational", "warm"],
            segment_order_notes="Priority ordering — most important first.",
            closing_strategy="Anchor delivers kicker, host wraps with headline recap.",
            check_interval=2,
        )

    return NewsConfig(
        brief="Test news broadcast",
        title="The Evening Brief",
        topic="Today's top stories",
        tone="authoritative",
        setting="Modern news studio",
        anchors=anchors,
        segments=segments,
        show_host=show_host,
        director_brief=director_brief,
        target_segments=num_segments,
    )


def make_anchor_json(text="Here's what happened today.", **overrides):
    """Build a canned anchor JSON response."""
    data = {
        "text": text,
        "tone": "authoritative",
        "volume": "moderate",
        "pace": "medium",
        "pause_before": "short",
        "pause_after": "medium",
    }
    data.update(overrides)
    return json.dumps(data)


def make_host_json(text="Welcome to the show."):
    """Build a canned host narration JSON response."""
    return json.dumps({
        "text": text,
        "tone": "warm",
        "volume": "moderate",
        "pace": "medium",
        "pause_before": "none",
        "pause_after": "long",
    })


def make_director_json(mood_note="Keep the energy up.", nudge=None, wrap_up=False):
    """Build a canned director check-in response."""
    return json.dumps({
        "mood_note": mood_note,
        "nudge": nudge,
        "wrap_up": wrap_up,
    })


SUMMARY_RESPONSE = "Covered Story 1 and Story 2 so far."


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: Config Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestNewsConfig:
    """Tests for NewsConfig.validate()."""

    def test_valid_config_has_no_problems(self):
        config = make_test_config()
        assert config.validate() == []

    def test_missing_brief(self):
        config = make_test_config()
        config.brief = ""
        problems = config.validate()
        assert any("brief" in p.lower() for p in problems)

    def test_no_anchors(self):
        config = make_test_config()
        config.anchors = []
        problems = config.validate()
        assert any("anchor" in p.lower() for p in problems)

    def test_anchor_no_name(self):
        config = make_test_config()
        config.anchors = [AnchorConfig(name="", style="test")]
        problems = config.validate()
        assert any("name" in p.lower() for p in problems)

    def test_no_segments(self):
        config = make_test_config()
        config.segments = []
        problems = config.validate()
        assert any("segment" in p.lower() for p in problems)

    def test_segment_no_headline(self):
        config = make_test_config()
        config.segments = [SegmentConfig(headline="", angle="test")]
        problems = config.validate()
        assert any("headline" in p.lower() for p in problems)

    def test_duplicate_anchor_names(self):
        config = make_test_config()
        config.anchors = [
            AnchorConfig(name="Jordan Lee"),
            AnchorConfig(name="Jordan Lee"),
        ]
        problems = config.validate()
        assert any("duplicate" in p.lower() for p in problems)

    def test_content_type_is_news(self):
        config = make_test_config()
        assert config.content_type == "news"


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: Prompt Building
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptBuilding:
    """Tests that system prompts are correctly assembled from config."""

    def test_anchor_prompt_includes_name_and_topic(self):
        config = make_test_config()
        conductor = NewsConductor(config, client=Mock())
        conductor._build_agents()

        prompt = conductor._anchor_prompts["Jordan Lee"]
        assert "Jordan Lee" in prompt
        assert "Today's top stories" in prompt

    def test_anchor_prompt_includes_style(self):
        config = make_test_config()
        conductor = NewsConductor(config, client=Mock())
        conductor._build_agents()

        prompt = conductor._anchor_prompts["Jordan Lee"]
        assert "NPR" in prompt

    def test_host_prompt_includes_name_and_role(self):
        config = make_test_config()
        conductor = NewsConductor(config, client=Mock())
        conductor._build_agents()

        assert conductor._host_prompt is not None
        assert "Marcus Webb" in conductor._host_prompt
        assert "EMCEE" in conductor._host_prompt

    def test_no_host_means_no_prompt(self):
        config = make_test_config(with_host=False)
        conductor = NewsConductor(config, client=Mock())
        conductor._build_agents()

        assert conductor._host_prompt is None


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: Response Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseParsing:
    """Tests for _parse_response — extracting JSON from LLM output."""

    def setup_method(self):
        self.conductor = NewsConductor(make_test_config(), client=Mock())

    def test_parse_clean_json(self):
        result = self.conductor._parse_response(make_anchor_json("Breaking news."))
        assert result["text"] == "Breaking news."

    def test_parse_markdown_wrapped(self):
        wrapped = f"```json\n{make_anchor_json('Wrapped.')}\n```"
        result = self.conductor._parse_response(wrapped)
        assert result["text"] == "Wrapped."

    def test_parse_with_preamble(self):
        preambled = f"Here's my delivery:\n{make_anchor_json('After preamble.')}"
        result = self.conductor._parse_response(preambled)
        assert result["text"] == "After preamble."

    def test_parse_garbage_falls_back(self):
        result = self.conductor._parse_response("I refuse to use JSON!")
        assert result["text"] == "I refuse to use JSON!"
        assert result["tone"] == "neutral"

    def test_parse_truncates_long_fallback(self):
        long_text = "A" * 600
        result = self.conductor._parse_response(long_text)
        assert len(result["text"]) <= 500


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: Full Conductor Run (Mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestConductorRun:
    """Tests for the full conductor lifecycle with mocked LLM calls."""

    def _make_mock_client(self, num_segments=3, with_host=True, with_director=True):
        """Build a mock client that returns appropriate canned responses.

        Response order (with host + 3 segments + director at interval 2):
          1. Host intro
          2. Segment 1 (anchor)
          3. Host transition
          4. Segment 2 (anchor)
          5. Director check (after segment 2)
          6. Host transition
          7. Segment 3 (anchor)
          8. Director check (after segment 3... wait, 3 % 2 != 0, skip)
          ... actually check_interval=2, so check at segment 2 (2%2==0)
             but we also check at 2 because seg_num=2 and 2 % 2 == 0.
             We need to be careful about the exact calling order.
        """
        mock_client = Mock()

        responses = []

        # Host intro (if host)
        if with_host:
            responses.append(make_host_json("Good evening, welcome to The Evening Brief."))

        for i in range(num_segments):
            # Host transition (except before first segment)
            if with_host and i > 0:
                responses.append(make_host_json(f"Now let's turn to our next story."))

            # Anchor delivery
            responses.append(make_anchor_json(f"Segment {i + 1} delivery content."))

            # Director check at intervals
            seg_num = i + 1
            if with_director and seg_num > 0 and seg_num % 2 == 0:
                responses.append(make_director_json())

        # Host outro (if host)
        if with_host:
            responses.append(make_host_json("That's all for tonight. Good night."))

        mock_client.chat.side_effect = responses
        return mock_client

    def test_basic_run_produces_events(self):
        """A basic run should produce events for each segment plus host narration."""
        config = make_test_config(num_segments=3, with_director=False)
        mock_client = self._make_mock_client(num_segments=3, with_director=False)
        conductor = NewsConductor(config, client=mock_client)

        result = conductor.run()

        assert isinstance(result, NewsConductorResult)
        assert len(result.news_events) > 0
        # Should have: host intro + (seg1) + host trans + (seg2) + host trans + (seg3) + host outro
        # = 1 + 1 + 1 + 1 + 1 + 1 + 1 = 7 events
        assert len(result.news_events) == 7

    def test_run_without_host(self):
        """Without a host, should have only segment events."""
        config = make_test_config(num_segments=2, with_host=False, with_director=False)
        mock_client = self._make_mock_client(num_segments=2, with_host=False, with_director=False)
        conductor = NewsConductor(config, client=mock_client)

        result = conductor.run()

        # Just 2 segment events.
        assert len(result.news_events) == 2
        for evt in result.news_events:
            assert evt.speaker_role == "anchor"

    def test_events_have_global_order(self):
        """Events should have sequential global_order."""
        config = make_test_config(num_segments=2, with_host=False, with_director=False)
        mock_client = self._make_mock_client(num_segments=2, with_host=False, with_director=False)
        conductor = NewsConductor(config, client=mock_client)

        result = conductor.run()

        orders = [e.global_order for e in result.news_events]
        assert orders == sorted(orders)
        assert orders == list(range(len(orders)))

    def test_anchor_events_have_segment_id(self):
        """Anchor events should carry segment_id metadata."""
        config = make_test_config(num_segments=2, with_host=False, with_director=False)
        mock_client = self._make_mock_client(num_segments=2, with_host=False, with_director=False)
        conductor = NewsConductor(config, client=mock_client)

        result = conductor.run()

        for evt in result.news_events:
            if evt.speaker_role == "anchor":
                assert evt.segment_id is not None

    def test_host_events_are_narration_kind(self):
        """Host events should have event_kind='narration'."""
        config = make_test_config(num_segments=1, with_director=False)
        mock_client = self._make_mock_client(num_segments=1, with_director=False)
        conductor = NewsConductor(config, client=mock_client)

        result = conductor.run()

        host_events = [e for e in result.news_events if e.speaker_role == "host"]
        assert len(host_events) >= 2  # at least intro + outro
        for evt in host_events:
            assert evt.event_kind == "narration"

    def test_llm_calls_counted(self):
        """total_llm_calls should match the number of mock calls made."""
        config = make_test_config(num_segments=2, with_host=False, with_director=False)
        mock_client = self._make_mock_client(num_segments=2, with_host=False, with_director=False)
        conductor = NewsConductor(config, client=mock_client)

        result = conductor.run()

        assert result.total_llm_calls == 2  # one per segment

    def test_show_id_assigned(self):
        """Result should have a non-empty show_id."""
        config = make_test_config(num_segments=1, with_host=False, with_director=False)
        mock_client = self._make_mock_client(num_segments=1, with_host=False, with_director=False)
        conductor = NewsConductor(config, client=mock_client)

        result = conductor.run()

        assert result.show_id
        assert len(result.show_id) == 8


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: Result Packaging
# ═══════════════════════════════════════════════════════════════════════════

class TestResultPackaging:
    """Tests for to_merged_news_show() conversion."""

    def test_to_merged_news_show_basic(self):
        """MergedNewsShow should have correct metadata."""
        config = make_test_config(num_segments=2, with_host=False, with_director=False)
        mock_client = Mock()
        mock_client.chat.side_effect = [
            make_anchor_json("Segment 1."),
            make_anchor_json("Segment 2."),
        ]
        conductor = NewsConductor(config, client=mock_client)
        result = conductor.run()

        merged = result.to_merged_news_show()

        assert isinstance(merged, MergedNewsShow)
        assert merged.title == "The Evening Brief"
        assert merged.topic == "Today's top stories"
        assert "Jordan Lee" in merged.anchors
        assert merged.segment_count == 2

    def test_merged_events_match_result_events(self):
        """MergedNewsShow events should match the result."""
        config = make_test_config(num_segments=1, with_host=False, with_director=False)
        mock_client = Mock()
        mock_client.chat.return_value = make_anchor_json("The story.")
        conductor = NewsConductor(config, client=mock_client)
        result = conductor.run()

        merged = result.to_merged_news_show()

        assert len(merged.events) == len(result.news_events)

    def test_merged_to_dict_is_serializable(self):
        """to_dict() should produce a JSON-serializable structure."""
        config = make_test_config(num_segments=1, with_host=False, with_director=False)
        mock_client = Mock()
        mock_client.chat.return_value = make_anchor_json("Test.")
        conductor = NewsConductor(config, client=mock_client)
        result = conductor.run()

        merged = result.to_merged_news_show()
        d = merged.to_dict()

        # Should not raise.
        json_str = json.dumps(d)
        assert "Test." in json_str


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 6: Serializer
# ═══════════════════════════════════════════════════════════════════════════

class TestNewsSerializer:
    """Tests for save/load round-trip."""

    def test_save_and_load_round_trip(self):
        """Saving and loading should preserve data."""
        config = make_test_config(num_segments=1, with_host=False, with_director=False)
        mock_client = Mock()
        mock_client.chat.return_value = make_anchor_json("Round trip test.")
        conductor = NewsConductor(config, client=mock_client)
        result = conductor.run()

        merged = result.to_merged_news_show()
        serializer = NewsSerializer()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_news.json"
            serializer.save(merged, path)

            assert path.exists()

            loaded = serializer.load(path)
            assert loaded["title"] == "The Evening Brief"
            assert len(loaded["events"]) == 1
            assert loaded["events"][0]["text"] == "Round trip test."

    def test_to_json_string(self):
        """to_json_string should return valid JSON."""
        merged = MergedNewsShow(
            show_id="test123",
            title="Test Show",
            events=(),
            segments=(),
            anchors=("Test Anchor",),
        )
        serializer = NewsSerializer()
        json_str = serializer.to_json_string(merged)
        data = json.loads(json_str)
        assert data["show_id"] == "test123"


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 7: Bootstrap Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapParsing:
    """Tests for bootstrap architect and director parsing with canned JSON."""

    def test_parse_architect_response(self):
        """Canned architect JSON should produce a valid NewsConfig."""
        bootstrap = NewsBootstrap(client=Mock())

        raw = json.dumps({
            "title": "Tech Weekly",
            "topic": "This week in technology",
            "tone": "conversational",
            "setting": "Modern podcast studio",
            "target_segments": 4,
            "anchors": [
                {"name": "Alex Rivera", "style": "Tech-savvy, fast-paced"}
            ],
            "segments": [
                {
                    "headline": "AI Breakthrough",
                    "angle": "Impact on healthcare",
                    "key_facts": ["GPT-5 released", "FDA approval pending"],
                    "tone": "excited",
                    "duration_hint": "medium",
                    "segment_type": "anchor_read",
                },
                {
                    "headline": "Tech Layoffs",
                    "angle": "Effect on Bay Area",
                    "key_facts": ["10,000 jobs cut"],
                    "tone": "somber",
                    "duration_hint": "medium",
                    "segment_type": "anchor_read",
                },
            ],
            "show_host": {
                "name": "The Narrator",
                "role": "narrator",
                "style": "Documentary tone",
            },
        })

        config = bootstrap._parse_architect_response(
            raw, brief="Test brief", fact_check="off",
        )

        assert config.title == "Tech Weekly"
        assert config.topic == "This week in technology"
        assert len(config.anchors) == 1
        assert config.anchors[0].name == "Alex Rivera"
        assert len(config.segments) == 2
        assert config.segments[0].headline == "AI Breakthrough"
        assert config.show_host is not None
        assert config.show_host.name == "The Narrator"

    def test_parse_director_response(self):
        """Canned director JSON should produce a valid DirectorBrief."""
        bootstrap = NewsBootstrap(client=Mock())

        raw = json.dumps({
            "show_arc": "Lead heavy, middle light, close warm.",
            "mood_progression": ["urgent", "analytical", "warm"],
            "segment_order_notes": "Priority ordering.",
            "closing_strategy": "Recap and sign off.",
        })

        brief = bootstrap._parse_director_response(raw)

        assert brief.show_arc == "Lead heavy, middle light, close warm."
        assert len(brief.mood_progression) == 3
        assert brief.closing_strategy == "Recap and sign off."

    def test_architect_invalid_json_raises(self):
        """Invalid JSON from architect should raise ValueError."""
        bootstrap = NewsBootstrap(client=Mock())
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_architect_response("not json", brief="test")

    def test_director_invalid_json_raises(self):
        """Invalid JSON from director should raise ValueError."""
        bootstrap = NewsBootstrap(client=Mock())
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_director_response("not json either")


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 8: Models
# ═══════════════════════════════════════════════════════════════════════════

class TestNewsModels:
    """Tests for NewsSegment and MergedNewsShow data models."""

    def test_news_event_alias(self):
        """NewsEvent should be AudioEvent."""
        from vox_romana.core.models import AudioEvent
        assert NewsEvent is AudioEvent

    def test_segment_to_dict(self):
        """NewsSegment.to_dict() should work."""
        seg = NewsSegment(
            segment_id="s1",
            segment_order=0,
            headline="Test",
            events=(),
        )
        d = seg.to_dict()
        assert d["segment_id"] == "s1"
        assert d["headline"] == "Test"
        assert d["events"] == []

    def test_merged_show_to_dict(self):
        """MergedNewsShow.to_dict() should produce a complete structure."""
        seg = NewsSegment(
            segment_id="s1",
            segment_order=0,
            headline="Story 1",
            events=(),
        )
        show = MergedNewsShow(
            show_id="abc",
            title="Test Show",
            segments=(seg,),
            events=(),
            anchors=("Anchor 1",),
            segment_count=1,
        )
        d = show.to_dict()
        assert d["show_id"] == "abc"
        assert d["title"] == "Test Show"
        assert len(d["segments"]) == 1
        assert d["anchors"] == ["Anchor 1"]


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 9: Director Check-in
# ═══════════════════════════════════════════════════════════════════════════

class TestDirectorCheckIn:
    """Tests for the Director check-in mechanism."""

    def test_director_check_updates_note(self):
        """Director check-in should update _director_note."""
        config = make_test_config(num_segments=2, with_host=False)
        mock_client = Mock()
        # _director_check calls client.chat() once — return the director response.
        mock_client.chat.return_value = make_director_json(
            "Bring more energy.", nudge="Speed up the next segment.",
        )
        conductor = NewsConductor(config, client=mock_client)
        conductor._build_agents()
        conductor._show_id = "test1234"

        # Manually add something to transcript for context.
        conductor._transcript.append({"speaker": "Jordan Lee", "text": "Test."})

        result = conductor._director_check(2, 3)

        assert "energy" in conductor._director_note.lower() or "speed" in conductor._director_note.lower()

    def test_no_director_returns_empty(self):
        """Without a director brief, check should return empty dict."""
        config = make_test_config(with_director=False)
        conductor = NewsConductor(config, client=Mock())

        result = conductor._director_check(1, 3)
        assert result == {}
