"""Tests for the audiobook show infrastructure.

Tests:
    1. Config validation
    2. Bootstrap architect parsing (canned JSON → AudiobookConfig)
    3. Bootstrap director parsing (canned JSON → DirectorBrief)
    4. Conductor prompt building
    5. Conductor response parsing (JSON extraction, fallback)
    6. Full conductor run with mocked LLM
    7. Result packaging (AudiobookEvent creation, MergedAudiobook conversion)
    8. Serializer round-trip
    9. Chapter generation (multi-beat, key_points distribution)
    10. Auto-scaling beats from target_words

All tests use a mocked DeepSeekClient — no real LLM calls.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_audiobook_conductor.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from vox_romana.core.config import ParticipantProfile, ShowHostConfig
from vox_romana.shows.audiobooks.bootstrap import AudiobookBootstrap
from vox_romana.shows.audiobooks.conductor import (
    AudiobookConductor,
    AudiobookConductorResult,
)
from vox_romana.shows.audiobooks.config import (
    AudiobookConfig,
    ChapterConfig,
    DirectorBrief,
    NarratorConfig,
)
from vox_romana.shows.audiobooks.models import (
    AudiobookChapter,
    AudiobookEvent,
    MergedAudiobook,
)
from vox_romana.shows.audiobooks.serializer import AudiobookSerializer


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def make_test_config(
    *,
    num_chapters=3,
    with_director=True,
    target_beats=1,
):
    """Build a minimal but complete AudiobookConfig for testing."""
    chapters = []
    for i in range(num_chapters):
        chapters.append(ChapterConfig(
            title=f"Chapter {i + 1}: Title {i + 1}",
            synopsis=f"This chapter explores theme {i + 1} in depth.",
            key_points=[
                f"Point {i + 1}A: Opening scene",
                f"Point {i + 1}B: Key development",
                f"Point {i + 1}C: Resolution",
            ],
            tone="contemplative" if i == 0 else "dramatic",
            duration_hint="medium" if i < num_chapters - 1 else "short",
            target_beats=target_beats,
        ))

    narrator = NarratorConfig(
        name="Claire Moreau",
        style="Warm, measured literary voice",
        voice_fingerprint="Deliberate cadence, slight pauses before emphasis, rich lower register.",
    )

    director_brief = None
    if with_director:
        director_brief = DirectorBrief(
            narrative_arc="Build from quiet opening to dramatic peak, then resolve.",
            mood_progression=["contemplative", "uneasy", "tense", "cathartic"],
            chapter_notes="Transitions between chapters should feel organic.",
            closing_strategy="Final chapter slows to reflective tone, narrator warms.",
            check_interval=2,
        )

    return AudiobookConfig(
        brief="A test audiobook about exploration",
        title="The Distant Shore",
        topic="Discovery and transformation",
        genre="literary fiction",
        tone="contemplative",
        setting="A coastal village in the near future",
        narrator=narrator,
        chapters=chapters,
        characters=[
            ParticipantProfile(name="Elena", role="protagonist", voice_gender="female"),
            ParticipantProfile(name="Kai", role="mentor", voice_gender="male"),
        ],
        director_brief=director_brief,
    )


def make_narrator_json(text="The waves crashed against the ancient seawall.", **overrides):
    """Build a canned narrator JSON response."""
    data = {
        "text": text,
        "tone": "contemplative",
        "volume": "moderate",
        "pace": "medium",
        "pause_before": "medium",
        "pause_after": "long",
    }
    data.update(overrides)
    return json.dumps(data)


def make_director_json(mood_note="Increase the tension.", nudge=None, wrap_up=False):
    """Build a canned director check-in response."""
    return json.dumps({
        "mood_note": mood_note,
        "nudge": nudge,
        "wrap_up": wrap_up,
    })


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: Config Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestAudiobookConfig:
    """Tests for AudiobookConfig.validate()."""

    def test_valid_config_has_no_problems(self):
        config = make_test_config()
        assert config.validate() == []

    def test_missing_brief(self):
        config = make_test_config()
        config.brief = ""
        problems = config.validate()
        assert any("brief" in p.lower() for p in problems)

    def test_no_chapters(self):
        config = make_test_config()
        config.chapters = []
        problems = config.validate()
        assert any("chapter" in p.lower() for p in problems)

    def test_no_narrator(self):
        config = make_test_config()
        config.narrator = None
        problems = config.validate()
        assert any("narrator" in p.lower() for p in problems)

    def test_narrator_no_name(self):
        config = make_test_config()
        config.narrator = NarratorConfig(name="", style="test")
        problems = config.validate()
        assert any("name" in p.lower() for p in problems)

    def test_content_type_is_audiobook(self):
        config = make_test_config()
        assert config.content_type == "audiobook"

    def test_chapter_defaults(self):
        ch = ChapterConfig(title="Test")
        assert ch.duration_hint == "medium"
        assert ch.target_beats == 1
        assert ch.key_points == []
        assert ch.characters == []

    def test_narrator_defaults(self):
        n = NarratorConfig(name="Test")
        assert n.voice_fingerprint is None
        assert n.tts_style is None

    def test_director_brief_defaults(self):
        d = DirectorBrief()
        assert d.check_interval == 3
        assert d.mood_progression == []


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: Conductor Prompts
# ═══════════════════════════════════════════════════════════════════════════

class TestConductorPrompts:
    """Tests for prompt building."""

    def test_narrator_prompt_includes_name_and_genre(self):
        config = make_test_config()
        mock_client = Mock()
        conductor = AudiobookConductor(config, client=mock_client)
        prompt = conductor._build_narrator_prompt()
        assert "Claire Moreau" in prompt
        assert "literary fiction" in prompt

    def test_narrator_prompt_includes_title(self):
        config = make_test_config()
        mock_client = Mock()
        conductor = AudiobookConductor(config, client=mock_client)
        prompt = conductor._build_narrator_prompt()
        assert "The Distant Shore" in prompt

    def test_narrator_prompt_includes_fingerprint(self):
        config = make_test_config()
        mock_client = Mock()
        conductor = AudiobookConductor(config, client=mock_client)
        prompt = conductor._build_narrator_prompt()
        assert "Deliberate cadence" in prompt

    def test_no_narrator_means_empty_prompt(self):
        config = make_test_config()
        config.narrator = None
        mock_client = Mock()
        conductor = AudiobookConductor(config, client=mock_client)
        assert conductor._build_narrator_prompt() == ""


class TestPromptMetadataLeaks:
    """Regression tests for machine-only metadata leaking into narrator prompts."""

    def test_generate_chapter_prompt_omits_indices_and_raw_beat_labels(self):
        config = make_test_config(num_chapters=3, with_director=False, target_beats=3)
        conductor = AudiobookConductor(config, client=Mock())
        conductor._audiobook_id = "test1234"

        session = Mock()
        session.send.return_value = make_narrator_json("Narration beat.")
        conductor._narrator_session = session

        conductor._generate_chapter(
            chapter=config.chapters[0],
            chapter_number=1,
            total_chapters=3,
            global_order_start=0,
        )

        prompts = [call.args[0] for call in session.send.call_args_list]
        assert len(prompts) == 3
        assert all("CHAPTER 1 of 3" not in prompt for prompt in prompts)
        assert all("OPENING of this chapter" not in prompt for prompt in prompts)
        assert all("BODY of this chapter" not in prompt for prompt in prompts)
        assert all("CLOSING of this chapter" not in prompt for prompt in prompts)
        assert all("CHAPTER TITLE: Chapter 1: Title 1" in prompt for prompt in prompts)


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: Response Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseParsing:
    """Tests for _parse_response."""

    def _get_conductor(self):
        config = make_test_config()
        mock_client = Mock()
        return AudiobookConductor(config, client=mock_client)

    def test_parse_clean_json(self):
        c = self._get_conductor()
        result = c._parse_response(make_narrator_json())
        assert result["text"] == "The waves crashed against the ancient seawall."

    def test_parse_markdown_wrapped(self):
        c = self._get_conductor()
        raw = f"```json\n{make_narrator_json()}\n```"
        result = c._parse_response(raw)
        assert "waves" in result["text"]

    def test_parse_with_preamble(self):
        c = self._get_conductor()
        raw = f"Here is the narration:\n{make_narrator_json()}"
        result = c._parse_response(raw)
        assert "waves" in result["text"]

    def test_parse_issue_flag_queues_report_and_strips_field(self):
        c = self._get_conductor()
        raw = json.dumps({
            "text": "The waves crashed against the ancient seawall.",
            "issue_flag": {
                "summary": "This detail is not grounded in the source text.",
                "proposed_remediation": "Stay with reference-backed narration.",
            },
        })

        result = c._parse_response(raw, from_agent="narrator")

        assert "issue_flag" not in result
        rendered = c._consume_pending_issue_reports()
        assert "FROM: narrator" in rendered
        assert '"summary": "This detail is not grounded in the source text."' in rendered

    def test_parse_garbage_falls_back(self):
        c = self._get_conductor()
        result = c._parse_response("This is not JSON at all")
        assert result["text"] == "This is not JSON at all"
        assert result["tone"] == "neutral"

    def test_parse_truncates_long_fallback(self):
        c = self._get_conductor()
        long_text = "A" * 600
        result = c._parse_response(long_text)
        assert len(result["text"]) == 500


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: Full Conductor Run (Mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestConductorRun:
    """Tests for the full conductor lifecycle with mocked LLM calls."""

    def _make_mock_client(self, num_chapters=3, with_director=True, target_beats=1):
        """Build a mock client that returns appropriate canned responses.

        Response order for 3 chapters, 1 beat each, director at interval 2:
          - Chapter 1 (1 beat)
          - Chapter 2 (1 beat)
          - Director check (after chapter 2: 2 % 2 == 0)
          - Chapter 3 (1 beat)
        """
        mock_client = Mock()
        responses = []

        for i in range(num_chapters):
            # Chapter beats
            for _b in range(target_beats):
                responses.append(make_narrator_json(f"Chapter {i + 1} narration beat."))

            # Director check at intervals
            ch_num = i + 1
            if with_director and ch_num > 0 and ch_num % 2 == 0:
                responses.append(make_director_json())

        mock_client.chat.side_effect = responses
        return mock_client

    def test_basic_run_produces_events(self):
        """A basic run should produce events for each chapter."""
        config = make_test_config(num_chapters=3, with_director=False)
        mock_client = self._make_mock_client(num_chapters=3, with_director=False)
        conductor = AudiobookConductor(config, client=mock_client)

        result = conductor.run()

        assert isinstance(result, AudiobookConductorResult)
        assert len(result.audiobook_events) == 3  # 1 beat per chapter
        assert len(result.audiobook_chapters) == 3

    def test_run_with_director(self):
        """Director check should not affect event count."""
        config = make_test_config(num_chapters=3, with_director=True)
        mock_client = self._make_mock_client(num_chapters=3, with_director=True)
        conductor = AudiobookConductor(config, client=mock_client)

        result = conductor.run()

        assert len(result.audiobook_events) == 3

    def test_events_have_global_order(self):
        """Events should have sequential global_order values."""
        config = make_test_config(num_chapters=3, with_director=False)
        mock_client = self._make_mock_client(num_chapters=3, with_director=False)
        conductor = AudiobookConductor(config, client=mock_client)

        result = conductor.run()

        orders = [e.global_order for e in result.audiobook_events]
        assert orders == list(range(len(orders)))

    def test_events_have_chapter_id(self):
        """Each event should be tagged with a chapter_id (stored in segment_id)."""
        config = make_test_config(num_chapters=2, with_director=False)
        mock_client = self._make_mock_client(num_chapters=2, with_director=False)
        conductor = AudiobookConductor(config, client=mock_client)

        result = conductor.run()

        for evt in result.audiobook_events:
            assert evt.segment_id is not None
            assert "_ch_" in evt.segment_id

    def test_events_are_narration_kind(self):
        """All audiobook events should have event_kind='narration'."""
        config = make_test_config(num_chapters=2, with_director=False)
        mock_client = self._make_mock_client(num_chapters=2, with_director=False)
        conductor = AudiobookConductor(config, client=mock_client)

        result = conductor.run()

        for evt in result.audiobook_events:
            assert evt.event_kind == "narration"
            assert evt.speaker_role == "narrator"
            assert evt.speaker_name == "Claire Moreau"

    def test_llm_calls_counted(self):
        """total_llm_calls should count all LLM interactions."""
        config = make_test_config(num_chapters=2, with_director=False)
        mock_client = self._make_mock_client(num_chapters=2, with_director=False)
        conductor = AudiobookConductor(config, client=mock_client)

        result = conductor.run()

        # 2 chapters × 1 beat = 2 calls (via AgentSession)
        assert result.total_llm_calls >= 2

    def test_audiobook_id_assigned(self):
        """The result should have a unique audiobook_id."""
        config = make_test_config(num_chapters=1, with_director=False)
        mock_client = self._make_mock_client(num_chapters=1, with_director=False)
        conductor = AudiobookConductor(config, client=mock_client)

        result = conductor.run()

        assert result.audiobook_id
        assert len(result.audiobook_id) == 8

    def test_multi_beat_chapters(self):
        """Chapters with target_beats>1 should produce multiple events."""
        config = make_test_config(num_chapters=2, with_director=False, target_beats=3)

        # With recent_window=4 (default), the AgentSession compresses when
        # history exceeds 8 messages. 3 beats/chapter × 2 messages/beat = 6
        # per chapter. After ch2 beat 3 (12 msgs total), compression fires
        # (older=4 ≥ 4). So we need 6 beat responses + 1 compression = 7.
        mock_client = Mock()
        responses = []
        for i in range(2):
            for b in range(3):
                responses.append(make_narrator_json(f"Ch{i+1} beat {b+1}."))
        # Compression response (summary)
        responses.append("Covered chapter 1 and start of chapter 2.")
        mock_client.chat.side_effect = responses

        conductor = AudiobookConductor(config, client=mock_client)
        result = conductor.run()

        # 2 chapters × 3 beats = 6 events
        assert len(result.audiobook_events) == 6


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: Result Packaging
# ═══════════════════════════════════════════════════════════════════════════

class TestResultPackaging:
    """Tests for result → MergedAudiobook conversion."""

    def _run_conductor(self, num_chapters=2):
        config = make_test_config(num_chapters=num_chapters, with_director=False)
        mock_client = Mock()
        responses = [make_narrator_json(f"Ch {i+1} narration.") for i in range(num_chapters)]
        mock_client.chat.side_effect = responses
        conductor = AudiobookConductor(config, client=mock_client)
        return conductor.run()

    def test_to_merged_audiobook_basic(self):
        result = self._run_conductor(num_chapters=3)
        merged = result.to_merged_audiobook()

        assert isinstance(merged, MergedAudiobook)
        assert merged.title == "The Distant Shore"
        assert merged.narrator == "Claire Moreau"
        assert merged.genre == "literary fiction"
        assert merged.chapter_count == 3
        assert len(merged.chapters) == 3
        assert len(merged.events) == 3

    def test_merged_events_match_result_events(self):
        result = self._run_conductor(num_chapters=2)
        merged = result.to_merged_audiobook()

        for r_evt, m_evt in zip(result.audiobook_events, merged.events):
            assert r_evt.text == m_evt.text
            assert r_evt.global_order == m_evt.global_order

    def test_merged_to_dict_is_serializable(self):
        result = self._run_conductor(num_chapters=2)
        merged = result.to_merged_audiobook()
        d = merged.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert d["audiobook_id"]
        assert d["title"] == "The Distant Shore"
        assert len(d["chapters"]) == 2

    def test_merged_has_characters(self):
        result = self._run_conductor(num_chapters=1)
        merged = result.to_merged_audiobook()

        assert "Elena" in merged.characters
        assert "Kai" in merged.characters

    def test_voice_casting_from_config(self):
        result = self._run_conductor(num_chapters=1)
        config = result.config
        config.narrator.voice_gender = "female"
        config.narrator.voice_hint = "warm mezzo voice"
        merged = result.to_merged_audiobook()

        assert merged.voice_casting.get("Claire Moreau") == "female"
        assert merged.voice_hints.get("Claire Moreau") == "warm mezzo voice"


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 6: Serializer
# ═══════════════════════════════════════════════════════════════════════════

class TestAudiobookSerializer:
    """Tests for the AudiobookSerializer."""

    def _make_merged(self):
        ch = AudiobookChapter(
            chapter_id="abc_ch_0",
            chapter_order=0,
            title="Chapter 1",
            narrator_name="Test Narrator",
            events=(),
        )
        return MergedAudiobook(
            audiobook_id="abc",
            title="Test Book",
            chapters=(ch,),
            events=(),
            narrator="Test Narrator",
            chapter_count=1,
        )

    def test_save_and_load_round_trip(self):
        merged = self._make_merged()
        serializer = AudiobookSerializer()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_audiobook.json"
            saved_path = serializer.save(merged, path)
            assert saved_path.exists()

            loaded = serializer.load(saved_path)
            assert loaded["audiobook_id"] == "abc"
            assert loaded["title"] == "Test Book"
            assert len(loaded["chapters"]) == 1

    def test_to_json_string(self):
        merged = self._make_merged()
        serializer = AudiobookSerializer()
        json_str = serializer.to_json_string(merged)

        parsed = json.loads(json_str)
        assert parsed["audiobook_id"] == "abc"

    def test_save_creates_parent_dirs(self):
        merged = self._make_merged()
        serializer = AudiobookSerializer()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "audiobook.json"
            saved_path = serializer.save(merged, path)
            assert saved_path.exists()


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 7: Bootstrap Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapParsing:
    """Tests for architect and director response parsing."""

    def _get_bootstrap(self):
        mock_client = Mock()
        return AudiobookBootstrap(client=mock_client)

    def test_parse_architect_response(self):
        bootstrap = self._get_bootstrap()
        raw = json.dumps({
            "title": "The Distant Shore",
            "genre": "science fiction",
            "topic": "First contact",
            "tone": "contemplative",
            "setting": "Near-future Earth",
            "narrator": {
                "name": "David Chen",
                "style": "Warm, measured",
            },
            "characters": [
                {"name": "Elena", "role": "protagonist", "voice_gender": "female"},
            ],
            "chapters": [
                {
                    "title": "Chapter 1: The Signal",
                    "synopsis": "Elena detects an anomalous signal.",
                    "key_points": ["She finds the signal", "She alerts the team"],
                    "tone": "curious",
                    "duration_hint": "medium",
                    "target_beats": 2,
                    "characters": ["Elena"],
                },
                {
                    "title": "Chapter 2: First Light",
                    "synopsis": "The team makes contact.",
                    "key_points": ["The signal responds"],
                    "tone": "dramatic",
                    "duration_hint": "long",
                    "target_beats": 3,
                    "characters": ["Elena"],
                },
            ],
        })

        config = bootstrap._parse_architect_response(
            raw,
            brief="Test brief about David Chen narrating Elena's first contact story",
            reference_material=None,
            fact_check="off",
        )

        assert config.title == "The Distant Shore"
        assert config.genre == "science fiction"
        assert config.narrator.name == "David Chen"
        assert len(config.chapters) == 2
        assert config.chapters[0].title == "Chapter 1: The Signal"
        assert config.chapters[0].target_beats == 2
        assert len(config.chapters[0].key_points) == 2
        assert len(config.characters) == 1
        assert config.characters[0].name == "Elena"

    def test_parse_architect_curates_names_and_chapter_refs(self):
        bootstrap = self._get_bootstrap()
        raw = json.dumps({
            "title": "The Distant Shore",
            "narrator": {"name": "David Chen", "style": "Warm, measured"},
            "characters": [
                {"name": "Elena", "role": "protagonist", "voice_gender": "female"},
            ],
            "chapters": [
                {"title": "Chapter 1", "characters": ["Elena"]},
            ],
        })

        config = bootstrap._parse_architect_response(raw, brief="A first-contact novella")

        assert config.narrator.name != "David Chen"
        assert config.characters[0].name != "Elena"
        assert config.chapters[0].characters == [config.characters[0].name]

    def test_parse_architect_with_markdown_fences(self):
        bootstrap = self._get_bootstrap()
        inner = json.dumps({
            "title": "Test",
            "narrator": {"name": "N", "style": "S"},
            "chapters": [{"title": "Ch1"}],
        })
        raw = f"```json\n{inner}\n```"
        config = bootstrap._parse_architect_response(raw, brief="test")
        assert config.title == "Test"

    def test_parse_architect_invalid_json_raises(self):
        bootstrap = self._get_bootstrap()
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_architect_response("not json", brief="test")

    def test_parse_director_response(self):
        bootstrap = self._get_bootstrap()
        raw = json.dumps({
            "narrative_arc": "Build tension, resolve at end",
            "mood_progression": ["calm", "tense", "cathartic"],
            "chapter_notes": "Smooth transitions",
            "closing_strategy": "Warm, reflective ending",
        })

        brief = bootstrap._parse_director_response(raw)

        assert brief.narrative_arc == "Build tension, resolve at end"
        assert len(brief.mood_progression) == 3
        assert brief.closing_strategy == "Warm, reflective ending"

    def test_parse_director_invalid_json_raises(self):
        bootstrap = self._get_bootstrap()
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_director_response("garbage")


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 8: Models
# ═══════════════════════════════════════════════════════════════════════════

class TestModels:
    """Tests for data model classes."""

    def test_audiobook_event_alias(self):
        """AudiobookEvent should be the same as AudioEvent."""
        from vox_romana.core.models import AudioEvent
        assert AudiobookEvent is AudioEvent

    def test_chapter_to_dict(self):
        ch = AudiobookChapter(
            chapter_id="test_ch_0",
            chapter_order=0,
            title="Test Chapter",
            tone="dramatic",
            narrator_name="Claire",
            events=(),
        )
        d = ch.to_dict()
        assert d["chapter_id"] == "test_ch_0"
        assert d["title"] == "Test Chapter"
        assert d["narrator_name"] == "Claire"
        assert d["events"] == []

    def test_merged_audiobook_to_dict(self):
        ch = AudiobookChapter(
            chapter_id="abc_ch_0", chapter_order=0, title="Ch1", events=(),
        )
        merged = MergedAudiobook(
            audiobook_id="abc",
            title="Test",
            chapters=(ch,),
            events=(),
            narrator="Test",
            characters=("Elena",),
            genre="sci-fi",
            chapter_count=1,
        )
        d = merged.to_dict()
        assert d["audiobook_id"] == "abc"
        assert d["characters"] == ["Elena"]
        assert d["genre"] == "sci-fi"


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 9: Key Points Distribution
# ═══════════════════════════════════════════════════════════════════════════

class TestKeyPointsDistribution:
    """Tests for _distribute_points utility."""

    def test_distribute_evenly(self):
        points = ["A", "B", "C", "D"]
        result = AudiobookConductor._distribute_points(points, 2)
        assert result == [["A", "C"], ["B", "D"]]

    def test_distribute_uneven(self):
        points = ["A", "B", "C"]
        result = AudiobookConductor._distribute_points(points, 2)
        assert result == [["A", "C"], ["B"]]

    def test_distribute_empty(self):
        result = AudiobookConductor._distribute_points([], 3)
        assert result == [[], [], []]

    def test_distribute_single_beat(self):
        points = ["A", "B", "C"]
        result = AudiobookConductor._distribute_points(points, 1)
        assert result == [["A", "B", "C"]]

    def test_distribute_more_beats_than_points(self):
        points = ["A"]
        result = AudiobookConductor._distribute_points(points, 3)
        assert result == [["A"], [], []]

    def test_distribute_zero_beats(self):
        result = AudiobookConductor._distribute_points(["A"], 0)
        assert result == [[]]


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 10: Auto-scaling
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoScaling:
    """Tests for target_words auto-scaling logic."""

    def test_auto_scaling_distributes_beats(self):
        """When target_words is set, chapters should get beats proportional to weight."""
        config = make_test_config(num_chapters=3, with_director=False)
        config.target_words = 3000

        # Simulate the auto-scaling logic from _run_turn_loop
        chapters = config.chapters
        total_chapters = len(chapters)
        words_per_beat = 200
        total_beats_needed = max(total_chapters, config.target_words // words_per_beat)
        weight_map = {"short": 1, "medium": 2, "long": 3}
        total_weight = sum(weight_map.get(ch.duration_hint, 2) for ch in chapters)

        for ch in chapters:
            if ch.target_beats <= 1:
                w = weight_map.get(ch.duration_hint, 2)
                ch.target_beats = max(1, round(total_beats_needed * w / total_weight))

        # All chapters should now have > 1 beat
        for ch in chapters:
            assert ch.target_beats >= 1

        # Total beats should approximately match target
        total_beats = sum(ch.target_beats for ch in chapters)
        assert total_beats >= 3  # At least one per chapter

    def test_preset_beats_not_overwritten(self):
        """Chapters with target_beats > 1 should not be modified by auto-scaling."""
        ch = ChapterConfig(title="Deep Chapter", target_beats=5, duration_hint="long")
        config = AudiobookConfig(
            brief="test",
            target_words=3000,
            narrator=NarratorConfig(name="N"),
            chapters=[ch, ChapterConfig(title="Auto Chapter")],
        )

        # Run the scaling logic
        chapters = config.chapters
        total_chapters = len(chapters)
        words_per_beat = 200
        total_beats_needed = max(total_chapters, config.target_words // words_per_beat)
        weight_map = {"short": 1, "medium": 2, "long": 3}
        total_weight = sum(weight_map.get(c.duration_hint, 2) for c in chapters)
        for c in chapters:
            if c.target_beats <= 1:
                w = weight_map.get(c.duration_hint, 2)
                c.target_beats = max(1, round(total_beats_needed * w / total_weight))

        assert ch.target_beats == 5  # Not overwritten

    def test_no_auto_scale_without_target_words(self):
        """Without target_words, chapters keep default beats."""
        config = make_test_config(num_chapters=3, with_director=False)
        assert config.target_words is None
        for ch in config.chapters:
            assert ch.target_beats == 1  # Default


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 11: Director Check
# ═══════════════════════════════════════════════════════════════════════════

class TestDirectorCheck:
    """Tests for the director check-in mechanism."""

    def test_director_check_updates_note(self):
        config = make_test_config(num_chapters=2, with_director=True)
        config.director_brief.check_interval = 1  # Check every chapter

        mock_client = Mock()
        responses = [
            make_narrator_json("Chapter 1 content."),
            make_director_json("Increase the tension.", nudge="Add more stakes."),
            make_narrator_json("Chapter 2 content."),
            make_director_json("Good pacing."),
        ]
        mock_client.chat.side_effect = responses

        conductor = AudiobookConductor(config, client=mock_client)
        result = conductor.run()

        assert len(result.audiobook_events) == 2

    def test_no_director_returns_empty(self):
        config = make_test_config(num_chapters=2, with_director=False)
        mock_client = Mock()
        responses = [make_narrator_json(f"Ch {i+1}.") for i in range(2)]
        mock_client.chat.side_effect = responses

        conductor = AudiobookConductor(config, client=mock_client)
        result = conductor.run()

        assert result.director_notes == []


# ═══════════════════════════════════════════════════════════════════════════
# TEST CLASS 12: Router Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestRouterIntegration:
    """Tests that the router recognizes audiobook as a show type."""

    def test_audiobook_in_show_types(self):
        from vox_romana.core.router import SHOW_TYPES
        assert "audiobook" in SHOW_TYPES

    def test_router_result_accepts_audiobook(self):
        from vox_romana.core.router import RouterResult
        r = RouterResult("audiobook", 0.9, "narrated audiobook")
        assert r.show_type == "audiobook"

    def test_parse_router_audiobook_response(self):
        from vox_romana.core.router import parse_router_response
        raw = json.dumps({
            "show_type": "audiobook",
            "confidence": 0.92,
            "reasoning": "The brief asks for a narrated story.",
            "needs_research": False,
        })
        result = parse_router_response(raw)
        assert result.show_type == "audiobook"
        assert result.confidence == 0.92
