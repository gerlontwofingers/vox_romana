"""Tests for the lecture show infrastructure.

Covers:
    1. Config validation (sections + professor required, valid section_type)
    2. Bootstrap architect parsing (canned JSON \u2192 LectureConfig)
    3. Bootstrap director parsing (canned JSON \u2192 LectureDirectorBrief)
    4. Conductor prompt building (professor system prompt format)
    5. Conductor response parsing (JSON extraction + fallback)
    6. Full conductor run with mocked LLM
    7. Result packaging (LectureEvent + MergedLecture)
    8. Type overlay selection by section_type
    9. Serializer round-trip

All tests use a mocked DeepSeekClient \u2014 no real LLM calls.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from vox_romana.shows.lecture.bootstrap import LectureBootstrap
from vox_romana.shows.lecture.conductor import (
    LectureConductor,
    LectureConductorResult,
)
from vox_romana.shows.lecture.config import (
    LectureConfig,
    LectureDirectorBrief,
    LectureSection,
    ProfessorConfig,
)
from vox_romana.shows.lecture.models import (
    LectureSection as LectureSectionModel,
    MergedLecture,
)
from vox_romana.shows.lecture.prompts import OVERLAY_BY_TYPE
from vox_romana.shows.lecture.serializer import LectureSerializer


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


def make_test_config(*, num_sections=3, with_director=True, target_beats=1):
    """Build a minimal but complete LectureConfig for testing."""
    section_types = ["intro", "concept", "case_study", "recap", "outro"]
    sections = []
    for i in range(num_sections):
        sections.append(LectureSection(
            title=f"Section {i + 1}",
            section_type=section_types[i % len(section_types)],
            learning_objective=f"Listener should grasp idea {i + 1}",
            key_points=[f"Point {i+1}A", f"Point {i+1}B"],
            examples=[f"Example {i+1}"],
            duration_hint="medium",
            target_beats=target_beats,
        ))

    professor = ProfessorConfig(
        name="Prof. Aria Stone",
        title="Professor of Cognitive Science",
        style="warm, methodical, fond of analogies",
        expertise=["cognitive science", "philosophy of mind"],
        voice_fingerprint="Measured pace, slight pauses before key terms.",
    )

    director_brief = None
    if with_director:
        director_brief = LectureDirectorBrief(
            thesis="Understanding emerges from layered abstraction.",
            progression=["foundations", "complications", "synthesis"],
            pedagogical_notes="Use analogies generously; recap every other section.",
            closing_strategy="Open question to take away.",
            check_interval=2,
        )

    return LectureConfig(
        brief="A lecture on cognitive abstraction",
        title="Layered Minds",
        topic="Cognitive abstraction and understanding",
        subject_area="cognitive science",
        tone="warm and curious",
        setting="intimate seminar",
        professor=professor,
        sections=sections,
        director_brief=director_brief,
    )


def make_professor_json(text="Today we explore cognitive abstraction.", **overrides):
    data = {
        "text": text,
        "tone": "warm",
        "volume": "moderate",
        "pace": "medium",
        "pause_before": "medium",
        "pause_after": "long",
        "nonverbal": [],
    }
    data.update(overrides)
    return json.dumps(data)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIG VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestLectureConfigValidation:
    def test_valid_config_has_no_problems(self):
        config = make_test_config()
        assert config.validate() == []

    def test_missing_sections_is_invalid(self):
        config = LectureConfig(
            brief="x", professor=ProfessorConfig(name="P"), sections=[],
        )
        problems = config.validate()
        assert any("section" in p.lower() for p in problems)

    def test_missing_professor_is_invalid(self):
        config = LectureConfig(
            brief="x", professor=None,
            sections=[LectureSection(title="S")],
        )
        problems = config.validate()
        assert any("professor" in p.lower() for p in problems)

    def test_professor_without_name_is_invalid(self):
        config = LectureConfig(
            brief="x", professor=ProfessorConfig(name=""),
            sections=[LectureSection(title="S")],
        )
        problems = config.validate()
        assert any("name" in p.lower() for p in problems)

    def test_invalid_section_type_is_caught(self):
        config = LectureConfig(
            brief="x", professor=ProfessorConfig(name="P"),
            sections=[LectureSection(title="S", section_type="rant")],
        )
        problems = config.validate()
        assert any("section_type" in p for p in problems)

    def test_content_type_is_lecture(self):
        config = make_test_config()
        assert config.content_type == "lecture"


# ═══════════════════════════════════════════════════════════════════════════
# 2. BOOTSTRAP ARCHITECT PARSING
# ═══════════════════════════════════════════════════════════════════════════


class TestLectureArchitectParsing:
    def test_parses_full_architect_response(self):
        client = Mock()
        bootstrap = LectureBootstrap(client=client)

        canned = json.dumps({
            "title": "Why Stoicism Endures",
            "topic": "Stoic ethics",
            "subject_area": "philosophy",
            "tone": "calm and grounded",
            "setting": "small seminar",
            "professor": {
                "name": "Prof. Marcus Aurelio",
                "title": "Professor of Philosophy",
                "style": "patient and analogical",
                "expertise": ["ancient ethics", "Roman thought"],
                "voice_gender": "male",
            },
            "sections": [
                {"title": "Intro", "section_type": "intro",
                 "learning_objective": "frame the lecture",
                 "key_points": ["hook"], "examples": [], "target_beats": 1},
                {"title": "Core idea", "section_type": "concept",
                 "learning_objective": "define apatheia",
                 "key_points": ["definition", "example"],
                 "examples": ["Marcus on grief"], "target_beats": 2},
                {"title": "Wrap", "section_type": "outro",
                 "learning_objective": "tie it together",
                 "key_points": ["recap"], "examples": [], "target_beats": 1},
            ],
        })

        config = bootstrap._parse_architect_response(
            canned,
            brief="A lecture by Prof. Marcus Aurelio on Stoic ethics",
            reference_material=None,
            fact_check="off",
        )

        assert config.title == "Why Stoicism Endures"
        assert config.subject_area == "philosophy"
        assert config.professor.name == "Prof. Marcus Aurelio"
        assert config.professor.voice_gender == "male"
        assert len(config.sections) == 3
        assert config.sections[0].section_type == "intro"
        assert config.sections[1].target_beats == 2
        assert "Marcus on grief" in config.sections[1].examples

    def test_handles_markdown_fences(self):
        client = Mock()
        bootstrap = LectureBootstrap(client=client)
        wrapped = "```json\n" + json.dumps({
            "title": "X", "topic": "Y", "subject_area": "Z",
            "professor": {"name": "P"},
            "sections": [{"title": "S", "section_type": "concept"}],
        }) + "\n```"
        config = bootstrap._parse_architect_response(wrapped, brief="Lecture by P")
        assert config.title == "X"
        assert config.professor.name == "P"

    def test_curates_professor_name_when_brief_is_generic(self):
        client = Mock()
        bootstrap = LectureBootstrap(client=client)
        canned = json.dumps({
            "title": "Why Stoicism Endures",
            "professor": {"name": "Prof. Marcus Aurelio", "voice_gender": "male"},
            "sections": [{"title": "Intro", "section_type": "intro"}],
        })

        config = bootstrap._parse_architect_response(canned, brief="A lecture on Stoicism")

        assert config.professor.name != "Prof. Marcus Aurelio"
        assert config.professor.name

    def test_invalid_json_raises(self):
        client = Mock()
        bootstrap = LectureBootstrap(client=client)
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_architect_response("not json at all")


# ═══════════════════════════════════════════════════════════════════════════
# 3. BOOTSTRAP DIRECTOR PARSING
# ═══════════════════════════════════════════════════════════════════════════


class TestLectureDirectorParsing:
    def test_parses_director_response(self):
        client = Mock()
        bootstrap = LectureBootstrap(client=client)
        canned = json.dumps({
            "thesis": "Stoicism is about agency, not suppression.",
            "progression": ["frame", "deepen", "synthesize"],
            "pedagogical_notes": "Use Roman examples.",
            "closing_strategy": "End with an open question.",
        })

        brief = bootstrap._parse_director_response(canned)

        assert brief.thesis.startswith("Stoicism")
        assert brief.progression == ["frame", "deepen", "synthesize"]
        assert brief.closing_strategy.endswith("question.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONDUCTOR PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════


class TestProfessorPromptBuilding:
    def test_includes_name_topic_voice_fingerprint(self):
        config = make_test_config()
        client = Mock()
        conductor = LectureConductor(config, client=client)
        prompt = conductor._build_professor_prompt()

        assert "Prof. Aria Stone" in prompt
        assert "Cognitive Science" in prompt
        assert "Cognitive abstraction" in prompt
        assert "Measured pace" in prompt

    def test_handles_missing_title_gracefully(self):
        config = make_test_config()
        config.professor.title = None
        conductor = LectureConductor(config, client=Mock())
        prompt = conductor._build_professor_prompt()
        # Title clause becomes empty; sentence still well-formed.
        assert "Prof. Aria Stone" in prompt
        assert "delivering a lecture" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 5. RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════════════════


class TestResponseParsing:
    def test_parses_valid_json(self):
        config = make_test_config()
        conductor = LectureConductor(config, client=Mock())
        result = conductor._parse_response(make_professor_json(text="Hello."))
        assert result["text"] == "Hello."
        assert result["tone"] == "warm"

    def test_strips_markdown_fences(self):
        config = make_test_config()
        conductor = LectureConductor(config, client=Mock())
        wrapped = "```json\n" + make_professor_json(text="Wrapped.") + "\n```"
        result = conductor._parse_response(wrapped)
        assert result["text"] == "Wrapped."

    def test_issue_flag_queues_report_and_strips_field(self):
        config = make_test_config()
        conductor = LectureConductor(config, client=Mock())
        raw = json.dumps({
            "text": "Hello.",
            "issue_flag": {
                "summary": "This explanation needs a sourced example.",
                "proposed_remediation": "Use a simpler verified example.",
            },
        })

        result = conductor._parse_response(raw, from_agent="professor")

        assert "issue_flag" not in result
        rendered = conductor._consume_pending_issue_reports()
        assert "FROM: professor" in rendered
        assert '"summary": "This explanation needs a sourced example."' in rendered

    def test_fallback_on_invalid_json(self):
        config = make_test_config()
        conductor = LectureConductor(config, client=Mock())
        result = conductor._parse_response("just plain text response")
        assert "text" in result
        assert "tone" in result


# ═══════════════════════════════════════════════════════════════════════════
# 6. FULL CONDUCTOR RUN (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════


class TestConductorRun:
    def test_produces_one_event_per_section_when_target_beats_is_one(self):
        config = make_test_config(num_sections=3, target_beats=1, with_director=False)
        client = Mock()
        client.chat = Mock(side_effect=[
            make_professor_json(text=f"Section {i+1} content.") for i in range(3)
        ])

        conductor = LectureConductor(config, client=client)
        result = conductor.run()

        assert isinstance(result, LectureConductorResult)
        assert len(result.lecture_events) == 3
        assert all(e.speaker_role == "professor" for e in result.lecture_events)
        assert all(e.event_kind == "narration" for e in result.lecture_events)
        # Each event tagged with its section_id
        assert {e.segment_id for e in result.lecture_events} == {
            f"{result.lecture_id}_sec_{i}" for i in range(3)
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7. RESULT PACKAGING / MERGED LECTURE
# ═══════════════════════════════════════════════════════════════════════════


class TestMergedLecture:
    def test_to_merged_lecture_collects_voice_metadata(self):
        config = make_test_config(num_sections=2, target_beats=1, with_director=False)
        config.professor.voice_gender = "female"
        config.professor.voice_hint = "warm mezzo, 50s"
        config.professor.tts_style = "speak with patience"

        client = Mock()
        client.chat = Mock(side_effect=[
            make_professor_json(text=f"Section {i+1} content.") for i in range(2)
        ])
        conductor = LectureConductor(config, client=client)
        result = conductor.run()

        merged = result.to_merged_lecture()

        assert isinstance(merged, MergedLecture)
        assert merged.title == "Layered Minds"
        assert merged.professor == "Prof. Aria Stone"
        assert merged.subject_area == "cognitive science"
        assert merged.section_count == 2
        assert len(merged.events) == 2
        assert merged.voice_casting["Prof. Aria Stone"] == "female"
        assert merged.voice_hints["Prof. Aria Stone"] == "warm mezzo, 50s"
        assert merged.tts_styles["Prof. Aria Stone"] == "speak with patience"


# ═══════════════════════════════════════════════════════════════════════════
# 8. TYPE OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════


class TestSectionTypeOverlays:
    @pytest.mark.parametrize("section_type", [
        "intro", "concept", "case_study", "exercise", "recap", "outro",
    ])
    def test_each_section_type_has_an_overlay(self, section_type):
        assert section_type in OVERLAY_BY_TYPE
        assert "SECTION MODE" in OVERLAY_BY_TYPE[section_type]


# ═══════════════════════════════════════════════════════════════════════════
# 9. SERIALIZER ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════


class TestSerializerRoundTrip:
    def test_serializer_save_writes_json(self, tmp_path):
        config = make_test_config(num_sections=2, target_beats=1, with_director=False)
        client = Mock()
        client.chat = Mock(side_effect=[
            make_professor_json(text=f"Section {i+1}.") for i in range(2)
        ])
        result = LectureConductor(config, client=client).run()
        merged = result.to_merged_lecture()

        out = tmp_path / "lecture.json"
        LectureSerializer().save(merged, out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["title"] == "Layered Minds"
        assert data["section_count"] == 2
        assert len(data["events"]) == 2
