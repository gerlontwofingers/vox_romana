"""Tests for the radio drama show infrastructure (shot 5.1-a).

Covers the dataclasses, models, and serializer that exist after shot 5.1-a:
    1. Config validation (cast/scenes/era/series-mode checks)
    2. CharacterConfig / SceneConfig / RadioDramaDirectorBrief defaults
    3. Model to_dict + JSON round-trip (RadioDramaScene, MergedRadioDrama)
    4. Serializer save/load round-trip
    5. Package imports

Bootstrap/conductor/prompts land in later shots (5.1-b/5.1-c/5.1-d); those
tests arrive with them.

Run with:
    python -m pytest tests/test_radio_drama.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vox_romana.core.models import AudioEvent
from vox_romana.shows.radio_dramas import (
    ERAS,
    SERIES_MODES,
    CharacterConfig,
    MergedRadioDrama,
    RadioDramaConfig,
    RadioDramaDirectorBrief,
    RadioDramaEvent,
    RadioDramaScene,
    RadioDramaSerializer,
    SceneConfig,
)
from vox_romana.shows.radio_dramas import prompts as rd_prompts
from vox_romana.core.messages import ISSUE_FLAG_PROMPT_NOTE


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def make_character(name="Sam Vale", gender="male", role="detective"):
    return CharacterConfig(
        name=name,
        role=role,
        archetype="hardboiled detective",
        description="A weary private eye with a soft spot for lost causes.",
        speech_style="terse, world-weary",
        voice_gender=gender,
        voice_hint="gravelly baritone",
        tts_style="Speak low and measured.",
    )


def make_config(*, series_mode="pilot", series_id=None, num_scenes=3):
    characters = [
        make_character("Sam Vale", "male", "detective"),
        make_character("Dolores Kane", "female", "femme fatale"),
    ]
    announcer = CharacterConfig(
        name="The Announcer",
        role="announcer",
        voice_gender="male",
        voice_hint="boom-voiced 1940s radio announcer",
    )
    scenes = [
        SceneConfig(
            title=f"Scene {i + 1}",
            synopsis=f"What happens in scene {i + 1}.",
            location="A rain-slicked dock",
            characters=["Sam Vale"],
            duration_hint="medium",
        )
        for i in range(num_scenes)
    ]
    return RadioDramaConfig(
        brief="A noir mystery about a missing heiress.",
        title="The Dock Street Case",
        genre="noir",
        era="1940s",
        setting="Philadelphia, 1946",
        announcer=announcer,
        characters=characters,
        scenes=scenes,
        series_mode=series_mode,
        series_id=series_id,
    )


def make_event(order=0, speaker="Sam Vale", text="The rain kept falling."):
    return AudioEvent(
        global_order=order,
        local_order=order,
        speaker_role="detective",
        speaker_name=speaker,
        target=None,
        text=text,
        tone=("tense",),
        volume="low",
        pace="measured",
        pause_before="0.3",
        pause_after="0.2",
        event_kind="utterance",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Config validation
# ═══════════════════════════════════════════════════════════════════════════

class TestRadioDramaConfig:
    def test_valid_config_no_problems(self):
        config = make_config()
        # base validate() flags fact_check + reference_material mismatch only;
        # our brief is set and fact_check is off, so it should be clean.
        assert config.validate() == []

    def test_empty_brief_flagged(self):
        config = make_config()
        config.brief = ""
        problems = config.validate()
        assert any("brief" in p for p in problems)

    def test_no_cast_flagged(self):
        config = make_config()
        config.characters = []
        config.announcer = None
        problems = config.validate()
        assert any("character" in p.lower() for p in problems)

    def test_no_scenes_flagged(self):
        config = make_config()
        config.scenes = []
        problems = config.validate()
        assert any("scene" in p for p in problems)

    def test_invalid_era_flagged(self):
        config = make_config()
        config.era = "medieval"
        problems = config.validate()
        assert any("era" in p for p in problems)

    def test_invalid_series_mode_flagged(self):
        config = make_config()
        config.series_mode = "anthology"
        problems = config.validate()
        assert any("series_mode" in p for p in problems)

    def test_serial_requires_series_id(self):
        config = make_config(series_mode="serial", series_id=None)
        problems = config.validate()
        assert any("series_id" in p for p in problems)

    def test_serial_with_series_id_ok(self):
        config = make_config(series_mode="serial", series_id="dock_street")
        problems = config.validate()
        assert not any("series_id" in p for p in problems)


# ═══════════════════════════════════════════════════════════════════════════
# Character / Scene / DirectorBrief defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestCharacterConfig:
    def test_defaults(self):
        c = CharacterConfig(name="Eve")
        assert c.role == "character"
        assert c.archetype is None
        # VoiceProfile inheritance:
        assert c.voice_gender is None
        assert c.voice_hint is None
        assert c.tts_style is None
        assert c.voice_fingerprint is None

    def test_voice_fields_set(self):
        c = make_character()
        assert c.voice_gender == "male"
        assert c.voice_hint == "gravelly baritone"


class TestSceneConfig:
    def test_defaults(self):
        s = SceneConfig()
        assert s.duration_hint == "medium"
        assert s.characters == []
        assert s.sfx_hints == []
        assert s.cliffhanger is False

    def test_sfx_hints_present(self):
        s = SceneConfig(sfx_hints=["rain", "footsteps"], cliffhanger=True)
        assert s.sfx_hints == ["rain", "footsteps"]
        assert s.cliffhanger is True


class TestDirectorBrief:
    def test_defaults(self):
        b = RadioDramaDirectorBrief()
        assert b.narrative_arc == ""
        assert b.mood_progression == []
        assert b.scene_notes == {}
        assert b.check_interval == 3

    def test_scene_notes_int_keys(self):
        b = RadioDramaDirectorBrief(scene_notes={0: "hook", 2: "cliffhanger"})
        assert b.scene_notes[0] == "hook"
        assert b.scene_notes[2] == "cliffhanger"


# ═══════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════

class TestRadioDramaModels:
    def test_event_alias(self):
        assert RadioDramaEvent is AudioEvent

    def test_scene_to_dict(self):
        scene = RadioDramaScene(
            scene_id="scene_000",
            scene_order=1,
            title="The Dock",
            location="A rain-slicked dock",
            characters=("Sam Vale",),
            events=(make_event(0), make_event(1, "Dolores Kane", "Don't.")),
        )
        d = scene.to_dict()
        assert d["scene_id"] == "scene_000"
        assert d["title"] == "The Dock"
        assert d["characters"] == ["Sam Vale"]
        assert len(d["events"]) == 2
        assert d["events"][0]["speaker_name"] == "Sam Vale"
        # JSON-serializable
        json.dumps(d)

    def test_merged_to_dict(self):
        scene = RadioDramaScene(
            scene_id="scene_000",
            scene_order=1,
            title="The Dock",
            events=(make_event(0),),
        )
        merged = MergedRadioDrama(
            drama_id="dock_street_001",
            title="The Dock Street Case",
            series_mode="serial",
            series_id="dock_street",
            episode_number=1,
            genre="noir",
            era="1940s",
            scenes=(scene,),
            events=scene.events,
            announcer="The Announcer",
            characters=("Sam Vale", "Dolores Kane"),
            voice_casting={"Sam Vale": "male", "Dolores Kane": "female"},
            language="en",
            music_style="1940s big band",
        )
        d = merged.to_dict()
        assert d["drama_id"] == "dock_street_001"
        assert d["series_mode"] == "serial"
        assert d["series_id"] == "dock_street"
        assert d["episode_number"] == 1
        assert d["era"] == "1940s"
        assert d["characters"] == ["Sam Vale", "Dolores Kane"]
        json.dumps(d)  # must be JSON-serializable


# ═══════════════════════════════════════════════════════════════════════════
# Serializer
# ═══════════════════════════════════════════════════════════════════════════

class TestRadioDramaSerializer:
    def test_save_and_load_roundtrip(self):
        scene = RadioDramaScene(
            scene_id="scene_000",
            scene_order=1,
            title="The Dock",
            events=(make_event(0),),
        )
        drama = MergedRadioDrama(
            drama_id="round_trip_test",
            title="Round Trip",
            series_mode="pilot",
            era="1940s",
            scenes=(scene,),
            events=scene.events,
            announcer="The Announcer",
            characters=("Sam Vale",),
            voice_casting={"Sam Vale": "male"},
        )
        serializer = RadioDramaSerializer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drama.json"
            saved = serializer.save(drama, path)
            assert saved.exists()

            loaded = serializer.load(saved)
            assert loaded["drama_id"] == "round_trip_test"
            assert loaded["series_mode"] == "pilot"
            assert len(loaded["scenes"]) == 1
            assert loaded["scenes"][0]["title"] == "The Dock"
            assert loaded["voice_casting"]["Sam Vale"] == "male"

    def test_to_json_string(self):
        drama = MergedRadioDrama(drama_id="s", title="T", era="present")
        serializer = RadioDramaSerializer()
        result = serializer.to_json_string(drama)
        assert '"drama_id": "s"' in result
        assert json.loads(result)["title"] == "T"


# ═══════════════════════════════════════════════════════════════════════════
# Package imports
# ═══════════════════════════════════════════════════════════════════════════

class TestRadioDramaImports:
    def test_public_symbols_importable(self):
        from vox_romana.shows.radio_dramas import (
            CharacterConfig,
            MergedRadioDrama,
            RadioDramaConfig,
            RadioDramaDirectorBrief,
            RadioDramaEvent,
            RadioDramaScene,
            RadioDramaSerializer,
            SceneConfig,
        )
        assert RadioDramaConfig is not None
        assert RadioDramaSerializer is not None

    def test_constants(self):
        assert ERAS == ("1940s", "1970s", "present", "future")
        assert SERIES_MODES == ("pilot", "serial")


# ═══════════════════════════════════════════════════════════════════════════
# Prompts (shot 5.1-b)
# ═══════════════════════════════════════════════════════════════════════════

class TestRadioDramaPrompts:
    def test_era_table_has_all_eras(self):
        assert set(rd_prompts.ERA_TABLE) == set(ERAS)

    def test_era_guidance_is_brace_free(self):
        # era_guidance is spliced into str.format templates as a value, so it
        # must contain no braces.
        for era in ERAS:
            g = rd_prompts.era_guidance(era)
            assert "{" not in g
            assert "}" not in g

    def test_era_guidance_reflects_era(self):
        g = rd_prompts.era_guidance("1940s")
        assert "Golden Age" in g
        assert "ANNOUNCER REGISTER" in g
        assert "MUSIC STYLE" in g
        # Unknown era falls back to present.
        assert "Present day" in rd_prompts.era_guidance("bogus")

    def _scriptwriter_kwargs(self):
        return dict(
            show_name="The Dock Street Files",
            genre="noir mystery",
            title="The Missing Heiress",
            setting="Philadelphia, 1946",
            era_guidance=rd_prompts.era_guidance("1940s"),
            cast_descriptions=(
                "  Sam Vale (detective): terse, world-weary\n"
                "  The Announcer (announcer): boom-voiced"
            ),
            character_names='"Sam Vale", "The Announcer"',
        )

    def test_scriptwriter_formats_and_advertises_issue_flag(self):
        prompt = rd_prompts.SCRIPTWRITER_SYSTEM_PROMPT.format(
            **self._scriptwriter_kwargs()
        )
        assert "Sam Vale" in prompt
        assert "The Announcer" in prompt
        assert "Golden Age of Radio" in prompt
        assert '"issue_flag"' in prompt
        assert prompt.endswith(ISSUE_FLAG_PROMPT_NOTE)

    def test_scriptwriter_prompt_mentions_sfx_and_music_cues(self):
        prompt = rd_prompts.SCRIPTWRITER_SYSTEM_PROMPT.format(
            **self._scriptwriter_kwargs()
        )
        assert "[music:" in prompt
        assert "[sfx:" in prompt

    def test_scene_delivery_formats(self):
        prompt = rd_prompts.SCENE_DELIVERY_PROMPT.format(
            scene_title="The Dock",
            location="A rain-slicked dock",
            synopsis="Sam confronts a witness.",
            characters_present="Sam Vale",
            scene_tone="tense",
            length_contract="LENGTH REQUIREMENT (HARD): ...",
            context_summary="(First scene.)",
            director_note="",
            runtime_status="",
            cliffhanger_instruction="",
        )
        assert "The Dock" in prompt
        assert "Sam Vale" in prompt

    def test_architect_prompt_has_no_orphan_placeholders(self):
        # The architect prompt uses only {{ }} JSON braces and no {field}
        # placeholders, so .format() with no kwargs must succeed.
        rendered = rd_prompts.ARCHITECT_PROMPT.format()
        assert "series_mode" in rendered
        assert "recurring_cast" in rendered
        assert "cliffhanger" in rendered

    def test_character_profiler_formats(self):
        prompt = rd_prompts.CHARACTER_PROFILER_PROMPT.format(
            name="Sam Vale",
            role="detective",
            archetype="hardboiled detective",
            description="A weary private eye.",
            genre="noir mystery",
            era="1940s",
        )
        assert "Sam Vale" in prompt
        assert "1940s" in prompt

    def test_director_prompts_format(self):
        check = rd_prompts.DIRECTOR_CHECK_PROMPT.format(
            scene_number=2,
            total_scenes=5,
            arc="Tension builds then resolves.",
            current_mood="tense",
            closing_strategy="Cliffhanger hook.",
            recent_content="Sam found the note.",
        )
        assert "tense" in check

        summary = rd_prompts.CONTEXT_SUMMARY_PROMPT.format(
            transcript="Sam entered the dock office."
        )
        assert "Sam entered" in summary
