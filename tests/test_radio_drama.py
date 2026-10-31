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
