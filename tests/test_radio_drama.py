"""Tests for the radio drama show infrastructure.

Covers:
    1. Config validation (cast/scenes/era/series-mode checks)
    2. CharacterConfig / SceneConfig / RadioDramaDirectorBrief defaults
    3. Model to_dict + JSON round-trip (RadioDramaScene, MergedRadioDrama)
    4. Serializer save/load round-trip
    5. Prompts (era table, format-safety)
    6. Bootstrap architect/director/character parsing + pilot-vs-serial planner
    7. Package imports

Conductor tests land with shot 5.1-d.

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
    RadioDramaBootstrap,
    RadioDramaConductor,
    RadioDramaConductorResult,
    RadioDramaConfig,
    RadioDramaDirectorBrief,
    RadioDramaEvent,
    RadioDramaScene,
    RadioDramaSerializer,
    SceneConfig,
)
from vox_romana.shows.radio_dramas import prompts as rd_prompts
from vox_romana.core.messages import ISSUE_FLAG_PROMPT_NOTE

from unittest.mock import Mock


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


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap (shot 5.1-c)
# ═══════════════════════════════════════════════════════════════════════════

def _architect_json(series_mode="pilot"):
    return json.dumps({
        "title": "The Missing Heiress",
        "genre": "noir mystery",
        "setting": "Philadelphia, 1946",
        "tone": "tense",
        "announcer": {"name": "The Announcer", "voice_gender": "male"},
        "characters": [
            {
                "name": "Sam Vale",
                "role": "detective",
                "archetype": "hardboiled detective",
                "description": "A weary private eye.",
                "speech_style": "terse",
                "voice_gender": "male",
            },
            {
                "name": "Dolores Kane",
                "role": "femme fatale",
                "archetype": "femme fatale",
                "description": "The missing heiress's sister.",
                "speech_style": "smooth, guarded",
                "voice_gender": "female",
            },
        ],
        "scenes": [
            {
                "title": "The Dock",
                "synopsis": "Sam confronts a witness.",
                "location": "A rain-slicked dock",
                "characters": ["Sam Vale"],
                "duration_hint": "medium",
                "music_hint": "tense string swell",
                "cliffhanger": False,
            },
            {
                "title": "The Letter",
                "synopsis": "Dolores reveals a secret.",
                "location": "A smoky office",
                "characters": ["Sam Vale", "Dolores Kane"],
                "duration_hint": "medium",
                "music_hint": "dramatic brass hit",
                "cliffhanger": False,
            },
        ],
        "series_mode": series_mode,
        "series_id": "dock-street-files" if series_mode == "serial" else None,
        "episode_number": 1 if series_mode == "serial" else None,
        "recurring_cast": ["Sam Vale", "Dolores Kane"] if series_mode == "serial" else [],
        "language": "en",
        "music_style": "1940s big band",
    })


def _profiler_json():
    return json.dumps({
        "voice_gender": "male",
        "voice_hint": "gravelly baritone, 50s",
        "tts_style": "Speak low and measured.",
        "voice_fingerprint": "Terse, clipped sentences. World-weary.",
    })


def _director_json():
    return json.dumps({
        "narrative_arc": "Open tense, build, resolve with a sting.",
        "mood_progression": ["uneasy", "tense", "shocked", "resolved"],
        "scene_notes": "Tighten transitions between scenes 1 and 2.",
        "closing_strategy": "Full resolution.",
    })


class TestRadioDramaBootstrap:
    def _make_bootstrap(self):
        return RadioDramaBootstrap(client=Mock())

    def test_parse_architect_pilot(self):
        bootstrap = self._make_bootstrap()
        config = bootstrap._parse_architect_response(
            _architect_json("pilot"),
            brief="A noir mystery about a missing heiress.",
            fact_check="off",
        )
        assert isinstance(config, RadioDramaConfig)
        assert config.title == "The Missing Heiress"
        assert config.genre == "noir mystery"
        assert config.series_mode == "pilot"
        assert config.series_id is None
        assert config.episode_number is None
        assert config.recurring_cast == []
        assert len(config.characters) == 2
        assert config.announcer is not None
        assert config.announcer.role == "announcer"
        assert len(config.scenes) == 2
        assert config.scenes[0].title == "The Dock"

    def test_parse_architect_serial_records_planner_fields(self):
        bootstrap = self._make_bootstrap()
        config = bootstrap._parse_architect_response(
            _architect_json("serial"),
            brief="A recurring noir series.",
        )
        assert config.series_mode == "serial"
        assert config.series_id == "dock-street-files"
        assert config.episode_number == 1
        # recurring_cast is name-curated (diverse names), so it must match
        # the curated character names — not the raw architect names.
        assert config.recurring_cast == [c.name for c in config.characters]

    def test_parse_architect_invalid_series_mode_falls_back_to_pilot(self):
        data = json.loads(_architect_json("pilot"))
        data["series_mode"] = "anthology"
        bootstrap = self._make_bootstrap()
        config = bootstrap._parse_architect_response(json.dumps(data), brief="x")
        assert config.series_mode == "pilot"

    def test_parse_architect_invalid_json_raises(self):
        bootstrap = self._make_bootstrap()
        import pytest
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_architect_response("not json", brief="x")

    def test_parse_architect_markdown_fenced(self):
        bootstrap = self._make_bootstrap()
        raw = "```json\n" + _architect_json("pilot") + "\n```"
        config = bootstrap._parse_architect_response(raw, brief="x")
        assert config.title == "The Missing Heiress"

    def test_parse_director(self):
        bootstrap = self._make_bootstrap()
        brief = bootstrap._parse_director_response(_director_json())
        assert isinstance(brief, RadioDramaDirectorBrief)
        assert brief.narrative_arc.startswith("Open tense")
        assert brief.mood_progression == ["uneasy", "tense", "shocked", "resolved"]
        assert brief.closing_strategy == "Full resolution."

    def test_parse_director_invalid_json_raises(self):
        bootstrap = self._make_bootstrap()
        import pytest
        with pytest.raises(ValueError, match="invalid JSON"):
            bootstrap._parse_director_response("not json")

    def test_profile_character_fills_voice_fields(self):
        bootstrap = self._make_bootstrap()
        bootstrap._client.chat.return_value = _profiler_json()
        character = make_character()
        profiled = bootstrap._profile_character(
            character, genre="noir mystery", era="1940s",
        )
        assert profiled.voice_fingerprint == "Terse, clipped sentences. World-weary."
        assert profiled.voice_hint == "gravelly baritone, 50s"
        assert profiled.tts_style == "Speak low and measured."

    def test_profile_character_bad_json_falls_back(self):
        bootstrap = self._make_bootstrap()
        bootstrap._client.chat.return_value = "not json"
        character = make_character()
        profiled = bootstrap._profile_character(
            character, genre="noir", era="1940s",
        )
        assert profiled.voice_fingerprint == "(distinct dramatic voice)"

    def test_from_brief_full_chain(self):
        """Architect → 2 character profiles → announcer profile → director."""
        bootstrap = self._make_bootstrap()
        # chat() call order:
        #   1. architect
        #   2. announcer profile
        #   3. character 1 profile
        #   4. character 2 profile
        #   5. director
        bootstrap._client.chat.side_effect = [
            _architect_json("serial"),
            _profiler_json(),  # announcer
            _profiler_json(),  # Sam
            _profiler_json(),  # Dolores
            _director_json(),
        ]
        config = bootstrap.from_brief("A recurring noir series.")
        assert isinstance(config, RadioDramaConfig)
        assert config.series_mode == "serial"
        assert config.director_brief is not None
        assert config.director_brief.mood_progression == ["uneasy", "tense", "shocked", "resolved"]
        # Every cast member got a voice fingerprint.
        assert config.announcer.voice_fingerprint is not None
        assert all(c.voice_fingerprint is not None for c in config.characters)
        assert bootstrap._client.chat.call_count == 5

    def test_from_brief_applies_overrides(self):
        bootstrap = self._make_bootstrap()
        bootstrap._client.chat.side_effect = [
            _architect_json("pilot"),
            _profiler_json(),  # announcer
            _profiler_json(),  # Sam
            _profiler_json(),  # Dolores
            _director_json(),
        ]
        config = bootstrap.from_brief(
            "A noir mystery.",
            overrides={"era": "1970s", "genre": "paranoid thriller"},
        )
        assert config.era == "1970s"
        assert config.genre == "paranoid thriller"


# ═══════════════════════════════════════════════════════════════════════════
# Conductor (shot 5.1-d)
# ═══════════════════════════════════════════════════════════════════════════

def make_scene_response(speakers: list[str], num_turns: int = 4) -> str:
    """Build a canned JSON array response for a scene."""
    filler = (
        " The rain kept falling and every shadow in the doorway looked like a "
        "man with a grudge, so we kept our voices low and our eyes on the "
        "alley, waiting for the next pair of footsteps to give the game away "
        "before the clock in the tower struck midnight."
    )
    turns = []
    for i in range(num_turns):
        speaker = speakers[i % len(speakers)]
        turns.append({
            "speaker_name": speaker,
            "text": f"This is line {i + 1} from {speaker}.{filler}",
            "tone": "tense" if i % 2 == 0 else "somber",
            "volume": "moderate",
            "pace": "medium",
        })
    return json.dumps(turns, indent=2)


class TestRadioDramaConductor:
    def _make_conductor(self, config=None):
        return RadioDramaConductor(config or make_config(), client=Mock())

    def test_build_agents_sets_system_prompt(self):
        conductor = self._make_conductor()
        conductor._build_agents()
        assert "The Dock Street Case" in conductor._system_prompt
        assert "Sam Vale" in conductor._system_prompt
        assert "Golden Age of Radio" in conductor._system_prompt
        assert conductor._session is not None

    def test_parse_scene_response(self):
        conductor = self._make_conductor()
        raw = json.dumps([
            {"speaker_name": "Sam Vale", "text": "The rain kept falling.", "tone": "tense"},
            {"speaker_name": "Dolores Kane", "text": "You're lying.", "tone": "somber"},
        ])
        events = conductor._parse_scene_response(
            raw, scene_id="scene_000", global_order_start=0,
        )
        assert len(events) == 2
        assert events[0]["speaker_name"] == "Sam Vale"
        assert events[1]["speaker_name"] == "Dolores Kane"
        assert events[0]["scene_id"] == "scene_000"
        assert events[0]["global_order"] == 0
        assert events[1]["global_order"] == 1

    def test_parse_scene_fallback_on_bad_json(self):
        conductor = self._make_conductor()
        events = conductor._parse_scene_response(
            "not json at all", scene_id="scene_000", global_order_start=0,
        )
        assert len(events) == 1
        # Falls back to the announcer name.
        assert events[0]["speaker_name"] == "The Announcer"

    def test_parse_scene_issue_flag_stripped(self):
        conductor = self._make_conductor()
        raw = json.dumps([
            {
                "speaker_name": "Sam Vale",
                "text": "Hello.",
                "issue_flag": "The outline contradicts the setting.",
            },
        ])
        events = conductor._parse_scene_response(
            raw, scene_id="scene_000", global_order_start=0,
        )
        assert "issue_flag" not in events[0]
        rendered = conductor._consume_pending_issue_reports()
        assert "character:Sam Vale" in rendered

    def test_full_run_produces_result(self):
        config = make_config(num_scenes=2)
        # Small explicit targets keep the canned responses above the 70%
        # retry threshold so this test exercises wiring, not length logic.
        for scene in config.scenes:
            scene.target_words = 150
        conductor = self._make_conductor(config)
        conductor._build_agents()

        host_names = ["Sam Vale", "Dolores Kane"]
        responses = [
            make_scene_response(host_names, num_turns=3),
            make_scene_response(host_names, num_turns=3),
        ]
        call_idx = [0]

        def mock_send(msg):
            idx = call_idx[0]
            call_idx[0] += 1
            return responses[idx] if idx < len(responses) else responses[-1]

        conductor._session.send = mock_send

        raw_events = conductor._run_turn_loop()
        result = conductor._package_result(raw_events)

        assert isinstance(result, RadioDramaConductorResult)
        assert len(result.drama_events) == 6
        assert len(result.drama_scenes) == 2
        assert result.drama_scenes[0].title == "Scene 1"

    def test_to_merged_radio_drama_voice_maps(self):
        config = make_config(series_mode="serial", series_id="dock-street")
        conductor = self._make_conductor(config)
        conductor._drama_id = "test_123"

        result = RadioDramaConductorResult(
            config=config,
            events=[],
            transcript=[],
            total_llm_calls=1,
            drama_events=[],
            drama_scenes=[],
            drama_id="test_123",
        )
        merged = result.to_merged_radio_drama()
        assert isinstance(merged, MergedRadioDrama)
        assert merged.drama_id == "test_123"
        assert merged.series_mode == "serial"
        assert merged.series_id == "dock-street"
        assert merged.voice_casting["Sam Vale"] == "male"
        assert merged.voice_casting["Dolores Kane"] == "female"
        assert "The Announcer" in merged.voice_casting

    def test_summarize_scene(self):
        config = make_config()
        events = [
            {"speaker_name": "Sam Vale", "text": "Hello, detective."},
            {"speaker_name": "Dolores Kane", "text": "Don't trust him."},
        ]
        scene = config.scenes[0]
        summary = RadioDramaConductor._summarize_scene(events, scene)
        assert "Scene 1" in summary
        assert "Sam Vale" in summary

    def test_build_context_summary(self):
        conductor = self._make_conductor()
        conductor._context_log = ["Scene 1 — speakers: Sam Vale."]
        result = conductor._build_context_summary()
        assert "Scene 1" in result


class TestRadioDramaConductorGaps:
    """Targeted coverage for conductor paths not exercised by the happy path."""

    def _make_conductor(self, config=None):
        return RadioDramaConductor(config or make_config(), client=Mock())

    def test_build_cast_descriptions_includes_announcer_and_cast(self):
        config = make_config()
        desc = RadioDramaConductor._build_cast_descriptions(config)
        assert "The Announcer" in desc
        assert "Sam Vale" in desc
        assert "Dolores Kane" in desc

    def test_build_cast_descriptions_empty_cast(self):
        config = RadioDramaConfig(brief="x", scenes=[SceneConfig()])
        desc = RadioDramaConductor._build_cast_descriptions(config)
        assert desc == "(no cast)"

    def test_parse_scene_empty_array_falls_back(self):
        conductor = self._make_conductor()
        events = conductor._parse_scene_response(
            "[]", scene_id="scene_000", global_order_start=0,
        )
        assert len(events) == 1
        assert events[0]["speaker_name"] == "The Announcer"

    def test_parse_scene_skips_non_dict_items(self):
        conductor = self._make_conductor()
        raw = json.dumps([
            {"speaker_name": "Sam Vale", "text": "Line one."},
            "not a dict",
            42,
            {"speaker_name": "Dolores Kane", "text": "Line two."},
        ])
        events = conductor._parse_scene_response(
            raw, scene_id="scene_000", global_order_start=0,
        )
        assert len(events) == 2
        assert events[0]["speaker_name"] == "Sam Vale"
        assert events[1]["speaker_name"] == "Dolores Kane"

    def test_director_check_applies_mood_note(self):
        config = make_config()
        config.director_brief = RadioDramaDirectorBrief(
            narrative_arc="Build then resolve.",
            mood_progression=["uneasy", "tense", "resolved"],
            closing_strategy="Full resolution.",
        )
        conductor = self._make_conductor(config)
        conductor._client.chat.return_value = json.dumps({
            "mood_note": "Keep it tense.",
            "nudge": "Raise the stakes.",
            "wrap_up": False,
        })
        result = conductor._director_check(2, 3)
        assert result["mood_note"] == "Keep it tense."
        assert conductor._director_note == "Keep it tense. | Raise the stakes."

    def test_director_check_no_brief_returns_empty(self):
        conductor = self._make_conductor(make_config())
        assert conductor._director_check(1, 3) == {}

    def test_director_check_bad_json_returns_empty(self):
        config = make_config()
        config.director_brief = RadioDramaDirectorBrief(
            narrative_arc="arc", mood_progression=["tense"],
        )
        conductor = self._make_conductor(config)
        conductor._client.chat.return_value = "not json"
        assert conductor._director_check(2, 3) == {}

    def test_compress_context_writes_compressed(self):
        conductor = self._make_conductor()
        conductor._context_log = [
            f"Scene {i} — speakers: Sam Vale." for i in range(1, 6)
        ]
        conductor._client.chat.return_value = "Earlier scenes built tension."
        conductor._compress_context()
        assert conductor._compressed_ctx == "Earlier scenes built tension."
        # Keeps only the 2 most recent verbatim.
        assert len(conductor._context_log) == 2

    def test_compress_context_noop_when_too_short(self):
        conductor = self._make_conductor()
        conductor._context_log = ["Scene 1 — speakers: Sam Vale."]
        conductor._compress_context()
        assert conductor._compressed_ctx == ""
        assert len(conductor._context_log) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Router + runners wiring (shot 5.1-e)
# ═══════════════════════════════════════════════════════════════════════════

class TestRadioDramaWiring:
    def test_router_has_radio_drama(self):
        from vox_romana.core.router import SHOW_TYPES
        assert "radio_drama" in SHOW_TYPES

    def test_runners_dispatch_table_has_radio_drama(self):
        from vox_romana.cli.runners import _RUNNERS, _CONTENT_TYPE_KEYS
        assert "radio_drama" in _RUNNERS
        assert _CONTENT_TYPE_KEYS.get("drama_id") == "radio_drama"

    def test_duration_wpm_has_radio_drama(self):
        from vox_romana.core.duration import get_wpm
        assert get_wpm("radio_drama") == 150
