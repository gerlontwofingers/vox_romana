"""Tests for vox_romana.animation.director.

Verifies the AnimationDirector LLM agent: prompt construction, response
parsing (including markdown fence stripping), enum validation with
fallbacks, scene parsing, and event summarization.

All tests use a mock DeepSeekClient — no real LLM calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vox_romana.animation.director import (
    ANIMATION_DIRECTOR_SYSTEM_PROMPT,
    AnimationDirector,
)
from vox_romana.animation.models import (
    AnimationPlan,
    HardwareProfile,
    SceneSpec,
)
from vox_romana.core.models import AudioEvent


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def hw_profile() -> HardwareProfile:
    """Standard MPS profile matching devstation_01."""
    return HardwareProfile(
        device="mps",
        vram_gb=32.0,
        total_ram_gb=32.0,
        recommended_sd_model="stabilityai/stable-diffusion-xl-base-1.0",
        recommended_resolution=(1920, 1080),
        recommended_fps=30,
        low_disk_warning=False,
        free_disk_gb=37.0,
    )


@pytest.fixture
def cpu_profile() -> HardwareProfile:
    """Low-end fallback profile."""
    return HardwareProfile(
        device="cpu",
        vram_gb=None,
        total_ram_gb=8.0,
        recommended_sd_model="runwayml/stable-diffusion-v1-5",
        recommended_resolution=(512, 512),
        recommended_fps=24,
        low_disk_warning=True,
        free_disk_gb=5.0,
    )


@pytest.fixture
def sample_events() -> list[AudioEvent]:
    """A small set of AudioEvents."""
    return [
        AudioEvent(
            global_order=0, local_order=0,
            speaker_role="moderator", speaker_name="Sarah Chen",
            target=None, text="Welcome to today's debate on universal basic income.",
            event_kind="utterance",
        ),
        AudioEvent(
            global_order=1, local_order=1,
            speaker_role="participant", speaker_name="Dr. Vance",
            target="all", text="UBI is the most important policy of our generation.",
            event_kind="utterance",
        ),
        AudioEvent(
            global_order=2, local_order=2,
            speaker_role="participant", speaker_name="Dr. Marlowe",
            target="Dr. Vance", text="That's a wildly optimistic claim with no empirical backing.",
            event_kind="utterance",
        ),
    ]


@pytest.fixture
def canned_plan_json() -> str:
    """A well-formed director response covering 3 events in 2 scenes."""
    return json.dumps({
        "visual_instructions": "Documentary feel with two-shot interview framing.",
        "style": "documentary",
        "render_backend": "moviepy",
        "color_palette": ["#1a3a5c", "#d4af37", "#f5f5f5"],
        "font_family": "Helvetica",
        "fps": 30,
        "resolution": [1920, 1080],
        "image_model": None,
        "scenes": [
            {
                "scene_id": "scene_01",
                "event_global_orders": [0],
                "image_prompt": "modern television studio, anchor desk, warm lighting",
                "negative_prompt": "blurry, watermark, low quality",
                "text_overlay": "Universal Basic Income — A Debate",
                "speaker_label": "Sarah Chen — Moderator",
                "transition_in": "fade",
                "transition_out": "cut",
                "animation_hint": "ken_burns_zoom_in",
            },
            {
                "scene_id": "scene_02",
                "event_global_orders": [1, 2],
                "image_prompt": "two economists in heated discussion at panel desk",
                "negative_prompt": "blurry, distorted",
                "text_overlay": None,
                "speaker_label": "Dr. Vance — Economist",
                "transition_in": "cut",
                "transition_out": "crossfade",
                "animation_hint": "static",
            },
        ],
    })


@pytest.fixture
def mock_client(canned_plan_json: str) -> MagicMock:
    """A DeepSeekClient mock whose .chat() returns canned JSON."""
    client = MagicMock()
    client.chat.return_value = canned_plan_json
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_init_stores_client(self) -> None:
        client = MagicMock()
        director = AnimationDirector(client)
        assert director._client is client

    def test_system_prompt_is_module_constant(self) -> None:
        assert isinstance(ANIMATION_DIRECTOR_SYSTEM_PROMPT, str)
        assert "Animation Director" in ANIMATION_DIRECTOR_SYSTEM_PROMPT
        # Must explicitly request raw JSON (no markdown) for parsing reliability.
        assert "JSON" in ANIMATION_DIRECTOR_SYSTEM_PROMPT
        assert "scenes" in ANIMATION_DIRECTOR_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# run() — message construction & client call
# ─────────────────────────────────────────────────────────────────────────────

class TestRun:
    def test_run_returns_animation_plan(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        plan = director.run("brief", sample_events, hardware_profile=hw_profile)
        assert isinstance(plan, AnimationPlan)

    def test_run_uses_temperature_zero(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        director.run("brief", sample_events, hardware_profile=hw_profile)
        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    def test_run_sends_system_and_user_messages(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        director.run("brief about UBI", sample_events, hardware_profile=hw_profile)
        msgs = mock_client.chat.call_args.kwargs["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == ANIMATION_DIRECTOR_SYSTEM_PROMPT
        assert msgs[1]["role"] == "user"
        assert "brief about UBI" in msgs[1]["content"]

    def test_run_includes_visual_brief_when_provided(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        director.run(
            "brief", sample_events, hardware_profile=hw_profile,
            visual_brief="noir aesthetic, high contrast",
        )
        user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
        assert "VISUAL_BRIEF" in user_msg
        assert "noir aesthetic" in user_msg
        assert "authoritative" in user_msg.lower()

    def test_run_omits_visual_brief_section_when_absent(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        director.run("brief", sample_events, hardware_profile=hw_profile)
        user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
        assert "VISUAL_BRIEF" not in user_msg

    def test_run_omits_visual_brief_when_blank(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        director.run(
            "brief", sample_events, hardware_profile=hw_profile,
            visual_brief="   ",
        )
        user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
        assert "VISUAL_BRIEF" not in user_msg

    def test_run_includes_hardware_summary(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        director.run("brief", sample_events, hardware_profile=hw_profile)
        user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
        assert "HARDWARE" in user_msg
        assert "mps" in user_msg
        assert "1920" in user_msg

    def test_run_includes_available_backends(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        director.run(
            "brief", sample_events, hardware_profile=hw_profile,
            available_backends=("moviepy", "manim"),
        )
        user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
        assert "AVAILABLE_BACKENDS" in user_msg
        assert "moviepy" in user_msg
        assert "manim" in user_msg

    def test_run_includes_event_summary(
        self, mock_client: MagicMock, sample_events: list[AudioEvent],
        hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        director.run("brief", sample_events, hardware_profile=hw_profile)
        user_msg = mock_client.chat.call_args.kwargs["messages"][1]["content"]
        assert "EVENTS" in user_msg
        assert "Sarah Chen" in user_msg
        assert "Dr. Vance" in user_msg
        assert "Dr. Marlowe" in user_msg

    def test_run_accepts_dict_events(
        self, mock_client: MagicMock, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(mock_client)
        events_as_dicts = [
            {"global_order": 0, "speaker_name": "Alice",
             "speaker_role": "host", "text": "hi", "event_kind": "utterance"}
        ]
        plan = director.run("brief", events_as_dicts, hardware_profile=hw_profile)
        assert isinstance(plan, AnimationPlan)


# ─────────────────────────────────────────────────────────────────────────────
# parse_response() — happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestParseResponseHappyPath:
    def test_parses_well_formed_json(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            canned_plan_json, brief="brief", hardware_profile=hw_profile,
        )
        assert plan.style == "documentary"
        assert plan.render_backend == "moviepy"
        assert plan.fps == 30
        assert plan.resolution == (1920, 1080)
        assert plan.font_family == "Helvetica"
        assert len(plan.scenes) == 2

    def test_image_model_falls_back_to_hardware_default_when_null(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            canned_plan_json, brief="brief", hardware_profile=hw_profile,
        )
        assert plan.image_model == hw_profile.recommended_sd_model

    def test_first_scene_fields(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            canned_plan_json, brief="brief", hardware_profile=hw_profile,
        )
        s = plan.scenes[0]
        assert s.scene_id == "scene_01"
        assert s.event_global_orders == [0]
        assert "television studio" in s.image_prompt
        assert s.text_overlay == "Universal Basic Income — A Debate"
        assert s.speaker_label == "Sarah Chen — Moderator"
        assert s.transition_in == "fade"
        assert s.animation_hint == "ken_burns_zoom_in"

    def test_second_scene_groups_two_events(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            canned_plan_json, brief="brief", hardware_profile=hw_profile,
        )
        s = plan.scenes[1]
        assert s.event_global_orders == [1, 2]
        assert s.text_overlay is None  # null in JSON → None
        assert s.transition_out == "crossfade"

    def test_plan_id_is_unique(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        a = director.parse_response(canned_plan_json, brief="b", hardware_profile=hw_profile)
        b = director.parse_response(canned_plan_json, brief="b", hardware_profile=hw_profile)
        assert a.plan_id != b.plan_id

    def test_plan_stores_brief_and_hardware_profile(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            canned_plan_json, brief="my brief", hardware_profile=hw_profile,
        )
        assert plan.brief == "my brief"
        assert plan.hardware_profile is hw_profile

    def test_color_palette_parsed(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            canned_plan_json, brief="b", hardware_profile=hw_profile,
        )
        assert plan.color_palette == ["#1a3a5c", "#d4af37", "#f5f5f5"]


# ─────────────────────────────────────────────────────────────────────────────
# parse_response() — markdown fences
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkdownFenceStripping:
    def test_strips_json_fence(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        wrapped = f"```json\n{canned_plan_json}\n```"
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(wrapped, brief="b", hardware_profile=hw_profile)
        assert plan.style == "documentary"

    def test_strips_plain_fence(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        wrapped = f"```\n{canned_plan_json}\n```"
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(wrapped, brief="b", hardware_profile=hw_profile)
        assert plan.style == "documentary"

    def test_handles_leading_trailing_whitespace(
        self, canned_plan_json: str, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            f"\n\n{canned_plan_json}\n\n",
            brief="b", hardware_profile=hw_profile,
        )
        assert plan.style == "documentary"


# ─────────────────────────────────────────────────────────────────────────────
# parse_response() — error paths
# ─────────────────────────────────────────────────────────────────────────────

class TestParseResponseErrors:
    def test_malformed_json_raises_value_error(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        with pytest.raises(ValueError, match="invalid JSON"):
            director.parse_response("not json{{", brief="b", hardware_profile=hw_profile)

    def test_non_object_root_raises(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        with pytest.raises(ValueError, match="expected a JSON object"):
            director.parse_response("[1, 2, 3]", brief="b", hardware_profile=hw_profile)

    def test_missing_scenes_raises(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({"style": "documentary"})
        with pytest.raises(ValueError, match="scenes"):
            director.parse_response(bad, brief="b", hardware_profile=hw_profile)

    def test_empty_scenes_raises(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({"style": "documentary", "scenes": []})
        with pytest.raises(ValueError, match="scenes"):
            director.parse_response(bad, brief="b", hardware_profile=hw_profile)


# ─────────────────────────────────────────────────────────────────────────────
# Enum validation & fallbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestEnumValidation:
    def _minimal_response(self, **overrides: object) -> str:
        base: dict[str, object] = {
            "style": "documentary",
            "render_backend": "moviepy",
            "scenes": [{
                "scene_id": "s1",
                "event_global_orders": [0],
                "image_prompt": "x",
                "transition_in": "cut",
                "transition_out": "cut",
                "animation_hint": "static",
            }],
        }
        base.update(overrides)
        return json.dumps(base)

    def test_invalid_style_falls_back_to_cinematic(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            self._minimal_response(style="bogus"),
            brief="b", hardware_profile=hw_profile,
        )
        assert plan.style == "cinematic"

    def test_invalid_backend_falls_back_to_first_available(
        self, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        plan = director.parse_response(
            self._minimal_response(render_backend="manim"),
            brief="b", hardware_profile=hw_profile,
            available_backends=("moviepy",),
        )
        assert plan.render_backend == "moviepy"

    def test_invalid_scene_transition_falls_back_to_cut(
        self, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({
            "style": "documentary",
            "render_backend": "moviepy",
            "scenes": [{
                "scene_id": "s1",
                "event_global_orders": [0],
                "image_prompt": "x",
                "transition_in": "swirl",
                "transition_out": "explode",
                "animation_hint": "rocket",
            }],
        })
        plan = director.parse_response(bad, brief="b", hardware_profile=hw_profile)
        assert plan.scenes[0].transition_in == "cut"
        assert plan.scenes[0].transition_out == "cut"
        assert plan.scenes[0].animation_hint == "static"


# ─────────────────────────────────────────────────────────────────────────────
# Numeric fallbacks (fps, resolution)
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericFallbacks:
    def test_missing_fps_uses_hardware_default(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({
            "style": "documentary",
            "render_backend": "moviepy",
            "scenes": [{"scene_id": "s1", "event_global_orders": [0],
                        "image_prompt": "x"}],
        })
        plan = director.parse_response(bad, brief="b", hardware_profile=hw_profile)
        assert plan.fps == hw_profile.recommended_fps

    def test_invalid_fps_uses_hardware_default(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({
            "style": "documentary", "render_backend": "moviepy",
            "fps": "fast",
            "scenes": [{"scene_id": "s1", "event_global_orders": [0],
                        "image_prompt": "x"}],
        })
        plan = director.parse_response(bad, brief="b", hardware_profile=hw_profile)
        assert plan.fps == hw_profile.recommended_fps

    def test_invalid_resolution_uses_hardware_default(
        self, cpu_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({
            "style": "documentary", "render_backend": "moviepy",
            "resolution": "huge",
            "scenes": [{"scene_id": "s1", "event_global_orders": [0],
                        "image_prompt": "x"}],
        })
        plan = director.parse_response(bad, brief="b", hardware_profile=cpu_profile)
        assert plan.resolution == cpu_profile.recommended_resolution


# ─────────────────────────────────────────────────────────────────────────────
# Scene parsing edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSceneParsing:
    def test_non_dict_scene_yields_empty_spec(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({
            "style": "documentary", "render_backend": "moviepy",
            "scenes": ["not a dict"],
        })
        plan = director.parse_response(bad, brief="b", hardware_profile=hw_profile)
        assert isinstance(plan.scenes[0], SceneSpec)
        assert plan.scenes[0].scene_id == "scene_00"

    def test_missing_scene_id_gets_indexed_default(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({
            "style": "documentary", "render_backend": "moviepy",
            "scenes": [{"event_global_orders": [0], "image_prompt": "x"}],
        })
        plan = director.parse_response(bad, brief="b", hardware_profile=hw_profile)
        assert plan.scenes[0].scene_id == "scene_00"

    def test_empty_string_text_overlay_becomes_none(
        self, hw_profile: HardwareProfile,
    ) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({
            "style": "documentary", "render_backend": "moviepy",
            "scenes": [{"scene_id": "s1", "event_global_orders": [0],
                        "image_prompt": "x", "text_overlay": "  "}],
        })
        plan = director.parse_response(bad, brief="b", hardware_profile=hw_profile)
        assert plan.scenes[0].text_overlay is None

    def test_event_orders_filters_non_numeric(self, hw_profile: HardwareProfile) -> None:
        director = AnimationDirector(MagicMock())
        bad = json.dumps({
            "style": "documentary", "render_backend": "moviepy",
            "scenes": [{
                "scene_id": "s1",
                "event_global_orders": [0, "skip", 2, None, 3],
                "image_prompt": "x",
            }],
        })
        plan = director.parse_response(bad, brief="b", hardware_profile=hw_profile)
        assert plan.scenes[0].event_global_orders == [0, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Event summarisation
# ─────────────────────────────────────────────────────────────────────────────

class TestEventSummarization:
    def test_empty_events_returns_placeholder(self) -> None:
        result = AnimationDirector._summarize_events([])
        assert "no events" in result.lower()

    def test_long_text_is_truncated(self) -> None:
        ev = AudioEvent(
            global_order=0, local_order=0,
            speaker_role="host", speaker_name="A", target=None,
            text="x" * 500,
        )
        result = AnimationDirector._summarize_events([ev])
        # Truncation marker present
        assert "…" in result
        # Original 500-char text not preserved verbatim
        assert "x" * 500 not in result

    def test_newlines_collapsed(self) -> None:
        ev = AudioEvent(
            global_order=0, local_order=0,
            speaker_role="host", speaker_name="A", target=None,
            text="line one\nline two",
        )
        result = AnimationDirector._summarize_events([ev])
        assert "line one line two" in result

    def test_includes_global_order_and_speaker(self) -> None:
        ev = AudioEvent(
            global_order=42, local_order=0,
            speaker_role="participant", speaker_name="Bob", target=None,
            text="hi",
        )
        result = AnimationDirector._summarize_events([ev])
        assert "42" in result
        assert "Bob" in result
        assert "participant" in result

    def test_samples_when_too_many_events(self) -> None:
        many = [
            AudioEvent(
                global_order=i, local_order=i,
                speaker_role="host", speaker_name=f"S{i}", target=None,
                text=f"text {i}",
            )
            for i in range(500)
        ]
        result = AnimationDirector._summarize_events(many)
        # Should not have one line per event
        assert result.count("\n") < 200
        # Last event should still appear
        assert "S499" in result

    def test_skips_unrecognized_items(self) -> None:
        ev = AudioEvent(
            global_order=0, local_order=0,
            speaker_role="host", speaker_name="Alice", target=None, text="hi",
        )
        result = AnimationDirector._summarize_events([ev, 42, None, "string"])  # type: ignore[list-item]
        assert "Alice" in result


# ─────────────────────────────────────────────────────────────────────────────
# Static helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestStaticHelpers:
    def test_optional_str_returns_none_for_blank(self) -> None:
        assert AnimationDirector._optional_str(None) is None
        assert AnimationDirector._optional_str("") is None
        assert AnimationDirector._optional_str("   ") is None
        assert AnimationDirector._optional_str(42) is None

    def test_optional_str_returns_stripped_string(self) -> None:
        assert AnimationDirector._optional_str("  hello  ") == "hello"

    def test_coerce_int_uses_default_on_failure(self) -> None:
        assert AnimationDirector._coerce_int("nope", default=24) == 24
        assert AnimationDirector._coerce_int(None, default=24) == 24
        assert AnimationDirector._coerce_int("30", default=24) == 30

    def test_coerce_resolution_handles_lists(self) -> None:
        assert AnimationDirector._coerce_resolution(
            [800, 600], default=(1920, 1080),
        ) == (800, 600)

    def test_coerce_resolution_uses_default_on_bad_input(self) -> None:
        assert AnimationDirector._coerce_resolution(
            "huge", default=(1920, 1080),
        ) == (1920, 1080)
        assert AnimationDirector._coerce_resolution(
            [1, 2, 3], default=(1920, 1080),
        ) == (1920, 1080)

    def test_validate_enum_returns_value_when_valid(self) -> None:
        assert AnimationDirector._validate_enum(
            "fade", {"cut", "fade"}, default="cut", field_name="t",
        ) == "fade"

    def test_validate_enum_returns_default_when_invalid(self) -> None:
        assert AnimationDirector._validate_enum(
            "swirl", {"cut", "fade"}, default="cut", field_name="t",
        ) == "cut"
