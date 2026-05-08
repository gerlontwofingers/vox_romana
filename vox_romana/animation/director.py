"""The AnimationDirector — turns a brief + AudioEvents into an AnimationPlan.

If the animation pipeline is a film production, the AnimationDirector is the
director of photography + production designer + storyboard artist rolled into
one. They read the script (the AudioEvents JSON), absorb the brief's vibe,
and produce a complete shooting plan: visual style, color palette, scene
boundaries, image prompts, transitions, motion hints.

Everything downstream — the image generator, the timeline syncer, the video
compositor — just executes what this director decides.

SINGLE-SHOT DESIGN:

    Like the DebateArchitect, the AnimationDirector makes ONE call to the
    LLM. Inputs: brief + events summary + hardware profile + visual_brief.
    Output: a single JSON document parsed into an AnimationPlan.

    No state, no follow-ups. If it screws up, retry from scratch.

USER INPUT FLOWS:

    Two channels carry user intent into the director:

    1. ``brief`` — the original content brief from run.py. Style cues
       embedded there ("BBC documentary feel", "noir aesthetic", "stark
       and minimalist") are picked up by the LLM naturally.

    2. ``visual_brief`` — the dedicated animation-override channel from
       AnimationConfig.visual_brief. This is what a future
       ``--animation-brief "..."`` CLI flag will populate. Treated as
       authoritative — overrides anything in the brief that conflicts.

EVENTS SUMMARY (NOT FULL TRANSCRIPT):

    The full AudioEvents JSON for a 20-minute show can be 100+ KB.
    Sending all of it to the director wastes tokens for no benefit —
    the director needs the SHAPE of the show (who speaks when, what
    the structural beats are), not every word.

    ``_summarize_events()`` reduces the events list to a compact form:
        - speaker name + role
        - text snippet (first ~80 chars)
        - event_kind / segment_id
        - global_order

    The director uses this to decide scene boundaries and write image
    prompts that match what's being said.

HARDWARE-AWARE DECISIONS:

    The director receives the HardwareProfile so it can:
        - Choose a render backend the machine can actually run
        - Match the SD model to the recommended tier
        - Set fps/resolution to the recommended values (or downgrade
          if the LLM thinks the brief calls for something simpler)

EXAMPLE:

    from vox_romana.core import DeepSeekClient
    from vox_romana.animation.hardware import HardwareScanner
    from vox_romana.animation.director import AnimationDirector

    client = DeepSeekClient()
    director = AnimationDirector(client)

    profile = HardwareScanner.scan()
    plan = director.run(
        brief="Two economists debate UBI in a university lecture hall",
        events=audio_events,  # list of AudioEvent or dicts
        hardware_profile=profile,
        visual_brief="academic documentary style, warm lighting",
    )

    print(plan.style)              # "documentary"
    print(plan.total_scenes)       # 12
    print(plan.scenes[0].image_prompt)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from ..core.client import DeepSeekClient
from ..core.models import AudioEvent
from .models import (
    AnimationHint,
    AnimationPlan,
    AnimationStyle,
    HardwareProfile,
    RenderBackend,
    SceneSpec,
    TransitionKind,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Valid enum values — used for validation when parsing the LLM response.
_VALID_STYLES: set[str] = {
    "cinematic", "documentary", "kinetic_text", "illustrated", "talking_heads",
}

_VALID_BACKENDS: set[str] = {"moviepy", "manim", "ffmpeg"}

_VALID_HINTS: set[str] = {
    "ken_burns_zoom_in", "ken_burns_zoom_out",
    "ken_burns_pan_left", "ken_burns_pan_right",
    "parallax", "static", "text_pulse", "talking_head_sync",
}

_VALID_TRANSITIONS: set[str] = {"cut", "fade", "crossfade", "wipe"}

# How many characters of text per event to include in the events summary.
# Long enough for the LLM to grasp tone; short enough to keep the prompt lean.
_EVENT_TEXT_PREVIEW_CHARS: int = 80

# Cap on number of events sent to the director. Beyond this, we sample
# every Nth event so the director still sees the show's shape.
_MAX_EVENTS_IN_PROMPT: int = 80


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

ANIMATION_DIRECTOR_SYSTEM_PROMPT = """\
You are the Animation Director for a video generation system. Your job is to
read a content brief and a summary of the show's audio events, then produce
a complete visual plan that downstream tools will execute literally.

You will receive:
  1. BRIEF — the original content brief (may contain visual style cues).
  2. VISUAL_BRIEF — optional dedicated visual-override instructions
     (when present, treat as authoritative — overrides anything in BRIEF
     that conflicts).
  3. EVENTS — a compact summary of the show's audio events: who speaks
     when, what they say (truncated), and the structural beats.
  4. HARDWARE — the machine's capabilities. Choose a render backend it
     can actually run and match the SD model / resolution / fps to what
     the hardware can handle.
  5. AVAILABLE_BACKENDS — the list of render backends the system has
     code for. ONLY pick from this list.

You must produce a JSON object with EXACTLY this schema. Do NOT include any
text outside the JSON block. No markdown, no commentary, just raw JSON.

{
  "visual_instructions": "string — your interpretation of the visual brief in 1-3 sentences",
  "style": "string — one of: cinematic, documentary, kinetic_text, illustrated, talking_heads",
  "render_backend": "string — one of the AVAILABLE_BACKENDS values",
  "color_palette": ["#hex1", "#hex2", "#hex3"],
  "font_family": "string or null — font for text overlays (system font name; null for default)",
  "fps": integer (24 or 30),
  "resolution": [width_int, height_int],
  "image_model": "string or null — SD model identifier (null = use hardware default)",
  "scenes": [
    {
      "scene_id": "string — short unique id like 'scene_01'",
      "event_global_orders": [int, int, ...],
      "image_prompt": "string — Stable Diffusion prompt; rich, specific, evocative",
      "negative_prompt": "string or null — SD negative prompt (things to avoid)",
      "text_overlay": "string or null — caption/subtitle text on this scene",
      "speaker_label": "string or null — name card text (e.g. 'Dr. Vance — Economist')",
      "transition_in": "string — one of: cut, fade, crossfade, wipe",
      "transition_out": "string — one of: cut, fade, crossfade, wipe",
      "animation_hint": "string — one of: ken_burns_zoom_in, ken_burns_zoom_out, ken_burns_pan_left, ken_burns_pan_right, parallax, static, text_pulse, talking_head_sync"
    }
  ]
}

DIRECTING RULES:

- STYLE picks the overall look. Rough guide:
    cinematic    — narrative shows, debates, fiction; wide shots, atmosphere
    documentary  — news, factual debates; anchor cards, B-roll, lower thirds
    kinetic_text — text-heavy, podcast-style; typography drives the visuals
    illustrated  — fiction, audiobooks; painterly, illustrated frames
    talking_heads — podcasts, interviews; speaker portraits, name plates

- COLOR_PALETTE: 3-5 hex strings. Should match the style and the brief's mood.
  Noir → dark cool tones. Bright satire → saturated playful tones.
  Documentary → muted naturals.

- SCENES: group consecutive events that share visual context. A scene
  typically covers 1-4 events (one speaker turn, or a tight back-and-forth).
  Aim for 1 scene every 8-15 seconds of expected audio. A 5-minute show
  → roughly 20-30 scenes. A 20-minute show → roughly 60-90 scenes.

- IMAGE_PROMPT for each scene must be:
    * Specific (subject, setting, lighting, mood, framing)
    * Style-consistent across the whole plan
    * 15-40 words; avoid bullet-point style
    * ALWAYS include one COMPOSITION term — pick the one that serves the
      scene's emotional purpose:
          rule of thirds, leading lines, foreground interest,
          shallow depth of field, deep focus, wide establishing shot,
          low angle, high angle, overhead shot, extreme close-up,
          over-the-shoulder, symmetrical framing, negative space,
          Dutch angle (for tension), centered hero shot
    * ALWAYS include one LIGHTING quality term:
          golden hour, magic hour, overcast diffuse, hard side-light,
          soft three-point studio, backlit silhouette, rim light,
          neon fill, candlelight, harsh noon sun, blue-hour twilight,
          motivated window light, chiaroscuro
  Example (documentary): "modern television studio, anchor desk, rule of
  thirds framing, soft three-point studio lighting, professional camera
  angle, broadcast news set, blue and grey color scheme, 35mm lens"

- NEGATIVE_PROMPT: short list of what to exclude. Reuse the same negative
  prompt across scenes when possible. Default: "blurry, watermark, text,
  low quality, distorted, oversaturated, bad composition, poorly framed,
  cropped subject, cluttered background, flat lighting"

- TEXT_OVERLAY: use sparingly. Best for headlines, key facts, pull quotes.
  null for most scenes (audio carries the content). Maximum ~80 chars.

- SPEAKER_LABEL: include for the FIRST scene each new speaker appears in.
  Format: "Name — Role" (e.g. "Dr. Elena Vasquez — Economist"). null for
  subsequent scenes with the same speaker.

- TRANSITIONS: default to "cut". Use "fade" for opening/closing scenes.
  Use "crossfade" between major topic shifts. Use "wipe" sparingly for
  satirical/kinetic styles.

- ANIMATION_HINT: most scenes should be "static" or a Ken Burns variant
  (the slow zoom adds life to still images). Use "text_pulse" only when
  text_overlay is the focus. "parallax" is reserved for cinematic style.
  Use "talking_head_sync" ONLY when style == "talking_heads" AND the scene
  covers a single speaker's utterance — it tells the renderer to lip-sync
  the portrait to that scene's audio. Never use "talking_head_sync" for
  intro/outro/B-roll scenes or for any other style.

- TALKING_HEADS STYLE — when you choose this style:
    * IMAGE_PROMPT for utterance scenes must describe a head-and-shoulders
      portrait of the speaker, front-facing, neutral background, even
      lighting, eye-level camera, mouth visible (the lip-sync model needs
      a clear mouth region). 15-30 words is plenty.
      Example: "head and shoulders portrait of a 50-year-old male news
      anchor, dark suit, neutral grey backdrop, soft three-point
      lighting, front-facing, neutral expression, eye-level camera, 50mm"
    * Set ANIMATION_HINT to "talking_head_sync" for every spoken-event
      scene. Use "static" or a Ken Burns variant for intro/outro/bumper
      scenes that have no speaker.
    * Keep SPEAKER_LABEL consistent for the same speaker ("Name — Role")
      and reuse the SAME image_prompt subject description across that
      speaker's scenes so the portrait stays recognizably the same
      person across cuts.
    * Group consecutive lines from the same speaker into ONE scene when
      possible — it lowers the number of SadTalker invocations and
      yields smoother delivery.

- FPS: 24 for cinematic/documentary/illustrated. 30 for kinetic/podcast.

- RESOLUTION: take the HARDWARE.recommended_resolution unless the brief
  specifically calls for a different aspect ratio (vertical 1080x1920
  for social media, square 1080x1080 for clips).

- IMAGE_MODEL: leave as null unless the brief specifically calls for a
  particular SD model. The system will use the hardware-recommended one.

- RENDER_BACKEND: pick from AVAILABLE_BACKENDS. "moviepy" is the safest
  choice for any style. Only choose other backends if the brief
  explicitly calls for their unique capabilities.

- Every event in EVENTS must appear in exactly ONE scene's
  event_global_orders list. Don't skip events; don't double-count them.

ALWAYS respond with valid JSON. Nothing else.
"""


# ═══════════════════════════════════════════════════════════════════════════
# AnimationDirector
# ═══════════════════════════════════════════════════════════════════════════

class AnimationDirector:
    """Single-shot LLM agent that produces an AnimationPlan.

    Design mirrors DebateArchitect: dependency-injected client, ``run()``
    + ``parse_response()`` split for testability, no internal state.

    Attributes:
        _client: Injected DeepSeekClient. Production code passes the real
                 client; tests pass a mock.
    """

    def __init__(self, client: DeepSeekClient) -> None:
        """Initialize with an injected LLM client.

        Args:
            client: A DeepSeekClient (or any object exposing ``.chat()``).
        """
        self._client = client

    # ── Public entry point ────────────────────────────────────────────────

    def run(
        self,
        brief: str,
        events: Iterable[AudioEvent | dict[str, Any]],
        *,
        hardware_profile: HardwareProfile,
        visual_brief: str | None = None,
        available_backends: tuple[str, ...] = ("moviepy",),
    ) -> AnimationPlan:
        """Call the LLM once and return a populated AnimationPlan.

        Args:
            brief:
                The original content brief from run.py. The LLM reads
                style cues here naturally ("BBC documentary feel" etc.).

            events:
                The show's AudioEvents — either AudioEvent dataclass
                instances or already-deserialized dicts. The director
                summarises them; it does not need every word.

            hardware_profile:
                What the machine can do. Used to constrain backend,
                model, resolution, fps choices.

            visual_brief:
                Optional dedicated animation-instruction channel. Takes
                precedence over conflicting style cues in ``brief``.

            available_backends:
                Backends the system has code for. The LLM may only
                choose from this set. Default: ("moviepy",) — only the
                functional backend. Pass ("moviepy", "manim", "ffmpeg")
                later when those backends are implemented.

        Returns:
            A populated AnimationPlan ready for ImageGenerator + renderer.

        Raises:
            ValueError: If the LLM returns unparseable or invalid JSON.
        """
        events_summary = self._summarize_events(events)

        user_message = self._build_user_message(
            brief=brief,
            events_summary=events_summary,
            hardware_profile=hardware_profile,
            visual_brief=visual_brief,
            available_backends=available_backends,
        )

        logger.debug(
            "AnimationDirector: requesting plan for brief (%d chars), %d events",
            len(brief), len(events_summary),
        )

        raw_response = self._client.chat(
            messages=[
                {"role": "system", "content": ANIMATION_DIRECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        return self.parse_response(
            raw_response,
            brief=brief,
            hardware_profile=hardware_profile,
            available_backends=available_backends,
        )

    # ── Pure-function parsing (testable without LLM) ──────────────────────

    def parse_response(
        self,
        raw_json: str,
        *,
        brief: str,
        hardware_profile: HardwareProfile,
        available_backends: tuple[str, ...] = ("moviepy",),
    ) -> AnimationPlan:
        """Parse the LLM's JSON output into a validated AnimationPlan.

        Separated from ``run()`` so unit tests can feed canned JSON
        without hitting the LLM. Validates enum values; falls back to
        sensible defaults when fields are missing or invalid.

        Args:
            raw_json:
                The raw string returned by the LLM.
            brief:
                Original brief, stored on the plan for provenance.
            hardware_profile:
                The profile used for this plan (stored for reproducibility
                and used as a fallback for unset numeric fields).
            available_backends:
                Allowed backend choices. If the LLM picks one outside
                this set, we fall back to the first available backend.

        Returns:
            A populated, validated AnimationPlan.

        Raises:
            ValueError: If the JSON is malformed or has no ``scenes``.
        """
        cleaned = self._strip_markdown_fence(raw_json)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"AnimationDirector produced invalid JSON. "
                f"First 200 chars: {raw_json[:200]!r}\nParse error: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"AnimationDirector returned a {type(data).__name__}, "
                "expected a JSON object."
            )

        scenes_data = data.get("scenes")
        if not isinstance(scenes_data, list) or not scenes_data:
            raise ValueError(
                "AnimationDirector response missing required non-empty "
                "'scenes' array."
            )

        # ── Validated enum fields with fallbacks ──────────────────────
        style = self._validate_enum(
            data.get("style"), _VALID_STYLES, default="cinematic",
            field_name="style",
        )

        backend_choice = self._validate_enum(
            data.get("render_backend"), set(available_backends),
            default=available_backends[0], field_name="render_backend",
        )

        # ── Numeric fields with hardware-aware fallbacks ──────────────
        fps = self._coerce_int(
            data.get("fps"), default=hardware_profile.recommended_fps
        )
        resolution = self._coerce_resolution(
            data.get("resolution"),
            default=hardware_profile.recommended_resolution,
        )

        # ── Color palette + font ──────────────────────────────────────
        palette_data = data.get("color_palette")
        if isinstance(palette_data, list) and palette_data:
            color_palette = [str(c) for c in palette_data]
        else:
            color_palette = ["#1a1a2e", "#e94560", "#f5f5f5"]

        font_family = data.get("font_family")
        if font_family is not None and not isinstance(font_family, str):
            font_family = None

        # ── Image model: null means "use hardware default" ────────────
        image_model = data.get("image_model")
        if image_model is None or not isinstance(image_model, str):
            image_model = hardware_profile.recommended_sd_model

        # ── Scenes ────────────────────────────────────────────────────
        scenes = [
            self._parse_scene(scene_data, idx)
            for idx, scene_data in enumerate(scenes_data)
        ]

        return AnimationPlan(
            plan_id=str(uuid.uuid4()),
            brief=brief,
            visual_instructions=self._optional_str(data.get("visual_instructions")),
            style=style,  # type: ignore[arg-type]
            render_backend=backend_choice,  # type: ignore[arg-type]
            color_palette=color_palette,
            font_family=font_family,
            fps=fps,
            resolution=resolution,
            image_model=image_model,
            scenes=scenes,
            hardware_profile=hardware_profile,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── User-message construction ─────────────────────────────────────────

    def _build_user_message(
        self,
        *,
        brief: str,
        events_summary: str,
        hardware_profile: HardwareProfile,
        visual_brief: str | None,
        available_backends: tuple[str, ...],
    ) -> str:
        """Assemble the user message sent to the LLM.

        Sections are clearly labeled so the model can parse them. The
        visual_brief section appears immediately after BRIEF (and is
        marked authoritative) so its overrides take effect.
        """
        parts = [f"BRIEF:\n{brief}"]

        if visual_brief and visual_brief.strip():
            parts.append(
                f"\nVISUAL_BRIEF (authoritative — overrides BRIEF on conflicts):\n"
                f"{visual_brief.strip()}"
            )

        parts.append(f"\nEVENTS:\n{events_summary}")

        hw_summary = (
            f"device: {hardware_profile.device}, "
            f"vram_gb: {hardware_profile.vram_gb or 'unknown'}, "
            f"recommended_sd_model: {hardware_profile.recommended_sd_model}, "
            f"recommended_resolution: {list(hardware_profile.recommended_resolution)}, "
            f"recommended_fps: {hardware_profile.recommended_fps}"
        )
        parts.append(f"\nHARDWARE:\n{hw_summary}")

        parts.append(
            f"\nAVAILABLE_BACKENDS:\n{list(available_backends)}"
        )

        return "\n".join(parts)

    # ── Events → compact summary for the LLM ──────────────────────────────

    @staticmethod
    def _summarize_events(
        events: Iterable[AudioEvent | dict[str, Any]],
    ) -> str:
        """Reduce events to a compact, line-per-event summary.

        Each line includes the global_order, speaker, role, event_kind,
        and a truncated text preview. If there are more than
        _MAX_EVENTS_IN_PROMPT events, we sample evenly to keep the
        prompt reasonable while still showing the show's overall shape.
        """
        normalized: list[dict[str, Any]] = []
        for ev in events:
            if isinstance(ev, AudioEvent):
                normalized.append(ev.to_dict())
            elif isinstance(ev, dict):
                normalized.append(ev)
            else:
                # Skip unrecognized event objects rather than crash.
                logger.debug(
                    "AnimationDirector: skipping non-event item of type %s",
                    type(ev).__name__,
                )

        if not normalized:
            return "(no events provided)"

        # Sample down if too many events
        if len(normalized) > _MAX_EVENTS_IN_PROMPT:
            stride = len(normalized) / _MAX_EVENTS_IN_PROMPT
            sampled = [
                normalized[int(i * stride)]
                for i in range(_MAX_EVENTS_IN_PROMPT)
            ]
            # Always include the last event
            if sampled[-1] is not normalized[-1]:
                sampled.append(normalized[-1])
            normalized = sampled

        lines: list[str] = []
        for ev in normalized:
            order = ev.get("global_order", "?")
            speaker = ev.get("speaker_name") or "(unknown)"
            role = ev.get("speaker_role") or ""
            kind = ev.get("event_kind") or "utterance"
            text = ev.get("text") or ""
            if len(text) > _EVENT_TEXT_PREVIEW_CHARS:
                text = text[:_EVENT_TEXT_PREVIEW_CHARS].rstrip() + "…"
            text = text.replace("\n", " ").strip()
            role_part = f"/{role}" if role else ""
            lines.append(
                f"[{order:>3}] ({kind}) {speaker}{role_part}: {text}"
            )

        return "\n".join(lines)

    # ── Per-scene parsing ─────────────────────────────────────────────────

    @classmethod
    def _parse_scene(cls, data: Any, idx: int) -> SceneSpec:
        """Build a SceneSpec from one entry in the LLM's scenes array.

        Falls back to defaults for missing fields so a partial response
        still produces a renderable scene rather than crashing the run.
        """
        if not isinstance(data, dict):
            logger.warning(
                "AnimationDirector: scene %d is not a dict; using empty SceneSpec",
                idx,
            )
            return SceneSpec(scene_id=f"scene_{idx:02d}")

        # Event orders — keep only ints
        orders_raw = data.get("event_global_orders", [])
        if not isinstance(orders_raw, list):
            orders_raw = []
        event_orders = [
            int(o) for o in orders_raw if isinstance(o, (int, float))
        ]

        # Enum fields with fallbacks
        transition_in = cls._validate_enum(
            data.get("transition_in"), _VALID_TRANSITIONS,
            default="cut", field_name=f"scene[{idx}].transition_in",
        )
        transition_out = cls._validate_enum(
            data.get("transition_out"), _VALID_TRANSITIONS,
            default="cut", field_name=f"scene[{idx}].transition_out",
        )
        animation_hint = cls._validate_enum(
            data.get("animation_hint"), _VALID_HINTS,
            default="static", field_name=f"scene[{idx}].animation_hint",
        )

        return SceneSpec(
            scene_id=str(data.get("scene_id") or f"scene_{idx:02d}"),
            event_global_orders=event_orders,
            image_prompt=str(data.get("image_prompt", "")),
            negative_prompt=cls._optional_str(data.get("negative_prompt")),
            text_overlay=cls._optional_str(data.get("text_overlay")),
            speaker_label=cls._optional_str(data.get("speaker_label")),
            transition_in=transition_in,  # type: ignore[arg-type]
            transition_out=transition_out,  # type: ignore[arg-type]
            animation_hint=animation_hint,  # type: ignore[arg-type]
        )

    # ── Validation helpers ────────────────────────────────────────────────

    @staticmethod
    def _validate_enum(
        value: Any,
        allowed: set[str],
        *,
        default: str,
        field_name: str,
    ) -> str:
        """Return ``value`` if it's in ``allowed``; otherwise ``default``.

        Logs a warning when the LLM picks an invalid value so we can
        spot prompt drift.
        """
        if isinstance(value, str) and value in allowed:
            return value
        if value is not None:
            logger.warning(
                "AnimationDirector: invalid %s=%r (allowed=%s); using %r",
                field_name, value, sorted(allowed), default,
            )
        return default

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        """Coerce ``value`` to int, falling back to ``default``."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_resolution(
        value: Any,
        *,
        default: tuple[int, int],
    ) -> tuple[int, int]:
        """Coerce a list/tuple ``[w, h]`` into a (width, height) tuple."""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return (int(value[0]), int(value[1]))
            except (TypeError, ValueError):
                pass
        return default

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        """Return value as str if non-empty; else None.

        Treats null, empty strings, and non-string values as None so
        downstream code doesn't have to special-case them.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped if stripped else None

    @staticmethod
    def _strip_markdown_fence(raw: str) -> str:
        """Remove ``` ``` fences from an LLM response if present.

        Some models wrap JSON in ```json ... ``` even when told not to.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Drop first line (```json or ```) and last line (```)
            if len(lines) >= 2:
                cleaned = "\n".join(lines[1:-1]).strip()
        return cleaned
