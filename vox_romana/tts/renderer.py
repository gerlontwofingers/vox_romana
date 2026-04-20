"""The TTS Renderer — turns structured content JSON into audio.

The renderer reads a content script (debate JSON, news show JSON, etc.),
calls each speaker into the booth (TTS engine), collects their recordings,
and hands the tapes to the sound engineer (AudioMixer) for final assembly.

THE MOVIE DIRECTOR ANALOGY:

    The director doesn't operate the camera, doesn't act, and doesn't
    edit the film. The director COORDINATES:

    1. Reads the script     → loads the MergedDebate JSON
    2. Casts the actors     → maps speakers to TTS voices
    3. Directs each scene   → feeds each event to the TTS engine
    4. Hands off to editing → gives audio clips to the mixer
    5. Delivers the film    → exports the final audio file

    The director doesn't care if the camera is a RED or an ARRI (Piper
    or Kokoro). The director just says "shoot this scene" and the crew
    handles the rest.

VOICE ASSIGNMENT:

    The renderer needs to decide which TTS voice plays each speaker.
    There are three modes:

    CATALOG-MATCHED (when voice_hints are present):
        If the show JSON includes a "voice_hints" dict mapping speaker
        names to natural-language casting notes (e.g. "warm British
        baritone, late 40s, professorial"), the renderer uses the voice
        catalog to find the best-matching voice for each speaker based
        on tag overlap scoring. This ensures Frank doesn't sound like a
        teenage girl. Voices are assigned uniquely — no two speakers
        share a voice. Gender info from "voice_casting" is also used.

    AUTOMATIC + VOICE CASTING (fallback when no hints):
        If the show JSON includes a "voice_casting" dict mapping speaker
        names to genders ("male"/"female"), the renderer picks from the
        appropriate voice pool. Voices are grouped by gender prefix:
            af_*/bf_* = female, am_*/bm_* = male
        Speakers without casting info default to male voices.

    MANUAL:
        You pass a voice_map dict:
            {"Dr. Maya Chen": VoiceConfig(voice_id="af_heart"),
             "Dr. Robert Vance": VoiceConfig(voice_id="am_adam")}

        This gives you full control over who sounds like what.

TTS STYLE:

    If the show JSON includes a "tts_styles" dict mapping speaker names
    to persistent style instructions (e.g. "Speak with quiet authority
    and measured pacing"), these are prepended to every emotion hint for
    that speaker. This gives each character a consistent vocal baseline
    that persists across all their utterances.

EMOTION TRANSLATION:

    DebateEvents carry rich emotional metadata (tone, pace, volume,
    nonverbal cues). The renderer builds an "emotion hint" string from
    these fields and passes it to the engine:

        tone=["aggressive", "rising"], pace="fast", volume="loud"
        → emotion hint: "aggressive, rising; pace: fast; volume: loud"

    What the engine does with this is engine-specific (see engine.py).

THE PIPELINE:

    MergedDebate
        ↓
    [Event 1] → engine.synthesize() → SynthesizedAudio → mixer.add_clip()
    [Event 2] → engine.synthesize() → SynthesizedAudio → mixer.add_clip()
    [Event 3] → engine.synthesize() → SynthesizedAudio → mixer.add_clip()
    ...
        ↓
    mixer.export("debate.mp3")

EXAMPLE:
    from vox_romana.tts.renderer import TTSRenderer
    from vox_romana.tts.engines.piper import PiperEngine

    engine = PiperEngine(model_path="path/to/en_US-lessac-medium.onnx")
    renderer = TTSRenderer(engine)

    # From a JSON file:
    renderer.render_file("output/ubi_debate_test_01.json", "output/ubi_debate.mp3")

    # From a MergedDebate object:
    renderer.render(merged_debate, "output/ubi_debate.mp3")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import BaseTTSEngine, SynthesizedAudio, VoiceConfig
from .mixer import AudioMixer, DEFAULT_OVERLAP_MS, DEFAULT_SOFT_OVERLAP_MS


# ═══════════════════════════════════════════════════════════════════════════
# STAGE DIRECTION STRIPPING
# ═══════════════════════════════════════════════════════════════════════════

# Matches any (parenthetical) block in utterance text, e.g.:
#   (laughs)  (dramatic pause)  (Leaning in, voice low)
# These are LLM-generated performance directions. Standard TTS engines
# have no way to act on them, so they end up spoken literally.
_STAGE_DIR_RE = re.compile(r'\([^)]*\)')


def strip_stage_directions(text: str) -> str:
    """Remove parenthetical stage directions from TTS text.

    Strips all (parenthetical) blocks regardless of their position in the
    string, then collapses any resulting runs of whitespace.  Ellipsis
    clusters around the removed block are preserved.

    Examples::

        >>> strip_stage_directions("(laughs) Well, that's funny.")
        "Well, that's funny."
        >>> strip_stage_directions("Let me tell you... (dramatic pause) Gold.")
        'Let me tell you... Gold.'
        >>> strip_stage_directions("Just plain dialogue.")
        'Just plain dialogue.'
    """
    cleaned = _STAGE_DIR_RE.sub('', text)
    # Collapse multiple whitespace (including from removed leading direction)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    # Tidy up triple-dot + space created when a mid-sentence direction is removed
    cleaned = re.sub(r'\.\.\.\s+', '... ', cleaned)
    return cleaned.strip()


# ═══════════════════════════════════════════════════════════════════════════
# RENDER RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RenderResult:
    """Result of rendering a debate to audio.

    Contains both the output path and stats about the rendering process.
    Useful for logging, debugging, and knowing how long things took.

    Attributes:
        output_path:   Where the audio file was saved.
        total_events:  How many debate events were in the script.
        rendered:      How many events actually produced audio.
                       (Some events like stage directions have no text.)
        skipped:       How many events were skipped (no text, or non-utterance).
        duration_ms:   Total audio duration in milliseconds.
        voice_map:     The speaker→voice mapping that was used.
    """

    output_path: Path
    total_events: int = 0
    rendered: int = 0
    skipped: int = 0
    duration_ms: int = 0
    voice_map: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.voice_map is None:
            self.voice_map = {}


# Event kinds that contain spoken text and should be synthesized.
# Other kinds (break, bumper, transition) are non-spoken markers
# handled by the mixer or ignored.
SPOKEN_EVENT_KINDS = frozenset({"utterance", "narration", "interrupt"})


class TTSRenderer:
    """Renders audio content (debates, news shows, etc.) from event dicts.

    The renderer is the glue between structured content data, the TTS engine,
    and the audio mixer. It reads events, synthesizes speech, and
    assembles the final audio.

    Attributes:
        _engine:    The TTS engine to use for synthesis.
        _mixer:     Will be created fresh for each render.
        _voice_map: Speaker name → VoiceConfig mapping.
    """

    def __init__(
        self,
        engine: BaseTTSEngine,
        *,
        voice_map: dict[str, VoiceConfig] | None = None,
        strip_directions: bool = False,
    ) -> None:
        """Initialize the renderer.

        Args:
            engine:           Any BaseTTSEngine implementation (Piper, Kokoro, etc.).
            voice_map:        Optional manual speaker→voice mapping. If not provided,
                              voices are auto-assigned from the engine's available
                              voices when render() is called.
            strip_directions: If True, strip parenthetical stage directions
                              (e.g. "(laughs)", "(dramatic pause)") from each
                              utterance's text before synthesis.  The original
                              JSON is never modified — only the text sent to the
                              TTS engine is cleaned.

        TIP: For quick testing, skip voice_map entirely. The renderer
             will assign voices automatically. For production, craft a
             voice_map that matches each character's personality.
        """
        self._engine = engine
        self._voice_map = voice_map or {}
        self._strip_directions = strip_directions

    def render(
        self,
        debate_data: dict[str, Any],
        output_path: str | Path,
        *,
        format: str = "mp3",
        normalize: bool = True,
        quiet: bool = False,
    ) -> RenderResult:
        """Render a debate dict to an audio file.

        This is the main entry point. Pass the debate data (parsed from
        JSON) and an output path in, get an audio file out.

        Args:
            debate_data:  A dict with an "events" key (list of event dicts).
                          Optionally has "participants", "title", and other
                          metadata. Works with any show type's JSON output.
            output_path:  Where to save the audio file.
            format:       Audio format ("mp3", "wav", "flac").
            normalize:    Normalize volume across all speakers.
            quiet:        Suppress progress output.

        Returns:
            RenderResult with stats about the rendering.
        """
        output_path = Path(output_path)
        events = debate_data.get("events", [])
        participants = debate_data.get("participants", [])

        # ── STEP 1: Assign voices to speakers ────────────────────────
        # Build the voice map (auto-assign if not manually provided).
        # We need to know every speaker name that appears in events.
        speaker_names = self._extract_speakers(events)
        voice_casting = debate_data.get("voice_casting", {})
        voice_hints = debate_data.get("voice_hints", {})
        tts_styles = debate_data.get("tts_styles", {})
        voice_map = self._build_voice_map(speaker_names, voice_casting, voice_hints)

        if not quiet:
            print(f"  Voice assignments:")
            for speaker, voice_id in voice_map.items():
                print(f"    {speaker} → {voice_id}")

        # ── STEP 2: Render each event through the TTS engine ────────
        mixer = AudioMixer()
        rendered = 0
        skipped = 0

        for i, event in enumerate(events):
            # Only render events that have spoken text.
            # Non-spoken kinds (break, bumper, transition) are markers
            # for the mixer or downstream processing, not TTS.
            text = event.get("text", "")
            event_kind = event.get("event_kind", "utterance")

            # Strip parenthetical stage directions before synthesis if enabled.
            # We do this before the empty-text guard so that an utterance that
            # consists *only* of a direction (e.g. "(pause)") is also skipped.
            if self._strip_directions:
                text = strip_stage_directions(text)

            if not text or not text.strip() or event_kind not in SPOKEN_EVENT_KINDS:
                skipped += 1
                continue

            speaker = event.get("speaker_name", "Unknown")
            voice_id = voice_map.get(speaker, self._default_voice())

            # ── Build the emotion hint from event metadata ────────
            # We combine tone, pace, volume, and nonverbal cues into
            # a single string that the engine can interpret.
            # If a persistent tts_style is set for this speaker, it
            # is prepended as a baseline delivery instruction.
            emotion = self._build_emotion_hint(event, tts_styles.get(speaker))

            # ── Determine speed from pace metadata ────────────────
            speed = self._pace_to_speed(event.get("pace"))

            # ── Determine pause and overlap ───────────────────────
            pause_before = mixer.resolve_pause(event.get("pause_before"))
            overlap = 0
            if event.get("interruption"):
                overlap = DEFAULT_OVERLAP_MS
                pause_before = 0  # No silence before an interruption.
            elif event.get("overlap_with_previous"):
                overlap = DEFAULT_SOFT_OVERLAP_MS
                pause_before = 0

            # Get engine-specific extras from the voice config.
            voice_cfg = self._voice_map.get(speaker, VoiceConfig(voice_id=voice_id))
            extras = voice_cfg.extras

            # ── Synthesize! ───────────────────────────────────────
            if not quiet:
                truncated = text[:60] + "..." if len(text) > 60 else text
                label = f"[{i+1}/{len(events)}]"
                print(f"  {label} {speaker}: {truncated}")

            try:
                result = self._engine.synthesize(
                    text,
                    voice=voice_id,
                    emotion=emotion,
                    speed=speed,
                    **extras,
                )

                mixer.add_clip(
                    result.audio,
                    sample_rate=result.sample_rate,
                    pause_before=pause_before,
                    overlap=overlap,
                    speaker=speaker,
                )
                rendered += 1

            except Exception as e:
                # If one clip fails, skip it rather than crashing the
                # whole render. Log the error and keep going.
                # The final audio will have a gap where the failed clip
                # would have been.
                if not quiet:
                    print(f"    WARNING: Failed to synthesize: {e}")
                skipped += 1

        # ── STEP 3: Mix and export ───────────────────────────────────
        if not quiet:
            print(f"\n  Mixing {rendered} clips...")

        final_path = mixer.export(output_path, format=format, normalize=normalize)
        duration_ms = mixer.duration_ms

        return RenderResult(
            output_path=final_path,
            total_events=len(events),
            rendered=rendered,
            skipped=skipped,
            duration_ms=duration_ms,
            voice_map=voice_map,
        )

    def render_file(
        self,
        json_path: str | Path,
        output_path: str | Path,
        *,
        format: str = "mp3",
        normalize: bool = True,
        quiet: bool = False,
    ) -> RenderResult:
        """Convenience: load a debate JSON file and render it.

        This is the "I just want to turn a JSON into an MP3" method.

        Args:
            json_path:   Path to the debate JSON file.
            output_path: Where to save the audio.
            format:      Audio format.
            normalize:   Normalize volume.
            quiet:       Suppress progress output.

        Returns:
            RenderResult with stats.
        """
        json_path = Path(json_path)
        with open(json_path) as f:
            debate_data = json.load(f)

        if not quiet:
            title = debate_data.get("title", json_path.stem)
            print(f"\n  Rendering: {title}")
            print(f"  Source: {json_path}")
            print()

        return self.render(
            debate_data,
            output_path,
            format=format,
            normalize=normalize,
            quiet=quiet,
        )

    # ═══════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _extract_speakers(self, events: list[dict]) -> list[str]:
        """Get unique speaker names from events, in order of first appearance.

        We preserve order so voice assignment is deterministic — the first
        speaker always gets the first voice, etc.
        """
        seen: set[str] = set()
        speakers: list[str] = []
        for event in events:
            name = event.get("speaker_name", "")
            if name and name not in seen:
                seen.add(name)
                speakers.append(name)
        return speakers

    def _build_voice_map(
        self,
        speaker_names: list[str],
        voice_casting: dict[str, str] | None = None,
        voice_hints: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build a speaker→voice_id mapping.

        Priority order:
        1. Manual voice_map (passed to TTSRenderer constructor).
        2. Catalog-matched voice using voice_hints metadata. If any
           speaker has a voice_hint, the catalog's ``match_voices_for_cast``
           assigns distinct voices based on hint tag overlap.
        3. Gender-aware assignment using voice_casting metadata from the
           show JSON (maps speaker names to "male"/"female").
        4. Round-robin through available voices (fallback).

        Args:
            speaker_names:  Speaker names in order of first appearance.
            voice_casting:  Optional dict mapping speaker names to gender
                            strings ("male" or "female") from the show JSON.
            voice_hints:    Optional dict mapping speaker names to natural-
                            language casting notes for catalog matching.

        Returns:
            Dict mapping every speaker name to a voice_id string.
        """
        available = self._engine.available_voices()
        casting = voice_casting or {}
        hints = voice_hints or {}
        result: dict[str, str] = {}

        # ── Manual overrides first ────────────────────────────────────
        for name in speaker_names:
            if name in self._voice_map:
                result[name] = self._voice_map[name].voice_id

        # ── Catalog matching (when any speaker has a voice_hint) ──────
        # Build a cast dict for unassigned speakers that have hints.
        remaining = [n for n in speaker_names if n not in result]
        if remaining and any(n in hints for n in remaining):
            from .voice_catalog import match_voices_for_cast

            cast_info: dict[str, dict] = {}
            for name in remaining:
                info: dict[str, str] = {}
                if name in casting:
                    info["gender"] = casting[name]
                if name in hints:
                    info["voice_hint"] = hints[name]
                cast_info[name] = info

            # Catalog assigns unique voices for the entire cast.
            catalog_map = match_voices_for_cast(cast_info)
            for name, voice_id in catalog_map.items():
                if voice_id and name not in result:
                    result[name] = voice_id

        # ── Gender-only fallback for any still-unassigned speakers ────
        # Partition voices by gender using the Kokoro naming convention:
        #   af_* / bf_* = female voices
        #   am_* / bm_* = male voices
        male_voices = [v for v in available if v[:2] in ("am", "bm")]
        female_voices = [v for v in available if v[:2] in ("af", "bf")]
        male_idx = 0
        female_idx = 0
        fallback_idx = 0

        for name in speaker_names:
            if name in result:
                continue
            if name in casting and casting[name] == "female" and female_voices:
                result[name] = female_voices[female_idx % len(female_voices)]
                female_idx += 1
            elif name in casting and casting[name] == "male" and male_voices:
                result[name] = male_voices[male_idx % len(male_voices)]
                male_idx += 1
            elif male_voices:
                # No gender info — default to male (safer than round-robin
                # through all voices which starts with female).
                result[name] = male_voices[male_idx % len(male_voices)]
                male_idx += 1
            elif available:
                # No gendered voices found — raw round-robin fallback.
                result[name] = available[fallback_idx % len(available)]
                fallback_idx += 1
            else:
                result[name] = "default"

        return result

    def _default_voice(self) -> str:
        """Get the first available voice, or "default" if none."""
        voices = self._engine.available_voices()
        return voices[0] if voices else "default"

    @staticmethod
    def _build_emotion_hint(event: dict, tts_style: str | None = None) -> str:
        """Translate event metadata into a natural-language emotion hint.

        If a persistent ``tts_style`` is provided for the speaker, it is
        prepended as a baseline delivery instruction. Per-event tone, pace,
        volume, and nonverbal cues are appended after it.

        The engine receives this string and interprets it however it can.
        Kokoro and OpenAI TTS understand natural language instructions.
        Piper mostly ignores it. That's fine.

        Args:
            event:     A show event dict.
            tts_style: Optional persistent style prefix for this speaker.

        Returns:
            A human-readable emotion hint string, e.g.:
            "Speak with quiet authority. aggressive, rising; pace: fast"
        """
        parts: list[str] = []

        # Persistent style prefix — the character's baseline vocal personality.
        if tts_style:
            parts.append(tts_style.rstrip(". ") + ".")

        # Add tone descriptors (e.g. "aggressive", "contemplative").
        tone = event.get("tone", [])
        if tone:
            parts.append(", ".join(tone))

        # Add pace hint.
        pace = event.get("pace")
        if pace:
            parts.append(f"pace: {pace}")

        # Add volume hint.
        volume = event.get("volume")
        if volume:
            parts.append(f"volume: {volume}")

        # Add nonverbal cues (e.g. "leans forward", "sighs").
        nonverbal = event.get("nonverbal", [])
        if nonverbal:
            parts.append("; ".join(nonverbal))

        return "; ".join(parts)

    @staticmethod
    def _pace_to_speed(pace: str | None) -> float:
        """Convert a pace hint to a speed multiplier.

        The conductor writes pace as "slow", "medium", "fast". We need
        a float multiplier for the TTS engine.

        Args:
            pace: "slow", "medium", "fast", or None.

        Returns:
            Speed multiplier (0.85 to 1.15).
            We keep the range tight — extreme speeds sound robotic.
        """
        pace_map = {
            "slow": 0.85,
            "medium": 1.0,
            "fast": 1.15,
        }
        if pace:
            return pace_map.get(pace.lower(), 1.0)
        return 1.0
