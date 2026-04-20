"""The Audio Mixer — stitches TTS clips into a finished audio file.

If the TTS engine is the vocalist and the renderer is the director, the
mixer is the sound engineer in the booth. It takes individual audio clips
(one per debate turn) and assembles them into a polished final track.

THE RECORDING STUDIO ANALOGY:

    Imagine you've recorded each speaker's lines separately in an
    isolation booth. Now the sound engineer has a stack of labeled tapes:

        [Dr. Chen - Turn 1]  [Dr. Vance - Turn 2]  [Moderator - Turn 3]  ...

    The engineer's job:
    1. Play them in order (sequencing)
    2. Add silence between turns (natural pauses)
    3. Handle interruptions (overlap, cut-off)
    4. Normalize volume (nobody should be 3x louder)
    5. Bounce to a final file (mp3, wav, flac)

    That's exactly what AudioMixer does. It uses pydub under the hood,
    which uses your ffmpeg installation for the actual audio processing.

PAUSE HANDLING:

    Debates have natural pauses — between turns, before dramatic moments,
    after interruptions. The mixer reads pause hints from DebateEvents:

        pause_before: "short" | "medium" | "long" | None
        pause_after:  "short" | "medium" | "long" | None

    These get mapped to milliseconds:
        "short"  → 300ms  (quick breath, conversational)
        "medium" → 700ms  (thoughtful pause, topic shift)
        "long"   → 1200ms (dramatic beat, section break)
        None     → 400ms  (default gap between turns)

    TIP FOR HUMANS: These durations are tunable. If debates sound too
    rushed or too draggy, tweak DEFAULT_PAUSE_MS and the PAUSE_MAP.

INTERRUPTIONS:

    When interruption=True on a DebateEvent, the mixer overlaps the
    speaker's audio with the tail end of the previous clip. This creates
    that "talking over each other" effect. The overlap amount is
    configurable (default: 500ms).

    For overlap_with_previous=True, same idea but less aggressive —
    a shorter overlap that sounds like finishing each other's sentences.

DEPENDENCIES:
    - pydub (pip install pydub) — the audio manipulation library
    - ffmpeg (brew install ffmpeg) — the backend pydub relies on
    Both are already installed on this system.

EXAMPLE:
    from vox_romana.tts.mixer import AudioMixer

    mixer = AudioMixer()
    mixer.add_clip(audio_bytes_1, sample_rate=22050, pause_before=300)
    mixer.add_clip(audio_bytes_2, sample_rate=22050, pause_before=0, overlap=500)
    mixer.export("output/debate.mp3", format="mp3")
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from pydub import AudioSegment


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Default silence between turns when no pause hint is given.
# 400ms feels like a natural conversational gap — not too rushed,
# not awkwardly long. Adjust to taste.
DEFAULT_PAUSE_MS = 400

# Maps the string pause hints from DebateEvent to milliseconds.
# These are the "short/medium/long" labels from the conductor.
PAUSE_MAP: dict[str | None, int] = {
    "short": 300,     # Quick breath. "And another thing—"
    "medium": 700,    # Thoughtful beat. "Hmm. Well..."
    "long": 1200,     # Dramatic pause. Section break. Mic drop.
    "none": 0,        # No pause at all. Rapid-fire exchange.
    None: DEFAULT_PAUSE_MS,  # No hint given → use the default.
}

# How much audio to overlap when someone interrupts (milliseconds).
# 500ms is enough to hear the overlap without making it unintelligible.
DEFAULT_OVERLAP_MS = 500

# Shorter overlap for "finishing each other's sentences" (overlap_with_previous).
DEFAULT_SOFT_OVERLAP_MS = 200

# Target loudness for normalization (dBFS).
# -20 dBFS is a good podcast/broadcast level — loud enough to hear,
# quiet enough that peaks don't clip.
TARGET_DBFS = -20.0


# ═══════════════════════════════════════════════════════════════════════════
# CLIP CONTAINER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AudioClip:
    """One audio clip in the mix timeline.

    Each clip represents a single debate turn's audio, plus metadata
    about how it should be positioned relative to its neighbors.

    Attributes:
        segment:      The pydub AudioSegment (the actual audio data).
        pause_before: Silence to insert BEFORE this clip (ms).
        overlap:      How much to overlap with the PREVIOUS clip (ms).
                      0 = no overlap. Positive = interrupt effect.
        speaker:      Who said this (for logging/debugging).
    """

    segment: AudioSegment
    pause_before: int = DEFAULT_PAUSE_MS
    overlap: int = 0
    speaker: str = ""


class AudioMixer:
    """Assembles individual TTS clips into a finished audio file.

    Usage:
        mixer = AudioMixer()

        # Add clips in order (the mixer sequences them left-to-right).
        mixer.add_clip(wav_bytes_1, sample_rate=22050, pause_before=300)
        mixer.add_clip(wav_bytes_2, sample_rate=22050, overlap=500)

        # Export the final mix.
        mixer.export("output/debate.mp3", format="mp3")

    The mixer handles:
        - Sequencing clips in order
        - Adding silence (pauses) between them
        - Overlapping for interruptions
        - Normalizing volume to a consistent level
        - Exporting in any format ffmpeg supports
    """

    def __init__(self) -> None:
        """Initialize an empty mixer.

        Clips are added one at a time via add_clip(). The mix is built
        lazily when you call mix() or export().
        """
        self._clips: list[AudioClip] = []

    def add_clip(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int = 22050,
        pause_before: int = DEFAULT_PAUSE_MS,
        overlap: int = 0,
        speaker: str = "",
    ) -> None:
        """Add an audio clip to the mix timeline.

        Clips are played in the order they're added — like laying down
        tracks on a multitrack tape, left to right.

        Args:
            audio_bytes:  Raw WAV audio bytes (from SynthesizedAudio.audio).
            sample_rate:  Sample rate of the audio (must match the WAV header).
            pause_before: Milliseconds of silence before this clip.
                          Use 0 for no pause. Use PAUSE_MAP values for
                          standard debate pauses.
            overlap:      Milliseconds to overlap with the previous clip.
                          Used for interruptions. 0 = no overlap.
                          If both pause_before and overlap are set,
                          overlap wins (can't have silence AND overlap).
            speaker:      Speaker name for logging. Optional.
        """
        # Convert raw WAV bytes to a pydub AudioSegment.
        # pydub can ingest WAV from a file-like object.
        segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))

        self._clips.append(AudioClip(
            segment=segment,
            pause_before=pause_before,
            overlap=overlap,
            speaker=speaker,
        ))

    def add_silence(self, duration_ms: int) -> None:
        """Add a block of pure silence (e.g. for section breaks).

        Args:
            duration_ms: How long the silence should be, in milliseconds.
        """
        silence = AudioSegment.silent(duration=duration_ms)
        self._clips.append(AudioClip(
            segment=silence,
            pause_before=0,
            overlap=0,
            speaker="[silence]",
        ))

    def mix(self, *, normalize: bool = True) -> AudioSegment:
        """Assemble all clips into a single AudioSegment.

        This is where the magic happens. The mixer walks through every
        clip and either:
            - Appends it after a silence gap (normal turn), OR
            - Overlays it on top of the previous clip's tail (interruption)

        Args:
            normalize: If True, normalize the final mix to TARGET_DBFS.
                       This ensures consistent volume across the whole
                       debate. Highly recommended — without it, some
                       speakers might be much louder than others.

        Returns:
            A single AudioSegment containing the complete mixed audio.
            You can export this directly or do further processing.
        """
        if not self._clips:
            # No clips → return 1 second of silence.
            # (Better than crashing. The caller can check duration.)
            return AudioSegment.silent(duration=1000)

        # Start with the first clip (no pause before the very first speaker).
        mixed = self._clips[0].segment

        # Walk through remaining clips and stitch them together.
        for clip in self._clips[1:]:
            if clip.overlap > 0:
                # ── INTERRUPTION / OVERLAP ────────────────────────────
                # The current speaker starts talking BEFORE the previous
                # speaker finishes. We overlay the new clip on the tail
                # of the existing mix.
                #
                # Visual:
                #   mixed:  [=========AAAAAAA]
                #   clip:                [BBBBBBBBB]
                #   result: [=========AAABBBBBBBB]
                #                     ^^^^ overlap zone
                #
                # How pydub overlay works: it mixes two signals at a
                # given position. We position the new clip so it starts
                # `overlap` ms before the end of the current mix.
                overlap_ms = min(clip.overlap, len(mixed))
                position = len(mixed) - overlap_ms
                mixed = mixed.overlay(clip.segment, position=position)

                # Extend the mix if the new clip runs longer than the
                # remaining tail of the old mix.
                new_end = position + len(clip.segment)
                if new_end > len(mixed):
                    extra = clip.segment[len(mixed) - position:]
                    mixed = mixed + extra
            else:
                # ── NORMAL SEQUENCING ────────────────────────────────
                # Add silence, then the clip. Like a pause between
                # sentences in a conversation.
                if clip.pause_before > 0:
                    mixed += AudioSegment.silent(duration=clip.pause_before)
                mixed += clip.segment

        # ── NORMALIZE ────────────────────────────────────────────────
        # Make the whole debate a consistent volume. Without this,
        # some speakers might be whisper-quiet while others blow out
        # your headphones.
        if normalize and len(mixed) > 0:
            change_in_dbfs = TARGET_DBFS - mixed.dBFS
            mixed = mixed.apply_gain(change_in_dbfs)

        return mixed

    def export(
        self,
        path: str | Path,
        *,
        format: str = "mp3",
        normalize: bool = True,
        bitrate: str = "192k",
    ) -> Path:
        """Mix all clips and export to a file.

        This is the most common entry point — mix everything and save it.

        Args:
            path:      Output file path. Directory will be created if needed.
            format:    Audio format ("mp3", "wav", "flac", "ogg").
                       Anything ffmpeg supports.
            normalize: Normalize volume to TARGET_DBFS before export.
            bitrate:   Bitrate for lossy formats like mp3 (default: "192k").
                       Ignored for lossless formats.

        Returns:
            The Path to the exported file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mixed = self.mix(normalize=normalize)

        export_kwargs: dict = {"format": format}
        if format in ("mp3", "ogg", "aac"):
            export_kwargs["bitrate"] = bitrate

        mixed.export(str(output_path), **export_kwargs)
        return output_path

    @property
    def clip_count(self) -> int:
        """How many clips are in the mix timeline."""
        return len(self._clips)

    @property
    def duration_ms(self) -> int:
        """Estimated total duration in milliseconds.

        This is an estimate because it doesn't account for overlaps.
        For the exact duration, call mix() and check len(result).
        """
        total = sum(len(c.segment) + c.pause_before for c in self._clips)
        return total

    def resolve_pause(self, pause_hint: str | None) -> int:
        """Convert a DebateEvent pause hint to milliseconds.

        The conductor writes pause hints as strings ("short", "medium",
        "long"). This method maps them to actual durations.

        Args:
            pause_hint: "short", "medium", "long", "none", or None.

        Returns:
            Duration in milliseconds.

        TIP: If you want custom pause durations, modify PAUSE_MAP at the
             module level or subclass AudioMixer and override this method.
        """
        # Normalize the hint (lowercase, strip whitespace).
        if isinstance(pause_hint, str):
            pause_hint = pause_hint.strip().lower()
            if pause_hint == "":
                pause_hint = None

        return PAUSE_MAP.get(pause_hint, DEFAULT_PAUSE_MS)
