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

import io
import json
import logging
import re
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import BaseTTSEngine, SynthesizedAudio, VoiceConfig
from .mixer import AudioMixer, DEFAULT_OVERLAP_MS, DEFAULT_SOFT_OVERLAP_MS
from .voice_catalog import UNIFIED_VOICE_CATALOG

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE DIRECTION STRIPPING
# ═══════════════════════════════════════════════════════════════════════════

# Matches any (parenthetical) block in utterance text, e.g.:
#   (laughs)  (dramatic pause)  (Leaning in, voice low)
# These are LLM-generated performance directions. Standard TTS engines
# have no way to act on them, so they end up spoken literally.
_STAGE_DIR_RE = re.compile(r'\(([^)]*)\)')

# Matches square-bracket stage directions, e.g.:
#   [Dr. Waffleford holds the pizza aloft.]
#   [The audience gasps.]
# Audiobook conductors write these for scene actions. They are NEVER data —
# strip them unconditionally (unlike parentheticals which may contain acronyms).
_BRACKET_STAGE_DIR_RE = re.compile(r'\[[^\]]*\]')


def _is_stage_direction(content: str) -> bool:
    """Return True if a parenthetical looks like a performance direction.

    Stage directions contain no digits and no acronyms (two or more
    consecutive uppercase letters).  Numeric asides and acronym expansions
    are treated as prose and left untouched.  Long prose explanations with
    conjunctions and verb phrases are also left untouched.
    """
    stripped = content.strip()
    # Numeric asides: "(3.5 million)", "(down 12%)"
    if re.search(r'\d', stripped):
        return False
    # Acronym expansions: "(UBI)", "(GDP growth)"
    if re.search(r'[A-Z]{2,}', stripped):
        return False
    # Long prose parentheticals with conjunctions/relative clauses are not cues
    if len(stripped.split()) > 15:
        return False
    return True


def strip_stage_directions(text: str) -> str:
    """Remove parenthetical stage directions from TTS text.

    Only strips parentheticals that look like performance cues (short,
    no digits, no acronyms).  Prose parentheticals such as acronym
    expansions ``(UBI)`` or numeric asides ``(down 3%)`` are preserved.

    Strips all matching (parenthetical) blocks regardless of their position in
    the string, then collapses any resulting runs of whitespace.  Adjacent
    ellipsis and em-dash artifacts created by the removal are also merged.

    Examples::

        >>> strip_stage_directions("(laughs) Well, that's funny.")
        "Well, that's funny."
        >>> strip_stage_directions("Let me tell you... (dramatic pause) Gold.")
        'Let me tell you... Gold.'
        >>> strip_stage_directions("Just plain dialogue.")
        'Just plain dialogue.'
        >>> strip_stage_directions("Universal Basic Income (UBI) proposals")
        'Universal Basic Income (UBI) proposals'
    """
    def _replace(m: re.Match) -> str:
        if _is_stage_direction(m.group(1)):
            return ''
        return m.group(0)  # keep prose parentheticals intact

    cleaned = _STAGE_DIR_RE.sub(_replace, text)
    # Strip square-bracket stage directions unconditionally — these are always
    # scene actions written by the audiobook conductor, never data.
    cleaned = _BRACKET_STAGE_DIR_RE.sub('', cleaned)
    # Collapse ALL whitespace runs — including \n from removed inline directions
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Merge adjacent-ellipsis artifacts left by "... (direction) ...": "... ..." → "..."
    cleaned = re.sub(r'\.\.\.[ \t]*\.\.\.', '...', cleaned)
    # Merge adjacent em-dash artifacts left by "—(direction)—": "——" or "— —" → "—"
    cleaned = re.sub(r'\u2014[ \t]*\u2014', '\u2014', cleaned)
    return cleaned.strip()


# Matches Markdown emphasis markers: *word*, **word**, _word_, __word__
# These are written by the LLM for emphasis but have no meaning in TTS —
# the asterisks/underscores are read aloud literally by the speech engine.
_MD_EMPHASIS_RE = re.compile(r'(\*{1,2}|_{1,2})(.+?)\1')


# Matches inline music cues written by agents into utterance text:
#   "[music: upbeat jazz piano]"
#   "[Music: somber strings]"
# The captured group is the prompt passed to the music engine. These cues
# can appear in any event — not just bumper events — and trigger generative
# music when a music engine is wired into the renderer.
_MUSIC_CUE_RE = re.compile(r'\[\s*music\s*:\s*([^\]]+?)\s*\]', re.IGNORECASE)


# Nonverbal markers that signal a theme music cue at this event boundary.
# Lowercased substring match against each entry in event["nonverbal"].
_THEME_MUSIC_MARKERS: frozenset[str] = frozenset({"theme music", "theme song", "intro music", "outro music"})


def _ensure_music_bumpers(events: list[dict]) -> None:
    """Mutate *events* in-place to guarantee intro/outro music cues exist.

    Called by :meth:`TTSRenderer.render` when a music engine is active.  This
    is a safety net for show JSONs generated before the prompts carried the
    explicit cue instructions, for re-renders, and for LLM non-compliance.

    A cue is already considered present if:
    - the event's ``event_kind`` is ``"bumper"`` (an explicit bumper event), OR
    - ``"intro music"`` / ``"outro music"`` already appears in the event's
      ``nonverbal`` list (case-insensitive substring match).
    """
    if not events:
        return

    def _has_cue(event: dict, cue: str) -> bool:
        if event.get("event_kind") == "bumper":
            return True
        return any(cue in nv.lower() for nv in event.get("nonverbal") or [])

    first = events[0]
    if not _has_cue(first, "intro music"):
        nv = list(first.get("nonverbal") or [])
        nv.append("intro music")
        first["nonverbal"] = nv

    last = events[-1]
    if not _has_cue(last, "outro music"):
        nv = list(last.get("nonverbal") or [])
        nv.append("outro music")
        last["nonverbal"] = nv


def strip_markdown_emphasis(text: str) -> str:
    """Remove Markdown emphasis markers from TTS text, keeping the inner word.

    Strips *word*, **word**, _word_, and __word__ patterns, leaving the
    emphasised text intact without the surrounding markers.

    Examples::

        >>> strip_markdown_emphasis("anything still works to come back *to*")
        'anything still works to come back to'
        >>> strip_markdown_emphasis("a *critiquing* son")
        'a critiquing son'
        >>> strip_markdown_emphasis("Plain text unchanged.")
        'Plain text unchanged.'
    """
    return _MD_EMPHASIS_RE.sub(r'\2', text)


def apply_pronunciation_rewrites(text: str, overrides: dict[str, Any]) -> str:
    """Replace source phrases with their spoken forms (Piper text-rewrite path).

    Performs a single-pass, case-insensitive, whole-word substitution so that
    no replacement can be re-matched by a subsequent shorter rule.  Longer
    phrases are listed first in the alternation so "New York" wins over "York".
    """
    if not overrides:
        return text
    # Build lookup: normalised source -> spoken form (skip empty spoken)
    spoken_map: dict[str, str] = {}
    for phrase in sorted(overrides, key=len, reverse=True):
        entry = overrides[phrase]
        spoken = entry.get("spoken", "") if isinstance(entry, dict) else str(entry)
        if spoken:
            spoken_map[phrase] = spoken
    if not spoken_map:
        return text
    pattern = re.compile(
        r'(?<![^\W_])(?:' +
        '|'.join(re.escape(p) for p in spoken_map) +
        r')(?![^\W_])',
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: spoken_map[next(
        k for k in spoken_map if k.lower() == m.group().lower()
    )], text)


def build_pronunciation_hint(overrides: dict[str, Any]) -> str:
    """Build a natural-language pronunciation guidance suffix for Kokoro.

    Returns a string like::

        "pronounce Okonkwo as oh-KOHN-kwoh; pronounce Nguyen as nwin"

    appended to the emotion hint so Kokoro's instruct layer can shape
    phoneme production.  Returns an empty string when *overrides* is empty.
    """
    if not overrides:
        return ""
    parts = []
    for phrase, val in overrides.items():
        spoken = val.get("spoken", "") if isinstance(val, dict) else str(val)
        if spoken:
            parts.append(f"pronounce {phrase} as {spoken}")
    return "; ".join(parts)


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
    event_durations: dict[int, float] = field(default_factory=dict)
    """Per-event audio duration in seconds, keyed by event index.

    Populated only for spoken events that were successfully synthesised.
    Downstream callers (animation timeline sync) use this to align
    scene boundaries to real audio boundaries instead of relying on
    the 3-second fallback.
    """

    def __post_init__(self):
        if self.voice_map is None:
            self.voice_map = {}


# Event kinds that contain spoken text and should be synthesized.
# Other kinds (break, bumper, transition) are non-spoken markers
# handled by the mixer or ignored.
SPOKEN_EVENT_KINDS = frozenset({"utterance", "narration", "interrupt"})

# ── Content-type detection ─────────────────────────────────────────────────
# Infer content_type from the show JSON's top-level keys so render() doesn't
# need a new parameter and existing callers stay unchanged.
_CONTENT_TYPE_KEYS: dict[str, str] = {
    "podcast_id": "podcast",
    "debate_id": "debate",
    "audiobook_id": "audiobook",
    "show_id": "news",
}

# Lead-in silence (ms) prepended before the first utterance per content type.
# Prevents cold starts where audio jumps in with zero margin.
# Skipped when the first event already carries an intro music nonverbal cue
# (e.g. "intro music", "theme", "sting") to avoid double-stacking with a
# generated music bumper.
_LEAD_IN_MS: dict[str, int] = {
    "podcast": 1500,
    "news": 800,
    "debate": 1000,
    "audiobook": 2000,
}

# Nonverbal keywords that indicate a music intro is already present in the
# first event, making the lead-in silence redundant.
_INTRO_MUSIC_CUES: frozenset[str] = frozenset({
    "intro music", "theme music", "theme song", "theme", "sting", "bumper",
})

# Podcast-specific TTS speed multiplier applied on top of the per-event
# pace-derived speed.  Keeps conversational delivery measured rather than
# clipped.  Intentionally aligned with WPM_DEFAULTS["podcast"] = 135
# (see core/duration.py) so word-budget estimation and playback stay in sync.
PODCAST_DEFAULT_SPEED: float = 0.92


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
        registry: "VoiceRegistry | None" = None,
        series_key: str | None = None,
        exclude_whispery: bool = False,
        music_engine: "MusicGenEngine | None" = None,
        music_style: str = "",
        music_duration: float = 8.0,
        music_volume_db: float = 0.0,
        parallel_workers: int = 1,
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
            registry:         Optional VoiceRegistry for cross-episode character
                              consistency and global voice-usage tracking. When
                              supplied, recurring characters in the same series
                              are locked to the same voice and the gender-only
                              fallback prefers least-used voices. When None,
                              registry features are disabled (test-friendly).
            series_key:       Override for the series identifier used for
                              registry lookups. When None, ``render()`` derives
                              one from the input JSON (series_id → show_name →
                              "global").
            music_engine:     Optional MusicGenEngine for generative music at
                              bumper events (event_kind == "bumper") and
                              ``[music: …]`` cues found in any event's text.
                              When None, all such events are skipped silently
                              (preserves the legacy behaviour).
            music_style:      Default style hint appended to music prompts when
                              the cue itself contains no description. Examples:
                              "lo-fi podcast bed", "orchestral cinematic".
                              Ignored when ``music_engine`` is None.
            music_duration:   Default length (seconds) of generated music clips.
                              Used for any cue that doesn't specify its own.
                              Ignored when ``music_engine`` is None.
            parallel_workers: Number of worker threads used to synthesize
                              spoken events concurrently. Default 1 preserves
                              the existing sequential behaviour. Values > 1
                              submit all spoken events to a ThreadPoolExecutor
                              up-front and consume results in event order, so
                              audio mixing order is unchanged. Music cues
                              always render sequentially in the main loop.

                              IMPORTANT: Kokoro (MLX/MPS) has not been
                              validated with workers > 1 — the MLX model may
                              serialize internally and parallel calls could
                              be no-ops or cause GPU contention. Piper is
                              safe at any worker count (each call is a fresh
                              subprocess with its own model load).

        TIP: For quick testing, skip voice_map entirely. The renderer
             will assign voices automatically. For production, craft a
             voice_map that matches each character's personality.
        """
        self._engine = engine
        self._voice_map = voice_map or {}
        self._strip_directions = strip_directions
        self._registry = registry
        self._series_key_override = series_key
        self._exclude_whispery = exclude_whispery
        self._music_engine = music_engine
        self._music_style = music_style
        self._music_duration = music_duration
        self._music_volume_db = music_volume_db
        if parallel_workers < 1:
            raise ValueError(f"parallel_workers must be >= 1, got {parallel_workers}")
        self._parallel_workers = parallel_workers

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
        events = list(debate_data.get("events", []))
        if self._music_engine is not None and events:
            events = [dict(e) for e in events]  # shallow-copy — do not mutate caller's data
            _ensure_music_bumpers(events)
        participants = debate_data.get("participants", [])
        # The show's language drives voice selection and is passed to every
        # synthesize() call so the engine can validate / route appropriately.
        language = debate_data.get("language", "en")
        character_langs = dict(debate_data.get("character_langs", {}))

        # ── Infer content type from show JSON keys ────────────────────
        content_type = next(
            (ct for key, ct in _CONTENT_TYPE_KEYS.items() if key in debate_data),
            "generic",
        )

        # ── STEP 1: Assign voices to speakers ────────────────────────
        # Build the voice map (auto-assign if not manually provided).
        # We need to know every speaker name that appears in events.
        speaker_names = self._extract_speakers(events)
        voice_casting = debate_data.get("voice_casting", {})
        voice_hints = debate_data.get("voice_hints", {})
        tts_styles = debate_data.get("tts_styles", {})
        pronunciation_overrides = debate_data.get("pronunciation_overrides", {})

        # Pre-build the Kokoro instruct suffix once — same string every utterance.
        _pron_hint_suffix = build_pronunciation_hint(pronunciation_overrides)
        _title = debate_data.get("title", "")
        _seed = sum(ord(c) for c in _title[:32]) if _title else 0

        # Resolve the series key for registry lookups. Priority:
        # explicit constructor override > JSON series_id > JSON show_name > "global".
        series_key = (
            self._series_key_override
            or debate_data.get("series_id")
            or debate_data.get("show_name")
            or "global"
        )

        voice_map = self._build_voice_map(
            speaker_names, voice_casting, voice_hints,
            language=language, character_langs=character_langs, seed=_seed,
            registry=self._registry, series_key=series_key,
        )

        # _build_voice_map normalises speaker names to canonical host names
        # (e.g. "Lord Bridgerton" → "Viscount Anthony Bridgerton").  Events in
        # the JSON still carry the original speaker_name, so add alias entries
        # so the main render loop can look up either form.
        _raw_speakers = self._extract_speakers(events)
        for raw_name in _raw_speakers:
            canonical = voice_map.get(raw_name)
            if canonical is None:
                # Try suffix-token match against existing keys.
                raw_last = raw_name.lower().split()[-1] if raw_name else ""
                for key, vid in voice_map.items():
                    if raw_last and raw_last in key.lower().split():
                        voice_map[raw_name] = vid
                        break
            if raw_name in character_langs:
                continue
            raw_last = raw_name.lower().split()[-1] if raw_name else ""
            for key, code in list(character_langs.items()):
                if raw_last and raw_last in key.lower().split():
                    character_langs[raw_name] = code
                    break

        # Build a VoiceConfig map covering all auto-assigned speakers so that
        # engine-specific extras (e.g. Piper speaker-id) are populated even
        # when the voice was chosen automatically.  Manual overrides from
        # self._voice_map (which carry extras set by the caller) take precedence.
        voice_cfg_map: dict[str, VoiceConfig] = {
            speaker: self._voice_map.get(speaker, VoiceConfig(voice_id=vid))
            for speaker, vid in voice_map.items()
        }

        if not quiet:
            print(f"  Voice assignments:")
            for speaker, voice_id in voice_map.items():
                print(f"    {speaker} → {voice_id}")

        # ── STEP 2: Optionally pre-submit spoken events to a thread pool ─
        # When parallel_workers > 1 we identify every spoken event up-front
        # and submit a synthesize() call for each into a ThreadPoolExecutor.
        # The main loop below still walks events in script order; for each
        # spoken event it pulls the matching future's .result(). This
        # preserves mixing order (and therefore audio order) while letting
        # synthesis overlap.
        #
        # Music cues stay in the sequential main loop — MusicGen is also
        # MPS-bound and concurrent generation has not been validated.
        # Kokoro (MLX/MPS) similarly has not been tested with workers > 1
        # and may serialize internally; Piper (subprocess-per-call) is
        # safe at any worker count.
        synth_futures: dict[int, Future] = {}
        executor: ThreadPoolExecutor | None = None
        if self._parallel_workers > 1:
            executor = ThreadPoolExecutor(max_workers=self._parallel_workers)
            for i, event in enumerate(events):
                # Skip music cues — they render in the main loop.
                if self._music_engine is not None and self._is_music_cue(event):
                    continue
                synth_kwargs = self._prepare_synth_kwargs(
                    event,
                    voice_map,
                    voice_cfg_map,
                    tts_styles,
                    character_langs,
                    language,
                )
                if synth_kwargs is None:
                    continue
                synth_futures[i] = executor.submit(
                    self._engine.synthesize, **synth_kwargs,
                )

        # ── STEP 3: Render each event through the TTS engine ────────
        mixer = AudioMixer()
        rendered = 0
        skipped = 0
        event_durations: dict[int, float] = {}

        # ── C1: Cold-start lead-in silence ───────────────────────────
        # Prepend a brief silence before the first spoken clip so audio
        # doesn't jump in with zero margin ("cold start" artefact).
        # Skip if the first event already has an intro music nonverbal cue
        # so we don't stack silence on top of a generated bumper.
        lead_in_ms = _LEAD_IN_MS.get(content_type, 0)
        if lead_in_ms > 0:
            first_event = next(
                (e for e in events if e.get("text", "").strip()),
                None,
            )
            first_nonverbals = {
                n.lower()
                for n in (first_event.get("nonverbal") or [])
            } if first_event else set()
            if not (_INTRO_MUSIC_CUES & first_nonverbals):
                mixer.add_silence(lead_in_ms)

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

            # Always strip Markdown emphasis markers (*word*, _word_, etc.).
            # TTS engines have no concept of Markdown — the asterisks are read
            # aloud literally.  The inner word is preserved.
            text = strip_markdown_emphasis(text)

            # ── Music intercept ──────────────────────────────────────
            # Bumper events and inline [music: …] cues route to the music
            # engine instead of the speech engine. We do this BEFORE the
            # SPOKEN_EVENT_KINDS guard below so that bumper events (which
            # have no spoken text) are caught here rather than skipped.
            #
            # When no music engine is wired in, these events fall through
            # to the existing skip-or-synthesize logic — preserving the
            # legacy behaviour where bumpers were silently dropped.
            if self._music_engine is not None and self._is_music_cue(event):
                prompt = self._resolve_music_prompt(event)
                if not quiet:
                    label = f"[{i+1}/{len(events)}]"
                    print(f"  {label} 🎵 music: {prompt}")
                try:
                    music_result = self._music_engine.generate(
                        prompt,
                        duration=self._music_duration,
                    )
                    mixer.add_music_clip(
                        music_result.audio,
                        sample_rate=music_result.sample_rate,
                        gain_db=self._music_volume_db,
                    )
                    # Record music clip duration so TimelineSync can account
                    # for it when building scene timestamps.  Without this,
                    # music events are invisible to the timeline and all
                    # subsequent scenes start too early in video time.
                    try:
                        with wave.open(io.BytesIO(music_result.audio)) as _wf:
                            event_durations[i] = _wf.getnframes() / _wf.getframerate()
                    except Exception:
                        # Fall back to the requested duration if WAV probe fails.
                        event_durations[i] = self._music_duration
                    rendered += 1
                except Exception as exc:
                    logger.warning(
                        "Music generation failed for event %d (%r): %s",
                        i, prompt, exc,
                    )
                    skipped += 1
                continue

            if not text or not text.strip() or event_kind not in SPOKEN_EVENT_KINDS:
                skipped += 1
                continue

            speaker = event.get("speaker_name", "Unknown")
            event_language = self._event_language(
                event,
                character_langs,
                default_language=language,
            )
            voice_id = voice_map.get(speaker)
            if voice_id is None:
                voice_id = self._default_voice()

            # ── Pronunciation overrides ───────────────────────────
            # Determine engine for this voice once per utterance.
            # Piper: rewrite source phrase directly in the text.
            # Kokoro: append a guidance suffix to the emotion hint.
            _voice_engine = "kokoro"
            if pronunciation_overrides or _pron_hint_suffix:
                _desc = UNIFIED_VOICE_CATALOG.get(voice_id)
                if _desc is not None:
                    _voice_engine = _desc.engine
                if _voice_engine == "piper" and pronunciation_overrides:
                    text = apply_pronunciation_rewrites(text, pronunciation_overrides)

            # ── Build the emotion hint from event metadata ────────
            # We combine tone, pace, volume, and nonverbal cues into
            # a single string that the engine can interpret.
            # If a persistent tts_style is set for this speaker, it
            # is prepended as a baseline delivery instruction.
            emotion = self._build_emotion_hint(event, tts_styles.get(speaker))
            if _pron_hint_suffix and _voice_engine == "kokoro":
                emotion = (emotion + "; " + _pron_hint_suffix).lstrip("; ")

            # ── Determine speed from pace metadata ────────────────
            speed = self._pace_to_speed(event.get("pace"))
            # C2: Apply podcast-specific speed baseline so delivery stays
            # measured.  Aligned with WPM_DEFAULTS["podcast"] = 135.
            if content_type == "podcast":
                speed *= PODCAST_DEFAULT_SPEED

            # ── Determine pause and overlap ───────────────────────
            pause_before = mixer.resolve_pause(event.get("pause_before"))
            pause_after = mixer.resolve_pause(event.get("pause_after")) if event.get("pause_after") else 0
            overlap = 0
            if event.get("interruption"):
                overlap = DEFAULT_OVERLAP_MS
                pause_before = 0  # No silence before an interruption.
            elif event.get("overlap_with_previous"):
                overlap = DEFAULT_SOFT_OVERLAP_MS
                pause_before = 0

            # Get engine-specific extras from the voice config.
            voice_cfg = voice_cfg_map.get(speaker, VoiceConfig(voice_id=voice_id))
            extras = voice_cfg.extras

            # ── Synthesize! ───────────────────────────────────────
            if not quiet:
                truncated = text[:60] + "..." if len(text) > 60 else text
                label = f"[{i+1}/{len(events)}]"
                print(f"  {label} {speaker}: {truncated}")

            try:
                future = synth_futures.get(i)
                if future is not None:
                    result = future.result()
                else:
                    result = self._engine.synthesize(
                        text,
                        voice=voice_id,
                        emotion=emotion,
                        speed=speed,
                        language=event_language,
                        **extras,
                    )

                mixer.add_clip(
                    result.audio,
                    sample_rate=result.sample_rate,
                    pause_before=pause_before,
                    pause_after=pause_after,
                    overlap=overlap,
                    speaker=speaker,
                )
                rendered += 1

                # Record per-event duration for animation timeline sync.
                try:
                    with wave.open(io.BytesIO(result.audio)) as _wf:
                        event_durations[i] = _wf.getnframes() / _wf.getframerate()
                except Exception:
                    pass  # Duration unavailable; fallback applies downstream.

                # Bump the voice's global usage counter only after a
                # successful synthesis.  This way, voices that fail to
                # render (missing model, engine error, etc.) don't get
                # penalised by the gender-fallback's least-used picker.
                if self._registry is not None and voice_id and voice_id != "default":
                    try:
                        self._registry.record_usage(voice_id)
                    except Exception:
                        # Registry failures must never break a render.
                        pass

            except Exception as e:
                # If one clip fails, skip it rather than crashing the
                # whole render. Log the error and keep going.
                # The final audio will have a gap where the failed clip
                # would have been.
                if not quiet:
                    print(f"    WARNING: Failed to synthesize: {e}")
                skipped += 1

        if executor is not None:
            executor.shutdown(wait=True)

        # ── STEP 4: Mix and export ───────────────────────────────────
        if not quiet:
            print(f"\n  Mixing {rendered} clips...")

        final_path = mixer.export(output_path, format=format, normalize=normalize)
        duration_ms = mixer.last_mix_duration_ms or mixer.duration_ms

        return RenderResult(
            output_path=final_path,
            total_events=len(events),
            rendered=rendered,
            skipped=skipped,
            duration_ms=duration_ms,
            voice_map=voice_map,
            event_durations=event_durations,
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
        *,
        language: str = "en",
        character_langs: dict[str, str] | None = None,
        seed: int = 0,
        registry: "VoiceRegistry | None" = None,
        series_key: str | None = None,
    ) -> dict[str, str]:
        """Build a speaker→voice_id mapping.

        Priority order:
        1. Manual voice_map (passed to TTSRenderer constructor).
        2. Registry lookup — lock recurring characters to their previously
           assigned voice within ``series_key`` (when registry is provided).
        3. Catalog-matched voice using voice_hints metadata. If any
           speaker has a voice_hint, the catalog's ``match_voices_for_cast``
           assigns distinct voices based on hint tag overlap.
        4. Gender-aware assignment using voice_casting metadata from the
           show JSON (maps speaker names to "male"/"female"), preferring
           least-used voices when a registry is provided.
        5. Round-robin through available voices (fallback).

        After assignment, every newly chosen voice is written back to the
        registry (if any) so future renders honor the lock and so the
        usage-count-based fallback distributes work across the full pool.

        Args:
            speaker_names:  Speaker names in order of first appearance.
            voice_casting:  Optional dict mapping speaker names to gender
                            strings ("male" or "female") from the show JSON.
            voice_hints:    Optional dict mapping speaker names to natural-
                            language casting notes for catalog matching.
            language:       BCP-47 code for the show. When non-English on a
                            Piper backend, the candidate voice pool is
                            pre-filtered to voices whose id encodes that
                            language; catalog matching is skipped (the
                            built-in catalogs only cover English voices).
            character_langs: Optional per-speaker language overrides.

        Returns:
            Dict mapping every speaker name to a voice_id string.
        """
        available = self._engine.available_voices()
        casting = voice_casting or {}
        hints = voice_hints or {}
        language_overrides = character_langs or {}
        result: dict[str, str] = {}

        # ── Normalize speaker names to canonical host names ────────────
        # The conductor LLM sometimes uses shortened or informal variants of
        # a host's canonical name (e.g. "Lord Bridgerton" instead of
        # "Viscount Anthony Bridgerton"). This causes casting / hints lookups
        # to silently miss, dropping gender info and producing wrong voices.
        # Build a name_map that resolves event speaker_names to the canonical
        # key used in voice_casting / voice_hints.
        #
        # Match priority (first hit wins):
        #   1. Exact match — no-op, already correct.
        #   2. Case-insensitive exact match.
        #   3. Suffix token match: last whitespace-delimited token of the
        #      event name appears as a token in exactly one canonical name
        #      (e.g. "Bridgerton" → "Viscount Anthony Bridgerton").
        canonical_names: set[str] = set(casting) | set(hints)

        def _resolve_name(name: str) -> str:
            if name in canonical_names:
                return name
            name_lower = name.lower()
            for cname in canonical_names:
                if cname.lower() == name_lower:
                    return cname
            # Suffix token match: last token of event name appears in canonical
            tokens_event = name.lower().split()
            if tokens_event:
                last_token = tokens_event[-1]
                matches = [
                    c for c in canonical_names
                    if last_token in c.lower().split()
                ]
                if len(matches) == 1:
                    return matches[0]
            return name

        # Remap speaker_names list and build a translation table so that
        # result keys are always the canonical form used by casting/hints.
        name_map: dict[str, str] = {n: _resolve_name(n) for n in speaker_names}
        speaker_names = [name_map[n] for n in speaker_names]
        # Deduplicate while preserving order (two variants of the same host).
        seen_canonical: set[str] = set()
        speaker_names = [
            n for n in speaker_names
            if not (n in seen_canonical or seen_canonical.add(n))  # type: ignore[func-returns-value]
        ]
        speaker_languages: dict[str, str] = {}
        for raw_name, canonical_name in name_map.items():
            effective_language = (
                language_overrides.get(canonical_name)
                or language_overrides.get(raw_name)
                or language
            )
            speaker_languages.setdefault(canonical_name, effective_language)

        # Detect which engine is active and pick the right catalog.
        from .engine import BaseTTSEngine
        from .voice_catalog import (
            KOKORO_VOICE_CATALOG,
            PIPER_VOICE_CATALOG,
            UNIFIED_VOICE_CATALOG,
        )

        try:
            from .engines.piper_engine import PiperEngine, _lang_from_voice_id
            _is_piper = isinstance(self._engine, PiperEngine)
        except Exception:
            _is_piper = False
            _lang_from_voice_id = lambda v: None  # type: ignore[assignment]

        try:
            from .engines.multi_engine import MultiEngine
            _is_multi = isinstance(self._engine, MultiEngine)
        except Exception:
            _is_multi = False

        if _is_multi:
            # Both Kokoro + Piper voices are valid candidates; the engine
            # will route synthesis to the right backend based on voice_id.
            active_catalog = UNIFIED_VOICE_CATALOG
        elif _is_piper:
            active_catalog = PIPER_VOICE_CATALOG
        else:
            active_catalog = KOKORO_VOICE_CATALOG

        try:
            from .engines.kokoro_engine import KOKORO_LANGUAGES as _KOKORO_LANGUAGES
        except Exception:
            _KOKORO_LANGUAGES = frozenset()

        def _candidate_pool_for_language(speaker_language: str) -> list[str]:
            normalized_language = speaker_language.strip().lower().replace("_", "-")
            primary_language = normalized_language.split("-", 1)[0]

            if primary_language == "en" or not (_is_piper or _is_multi):
                return list(available)

            if _is_multi:
                kokoro_supported = (
                    normalized_language in _KOKORO_LANGUAGES
                    or primary_language in _KOKORO_LANGUAGES
                )
                return [
                    voice_id for voice_id in available
                    if voice_id.startswith("vc_")
                    or _lang_from_voice_id(voice_id) == primary_language
                    or (
                        kokoro_supported
                        and voice_id in active_catalog
                        and active_catalog[voice_id].engine == "kokoro"
                    )
                ]

            return [
                voice_id for voice_id in available
                if _lang_from_voice_id(voice_id) == primary_language
            ]

        # ── Manual overrides first ────────────────────────────────────
        for name in speaker_names:
            if name in self._voice_map:
                result[name] = self._voice_map[name].voice_id

        # ── Registry lookup: lock recurring characters ──────────────
        # If we have a registry and the previously assigned voice is still
        # in the available pool, reuse it so the same character keeps the
        # same voice across episodes within the series.
        if registry is not None and series_key:
            for name in speaker_names:
                if name in result:
                    continue
                available_set = set(
                    _candidate_pool_for_language(speaker_languages.get(name, language))
                )
                try:
                    locked = registry.resolve(name, series_key)
                except Exception:
                    locked = None
                if locked and locked in available_set:
                    result[name] = locked

        # Snapshot global usage counts once — used by both the catalog
        # matcher and the gender-only fallback to prefer least-used voices.
        usage_counts: dict[str, int] = {}
        if registry is not None:
            try:
                usage_counts = registry.usage_counts()
            except Exception:
                usage_counts = {}

        # ── Catalog matching (when any speaker has a voice_hint) ──────
        # Build a cast dict for unassigned speakers that have hints.
        remaining = [n for n in speaker_names if n not in result]
        catalog_remaining = [
            n for n in remaining
            if speaker_languages.get(n, language) == "en"
        ]
        if catalog_remaining and any(n in hints for n in catalog_remaining):
            from .voice_catalog import match_voices_for_cast

            cast_info: dict[str, dict] = {}
            for name in catalog_remaining:
                info: dict[str, str] = {}
                if name in casting:
                    info["gender"] = casting[name]
                if name in hints:
                    info["voice_hint"] = hints[name]
                cast_info[name] = info

            # Catalog assigns unique voices for the entire cast. Voices
            # already locked by the registry must not be reassigned.
            already_used = set(result.values())
            catalog_map = match_voices_for_cast(
                cast_info,
                catalog=active_catalog,
                usage_counts=usage_counts,
                exclude_whispery=self._exclude_whispery,
            )
            for name, voice_id in catalog_map.items():
                if voice_id and name not in result and voice_id not in already_used:
                    result[name] = voice_id
                    already_used.add(voice_id)

        # ── Gender-only fallback for any still-unassigned speakers ────
        # For Piper voices, the naming convention is:
        #   en_US-lessac-high, en_US-ryan-high, en_GB-cori-high, etc.
        # We use the catalog to split by gender; fall back to keyword
        # heuristics for voices not in the catalog.
        if _is_multi:
            # Drive purely off the catalog — it covers every voice in
            # both Kokoro and Piper, so no keyword heuristics needed.
            male_voices = [
                v for v in available
                if v in active_catalog and active_catalog[v].gender == "male"
            ]
            female_voices = [
                v for v in available
                if v in active_catalog and active_catalog[v].gender == "female"
            ]
        elif _is_piper:
            male_voices = [
                v for v in available
                if v in active_catalog and active_catalog[v].gender == "male"
                or (v not in active_catalog and any(
                    kw in v for kw in ("ryan", "joe", "danny", "norman", "bryce",
                                       "john", "hfc_male", "sam", "alan",
                                       "northern_english_male")
                ))
            ]
            female_voices = [
                v for v in available
                if v in active_catalog and active_catalog[v].gender == "female"
                or (v not in active_catalog and any(
                    kw in v for kw in ("lessac", "amy", "kathleen", "kristin",
                                       "hfc_female", "ljspeech", "alba", "cori",
                                       "jenny_dioco", "southern_english_female")
                ))
            ]
        else:
            # Kokoro naming convention: af_* / bf_* = female; am_* / bm_* = male
            male_voices = [v for v in available if v[:2] in ("am", "bm")]
            female_voices = [v for v in available if v[:2] in ("af", "bf")]

        # Filter whispery voices from all pools when requested.
        if self._exclude_whispery:
            def _not_whispery(v: str) -> bool:
                desc = active_catalog.get(v)
                return desc is None or not desc.is_whispery
            male_voices = [v for v in male_voices if _not_whispery(v)]
            female_voices = [v for v in female_voices if _not_whispery(v)]
            available = [v for v in available if _not_whispery(v)]

        # When usage counts are available, sort each gender pool by
        # ascending usage so the next unassigned speaker draws from the
        # least-used voices first. Python's sort is stable, so voices
        # with equal counts keep their original (catalog) order — this
        # makes behavior deterministic and reproducible.
        if usage_counts:
            male_voices = sorted(male_voices, key=lambda v: usage_counts.get(v, 0))
            female_voices = sorted(female_voices, key=lambda v: usage_counts.get(v, 0))
            available = sorted(available, key=lambda v: usage_counts.get(v, 0))

        male_idx = seed % len(male_voices) if male_voices else 0
        female_idx = seed % len(female_voices) if female_voices else 0
        fallback_idx = seed % len(available) if available else 0

        for name in speaker_names:
            if name in result:
                continue
            speaker_available = _candidate_pool_for_language(
                speaker_languages.get(name, language)
            )
            speaker_available_set = set(speaker_available)
            speaker_male_voices = [v for v in male_voices if v in speaker_available_set]
            speaker_female_voices = [v for v in female_voices if v in speaker_available_set]
            if name in casting and casting[name] == "female" and speaker_female_voices:
                result[name] = speaker_female_voices[female_idx % len(speaker_female_voices)]
                female_idx += 1
            elif name in casting and casting[name] == "male" and speaker_male_voices:
                result[name] = speaker_male_voices[male_idx % len(speaker_male_voices)]
                male_idx += 1
            elif speaker_male_voices:
                # No gender info — default to male (safer than round-robin
                # through all voices which starts with female).
                result[name] = speaker_male_voices[male_idx % len(speaker_male_voices)]
                male_idx += 1
            elif speaker_available:
                # No gendered voices found — raw round-robin fallback.
                result[name] = speaker_available[fallback_idx % len(speaker_available)]
                fallback_idx += 1
            else:
                result[name] = "default"

        # ── Persist assignments to the registry ──────────────────
        # register() is idempotent: it locks the voice on first sight and
        # only refreshes last_seen on subsequent calls (never overwrites).
        # NOTE: usage counts are NOT bumped here — only after a clip is
        # successfully synthesized (see the main render loop).  Counting
        # at assignment time would penalise voices for failed renders.
        if registry is not None and series_key:
            for name, voice_id in result.items():
                if not voice_id or voice_id == "default":
                    continue
                try:
                    registry.register(name, voice_id, series_key)
                except Exception:
                    # Registry failures must never break a render.
                    pass

        return result

    def _default_voice(self) -> str:
        """Get the first available voice, or "default" if none."""
        voices = self._engine.available_voices()
        return voices[0] if voices else "default"

    @staticmethod
    def _event_language(
        event: dict,
        character_langs: dict[str, str],
        *,
        default_language: str,
    ) -> str:
        """Resolve the effective language for a single event."""
        speaker = event.get("speaker_name", "Unknown")
        return character_langs.get(speaker, default_language)

    def _prepare_synth_kwargs(
        self,
        event: dict,
        voice_map: dict[str, str],
        voice_cfg_map: dict[str, VoiceConfig],
        tts_styles: dict[str, str],
        character_langs: dict[str, str],
        default_language: str,
    ) -> dict | None:
        """Build the kwargs dict for an engine.synthesize() call.

        Returns ``None`` if the event should be skipped (non-spoken kind,
        empty text, or text consisting only of stripped stage directions).

        This logic MUST stay in sync with the per-event prep in render()'s
        main loop — both paths feed the same engine with the same kwargs,
        so drift here would cause the parallel pre-pass to disagree with
        the sequential fallback. When parallel_workers > 1 this helper is
        called once during the pre-pass; the main loop's own inline
        computation provides the matching kwargs for sequential mode.
        """
        text = event.get("text", "")
        event_kind = event.get("event_kind", "utterance")
        if self._strip_directions:
            text = strip_stage_directions(text)
        text = strip_markdown_emphasis(text)
        if not text or not text.strip() or event_kind not in SPOKEN_EVENT_KINDS:
            return None

        speaker = event.get("speaker_name", "Unknown")
        voice_id = voice_map.get(speaker) or self._default_voice()
        emotion = self._build_emotion_hint(event, tts_styles.get(speaker))
        speed = self._pace_to_speed(event.get("pace"))
        voice_cfg = voice_cfg_map.get(speaker, VoiceConfig(voice_id=voice_id))
        event_language = self._event_language(
            event,
            character_langs,
            default_language=default_language,
        )
        kwargs = {
            "text": text,
            "voice": voice_id,
            "emotion": emotion,
            "speed": speed,
            "language": event_language,
        }
        kwargs.update(voice_cfg.extras)
        return kwargs

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

    # ── Music helpers ─────────────────────────────────────────────────────────

    def _is_music_cue(self, event: dict) -> bool:
        """Return True if this event should produce generative music.

        Three signals trigger music routing:
          1. ``event_kind == "bumper"`` — explicit bumper marker from the
             show script. The renderer historically skipped these silently.
          2. ``[music: …]`` cue embedded in the event text — the agent
             writes an inline music direction (any event kind).
          3. A theme-music marker in ``event.nonverbal`` — e.g. agents
             wrote ``"theme music"``, ``"intro music"``, etc.

        Args:
            event: A raw event dict (as found in the JSON ``events`` list).

        Returns:
            True if any of the three signals match. False otherwise.
        """
        if (event.get("event_kind") or "").lower() == "bumper":
            return True

        text = event.get("text") or ""
        if _MUSIC_CUE_RE.search(text):
            return True

        nonverbal = event.get("nonverbal") or ()
        for marker in nonverbal:
            marker_lc = (marker or "").lower()
            if any(theme in marker_lc for theme in _THEME_MUSIC_MARKERS):
                return True

        return False

    def _resolve_music_prompt(self, event: dict) -> str:
        """Build the text prompt to pass to the music engine for this event.

        Resolution priority:
          1. The first ``[music: …]`` capture in the event text — the agent
             explicitly described what they want.
          2. The first matching theme-music marker in ``event.nonverbal``.
          3. The renderer's ``music_style`` default (set in __init__).
          4. A safe generic fallback so we always pass something to the model.

        Args:
            event: A raw event dict.

        Returns:
            A non-empty prompt string suitable for ``MusicGenEngine.generate()``.
        """
        text = event.get("text") or ""
        m = _MUSIC_CUE_RE.search(text)
        if m:
            inline = m.group(1).strip()
            if inline:
                return inline

        nonverbal = event.get("nonverbal") or ()
        for marker in nonverbal:
            marker_lc = (marker or "").lower()
            if any(theme in marker_lc for theme in _THEME_MUSIC_MARKERS):
                # The nonverbal marker is a *trigger* signal ("intro music",
                # "outro music", etc.), not a descriptive MusicGen prompt.
                # Fall through to music_style / generic fallback below.
                break

        if self._music_style:
            return self._music_style

        return "podcast background music"
