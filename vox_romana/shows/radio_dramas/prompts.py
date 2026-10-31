"""Radio drama system prompt templates.

Radio dramas are SCRIPTED multi-voice narrative fiction — golden-age radio
serials (Sam Spade, Lights Out, Suspense, X Minus One) and their modern
descendants. Like podcasts, the generation agent voices ALL characters in a
single response (a JSON array of lines); unlike podcasts, the content is
plotted fiction (scenes, not conversational segments) and carries an
announcer cold-open/sign-off plus era-styled music cues.

These templates use str.format() placeholders. The conductor/bootstrap fill
them in at runtime with data from the RadioDramaConfig.

ERA DIAL:
    The `--era` knob (1940s | 1970s | present | future) is data-driven via
    :data:`ERA_TABLE`. It influences three things — the announcer's register,
    the music-style seed, and the vocabulary guidance — so no scattered
    if/else blocks in the prompt text.

NOTE (5.1-a scope):
    SFX is DEFERRED for the MVP. The scriptwriter may emit `[sfx: ...]`
    bracket cues (the TTS renderer already strips them), but no discrete SFX
    audio is synthesized yet.
"""

from __future__ import annotations

from ...core.messages import ISSUE_FLAG_PROMPT_NOTE

# ═══════════════════════════════════════════════════════════════════════════
# ERA DIAL — data-driven table
# ═══════════════════════════════════════════════════════════════════════════

ERA_TABLE = {
    "1940s": {
        "label": "Golden Age of Radio (1930s–50s)",
        "announcer_register": (
            "Formal, boom-voiced, theatrical. Opens by announcing the show "
            "title, the episode, and the cast; closes with a warm, era-"
            "appropriate sign-off."
        ),
        "music_seed": (
            "1940s big-band and orchestral radio-drama stings: dramatic "
            "brass hits, sweeping strings, suspense organ stabs"
        ),
        "vocabulary": (
            "period-correct mid-century American idiom; hardboiled/noir or "
            "wholesome serial diction as the genre demands"
        ),
    },
    "1970s": {
        "label": "1970s documentary / late-night drama",
        "announcer_register": (
            "Laid-back, warm, documentary-style. Measured and intimate, "
            "like a late-night FM host setting up a story."
        ),
        "music_seed": (
            "1970s funk and synth-strings underscores, moody bass lines, "
            "analog synth drones"
        ),
        "vocabulary": (
            "1970s idiom; paranoia, conspiracy, and social realism flavors "
            "as the genre demands"
        ),
    },
    "present": {
        "label": "Present day",
        "announcer_register": (
            "Neutral, modern, clean. Concise setup and sign-off without "
            "period affectation."
        ),
        "music_seed": (
            "modern cinematic underscore, tense drones, contemporary "
            "production texture"
        ),
        "vocabulary": "contemporary, natural, present-day speech",
    },
    "future": {
        "label": "Speculative future",
        "announcer_register": (
            "Clipped, precise, slightly cold. Announcements read like a "
            "ship's computer briefing — information-first, minimal warmth."
        ),
        "music_seed": (
            "cold futuristic ambient, synthetic pads, glitchy stings, "
            "space-y atmospherics"
        ),
        "vocabulary": (
            "speculative-future idiom; technical neologisms and clipped "
            "syntax as the genre demands"
        ),
    },
}


def era_guidance(era: str) -> str:
    """Return a formatted, brace-free block describing the chosen era.

    Safe to splice into any str.format() template as a pre-rendered string.
    """
    entry = ERA_TABLE.get(era, ERA_TABLE["present"])
    return (
        f"ERA: {entry['label']}\n"
        f"ANNOUNCER REGISTER: {entry['announcer_register']}\n"
        f"MUSIC STYLE: {entry['music_seed']}\n"
        f"VOCABULARY: {entry['vocabulary']}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SCRIPTWRITER SYSTEM PROMPT — the primary generation agent
# ═══════════════════════════════════════════════════════════════════════════
# This is the "cast sheet + show bible" injected as the system prompt for the
# scene-generating agent. The agent writes lines for ALL characters (and the
# announcer) in a single response — same shape as the podcast host prompt.

SCRIPTWRITER_SYSTEM_PROMPT = """\
You are writing a radio drama called "{show_name}".

GENRE: {genre}
TITLE: "{title}"
SETTING: {setting}

{era_guidance}

THE CAST (you write lines for ALL of them):
{cast_descriptions}

RULES:
- Write scripted dramatic fiction suitable for audio only. The listener
  cannot see anything — setting, action, and emotion must be carried by
  dialogue, sound, and the announcer.
- Stay in character. Each cast member has a distinct voice and agenda —
  make them sound DISTINCT from each other.
- Characters RESPOND to each other. This is staged drama, not parallel
  monologues. They interrupt, deflect, confess, lie, and react.
- Use characters' NAMES naturally in conversation — this is audio-only, so
  the listener needs periodic name cues to track who is speaking.
- Advance the plot. Every scene must move the story forward: reveal
  something, raise the stakes, or change a relationship.
- The announcer's lines bookend the show — a cold open that names the show,
  the episode, and the cast, and a sign-off at the end. The announcer never
  participates in a scene's dialogue.
- MUSIC: request era-styled music with an inline cue in the text, e.g.
  "[music: dramatic 1940s orchestral sting]". Use a music cue at the cold
  open, at scene transitions, and at the final sign-off — not every line.
- SFX: you MAY write sound effects as inline cues, e.g. "[sfx: rain]",
  "[sfx: footsteps on gravel]", "[sfx: gunshot]". Use sparingly — only
  when the sound genuinely carries the moment. (These cues are stripped
  from the spoken audio; no sound is synthesized yet.)
- STAGE DIRECTIONS: if a scene action is unavoidable, use ONLY [square
  brackets], never (parentheses). Delivery hints (tone, pace, pauses)
  belong in the structured fields, NOT embedded in the text.
- Vary energy. Not every line is a crescendo — mix tense silences, quiet
  confessions, sudden outbursts.

SPEAKER NAMES — CRITICAL:
The `speaker_name` field MUST be EXACTLY one of: {character_names}
Do NOT use shortened forms, titles, nicknames, or informal variants. Use
the full canonical name exactly as listed, character by character.

OUTPUT FORMAT:
Respond with ONLY a JSON array of line objects (no other text):
[
  {{
    "speaker_name": "exact cast name",
    "text": "the spoken line, or a music/sfx cue",
    "tone": "neutral|calm|warm|excited|skeptical|passionate|playful|contemplative|urgent|dramatic|somber|menacing|desperate",
    "volume": "soft|moderate|loud|rising|falling",
    "pace": "slow|medium|fast|accelerating|decelerating",
    "pause_before": "none|short|medium|long",
    "pause_after": "none|short|medium|long",
    "nonverbal": [],
    "event_kind": "utterance"
  }},
  ...
]

NONVERBAL FIELD — audible cues only (no visual gestures). Use sparingly:
e.g. ["laughs"], ["sighs"], ["gasps"], ["sobs"], ["screams"], ["deep
breath"]. Most lines leave it empty ([]). Never use visual cues like
"nods" or "leans in" — the listener cannot see them.
""" + ISSUE_FLAG_PROMPT_NOTE


# ═══════════════════════════════════════════════════════════════════════════
# SCENE DELIVERY PROMPT — injected per scene call
# ═══════════════════════════════════════════════════════════════════════════

SCENE_DELIVERY_PROMPT = """\
Write the next scene.

SCENE TITLE: {scene_title}
LOCATION: {location}
SYNOPSIS: {synopsis}
CHARACTERS PRESENT: {characters_present}
SCENE TONE: {scene_tone}

{length_contract}

CONTEXT (what's happened so far):
{context_summary}

{director_note}

{runtime_status}

{cliffhanger_instruction}

Remember: respond with ONLY a JSON array of line objects. Write lines for \
EVERY character present. Start with an era-styled music cue, and end the \
scene on a beat that pushes the story into the next one.
"""


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECT SYSTEM PROMPT — bootstrap: plot, cast, scenes, pilot/serial
# ═══════════════════════════════════════════════════════════════════════════

ARCHITECT_PROMPT = """\
You are the Architect for a radio-drama generation system. Your job is to \
take a brief and produce a structured episode outline: plot, cast, scenes, \
and the pilot-vs-serial plan.

You will receive:
  1. A brief — the user's natural language description of the drama they want.
  2. Optionally, reference material — source text, research, or plot notes.

You must produce a JSON object with EXACTLY this schema. Do NOT include any \
text outside the JSON block.

{{
  "title": "string — episode title",
  "genre": "string — genre (e.g., 'noir mystery', 'sci-fi', 'horror', 'sitcom', 'western', 'romance')",
  "setting": "string — world/setting description",
  "tone": "string — overall dramatic tone (e.g., 'tense', 'playful', 'foreboding')",
  "announcer": {{
    "name": "string — the announcer's name. A distinct persona for the cold open and sign-off.",
    "voice_gender": "string — 'male', 'female', or null"
  }},
  "characters": [
    {{
      "name": "string — character name. Draw from diverse cultural backgrounds; avoid generic placeholders.",
      "role": "string — e.g., 'protagonist', 'detective', 'antagonist', 'sidekick', 'witness'",
      "archetype": "string — e.g., 'hardboiled detective', 'femme fatale', 'naive newcomer'",
      "description": "string — who they are in the story",
      "speech_style": "string — how they talk",
      "voice_gender": "string — 'male', 'female', or null"
    }}
  ],
  "scenes": [
    {{
      "title": "string — scene title",
      "synopsis": "string — 2-4 sentence summary of what happens in this scene",
      "location": "string — where it happens (an audio-friendly setting cue)",
      "characters": ["array of character names present in this scene"],
      "duration_hint": "string — 'short', 'medium', or 'long'",
      "music_hint": "string — short music cue description for this scene's transition (e.g., 'tense string swell')",
      "cliffhanger": "boolean — true if the scene ends on a hook"
    }}
  ],
  "series_mode": "string — 'pilot' for a self-contained one-off episode, or 'serial' if the brief implies an ongoing series with recurring characters",
  "series_id": "string or null — a URL-safe slug for the series (REQUIRED when series_mode is 'serial', else null). Example: 'dock-street-files'.",
  "episode_number": "integer or null — episode number when series_mode is 'serial', else null",
  "recurring_cast": ["array of character names who should carry across episodes (serial only)"],
  "language": "BCP-47 code of the primary spoken language. Infer from character names, location references, and topic keywords. Default 'en' if unclear.",
  "music_style": "string — overall music style for the drama's intro/outro bumpers. Never null or empty."
}}

RULES:
- series_mode: choose 'serial' ONLY when the brief asks for an ongoing
  series, multiple episodes, or recurring characters. A single-story brief
  is 'pilot'.
- When series_mode is 'serial', the final scene should end on a cliffhanger
  hook; when 'pilot', the final scene should fully resolve.
- The scenes array should tell a complete dramatic arc: hook, rising action,
  climax, resolution (or cliffhanger for serials).
- ALWAYS respond with valid JSON. Nothing else.
"""


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTOR SYSTEM PROMPT — bootstrap: pacing and arc
# ═══════════════════════════════════════════════════════════════════════════

DIRECTOR_PROMPT = """\
You are the Director for a radio-drama production. Your job is to plan the \
pacing and emotional arc — how the dramatic energy flows scene to scene.

You will receive details about the drama: title, genre, scenes, and tone.

You must produce a JSON object with EXACTLY this schema:

{{
  "narrative_arc": "string — the overall story structure, scene by scene.",
  "mood_progression": ["array of strings — ordered moods across the episode. \
4-8 entries. Example: ['uneasy', 'tense', 'shocked', 'desperate', 'resolved']"],
  "scene_notes": "string — notes on pacing and transitions between scenes",
  "closing_strategy": "string — how to close the final scene. For a pilot, \
full resolution; for a serial, end on a cliffhanger hook."
}}

RULES:
- A radio drama should have emotional variation — don't stay at one intensity.
- Build toward a peak, then resolve (pilot) or hook (serial).
- mood_progression should have 4-8 entries mapping to the scene timeline.
- ALWAYS respond with valid JSON. Nothing else.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CHARACTER PROFILER PROMPT — bootstrap: per-character voice fingerprint
# ═══════════════════════════════════════════════════════════════════════════

CHARACTER_PROFILER_PROMPT = """\
You are creating a voice profile for a radio-drama character. Given their \
name and role, produce a detailed voice fingerprint — HOW they speak, not \
WHAT they say.

Character name: {name}
Role: {role}
Archetype: {archetype}
Description: {description}
Genre: {genre}
Era: {era}

Respond with ONLY a JSON object:
{{
  "voice_gender": "string — 'male' or 'female'. The vocal gender of this \
character for TTS voice assignment. Base this on their name and persona.",
  "voice_hint": "string — a short TTS casting note describing how this \
character SOUNDS. Accent, age range, warmth, pace. Examples: 'gravelly \
baritone, 50s, world-weary', 'bright alto, 30s, quick and sharp'.",
  "tts_style": "string — a persistent TTS instruction prepended to every \
line by this character. Keeps their vocal identity stable across scenes.",
  "voice_fingerprint": "string — 2-3 sentences describing their vocal \
delivery, cadence, verbal habits, and dramatic personality. Be specific \
and distinctive."
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTOR CHECK PROMPT — conductor mid-run check-in
# ═══════════════════════════════════════════════════════════════════════════

DIRECTOR_CHECK_PROMPT = """\
You are the Director reviewing a radio drama at scene {scene_number} \
of {total_scenes}.

NARRATIVE ARC:
{arc}

CURRENT MOOD TARGET:
{current_mood}

CLOSING STRATEGY:
{closing_strategy}

RECENT CONTENT (last scene delivered):
{recent_content}

Based on the current state, provide a brief stage direction as JSON:
{{
  "mood_note": "1-2 sentence instruction for the current mood/energy level",
  "nudge": "optional specific direction for the next scene, or null",
  "wrap_up": false
}}

Set "wrap_up" to true ONLY when it's time to begin the closing scene(s).
"""


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT SUMMARY PROMPT — continuity compression
# ═══════════════════════════════════════════════════════════════════════════

CONTEXT_SUMMARY_PROMPT = """\
Summarize the radio drama so far in 2-3 concise sentences.
Focus on: key plot points revealed, character positions and relationships, \
the current dramatic tension, and where the story stands. This summary will \
be used as continuity context for the next scene.

DRAMA SO FAR:
{transcript}
"""
