"""Audiobook bootstrap — brief → fully populated AudiobookConfig.

The audiobook bootstrap plans the chapter structure and narrator voice:
    1. ARCHITECT call: brief → title, genre, chapters, narrator, setting
    2. NARRATOR PROFILER call: name + style → voice fingerprint
    3. DIRECTOR call: chapter list → pacing arc

CHAIN:
    brief → Architect → Narrator profiler → Director → AudiobookConfig

EXAMPLE:
    bootstrap = AudiobookBootstrap(client)
    config = bootstrap.from_brief("A short sci-fi story about first contact")
"""

from __future__ import annotations

import json

from ...core.bootstrap import BaseShowBootstrap
from ...core.client import DeepSeekClient
from ...core.config import ParticipantProfile, ShowHostConfig
from ...core.names import build_curated_name_map, remap_name, remap_names
from ...core.search import BaseSearchProvider
from .config import (
    AudiobookConfig,
    ChapterConfig,
    AudiobookDirectorBrief,
    NarratorConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECT SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

AUDIOBOOK_ARCHITECT_PROMPT = """\
You are the Architect for an audiobook generation system. Your job is to take
a brief and produce a structured audiobook outline: chapters, narrator, genre,
and the overall narrative plan.

You will receive:
  1. A brief — the user's natural language description of the audiobook they want.
  2. Optionally, reference material — source text, research, or plot notes.

You must produce a JSON object with EXACTLY this schema. Do NOT include any
text outside the JSON block.

{{
  "title": "string — audiobook title",
  "genre": "string — genre (e.g., 'science fiction', 'history', 'self-help', 'thriller')",
  "topic": "string — the core subject or theme",
  "tone": "string — overall narrative tone (e.g., 'contemplative', 'urgent', 'warm')",
  "setting": "string — world or setting description",
  "narrator": {{
    "name": "string — narrator's name (can be a fictional persona). Pick a memorable, varied name. AVOID overused defaults like 'Sarah Chen', 'Marcus Webb', 'Elena Vasquez', 'Auntie Star', or any 'Chen' surname. Draw from diverse cultural backgrounds.",
    "style": "string — narration style description"
  }},
  "characters": [
    {{
      "name": "string — character name. Draw from diverse cultural backgrounds; avoid generic placeholder names.",
      "role": "string — role in the story (e.g., 'protagonist', 'antagonist', 'mentor')",
      "voice_gender": "string — 'male', 'female', or null"
    }}
  ],
  "chapters": [
    {{
      "title": "string — chapter title",
      "synopsis": "string — 2-4 sentence summary of what happens in this chapter",
      "key_points": ["array of narrative beats in order — each is a sentence describing a scene or point"],
      "tone": "string — chapter-specific tone",
      "duration_hint": "string — 'short', 'medium', or 'long'",
      "target_beats": "integer — how many generation beats (1=short, 2-3=standard, 4+=deep). Default 2.",
      "characters": ["array of character names who appear in this chapter"]
    }}
  ],
  "language": "BCP-47 code of the primary spoken language. Infer from character names, location references, and topic keywords in the brief. Examples: 'es' (Spanish), 'fr' (French), 'de' (German), 'pt' (Portuguese), 'it' (Italian), 'zh' (Mandarin). Default 'en' if unclear.",
  "music_style": "string — music style for this audiobook's intro/outro bumpers. Infer from the genre, tone, and setting. Be specific and evocative. Examples: 'atmospheric orchestral, cinematic' (thriller), 'gentle acoustic guitar bed' (self-help), 'sweeping fantasy orchestra with choir' (epic fantasy). Never null or empty.",
  "perspective": "string or null — the primary geographic, cultural, or institutional vantage point from which this audiobook is narrated. This anchors observer-relative language ('the northern border', 'our city', 'the empire to the east') so downstream fact-checkers do not flag perspectival phrasing as fabrication. Examples: 'from Guatemala' (Mexico is to its NORTH, El Salvador/Honduras to its south), 'from the perspective of the Roman Senate, late Republic', 'from a 19th-century New England whaling town'. For fiction, give the in-world vantage point. Use null ONLY if the work is genuinely placeless / omniscient with no positional framing."
- Characters array should list all named characters with roles.
- ALWAYS respond with valid JSON. Nothing else.
"""


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTOR SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

AUDIOBOOK_DIRECTOR_PROMPT = """\
You are the Director for an audiobook production. Your job is to plan the
pacing and emotional arc — how the narrative energy flows chapter to chapter.

You will receive details about the audiobook: title, genre, chapters, and tone.

You must produce a JSON object with EXACTLY this schema:

{{
  "narrative_arc": "string — the overall story structure. Example: 'Open with \
a quiet hook, build tension through the middle chapters, peak at the climax \
in chapter N, then resolve with a reflective denouement.'",
  "mood_progression": ["array of strings — ordered moods across the book. \
4-8 entries. Example: ['curious', 'uneasy', 'tense', 'desperate', 'cathartic', 'peaceful']"],
  "chapter_notes": "string — notes on pacing and transitions between chapters",
  "closing_strategy": "string — how to close the audiobook. Example: 'The final \
chapter slows to a reflective pace, the narrator's tone shifts to warm and \
contemplative, ending with a moment of quiet resolution.'"
}}

RULES:
- An audiobook should have emotional variation — don't stay at one intensity.
- Build toward a peak, then resolve. Classic dramatic structure.
- mood_progression should have 4-8 entries mapping to the narrative timeline.
- ALWAYS respond with valid JSON. Nothing else.
"""


# ═══════════════════════════════════════════════════════════════════════════
# NARRATOR PROFILER PROMPT
# ═══════════════════════════════════════════════════════════════════════════

NARRATOR_PROFILER_PROMPT = """\
You are creating a voice profile for an audiobook narrator. Given their name
and style, produce a detailed voice fingerprint — HOW they narrate, not WHAT
they say.

Narrator name: {name}
Narrator style: {style}
Genre: {genre}
Tone: {tone}

Respond with ONLY a JSON object:
{{
  "voice_gender": "string — 'male' or 'female'. The vocal gender of this \
narrator for TTS voice assignment. Base this on their name and persona.",
  "voice_hint": "string — a short TTS casting note describing how this narrator \
SOUNDS. Accent, age range, warmth, pace. Examples: 'warm baritone, 50s, \
BBC documentary cadence', 'clear mezzo-soprano, 40s, steady and unhurried', \
'bright tenor, 30s, intimate bedside-manner delivery'.",
  "tts_style": "string — a persistent TTS instruction prepended to every \
utterance by this narrator. Keeps their vocal identity stable across all chapters. \
Examples: 'Narrate with quiet authority and unhurried gravitas, letting each \
sentence breathe.', 'Read with warm intimacy, as if telling a story by firelight.'",
  "voice_fingerprint": "string — 2-3 sentences describing their vocal delivery, \
cadence, verbal habits, and narrative personality. Be specific and distinctive."
}}
"""


class AudiobookBootstrap(BaseShowBootstrap):
    """Orchestrates the brief-to-config bootstrap chain for audiobooks.

    Chain: brief → [Researcher] → Architect → Narrator profiler → Director → config
    """

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        super().__init__(client, api_key=api_key)

    def from_brief(
        self,
        brief: str,
        *,
        reference_material: str | None = None,
        fact_check: str = "off",
        overrides: dict | None = None,
        auto_research: bool = False,
        search_provider: BaseSearchProvider | None = None,
        target_words: int | None = None,
        memory=None,
        series_context: str | None = None,
    ) -> AudiobookConfig:
        """Run the full bootstrap chain: brief → fully populated AudiobookConfig.

        Args:
            brief:              Natural language description of the audiobook.
            reference_material: Source text, plot notes, or research.
            fact_check:         "off", "quick", or "thorough".
            overrides:          Dict of AudiobookConfig field names to override.
            auto_research:      If True, research the topic before generating.
            search_provider:    Search provider for web research.
            target_words:       Target word count for the audiobook.

        Returns:
            A fully populated AudiobookConfig.
        """
        overrides = overrides or {}

        if series_context:
            brief = f"{series_context}\n\n{brief}"

        # ── Stage 0: Auto-research (optional) ────────────────────────
        if auto_research and not reference_material:
            from ...core.search import build_research_provider
            from ..debates.researcher import DebateResearcher
            effective_provider = build_research_provider(search_provider, memory)
            if effective_provider is None:
                researcher = DebateResearcher(self._client)
            else:
                researcher = DebateResearcher(self._client, effective_provider)
            reference_material = researcher.run(
                brief,
                perspective=overrides.get("perspective"),
            ) or None

        # ── Stage 1: Architect ──────────────────────────────────────
        config = self._run_architect(
            brief, reference_material, fact_check, target_words=target_words,
        )

        # ── Stage 2: Narrator profiling ──────────────────────────────
        if config.narrator and config.narrator.name:
            config.narrator = self._profile_narrator(
                config.narrator,
                genre=config.genre or "general",
                tone=config.tone or "engaging",
            )

        # ── Stage 3: Director ─────────────────────────────────────────
        config.director_brief = self._run_director(config)

        # ── Stage 4: Apply overrides ─────────────────────────────────
        self._apply_overrides(config, overrides)

        return config

    def _run_architect(
        self,
        brief: str,
        reference_material: str | None,
        fact_check: str,
        *,
        target_words: int | None = None,
    ) -> AudiobookConfig:
        """Call the Architect LLM to produce a skeleton AudiobookConfig."""
        user_parts = [f"BRIEF:\n{brief}"]
        if reference_material:
            user_parts.append(f"\nREFERENCE MATERIAL:\n{reference_material}")
        if target_words:
            # Derive chapter count and beats-per-chapter from the word budget
            # so the architect gets explicit, non-contradicting targets.
            # Formula mirrors what the system prompt documents:
            #   ideal_chapters    = max(3, target_words // 2000)
            #   beats_per_chapter = max(2, target_words // (200 * ideal_chapters))
            ideal_chapters = max(3, target_words // 2000)
            beats_per_chapter = max(2, target_words // (200 * ideal_chapters))
            user_parts.append(
                f"\nTARGET LENGTH: {target_words:,} words. "
                f"Plan approximately {ideal_chapters} chapters with "
                f"~{beats_per_chapter} beats each "
                f"({ideal_chapters} \u00d7 {beats_per_chapter} \u00d7 ~200 words/beat "
                f"\u2248 {ideal_chapters * beats_per_chapter * 200:,} words). "
                f"Adjust chapter count or beats-per-chapter as needed to reach "
                f"the target — do NOT use fewer total beats than "
                f"{ideal_chapters * beats_per_chapter}."
            )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": AUDIOBOOK_ARCHITECT_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            temperature=0,
        )

        return self._parse_architect_response(
            raw, brief=brief, reference_material=reference_material,
            fact_check=fact_check,
        )

    def _parse_architect_response(
        self,
        raw_json: str,
        *,
        brief: str = "",
        reference_material: str | None = None,
        fact_check: str = "off",
    ) -> AudiobookConfig:
        """Parse the Architect's JSON into an AudiobookConfig."""
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        # Find the JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Audiobook Architect produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        narrator_data = data.get("narrator", {})
        name_specs = [
            (narrator_data.get("name", ""), None, "audiobook.narrator"),
        ]
        name_specs.extend(
            (c.get("name", ""), c.get("voice_gender"), f"audiobook.character.{idx}")
            for idx, c in enumerate(data.get("characters", []))
        )
        name_map = build_curated_name_map(name_specs, brief=brief)

        # Build narrator.
        narrator = NarratorConfig(
            name=remap_name(narrator_data.get("name", ""), name_map) or "",
            style=narrator_data.get("style"),
        )

        # Build characters.
        characters = []
        for c in data.get("characters", []):
            characters.append(ParticipantProfile(
                name=remap_name(c.get("name", ""), name_map) or "",
                role=c.get("role", "character"),
                voice_gender=c.get("voice_gender"),
            ))

        # Build chapters.
        chapters = []
        for ch in data.get("chapters", []):
            chapters.append(ChapterConfig(
                title=ch.get("title", ""),
                synopsis=ch.get("synopsis"),
                key_points=ch.get("key_points", []),
                tone=ch.get("tone"),
                duration_hint=ch.get("duration_hint", "medium"),
                target_beats=ch.get("target_beats", 2),
                characters=remap_names(ch.get("characters", []), name_map),
            ))

        return AudiobookConfig(
            brief=brief,
            reference_material=reference_material,
            fact_check=fact_check,
            title=data.get("title"),
            topic=data.get("topic"),
            genre=data.get("genre"),
            tone=data.get("tone"),
            setting=data.get("setting"),
            narrator=narrator,
            characters=characters,
            chapters=chapters,
            language=(data.get("language") or "en").strip().lower(),
            music_style=data.get("music_style", ""),
            perspective=(data.get("perspective") or None),
        )

    def _profile_narrator(
        self,
        narrator: NarratorConfig,
        *,
        genre: str,
        tone: str,
    ) -> NarratorConfig:
        """Give the narrator a voice_fingerprint via LLM."""
        prompt = NARRATOR_PROFILER_PROMPT.format(
            name=narrator.name,
            style=narrator.style or "engaging and clear",
            genre=genre,
            tone=tone,
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": "You produce JSON voice profiles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
            narrator.voice_fingerprint = data.get("voice_fingerprint", "")
            if data.get("voice_gender"):
                narrator.voice_gender = data["voice_gender"]
            if data.get("voice_hint"):
                narrator.voice_hint = data["voice_hint"]
            if data.get("tts_style"):
                narrator.tts_style = data["tts_style"]
        except (json.JSONDecodeError, TypeError):
            narrator.voice_fingerprint = "(warm, clear audiobook narration)"

        return narrator

    def _run_director(self, config: AudiobookConfig) -> AudiobookDirectorBrief:
        """Call the Director LLM to plan the narrative arc."""
        chapter_list = "\n".join(
            f"  {i+1}. {ch.title} ({ch.duration_hint}, {ch.tone or config.tone or 'neutral'})"
            for i, ch in enumerate(config.chapters)
        )

        user_message = (
            f"TITLE: {config.title or 'Untitled'}\n"
            f"GENRE: {config.genre or 'general'}\n"
            f"TONE: {config.tone or 'engaging'}\n"
            f"TOTAL CHAPTERS: {len(config.chapters)}\n"
            f"CHAPTERS:\n{chapter_list}"
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": AUDIOBOOK_DIRECTOR_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        return self._parse_director_response(raw)

    def _parse_director_response(self, raw_json: str) -> AudiobookDirectorBrief:
        """Parse the Director's JSON into an AudiobookDirectorBrief."""
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        # Find JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Audiobook Director produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        return AudiobookDirectorBrief(
            narrative_arc=data.get("narrative_arc", ""),
            mood_progression=data.get("mood_progression", []),
            chapter_notes=data.get("chapter_notes"),
            closing_strategy=data.get("closing_strategy"),
        )
