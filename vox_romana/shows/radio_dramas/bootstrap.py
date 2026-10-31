"""Radio drama bootstrap — brief → fully populated RadioDramaConfig.

The radio-drama bootstrap plans the plot, cast, and scenes, then profiles
every voice and plans the dramatic arc. It mirrors the audiobook bootstrap
chain but adds the pilot-vs-serial planner (shot 5.1-c).

CHAIN:
    brief → Architect → Character profiler (announcer + each character)
          → Director → RadioDramaConfig

EXAMPLE:
    bootstrap = RadioDramaBootstrap(client)
    config = bootstrap.from_brief("A noir mystery set in 1940s Philadelphia")
"""

from __future__ import annotations

import json

from ...core.bootstrap import BaseShowBootstrap
from ...core.client import DeepSeekClient
from ...core.names import build_curated_name_map, remap_name, remap_names
from ...core.search import BaseSearchProvider
from .config import (
    CharacterConfig,
    RadioDramaConfig,
    RadioDramaDirectorBrief,
    SceneConfig,
)
from .prompts import (
    ARCHITECT_PROMPT,
    CHARACTER_PROFILER_PROMPT,
    DIRECTOR_PROMPT,
)


class RadioDramaBootstrap(BaseShowBootstrap):
    """Orchestrates the brief-to-config bootstrap chain for radio dramas.

    Chain: brief → Architect → Character profiler (per cast member) →
    Director → config. The architect output carries the pilot-vs-serial
    decision; the parser records it onto the config verbatim.
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
    ) -> RadioDramaConfig:
        """Run the full bootstrap chain: brief → fully populated RadioDramaConfig.

        Args:
            brief:              Natural language description of the drama.
            reference_material: Source text, plot notes, or research.
            fact_check:         "off", "quick", or "thorough".
            overrides:          Dict of RadioDramaConfig field names to override.
            auto_research:      If True, research the topic before generating.
            search_provider:    Search provider for web research.
            target_words:       Target word count for the episode.
            memory:             Optional memory handle for research.
            series_context:     Optional series-bible context prepended to the brief.

        Returns:
            A fully populated RadioDramaConfig.
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

        # ── Stage 2: Character profiling ────────────────────────────
        if config.announcer and config.announcer.name:
            config.announcer = self._profile_character(
                config.announcer,
                genre=config.genre or "drama",
                era=config.era,
            )
        profiled: list[CharacterConfig] = []
        for character in config.characters:
            profiled.append(self._profile_character(
                character,
                genre=config.genre or "drama",
                era=config.era,
            ))
        config.characters = profiled

        # ── Stage 3: Director ───────────────────────────────────────
        config.director_brief = self._run_director(config)

        # ── Stage 4: Apply overrides ────────────────────────────────
        self._apply_overrides(config, overrides)

        return config

    def _run_architect(
        self,
        brief: str,
        reference_material: str | None,
        fact_check: str,
        *,
        target_words: int | None = None,
    ) -> RadioDramaConfig:
        """Call the Architect LLM to produce a skeleton RadioDramaConfig."""
        user_parts = [f"BRIEF:\n{brief}"]
        if reference_material:
            user_parts.append(f"\nREFERENCE MATERIAL:\n{reference_material}")
        if target_words:
            # Derive scene count from the word budget so the architect gets
            # explicit, non-contradicting targets. ~350 words per scene is a
            # reasonable radio-drama scene length.
            ideal_scenes = max(3, target_words // 350)
            user_parts.append(
                f"\nTARGET LENGTH: {target_words:,} words. "
                f"Plan approximately {ideal_scenes} scenes "
                f"(~{ideal_scenes} × ~350 words/scene "
                f"≈ {ideal_scenes * 350:,} words). Adjust scene count as "
                f"needed to reach the target — do NOT use fewer than "
                f"{ideal_scenes} scenes."
            )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": ARCHITECT_PROMPT},
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
    ) -> RadioDramaConfig:
        """Parse the Architect's JSON into a RadioDramaConfig.

        The pilot-vs-serial planner fields (series_mode, series_id,
        episode_number, recurring_cast) are recorded verbatim from the
        architect output; config.validate() enforces their consistency
        (serial requires series_id).
        """
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
                f"RadioDrama Architect produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        # Curate names so cast members are diverse and non-generic.
        announcer_data = data.get("announcer", {})
        name_specs = [
            (announcer_data.get("name", ""), None, "radio_drama.announcer"),
        ]
        name_specs.extend(
            (c.get("name", ""), c.get("voice_gender"), f"radio_drama.character.{idx}")
            for idx, c in enumerate(data.get("characters", []))
        )
        name_map = build_curated_name_map(name_specs, brief=brief)

        announcer = None
        if announcer_data:
            announcer = CharacterConfig(
                name=remap_name(announcer_data.get("name", ""), name_map) or "",
                role="announcer",
                voice_gender=announcer_data.get("voice_gender"),
            )

        characters = []
        for c in data.get("characters", []):
            characters.append(CharacterConfig(
                name=remap_name(c.get("name", ""), name_map) or "",
                role=c.get("role", "character"),
                archetype=c.get("archetype"),
                description=c.get("description"),
                speech_style=c.get("speech_style"),
                voice_gender=c.get("voice_gender"),
            ))

        scenes = []
        for sc in data.get("scenes", []):
            scenes.append(SceneConfig(
                title=sc.get("title", ""),
                synopsis=sc.get("synopsis"),
                location=sc.get("location"),
                characters=remap_names(sc.get("characters", []), name_map),
                duration_hint=sc.get("duration_hint", "medium"),
                music_hint=sc.get("music_hint"),
                cliffhanger=bool(sc.get("cliffhanger", False)),
            ))

        series_mode = data.get("series_mode", "pilot")
        if series_mode not in ("pilot", "serial"):
            series_mode = "pilot"

        return RadioDramaConfig(
            brief=brief,
            reference_material=reference_material,
            fact_check=fact_check,
            title=data.get("title"),
            genre=data.get("genre"),
            era="1940s",  # overridden by --era in runners (shot 5.1-e)
            setting=data.get("setting"),
            tone=data.get("tone"),
            announcer=announcer,
            characters=characters,
            scenes=scenes,
            language=(data.get("language") or "en").strip().lower(),
            music_style=data.get("music_style", ""),
            series_mode=series_mode,
            series_id=data.get("series_id") or None,
            episode_number=data.get("episode_number"),
            recurring_cast=remap_names(data.get("recurring_cast", []), name_map),
        )

    def _profile_character(
        self,
        character: CharacterConfig,
        *,
        genre: str,
        era: str,
    ) -> CharacterConfig:
        """Give a character a voice_fingerprint via LLM."""
        prompt = CHARACTER_PROFILER_PROMPT.format(
            name=character.name,
            role=character.role,
            archetype=character.archetype or "unspecified",
            description=character.description or "unspecified",
            genre=genre,
            era=era,
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
            character.voice_fingerprint = data.get("voice_fingerprint", "")
            if data.get("voice_gender"):
                character.voice_gender = data["voice_gender"]
            if data.get("voice_hint"):
                character.voice_hint = data["voice_hint"]
            if data.get("tts_style"):
                character.tts_style = data["tts_style"]
        except (json.JSONDecodeError, TypeError):
            character.voice_fingerprint = "(distinct dramatic voice)"

        return character

    def _run_director(self, config: RadioDramaConfig) -> RadioDramaDirectorBrief:
        """Call the Director LLM to plan the dramatic arc."""
        scene_list = "\n".join(
            f"  {i + 1}. {sc.title} ({sc.duration_hint}, "
            f"{sc.synopsis or 'no synopsis'})"
            for i, sc in enumerate(config.scenes)
        )

        user_message = (
            f"TITLE: {config.title or 'Untitled'}\n"
            f"GENRE: {config.genre or 'drama'}\n"
            f"TONE: {config.tone or 'engaging'}\n"
            f"SERIES MODE: {config.series_mode}\n"
            f"TOTAL SCENES: {len(config.scenes)}\n"
            f"SCENES:\n{scene_list}"
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": DIRECTOR_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        return self._parse_director_response(raw)

    def _parse_director_response(self, raw_json: str) -> RadioDramaDirectorBrief:
        """Parse the Director's JSON into a RadioDramaDirectorBrief."""
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
                f"RadioDrama Director produced invalid JSON. First 200 chars: "
                f"{raw_json[:200]!r}\nParse error: {e}"
            ) from e

        # scene_notes is a free-text string in the director output; config
        # carries an int-keyed dict for segment-specific notes. The conductor
        # can stash notes separately, so we keep the string on narrative
        # context via narrative_arc and leave scene_notes empty here.
        return RadioDramaDirectorBrief(
            narrative_arc=data.get("narrative_arc", ""),
            mood_progression=data.get("mood_progression", []),
            closing_strategy=data.get("closing_strategy"),
        )
