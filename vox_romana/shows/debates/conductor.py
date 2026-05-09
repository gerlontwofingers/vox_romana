"""The Debate Conductor — the runtime engine that generates a debate.

This is where the magic happens. The conductor takes a fully populated
DebateConfig (from the bootstrap chain or manually built) and generates
a complete debate by orchestrating multiple LLM agents in conversation.

MENTAL MODEL — A JAZZ ENSEMBLE:

    Think of the conductor as a bandleader. Each musician (agent) has their
    own personality, style, and sheet music (system prompt). The bandleader
    doesn't play any instruments — they point at who plays next, set the
    tempo, and keep everyone roughly on the same song.

    - The MODERATOR is the rhythm section — keeps time, calls solos.
    - The PARTICIPANTS are soloists — each with a distinct voice.
    - The DIRECTOR is the composer watching from the booth — occasionally
      sends a note saying "bring the energy up" or "it's time to resolve."
    - The AUDIENCE (if present) is the crowd — reactive, not scripted.

THE TURN LOOP:

    The loop is STRICTLY SEQUENTIAL. Agent B's response depends on Agent A's
    output. There is no parallelism here, and that's correct — a conversation
    is inherently serial.

    Each iteration:
    1. Decide who speaks next (moderator decides, or round-robin fallback)
    2. Build context for the selected speaker (summary + recent turns + director note)
    3. The speaker generates their line as JSON
    4. Parse the JSON into a DebateEvent
    5. Append to the transcript
    6. Every N turns: compress the older transcript into a summary
    7. Every check_interval turns: ask the Director for a stage note
    8. Repeat until target_turns reached, then trigger closing sequence

CONTEXT WINDOW MANAGEMENT:

    LLMs have finite context windows. A 35-turn debate would overflow if
    we stuffed every utterance into every prompt. So we use a sliding window:

    ┌─────────────────────────────────────────────────────────┐
    │ [System prompt]  ~500 tokens, fixed                     │
    │ [Compressed summary of turns 1..N-K]  ~300-500 tokens   │
    │ [Director's current mood note]  ~50 tokens              │
    │ [Last K verbatim turns]  ~800-1200 tokens               │
    │ [Turn prompt: "It's your turn, respond to X"]  ~50 tok  │
    └─────────────────────────────────────────────────────────┘
    Total per call: ~1700-2300 tokens input + ~150-300 tokens output
    Total for 35-turn debate: ~50-80k tokens ≈ $0.01-0.03 at DeepSeek rates

OUTPUT FORMAT:

    The conductor produces a DebateConductorResult containing:
    - events: list of DebateEvent (the full structured transcript)
    - config: the DebateConfig that was used
    - transcript: human-readable lines for debugging
    - A to_merged_debate() method for serialization

USAGE:

    # Option A — Auto from brief (bootstrap + conduct in one shot):
    conductor = DebateConductor.from_brief(
        "Should cities ban cars from downtown?",
        reference_material="Barcelona superblocks data...",
    )
    result = conductor.run()

    # Option B — From an existing config:
    config = DebateBootstrap().from_brief("...")
    conductor = DebateConductor(config)
    result = conductor.run()

    # Save to JSON:
    merged = result.to_merged_debate()
    DebateSerializer.save(merged, "output/debate.json")

    # Inspect:
    for event in result.events:
        print(f"[{event.speaker_name}] {event.text}")

TESTING:

    The conductor accepts a `client` parameter. In tests, inject a mock
    DeepSeekClient whose `.chat()` returns canned JSON responses. The
    conductor never creates its own client — dependency injection all the way.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...core.client import DeepSeekClient
from ...core.conductor import BaseConductor, ConductorResult
from ...core.fact_checker import FactChecker
from ...core.session import AgentSession
from .bootstrap import DebateBootstrap
from .config import DebateConfig, ParticipantProfile
from .models import DebateEvent, DebateSegment, MergedDebate
from .prompts import (
    CONTEXT_SUMMARY_PROMPT,
    DIRECTOR_CHECK_PROMPT,
    EMCEE_ROLE_INSTRUCTIONS,
    MODERATOR_SYSTEM_PROMPT,
    NARRATOR_ROLE_INSTRUCTIONS,
    PARTICIPANT_SYSTEM_PROMPT,
    SHOW_HOST_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from ...core.lens import LLMLensRack

logger = logging.getLogger(__name__)


def _build_language_instruction(language: str | None) -> str:
    """Return a prompt instruction for a character's assigned language.

    Returns an empty string when *language* is ``None`` or ``"en"`` so that
    the ``{language_instruction}`` placeholder leaves no trace in the prompt.
    For any other language code, returns an explicit instruction block that
    tells the LLM to write all dialogue in that language.
    """
    if not language or language == "en":
        return ""
    from ...core.languages import LANGUAGE_NAMES
    lang_name = LANGUAGE_NAMES.get(language, language)
    return (
        f"\nLANGUAGE CONSTRAINT (critical):\n"
        f"- You MUST write ALL of your dialogue in {lang_name} ({language}).\n"
        f"- Every word in the \"text\" field must be in {lang_name}.\n"
        f"- The debate topic, names, and other speakers' words may be in English —\n"
        f"  but YOUR spoken lines are always in {lang_name}.\n"
        f"- JSON keys and tone/volume/pace values remain in English as usual."
    )


# ─────────────────────────────────────────────────────────────────────────
# How many recent utterances to include verbatim in each agent's context.
# Older turns get compressed into a summary. This is the "sliding window."
DEFAULT_RECENT_WINDOW = 6

# How often to re-compress the summary (every N turns).
DEFAULT_COMPRESS_INTERVAL = 8

# How often the moderator interjects (every N participant turns).
DEFAULT_MODERATOR_INTERVAL = 3


# ═══════════════════════════════════════════════════════════════════════════
# DebateConductorResult
# ═══════════════════════════════════════════════════════════════════════════
# Extends ConductorResult with debate-specific typed output and a
# convenience method to produce a MergedDebate for serialization.

@dataclass
class DebateConductorResult(ConductorResult):
    """Typed result from a DebateConductor run.

    Adds:
        debate_events: The full list of DebateEvent objects (typed, structured).
        debate_id:     A unique ID for this debate run.

    The base class `events` holds the raw dicts. `debate_events` holds the
    typed DebateEvent objects built from those dicts. Use whichever is
    more convenient.

    USAGE:
        result = conductor.run()

        # Iterate typed events:
        for event in result.debate_events:
            print(f"[{event.speaker_name}] {event.text}")

        # Convert to MergedDebate for serialization:
        merged = result.to_merged_debate()
        DebateSerializer.save(merged, "output/debate.json")
    """

    debate_events: list[DebateEvent] = field(default_factory=list)
    debate_id: str = ""

    def to_merged_debate(self, series_id: str | None = None) -> MergedDebate:
        """Convert this result into a MergedDebate for serialization.

        This bridges the conductor's output to the existing pipeline.
        The pipeline was designed for parsing externally-generated text,
        but the conductor generates events directly. This method packages
        them into the same MergedDebate format the serializer expects.

        Returns:
            A MergedDebate ready for DebateSerializer.save().
        """
        config = self.config
        if not isinstance(config, DebateConfig):
            raise TypeError("Expected a DebateConfig, got {type(config)}")

        events_tuple = tuple(self.debate_events)

        # Build a single DebateSegment wrapping all events.
        chunk = DebateSegment(
            template_id="conductor",
            debate_id=self.debate_id,
            title=config.title or "Untitled Debate",
            chunk_order=1,
            is_final_chunk=True,
            participants=tuple(p.name for p in config.participants),
            moderators=(
                (config.moderator.name,) if config.moderator else ()
            ),
            hosts=(
                (config.show_host.name,) if config.show_host else ()
            ),
            audience_present=(
                config.audience.present if config.audience else False
            ),
            audience_participation=(
                config.audience.participation if config.audience else None
            ),
            topic=config.topic,
            debate_format=config.debate_format,
            setting=config.setting,
            moderator_mode=(
                config.moderator.mode if config.moderator else None
            ),
            events=events_tuple,
        )

        # Build voice casting from profile voice_gender fields.
        voice_casting: dict[str, str] = {}
        voice_hints: dict[str, str] = {}
        tts_styles: dict[str, str] = {}
        character_langs: dict[str, str] = {}
        for p in config.participants:
            if p.voice_gender:
                voice_casting[p.name] = p.voice_gender
            if p.voice_hint:
                voice_hints[p.name] = p.voice_hint
            if p.tts_style:
                tts_styles[p.name] = p.tts_style
            if p.language and p.language != "en":
                character_langs[p.name] = p.language
        if config.moderator and config.moderator.name:
            # Moderators don't have profiles, default to male.
            voice_casting.setdefault(config.moderator.name, "male")
        if config.show_host and config.show_host.voice_gender:
            voice_casting[config.show_host.name] = config.show_host.voice_gender
        if config.show_host and config.show_host.voice_hint:
            voice_hints[config.show_host.name] = config.show_host.voice_hint
        if config.show_host and config.show_host.tts_style:
            tts_styles[config.show_host.name] = config.show_host.tts_style

        return MergedDebate(
            template_id="conductor",
            debate_id=self.debate_id,
            title=config.title or "Untitled Debate",
            chunks=(chunk,),
            events=events_tuple,
            parser_names=("conductor",),
            participants=tuple(p.name for p in config.participants),
            moderators=(
                (config.moderator.name,) if config.moderator else ()
            ),
            hosts=(
                (config.show_host.name,) if config.show_host else ()
            ),
            audience_present=(
                config.audience.present if config.audience else False
            ),
            audience_participation=(
                config.audience.participation if config.audience else None
            ),
            topic=config.topic,
            debate_format=config.debate_format,
            setting=config.setting,
            moderator_mode=(
                config.moderator.mode if config.moderator else None
            ),
            source_count=1,
            voice_casting=voice_casting,
            voice_hints=voice_hints,
            tts_styles=tts_styles,
            character_langs=character_langs,
            series_id=series_id,
        )


# ═══════════════════════════════════════════════════════════════════════════
# THE DEBATE CONDUCTOR
# ═══════════════════════════════════════════════════════════════════════════

class DebateConductor(BaseConductor):
    """Generates a complete debate by orchestrating LLM agents.

    This is the main runtime engine. Give it a DebateConfig, call .run(),
    and get back a DebateConductorResult with the full structured transcript.

    The turn loop is strictly sequential — each agent's response depends on
    the previous agent's output. No parallelism, and that's by design.

    Attributes:
        config:             The DebateConfig driving this debate.
        _client:            The shared DeepSeekClient for all LLM calls.
        _participant_prompts: System prompts for each participant, keyed by name.
        _moderator_prompt:  System prompt for the moderator (if any).
        _transcript:        List of (speaker_name, text) tuples for context.
        _summary:           Compressed summary of older turns.
        _director_note:     Current stage direction from the Director.
        _llm_calls:         Counter for total LLM API calls made.
    """

    _bootstrap_cls = DebateBootstrap

    def __init__(
        self,
        config: DebateConfig,
        *,
        client: DeepSeekClient | None = None,
        api_key: str | None = None,
        lens_rack: "LLMLensRack | None" = None,
        **base_kwargs,
    ) -> None:
        """Initialize the conductor.

        Args:
            config:    A fully populated DebateConfig (from bootstrap or manual).
            client:    An existing DeepSeekClient to use (for testing: inject a mock).
            api_key:   API key for DeepSeekClient. Only used if client is None.
            lens_rack: Optional LLMLensRack for per-character LLM routing. When
                       a route ``character:<Name>`` exists in the rack, that
                       character's AgentSession uses the routed lens's client
                       instead of the shared default client. This lets you run
                       Salvatore on ``local-llama-dim`` while the moderator still
                       uses cloud-deepseek.
            **base_kwargs: Forwarded to BaseConductor (checkpoint_renderer,
                           budget_tracker, notes_dir, recent_window).
        """
        resolved_client = client or DeepSeekClient(api_key=api_key)
        super().__init__(config, client=resolved_client, **base_kwargs)

        self._lens_rack: "LLMLensRack | None" = lens_rack

        # These get populated by _build_agents():
        self._participant_prompts: dict[str, str] = {}
        self._moderator_prompt: str | None = None
        self._host_prompt: str | None = None
        self._agent_sessions: dict[str, "AgentSession"] = {}
        self._fact_checker: FactChecker | None = None

        # Runtime state (reset each run):
        self._transcript: list[dict] = []  # [{speaker, text, ...}, ...]
        self._summary: str = ""
        self._director_note: str = ""
        self._llm_calls: int = 0
        self._debate_id: str = ""
        self._has_spoken: set[str] = set()  # tracks first-appearance warm opens

    @classmethod
    def from_brief(
        cls,
        brief: str,
        *,
        reference_material: str | None = None,
        fact_check: str = "off",
        overrides: dict | None = None,
        client: DeepSeekClient | None = None,
        api_key: str | None = None,
    ) -> DebateConductor:
        """Bootstrap a DebateConfig from a brief and create a conductor.

        Thin wrapper over :meth:`BaseConductor.from_brief` that fixes the
        return type for callers and tooling.
        """
        return super().from_brief(  # type: ignore[return-value]
            brief,
            reference_material=reference_material,
            fact_check=fact_check,
            overrides=overrides,
            client=client,
            api_key=api_key,
        )

    # ═══════════════════════════════════════════════════════════════════
    # BaseConductor interface implementation
    # ═══════════════════════════════════════════════════════════════════

    def _build_agents(self) -> None:
        """Build system prompts and AgentSessions for each participant.

        Each participant and the moderator (if present) gets their own
        :class:`~vox_romana.core.session.AgentSession` with an independent
        sliding-window context. Sessions are created here rather than at
        init time because the config may not have been fully populated when
        the conductor was constructed (e.g. after ``from_brief()``).  We do
        NOT build one monolithic "DebateAgent" object — the multi-agent
        approach (one session per character) is what produces the distinct
        voice and perspective separation that makes debates work.
        """
        config = self.config
        if not isinstance(config, DebateConfig):
            raise TypeError(f"DebateConductor requires DebateConfig, got {type(config)}")

        # ── Build a system prompt for each participant ────────────────
        for p in config.participants:
            self._participant_prompts[p.name] = self._build_participant_prompt(p)

        # ── Build the moderator prompt (if any) ──────────────────────
        if config.moderator and config.moderator.mode != "none":
            self._moderator_prompt = self._build_moderator_prompt()

        # ── Build the show host prompt (if any) ──────────────────────
        # The host is the narrative voice that wraps the debate:
        # intro → transitions → outro. They get their own system prompt
        # built from ShowHostConfig data + the host role instructions.
        if config.show_host and config.show_host.name:
            self._host_prompt = self._build_host_prompt()

        # ── Create AgentSessions for persistent agents ───────────────
        # Each participant and moderator gets their own session with
        # sliding-window compression. This replaces the manual
        # _compress_context() approach — sessions auto-compress.
        #
        # Per-character lens routing: when a LLMLensRack is attached and
        # has a route for ``character:<Name>`` (or a catch-all
        # ``character`` route), that character's session uses the routed
        # lens's client instead of the shared default. This lets you run
        # a lightweight local model for minor characters while the main
        # participants use a cloud model.
        self._agent_sessions = {}
        for name, prompt in self._participant_prompts.items():
            char_client: DeepSeekClient | None = None
            if self._lens_rack is not None:
                char_lens = self._lens_rack.select(
                    role="character", character=name,
                )
                # Only override if the rack actually resolved a
                # character-specific route (not just the rack default).
                char_key = f"character:{name}".lower()
                catch_all = "character"
                routes = self._lens_rack.routes()
                if char_key in routes or catch_all in routes:
                    char_client = char_lens.client  # type: ignore[assignment]
                    logger.debug(
                        "DebateConductor: %s → lens %s (%s)",
                        name, char_lens.name, char_lens.model,
                    )
            self._agent_sessions[name] = self._create_session(
                prompt,
                client=char_client,
                recent_window=DEFAULT_RECENT_WINDOW,
            )
        if self._moderator_prompt and config.moderator:
            self._agent_sessions[config.moderator.name] = self._create_session(
                self._moderator_prompt, recent_window=DEFAULT_RECENT_WINDOW,
            )

        # ── Reset runtime state ──────────────────────────────────────
        self._transcript = []
        self._summary = ""
        self._director_note = ""
        self._llm_calls = 0
        self._debate_id = str(uuid.uuid4())[:8]
        self._has_spoken = set()  # reset warm-open tracking for each run

        # ── Build FactChecker if fact-checking is enabled ────────────
        if config.fact_check != "off":
            self._fact_checker = FactChecker.build_default(self._client)
        else:
            self._fact_checker = None

    def _preflight_check_bootstrap(self) -> None:
        """Fact-check the bootstrap output before generation begins.

        Scans the debate topic and director arc for fabricated claims and
        aborts (or warns) based on ``config.allow_bootstrap_fabrications``.
        """
        if self._fact_checker is None:
            return
        config = self.config
        if not isinstance(config, DebateConfig):
            return
        surfaces: list[str] = []
        if config.topic:
            surfaces.append(config.topic)
        if config.director_brief and config.director_brief.dramatic_arc:
            surfaces.append(config.director_brief.dramatic_arc)
        bootstrap_text = " ".join(surfaces).strip()
        if not bootstrap_text:
            return
        try:
            report = self._fact_checker.check_content(
                bootstrap_text,
                content_id="bootstrap",
                reference_material=config.reference_material,
                mode=config.fact_check,
                content_mode="factual",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DebateConductor: bootstrap preflight raised: %r", exc)
            return
        if report.has_fabrications:
            print(
                "\n\u26a0  BOOTSTRAP PRE-FLIGHT: Fabricated facts detected in "
                "the generated outline."
            )
            print(
                "   The fact-checker flagged the following based on LLM knowledge:"
            )
            for r in report.fabricated_claims:
                print(f"   \u2022 {r.claim}")
                print(f"     \u2192 {r.explanation}")
            if not config.allow_bootstrap_fabrications:
                print(
                    "   Generation aborted. Fix your brief or supply "
                    "--reference-material."
                )
                print(
                    "   Use --allow-bootstrap-fabrications to override "
                    "(not recommended).\n"
                )
                raise SystemExit(1)
            else:
                print(
                    "   Continuing anyway (--allow-bootstrap-fabrications is set).\n"
                )

    def _run_turn_loop(self) -> list[dict]:
        """The main generation loop. Sequential. No parallelism.

        This is the beating heart of the conductor. Each turn:
        1. Pick who speaks next
        2. Build their context (summary + recent turns + director note)
        3. Call the LLM
        4. Parse the response
        5. Record the event
        6. Compress / direct as needed
        7. Repeat

        SHOW HOST INTEGRATION:
        When a show_host is configured, the host gets injected at structural
        beats — like a film's establishing shots between dialogue scenes:
          - BEFORE the debate: intro narration (cold open + speaker intros)
          - AT ACT BREAKS: transition narration (at director key moments)
          - AFTER the debate: outro narration (summary + sign-off)

        Host events use event_kind="narration" so the renderer can distinguish
        them from regular debate utterances (event_kind="utterance").

        Returns:
            List of raw event dicts (one per speaking turn + host narrations).
        """
        config = self.config
        target_turns = config.target_turns or 35
        check_interval = (
            config.director_brief.check_interval
            if config.director_brief
            else 7
        )

        raw_events: list[dict] = []
        participant_names = [p.name for p in config.participants]
        participant_index = 0  # Round-robin pointer
        turns_since_moderator = 0
        wrapping_up = False
        closing_speakers: set[str] = set()  # Track who's given a closing statement
        has_host = self._host_prompt is not None

        # ── PRE-DEBATE: Show host intro ──────────────────────────────
        # The host opens the show BEFORE any debate turns. This is the
        # cold open, show title, topic intro, and speaker introductions.
        # Think of it as the "Previously on..." + "Tonight on..." segment.
        if has_host:
            # Build rich speaker intros so the listener can anchor each
            # TTS voice to a character. These aren't the real actors —
            # the listener needs help knowing who's who.
            speaker_intros = "; ".join(
                f"{p.name} (arguing {p.side})"
                for p in config.participants
            )
            intro_event = self._generate_host_event(
                beat="intro",
                context_hint=(
                    f"This is the opening of the show. The topic is: {config.topic}. "
                    f"The format is: {config.debate_format}. "
                    f"The setting is: {config.setting}. "
                    f"\n\nINTRO STRUCTURE (follow this order):\n"
                    f"1. Hook the listener with the topic. Name the show ('Vox Romana').\n"
                    f"2. Set the scene — describe the setting vividly.\n"
                    f"3. Introduce EACH speaker BY NAME with a brief character sketch "
                    f"(who they are, what they represent, what side they're on). "
                    f"The listener is hearing these voices for the FIRST TIME and needs "
                    f"to mentally tag each voice to a character. Give them something "
                    f"memorable to anchor each name to.\n"
                    f"4. Build anticipation for the debate.\n\n"
                    f"Speakers: {speaker_intros}"
                ),
                word_budget=self._host_intro_budget(),
            )
            raw_events.append(intro_event)

        # ── Track which key moments we've narrated transitions for ───
        # We use director brief key_moments as transition points. When
        # a key moment's approximate turn range is reached, the host
        # narrates a transition. Each moment only triggers once.
        narrated_transitions: set[int] = set()

        for turn_number in range(1, target_turns + 10):
            # +10 buffer: we allow a few extra turns for closing statements
            # rather than cutting off mid-argument.

            # ── Should we stop? ───────────────────────────────────────
            if turn_number > target_turns and not wrapping_up:
                wrapping_up = True
            if not wrapping_up and self._should_wrap_up():
                wrapping_up = True
            if turn_number > target_turns + 3:
                break  # Hard cap: don't let closing statements go on forever.
            if self._over_budget():
                break  # Hard cap: word budget exceeded.

            # ── All closings done? ───────────────────────────────────
            # Once every participant (and moderator if present) has given
            # exactly one closing statement, stop. No repeat closings.
            if wrapping_up:
                has_moderator = self._moderator_prompt is not None
                all_closed = all(
                    name in closing_speakers for name in participant_names
                )
                if has_moderator:
                    all_closed = all_closed and (
                        config.moderator.name in closing_speakers
                    )
                if all_closed:
                    break

            # ── Director check-in ────────────────────────────────────
            if (
                turn_number > 1
                and config.director_brief
                and turn_number % check_interval == 0
            ):
                director_result = self._director_check(turn_number, target_turns)
                if director_result.get("wrap_up"):
                    wrapping_up = True

            # ── Mid-debate host transition ───────────────────────────
            # At roughly the midpoint of the debate, inject a host
            # transition narration. This breaks up long debates and
            # gives the listener a "chapter marker." We trigger at the
            # ~40% and ~70% marks for a natural two-transition rhythm.
            if has_host and not wrapping_up:
                progress = turn_number / max(target_turns, 1)
                for checkpoint_idx, checkpoint_pct in enumerate([0.4, 0.7]):
                    if (
                        checkpoint_idx not in narrated_transitions
                        and abs(progress - checkpoint_pct) < (1.0 / max(target_turns, 1))
                    ):
                        narrated_transitions.add(checkpoint_idx)
                        # Build a contextual hint from recent transcript
                        recent_summary = " ".join(
                            f"{t['speaker']} argued: {t['text'][:80]}"
                            for t in self._transcript[-3:]
                        ) if self._transcript else "The debate is underway."
                        transition_event = self._generate_host_event(
                            beat="transition",
                            context_hint=(
                                f"We're at the {int(checkpoint_pct * 100)}% mark of the debate. "
                                f"Recent discussion: {recent_summary}. "
                                f"Provide a brief narrative transition — acknowledge what's "
                                f"been discussed and tee up the next phase. 2-3 sentences max. "
                                f"IMPORTANT: Name the speakers when describing their positions. "
                                f"The listener needs periodic reminders of WHO is arguing WHAT, "
                                f"since these are unfamiliar voices. Attribute key arguments "
                                f"to specific speakers by name (e.g., 'Teal'c has been "
                                f"arguing...', 'Ronon pushes back with...')."
                            ),
                        )
                        raw_events.append(transition_event)

            # ── Pick the next speaker ────────────────────────────────
            # Simple protocol: moderator interjects every N participant turns.
            # Between moderator turns, participants go round-robin.
            speaker_name, speaker_role = self._pick_next_speaker(
                participant_names=participant_names,
                participant_index=participant_index,
                turns_since_moderator=turns_since_moderator,
                wrapping_up=wrapping_up,
                turn_number=turn_number,
                target_turns=target_turns,
            )

            # ── Progress ─────────────────────────────────────────────
            print(
                f"  [turn {turn_number}/{target_turns}] {speaker_name}",
                flush=True,
            )

            if speaker_role == "moderator":
                turns_since_moderator = 0
            else:
                turns_since_moderator += 1
                # Advance round-robin pointer only for participant turns.
                participant_index = (
                    (participant_names.index(speaker_name) + 1)
                    % len(participant_names)
                )

            # ── Build context and generate ───────────────────────────
            remaining_turns = max(0, target_turns - turn_number + 1)
            event_dict = self._generate_turn(
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                turn_number=turn_number,
                wrapping_up=wrapping_up,
                remaining_turns=remaining_turns,
            )

            raw_events.append(event_dict)

            # ── Record in transcript ─────────────────────────────────
            text = event_dict.get("text", "")
            self._transcript.append({
                "speaker": speaker_name,
                "role": speaker_role,
                "text": text,
                "turn": turn_number,
            })

            # ── Mark first-appearance (warm-open) tracking ───────────
            self._has_spoken.add(speaker_name)

            # ── Adaptive duration tracking ───────────────────────────
            self._record_words(text)

            # ── Track closing statements ─────────────────────────────
            if wrapping_up:
                closing_speakers.add(speaker_name)

            # ── Check for end conditions ─────────────────────────────
            # (Handled by the all_closed check at the top of the loop.
            # The old moderator-last-word check is no longer needed since
            # closing_speakers tracking handles all configurations.
            # Context compression is handled automatically by each agent's
            # AgentSession — no manual _compress_context() needed.)

        # ── POST-DEBATE: Show host outro ─────────────────────────────
        # The host signs off AFTER the debate wraps. This is the closing
        # summary, reflection, and sign-off. The "Thanks for listening"
        # moment that makes the show feel complete and produced.
        if has_host:
            # Build a brief summary of the debate for the host's context
            debate_summary = " ".join(
                f"{t['speaker']}: {t['text'][:60]}"
                for t in self._transcript[-6:]
            ) if self._transcript else "The debate has concluded."

            outro_event = self._generate_host_event(
                beat="outro",
                context_hint=(
                    f"The debate has concluded. Final exchanges: {debate_summary}. "
                    f"Provide a closing narration: briefly reflect on what was discussed, "
                    f"note any key moments or surprising turns, and sign off the show. "
                    f"Name the speakers one last time when summarizing their positions — "
                    f"the listener benefits from hearing who argued what. "
                    f"End with something like 'This has been Vox Romana' or similar. "
                    f"3-5 sentences."
                ),
                word_budget=self._host_outro_budget(),
            )
            raw_events.append(outro_event)

        return raw_events

    def _package_result(self, raw_events: list[dict]) -> DebateConductorResult:
        """Convert raw event dicts into typed DebateEvents and package the result.

        This is where the conductor's raw JSON dicts become the same DebateEvent
        objects that the parser pipeline uses. Same model, different origin.

        Host narration events (event_kind="narration") are packaged alongside
        regular utterance events. The renderer uses event_kind to distinguish
        them — narration events get the host's distinct voice.
        """
        config = self.config
        debate_events = []

        for i, evt in enumerate(raw_events):
            # Host narration events have event_kind="narration" set in
            # _generate_host_event(). Regular debate turns are "utterance".
            event_kind = evt.get("event_kind", "utterance")

            debate_events.append(DebateEvent(
                global_order=i + 1,
                local_order=i + 1,
                speaker_role=evt.get("role", "participant"),
                speaker_name=evt.get("speaker", "Unknown"),
                target=evt.get("target"),
                text=evt.get("text", ""),
                tone=tuple(evt.get("tone", "neutral").split(";")),
                volume=evt.get("volume", "moderate"),
                pace=evt.get("pace", "medium"),
                pause_before=evt.get("pause_before", "none"),
                pause_after=evt.get("pause_after", "none"),
                interruption=evt.get("interruption", False),
                nonverbal=tuple(evt.get("nonverbal", [])),
                event_kind=event_kind,
            ))

        transcript_lines = [
            f"[{t['speaker']}] {t['text']}" for t in self._transcript
        ]

        return DebateConductorResult(
            config=config,
            events=raw_events,
            transcript=transcript_lines,
            total_llm_calls=self._total_session_calls + self._llm_calls,
            director_notes=[self._director_note] if self._director_note else [],
            debate_events=debate_events,
            debate_id=self._debate_id,
        )

    # ═══════════════════════════════════════════════════════════════════
    # PROMPT BUILDERS
    # ═══════════════════════════════════════════════════════════════════
    # These take config data and produce the system prompt strings that
    # define each agent's personality and rules.

    def _build_participant_prompt(self, profile: ParticipantProfile) -> str:
        """Build the system prompt for one participant.

        Takes their full personality profile and injects it into the
        PARTICIPANT_SYSTEM_PROMPT template. The result is a detailed
        character sheet that tells the LLM exactly who to be.
        """
        config = self.config

        # ── Build ground rules (the "room" the participant walks into) ──
        ground_rules = self._build_ground_rules(profile)

        # Format key positions as a numbered list.
        positions_str = "\n".join(
            f"  {i+1}. {pos}" for i, pos in enumerate(profile.key_positions)
        ) or "  (none specified)"

        # Format factual claims as a numbered list.
        claims_str = "\n".join(
            f"  {i+1}. {claim}" for i, claim in enumerate(profile.factual_claims)
        ) or "  (none — argue from principle, not statistics)"

        # Format tangent topics as a numbered list.
        tangents_str = "\n".join(
            f"  {i+1}. {topic}" for i, topic in enumerate(profile.tangent_topics)
        ) or "  (draw from your personality and background)"

        # Reference corpus section (optional few-shot examples).
        if profile.reference_corpus:
            corpus_lines = "\n".join(
                f'  "{q}"' for q in profile.reference_corpus
            )
            corpus_section = (
                f"EXAMPLE PHRASES (to match your speaking style):\n{corpus_lines}"
            )
        else:
            corpus_section = ""

        return PARTICIPANT_SYSTEM_PROMPT.format(
            name=profile.name,
            role=profile.role,
            debate_format=config.debate_format or "debate",
            topic=config.topic or config.brief,
            side=profile.side or "unspecified",
            ground_rules=ground_rules,
            voice_fingerprint=profile.voice_fingerprint or "(no specific voice style)",
            behavioral_triggers=profile.behavioral_triggers or "(no specific triggers)",
            emotional_arc=profile.emotional_arc or "(natural progression)",
            key_positions=positions_str,
            factual_claims=claims_str,
            tangent_topics=tangents_str,
            reference_corpus_section=corpus_section,
            language_instruction=_build_language_instruction(profile.language),
        )

    def _build_ground_rules(self, profile: ParticipantProfile) -> str:
        """Build the ground rules a participant knows walking into the room.

        With a moderator, rules are explicit: someone's running the show,
        calling on speakers, enforcing time. Without a moderator, the rules
        are subtextual — social norms of a conversation, not a formal debate.

        Either way, participants know: who else is here, where they are,
        roughly how long this is, and what the vibe is.
        """
        config = self.config
        lines: list[str] = []

        # Who else is in the room
        others = [
            f"{p.name} (arguing {p.side})"
            for p in config.participants
            if p.name != profile.name
        ]
        if others:
            lines.append(f"You're debating: {', '.join(others)}")

        # The setting
        if config.setting:
            lines.append(f"Setting: {config.setting}")

        # Audience
        if config.audience and config.audience.present:
            size = config.audience.size_description or "an audience"
            if config.audience.participation:
                lines.append(f"There's {size} — and they can jump in.")
            else:
                lines.append(f"There's {size} watching.")
        else:
            lines.append("No live audience.")

        # Approximate length (so they know to pace themselves)
        target = config.target_turns or 35
        per_speaker = target // max(len(config.participants), 1)
        lines.append(
            f"This is roughly a {per_speaker}-turn conversation for you — "
            "pace yourself, don't blow all your arguments up front."
        )

        # Tone
        if config.tone:
            lines.append(f"The tone is: {config.tone}")

        # Moderator vs. no moderator — this is where framing diverges
        if config.moderator and config.moderator.mode != "none":
            mod = config.moderator
            lines.append(
                f"MODERATOR: {mod.name} is running the show "
                f"({mod.mode} style — {mod.style or 'keeps things moving'}). "
                "They'll call on speakers and may push back on weak points. "
                "Address them when appropriate."
            )
        else:
            lines.append(
                "NO MODERATOR — this is a direct conversation. There's nobody "
                "directing traffic, so the social rules are subtextual: "
                "take turns naturally, don't monologue, and if you want to "
                "redirect, do it the way people actually do in heated "
                "conversations — interrupt, pivot, ask a pointed question, "
                "or concede a small point to claim a bigger one."
            )

        return "\n".join(lines)

    def _build_moderator_prompt(self) -> str:
        """Build the system prompt for the moderator."""
        config = self.config
        mod = config.moderator

        participant_list = ", ".join(
            f"{p.name} ({p.side or 'unspecified'})"
            for p in config.participants
        )

        return MODERATOR_SYSTEM_PROMPT.format(
            name=mod.name,
            debate_format=config.debate_format or "debate",
            topic=config.topic or config.brief,
            style=mod.style or "Neutral and professional",
            mode=mod.mode,
            setting=config.setting or "unspecified",
            participant_list=participant_list,
        )

    def _build_host_prompt(self) -> str:
        """Build the system prompt for the show host.

        The host gets a different system prompt from participants and the
        moderator because their job is fundamentally different — they
        narrate AROUND the debate, not within it.
        """
        config = self.config
        host = config.show_host

        participant_list = ", ".join(
            f"{p.name} ({p.side or 'unspecified'})"
            for p in config.participants
        )

        # Pick the right role-specific instructions.
        # "narrator" = third-person omniscient, "emcee" = first-person host.
        role_instructions = (
            NARRATOR_ROLE_INSTRUCTIONS
            if host.role == "narrator"
            else EMCEE_ROLE_INSTRUCTIONS
        )

        return SHOW_HOST_SYSTEM_PROMPT.format(
            name=host.name,
            role=host.role,
            style=host.style or "Professional and engaging",
            voice_fingerprint=host.voice_fingerprint or "(no specific voice style)",
            debate_format=config.debate_format or "debate",
            topic=config.topic or config.brief,
            setting=config.setting or "unspecified",
            participant_list=participant_list,
            role_upper=host.role.upper(),
            role_instructions=role_instructions,
        )

    def _generate_host_event(
        self, *, beat: str, context_hint: str, word_budget: int | None = None
    ) -> dict:
        """Generate one narration event from the show host.

        This is similar to _generate_turn() but simpler — the host doesn't
        need the full debate context window. They get their system prompt
        plus a specific instruction for this particular beat.

        Args:
            beat:         One of "intro", "transition", "outro". Stored as
                          the narration_beat in the event dict for debugging.
            context_hint: Specific instructions for what this narration
                          should contain. Built by the caller based on the
                          current state of the debate.
            word_budget:  Soft cap on narration length. When set, a length
                          contract is appended to the prompt so short shows
                          don't get blown apart by long host monologues.

        Returns:
            A raw event dict with event_kind="narration" and role="host".
        """
        config = self.config
        host = config.show_host

        # The host sees: their system prompt (personality) + a beat-specific
        # instruction (what to narrate right now).
        user_message = f"NARRATION BEAT: {beat.upper()}\n\n{context_hint}"
        if word_budget and word_budget > 0:
            user_message += (
                f"\n\nLENGTH CONTRACT: keep this narration to about {word_budget} words "
                f"(hard ceiling: {int(word_budget * 1.3)}). Do not exceed it."
            )

        raw_response = self._client.chat(
            messages=[
                {"role": "system", "content": self._host_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        self._llm_calls += 1

        # Parse the JSON response (same extraction logic as debate turns).
        event_dict = self._parse_turn_response(raw_response)

        # Attach host-specific metadata. The key differentiator from regular
        # debate events is event_kind="narration" — this tells the renderer
        # to use the host's voice and potentially different audio treatment.
        event_dict["speaker"] = host.name
        event_dict["role"] = "host"
        event_dict["event_kind"] = "narration"
        event_dict["narration_beat"] = beat
        event_dict["target"] = "audience"  # Host always addresses the listener

        # Host narrations get a long pause after (gives the debate room to breathe)
        # and a medium pause before transitions/outros.
        if beat == "intro":
            event_dict.setdefault("pause_after", "long")
        elif beat == "transition":
            event_dict.setdefault("pause_before", "medium")
            event_dict.setdefault("pause_after", "medium")
        elif beat == "outro":
            event_dict.setdefault("pause_before", "long")

        return event_dict

    def _host_intro_budget(self) -> int | None:
        """Soft word cap for the host intro narration.

        Scaled to the overall word target so a 3-min show doesn't get a
        200-word opener while a 30-min show isn't forced into 3 sentences.
        Returns None when no target is set (let the LLM pick).
        """
        if self._target_words is None:
            return None
        return max(60, min(180, int(self._target_words * 0.18)))

    def _host_outro_budget(self) -> int | None:
        """Soft word cap for the host outro narration. Smaller than intro."""
        if self._target_words is None:
            return None
        return max(40, min(120, int(self._target_words * 0.10)))

    # ═══════════════════════════════════════════════════════════════════
    # TURN GENERATION
    # ═══════════════════════════════════════════════════════════════════

    def _pick_next_speaker(
        self,
        *,
        participant_names: list[str],
        participant_index: int,
        turns_since_moderator: int,
        wrapping_up: bool,
        turn_number: int,
        target_turns: int,
    ) -> tuple[str, str]:
        """Decide who speaks next.

        Current strategy is simple and deterministic:
        - Moderator opens the debate (turn 1).
        - Participants go round-robin.
        - Moderator interjects every DEFAULT_MODERATOR_INTERVAL participant turns.
        - During wrap-up, alternate: each participant gets a closing turn,
          then moderator closes.

        Returns:
            (speaker_name, speaker_role) tuple.
        """
        config = self.config
        has_moderator = self._moderator_prompt is not None

        # ── Turn 1: moderator opens (if present) ─────────────────────
        if turn_number == 1 and has_moderator:
            return config.moderator.name, "moderator"

        # ── Wrap-up phase: moderator gets last word ──────────────────
        if wrapping_up and has_moderator and turns_since_moderator >= len(participant_names):
            return config.moderator.name, "moderator"

        # ── Regular turns: moderator interjects periodically ─────────
        if (
            has_moderator
            and turns_since_moderator >= DEFAULT_MODERATOR_INTERVAL
            and turn_number > 1
        ):
            return config.moderator.name, "moderator"

        # ── Participant's turn (round-robin) ─────────────────────────
        name = participant_names[participant_index % len(participant_names)]
        return name, "participant"

    def _generate_turn(
        self,
        *,
        speaker_name: str,
        speaker_role: str,
        turn_number: int,
        wrapping_up: bool,
        remaining_turns: int = 0,
    ) -> dict:
        """Generate one speaking turn by calling the LLM via AgentSession.

        Builds the context message (director note + recent turns + instruction),
        sends it through the speaker's session (which manages its own history
        and compression), and parses the JSON response.
        """
        # ── Compute per-turn word target ────────────────
        # Total budget / remaining turns gives a fair share that auto-
        # compensates for prior turns running short or long.
        per_turn_target = 0
        if self._target_words is not None and remaining_turns > 0:
            per_turn_target = max(
                30, self._remaining_words // max(remaining_turns, 1)
            )

        # ── Build the user message (context + turn prompt) ───────────
        user_message = self._build_turn_context(
            speaker_name=speaker_name,
            speaker_role=speaker_role,
            turn_number=turn_number,
            wrapping_up=wrapping_up,
            remaining_turns=remaining_turns,
            target_words=per_turn_target,
        )

        # ── Send via the speaker's AgentSession ─────────────────────
        session = self._agent_sessions.get(speaker_name)
        if session is not None:
            raw_response = session.send(user_message)
        else:
            # Fallback for agents without sessions (shouldn't happen in
            # normal flow, but keeps backward compat).
            system_prompt = (
                self._moderator_prompt
                if speaker_role == "moderator"
                else self._participant_prompts.get(speaker_name, "")
            )
            raw_response = self._client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            )
            self._llm_calls += 1

        # ── Parse the JSON response ──────────────────────────────────
        event_dict = self._parse_turn_response(raw_response)

        # ── Length contract validation ──────────────────
        if per_turn_target > 0:
            actual = len((event_dict.get("text") or "").split())
            warning = self._validate_word_count(
                declared=event_dict.get("word_count"),
                actual=actual,
                target_words=per_turn_target,
                tolerance_pct=15.0,
            )
            if warning:
                self._length_warnings.append(
                    f"[turn {turn_number} {speaker_name}] {warning}"
                )

        # ── Checkpoint render: measure THIS turn's audio length ─────
        # Use turn_number-1 as the zero-based index (matches BudgetTracker).
        self._checkpoint_segment(
            turn_number - 1, event_dict.get("text") or "",
        )

        # Attach metadata that the agent doesn't produce.
        event_dict["speaker"] = speaker_name
        event_dict["role"] = speaker_role
        event_dict["turn"] = turn_number

        # ── Fact-check gate (blocking) ────────────────────────
        # Policy: when fact-checking is enabled, we run the freshly generated
        # turn through the fact-checker. If it flags fabrications or
        # contradictions, we feed those flags back to the *same* AgentSession
        # so the speaker can restate without losing in-character tone or
        # session memory. We retry up to `config.max_retries` times. If the
        # final attempt still fails, we publish anyway and append a
        # FACT_CHECK_WARN entry to `_length_warnings` so downstream
        # rendering/CLI can surface the issue. "Blocking" here means we hold
        # the turn — no other turn runs until this gate either passes or
        # exhausts retries — but we never silently drop content.
        config = self.config
        if (
            self._fact_checker is not None
            and isinstance(config, DebateConfig)
            and config.fact_check != "off"
        ):
            turn_text = event_dict.get("text") or ""
            for fc_attempt in range(config.max_retries):
                try:
                    fc_report = self._fact_checker.check_content(
                        turn_text,
                        content_id=f"turn_{turn_number}_{speaker_name}",
                        reference_material=config.reference_material,
                        mode=config.fact_check,
                        content_mode="factual",
                        prior_context=self._build_prior_context(),
                    )
                except Exception as exc:
                    logger.warning(
                        "DebateConductor: fact-check raised for turn %d %s: %r",
                        turn_number, speaker_name, exc,
                    )
                    break

                if not (fc_report.has_fabrications or fc_report.has_contradictions):
                    break  # Turn passed — move on.

                session = self._agent_sessions.get(speaker_name)
                blocking = (
                    fc_report.fabricated_claims + fc_report.contradicted_claims
                )
                if fc_attempt < config.max_retries - 1 and session is not None:
                    claims_lines = "\n".join(
                        f"  • {r.claim}: {r.explanation}"
                        for r in blocking
                    )
                    rejection_msg = (
                        f"FACT-CHECK REJECTION (attempt {fc_attempt + 1}/"
                        f"{config.max_retries}): Your previous response "
                        f"contained fabricated facts or contradictions with "
                        f"earlier statements:\n{claims_lines}\n\n"
                        f"Please restate your position without these issues. "
                        f"Only include information you can verify and that is "
                        f"consistent with what you have already said."
                    )
                    print(
                        f"  [turn {turn_number} {speaker_name}] "
                        f"fact-check fail — retry {fc_attempt + 1}/"
                        f"{config.max_retries}",
                        flush=True,
                    )
                    try:
                        retry_raw = session.send(rejection_msg)
                        retry_dict = self._parse_turn_response(retry_raw)
                        retry_dict["speaker"] = speaker_name
                        retry_dict["role"] = speaker_role
                        retry_dict["turn"] = turn_number
                        event_dict = retry_dict
                        turn_text = event_dict.get("text") or ""
                    except Exception as exc:
                        logger.warning(
                            "DebateConductor: fact-check retry send failed "
                            "for turn %d %s: %r", turn_number, speaker_name, exc,
                        )
                        break
                else:
                    # Exhausted retries — publish with warning.
                    for r in blocking:
                        self._length_warnings.append(
                            f"[turn_{turn_number}_{speaker_name}] FACT_CHECK_WARN: "
                            f"Published with unresolved claim after "
                            f"{config.max_retries} attempts: {r.claim}"
                        )
                    print(
                        f"  [turn {turn_number} {speaker_name}] "
                        f"fact-check EXHAUSTED — "
                        f"{len(blocking)} issue(s) remain",
                        flush=True,
                    )
                    break

        return event_dict

    def _build_prior_context(self) -> str | None:
        """Return transcript turns formatted for intra-show consistency checking.

        Formats the debate transcript as labeled speaker turns so the
        consistency verifier can apply the same-speaker rule correctly
        (only flag when the SAME speaker asserts incompatible facts).
        Returns ``None`` when the transcript is empty (first turn).
        """
        if not self._transcript:
            return None
        lines = [
            f"{t['speaker']}: {t['text']}"
            for t in self._transcript
        ]
        return "\n".join(lines)

    def _build_turn_context(
        self,
        *,
        speaker_name: str,
        speaker_role: str,
        turn_number: int,
        wrapping_up: bool,
        remaining_turns: int = 0,
        target_words: int = 0,
    ) -> str:
        """Build the user message for a speaking turn.

        Assembles: director note + budget status + position phase +
        warm-open nudge (first appearance only) + recent turns +
        turn instruction.

        The per-agent AgentSession handles long-range memory via its own
        sliding-window compression, so we no longer include the shared
        summary here.
        """
        parts = []

        # ── Director's current mood note ─────────────────────────────
        if self._director_note:
            parts.append(f"DIRECTOR'S NOTE (mood/pacing guidance):\n{self._director_note}\n")

        # ── Runtime budget status ────────────────────────────────────
        total_turns = remaining_turns + turn_number - 1
        budget_status = self._budget_status_for_prompt(
            remaining_turns, total_steps=total_turns,
        )
        if budget_status:
            parts.append(budget_status)

        # ── Hard length contract for THIS turn ───────────
        # Wrap-up turns and the moderator's opening get a shorter target.
        # Closing statements are also hard-capped at 60 words to prevent
        # the LLM from delivering verbose summaries that blow the budget.
        turn_target = target_words
        if wrapping_up:
            half = max(20, target_words // 2) if target_words else 0
            turn_target = min(60, half) if half else 0
        if turn_target > 0:
            parts.append(
                self._length_contract_block(turn_target, tolerance_pct=15.0)
            )

        # ── Position-aware debate phase ──────────────────────────────
        # Tells each speaker where they are in the arc so they can
        # calibrate their strategy: lay foundations early, fight hard in
        # the middle, distil late. Moderators and wrap-up turns skip this.
        config = self.config
        if not wrapping_up and speaker_role != "moderator":
            t_total = (remaining_turns + turn_number - 1) or 1
            progress = turn_number / t_total
            if progress <= 0.25:
                phase_hint = (
                    "DEBATE PHASE — OPENING: You're early in the debate. "
                    "Lay out your core position clearly. "
                    "Don't waste your best arguments yet — establish the ground."
                )
            elif progress <= 0.65:
                phase_hint = (
                    "DEBATE PHASE — DEVELOPMENT: You're in the heart of the debate. "
                    "Challenge your opponent's logic, probe their weakest points, "
                    "counter with evidence and examples. "
                    "Advance, don't repeat."
                )
            else:
                phase_hint = (
                    "DEBATE PHASE — LATE: The debate is nearing its end. "
                    "Narrow to your single strongest argument. "
                    "Make a concrete, memorable claim your opponent can't easily dismiss. "
                    "Stop introducing new topics — sharpen what's already been said."
                )
            parts.append(phase_hint)

        # ── Warm-open: first-appearance nudge ────────────────────────
        # On a speaker's very first turn, prompt them to weave their name
        # in naturally so audio listeners can anchor voice to character.
        # This is invisible to the LLM as system-level meta — it reads
        # as a brief instruction in the user message.
        if speaker_role == "participant" and speaker_name not in self._has_spoken:
            parts.append(
                f"FIRST APPEARANCE: This is your first time speaking. "
                f"Weave your name naturally into your opening words so the "
                f"listener can anchor your voice to your character. "
                f"E.g. 'As someone who {{}}, I believe...' where you mention "
                f"who you are briefly — don't do a formal introduction, just "
                f"one natural self-reference."
            )

        # ── Recent verbatim turns ────────────────────────────────────
        recent = self._transcript[-DEFAULT_RECENT_WINDOW:]
        if recent:
            lines = []
            for t in recent:
                lines.append(f"[{t['speaker']}]: {t['text']}")
            parts.append("RECENT DIALOGUE:\n" + "\n".join(lines) + "\n")

        # ── Turn instruction ─────────────────────────────────────────
        if wrapping_up:
            if speaker_role == "moderator":
                parts.append(
                    "INSTRUCTION: The debate is ending. Deliver your closing — "
                    "summarize the key points of disagreement and thank the participants. "
                    "This is your ONE closing statement."
                )
            else:
                parts.append(
                    f"INSTRUCTION: The debate is wrapping up. {speaker_name}, "
                    "deliver your ONE closing statement (2-3 sentences). "
                    "Do NOT repeat your earlier arguments verbatim — instead, "
                    "distill your single most powerful insight or make a "
                    "forward-looking prediction. Make it count."
                )
        else:
            if turn_number == 1 and speaker_role == "moderator":
                parts.append(
                    "INSTRUCTION: Open the debate. Welcome the audience, "
                    "introduce the topic, and invite the first participant to begin."
                )
            elif speaker_role == "moderator":
                parts.append(
                    "INSTRUCTION: It's your turn to moderate. You may ask a "
                    "follow-up, redirect the conversation, or invite the next speaker. "
                    "When you call on someone, say their name."
                )
            else:
                # Build a speaker-specific attribution nudge using the last
                # speaker's name so the LLM knows exactly who to address.
                last_speaker = (
                    self._transcript[-1]["speaker"]
                    if self._transcript
                    else None
                )
                attribution = (
                    f" Address {last_speaker} by name in your response."
                    if last_speaker and last_speaker != speaker_name
                    else ""
                )
                parts.append(
                    f"INSTRUCTION: It's your turn, {speaker_name}. "
                    f"Respond to what was just said.{attribution} "
                    "Remember: do NOT restate arguments you've already made — "
                    "advance to a new angle, example, or question. "
                    "Vary your sentence opener."
                )

        return "\n".join(parts)

    def _parse_turn_response(self, raw: str) -> dict:
        """Parse an agent's JSON response into an event dict.

        LLMs sometimes wrap JSON in markdown fences or add preamble text.
        This method extracts the JSON object robustly.

        If parsing fails entirely, returns a minimal dict with the raw text
        so the debate can continue rather than crashing mid-generation.
        """
        cleaned = raw.strip()

        # Strip markdown fences if present.
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        # Try to find a JSON object in the response.
        # Sometimes the LLM adds text before/after the JSON.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            data = json.loads(cleaned)
            # Ensure nonverbal is always a list.
            if isinstance(data.get("nonverbal"), str):
                data["nonverbal"] = [data["nonverbal"]] if data["nonverbal"] else []
            return data
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat the entire response as spoken text.
            # This keeps the debate going even if the LLM flubbed the format.
            return {
                "text": raw.strip()[:500],  # Cap length for safety.
                "tone": "neutral",
                "volume": "moderate",
                "pace": "medium",
                "target": "all",
                "pause_before": "none",
                "pause_after": "none",
                "interruption": False,
                "nonverbal": [],
            }

    # ═══════════════════════════════════════════════════════════════════
    # CONTEXT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _compress_context(self) -> None:
        """Compress older transcript turns into a summary.

        Takes all turns EXCEPT the recent window, asks the LLM to summarize
        them, and stores the result. Next turn, agents see:
            [summary of turns 1..N-K] + [last K verbatim turns]

        This keeps the context window roughly flat regardless of debate length.
        """
        # Only summarize turns outside the recent window.
        older_turns = self._transcript[:-DEFAULT_RECENT_WINDOW]
        if len(older_turns) < 4:
            return  # Not enough to summarize yet.

        transcript_text = "\n".join(
            f"[{t['speaker']}]: {t['text']}" for t in older_turns
        )

        prompt = CONTEXT_SUMMARY_PROMPT.format(transcript=transcript_text)

        self._summary = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        self._llm_calls += 1

    def _director_check(self, turn_number: int, target_turns: int) -> dict:
        """Ask the Director to review the debate and issue a stage note.

        Called every `check_interval` turns. The Director reads the recent
        transcript and the arc plan, then issues a brief mood/pacing note
        that gets injected into agents' context on subsequent turns.

        Returns:
            A dict with "mood_note", "nudge", and "wrap_up" keys.
        """
        config = self.config
        director = config.director_brief

        if not director:
            return {}

        # ── Figure out current mood target ───────────────────────────
        # Map the turn number to a position in the mood_progression list.
        if director.mood_progression:
            progress = turn_number / max(target_turns, 1)
            mood_index = min(
                int(progress * len(director.mood_progression)),
                len(director.mood_progression) - 1,
            )
            current_mood = director.mood_progression[mood_index]
        else:
            current_mood = "neutral"

        # ── Find remaining key moments ───────────────────────────────
        # Filter out moments whose target turn range has already passed.
        # Each moment string may mention a turn number like "Turn 15:" or
        # "around turn 20".  We keep moments that either reference a future
        # turn or contain no turn number at all (always eligible).
        remaining = []
        for moment in director.key_moments:
            # Look for an explicit turn number in the moment text.
            match = re.search(r"[Tt]urn\s*(\d+)", moment)
            if match:
                moment_turn = int(match.group(1))
                if moment_turn >= turn_number:
                    remaining.append(moment)
            else:
                # No turn number mentioned — always include.
                remaining.append(moment)

        # ── Build recent transcript ──────────────────────────────────
        recent = self._transcript[-DEFAULT_RECENT_WINDOW:]
        recent_text = "\n".join(
            f"Turn {t['turn']} [{t['speaker']}]: {t['text']}" for t in recent
        )

        # ── Extract arguments already made from the transcript ────────
        # Build a compact list of what each speaker has argued so the
        # director can flag recycled points.
        older_turns = self._transcript[:-DEFAULT_RECENT_WINDOW]
        if older_turns:
            arguments_used = "ARGUMENTS USED:\n" + "\n".join(
                f"  [{t['speaker']}]: {t['text'][:100]}" for t in older_turns[-20:]
            )
        else:
            arguments_used = "(debate just started — no arguments yet)"

        prompt = DIRECTOR_CHECK_PROMPT.format(
            turn_number=turn_number,
            target_turns=target_turns,
            dramatic_arc=director.dramatic_arc or "(no specific arc)",
            current_mood=current_mood,
            remaining_moments="\n".join(f"- {m}" for m in remaining) or "(none)",
            closing_strategy=director.closing_strategy or "(natural ending)",
            arguments_used=arguments_used,
            recent_transcript=recent_text or "(debate just started)",
            window_size=len(recent),
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a debate director. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        self._llm_calls += 1

        # Parse the director's response.
        result = self._parse_turn_response(raw)

        # Update the director note that agents will see.
        mood = result.get("mood_note", "")
        nudge = result.get("nudge", "")
        new_topic = result.get("new_topic_suggestion", "")
        rep_warning = result.get("repetition_warning", "")

        parts = []
        if mood:
            parts.append(mood)
        if nudge:
            parts.append(f"Direction: {nudge}")
        if new_topic:
            parts.append(f"Explore next: {new_topic}")
        if rep_warning:
            parts.append(f"WARNING — AVOID REPEATING: {rep_warning}")
        self._director_note = " ".join(parts) if parts else ""

        return result
