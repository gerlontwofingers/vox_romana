"""Base conductor — the abstract contract for reactive multi-agent generation.

A conductor is the runtime engine that turns a config into content. It
orchestrates multiple LLM agents in conversation with each other, managing
turn order, context windows, and dramatic pacing.

THE TWO GENERATION PATTERNS:

    1. REACTIVE (BaseConductor): Multi-agent turn loop. Each agent responds
       to what the previous agent said. Used for debates, interviews, talk
       shows, news panels, podcast conversations — anything where the
       content emerges from the interaction.

    2. SCRIPTED (BaseRenderer — future): Pre-planned sequential rendering.
       A single agent generates from an outline. Used for audiobooks, news
       anchor reads, narrated docs, monologues.

    Some shows are hybrids: news = scripted anchoring + reactive panel.
    Both patterns produce the same Event objects → same pipeline → same TTS.

This file defines BaseConductor. Content-type implementations (DebateConductor,
InterviewConductor, etc.) subclass it and fill in the specifics.

WHY AN ABSTRACT BASE?
    Because every reactive show follows the same lifecycle:
        1. Build agents from config (system prompts + personalities)
        2. Run the turn loop (decide who speaks, generate, record)
        3. Compress context periodically (keep token budget flat)
        4. Check in with a director for pacing (optional)
        5. Package the output

    The specifics — WHO speaks, WHAT the prompts say, HOW turns are routed —
    differ by content type. The lifecycle doesn't.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .archive import AgentNote, AgentNoteStore
from .client import DeepSeekClient
from .config import ContentConfig
from .duration import (
    SpeechRateAdvisor,
    estimate_total_words,
    get_wpm,
    seconds_to_mmss,
    tolerance_window,
    words_to_seconds,
)
from .session import AgentSession


@dataclass
class ConductorResult:
    """The output of a conductor run.

    This is a generic container. Content-type conductors attach their own
    domain-specific data (e.g., DebateEvent objects) and provide convenience
    methods to convert into their pipeline's expected format.

    Attributes:
        config:         The config that was used for this run.
        events:         List of event dicts. Each content type defines its
                        own event structure. Stored as dicts here for
                        generality; subclasses cast to typed objects.
        transcript:     The full human-readable transcript (for debugging).
        total_llm_calls: How many LLM API calls were made.
        director_notes: Stage directions issued during the run.
    """

    config: ContentConfig
    events: list[dict] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    total_llm_calls: int = 0
    director_notes: list[str] = field(default_factory=list)


class BaseConductor(ABC):
    """Abstract base class for reactive multi-agent content generation.

    Subclasses must implement:
        _build_agents()     — Create LLM agents from the config.
        _run_turn_loop()    — The main generation loop.
        _package_result()   — Convert raw output into a ConductorResult.

    The public interface is just `run()`, which calls these in order.

    Optional infrastructure (available to subclasses):
        _create_session()   — Build an AgentSession with sliding-window
                              compression. Tracked internally for LLM call
                              accounting.
        _note_store         — AgentNoteStore for inter-agent communication
                              (set via notes_dir constructor parameter).
        _write_note()       — Convenience for appending an AgentNote.
    """

    def __init__(
        self,
        config: ContentConfig,
        *,
        client: DeepSeekClient | None = None,
        notes_dir: str | Path | None = None,
        recent_window: int = 10,
    ) -> None:
        self.config = config
        self._client = client
        self._recent_window = recent_window
        self._sessions: list[AgentSession] = []

        # ── Adaptive duration tracking ───────────────────────────────
        self._wpm: int = get_wpm(config.content_type)
        self._words_generated: int = 0
        self._target_words: int | None = None
        self._word_lo: int | None = None
        self._word_hi: int | None = None

        self._init_duration_targets()

        if notes_dir is not None:
            path = Path(notes_dir) / "agent_notes.jsonl"
            self._note_store: AgentNoteStore | None = AgentNoteStore(path)
        else:
            self._note_store = None

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _create_session(
        self,
        system_prompt: str,
        *,
        client: DeepSeekClient | None = None,
        recent_window: int | None = None,
    ) -> AgentSession:
        """Create a tracked AgentSession.

        The session is registered internally so that `_total_llm_calls`
        can aggregate call counts from all sessions.
        """
        c = client or self._client
        if c is None:
            raise RuntimeError(
                "No DeepSeekClient available. Pass client= to the conductor "
                "or to _create_session()."
            )
        session = AgentSession(
            client=c,
            system_prompt=system_prompt,
            recent_window=recent_window or self._recent_window,
        )
        self._sessions.append(session)
        return session

    @property
    def _total_session_calls(self) -> int:
        """Total LLM calls across all tracked sessions."""
        return sum(s.total_calls for s in self._sessions)

    # ------------------------------------------------------------------
    # Note helpers
    # ------------------------------------------------------------------

    def _write_note(
        self,
        *,
        run_id: str,
        agent: str,
        note_type: str,
        key: str,
        value: str,
    ) -> None:
        """Write an inter-agent note if a note store is configured."""
        if self._note_store is None:
            return
        self._note_store.append(
            AgentNote(
                run_id=run_id,
                agent=agent,
                note_type=note_type,
                key=key,
                value=value,
            )
        )

    # ------------------------------------------------------------------
    # Duration / adaptive length helpers
    # ------------------------------------------------------------------

    def _init_duration_targets(self) -> None:
        """Compute word-count targets from the config's duration fields.

        Priority:
            1. config.target_words  (explicit word count)
            2. config.target_duration_minutes  (converted via WPM)
            3. config.estimated_duration_minutes  (legacy field, same calc)
            4. None — no enforcement (subclass decides loop length itself)
        """
        cfg = self.config

        if cfg.target_words is not None:
            self._target_words = cfg.target_words
        elif cfg.target_duration_minutes is not None:
            self._target_words = estimate_total_words(
                cfg.target_duration_minutes,
                cfg.content_type,
                wpm=self._wpm,
            )
        elif cfg.estimated_duration_minutes is not None:
            self._target_words = estimate_total_words(
                float(cfg.estimated_duration_minutes),
                cfg.content_type,
                wpm=self._wpm,
            )

        if self._target_words is not None:
            self._word_lo, self._word_hi = tolerance_window(
                self._target_words,
                cfg.tolerance_percent,
            )

    def _record_words(self, text: str) -> None:
        """Track words generated so far."""
        self._words_generated += len(text.split())

    @property
    def _remaining_words(self) -> int | None:
        """Words left before the target.  None if no target set."""
        if self._target_words is None:
            return None
        return max(0, self._target_words - self._words_generated)

    @property
    def _progress_fraction(self) -> float | None:
        """Fraction of target words generated (0.0–1.0+). None if no target."""
        if self._target_words is None or self._target_words == 0:
            return None
        return self._words_generated / self._target_words

    def _in_tolerance(self) -> bool | None:
        """True if current word count is within the tolerance window.

        Returns None when no target is set.
        """
        if self._word_lo is None or self._word_hi is None:
            return None
        return self._word_lo <= self._words_generated <= self._word_hi

    def _should_wrap_up(self) -> bool:
        """True when the conductor should start wrapping up.

        Uses 90% of the target as the trigger point so the closing
        sequence doesn't overshoot.  Returns False if no target is set.
        """
        if self._target_words is None:
            return False
        return self._words_generated >= self._target_words * 0.90

    def _over_budget(self) -> bool:
        """True when the word count has exceeded the upper tolerance bound."""
        if self._word_hi is None:
            return False
        return self._words_generated > self._word_hi

    def _progress_summary(self) -> str:
        """Human-readable progress string for logging."""
        if self._target_words is None:
            return f"{self._words_generated:,} words (no target)"
        pct = (self._words_generated / self._target_words) * 100
        est_secs = words_to_seconds(self._words_generated, wpm=self._wpm)
        return (
            f"{self._words_generated:,}/{self._target_words:,} words "
            f"({pct:.0f}%) ≈ {seconds_to_mmss(est_secs)}"
        )

    def _budget_status_for_prompt(
        self,
        remaining_steps: int,
        total_steps: int | None = None,
    ) -> str:
        """Runtime budget status block for injection into LLM prompts.

        Returns an empty string when no target is set, so callers can
        unconditionally embed it without conditional logic.

        Args:
            remaining_steps: How many generation steps remain (segments,
                turns, beats, chapters — whatever the conductor uses).
            total_steps:     Total steps in the run. When provided,
                enables behind-pace detection.
        """
        if self._target_words is None:
            return ""
        remaining = self._remaining_words
        pct = (self._words_generated / self._target_words) * 100
        status = (
            f"BUDGET STATUS: {self._words_generated:,} / {self._target_words:,} "
            f"words generated ({pct:.0f}%). "
            f"~{remaining:,} words remaining across {remaining_steps} step(s)."
        )
        # Warn if behind pace: expected fraction done vs actual.
        if total_steps and total_steps > 0 and remaining_steps > 0:
            steps_done = total_steps - remaining_steps
            expected_pct = (steps_done / total_steps) * 100
            if pct < expected_pct - 5:  # 5% grace margin
                status += (
                    "\n⚠ BEHIND PACE — write a LONGER response to stay on track."
                )
        return status

    def _rebalanced_target(self, planned_words: int, remaining_steps: int) -> int:
        """Redistribute unspent word budget across remaining steps.

        When earlier steps run short, later steps need to compensate.
        Returns ``planned_words`` unchanged when no target is set.

        Args:
            planned_words:   The originally planned word count for one step.
            remaining_steps: How many steps remain (including the current one).
        """
        if self._target_words is None:
            return planned_words
        if remaining_steps <= 0:
            return planned_words
        fair_share = self._remaining_words // remaining_steps
        return max(planned_words, fair_share)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> ConductorResult:
        """Execute the full generation pipeline.

        Returns:
            A ConductorResult with all generated events.
        """
        self._build_agents()
        raw = self._run_turn_loop()
        return self._package_result(raw)

    @abstractmethod
    def _build_agents(self) -> None:
        """Create and configure LLM agents from self.config.

        Called once before the turn loop starts. Implementations should
        store the agents as instance attributes.
        """

    @abstractmethod
    def _run_turn_loop(self) -> list[dict]:
        """Run the main generation loop.

        Returns:
            A list of raw event dicts (one per speaking turn).
        """

    @abstractmethod
    def _package_result(self, raw_events: list[dict]) -> ConductorResult:
        """Convert raw events into a typed ConductorResult.

        Args:
            raw_events: The list of event dicts from _run_turn_loop().

        Returns:
            A ConductorResult ready for downstream consumption.
        """
