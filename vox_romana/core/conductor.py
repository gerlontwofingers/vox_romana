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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .archive import AgentNote, AgentNoteStore
from .client import DeepSeekClient
from .config import ContentConfig
from .crew import CrewMember
from .duration import (
    SpeechRateAdvisor,
    estimate_total_words,
    get_wpm,
    seconds_to_mmss,
    tolerance_window,
    words_to_seconds,
)
from .messages import AgentMessage, AgentRouter, MessageType
from .session import AgentSession
from .stage_directions import find_parenthetical_stage_directions

if TYPE_CHECKING:
    from .budget_tracker import BudgetTracker
    from .editing.models import EditorVerdict
    from .editing.models import FactCheckReport
    from .fact_checker import FactChecker
    from ..tts.checkpoint import CheckpointRenderer


logger = logging.getLogger(__name__)


# Maximum number of expansion passes BaseConductor.run() will attempt
# when the content is short of target. Each pass adds at most one bonus
# segment / turn worth of content (~1,000–1,500 words). For a 3-hour
# podcast (27k words) finishing at 7k words, we need ~17–20 passes to
# close the gap. 30 gives headroom without runaway cost.
MAX_EXPANSION_PASSES = 30


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
    production_log: list[dict[str, Any]] = field(default_factory=list)
    production_log_path: str | None = None


class BaseConductor(AgentRouter, ABC):
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

    # Empirical correction for systematic per-turn overshoot. Battery
    # studies (Apr 2026) showed conductors landing ~8–15% over their
    # target word counts because LLMs treat soft per-turn targets as
    # floors rather than midpoints. Multiplying the rebalanced fair
    # share by this factor instructs each turn to aim slightly low so
    # the cumulative output lands near the contracted target.
    # Subclasses may override.
    OVERSHOOT_COMPENSATION: float = 0.92

    # Fraction of target_words at which _should_wrap_up() fires.
    # Leaves headroom for closing statements (+25–40% empirically).
    # Subclasses may override.
    WRAP_UP_THRESHOLD: float = 0.70

    def __init__(
        self,
        config: ContentConfig,
        *,
        client: DeepSeekClient | None = None,
        notes_dir: str | Path | None = None,
        recent_window: int = 10,
        checkpoint_renderer: "CheckpointRenderer | None" = None,
        budget_tracker: "BudgetTracker | None" = None,
        crew_members: list[CrewMember] | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._recent_window = recent_window
        self._sessions: list[AgentSession] = []
        self._session_registry: dict[str, AgentSession] = {}
        self._pending_issue_reports: list[AgentMessage] = []
        self._production_log: list[dict[str, Any]] = []

        # ── Crew profiles (role → formatted profile text) ────────────
        # Built from the optional crew_members list.  Subclasses call
        # ``_augmented_director_system_prompt(base)`` to prepend the
        # director's profile to any per-call director system prompt.
        # Layer-1 identity and Layer-2 accumulated wisdom are combined
        # here so every injection site gets both without extra wiring.
        self._crew_profiles: dict[str, str] = {}
        if crew_members:
            for member in crew_members:
                blocks = [
                    member.profile_text(),
                    member.memory_text(show_type=config.content_type),
                ]
                text = "\n\n".join(b for b in blocks if b)
                if text:
                    self._crew_profiles[member.role] = text

        # ── Adaptive duration tracking ───────────────────────────────
        self._wpm: int = get_wpm(config.content_type)
        self._words_generated: int = 0
        self._target_words: int | None = None
        self._word_lo: int | None = None
        self._word_hi: int | None = None

        # ── Length-contract telemetry ────────────────────────────────
        # Diagnostic strings, one per segment whose actual word count
        # fell outside the target window. Inspected by the renderer/CLI.
        self._length_warnings: list[str] = []

        # ── Checkpoint rendering + budget tracking ─────────────────────
        # Both are opt-in. When a CheckpointRenderer is provided, each
        # segment's text can be measured for actual TTS duration. When a
        # BudgetTracker is also provided, those measurements feed back
        # into per-segment time targets for dynamic rebalancing.
        # When either is None, that feature is simply skipped — the
        # conductor generates content without TTS measurement or budget
        # feedback. None is the normal state for CLI runs that don't pass
        # --checkpoint-renderer / --budget-tracker.
        self._checkpoint_renderer = checkpoint_renderer
        self._budget_tracker = budget_tracker

        self._init_duration_targets()

        if notes_dir is not None:
            path = Path(notes_dir) / "agent_notes.jsonl"
            self._note_store: AgentNoteStore | None = AgentNoteStore(path)
        else:
            self._note_store = None

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _append_production_log(self, entry: dict[str, Any]) -> None:
        """Append a structured production-log entry for the current run."""
        stamped = dict(entry)
        stamped.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
        self._production_log.append(stamped)

    def _augmented_director_system_prompt(self, base_prompt: str) -> str:
        """Return *base_prompt* prepended with the director's crew profile.

        When no director crew member was supplied (the common case), returns
        *base_prompt* unchanged so callers need no conditional logic.

        The crew profile block is separated from the base prompt by a blank
        line so the LLM sees them as distinct sections::

            CREW IDENTITY: Marcus Webb
            Aesthetic philosophy: ...
            Pacing approach: ...

            You are a debate director. Respond with JSON only.

        Args:
            base_prompt: The show-specific director system prompt string.

        Returns:
            Augmented prompt string, or *base_prompt* if no profile exists.
        """
        profile_text = self._crew_profiles.get("director", "")
        if not profile_text:
            return base_prompt
        return f"{profile_text}\n\n{base_prompt}"

    @staticmethod
    def _outline_label(item: Any) -> str:
        """Return the best single-line label for an outline item."""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("title", "headline", "topic", "summary", "description", "name"):
                value = item.get(key)
                if value:
                    return str(value).strip()
            return ""
        for attr in ("title", "headline", "topic", "summary", "description", "name"):
            value = getattr(item, attr, None)
            if value:
                return str(value).strip()
        return ""

    def _build_initial_outline_context(self) -> str:
        """Render the episode context that every session should see up front."""
        config = self.config
        lines = [
            "INITIAL OUTLINE CONTEXT",
            "=======================",
            f"Show type: {config.content_type}",
        ]

        for label, value in (
            ("Title", getattr(config, "title", None)),
            ("Topic", getattr(config, "topic", None)),
            ("Tone", getattr(config, "tone", None)),
            ("Setting", getattr(config, "setting", None)),
            ("Perspective", getattr(config, "perspective", None)),
        ):
            if value:
                lines.append(f"{label}: {value}")

        brief = str(getattr(config, "brief", "") or "").strip()
        if brief:
            premise = brief.replace("\n", " ")[:280].strip()
            suffix = "..." if len(brief) > len(premise) else ""
            lines.append(f"Premise: {premise}{suffix}")

        cast_names: list[str] = []
        for field_name in ("participants", "anchors", "hosts", "characters"):
            entries = getattr(config, field_name, None) or []
            for entry in entries:
                name = getattr(entry, "name", None)
                if name:
                    cast_names.append(str(name))
        for field_name in ("moderator", "show_host", "narrator", "professor"):
            entry = getattr(config, field_name, None)
            name = getattr(entry, "name", None) if entry is not None else None
            if name:
                cast_names.append(str(name))
        if cast_names:
            lines.append(f"Cast: {', '.join(list(dict.fromkeys(cast_names)))}")

        outline_items: list[str] = []
        for field_name, label in (
            ("segments", "Segment"),
            ("chapters", "Chapter"),
            ("sections", "Section"),
            ("beats", "Beat"),
        ):
            entries = getattr(config, field_name, None) or []
            if not entries:
                continue
            for idx, entry in enumerate(entries[:8], start=1):
                item_label = self._outline_label(entry)
                outline_items.append(
                    f"{label} {idx}: {item_label}" if item_label else f"{label} {idx}"
                )
            break

        director_brief = getattr(config, "director_brief", None)
        if director_brief is not None:
            dramatic_arc = getattr(director_brief, "dramatic_arc", None)
            if dramatic_arc:
                lines.append(f"Arc: {dramatic_arc}")
            key_moments = getattr(director_brief, "key_moments", None) or []
            for moment in key_moments[:5]:
                text = str(moment).strip()
                if text:
                    outline_items.append(f"Key moment: {text}")

        if not outline_items:
            for label, value in (
                ("Planned turns", getattr(config, "target_turns", None)),
                ("Planned segments", getattr(config, "target_segments", None)),
                ("Planned chapters", getattr(config, "target_chapters", None)),
                ("Planned sections", getattr(config, "target_sections", None)),
            ):
                if value:
                    outline_items.append(f"{label}: {value}")

        if outline_items:
            lines.append("Outline:")
            lines.extend(f"- {item}" for item in outline_items[:8])

        return "\n".join(lines)

    def _broadcast_initial_outline(self) -> None:
        """Give every registered session the episode context before generation."""
        if not self._session_registry:
            return
        outline_context = self._build_initial_outline_context().strip()
        if not outline_context:
            return
        self._broadcast(
            AgentMessage(
                type=MessageType.BROADCAST,
                from_agent="conductor",
                payload={
                    "phase": "table_read",
                    "outline_context": outline_context,
                },
            )
        )

    def _register_session(self, name: str, session: AgentSession) -> AgentSession:
        """Register a named session for future message routing."""
        existing = self._session_registry.get(name)
        if existing is not None and existing is not session:
            raise ValueError(f"Session '{name}' is already registered.")
        self._session_registry[name] = session
        return session

    def _create_session(
        self,
        system_prompt: str,
        *,
        client: DeepSeekClient | None = None,
        recent_window: int | None = None,
        session_name: str | None = None,
    ) -> AgentSession:
        """Create a tracked AgentSession.

        The session is registered internally so that `_total_llm_calls`
        can aggregate call counts from all sessions. When ``session_name``
        is provided, the session is also added to the router registry.
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
        if session_name is not None:
            self._register_session(session_name, session)
        return session

    @property
    def _total_session_calls(self) -> int:
        """Total LLM calls across all tracked sessions."""
        return sum(s.total_calls for s in self._sessions)

    # ------------------------------------------------------------------
    # Fact-check challenge loop
    # ------------------------------------------------------------------

    def _challenge_fabrications(
        self,
        report: "FactCheckReport",
        *,
        generator_agent: str,
        fact_checker: "FactChecker | None" = None,
        perspective: str | None = None,
    ) -> "FactCheckReport":
        """Run the challenge loop on each fabricated claim in *report*.

        For every claim with verdict == "fabricated", the conductor sends a
        CHALLENGE message to the registered ``generator_agent`` session,
        asking it to justify or correct the claim. The generator's reply
        is then handed to the fact checker's adjudicator, which may
        downgrade the verdict to "verified" or "unverifiable" (or leave it
        as "fabricated", in which case the segment will still be blocked).

        The original report is mutated in place AND returned for chaining.

        No-ops gracefully when:
          - ``generator_agent`` is not in the session registry (silent skip),
          - ``fact_checker`` is None and no ``self._fact_checker`` exists,
          - the report contains no fabricated claims.
        """
        if not report.fabricated_claims:
            return report

        checker = fact_checker or getattr(self, "_fact_checker", None)
        if checker is None:
            return report

        if generator_agent not in self._session_registry:
            logger.debug(
                "_challenge_fabrications: generator '%s' not registered; "
                "skipping challenge loop.",
                generator_agent,
            )
            return report

        challenger = "fact_checker"
        for idx, result in enumerate(report.results):
            if result.verdict != "fabricated":
                continue
            payload = {
                "claim": result.claim,
                "verifier_explanation": result.explanation,
                "instructions": (
                    "The fact-checker flagged this claim as fabricated. "
                    "Either justify it (cite a source, quote canon, or "
                    "explain how it aligns with the show's stated "
                    "perspective) OR provide a corrected version. Be brief "
                    "and concrete. If you cannot justify or correct, say so."
                ),
            }
            if perspective:
                payload["perspective"] = perspective
            try:
                response = self._challenge_loop(
                    from_agent=challenger,
                    to_agent=generator_agent,
                    payload=payload,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_challenge_fabrications: challenge round-trip failed "
                    "for claim %r: %r",
                    result.claim, exc,
                )
                continue

            justification = str(response.payload.get("response", "")).strip()
            if not justification:
                continue
            new_result = checker.adjudicate_justification(
                result, justification, perspective=perspective,
            )
            report.results[idx] = new_result
            logger.info(
                "_challenge_fabrications: claim %r -> %s (was fabricated).",
                result.claim, new_result.verdict,
            )

        return report

    def _broadcast_director_note(
        self,
        director_note: str,
        *,
        summary: str | None = None,
        changed_sections: list[str] | None = None,
    ) -> None:
        """Broadcast the latest director guidance to all registered sessions."""
        payload: dict[str, object] = {"director_note": director_note}
        if summary:
            payload["summary"] = summary
        if changed_sections:
            payload["changed_sections"] = changed_sections
        self._broadcast(
            AgentMessage(
                type=MessageType.BROADCAST,
                from_agent="director",
                payload=payload,
            )
        )

    def _route_editor_rejection(
        self,
        verdict: "EditorVerdict",
        *,
        chunk_id: str,
        target_agents: list[str] | None = None,
    ) -> None:
        """Route editor rejection notes down the chain of command."""
        rejection_payload = {
            "chunk_id": chunk_id,
            "outline_version": verdict.outline_version,
            "status": verdict.status,
            "summary": verdict.summary,
            "issues": verdict.issues,
            "recommendations": verdict.recommendations,
        }
        directive_payload = {
            "chunk_id": chunk_id,
            "outline_version": verdict.outline_version,
            "instructions": (
                "The editor rejected the last chunk. Regenerate from scratch "
                "and address every issue below before proceeding."
            ),
            "editor_rejection": rejection_payload,
        }
        if target_agents:
            for target in target_agents:
                self._route_message(
                    AgentMessage(
                        type=MessageType.DIRECTIVE,
                        from_agent="director",
                        to_agent=target,
                        payload=directive_payload,
                    )
                )
            return
        self._broadcast(
            AgentMessage(
                type=MessageType.DIRECTIVE,
                from_agent="director",
                payload=directive_payload,
            )
        )

    def _report_issue_to_director(
        self,
        *,
        from_agent: str,
        summary: str,
        proposed_remediation: str | None = None,
    ) -> AgentMessage:
        """Queue a cast issue report so the next director check can see it."""
        payload = {"summary": summary}
        if proposed_remediation:
            payload["proposed_remediation"] = proposed_remediation
        message = AgentMessage(
            type=MessageType.ISSUE_REPORT,
            from_agent=from_agent,
            to_agent="director",
            payload=payload,
        )
        self._pending_issue_reports.append(message)
        self._append_production_log({
            "event_type": "issue_report",
            "status": "queued",
            "recipients": ["director"],
            "message": message.to_dict(),
        })
        return message

    def _capture_issue_flag(
        self,
        payload: dict[str, Any],
        *,
        from_agent: str | None = None,
    ) -> dict[str, Any]:
        """Strip optional issue metadata from parsed JSON and queue a report.

        Also runs the stage-direction format check (B-7) on ``text``, since
        every show conductor already funnels its per-event payload through
        here right after parsing — the natural shared hook point.
        """
        self._check_stage_direction_format(payload, from_agent=from_agent)

        issue_flag = payload.pop("issue_flag", None)
        if not issue_flag or not from_agent:
            return payload

        summary = ""
        proposed_remediation = None
        if isinstance(issue_flag, str):
            summary = issue_flag.strip()
        elif isinstance(issue_flag, dict):
            summary = str(
                issue_flag.get("summary")
                or issue_flag.get("issue")
                or issue_flag.get("message")
                or ""
            ).strip()
            raw_remediation = (
                issue_flag.get("proposed_remediation")
                or issue_flag.get("remediation")
                or issue_flag.get("suggested_fix")
            )
            if raw_remediation:
                proposed_remediation = str(raw_remediation).strip()
        elif isinstance(issue_flag, (list, tuple)):
            summary = "; ".join(
                str(item).strip() for item in issue_flag if str(item).strip()
            )
        else:
            summary = str(issue_flag).strip()

        if summary:
            self._report_issue_to_director(
                from_agent=from_agent,
                summary=summary,
                proposed_remediation=proposed_remediation,
            )
        return payload

    def _check_stage_direction_format(
        self,
        payload: dict[str, Any],
        *,
        from_agent: str | None = None,
    ) -> None:
        """Flag (non-blocking) when ``text`` uses parentheses for a scene
        action instead of the canonical [square brackets] convention.

        The TTS renderer strips both forms either way, so this never blocks
        or mutates generation — it's pure observability into prompt-
        compliance drift, surfaced via the existing length-warnings channel.
        """
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            return
        violations = find_parenthetical_stage_directions(text)
        if not violations:
            return
        speaker = from_agent or payload.get("speaker_name") or "unknown speaker"
        self._length_warnings.append(
            f"[stage-direction format] {speaker}: found (parenthetical) "
            f"scene action(s) {violations!r} — canonical format is "
            f"[square brackets] for scene actions."
        )

    def _consume_pending_issue_reports(self) -> str:
        """Return queued issue reports as a text block and clear them."""
        if not self._pending_issue_reports:
            return ""
        rendered = "\n\n".join(
            message.format_for_context()
            for message in self._pending_issue_reports
        )
        self._pending_issue_reports.clear()
        return rendered

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

        Uses 70% of the target as the trigger point.  This leaves
        headroom for closing statements + host outro, which empirically
        add another 25-40% on top of the trigger word count.  Returns
        False if no target is set.
        """
        if self._target_words is None:
            return False
        return self._words_generated >= self._target_words * self.WRAP_UP_THRESHOLD

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
        # Apply overshoot compensation so the per-turn ask aims slightly
        # below the naive average. Empirically the model produces ~8–15%
        # more than asked for, so scaling the budget down lands the
        # cumulative output near target. See class-level
        # OVERSHOOT_COMPENSATION docstring.
        adjusted_fair_share = int(fair_share * self.OVERSHOOT_COMPENSATION)
        return max(planned_words, adjusted_fair_share)

    # ── Hard length contract helpers ──────────────────────────────────

    @staticmethod
    def _length_contract_block(
        target_words: int,
        *,
        tolerance_pct: float = 12.0,
        require_word_count: bool = True,
    ) -> str:
        """Build the strict length-contract text for injection into prompts.

        This replaces soft language ("aim for ~X words") with a hard range
        ("MUST be between MIN and MAX words") and instructs the model to
        return its declared word count for accountability.

        Args:
            target_words:       The midpoint of the desired word range.
            tolerance_pct:      Acceptable +/- percentage around target.
            require_word_count: When True, instructs the model to include
                                a 'word_count' field in its JSON response.

        Returns:
            A formatted block of text ready to splice into the prompt.
            Empty string when target_words <= 0.
        """
        if target_words <= 0:
            return ""
        margin = max(1, int(target_words * (tolerance_pct / 100.0)))
        lo = max(1, target_words - margin)
        hi = target_words + margin
        lines = [
            f"LENGTH REQUIREMENT (HARD): your response MUST be between "
            f"{lo} and {hi} words (target: {target_words}). "
            "This is a hard constraint — pad with detail, examples, or "
            "follow-up beats if you finish your point early. Do NOT cut "
            "short. Do NOT exceed the upper bound."
        ]
        if require_word_count:
            lines.append(
                "Your response JSON MUST include a numeric 'word_count' "
                "field counting the words in the spoken text. The system "
                "will validate this — under-count and you will be retried."
            )
        return "\n".join(lines)

    @staticmethod
    def _validate_word_count(
        declared: int | None,
        actual: int,
        *,
        target_words: int,
        tolerance_pct: float = 12.0,
    ) -> str | None:
        """Return a length-warning message if the response misses target.

        Returns None when the response is within tolerance.
        """
        if target_words <= 0:
            return None
        margin = max(1, int(target_words * (tolerance_pct / 100.0)))
        lo = max(1, target_words - margin)
        hi = target_words + margin
        if not (lo <= actual <= hi):
            tag = ""
            if declared is not None and abs(declared - actual) > 5:
                tag = f" (declared word_count={declared}, actual={actual})"
            return (
                f"length out of range: actual={actual} target={target_words} "
                f"window=[{lo},{hi}]{tag}"
            )
        return None

    # ── Checkpoint rendering ───────────────────────────────────────────

    def _checkpoint_segment(
        self,
        segment_index: int,
        text: str,
        *,
        voice: str | None = None,
    ) -> float | None:
        """Measure the actual TTS duration of a freshly-generated segment.

        Records the result with the BudgetTracker (when one is attached)
        so that subsequent segments can rebalance against the real drift.

        Returns:
            The measured duration in seconds, or None when no
            CheckpointRenderer is configured (caller should fall back to
            word-count estimation).
        """
        if self._checkpoint_renderer is None:
            return None
        if not text or not text.strip():
            actual_seconds = 0.0
        else:
            try:
                actual_seconds = self._checkpoint_renderer.measure(
                    text, voice=voice,
                )
            except Exception as exc:  # noqa: BLE001 — TTS is fragile
                # We never want a checkpoint failure to abort generation.
                self._length_warnings.append(
                    f"[checkpoint segment {segment_index}] "
                    f"measurement failed: {exc!r}"
                )
                return None

        if self._budget_tracker is not None:
            try:
                self._budget_tracker.record_segment(
                    segment_index, actual_seconds=actual_seconds,
                )
            except (IndexError, ValueError) as exc:
                # Out-of-range index = our caller miscounted; warn but keep going.
                self._length_warnings.append(
                    f"[checkpoint segment {segment_index}] "
                    f"tracker rejected: {exc!r}"
                )

        return actual_seconds

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> ConductorResult:
        """Execute the full generation pipeline.

        Returns:
            A ConductorResult with all generated events.
        """
        self._build_agents()
        self._broadcast_initial_outline()
        self._preflight_check_bootstrap()
        raw = self._run_turn_loop()
        # ── Expansion loop ────────────────────────────────────────────
        # If we finished short of target, give the conductor multiple
        # chances to extend the content. Each pass recalculates the
        # remaining deficit from fresh state. Subclasses opt in by
        # overriding ``_expansion_pass()``; the default does nothing.
        max_passes = MAX_EXPANSION_PASSES
        pass_count = 0
        while pass_count < max_passes:
            extra = self._maybe_expand(raw)
            if not extra:
                break
            raw.extend(extra)
            # Update word counter so the next pass's deficit calculation
            # reflects the events we just appended. Without this the loop
            # would keep firing until it hit the cap even when expansion
            # did real work.
            for ev in extra:
                text = ""
                if isinstance(ev, dict):
                    text = ev.get("text", "") or ""
                else:
                    text = getattr(ev, "text", "") or ""
                if text:
                    self._words_generated += len(text.split())
            pass_count += 1
        if pass_count >= max_passes:
            self._length_warnings.append(
                f"[expansion] hit max-pass cap ({max_passes}) — "
                f"final word count {self._words_generated:,} may still be short."
            )
        elif pass_count > 0:
            self._length_warnings.append(
                f"[expansion] closed deficit in {pass_count} pass(es) — "
                f"final word count {self._words_generated:,}."
            )
        result = self._package_result(raw)
        result.production_log = list(self._production_log)
        return result

    def _maybe_expand(self, raw_events: list[dict]) -> list[dict]:
        """Decide whether to call ``_expansion_pass`` and run it if so.

        Returns the list of additional events produced (possibly empty).
        Driven by the BudgetTracker when present, otherwise by the word
        counter.
        """
        # Decide based on time when we have a tracker, else words.
        deficit_seconds: float = 0.0
        deficit_words: int = 0

        if self._budget_tracker is not None and self._budget_tracker.records:
            tolerance_s = max(15.0, self._budget_tracker.total_target_seconds * 0.05)
            if self._budget_tracker.is_short_by(tolerance_s):
                deficit_seconds = -self._budget_tracker.cumulative_delta()
                # Convert to words via the conductor's wpm estimate.
                deficit_words = max(20, int(deficit_seconds / 60.0 * self._wpm))
        elif self._target_words is not None and self._word_lo is not None:
            if self._words_generated < self._word_lo:
                deficit_words = self._word_lo - self._words_generated
                deficit_seconds = deficit_words / max(self._wpm, 1) * 60.0

        if deficit_words <= 0:
            return []

        try:
            return self._expansion_pass(
                deficit_words=deficit_words,
                deficit_seconds=deficit_seconds,
            ) or []
        except Exception as exc:  # noqa: BLE001
            self._length_warnings.append(
                f"[expansion] failed: {exc!r} — proceeding without extension."
            )
            return []

    def _expansion_pass(
        self,
        *,
        deficit_words: int,
        deficit_seconds: float,
    ) -> list[dict]:
        """Hook: produce additional events to close a length deficit.

        The default implementation is a no-op. Subclasses override this
        to add content that fits their format (e.g. a debate gets one
        more exchange round, an audiobook gets a narrative continuation).

        Args:
            deficit_words:   Words needed to reach the lower length bound.
            deficit_seconds: Seconds of audio needed (estimated).

        Returns:
            A list of new event dicts to append. May be empty if the
            subclass declines or cannot extend further.
        """
        return []

    def _preflight_check_bootstrap(self) -> None:
        """Fact-check bootstrap output before the generation loop begins.

        Called in ``run()`` after ``_build_agents()`` (so
        ``self._fact_checker`` and ``self._content_mode`` are set) but
        before ``_run_turn_loop()``. The base implementation is a no-op.
        Subclasses that produce bootstrap output (PodcastConductor,
        NewsConductor, etc.) override this to scan the generated config
        for fabricated claims and abort or warn accordingly.
        """

    # ------------------------------------------------------------------
    # Bootstrap convenience
    # ------------------------------------------------------------------

    # Subclasses set this to their show-specific bootstrap class so that
    # ``BaseConductor.from_brief`` can build a config without knowing the
    # show type. Left as ``None`` on the base means subclasses must
    # override ``from_brief`` directly (which is fine, but uncommon).
    _bootstrap_cls: type | None = None

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
        target_words: int | None = None,
    ) -> "BaseConductor":
        """Bootstrap a config from a brief and create a conductor.

        Template implementation: each subclass sets ``_bootstrap_cls`` to
        its show-specific bootstrap class, and the rest of the wiring —
        sharing the LLM client between bootstrap and conductor, forwarding
        ``overrides`` and (when supported) ``target_words`` — happens here.

        Args:
            brief:              Natural language description of the show.
            reference_material: Factual source text to ground the output.
            fact_check:         "off", "quick", or "thorough".
            overrides:          Dict of config-field overrides (subclass-specific).
            client:             Shared DeepSeekClient. One client is used for
                                BOTH the bootstrap agents and the conductor.
            api_key:            API key if ``client`` is not provided.
            target_words:       Target word count for the show. Forwarded to
                                bootstraps that accept it (news, podcast,
                                audiobooks); ignored for debates.

        Returns:
            A subclass instance ready to ``run()``.
        """
        if cls._bootstrap_cls is None:
            raise NotImplementedError(
                f"{cls.__name__}._bootstrap_cls is not set. Override "
                "from_brief() or set _bootstrap_cls on the subclass."
            )
        shared_client = client or DeepSeekClient(api_key=api_key)
        bootstrap = cls._bootstrap_cls(client=shared_client)

        kwargs: dict = {
            "reference_material": reference_material,
            "fact_check": fact_check,
            "overrides": overrides,
        }
        # Only forward target_words when the bootstrap accepts it. Debates
        # don't take a word target at the bootstrap stage; everyone else does.
        from inspect import signature
        if "target_words" in signature(bootstrap.from_brief).parameters:
            kwargs["target_words"] = target_words

        config = bootstrap.from_brief(brief, **kwargs)
        return cls(config, client=shared_client)

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
