"""The Podcast Conductor — runtime engine that generates a podcast episode.

This is the podcast counterpart to DebateConductor and AudiobookConductor.
It takes a fully populated PodcastConfig and generates a complete podcast
episode by orchestrating a multi-host conversation segment by segment.

STRUCTURAL DIFFERENCE FROM OTHER SHOW TYPES:

    Debate:    Reactive. Multiple agents argue turn-by-turn.
    Audiobook: Scripted. A single narrator generates chapter-by-chapter.
    Podcast:   SEGMENT-LEVEL. One LLM call produces an entire segment of
               multi-speaker conversation (3-15 utterances from multiple
               hosts in a single response).

    This is more cost-efficient than debate-style turn-by-turn generation
    and produces more natural conversational flow — hosts can react to each
    other within the same LLM context window.

THE SEGMENT LOOP:

    For each segment in the outline:
    1. Build context: previous summary + director note for this segment
    2. Pick the lead host (or rotate if no lead is specified)
    3. Inject the SEGMENT_PROMPT with type-specific overlay
    4. Parse response: JSON array → list of PodcastEvent dicts
    5. Record segment summary for continuity
    6. Compress context periodically
    7. Director check-in at intervals

CONTEXT MANAGEMENT:

    The conductor maintains a running context log of segment summaries.
    Every N segments (configurable via director_brief.compression_interval),
    the summaries are compressed into a shorter context block via LLM.
    This keeps the context window bounded for long episodes.

USAGE:

    # One-shot from brief:
    conductor = PodcastConductor.from_brief(
        "Three women discuss soccer rivalries",
        target_words=5000,
    )
    result = conductor.run()

    # From existing config:
    config = PodcastBootstrap().from_brief("...")
    conductor = PodcastConductor(config)
    result = conductor.run()

    # Save:
    merged = result.to_merged_podcast()
    PodcastSerializer.save(merged, "output/text/podcasts/soccer.json")
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from dataclasses import dataclass, field

from ...core.client import DeepSeekClient
from .bootstrap import PodcastBootstrap
from ...core.conductor import BaseConductor, ConductorResult
from ...core.content_mode import ContentModeClassifier
from ...core.fact_checker import FactChecker
from ...core.session import AgentSession
from .config import PodcastConfig, PodcastHostProfile, PodcastSegmentConfig

logger = logging.getLogger(__name__)
from .models import MergedPodcast, PodcastEvent, PodcastSegment
from .prompts import (
    CONTEXT_SUMMARY_PROMPT,
    DIRECTOR_CHECK_PROMPT,
    HOST_DESCRIPTION_TEMPLATE,
    PODCAST_HOST_SYSTEM_PROMPT,
    SEGMENT_PROMPT,
    SEGMENT_TYPE_OVERLAYS,
)


# ── Defaults ─────────────────────────────────────────────────────────────
# How many recent segment summaries to keep as verbatim context.
DEFAULT_RECENT_WINDOW = 6

# Fallback word targets by duration_hint when no target_words is set.
# These are also used as ceilings when the architect omits a target —
# tuned to match the architect's per-segment budget (~1,100 words medium).
DURATION_WORD_TARGETS = {
    "short": 600,
    "medium": 1100,
    "long": 1600,
}

# When a segment comes back below this fraction of its word target,
# the conductor sends one continuation request to bring it up to spec.
SEGMENT_RETRY_THRESHOLD = 0.70

# Continuation prompt sent when a segment came back too short. The {actual},
# {target}, and {deficit} placeholders are filled by ``_run_turn_loop``.
_SEGMENT_RETRY_TEMPLATE = (
    "Your previous response was only {actual} words but the target was "
    "{target}. Continue the conversation from EXACTLY where you left off "
    "— add roughly {deficit} more words of substantive new content (more "
    "detail, a counterpoint, an example, a quick anecdote). Do NOT recap. "
    "Do NOT start over. Stay in the same JSON format."
)


# ═══════════════════════════════════════════════════════════════════════════
# PodcastConductorResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PodcastConductorResult(ConductorResult):
    """Typed result from a PodcastConductor run.

    Adds:
        podcast_events:   List of typed PodcastEvent objects.
        podcast_segments: List of completed PodcastSegment objects.
        podcast_id:       Unique ID for this episode.

    USAGE:
        result = conductor.run()

        # Iterate typed events:
        for event in result.podcast_events:
            print(f"[{event.speaker_name}] {event.text}")

        # Convert to MergedPodcast for serialization:
        merged = result.to_merged_podcast()
        PodcastSerializer.save(merged, "output/podcast.json")
    """

    podcast_events: list[PodcastEvent] = field(default_factory=list)
    podcast_segments: list[PodcastSegment] = field(default_factory=list)
    podcast_id: str = ""

    def to_merged_podcast(self, series_id: str | None = None) -> MergedPodcast:
        """Convert this result into a MergedPodcast for serialization.

        Bridges conductor output to the serialization format. Builds
        voice_casting, voice_hints, and tts_styles from host profiles.
        """
        config = self.config
        if not isinstance(config, PodcastConfig):
            raise TypeError(f"Expected a PodcastConfig, got {type(config)}")

        # Build voice maps from host profiles.
        voice_casting: dict[str, str] = {}
        voice_hints: dict[str, str] = {}
        tts_styles: dict[str, str] = {}
        for host in config.hosts:
            if host.gender:
                voice_casting[host.name] = host.gender
            if host.voice_hint:
                voice_hints[host.name] = host.voice_hint
            if host.tts_style:
                tts_styles[host.name] = host.tts_style

        return MergedPodcast(
            podcast_id=self.podcast_id,
            title=config.title or "Untitled Podcast",
            show_name=config.show_name,
            topic=config.topic,
            tone=config.tone,
            hosts=tuple(h.name for h in config.hosts),
            segments=tuple(self.podcast_segments),
            events=tuple(self.podcast_events),
            voice_casting=voice_casting,
            voice_hints=voice_hints,
            tts_styles=tts_styles,
            series_id=series_id,
        )


# ═══════════════════════════════════════════════════════════════════════════
# THE PODCAST CONDUCTOR
# ═══════════════════════════════════════════════════════════════════════════

class PodcastConductor(BaseConductor):
    """Generates a complete podcast by orchestrating multi-host segments.

    The segment loop walks through the episode outline, generating each
    segment as a complete multi-speaker conversation in a single LLM call.
    This is more efficient than turn-by-turn generation and produces more
    natural conversational flow.

    Attributes:
        config:           The PodcastConfig driving this episode.
        _client:          The shared DeepSeekClient.
        _session:         The single AgentSession for generation.
        _context_log:     Running list of segment summaries for continuity.
        _compressed_ctx:  Compressed version of older segment summaries.
        _director_note:   Current mood/pacing note from the Director.
        _llm_calls:       Counter for direct LLM calls (non-session).
    """

    _bootstrap_cls = PodcastBootstrap

    def __init__(
        self,
        config: PodcastConfig,
        *,
        client: DeepSeekClient | None = None,
        api_key: str | None = None,
        **base_kwargs,
    ) -> None:
        """Initialize the conductor.

        Args:
            config:  A fully populated PodcastConfig (from bootstrap or manual).
            client:  An existing DeepSeekClient (for testing: inject a mock).
            api_key: API key for DeepSeekClient. Only used if client is None.
            **base_kwargs: Forwarded to BaseConductor (checkpoint_renderer,
                           budget_tracker, notes_dir, recent_window).
        """
        resolved_client = client or DeepSeekClient(api_key=api_key)
        super().__init__(config, client=resolved_client, **base_kwargs)

        # Populated by _build_agents():
        self._system_prompt: str = ""
        self._session: AgentSession | None = None
        self._fact_checker: FactChecker | None = None
        self._content_mode: str = "factual"

        # Runtime state (reset each run):
        self._context_log: list[str] = []  # Segment summaries
        self._compressed_ctx: str = ""  # Compressed older summaries
        self._director_note: str = ""
        self._llm_calls: int = 0
        self._podcast_id: str = ""
        self._approved_segment_texts: list[str] = []
        # Counter that lets _expansion_pass rotate through the original
        # outline rather than fixating on the last segment's topic.
        self._expansion_idx: int = 0

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
    ) -> PodcastConductor:
        """Bootstrap a PodcastConfig from a brief and create a conductor.

        Thin wrapper over :meth:`BaseConductor.from_brief`.
        """
        return super().from_brief(  # type: ignore[return-value]
            brief,
            reference_material=reference_material,
            fact_check=fact_check,
            overrides=overrides,
            client=client,
            api_key=api_key,
            target_words=target_words,
        )

    # ═══════════════════════════════════════════════════════════════════
    # BaseConductor interface implementation
    # ═══════════════════════════════════════════════════════════════════

    def _build_agents(self) -> None:
        """Build the system prompt and session for segment generation.

        Unlike debates (one session per speaker), podcasts use ONE session
        that generates lines for ALL hosts. The system prompt describes
        the full cast, and each segment prompt says "now have them discuss X."
        """
        config = self.config
        if not isinstance(config, PodcastConfig):
            raise TypeError(
                f"PodcastConductor requires PodcastConfig, got {type(config)}"
            )

        # Build host descriptions for the system prompt.
        host_descriptions = self._build_host_descriptions(config.hosts)

        self._system_prompt = PODCAST_HOST_SYSTEM_PROMPT.format(
            show_name=config.show_name or "Untitled Podcast",
            topic=config.topic or config.brief[:200],
            tone=config.tone or "engaging",
            host_descriptions=host_descriptions,
            current_date=datetime.date.today().isoformat(),
        )

        # Create a single AgentSession for all segment generation.
        self._session = self._create_session(
            self._system_prompt, recent_window=DEFAULT_RECENT_WINDOW,
        )

        # Build FactChecker if fact-checking is enabled.
        if config.fact_check != "off":
            self._fact_checker = FactChecker.build_default(self._client)
            # Use explicitly-set content_mode; only call classifier when
            # config.content_mode is still the default ("factual") meaning
            # the user hasn't overridden it and auto-detection should run.
            if config.content_mode != "factual":
                self._content_mode = config.content_mode
            else:
                classifier = ContentModeClassifier(self._client)
                self._content_mode = classifier.classify(
                    brief=config.brief,
                    show_type="podcast",
                )
        else:
            self._fact_checker = None
            self._content_mode = config.content_mode

        # Reset runtime state.
        self._context_log = []
        self._compressed_ctx = ""
        self._director_note = ""
        self._llm_calls = 0
        self._podcast_id = str(uuid.uuid4())[:8]
        self._approved_segment_texts = []
        self._expansion_idx = 0

    def _preflight_check_bootstrap(self) -> None:
        """Fact-check the bootstrap output before generation begins.

        Collects the show topic, each segment topic, and the director arc
        into a single text block and runs it through the fact-checker.
        Aborts (or warns) on fabricated claims based on
        ``config.allow_bootstrap_fabrications``.
        """
        if self._fact_checker is None:
            return
        config = self.config
        if not isinstance(config, PodcastConfig):
            return
        surfaces: list[str] = []
        if config.topic:
            surfaces.append(config.topic)
        for seg in config.segments:
            if seg.topic:
                surfaces.append(seg.topic)
        if config.director_brief and config.director_brief.arc:
            surfaces.append(config.director_brief.arc)
        bootstrap_text = " ".join(surfaces).strip()
        if not bootstrap_text:
            return
        try:
            report = self._fact_checker.check_content(
                bootstrap_text,
                content_id="bootstrap",
                reference_material=config.reference_material,
                mode=config.fact_check,
                content_mode=self._content_mode,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PodcastConductor: bootstrap preflight raised: %r", exc)
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
        """The main segment-by-segment generation loop.

        For each segment in the config:
        1. Build context from previous segments + director note
        2. Build the segment prompt with type-specific overlay
        3. Send to the LLM (via AgentSession)
        4. Parse the JSON array response into event dicts
        5. Record segment summary for continuity
        6. Compress context periodically
        7. Director check-in at intervals

        Returns:
            List of raw event dicts (all segments, flattened).
        """
        config = self.config
        if not isinstance(config, PodcastConfig):
            raise TypeError(
                f"PodcastConductor requires PodcastConfig, got {type(config)}"
            )

        all_events: list[dict] = []
        segments = config.segments
        total_segments = len(segments)
        global_order = 0
        host_names = [h.name for h in config.hosts]

        # Compression interval from director brief, or default.
        compress_interval = 5
        if config.director_brief:
            compress_interval = config.director_brief.compression_interval

        for seg_idx, segment in enumerate(segments):
            seg_id = f"seg_{seg_idx:03d}"

            # ── Progress ─────────────────────────────────────────────
            topic_preview = (segment.topic or "")[:60]
            print(
                f"  [segment {seg_idx + 1}/{total_segments}] {topic_preview}",
                flush=True,
            )

            # ── Build context summary ────────────────────────────────
            context_summary = self._build_context_summary()

            # ── Director note for this segment ───────────────────────
            director_note_str = ""
            if self._director_note:
                director_note_str = f"DIRECTOR'S NOTE: {self._director_note}"
            # Also check for segment-specific notes.
            if config.director_brief and seg_idx in config.director_brief.segment_notes:
                specific_note = config.director_brief.segment_notes[seg_idx]
                director_note_str += f"\nSEGMENT NOTE: {specific_note}"

            # ── Determine target words ───────────────────────────────
            target_words = segment.target_words
            if target_words is None:
                target_words = DURATION_WORD_TARGETS.get(
                    segment.duration_hint, 400
                )
            remaining_segments = total_segments - seg_idx
            target_words = self._rebalanced_target(target_words, remaining_segments)

            # ── Runtime budget status for the LLM ────────────────────
            runtime_status = self._budget_status_for_prompt(
                remaining_segments, total_steps=total_segments,
            )

            # ── Determine lead host ──────────────────────────────────
            lead_host = segment.lead_host
            if not lead_host and host_names:
                # Rotate: pick based on segment index.
                lead_host = host_names[seg_idx % len(host_names)]

            # ── Get type-specific overlay ────────────────────────────
            type_overlay = SEGMENT_TYPE_OVERLAYS.get(
                segment.segment_type, ""
            )

            # ── Build the segment prompt ─────────────────────────────
            length_contract = self._length_contract_block(target_words)
            user_message = SEGMENT_PROMPT.format(
                segment_type=segment.segment_type.upper(),
                segment_topic=segment.topic,
                lead_host=lead_host or "any host",
                length_contract=length_contract,
                type_overlay=type_overlay,
                context_summary=context_summary or "(This is the first segment.)",
                director_note=director_note_str,
                runtime_status=runtime_status,
            )

            # ── Generate via AgentSession ────────────────────────────
            if self._session is not None:
                raw_response = self._session.send(user_message)
            else:
                # Fallback: direct LLM call (shouldn't happen in normal flow).
                raw_response = self._client.chat(
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
                self._llm_calls += 1

            # ── Parse response into event dicts ─────────────────────
            segment_events = self._parse_segment_response(
                raw_response,
                segment_id=seg_id,
                segment_type=segment.segment_type,
                lead_host=lead_host or "",
                global_order_start=global_order,
            )

            # ── Length contract validation ──────────────
            actual_words = sum(
                len((evt.get("text") or "").split()) for evt in segment_events
            )
            # ── Retry once if the segment came back too short ────────
            # The model often under-shoots the LENGTH REQUIREMENT block.
            # One follow-up that explicitly cites the deficit reliably
            # adds 60-90% of the missing content without altering tone.
            #
            # Threshold (70%): empirically, segments in the 70-100% band
            # of target read fine — we accept those rather than risk a
            # tone-breaking continuation. Below 70%, the deficit is
            # severe enough that continuing is worth the risk.
            #
            # Deficit math: `max(50, target - actual)` floors the ask at
            # 50 words so we don't request a useless 5-word stub when
            # the gap is small but still under-threshold.
            #
            # Extend, don't replace: we APPEND the continuation events to
            # the existing `segment_events` rather than discarding the
            # original. The original is already valid speech — throwing
            # it out would waste a successful generation and force the
            # model to start a new arc cold. Continuation preserves the
            # in-character voice and conversational momentum.
            if (
                self._session is not None
                and target_words > 0
                and actual_words < target_words * SEGMENT_RETRY_THRESHOLD
            ):
                deficit = max(50, target_words - actual_words)
                retry_msg = _SEGMENT_RETRY_TEMPLATE.format(
                    actual=actual_words,
                    target=target_words,
                    deficit=deficit,
                )
                print(
                    f"  [segment {seg_idx + 1}/{total_segments}] retry "
                    f"(was {actual_words}, target {target_words})",
                    flush=True,
                )
                try:
                    retry_raw = self._session.send(retry_msg)
                except Exception as exc:  # noqa: BLE001
                    self._length_warnings.append(
                        f"[segment {seg_idx + 1}] retry LLM call failed: {exc!r}"
                    )
                    retry_raw = ""
                if retry_raw:
                    extra_events = self._parse_segment_response(
                        retry_raw,
                        segment_id=seg_id,
                        segment_type=segment.segment_type,
                        lead_host=lead_host or "",
                        global_order_start=global_order + len(segment_events),
                    )
                    segment_events.extend(extra_events)
                    actual_words = sum(
                        len((evt.get("text") or "").split())
                        for evt in segment_events
                    )
            warning = self._validate_word_count(
                declared=None,
                actual=actual_words,
                target_words=target_words,
            )
            if warning:
                # Stash on the conductor for downstream visibility; do NOT
                # inject as a fake event (would skew counts and renders).
                self._length_warnings.append(
                    f"[segment {seg_idx + 1}/{total_segments}] {warning}"
                )

            # ── Fact-check gate (blocking) ────────────────────────
            if self._fact_checker is not None and config.fact_check != "off":
                for fc_attempt in range(config.max_retries):
                    seg_text_for_check = " ".join(
                        (evt.get("text") or "").strip() for evt in segment_events
                    ).strip()
                    try:
                        fc_report = self._fact_checker.check_content(
                            seg_text_for_check,
                            content_id=seg_id,
                            reference_material=config.reference_material,
                            mode=config.fact_check,
                            content_mode=self._content_mode,
                            prior_context=self._build_prior_context(),
                        )
                    except Exception as exc:
                        logger.warning(
                            "PodcastConductor: fact-check raised for %s: %r",
                            seg_id, exc,
                        )
                        break

                    if not (fc_report.has_fabrications or fc_report.has_contradictions):
                        break  # Segment passed — move on.

                    blocking = (
                        fc_report.fabricated_claims + fc_report.contradicted_claims
                    )
                    if fc_attempt < config.max_retries - 1:
                        # Build rejection note and regenerate.
                        claims_lines = "\n".join(
                            f"  • {r.claim}: {r.explanation}"
                            for r in blocking
                        )
                        rejection_msg = (
                            f"FACT-CHECK REJECTION (attempt {fc_attempt + 1}/"
                            f"{config.max_retries}): The previous segment "
                            f"contained fabricated facts or contradictions with "
                            f"previously approved content:\n{claims_lines}\n\n"
                            f"Please rewrite the entire segment from scratch. "
                            f"Remove all fabricated claims and contradictions. Only "
                            f"state things confirmed by the reference material or "
                            f"consistent with what has already been said."
                        )
                        print(
                            f"  [segment {seg_idx + 1}/{total_segments}] "
                            f"fact-check fail ({len(blocking)} issue(s)) "
                            f"— retry {fc_attempt + 1}/{config.max_retries}",
                            flush=True,
                        )
                        try:
                            retry_raw = self._session.send(rejection_msg) if self._session else ""
                            if retry_raw:
                                segment_events = self._parse_segment_response(
                                    retry_raw,
                                    segment_id=seg_id,
                                    segment_type=segment.segment_type,
                                    lead_host=lead_host or "",
                                    global_order_start=global_order,
                                )
                        except Exception as exc:
                            logger.warning(
                                "PodcastConductor: fact-check retry send failed "
                                "for %s: %r", seg_id, exc,
                            )
                            break
                    else:
                        # Retries exhausted — publish with warnings attached.
                        for r in blocking:
                            self._length_warnings.append(
                                f"[{seg_id}] FACT_CHECK_WARN: Published with "
                                f"unresolved claim after "
                                f"{config.max_retries} attempts: {r.claim}"
                            )
                        print(
                            f"  [segment {seg_idx + 1}/{total_segments}] "
                            f"fact-check EXHAUSTED — "
                            f"{len(blocking)} issue(s) remain",
                            flush=True,
                        )

            # ── Checkpoint render to measure actual TTS duration ────
            # The combined text of all utterances in this segment is what
            # will eventually be spoken. Measuring it gives the budget
            # tracker real numbers to rebalance against.
            segment_text = " ".join(
                (e.get("text") or "").strip() for e in segment_events
            ).strip()
            self._checkpoint_segment(seg_idx, segment_text)

            # ── Record events ────────────────────────────────────────
            for evt in segment_events:
                all_events.append(evt)
                global_order += 1
                # ── Adaptive duration tracking ───────────────────────
                self._record_words(evt.get("text", ""))

            # ── Append approved text to prior-context buffer ─────────
            if segment_text:
                self._approved_segment_texts.append(segment_text)

            # ── Check if we've hit the budget ────────────────────────
            if self._over_budget():
                break

            # ── Record segment summary for continuity ────────────────
            summary = self._summarize_segment(segment_events, segment)
            self._context_log.append(summary)

            # ── Compress context periodically ────────────────────────
            if (
                compress_interval > 0
                and len(self._context_log) > compress_interval
                and len(self._context_log) % compress_interval == 0
            ):
                self._compress_context()

            # ── Director check-in ────────────────────────────────────
            # Periodically pause segment generation to ask the Director
            # agent for pacing/tone guidance. The note feeds into the next
            # segment's prompt so the show can self-correct mid-run.
            #
            # Cadence:
            #   * `check_interval` defaults to 3 segments. The director
            #     brief can override it (e.g. shorter for tight 10-segment
            #     shows, longer for sprawling 20+).
            #   * `seg_idx > 0` skips the very first segment — there's
            #     nothing produced yet for the Director to react to.
            #   * `(seg_idx + 1) % check_interval == 0` fires at the end
            #     of segment N, N+check_interval, N+2*check_interval, …
            #     We use `seg_idx + 1` (the 1-based count of completed
            #     segments) so that with check_interval=3 the first
            #     check-in happens AFTER segment 3, not 4.
            check_interval = 3
            if config.director_brief and config.director_brief.compression_interval:
                check_interval = config.director_brief.compression_interval
            if (
                config.director_brief
                and seg_idx > 0
                and (seg_idx + 1) % check_interval == 0
            ):
                self._director_check(seg_idx + 1, total_segments)

        return all_events

    # ═══════════════════════════════════════════════════════════════════
    # EXPANSION PASS
    # ═══════════════════════════════════════════════════════════════════

    def _expansion_pass(
        self,
        *,
        deficit_words: int,
        deficit_seconds: float,
    ) -> list[dict]:
        """Generate one bonus segment to close a length deficit.

        Reuses the existing segment prompt machinery — produces an extra
        ``deep_dive`` segment continuing the existing topic. Returns the
        new events for the conductor to splice in.
        """
        if self._session is None:
            return []
        config = self.config
        if not isinstance(config, PodcastConfig):
            return []

        # Pick a topic continuation — rotate through the original outline
        # so that multi-pass expansions don't keep extending the same beat.
        last_topic = ""
        if config.segments:
            idx = self._expansion_idx % len(config.segments)
            last_topic = config.segments[idx].topic or ""
            self._expansion_idx += 1
        host_names = [h.name for h in config.hosts]
        lead_host = host_names[self._expansion_idx % len(host_names)] if host_names else ""
        seg_id = f"seg_bonus_{self._expansion_idx:03d}"

        length_contract = self._length_contract_block(deficit_words)
        user_message = SEGMENT_PROMPT.format(
            segment_type="DEEP_DIVE",
            segment_topic=(
                f"BONUS — extend the conversation. Previous topic: {last_topic}. "
                "Pick up where you left off and add new substance: a fresh "
                "angle, a counterpoint, a real-world example. Do NOT recap."
            ),
            lead_host=lead_host or "any host",
            length_contract=length_contract,
            type_overlay=SEGMENT_TYPE_OVERLAYS.get("deep_dive", ""),
            context_summary=self._build_context_summary() or "(see prior segments)",
            director_note=(
                "DIRECTOR'S NOTE: We need extra runtime — give this segment "
                "real substance, not filler. Hosts should genuinely engage."
            ),
            runtime_status=(
                f"BONUS SEGMENT — episode is short by ~{deficit_seconds:.0f}s. "
                f"Target ~{deficit_words} additional words."
            ),
        )

        try:
            raw_response = self._session.send(user_message)
        except Exception as exc:  # noqa: BLE001
            self._length_warnings.append(
                f"[expansion] LLM call failed: {exc!r}"
            )
            return []

        new_events = self._parse_segment_response(
            raw_response,
            segment_id=seg_id,
            segment_type="deep_dive",
            lead_host=lead_host or "",
            global_order_start=0,  # Order is appended; absolute index reset.
        )
        for evt in new_events:
            self._record_words(evt.get("text", ""))
        return new_events

    def _package_result(self, raw_events: list[dict]) -> PodcastConductorResult:
        """Convert raw event dicts into a typed PodcastConductorResult.

        Groups events by segment_id, builds PodcastSegment objects, and
        constructs the voice_casting / voice_hints / tts_styles dicts.
        """
        config = self.config
        if not isinstance(config, PodcastConfig):
            raise TypeError(
                f"PodcastConductor requires PodcastConfig, got {type(config)}"
            )

        # Build typed PodcastEvent objects.
        podcast_events: list[PodcastEvent] = []
        for i, evt in enumerate(raw_events):
            tone_raw = evt.get("tone", "neutral")
            if isinstance(tone_raw, str):
                tone_tuple = (tone_raw,)
            elif isinstance(tone_raw, list):
                tone_tuple = tuple(tone_raw)
            else:
                tone_tuple = ("neutral",)

            podcast_events.append(PodcastEvent(
                global_order=evt.get("global_order", i),
                local_order=evt.get("local_order", i),
                speaker_role=evt.get("role", "host"),
                speaker_name=evt.get("speaker_name", evt.get("speaker", "Unknown")),
                target=evt.get("target"),
                text=evt.get("text", ""),
                tone=tone_tuple,
                volume=evt.get("volume"),
                pace=evt.get("pace"),
                pause_before=evt.get("pause_before"),
                pause_after=evt.get("pause_after"),
                interruption=evt.get("interruption", False),
                nonverbal=tuple(evt.get("nonverbal", [])),
                event_kind=evt.get("event_kind", "utterance"),
                segment_id=evt.get("segment_id"),
            ))

        # Group events into PodcastSegment objects.
        segment_map: dict[str, list[PodcastEvent]] = {}
        for evt in podcast_events:
            if evt.segment_id:
                segment_map.setdefault(evt.segment_id, []).append(evt)

        podcast_segments: list[PodcastSegment] = []
        for seg_idx, seg_cfg in enumerate(config.segments):
            seg_id = f"seg_{seg_idx:03d}"
            evts = segment_map.get(seg_id, [])
            podcast_segments.append(PodcastSegment(
                segment_id=seg_id,
                segment_order=seg_idx + 1,
                segment_type=seg_cfg.segment_type,
                topic=seg_cfg.topic,
                events=tuple(evts),
            ))

        transcript_lines = [
            f"[{evt.get('speaker_name', evt.get('speaker', '?'))}] "
            f"{evt.get('text', '')}"
            for evt in raw_events
        ]

        return PodcastConductorResult(
            config=config,
            events=raw_events,
            transcript=transcript_lines,
            total_llm_calls=self._total_session_calls + self._llm_calls,
            director_notes=[self._director_note] if self._director_note else [],
            podcast_events=podcast_events,
            podcast_segments=podcast_segments,
            podcast_id=self._podcast_id,
        )

    # ═══════════════════════════════════════════════════════════════════
    # HOST DESCRIPTION BUILDER
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_host_descriptions(hosts: list[PodcastHostProfile]) -> str:
        """Build the host description block for the system prompt.

        Each host gets a structured description injected into the system
        prompt so the LLM knows the full cast it's voice-acting.
        """
        descriptions = []
        for host in hosts:
            desc = HOST_DESCRIPTION_TEMPLATE.format(
                name=host.name,
                role=host.role,
                speech_style=host.speech_style or "natural and engaging",
                voice_fingerprint=host.voice_fingerprint or "(no specific voice style)",
                knowledge_domains=", ".join(host.knowledge_domains) or "general",
                forbidden_knowledge=", ".join(host.forbidden_knowledge) or "none",
                interruption_tendency=host.interruption_tendency,
                behavioral_triggers=host.behavioral_triggers or "(none specified)",
                tangent_topics=", ".join(host.tangent_topics) or "none",
            )
            descriptions.append(desc)
        return "\n".join(descriptions)

    # ═══════════════════════════════════════════════════════════════════
    # RESPONSE PARSING
    # ═══════════════════════════════════════════════════════════════════

    def _parse_segment_response(
        self,
        raw: str,
        *,
        segment_id: str,
        segment_type: str,
        lead_host: str,
        global_order_start: int,
    ) -> list[dict]:
        """Parse the LLM's JSON array response into a list of event dicts.

        Expected format: a JSON array of utterance objects:
        [
          {"speaker_name": "Host A", "text": "...", "tone": "...", ...},
          {"speaker_name": "Host B", "text": "...", "tone": "...", ...},
        ]

        Fallback: if parsing fails entirely, return a single event with the
        raw text attributed to the lead host.

        Args:
            raw:                 Raw LLM response string.
            segment_id:          ID for this segment.
            segment_type:        Type of segment (for event_kind tagging).
            lead_host:           Name of the lead host (fallback speaker).
            global_order_start:  Starting global_order index.

        Returns:
            List of event dicts, each with speaker_name, text, tone, etc.
        """
        cleaned = raw.strip()

        # Strip markdown fences if present.
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        # Find the JSON array.
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat the raw text as a single utterance by lead host.
            return [{
                "speaker_name": lead_host,
                "text": raw.strip()[:1000],
                "tone": "neutral",
                "volume": "moderate",
                "pace": "medium",
                "role": "host",
                "event_kind": "utterance",
                "segment_id": segment_id,
                "global_order": global_order_start,
                "local_order": 0,
            }]

        events = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            events.append({
                "speaker_name": item.get("speaker_name", lead_host),
                "text": item.get("text", ""),
                "tone": item.get("tone", "neutral"),
                "volume": item.get("volume", "moderate"),
                "pace": item.get("pace", "medium"),
                "pause_before": item.get("pause_before", "none"),
                "pause_after": item.get("pause_after", "none"),
                "interruption": item.get("interruption", False),
                "nonverbal": item.get("nonverbal", []),
                "role": "host",
                "event_kind": "utterance",
                "segment_id": segment_id,
                "global_order": global_order_start + i,
                "local_order": i,
            })

        # If we got an empty array somehow, fallback.
        if not events:
            events.append({
                "speaker_name": lead_host,
                "text": "(segment generation produced no content)",
                "tone": "neutral",
                "volume": "moderate",
                "pace": "medium",
                "role": "host",
                "event_kind": "utterance",
                "segment_id": segment_id,
                "global_order": global_order_start,
                "local_order": 0,
            })

        return events

    # ═══════════════════════════════════════════════════════════════════
    # CONTEXT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _build_prior_context(self) -> str | None:
        """Return concatenated approved segment texts for consistency checking.

        Returns ``None`` when no segments have been approved yet (skips the
        consistency pass entirely, avoiding an unnecessary LLM call).
        Oldest-first joining gives the consistency verifier a chronological
        view; :func:`~vox_romana.core.fact_checker._trim_prior_context` will
        trim to the most-recent 4 000 chars when needed.
        """
        if not self._approved_segment_texts:
            return None
        return "\n---\n".join(self._approved_segment_texts)

    @staticmethod
    def _summarize_segment(
        events: list[dict],
        segment_config: PodcastSegmentConfig,
    ) -> str:
        """Create a brief summary of a segment for the context log.

        This is NOT an LLM call — it's a simple string describing
        what happened in the segment, used for continuity tracking.
        """
        speakers = set()
        text_snippets = []
        for evt in events:
            name = evt.get("speaker_name", evt.get("speaker", "?"))
            speakers.add(name)
            text = evt.get("text", "")
            if text:
                text_snippets.append(f"{name}: {text[:80]}")

        speaker_list = ", ".join(sorted(speakers))
        snippet_text = " | ".join(text_snippets[:3])  # First 3 turns
        return (
            f"{segment_config.topic} — "
            f"speakers: {speaker_list}. {snippet_text}"
        )

    def _build_context_summary(self) -> str:
        """Build the context summary from compressed + recent log entries.

        Combines the compressed older context with the most recent segment
        summaries for continuity.
        """
        parts = []
        if self._compressed_ctx:
            parts.append(f"Earlier: {self._compressed_ctx}")

        # Keep the last few segment summaries verbatim.
        recent_count = 3
        recent = self._context_log[-recent_count:] if self._context_log else []
        if recent:
            parts.append("The conversation so far:\n" + "\n".join(f"  - {s}" for s in recent))

        return "\n\n".join(parts) if parts else ""

    def _compress_context(self) -> None:
        """Compress older segment summaries into a shorter context block.

        Uses the LLM to summarize the accumulated context log into 2-3
        sentences. The recent entries are kept verbatim; older ones are
        replaced by the compressed summary.
        """
        if not self._context_log:
            return

        # Keep the last 2 entries, compress everything before them.
        keep_recent = 2
        to_compress = self._context_log[:-keep_recent] if len(self._context_log) > keep_recent else []
        if not to_compress:
            return

        transcript_block = "\n".join(to_compress)
        prompt = CONTEXT_SUMMARY_PROMPT.format(transcript=transcript_block)

        if self._session is not None:
            # Use a separate one-shot call for compression (don't pollute
            # the main session's history with meta-prompts).
            raw = self._client.chat(
                messages=[
                    {"role": "system", "content": "You summarize podcast conversations concisely."},
                    {"role": "user", "content": prompt},
                ]
            )
            self._llm_calls += 1
        else:
            raw = self._client.chat(
                messages=[
                    {"role": "system", "content": "You summarize podcast conversations concisely."},
                    {"role": "user", "content": prompt},
                ]
            )
            self._llm_calls += 1

        self._compressed_ctx = raw.strip()[:500]
        # Replace compressed entries with just the recent ones.
        self._context_log = self._context_log[-keep_recent:]

    # ═══════════════════════════════════════════════════════════════════
    # DIRECTOR CHECK
    # ═══════════════════════════════════════════════════════════════════

    def _director_check(self, segment_number: int, total_segments: int) -> dict:
        """Ask the Director to review progress and issue a pacing note.

        The Director reads the episode arc, current mood target, and recent
        content, then provides a mood_note and optional nudge.
        """
        config = self.config
        if not isinstance(config, PodcastConfig):
            return {}

        director = config.director_brief
        if not director:
            return {}

        # Current mood target from the progression.
        if director.mood_progression:
            progress = segment_number / max(total_segments, 1)
            mood_index = min(
                int(progress * len(director.mood_progression)),
                len(director.mood_progression) - 1,
            )
            current_mood = director.mood_progression[mood_index]
        else:
            current_mood = "neutral"

        # Recent content from context log.
        recent = self._context_log[-2:] if self._context_log else ["(no content yet)"]
        recent_text = "\n".join(recent)

        prompt = DIRECTOR_CHECK_PROMPT.format(
            segment_number=segment_number,
            total_segments=total_segments,
            arc=director.arc,
            current_mood=current_mood,
            closing_strategy=director.closing_strategy or "(not specified)",
            recent_content=recent_text,
        )

        raw = self._client.chat(
            messages=[
                {"role": "system", "content": "You are a podcast Director. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        self._llm_calls += 1

        # Parse director response.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return {}

        # Apply the director's note.
        mood_note = data.get("mood_note", "")
        nudge = data.get("nudge")
        if mood_note:
            self._director_note = mood_note
            if nudge:
                self._director_note += f" | {nudge}"

        return data
