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

from .config import ContentConfig


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
    """

    def __init__(self, config: ContentConfig) -> None:
        self.config = config

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
