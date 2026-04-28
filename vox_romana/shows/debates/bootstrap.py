"""The Bootstrap orchestrator — chains Architect → Profile Builder → Director.

This is the conductor of the bootstrap assembly line (not to be confused with
the DebateConductor, which runs the actual debate). It coordinates the three
bootstrap agents in sequence:

    1. ARCHITECT: brief → skeleton config (cast, format, ground rules)
    2. PROFILE BUILDER: stub → full personality (one per participant, parallelizable)
    3. DIRECTOR: config → dramatic arc (tension, pacing, resolution)

After all three stages, any user-provided overrides are applied on top. The
result is a fully populated DebateConfig ready for the DebateConductor.

OVERRIDE SYSTEM:
    Overrides let you auto-generate most of the config but manually tweak
    specific parts. Think of it like ordering a pizza with modifications:

        config = bootstrap.from_brief(
            "Two economists argue about UBI",
            overrides={
                "debate_format": "oxford",           # override the architect's choice
                "target_turns": 50,                   # longer than default
                "tone": "cable news energy",          # spicier than the architect picked
            },
        )

    The override keys must match DebateConfig field names. Unknown keys are
    silently ignored (they might be for a future version).

WHY NOT ASYNC?
    Profile building COULD be async (each participant is independent), but
    the current implementation is synchronous for simplicity. The plan says
    to add AsyncOpenAI later when batch production warrants it. Right now,
    a 3-participant bootstrap takes ~3 sequential LLM calls for profiles
    (~5-10 seconds total). Not worth the async complexity yet.

EXAMPLE — full auto:
    from vox_romana.shows.debates.bootstrap import DebateBootstrap

    bootstrap = DebateBootstrap()     # uses DEEPSEEK_API_KEY from env
    config = bootstrap.from_brief(
        "Should cities ban cars from downtown areas?",
        reference_material="Barcelona's superblocks program reduced...",
        fact_check="quick",
    )
    # config is now fully populated and ready for DebateConductor(config)

EXAMPLE — auto with overrides:
    config = bootstrap.from_brief(
        "Pineapple on pizza: culinary innovation or war crime?",
        overrides={"tone": "unhinged cable news", "target_turns": 20},
    )

EXAMPLE — mock client for testing:
    from unittest.mock import Mock
    mock_client = Mock()
    mock_client.chat.return_value = '{"title": "Test", ...}'
    bootstrap = DebateBootstrap(client=mock_client)
"""

from __future__ import annotations

from typing import Any

from ...core.bootstrap import BaseShowBootstrap
from ...core.client import DeepSeekClient
from ...core.search import BaseSearchProvider, build_research_provider
from .architect import DebateArchitect
from .config import DebateConfig, ParticipantProfile
from .director import DebateDirector
from .profile_builder import DebateProfileBuilder
from .researcher import DebateResearcher


class DebateBootstrap(BaseShowBootstrap):
    """Orchestrates the brief-to-config bootstrap chain.

    Takes a human's natural language brief and produces a fully populated
    DebateConfig by chaining:  Architect → Profile Builder(s) → Director.

    Attributes:
        _client:    Shared DeepSeekClient passed to all bootstrap agents.
        _architect: The Architect agent instance.
        _profiler:  The Profile Builder agent instance.
        _director:  The Director Brief agent instance.
    """

    def __init__(self, client: DeepSeekClient | None = None, *, api_key: str | None = None) -> None:
        """Initialize the bootstrap orchestrator.

        Args:
            client:  An existing DeepSeekClient to use. If not provided,
                     creates a new one using api_key or DEEPSEEK_API_KEY env var.
            api_key: API key for DeepSeekClient. Only used if client is None.
        """
        # All three agents share the same client. In production, this means
        # they share the same HTTP connection pool. In tests, you inject one
        # mock that handles all three stages.
        super().__init__(client, api_key=api_key)
        self._architect = DebateArchitect(self._client)
        self._profiler = DebateProfileBuilder(self._client)
        self._director = DebateDirector(self._client)
        # The researcher is created lazily in from_brief() because it
        # needs a search_provider, which is optional and passed per-call.
        self._researcher: DebateResearcher | None = None

    def from_brief(
        self,
        brief: str,
        *,
        reference_material: str | None = None,
        fact_check: str = "off",
        overrides: dict | None = None,
        auto_research: bool = False,
        search_provider: BaseSearchProvider | None = None,
        memory: Any | None = None,
        series_context: str | None = None,
    ) -> DebateConfig:
        """Run the full bootstrap chain: brief → fully populated DebateConfig.

        This is the main entry point. Give it a brief (and optionally some
        reference material and overrides), get back a complete config.

        Args:
            brief:              Natural language description of the debate.
            reference_material: Factual source text to ground the output.
            fact_check:         "off", "quick", or "thorough".
            overrides:          Dict of DebateConfig field names to override.
                                Applied AFTER all agents run.
            auto_research:      If True AND reference_material is empty/None,
                                automatically research the topic via web search
                                before running the Architect. Requires a search_provider.
            search_provider:    A BaseSearchProvider instance (e.g. GoogleSearchProvider).
                                Required when auto_research=True. Ignored otherwise.
            memory:             Optional MemoryRecall instance. When provided,
                                auto-research consults the local knowledge base
                                FIRST and only falls through to ``search_provider``
                                if memory doesn't have enough material. Free,
                                fast, and domain-tuned. See
                                ``vox_romana.core.search.MemorySearchProvider``.

        Returns:
            A fully populated DebateConfig with:
            - Cast, format, ground rules (from the Architect)
            - Full personality profiles (from the Profile Builder)
            - Dramatic arc plan (from the Director)
            - User overrides applied on top

        Raises:
            ValueError: If any agent returns unparseable output, or if
                        auto_research=True but no search_provider is given.
        """
        overrides = overrides or {}

        # If a series bible/context block was supplied, prepend it to the
        # brief so every downstream agent (Architect, Profile Builder,
        # Director) sees the recurring tone, world facts, and prior-episode
        # canon before they read the user's specific prompt.
        if series_context:
            brief = f"{series_context}\n\n{brief}"

        # ══════════════════════════════════════════════════════════════
        # STAGE 0: AUTO-RESEARCH (optional)
        # ══════════════════════════════════════════════════════════════
        # If the caller wants auto research AND didn't provide their own
        # reference material, we fire up the DebateResearcher to search
        # the web and compile a fact sheet. This becomes the reference
        # material for every subsequent stage.
        #
        # Think of it like a production assistant running to the library
        # before the show starts — getting everyone their prep packet.
        if auto_research and not reference_material:
            # Build the effective search provider. Memory comes FIRST when
            # available — it's free, sub-millisecond, and domain-tuned. We
            # only fall through to the (paid, slower) web search when memory
            # doesn't have enough material to ground the topic.
            effective_provider = build_research_provider(search_provider, memory)
            if effective_provider is None:
                # No search provider AND no memory → LLM-only research
                # (DeepSeek compiles from training knowledge). Still useful.
                researcher = DebateResearcher(self._client)
            else:
                researcher = DebateResearcher(self._client, effective_provider)
            reference_material = researcher.run(brief) or None

        # ══════════════════════════════════════════════════════════════
        # STAGE 1: ARCHITECT
        # ══════════════════════════════════════════════════════════════
        # The Architect reads the brief and produces the skeleton:
        # title, topic, format, participant stubs, moderator, audience.
        # Think of it as the casting call + venue booking.
        config = self._architect.run(
            brief,
            reference_material=reference_material,
            fact_check=fact_check,
        )

        # ══════════════════════════════════════════════════════════════
        # STAGE 2: PROFILE BUILDER (one per participant)
        # ══════════════════════════════════════════════════════════════
        # Now we take each participant stub and ask the Profile Builder
        # to flesh it out: voice fingerprint, behavioral triggers,
        # emotional arc, key positions, factual claims.
        #
        # This is currently sequential. When we add async support, this
        # loop becomes an asyncio.gather() — each profile is independent.
        fleshed_out = []
        for stub in config.participants:
            full_profile = self._profiler.run(
                stub,
                topic=config.topic or brief,
                reference_material=reference_material,
            )
            fleshed_out.append(full_profile)

        # Replace the stubs with the fully fleshed-out profiles.
        config.participants = fleshed_out

        # ══════════════════════════════════════════════════════════════
        # STAGE 2.5: SHOW HOST PROFILING (if the Architect included one)
        # ══════════════════════════════════════════════════════════════
        # If the Architect decided this show needs a narrative host (it
        # usually does — they make shows sound polished), we need to give
        # them a voice_fingerprint. The host gets a single LLM call via
        # the Profile Builder, same as participants, but with role="host".
        #
        # Think of it like hiring the announcer AFTER you've cast the
        # actors — the announcer's style should complement the cast.
        if config.show_host and config.show_host.name:
            # Build a stub ParticipantProfile for the host so we can reuse
            # the Profile Builder. The host isn't a debater, but the Profile
            # Builder's voice_fingerprint output is exactly what we need.
            host_stub = ParticipantProfile(
                name=config.show_host.name,
                side=None,
                role="host",
            )
            host_profile = self._profiler.run(
                host_stub,
                topic=config.topic or brief,
                reference_material=reference_material,
            )
            # Copy voice casting fields from the profile. The host
            # doesn't have "key positions" or "factual claims" — they narrate,
            # they don't argue.
            config.show_host.voice_fingerprint = host_profile.voice_fingerprint
            config.show_host.voice_hint = host_profile.voice_hint
            config.show_host.voice_gender = host_profile.voice_gender

        # ══════════════════════════════════════════════════════════════
        # STAGE 3: DIRECTOR BRIEF
        # ══════════════════════════════════════════════════════════════
        # The Director reads the full config (with fleshed-out participants)
        # and plans the dramatic arc: mood progression, key moments, and
        # closing strategy.
        #
        # We pass participant summaries (name + side) so the Director can
        # design moments that leverage specific personalities.
        participant_summaries = [
            f"{p.name} ({p.side or 'unspecified'})" for p in config.participants
        ]
        config.director_brief = self._director.run(
            topic=config.topic or brief,
            participants=participant_summaries,
            debate_format=config.debate_format or "public_forum",
            tone=config.tone or "balanced",
            target_turns=config.target_turns or 35,
        )

        # ══════════════════════════════════════════════════════════════
        # STAGE 4: APPLY OVERRIDES
        # ══════════════════════════════════════════════════════════════
        # The user gets the last word. Any field in `overrides` replaces
        # whatever the agents produced. This is how you say "I like what
        # you came up with, but make it 50 turns and set the tone to
        # 'cable news energy'."
        self._apply_overrides(config, overrides)

        return config

    @staticmethod
    def _apply_overrides(config: DebateConfig, overrides: dict | None) -> None:
        """Apply user overrides to a DebateConfig (delegates to base).

        Kept for backward compatibility — callers/tests may import
        :meth:`DebateBootstrap._apply_overrides` directly. The actual
        implementation lives on :class:`BaseShowBootstrap`.
        """
        BaseShowBootstrap._apply_overrides(config, overrides)
