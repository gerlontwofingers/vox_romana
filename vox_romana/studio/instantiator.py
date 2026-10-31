"""Studio instantiator — orchestrate staged inference into a real studio.

Shot 5.2-b of the Studio Instantiation feature.  This module chains the
three staged prompts from :mod:`vox_romana.studio.prompts` into a single
orchestration that:

    1. Runs META → BIBLE → AUDIENCE inference (3 LLM calls).
    2. Creates the :class:`~vox_romana.core.catalog.SeriesRecord` in the
       Catalog (the source of truth), attaching the audience profile.
    3. Optionally drafts episode 1's brief via
       :class:`~vox_romana.planning.episode_planner.EpisodePlanner`.

Mock-friendly: the client is anything with a ``.chat()`` shape; the
catalog is anything exposing ``create_series`` / ``next_episode_number`` /
``get_recent_episode_summaries``.  Tests pass stubs; production passes
``DeepSeekClient`` and ``Catalog``.

NOTE on series identity:
    The Catalog assigns a UUID as the authoritative ``series_id`` (its
    ``create_series`` ignores any caller-supplied key).  The human-friendly
    slug from Stage-1 META is kept separately as ``slug`` — used later (shot
    5.2-c) to name the on-disk bundle directory, while the UUID remains the
    canonical key recorded inside ``studio.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..core.audience import AudienceProfile
from ..planning.episode_planner import EpisodeBrief, EpisodePlanner
from .prompts import (
    AUDIENCE_SYSTEM_PROMPT,
    BIBLE_SYSTEM_PROMPT,
    META_SYSTEM_PROMPT,
    StudioMeta,
    build_audience_user_prompt,
    build_bible_user_prompt,
    build_meta_user_prompt,
    parse_audience,
    parse_bible,
    parse_meta,
)


# ─────────────────────────────────────────────────────────────────────────────
# Protocols (keeps tests trivial)
# ─────────────────────────────────────────────────────────────────────────────


class _ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class _Catalog(Protocol):
    def create_series(
        self,
        *,
        title: str,
        show_type: str | None = None,
        tone: str | None = None,
        bible: str = "",
        audience_profile: AudienceProfile | None = None,
    ) -> str: ...

    def next_episode_number(self, series_id: str) -> int: ...

    def get_recent_episode_summaries(
        self, series_id: str, n: int = 5
    ) -> list[dict[str, Any]]: ...


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StudioInstantiation:
    """Everything produced by a single :meth:`StudioInstantiator.instantiate`.

    Attributes:
        meta:          The Stage-1 metadata (title, show_type, slug, …).
        series_id:     The Catalog-assigned UUID — the canonical key.
        bible:         The Stage-2 writers' bible text.
        audience:      The Stage-3 audience profile.
        episode_brief: Episode 1's drafted brief, or None when drafting
                       was skipped (or failed non-fatally).
    """

    meta: StudioMeta
    series_id: str
    bible: str
    audience: AudienceProfile
    episode_brief: EpisodeBrief | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Instantiator
# ─────────────────────────────────────────────────────────────────────────────


class StudioInstantiator:
    """Turn a one-sentence brief into a registered studio.

    Stateless w.r.t. the client: one instantiator can create many studios.
    """

    def __init__(
        self,
        client: _ChatClient,
        *,
        catalog: _Catalog | None = None,
    ) -> None:
        """Initialize with an LLM client and an optional catalog.

        Args:
            client:  The LLM chat client (DeepSeekClient-shaped).
            catalog: Optional catalog.  When omitted, a real
                     :class:`~vox_romana.core.catalog.Catalog` is created
                     lazily at instantiate() time.
        """
        self._client = client
        self._catalog = catalog

    # ── Stage helpers (each is independently testable) ──────────────────

    def _infer_meta(self, brief: str) -> StudioMeta:
        raw = self._client.chat(
            [
                {"role": "system", "content": META_SYSTEM_PROMPT},
                {"role": "user", "content": build_meta_user_prompt(brief)},
            ],
            temperature=0,
        )
        return parse_meta(raw)

    def _infer_bible(self, meta: StudioMeta) -> str:
        raw = self._client.chat(
            [
                {"role": "system", "content": BIBLE_SYSTEM_PROMPT},
                {"role": "user", "content": build_bible_user_prompt(meta)},
            ],
            temperature=0.6,
        )
        return parse_bible(raw)

    def _infer_audience(self, meta: StudioMeta) -> AudienceProfile:
        raw = self._client.chat(
            [
                {"role": "system", "content": AUDIENCE_SYSTEM_PROMPT},
                {"role": "user", "content": build_audience_user_prompt(meta)},
            ],
            temperature=0,
        )
        return parse_audience(raw)

    def _draft_episode(
        self,
        *,
        meta: StudioMeta,
        series_id: str,
        bible: str,
        catalog: _Catalog,
        topic: str | None,
    ) -> EpisodeBrief | None:
        """Draft episode 1 via EpisodePlanner (non-fatal on failure)."""
        planner = EpisodePlanner(self._client)
        recent = catalog.get_recent_episode_summaries(series_id, n=5)
        episode_number = catalog.next_episode_number(series_id)
        try:
            return planner.plan_next(
                series_title=meta.title,
                series_show_type=meta.show_type,
                series_tone=meta.tone,
                bible=bible,
                recent_episodes=recent,
                next_episode_number=episode_number,
                topic=topic,
            )
        except Exception:
            # Drafting episode 1 is a nice-to-have, never a hard failure —
            # the studio record is already valid without it.
            return None

    # ── Public orchestration ────────────────────────────────────────────

    def instantiate(
        self,
        brief: str,
        *,
        draft_episode: bool = True,
        topic: str | None = None,
    ) -> StudioInstantiation:
        """Run the full staged inference and register the studio.

        Args:
            brief:          One-sentence description of the studio/show.
            draft_episode:  When True (default), also draft episode 1's
                            brief via EpisodePlanner.
            topic:          Optional topic constraint passed through to the
                            episode planner.

        Returns:
            A :class:`StudioInstantiation` carrying meta, series_id, bible,
            audience, and (optionally) the episode-1 brief.

        Raises:
            ValueError:  If Stage-1 meta inference returns invalid JSON.
        """
        # ── Stage 1: meta ────────────────────────────────────────────
        meta = self._infer_meta(brief)

        # ── Stage 2: bible ───────────────────────────────────────────
        bible = self._infer_bible(meta)

        # ── Stage 3: audience ────────────────────────────────────────
        audience = self._infer_audience(meta)

        # ── Register in the catalog (source of truth) ────────────────
        catalog = self._catalog
        if catalog is None:
            from ..core.catalog import Catalog
            catalog = Catalog()

        series_id = catalog.create_series(
            title=meta.title,
            show_type=meta.show_type,
            tone=meta.tone,
            bible=bible,
            audience_profile=audience,
        )

        # ── Optional: draft episode 1 ────────────────────────────────
        episode_brief = None
        if draft_episode:
            episode_brief = self._draft_episode(
                meta=meta,
                series_id=series_id,
                bible=bible,
                catalog=catalog,
                topic=topic,
            )

        return StudioInstantiation(
            meta=meta,
            series_id=series_id,
            bible=bible,
            audience=audience,
            episode_brief=episode_brief,
        )
