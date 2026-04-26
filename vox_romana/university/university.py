"""University — multi-Scholar deep-research orchestrator.

The University coordinates one or more :class:`Scholar` instances and
links each completed session into the catalog so downstream show
generation can find and cite the research.

Three public methods cover the planned use cases:

* :meth:`University.study` — single Scholar on one topic. Auto-logs to
  the catalog on completion (if a :class:`Catalog` was provided).
* :meth:`University.study_many` — sequentially runs one Scholar per
  sub-topic. Each session checkpoints independently so a Ctrl-C only
  loses the in-flight Scholar's most-recent article.
* :meth:`University.cross_synthesize` — LLM-driven merge of several
  session syntheses into one document, useful when one show needs
  research from several angles (e.g. a debate covering both sides).

USAGE::

    from vox_romana.university import University, WikipediaSource, LocalSource
    from vox_romana.local import OllamaClient
    from vox_romana.core.catalog import Catalog

    uni = University(
        client=OllamaClient.from_config(),
        sources=[WikipediaSource(), LocalSource(memory_recall)],
        catalog=Catalog(),
    )

    # Single study
    session = uni.study("Roman engineering", budget_hours=2.0)
    print(session.synthesis)
    print("study_id:", session.catalog_study_id)

    # Multi-angle study
    sessions = uni.study_many(
        ["pro arguments for nuclear power", "anti arguments for nuclear power"],
        budget_hours_each=1.0,
    )
    debate_brief = uni.cross_synthesize(
        sessions, directive="Frame this as a balanced debate brief."
    )

Concurrency
-----------
Scholars run **sequentially** in this phase. A future phase may run them
in parallel — the API is already shaped for that (each Scholar gets its
own checkpointed session, and ``study_many`` returns a list).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from vox_romana.university.config import UniversityConfig
from vox_romana.university.scholar import (
    Scholar,
    _MAX_SYNTHESIS_CHARS_IN_PROMPT,
    _strip_think,
)
from vox_romana.university.session import ScholarSession

if TYPE_CHECKING:
    from vox_romana.core.catalog import Catalog
    from vox_romana.core.client import DeepSeekClient
    from vox_romana.university.sources import KnowledgeSource

logger = logging.getLogger(__name__)


class University:
    """Multi-Scholar research orchestrator.

    Attributes:
        client:        LLM client passed to every spawned Scholar (and used
                       for the cross-synthesis pass).
        sources:       KnowledgeSources shared across Scholars.
        config:        UniversityConfig.
        catalog:       Optional :class:`Catalog`; when present, every
                       completed study is logged via ``log_study()`` and
                       the resulting study_id is written back onto the
                       :class:`ScholarSession`.
        run_id:        Optional run_id to link studies to a parent show
                       run (forwarded into ``log_study(run_id=...)``).
        content_type:  Optional content-type tag forwarded into
                       ``log_study(content_type=...)``.
    """

    def __init__(
        self,
        client: "DeepSeekClient",
        sources: "list[KnowledgeSource]",
        config: UniversityConfig | None = None,
        *,
        catalog: "Catalog | None" = None,
        run_id: str | None = None,
        content_type: str = "",
    ):
        if not sources:
            raise ValueError("University needs at least one KnowledgeSource.")
        self.client = client
        self.sources = list(sources)
        self.config = config or UniversityConfig.from_project_yaml()
        self.catalog = catalog
        self.run_id = run_id
        self.content_type = content_type

    # ── Scholar lifecycle ───────────────────────────────────────────────

    def spawn_scholar(self) -> Scholar:
        """Build a fresh Scholar with this University's wiring.

        A new Scholar is returned every call — Scholars are not safe to
        share across concurrent studies.
        """
        return Scholar(self.client, self.sources, self.config)

    # ── single study ────────────────────────────────────────────────────

    def study(
        self,
        topic: str,
        *,
        budget_hours: float | None = None,
        save_dir: Path | None = None,
        seed_article_ids: list[str] | None = None,
    ) -> ScholarSession:
        """Run one Scholar on ``topic``.

        Equivalent to ``self.spawn_scholar().study(topic, ...)`` plus
        catalog logging on completion.

        Args:
            topic:            Research directive.
            budget_hours:     Hard time budget. ``None`` = Scholar self-estimates.
            save_dir:         Root for session persistence (default
                              :data:`UNIVERSITY_ROOT`).
            seed_article_ids: Optional pre-seeded queue.

        Returns:
            The completed :class:`ScholarSession`.
        """
        scholar = self.spawn_scholar()
        session = scholar.study(
            topic,
            budget_hours=budget_hours,
            save_dir=save_dir,
            seed_article_ids=seed_article_ids,
        )
        self._catalog_log(session, save_dir=save_dir)
        return session

    def extend(
        self,
        session: ScholarSession,
        *,
        additional_hours: float | None = None,
        save_dir: Path | None = None,
    ) -> ScholarSession:
        """Resume a paused/interrupted session for more research time."""
        scholar = self.spawn_scholar()
        session = scholar.extend(
            session,
            additional_hours=additional_hours,
            save_dir=save_dir,
        )
        self._catalog_log(session, save_dir=save_dir)
        return session

    # ── multi-topic study ───────────────────────────────────────────────

    def study_many(
        self,
        topics: list[str],
        *,
        budget_hours_each: float | None = None,
        save_dir: Path | None = None,
    ) -> list[ScholarSession]:
        """Run one Scholar per topic, sequentially.

        Each session is checkpointed independently. If a Scholar is
        interrupted (SIGINT) the loop stops and returns the sessions
        completed so far plus the interrupted one.

        Args:
            topics:            One topic per sub-study. Empty topics are skipped.
            budget_hours_each: Per-topic budget. ``None`` = each Scholar
                               self-estimates.
            save_dir:          Root for session persistence.

        Returns:
            List of :class:`ScholarSession` in topic order.
        """
        sessions: list[ScholarSession] = []
        for i, topic in enumerate(topics):
            topic = (topic or "").strip()
            if not topic:
                logger.warning("Skipping empty topic at index %d", i)
                continue

            logger.info(
                "University: starting study %d/%d — %r", i + 1, len(topics), topic
            )
            session = self.study(
                topic,
                budget_hours=budget_hours_each,
                save_dir=save_dir,
            )
            sessions.append(session)

            # If a Scholar was interrupted, stop the multi-topic loop —
            # the user clearly wants to bail out, not move to the next topic.
            if session.status == "interrupted":
                logger.info(
                    "University: Scholar interrupted — stopping multi-topic loop "
                    "after %d/%d studies.", i + 1, len(topics)
                )
                break

        return sessions

    # ── cross-synthesis ─────────────────────────────────────────────────

    def cross_synthesize(
        self,
        sessions: list[ScholarSession],
        directive: str = "",
    ) -> str:
        """Merge several session syntheses into one document.

        Useful when a downstream show needs a unified brief built from
        multiple angles (e.g. pro/anti for a debate, multiple subtopics
        for a podcast).

        Args:
            sessions:  Sessions to merge. Empty sessions are skipped.
                       Order is preserved as a hint to the LLM.
            directive: Optional framing instruction (e.g. "Write as a
                       balanced debate brief").  When blank, a generic
                       merge instruction is used.

        Returns:
            The merged synthesis as plain Markdown. Empty string if
            there's nothing to merge or the LLM call fails.
        """
        usable = [s for s in sessions if (s.synthesis or s.notes)]
        if not usable:
            logger.warning("cross_synthesize called with no usable sessions.")
            return ""

        # Build per-session blocks. Cap each block so the merged prompt
        # stays a reasonable size even with 4–6 sessions.
        per_block_cap = max(800, _MAX_SYNTHESIS_CHARS_IN_PROMPT // max(1, len(usable)))
        blocks: list[str] = []
        for s in usable:
            synth = s.synthesis or "[no synthesis written — using notes only]"
            if len(synth) > per_block_cap:
                synth = synth[:per_block_cap] + "\n…[truncated]"
            blocks.append(
                f"### Session: {s.topic}\n"
                f"  notes: {len(s.notes)} | sources: "
                f"{', '.join(s.sources_used) or '—'}\n\n"
                f"{synth}"
            )

        directive = directive.strip() or (
            "Merge the per-session syntheses below into a single, coherent "
            "Markdown document. Preserve the distinct sub-topics as sections; "
            "highlight cross-cutting themes; do not invent material that is "
            "not in the syntheses."
        )

        system = (
            "You are the University's editor, merging research from several "
            "Scholars into one document. Be faithful: every claim must trace "
            "to one of the per-session syntheses. Cite the originating "
            "session topic inline like [Session: <topic>]."
        )
        user = (
            f"DIRECTIVE: {directive}\n\n"
            f"SESSIONS TO MERGE ({len(usable)}):\n"
            + "\n\n".join(blocks)
            + "\n\nProduce the merged document in plain Markdown. "
            "No JSON, no code fences."
        )

        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.config.temperature,
            )
        except Exception:
            logger.exception("cross_synthesize: LLM call failed")
            return ""

        return _strip_think(raw or "")

    # ── catalog wiring ──────────────────────────────────────────────────

    def _catalog_log(
        self,
        session: ScholarSession,
        *,
        save_dir: Path | None = None,
    ) -> None:
        """Log a completed session to the catalog (no-op if no catalog).

        The session's ``catalog_study_id`` is set in-place and the session
        is re-saved so the linkage is durable.

        Notes are passed through as ``fact_cards`` (each note is a small
        ScholarNote dict; the catalog stores them as JSON).  This makes
        every Scholar note discoverable via the existing study viewers
        without any new code.
        """
        if self.catalog is None:
            return

        # Don't log empty studies — the catalog should reflect real work.
        if not session.notes and not session.synthesis:
            logger.info(
                "Skipping catalog log for session %s — no notes and no synthesis.",
                session.session_id,
            )
            return

        # Skip if already logged (e.g. extend() of an already-cataloged session).
        if session.catalog_study_id:
            logger.debug(
                "Session %s already cataloged as %s — skipping.",
                session.session_id, session.catalog_study_id,
            )
            return

        report_path = ""
        try:
            report_path = str(session.session_dir(save_dir))
        except Exception:
            pass

        try:
            study_id = self.catalog.log_study(
                run_id=self.run_id,
                content_type=self.content_type,
                kind="university_research",
                dataset=session.topic,
                method=f"scholar/{self.config.model}",
                findings=session.synthesis,
                fact_cards=[n.to_dict() for n in session.notes],
                report_path=report_path,
            )
        except Exception:
            logger.exception(
                "Failed to log session %s to catalog", session.session_id
            )
            return

        session.catalog_study_id = study_id
        try:
            session.save(save_dir)
        except Exception:
            logger.exception(
                "Failed to re-save session %s after catalog logging",
                session.session_id,
            )
