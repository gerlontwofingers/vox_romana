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
    _extract_json,
    _strip_think,
)
from vox_romana.university.session import (
    ColloquiumExchange,
    ColloquiumRound,
    ScholarSession,
    _now_iso,
)

if TYPE_CHECKING:
    from vox_romana.core.catalog import Catalog
    from vox_romana.core.client import DeepSeekClient
    from vox_romana.university.sources import KnowledgeSource

logger = logging.getLogger(__name__)


class University:
    """Multi-Scholar research orchestrator.

    Attributes:
        client:           LLM reading client passed to every spawned Scholar
                          (fast model for per-article note extraction).
        synthesis_client: LLM synthesis client used for Scholar synthesis passes
                          and cross-synthesis (quality model).
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
        synthesis_client: "DeepSeekClient | None" = None,
        catalog: "Catalog | None" = None,
        run_id: str | None = None,
        content_type: str = "",
    ):
        if not sources:
            raise ValueError("University needs at least one KnowledgeSource.")
        self.client = client
        self.synthesis_client = synthesis_client or client
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
        return Scholar(self.client, self.sources, self.config,
                       synthesis_client=self.synthesis_client)

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
            raw = self.synthesis_client.chat(
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

    # ── seminar / colloquium ────────────────────────────────────────────

    def seminar(
        self,
        topics: list[str],
        *,
        budget_hours_each: float | None = None,
        save_dir: Path | None = None,
        colloquium_enabled: bool | None = None,
        colloquium_interval: int | None = None,
    ) -> list[ScholarSession]:
        """Run several Scholars **in interleaved rounds** with peer colloquia.

        This is a richer alternative to :meth:`study_many`: instead of
        running each Scholar to completion and only merging at the end,
        the Scholars work in fixed-size rounds (``colloquium_interval``
        articles each).  Between rounds, every Scholar shares a snippet
        of its current synthesis and its peers' LLMs propose follow-up
        search queries that get injected into the peers' queues.  The
        result is cross-pollination of leads — sub-topics inform each
        other in real time, not just at the end.

        Args:
            topics:              One topic per Scholar.  Empty topics are skipped.
            budget_hours_each:   Per-Scholar time budget. ``None`` lets
                                 each Scholar self-estimate.
            save_dir:            Root directory for session persistence.
            colloquium_enabled:  Override
                                 :attr:`UniversityConfig.colloquium_enabled`.
                                 When ``False``, this method is equivalent
                                 to :meth:`study_many` (no exchanges).
            colloquium_interval: Override
                                 :attr:`UniversityConfig.colloquium_interval`.

        Returns:
            One :class:`ScholarSession` per non-empty topic, in order.
            Each session's ``colloquium_rounds`` records the leads it
            received from peers, for full auditability.
        """
        # Resolve effective colloquium settings.
        enabled = (
            self.config.colloquium_enabled
            if colloquium_enabled is None else colloquium_enabled
        )
        interval = max(
            1,
            self.config.colloquium_interval
            if colloquium_interval is None else colloquium_interval,
        )

        # Validate topics; preserve original order for stable session IDs.
        clean_topics = [(t or "").strip() for t in topics]
        clean_topics = [t for t in clean_topics if t]
        if not clean_topics:
            logger.warning("seminar() called with no usable topics.")
            return []

        # Without colloquium and with a single topic, fall back to study()
        # so behaviour matches user expectation (no spurious round bookkeeping).
        if not enabled or len(clean_topics) < 2:
            if not enabled:
                logger.info(
                    "seminar(): colloquium disabled — falling back to study_many()."
                )
            return self.study_many(
                clean_topics,
                budget_hours_each=budget_hours_each,
                save_dir=save_dir,
            )

        # Spawn one Scholar + session per topic.  Each Scholar holds its own
        # save_dir state; pairs are kept aligned by index.
        scholars: list[Scholar] = [self.spawn_scholar() for _ in clean_topics]
        sessions: list[ScholarSession] = []

        # Round 0 — seed each session with its initial queue, then run the
        # first bounded slice of articles.
        logger.info(
            "Seminar: %d Scholars, %d articles/round, colloquium ON",
            len(clean_topics), interval,
        )
        for scholar, topic in zip(scholars, clean_topics):
            session = scholar.study(
                topic,
                budget_hours=budget_hours_each,
                save_dir=save_dir,
                max_articles=interval,
            )
            sessions.append(session)

        round_n = 1
        # Subsequent rounds — colloquium → bounded extend → repeat.
        while self._any_active(sessions):
            logger.info(
                "Seminar: colloquium round %d (%d/%d Scholars active)",
                round_n,
                sum(1 for s in sessions if s.status == "active"),
                len(sessions),
            )
            self._run_colloquium_round(sessions, round_n=round_n)

            # Each still-active Scholar reads another bounded slice.  An
            # interrupted Scholar stops the seminar (Ctrl-C semantics from
            # study_many are preserved).
            interrupted = False
            for scholar, session in zip(scholars, sessions):
                if session.status != "active":
                    continue
                scholar.extend(
                    session,
                    additional_hours=None,  # keep existing budget
                    save_dir=save_dir,
                    max_articles=interval,
                )
                if session.status == "interrupted":
                    interrupted = True
                    break
            if interrupted:
                logger.info("Seminar: Scholar interrupted — stopping.")
                break
            round_n += 1

        # Catalog-log every session that produced anything.  Done at the
        # end (not per-round) so each catalog entry reflects the final state.
        for session in sessions:
            self._catalog_log(session, save_dir=save_dir)

        return sessions

    @staticmethod
    def _any_active(sessions: list[ScholarSession]) -> bool:
        return any(s.status == "active" and s.articles_queued for s in sessions)

    def _run_colloquium_round(
        self,
        sessions: list[ScholarSession],
        *,
        round_n: int,
    ) -> int:
        """Have each Scholar suggest leads to its peers based on findings.

        For every (sender, receiver) pair where ``sender != receiver``,
        the receiver's synthesis-tier LLM is shown a snippet of the
        sender's running synthesis and asked which 1–N search queries it
        would add to *its own* queue to enrich its topic.  The queries
        are run through every :class:`KnowledgeSource` and unique results
        are appended to the receiver's queue.

        Records a :class:`ColloquiumRound` on every receiver session so
        the cross-pollination is fully auditable.

        Args:
            sessions: All Scholar sessions in the seminar (order matters
                      only for deterministic peer ordering in tests).
            round_n:  1-indexed round number used in the log entry.

        Returns:
            Total number of qids added across all receivers (useful for
            tests/diagnostics).
        """
        max_leads = max(1, self.config.colloquium_max_leads_per_peer)
        snippet_chars = max(200, self.config.colloquium_snippet_chars)
        total_added = 0

        # Build sender snippets up-front so each receiver sees a stable view.
        sender_views: list[tuple[ScholarSession, str]] = []
        for sess in sessions:
            snippet = (sess.synthesis or "").strip()
            if not snippet:
                # No synthesis yet — nothing this peer can usefully share.
                continue
            if len(snippet) > snippet_chars:
                snippet = snippet[:snippet_chars] + "\n…[truncated]"
            sender_views.append((sess, snippet))

        if len(sender_views) < 2:
            # Need at least two Scholars with synthesis to have an exchange.
            logger.debug(
                "Colloquium round %d skipped — fewer than 2 Scholars have synthesis.",
                round_n,
            )
            return 0

        # Build a fresh round on every session that may receive exchanges.
        round_by_session: dict[str, ColloquiumRound] = {
            sess.session_id: ColloquiumRound(round_n=round_n)
            for sess in sessions
        }

        for receiver in sessions:
            for sender_sess, snippet in sender_views:
                if sender_sess.session_id == receiver.session_id:
                    continue
                exchange = self._colloquium_exchange(
                    sender=sender_sess,
                    receiver=receiver,
                    snippet=snippet,
                    max_leads=max_leads,
                )
                if exchange is None:
                    continue
                round_by_session[receiver.session_id].exchanges.append(exchange)
                # Inject the suggested queries as queue items via every source.
                if exchange.suggested_queries:
                    new_qids = self._safe_search_to_qids(
                        list(exchange.suggested_queries),
                        receiver=receiver,
                        max_per_peer=max_leads,
                    )
                    receiver.articles_queued.extend(new_qids)
                    total_added += len(new_qids)

        # Persist the round on each receiver that actually got exchanges.
        for receiver in sessions:
            r = round_by_session[receiver.session_id]
            if r.exchanges:
                receiver.colloquium_rounds.append(r)

        return total_added

    def _colloquium_exchange(
        self,
        *,
        sender: ScholarSession,
        receiver: ScholarSession,
        snippet: str,
        max_leads: int,
    ) -> ColloquiumExchange | None:
        """Run one peer-to-peer LLM exchange. Returns ``None`` on failure."""
        system = (
            "You are a Scholar in a research seminar.  A peer Scholar is "
            "studying a related topic and has just shared an excerpt of "
            "their running synthesis.  Your job is to identify search "
            "queries that, given the peer's discoveries, would enrich "
            "YOUR OWN research on YOUR topic — not theirs.  Be concrete: "
            "queries should be the kind a researcher would type into an "
            "encyclopaedia or academic database.  If nothing the peer "
            "shared is genuinely useful for your topic, return an empty "
            "list — do not invent connections."
        )
        user = (
            f"YOUR TOPIC: {receiver.topic}\n"
            f"PEER TOPIC: {sender.topic}\n\n"
            f"PEER'S SYNTHESIS EXCERPT:\n{snippet}\n\n"
            f"Output JSON ONLY (no prose, no code fence) with shape:\n"
            f'  {{"comment": "<one sentence on the connection or '
            f'\\"no relevant connection\\">", '
            f'"queries": ["query 1", "query 2", ...]}}\n'
            f"At most {max_leads} queries."
        )
        try:
            raw = self.synthesis_client.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.config.temperature,
            )
        except Exception:
            logger.exception(
                "Colloquium: LLM call failed for %s -> %s",
                sender.topic, receiver.topic,
            )
            return None

        data = _extract_json(_strip_think(raw or ""))
        comment = ""
        queries: list[str] = []
        if isinstance(data, dict):
            comment = str(data.get("comment", "")).strip()
            raw_queries = data.get("queries") or []
            if isinstance(raw_queries, list):
                for q in raw_queries[:max_leads]:
                    qs = str(q).strip()
                    if qs:
                        queries.append(qs)

        return ColloquiumExchange(
            from_topic=sender.topic,
            to_topic=receiver.topic,
            snippet=snippet,
            comment=comment,
            suggested_queries=tuple(queries),
            timestamp=_now_iso(),
        )

    def _safe_search_to_qids(
        self,
        queries: list[str],
        *,
        receiver: ScholarSession,
        max_per_peer: int,
    ) -> list[str]:
        """Convert peer-suggested queries into qids, avoiding duplicates.

        Skips qids that are already in ``receiver.articles_read`` or
        ``receiver.articles_queued`` so a colloquium round never makes
        the same Scholar read the same article twice.
        """
        from vox_romana.university.scholar import _make_qid

        existing: set[str] = set(receiver.articles_read) | set(receiver.articles_queued)
        added: list[str] = []
        per_query_cap = max(1, max_per_peer)
        for q in queries:
            for src in self.sources:
                try:
                    hits = src.search(q, n=per_query_cap)
                except Exception:
                    logger.exception(
                        "Colloquium source %s.search raised on %r", src.name, q
                    )
                    continue
                for art in hits:
                    qid = _make_qid(art.source_name or src.name, art.title)
                    if qid in existing:
                        continue
                    existing.add(qid)
                    added.append(qid)
                    if len(added) >= self.config.max_queue_size:
                        return added
        return added

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
                method=f"scholar/{self.config.reading_model}+{self.config.synthesis_model}",
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

        # Auto-ingest notes into the local knowledge store so future
        # --research runs can consult this session's findings immediately.
        try:
            notes_path = Path(session.session_dir(save_dir)) / "notes.jsonl"
            if notes_path.exists():
                from vox_romana.memory.ingest import DocumentIngester
                DocumentIngester().ingest(notes_path, tag="university")
                logger.debug(
                    "Auto-ingested %d notes from session %s",
                    len(session.notes),
                    session.session_id,
                )
        except Exception:
            logger.warning(
                "Auto-ingest of university notes failed (non-fatal)",
                exc_info=True,
            )
