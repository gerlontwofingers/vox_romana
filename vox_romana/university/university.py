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

import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING

from vox_romana.university.config import UniversityConfig
from vox_romana.university.seminar_plan import SeminarPlan
from vox_romana.university.scholar import (
    MAX_SYNTHESIS_CHARS_IN_PROMPT,
    Scholar,
    extract_json,
    strip_think,
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


# ── Seminar telemetry helpers ─────────────────────────────────────────────────

def _seminar_log_path(save_dir: Path | None) -> Path | None:
    """Return the seminar event-log path, or None when no save_dir is set."""
    if save_dir is None:
        return None
    return save_dir / "seminar_log.jsonl"


def _log_seminar_event(
    path: Path | None,
    event_type: str,
    **data,
) -> None:
    """Append one structured event to the seminar JSONL log.

    Each entry is ``{"logged_at": ISO, "event_type": ..., ...data}``.
    Uses append-mode + immediate flush so partial runs are still readable.
    No-ops when *path* is None (no save_dir configured).
    """
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"logged_at": _now_iso(), "event_type": event_type, **data}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=True, default=str))
            fh.write("\n")
            fh.flush()
    except Exception:
        logger.debug("_log_seminar_event: could not write to %s", path, exc_info=True)


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
        per_block_cap = max(800, MAX_SYNTHESIS_CHARS_IN_PROMPT // max(1, len(usable)))
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

        return strip_think(raw or "")

    # ── seminar / colloquium ────────────────────────────────────────────

    @staticmethod
    def _start_deadline_watchdog(
        deadline_monotonic: float,
        grace_secs: float = 90.0,
        log_path: "Path | None" = None,
    ) -> None:
        """Spawn a daemon thread that hard-exits the process if still running
        after ``deadline_monotonic + grace_secs``.

        ``grace_secs`` (default 90 s) gives the in-flight LLM call time to
        finish writing its response before we pull the plug.  It will not
        fire during a normal run — if ``seminar()`` returns before the
        grace window, the daemon thread is killed automatically.
        """

        def _watch() -> None:
            kill_at = deadline_monotonic + grace_secs
            remaining = kill_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            if time.monotonic() >= kill_at:
                elapsed = time.monotonic() - (kill_at - grace_secs)
                print(
                    f"\n[Seminar/WATCHDOG] deadline + {grace_secs:.0f}s grace exceeded "
                    f"(~{elapsed:.0f}s over budget) — forcing exit",
                    flush=True,
                )
                _log_seminar_event(
                    log_path, "seminar_watchdog_kill",
                    grace_secs=grace_secs,
                    elapsed_over_budget_secs=round(elapsed, 1),
                )
                os._exit(1)

        t = threading.Thread(target=_watch, daemon=True, name="seminar-watchdog")
        t.start()

    def seminar(
        self,
        topics: list[str],
        *,
        budget_hours_each: float | None = None,
        save_dir: Path | None = None,
        colloquium_enabled: bool | None = None,
        colloquium_interval: int | None = None,
        notes_per_synthesis: int | None = None,
        colloquium_max_rounds: int | None = None,
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

        log_path = _seminar_log_path(save_dir)
        _log_seminar_event(
            log_path, "seminar_started",
            topics=clean_topics,
            n_scholars=len(clean_topics),
            budget_hours_each=budget_hours_each,
            colloquium_enabled=enabled,
        )

        # Build a budget-aware plan.  CLI overrides take priority; math fills
        # in everything else so the user never has to tune the yaml manually.
        plan: SeminarPlan | None = None
        if budget_hours_each is not None:
            plan = SeminarPlan.suggest(
                budget_hours=budget_hours_each,
                n_scholars=len(clean_topics),
                config=self.config,
                colloquium_every=colloquium_interval,  # None → derived
                notes_per_synthesis=notes_per_synthesis,
                colloquium_max_rounds=colloquium_max_rounds,
            )
            interval = plan.colloquium_every
            print(f"[Seminar] plan: {plan.summary()}", flush=True)
        else:
            interval = max(
                1,
                self.config.colloquium_interval
                if colloquium_interval is None else colloquium_interval,
            )

        # Round 0 — seed each session with its initial queue, then run the
        # first bounded slice of articles.  All Scholars start in parallel;
        # colloquium requires all sessions to exist before it can run.
        logger.info(
            "Seminar: %d Scholars, %d articles/round, colloquium ON",
            len(clean_topics), interval,
        )

        # Compute a single shared deadline anchored to NOW so that the total
        # seminar wall-clock matches --hours regardless of how many scholars
        # are interleaved or whether Ollama serialises their LLM calls.
        # colloquium_extra_secs reserves additional wall-clock time for the
        # cross-pollination phase so the seed budget is not shared with it.
        _collo_extra = self.config.colloquium_extra_secs if enabled else 0.0
        seminar_deadline: float | None = (
            time.monotonic() + budget_hours_each * 3600.0 + _collo_extra
            if budget_hours_each is not None else None
        )

        # Hard-kill watchdog: if we're still running grace_secs after the
        # deadline, something is stuck in an LLM call and can't self-stop.
        # The daemon thread forces os._exit() so the user is never left hanging.
        if seminar_deadline is not None:
            self._start_deadline_watchdog(seminar_deadline, log_path=log_path)

        def _seed(args: tuple) -> ScholarSession:
            scholar, topic = args
            return scholar.study(
                topic,
                budget_hours=budget_hours_each,
                save_dir=save_dir,
                max_articles=interval,
                notes_per_synthesis=plan.notes_per_synthesis if plan is not None else None,
                _deadline_monotonic=seminar_deadline,
            )

        with ThreadPoolExecutor(max_workers=len(scholars)) as pool:
            sessions = list(pool.map(_seed, zip(scholars, clean_topics)))

        print(
            f"[Seminar] seed phase complete — "
            f"{sum(s.status == 'active' for s in sessions)}/{len(sessions)} active, "
            f"{sum(s.status == 'paused' for s in sessions)} paused",
            flush=True,
        )
        _log_seminar_event(
            log_path, "seed_phase_complete",
            n_active=sum(s.status == 'active' for s in sessions),
            n_paused=sum(s.status == 'paused' for s in sessions),
            session_ids=[s.session_id for s in sessions],
        )

        # Always attempt at least one colloquium round after the seed,
        # regardless of session status.  With tight budgets the seed phase
        # may consume the whole window and leave scholars paused; we still
        # want the cross-pollination on whatever they managed to read.
        with_synthesis = sum(1 for s in sessions if (s.synthesis or "").strip())
        if with_synthesis >= 2:
            self._run_colloquium_round(
                sessions, round_n=1, deadline_monotonic=seminar_deadline,
                log_path=log_path,
            )
            for s in sessions:
                s.save(save_dir)
        else:
            print(
                f"[Seminar] skipping post-seed colloquium — only {with_synthesis} scholar(s) have synthesis",
                flush=True,
            )

        round_n = 2
        max_collo_rounds: int | None = plan.colloquium_max_rounds if plan is not None else None
        # Subsequent rounds — colloquium (serial) → bounded extend (parallel).
        while self._any_active(sessions):
            # Hard deadline: stop before starting another expensive round.
            if seminar_deadline is not None and time.monotonic() >= seminar_deadline:
                logger.info("Seminar: wall-clock budget exhausted — stopping before round %d.", round_n)
                _log_seminar_event(log_path, "seminar_budget_exhausted", round_n=round_n)
                for s in sessions:
                    if s.status == "active":
                        s.status = "paused"
                break
            # Plan-derived cap: never exceed the number of rounds the budget can afford.
            if max_collo_rounds is not None and round_n > max_collo_rounds:
                logger.info(
                    "Seminar: colloquium_max_rounds (%d) reached — stopping.",
                    max_collo_rounds,
                )
                break
            logger.info(
                "Seminar: colloquium round %d (%d/%d Scholars active)",
                round_n,
                sum(1 for s in sessions if s.status == "active"),
                len(sessions),
            )
            self._run_colloquium_round(
                sessions, round_n=round_n, deadline_monotonic=seminar_deadline,
                log_path=log_path,
            )

            # All still-active Scholars read their next bounded slice in
            # parallel.  We wait for all to finish before the next colloquium
            # so every peer sees a stable synthesis snapshot.
            active_pairs = [
                (scholar, session)
                for scholar, session in zip(scholars, sessions)
                if session.status == "active"
            ]

            def _extend(args: tuple) -> None:
                scholar, session = args
                scholar.extend(
                    session,
                    additional_hours=None,  # keep existing budget
                    save_dir=save_dir,
                    max_articles=interval,
                    notes_per_synthesis=plan.notes_per_synthesis if plan is not None else None,
                )

            with ThreadPoolExecutor(max_workers=len(active_pairs) or 1) as pool:
                list(pool.map(_extend, active_pairs))

            if any(s.status == "interrupted" for s in sessions):
                logger.info("Seminar: Scholar interrupted — stopping.")
                break
            round_n += 1

        # Catalog-log every session that produced anything.  Done at the
        # end (not per-round) so each catalog entry reflects the final state.
        for session in sessions:
            self._catalog_log(session, save_dir=save_dir)

        _log_seminar_event(
            log_path, "seminar_complete",
            sessions=[
                {
                    "session_id": s.session_id,
                    "status": s.status,
                    "n_notes": len(s.notes),
                    "n_articles": len(s.articles_read),
                    "has_synthesis": bool((s.synthesis or "").strip()),
                    "n_colloquium_rounds": len(s.colloquium_rounds),
                }
                for s in sessions
            ],
        )
        return sessions

    @staticmethod
    def _any_active(sessions: list[ScholarSession]) -> bool:
        return any(s.status == "active" and s.articles_queued for s in sessions)

    def _run_colloquium_round(
        self,
        sessions: list[ScholarSession],
        *,
        round_n: int,
        deadline_monotonic: float | None = None,
        log_path: Path | None = None,
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
        n_with_synthesis = sum(1 for s in sessions if (s.synthesis or "").strip())
        print(
            f"[Seminar/COLLOQUIUM] round {round_n} starting — "
            f"{n_with_synthesis}/{len(sessions)} scholars have synthesis",
            flush=True,
        )
        _log_seminar_event(
            log_path, "colloquium_round_started",
            round_n=round_n,
            n_scholars_with_synthesis=n_with_synthesis,
            n_pairs=len(sessions) * (len(sessions) - 1),
        )

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

        # Run all (sender, receiver) LLM calls concurrently — each pair is
        # independent.  We collect results first, then apply mutations to
        # session state serially so there are no write races.
        pairs: list[tuple[ScholarSession, ScholarSession, str]] = []
        for receiver in sessions:
            for sender_sess, snippet in sender_views:
                if sender_sess.session_id == receiver.session_id:
                    continue
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    print(
                        f"[Seminar/COLLOQUIUM] round {round_n} aborted — deadline reached",
                        flush=True,
                    )
                    _log_seminar_event(
                        log_path, "colloquium_round_aborted",
                        round_n=round_n,
                        reason="deadline_reached",
                    )
                    break
                pairs.append((sender_sess, receiver, snippet))
            else:
                continue
            break

        if not pairs:
            print(
                f"[Seminar/COLLOQUIUM] round {round_n} done — added 0 qids across peers",
                flush=True,
            )
            return 0

        print(
            f"[Seminar/COLLOQUIUM] round {round_n}: {len(pairs)} exchanges "
            f"firing in parallel",
            flush=True,
        )

        def _one_exchange(
            args: tuple[ScholarSession, ScholarSession, str],
        ) -> tuple[str, "ColloquiumExchange | None", list[str]]:
            """Run one exchange; return (receiver_id, exchange, new_qids)."""
            sender_s, recv_s, snip = args
            print(
                f"[Seminar/COLLOQUIUM] round {round_n}: "
                f"{sender_s.session_id[:8]} \u2192 {recv_s.session_id[:8]}",
                flush=True,
            )
            exc = self._colloquium_exchange(
                sender=sender_s,
                receiver=recv_s,
                snippet=snip,
                max_leads=max_leads,
            )
            # Log the LLM exchange result before source fetching so the entry
            # survives a watchdog kill during _safe_search_to_qids.
            _log_seminar_event(
                log_path, "colloquium_exchange_result",
                round_n=round_n,
                sender_id=sender_s.session_id[:8],
                receiver_id=recv_s.session_id[:8],
                n_suggested_queries=len(exc.suggested_queries) if exc is not None else 0,
            )
            qids: list[str] = []
            if exc is not None and exc.suggested_queries:
                qids = self._safe_search_to_qids(
                    list(exc.suggested_queries),
                    receiver=recv_s,
                    max_per_peer=max_leads,
                )
            return recv_s.session_id, exc, qids

        n_workers = min(len(pairs), max(1, len(sessions) * (len(sessions) - 1)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_one_exchange, pairs))

        # Apply mutations serially — no concurrent writes to session state.
        for receiver_id, exc, new_qids in results:
            if exc is None:
                continue
            round_by_session[receiver_id].exchanges.append(exc)
            if new_qids:
                recv_sess = next(s for s in sessions if s.session_id == receiver_id)
                recv_sess.articles_queued.extend(new_qids)
                total_added += len(new_qids)

        # Persist the round on each receiver that actually got exchanges.
        for receiver in sessions:
            r = round_by_session[receiver.session_id]
            if r.exchanges:
                receiver.colloquium_rounds.append(r)

        print(
            f"[Seminar/COLLOQUIUM] round {round_n} done — added {total_added} qids across peers",
            flush=True,
        )
        _log_seminar_event(
            log_path, "colloquium_round_complete",
            round_n=round_n,
            total_qids_added=total_added,
        )
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

        data = extract_json(strip_think(raw or ""))
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
