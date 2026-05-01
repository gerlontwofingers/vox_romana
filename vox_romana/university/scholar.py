"""Scholar — the deep-research agent at the heart of the University.

A Scholar runs a time-budgeted, interruptable read/synthesise loop over one
or more :class:`KnowledgeSource` instances, building structured
:class:`ScholarNote` records and a running synthesis.  All LLM calls go
through a :class:`vox_romana.core.client.DeepSeekClient` (typically an
:class:`vox_romana.local.ollama.OllamaClient` for fully-local work).

Lifecycle
---------
::

    scholar = Scholar(client, sources=[WikipediaSource(), LocalSource(recall)])
    session = scholar.study("Roman engineering", budget_hours=2.0)
    # …or self-determined duration:
    session = scholar.study("Roman engineering")
    # …or resume:
    scholar.extend(session, additional_hours=1.0)
    # …or interactive Q&A:
    answer = scholar.ask("What materials did Roman roads use?", session=session)

The session is checkpointed to disk after every article so a SIGINT (Ctrl-C)
or process kill never costs more than the most recent article's work.

Identifier convention
---------------------
The article queue holds strings of form ``"{source_name}:{identifier}"``,
e.g. ``"wikipedia:Roman roads"`` or ``"local:study_abc123"``.  This keeps
multi-source queues unambiguous and makes ``session.json`` human-readable.
"""

from __future__ import annotations

import json
import logging
import re
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vox_romana.core.client import DeepSeekClient
from vox_romana.university.config import UniversityConfig
from vox_romana.university.session import ScholarNote, ScholarSession, _now_iso
from vox_romana.university.sources import Article, KnowledgeSource

logger = logging.getLogger(__name__)


# Per-prompt char caps — keep small models happy and prompts well under 8k tokens.
_MAX_ARTICLE_CHARS = 6000
_MAX_SYNTHESIS_CHARS_IN_PROMPT = 2500
_MAX_LINKS_IN_PROMPT = 50
_MAX_NOTES_FOR_SYNTHESIS = 40       # most-recent-N notes packed into the synthesis prompt
_MAX_NOTES_FOR_FINALIZE = 200       # final pass gets a wider window


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE | re.IGNORECASE)


def _strip_think(raw: str) -> str:
    """Strip DeepSeek-R1 ``<think>…</think>`` blocks from a response."""
    return _THINK_RE.sub("", raw or "").strip()


def _extract_json(raw: str) -> Any | None:
    """Best-effort JSON extraction from an LLM response.

    Strips ``<think>`` blocks (DeepSeek-R1), code fences, and tries to
    locate the first ``{…}`` or ``[…]`` block if surrounding prose remains.
    Returns ``None`` if nothing parses — callers must handle that.
    """
    if not raw:
        return None
    text = _strip_think(raw)
    # Strip ```json … ``` fences if present.
    text = _FENCE_RE.sub("", text).strip()
    # Direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Locate the first balanced JSON block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    logger.warning("Could not parse JSON from response: %.200s", text)
    return None


def _format_hms(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` (no negatives)."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _split_qid(qid: str) -> tuple[str | None, str]:
    """Split a queue identifier ``source:identifier`` → ``(source, ident)``.

    If no colon is present, returns ``(None, qid)`` so the caller can try
    every source as a fallback.
    """
    if ":" not in qid:
        return None, qid
    src, ident = qid.split(":", 1)
    return src or None, ident


def _make_qid(source_name: str, identifier: str) -> str:
    """Build a queue identifier."""
    return f"{source_name}:{identifier}"


# ─────────────────────────────────────────────────────────────────────────────
# Scholar
# ─────────────────────────────────────────────────────────────────────────────

class Scholar:
    """Time-budgeted, interruptable deep-research agent.

    The Scholar is the only University component that talks to the LLM.
    All knowledge access goes through :class:`KnowledgeSource` instances
    passed at construction time.

    Attributes:
        reading_client:   LLM client for per-article note extraction (fast model).
        synthesis_client: LLM client for synthesis, prioritisation, finalize (quality model).
        sources: Ordered list of KnowledgeSource. First match wins on fetch.
        config:  UniversityConfig (defaults from ``vox_romana.yaml``).

    Thread-safety:
        One Scholar instance is **not** safe to share across concurrent
        ``study()`` calls. Spawn one Scholar per study.
    """

    def __init__(
        self,
        client: DeepSeekClient,
        sources: list[KnowledgeSource],
        config: UniversityConfig | None = None,
        synthesis_client: DeepSeekClient | None = None,
    ):
        if not sources:
            raise ValueError("Scholar needs at least one KnowledgeSource.")
        self.reading_client = client
        self.synthesis_client = synthesis_client or client
        self.sources = list(sources)
        self.config = config or UniversityConfig.from_project_yaml()

        # Per-study runtime state (reset by study()/extend()).
        self._stop = threading.Event()
        self._started_monotonic: float = 0.0
        self._save_dir: Path | None = None
        self._sources_by_name = {s.name: s for s in self.sources}

    # ── public API ──────────────────────────────────────────────────────

    def study(
        self,
        topic: str,
        *,
        budget_hours: float | None = None,
        save_dir: Path | None = None,
        seed_article_ids: list[str] | None = None,
        max_articles: int | None = None,
        notes_per_synthesis: int | None = None,
        _deadline_monotonic: float | None = None,
    ) -> ScholarSession:
        """Run a fresh deep-research study on ``topic``.

        Args:
            topic:            Free-form research directive.
            budget_hours:     Hard time budget. ``None`` = let the Scholar
                              estimate how long the topic deserves.
            save_dir:         Root directory for session persistence.
                              Defaults to :data:`UNIVERSITY_ROOT`.
            seed_article_ids: Optional pre-seeded queue (qids of form
                              ``"source:identifier"``). When omitted, the
                              Scholar generates initial queries via
                              :meth:`_plan_initial_queries`.
            max_articles:     Optional cap on articles processed in this
                              call. ``None`` = no cap (default). Used by
                              :meth:`University.seminar` to interleave
                              several Scholars in fixed-size rounds.

        Returns:
            The fully-populated :class:`ScholarSession`. Status will be
            ``"complete"``, ``"interrupted"``, or ``"paused"`` (budget
            exhausted) depending on how the loop terminated.
        """
        session = ScholarSession(topic=topic, budget_hours=budget_hours)
        self._save_dir = save_dir
        problems = session.validate()
        if problems:
            raise ValueError(f"Invalid session: {'; '.join(problems)}")

        # Self-determined duration if no explicit budget.
        if budget_hours is None:
            session.estimated_hours = self._determine_duration(topic)
            logger.info(
                "Scholar self-estimated %.1fh for topic %r", session.estimated_hours, topic
            )
        else:
            session.estimated_hours = budget_hours

        # Set the absolute deadline ONCE here. It is never reset by extend()
        # so the total budget is honoured across seminar rounds.
        # A caller (e.g. University.seminar) may supply a shared deadline so
        # that all scholars stop at the same wall-clock moment.
        if _deadline_monotonic is not None:
            session.deadline_monotonic = _deadline_monotonic
        elif budget_hours is not None:
            session.deadline_monotonic = time.monotonic() + budget_hours * 3600.0

        # Seed the queue.
        if seed_article_ids:
            session.articles_queued.extend(seed_article_ids)
        else:
            queries = self._plan_initial_queries(topic)
            session.articles_queued.extend(self._search_to_qids(queries))

        session.save(self._save_dir)
        return self._run_loop(session, max_articles=max_articles, notes_per_synthesis=notes_per_synthesis)

    def extend(
        self,
        session: ScholarSession,
        *,
        additional_hours: float | None = None,
        save_dir: Path | None = None,
        max_articles: int | None = None,
        notes_per_synthesis: int | None = None,
    ) -> ScholarSession:
        """Resume an existing session for more research time.

        The session's queue is preserved as-is. If the queue is empty,
        the Scholar replans initial queries from the topic.

        Args:
            session:          Previously-saved ScholarSession.
            additional_hours: Extra time budget to add. ``None`` =
                              self-estimate again from topic.
            save_dir:         Root directory for persistence.
            max_articles:     Optional cap on articles processed in this
                              call.  See :meth:`study` for the use case.
        """
        self._save_dir = save_dir
        session.status = "active"
        if additional_hours is not None:
            session.budget_hours = (session.budget_hours or 0.0) + additional_hours
            session.estimated_hours = additional_hours
        elif session.estimated_hours is None:
            # Only spend an LLM call on duration when we've never estimated.
            session.estimated_hours = self._determine_duration(session.topic)

        if not session.articles_queued:
            queries = self._plan_initial_queries(session.topic)
            session.articles_queued.extend(self._search_to_qids(queries))

        session.save(self._save_dir)
        return self._run_loop(session, max_articles=max_articles, notes_per_synthesis=notes_per_synthesis)

    def ask(self, question: str, *, session: ScholarSession) -> str:
        """One-shot Q&A against a session's synthesis + notes.

        Does not modify the session.  Uses the most-relevant notes
        (top-N by relevance score) to ground the answer.
        """
        # Pick the most relevant notes (cap to keep prompt small).
        ranked = sorted(session.notes, key=lambda n: n.relevance_score, reverse=True)
        excerpts = self._format_notes_for_prompt(ranked[:20])

        synthesis = session.synthesis or "[no synthesis written yet]"
        if len(synthesis) > _MAX_SYNTHESIS_CHARS_IN_PROMPT:
            synthesis = synthesis[:_MAX_SYNTHESIS_CHARS_IN_PROMPT] + "\n…[truncated]"

        system = (
            "You are a Scholar answering a question grounded ONLY in the "
            "research notes you collected. If the notes do not contain the "
            "answer, say so plainly — do not invent facts."
        )
        user = (
            f"RESEARCH TOPIC: {session.topic}\n\n"
            f"YOUR SYNTHESIS:\n{synthesis}\n\n"
            f"YOUR MOST RELEVANT NOTES:\n{excerpts}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer in 1–4 paragraphs.  Cite article titles inline like "
            "[Roman roads] when drawing on a specific note."
        )
        raw = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=self.config.temperature,
        )
        return _strip_think(raw)

    def chat(self, session: ScholarSession) -> None:
        """Interactive REPL against a session. Exits on EOF/Ctrl-D."""
        print(f"Chatting about: {session.topic}")
        print(f"  notes: {len(session.notes)}  |  synthesis: "
              f"{len(session.synthesis)} chars")
        print("Enter blank line or Ctrl-D to exit.\n")
        while True:
            try:
                q = input("you> ").strip()
            except EOFError:
                print()
                return
            if not q:
                return
            try:
                answer = self.ask(q, session=session)
            except Exception as exc:
                logger.exception("Scholar.ask failed")
                print(f"[error: {exc}]")
                continue
            print(f"\nscholar> {answer}\n")

    def interrupt(self) -> None:
        """Request a graceful stop at the next safe checkpoint."""
        self._stop.set()

    # ── main loop ───────────────────────────────────────────────────────

    def _run_loop(
        self,
        session: ScholarSession,
        *,
        max_articles: int | None = None,
        notes_per_synthesis: int | None = None,
    ) -> ScholarSession:
        """Core read/synthesise loop with SIGINT handling.

        Args:
            session:      The session to drive.
            max_articles: Optional cap on articles processed in this call.
                          ``None`` = no cap.
        """
        self._stop.clear()
        self._started_monotonic = time.monotonic()
        prev_handler = signal.getsignal(signal.SIGINT)

        def _on_sigint(signum, frame):  # pragma: no cover — signal path
            print("\n[Scholar] SIGINT received — finishing current article and saving…")
            self._stop.set()

        # Only install if we're in the main thread (otherwise signal raises).
        signal_installed = False
        try:
            try:
                signal.signal(signal.SIGINT, _on_sigint)
                signal_installed = True
            except (ValueError, OSError):
                # Not the main thread — caller can still call .interrupt().
                pass

            self._iterate(session, max_articles=max_articles, notes_per_synthesis=notes_per_synthesis)
        finally:
            if signal_installed:
                try:
                    signal.signal(signal.SIGINT, prev_handler)
                except (ValueError, OSError):
                    pass

        return session

    def _iterate(
        self,
        session: ScholarSession,
        *,
        max_articles: int | None = None,
        notes_per_synthesis: int | None = None,
    ) -> None:
        """The actual loop body — separated for testability."""
        articles_since_synthesis = 0
        articles_since_priority = 0
        articles_this_call = 0
        seen_qids: set[str] = {_make_qid(s, t) for s in self._sources_by_name for t in session.articles_read}
        seen_qids.update(session.articles_read)

        while session.articles_queued and not self._stop.is_set():
            if max_articles is not None and articles_this_call >= max_articles:
                # Bounded slice complete — leave the session ``active`` so the
                # caller (typically University.seminar) can resume next round.
                break
            if self._budget_exhausted(session):
                logger.info("Scholar budget exhausted (%.1fh).", self._elapsed_hours())
                session.status = "paused"
                break

            qid = session.articles_queued.pop(0)
            if qid in seen_qids:
                continue
            seen_qids.add(qid)

            self._print_progress(session)

            article = self._fetch(qid)
            if article is None:
                continue

            if article.source_name and article.source_name not in session.sources_used:
                session.sources_used.append(article.source_name)

            note = self._read_and_note(article, session)
            session.articles_read.append(qid)

            if note is None:
                # LLM/parse failure — skip but checkpoint occasionally.
                continue

            if note.relevance_score >= self.config.relevance_threshold:
                session.notes.append(note)
                # Expand the queue with the LLM's chosen links.
                for link in note.links_found:
                    new_qid = _make_qid(article.source_name, link)
                    if new_qid in seen_qids or new_qid in session.articles_queued:
                        continue
                    if len(session.articles_queued) >= self.config.max_queue_size:
                        break
                    session.articles_queued.append(new_qid)

            articles_since_synthesis += 1
            articles_since_priority += 1
            articles_this_call += 1

            _nps = notes_per_synthesis if notes_per_synthesis is not None else self.config.notes_per_synthesis
            if articles_since_synthesis >= _nps:
                session.synthesis = self._update_synthesis(session)
                articles_since_synthesis = 0
                session.save(self._save_dir)
                session.save_notes_jsonl(self._save_dir)

            if articles_since_priority >= self.config.priority_interval:
                session.articles_queued = self._prioritize_queue(session)
                articles_since_priority = 0

            # Always checkpoint per article (cheap; protects against crash).
            session.save(self._save_dir)

        # Loop exit.
        cap_hit = (
            max_articles is not None and articles_this_call >= max_articles
        )
        if self._stop.is_set():
            session.status = "interrupted"
        elif cap_hit and session.articles_queued:
            # Bounded slice ended early on purpose; leave session active so
            # it can be resumed by the next round of a seminar.
            session.status = "active"
        elif session.status == "active":
            session.status = "complete"

        # Mid-slice synthesis flush — when a bounded slice ends with new
        # notes (whether from cap_hit or from the budget tripping), make
        # sure the synthesis reflects them.  Without this, a colloquium
        # round that follows has no material to share between peers, since
        # ``articles_since_synthesis`` is local and resets every call.
        if articles_since_synthesis > 0 and session.notes:
            print(
                f"[Scholar/SYNTH] {session.session_id[:8]} flushing synthesis "
                f"({len(session.notes)} notes)…",
                flush=True,
            )
            session.synthesis = self._update_synthesis(session)

        # Final synthesis pass — but only if we actually have notes AND the
        # session is truly finished. A bounded slice that paused mid-seminar
        # should not waste a synthesis pass between rounds.
        if session.notes and session.status not in ("active", "paused"):
            print(
                f"[Scholar/SYNTH] {session.session_id[:8]} finalizing synthesis…",
                flush=True,
            )
            session.synthesis = self._finalize(session)

        session.save(self._save_dir)
        session.save_notes_jsonl(self._save_dir)
        self._print_progress(session, final=True)

    # ── timing helpers ──────────────────────────────────────────────────

    def _elapsed_hours(self) -> float:
        return (time.monotonic() - self._started_monotonic) / 3600.0

    def _budget_exhausted(self, session: ScholarSession) -> bool:
        # Prefer the absolute deadline if set — it survives across seminar
        # extend() rounds and gives a true wall-clock budget.
        if session.deadline_monotonic is not None:
            return time.monotonic() >= session.deadline_monotonic
        if session.budget_hours is None:
            return False
        return self._elapsed_hours() >= session.budget_hours

    def _print_progress(self, session: ScholarSession, *, final: bool = False) -> None:
        elapsed = time.monotonic() - self._started_monotonic
        budget_secs = (session.budget_hours or session.estimated_hours or 0.0) * 3600.0
        remaining = max(0.0, budget_secs - elapsed) if budget_secs else 0.0
        tag = "DONE" if final else "WORK"
        msg = (
            f"[Scholar/{tag}] {_format_hms(elapsed)} elapsed | "
            f"~{_format_hms(remaining)} remaining | "
            f"{len(session.articles_read)} read | "
            f"{len(session.articles_queued)} queued | "
            f"{len(session.notes)} notes"
        )
        print(msg)

    # ── source dispatch ─────────────────────────────────────────────────

    def _fetch(self, qid: str) -> Article | None:
        """Resolve a queue identifier to an Article via the right source."""
        src_name, ident = _split_qid(qid)

        # Preferred source first, then fall back to the rest.
        ordered: list[KnowledgeSource] = []
        if src_name and src_name in self._sources_by_name:
            ordered.append(self._sources_by_name[src_name])
        for src in self.sources:
            if src not in ordered:
                ordered.append(src)

        for src in ordered:
            try:
                article = src.fetch(ident)
            except Exception:
                logger.exception("Source %s.fetch raised on %r", src.name, ident)
                article = None
            if article is not None and article.text:
                return article
        return None

    def _search_to_qids(self, queries: list[str]) -> list[str]:
        """Run each query through every source and collect qids."""
        qids: list[str] = []
        seen: set[str] = set()
        for q in queries:
            for src in self.sources:
                try:
                    hits = src.search(q, n=5)
                except Exception:
                    logger.exception("Source %s.search raised on %r", src.name, q)
                    continue
                for art in hits:
                    qid = _make_qid(art.source_name or src.name, art.title)
                    if qid in seen:
                        continue
                    seen.add(qid)
                    qids.append(qid)
                    if len(qids) >= self.config.max_queue_size:
                        return qids
        return qids

    # ── LLM-driven helpers ──────────────────────────────────────────────

    def _chat(self, messages: list[dict[str, str]], *, synthesis: bool = False, **kwargs) -> str:
        """Wrap client.chat with config-default temperature and a guard.

        Uses :attr:`synthesis_client` when *synthesis* is ``True`` (synthesis
        passes, queue prioritisation, finalize, duration estimation) and
        :attr:`reading_client` otherwise (per-article note extraction).
        """
        if "temperature" not in kwargs:
            kwargs["temperature"] = self.config.temperature
        client = self.synthesis_client if synthesis else self.reading_client
        try:
            return client.chat(messages=messages, **kwargs)
        except Exception:
            logger.exception("LLM call failed")
            return ""

    def _determine_duration(self, topic: str) -> float:
        """Ask the LLM how many hours the topic deserves.

        Returns a float in hours, clamped to [0.25, 24.0].  Falls back to
        1.0h if the response is unparseable.
        """
        system = (
            "You are a Scholar planning a deep-research study. Given a "
            "topic, estimate how many hours of focused reading + note-taking "
            "the topic deserves to produce a well-grounded synthesis. "
            "Be honest: a narrow factual question deserves 0.25–1h; a broad "
            "historical or technical topic deserves 2–8h; a multi-disciplinary "
            "survey may deserve 12–24h."
        )
        user = (
            f"TOPIC: {topic}\n\n"
            "Reply with JSON only:\n"
            '{ "estimated_hours": <float>, "rationale": "<one sentence>" }'
        )
        raw = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.3,
            synthesis=True,
        )
        data = _extract_json(raw)
        if isinstance(data, dict) and "estimated_hours" in data:
            try:
                hours = float(data["estimated_hours"])
                return max(0.25, min(24.0, hours))
            except (TypeError, ValueError):
                pass
        logger.warning("Could not parse estimated_hours; defaulting to 1.0h.")
        return 1.0

    def _plan_initial_queries(self, topic: str) -> list[str]:
        """Generate seed search queries for the initial queue."""
        system = (
            "You are a Scholar opening a new research session. Generate "
            "5–8 distinct search queries that, taken together, give broad "
            "first-pass coverage of the topic. Prefer concrete proper nouns "
            "and named concepts over abstract phrases — search engines and "
            "encyclopaedias respond better to those."
        )
        user = (
            f"TOPIC: {topic}\n\n"
            "Reply with JSON only:\n"
            '{ "queries": ["query 1", "query 2", ...] }'
        )
        raw = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.5,
            synthesis=True,
        )
        data = _extract_json(raw)
        if isinstance(data, dict):
            queries = data.get("queries") or []
            if isinstance(queries, list):
                cleaned = [str(q).strip() for q in queries if str(q).strip()]
                if cleaned:
                    return cleaned[:10]
        # Fallback: use the topic itself as a single query.
        logger.warning("Could not parse initial queries; falling back to topic as query.")
        return [topic]

    def _read_and_note(
        self, article: Article, session: ScholarSession
    ) -> ScholarNote | None:
        """Extract a structured note from one article."""
        synth = session.synthesis or "[nothing yet — this is your first article]"
        if len(synth) > _MAX_SYNTHESIS_CHARS_IN_PROMPT:
            synth = synth[:_MAX_SYNTHESIS_CHARS_IN_PROMPT] + "\n…[truncated]"

        text = article.text[:_MAX_ARTICLE_CHARS]
        if len(article.text) > _MAX_ARTICLE_CHARS:
            text += "\n…[article truncated for prompt]"

        link_options = list(article.links[:_MAX_LINKS_IN_PROMPT])
        link_block = (
            "\n".join(f"- {ln}" for ln in link_options)
            if link_options
            else "[no outbound links available]"
        )

        max_links = self.config.max_links_per_article

        system = (
            "You are a Scholar reading an article in service of a research "
            "topic. Your job is to extract ONLY what advances understanding "
            "of the topic, and to identify a small number of links genuinely "
            "worth following next. Be ruthless: do not include encyclopaedic "
            "background unless it directly informs the topic."
        )
        user = (
            f"RESEARCH TOPIC: {session.topic}\n\n"
            f"WHAT YOU'VE LEARNED SO FAR:\n{synth}\n\n"
            f"ARTICLE TITLE: {article.title}\n"
            f"ARTICLE URL: {article.url}\n\n"
            f"ARTICLE TEXT:\n{text}\n\n"
            f"OUTBOUND LINKS YOU COULD EXPLORE NEXT:\n{link_block}\n\n"
            "TASK:\n"
            f"1. Extract 1–10 key facts that ADVANCE understanding of the topic.\n"
            f"2. Write a 2–4 sentence summary of what this article contributed.\n"
            f"3. Pick up to {max_links} links from the list above genuinely worth "
            "following — or fewer, or none, if the article is a dead end. "
            "Use the EXACT title strings shown.\n"
            "4. Score relevance 0–10 (10=central, 5=useful context, 0=off-topic).\n\n"
            "Reply with JSON only:\n"
            "{\n"
            '  "key_facts": ["...", "..."],\n'
            '  "summary": "...",\n'
            '  "links_to_explore": ["...", "..."],\n'
            '  "relevance_score": <float 0-10>\n'
            "}"
        )
        raw = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=self.config.temperature,
            synthesis=False,
        )
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return None

        key_facts = data.get("key_facts") or []
        if not isinstance(key_facts, list):
            key_facts = []
        summary = str(data.get("summary") or "").strip()
        links = data.get("links_to_explore") or []
        if not isinstance(links, list):
            links = []
        # Trim to the configured cap and keep only links the article actually had.
        valid_links = set(article.links)
        links = [str(l).strip() for l in links if str(l).strip() in valid_links]
        links = links[: self.config.max_links_per_article]

        try:
            score = float(data.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(10.0, score))

        return ScholarNote(
            article_title=article.title,
            source_url=article.url,
            source_name=article.source_name,
            key_facts=tuple(str(f).strip() for f in key_facts if str(f).strip()),
            summary=summary,
            links_found=tuple(links),
            relevance_score=score,
            timestamp=_now_iso(),
        )

    def _update_synthesis(self, session: ScholarSession) -> str:
        """Refresh the running synthesis from the most recent notes."""
        recent = session.notes[-_MAX_NOTES_FOR_SYNTHESIS:]
        notes_block = self._format_notes_for_prompt(recent)
        prior = session.synthesis or "[no prior synthesis]"
        if len(prior) > _MAX_SYNTHESIS_CHARS_IN_PROMPT:
            prior = prior[:_MAX_SYNTHESIS_CHARS_IN_PROMPT] + "\n…[truncated]"

        system = (
            "You are a Scholar mid-study, integrating new notes into a "
            "running synthesis. Keep the synthesis concise (3–6 paragraphs), "
            "well-organised by sub-theme, and grounded — every claim should "
            "trace back to a specific note (cite article titles inline like "
            "[Roman roads])."
        )
        user = (
            f"TOPIC: {session.topic}\n\n"
            f"PRIOR SYNTHESIS:\n{prior}\n\n"
            f"RECENT NOTES:\n{notes_block}\n\n"
            "Produce an updated synthesis in plain Markdown. No JSON, no "
            "code fences. Aim for 3–6 paragraphs."
        )
        raw = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=self.config.temperature,
            synthesis=True,
        )
        return _strip_think(raw) or session.synthesis

    def _prioritize_queue(self, session: ScholarSession) -> list[str]:
        """Re-rank the queue by promise relative to current synthesis."""
        if not session.articles_queued:
            return session.articles_queued

        # Cap candidates we send to LLM — long queues are common.
        head = session.articles_queued[:80]
        tail = session.articles_queued[80:]

        synth = session.synthesis or "[no synthesis yet]"
        if len(synth) > _MAX_SYNTHESIS_CHARS_IN_PROMPT:
            synth = synth[:_MAX_SYNTHESIS_CHARS_IN_PROMPT] + "\n…[truncated]"

        listing = "\n".join(f"- {q}" for q in head)
        system = (
            "You are a Scholar pruning and re-ordering a queue of "
            "candidate articles. Promote articles that fill genuine gaps "
            "in the current synthesis; demote duplicates and tangents."
        )
        user = (
            f"TOPIC: {session.topic}\n\n"
            f"CURRENT SYNTHESIS:\n{synth}\n\n"
            f"CANDIDATE QUEUE (each line is a queue identifier):\n{listing}\n\n"
            "Reply with JSON only — a re-ordered list of the SAME identifiers "
            "(you may drop tangents but must NOT invent new ones):\n"
            '{ "queue": ["source:identifier", "source:identifier", ...] }'
        )
        raw = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.3,
            synthesis=True,
        )
        data = _extract_json(raw)
        if isinstance(data, dict):
            new_head = data.get("queue") or []
            if isinstance(new_head, list):
                valid = set(head)
                cleaned = [str(q) for q in new_head if str(q) in valid]
                # Drop duplicates while preserving LLM order.
                seen: set[str] = set()
                deduped = []
                for q in cleaned:
                    if q not in seen:
                        seen.add(q)
                        deduped.append(q)
                if deduped:
                    return deduped + tail
        return session.articles_queued

    def _finalize(self, session: ScholarSession) -> str:
        """Write the final synthesis at session end."""
        notes = session.notes[-_MAX_NOTES_FOR_FINALIZE:]
        notes_block = self._format_notes_for_prompt(notes)

        system = (
            "You are a Scholar concluding a research study. Produce a "
            "polished final synthesis from your notes. Be substantive, "
            "well-structured, and rigorously grounded — every claim should "
            "trace to a note. Cite article titles inline like [Title]."
        )
        user = (
            f"TOPIC: {session.topic}\n\n"
            f"ALL NOTES (most-recent {len(notes)}):\n{notes_block}\n\n"
            "Write a final synthesis in plain Markdown. Suggested structure:\n"
            "## Overview (1 paragraph)\n"
            "## Key findings (3–6 paragraphs, organised by sub-theme)\n"
            "## Open questions (1 paragraph)\n"
            "Do not output JSON or code fences."
        )
        raw = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=self.config.temperature,
            synthesis=True,
        )
        return _strip_think(raw) or session.synthesis

    # ── prompt formatting ───────────────────────────────────────────────

    @staticmethod
    def _format_notes_for_prompt(notes: list[ScholarNote]) -> str:
        """Render a list of notes as a compact prompt block."""
        if not notes:
            return "[no notes yet]"
        lines: list[str] = []
        for n in notes:
            lines.append(f"### {n.article_title}  (relevance {n.relevance_score:.1f})")
            if n.summary:
                lines.append(n.summary)
            for f in n.key_facts:
                lines.append(f"  - {f}")
            lines.append("")
        return "\n".join(lines)
