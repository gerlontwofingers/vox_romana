"""run_university.py — command-line interface for the University research system.

USAGE
-----

::

    # Run a new deep-research study
    python run_university.py study "Roman aqueducts and hydraulic engineering"

    # Study with explicit budget
    python run_university.py study "Universal Basic Income" --hours 2.0

    # List all sessions (from index.json)
    python run_university.py list

    # Inspect a session (synthesis + notes)
    python run_university.py inspect a1b2c3d4e5f6

    # One-shot question against a completed session
    python run_university.py ask a1b2c3d4e5f6 "What materials did Roman roads use?"

    # Interactive chat REPL against a session
    python run_university.py chat a1b2c3d4e5f6

    # Extend a paused/interrupted session
    python run_university.py extend a1b2c3d4e5f6 --hours 1.0

    # Export session to Markdown or JSONL
    python run_university.py export a1b2c3d4e5f6
    python run_university.py export a1b2c3d4e5f6 --format jsonl

SUBCOMMANDS
-----------
study      Run a new Scholar on one topic.
ask        Answer a one-shot question grounded in a session's notes.
chat       Interactive Q&A REPL against a session.
list       List all sessions from the session registry.
inspect    Print synthesis and notes for a session.
extend     Resume a paused/interrupted session for more research time.
export     Write a session to Markdown or JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_session(session_id: str):
    """Load a ScholarSession by ID.  Exits with a clear error if not found."""
    from vox_romana.core.paths import UNIVERSITY_ROOT
    from vox_romana.university.session import ScholarSession

    session_dir = UNIVERSITY_ROOT / session_id
    try:
        return ScholarSession.load(session_dir)
    except FileNotFoundError:
        print(
            f"Error: session '{session_id}' not found in {UNIVERSITY_ROOT}.\n"
            "Run `python run_university.py list` to see available sessions.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: could not load session '{session_id}': {exc}", file=sys.stderr)
        sys.exit(1)


def _build_university():
    """Build a University with OllamaClient + WikipediaSource defaults."""
    from vox_romana.local.ollama import OllamaClient
    from vox_romana.university.sources import WikipediaSource
    from vox_romana.university.university import University

    client = OllamaClient.from_config()
    sources = [WikipediaSource()]
    return University(client=client, sources=sources)


def _build_scholar():
    """Build a bare Scholar (no catalog wiring) for ask/chat/extend."""
    from vox_romana.local.ollama import OllamaClient
    from vox_romana.university.scholar import Scholar
    from vox_romana.university.sources import WikipediaSource

    client = OllamaClient.from_config()
    sources = [WikipediaSource()]
    return Scholar(client=client, sources=sources)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_study(args: argparse.Namespace) -> None:
    """Run a new deep-research study on a topic."""
    uni = _build_university()
    topic = args.topic
    hours = args.hours

    if hours is not None and hours <= 0:
        print("Error: --hours must be positive.", file=sys.stderr)
        sys.exit(1)

    print(f"\nUniversity: starting study — {topic!r}")
    if hours:
        print(f"  budget: {hours:.1f}h")
    else:
        print("  budget: self-determined by Scholar")
    print()

    session = uni.study(topic, budget_hours=hours)

    print(f"\n{'─' * 60}")
    print(f"Session ID : {session.session_id}")
    print(f"Topic      : {session.topic}")
    print(f"Status     : {session.status}")
    print(f"Notes      : {len(session.notes)}")
    print(f"Articles   : {len(session.articles_read)} read")
    print(f"Sources    : {', '.join(session.sources_used) or 'none'}")
    if session.catalog_study_id:
        print(f"Catalog ID : {session.catalog_study_id}")
    print()
    if session.synthesis:
        print("Synthesis:")
        print(f"  {session.synthesis[:500]}{'…' if len(session.synthesis) > 500 else ''}")
    print()
    print(f"Session saved.  Inspect: python run_university.py inspect {session.session_id}")


def _cmd_ask(args: argparse.Namespace) -> None:
    """Answer a one-shot question grounded in a session."""
    session = _load_session(args.session_id)
    scholar = _build_scholar()
    question = args.question

    print(f"\nQuestion: {question}")
    print(f"Session : {session.session_id} — {session.topic!r}")
    print()
    answer = scholar.ask(question, session=session)
    print(answer)
    print()


def _cmd_chat(args: argparse.Namespace) -> None:
    """Interactive Q&A REPL against a session."""
    session = _load_session(args.session_id)
    scholar = _build_scholar()
    scholar.chat(session)


def _cmd_list(args: argparse.Namespace) -> None:
    """List all sessions from the index."""
    from vox_romana.core.paths import UNIVERSITY_ROOT

    index_path = UNIVERSITY_ROOT / "index.json"
    if not index_path.exists():
        print("No sessions found.  Run `python run_university.py study <topic>` to start.")
        return

    try:
        with index_path.open(encoding="utf-8") as fh:
            records = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error reading index: {exc}", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("No sessions found.")
        return

    # Filter by status if requested.
    status_filter = getattr(args, "status", None)
    if status_filter:
        records = [r for r in records if r.get("status") == status_filter]
        if not records:
            print(f"No sessions with status '{status_filter}'.")
            return

    col_w = {"id": 14, "status": 11, "notes": 6, "articles": 8, "updated": 20}
    header = (
        f"{'SESSION ID':<{col_w['id']}}  "
        f"{'STATUS':<{col_w['status']}}  "
        f"{'NOTES':>{col_w['notes']}}  "
        f"{'ARTICLES':>{col_w['articles']}}  "
        f"{'UPDATED':<{col_w['updated']}}  TOPIC"
    )
    print(header)
    print("─" * (len(header) + 10))

    for r in records:
        sid = r.get("session_id", "?")[:14]
        status = r.get("status", "?")[:11]
        notes = str(r.get("notes_count", 0))
        arts = str(r.get("articles_read", 0))
        updated = r.get("updated_at", "")[:19].replace("T", " ")
        topic = r.get("topic", "")
        print(
            f"{sid:<{col_w['id']}}  "
            f"{status:<{col_w['status']}}  "
            f"{notes:>{col_w['notes']}}  "
            f"{arts:>{col_w['articles']}}  "
            f"{updated:<{col_w['updated']}}  {topic}"
        )


def _cmd_inspect(args: argparse.Namespace) -> None:
    """Print full synthesis and notes for a session."""
    session = _load_session(args.session_id)

    print(f"\n{'═' * 60}")
    print(f"Session ID : {session.session_id}")
    print(f"Topic      : {session.topic}")
    print(f"Status     : {session.status}")
    print(f"Started    : {session.started_at}")
    print(f"Updated    : {session.updated_at}")
    print(f"Notes      : {len(session.notes)}")
    print(f"Articles   : {len(session.articles_read)} read  |  {len(session.articles_queued)} queued")
    print(f"Sources    : {', '.join(session.sources_used) or 'none'}")
    if session.budget_hours is not None:
        print(f"Budget     : {session.budget_hours:.1f}h")
    if session.catalog_study_id:
        print(f"Catalog ID : {session.catalog_study_id}")
    print(f"{'═' * 60}")

    if session.synthesis:
        print("\nSYNTHESIS")
        print("─" * 60)
        print(session.synthesis)
    else:
        print("\n[No synthesis yet]")

    if not args.no_notes and session.notes:
        print(f"\nNOTES ({len(session.notes)})")
        print("─" * 60)
        for i, note in enumerate(session.notes, 1):
            print(f"\n[{i}] {note.article_title}")
            print(f"    source : {note.source_name}  |  relevance: {note.relevance_score:.1f}")
            if note.source_url:
                print(f"    url    : {note.source_url}")
            if note.summary:
                summary = note.summary
                if len(summary) > 200:
                    summary = summary[:199] + "…"
                print(f"    summary: {summary}")
            if note.key_facts:
                print("    facts  :")
                for fact in note.key_facts[:5]:
                    print(f"      • {fact}")
    print()


def _cmd_extend(args: argparse.Namespace) -> None:
    """Resume a paused/interrupted session."""
    session = _load_session(args.session_id)
    hours = args.hours

    if hours is not None and hours <= 0:
        print("Error: --hours must be positive.", file=sys.stderr)
        sys.exit(1)

    print(f"\nExtending session {session.session_id} — {session.topic!r}")
    print(f"  current status : {session.status}")
    print(f"  notes so far   : {len(session.notes)}")
    if hours:
        print(f"  additional budget: {hours:.1f}h")
    else:
        print("  additional budget: self-determined by Scholar")
    print()

    scholar = _build_scholar()
    session = scholar.extend(session, additional_hours=hours)

    print(f"\n{'─' * 60}")
    print(f"Session ID : {session.session_id}")
    print(f"Status     : {session.status}")
    print(f"Notes      : {len(session.notes)}")
    print(f"Articles   : {len(session.articles_read)} read")
    print()
    if session.synthesis:
        print("Updated synthesis:")
        print(f"  {session.synthesis[:400]}{'…' if len(session.synthesis) > 400 else ''}")
    print()


def _cmd_export(args: argparse.Namespace) -> None:
    """Export a session to Markdown or JSONL."""
    session = _load_session(args.session_id)
    fmt = args.format
    output_path = args.output

    if fmt == "markdown":
        content = _export_markdown(session)
        default_name = f"university_{session.session_id}.md"
    else:
        content = _export_jsonl(session)
        default_name = f"university_{session.session_id}.jsonl"

    if output_path:
        dest = Path(output_path)
    else:
        dest = Path(default_name)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    print(f"Exported {len(content):,} chars → {dest}")


def _export_markdown(session) -> str:
    """Render a session to Markdown."""
    lines: list[str] = []
    lines.append(f"# University Research: {session.topic}")
    lines.append("")
    lines.append(f"**Session ID:** `{session.session_id}`  ")
    lines.append(f"**Status:** {session.status}  ")
    lines.append(f"**Started:** {session.started_at}  ")
    lines.append(f"**Updated:** {session.updated_at}  ")
    lines.append(f"**Articles read:** {len(session.articles_read)}  ")
    lines.append(f"**Notes:** {len(session.notes)}  ")
    if session.sources_used:
        lines.append(f"**Sources:** {', '.join(session.sources_used)}  ")
    lines.append("")

    if session.synthesis:
        lines.append("## Synthesis")
        lines.append("")
        lines.append(session.synthesis)
        lines.append("")

    if session.notes:
        lines.append("## Notes")
        lines.append("")
        for i, note in enumerate(session.notes, 1):
            lines.append(f"### [{i}] {note.article_title}")
            if note.source_url:
                lines.append(f"- **URL:** {note.source_url}")
            lines.append(f"- **Source:** {note.source_name}")
            lines.append(f"- **Relevance:** {note.relevance_score:.1f}/10")
            if note.summary:
                lines.append(f"- **Summary:** {note.summary}")
            if note.key_facts:
                lines.append("- **Key facts:**")
                for fact in note.key_facts:
                    lines.append(f"  - {fact}")
            lines.append("")

    return "\n".join(lines)


def _export_jsonl(session) -> str:
    """Render a session's notes as DocumentChunk JSONL (for ingestion)."""
    chunks = session.to_jsonl_chunks()
    return "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_university.py",
        description="University deep-research CLI — powered by a local Ollama/Scholar agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_university.py study 'Roman aqueducts'\n"
            "  python run_university.py study 'Universal Basic Income' --hours 2.0\n"
            "  python run_university.py list\n"
            "  python run_university.py inspect a1b2c3d4e5f6\n"
            "  python run_university.py ask a1b2c3d4e5f6 'What did Roman roads use?'\n"
            "  python run_university.py chat a1b2c3d4e5f6\n"
            "  python run_university.py extend a1b2c3d4e5f6 --hours 1.0\n"
            "  python run_university.py export a1b2c3d4e5f6 --format markdown\n"
        ),
    )

    sub = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True

    # ── study ────────────────────────────────────────────────────────────
    sp_study = sub.add_parser(
        "study",
        help="Run a new deep-research study on a topic.",
        description=(
            "Spawn a Scholar and run a time-budgeted read/synthesise loop over "
            "Wikipedia (and local memory when configured).  The session is "
            "checkpointed to disk after every article."
        ),
    )
    sp_study.add_argument(
        "topic",
        type=str,
        help="Research directive (free-form text). E.g. 'Roman engineering'.",
    )
    sp_study.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Time budget in hours. Omit to let the Scholar self-determine.",
    )
    sp_study.set_defaults(func=_cmd_study)

    # ── ask ──────────────────────────────────────────────────────────────
    sp_ask = sub.add_parser(
        "ask",
        help="Answer a one-shot question grounded in a session's notes.",
        description=(
            "Pose a single question to the Scholar using the notes and synthesis "
            "from a completed (or paused) session.  Does not modify the session."
        ),
    )
    sp_ask.add_argument(
        "session_id",
        type=str,
        help="12-char session ID (from `run_university.py list`).",
    )
    sp_ask.add_argument(
        "question",
        type=str,
        help="Question to answer (free-form text).",
    )
    sp_ask.set_defaults(func=_cmd_ask)

    # ── chat ─────────────────────────────────────────────────────────────
    sp_chat = sub.add_parser(
        "chat",
        help="Interactive Q&A REPL against a session.",
        description=(
            "Start an interactive chat session grounded in a Scholar session's "
            "notes and synthesis.  Enter a blank line or Ctrl-D to exit."
        ),
    )
    sp_chat.add_argument(
        "session_id",
        type=str,
        help="12-char session ID.",
    )
    sp_chat.set_defaults(func=_cmd_chat)

    # ── list ─────────────────────────────────────────────────────────────
    sp_list = sub.add_parser(
        "list",
        help="List all sessions from the session registry.",
    )
    sp_list.add_argument(
        "--status",
        type=str,
        default=None,
        choices=["active", "complete", "paused", "interrupted"],
        help="Filter by session status.",
    )
    sp_list.set_defaults(func=_cmd_list)

    # ── inspect ──────────────────────────────────────────────────────────
    sp_inspect = sub.add_parser(
        "inspect",
        help="Print synthesis and notes for a session.",
    )
    sp_inspect.add_argument(
        "session_id",
        type=str,
        help="12-char session ID.",
    )
    sp_inspect.add_argument(
        "--no-notes",
        action="store_true",
        dest="no_notes",
        help="Print only the synthesis (skip per-note detail).",
    )
    sp_inspect.set_defaults(func=_cmd_inspect)

    # ── extend ───────────────────────────────────────────────────────────
    sp_extend = sub.add_parser(
        "extend",
        help="Resume a paused/interrupted session for more research time.",
    )
    sp_extend.add_argument(
        "session_id",
        type=str,
        help="12-char session ID.",
    )
    sp_extend.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Additional time budget in hours. Omit for Scholar self-estimate.",
    )
    sp_extend.set_defaults(func=_cmd_extend)

    # ── export ───────────────────────────────────────────────────────────
    sp_export = sub.add_parser(
        "export",
        help="Write a session to Markdown or JSONL.",
        description=(
            "Export a session's synthesis and notes.\n\n"
            "  markdown  — human-readable Markdown (default)\n"
            "  jsonl     — DocumentChunk JSONL compatible with `run.py --mode ingest`"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp_export.add_argument(
        "session_id",
        type=str,
        help="12-char session ID.",
    )
    sp_export.add_argument(
        "--format", "-f",
        type=str,
        default="markdown",
        choices=["markdown", "jsonl"],
        dest="format",
        help="Output format (default: markdown).",
    )
    sp_export.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path. Default: university_<session_id>.md / .jsonl",
    )
    sp_export.set_defaults(func=_cmd_export)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
