"""Series subcommand CLI — manage writers'-bible records.

A "series" ties multiple episodes together with a recurring writers'-bible
(tone, world facts, character canon). When a show is generated with
``--series ID``, the bible plus the last few episode summaries are
formatted into a SERIES CONTEXT block that gets prepended to the brief
before the Architect runs.

Subcommands:

    create TITLE [--type T] [--tone T] [--bible PATH|TEXT]
        Create a new series. Prints the generated UUID on stdout.

    list
        List all series, newest first.

    inspect SERIES_ID
        Print the full record for one series, including bible text and
        the last 5 episodes generated under it.

    assets SERIES_ID [--type voice|music|visual]
        List the approved asset pool for a series.

    approve SERIES_ID --audition-id SESSION_ID [--approve-all]
        Approve asset(s) from an audition session into the pool.
        --approve-all approves every candidate in the session.

    remove-asset SERIES_ID --asset-id PK
        Remove an asset from the pool by its integer PK.

    next SERIES_ID [--topic TEXT] [--brief-output PATH] [--run]
        Auto-draft the brief for the next episode, using the bible +
        recent-episode summaries.  Writes the brief to disk and prints
        its path.  Pass --run to also invoke the show pipeline on it
        via a fresh ``python run.py`` subprocess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vox_romana.core.audience import AudienceProfile
from vox_romana.core.catalog import Catalog


def _read_bible(value: str | None) -> str:
    """Treat ``value`` as a file path if it exists, otherwise as raw text."""
    if not value:
        return ""
    p = Path(value)
    if p.exists() and p.is_file():
        return p.read_text().strip()
    return value


def _cmd_create(args: argparse.Namespace) -> int:
    bible = _read_bible(args.bible)
    # Build AudienceProfile if any audience flag was provided.
    audience_flags = (
        args.audience_age,
        args.audience_education,
        args.audience_vocab,
        args.audience_tone,
        args.audience_prior,
    )
    audience_profile = None
    if any(f is not None for f in audience_flags):
        audience_profile = AudienceProfile(
            age_range=args.audience_age,
            education_level=args.audience_education,
            vocabulary_level=args.audience_vocab or "conversational",
            preferred_tone=args.audience_tone,
            prior_knowledge=args.audience_prior,
        )
    with Catalog() as cat:
        series_id = cat.create_series(
            title=args.title,
            show_type=args.type,
            tone=args.tone,
            bible=bible,
            audience_profile=audience_profile,
        )
    print(series_id)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with Catalog() as cat:
        series_list = cat.list_series()
    if not series_list:
        print("No series yet. Create one with: python run.py --mode series create \"My Show\"")
        return 0
    for s in series_list:
        type_part = f" [{s.show_type}]" if s.show_type else ""
        tone_part = f"  tone={s.tone}" if s.tone else ""
        print(f"{s.series_id}  {s.title}{type_part}{tone_part}  ({s.created_at})")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    with Catalog() as cat:
        series = cat.get_series(args.series_id)
        if series is None:
            print(f"Error: unknown series: {args.series_id}", file=sys.stderr)
            return 1
        episodes = cat.get_recent_episode_summaries(args.series_id, n=5)

    print(f"Series ID:   {series.series_id}")
    print(f"Title:       {series.title}")
    print(f"Show type:   {series.show_type or '(any)'}")
    print(f"Tone:        {series.tone or '(none)'}")
    print(f"Created:     {series.created_at}")
    print()
    print("Bible:")
    print("─" * 60)
    print(series.bible or "(empty)")
    print("─" * 60)
    if series.audience_profile is not None:
        ap = series.audience_profile
        print()
        print("Audience profile:")
        print("─" * 60)
        print(ap.to_prompt_text())
        print("─" * 60)
    if episodes:
        print()
        print("Recent episodes:")
        for ep in episodes:
            ep_no = ep["episode_number"] or "—"
            print(f"  • [{ep_no}] {ep['title']}  ({ep['created_at']})")
    return 0


def _cmd_assets(args: argparse.Namespace) -> int:
    """List the approved asset pool for a series."""
    asset_type = getattr(args, "asset_type", None)
    with Catalog() as cat:
        series = cat.get_series(args.series_id)
        if series is None:
            print(f"Error: unknown series: {args.series_id}", file=sys.stderr)
            return 1
        assets = cat.list_assets(args.series_id, asset_type=asset_type)

    if not assets:
        filter_note = f" (type={asset_type})" if asset_type else ""
        print(f"No assets in pool for series {args.series_id}{filter_note}.")
        return 0

    print(f"Assets for series: {series.title} ({args.series_id})")
    print(f"{'─' * 70}")
    print(f"  {'ID':>4}  {'Type':<8}  {'Asset ID':<24}  {'Character':<20}  Notes")
    print(f"  {'─'*4}  {'─'*8}  {'─'*24}  {'─'*20}  {'─'*20}")
    for a in assets:
        char = a.character_name or "—"
        notes = a.notes[:20] if a.notes else "—"
        print(f"  {a.id:>4}  {a.asset_type:<8}  {a.asset_id:<24}  {char:<20}  {notes}")
    print(f"{'─' * 70}")
    print(f"  {len(assets)} asset(s) total.")
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    """Approve asset(s) from an audition session into the pool."""
    with Catalog() as cat:
        series = cat.get_series(args.series_id)
        if series is None:
            print(f"Error: unknown series: {args.series_id}", file=sys.stderr)
            return 1

        session_id = args.audition_id
        audition = cat.get_audition(session_id)
        if audition is None:
            print(f"Error: audition session not found: {session_id}", file=sys.stderr)
            return 1

        if args.approve_all:
            pks = cat.approve_all_from_audition(session_id, series_id=args.series_id)
            print(f"Approved {len(pks)} asset(s) from session {session_id} into series {args.series_id}.")
            for pk in pks:
                print(f"  asset id={pk}")
        else:
            # Single-asset approval: require --asset-id flag
            asset_id_val = getattr(args, "asset_id_val", None)
            if not asset_id_val:
                print(
                    "Error: supply --asset-id ASSET_ID to approve a single asset, "
                    "or use --approve-all to approve the entire session.",
                    file=sys.stderr,
                )
                return 1
            pk = cat.approve_asset(
                series_id=args.series_id,
                asset_type=audition.audition_type,
                asset_id=asset_id_val,
                character_name=getattr(args, "character_name", None),
                source_path=getattr(args, "source_path", "") or "",
                notes=getattr(args, "notes", "") or "",
            )
            # Mark chosen on the audition row.
            with cat._conn:
                cat._conn.execute(
                    "UPDATE auditions SET chosen = ? WHERE session_id = ?",
                    (asset_id_val, session_id),
                )
            print(f"Approved asset '{asset_id_val}' → pool id={pk} (series {args.series_id}).")
    return 0


def _cmd_remove_asset(args: argparse.Namespace) -> int:
    """Remove an asset from the series pool by PK."""
    with Catalog() as cat:
        series = cat.get_series(args.series_id)
        if series is None:
            print(f"Error: unknown series: {args.series_id}", file=sys.stderr)
            return 1
        cat.remove_asset(args.asset_pk)
    print(f"Removed asset id={args.asset_pk} from series {args.series_id}.")
    return 0


def _cmd_next(args: argparse.Namespace) -> int:
    """Auto-draft the brief for the next episode in a series.

    Pulls the bible + recent-episode summaries from the catalog, calls
    :class:`EpisodePlanner` for an LLM-drafted brief, writes the brief
    text to disk, and prints the path.  When ``--run`` is set we then
    invoke the main pipeline as a subprocess so the user gets the
    finished show without a second copy-paste step.

    Returns:
        0 on success, 1 on any guard failure (unknown series, planner
        error, missing API key, subprocess non-zero).
    """
    # Imports are deferred so series list/inspect/etc. don't pay the
    # price of loading the LLM client.
    from vox_romana.core.client import DeepSeekClient
    from vox_romana.planning.episode_planner import EpisodePlanner

    with Catalog() as cat:
        series = cat.get_series(args.series_id)
        if series is None:
            print(f"Error: unknown series: {args.series_id}", file=sys.stderr)
            return 1
        recent = cat.get_recent_episode_summaries(args.series_id, n=5)
        ep_no = cat.next_episode_number(args.series_id)

    try:
        client = DeepSeekClient()
    except ValueError as exc:
        # Missing DEEPSEEK_API_KEY — return cleanly so callers see the
        # cause instead of a stack trace.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    planner = EpisodePlanner(client)
    try:
        brief = planner.plan_next(
            series_title=series.title,
            series_show_type=series.show_type,
            series_tone=series.tone,
            bible=series.bible,
            recent_episodes=recent,
            next_episode_number=ep_no,
            topic=args.topic,
        )
    except RuntimeError as exc:
        print(f"Error: episode planner failed: {exc}", file=sys.stderr)
        return 1

    # ── Resolve output path for the brief text ────────────────────────
    if args.brief_output:
        brief_path = Path(args.brief_output)
    else:
        # Default: data/series/{id}/briefs/ep_{NN}.txt — same convention
        # used elsewhere for series-scoped artifacts.
        from vox_romana.core.paths import DATA_ROOT
        brief_dir = DATA_ROOT / "series" / args.series_id / "briefs"
        brief_dir.mkdir(parents=True, exist_ok=True)
        brief_path = brief_dir / f"ep_{ep_no:02d}.txt"

    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(brief.prompt_text, encoding="utf-8")

    # ── Print a human-readable summary ────────────────────────────────
    print(f"Episode #{ep_no} brief drafted for series {args.series_id} ({series.title})")
    print(f"  Title:    {brief.title}")
    if brief.angle:
        print(f"  Angle:    {brief.angle}")
    if brief.hooks:
        print(f"  Hooks:    {len(brief.hooks)}")
        for h in brief.hooks:
            print(f"    • {h}")
    if brief.guests:
        print(f"  Guests:   {', '.join(brief.guests)}")
    print(f"  Duration: ~{brief.estimated_duration:.0f} min (estimated)")
    print(f"  Brief →   {brief_path}")

    if not args.run:
        print(
            "\nNext: run the pipeline yourself with:\n"
            f"  python run.py {brief_path} --series {args.series_id}"
        )
        return 0

    # ── Run the pipeline as a subprocess ──────────────────────────────
    # We deliberately invoke a fresh ``python run.py`` rather than
    # importing run.py here, because run.py's main() expects to be the
    # process entry point (it calls sys.exit on errors, configures
    # logging, etc.).  Subprocess isolation keeps this clean.
    import subprocess

    cmd = [
        sys.executable,
        "run.py",
        str(brief_path),
        "--series",
        args.series_id,
    ]
    if series.show_type:
        cmd += ["--type", series.show_type]
    print(f"\nLaunching pipeline: {' '.join(cmd)}\n")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"Error: pipeline exited with code {rc}", file=sys.stderr)
        return 1
    return 0


def _cmd_instantiate(args: argparse.Namespace) -> int:
    """Instantiate a studio from a one-sentence brief.

    Runs the staged inference, registers the SeriesRecord, drafts episode 1
    (unless ``--no-draft``), materializes a portable bundle, and zips it.
    """
    from vox_romana.core.client import DeepSeekClient
    from vox_romana.studio import StudioInstantiator, materialize_bundle, zip_bundle

    try:
        client = DeepSeekClient()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    instantiator = StudioInstantiator(client)
    try:
        result = instantiator.instantiate(
            args.brief,
            draft_episode=not args.no_draft,
            topic=args.topic,
        )
    except ValueError as exc:
        print(f"Error: studio inference failed: {exc}", file=sys.stderr)
        return 1

    meta = result.meta
    print(f"Studio instantiated: {meta.title} [{meta.show_type}]")
    print(f"  Series ID: {result.series_id}")
    if meta.tone:
        print(f"  Tone:      {meta.tone}")
    print(f"  Slug:      {meta.series_id}")

    # ── Materialize + zip the portable bundle ────────────────────────
    with Catalog() as cat:
        series = cat.get_series(result.series_id)
    if series is None:
        print(
            f"Error: could not re-read series {result.series_id} from catalog.",
            file=sys.stderr,
        )
        return 1

    dest_root = Path(args.bundle_dir) if args.bundle_dir else Path.cwd()
    bundle_dir = materialize_bundle(
        series,
        dest_root,
        slug=meta.series_id,
        episode_brief=result.episode_brief,
    )
    archive = zip_bundle(bundle_dir)
    print(f"  Bundle:    {bundle_dir}")
    print(f"  Zip:       {archive}")

    if result.episode_brief is not None:
        print(f"  Episode 1: {result.episode_brief.title}")

    print(
        "\nNext: run the pipeline with:\n"
        f"  python run.py prompts/episode_1_brief.txt --series {result.series_id}"
    )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Materialize + zip an existing series into a portable bundle."""
    from vox_romana.studio import materialize_bundle, zip_bundle

    with Catalog() as cat:
        series = cat.get_series(args.series_id)
        if series is None:
            print(f"Error: unknown series: {args.series_id}", file=sys.stderr)
            return 1

    dest_root = Path(args.bundle_dir) if args.bundle_dir else Path.cwd()
    bundle_dir = materialize_bundle(series, dest_root)
    archive = zip_bundle(bundle_dir)

    print(f"Bundle: {bundle_dir}")
    print(f"Zip:    {archive}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vox_romana series",
        description="Manage series writers'-bible records.",
    )
    sub = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True

    sp_create = sub.add_parser("create", help="Create a new series.")
    sp_create.add_argument("title", type=str, help="Series title.")
    sp_create.add_argument("--type", type=str, default=None,
                           choices=["debate", "news", "podcast", "audiobook"],
                           help="Default show type for episodes in this series.")
    sp_create.add_argument("--tone", type=str, default=None,
                           help="Recurring tone (e.g. 'serious', 'satirical').")
    sp_create.add_argument("--bible", type=str, default=None,
                           metavar="PATH_OR_TEXT",
                           help="Bible text, or a path to a text file.")
    sp_create.add_argument("--audience-age", dest="audience_age",
                           type=str, default=None, metavar="RANGE",
                           help="Target age range, e.g. '18-35'.")
    sp_create.add_argument("--audience-education", dest="audience_education",
                           type=str, default=None,
                           choices=["high-school", "undergraduate", "graduate", "expert"],
                           help="Assumed education level.")
    sp_create.add_argument("--audience-vocab", dest="audience_vocab",
                           type=str, default=None,
                           choices=["plain", "conversational", "educated", "expert"],
                           help="Vocabulary complexity level (default: conversational).")
    sp_create.add_argument("--audience-tone", dest="audience_tone",
                           type=str, default=None, metavar="DESCRIPTION",
                           help="Preferred tone, e.g. 'curious, skeptical'.")
    sp_create.add_argument("--audience-prior", dest="audience_prior",
                           type=str, default=None, metavar="TEXT",
                           help="Assumed prior knowledge, e.g. 'knows Python, new to ML'.")
    sp_create.set_defaults(func=_cmd_create)

    sp_list = sub.add_parser("list", help="List all series.")
    sp_list.set_defaults(func=_cmd_list)

    sp_inspect = sub.add_parser("inspect", help="Show one series in detail.")
    sp_inspect.add_argument("series_id", type=str, help="Series UUID.")
    sp_inspect.set_defaults(func=_cmd_inspect)

    sp_assets = sub.add_parser("assets", help="List the approved asset pool for a series.")
    sp_assets.add_argument("series_id", type=str, help="Series UUID.")
    sp_assets.add_argument(
        "--type",
        dest="asset_type",
        type=str,
        default=None,
        choices=["voice", "music", "visual"],
        help="Filter by asset type.",
    )
    sp_assets.set_defaults(func=_cmd_assets)

    sp_approve = sub.add_parser(
        "approve",
        help="Approve asset(s) from an audition session into the pool.",
    )
    sp_approve.add_argument("series_id", type=str, help="Series UUID.")
    sp_approve.add_argument(
        "--audition-id",
        dest="audition_id",
        type=str,
        required=True,
        metavar="SESSION_ID",
        help="Audition session ID to approve from.",
    )
    sp_approve.add_argument(
        "--approve-all",
        dest="approve_all",
        action="store_true",
        default=False,
        help="Approve every candidate in the session.",
    )
    sp_approve.add_argument(
        "--asset-id",
        dest="asset_id_val",
        type=str,
        default=None,
        metavar="ASSET_ID",
        help="Specific asset ID to approve (required when --approve-all is not set).",
    )
    sp_approve.add_argument(
        "--character",
        dest="character_name",
        type=str,
        default=None,
        metavar="NAME",
        help="Character name for voice assets.",
    )
    sp_approve.add_argument(
        "--source-path",
        dest="source_path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the reference clip or file.",
    )
    sp_approve.add_argument(
        "--notes",
        dest="notes",
        type=str,
        default=None,
        metavar="TEXT",
        help="Optional notes about this asset.",
    )
    sp_approve.set_defaults(func=_cmd_approve)

    sp_remove = sub.add_parser("remove-asset", help="Remove an asset from the pool.")
    sp_remove.add_argument("series_id", type=str, help="Series UUID.")
    sp_remove.add_argument(
        "--asset-id",
        dest="asset_pk",
        type=int,
        required=True,
        metavar="PK",
        help="Integer PK (id column) of the asset to remove.",
    )
    sp_remove.set_defaults(func=_cmd_remove_asset)

    sp_next = sub.add_parser(
        "next",
        help="Auto-draft the brief for the next episode using the bible + history.",
    )
    sp_next.add_argument("series_id", type=str, help="Series UUID.")
    sp_next.add_argument(
        "--topic",
        dest="topic",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "Optional topic constraint, e.g. "
            "\"AI copyright in creative industries\"."
        ),
    )
    sp_next.add_argument(
        "--brief-output",
        dest="brief_output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Where to write the drafted brief.  Defaults to "
            "DATA_ROOT/series/{id}/briefs/ep_NN.txt."
        ),
    )
    sp_next.add_argument(
        "--run",
        dest="run",
        action="store_true",
        default=False,
        help=(
            "After drafting the brief, invoke 'python run.py BRIEF "
            "--series ID' as a subprocess to actually produce the "
            "episode.  Default: just write the brief and exit."
        ),
    )
    sp_next.set_defaults(func=_cmd_next)

    sp_instantiate = sub.add_parser(
        "instantiate",
        help="Create a full studio (series) from a one-sentence brief.",
    )
    sp_instantiate.add_argument("brief", type=str, help="One-sentence studio brief.")
    sp_instantiate.add_argument(
        "--no-draft",
        dest="no_draft",
        action="store_true",
        default=False,
        help="Skip drafting episode 1's brief.",
    )
    sp_instantiate.add_argument(
        "--topic",
        dest="topic",
        type=str,
        default=None,
        metavar="TEXT",
        help="Optional topic constraint for episode 1.",
    )
    sp_instantiate.add_argument(
        "--bundle-dir",
        dest="bundle_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Parent directory for the portable bundle (default: "
            "current working directory)."
        ),
    )
    sp_instantiate.set_defaults(func=_cmd_instantiate)

    sp_export = sub.add_parser(
        "export",
        help="Materialize an existing series into a portable bundle + zip.",
    )
    sp_export.add_argument("series_id", type=str, help="Series UUID.")
    sp_export.add_argument(
        "--bundle-dir",
        dest="bundle_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Parent directory for the portable bundle (default: "
            "current working directory)."
        ),
    )
    sp_export.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    if rc:
        sys.exit(rc)
