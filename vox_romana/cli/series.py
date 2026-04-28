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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    if rc:
        sys.exit(rc)
