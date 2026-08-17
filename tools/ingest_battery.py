#!/usr/bin/env python3
"""Ingest a battery study directory into the longitudinal SQLite database.

Usage::

    python tools/ingest_battery.py study_results/20260524_123833
    python tools/ingest_battery.py study_results/20260524_123833 \\
        --production-id my_studio \\
        --db study_results/longitudinal.db

After ingestion you can inspect the results with::

    sqlite3 study_results/longitudinal.db \\
        "SELECT show_type, avg_word_ratio, n_pieces FROM runs WHERE run_id='20260524_123833'"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vox_romana.analysis.longitudinal import ingest_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a study_battery run directory into longitudinal.db.",
    )
    parser.add_argument(
        "study_dir",
        type=Path,
        help="Path to the study_battery output directory (contains index.json).",
    )
    parser.add_argument(
        "--production-id",
        default="default",
        metavar="ID",
        help="Production identifier to tag this run (default: 'default').",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=REPO_ROOT / "study_results" / "longitudinal.db",
        metavar="PATH",
        help="Path to the SQLite longitudinal DB (created if absent).",
    )
    args = parser.parse_args(argv)

    study_dir = args.study_dir.resolve()
    if not study_dir.exists():
        print(f"error: study_dir not found: {study_dir}", file=sys.stderr)
        return 1

    index_path = study_dir / "index.json"
    if not index_path.exists():
        print(f"error: {index_path} not found — is this a study_battery output dir?", file=sys.stderr)
        return 1

    print(f"Ingesting {study_dir.name} → {args.db}")
    print(f"  production_id: {args.production_id!r}")

    try:
        inserted = ingest_run(
            db_path=args.db,
            study_dir=study_dir,
            production_id=args.production_id,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total = sum(inserted.values())
    print(f"\nInserted {total} pieces across {len(inserted)} show types:")
    for show_type, n in sorted(inserted.items()):
        print(f"  {show_type:12s} {n} pieces")
    print(f"\nDB: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
