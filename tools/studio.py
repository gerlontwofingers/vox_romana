#!/usr/bin/env python3
"""Unified CLI for the vox_romana longitudinal data store.

Commands
--------
ingest   <study_dir>   Ingest a study_battery run into longitudinal.db.
status                 Show DB overview: runs, word ratios, calibration state.
decision               Record a deliberate parameter change and its hypothesis.
calibrate              Preview or apply parameter calibration from DB history.

Examples
--------
::

    # Ingest a battery run using the production_id from vox_romana.yaml
    python tools/studio.py ingest study_results/20260524_123833

    # Override production_id
    python tools/studio.py ingest study_results/20260524_123833 --production-id home_studio

    # Show DB status for all productions
    python tools/studio.py status

    # Filter status by production
    python tools/studio.py status --production-id home_studio

    # Record that you changed a parameter and why
    python tools/studio.py decision \\
        --show-type debate \\
        --hypothesis "Raising wrap_up_threshold to 0.80 will reduce debate overshoot" \\
        --parameters '{"wrap_up_threshold": 0.80}' \\
        --rationale "avg_word_ratio was 1.83 in run 20260524_123833; want to cut closing bloat." \\
        --run-before 20260524_123833

    # Preview calibration recommendations
    python tools/studio.py calibrate

    # Persist calibration_state rows + calibration.yaml
    python tools/studio.py calibrate --apply
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "study_results" / "longitudinal.db"

sys.path.insert(0, str(REPO_ROOT))

from vox_romana.analysis.longitudinal import init_db, ingest_run  # noqa: E402
from tools.calibrate import DEFAULT_OUTPUT, run_calibration  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_production_id() -> str:
    """Read production_id from vox_romana.yaml; fall back to 'default'."""
    for cfg_path in [
        REPO_ROOT / "vox_romana.yaml",
        Path.home() / ".vox_romana" / "config.yaml",
    ]:
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                val = cfg.get("production_id")
                if val:
                    return str(val)
            except Exception:
                pass
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# ingest
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> int:
    study_dir = args.study_dir.resolve()
    if not (study_dir / "index.json").exists():
        print(
            f"error: index.json not found in {study_dir} — "
            "is this a study_battery output directory?",
            file=sys.stderr,
        )
        return 1

    production_id = args.production_id or _read_production_id()
    print(f"Ingesting {study_dir.name} → {args.db}")
    print(f"  production_id: {production_id!r}")

    try:
        inserted = ingest_run(
            db_path=args.db,
            study_dir=study_dir,
            production_id=production_id,
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


# ─────────────────────────────────────────────────────────────────────────────
# status
# ─────────────────────────────────────────────────────────────────────────────

def cmd_status(args: argparse.Namespace) -> int:
    db_path = args.db
    if not db_path.exists():
        print(f"No DB found at {db_path}. Run 'studio ingest' first.")
        return 0

    production_filter = args.production_id
    where = "WHERE production_id = ?" if production_filter else ""
    bind: tuple = (production_filter,) if production_filter else ()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # ── runs summary ────────────────────────────────────────────────────
        run_rows = conn.execute(
            f"""
            SELECT production_id, show_type,
                   COUNT(*)                     AS n_runs,
                   SUM(n_pieces)                AS total_pieces,
                   SUM(n_ok)                    AS total_ok,
                   ROUND(AVG(avg_word_ratio), 3) AS mean_ratio,
                   MAX(run_id)                  AS latest_run
            FROM runs
            {where}
            GROUP BY production_id, show_type
            ORDER BY production_id, show_type
            """,
            bind,
        ).fetchall()

        if not run_rows:
            print("No runs ingested yet.")
            return 0

        print("═" * 80)
        print("LONGITUDINAL DB STATUS")
        print(f"DB: {db_path}")
        print("─" * 80)
        print(
            f"{'production':16s} {'show_type':12s} {'runs':>5s} "
            f"{'pieces':>7s} {'ok':>5s} {'mean_ratio':>10s}  latest_run"
        )
        print("─" * 80)
        for r in run_rows:
            ratio_s = f"{r['mean_ratio']:.3f}" if r["mean_ratio"] is not None else "   N/A"
            print(
                f"{r['production_id']:16s} {r['show_type']:12s} {r['n_runs']:>5d}"
                f" {r['total_pieces']:>7d} {r['total_ok']:>5d}"
                f" {ratio_s:>10s}  {r['latest_run']}"
            )
        print("═" * 80)

        # ── recent decisions ─────────────────────────────────────────────────
        dec_rows = conn.execute(
            f"""
            SELECT production_id, show_type, timestamp, hypothesis, outcome
            FROM decisions
            {where}
            ORDER BY decision_id DESC
            LIMIT 5
            """,
            bind,
        ).fetchall()

        if dec_rows:
            print("\nRECENT DECISIONS (last 5):")
            print("─" * 80)
            for d in dec_rows:
                ts = (d["timestamp"] or "")[:10]
                show = d["show_type"] or "all"
                hyp = (d["hypothesis"] or "")[:65]
                outcome = d["outcome"] or "(pending)"
                print(f"  [{ts}] [{show:9s}] {hyp}")
                print(f"    outcome: {outcome}")
            print("─" * 80)

        # ── calibration state ────────────────────────────────────────────────
        cal_rows = conn.execute(
            f"""
            SELECT production_id, show_type,
                   overshoot_compensation, avg_words_per_turn,
                   avg_words_per_segment, r_squared, n_runs_used, updated_at
            FROM calibration_state
            {where}
            ORDER BY production_id, show_type
            """,
            bind,
        ).fetchall()

        if cal_rows:
            print("\nCALIBRATION STATE:")
            print("─" * 80)
            for c in cal_rows:
                ts = (c["updated_at"] or "")[:10]
                r2 = f"{c['r_squared']:.3f}" if c["r_squared"] is not None else " N/A"
                print(
                    f"  {c['production_id']:16s} {c['show_type']:12s}"
                    f"  oc={c['overshoot_compensation']}  r²={r2}"
                    f"  n_runs={c['n_runs_used']}  updated={ts}"
                )
            print("─" * 80)

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# decision
# ─────────────────────────────────────────────────────────────────────────────

def cmd_decision(args: argparse.Namespace) -> int:
    production_id = args.production_id or _read_production_id()
    db_path = args.db

    if not db_path.exists():
        init_db(db_path)

    parameters_json: str | None = None
    if args.parameters:
        try:
            parsed = json.loads(args.parameters)
            parameters_json = json.dumps(parsed)
        except json.JSONDecodeError as exc:
            print(f"error: --parameters is not valid JSON: {exc}", file=sys.stderr)
            return 1

    timestamp = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO decisions (
                production_id, show_type, timestamp,
                run_id_before, run_id_after,
                hypothesis, parameters_changed, rationale, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                production_id,
                args.show_type,
                timestamp,
                args.run_before,
                args.run_after,
                args.hypothesis,
                parameters_json,
                args.rationale,
                args.outcome,
            ),
        )
        decision_id = cur.lastrowid
        conn.commit()

    print(f"Decision #{decision_id} recorded at {timestamp[:19]}Z.")
    print(f"  production_id : {production_id}")
    print(f"  show_type     : {args.show_type or '(all shows)'}")
    print(f"  hypothesis    : {args.hypothesis}")
    if parameters_json:
        print(f"  parameters    : {parameters_json}")
    if args.run_before:
        print(f"  run_before    : {args.run_before}")
    if args.run_after:
        print(f"  run_after     : {args.run_after}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# calibrate
# ─────────────────────────────────────────────────────────────────────────────

def cmd_calibrate(args: argparse.Namespace) -> int:
    production_id = args.production_id or _read_production_id()
    return run_calibration(
        db_path=args.db,
        production_id=production_id,
        apply=args.apply,
        output_path=args.output,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--production-id",
        default=None,
        metavar="ID",
        help="Production identifier (default: production_id from vox_romana.yaml).",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        metavar="PATH",
        help=f"Path to longitudinal.db (default: study_results/longitudinal.db).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studio",
        description="Manage the vox_romana longitudinal data store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # ingest
    p_ingest = sub.add_parser(
        "ingest",
        help="Ingest a study_battery output directory into longitudinal.db.",
    )
    p_ingest.add_argument(
        "study_dir",
        type=Path,
        help="Path to the study_battery output directory (contains index.json).",
    )
    _add_common(p_ingest)

    # status
    p_status = sub.add_parser("status", help="Show DB overview.")
    _add_common(p_status)

    # decision
    p_dec = sub.add_parser(
        "decision",
        help="Record a deliberate parameter change and its expected outcome.",
    )
    _add_common(p_dec)
    p_dec.add_argument(
        "--show-type",
        default=None,
        metavar="TYPE",
        help="Show type this decision targets (omit for cross-cutting changes).",
    )
    p_dec.add_argument(
        "--hypothesis",
        required=True,
        metavar="TEXT",
        help="What you expect this parameter change to achieve.",
    )
    p_dec.add_argument(
        "--parameters",
        default=None,
        metavar="JSON",
        help='Changed parameters as JSON, e.g. \'{"wrap_up_threshold": 0.80}\'.',
    )
    p_dec.add_argument(
        "--rationale",
        default=None,
        metavar="TEXT",
        help="Why you are making this change.",
    )
    p_dec.add_argument(
        "--run-before",
        default=None,
        metavar="RUN_ID",
        help="Run ID that motivated the change.",
    )
    p_dec.add_argument(
        "--run-after",
        default=None,
        metavar="RUN_ID",
        help="Run ID that measures the outcome (fill in after the next battery).",
    )
    p_dec.add_argument(
        "--outcome",
        default=None,
        metavar="TEXT",
        help="Observed outcome (fill in after the next battery run).",
    )

    # calibrate
    p_cal = sub.add_parser(
        "calibrate",
        help="Preview or apply calibration recommendations.",
    )
    _add_common(p_cal)
    p_cal.add_argument(
        "--apply",
        action="store_true",
        help="Persist calibration_state rows and write calibration.yaml.",
    )
    p_cal.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="Path to calibration.yaml when using --apply.",
    )

    args = parser.parse_args(argv)
    dispatch = {
        "ingest": cmd_ingest,
        "status": cmd_status,
        "decision": cmd_decision,
        "calibrate": cmd_calibrate,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
