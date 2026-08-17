#!/usr/bin/env python3
"""Calibrate creative parameters from longitudinal battery history."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "study_results" / "longitudinal.db"
DEFAULT_OUTPUT = REPO_ROOT / "calibration.yaml"

sys.path.insert(0, str(REPO_ROOT))

from vox_romana.analysis.longitudinal import init_db  # noqa: E402
from vox_romana.cli.runners import (  # noqa: E402
    _DEBATE_AVG_WORDS_PER_TURN,
    _DEBATE_HOST_OVERHEAD_WORDS,
    _NEWS_AVG_WORDS_PER_SEGMENT,
    _NEWS_HOST_OVERHEAD_WORDS,
)
from vox_romana.core.conductor import BaseConductor  # noqa: E402
from vox_romana.core.duration import WPM_DEFAULTS  # noqa: E402
from vox_romana.shows.lecture.conductor import MAX_BEAT_WORDS as LECTURE_MAX_BEAT_WORDS  # noqa: E402
from vox_romana.shows.news.conductor import MAX_BEAT_WORDS as NEWS_MAX_BEAT_WORDS  # noqa: E402


_RATIO_LOW = 0.85
_RATIO_HIGH = 1.15
_GLOBAL_RATIO_LOW = 0.90
_GLOBAL_RATIO_HIGH = 1.10
_DEBATE_MIN_TURNS = 4
_NEWS_MIN_SEGMENTS = 1


@dataclass(slots=True)
class Recommendation:
    production_id: str
    show_type: str
    source: str
    strategy: str
    n_runs: int
    n_pieces: int
    mean_ratio: float | None
    current: dict[str, Any]
    recommended: dict[str, Any]
    notes: list[str]
    r_squared: float | None = None


def _read_production_id() -> str:
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


def _live_defaults(show_type: str) -> dict[str, Any]:
    defaults = {
        "overshoot_compensation": BaseConductor.OVERSHOOT_COMPENSATION,
        "wrap_up_threshold": BaseConductor.WRAP_UP_THRESHOLD,
        "wpm_default": WPM_DEFAULTS.get(show_type),
        "max_beat_words": None,
        "avg_words_per_turn": None,
        "avg_words_per_segment": None,
    }
    if show_type == "debate":
        defaults["avg_words_per_turn"] = _DEBATE_AVG_WORDS_PER_TURN
    elif show_type == "news":
        defaults["max_beat_words"] = NEWS_MAX_BEAT_WORDS
        defaults["avg_words_per_segment"] = _NEWS_AVG_WORDS_PER_SEGMENT
    elif show_type == "lecture":
        defaults["max_beat_words"] = LECTURE_MAX_BEAT_WORDS
    return defaults


def _clamp_float(value: float, low: float, high: float, digits: int = 3) -> float:
    return round(min(max(value, low), high), digits)


def _clamp_int(value: float, low: int, high: int) -> int:
    return int(min(max(round(value), low), high))


def _query_rows(conn: sqlite3.Connection, query: str, bind: tuple[Any, ...]) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(query, bind).fetchall()


def _get_current_params(
    conn: sqlite3.Connection,
    production_id: str,
    show_type: str,
) -> tuple[dict[str, Any], str]:
    row = conn.execute(
        """
        SELECT overshoot_compensation, wrap_up_threshold, wpm_default,
               max_beat_words, avg_words_per_turn, avg_words_per_segment
        FROM calibration_state
        WHERE production_id = ? AND show_type = ?
        """,
        (production_id, show_type),
    ).fetchone()
    if row:
        current = dict(row)
        merged = _live_defaults(show_type)
        merged.update({k: v for k, v in current.items() if v is not None})
        return merged, "calibration_state"

    row = conn.execute(
        """
        SELECT overshoot_compensation, wrap_up_threshold, wpm_default,
               max_beat_words, avg_words_per_turn, avg_words_per_segment
        FROM runs
        WHERE production_id = ? AND show_type = ?
        ORDER BY run_id DESC
        LIMIT 1
        """,
        (production_id, show_type),
    ).fetchone()
    if row:
        current = dict(row)
        merged = _live_defaults(show_type)
        merged.update({k: v for k, v in current.items() if v is not None})
        return merged, "latest_run"

    baseline = _global_baseline(conn, show_type)
    if baseline:
        merged = _live_defaults(show_type)
        merged.update(baseline)
        return merged, "global_baseline"

    return _live_defaults(show_type), "live_defaults"


def _global_baseline(conn: sqlite3.Connection, show_type: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT AVG(r.overshoot_compensation) AS overshoot_compensation,
               AVG(r.wrap_up_threshold) AS wrap_up_threshold,
               AVG(r.wpm_default) AS wpm_default,
               AVG(r.max_beat_words) AS max_beat_words,
               AVG(r.avg_words_per_turn) AS avg_words_per_turn,
               AVG(r.avg_words_per_segment) AS avg_words_per_segment,
               COUNT(DISTINCT r.run_id) AS n_runs,
               COUNT(*) AS n_pieces
        FROM pieces AS p
        JOIN runs AS r
          ON r.run_id = p.run_id
         AND r.show_type = p.show_type
        WHERE p.show_type = ?
          AND p.exit_code = 0
          AND p.word_ratio BETWEEN ? AND ?
        """,
        (show_type, _GLOBAL_RATIO_LOW, _GLOBAL_RATIO_HIGH),
    ).fetchone()
    if not row or not row["n_pieces"]:
        return None
    return {
        "overshoot_compensation": row["overshoot_compensation"],
        "wrap_up_threshold": row["wrap_up_threshold"],
        "wpm_default": round(row["wpm_default"]) if row["wpm_default"] is not None else None,
        "max_beat_words": round(row["max_beat_words"]) if row["max_beat_words"] is not None else None,
        "avg_words_per_turn": round(row["avg_words_per_turn"]) if row["avg_words_per_turn"] is not None else None,
        "avg_words_per_segment": round(row["avg_words_per_segment"]) if row["avg_words_per_segment"] is not None else None,
        "_n_runs": row["n_runs"],
        "_n_pieces": row["n_pieces"],
    }


def _piece_rows(
    conn: sqlite3.Connection,
    production_id: str,
    show_type: str,
) -> list[sqlite3.Row]:
    return _query_rows(
        conn,
        """
        SELECT p.run_id, p.word_ratio, p.target_duration_mins, p.target_words,
               p.architect_section_count, p.exit_code,
               r.overshoot_compensation, r.max_beat_words,
               r.avg_words_per_turn, r.avg_words_per_segment
        FROM pieces AS p
        JOIN runs AS r
          ON r.run_id = p.run_id
         AND r.show_type = p.show_type
        WHERE p.production_id = ?
          AND p.show_type = ?
          AND p.exit_code = 0
          AND p.word_ratio IS NOT NULL
        ORDER BY p.piece_id
        """,
        (production_id, show_type),
    )


def _show_types_for_production(conn: sqlite3.Connection, production_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT show_type FROM runs WHERE production_id = ? ORDER BY show_type",
        (production_id,),
    ).fetchall()
    if rows:
        return [r[0] for r in rows]

    rows = conn.execute(
        """
        SELECT DISTINCT p.show_type
        FROM pieces AS p
        WHERE p.exit_code = 0
          AND p.word_ratio BETWEEN ? AND ?
        ORDER BY p.show_type
        """,
        (_GLOBAL_RATIO_LOW, _GLOBAL_RATIO_HIGH),
    ).fetchall()
    return [r[0] for r in rows]


def _detect_binding_constraint(
    show_type: str,
    rows: list[sqlite3.Row],
    current: dict[str, Any],
) -> tuple[bool, str | None, int | None]:
    if show_type == "debate":
        turn_words = current.get("avg_words_per_turn") or _DEBATE_AVG_WORDS_PER_TURN
        floor_ratios: list[float] = []
        for row in rows:
            target_words = row["target_words"]
            if target_words:
                floor_words = _DEBATE_HOST_OVERHEAD_WORDS + (_DEBATE_MIN_TURNS * turn_words)
                floor_ratios.append(floor_words / target_words)
        if floor_ratios and sum(floor_ratios) / len(floor_ratios) > _RATIO_HIGH:
            avg_target = sum(r["target_words"] for r in rows if r["target_words"]) / len(
                [r for r in rows if r["target_words"]]
            )
            feasible = int((avg_target - _DEBATE_HOST_OVERHEAD_WORDS) / _DEBATE_MIN_TURNS)
            if feasible < 1:
                return True, (
                    "Format floor exceeds the target word budget: host overhead alone"
                    " dominates short debate runs. Treat this as a format constraint,"
                    " not an overshoot_compensation problem."
                ), None
            return True, (
                "Format floor is binding for the observed target durations; calibrate"
                " avg_words_per_turn before overshoot_compensation."
            ), _clamp_int(feasible, 1, 120)

    if show_type == "news":
        segment_words = current.get("avg_words_per_segment") or _NEWS_AVG_WORDS_PER_SEGMENT
        floor_ratios = []
        for row in rows:
            target_words = row["target_words"]
            if target_words:
                floor_words = _NEWS_HOST_OVERHEAD_WORDS + (_NEWS_MIN_SEGMENTS * segment_words)
                floor_ratios.append(floor_words / target_words)
        if floor_ratios and sum(floor_ratios) / len(floor_ratios) > _RATIO_HIGH:
            avg_target = sum(r["target_words"] for r in rows if r["target_words"]) / len(
                [r for r in rows if r["target_words"]]
            )
            feasible = int(avg_target - _NEWS_HOST_OVERHEAD_WORDS)
            if feasible < 1:
                return True, (
                    "Format floor exceeds the target word budget: host overhead alone"
                    " dominates short news runs. Treat this as a format constraint,"
                    " not a max_beat_words problem."
                ), None
            return True, (
                "Format floor is binding for the observed target durations; calibrate"
                " avg_words_per_segment before max_beat_words."
            ), _clamp_int(feasible, 1, 500)

    return False, None, None


def _can_fit_ols(rows: list[sqlite3.Row]) -> tuple[bool, str]:
    if len(rows) < 10:
        return False, "fewer than 10 successful pieces"

    adjustable_columns = ["overshoot_compensation", "max_beat_words"]
    varied = []
    for col in adjustable_columns:
        values = [row[col] for row in rows if row[col] is not None]
        if len(values) == len(rows) and len(set(values)) > 1:
            varied.append(col)

    if not varied:
        return False, "no variation in overshoot_compensation or max_beat_words"
    return True, ""


def _fit_ols(
    show_type: str,
    rows: list[sqlite3.Row],
    current: dict[str, Any],
) -> tuple[dict[str, Any] | None, float | None, str | None]:
    candidate_columns = [
        "overshoot_compensation",
        "max_beat_words",
        "target_duration_mins",
        "architect_section_count",
    ]
    usable_columns = []
    complete_rows = rows
    for col in candidate_columns:
        values = [row[col] for row in complete_rows if row[col] is not None]
        if len(values) == len(complete_rows) and len(set(values)) > 1:
            usable_columns.append(col)

    adjustable = [c for c in usable_columns if c in {"overshoot_compensation", "max_beat_words"}]
    if not adjustable:
        return None, None, "OLS skipped: adjustable parameters did not vary."

    x_rows = []
    y_rows = []
    for row in complete_rows:
        x_rows.append([1.0, *[float(row[c]) for c in usable_columns]])
        y_rows.append(float(row["word_ratio"]))

    x = np.array(x_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    residual = float(np.sum((y - pred) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total == 0.0 else max(0.0, 1.0 - (residual / total))

    if show_type in {"news", "lecture"} and "max_beat_words" in adjustable:
        primary = "max_beat_words"
    elif "overshoot_compensation" in adjustable:
        primary = "overshoot_compensation"
    else:
        primary = adjustable[0]

    current_point: dict[str, float] = {}
    for col in usable_columns:
        value = current.get(col)
        if value is None:
            value = float(np.mean([row[col] for row in complete_rows if row[col] is not None]))
        current_point[col] = float(value)

    base_pred = beta[0]
    for idx, col in enumerate(usable_columns, start=1):
        base_pred += beta[idx] * current_point[col]

    primary_idx = usable_columns.index(primary) + 1
    coef = float(beta[primary_idx])
    if abs(coef) < 1e-6:
        return None, r_squared, f"OLS skipped: {primary} coefficient is too small to invert."

    recommended_value = current_point[primary] + ((1.0 - base_pred) / coef)
    recommended = dict(current)
    if primary == "overshoot_compensation":
        recommended[primary] = _clamp_float(recommended_value, 0.1, 1.5)
    else:
        recommended[primary] = _clamp_int(recommended_value, 10, 500)
    return recommended, r_squared, None


def _ratio_adjusted(current: dict[str, Any], mean_ratio: float) -> dict[str, Any]:
    factor = 1.0 / mean_ratio
    recommended = dict(current)
    overshoot = current.get("overshoot_compensation")
    if overshoot is not None:
        recommended["overshoot_compensation"] = _clamp_float(overshoot * factor, 0.1, 1.5)

    max_beat_words = current.get("max_beat_words")
    if max_beat_words is not None:
        recommended["max_beat_words"] = _clamp_int(max_beat_words * factor, 10, 500)
    return recommended


def _recommend_show(
    conn: sqlite3.Connection,
    production_id: str,
    show_type: str,
) -> Recommendation:
    current, source = _get_current_params(conn, production_id, show_type)
    rows = _piece_rows(conn, production_id, show_type)
    if not rows:
        if source == "global_baseline":
            baseline = _global_baseline(conn, show_type) or {}
            notes = [
                "No local successful pieces with word_ratio data.",
                "Using _global baseline assembled from in-range runs across productions.",
            ]
            return Recommendation(
                production_id=production_id,
                show_type=show_type,
                source=source,
                strategy="global_baseline",
                n_runs=int(baseline.get("_n_runs", 0)),
                n_pieces=int(baseline.get("_n_pieces", 0)),
                mean_ratio=None,
                current=current,
                recommended={k: v for k, v in current.items() if not k.startswith("_")},
                notes=notes,
            )
        return Recommendation(
            production_id=production_id,
            show_type=show_type,
            source=source,
            strategy="no_data",
            n_runs=0,
            n_pieces=0,
            mean_ratio=None,
            current=current,
            recommended=current,
            notes=["No successful pieces with word_ratio data for this production/show."],
        )

    n_runs = len({row["run_id"] for row in rows})
    n_pieces = len(rows)
    mean_ratio = sum(row["word_ratio"] for row in rows) / n_pieces
    recommended = dict(current)
    notes: list[str] = []

    if _RATIO_LOW <= mean_ratio <= _RATIO_HIGH:
        notes.append("Observed mean_ratio is already within the target band.")
        return Recommendation(
            production_id=production_id,
            show_type=show_type,
            source=source,
            strategy="in_range",
            n_runs=n_runs,
            n_pieces=n_pieces,
            mean_ratio=mean_ratio,
            current=current,
            recommended=recommended,
            notes=notes,
        )

    binding, binding_note, structural_value = _detect_binding_constraint(show_type, rows, current)
    if binding:
        if show_type == "debate" and structural_value is not None:
            recommended["avg_words_per_turn"] = structural_value
        elif show_type == "news" and structural_value is not None:
            recommended["avg_words_per_segment"] = structural_value
        if binding_note:
            notes.append(binding_note)
        return Recommendation(
            production_id=production_id,
            show_type=show_type,
            source=source,
            strategy="binding",
            n_runs=n_runs,
            n_pieces=n_pieces,
            mean_ratio=mean_ratio,
            current=current,
            recommended=recommended,
            notes=notes,
        )

    can_fit, reason = _can_fit_ols(rows)
    if can_fit:
        ols_recommended, r_squared, ols_note = _fit_ols(show_type, rows, current)
        if ols_recommended is not None:
            notes.append("Using OLS because there are at least 10 successful pieces with parameter variation.")
            return Recommendation(
                production_id=production_id,
                show_type=show_type,
                source=source,
                strategy="ols",
                n_runs=n_runs,
                n_pieces=n_pieces,
                mean_ratio=mean_ratio,
                current=current,
                recommended=ols_recommended,
                notes=notes,
                r_squared=r_squared,
            )
        if ols_note:
            notes.append(ols_note)
    else:
        notes.append(f"OLS skipped: {reason}.")

    notes.append("Falling back to ratio-based update.")
    return Recommendation(
        production_id=production_id,
        show_type=show_type,
        source=source,
        strategy="ratio",
        n_runs=n_runs,
        n_pieces=n_pieces,
        mean_ratio=mean_ratio,
        current=current,
        recommended=_ratio_adjusted(current, mean_ratio),
        notes=notes,
    )


def build_recommendations(db_path: str | Path, production_id: str) -> list[Recommendation]:
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        init_db(db_path)
        show_types = _show_types_for_production(conn, production_id)
        return [_recommend_show(conn, production_id, show_type) for show_type in show_types]


def _apply_recommendations(
    db_path: str | Path,
    production_id: str,
    recommendations: list[Recommendation],
    output_path: str | Path,
) -> None:
    db_path = Path(db_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        init_db(db_path)
        for rec in recommendations:
            if rec.strategy == "no_data":
                continue
            conn.execute(
                """
                INSERT INTO calibration_state (
                    production_id, show_type,
                    overshoot_compensation, wrap_up_threshold, wpm_default,
                    max_beat_words, avg_words_per_turn, avg_words_per_segment,
                    r_squared, n_runs_used, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(production_id, show_type) DO UPDATE SET
                    overshoot_compensation = excluded.overshoot_compensation,
                    wrap_up_threshold = excluded.wrap_up_threshold,
                    wpm_default = excluded.wpm_default,
                    max_beat_words = excluded.max_beat_words,
                    avg_words_per_turn = excluded.avg_words_per_turn,
                    avg_words_per_segment = excluded.avg_words_per_segment,
                    r_squared = excluded.r_squared,
                    n_runs_used = excluded.n_runs_used,
                    updated_at = excluded.updated_at
                """,
                (
                    production_id,
                    rec.show_type,
                    rec.recommended.get("overshoot_compensation"),
                    rec.recommended.get("wrap_up_threshold"),
                    rec.recommended.get("wpm_default"),
                    rec.recommended.get("max_beat_words"),
                    rec.recommended.get("avg_words_per_turn"),
                    rec.recommended.get("avg_words_per_segment"),
                    rec.r_squared,
                    rec.n_runs,
                    now,
                ),
            )
        conn.commit()

    payload: dict[str, Any] = {"calibration": {}}
    for rec in recommendations:
        if rec.strategy == "no_data":
            continue
        show_payload = {}
        for key in [
            "overshoot_compensation",
            "wrap_up_threshold",
            "max_beat_words",
            "avg_words_per_turn",
            "avg_words_per_segment",
        ]:
            value = rec.recommended.get(key)
            if value is not None:
                show_payload[key] = value
        if show_payload:
            payload["calibration"][rec.show_type] = show_payload

    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _print_recommendations(
    recommendations: list[Recommendation],
    production_id: str,
    db_path: Path,
    apply: bool,
    output_path: Path,
) -> None:
    print("═" * 78)
    print("CALIBRATION REPORT")
    print(f"production_id : {production_id!r}")
    print(f"DB            : {db_path}")
    print(f"mode          : {'apply' if apply else 'preview'}")
    if apply:
        print(f"output        : {output_path}")
    print("─" * 78)

    if not recommendations:
        print("No data available. Ingest a battery run first.")
        return

    for rec in recommendations:
        ratio_s = f"{rec.mean_ratio:.3f}" if rec.mean_ratio is not None else "N/A"
        print(
            f"\n[{rec.show_type}] strategy={rec.strategy} source={rec.source}"
            f" n_runs={rec.n_runs} n_pieces={rec.n_pieces} mean_ratio={ratio_s}"
        )
        current_bits = []
        recommended_bits = []
        for key in [
            "overshoot_compensation",
            "max_beat_words",
            "avg_words_per_turn",
            "avg_words_per_segment",
        ]:
            cur = rec.current.get(key)
            new = rec.recommended.get(key)
            if cur is not None:
                current_bits.append(f"{key}={cur}")
            if new is not None:
                recommended_bits.append(f"{key}={new}")
        if current_bits:
            print(f"  current     : {', '.join(current_bits)}")
        if recommended_bits:
            print(f"  recommended : {', '.join(recommended_bits)}")
        if rec.r_squared is not None:
            print(f"  r_squared   : {rec.r_squared:.3f}")
        for note in rec.notes:
            print(f"  note        : {note}")

    print()
    if apply:
        print("Applied recommendations to calibration_state and wrote calibration.yaml.")
    else:
        print("Preview only. Re-run with --apply to persist calibration_state and calibration.yaml.")


def run_calibration(
    *,
    db_path: str | Path,
    production_id: str,
    apply: bool = False,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> int:
    db_path = Path(db_path)
    output_path = Path(output_path)
    recommendations = build_recommendations(db_path, production_id)
    _print_recommendations(recommendations, production_id, db_path, apply, output_path)
    if apply and recommendations:
        _apply_recommendations(db_path, production_id, recommendations, output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description="Recommend and optionally persist creative parameter calibration.",
    )
    parser.add_argument(
        "--production-id",
        default=None,
        metavar="ID",
        help="Production identifier (default: production_id from vox_romana.yaml).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        metavar="PATH",
        help="Path to longitudinal.db.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist the recommendations to calibration_state and calibration.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="Path to calibration.yaml (used with --apply).",
    )
    args = parser.parse_args(argv)
    production_id = args.production_id or _read_production_id()
    return run_calibration(
        db_path=args.db,
        production_id=production_id,
        apply=args.apply,
        output_path=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())